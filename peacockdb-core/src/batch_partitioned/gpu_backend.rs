//! The GPU backend's executors: a node's recipe is the instruction set, and running it is
//! all these do.
//!
//! No legacy operator code is reached from here, deliberately: the recipe already says
//! which seqs to address and in what order, so an executor that also built cuDF calls
//! would be a second path to the same kernels, and the two would drift where the plan
//! golden pins only the recipe.
//!
//! So an executor holds a borrowed session pointer, its recipe's calls, and the schema its
//! output is priced by. Handles thread from one call to the next.

pub mod accumulate;
pub mod backend;
pub mod emit;
pub mod join;
pub mod source;

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::compute::{cast, concat_batches};
use datafusion::arrow::datatypes::{Field, Schema as ArrowSchema, SchemaRef};
use datafusion::arrow::ipc::reader::StreamReader;

use peacockdb_ffi::raw::{
    PeacockExecutor, PeacockNodeStats, peacock_executor_execute_node, peacock_last_error,
    peacock_result_free, peacock_result_from_handle,
};

use crate::memory::logical_size_from_schema;

use super::cpu_batch::CpuBatch;
use super::error::PlanError;
use super::executor::{BackendError, CallResult, CallStats, RowRange};
use super::exports::Exports;
use super::gpu_batch::GpuBatch;
use super::recipe::{CallPattern, FbKind, Input, Recipe, Seq};

/// A node's calls, in order — the batch into the first, each output into the next.
///
/// The session pointer is BORROWED, as everywhere on the GPU path: the session outlives
/// every executor drawn from it, and the handles it hands back.
pub struct GpuExec {
    executor: *mut PeacockExecutor,
    calls: Vec<(Seq, FbKind)>,
    schema: SchemaRef,
}

impl GpuExec {
    /// `schema` is what the node declares it produces, which is what prices the batch —
    /// the ABI reports rows and varlen content, and the fixed width per row is the
    /// schema's.
    pub fn new(
        executor: *mut PeacockExecutor,
        recipe: &Recipe,
        schema: &ArrowSchema,
    ) -> Result<Self, PlanError> {
        let mut calls = Vec::with_capacity(recipe.calls.len());
        for (position, call) in recipe.calls.iter().enumerate() {
            if call.when != CallPattern::PerBatch {
                return Err(PlanError::Invalid(format!(
                    "an exec node calls per batch and this recipe's call {position} is \
                     {:?} — a node whose calls wait for done is an accumulator",
                    call.when
                )));
            }
            let expected = if position == 0 {
                Input::Batch
            } else {
                Input::PriorOutput
            };
            if call.inputs.as_slice() != [expected] {
                return Err(PlanError::Invalid(format!(
                    "an exec node's calls are a straight line — the batch into the first \
                     and each output into the next — and call {position} takes {:?}",
                    call.inputs
                )));
            }
            let (seq, kind) = call.target.ok_or_else(|| {
                PlanError::Invalid(format!(
                    "call {position} takes runtime bounds rather than a seq, which no exec \
                     node does"
                ))
            })?;
            calls.push((seq, kind));
        }
        Ok(Self {
            executor,
            calls,
            schema: Arc::new(schema.clone()),
        })
    }

    /// One batch in, one batch out. The input handle is consumed by the first call and
    /// every intermediate by the call after it, so what is released here is nothing: a
    /// failed call ends the query, and the session it belonged to is torn down with it.
    pub fn exec(&mut self, batch: GpuBatch) -> CallResult<GpuBatch> {
        // The session is this executor's, not the batch's: a batch carries the pointer so
        // that dropping it can release its handle, and every batch reaching a node was
        // drawn from the session the node was built against.
        let (_, mut handle) = batch.consume();
        let mut stats = PeacockNodeStats::default();
        for (seq, kind) in &self.calls {
            let (produced, node_stats) = execute_node(self.executor, *seq, *kind, &[vec![handle]])?;
            handle = produced;
            stats = node_stats;
        }
        Ok((
            produced(self.executor, handle, stats, &self.schema),
            CallStats::default(),
        ))
    }
}

/// Where the data leaves the device: one export per handle, over the row range the driver
/// supplies. Named for the call rather than for the node, since `GpuUnload` is the node.
pub struct GpuExport {
    executor: *mut PeacockExecutor,
    schema: SchemaRef,
    /// The columns the device hands back as `Utf8` where the sink declared a wider string
    /// — the one divergence neither side can avoid, and so the only one absorbed here.
    casts: Vec<u32>,
}

impl GpuExport {
    /// A sink declares no schema of its own, so this is its input's — the columns that
    /// cross the boundary. `exports` is the plan's prediction about those same columns,
    /// derived where the plan was built.
    pub fn new(executor: *mut PeacockExecutor, schema: &ArrowSchema, exports: &Exports) -> Self {
        let casts = exports.cast_ordinals();
        // The two arguments are two derivations of one node's input, and nothing in the
        // types says so: a pair from different nodes would index past the end at the cast.
        assert!(
            casts.iter().all(|at| (*at as usize) < schema.fields().len()),
            "the sink's exports address column {} and its schema declares {} — the schema and              the exports are about different nodes",
            casts.iter().max().copied().unwrap_or(0),
            schema.fields().len()
        );
        Self {
            executor,
            schema: Arc::new(schema.clone()),
            casts,
        }
    }

    /// The export does not consume the handle, so the batch is released here by going out
    /// of scope — which is the whole of what the row range buys: the rows wanted cross
    /// PCIe rather than the batch they sit in.
    pub fn unload(&mut self, batch: GpuBatch, rows: RowRange) -> CallResult<CpuBatch> {
        let mut ipc: *mut u8 = std::ptr::null_mut();
        let mut len = 0u64;
        let rc = unsafe {
            peacock_result_from_handle(
                self.executor,
                batch.handle(),
                rows.offset,
                rows.length,
                &mut ipc,
                &mut len,
            )
        };
        if rc != 0 {
            return Err(BackendError::new(format!(
                "result_from_handle({}, {}..+{}): {}",
                batch.handle(),
                rows.offset,
                rows.length,
                last_error(self.executor)
            )));
        }
        // A range naming no rows exports nothing at all, and there is nothing to free.
        if len == 0 {
            return Ok((
                CpuBatch::new(RecordBatch::new_empty(self.schema.clone())),
                CallStats::default(),
            ));
        }
        let decoded = decode(unsafe { std::slice::from_raw_parts(ipc, len as usize) });
        unsafe { peacock_result_free(ipc) };
        let batches = self.absorb(decoded?)?;
        let batch = concat_batches(&self.schema, batches.iter()).map_err(not_the_sinks_rows)?;
        Ok((CpuBatch::new(batch), CallStats::default()))
    }

    /// The decoded batches under the sink's own schema, casting the columns the plan
    /// predicted would need it and leaving every other difference to fail.
    ///
    /// Narrow on purpose: the concat below is the only thing checking that the device
    /// produced what the plan declared, and a blanket cast to the schema would fix #183
    /// by removing the check that found #187.
    fn absorb(&self, batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>, BackendError> {
        if self.casts.is_empty() {
            return Ok(batches);
        }
        batches
            .into_iter()
            .map(|batch| self.cast_declared(batch))
            .collect()
    }

    /// The exported batch under its own schema with the cast columns retyped — not under
    /// the sink's. Handing it the sink's schema would name the columns rather than compare
    /// them, and the concat below compares names, nullability and every type this did not
    /// touch.
    fn cast_declared(&self, batch: RecordBatch) -> Result<RecordBatch, BackendError> {
        // An ordinal comes from the declared schema, so it addresses the exported batch
        // only while the two are the same width. Zipping instead would truncate to the
        // shorter and lose the check the concat makes when nothing is cast.
        if batch.num_columns() != self.schema.fields().len() {
            return Err(not_the_sinks_rows(format!(
                "{} exported columns against {} declared",
                batch.num_columns(),
                self.schema.fields().len()
            )));
        }
        let mut columns = batch.columns().to_vec();
        let mut fields: Vec<Field> = batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect();
        for ordinal in &self.casts {
            let at = *ordinal as usize;
            let declared = self.schema.field(at).data_type();
            columns[at] = cast(&columns[at], declared).map_err(not_the_sinks_rows)?;
            fields[at] = fields[at].clone().with_data_type(declared.clone());
        }
        let schema = ArrowSchema::new_with_metadata(fields, batch.schema().metadata().clone());
        RecordBatch::try_new(Arc::new(schema), columns).map_err(not_the_sinks_rows)
    }
}

fn not_the_sinks_rows(error: impl std::fmt::Display) -> BackendError {
    BackendError::new(format!(
        "the exported stream is not the sink's rows: {error}"
    ))
}

fn decode(bytes: &[u8]) -> Result<Vec<RecordBatch>, BackendError> {
    StreamReader::try_new(std::io::Cursor::new(bytes), None)
        .and_then(|stream| stream.collect::<Result<Vec<RecordBatch>, _>>())
        .map_err(|error| BackendError::new(format!("decoding the exported IPC stream: {error}")))
}

/// What a call produced, priced by the schema the node declares: the ABI reports rows and
/// varlen content, and the fixed width per row is the schema's.
pub(super) fn produced(
    executor: *mut PeacockExecutor,
    handle: u64,
    stats: PeacockNodeStats,
    schema: &SchemaRef,
) -> GpuBatch {
    GpuBatch::new(
        executor,
        handle,
        stats.rows as usize,
        logical_size_from_schema(
            schema,
            stats.rows as usize,
            stats.varlen_content_bytes as usize,
        ),
    )
}

/// One `execute_node` against the seq a recipe named, its handles grouped by the child
/// slot each fills. The call CONSUMES them, so a caller hands over batches it will not
/// release itself.
pub(super) fn execute_node(
    executor: *mut PeacockExecutor,
    seq: Seq,
    kind: FbKind,
    inputs: &[Vec<u64>],
) -> Result<(u64, PeacockNodeStats), BackendError> {
    let [one] = <[(u64, PeacockNodeStats); 1]>::try_from(execute_node_many(
        executor, seq, kind, inputs, 1,
    )?)
    .map_err(|produced| {
        BackendError::new(format!(
            "execute_node(#{seq} {kind}) answered with {} handles — a node driven here maps \
             one call to one output",
            produced.len()
        ))
    })?;
    Ok(one)
}

/// The same call where the output count is a plan value: a scatter's N lanes. Every other
/// node this backend drives takes the one-output form above.
pub(super) fn execute_node_many(
    executor: *mut PeacockExecutor,
    seq: Seq,
    kind: FbKind,
    inputs: &[Vec<u64>],
    out_cap: usize,
) -> Result<Vec<(u64, PeacockNodeStats)>, BackendError> {
    // Grouped by the child slot each fills, not flattened: the C++ reads its output count
    // off child 0's, so a join's two handles in one group would ask for two outputs and be
    // refused for a buffer it never needed.
    let counts: Vec<u64> = inputs.iter().map(|group| group.len() as u64).collect();
    let inputs: Vec<u64> = inputs.concat();
    let mut handles = vec![0u64; out_cap];
    let mut stats = vec![PeacockNodeStats::default(); out_cap];
    let mut produced = 0u64;
    let rc = unsafe {
        peacock_executor_execute_node(
            executor,
            seq as u64,
            inputs.as_ptr(),
            counts.as_ptr(),
            counts.len() as u64,
            handles.as_mut_ptr(),
            handles.len() as u64,
            &mut produced,
            stats.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(BackendError::new(format!(
            "execute_node(#{seq} {kind}): {}",
            last_error(executor)
        )));
    }
    handles.truncate(produced as usize);
    stats.truncate(produced as usize);
    Ok(handles.into_iter().zip(stats).collect())
}

pub(super) fn last_error(executor: *mut PeacockExecutor) -> String {
    let message = unsafe { peacock_last_error(executor) };
    if message.is_null() {
        return "no message".to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}
