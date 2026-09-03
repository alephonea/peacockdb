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
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef};
use datafusion::arrow::ipc::reader::StreamReader;

use peacockdb_ffi::raw::{
    PeacockExecutor, PeacockNodeRegion, PeacockNodeStats, peacock_executor_collect_node_regions,
    peacock_executor_execute_node, peacock_last_error, peacock_result_free,
    peacock_result_from_handle,
};

use crate::executors::node_timing_on;
use crate::memory::logical_size_from_schema;

use super::batch::Batch;
use super::driver::Region;
use super::cpu_batch::CpuBatch;
use super::error::PlanError;
use super::executor::{AbiCalls, BackendError, CallResult, CallStats, RowRange};
use super::gpu_batch::GpuBatch;
use super::recipe::{CallPattern, FbKind, Input, Recipe, Seq};

/// A node's calls, in order — the batch into the first, each output into the next.
///
/// The session pointer is BORROWED, as everywhere on the GPU path: the session outlives
/// every executor drawn from it, and the handles it hands back.
pub struct GpuExec {
    dev: Device,
    calls: Vec<(Seq, FbKind)>,
    schema: SchemaRef,
}

impl GpuExec {
    /// `schema` is what the node declares it produces, which is what prices the batch —
    /// the ABI reports rows and varlen content, and the fixed width per row is the
    /// schema's.
    pub fn new(dev: Device, recipe: &Recipe, schema: &ArrowSchema) -> Result<Self, PlanError> {
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
            dev,
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
        let mut calls = AbiCalls::armed(node_timing_on());
        // Only the first call reads a batch this side priced; every later one reads the
        // one before it, which the recipe names `PriorOutput` and only C++ measured.
        let taken = calls
            .is_armed()
            .then(|| Consumed::of(&batch))
            .unwrap_or_default();
        let mut input = (taken.rows, Some(taken.bytes));
        let (_, mut handle) = batch.consume();
        let mut stats = PeacockNodeStats::default();
        for (seq, kind) in &self.calls {
            let (produced, node_stats) = execute_node(self.dev, *seq, *kind, &[vec![handle]])?;
            calls.record(*seq, *kind, input.0, input.1);
            handle = produced;
            stats = node_stats;
            input = (node_stats.rows, None);
        }
        Ok((
            produced(self.dev.executor, handle, stats, &self.schema),
            CallStats {
                scratch_bytes: None,
                calls,
            },
        ))
    }
}

/// Where the data leaves the device: one export per handle, over the row range the driver
/// supplies. Named for the call rather than for the node, since `GpuUnload` is the node.
pub struct GpuExport {
    dev: Device,
    schema: SchemaRef,
}

impl GpuExport {
    /// A sink declares no schema of its own, so this is its input's — the columns that
    /// cross the boundary.
    pub fn new(dev: Device, schema: &ArrowSchema) -> Self {
        Self {
            dev,
            schema: Arc::new(schema.clone()),
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
                self.dev.executor,
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
                last_error(self.dev.executor)
            )));
        }
        // A range naming no rows exports nothing at all, and there is nothing to free.
        if len == 0 {
            return Ok((
                CpuBatch::new(RecordBatch::new_empty(self.schema.clone())),
                no_abi_calls(),
            ));
        }
        let decoded = decode(unsafe { std::slice::from_raw_parts(ipc, len as usize) });
        unsafe { peacock_result_free(ipc) };
        let batches = decoded?;
        let batch = concat_batches(&self.schema, batches.iter()).map_err(|error| {
            BackendError::new(format!(
                "the exported stream is not the sink's rows: {error}"
            ))
        })?;
        Ok((CpuBatch::new(batch), no_abi_calls()))
    }
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

/// The session an executor drives, and where in the plan it is driving it.
///
/// One value rather than a pointer and two numbers threaded separately: every executor
/// already carried the pointer, so this costs no argument, and a call that knows only its
/// seq cannot say which lane it was for. C++ answers with `(seq, partition, call_index)`,
/// where `partition` is the output slot inside one call — at four lanes every one of them
/// reports 0, so the region alone cannot tell lane 1 batch 0 from lane 0 batch 1.
#[derive(Clone, Copy)]
pub struct Device {
    pub executor: *mut PeacockExecutor,
    /// Post-order position in the plan tree — the index recipes and the report share.
    pub node: usize,
    pub lane: usize,
}

/// Drain what the session recorded, after the root export and before the plan ends —
/// the events die with the plan, and the device times do not exist until then.
///
/// `cap` must bound what the run produced. C++ FAILS rather than truncating, and by then
/// the drain has happened, so a cap that is too small loses the measurement instead of
/// reporting less of it. On this path the bound is the calls the driver recorded times the
/// widest output any single call can have.
pub fn collect_regions(
    executor: *mut PeacockExecutor,
    cap: usize,
) -> Result<Vec<Region>, BackendError> {
    let mut buf = vec![PeacockNodeRegion::default(); cap];
    let mut count = 0u64;
    let rc = unsafe {
        peacock_executor_collect_node_regions(executor, buf.as_mut_ptr(), cap as u64, &mut count)
    };
    if rc != 0 {
        return Err(BackendError::new(format!(
            "collect_node_regions(cap={cap}): {}",
            last_error(executor)
        )));
    }
    Ok(buf[..count as usize]
        .iter()
        .map(|region| Region {
            seq: region.seq as Seq,
            partition: region.partition as usize,
            call_index: region.call_index,
            host_setup_us: region.host_setup_us,
            host_submit_us: region.host_submit_us,
            device_us: region.device_us,
            out_rows: region.rows,
            logical_bytes: region.logical_bytes,
            schema_faithful: region.schema_faithful != 0,
        })
        .collect())
}

/// What a call reports when it made no ABI call of its own — an accumulator that only
/// took the batch, a scan that had nothing left to read.
///
/// Not `CallStats::default()`: that says nobody was measuring, and on a measured run an
/// empty list is the true answer rather than the absent one.
pub(super) fn no_abi_calls() -> CallStats {
    CallStats {
        scratch_bytes: None,
        calls: AbiCalls::armed(node_timing_on()),
    }
}

/// What a call was handed, as the caller priced it. Read where the handles are given
/// up: `GpuBatch::consume` is where a batch's own figures stop being reachable.
#[derive(Clone, Copy, Default)]
pub(super) struct Consumed {
    pub rows: u64,
    pub bytes: u64,
}

impl Consumed {
    pub(super) fn of(batch: &GpuBatch) -> Self {
        Self {
            rows: batch.num_rows() as u64,
            bytes: batch.byte_size() as u64,
        }
    }

    /// A whole handover priced in one pass, for a caller that has already checked its log
    /// is armed — an unmeasured run does not walk the batches at all.
    pub(super) fn sum(batches: &[GpuBatch]) -> Self {
        batches.iter().fold(Self::default(), |mut total, batch| {
            total.rows += batch.num_rows() as u64;
            total.bytes += batch.byte_size() as u64;
            total
        })
    }
}

/// One `execute_node` against the seq a recipe named, its handles grouped by the child
/// slot each fills. The call CONSUMES them, so a caller hands over batches it will not
/// release itself.
pub(super) fn execute_node(
    dev: Device,
    seq: Seq,
    kind: FbKind,
    inputs: &[Vec<u64>],
) -> Result<(u64, PeacockNodeStats), BackendError> {
    let [one] = <[(u64, PeacockNodeStats); 1]>::try_from(execute_node_many(
        dev, seq, kind, inputs, 1,
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
    dev: Device,
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
            dev.executor,
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
            "execute_node(#{seq} {kind}) for node {} lane {}: {}",
            dev.node,
            dev.lane,
            last_error(dev.executor)
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
