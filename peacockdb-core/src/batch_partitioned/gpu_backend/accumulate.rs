//! The accumulators on a device: what each one holds between calls, and which of its
//! recipe's calls it makes when.
//!
//! Holding is the whole difference from an exec node. A handle an FFI call consumes is
//! gone by move, and one this executor is sitting on is a `GpuBatch` — so a lane the
//! driver abandons releases what it held rather than leaking it.
//!
//! One type with a variant per node, since a backend names one type for the category.

use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef};

use peacockdb_ffi::raw::{PeacockExecutor, peacock_executor_slice_handle};

use crate::memory::logical_size_from_schema;

use super::super::batch::Batch;
use super::super::error::PlanError;
use super::super::executor::{BackendError, CallResult, CallStats, LaneEvent};
use super::super::gpu_batch::GpuBatch;
use super::super::node::RowInterval;
use super::super::recipe::{AbiSymbol, Call, CallPattern, FbKind, Input, Recipe, Seq};
use super::{execute_node, last_error, produced};

/// A `BatchAccumulator` node's executor, one variant per node.
pub enum GpuAccumulator {
    Coalesce(Collapse),
    Sorted(SortedRuns),
    Aggregate(AggregateBatches),
    Limit(LimitStream),
}

impl GpuAccumulator {
    /// One call at done, over everything the lane accumulated.
    pub fn coalesce(
        executor: *mut PeacockExecutor,
        recipe: &Recipe,
        schema: &ArrowSchema,
    ) -> Result<Self, PlanError> {
        let [call] = recipe.calls.as_slice() else {
            return Err(shape("a coalesce makes one call, at done", recipe));
        };
        Ok(Self::Coalesce(Collapse {
            executor,
            collapse: addressed(call, CallPattern::AtDone, Input::LaneBatches)?,
            held: Vec::new(),
            schema: Arc::new(schema.clone()),
        }))
    }

    /// A sort per batch, then one merge over the runs at done.
    pub fn sorted(
        executor: *mut PeacockExecutor,
        recipe: &Recipe,
        schema: &ArrowSchema,
    ) -> Result<Self, PlanError> {
        let [per_batch, at_done] = recipe.calls.as_slice() else {
            return Err(shape(
                "an accumulating sort makes two calls: a sort per batch and a merge at done",
                recipe,
            ));
        };
        Ok(Self::Sorted(SortedRuns {
            executor,
            sort: addressed(per_batch, CallPattern::PerBatch, Input::Batch)?,
            merge: addressed(at_done, CallPattern::AtDone, Input::LaneBatches)?,
            held: Vec::new(),
            schema: Arc::new(schema.clone()),
        }))
    }

    /// A concat and a merge per compaction and again at done, plus the finalize project
    /// where this node finishes the aggregate. `compact_bytes` is the held size that
    /// triggers a compaction; it comes from the budget rule, which is the driver's.
    pub fn aggregate(
        executor: *mut PeacockExecutor,
        recipe: &Recipe,
        state: &ArrowSchema,
        output: &ArrowSchema,
        compact_bytes: usize,
    ) -> Result<Self, PlanError> {
        let (concat, merge, finalize) = match recipe.calls.as_slice() {
            [concat, merge] => (concat, merge, None),
            [concat, merge, finalize] => (
                concat,
                merge,
                Some(addressed(
                    finalize,
                    CallPattern::AtDone,
                    Input::PriorOutput,
                )?),
            ),
            _ => {
                return Err(shape(
                    "a batch aggregate makes a concat and a merge, and a finalize project \
                     where it finishes the aggregate",
                    recipe,
                ));
            }
        };
        Ok(Self::Aggregate(AggregateBatches {
            executor,
            concat: addressed(concat, CallPattern::PerCompaction, Input::LaneBatches)?,
            merge: addressed(merge, CallPattern::PerCompaction, Input::PriorOutput)?,
            finalize,
            state: None,
            pending: Vec::new(),
            threshold: compact_bytes,
            compactions: 0,
            held: Arc::new(state.clone()),
            output: Arc::new(output.clone()),
        }))
    }

    /// No seq at all: a limit's bounds are runtime values, so it addresses the slice
    /// symbol rather than a node.
    pub fn limit(
        executor: *mut PeacockExecutor,
        recipe: &Recipe,
        interval: RowInterval,
        schema: &ArrowSchema,
    ) -> Result<Self, PlanError> {
        let [call] = recipe.calls.as_slice() else {
            return Err(shape(
                "a limit makes one call, per straddling batch",
                recipe,
            ));
        };
        if call.symbol != AbiSymbol::SliceHandle || call.when != CallPattern::PerStraddlingBatch {
            return Err(shape(
                "a limit slices the batches that straddle its ends and calls nothing else",
                recipe,
            ));
        }
        Ok(Self::Limit(LimitStream {
            executor,
            interval,
            seen: 0,
            schema: Arc::new(schema.clone()),
        }))
    }

    pub fn accumulate_and_fetch(&mut self, batch: GpuBatch) -> CallResult<Vec<GpuBatch>> {
        match self {
            Self::Coalesce(state) => state.accumulate_and_fetch(batch),
            Self::Sorted(state) => state.accumulate_and_fetch(batch),
            Self::Aggregate(state) => state.accumulate_and_fetch(batch),
            Self::Limit(state) => state.accumulate_and_fetch(batch),
        }
    }

    pub fn mark_done_and_fetch(self) -> CallResult<Vec<GpuBatch>> {
        match self {
            Self::Coalesce(state) => state.mark_done_and_fetch(),
            Self::Sorted(state) => state.mark_done_and_fetch(),
            Self::Aggregate(state) => state.mark_done_and_fetch(),
            Self::Limit(state) => state.mark_done_and_fetch(),
        }
    }
}

/// The seq a call addresses, once the call is what this node's recipe should hold. A
/// recipe that names a different pattern or a different input is one this executor would
/// drive at the wrong moment, which is a wrong answer rather than a failed call.
fn addressed(call: &Call, when: CallPattern, input: Input) -> Result<(Seq, FbKind), PlanError> {
    if call.when != when || call.inputs.as_slice() != [input] {
        return Err(PlanError::Invalid(format!(
            "this node's call is {call:?} where the executor drives {when:?} over {input:?}"
        )));
    }
    call.target
        .ok_or_else(|| PlanError::Invalid(format!("{call:?} addresses no seq")))
}

fn shape(expected: &str, recipe: &Recipe) -> PlanError {
    PlanError::Invalid(format!("{expected}, and this one is `{recipe}`"))
}

/// The handles a call is about to consume, and the batches they came from dropped without
/// releasing: C++ erases each registry entry as it takes it.
fn hand_over(batches: Vec<GpuBatch>) -> Vec<u64> {
    batches.into_iter().map(|batch| batch.consume().1).collect()
}

/// A lane's batches concatenated into one at done, and nothing at all where none arrived.
///
/// The collapse arm does answer a call with no handles — but with a table of no columns,
/// which is not the batch of the node's schema a SingleBatch output owes downstream. So
/// this backend emits nothing and the empty batch is the driver's to supply, which is the
/// one place that knows the schema without asking the device.
pub struct Collapse {
    executor: *mut PeacockExecutor,
    collapse: (Seq, FbKind),
    held: Vec<GpuBatch>,
    schema: SchemaRef,
}

impl Collapse {
    fn accumulate_and_fetch(&mut self, batch: GpuBatch) -> CallResult<Vec<GpuBatch>> {
        self.held.push(batch);
        Ok((Vec::new(), CallStats::default()))
    }

    fn mark_done_and_fetch(self) -> CallResult<Vec<GpuBatch>> {
        if self.held.is_empty() {
            return Ok((Vec::new(), CallStats::default()));
        }
        let (seq, kind) = self.collapse;
        let handles = hand_over(self.held);
        let (handle, stats) = execute_node(self.executor, seq, kind, &[handles])?;
        Ok((
            vec![produced(self.executor, handle, stats, &self.schema)],
            CallStats::default(),
        ))
    }
}

/// Each batch sorted as it arrives, the runs merged into one at done — and nothing at all
/// where none arrived, since a merge of no runs is the collapse of nothing by another name
/// and the device refuses that (#173).
pub struct SortedRuns {
    executor: *mut PeacockExecutor,
    sort: (Seq, FbKind),
    merge: (Seq, FbKind),
    held: Vec<GpuBatch>,
    schema: SchemaRef,
}

impl SortedRuns {
    fn accumulate_and_fetch(&mut self, batch: GpuBatch) -> CallResult<Vec<GpuBatch>> {
        let (seq, kind) = self.sort;
        let (handle, stats) = execute_node(self.executor, seq, kind, &[hand_over(vec![batch])])?;
        self.held
            .push(produced(self.executor, handle, stats, &self.schema));
        Ok((Vec::new(), CallStats::default()))
    }

    fn mark_done_and_fetch(self) -> CallResult<Vec<GpuBatch>> {
        if self.held.is_empty() {
            return Ok((Vec::new(), CallStats::default()));
        }
        let (seq, kind) = self.merge;
        let handles = hand_over(self.held);
        let (handle, stats) = execute_node(self.executor, seq, kind, &[handles])?;
        Ok((
            vec![produced(self.executor, handle, stats, &self.schema)],
            CallStats::default(),
        ))
    }
}

/// Pre-aggregated state merged into pre-aggregated state, emitted at done.
///
/// The threshold doubles when a compaction fails to shrink what it folded — see the CPU
/// backend's copy of this rule, which is the same rule and the same reason.
pub struct AggregateBatches {
    executor: *mut PeacockExecutor,
    concat: (Seq, FbKind),
    merge: (Seq, FbKind),
    finalize: Option<(Seq, FbKind)>,
    state: Option<GpuBatch>,
    pending: Vec<GpuBatch>,
    threshold: usize,
    compactions: usize,
    held: SchemaRef,
    output: SchemaRef,
}

impl AggregateBatches {
    pub fn compactions(&self) -> usize {
        self.compactions
    }

    fn held_bytes(&self) -> usize {
        self.state.as_ref().map(Batch::byte_size).unwrap_or(0)
            + self.pending.iter().map(Batch::byte_size).sum::<usize>()
    }

    fn accumulate_and_fetch(&mut self, batch: GpuBatch) -> CallResult<Vec<GpuBatch>> {
        self.pending.push(batch);
        if self.held_bytes() >= self.threshold {
            self.compact()?;
        }
        Ok((Vec::new(), CallStats::default()))
    }

    fn compact(&mut self) -> Result<(), BackendError> {
        let mut folding = Vec::with_capacity(self.pending.len() + 1);
        folding.extend(self.state.take());
        folding.append(&mut self.pending);
        let (seq, kind) = self.concat;
        let (concatenated, _) = execute_node(self.executor, seq, kind, &[hand_over(folding)])?;
        let (seq, kind) = self.merge;
        let (handle, stats) = execute_node(self.executor, seq, kind, &[vec![concatenated]])?;
        let state = produced(self.executor, handle, stats, &self.held);
        self.threshold = self.threshold.max(2 * state.byte_size());
        self.state = Some(state);
        self.compactions += 1;
        Ok(())
    }

    fn mark_done_and_fetch(mut self) -> CallResult<Vec<GpuBatch>> {
        if !self.pending.is_empty() {
            self.compact()?;
        }
        let Some(state) = self.state.take() else {
            return Ok((Vec::new(), CallStats::default()));
        };
        let Some((seq, kind)) = self.finalize else {
            return Ok((vec![state], CallStats::default()));
        };
        let (handle, stats) = execute_node(self.executor, seq, kind, &[hand_over(vec![state])])?;
        Ok((
            vec![produced(self.executor, handle, stats, &self.output)],
            CallStats::default(),
        ))
    }
}

/// The one node of the partition-accumulator category: every lane's sorted run merged into
/// one at the last lane's done, and nothing where no lane sent anything — a merge of no
/// runs is the collapse of nothing under another name, and the device refuses that (#173).
///
/// One call per lane event, since that is what round-robin driving produces, and the call
/// carrying the last `Done` is the emitting one. The handles go into the merge in lane
/// order, which is what makes a tie partition-major rather than arrival-ordered.
pub struct GpuPartitionAccumulator {
    executor: *mut PeacockExecutor,
    merge: (Seq, FbKind),
    per_lane: Vec<Vec<GpuBatch>>,
    live: usize,
    schema: SchemaRef,
}

impl GpuPartitionAccumulator {
    pub fn merge_sorted(
        executor: *mut PeacockExecutor,
        recipe: &Recipe,
        lanes: usize,
        schema: &ArrowSchema,
    ) -> Result<Self, PlanError> {
        let [call] = recipe.calls.as_slice() else {
            return Err(shape(
                "a merge of sorted partitions makes one call, at done",
                recipe,
            ));
        };
        Ok(Self {
            executor,
            merge: addressed(call, CallPattern::AtDone, Input::AllLanes)?,
            per_lane: (0..lanes).map(|_| Vec::new()).collect(),
            live: lanes,
            schema: Arc::new(schema.clone()),
        })
    }

    pub fn accumulate_and_fetch(
        &mut self,
        partition: usize,
        event: LaneEvent<GpuBatch>,
    ) -> CallResult<Vec<GpuBatch>> {
        match event {
            LaneEvent::Batch(batch) => {
                self.per_lane[partition].push(batch);
                return Ok((Vec::new(), CallStats::default()));
            }
            LaneEvent::Done => self.live -= 1,
        }
        if self.live > 0 {
            return Ok((Vec::new(), CallStats::default()));
        }
        let (seq, kind) = self.merge;
        let held: Vec<GpuBatch> = std::mem::take(&mut self.per_lane)
            .into_iter()
            .flatten()
            .collect();
        if held.is_empty() {
            return Ok((Vec::new(), CallStats::default()));
        }
        let handles = hand_over(held);
        let (handle, stats) = execute_node(self.executor, seq, kind, &[handles])?;
        Ok((
            vec![produced(self.executor, handle, stats, &self.schema)],
            CallStats::default(),
        ))
    }
}

/// The mid-plan limit: it streams, holding nothing.
///
/// A batch entirely outside the interval is released where it stands, one entirely inside
/// is forwarded untouched, and only the two that straddle its ends are sliced. Its input
/// is one lane — the node checks that — so the count it keeps is the stream's.
pub struct LimitStream {
    executor: *mut PeacockExecutor,
    interval: RowInterval,
    seen: u64,
    schema: SchemaRef,
}

impl LimitStream {
    pub fn seen(&self) -> u64 {
        self.seen
    }

    fn accumulate_and_fetch(&mut self, batch: GpuBatch) -> CallResult<Vec<GpuBatch>> {
        let n_rows = batch.num_rows() as u64;
        let rows = self.interval.range_of(self.seen, n_rows);
        self.seen += n_rows;
        let Some(rows) = rows else {
            // Released here rather than shipped and trimmed, which is the whole saving on
            // an offset: the prefix a skip drops is unbounded.
            return Ok((Vec::new(), CallStats::default()));
        };
        if rows.covers(n_rows) {
            return Ok((vec![batch], CallStats::default()));
        }
        // The session is this executor's, as everywhere on this path: a batch carries the
        // pointer only so that dropping it can release its handle.
        let (_, handle) = batch.consume();
        let mut sliced = 0u64;
        let rc = unsafe {
            peacock_executor_slice_handle(
                self.executor,
                handle,
                rows.offset,
                rows.length,
                &mut sliced,
            )
        };
        if rc != 0 {
            return Err(BackendError::new(format!(
                "slice_handle({handle}, {}..+{}): {}",
                rows.offset,
                rows.length,
                last_error(self.executor)
            )));
        }
        // The slice reports no stats, so the batch is priced from the rows asked for: a
        // range that clamps is one this node computed against the batch it holds.
        let kept = GpuBatch::new(
            self.executor,
            sliced,
            rows.length as usize,
            logical_size_from_schema(&self.schema, rows.length as usize, 0),
        );
        Ok((vec![kept], CallStats::default()))
    }

    fn mark_done_and_fetch(self) -> CallResult<Vec<GpuBatch>> {
        Ok((Vec::new(), CallStats::default()))
    }
}
