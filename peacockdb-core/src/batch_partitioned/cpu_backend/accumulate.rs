//! The accumulators: the nodes that hold a lane's batches back, and the mid-plan limit,
//! which is one by category and holds nothing at all.
//!
//! One type with a variant per behaviour, because a backend names one type for the
//! category and what these four do differs completely. The variant is what a call site
//! reads: a coalesce that concatenates, a sort that orders the whole stream, a merge that
//! folds state on a threshold, and a limit that forwards, slices or drops.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;

use super::super::cpu_batch::CpuBatch;
use super::super::error::PlanError;
use super::super::executor::{BackendError, CallResult, CallStats, LaneEvent};
use super::super::expr_physical::physical_projection;
use super::super::node::{GpuNode, RowInterval};
use super::super::nodes::aggregate::{Phase, finalize_columns};
use super::super::nodes::{
    GpuAccumulateBatchesAndSort, GpuAggregateBatches, GpuCoalesceAllBatches, GpuLimit,
    GpuMergeSortedPartitions,
};
use super::{aggregate_exec, declared_as, lex_ordering, placeholder, run_node};

/// A `BatchAccumulator` node's executor, one variant per node.
pub enum CpuAccumulator {
    Coalesce(Coalesce),
    Sorted(SortedRuns),
    Aggregate(AggregateBatches),
    Limit(LimitStream),
}

impl CpuAccumulator {
    pub fn coalesce(_node: &GpuCoalesceAllBatches, input: &ArrowSchema) -> Self {
        Self::Coalesce(Coalesce {
            held: Vec::new(),
            schema: Arc::new(input.clone()),
        })
    }

    pub fn sorted(
        node: &GpuAccumulateBatchesAndSort,
        input: &ArrowSchema,
        ctx: Arc<TaskContext>,
    ) -> Result<Self, PlanError> {
        let ordering = lex_ordering(&node.keys, input)?;
        Ok(Self::Sorted(SortedRuns {
            held: Vec::new(),
            sort: Arc::new(SortExec::new(ordering, placeholder(input))),
            fetch: node.fetch,
            schema: Arc::new(input.clone()),
            ctx,
        }))
    }

    /// `compact_bytes` is the held-state size at which a compaction runs. It has no
    /// default here: the number comes from the same budget rule that sizes loader batches,
    /// which is the driver's (T17), and a stand-in default would be a policy nobody chose.
    pub fn aggregate(
        node: &GpuAggregateBatches,
        input: &ArrowSchema,
        ctx: Arc<TaskContext>,
        compact_bytes: usize,
    ) -> Result<Self, PlanError> {
        let body = &node.body;
        let state = node.intermediate();
        let merge = aggregate_exec(body, Phase::Merge, input, state, ctx.as_ref())?;
        let output = node.kind().schema().expect("an aggregate is not a sink");
        let finalize = match body.finalize {
            Some(_) => {
                let columns = finalize_columns(body, state, output)?;
                let exprs = physical_projection(&columns, &state.fields, ctx.as_ref())?;
                Some(Arc::new(
                    ProjectionExec::try_new(exprs, placeholder(&state.fields)).map_err(
                        |error| PlanError::Invalid(format!("the finalize project: {error}")),
                    )?,
                ) as Arc<dyn ExecutionPlan>)
            }
            None => None,
        };
        let ctx = super::always_aggregating(ctx);
        Ok(Self::Aggregate(AggregateBatches {
            merge,
            finalize,
            state: None,
            pending: Vec::new(),
            pending_bytes: 0,
            threshold: compact_bytes,
            compactions: 0,
            grouped: !body.group_by.is_empty(),
            held: state.fields.clone(),
            output: output.fields.clone(),
            ctx,
        }))
    }

    pub fn limit(node: &GpuLimit) -> Self {
        Self::Limit(LimitStream {
            interval: node.interval,
            seen: 0,
        })
    }

    pub fn accumulate_and_fetch(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        match self {
            Self::Coalesce(state) => state.accumulate_and_fetch(batch),
            Self::Sorted(state) => state.accumulate_and_fetch(batch),
            Self::Aggregate(state) => state.accumulate_and_fetch(batch),
            Self::Limit(state) => state.accumulate_and_fetch(batch),
        }
    }

    pub fn mark_done_and_fetch(self) -> CallResult<Vec<CpuBatch>> {
        match self {
            Self::Coalesce(state) => state.mark_done_and_fetch(),
            Self::Sorted(state) => state.mark_done_and_fetch(),
            Self::Aggregate(state) => state.mark_done_and_fetch(),
            Self::Limit(state) => state.mark_done_and_fetch(),
        }
    }
}

/// What an accumulator answers with at done: one batch of everything it held, and nothing
/// at all where nothing arrived.
///
/// Both backends emit nothing for an empty lane, which is what makes them one engine here:
/// the device's collapse of no handles is a refusal ([#173](../../../llm-wiki/tickets.md)),
/// so a batch invented on this side would be a row the other cannot produce. A grouped
/// merge over no arrivals owes no groups, so nothing is also its answer. The exception is
/// a global aggregate, which owes its identity row whatever arrived — see
/// [`AggregateBatches::mark_done_and_fetch`].
fn one_batch(schema: &SchemaRef, held: &[RecordBatch]) -> CallResult<Vec<CpuBatch>> {
    if held.is_empty() {
        return Ok((Vec::new(), CallStats::default()));
    }
    let batch = concat_batches(schema, held.iter())
        .map_err(|error| BackendError::new(format!("joining the lane's batches: {error}")))?;
    Ok((vec![CpuBatch::new(batch)], CallStats::default()))
}

/// A lane's batches concatenated into one at done.
pub struct Coalesce {
    held: Vec<RecordBatch>,
    schema: SchemaRef,
}

impl Coalesce {
    pub fn held(&self) -> &[RecordBatch] {
        &self.held
    }

    fn accumulate_and_fetch(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        self.held.push(batch.into_record_batch());
        Ok((Vec::new(), CallStats::default()))
    }

    fn mark_done_and_fetch(self) -> CallResult<Vec<CpuBatch>> {
        one_batch(&self.schema, &self.held)
    }
}

/// A lane's batches ordered into one at done.
///
/// The device sorts each batch and merges the runs, which is two calls; here it is one sort
/// over the concatenation, and the same answer — a stable sort of the runs in arrival order
/// puts tied rows exactly where merging them would.
///
/// The fetch is applied by slicing rather than by `SortExec::with_fetch`, and that is the
/// point: the top-N path keeps a bounded heap and does not preserve arrival order among
/// ties, so which of two tied rows a limit kept would depend on the heap. An oracle cannot
/// answer that differently from run to run.
pub struct SortedRuns {
    held: Vec<RecordBatch>,
    sort: Arc<dyn ExecutionPlan>,
    fetch: Option<usize>,
    schema: SchemaRef,
    ctx: Arc<TaskContext>,
}

impl SortedRuns {
    pub fn held(&self) -> &[RecordBatch] {
        &self.held
    }

    fn accumulate_and_fetch(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        self.held.push(batch.into_record_batch());
        Ok((Vec::new(), CallStats::default()))
    }

    fn mark_done_and_fetch(self) -> CallResult<Vec<CpuBatch>> {
        if self.held.is_empty() {
            return Ok((Vec::new(), CallStats::default()));
        }
        let sorted = run_node(&self.sort, vec![self.held], &self.ctx)?;
        let (ordered, _) = one_batch(&self.schema, &sorted)?;
        Ok((
            ordered
                .into_iter()
                .map(|batch| first_rows(batch, self.fetch))
                .collect(),
            CallStats::default(),
        ))
    }
}

/// The one node of the partition-accumulator category: every lane's sorted stream merged
/// into one at the last lane's done.
///
/// It takes one call per lane event because that is what round-robin driving produces, and
/// the call carrying the last `Done` is the emitting one. Ties are broken partition-major
/// — lane order, then arrival order inside a lane — which is what concatenating in lane
/// order and sorting stably gives, and what a k-way merge over the same runs gives.
pub struct CpuPartitionAccumulator {
    per_lane: Vec<Vec<RecordBatch>>,
    live: usize,
    sort: Arc<dyn ExecutionPlan>,
    fetch: Option<usize>,
    schema: SchemaRef,
    ctx: Arc<TaskContext>,
}

impl CpuPartitionAccumulator {
    pub fn merge_sorted(
        node: &GpuMergeSortedPartitions,
        lanes: usize,
        input: &ArrowSchema,
        ctx: Arc<TaskContext>,
    ) -> Result<Self, PlanError> {
        let ordering = lex_ordering(&node.keys, input)?;
        Ok(Self {
            per_lane: vec![Vec::new(); lanes],
            live: lanes,
            sort: Arc::new(SortExec::new(ordering, placeholder(input))),
            fetch: node.fetch,
            schema: Arc::new(input.clone()),
            ctx,
        })
    }

    /// What each lane is holding, for the accounting the driver sums.
    pub fn per_lane(&self) -> impl Iterator<Item = &[RecordBatch]> {
        self.per_lane.iter().map(|lane| lane.as_slice())
    }

    pub fn accumulate_and_fetch(
        &mut self,
        partition: usize,
        event: LaneEvent<CpuBatch>,
    ) -> CallResult<Vec<CpuBatch>> {
        match event {
            LaneEvent::Batch(batch) => {
                self.per_lane[partition].push(batch.into_record_batch());
                return Ok((Vec::new(), CallStats::default()));
            }
            LaneEvent::Done => self.live -= 1,
        }
        if self.live > 0 {
            return Ok((Vec::new(), CallStats::default()));
        }
        let partition_major: Vec<RecordBatch> = std::mem::take(&mut self.per_lane)
            .into_iter()
            .flatten()
            .collect();
        if partition_major.is_empty() {
            return Ok((Vec::new(), CallStats::default()));
        }
        let sorted = run_node(&self.sort, vec![partition_major], &self.ctx)?;
        let (ordered, _) = one_batch(&self.schema, &sorted)?;
        Ok((
            ordered
                .into_iter()
                .map(|batch| first_rows(batch, self.fetch))
                .collect(),
            CallStats::default(),
        ))
    }
}

/// The fetch as a slice of an ordered batch — see [`SortedRuns`] for why it is not
/// DataFusion's top-N.
fn first_rows(batch: CpuBatch, fetch: Option<usize>) -> CpuBatch {
    let batch = batch.into_record_batch();
    match fetch {
        Some(fetch) if fetch < batch.num_rows() => CpuBatch::new(batch.slice(0, fetch)),
        _ => CpuBatch::new(batch),
    }
}

/// Pre-aggregated state merged into pre-aggregated state, emitted at done.
///
/// Compaction runs on a threshold that doubles when it fails to pay. Compacting on every
/// arrival re-scans the whole state once per batch, which is quadratic where the groups
/// are disjoint and nothing merges; never compacting holds the whole input where the
/// cardinality is high. So arrivals are held until they cross the threshold, folded once,
/// and the threshold set to twice what that fold left behind.
pub struct AggregateBatches {
    merge: Arc<dyn ExecutionPlan>,
    /// The finalize project, where this node finishes the aggregate. Its presence is the
    /// only thing that distinguishes a merging node from a finalizing one.
    finalize: Option<Arc<dyn ExecutionPlan>>,
    state: Option<RecordBatch>,
    pending: Vec<RecordBatch>,
    pending_bytes: usize,
    threshold: usize,
    /// How many times the threshold was reached. Read by the tests, which are what pins
    /// the doubling: without it a disjoint-key aggregate compacts once per arrival.
    compactions: usize,
    /// Whether this aggregate has group keys. A global one owes a row even for an empty
    /// lane; a grouped one owes no groups.
    grouped: bool,
    held: SchemaRef,
    output: SchemaRef,
    ctx: Arc<TaskContext>,
}

impl AggregateBatches {
    pub fn compactions(&self) -> usize {
        self.compactions
    }

    pub fn held_bytes(&self) -> usize {
        self.state
            .as_ref()
            .map(RecordBatch::get_array_memory_size)
            .unwrap_or(0)
            + self.pending_bytes
    }

    fn accumulate_and_fetch(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        let batch = batch.into_record_batch();
        self.pending_bytes += batch.get_array_memory_size();
        self.pending.push(batch);
        if self.held_bytes() >= self.threshold {
            self.compact()?;
        }
        Ok((Vec::new(), CallStats::default()))
    }

    /// Fold every pending arrival into the state, and raise the bar. A compaction that did
    /// not shrink will not shrink next time either — the keys are simply distinct — so the
    /// threshold moves to twice what is now held rather than re-scanning it per arrival.
    fn compact(&mut self) -> Result<(), BackendError> {
        let mut inputs = Vec::with_capacity(self.pending.len() + 1);
        inputs.extend(self.state.take());
        inputs.append(&mut self.pending);
        self.pending_bytes = 0;
        let merged = run_node(&self.merge, vec![inputs], &self.ctx)?;
        let merged = declared(merged, &self.held)?;
        let state = concat_batches(&self.held, merged.iter())
            .map_err(|error| BackendError::new(format!("joining the merged state: {error}")))?;
        self.threshold = self.threshold.max(2 * state.get_array_memory_size());
        self.state = Some(state);
        self.compactions += 1;
        Ok(())
    }

    /// A lane that received nothing usually owes nothing — but a global aggregate owes its
    /// identity row, `count 0` rather than no row, and a mid-plan limit that dropped every
    /// batch of its one lane is how that lane comes to be empty. So the merge runs over no
    /// input rather than being skipped, and the operator answers: one row where there are
    /// no group keys, none where there are.
    fn mark_done_and_fetch(mut self) -> CallResult<Vec<CpuBatch>> {
        if !self.pending.is_empty() || (self.state.is_none() && !self.grouped) {
            self.compact()?;
        }
        let Some(state) = self.state else {
            return one_batch(&self.output, &[]);
        };
        let Some(finalize) = self.finalize else {
            return one_batch(&self.held, &[state]);
        };
        let finalized = run_node(&finalize, vec![vec![state]], &self.ctx)?;
        one_batch(&self.output, &declared(finalized, &self.output)?)
    }
}

/// The mid-plan limit: it streams, holding nothing.
///
/// Per batch, from a running count of the rows that have gone past: one entirely outside
/// the interval is dropped, one entirely inside is forwarded untouched, and only the two
/// that straddle its ends are sliced. Its input is one lane — the node checks that — so
/// the count it keeps is the count of the stream.
pub struct LimitStream {
    interval: RowInterval,
    seen: u64,
}

impl LimitStream {
    /// How many rows have gone past, which is what a driver reads to know the interval is
    /// satisfied and the pulls can stop.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    fn accumulate_and_fetch(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        let batch = batch.into_record_batch();
        let n_rows = batch.num_rows() as u64;
        let rows = self.interval.range_of(self.seen, n_rows);
        self.seen += n_rows;
        let Some(rows) = rows else {
            return Ok((Vec::new(), CallStats::default()));
        };
        let kept = if rows.covers(n_rows) {
            batch
        } else {
            batch.slice(rows.offset as usize, rows.length as usize)
        };
        Ok((vec![CpuBatch::new(kept)], CallStats::default()))
    }

    fn mark_done_and_fetch(self) -> CallResult<Vec<CpuBatch>> {
        Ok((Vec::new(), CallStats::default()))
    }
}

fn declared(
    batches: Vec<RecordBatch>,
    schema: &SchemaRef,
) -> Result<Vec<RecordBatch>, BackendError> {
    batches
        .into_iter()
        .map(|batch| declared_as(batch, schema))
        .collect()
}
