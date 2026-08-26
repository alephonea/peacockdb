//! `CpuBackend`: the seven executor types this backend names, and the memory each reports.
//!
//! Every executor was written to the trait's shape before a `Backend` existed to check it,
//! so this file is where the compiler reads them. What it adds is `Executor` — the held
//! bytes and the pre-call transient the enforcer sums — which no earlier task could
//! answer, because nothing was accounting yet.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::execution::TaskContext;

use super::super::backend::{Backend, NodeExecutors};
use super::super::cpu_batch::CpuBatch;
use super::super::error::PlanError;
use super::super::executor::{
    BackendError, BatchAccumulatorExecutor, CallResult, ExecExecutor, Executor, JoinExecutor,
    LaneEvent, PartitionAccumulatorExecutor, PartitionEmitterExecutor, ProbingJoin, RowRange,
    SourceExecutor, SourceStep, UnloadExecutor,
};
use super::super::forwarder::forwarder_for;
use super::super::node::GpuNode;
use super::super::nodes::{NodeRef, as_node_ref};
use super::accumulate::{CpuAccumulator, CpuPartitionAccumulator};
use super::emit::CpuEmitter;
use super::join::{CpuJoin, CpuProbingJoin};
use super::source::CpuSource;
use super::{CpuExec, CpuUnload};

/// The threshold a batch aggregate compacts at, until the driver derives one from the
/// budget the way the loader's batch size is derived (#142).
const COMPACT_BYTES: usize = 1 << 20;

pub struct CpuBackend;

impl Backend for CpuBackend {
    type Context = Arc<TaskContext>;
    type Batch = CpuBatch;
    type Source = CpuSource;
    type Exec = CpuExec;
    type BatchAcc = CpuAccumulator;
    type PartAcc = CpuPartitionAccumulator;
    type Emitter = CpuEmitter;
    type Join = CpuJoin;
    type Unload = CpuUnload;

    fn executors_for(
        ctx: &Arc<TaskContext>,
        node: &dyn GpuNode,
        _post_order: usize,
        lane: usize,
    ) -> Result<NodeExecutors<Self>, PlanError> {
        // No recipe is read here: this backend asks DataFusion for the operator a node
        // describes, and the recipe is what the other one needs to address a seq.
        let input = |ordinal: usize| -> ArrowSchema {
            let child = node.children()[ordinal];
            child
                .kind()
                .schema()
                .expect("a node's input is not a sink")
                .fields
                .as_ref()
                .clone()
        };
        let lanes = || node.kind().layout().expect("an emitter is not a sink").n;
        Ok(match as_node_ref(node) {
            NodeRef::LoadParquet(load) => NodeExecutors::Source(CpuSource::new(
                load,
                lane,
                &node.kind().schema().expect("a source is not a sink").fields,
            )?),
            NodeRef::Filter(filter) => {
                NodeExecutors::Exec(CpuExec::filter(filter, &input(0), ctx.clone())?)
            }
            NodeRef::Project(project) => {
                NodeExecutors::Exec(CpuExec::project(project, &input(0), ctx.clone())?)
            }
            NodeRef::Sort(sort) => {
                NodeExecutors::Exec(CpuExec::sort(sort, &input(0), ctx.clone())?)
            }
            NodeRef::Aggregate(aggregate) => {
                NodeExecutors::Exec(CpuExec::aggregate(aggregate, &input(0), ctx.clone())?)
            }
            NodeRef::CoalesceAllBatches(coalesce) => {
                NodeExecutors::BatchAccumulator(CpuAccumulator::coalesce(coalesce, &input(0)))
            }
            NodeRef::AccumulateBatchesAndSort(sort) => NodeExecutors::BatchAccumulator(
                CpuAccumulator::sorted(sort, &input(0), ctx.clone())?,
            ),
            NodeRef::AggregateBatches(merge) => NodeExecutors::BatchAccumulator(
                CpuAccumulator::aggregate(merge, &input(0), ctx.clone(), COMPACT_BYTES)?,
            ),
            NodeRef::Limit(limit) => NodeExecutors::BatchAccumulator(CpuAccumulator::limit(limit)),
            NodeRef::MergeSortedPartitions(merge) => {
                let lanes = node.children()[0]
                    .kind()
                    .layout()
                    .expect("a merge's input is not a sink")
                    .n;
                NodeExecutors::PartitionAccumulator(CpuPartitionAccumulator::merge_sorted(
                    merge,
                    lanes,
                    &input(0),
                    ctx.clone(),
                )?)
            }
            NodeRef::EmitPartitions(emit) => {
                NodeExecutors::PartitionEmitter(CpuEmitter::new(emit, lanes(), &input(0))?)
            }
            NodeRef::Join(join) => {
                NodeExecutors::Join(CpuJoin::hash(join, &input(0), &input(1), ctx.clone())?)
            }
            NodeRef::CrossJoin(join) => {
                NodeExecutors::Join(CpuJoin::cross(join, &input(0), &input(1), ctx.clone())?)
            }
            NodeRef::NestedLoopJoin(join) => NodeExecutors::Join(CpuJoin::nested_loop(
                join,
                &input(0),
                &input(1),
                ctx.clone(),
            )?),
            NodeRef::Unload(_) => NodeExecutors::Unload(CpuUnload),
            NodeRef::MergePartitions(_) | NodeRef::Union(_) | NodeRef::Interleave(_) => {
                NodeExecutors::BatchForwarder(forwarder_for(node))
            }
        })
    }
}

fn bytes_of(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::get_array_memory_size).sum()
}

/// A source holds nothing between calls: the batch it reads is its output, and the reader
/// it opens lives inside one call.
impl Executor for CpuSource {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, _n_bytes: usize) -> usize {
        0
    }
}

impl SourceExecutor<CpuBackend> for CpuSource {
    fn next_batch(mut self) -> Result<SourceStep<CpuBackend>, BackendError> {
        match self.read_next()? {
            Some((batch, stats)) => Ok(SourceStep::Batch {
                batch,
                stats,
                source: self,
            }),
            None => Ok(SourceStep::Exhausted),
        }
    }
}

/// An exec node holds nothing between calls, and its transient is the answer it is about
/// to build out of the batch it was handed.
impl Executor for CpuExec {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        n_bytes
    }
}

impl ExecExecutor<CpuBackend> for CpuExec {
    fn exec(&mut self, batch: CpuBatch) -> CallResult<CpuBatch> {
        CpuExec::exec(self, batch)
    }
}

impl Executor for CpuAccumulator {
    fn resident_bytes(&self) -> usize {
        self.held_bytes()
    }
    /// What it holds plus what arrives: a compaction reads both at once, and a merge that
    /// is about to fold the two is the transient the enforcer has to have room for.
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        self.held_bytes() + n_bytes
    }
}

impl BatchAccumulatorExecutor<CpuBackend> for CpuAccumulator {
    fn accumulate_and_fetch(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        CpuAccumulator::accumulate_and_fetch(self, batch)
    }
    fn mark_done_and_fetch(self) -> CallResult<Vec<CpuBatch>> {
        CpuAccumulator::mark_done_and_fetch(self)
    }
}

impl Executor for CpuPartitionAccumulator {
    fn resident_bytes(&self) -> usize {
        self.held_bytes()
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        self.held_bytes() + n_bytes
    }
}

impl PartitionAccumulatorExecutor<CpuBackend> for CpuPartitionAccumulator {
    fn accumulate_and_fetch(
        &mut self,
        partition: usize,
        event: LaneEvent<CpuBatch>,
    ) -> CallResult<Vec<CpuBatch>> {
        CpuPartitionAccumulator::accumulate_and_fetch(self, partition, event)
    }
}

/// A scatter holds nothing: its N outputs are the batch it was handed, redistributed.
impl Executor for CpuEmitter {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        n_bytes
    }
}

impl PartitionEmitterExecutor<CpuBackend> for CpuEmitter {
    fn emit(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        CpuEmitter::emit(self, batch)
    }
}

/// Before its build side arrives a join holds only its operators.
impl Executor for CpuJoin {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        n_bytes
    }
}

impl JoinExecutor<CpuBackend> for CpuJoin {
    type Probing = CpuProbingJoin;
    fn set_build(self, batch: CpuBatch) -> CallResult<CpuProbingJoin> {
        CpuJoin::set_build(self, batch)
    }
}

impl Executor for CpuProbingJoin {
    /// The build side for as long as this join lives, and the probe keys a finishing type
    /// keeps until its finish pass runs.
    fn resident_bytes(&self) -> usize {
        self.build_bytes() + self.accumulated_bytes()
    }
    /// The build side only where a probe call reads it: the build-side semi family's probe
    /// call is the key project, and charging it the build side would refuse a query that
    /// fits, since the enforcer reads this before the call rather than after it.
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        if self.probe_reads_build() {
            self.build_bytes() + n_bytes
        } else {
            n_bytes
        }
    }
}

impl ProbingJoin<CpuBackend> for CpuProbingJoin {
    fn probe_and_fetch(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        CpuProbingJoin::probe_and_fetch(self, batch)
    }
    fn finish_and_fetch(self) -> CallResult<Vec<CpuBatch>> {
        CpuProbingJoin::finish_and_fetch(self)
    }
}

/// The unload holds nothing and its transient is the rows crossing out, which on this
/// backend is a slice of a batch already here.
impl Executor for CpuUnload {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, _n_bytes: usize) -> usize {
        0
    }
}

impl UnloadExecutor<CpuBackend> for CpuUnload {
    fn unload(&mut self, batch: CpuBatch, rows: RowRange) -> CallResult<CpuBatch> {
        CpuUnload::unload(self, batch, rows)
    }
}

/// Held bytes, which every accumulator answers and only the aggregate tracks incrementally.
trait HeldBytes {
    fn held_bytes(&self) -> usize;
}

impl HeldBytes for CpuAccumulator {
    fn held_bytes(&self) -> usize {
        match self {
            Self::Coalesce(state) => bytes_of(state.held()),
            Self::Sorted(state) => bytes_of(state.held()),
            Self::Aggregate(state) => state.held_bytes(),
            // A limit holds nothing, which is the whole point of the slice symbol.
            Self::Limit(_) => 0,
        }
    }
}

impl HeldBytes for CpuPartitionAccumulator {
    fn held_bytes(&self) -> usize {
        self.per_lane().map(bytes_of).sum()
    }
}
