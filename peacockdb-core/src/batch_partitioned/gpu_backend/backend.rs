//! `GpuBackend`: the seven executor types this backend names, and the memory each reports.
//!
//! Every executor here runs the recipe its node published, so the whole of building one is
//! finding that recipe — which is what the context carries beside the session pointer, and
//! why `executors_for` takes a post-order rather than deriving one.

use datafusion::arrow::datatypes::Schema as ArrowSchema;

use peacockdb_ffi::raw::PeacockExecutor;

use super::super::backend::{Backend, NodeExecutors};
use super::super::batch::Batch;
use super::super::cpu_batch::CpuBatch;
use super::super::error::PlanError;
use super::super::executor::{
    BackendError, BatchAccumulatorExecutor, CallResult, ExecExecutor, Executor, JoinExecutor,
    LaneEvent, PartitionAccumulatorExecutor, PartitionEmitterExecutor, ProbingJoin, RowRange,
    SourceExecutor, SourceStep, UnloadExecutor,
};
use super::super::forwarder::forwarder_for;
use super::super::gpu_batch::GpuBatch;
use super::super::node::GpuNode;
use super::super::nodes::join::per_call_join_type;
use super::super::nodes::{NodeRef, as_node_ref};
use super::super::recipe::RecipePlan;
use super::accumulate::{GpuAccumulator, GpuPartitionAccumulator};
use super::emit::GpuEmitter;
use super::join::{GpuJoin, GpuProbingJoin};
use super::source::GpuSource;
use super::{GpuExec, GpuExport};

/// The threshold a batch aggregate compacts at, until the driver derives one from the
/// budget the way the loader's batch size is derived (#142).
const COMPACT_BYTES: usize = 1 << 26;

/// What an executor on this backend is built from: the open session, and the recipes whose
/// seqs address the plan that session was given.
///
/// The pointer is BORROWED, as everywhere on this path — the session outlives every
/// executor drawn from it, and the handles they hand each other.
pub struct GpuContext {
    pub executor: *mut PeacockExecutor,
    pub recipes: RecipePlan,
}

pub struct GpuBackend;

impl Backend for GpuBackend {
    type Context = GpuContext;
    type Batch = GpuBatch;
    type Source = GpuSource;
    type Exec = GpuExec;
    type BatchAcc = GpuAccumulator;
    type PartAcc = GpuPartitionAccumulator;
    type Emitter = GpuEmitter;
    type Join = GpuJoin;
    type Unload = GpuExport;

    fn executors_for(
        ctx: &GpuContext,
        node: &dyn GpuNode,
        post_order: usize,
        lane: usize,
    ) -> Result<NodeExecutors<Self>, PlanError> {
        // Routing is the driver's, so a forwarder is answered before a recipe is looked
        // for: its absence is what a forwarder means rather than a gap to report.
        if let NodeRef::MergePartitions(_) | NodeRef::Union(_) | NodeRef::Interleave(_) =
            as_node_ref(node)
        {
            return Ok(NodeExecutors::BatchForwarder(forwarder_for(node)));
        }
        let recipe = ctx.recipes.get(post_order).ok_or_else(|| {
            PlanError::Invalid(format!(
                "{} is at {post_order} in the tree, where the recipes hold no entry — a node \
                 that makes ABI calls has one",
                node.name()
            ))
        })?;
        let executor = ctx.executor;
        let out = |node: &dyn GpuNode| -> ArrowSchema {
            node.kind()
                .schema()
                .expect("a node that is not a sink declares its columns")
                .fields
                .as_ref()
                .clone()
        };
        let input = |ordinal: usize| out(node.children()[ordinal]);
        Ok(match as_node_ref(node) {
            NodeRef::LoadParquet(load) => {
                NodeExecutors::Source(GpuSource::new(executor, recipe, load, lane, &out(node))?)
            }
            NodeRef::Filter(_) | NodeRef::Project(_) | NodeRef::Sort(_) | NodeRef::Aggregate(_) => {
                NodeExecutors::Exec(GpuExec::new(executor, recipe, &out(node))?)
            }
            NodeRef::CoalesceAllBatches(_) => NodeExecutors::BatchAccumulator(
                GpuAccumulator::coalesce(executor, recipe, &out(node))?,
            ),
            NodeRef::AccumulateBatchesAndSort(_) => NodeExecutors::BatchAccumulator(
                GpuAccumulator::sorted(executor, recipe, &out(node))?,
            ),
            NodeRef::AggregateBatches(merge) => {
                NodeExecutors::BatchAccumulator(GpuAccumulator::aggregate(
                    executor,
                    recipe,
                    &merge.intermediate().fields.as_ref().clone(),
                    &out(node),
                    COMPACT_BYTES,
                )?)
            }
            NodeRef::Limit(limit) => NodeExecutors::BatchAccumulator(GpuAccumulator::limit(
                executor,
                recipe,
                limit.interval,
                &out(node),
            )?),
            NodeRef::MergeSortedPartitions(_) => {
                let lanes = node.children()[0]
                    .kind()
                    .layout()
                    .expect("a merge's input is not a sink")
                    .n;
                NodeExecutors::PartitionAccumulator(GpuPartitionAccumulator::merge_sorted(
                    executor,
                    recipe,
                    lanes,
                    &out(node),
                )?)
            }
            NodeRef::EmitPartitions(_) => {
                NodeExecutors::PartitionEmitter(GpuEmitter::new(executor, recipe, &out(node))?)
            }
            NodeRef::Join(join) => {
                // A join that answers in one call has no finish pass, so what its per-call
                // type would be is a question with no answer — asking it is the defect, and
                // the `unreachable!` behind it is doing its job. The cpu backend guards on
                // the same capability at `cpu_backend/join.rs`, and one question answered
                // two ways is how the two backends drift.
                let one_call = join.capability()?.answers_in_one_call();
                // The key table this lane accumulates is the probe's key columns under the
                // probe's names, which is what the key project the recipe names emits.
                let keys = (!one_call && per_call_join_type(join.join_type).is_none())
                    .then(|| key_schema(&input(1), &join.keys));
                NodeExecutors::Join(GpuJoin::new(
                    executor,
                    recipe,
                    Some(join.join_type),
                    keys.as_ref(),
                    &out(node),
                )?)
            }
            NodeRef::CrossJoin(_) | NodeRef::NestedLoopJoin(_) => {
                // Neither has a join type of its own on the wire, and neither publishes a
                // finish, so there is no answer for one over no keys to be wrong about.
                NodeExecutors::Join(GpuJoin::new(executor, recipe, None, None, &out(node))?)
            }
            NodeRef::Unload(_) => {
                NodeExecutors::Unload(GpuExport::new(executor, &input(0), &recipe.exports))
            }
            NodeRef::MergePartitions(_) | NodeRef::Union(_) | NodeRef::Interleave(_) => {
                unreachable!("routing nodes are answered above")
            }
        })
    }
}

/// The probe's key columns in the order the join hashes them, keeping the probe's own
/// names — the shape `recipe::join::key_project` writes, and the schema its output is
/// priced by.
fn key_schema(probe: &ArrowSchema, keys: &[(u32, u32)]) -> ArrowSchema {
    ArrowSchema::new(
        keys.iter()
            .map(|(_, probe_ordinal)| probe.field(*probe_ordinal as usize).clone())
            .collect::<Vec<_>>(),
    )
}

/// A source holds nothing between calls: the handle a read produces is its output, and
/// the row groups it still owes are a list, not a table.
impl Executor for GpuSource {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, _n_bytes: usize) -> usize {
        0
    }
}

impl SourceExecutor<GpuBackend> for GpuSource {
    fn next_batch(mut self) -> Result<SourceStep<GpuBackend>, BackendError> {
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

/// An exec node holds nothing between calls; its transient is the table the call is about
/// to build, which is bounded by the batch it reads.
impl Executor for GpuExec {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        n_bytes
    }
}

impl ExecExecutor<GpuBackend> for GpuExec {
    fn exec(&mut self, batch: GpuBatch) -> CallResult<GpuBatch> {
        GpuExec::exec(self, batch)
    }
}

impl Executor for GpuAccumulator {
    fn resident_bytes(&self) -> usize {
        self.held_bytes()
    }
    /// What it holds plus what arrives: a compaction reads both at once, and the merge
    /// about to fold them is the transient the enforcer has to have room for.
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        self.held_bytes() + n_bytes
    }
}

impl BatchAccumulatorExecutor<GpuBackend> for GpuAccumulator {
    fn accumulate_and_fetch(&mut self, batch: GpuBatch) -> CallResult<Vec<GpuBatch>> {
        GpuAccumulator::accumulate_and_fetch(self, batch)
    }
    fn mark_done_and_fetch(self) -> CallResult<Vec<GpuBatch>> {
        GpuAccumulator::mark_done_and_fetch(self)
    }
}

impl Executor for GpuPartitionAccumulator {
    fn resident_bytes(&self) -> usize {
        self.held_bytes()
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        self.held_bytes() + n_bytes
    }
}

impl PartitionAccumulatorExecutor<GpuBackend> for GpuPartitionAccumulator {
    fn accumulate_and_fetch(
        &mut self,
        partition: usize,
        event: LaneEvent<GpuBatch>,
    ) -> CallResult<Vec<GpuBatch>> {
        GpuPartitionAccumulator::accumulate_and_fetch(self, partition, event)
    }
}

/// A scatter holds nothing: its N outputs are the batch it was handed, redistributed.
impl Executor for GpuEmitter {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        n_bytes
    }
}

impl PartitionEmitterExecutor<GpuBackend> for GpuEmitter {
    fn emit(&mut self, batch: GpuBatch) -> CallResult<Vec<GpuBatch>> {
        GpuEmitter::emit(self, batch)
    }
}

/// Before its build side arrives a join holds only the calls it will make.
impl Executor for GpuJoin {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        n_bytes
    }
}

impl JoinExecutor<GpuBackend> for GpuJoin {
    type Probing = GpuProbingJoin;
    fn set_build(self, batch: GpuBatch) -> CallResult<GpuProbingJoin> {
        GpuJoin::set_build(self, batch)
    }
    fn without_build(self) -> Result<(), BackendError> {
        GpuJoin::without_build(self)
    }
}

impl Executor for GpuProbingJoin {
    /// The build side for as long as this join holds it, and the probe keys a finishing
    /// type keeps until its finish pass runs.
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

impl ProbingJoin<GpuBackend> for GpuProbingJoin {
    fn probe_and_fetch(&mut self, batch: GpuBatch) -> CallResult<Vec<GpuBatch>> {
        GpuProbingJoin::probe_and_fetch(self, batch)
    }
    fn finish_and_fetch(self) -> CallResult<Vec<GpuBatch>> {
        GpuProbingJoin::finish_and_fetch(self)
    }
}

/// The export holds nothing and its transient is the rows crossing out — the IPC buffer
/// the range names, which is why a limit's range is worth having.
impl Executor for GpuExport {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        n_bytes
    }
}

impl UnloadExecutor<GpuBackend> for GpuExport {
    fn unload(&mut self, batch: GpuBatch, rows: RowRange) -> CallResult<CpuBatch> {
        GpuExport::unload(self, batch, rows)
    }
}

/// Held bytes, which every accumulator answers off the handles it is keeping.
trait HeldBytes {
    fn held_bytes(&self) -> usize;
}

impl HeldBytes for GpuAccumulator {
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

impl HeldBytes for GpuPartitionAccumulator {
    fn held_bytes(&self) -> usize {
        self.per_lane().map(bytes_of).sum()
    }
}

fn bytes_of(batches: &[GpuBatch]) -> usize {
    batches.iter().map(GpuBatch::byte_size).sum()
}
