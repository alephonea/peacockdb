//! Layout injection: rewriting a planned tree into shapes no planner would emit, and
//! demanding the same rows out of every one of them.
//!
//! Two rewrites and two wraps. A rebatcher and a drained lane change the tree — one
//! inserts a node, the other moves a lane's row groups to its neighbour — and are built on
//! [`rebuild`](super::rebuild::rebuild). Zero-row batches and a degenerate hash are
//! behaviour no node carries, so they wrap the source and emit executors of a backend
//! otherwise the CPU one. Lane counts are deliberately absent: `target_partitions` varies
//! them through the planner already.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::execution::TaskContext;
use datafusion::execution::context::SessionContext;

use peacockdb_core::batch_partitioned::GpuNode;
use peacockdb_core::batch_partitioned::backend::{Backend, NodeExecutors};
use peacockdb_core::batch_partitioned::cpu_backend::accumulate::{
    CpuAccumulator, CpuPartitionAccumulator,
};
use peacockdb_core::batch_partitioned::cpu_backend::backend::CpuBackend;
use peacockdb_core::batch_partitioned::cpu_backend::emit::CpuEmitter;
use peacockdb_core::batch_partitioned::cpu_backend::join::{CpuJoin, CpuProbingJoin};
use peacockdb_core::batch_partitioned::cpu_backend::source::CpuSource;
use peacockdb_core::batch_partitioned::cpu_backend::{CpuExec, CpuUnload};
use peacockdb_core::batch_partitioned::cpu_batch::CpuBatch;
use peacockdb_core::batch_partitioned::error::PlanError;
use peacockdb_core::batch_partitioned::executor::{
    BackendError, BatchAccumulatorExecutor, CallResult, CallStats, ExecExecutor, Executor,
    JoinExecutor, LaneEvent, PartitionAccumulatorExecutor, PartitionEmitterExecutor, ProbingJoin,
    RowRange, SourceExecutor, SourceStep, UnloadExecutor,
};
use peacockdb_core::batch_partitioned::nodes::join::empty_build_answers_nothing;
use peacockdb_core::batch_partitioned::nodes::{
    GpuCoalesceAllBatches, GpuEmitPartitions, GpuLoadParquet, GpuMergeSortedPartitions, GpuUnload,
    NodeRef, as_node_ref,
};
use peacockdb_core::batch_partitioned::schema::Schema;

use super::rebuild::{key, lanes_of, rebuild, scan_of, schema_of, sorted, source};

/// At most this many injected runs per query: 30 is 212 s for the eleven queries serially,
/// and the cover the selector guarantees is 13 of them.
pub const CAP: usize = 30;

/// One seed for the tier: which candidates are chosen and which calls emit an empty batch
/// are both functions of it, so a failure reproduces.
pub const SEED: u64 = 17;

// ── the decorator ──────────────────────────────────────────────────────────

/// Whether a source emits a zero-row batch instead of advancing. Never twice in a row:
/// the driver pulls until a source is exhausted, and a source that can always answer
/// without advancing never is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Empties {
    Never,
    /// Percent of calls, decided from the seed and the source's address rather than from
    /// a shared generator — two runs of one setting make the same calls.
    Sometimes(u32),
}

impl Empties {
    fn fires(self, stamp: u64, call: u32) -> bool {
        match self {
            Self::Never => false,
            Self::Sometimes(percent) => mix(stamp, u64::from(call)) % 100 < u64::from(percent),
        }
    }
}

/// Where the emitter puts a row. Both are legal: a shuffle's contract is co-location, and
/// every key into one lane satisfies it — nothing above a scatter may depend on how evenly
/// the lanes were loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Hash {
    AsPlanned,
    /// Every key into lane 0. Applied at every emitter, so both sides of a join still meet.
    Degenerate,
}

/// The CPU backend with two of its executors wrapped. Everything else is the CPU type
/// itself, reached through a second impl of its trait rather than through a wrapper that
/// would only forward.
pub struct Injected;

pub struct InjectedContext {
    pub task: Arc<TaskContext>,
    pub empties: Empties,
    pub hash: Hash,
    pub seed: u64,
    /// What the sources actually did, counted where it happens. A seed under which no call
    /// fires is possible, the answer is unchanged either way, and the run would pass having
    /// injected nothing — so the wrapper counts and the caller asserts.
    emitted: Arc<AtomicUsize>,
}

impl InjectedContext {
    pub fn new(task: Arc<TaskContext>, injection: Injection, seed: u64) -> Self {
        Self {
            task,
            empties: injection.empties,
            hash: injection.hash,
            seed,
            emitted: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Zero-row batches emitted instead of advancing, over every source and lane.
    pub fn empty_batches(&self) -> usize {
        self.emitted.load(Ordering::Relaxed)
    }
}

impl Backend for Injected {
    type Context = InjectedContext;
    type Batch = CpuBatch;
    type Source = InjectedSource;
    type Exec = CpuExec;
    type BatchAcc = CpuAccumulator;
    type PartAcc = CpuPartitionAccumulator;
    type Emitter = InjectedEmitter;
    type Join = CpuJoin;
    type Unload = CpuUnload;

    fn executors_for(
        ctx: &InjectedContext,
        node: &dyn GpuNode,
        post_order: usize,
        lane: usize,
    ) -> Result<NodeExecutors<Self>, PlanError> {
        Ok(
            match CpuBackend::executors_for(&ctx.task, node, post_order, lane)? {
                NodeExecutors::Source(inner) => NodeExecutors::Source(InjectedSource {
                    inner,
                    schema: node
                        .kind()
                        .schema()
                        .expect("a source is not a sink")
                        .fields
                        .clone(),
                    empties: ctx.empties,
                    emitted: ctx.emitted.clone(),
                    stamp: mix(mix(ctx.seed, post_order as u64), lane as u64),
                    call: 0,
                    was_empty: false,
                }),
                NodeExecutors::PartitionEmitter(inner) => {
                    NodeExecutors::PartitionEmitter(InjectedEmitter {
                        inner,
                        lanes: lanes_of(node),
                        hash: ctx.hash,
                    })
                }
                NodeExecutors::Exec(exec) => NodeExecutors::Exec(exec),
                NodeExecutors::BatchAccumulator(acc) => NodeExecutors::BatchAccumulator(acc),
                NodeExecutors::PartitionAccumulator(acc) => {
                    NodeExecutors::PartitionAccumulator(acc)
                }
                NodeExecutors::Join(join) => NodeExecutors::Join(join),
                NodeExecutors::Unload(unload) => NodeExecutors::Unload(unload),
                NodeExecutors::BatchForwarder(forwarder) => {
                    NodeExecutors::BatchForwarder(forwarder)
                }
            },
        )
    }
}

pub struct InjectedSource {
    inner: CpuSource,
    /// An empty batch still declares the columns its consumers read.
    schema: Arc<ArrowSchema>,
    empties: Empties,
    emitted: Arc<AtomicUsize>,
    stamp: u64,
    call: u32,
    was_empty: bool,
}

impl Executor for InjectedSource {
    fn resident_bytes(&self) -> usize {
        self.inner.resident_bytes()
    }

    fn scratch_bytes(&self, n_rows: u64, n_bytes: usize) -> usize {
        self.inner.scratch_bytes(n_rows, n_bytes)
    }
}

impl SourceExecutor<Injected> for InjectedSource {
    fn next_batch(self) -> Result<SourceStep<Injected>, BackendError> {
        let Self {
            inner,
            schema,
            empties,
            emitted,
            stamp,
            call,
            was_empty,
        } = self;
        let again = |inner, was_empty| Self {
            inner,
            schema: schema.clone(),
            empties,
            emitted: emitted.clone(),
            stamp,
            call: call + 1,
            was_empty,
        };
        if !was_empty && empties.fires(stamp, call) {
            emitted.fetch_add(1, Ordering::Relaxed);
            // The stats a real read reports: `CpuSource` measures nothing either, so an
            // injected call gives the accountant what an uninjected one does.
            return Ok(SourceStep::Batch {
                batch: CpuBatch::new(RecordBatch::new_empty(schema.clone())),
                stats: CallStats::default(),
                source: again(inner, true),
            });
        }
        Ok(match SourceExecutor::<CpuBackend>::next_batch(inner)? {
            SourceStep::Batch {
                batch,
                stats,
                source,
            } => SourceStep::Batch {
                batch,
                stats,
                source: again(source, false),
            },
            SourceStep::Exhausted => SourceStep::Exhausted,
        })
    }
}

pub struct InjectedEmitter {
    inner: CpuEmitter,
    lanes: usize,
    hash: Hash,
}

impl Executor for InjectedEmitter {
    fn resident_bytes(&self) -> usize {
        self.inner.resident_bytes()
    }

    fn scratch_bytes(&self, n_rows: u64, n_bytes: usize) -> usize {
        self.inner.scratch_bytes(n_rows, n_bytes)
    }
}

impl PartitionEmitterExecutor<Injected> for InjectedEmitter {
    fn emit(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        if self.hash == Hash::AsPlanned {
            return PartitionEmitterExecutor::<CpuBackend>::emit(&mut self.inner, batch);
        }
        // Exactly N in lane order is the emitter's contract: lane 0 takes the batch, the
        // rest are the empties a spreading hash would have produced where it sent none.
        // The inner emitter is not called and the figures still hold — a `CpuEmitter`
        // holds no state, reporting 0 resident and its input's bytes as the transient.
        let rows = batch.into_record_batch();
        let schema = rows.schema();
        let mut lanes = Vec::with_capacity(self.lanes);
        lanes.push(CpuBatch::new(rows));
        lanes
            .extend((1..self.lanes).map(|_| CpuBatch::new(RecordBatch::new_empty(schema.clone()))));
        Ok((lanes, CallStats::default()))
    }
}

// The five categories nothing here changes, reached through a second impl of each trait
// rather than through a wrapper that would only forward. The batch type is the same, so
// each body is the CPU one.
impl ExecExecutor<Injected> for CpuExec {
    fn exec(&mut self, batch: CpuBatch) -> CallResult<CpuBatch> {
        ExecExecutor::<CpuBackend>::exec(self, batch)
    }
}

impl BatchAccumulatorExecutor<Injected> for CpuAccumulator {
    fn accumulate_and_fetch(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        BatchAccumulatorExecutor::<CpuBackend>::accumulate_and_fetch(self, batch)
    }

    fn mark_done_and_fetch(self) -> CallResult<Vec<CpuBatch>> {
        BatchAccumulatorExecutor::<CpuBackend>::mark_done_and_fetch(self)
    }
}

impl PartitionAccumulatorExecutor<Injected> for CpuPartitionAccumulator {
    fn accumulate_and_fetch(
        &mut self,
        partition: usize,
        event: LaneEvent<CpuBatch>,
    ) -> CallResult<Vec<CpuBatch>> {
        PartitionAccumulatorExecutor::<CpuBackend>::accumulate_and_fetch(self, partition, event)
    }
}

impl JoinExecutor<Injected> for CpuJoin {
    type Probing = CpuProbingJoin;

    fn set_build(self, batch: CpuBatch) -> CallResult<CpuProbingJoin> {
        JoinExecutor::<CpuBackend>::set_build(self, batch)
    }

    fn without_build(self) -> Result<(), BackendError> {
        JoinExecutor::<CpuBackend>::without_build(self)
    }
}

impl ProbingJoin<Injected> for CpuProbingJoin {
    fn probe_and_fetch(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        ProbingJoin::<CpuBackend>::probe_and_fetch(self, batch)
    }

    fn finish_and_fetch(self) -> CallResult<Vec<CpuBatch>> {
        ProbingJoin::<CpuBackend>::finish_and_fetch(self)
    }
}

impl UnloadExecutor<Injected> for CpuUnload {
    fn unload(&mut self, batch: CpuBatch, rows: RowRange) -> CallResult<CpuBatch> {
        UnloadExecutor::<CpuBackend>::unload(self, batch, rows)
    }
}

/// A hash, so that what a run does is a function of its seed and the address of the call
/// rather than of how many calls came before it anywhere else.
fn mix(seed: u64, value: u64) -> u64 {
    let mut hash = seed ^ 0x9e37_79b9_7f4a_7c15;
    for byte in value.to_le_bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

// ── the rewrites, and the settings that name them ──────────────────────────

/// Where a rebatcher goes. One direction only: nothing below the loader splits a batch
/// ([#142](../../../llm-wiki/tickets.md#t142)), so the node is `GpuCoalesceAllBatches`
/// merging a lane to one, and the finer direction is the mode axis already.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rebatch {
    None,
    AboveSources,
    AboveInterior,
}

/// Whether a lane is left live and empty. The mapping moves rather than disappears — a
/// lane that produced nothing because its row groups went to its neighbour is a drained
/// lane, and one whose rows were dropped is a wrong answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Drain {
    None,
    FirstLane,
}

/// One injected shape: two rewrites of the tree and two wraps of the calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Injection {
    pub rebatch: Rebatch,
    pub drain: Drain,
    pub empties: Empties,
    pub hash: Hash,
}

impl Injection {
    /// The plan as planned and the calls as written — what every mode already runs, and
    /// the row every crossing is read against.
    pub const NONE: Self = Self {
        rebatch: Rebatch::None,
        drain: Drain::None,
        empties: Empties::Never,
        hash: Hash::AsPlanned,
    };

    /// A small number standing for this setting, so a choice made from the seed differs
    /// between two candidates of one plan rather than only between plans.
    fn stamp(&self) -> u64 {
        let rebatch = match self.rebatch {
            Rebatch::None => 0,
            Rebatch::AboveSources => 1,
            Rebatch::AboveInterior => 2,
        };
        let empties = match self.empties {
            Empties::Never => 0,
            Empties::Sometimes(percent) => u64::from(percent) + 1,
        };
        rebatch
            + 8 * u64::from(self.drain == Drain::FirstLane)
            + 16 * empties
            + 4096 * u64::from(self.hash == Hash::Degenerate)
    }

    pub fn label(&self) -> String {
        let mut parts = Vec::new();
        match self.rebatch {
            Rebatch::None => {}
            Rebatch::AboveSources => parts.push("rebatch=sources".to_string()),
            Rebatch::AboveInterior => parts.push("rebatch=interior".to_string()),
        }
        if self.drain == Drain::FirstLane {
            parts.push("drain=lane0".to_string());
        }
        if let Empties::Sometimes(percent) = self.empties {
            parts.push(format!("empties={percent}%"));
        }
        if self.hash == Hash::Degenerate {
            parts.push("hash=one-lane".to_string());
        }
        if parts.is_empty() {
            "as-planned".to_string()
        } else {
            parts.join("/")
        }
    }
}

/// One edge of the tree, addressed by the pre-order position of the node below it — the
/// numbering `PlanIndex` uses, so an edge names the same node the driver would.
pub struct Edge {
    pub child: usize,
    pub node: &'static str,
    /// Why a rebatcher may not go here: the child declares an order and a coalesce clears
    /// it, so a parent that merges on the order refuses the plan and one that merely
    /// carries rows past it would take a different prefix under a row interval. Named
    /// rather than skipped — an edge passed over quietly reads as one that was covered.
    pub refused: Option<&'static str>,
}

const REFUSED_BY_ORDER: &str = "a coalesce clears the order this stream declares";

/// Every edge whose child is not a source, in pre-order.
pub fn interior_edges(root: &dyn GpuNode) -> Vec<Edge> {
    let mut edges = Vec::new();
    let mut next = 0;
    walk_edges(root, true, &mut next, &mut edges);
    edges
}

fn walk_edges(node: &dyn GpuNode, is_root: bool, next: &mut usize, edges: &mut Vec<Edge>) {
    let here = *next;
    *next += 1;
    let sorted = node
        .kind()
        .layout()
        .is_some_and(|layout| layout.sort_order.is_batch_sorted());
    let interior = !is_root && !node.children().is_empty();
    if interior {
        edges.push(Edge {
            child: here,
            node: node.name(),
            refused: sorted.then_some(REFUSED_BY_ORDER),
        });
    }
    for child in node.children() {
        walk_edges(child, false, next, edges);
    }
}

/// How many nodes a tree has, which is what says a rebatcher was actually inserted: an
/// edge set with nothing eligible in it injects nothing, and a run that injected nothing
/// is a run whose label claims a dimension it did not carry.
pub fn node_count(root: &dyn GpuNode) -> usize {
    1 + root
        .children()
        .into_iter()
        .map(node_count)
        .sum::<usize>()
}

/// `root` rewritten into one injected shape. The tree comes back rebuilt whether or not
/// anything was injected, so every run in a crossing is compared against a plan that took
/// the same path.
pub fn apply(root: &dyn GpuNode, injection: Injection, seed: u64) -> Box<dyn GpuNode> {
    let at = match injection.rebatch {
        // From the seed rather than the first eligible edge, which is the top of the tree
        // on every plan: the loader edges belong to the other setting, so taking the top
        // here would leave the middle — where the accumulators and the join build sides
        // are — never rebatched at all.
        Rebatch::AboveInterior => {
            let eligible: Vec<usize> = interior_edges(root)
                .into_iter()
                .filter(|edge| edge.refused.is_none())
                .map(|edge| edge.child)
                .collect();
            (!eligible.is_empty()).then(|| {
                eligible[(mix(seed, injection.stamp()) % eligible.len() as u64) as usize]
            })
        }
        _ => None,
    };
    let mut next = 0;
    rewrite(root, injection, at, &mut next)
}

fn rewrite(
    node: &dyn GpuNode,
    injection: Injection,
    at: Option<usize>,
    next: &mut usize,
) -> Box<dyn GpuNode> {
    let here = *next;
    *next += 1;
    let children: Vec<Box<dyn GpuNode>> = node
        .children()
        .into_iter()
        .map(|child| rewrite(child, injection, at, next))
        .collect();
    let rebuilt = match as_node_ref(node) {
        NodeRef::LoadParquet(load) if injection.drain == Drain::FirstLane => {
            drained(load, schema_of(node))
        }
        _ => rebuild(node, children),
    };
    let rebatch_here = match injection.rebatch {
        Rebatch::None => false,
        Rebatch::AboveSources => matches!(as_node_ref(node), NodeRef::LoadParquet(_)),
        Rebatch::AboveInterior => at == Some(here),
    };
    if rebatch_here {
        Box::new(GpuCoalesceAllBatches::new(rebuilt))
    } else {
        rebuilt
    }
}

/// A rebatcher above a named edge whatever the eligibility rule says. The refusal a
/// forbidden edge earns is demonstrated with this rather than predicted — the rule and
/// the engine agreeing is the claim.
pub fn rebatch_at(root: &dyn GpuNode, child: usize) -> Box<dyn GpuNode> {
    let mut next = 0;
    rewrite(
        root,
        Injection {
            rebatch: Rebatch::AboveInterior,
            ..Injection::NONE
        },
        Some(child),
        &mut next,
    )
}

/// A plan whose every interior edge is a forbidden one: a sort, the accumulator that
/// merges its batches and the k-way merge above them each declare an order, and a coalesce
/// at any of those edges clears it.
pub fn merge_over_sorted() -> Box<dyn GpuNode> {
    Box::new(GpuUnload::new(
        Box::new(GpuMergeSortedPartitions::new(
            sorted(source(None), key(0)),
            vec![key(0)],
            None,
        )),
        None,
    ))
}

/// Lane 0's batches handed to lane 1, so the lane stays in the mapping and stops
/// producing. Every row group is still read exactly once, which is what keeps the answer
/// the oracle's.
fn drained(load: &GpuLoadParquet, schema: Schema) -> Box<dyn GpuNode> {
    let mut groups = load.partition_groups.clone();
    if groups.len() > 1 {
        let moved = std::mem::take(&mut groups[0]);
        groups[1].extend(moved);
    }
    Box::new(GpuLoadParquet::new(
        load.table.clone(),
        load.projection.clone(),
        groups,
        &scan_of(load),
        load.limit,
        schema,
    ))
}

// ── the candidates, and choosing among them ────────────────────────────────

/// What a mode turned out to be for one query, which is what decides whether an injection
/// setting means anything there: draining needs a second lane to move the rows to, and a
/// degenerate hash needs a scatter to be degenerate at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedMode {
    pub name: &'static str,
    pub lanes: usize,
    pub shuffles: bool,
    /// Whether the plan carries a join whose answer to an empty build side is its probe
    /// side — Right, Full and RightAnti. Those refuse the call outright
    /// ([#175](../../../llm-wiki/tickets.md#t175)), so a hash that leaves every lane but
    /// one empty is a refusal rather than a shape, and the dimension has no meaning here.
    pub owes_probe_when_empty: bool,
}

/// What this query's plan at this mode turned out to be. Read off the tree rather than
/// off the knobs: `target_partitions` is what was asked for, and the small-table rule can
/// leave a query at one lane anyway.
pub fn planned_mode(name: &'static str, root: &dyn GpuNode) -> PlannedMode {
    fn walk(node: &dyn GpuNode, mode: &mut PlannedMode) {
        match as_node_ref(node) {
            NodeRef::LoadParquet(load) => mode.lanes = mode.lanes.max(load.partition_groups.len()),
            NodeRef::EmitPartitions(_) => mode.shuffles = true,
            NodeRef::Join(join) if !empty_build_answers_nothing(join.join_type) => {
                mode.owes_probe_when_empty = true
            }
            _ => {}
        }
        for child in node.children() {
            walk(child, mode);
        }
    }
    let mut mode = PlannedMode {
        name,
        lanes: 1,
        shuffles: false,
        owes_probe_when_empty: false,
    };
    walk(root, &mut mode);
    mode
}

/// The values each dimension takes. The high empty-batch setting is a percentage rather
/// than every call: a source that never advances is a source the driver never exhausts.
#[derive(Debug, Clone)]
pub struct Dimensions {
    pub rebatch: Vec<Rebatch>,
    pub drain: Vec<Drain>,
    pub empties: Vec<Empties>,
    pub hash: Vec<Hash>,
}

impl Default for Dimensions {
    fn default() -> Self {
        Self {
            rebatch: vec![Rebatch::None, Rebatch::AboveSources, Rebatch::AboveInterior],
            drain: vec![Drain::None, Drain::FirstLane],
            empties: vec![Empties::Never, Empties::Sometimes(50)],
            hash: vec![Hash::AsPlanned, Hash::Degenerate],
        }
    }
}

/// One run: the mode that planned it and what was injected into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub mode: usize,
    pub injection: Injection,
}

impl Candidate {
    pub fn label(&self, modes: &[PlannedMode]) -> String {
        format!("{} {}", modes[self.mode].name, self.injection.label())
    }
}

/// Every mode crossed with every setting, minus the combinations that mean nothing at
/// that mode: one lane has no lane to drain, and a plan with no scatter has no hash.
pub fn candidates(modes: &[PlannedMode], dimensions: &Dimensions) -> Vec<Candidate> {
    let mut out = Vec::new();
    for (index, mode) in modes.iter().enumerate() {
        for rebatch in &dimensions.rebatch {
            for drain in &dimensions.drain {
                for empties in &dimensions.empties {
                    for hash in &dimensions.hash {
                        if *drain == Drain::FirstLane && mode.lanes < 2 {
                            continue;
                        }
                        if *hash == Hash::Degenerate
                            && (!mode.shuffles || mode.owes_probe_when_empty)
                        {
                            continue;
                        }
                        out.push(Candidate {
                            mode: index,
                            injection: Injection {
                                rebatch: *rebatch,
                                drain: *drain,
                                empties: *empties,
                                hash: *hash,
                            },
                        });
                    }
                }
            }
        }
    }
    out
}

/// At most `cap` of them, covering every mode and every dimension value that the modes
/// make reachable. Cover first and fill after, so a smaller cap is still a cover — which
/// is what lets the run count be cut without cutting what is proved.
///
/// Seeded rather than sampled: the order is a hash of the seed and the candidate's
/// position, so two runs choose the same set and a failure is reproducible.
pub fn select(
    modes: &[PlannedMode],
    dimensions: &Dimensions,
    cap: usize,
    seed: u64,
) -> Vec<Candidate> {
    let all = candidates(modes, dimensions);
    let mut order: Vec<usize> = (0..all.len()).collect();
    order.sort_by_key(|index| (mix(seed, *index as u64), *index));

    let mut chosen: Vec<usize> = Vec::new();
    for (what, holds) in requirements(modes, dimensions) {
        if chosen.iter().any(|index| holds(&all[*index])) {
            continue;
        }
        let found = order
            .iter()
            .find(|index| holds(&all[**index]))
            .unwrap_or_else(|| panic!("no candidate carries {what}"));
        chosen.push(*found);
    }
    assert!(
        chosen.len() <= cap,
        "the cover alone is {} runs against a cap of {cap}, and a cap may cut runs rather \
         than cover",
        chosen.len()
    );
    for index in &order {
        if chosen.len() == cap.min(all.len()) {
            break;
        }
        if !chosen.contains(index) {
            chosen.push(*index);
        }
    }
    // Mode order, then the injection's, so a run reads down the modes rather than down
    // the seed — the seed decides which are chosen, not what order they are run in.
    let mut selected: Vec<Candidate> = chosen.into_iter().map(|index| all[index]).collect();
    selected.sort_by_key(|candidate| (candidate.mode, candidate.injection));
    selected
}

type Holds = Box<dyn Fn(&Candidate) -> bool>;

/// What a selection must carry: every mode, and every value of every dimension the modes
/// make reachable. Derived from the settings here, and written down again in the
/// selector's own case — a requirement read off the settings cannot catch a settings list
/// that lost a value.
fn requirements(modes: &[PlannedMode], dimensions: &Dimensions) -> Vec<(String, Holds)> {
    let mut out: Vec<(String, Holds)> = Vec::new();
    for (index, mode) in modes.iter().enumerate() {
        out.push((
            format!("the mode {}", mode.name),
            Box::new(move |candidate: &Candidate| candidate.mode == index),
        ));
    }
    for value in dimensions.rebatch.clone() {
        out.push((
            format!("a rebatcher {value:?}"),
            Box::new(move |candidate: &Candidate| candidate.injection.rebatch == value),
        ));
    }
    if modes.iter().any(|mode| mode.lanes > 1) {
        for value in dimensions.drain.clone() {
            out.push((
                format!("a drain {value:?}"),
                Box::new(move |candidate: &Candidate| candidate.injection.drain == value),
            ));
        }
    }
    for value in dimensions.empties.clone() {
        out.push((
            format!("empty batches {value:?}"),
            Box::new(move |candidate: &Candidate| candidate.injection.empties == value),
        ));
    }
    if modes
        .iter()
        .any(|mode| mode.shuffles && !mode.owes_probe_when_empty)
    {
        for value in dimensions.hash.clone() {
            out.push((
                format!("a hash {value:?}"),
                Box::new(move |candidate: &Candidate| candidate.injection.hash == value),
            ));
        }
    }
    out
}

/// One scatter's output for a batch of keys, under either hash. Reached through
/// `executors_for` rather than built by hand, so what it measures is the executor the
/// driver would have been handed.
pub fn emitter_over_four_lanes(keys: &[i64], hash: Hash) -> Vec<CpuBatch> {
    let node = GpuEmitPartitions::new(source(None), vec![0], 4);
    let ctx = InjectedContext::new(
        SessionContext::new().task_ctx(),
        Injection {
            hash,
            ..Injection::NONE
        },
        0,
    );
    let NodeExecutors::PartitionEmitter(mut emitter) =
        Injected::executors_for(&ctx, &node, 0, 0).expect("a scatter has an executor")
    else {
        panic!("a scatter is a partition emitter");
    };
    let column: ArrayRef = Arc::new(Int64Array::from(keys.to_vec()));
    let batch = RecordBatch::try_new(
        node.children()[0]
            .kind()
            .schema()
            .expect("the input is not a sink")
            .fields
            .clone(),
        vec![column.clone(), column],
    )
    .expect("a batch of keys");
    emitter
        .emit(CpuBatch::new(batch))
        .expect("the scatter runs")
        .0
}
