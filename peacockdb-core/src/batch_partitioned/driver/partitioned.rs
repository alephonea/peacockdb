//! `batch_partitioned_driver` — the tree, the queues, the schedule, and everything
//! cross-partition.
//!
//! One step runs every lane of the node the schedule picked: smallest height, ties
//! leftmost. That single rule is what makes this a push model — a batch's parent is
//! strictly lower, so it is carried up before anything below produces again — and what
//! bounds every queue at one batch per lane without a cap. Lane-scoped work is delegated
//! to [`super::single_partition`]; the three cross-lane categories are here, along with
//! the one node the driver special-cases, a `GpuUnload` carrying a limit.

use std::collections::VecDeque;

use super::accounting::{Held, ResidentAccountant, Slot, Trip};
use super::index::{PROBE_CHILD, PlanIndex, ROOT};
use super::scheduler::Scheduler;
use super::single_partition::{Avail, LaneCall, LaneDriver, LaneOutputs, LaneSite};
use super::{CallKind, StepError, TraceEvent, Underestimate};
use crate::batch_partitioned::backend::{Backend, NodeExecutors};
use crate::batch_partitioned::batch::Batch;
use crate::batch_partitioned::cpu_batch::CpuBatch;
use crate::batch_partitioned::error::{PlanError, RunError};
use crate::batch_partitioned::executor::{
    BackendError, LaneEvent, PartitionAccumulatorExecutor, PartitionEmitterExecutor, RowRange,
};
use crate::batch_partitioned::forwarder::{BatchForwarder, Forwarder};
use crate::batch_partitioned::node::GpuNode;
use crate::batch_partitioned::nodes::ExecutorCategory;
use crate::batch_partitioned::validate::check_canonical_form;

/// A step neither moves a batch nor finalizes a lane only if the schedule is wrong, so a
/// run that does not end is a bug here rather than a query that is merely large.
const DEFAULT_MAX_STEPS: usize = 1_000_000;

#[derive(Debug)]
pub struct RunReport {
    pub batches: Vec<CpuBatch>,
    pub peak_bytes: usize,
    /// Zero at the end of any correct run: a batch was held and never released otherwise.
    pub in_flight_bytes: usize,
    pub steps: usize,
    pub calls: usize,
    /// Batches held and batches released. Equal at the end of every run, on both the
    /// drained path and the early-exit one.
    pub holds: usize,
    pub releases: usize,
    pub trace: Vec<TraceEvent>,
    pub underestimates: Vec<Underestimate>,
    /// How many calls reported a measured transient rather than `None`. What makes an
    /// empty `underestimates` mean the model held: a backend measuring nothing produces
    /// the same empty list, and the two are indistinguishable without this.
    pub measured_calls: usize,
    /// Per node, rows released without an unload call — the saving a limit buys, made
    /// visible, since the rows returned look the same either way.
    pub rows_skipped: Vec<u64>,
    /// Per node, the most batches its queues held at once.
    pub peak_queued: Vec<usize>,
    /// Per node, its output lane count — what `peak_queued` is bounded by.
    pub lanes_of: Vec<usize>,
    /// Per node, per output lane, the batches it emitted in order.
    pub emitted: Vec<Vec<Vec<EmittedBatch>>>,
    /// Per node, per output lane, rows it emitted that nobody consumed — the queues an early
    /// exit left standing. Zero everywhere on a run that drained, and what closes
    /// `consumed + abandoned == the child's emitted` into an equality on every run.
    pub abandoned: Vec<Vec<u64>>,
    /// Per node, per child, per that child's lane, the rows this node consumed from it.
    /// Indexed by the child's lane rather than the consumer's, so it lines up with that
    /// child's own [`emitted`](RunReport::emitted) where the two differ — an emitter
    /// redistributes, so nothing else would sum.
    pub consumed: Vec<Vec<Vec<u64>>>,
    /// The nodes whose row interval was satisfied, in index order — empty on a run that
    /// drained. What the golden's `early_exit=` marker names, and the reason a lane can be
    /// short of what its plan called for: not a bool, because a reader of a smaller number
    /// needs to know which limit produced it.
    pub satisfied: Vec<usize>,
}

/// One batch a node emitted. The driver reads both figures already — the rows for a limit
/// interval, the bytes for the accountant — and kept only totals until the corpus goldens
/// needed the sizes themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmittedBatch {
    pub rows: u64,
    pub bytes: usize,
}

/// Run `root` to completion on `B`. `budget` of `None` accounts without ever tripping.
pub fn batch_partitioned_driver<B: Backend>(
    root: &dyn GpuNode,
    ctx: &B::Context,
    budget: Option<usize>,
) -> Result<RunReport, RunError> {
    Driver::<B>::new(root, ctx, budget)?.run(DEFAULT_MAX_STEPS)
}

pub(crate) struct Driver<'a, B: Backend> {
    index: PlanIndex<'a>,
    ctx: &'a B::Context,
    scheduler: Scheduler,
    acct: ResidentAccountant,
    states: Vec<NodeState<B>>,
    results: Vec<CpuBatch>,
    trace: Vec<TraceEvent>,
    steps: usize,
    /// Per node, rows of its input stream seen so far, summed over every lane. Only the
    /// driver can hold this: an unload instance is one lane's and the count is not.
    rows_seen: Vec<u64>,
    rows_skipped: Vec<u64>,
    peak_queued: Vec<usize>,
    emitted: Vec<Vec<Vec<EmittedBatch>>>,
    abandoned: Vec<Vec<u64>>,
    consumed: Vec<Vec<Vec<u64>>>,
}

/// Output queues live on the producing node, which is what makes lane remapping free: an
/// emitter reads its child's one lane, a forwarder reads whichever pair its map names, a
/// join reads lane p of each side, and no producer knows who consumes it.
struct NodeState<B: Backend> {
    out_queues: Vec<VecDeque<Held<B::Batch>>>,
    out_done: Vec<bool>,
    lanes: Vec<LaneDriver<B>>,
    cross: Option<CrossExecutor<B>>,
    /// `sources_of` materialized once per out lane: the predicate asks for it on every
    /// step, and the trait's `Vec` would be an allocation each time.
    forward_map: Vec<Vec<(usize, usize)>>,
    cursors: Vec<usize>,
    retired: Vec<Vec<bool>>,
    lane_done_sent: Vec<bool>,
    emitter_finished: bool,
}

enum CrossExecutor<B: Backend> {
    Accumulator(B::PartAcc),
    Emitter(B::Emitter),
}

impl<'a, B: Backend> Driver<'a, B> {
    pub(crate) fn new(
        root: &'a dyn GpuNode,
        ctx: &'a B::Context,
        budget: Option<usize>,
    ) -> Result<Self, RunError> {
        // The rule the planner applies, applied to a tree that did not come from it: a
        // hand-built plan meets the same refusal a planned one would, and there is one
        // statement of it rather than two.
        check_canonical_form(root).map_err(RunError::Backend)?;
        let index = PlanIndex::build(root).map_err(RunError::Backend)?;
        let scheduler = Scheduler::new(&index.shape);
        let acct = ResidentAccountant::new(index.slots, budget);
        let states = index
            .nodes
            .iter()
            .map(|node| NodeState {
                out_queues: (0..node.lanes).map(|_| VecDeque::new()).collect(),
                out_done: vec![false; node.lanes],
                lanes: (0..node.lanes).map(|_| LaneDriver::default()).collect(),
                cross: None,
                forward_map: Vec::new(),
                cursors: vec![0; node.lanes],
                retired: Vec::new(),
                lane_done_sent: vec![false; node.input_lanes],
                emitter_finished: false,
            })
            .collect();
        let nodes = index.len();
        let emitted = index
            .nodes
            .iter()
            .map(|node| vec![Vec::new(); node.lanes])
            .collect();
        let abandoned = index
            .nodes
            .iter()
            .map(|node| vec![0u64; node.lanes])
            .collect();
        let consumed = index
            .nodes
            .iter()
            .map(|node| {
                node.children
                    .iter()
                    .map(|child| vec![0u64; index.nodes[*child].lanes])
                    .collect()
            })
            .collect();
        let mut driver = Self {
            index,
            ctx,
            scheduler,
            acct,
            states,
            results: Vec::new(),
            trace: Vec::new(),
            steps: 0,
            rows_seen: vec![0; nodes],
            rows_skipped: vec![0; nodes],
            peak_queued: vec![0; nodes],
            emitted,
            abandoned,
            consumed,
        };
        driver.wire_forwarders()?;
        Ok(driver)
    }

    pub(crate) fn run(mut self, max_steps: usize) -> Result<RunReport, RunError> {
        let outcome = self.drive(max_steps);
        if outcome.is_err() {
            // A query that ended badly still gives back what it was holding, by the same
            // path the early exit uses: no handle is touched again, but the totals still
            // reconcile, so held and released stay equal on every path out of here.
            let _ = self.release_in_flight();
        }
        match outcome {
            Ok(()) => Ok(self.report()),
            Err(StepError::Run(error)) => Err(error),
            Err(StepError::Trip(trip)) => Err(self.budget_error(trip)),
        }
    }

    fn drive(&mut self, max_steps: usize) -> Result<(), StepError> {
        self.seed();
        while self.step()? {
            if self.steps > max_steps {
                return Err(
                    RunError::Protocol(format!("no termination after {max_steps} steps")).into(),
                );
            }
        }
        self.finish()
    }

    /// Readiness for every node, and the limits settled: a zero-row interval is satisfied
    /// before anything runs, and the plan still has to complete rather than stall.
    pub(crate) fn seed(&mut self) {
        for node in 0..self.index.len() {
            self.refresh(node);
            self.settle_limit(node);
        }
    }

    /// Run one node — every lane of it. `false` when nothing is runnable, which is how a
    /// run ends.
    pub(crate) fn step(&mut self) -> Result<bool, StepError> {
        let Some(node) = self.scheduler.next() else {
            return Ok(false);
        };
        self.steps += 1;
        self.run_node(node)?;
        self.settle_limit(node);
        self.refresh(node);
        if let Some(parent) = self.index.nodes[node].parent {
            self.refresh(parent);
        }
        Ok(true)
    }

    /// A run that ended because a limit was satisfied legitimately leaves lanes not done
    /// and queues non-empty; any other one has to have drained.
    pub(crate) fn finish(&mut self) -> Result<(), StepError> {
        if self.scheduler.any_satisfied() {
            self.release_in_flight()
        } else {
            self.assert_drained()
        }
    }

    fn run_node(&mut self, node: usize) -> Result<(), StepError> {
        match self.index.nodes[node].category {
            category if category.is_lane_scoped() => self.run_lane_scoped(node)?,
            ExecutorCategory::PartitionEmitter => self.run_emitter(node)?,
            ExecutorCategory::PartitionAccumulator => self.run_partition_accumulator(node)?,
            ExecutorCategory::BatchForwarder => self.run_forwarder(node),
            other => {
                return Err(RunError::Protocol(format!("{other:?} has no driver arm")).into());
            }
        }
        self.peak_queued[node] = self.peak_queued[node].max(self.queued(node));
        Ok(())
    }

    // -- lane-scoped -------------------------------------------------------------

    fn run_lane_scoped(&mut self, node: usize) -> Result<(), StepError> {
        let indexed = &self.index.nodes[node];
        let (category, interval, gpu_node) = (indexed.category, indexed.interval, indexed.node);
        let unloading = category == ExecutorCategory::Unload;
        for lane in 0..self.index.nodes[node].lanes {
            let avail = self.avail(node, lane);
            if !self.states[node].lanes[lane].can_step(category, &avail) {
                continue;
            }
            // A protocol violation is about one lane of one node, and the lane driver
            // knows neither: without this the message names a shape and no site.
            let name = self.index.nodes[node].node.name();
            let call = self.states[node].lanes[lane]
                .select(category, &avail)
                .map_err(|error| match error {
                    RunError::Protocol(what) => {
                        RunError::Protocol(format!("{name} lane {lane}: {what}"))
                    }
                    other => other,
                })?;
            // Every branch below is about the batch this call consumes, and the calls that
            // end a lane consume none: an interval decides nothing about them.
            let consuming = call.consumes().is_some();
            let arriving = match (interval, consuming) {
                (Some(_), true) => self.peek_rows(node, lane, call),
                _ => 0,
            };
            // Only an unload's decision is made here, its range being an argument of the
            // driver's own call. A mid-plan limit makes the same three-way choice inside
            // its executor, so for that one the driver only counts.
            let mut rows = RowRange::WHOLE;
            if let Some(interval) = interval.filter(|_| unloading && consuming) {
                match interval.range_of(self.rows_seen[node], arriving) {
                    // Not one row is wanted, so it never crosses the boundary: the batch
                    // is released where it stands, and the saving on the skip prefix is
                    // unbounded.
                    None => {
                        let unwanted = self.take_input(node, lane, call)?;
                        self.rows_seen[node] += arriving;
                        self.rows_skipped[node] += arriving;
                        self.acct.release(unwanted.bytes)?;
                        self.record(node, lane, CallKind::ReleaseUnwanted, 0);
                        continue;
                    }
                    Some(range) if !range.covers(arriving) => rows = range,
                    Some(_) => {}
                }
                // The ABI clamps an offset past the end and a length past it, because a C
                // ABI has to be total — and `range_of` can produce neither, so reaching
                // that tolerance means `rows_seen` has drifted. What that looks like from
                // the outside is a LIMIT quietly returning short, so the arithmetic is
                // checked here where it is done rather than absorbed where it is used.
                if rows != RowRange::WHOLE
                    && (rows.length == 0 || rows.offset + rows.length > arriving)
                {
                    return Err(RunError::Protocol(format!(
                        "{}: the rows wanted of this batch are {}..+{} of {arriving}",
                        self.index.nodes[node].node.name(),
                        rows.offset,
                        rows.length
                    ))
                    .into());
                }
            }
            let input = consuming
                .then(|| self.take_input(node, lane, call))
                .transpose()?;
            let site = LaneSite::<B> {
                ctx: self.ctx,
                node: gpu_node,
                category,
                post_order: self.index.nodes[node].post_order,
                lane,
                slot: self.slot(node, lane),
            };
            let outcome =
                self.states[node].lanes[lane].run(&site, call, input, rows, &mut self.acct)?;
            self.rows_seen[node] += arriving;
            let produced = match outcome.outputs {
                LaneOutputs::Device(batches) => {
                    let produced = batches.len();
                    for batch in batches {
                        self.record_emitted(node, lane, &batch);
                        self.states[node].out_queues[lane].push_back(batch);
                    }
                    produced
                }
                // The unload is the crossing: these bytes are on the host now, so they
                // leave the accounted set as they are handed to the caller.
                LaneOutputs::Host(batches) => {
                    let produced = batches.len();
                    for batch in batches {
                        self.record_emitted(node, lane, &batch);
                        self.acct.release(batch.bytes)?;
                        self.results.push(batch.batch);
                    }
                    produced
                }
            };
            // Both calls end a lane's build phase, and the hold lifts when every lane
            // has: a lane that owed nothing still has to say so, or the probe subtree is
            // held by a lane that will never build.
            if matches!(call, LaneCall::SetBuild | LaneCall::NoBuild) {
                self.scheduler.lane_left_build(node);
            }
            if outcome.finished {
                self.states[node].out_done[lane] = true;
            }
            self.record(node, lane, outcome.call, produced);
        }
        Ok(())
    }

    // -- cross-lane --------------------------------------------------------------

    fn run_emitter(&mut self, node: usize) -> Result<(), StepError> {
        let child = self.index.nodes[node].children[0];
        let lanes = self.index.nodes[node].lanes;
        let slot = self.slot(node, 0);
        if self.states[child].out_queues[0].is_empty() {
            self.states[node].emitter_finished = true;
            self.states[node].out_done = vec![true; lanes];
            self.acct.forget(slot);
            self.record(node, 0, CallKind::EmitDone, 0);
            return Ok(());
        }
        let batch = self.states[child].out_queues[0]
            .pop_front()
            .expect("a batch");
        self.record_consumed(node, 0, 0, batch.rows());
        self.build_cross(node)?;
        let Some(CrossExecutor::Emitter(emitter)) = &mut self.states[node].cross else {
            return Err(wrong_cross(self.index.nodes[node].node).into());
        };
        let modelled = self
            .acct
            .begin_call(slot, emitter, batch.rows(), batch.bytes)?;
        let emitted = emitter.emit(batch.batch);
        // The batch went into the call either way, so its bytes leave here whether or not
        // the call came back.
        self.acct.release(batch.bytes)?;
        let (outputs, stats) = emitted.map_err(|e| self.call_failed(node, 0, e))?;
        if outputs.len() != lanes {
            return Err(RunError::Protocol(format!(
                "{}: emit returned {} lanes against the {lanes} the plan declares",
                self.index.nodes[node].node.name(),
                outputs.len()
            ))
            .into());
        }
        let mut emitted = 0;
        for (lane, out) in outputs.into_iter().enumerate() {
            // Empty scatter outputs are dropped here, so nothing empty traverses a chain
            // because of hash skew.
            if out.num_rows() == 0 {
                continue;
            }
            let held = Held::of(out);
            self.acct.hold(held.bytes);
            self.record_emitted(node, lane, &held);
            self.states[node].out_queues[lane].push_back(held);
            emitted += 1;
        }
        let Some(CrossExecutor::Emitter(emitter)) = &self.states[node].cross else {
            return Err(wrong_cross(self.index.nodes[node].node).into());
        };
        self.acct.end_call(slot, emitter, stats, modelled)?;
        self.record(node, 0, CallKind::Emit, emitted);
        Ok(())
    }

    fn run_partition_accumulator(&mut self, node: usize) -> Result<(), StepError> {
        let child = self.index.nodes[node].children[0];
        let slot = self.slot(node, 0);
        for lane in 0..self.index.nodes[node].input_lanes {
            let has = !self.states[child].out_queues[lane].is_empty();
            let done = self.states[child].out_done[lane] && !self.states[node].lane_done_sent[lane];
            if !has && !done {
                continue;
            }
            let batch = has.then(|| {
                self.states[child].out_queues[lane]
                    .pop_front()
                    .expect("a batch")
            });
            if batch.is_none() {
                self.states[node].lane_done_sent[lane] = true;
            }
            let (rows, bytes) = batch
                .as_ref()
                .map_or((0, 0), |held| (held.rows(), held.bytes));
            if batch.is_some() {
                self.record_consumed(node, 0, lane, rows);
            }
            self.build_cross(node)?;
            let Some(CrossExecutor::Accumulator(accumulator)) = &mut self.states[node].cross else {
                return Err(wrong_cross(self.index.nodes[node].node).into());
            };
            let modelled = self.acct.begin_call(slot, accumulator, rows, bytes)?;
            let had_batch = batch.is_some();
            let (event, kind) = match batch {
                Some(held) => (LaneEvent::Batch(held.batch), CallKind::LaneEvent),
                None => (LaneEvent::Done, CallKind::LaneDone),
            };
            let accumulated = accumulator.accumulate_and_fetch(lane, event);
            // A lane's end carries no batch, so there is nothing to release: counting one
            // anyway made holds and releases disagree on every plan with a cross-lane
            // accumulator, which is what the report says they never do.
            if had_batch {
                self.acct.release(bytes)?;
            }
            let (outputs, stats) = accumulated.map_err(|e| self.call_failed(node, lane, e))?;
            let produced = outputs.len();
            for out in outputs {
                let held = Held::of(out);
                self.acct.hold(held.bytes);
                self.record_emitted(node, 0, &held);
                self.states[node].out_queues[0].push_back(held);
            }
            let Some(CrossExecutor::Accumulator(accumulator)) = &self.states[node].cross else {
                return Err(wrong_cross(self.index.nodes[node].node).into());
            };
            self.acct.end_call(slot, accumulator, stats, modelled)?;
            self.record(node, lane, kind, produced);
        }
        if self.states[node].lane_done_sent.iter().all(|sent| *sent) {
            self.states[node].out_done[0] = true;
            self.acct.forget(slot);
        }
        Ok(())
    }

    /// One batch per visit, cycling the lane's map in order from its cursor. The merge's
    /// round-robin and the interleave's child rotation are this one rule.
    fn run_forwarder(&mut self, node: usize) {
        for lane in 0..self.index.nodes[node].lanes {
            if self.states[node].out_done[lane] {
                continue;
            }
            match self.forward_one(node, lane) {
                true => self.record(node, lane, CallKind::Forward, 1),
                false if self.states[node].retired[lane].iter().all(|gone| *gone) => {
                    self.states[node].out_done[lane] = true;
                    self.record(node, lane, CallKind::ForwardDone, 0);
                }
                false => {}
            }
        }
    }

    fn forward_one(&mut self, node: usize, lane: usize) -> bool {
        let sources = self.states[node].forward_map[lane].len();
        let start = self.states[node].cursors[lane];
        for offset in 0..sources {
            let index = (start + offset) % sources;
            if self.states[node].retired[lane][index] {
                continue;
            }
            let (child_index, child_lane) = self.states[node].forward_map[lane][index];
            let child = self.index.nodes[node].children[child_index];
            // A move between queues: the batch stays in flight, so nothing is accounted.
            if let Some(batch) = self.states[child].out_queues[child_lane].pop_front() {
                self.record_consumed(node, child_index, child_lane, batch.rows());
                self.record_emitted(node, lane, &batch);
                self.states[node].out_queues[lane].push_back(batch);
                self.states[node].cursors[lane] = (index + 1) % sources;
                return true;
            }
            if self.states[child].out_done[child_lane] {
                self.states[node].retired[lane][index] = true;
            }
        }
        false
    }

    // -- schedule ----------------------------------------------------------------

    /// Recompute this node's ready lanes. Only the node just run and its parent can have
    /// changed: a node's readiness is a fact about its inputs, and nothing else moved.
    fn refresh(&mut self, node: usize) {
        for lane in 0..self.index.nodes[node].ready_lanes {
            let ready = self.lane_ready(node, lane);
            self.scheduler.set_lane_ready(node, lane, ready);
        }
    }

    fn lane_ready(&self, node: usize, lane: usize) -> bool {
        let indexed = &self.index.nodes[node];
        match indexed.category {
            category if category.is_lane_scoped() => {
                let avail = self.avail(node, lane);
                self.states[node].lanes[lane].can_step(category, &avail)
            }
            ExecutorCategory::PartitionEmitter => {
                let child = indexed.children[0];
                let has = !self.states[child].out_queues[0].is_empty();
                has || (self.states[child].out_done[0] && !self.states[node].emitter_finished)
            }
            ExecutorCategory::PartitionAccumulator => {
                let child = indexed.children[0];
                !self.states[child].out_queues[lane].is_empty()
                    || (self.states[child].out_done[lane]
                        && !self.states[node].lane_done_sent[lane])
            }
            ExecutorCategory::BatchForwarder => self.forwarder_lane_ready(node, lane),
            _ => false,
        }
    }

    /// A forwarder lane can step when a source has a batch, or when the last live source
    /// has finished — the visit that retires it is what marks the lane done.
    fn forwarder_lane_ready(&self, node: usize, lane: usize) -> bool {
        if self.states[node].out_done[lane] {
            return false;
        }
        let mut live = 0;
        for (index, (child_index, child_lane)) in
            self.states[node].forward_map[lane].iter().enumerate()
        {
            if self.states[node].retired[lane][index] {
                continue;
            }
            let child = self.index.nodes[node].children[*child_index];
            if !self.states[child].out_queues[*child_lane].is_empty() {
                return true;
            }
            if !self.states[child].out_done[*child_lane] {
                live += 1;
            }
        }
        live == 0
    }

    /// Enough rows have reached this node that no later one can change its answer. It is
    /// marked done as it is held, or the hold would stop it reporting and strand its
    /// parent — `LIMIT 0` is the case that forces it.
    fn settle_limit(&mut self, node: usize) {
        let Some(interval) = self.index.nodes[node].interval else {
            return;
        };
        if !interval.satisfied_by(self.rows_seen[node]) {
            return;
        }
        self.scheduler.satisfy(node);
        self.states[node].out_done = vec![true; self.index.nodes[node].lanes];
    }

    // -- queues ------------------------------------------------------------------

    fn avail(&self, node: usize, lane: usize) -> Avail {
        let mut avail = Avail::default();
        for (slot, child) in self.index.nodes[node].children.iter().enumerate().take(2) {
            let queue = &self.states[*child].out_queues[lane];
            avail.has[slot] = !queue.is_empty();
            avail.done[slot] = self.states[*child].out_done[lane] && queue.is_empty();
        }
        avail
    }

    fn peek_rows(&self, node: usize, lane: usize, call: LaneCall) -> u64 {
        let slot = call.consumes().expect("a consuming call");
        let child = self.index.nodes[node].children[slot];
        self.states[child].out_queues[lane]
            .front()
            .map_or(0, Held::rows)
    }

    fn take_input(
        &mut self,
        node: usize,
        lane: usize,
        call: LaneCall,
    ) -> Result<Held<B::Batch>, StepError> {
        let slot = call.consumes().expect("a consuming call");
        let child = self.index.nodes[node].children[slot];
        let batch = self.states[child].out_queues[lane]
            .pop_front()
            .ok_or_else(|| {
                StepError::Run(RunError::Protocol(format!(
                    "{}: the schedule offered a batch that is not there",
                    self.index.nodes[node].node.name()
                )))
            })?;
        self.record_consumed(node, slot, lane, batch.rows());
        Ok(batch)
    }

    #[cfg(test)]
    pub(crate) fn hops(&self) -> (usize, usize) {
        self.acct.hops()
    }

    #[cfg(test)]
    pub(crate) fn release_all(&mut self) -> Result<(), StepError> {
        self.release_in_flight()
    }

    #[cfg(test)]
    pub(crate) fn queue_len(&self, node: usize, lane: usize) -> usize {
        self.states[node].out_queues[lane].len()
    }

    #[cfg(test)]
    pub(crate) fn last_call(&self) -> Option<CallKind> {
        self.trace.last().map(|event| event.call)
    }

    fn queued(&self, node: usize) -> usize {
        self.states[node].out_queues.iter().map(VecDeque::len).sum()
    }

    fn release_in_flight(&mut self) -> Result<(), StepError> {
        for node in 0..self.index.len() {
            for lane in 0..self.states[node].out_queues.len() {
                while let Some(batch) = self.states[node].out_queues[lane].pop_front() {
                    self.abandoned[node][lane] += batch.rows();
                    self.acct.release(batch.bytes)?;
                }
            }
        }
        Ok(())
    }

    fn assert_drained(&self) -> Result<(), StepError> {
        // The root not being done is the same failure seen from the other end: a node that
        // never learned its input had ended never emitted, and no queue holds the evidence.
        if !self.states[ROOT].out_done.iter().all(|done| *done) {
            return Err(RunError::Protocol(format!(
                "nothing is runnable and {} has lanes that never ended",
                self.index.nodes[ROOT].node.name()
            ))
            .into());
        }
        let stranded: Vec<String> = (0..self.index.len())
            .flat_map(|node| {
                self.states[node]
                    .out_queues
                    .iter()
                    .enumerate()
                    .filter(|(_, queue)| !queue.is_empty())
                    .map(move |(lane, queue)| {
                        format!(
                            "{} lane {lane} holds {}",
                            self.index.nodes[node].node.name(),
                            queue.len()
                        )
                    })
            })
            .collect();
        if stranded.is_empty() {
            return Ok(());
        }
        Err(RunError::Protocol(format!(
            "nothing is runnable and batches remain: {}",
            stranded.join("; ")
        ))
        .into())
    }

    // -- wiring ------------------------------------------------------------------

    /// Forwarders are wired once: `sources_of` allocates, and the readiness predicate asks
    /// for it on every step. Checking the map against the tree here is also the only place
    /// a routing that names a child or a lane the plan does not have can be caught.
    fn wire_forwarders(&mut self) -> Result<(), RunError> {
        for node in 0..self.index.len() {
            if self.index.nodes[node].category != ExecutorCategory::BatchForwarder {
                continue;
            }
            let indexed = &self.index.nodes[node];
            let executors = B::executors_for(self.ctx, indexed.node, indexed.post_order, 0)
                .map_err(RunError::Backend)?;
            let NodeExecutors::BatchForwarder(forwarder) = executors else {
                return Err(RunError::Backend(PlanError::Invalid(format!(
                    "{}: the backend built a {:?} executor for a routing node",
                    indexed.node.name(),
                    executors.category()
                ))));
            };
            let map = self.forward_map_of(node, &forwarder)?;
            self.states[node].retired = map.iter().map(|lane| vec![false; lane.len()]).collect();
            self.states[node].forward_map = map;
        }
        Ok(())
    }

    fn forward_map_of(
        &self,
        node: usize,
        forwarder: &Forwarder,
    ) -> Result<Vec<Vec<(usize, usize)>>, RunError> {
        let indexed = &self.index.nodes[node];
        let invalid = |what: String| RunError::Backend(PlanError::Invalid(what));
        if let Forwarder::Union { lanes } = forwarder
            && lanes.len() != indexed.lanes
        {
            return Err(invalid(format!(
                "{}: the routing serves {} lanes and the layout declares {}",
                indexed.node.name(),
                lanes.len(),
                indexed.lanes
            )));
        }
        let mut map = Vec::with_capacity(indexed.lanes);
        for lane in 0..indexed.lanes {
            let sources = forwarder.sources_of(lane);
            if sources.is_empty() {
                return Err(invalid(format!(
                    "{}: lane {lane} is served by nothing, so it can never finish",
                    indexed.node.name()
                )));
            }
            for (child_index, child_lane) in &sources {
                let child = indexed.children.get(*child_index).ok_or_else(|| {
                    invalid(format!(
                        "{}: lane {lane} reads child {child_index} of {}",
                        indexed.node.name(),
                        indexed.children.len()
                    ))
                })?;
                if *child_lane >= self.index.nodes[*child].lanes {
                    return Err(invalid(format!(
                        "{}: lane {lane} reads lane {child_lane} of a {}-lane child",
                        indexed.node.name(),
                        self.index.nodes[*child].lanes
                    )));
                }
            }
            map.push(sources);
        }
        Ok(map)
    }

    fn build_cross(&mut self, node: usize) -> Result<(), RunError> {
        if self.states[node].cross.is_some() {
            return Ok(());
        }
        let indexed = &self.index.nodes[node];
        let executors = B::executors_for(self.ctx, indexed.node, indexed.post_order, 0)
            .map_err(RunError::Backend)?;
        if executors.category() != indexed.category {
            return Err(RunError::Backend(PlanError::Invalid(format!(
                "{}: the backend built a {:?} executor for a {:?} node",
                indexed.node.name(),
                executors.category(),
                indexed.category
            ))));
        }
        self.states[node].cross = Some(match executors {
            NodeExecutors::PartitionAccumulator(accumulator) => {
                CrossExecutor::Accumulator(accumulator)
            }
            NodeExecutors::PartitionEmitter(emitter) => CrossExecutor::Emitter(emitter),
            _ => return Err(wrong_cross(indexed.node)),
        });
        Ok(())
    }

    // -- reporting ---------------------------------------------------------------

    fn slot(&self, node: usize, lane: usize) -> Slot {
        Slot {
            index: self.index.slot(node, lane),
            node: node as u32,
            lane: lane as u32,
        }
    }

    fn record_emitted<T: crate::batch_partitioned::batch::Batch>(
        &mut self,
        node: usize,
        lane: usize,
        batch: &Held<T>,
    ) {
        self.emitted[node][lane].push(EmittedBatch {
            rows: batch.rows(),
            bytes: batch.bytes,
        });
    }

    fn record_consumed(&mut self, node: usize, slot: usize, child_lane: usize, rows: u64) {
        self.consumed[node][slot][child_lane] += rows;
    }

    fn record(&mut self, node: usize, lane: usize, call: CallKind, outputs: usize) {
        self.trace.push(TraceEvent {
            step: self.steps as u32,
            node: node as u32,
            lane: lane as u32,
            call,
            outputs: outputs as u32,
        });
    }

    /// A failed call ends the query: nothing further is scheduled and no handle is touched
    /// again. The node and the lane are what a message can name that a backend cannot.
    fn call_failed(&self, node: usize, lane: usize, error: BackendError) -> StepError {
        StepError::Run(RunError::CallFailed(format!(
            "{} lane {lane}: {error}",
            self.index.nodes[node].node.name()
        )))
    }

    fn budget_error(&self, trip: Trip) -> RunError {
        RunError::BudgetExceeded {
            when: trip.when,
            message: format!(
                "resident GPU memory budget exceeded at {} lane {}, {}: {} bytes > budget {} \
                 bytes",
                self.index.nodes[trip.slot.node as usize].node.name(),
                trip.slot.lane,
                trip.when.describe(),
                trip.bytes,
                trip.budget
            ),
        }
    }

    fn report(self) -> RunReport {
        RunReport {
            batches: self.results,
            peak_bytes: self.acct.peak(),
            in_flight_bytes: self.acct.in_flight(),
            steps: self.steps,
            calls: self.acct.calls(),
            holds: self.acct.hops().0,
            releases: self.acct.hops().1,
            underestimates: self.acct.underestimates().to_vec(),
            measured_calls: self.acct.measured_calls(),
            trace: self.trace,
            rows_skipped: self.rows_skipped,
            lanes_of: self.index.nodes.iter().map(|node| node.lanes).collect(),
            peak_queued: self.peak_queued,
            emitted: self.emitted,
            abandoned: self.abandoned,
            consumed: self.consumed,
            satisfied: (0..self.index.len())
                .filter(|node| self.scheduler.is_satisfied(*node))
                .collect(),
        }
    }
}

fn wrong_cross(node: &dyn GpuNode) -> RunError {
    RunError::Backend(PlanError::Invalid(format!(
        "{}: its executor is not the cross-lane one its category names",
        node.name()
    )))
}

/// The probe side is always the right child, which is the orientation the schedule's
/// leftmost tie-break turns into "the build subtree drains first".
const _: () = assert!(PROBE_CHILD == 1 && ROOT == 0);
