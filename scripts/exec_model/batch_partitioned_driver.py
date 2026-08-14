"""`batch_partitioned_driver` — the scheduler and everything cross-partition.

**The strategy.** Every node carries a height (distance to the root) and a left-to-right
order, both computed once in `plan.py`. A node is *runnable* when any of its partitions
can make progress: a source always can, and any other node can once its inputs for that
lane hold a batch or are known to be finished. Among all runnable nodes the driver picks
the one with the smallest height, breaking ties leftmost, and then runs **every** lane of
that node.

Min-height-first is what makes this a push model: as soon as a node produces a batch its
parent becomes runnable at a strictly lower height, so the batch is carried up the tree
before anything below produces again. It stops at a batch accumulator, a partition
accumulator, or the sink — which is also the livelock argument. The one place a batch can
be blocked is a join whose other side has not arrived; orienting the tree so the build
side is always the left child removes that wait, because at equal heights the leftmost
node wins and the build subtree drains first.

**Queues stay bounded at one batch per lane, with no explicit cap.** A producer's
out-queue is drained by its parent before the producer runs again, since the parent's
height is strictly lower. The one shape that breaks it is a join in its build phase — it
cannot consume a probe batch yet, so nothing drains its probe child — and that is closed
by holding the join's whole probe subtree until the build is set (`_held_by_a_join_build`).
With the hold in place the bound is unconditional, and the spec's cap-Q mechanism is
unnecessary.

Lane-scoped work is delegated to `batch_single_partition_driver`; this driver owns the
tree, the queues, the schedule, and the three cross-lane categories.
"""

from __future__ import annotations

from dataclasses import dataclass

from .accounting import ResidentAccountant
from .batch import Batch
from .errors import DriverError
from .executors import LaneEvent
from .node import LANE_SCOPED, BackendSelector, ExecutorCategory
from .plan import Plan, PlanNodeInfo
from .runtime import LaneInputs, NodeState
from .batch_single_partition_driver import (
    PROBE_SLOT,
    JoinPhase,
    batch_single_partition_driver,
)

#: Safety valve: a step that neither moves a batch nor finalizes a lane cannot happen,
#: so a run that never ends is a bug in the scheduler, and the prototype should say so.
DEFAULT_MAX_STEPS = 100_000


@dataclass(frozen=True)
class TraceEvent:
    step: int
    node: str
    lane: int
    call: str
    n_out: int


class BatchPartitionedDriver:
    def __init__(self, plan: Plan, selector: BackendSelector, budget: int | None = None):
        self.plan = plan
        self.selector = selector
        self.accountant = ResidentAccountant(budget)
        self.results: list[Batch] = []
        self.trace: list[TraceEvent] = []
        self.steps = 0
        #: per node: rows of its input stream seen so far, summed over every lane. Only
        #: the driver can hold this — an Unload instance is one lane's.
        self.rows_seen: list[int] = [0] * len(plan.nodes)
        #: rows released without an unload call, per node — the saving, made visible
        self.rows_skipped: list[int] = [0] * len(plan.nodes)
        self.states: list[NodeState] = [
            NodeState.create(info, self._input_lane_count(info)) for info in plan.nodes
        ]
        self.peak_queued: list[int] = [0] * len(plan.nodes)

    # -- public ------------------------------------------------------------------

    def run(self, max_steps: int = DEFAULT_MAX_STEPS) -> list[Batch]:
        self._settle_limits()   # a zero-row interval is satisfied before anything runs
        while self.step():
            if self.steps > max_steps:
                raise DriverError(f"no termination after {max_steps} steps")
        if self.early_exit:
            self._drop_in_flight()
        else:
            self._assert_drained()
        return self.results

    @property
    def early_exit(self) -> bool:
        """The run stopped because a limit was satisfied, not because work ran out."""
        return any(self._is_satisfied(state) for state in self.states)

    def step(self) -> bool:
        """Run one node — every lane of it. False when nothing is runnable."""
        chosen = self.choose()
        if chosen is None:
            return False
        self.steps += 1
        self._run(chosen)
        self._settle_limits()
        self._drain_root()
        return True

    def runnable_nodes(self) -> list[PlanNodeInfo]:
        return [s.info for s in self.states if self._runnable(s)]

    # -- scheduling --------------------------------------------------------------

    def choose(self) -> NodeState | None:
        """The scheduling decision: smallest height among runnable nodes, ties leftmost."""
        candidates = [s for s in self.states if self._runnable(s)]
        if not candidates:
            return None
        return min(candidates, key=lambda s: (s.info.height, s.info.order))

    def _runnable(self, state: NodeState) -> bool:
        if self._held_by_a_join_build(state) or self._held_by_a_satisfied_limit(state):
            return False
        category = state.info.category
        if category in LANE_SCOPED:
            return any(
                self._lane_driver(state, lane).can_step(self._lane_inputs(state, lane))
                for lane in range(state.info.n_lanes)
            )
        if category is ExecutorCategory.PARTITION_EMITTER:
            inputs = self._lane_inputs(state, 0)
            return inputs.has(0) or (inputs.done(0) and not state.emitter_finished)
        if category is ExecutorCategory.PARTITION_ACCUMULATOR:
            return any(
                self._accumulator_lane_ready(state, lane)
                for lane in range(len(state.lane_done_sent))
            )
        if category is ExecutorCategory.BATCH_FORWARDER:
            return any(
                self._forwarder_lane_ready(state, lane) for lane in range(state.info.n_lanes)
            )
        raise DriverError(f"{state.info}: unhandled category {category.value}")

    # -- backpressure ------------------------------------------------------------

    def _held_by_a_join_build(self, state: NodeState) -> bool:
        """A join in its build phase holds back its whole probe subtree.

        The join itself cannot consume a probe batch until `set_build` has run, so
        without this the probe side runs anyway and its output piles up. Blocking only
        the probe *child* would not help — its own child would keep producing and the
        pile would simply move one node down — so the hold is transitive over every edge
        on the path to the root.

        This cannot deadlock. Plans are trees, so a join's build subtree is disjoint from
        its probe subtree and is never held by this rule; the build side therefore always
        has a runnable node until it completes, and completing it is what lifts the hold.
        Nested joins resolve outermost-first for the same reason.
        """
        info = state.info
        while info.parent is not None:
            parent = self.states[info.parent]
            if (
                parent.info.category is ExecutorCategory.JOIN
                and info.child_slot == PROBE_SLOT
                and self._awaits_build(parent)
            ):
                return True
            info = parent.info
        return False

    def _held_by_a_satisfied_limit(self, state: NodeState) -> bool:
        """A satisfied limit holds its whole subtree, and itself.

        The same shape as the join hold, and for the same reason: blocking one node would
        leave its child producing into a queue nothing drains. It differs in never lifting
        — no later batch can change an answer that is already complete — so a run ends here
        with lanes not done and queues non-empty, which is what `_drop_in_flight` is for.
        """
        info = state.info
        while True:
            if self._is_satisfied(self.states[info.id]):
                return True
            if info.parent is None:
                return False
            info = self.states[info.parent].info

    def _is_satisfied(self, state: NodeState) -> bool:
        """Enough rows have reached this node that no later one can change its answer."""
        interval = state.info.node.row_interval()
        return interval is not None and interval.satisfied_by(self.rows_seen[state.info.id])

    def _settle_limits(self) -> None:
        """A satisfied limit will never produce again, so say so before anything waits.

        Without this the hold below would also stop the node itself from reporting done,
        and its parent would wait forever for a lane that had in fact finished. The
        pathological case is a zero-row interval, satisfied before a single step: the plan
        has to complete and return nothing, not stall.
        """
        for state in self.states:
            if self._is_satisfied(state) and not all(state.out_done):
                state.out_done = [True] * len(state.out_done)

    def _awaits_build(self, state: NodeState) -> bool:
        """True while any of the join's lanes has yet to leave its build phase.

        A lane with no driver yet has not been entered, so it is still in build.
        """
        for lane in range(state.info.n_lanes):
            driver = state.lane_drivers.get(lane)
            if driver is None:
                return True
            if not driver.finished and driver.join_phase is JoinPhase.BUILD:
                return True
        return False

    # -- running -----------------------------------------------------------------

    def _run(self, state: NodeState) -> None:
        category = state.info.category
        if category in LANE_SCOPED:
            self._run_lane_scoped(state)
        elif category is ExecutorCategory.PARTITION_EMITTER:
            self._run_emitter(state)
        elif category is ExecutorCategory.PARTITION_ACCUMULATOR:
            self._run_partition_accumulator(state)
        elif category is ExecutorCategory.BATCH_FORWARDER:
            self._run_forwarder(state)
        self.peak_queued[state.info.id] = max(
            self.peak_queued[state.info.id], state.queued_batches()
        )

    def _run_lane_scoped(self, state: NodeState) -> None:
        interval = self._interval_of(state)
        unloading = state.info.category is ExecutorCategory.UNLOAD
        for lane in range(state.info.n_lanes):
            driver = self._lane_driver(state, lane)
            inputs = self._lane_inputs(state, lane)
            if not driver.can_step(inputs):
                continue
            rows = None
            arriving = 0
            if interval is not None and inputs.has(0):
                arriving = inputs.peek(0).num_rows()
                # Only an unload's drop-narrow-or-pass decision is made here, because its
                # range is an argument of the driver's own call. A mid-plan limit makes
                # the same three-way decision inside its executor — releasing, forwarding,
                # or slicing through `peacock_executor_slice_handle` — so for it the
                # driver only counts, to feed `is_satisfied`. See `accumulators.LimitStream`.
                if unloading:
                    rows = interval.range_of(self.rows_seen[state.info.id], arriving)
                    if rows is None:
                        # Not one row is wanted, so it never crosses the boundary: the
                        # handle is released here. Unbounded saving on the skip prefix,
                        # and the property a test on the rows returned cannot see.
                        unwanted = inputs.take(0)
                        self.rows_seen[state.info.id] += arriving
                        self.rows_skipped[state.info.id] += unwanted.num_rows()
                        self.accountant.release(unwanted)
                        self._record(state, lane, "release/unwanted", 0)
                        continue
                    if rows.covers(arriving):
                        rows = None   # every row wanted: the fetch needs no range
            result = driver.step(inputs, rows)
            self.rows_seen[state.info.id] += arriving
            for batch in result.outputs:
                self._enqueue(state, lane, batch)
            if result.finished:
                state.out_done[lane] = True
            self._record(state, lane, result.call, len(result.outputs))

    def _interval_of(self, state: NodeState):
        """This node's row interval: an absorbed root-adjacent limit, or a mid-plan one."""
        return state.info.node.row_interval()

    def _run_emitter(self, state: NodeState) -> None:
        inputs = self._lane_inputs(state, 0)
        executor = self._cross_executor(state)
        if not inputs.has(0):
            state.emitter_finished = True
            for lane in range(state.info.n_lanes):
                state.out_done[lane] = True
            self.accountant.forget(str(state.info))
            self._record(state, 0, "emit/done", 0)
            return
        batch = inputs.take(0)
        label = str(state.info)
        modelled = self.accountant.begin_call(label, executor, batch.num_rows(), batch.byte_size())
        outputs, stats = executor.emit(batch)
        self.accountant.release(batch)
        if len(outputs) != state.info.n_lanes:
            raise DriverError(
                f"{state.info}: emit returned {len(outputs)} lanes, expected {state.info.n_lanes}"
            )
        emitted = 0
        for lane, out in enumerate(outputs):
            # Empty scatter outputs are dropped here, so nothing empty ever traverses a
            # chain because of hash skew.
            if out.num_rows() == 0:
                continue
            self._enqueue(state, lane, out)
            emitted += 1
        self.accountant.end_call(label, executor, stats, modelled)
        self._record(state, 0, "emit", emitted)

    def _run_partition_accumulator(self, state: NodeState) -> None:
        executor = self._cross_executor(state)
        child = self.states[state.info.children[0]]
        for lane in range(len(state.lane_done_sent)):
            inputs = LaneInputs([(child, lane)])
            label = str(state.info)
            if inputs.has(0):
                batch = inputs.take(0)
                modelled = self.accountant.begin_call(
                    label, executor, batch.num_rows(), batch.byte_size()
                )
                event, call = LaneEvent.of(batch), "accumulate_and_fetch"
            elif inputs.done(0) and not state.lane_done_sent[lane]:
                state.lane_done_sent[lane] = True
                modelled = self.accountant.begin_call(label, executor, 0, 0)
                event, call = LaneEvent.done(), "accumulate_and_fetch/done"
            else:
                continue
            outputs, stats = executor.accumulate_and_fetch(lane, event)
            if event.batch is not None:
                self.accountant.release(event.batch)
            for out in outputs:
                self._enqueue(state, 0, out)
            self.accountant.end_call(label, executor, stats, modelled)
            self._record(state, lane, call, len(outputs))
        if all(state.lane_done_sent):
            state.out_done[0] = True
            self.accountant.forget(str(state.info))

    def _run_forwarder(self, state: NodeState) -> None:
        forwarder = state.info.executors.forwarder
        for lane in range(state.info.n_lanes):
            if state.out_done[lane]:
                continue
            sources = forwarder.sources_of(lane)
            forwarded = self._forward_one(state, lane, sources)
            if forwarded is not None:
                self._record(state, lane, "forward", 1)
            elif len(state.retired[lane]) == len(sources):
                state.out_done[lane] = True
                self._record(state, lane, "forward/done", 0)

    def _forward_one(self, state: NodeState, lane: int, sources) -> Batch | None:
        """One batch per visit, cycling `sources_of` in order from the lane's cursor."""
        n = len(sources)
        start = state.cursors[lane]
        for offset in range(n):
            index = (start + offset) % n
            if index in state.retired[lane]:
                continue
            child_index, child_lane = sources[index]
            child = self.states[state.info.children[child_index]]
            if child.out_queues[child_lane]:
                # A move between queues: the batch stays in flight, so no accounting.
                batch = child.out_queues[child_lane].popleft()
                state.out_queues[lane].append(batch)
                state.cursors[lane] = (index + 1) % n
                return batch
            if child.out_done[child_lane]:
                state.retired[lane].add(index)
        return None

    # -- wiring ------------------------------------------------------------------

    def _input_lane_count(self, info: PlanNodeInfo) -> int:
        if info.category is not ExecutorCategory.PARTITION_ACCUMULATOR:
            return 0
        return self.plan.child(info.id, 0).n_lanes

    def _lane_driver(self, state: NodeState, lane: int):
        driver = state.lane_drivers.get(lane)
        if driver is None:
            backends = state.info.executors.backends
            driver = batch_single_partition_driver(
                state.info,
                lane,
                lambda: self.selector.select(state.info.category, backends, lane),
                self.accountant,
            )
            state.lane_drivers[lane] = driver
        return driver

    def _cross_executor(self, state: NodeState):
        if state.cross_executor is None:
            state.cross_executor = self.selector.select(
                state.info.category, state.info.executors.backends, None
            )
        return state.cross_executor

    def _lane_inputs(self, state: NodeState, lane: int) -> LaneInputs:
        sources = [(self.states[child], lane) for child in state.info.children]
        return LaneInputs(sources)

    def _accumulator_lane_ready(self, state: NodeState, lane: int) -> bool:
        child = self.states[state.info.children[0]]
        if child.out_queues[lane]:
            return True
        return child.out_done[lane] and not state.lane_done_sent[lane]

    def _forwarder_lane_ready(self, state: NodeState, lane: int) -> bool:
        if state.out_done[lane]:
            return False
        sources = state.info.executors.forwarder.sources_of(lane)
        live = 0
        for index, (child_index, child_lane) in enumerate(sources):
            if index in state.retired[lane]:
                continue
            child = self.states[state.info.children[child_index]]
            if child.out_queues[child_lane]:
                return True
            if not child.out_done[child_lane]:
                live += 1
        return live == 0

    def _enqueue(self, state: NodeState, lane: int, batch: Batch) -> None:
        state.out_queues[lane].append(batch)
        self.accountant.hold(batch)

    def _drain_root(self) -> None:
        root = self.states[self.plan.root]
        for queue in root.out_queues:
            while queue:
                batch = queue.popleft()
                self.accountant.release(batch)
                self.results.append(batch)

    def _drop_in_flight(self) -> None:
        """Release every batch still queued. Nothing will consume them now."""
        for state in self.states:
            for queue in state.out_queues:
                while queue:
                    self.accountant.release(queue.popleft())

    def _record(self, state: NodeState, lane: int, call: str, n_out: int) -> None:
        self.trace.append(TraceEvent(self.steps, str(state.info), lane, call, n_out))

    def _assert_drained(self) -> None:
        stranded = [
            f"{s.info} lane {lane} holds {len(q)}"
            for s in self.states
            for lane, q in enumerate(s.out_queues)
            if q
        ]
        if stranded:
            raise DriverError("nothing runnable but batches remain: " + "; ".join(stranded))


def batch_partitioned_driver(
    plan: Plan, selector: BackendSelector, budget: int | None = None
) -> BatchPartitionedDriver:
    """Constructor spelled as the driver name the spec uses."""
    return BatchPartitionedDriver(plan, selector, budget)
