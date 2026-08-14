"""`batch_single_partition_driver` — one lane of one lane-scoped node.

The four lane-scoped categories (Source, Exec, BatchAccumulator, Join) get one executor
instance per (node, lane), and this driver is that instance's state machine: it decides
which executor call the lane's current input state calls for, makes exactly one call, and
reports the outputs plus whether the lane will ever produce again.

Everything cross-partition — which node runs next, lane remapping, the fan-out at an
emitter, the rotation at a forwarder — is `batch_partitioned_driver`'s. The split is the
one the spec draws; what changed with the height scheduler is the *unit*: a chunk is one
node's lane rather than a chain of them, because min-height selection walks a batch up
the chain node by node all on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .accounting import ResidentAccountant
from .batch import Batch
from .errors import DriverError
from .executors import Executor
from .node import ExecutorCategory
from .plan import PlanNodeInfo
from .runtime import LaneInputs

BUILD_SLOT = 0
PROBE_SLOT = 1


class JoinPhase(Enum):
    BUILD = "build"
    PROBE = "probe"


@dataclass
class StepResult:
    outputs: list[Batch]
    #: the lane is finished — it will never produce another batch
    finished: bool
    #: what the executor was asked to do, for traces and tests
    call: str


class BatchSinglePartitionDriver:
    """One (node, lane). The executor is constructed on the first step, not before."""

    def __init__(
        self,
        info: PlanNodeInfo,
        lane: int,
        make_executor: Callable[[], Executor],
        accountant: ResidentAccountant,
    ):
        self.info = info
        self.lane = lane
        self._make_executor = make_executor
        self._accountant = accountant
        self._executor: Executor | None = None
        self.finished = False
        self.join_phase = JoinPhase.BUILD

    @property
    def label(self) -> str:
        return f"{self.info}/lane{self.lane}"

    @property
    def executor(self) -> Executor:
        if self._executor is None:
            self._executor = self._make_executor()
        return self._executor

    def can_step(self, inputs: LaneInputs) -> bool:
        if self.finished:
            return False
        category = self.info.category
        if category is ExecutorCategory.SOURCE:
            return True
        if category is ExecutorCategory.JOIN:
            slot = BUILD_SLOT if self.join_phase is JoinPhase.BUILD else PROBE_SLOT
            return inputs.has(slot) or inputs.done(slot)
        return inputs.has(0) or inputs.done(0)

    def step(self, inputs: LaneInputs, rows=None) -> StepResult:
        """`rows` is the `RowRange` for an unload; every other category ignores it."""
        if self.finished:
            raise DriverError(f"{self.label}: stepped after finishing")
        result = self._dispatch(inputs, rows)
        if result.finished:
            # A finished executor stops contributing to the accounted resident set: the
            # enforcer's total is over live executors.
            self._accountant.forget(self.label)
        return result

    def _dispatch(self, inputs: LaneInputs, rows) -> StepResult:
        category = self.info.category
        if category is ExecutorCategory.SOURCE:
            return self._step_source()
        if category is ExecutorCategory.UNLOAD:
            return self._step_unload(inputs, rows)
        if category is ExecutorCategory.EXEC:
            return self._step_exec(inputs)
        if category is ExecutorCategory.BATCH_ACCUMULATOR:
            return self._step_batch_accumulator(inputs)
        if category is ExecutorCategory.JOIN:
            return self._step_join(inputs)
        raise DriverError(f"{self.label}: {category.value} is not lane-scoped")

    # -- per category ------------------------------------------------------------

    def _step_source(self) -> StepResult:
        modelled = self._accountant.begin_call(self.label, self.executor, 0, 0)
        produced = self.executor.next_batch()
        if produced is None:
            self._accountant.end_call(self.label, self.executor)
            self.finished = True
            return StepResult([], True, "next_batch/exhausted")
        batch, stats = produced
        self._accountant.end_call(self.label, self.executor, stats, modelled)
        return StepResult([batch], False, "next_batch")

    def _step_exec(self, inputs: LaneInputs) -> StepResult:
        if not inputs.has(0):
            self.finished = True
            return StepResult([], True, "done")
        batch = inputs.take(0)
        outputs, _ = self._call(batch, self._exec_one)
        return StepResult(outputs, False, "exec")

    def _step_unload(self, inputs: LaneInputs, rows) -> StepResult:
        """The boundary crossing, over the row range the partitioned driver chose.

        The driver never sends a batch here that it wants none of — that batch is released
        without a call, which is the whole saving. So a range reaching this point always
        names at least one row.
        """
        if not inputs.has(0):
            self.finished = True
            return StepResult([], True, "done")
        batch = inputs.take(0)
        outputs, _ = self._call(batch, lambda b: self._unload_one(b, rows))
        return StepResult(outputs, False, "unload" if rows is None else "unload/range")

    def _unload_one(self, batch, rows):
        out, stats = self.executor.unload(batch, rows)
        return [out], stats

    def _exec_one(self, batch):
        """`exec` is the one method returning a single batch rather than a list."""
        out, stats = self.executor.exec(batch)
        return [out], stats

    def _step_batch_accumulator(self, inputs: LaneInputs) -> StepResult:
        if not inputs.has(0):
            outputs, _ = self._call(None, lambda _b: self.executor.mark_done_and_fetch())
            self.finished = True
            return StepResult(outputs, True, "mark_done_and_fetch")
        batch = inputs.take(0)
        outputs, _ = self._call(batch, lambda b: self.executor.accumulate_and_fetch(b))
        return StepResult(outputs, False, "accumulate_and_fetch")

    def _step_join(self, inputs: LaneInputs) -> StepResult:
        if self.join_phase is JoinPhase.BUILD:
            # The build side declares SingleBatch, so exactly one batch arrives here —
            # GpuCoalesceAllBatches emits one even when it accumulated nothing. Both
            # violations are the plan's fault, so both are loud.
            if not inputs.has(BUILD_SLOT):
                raise DriverError(f"{self.label}: build side finished without a batch")
            batch = inputs.take(BUILD_SLOT)
            self._call(batch, lambda b: ([], self.executor.set_build(b)))
            self.join_phase = JoinPhase.PROBE
            return StepResult([], False, "set_build")

        if inputs.has(BUILD_SLOT):
            raise DriverError(f"{self.label}: build side produced a second batch")
        if not inputs.has(PROBE_SLOT):
            outputs, _ = self._call(None, lambda _b: self.executor.finish_and_fetch())
            self.finished = True
            return StepResult(outputs, True, "finish_and_fetch")
        batch = inputs.take(PROBE_SLOT)
        outputs, _ = self._call(batch, lambda b: self.executor.probe_and_fetch(b))
        return StepResult(outputs, False, "probe_and_fetch")

    def _call(self, batch, invoke):
        """Pre-check, call, release the consumed input, then refresh residency and post-check.

        The input is released *after* the call, per the spec's accounting order: it is
        alive on the device while the call runs, so the pre-check counts it. `batch` is
        None for the calls the spec models with 0 rows and 0 bytes —
        `mark_done_and_fetch` and `finish_and_fetch`, which take no input.
        """
        n_rows = batch.num_rows() if batch is not None else 0
        n_bytes = batch.byte_size() if batch is not None else 0
        modelled = self._accountant.begin_call(self.label, self.executor, n_rows, n_bytes)
        outputs, stats = invoke(batch)
        if batch is not None:
            self._accountant.release(batch)
        self._accountant.end_call(self.label, self.executor, stats, modelled)
        return list(outputs), stats


def batch_single_partition_driver(
    info: PlanNodeInfo,
    lane: int,
    make_executor: Callable[[], Executor],
    accountant: ResidentAccountant,
) -> BatchSinglePartitionDriver:
    """Constructor spelled as the driver name the spec uses."""
    return BatchSinglePartitionDriver(info, lane, make_executor, accountant)
