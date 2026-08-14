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
            self._accountant.register(self.label, self._executor)
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

    def step(self, inputs: LaneInputs) -> StepResult:
        if self.finished:
            raise DriverError(f"{self.label}: stepped after finishing")
        category = self.info.category
        if category is ExecutorCategory.SOURCE:
            return self._step_source()
        if category is ExecutorCategory.EXEC:
            return self._step_exec(inputs)
        if category is ExecutorCategory.BATCH_ACCUMULATOR:
            return self._step_batch_accumulator(inputs)
        if category is ExecutorCategory.JOIN:
            return self._step_join(inputs)
        raise DriverError(f"{self.label}: {category.value} is not lane-scoped")

    # -- per category ------------------------------------------------------------

    def _step_source(self) -> StepResult:
        self._check(0, 0)
        produced = self.executor.next_batch()
        if produced is None:
            self.finished = True
            return StepResult([], True, "next_batch/exhausted")
        batch, _stats = produced
        return StepResult([batch], False, "next_batch")

    def _step_exec(self, inputs: LaneInputs) -> StepResult:
        if not inputs.has(0):
            self.finished = True
            return StepResult([], True, "done")
        batch = inputs.take(0)
        self._check(batch.num_rows(), batch.byte_size())
        out, _stats = self.executor.exec(batch)
        return StepResult([out], False, "exec")

    def _step_batch_accumulator(self, inputs: LaneInputs) -> StepResult:
        if not inputs.has(0):
            self._check(0, 0)
            outputs, _stats = self.executor.mark_done_and_fetch()
            self.finished = True
            return StepResult(list(outputs), True, "mark_done_and_fetch")
        batch = inputs.take(0)
        self._check(batch.num_rows(), batch.byte_size())
        outputs, _stats = self.executor.accumulate_and_fetch(batch)
        return StepResult(list(outputs), False, "accumulate_and_fetch")

    def _step_join(self, inputs: LaneInputs) -> StepResult:
        if self.join_phase is JoinPhase.BUILD:
            # The build side declares SingleBatch, so exactly one batch arrives here —
            # GpuCoalesceAllBatches emits one even when it accumulated nothing. Both
            # violations are the plan's fault, so both are loud.
            if not inputs.has(BUILD_SLOT):
                raise DriverError(f"{self.label}: build side finished without a batch")
            batch = inputs.take(BUILD_SLOT)
            self._check(batch.num_rows(), batch.byte_size())
            self.executor.set_build(batch)
            self.join_phase = JoinPhase.PROBE
            return StepResult([], False, "set_build")

        if inputs.has(BUILD_SLOT):
            raise DriverError(f"{self.label}: build side produced a second batch")
        if not inputs.has(PROBE_SLOT):
            self._check(0, 0)
            outputs, _stats = self.executor.finish_and_fetch()
            self.finished = True
            return StepResult(list(outputs), True, "finish_and_fetch")
        batch = inputs.take(PROBE_SLOT)
        self._check(batch.num_rows(), batch.byte_size())
        outputs, _stats = self.executor.probe_and_fetch(batch)
        return StepResult(list(outputs), False, "probe_and_fetch")

    def _check(self, n_rows: int, n_bytes: int) -> None:
        self._accountant.check_call(self.label, self.executor, n_rows, n_bytes)


def batch_single_partition_driver(
    info: PlanNodeInfo,
    lane: int,
    make_executor: Callable[[], Executor],
    accountant: ResidentAccountant,
) -> BatchSinglePartitionDriver:
    """Constructor spelled as the driver name the spec uses."""
    return BatchSinglePartitionDriver(info, lane, make_executor, accountant)
