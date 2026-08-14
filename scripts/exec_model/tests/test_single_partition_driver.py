"""`batch_single_partition_driver`: one lane's state machine, in isolation."""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from collections import deque

from .harness import main, raises
from ..accounting import ResidentAccountant
from ..batch_single_partition_driver import BatchSinglePartitionDriver, JoinPhase
from ..errors import DriverError
from ..layout import NodeKind, PartitionLayout
from ..node import ExecutorCategory
from ..plan import PlanNodeInfo
from ..runtime import LaneInputs
from .mocks import (
    CollectAccumulator,
    MapExec,
    MockBatch,
    MockNode,
    RecordingJoin,
    ScriptedSource,
)


class FakeProducer:
    """The two attributes `LaneInputs` reads off a producing node."""

    def __init__(self, batches=(), done=False):
        self.out_queues = [deque(batches)]
        self.out_done = [done]

    def finish(self):
        self.out_done[0] = True


class _Executors:
    """Stands in for `NodeExecutors`; the lane driver reads only `.category`."""

    def __init__(self, category):
        self.category = category
        self.backends = None
        self.forwarder = None


def make_info(category, name="node", kind=NodeKind.INTERMEDIATE):
    return PlanNodeInfo(
        id=0,
        node=MockNode(name, kind, PartitionLayout(n=1), category),
        executors=_Executors(category),
        height=0,
        order=0,
        parent=None,
        child_slot=None,
        children=(),
        n_lanes=1,
    )


def make_driver(category, executor, name="node"):
    accountant = ResidentAccountant()
    driver = BatchSinglePartitionDriver(
        make_info(category, name), 0, lambda: executor, accountant
    )
    return driver, accountant


def inputs_over(producers, accountant):
    return LaneInputs([(p, 0) for p in producers], accountant)


def test_source_runs_until_it_returns_none():
    executor = ScriptedSource([MockBatch("a", 1), MockBatch("b", 1)])
    driver, accountant = make_driver(ExecutorCategory.SOURCE, executor)
    empty = inputs_over([], accountant)

    tags = []
    while driver.can_step(empty):
        result = driver.step(empty)
        tags += [b.tag for b in result.outputs]
    assert tags == ["a", "b"]
    assert driver.finished


def test_exec_is_one_call_per_batch_then_propagates_done():
    producer = FakeProducer([MockBatch("in", 4)])
    driver, accountant = make_driver(ExecutorCategory.EXEC, MapExec("filter", selectivity=0.5))
    lane_inputs = inputs_over([producer], accountant)

    first = driver.step(lane_inputs)
    assert [b.tag for b in first.outputs] == ["in>filter"]
    assert first.outputs[0].num_rows() == 2
    assert not first.finished

    assert not driver.can_step(lane_inputs)  # queue empty, producer still live
    producer.finish()
    final = driver.step(lane_inputs)
    assert final.outputs == [] and final.finished


def test_batch_accumulator_emits_only_at_done():
    producer = FakeProducer([MockBatch("x", 3), MockBatch("y", 4)])
    driver, accountant = make_driver(
        ExecutorCategory.BATCH_ACCUMULATOR, CollectAccumulator("collect")
    )
    lane_inputs = inputs_over([producer], accountant)

    assert driver.step(lane_inputs).outputs == []
    assert driver.step(lane_inputs).outputs == []
    producer.finish()
    final = driver.step(lane_inputs)
    assert [b.tag for b in final.outputs] == ["[x+y]>collect"]
    assert final.outputs[0].num_rows() == 7
    assert final.finished


def test_join_sets_build_before_it_probes():
    build = FakeProducer([MockBatch("B", 2)], done=True)
    probe = FakeProducer([MockBatch("p0", 1), MockBatch("p1", 1)])
    driver, accountant = make_driver(ExecutorCategory.JOIN, RecordingJoin("join"))
    lane_inputs = inputs_over([build, probe], accountant)

    assert driver.join_phase is JoinPhase.BUILD
    assert driver.step(lane_inputs).call == "set_build"
    assert driver.join_phase is JoinPhase.PROBE

    tags = []
    for _ in range(2):
        tags += [b.tag for b in driver.step(lane_inputs).outputs]
    assert tags == ["(B⋈p0)>join", "(B⋈p1)>join"]

    probe.finish()
    final = driver.step(lane_inputs)
    assert final.call == "finish_and_fetch" and final.finished


def test_join_left_outer_emits_its_unmatched_build_rows_at_finish():
    build = FakeProducer([MockBatch("B", 2)], done=True)
    probe = FakeProducer([MockBatch("p", 1)], done=True)
    driver, accountant = make_driver(
        ExecutorCategory.JOIN, RecordingJoin("join", emit_on_finish=3)
    )
    lane_inputs = inputs_over([build, probe], accountant)

    driver.step(lane_inputs)  # set_build
    driver.step(lane_inputs)  # probe
    final = driver.step(lane_inputs)
    assert [b.num_rows() for b in final.outputs] == [3]


def test_join_build_side_that_produced_nothing_is_an_error():
    build = FakeProducer([], done=True)
    probe = FakeProducer([MockBatch("p", 1)])
    driver, accountant = make_driver(ExecutorCategory.JOIN, RecordingJoin("join"))
    lane_inputs = inputs_over([build, probe], accountant)

    with raises(DriverError, match="build side finished without a batch"):
        driver.step(lane_inputs)


def test_join_build_side_that_produced_two_batches_is_an_error():
    build = FakeProducer([MockBatch("B0", 1), MockBatch("B1", 1)], done=True)
    probe = FakeProducer([MockBatch("p", 1)])
    driver, accountant = make_driver(ExecutorCategory.JOIN, RecordingJoin("join"))
    lane_inputs = inputs_over([build, probe], accountant)

    driver.step(lane_inputs)
    with raises(DriverError, match="second batch"):
        driver.step(lane_inputs)


def test_stepping_a_finished_lane_is_an_error():
    executor = ScriptedSource([])
    driver, accountant = make_driver(ExecutorCategory.SOURCE, executor)
    empty = inputs_over([], accountant)
    driver.step(empty)
    with raises(DriverError, match="stepped after finishing"):
        driver.step(empty)


def test_the_executor_is_built_on_the_first_step_not_before():
    built = []

    def factory():
        built.append(1)
        return ScriptedSource([MockBatch("a", 1)])

    info = make_info(ExecutorCategory.SOURCE, "load", NodeKind.SOURCE)
    accountant = ResidentAccountant()
    driver = BatchSinglePartitionDriver(info, 0, factory, accountant)
    empty = LaneInputs([], accountant)

    assert driver.can_step(empty)
    assert built == []
    driver.step(empty)
    assert built == [1]


if __name__ == "__main__":
    raise SystemExit(main(globals()))
