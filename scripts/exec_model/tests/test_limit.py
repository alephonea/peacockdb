"""`start..limit` — both lowerings, and the transfers the root-adjacent one avoids.

A limit is decided by position. Feeding only `GpuUnload` it is not a node at all: `skip`
and `fetch` are the unload's, and the driver counts rows across lanes, releases batches it
wants nothing from, narrows the calls that straddle the interval's ends, and stops
scheduling once `is_satisfied` holds. Anywhere else it stays a node over one merged
partition and streams: batches outside the interval are released uncalled, batches inside
are forwarded untouched, and only the two that straddle its ends are sliced.

**These tests assert on the `unload` calls, not on the rows that come back.** Both are the
same for a correct implementation, but only the first can tell the difference between a
limit and a filter applied after the fact — and that difference is the whole feature. A
batch unloaded and then discarded has already crossed PCIe.

Mock nodes throughout: what is under test is scheduling, not arithmetic.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from . import mocks
from .harness import main, raises
from .mocks import MockSelector
from ..batch_partitioned_driver import batch_partitioned_driver
from ..errors import PlanError
from ..limit import RowInterval, RowRange
from ..plan import Plan


def run(root, budget=None):
    driver = batch_partitioned_driver(Plan.build(root), MockSelector(), budget)
    driver.run()
    return driver


def rows(driver) -> int:
    return sum(batch.num_rows() for batch in driver.results)


def unload_calls(driver) -> list:
    """Every range the sink's executors were asked for, lane by lane, in call order."""
    sink = driver.states[driver.plan.root]
    return [call for lane in sorted(sink.lane_drivers) for call in sink.lane_drivers[lane].executor.calls]


def limit_executor(driver, name: str = "limit"):
    """The mid-plan limit's executor, for asserting what it did rather than what came out."""
    state = next(s for s in driver.states if s.info.node.name() == name)
    return state.lane_drivers[0].executor


def source_calls(driver, name: str) -> int:
    return sum(1 for event in driver.trace if event.node.startswith(name + "#"))


# -- the root-adjacent lowering ---------------------------------------------------


def test_a_root_adjacent_limit_is_not_a_node():
    # The interval is the unload's, which is also what puts it in a plan golden rather
    # than in a side channel beside the plan.
    scan = mocks.source("scan", [[10, 10, 10]])
    plan = Plan.build(mocks.sink("unload", scan, fetch=5))
    assert plan.row_limit == RowInterval(0, 5)
    assert [info.node.name() for info in plan.nodes] == ["unload", "scan"]


def test_a_limit_node_feeding_only_the_sink_is_rejected():
    # Because it would be the wrong lowering: the driver could not release a batch it
    # wants nothing from if a node between it and the sink had already unloaded it.
    scan = mocks.source("scan", [[10]])
    limited = mocks.limit("limit", mocks.coalesce_all("collect", scan), fetch=5)
    with raises(PlanError, match="skip/fetch on GpuUnload"):
        Plan.build(mocks.sink("unload", limited))


def test_a_root_adjacent_limit_stops_the_run_early():
    scan = mocks.source("scan", [[4] * 10])
    driver = run(mocks.sink("unload", scan, fetch=5))
    assert rows(driver) == 5
    assert driver.early_exit
    assert source_calls(driver, "scan") == 2


def test_without_the_limit_the_same_plan_drains_everything():
    # The comparison that makes the test above mean something.
    scan = mocks.source("scan", [[4] * 10])
    driver = run(mocks.sink("unload", scan))
    assert rows(driver) == 40
    assert not driver.early_exit
    assert source_calls(driver, "scan") == 11    # ten batches, then the exhaustion pass


def test_the_skip_prefix_is_never_unloaded():
    # The unbounded saving, and the reason the interval sits on the unload at all: with a
    # trim after the fact all five leading batches would have crossed the boundary first.
    scan = mocks.source("scan", [[4] * 6])
    driver = run(mocks.sink("unload", scan, skip=20, fetch=3))
    assert unload_calls(driver) == [RowRange(0, 3)]
    assert driver.rows_skipped[driver.plan.root] == 20
    assert rows(driver) == 3


def test_only_the_straddling_batches_are_narrowed():
    # skip=2, fetch=8 over batches of 4: take the tail of the first, all of the second,
    # the head of the third, and never look at the fourth.
    scan = mocks.source("scan", [[4, 4, 4, 4]])
    driver = run(mocks.sink("unload", scan, skip=2, fetch=8))
    assert unload_calls(driver) == [RowRange(2, 2), None, RowRange(0, 2)]
    assert rows(driver) == 8


def test_a_batch_wholly_inside_the_interval_is_unloaded_whole():
    # No range means no narrowing: `peacock_result_from_handle` exports the whole table,
    # which is the case the new arguments must not make more expensive.
    scan = mocks.source("scan", [[4, 4]])
    driver = run(mocks.sink("unload", scan, fetch=100))
    assert unload_calls(driver) == [None, None]


def test_the_early_exit_reaches_through_a_shuffle():
    # The scheduler is what stops, so every node below stops with it however many there are.
    scan = mocks.source("scan", [[4] * 8, [4] * 8])
    merged = mocks.merge_partitions("merge", scan)
    emitted = mocks.emit_partitions("emit", merged, 2, mocks.even_router(2))
    collapsed = mocks.merge_partitions("collapse", emitted)
    driver = run(mocks.sink("unload", collapsed, fetch=6))
    assert rows(driver) == 6
    assert driver.early_exit
    assert source_calls(driver, "scan") < 16


def test_every_in_flight_batch_is_released_at_the_early_exit():
    # On the GPU each is a handle that must be neither leaked nor double-freed.
    scan = mocks.source("scan", [[4] * 10])
    driver = run(mocks.sink("unload", scan, fetch=5))
    assert driver.accountant.in_flight_bytes == 0


def test_the_count_is_across_lanes():
    # Four lanes of one batch each, and a limit of 6: two lanes are unloaded (one of them
    # narrowed) and the rest are released. A per-lane count would take 6 rows from every
    # lane and return 24.
    scan = mocks.source("scan", [[4], [4], [4], [4]])
    driver = run(mocks.sink("unload", scan, fetch=6))
    assert rows(driver) == 6
    assert unload_calls(driver) == [None, RowRange(0, 2)]


def test_a_limit_the_data_never_reaches_returns_everything_it_had():
    scan = mocks.source("scan", [[4, 4]])
    driver = run(mocks.sink("unload", scan, fetch=1000))
    assert rows(driver) == 8
    assert not driver.early_exit


def test_a_skip_past_the_end_returns_nothing_and_unloads_nothing():
    scan = mocks.source("scan", [[4, 4]])
    driver = run(mocks.sink("unload", scan, skip=100, fetch=10))
    assert driver.results == []
    assert unload_calls(driver) == []


def test_an_offset_with_no_fetch_never_exits_early():
    # No prefix determines an unbounded interval, so it can only drop and trim.
    scan = mocks.source("scan", [[4] * 5])
    driver = run(mocks.sink("unload", scan, skip=6))
    assert rows(driver) == 14
    assert not driver.early_exit
    assert driver.rows_skipped[driver.plan.root] == 4    # only the first batch is wholly skipped


def test_a_zero_fetch_unloads_nothing_at_all():
    scan = mocks.source("scan", [[4] * 10])
    driver = run(mocks.sink("unload", scan, fetch=0))
    assert driver.results == []
    assert unload_calls(driver) == []
    # Satisfied before anything ran: the scan is never asked for a batch at all.
    assert source_calls(driver, "scan") == 0


# -- the mid-plan lowering --------------------------------------------------------


def test_a_mid_plan_limit_stays_in_the_tree_and_answers_from_a_prefix():
    scan = mocks.source("scan", [[4, 4, 4]])
    limited = mocks.limit("limit", scan, fetch=5)
    driver = run(mocks.sink("unload", mocks.exec_node("after", limited)))
    assert driver.plan.row_limit is None            # the sink absorbed nothing
    assert rows(driver) == 5


def test_a_mid_plan_limit_does_not_read_its_whole_input():
    # The shape that motivates all of this:
    #     customer JOIN (SELECT * FROM orders LIMIT 100) ON ...
    # A hundred rows are wanted, so a hundred rows plus the batch they arrived in are
    # read. Requiring the limit's input to be a single batch would put a
    # GpuCoalesceAllBatches underneath and scan the whole of orders to answer.
    orders = mocks.source("orders", [[10] * 100])          # a thousand rows available
    limited = mocks.limit("limit", orders, fetch=100)
    build = mocks.coalesce_all("collect", limited)
    driver = run(mocks.sink("unload", build))
    assert rows(driver) == 100
    assert source_calls(driver, "orders") == 10            # ten batches, not a hundred
    assert driver.early_exit


def test_a_mid_plan_limit_holds_nothing_at_all():
    # It streams: batches outside the interval are released, batches inside are forwarded
    # untouched, and only the two that straddle its ends are sliced. Nothing is held, so
    # residency does not depend on the interval or on the input.
    orders = mocks.source("orders", [[10] * 100])
    limited = mocks.limit("limit", orders, fetch=100)
    driver = run(mocks.sink("unload", mocks.exec_node("after", limited)))
    executor = limit_executor(driver)
    assert executor.resident_bytes() == 0
    assert executor.sliced == []          # the boundaries happen to align
    assert executor.passed == 10
    assert rows(driver) == 100


def test_a_large_offset_streams_its_prefix_instead_of_holding_it():
    # The case the slice symbol exists for. With bounds frozen in the plan node they would
    # only be correct against a table starting at row 0 of the stream, so the whole offset
    # prefix would have to be held — for OFFSET 1000000 LIMIT 10, a million rows to return
    # ten. Runtime bounds mean the prefix is released a batch at a time.
    orders = mocks.source("orders", [[10] * 100])
    limited = mocks.limit("limit", orders, skip=55, fetch=10)
    driver = run(mocks.sink("unload", mocks.exec_node("after", limited)))
    executor = limit_executor(driver)
    assert executor.dropped == 5                                  # rows 0..49, never held
    assert executor.sliced == [RowRange(5, 5), RowRange(0, 5)]    # only the two edges
    assert executor.resident_bytes() == 0
    assert rows(driver) == 10
    assert source_calls(driver, "orders") == 7                    # not 100


def test_a_mid_plan_limit_needs_one_partition_but_not_one_batch():
    # An interval over four lanes names no rows, so the planner puts a merge under it.
    # Several batches on that one lane are fine, and are the point.
    scan = mocks.source("scan", [[4], [4], [4], [4]])
    with raises(PlanError, match="GpuMergePartitions"):
        Plan.build(mocks.sink("unload", mocks.limit("limit", scan, fetch=5)))

    merged = mocks.merge_partitions("merge", mocks.source("scan", [[4], [4], [4], [4]]))
    limited = mocks.limit("limit", merged, fetch=5)
    driver = run(mocks.sink("unload", mocks.exec_node("after", limited)))
    assert rows(driver) == 5


def test_a_mid_plan_limit_of_zero_does_not_freeze_the_plan():
    # Satisfied before a single node runs, which would hold its own subtree from step zero
    # and leave its parent waiting for a lane that had finished. The plan must complete and
    # return nothing, not stall — and it must not read anything either.
    scan = mocks.source("scan", [[4, 4, 4]])
    limited = mocks.limit("limit", scan, fetch=0)
    driver = run(mocks.sink("unload", mocks.exec_node("after", limited)))
    assert rows(driver) == 0
    assert source_calls(driver, "scan") == 0


# -- the interval itself ----------------------------------------------------------


def test_an_interval_rejects_negative_bounds():
    with raises(ValueError, match="skip"):
        RowInterval(skip=-1)
    with raises(ValueError, match="fetch"):
        RowInterval(fetch=-1)


def test_the_range_of_a_batch_is_relative_to_what_came_before():
    interval = RowInterval(skip=5, fetch=4)
    assert interval.range_of(seen=0, n_rows=4) is None       # wholly inside the skip
    assert interval.range_of(seen=4, n_rows=4) == RowRange(1, 3)
    assert interval.range_of(seen=8, n_rows=4) == RowRange(0, 1)
    assert interval.range_of(seen=12, n_rows=4) is None      # wholly past the stop


def test_an_unbounded_interval_only_ever_trims_its_front():
    interval = RowInterval(skip=3)
    assert interval.range_of(seen=0, n_rows=2) is None
    assert interval.range_of(seen=2, n_rows=4) == RowRange(1, 3)
    assert interval.range_of(seen=6, n_rows=4) == RowRange(0, 4)
    assert not interval.satisfied_by(1_000_000)


if __name__ == "__main__":
    raise SystemExit(main(globals()))
