"""End-to-end shapes through `batch_partitioned_driver`, with mock executors only."""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from .harness import main, raises
from ..batch_partitioned_driver import batch_partitioned_driver
from ..errors import ResidentBudgetExceeded
from ..plan import Plan
from .mocks import (
    MockSelector,
    coalesce_all,
    coalesce_target,
    emit_partitions,
    eager_merge_partitions,
    even_router,
    exec_node,
    interleave,
    join,
    merge_partitions,
    merge_sorted_partitions,
    sink,
    skew_router,
    source,
    union,
)


def run(root, budget=None):
    """Runs a plan and checks the queue bound on every shape the suite exercises.

    What this actually guards is **executor emission discipline** — that no executor
    returns more than one batch per call per output lane — not the scheduler. The
    scheduling half can no longer go red: a producer's parent is at strictly lower height
    so it drains first, and the transitive join hold closes the one exception, covering
    the producer too since anything under a held node is reached through the same probe
    edge. `test_the_queue_bound_assertion_is_live` is the input that does turn it red.
    """
    driver = batch_partitioned_driver(Plan.build(root), MockSelector(), budget)
    driver.run()
    for info in driver.plan.nodes:
        assert driver.peak_queued[info.id] <= info.n_lanes, f"{info}: queue bound broken"
    # Every plan here drains fully, so every executor finished — and a finished executor
    # stops contributing to the accounted resident set.
    assert driver.accountant.executor_bytes == 0, "a finished executor still counted"
    return driver


def total_rows(driver):
    return sum(b.num_rows() for b in driver.results)


def shuffle_aggregate_plan(lanes=4, batches_per_lane=2, rows=10):
    loaded = source("load", [[rows] * batches_per_lane for _ in range(lanes)])
    partial = exec_node("agg_partial", loaded)
    per_lane = coalesce_all("agg_batches", partial)
    shuffled = emit_partitions(
        "emit", merge_partitions("merge", per_lane), lanes, even_router(lanes)
    )
    return sink("unload", coalesce_all("agg_final", shuffled))


# -- shapes ----------------------------------------------------------------------


def test_scan_filter_aggregate_chain():
    driver = run(sink("unload", exec_node("agg", exec_node("filter", source("load", [[8, 8]])))))
    assert total_rows(driver) == 16
    assert len(driver.results) == 2


def test_two_sided_shuffle_aggregate_preserves_every_row():
    driver = run(shuffle_aggregate_plan())
    assert total_rows(driver) == 80
    assert len(driver.results) == 4  # one final batch per post-shuffle lane


def test_shuffle_join_runs_the_build_side_of_every_lane_first():
    build = coalesce_all("build_collect", source("build_scan", [[4], [4], [4], [4]]))
    probe = source("probe_scan", [[2, 2], [2, 2], [2, 2], [2, 2]])
    driver = run(sink("unload", join("join", build, probe)))

    assert total_rows(driver) == 16  # 8 probe batches × 2 rows
    calls = [e.call for e in driver.trace if e.node.startswith("join")]
    assert calls[:4] == ["set_build"] * 4
    assert calls[-4:] == ["finish_and_fetch"] * 4


def two_sided_shuffle_join(lanes=4):
    def shuffled(prefix, batches):
        loaded = source(f"{prefix}_scan", [batches] * lanes)
        return emit_partitions(
            f"{prefix}_emit",
            merge_partitions(f"{prefix}_merge", loaded),
            lanes,
            even_router(lanes),
        )

    build = coalesce_all("build_collect", shuffled("build", [4]))
    return sink("unload", join("join", build, shuffled("probe", [4, 4])))


def test_a_join_with_a_shuffle_on_both_sides_co_partitions_and_completes():
    driver = run(two_sided_shuffle_join())
    assert total_rows(driver) == 32  # 4 lanes × 2 batches × 4 rows, probe-side preserved


def test_probe_side_queues_stay_empty_until_the_build_is_set():
    # Without the join hold this shape put 32 batches (4 lanes × 2 batches, scattered 4
    # ways) into the probe-side emit before the first set_build, because the build-side
    # coalesce makes that subtree one level deeper. Held, the probe subtree has not run
    # at all when the build phase ends.
    driver = batch_partitioned_driver(Plan.build(two_sided_shuffle_join()), MockSelector())
    while not any(e.call == "set_build" for e in driver.trace):
        assert driver.step(), "the join never reached its build phase"

    probe_side = [s for s in driver.states if s.info.node.name().startswith("probe_")]
    assert probe_side
    assert all(state.queued_batches() == 0 for state in probe_side)

    driver.run()
    peak = {s.info.node.name(): driver.peak_queued[s.info.id] for s in driver.states}
    assert all(peak[name] <= 4 for name in peak), peak


def test_left_outer_join_finish_pass_reaches_the_root():
    build = coalesce_all("build_collect", source("build_scan", [[4]]))
    probe = source("probe_scan", [[2, 2]])
    driver = run(sink("unload", join("join", build, probe, emit_on_finish=3)))
    assert total_rows(driver) == 2 + 2 + 3


def test_top_n_sort_merges_every_lane_into_one():
    sorted_lanes = exec_node("sort", source("load", [[3, 3], [3], [3, 3, 3]]))
    driver = run(sink("unload", merge_sorted_partitions("merge_sorted", sorted_lanes)))
    assert len(driver.results) == 1
    assert total_rows(driver) == 18


def test_keyless_aggregate_skips_the_shuffle():
    collapsed = coalesce_all("agg_all", merge_partitions("merge", source("load", [[5], [5]])))
    driver = run(sink("unload", collapsed))
    assert len(driver.results) == 1
    assert total_rows(driver) == 10


def test_union_relabels_lanes_and_forwards_every_batch():
    left = exec_node("left_filter", source("left_scan", [[1, 1], [1]]))
    right = exec_node("right_filter", source("right_scan", [[1], [1, 1, 1]]))
    driver = run(sink("unload", union("union", [left, right])))
    assert len(driver.results) == 7
    assert driver.plan[driver.plan.root].n_lanes == 4


def test_a_three_branch_union_forwards_every_batch():
    branches = [exec_node(f"f{i}", source(f"s{i}", [[1, 1]])) for i in range(3)]
    driver = run(sink("unload", union("union", branches)))
    assert driver.plan[driver.plan.root].n_lanes == 3
    assert len(driver.results) == 6


def test_a_finished_join_releases_its_build_side():
    # The build side is executor residency while the join runs and must leave the
    # accounted total at finish — on the GPU nothing stays resident after the last call.
    build = coalesce_all("build_collect", source("build_scan", [[4]]))
    driver = run(sink("unload", join("join", build, source("probe_scan", [[2, 2]]))))
    state = next(s for s in driver.states if s.info.node.name() == "join")
    assert state.lane_drivers[0].executor.resident_bytes() == 0


def test_interleave_preserves_the_lane_count_and_rotates_children():
    left = exec_node("left_filter", source("left_scan", [[1, 1], [1, 1]]))
    right = exec_node("right_filter", source("right_scan", [[1, 1], [1, 1]]))
    driver = run(sink("unload", interleave("interleave", [left, right])))
    assert len(driver.results) == 8
    assert driver.plan[driver.plan.root].n_lanes == 2


def test_cross_join_beside_an_aggregate():
    scalar = coalesce_all("scalar_agg", merge_partitions("merge", source("agg_scan", [[1], [1]])))
    facts = merge_partitions("fact_merge", source("fact_scan", [[4, 4], [4]]))
    driver = run(sink("unload", join("cross", scalar, facts)))
    assert total_rows(driver) == 12


# -- stress ----------------------------------------------------------------------


def test_empty_partitions_do_not_stall_the_others():
    driver = run(sink("unload", exec_node("filter", source("load", [[], [3, 3], [], [3]]))))
    assert total_rows(driver) == 9
    assert len(driver.results) == 3


def test_operators_emitting_empty_batches_are_carried_through():
    driver = run(sink("unload", exec_node("filter", source("load", [[5, 5]]), selectivity=0.0)))
    assert len(driver.results) == 2
    assert total_rows(driver) == 0


def test_a_skewed_shuffle_drops_the_empty_lanes_at_the_emit():
    shuffled = emit_partitions(
        "emit", merge_partitions("merge", source("load", [[6], [6]])), 4, skew_router(4, 2)
    )
    driver = run(sink("unload", coalesce_all("agg_final", shuffled)))

    # Two input batches, all rows to lane 2: three lanes see nothing at all, and the one
    # final batch each of them emits carries no rows.
    assert total_rows(driver) == 12
    assert sorted(b.num_rows() for b in driver.results) == [0, 0, 0, 12]


def test_a_target_coalescer_is_tolerated_anywhere_in_the_tree():
    def above_source(loaded):
        return exec_node("project", exec_node("filter", coalesce_target("coalesce", loaded, 5)))

    def mid_chain(loaded):
        return exec_node("project", coalesce_target("coalesce", exec_node("filter", loaded), 5))

    def below_sink(loaded):
        return coalesce_target("coalesce", exec_node("project", exec_node("filter", loaded)), 5)

    for position in (above_source, mid_chain, below_sink):
        tree = position(source("load", [[2, 2, 2, 2], [2, 2]]))
        driver = run(sink("unload", tree))
        assert total_rows(driver) == 12, position.__name__


def test_a_target_coalescer_above_a_shuffle_still_delivers_every_row():
    shuffled = emit_partitions(
        "emit", merge_partitions("merge", source("load", [[8], [8], [8], [8]])), 4, even_router(4)
    )
    driver = run(sink("unload", coalesce_target("coalesce", shuffled, 3)))
    assert total_rows(driver) == 32


def test_nested_shuffles_hold_every_bound_at_once():
    first = emit_partitions(
        "emit1", merge_partitions("merge1", source("load", [[4], [4]])), 4, even_router(4)
    )
    second = emit_partitions(
        "emit2",
        merge_partitions("merge2", coalesce_all("agg1", first)),
        2,
        even_router(2),
    )
    driver = run(sink("unload", coalesce_all("agg2", second)))
    assert total_rows(driver) == 8
    for info in driver.plan.nodes:
        assert driver.peak_queued[info.id] <= info.n_lanes, info


def test_the_queue_bound_assertion_is_live():
    # Driven directly rather than through run(), because run() is the thing under test.
    # An accumulator emitting on every lane event puts 4 batches into its 1-lane output
    # queue in a single step, so the bound the helper asserts is genuinely reachable —
    # and #138's ranged merge emission would reach it the same way, from real code.
    plan = Plan.build(sink("unload", eager_merge_partitions("merge_eager", source("load", [[1], [1], [1], [1]]))))
    driver = batch_partitioned_driver(plan, MockSelector())
    driver.run()

    merge = next(i for i in plan.nodes if i.node.name() == "merge_eager")
    assert merge.n_lanes == 1
    assert driver.peak_queued[merge.id] == 4


def test_no_batch_is_ever_stranded():
    # `run()` fails loudly when nothing is runnable and a queue is non-empty, which is
    # the deadlock the height rule plus build-left orientation is meant to rule out.
    driver = run(shuffle_aggregate_plan(lanes=3, batches_per_lane=3, rows=7))
    assert all(state.all_done() for state in driver.states)


# -- accounting ------------------------------------------------------------------


def test_in_flight_bytes_return_to_zero_and_the_peak_is_recorded():
    driver = run(shuffle_aggregate_plan())
    assert driver.accountant.in_flight_bytes == 0
    assert driver.accountant.peak > 0


def test_the_enforcer_trips_cleanly_on_a_tight_budget():
    with raises(ResidentBudgetExceeded):
        run(shuffle_aggregate_plan(lanes=4, batches_per_lane=8, rows=100), budget=1_000)


def test_a_generous_budget_never_trips():
    driver = run(shuffle_aggregate_plan(), budget=10_000_000)
    assert driver.accountant.peak <= 10_000_000


if __name__ == "__main__":
    raise SystemExit(main(globals()))
