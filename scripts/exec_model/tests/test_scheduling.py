"""The scheduling rule itself: min height, ties leftmost, every lane of the chosen node."""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from .harness import main
from ..batch_partitioned_driver import batch_partitioned_driver
from ..plan import Plan
from .mocks import (
    MockSelector,
    coalesce_all,
    emit_partitions,
    even_router,
    exec_node,
    join,
    merge_partitions,
    merge_sorted_partitions,
    sink,
    source,
)


def node_names(driver):
    return [event.node.split("#")[0] for event in driver.trace]


def run(root, budget=None):
    driver = batch_partitioned_driver(Plan.build(root), MockSelector(), budget)
    driver.run()
    return driver


def test_a_batch_is_carried_to_the_root_before_the_next_one_is_produced():
    plan = sink("unload", exec_node("project", exec_node("filter", source("load", [[4, 4]]))))
    driver = run(plan)

    # Two batches, each walked load → filter → project → unload before the next load.
    assert node_names(driver)[:8] == [
        "load",
        "filter",
        "project",
        "unload",
        "load",
        "filter",
        "project",
        "unload",
    ]


def steps_of(driver):
    """The trace grouped by step: one entry per step, listing `node/pN` in visit order."""
    grouped = {}
    for event in driver.trace:
        grouped.setdefault(event.step, []).append(f"{event.node.split('#')[0]}/p{event.lane}")
    return [grouped[step] for step in sorted(grouped)]


def test_the_unit_carried_to_the_root_is_one_batch_per_lane():
    # With siblings present the thing that moves is a wavefront, not a batch: running a
    # node runs every lane of it, so a full N-wide wave walks to the root before the
    # source produces the next one. This is what makes the memory bound one batch per
    # lane rather than one batch.
    driver = run(sink("u", exec_node("project", exec_node("filter", source("load", [[1], [1], [1]])))))

    assert steps_of(driver)[:4] == [
        ["load/p0", "load/p1", "load/p2"],
        ["filter/p0", "filter/p1", "filter/p2"],
        ["project/p0", "project/p1", "project/p2"],
        ["u/p0", "u/p1", "u/p2"],
    ]


def batches_per_step(driver, node_name):
    """Batches `node_name` actually produced, per step it ran in.

    Not the same as how many lanes appear in the trace: a lane whose input is finished
    still runs, to propagate done, and emits an event carrying no batch.
    """
    per_step = {}
    for event in driver.trace:
        if event.node.split("#")[0] == node_name:
            per_step.setdefault(event.step, 0)
            per_step[event.step] += event.n_out
    return [per_step[step] for step in sorted(per_step)]


def test_an_exhausted_lane_narrows_the_wavefront_without_stalling_it():
    # Lane 1 holds one batch, the others two. The wave is 3 batches, then 2 — and the
    # third pass carries none at all, because by then every lane is only finalizing.
    driver = run(sink("u", exec_node("filter", source("load", [[1, 1], [1], [1, 1]]))))
    assert batches_per_step(driver, "load") == [3, 2, 0]


def test_a_collapse_point_narrows_the_wavefront_to_one_batch():
    # Above a merge there is one lane, so the 3-batch wave is consumed one at a time: the
    # merge forwards exactly one per visit and the nodes above it drain it before the
    # source is reached again. The source's second appearance is its exhaustion pass —
    # three lanes, no batches — not a fourth wave.
    driver = run(sink("u", exec_node("project", merge_partitions("merge", source("load", [[1], [1], [1]])))))
    steps = steps_of(driver)

    assert steps[0] == ["load/p0", "load/p1", "load/p2"]
    assert steps[1:10] == [["merge/p0"], ["project/p0"], ["u/p0"]] * 3
    assert steps[10] == ["load/p0", "load/p1", "load/p2"]
    assert batches_per_step(driver, "load") == [3, 0]


def test_choice_is_the_minimum_height_among_runnable_nodes():
    plan = Plan.build(sink("unload", exec_node("filter", source("load", [[1, 1]]))))
    driver = batch_partitioned_driver(plan, MockSelector())

    assert driver.choose().info.node.name() == "load"  # the only runnable node
    driver.step()

    runnable = {info.node.name() for info in driver.runnable_nodes()}
    assert runnable == {"load", "filter"}
    assert driver.choose().info.node.name() == "filter"  # height 1 beats height 2


def test_ties_break_leftmost_so_the_build_subtree_drains_first():
    build = coalesce_all("build_collect", source("build_scan", [[1], [1]]))
    probe = exec_node("probe_filter", source("probe_scan", [[1, 1], [1, 1]]))
    driver = run(sink("unload", join("join", build, probe)))

    names = node_names(driver)
    # build_scan and probe_scan sit at the same height; the build side is the left child,
    # so every build-side call precedes the first probe-side call.
    assert names.index("build_scan") < names.index("probe_scan")
    assert names.index("build_collect") < names.index("probe_filter")
    assert names.index("join") < names.index("probe_filter")


def test_running_a_node_runs_every_one_of_its_partitions():
    plan = Plan.build(sink("unload", source("load", [[1], [1], [1], [1]])))
    driver = batch_partitioned_driver(plan, MockSelector())
    driver.step()

    load = driver.states[[s.info.node.name() for s in driver.states].index("load")]
    assert [len(q) for q in load.out_queues] == [1, 1, 1, 1]
    assert len([e for e in driver.trace if e.node.startswith("load")]) == 4


def test_a_lane_that_cannot_step_is_skipped_not_stalled():
    # lane 1 is empty from the start; lane 0 has two batches.
    driver = run(sink("unload", exec_node("filter", source("load", [[2, 2], []]))))
    tags = sorted(b.tag for b in driver.results)
    assert tags == ["load.p0.b0>filter>unload", "load.p0.b1>filter>unload"]


def test_queues_stay_bounded_by_the_lane_count_without_an_explicit_cap():
    # A shuffle tree with no join: every producer is drained by a strictly lower node
    # before it runs again, so no queue can grow past one batch per lane.
    loaded = source("load", [[6], [6], [6], [6]])
    shuffled = emit_partitions(
        "emit", merge_partitions("merge", exec_node("filter", loaded)), 4, even_router(4)
    )
    driver = run(sink("unload", exec_node("agg", shuffled)))

    for info in driver.plan.nodes:
        assert driver.peak_queued[info.id] <= info.n_lanes, info


def test_a_join_in_its_build_phase_holds_back_its_probe_subtree():
    # Without the hold this is where the bound breaks: the build-side coalesce makes the
    # build subtree deeper, so min-height would hand the probe every choice and run it to
    # exhaustion into a queue the join cannot drain yet. Held, the probe does not start
    # until the build is set, and it never queues more than one batch per lane.
    deep_build = coalesce_all(
        "build_collect",
        exec_node("build_project", exec_node("build_filter", source("build_scan", [[1]]))),
    )
    driver = run(sink("unload", join("join", deep_build, source("probe_scan", [[1] * 8]))))

    probe_id = [s.info.node.name() for s in driver.states].index("probe_scan")
    assert driver.peak_queued[probe_id] == 1

    names = node_names(driver)
    calls = [e.call for e in driver.trace]
    assert names.index("probe_scan") > calls.index("set_build")


def test_the_hold_is_transitive_over_the_whole_probe_subtree():
    # The probe subtree must be SHALLOWER than the build's, or min-height prefers the
    # build unprompted and the test passes with no hold at all. Here probe_scan is at
    # height 4 against build_scan at 6, so every choice goes to the probe unless the hold
    # stops it — and holding only the join's direct child would move the pile one node
    # down rather than remove it, which is what the per-node assertions discriminate.
    build = coalesce_all(
        "build_collect",
        exec_node("b3", exec_node("b2", exec_node("b1", source("build_scan", [[1]])))),
    )
    probe = exec_node("probe_project", exec_node("probe_filter", source("probe_scan", [[1] * 6])))
    driver = run(sink("unload", join("join", build, probe)))

    heights = {s.info.node.name(): s.info.height for s in driver.states}
    assert heights["probe_scan"] < heights["build_scan"], heights

    peak = {s.info.node.name(): driver.peak_queued[s.info.id] for s in driver.states}
    assert peak["probe_project"] == 1  # red if the hold is dropped entirely
    assert peak["probe_filter"] == 1  # red if the hold covers only the direct child
    assert peak["probe_scan"] == 1  # red if it reaches only one level further


def test_the_hold_stays_on_while_any_join_lane_is_still_building():
    # The multi-lane case `_awaits_build`'s quantifier exists for. The build lanes hold
    # 1/2/3/4 batches, so they finish at different steps and set_build lands on four
    # separate ones — lanes already probing while others still build. The probe source is
    # shallower than the build's, so it would win every choice if the hold lifted at the
    # FIRST set_build instead of the last.
    build = coalesce_all("bc", source("bs", [[1], [1, 1], [1, 1, 1], [1, 1, 1, 1]]))
    driver = run(sink("u", join("j", build, source("ps", [[1, 1]] * 4))))

    set_build_steps = [e.step for e in driver.trace if e.call == "set_build"]
    assert len(set(set_build_steps)) == 4, set_build_steps  # genuinely staggered

    first_probe_step = next(e.step for e in driver.trace if e.node.startswith("ps"))
    assert first_probe_step > max(set_build_steps)


def test_a_join_inside_another_join_s_build_subtree_is_not_held():
    # The mirror of the nested-probe case, and the one an over-eager hold would deadlock:
    # the inner join is on the BUILD edge, so nothing holds it — and it has to complete,
    # because the outer join's build side is what lifts every hold below it.
    inner = join("j2", coalesce_all("b2c", source("b2", [[2]])), source("p2", [[3, 3]]))
    outer = join("j1", coalesce_all("b1c", inner), source("p1", [[4, 4, 4]]))
    driver = run(sink("unload", outer))

    assert sum(b.num_rows() for b in driver.results) == 12
    calls = [(e.node.split("#")[0], e.call) for e in driver.trace]
    assert calls.index(("j2", "set_build")) < calls.index(("j1", "set_build"))


def test_a_probe_subtree_containing_a_shuffle_resumes_when_the_hold_lifts():
    # The hold covers cross-lane nodes too, so it is released with a partition
    # accumulator mid-stream — it must resume rather than strand its lane events.
    probe = merge_sorted_partitions(
        "merge_sorted",
        emit_partitions(
            "emit", merge_partitions("merge", source("probe_scan", [[6], [6]])), 4, even_router(4)
        ),
    )
    driver = run(sink("unload", join("join", coalesce_all("bc", source("bs", [[5]])), probe)))
    assert sum(b.num_rows() for b in driver.results) == 12


def test_nested_joins_resolve_outermost_first_without_deadlocking():
    # The inner join lives inside the outer join's probe subtree, so it is held until the
    # outer build is set. The outer build subtree is disjoint from that region — which is
    # why the hold cannot deadlock — so it always makes progress and lifts the hold.
    inner = join(
        "inner_join",
        coalesce_all("inner_build_collect", source("inner_build_scan", [[2]])),
        source("inner_probe_scan", [[3, 3, 3]]),
    )
    outer = join("outer_join", coalesce_all("outer_build_collect", source("outer_build_scan", [[5]])), inner)
    driver = run(sink("unload", outer))

    assert sum(b.num_rows() for b in driver.results) == 9
    calls = [(e.node.split("#")[0], e.call) for e in driver.trace]
    assert calls.index(("outer_join", "set_build")) < calls.index(("inner_join", "set_build"))


if __name__ == "__main__":
    raise SystemExit(main(globals()))
