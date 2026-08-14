"""Two runs of the same plan produce the same batch trace and the same batches.

Scheduling has no freedom left: the choice key `(height, order)` is a total order over
the tree, lanes are visited by index, and a forwarder cycles `sources_of` from a cursor.
That matters beyond golden stability — float aggregation sums in stream order, so an
unpinned order changes low bits.
"""

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
    interleave,
    join,
    merge_partitions,
    merge_sorted_partitions,
    sink,
    skew_router,
    source,
    union,
)


def build_shuffle_join():
    build = coalesce_all(
        "build_collect",
        merge_partitions("build_merge", exec_node("build_filter", source("build_scan", [[3], [3]]))),
    )
    probe = merge_partitions("probe_merge", source("probe_scan", [[2, 2], [2]]))
    return sink("unload", join("join", build, probe))


def build_skewed_shuffle():
    shuffled = emit_partitions(
        "emit", merge_partitions("merge", source("load", [[9], [9], [9]])), 4, skew_router(4, 1)
    )
    return sink("unload", coalesce_all("agg_final", shuffled))


def build_union_over_shuffles():
    left = emit_partitions(
        "emit_l", merge_partitions("merge_l", source("left_scan", [[4], [4]])), 2, even_router(2)
    )
    right = interleave(
        "interleave",
        [exec_node("r0", source("r0_scan", [[1, 1], [1]])), exec_node("r1", source("r1_scan", [[1], [1, 1]]))],
    )
    return sink("unload", union("union", [left, right]))


def build_merge_sorted():
    return sink(
        "unload",
        merge_sorted_partitions("merge_sorted", exec_node("sort", source("load", [[3, 3], [3], []]))),
    )


PLANS = {
    "shuffle_join": build_shuffle_join,
    "skewed_shuffle": build_skewed_shuffle,
    "union_over_shuffles": build_union_over_shuffles,
    "merge_sorted": build_merge_sorted,
}


def run_once(builder):
    driver = batch_partitioned_driver(Plan.build(builder()), MockSelector())
    driver.run()
    return driver


def test_traces_are_identical_across_runs():
    for name, builder in PLANS.items():
        first, second = run_once(builder), run_once(builder)
        assert first.trace == second.trace, name


def test_results_are_identical_across_runs():
    for name, builder in PLANS.items():
        first, second = run_once(builder), run_once(builder)
        assert [b.tag for b in first.results] == [b.tag for b in second.results], name
        assert [b.num_rows() for b in first.results] == [b.num_rows() for b in second.results], name


def test_a_merge_forwarder_cycles_its_lanes_round_robin():
    plan = Plan.build(sink("unload", merge_partitions("merge", source("load", [[1, 1], [1, 1], [1, 1]]))))
    driver = batch_partitioned_driver(plan, MockSelector())
    driver.run()

    forwarded = [b.tag for b in driver.results]
    assert forwarded == [
        "load.p0.b0>unload",
        "load.p1.b0>unload",
        "load.p2.b0>unload",
        "load.p0.b1>unload",
        "load.p1.b1>unload",
        "load.p2.b1>unload",
    ]


def test_a_multi_child_forwarder_skips_a_pending_source_instead_of_waiting():
    # The rotation cursor advances only when it forwards, and a source that is merely
    # pending is skipped. Under min-height-first the left child wins every tie, so it is
    # the child that has a batch whenever the forwarder runs: the interleave drains
    # left-first rather than alternating. Deterministic, and left-biased — the round-robin
    # only really alternates where one child feeds all the lanes (GpuMergePartitions).
    left = exec_node("l", source("l_scan", [[1, 1], [1, 1]]))
    right = exec_node("r", source("r_scan", [[1, 1], [1, 1]]))
    plan = Plan.build(sink("unload", interleave("interleave", [left, right])))
    driver = batch_partitioned_driver(plan, MockSelector())
    driver.run()

    lane0 = [b.tag for b in driver.results if ".p0." in b.tag]
    assert lane0 == [
        "l_scan.p0.b0>l>unload",
        "l_scan.p0.b1>l>unload",
        "r_scan.p0.b0>r>unload",
        "r_scan.p0.b1>r>unload",
    ]


if __name__ == "__main__":
    raise SystemExit(main(globals()))
