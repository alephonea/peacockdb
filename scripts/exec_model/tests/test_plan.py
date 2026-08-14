"""Heights, left-to-right order, and the structural rules the scheduler assumes."""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from .harness import main, raises
from ..errors import PlanError
from ..layout import BatchLayout, NodeKind, PartitionLayout
from ..node import ExecutorBackends, ExecutorCategory, NodeExecutors
from ..plan import Plan
from .mocks import (
    MockNode,
    coalesce_all,
    emit_partitions,
    even_router,
    exec_node,
    join,
    merge_partitions,
    merge_sorted_partitions,
    sink,
    source,
    union,
)


def test_height_is_distance_to_root():
    plan = Plan.build(sink("unload", exec_node("filter", source("load", [[1, 1]]))))
    by_name = {info.node.name(): info for info in plan.nodes}
    assert by_name["unload"].height == 0
    assert by_name["filter"].height == 1
    assert by_name["load"].height == 2


def test_order_makes_the_build_side_leftmost_at_equal_height():
    build = coalesce_all("build_collect", source("build_scan", [[2]]))
    probe = exec_node("probe_filter", source("probe_scan", [[2]]))
    plan = Plan.build(sink("unload", join("join", build, probe)))
    by_name = {info.node.name(): info for info in plan.nodes}

    assert by_name["build_collect"].height == by_name["probe_filter"].height
    assert by_name["build_collect"].order < by_name["probe_filter"].order


def test_scheduling_key_is_a_total_order():
    plan = Plan.build(
        sink(
            "unload",
            join(
                "join",
                coalesce_all("build", source("build_scan", [[1]])),
                exec_node("probe", source("probe_scan", [[1]])),
            ),
        )
    )
    keys = [(info.height, info.order) for info in plan.nodes]
    assert len(set(keys)) == len(keys)


def test_sink_inherits_its_child_lane_count():
    plan = Plan.build(sink("unload", source("load", [[1], [1], [1]])))
    assert plan[plan.root].n_lanes == 3


def test_build_side_must_declare_single_batch():
    streaming_build = exec_node("build_filter", source("build_scan", [[1]]))
    with raises(PlanError, match="SingleBatch"):
        Plan.build(sink("unload", join("join", streaming_build, source("probe", [[1]]))))


def test_join_lane_counts_must_agree():
    build = coalesce_all("build", source("build_scan", [[1]]))
    probe = source("probe_scan", [[1], [1]])
    with raises(PlanError, match="join lanes must agree"):
        Plan.build(sink("unload", join("join", build, probe)))


def test_emitter_child_must_be_single_lane():
    with raises(PlanError, match="scatters 1"):
        Plan.build(
            sink(
                "unload",
                emit_partitions("emit", source("load", [[1], [1]]), 2, even_router(2)),
            )
        )


def test_exec_may_not_change_lane_count():
    bad = MockNode(
        "bad",
        NodeKind.INTERMEDIATE,
        PartitionLayout(n=4),
        ExecutorCategory.EXEC,
        children=[source("load", [[1]])],
        factory=lambda lane: None,
    )
    with raises(PlanError, match="1:1 per lane"):
        Plan.build(sink("unload", bad))


def test_plan_tree_is_binary():
    three = union("union", [source(f"s{i}", [[1]]) for i in range(3)])
    with raises(PlanError, match="binary"):
        Plan.build(sink("unload", three))


def test_only_join_and_forwarder_take_two_children():
    two_child_exec = MockNode(
        "bad_exec",
        NodeKind.INTERMEDIATE,
        PartitionLayout(n=1),
        ExecutorCategory.EXEC,
        children=[source("a", [[1]]), source("b", [[1]])],
        factory=lambda lane: None,
    )
    with raises(PlanError, match="at most one child"):
        Plan.build(sink("unload", two_child_exec))


def test_forwarder_lane_declaration_must_match_the_mapping():
    child = source("load", [[1], [1], [1]])
    bad = MockNode(
        "merge",
        NodeKind.INTERMEDIATE,
        PartitionLayout(n=2),
        ExecutorCategory.BATCH_FORWARDER,
        children=[child],
        forwarder=merge_partitions("m", child).make_executors().forwarder,
    )
    with raises(PlanError, match="forwarder declares"):
        Plan.build(sink("unload", bad))


def test_partition_accumulator_outputs_one_lane():
    node = merge_sorted_partitions("merge_sorted", source("load", [[1], [1]]))
    plan = Plan.build(sink("unload", node))
    assert plan[plan.root].n_lanes == 1


def test_routing_category_rejects_backends():
    with raises(ValueError):
        NodeExecutors(
            ExecutorCategory.BATCH_FORWARDER,
            backends=ExecutorBackends(cpu=lambda lane: None),
        )


def test_single_batch_canonicalizes_batch_sorted_to_partition_sorted():
    from ..layout import ColumnOrder, SortKind, SortOrder

    layout = PartitionLayout(
        n=1,
        sort_order=SortOrder.batch_sorted([ColumnOrder(0)]),
        batch_layout=BatchLayout.SINGLE_BATCH,
    )
    assert layout.sort_order.kind is SortKind.PARTITION_SORTED


if __name__ == "__main__":
    raise SystemExit(main(globals()))
