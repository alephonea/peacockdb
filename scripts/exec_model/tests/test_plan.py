"""Heights, left-to-right order, and the structural rules the scheduler assumes."""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

import pandas as pd

from .harness import main, raises
from ..errors import PlanError
from ..layout import BatchLayout, NodeKind, PartitionLayout, UniqueScope
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import Alias, Col
from ..operators.joins import JoinType
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


def test_a_forwarder_may_take_more_than_two_children():
    # DataFusion unions are n-ary, and the forwarder mappings generalize to any child
    # count, so the tree does too. Joins stay binary by their own arity check.
    three = union("union", [source(f"s{i}", [[1]]) for i in range(3)])
    plan = Plan.build(sink("unload", three))
    assert plan[plan.root].n_lanes == 3


def test_only_join_and_forwarder_take_multiple_children():
    def exec_over(children):
        return MockNode(
            "bad_exec",
            NodeKind.INTERMEDIATE,
            PartitionLayout(n=1),
            ExecutorCategory.EXEC,
            children=children,
            factory=lambda lane: None,
        )

    with raises(PlanError, match="at most one child"):
        Plan.build(sink("unload", exec_over([source("a", [[1]]), source("b", [[1]])])))
    # Three children must not slip past now that the tree itself is no longer binary.
    with raises(PlanError, match="at most one child"):
        Plan.build(sink("unload", exec_over([source(f"c{i}", [[1]]) for i in range(3)])))


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


# -- validate_schemas_and_partitions ------------------------------------------------
#
# What a node needs of its children, as opposed to plan.py's whole-tree rules. Each of
# these is a plan that runs and returns a wrong answer if the check is absent, which is
# why they are checks rather than comments. Built with the real (pandas) nodes, since the
# mocks declare no schema and no distribution.


def frame():
    return pd.DataFrame({"g": ["x", "y", "x"], "v": [1.0, 2.0, 3.0]})


def sum_aggs():
    return [A.Agg(A.SUM, "v", "s"), A.Agg(A.MEAN, "v", "m")]


def init_over(df, keys, aggs, lanes=1):
    return N.partial_aggregate("init", N.scan("scan", df, lanes, 2), keys, aggs)


def test_a_merge_must_group_on_what_its_partial_grouped_on():
    df, aggs = frame(), sum_aggs()
    with raises(PlanError, match="not grouping on what the partial grouped on|is not in its input"):
        Plan.build(N.unload("u", N.aggregate_batches("merge", init_over(df, ["g"], aggs),
                                                     ["v"], aggs)))


def test_a_merge_must_find_state_for_every_aggregate_it_declares():
    df = frame()
    partial = init_over(df, ["g"], [A.Agg(A.SUM, "v", "s")])
    with raises(PlanError, match="no state for 'm'"):
        Plan.build(N.unload("u", N.aggregate_batches("merge", partial, ["g"], sum_aggs())))


def test_a_merge_must_agree_with_its_partial_about_the_function():
    # The silent one: `s` exists and is a real column, but a sum read where a mean's
    # sum-half sits computes a wrong number from a right column.
    df = frame()
    partial = init_over(df, ["g"], [A.Agg(A.SUM, "v", "s")])
    with raises(PlanError, match="is a mean.* but its input declares sum"):
        Plan.build(N.unload("u", N.aggregate_batches("merge", partial, ["g"],
                                                     [A.Agg(A.MEAN, "v", "s")])))


def test_a_multi_lane_join_must_have_both_sides_hashed():
    # Two scans at four lanes each: the plan validates structurally — lane counts agree —
    # and would join lane p against lane p, losing every pair whose sides landed in
    # different lanes.
    df = frame()
    build = N.coalesce_all("build", N.scan("b", df, 4, 1), schema=dict(df.dtypes))
    probe = N.scan("p", df, 4, 1)
    with raises(PlanError, match="is not hash-distributed"):
        Plan.build(N.unload("u", N.hash_join("join", build, probe, JoinType.INNER,
                                             ["g"], ["g"])))


def test_a_single_lane_join_needs_no_distribution():
    # At one lane every row meets every other, so the rule does not apply.
    df = frame()
    build = N.coalesce_all("build", N.scan("b", df, 1, 4), schema=dict(df.dtypes))
    plan = Plan.build(N.unload("u", N.hash_join("join", build, N.scan("p", df, 1, 4),
                                                JoinType.INNER, ["g"], ["g"])))
    assert plan[plan.root].n_lanes == 1


def test_merging_sorted_partitions_requires_sorted_input():
    with raises(PlanError, match="requires BatchSorted input"):
        Plan.build(N.unload("u", N.merge_sorted_partitions(
            "merge_sorted", N.scan("scan", frame(), 2, 2), ["v"])))


def test_a_limit_after_a_per_batch_sort_is_rejected():
    # Sorted per batch and not across them, so a prefix is the head of whichever batches
    # arrived first rather than the top-N.
    keep = [Alias(Col("g"), "g"), Alias(Col("v"), "v")]
    sorted_batches = N.sort("sort", N.scan("scan", frame(), 1, 1), ["v"])
    # A project above it, since a limit feeding only the sink is the other lowering.
    with raises(PlanError, match="not a top-N"):
        Plan.build(N.unload("u", N.project("after", N.limit("limit", sorted_batches, fetch=2), keep)))

    # Stream-sorted, so the same limit is fine.
    stream = N.accumulate_and_sort("accum", sorted_batches, ["v"], schema=dict(frame().dtypes))
    Plan.build(N.unload("u", N.project("after", N.limit("limit", stream, fetch=2), keep)))


def test_an_aggregate_declares_the_uniqueness_of_its_own_output():
    # Not checked anywhere — declared so later work does not have to re-derive it.
    df, aggs = frame(), sum_aggs()
    init = init_over(df, ["g"], aggs, lanes=2)
    assert init.output_partitions().unique_keys[0].scope is UniqueScope.PER_BATCH

    per_lane = N.aggregate_batches("merge", init, ["g"], aggs)
    assert per_lane.output_partitions().unique_keys[0].scope is UniqueScope.PER_PARTITION

    shuffled = N.emit_partitions("emit", N.coalesce_all(
        "c", N.merge_partitions("m", per_lane)), ["g"], 2)
    final = N.aggregate_batches("final", shuffled, ["g"], aggs, A.finalize_exprs(aggs))
    assert final.output_partitions().unique_keys[0].scope is UniqueScope.GLOBAL


def test_routing_category_rejects_backends():
    with raises(ValueError):
        NodeExecutors(
            ExecutorCategory.BATCH_FORWARDER,
            backends=ExecutorBackends(cpu=lambda lane: None),
        )


def test_stream_sortedness_is_derived_from_the_batch_layout():
    # There is no PartitionSorted variant: a whole-stream order is BatchSorted meeting
    # SingleBatch, so the two cannot disagree.
    from ..layout import ColumnOrder, SortOrder

    sorted_by = SortOrder.batch_sorted([ColumnOrder(0)])
    assert PartitionLayout(n=1, sort_order=sorted_by, batch_layout=BatchLayout.SINGLE_BATCH).is_stream_sorted
    assert not PartitionLayout(n=1, sort_order=sorted_by).is_stream_sorted  # MultipleBatches
    assert not PartitionLayout(n=1, batch_layout=BatchLayout.SINGLE_BATCH).is_stream_sorted


if __name__ == "__main__":
    raise SystemExit(main(globals()))
