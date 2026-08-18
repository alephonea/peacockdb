"""Per-operator tests, weighted toward the places pandas and cuDF disagree.

The end-to-end suite proves the operators compose. This one pins the individual decisions
that would otherwise be pandas defaults quietly standing in for cuDF behaviour — each is a
divergence named in `operators/frame.py` or in architecture.md's cuDF options table.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

import numpy as np
import pandas as pd

from .harness import main, raises
from ..executors import LaneEvent
from ..operators import aggregates as A
from ..operators import source
from ..limit import RowRange
from ..operators.accumulators import (
    AccumulateBatchesAndSort,
    AggregateBatches,
    CoalesceAllBatches,
    LimitStream,
    MergeSortedPartitions,
    ReBatchToTarget,
)
from ..operators.exec_ops import FilterExec, ProjectExec, SortExec
from ..operators.expressions import Alias, Binary, Col, IsNotNull, Lit
from ..operators.frame import PandasBatch, concatenate
from ..operators.joins import HashJoin, JoinType
from ..operators.partition_ops import (
    EmitPartitions,
    first_lane_ids,
    last_lane_ids,
    partition_ids,
    skewed_ids,
)


def batch(frame, tag="t"):
    return PandasBatch(frame, tag)


# -- the cuDF subset rules --------------------------------------------------------


def test_concatenate_rejects_a_column_mismatch():
    # cudf::concatenate requires identical types; pandas would take the union of columns
    # and fill with NaN, which is a wrong answer rather than an error.
    left = pd.DataFrame({"a": [1], "b": [2]})
    right = pd.DataFrame({"b": [3], "a": [4]})
    with raises(ValueError, match="column mismatch"):
        concatenate([left, right])


def test_a_batch_carries_no_index():
    # Rule 1: a cudf::table is columns and nothing else. A filtered pandas frame keeps
    # the original labels, and arithmetic would then align on them.
    frame = pd.DataFrame({"v": [1, 2, 3, 4]})
    out, _ = FilterExec(Binary(">", Col("v"), Lit(2))).exec(batch(frame))
    assert list(out.frame.index) == [0, 1]


def test_a_batch_cannot_be_consumed_twice():
    b = batch(pd.DataFrame({"v": [1]}))
    b.consume()
    with raises(AssertionError, match="consumed twice"):
        b.consume()


def test_an_unsupported_operator_is_rejected():
    with raises(ValueError, match="no counterpart"):
        Binary("**", Col("v"), Lit(2))


# -- filter / project / sort / limit ----------------------------------------------


def test_a_null_predicate_does_not_pass_the_filter():
    frame = pd.DataFrame({"v": [1.0, np.nan, 5.0]})
    out, _ = FilterExec(Binary(">", Col("v"), Lit(2))).exec(batch(frame))
    assert list(out.frame.v) == [5.0]


def test_project_output_column_order_is_the_expression_order():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    exprs = [Alias(Col("b"), "b"), Alias(Binary("+", Col("a"), Col("b")), "sum")]
    out, _ = ProjectExec(exprs).exec(batch(frame))
    assert list(out.frame.columns) == ["b", "sum"]
    assert out.frame["sum"].iloc[0] == 3


def test_is_not_null_is_available_as_a_predicate():
    # The filter #137 wants inserted under a shuffle on placement-free sides.
    frame = pd.DataFrame({"k": [1.0, np.nan, 3.0]})
    out, _ = FilterExec(IsNotNull(Col("k"))).exec(batch(frame))
    assert list(out.frame.k) == [1.0, 3.0]


def test_sort_null_placement_is_explicit_in_both_directions():
    # cudf::order and cudf::null_order are separate arguments; pandas defaults nulls last.
    frame = pd.DataFrame({"v": [3.0, np.nan, 1.0]})
    last, _ = SortExec(["v"]).exec(batch(frame))
    assert np.isnan(last.frame.v.iloc[-1])
    first, _ = SortExec(["v"], nulls_first=True).exec(batch(frame))
    assert np.isnan(first.frame.v.iloc[0])


def test_per_batch_sort_fetch_is_a_top_n_within_the_batch():
    frame = pd.DataFrame({"v": [5, 1, 4, 2]})
    out, _ = SortExec(["v"], ascending=[False], fetch=2).exec(batch(frame))
    assert list(out.frame.v) == [5, 4]


def test_limit_streams_the_interval_out_of_its_input():
    # skip=2 fetch=3 across a 4-row batch and a 6-row one: both straddle, both are sliced,
    # and nothing is ever held — the interval comes out in the batches it arrived in.
    node = LimitStream(skip=2, fetch=3)
    first, _ = node.accumulate_and_fetch(batch(pd.DataFrame({"v": list(range(4))})))
    second, _ = node.accumulate_and_fetch(batch(pd.DataFrame({"v": list(range(4, 10))})))
    assert list(first[0].frame.v) == [2, 3]
    assert list(second[0].frame.v) == [4]
    assert node.sliced == [RowRange(2, 2), RowRange(0, 1)]
    assert node.resident_bytes() == 0


# -- aggregates -------------------------------------------------------------------


def test_the_null_group_survives_aggregation():
    # cuDF groups with null_policy::INCLUDE; pandas drops NaN keys unless told not to,
    # which is how tpcds q15's NULL ca_zip row disappears.
    frame = pd.DataFrame({"g": ["a", None, "a"], "v": [1, 2, 3]})
    out = A.single(frame, ["g"], [A.Agg(A.SUM, "v", "s")])
    assert len(out) == 2
    assert out.s.sum() == 6


def test_count_star_and_count_column_are_different_aggregates():
    frame = pd.DataFrame({"g": ["a", "a"], "v": [1.0, np.nan]})
    out = A.single(frame, ["g"], [A.Agg(A.COUNT, None, "rows"), A.Agg(A.COUNT, "v", "non_null")])
    assert out.rows.iloc[0] == 2
    assert out.non_null.iloc[0] == 1


def test_mean_decomposes_to_sum_and_count_not_a_mean_of_means():
    # Averaging partial means is wrong whenever the batches differ in size, which is the
    # multi-GPU rule in build-test.md. Two batches of 1 and 3 rows make it visible.
    frame = pd.DataFrame({"g": ["a"] * 4, "v": [10.0, 1.0, 1.0, 1.0]})
    first = A.partial(frame.iloc[:1], ["g"], [A.Agg(A.MEAN, "v", "m")])
    rest = A.partial(frame.iloc[1:], ["g"], [A.Agg(A.MEAN, "v", "m")])
    merged = A.final(concatenate([first, rest]), ["g"], [A.Agg(A.MEAN, "v", "m")])
    assert merged.m.iloc[0] == 3.25  # 13/4, not (10 + 1)/2
    assert list(first.columns) == ["g", "m$sum", "m$count"]


def test_a_final_phase_emits_its_declared_columns_when_empty():
    # An all-empty lane must not leak the partial's state columns into the final's output:
    # the frames are concatenated downstream and cudf::concatenate rejects a mismatch.
    aggs = [A.Agg(A.MEAN, "v", "m")]
    empty = A.partial(pd.DataFrame({"g": [], "v": []}), ["g"], aggs)
    out = A.final(empty, ["g"], aggs)
    assert list(out.columns) == ["g", "m"]


# -- the partitioner (T2 policy) --------------------------------------------------


def test_row_groups_split_into_contiguous_balanced_chunks():
    groups = source.split_row_groups(100, 10)
    mapping = source.partition_row_groups(groups, 4, None)
    assert [len(part[0]) for part in mapping] == [3, 3, 2, 2]
    flat = [g for part in mapping for b in part for g in b]
    assert flat == sorted(flat) == list(range(10))  # contiguous, file order, no gaps

    # The spec's stated bound: max − min partition rows ≤ one row group. Worth asserting
    # rather than eyeballing, because it is the property that makes contiguity acceptable.
    rows = [sum(groups[g][1] for b in part for g in b) for part in mapping]
    assert max(rows) - min(rows) <= max(length for _, length in groups)


def test_batching_off_gives_one_batch_per_chunk():
    mapping = source.partition_row_groups(source.split_row_groups(40, 10), 2, None)
    assert all(len(part) == 1 for part in mapping)


def test_batching_on_packs_greedily_under_the_target():
    mapping = source.partition_row_groups(source.split_row_groups(80, 10), 1, 25)
    assert [len(b) for b in mapping[0]] == [2, 2, 2, 2]


def test_a_row_group_over_target_still_becomes_its_own_batch():
    # Minimum granularity is one row group and the planner still produces a plan; the
    # runtime consequence belongs to the enforcer (#142).
    mapping = source.partition_row_groups(source.split_row_groups(30, 30), 1, 5)
    assert mapping == [[[0]]]


def test_fewer_row_groups_than_partitions_leaves_empty_partitions():
    mapping = source.partition_row_groups(source.split_row_groups(20, 10), 4, None)
    assert sum(1 for part in mapping if part and part[0]) == 2


def test_an_empty_survivor_set_is_an_explicit_error():
    # The fbs convention "empty map means legacy single partition" must not leak in here.
    with raises(ValueError, match="empty survivor set"):
        source.partition_row_groups([], 4, None)


def test_the_partitioner_is_a_pure_function():
    groups = source.split_row_groups(100, 7)
    assert source.partition_row_groups(groups, 3, 20) == source.partition_row_groups(groups, 3, 20)


# -- hash scatter -----------------------------------------------------------------


def test_all_null_keys_land_in_one_partition():
    # The kernel skips null columns (comet-mandated), so every all-null key row hashes to
    # the seed alone. That is the skew #137 is about, not a bug.
    frame = pd.DataFrame({"k": [np.nan] * 5})
    ids = partition_ids(frame, ["k"], 8)
    assert len(set(ids)) == 1


def test_the_scatter_preserves_every_row_exactly_once():
    frame = pd.DataFrame({"k": list(range(50)), "v": list(range(50))})
    outputs, _ = EmitPartitions(["k"], 4).emit(batch(frame))
    assert len(outputs) == 4
    total = pd.concat([o.frame for o in outputs], ignore_index=True)
    assert sorted(total.k) == list(range(50))


def test_the_same_key_always_lands_in_the_same_partition():
    left = partition_ids(pd.DataFrame({"k": [1, 2, 3]}), ["k"], 4)
    right = partition_ids(pd.DataFrame({"k": [3, 2, 1]}), ["k"], 4)
    assert left == right[::-1]


# -- joins ------------------------------------------------------------------------


def build_probe(join_type, null_equals_null=False):
    build = pd.DataFrame({"k": [1.0, 2.0, np.nan], "bv": ["a", "b", "c"]})
    probe = pd.DataFrame({"k": [2.0, np.nan, 9.0], "pv": [20, 99, 90]})
    join = HashJoin(join_type, ["k"], ["k"], null_equals_null)
    join.set_build(batch(build, "B"))
    out, _ = join.probe_and_fetch(batch(probe, "P"))
    finish, _ = join.finish_and_fetch()
    return out + finish


def test_null_keys_match_nothing_by_default():
    rows = sum(b.num_rows() for b in build_probe(JoinType.INNER))
    assert rows == 1  # k=2 only; the two nulls do not pair


def test_null_equals_null_makes_null_keys_match():
    # What a set operation lowered to a join asks for, and what pandas does by default —
    # which is why the flag has to be explicit rather than inherited from pandas.
    rows = sum(b.num_rows() for b in build_probe(JoinType.INNER, null_equals_null=True))
    assert rows == 2  # k=2, plus null=null


def test_left_outer_emits_unmatched_build_rows_only_at_finish():
    build = pd.DataFrame({"k": [1, 2], "bv": ["a", "b"]})
    join = HashJoin(JoinType.LEFT, ["k"], ["k"])
    join.set_build(batch(build, "B"))
    probes, _ = join.probe_and_fetch(batch(pd.DataFrame({"k": [2], "pv": [20]}), "P"))
    assert sum(b.num_rows() for b in probes) == 1
    finish, _ = join.finish_and_fetch()
    assert sum(b.num_rows() for b in finish) == 1  # k=1, null-padded


def test_a_build_row_matched_in_an_earlier_batch_is_not_re_emitted_at_finish():
    # The property the finish pass exists for, and the one a per-batch anti-join would get
    # wrong: matching is remembered across probe calls.
    build = pd.DataFrame({"k": [1, 2], "bv": ["a", "b"]})
    join = HashJoin(JoinType.LEFT_ANTI, ["k"], ["k"])
    join.set_build(batch(build, "B"))
    join.probe_and_fetch(batch(pd.DataFrame({"k": [1], "pv": [10]}), "P0"))
    join.probe_and_fetch(batch(pd.DataFrame({"k": [9], "pv": [90]}), "P1"))
    finish, _ = join.finish_and_fetch()
    assert list(finish[0].frame.k) == [2]


def test_probing_before_set_build_is_an_error():
    join = HashJoin(JoinType.INNER, ["k"], ["k"])
    with raises(AssertionError, match="probed before set_build"):
        join.probe_and_fetch(batch(pd.DataFrame({"k": [1]}), "P"))


def test_the_build_side_is_residency_and_is_reported():
    join = HashJoin(JoinType.INNER, ["k"], ["k"])
    assert join.resident_bytes() == 0
    join.set_build(batch(pd.DataFrame({"k": list(range(100))}), "B"))
    assert join.resident_bytes() > 0


def test_an_empty_partial_does_not_retype_the_key_it_is_concatenated_onto():
    # An empty batch reaching an aggregate makes it emit a frame with no rows, and that
    # frame still has to carry types — a cudf::column has one and cannot not have one.
    # pandas will happily default them to float64 and then let the concatenation retype
    # the key it lands on. That is not cosmetic: `partition_ids` stringifies 5.0 where it
    # stringifies 5, so a lane that saw an empty batch stops co-locating with one that
    # did not, and the shuffled aggregate quietly comes out short.
    aggs = [A.Agg(A.SUM, "v", "total")]
    rows = pd.DataFrame({"k": [5, 6], "v": [1.0, 2.0]})
    merged = concatenate([A.partial(rows, ["k"], aggs), A.partial(rows.iloc[0:0], ["k"], aggs)])
    assert merged.k.dtype == rows.k.dtype


# -- re-batching ------------------------------------------------------------------


def rows_out(outputs):
    return [out.num_rows() for out in outputs]


def test_re_batching_upward_merges_and_holds_the_remainder():
    node = ReBatchToTarget(10)
    for chunk in range(3):
        outputs, _ = node.accumulate_and_fetch(batch(pd.DataFrame({"v": [chunk] * 4})))
        # 4, then 8 — nothing crosses the target until the third arrival takes it to 12.
        assert rows_out(outputs) == ([10] if chunk == 2 else [])
    assert node.resident_bytes() > 0                       # the 2-row tail is held
    assert rows_out(node.mark_done_and_fetch()[0]) == [2]
    assert node.resident_bytes() == 0


def test_re_batching_downward_splits_one_batch_into_several():
    node = ReBatchToTarget(3)
    outputs, _ = node.accumulate_and_fetch(batch(pd.DataFrame({"v": list(range(10))})))
    assert rows_out(outputs) == [3, 3, 3]
    assert rows_out(node.mark_done_and_fetch()[0]) == [1]


def test_re_batching_conserves_rows_in_order():
    node = ReBatchToTarget(4)
    emitted = []
    for chunk in ([1, 2, 3], [4], [5, 6, 7, 8, 9]):
        outputs, _ = node.accumulate_and_fetch(batch(pd.DataFrame({"v": chunk})))
        emitted += [out.frame for out in outputs]
    emitted += [out.frame for out in node.mark_done_and_fetch()[0]]
    assert list(concatenate(emitted).v) == list(range(1, 10))


def test_re_batching_a_stream_of_empty_batches_emits_nothing():
    # An all-empty lane is ordinary — a filter that kept nothing — and the node must not
    # invent a batch for it, the way a join's build side deliberately does.
    node = ReBatchToTarget(4)
    for _ in range(3):
        outputs, _ = node.accumulate_and_fetch(batch(pd.DataFrame({"v": []})))
        assert outputs == []
    assert node.mark_done_and_fetch()[0] == []


# -- aggregate compaction policy ----------------------------------------------------
#
# Compacting per arrival bounds residency but re-scans the state once per batch, which is
# quadratic in the batch count when nothing merges; never compacting holds everything. The
# threshold-with-doubling rule has to behave correctly in both regimes and, above all,
# return the same answer whatever it decides.


def repeating(i):
    """Three groups, present in every batch — compaction shrinks."""
    return pd.DataFrame({"k": [0, 1, 2], "v": [float(i), float(i + 1), float(i + 2)]})


def disjoint(i):
    """Three groups nobody else has — compaction cannot shrink anything."""
    return pd.DataFrame({"k": [3 * i, 3 * i + 1, 3 * i + 2], "v": [1.0, 2.0, 3.0]})


def partial_of(rows, aggs):
    """What an init `GpuAggregate` hands the merge accumulator: state, not raw rows."""
    return A.partial(rows, ["k"], aggs)


def test_a_low_cardinality_aggregate_compacts_and_stays_at_group_cardinality():
    aggs = [A.Agg(A.SUM, "v", "total")]
    node = AggregateBatches(["k"], aggs, compact_bytes=400)
    peak = 0
    for i in range(20):
        node.accumulate_and_fetch(batch(partial_of(repeating(i), aggs)))
        peak = max(peak, node.resident_bytes())
    assert node.compactions > 0, "the threshold was never reached"
    # Three groups repeat forever, so each compaction leaves a 3-row state and the bar
    # never rises: residency is the threshold plus at most the arrival that crossed it.
    assert node.threshold == 400
    assert peak < 2 * 400


def test_a_high_cardinality_aggregate_doubles_its_way_out_instead_of_rescanning():
    aggs = [A.Agg(A.SUM, "v", "total")]
    node = AggregateBatches(["k"], aggs, compact_bytes=400)
    for i in range(40):
        node.accumulate_and_fetch(batch(partial_of(disjoint(i), aggs)))
    # Nothing merged, so the bar rose and the compactions are logarithmic in the arrivals
    # rather than one per arrival — the quadratic re-scan this rule exists to avoid.
    assert node.threshold > 400
    assert node.compactions <= 8, node.compactions


def test_the_compaction_policy_never_changes_the_answer():
    aggs = [A.Agg(A.SUM, "v", "total"), A.Agg(A.MEAN, "v", "avg"), A.Agg(A.COUNT, None, "n")]
    frames = [repeating(i) for i in range(9)]
    want = A.single(concatenate([f.copy() for f in frames]), ["k"], aggs)

    for threshold in (1, 300, 10**9):     # per-arrival, mid-stream, never-until-done
        node = AggregateBatches(["k"], aggs, A.finalize_exprs(aggs), compact_bytes=threshold)
        for f in frames:
            node.accumulate_and_fetch(batch(partial_of(f, aggs)))
        got = node.mark_done_and_fetch()[0][0].frame.sort_values("k").reset_index(drop=True)
        assert list(got.columns) == list(want.columns), threshold
        for column in want.columns:
            assert np.allclose(got[column].to_numpy(), want[column].to_numpy()), (
                f"threshold {threshold}: column {column}"
            )


# -- grouping sets -------------------------------------------------------------------


def rollup_frame():
    return pd.DataFrame({"a": [1, 1, 2, 2], "b": [10, 20, 10, 20], "v": [1.0, 2.0, 4.0, 8.0]})


def test_a_rollup_expands_into_one_batch_carrying_every_set():
    # k groupbys over the same input, tagged and concatenated — one frame out, never one
    # per set, because an executor may return at most one batch per call per output lane.
    aggs = [A.Agg(A.SUM, "v", "s")]
    out = A.partial_over_sets(rollup_frame(), ["a", "b"], aggs, A.rollup_masks(2))
    assert set(out[A.GROUPING_ID]) == {0, 2, 3}          # bitmask of masked positions
    assert len(out) == 4 + 2 + 1                          # (a,b) x4, (a) x2, () x1
    assert out[out[A.GROUPING_ID] == 3].s.iloc[0] == 15.0  # the grand total


def test_the_grouping_id_sits_between_the_keys_and_the_state():
    # The position the C++ gives it, and what fixes the ordinals in a plan: keys, then the
    # id, then the state columns.
    aggs = [A.Agg(A.SUM, "v", "s"), A.Agg(A.MEAN, "v", "m")]
    out = A.partial_over_sets(rollup_frame(), ["a", "b"], aggs, A.rollup_masks(2))
    assert list(out.columns) == ["a", "b", A.GROUPING_ID, "s", "m$sum", "m$count"]


def test_a_masked_key_is_null_rather_than_absent():
    # All sets share one schema — that is what lets them sit in a single table — so a
    # masked position is a NULL value, not a missing column.
    out = A.partial_over_sets(rollup_frame(), ["a", "b"], [A.Agg(A.SUM, "v", "s")],
                              A.rollup_masks(2))
    masked = out[out[A.GROUPING_ID] == 2]
    assert masked.b.isna().all() and masked.a.notna().all()
    assert out[out[A.GROUPING_ID] == 3][["a", "b"]].isna().all().all()


def test_the_merge_accepts_the_sets_in_any_order():
    # What the concatenation actually produces once a lane has several batches: each set's
    # rows are non-contiguous. Every consumer is a hash groupby over keys + id, so runs are
    # never a precondition — but order is pinned for float determinism, not for grouping.
    aggs = [A.Agg(A.SUM, "v", "s"), A.Agg(A.STDDEV, "v", "sd")]
    masks, keys = A.rollup_masks(2), ["a", "b"]
    with_id = keys + [A.GROUPING_ID]
    batches = [A.partial_over_sets(rollup_frame(), keys, aggs, masks) for _ in range(2)]
    interleaved = concatenate(batches)                                   # set-major per batch
    contiguous = interleaved.sort_values(with_id, na_position="last").reset_index(drop=True)
    shuffled = interleaved.sample(frac=1, random_state=5).reset_index(drop=True)

    results = [A.final(f, with_id, aggs).sort_values(with_id, na_position="last")
               .reset_index(drop=True) for f in (interleaved, contiguous, shuffled)]
    for other in results[1:]:
        assert other[with_id].equals(results[0][with_id])
        for column in ("s", "sd"):
            assert np.allclose(other[column], results[0][column], equal_nan=True)


def test_a_grouping_set_mask_must_match_the_key_count():
    with raises(ValueError, match="against 2 group keys"):
        A.partial_over_sets(rollup_frame(), ["a", "b"], [], [(True,)])
    with raises(ValueError, match="at least one set"):
        A.partial_over_sets(rollup_frame(), ["a", "b"], [], [])


# -- the empty single-batch contract ------------------------------------------------
#
# A SingleBatch accumulator owes downstream exactly one batch at done, even when it
# accumulated nothing (F7 — a join's build lane cannot tell an empty build side from a
# plan error otherwise), and the empty batch must be TYPED, or it retypes whatever it is
# later concatenated onto.


def test_an_empty_coalesce_all_still_emits_one_typed_batch():
    node = CoalesceAllBatches(schema={"k": "int64", "v": "float64"})
    outputs, _ = node.mark_done_and_fetch()
    assert len(outputs) == 1
    frame = outputs[0].frame
    assert len(frame) == 0
    assert list(frame.columns) == ["k", "v"]
    assert frame.k.dtype == "int64"


def test_an_empty_accumulate_and_sort_still_emits_one_typed_batch():
    node = AccumulateBatchesAndSort(["v"], schema={"v": "int64"})
    outputs, _ = node.mark_done_and_fetch()
    assert len(outputs) == 1 and outputs[0].num_rows() == 0
    assert outputs[0].frame.v.dtype == "int64"


def test_an_empty_merge_sorted_still_emits_one_typed_batch():
    merge = MergeSortedPartitions(2, ["v"], schema={"v": "int64"})
    first, _ = merge.accumulate_and_fetch(0, LaneEvent.done())
    assert first == []
    outputs, _ = merge.accumulate_and_fetch(1, LaneEvent.done())
    assert len(outputs) == 1 and outputs[0].num_rows() == 0
    assert outputs[0].frame.v.dtype == "int64"


def test_a_zero_input_aggregate_emits_its_declared_typed_empty():
    aggs = [A.Agg(A.SUM, "v", "total")]
    node = AggregateBatches(["k"], aggs, A.finalize_exprs(aggs),
                             schema={"k": "int64", "total": "float64"})
    outputs, _ = node.mark_done_and_fetch()
    frame = outputs[0].frame
    assert list(frame.columns) == ["k", "total"]
    assert len(frame) == 0
    assert frame.k.dtype == "int64"


def test_an_empty_single_batch_accumulator_without_a_schema_is_loud():
    for node in (
        CoalesceAllBatches(),
        AggregateBatches(["g"], [A.Agg(A.SUM, "v", "s")], A.finalize_exprs([A.Agg(A.SUM, "v", "s")])),
        AccumulateBatchesAndSort(["v"]),
    ):
        with raises(ValueError, match="no schema"):
            node.mark_done_and_fetch()
    merge = MergeSortedPartitions(1, ["v"])
    with raises(ValueError, match="no schema"):
        merge.accumulate_and_fetch(0, LaneEvent.done())


# -- the finish pass over an empty probe side ---------------------------------------


def test_a_left_outer_finish_with_no_probe_batches_pads_the_declared_schema():
    # A probe lane may legitimately deliver zero batches; the unmatched build rows must
    # still come out with the probe columns null-padded, not with a shape that silently
    # depends on whether a probe batch happened to arrive.
    build = pd.DataFrame({"k": [1, 2], "bv": ["a", "b"]})
    join = HashJoin(JoinType.LEFT, ["k"], ["k"], probe_schema=["k", "pv"])
    join.set_build(batch(build, "B"))
    finish, _ = join.finish_and_fetch()
    frame = finish[0].frame
    assert list(frame.columns) == ["k", "bv", "pv"]
    assert len(frame) == 2
    assert frame.pv.isna().all()


def test_an_outer_finish_with_no_probe_batches_and_no_schema_is_loud():
    join = HashJoin(JoinType.LEFT, ["k"], ["k"])
    join.set_build(batch(pd.DataFrame({"k": [1]}), "B"))
    with raises(ValueError, match="probe_schema"):
        join.finish_and_fetch()


def test_a_finished_join_holds_no_residency():
    join = HashJoin(JoinType.INNER, ["k"], ["k"])
    join.set_build(batch(pd.DataFrame({"k": list(range(50))}), "B"))
    assert join.resident_bytes() > 0
    join.finish_and_fetch()
    assert join.resident_bytes() == 0


# -- degenerate placements --------------------------------------------------------


def test_a_skewed_hash_still_co_locates_equal_keys():
    # The property a shuffle owes its callers is co-location, and nothing more. A
    # placement may load the lanes as unevenly as it likes and still be correct.
    frame = pd.DataFrame({"k": [i % 7 for i in range(60)]})
    for placement in (skewed_ids, first_lane_ids, last_lane_ids):
        ids = placement(frame, ["k"], 4)
        by_key = {}
        for key, lane in zip(frame.k, ids):
            by_key.setdefault(key, set()).add(lane)
        assert all(len(lanes) == 1 for lanes in by_key.values()), placement.__name__
        assert set(ids) <= set(range(4)), placement.__name__


def test_the_degenerate_placements_are_actually_degenerate():
    # Otherwise a sweep over them would be four runs of the same thing.
    frame = pd.DataFrame({"k": list(range(200))})
    assert len(set(partition_ids(frame, ["k"], 4))) == 4
    assert set(first_lane_ids(frame, ["k"], 4)) == {0}
    assert set(last_lane_ids(frame, ["k"], 4)) == {3}
    skewed = skewed_ids(frame, ["k"], 4)
    assert skewed.count(0) > len(skewed) // 2


if __name__ == "__main__":
    raise SystemExit(main(globals()))
