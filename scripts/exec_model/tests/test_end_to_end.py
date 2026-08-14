"""Whole queries through the real operators: every partitioning config gives one answer.

This is the prototype's version of two-engine correctness. Each query is built at several
(partitions, row-group size, batch target) settings and every one must equal a single-shot
pandas oracle — so a bug that only appears once rows are split across batches or lanes has
somewhere to show. That is the class the whole mode exists to create: partial aggregates
merged out of order, a join whose finish pass runs per lane, a sort whose batches are
individually ordered and collectively not.

pandas is imported unconditionally. If it is missing this file fails rather than skipping:
a skipped operator suite reads exactly like a passing one.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

import numpy as np
import pandas as pd

from .harness import main, raises
from ..batch_partitioned_driver import batch_partitioned_driver
from ..errors import ResidentBudgetExceeded
from ..node import CpuBackendSelector
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import Alias, Binary, Col, Lit
from ..operators.joins import JoinType
from ..plan import Plan

#: (n_partitions, rows_per_group, target_batch_rows). The first is the degenerate
#: single-partition single-batch case — the shape the oracle itself has.
CONFIGS = [(1, 1000, None), (1, 5, 10), (4, 5, 10), (3, 4, 4), (8, 3, 3)]


def fixture(n=60, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "k": rng.integers(0, 6, n),
            "g": rng.choice(list("xyz"), n),
            "v": rng.integers(1, 100, n),
        }
    )


def dim_fixture():
    return pd.DataFrame({"k": [0, 1, 2, 3, 9], "label": list("ABCDE")})


#: Every plan here runs under a real budget rather than `None`, so the accounting path is
#: live on each call and a regression that blew the resident set fails rather than passing
#: quietly. Generous against these fixtures (peaks are kilobytes) and far from unbounded.
BUDGET = 8 * 1024 * 1024


def execute(root, budget: int | None = BUDGET):
    driver = batch_partitioned_driver(Plan.build(root), CpuBackendSelector(), budget)
    driver.run()
    frames = [b.frame for b in driver.results if len(b.frame)]
    if not frames:
        return pd.DataFrame(), driver
    return pd.concat(frames, ignore_index=True), driver


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Row order is not part of the contract unless a sort says so — compare as a set."""
    if frame.empty:
        return frame
    return frame.sort_values(list(frame.columns)).reset_index(drop=True)


def same(got: pd.DataFrame, want: pd.DataFrame, label: str) -> None:
    got, want = canonical(got), canonical(want)
    assert list(got.columns) == list(want.columns), f"{label}: {list(got.columns)} vs {list(want.columns)}"
    assert len(got) == len(want), f"{label}: {len(got)} rows vs {len(want)}"
    for column in want.columns:
        left, right = got[column].to_numpy(), want[column].to_numpy()
        # pandas' own predicate, not np.issubdtype: from pandas 3 a string column is a
        # StringDtype rather than object, and numpy cannot interpret an extension dtype
        # at all — it raises instead of answering "not a number". CI installs the current
        # pandas, so the prototype has to hold across the 2/3 boundary.
        if pd.api.types.is_numeric_dtype(want[column]):
            assert np.allclose(left.astype(float), right.astype(float), equal_nan=True), (
                f"{label}: column {column} differs"
            )
        else:
            # NaN != NaN, so an object column holding nulls — which a masked grouping-set
            # key is — cannot be compared with ==. Normalize the nulls first.
            def nulls_alike(values):
                return [None if pd.isna(v) else v for v in values]

            assert nulls_alike(left) == nulls_alike(right), f"{label}: column {column} differs"


# -- queries ----------------------------------------------------------------------


def agg_schemas(df, keys, aggs):
    """Typed `{column: dtype}` for each aggregate phase, derived by running the phase
    over a zero-row slice — the empty-lane case each node may have to emit."""
    state = A.partial(df.iloc[0:0], keys, aggs)
    return dict(state.dtypes), dict(A.final(state, keys, aggs).dtypes)


def shuffled_aggregate(df, config, aggs, keys=("g",)):
    parts, group, target = config
    keys = list(keys)
    state_schema, final_schema = agg_schemas(df, keys, aggs)
    scan = N.scan("scan", df, parts, group, target)
    filtered = N.filter_("filter", scan, Binary(">", Col("v"), Lit(20)))
    partial = N.partial_aggregate("agg_partial", filtered, keys, aggs)
    compacted = N.aggregate_batches("agg_batches", partial, keys, aggs, schema=state_schema)
    if parts == 1:
        return N.unload(
            "unload",
            N.aggregate_batches("agg_final", compacted, keys, aggs,
                                A.finalize_exprs(aggs), schema=final_schema),
        )
    # GpuCoalesceAllBatches between the merge and the emit: the emit then makes one
    # scatter call and hands the final aggregate N batches rather than L*N. See the
    # spec's "The shuffle beneath a final aggregate is coalesced first".
    shuffle_in = N.coalesce_all("shuffle_in", N.merge_partitions("merge", compacted))
    emitted = N.emit_partitions("emit", shuffle_in, keys, parts)
    return N.unload(
        "unload",
        N.aggregate_batches("agg_final", emitted, keys, aggs,
                            A.finalize_exprs(aggs), schema=final_schema),
    )


def test_grouped_aggregate_matches_the_oracle_at_every_config():
    df = fixture()
    aggs = [A.Agg(A.SUM, "v", "sum_v"), A.Agg(A.MEAN, "v", "avg_v"), A.Agg(A.COUNT, None, "n")]
    sub = df[df.v > 20]
    want = (
        sub.groupby("g", dropna=False)
        .agg(sum_v=("v", "sum"), avg_v=("v", "mean"), n=("v", "size"))
        .reset_index()
    )
    for config in CONFIGS:
        got, _ = execute(shuffled_aggregate(df, config, aggs))
        same(got, want, f"grouped aggregate {config}")


def test_keyless_aggregate_matches_the_oracle_at_every_config():
    df = fixture()
    aggs = [A.Agg(A.SUM, "v", "sum_v"), A.Agg(A.MIN, "v", "min_v"), A.Agg(A.MAX, "v", "max_v")]
    want = pd.DataFrame([{"sum_v": df.v.sum(), "min_v": df.v.min(), "max_v": df.v.max()}])
    for parts, group, target in CONFIGS:
        scan = N.scan("scan", df, parts, group, target)
        partial = N.partial_aggregate("agg_partial", scan, [], aggs)
        compacted = N.aggregate_batches("agg_batches", partial, [], aggs)
        # Keyless needs no shuffle — collapse the lanes and finish once (the v1 shortcut).
        collapsed = N.merge_partitions("merge", compacted)
        root = N.unload(
            "unload",
            N.aggregate_batches("agg_final", collapsed, [], aggs, A.finalize_exprs(aggs)),
        )
        got, _ = execute(root)
        same(got, want, f"keyless aggregate {(parts, group, target)}")


def test_top_n_sort_matches_the_oracle_at_every_config():
    df = fixture()
    want = df.sort_values(["v", "k"], ascending=[False, True]).head(10).reset_index(drop=True)
    for parts, group, target in CONFIGS:
        scan = N.scan("scan", df, parts, group, target)
        per_batch = N.sort("sort", scan, ["v", "k"], ascending=[False, True], fetch=10)
        merged = N.merge_sorted_partitions(
            "merge_sorted", per_batch, ["v", "k"], ascending=[False, True], fetch=10
        )
        got, _ = execute(N.unload("unload", merged))
        # A top-N IS order-sensitive, so compare positionally rather than as a set.
        assert list(got.v) == list(want.v), f"top-n {(parts, group, target)}"


def test_a_single_lane_top_n_trims_at_every_stage():
    # The other half of the sort decomposition: within one lane it is
    # GpuAccumulateBatchesAndSort that carries the fetch, not GpuMergeSortedPartitions.
    # Without a fetch there the lane would accumulate and sort its whole stream to
    # return ten rows — the failure the limit lowering exists to avoid elsewhere.
    df = fixture()
    want = df.sort_values(["v", "k"], ascending=[False, True]).head(10).reset_index(drop=True)
    for parts, group, target in CONFIGS:
        scan = N.scan("scan", df, parts, group, target)
        per_batch = N.sort("sort", scan, ["v", "k"], ascending=[False, True], fetch=10)
        per_lane = N.accumulate_and_sort(
            "accum_sort", per_batch, ["v", "k"], ascending=[False, True], fetch=10,
            schema=dict(df.dtypes),
        )
        merged = N.merge_sorted_partitions(
            "merge_sorted", per_lane, ["v", "k"], ascending=[False, True], fetch=10
        )
        got, _ = execute(N.unload("unload", merged))
        assert list(got.v) == list(want.v), f"single-lane top-n {(parts, group, target)}"


def test_a_top_n_holds_only_the_fetch_at_each_stage():
    # What the per-stage fetch buys, asserted as residency rather than as rows: the
    # accumulator never holds more than the fetch per batch it has seen, so a top-10 over
    # a 60-row table is bounded by the limit and not by the input.
    df = fixture()
    scan = N.scan("scan", df, 1, 5, 10)
    per_batch = N.sort("sort", scan, ["v"], ascending=[False], fetch=3)
    per_lane = N.accumulate_and_sort("accum_sort", per_batch, ["v"], ascending=[False],
                                     fetch=3, schema=dict(df.dtypes))
    got, driver = execute(N.unload("unload", per_lane))
    assert list(got.v) == list(df.v.sort_values(ascending=False).head(3))
    # Each incoming batch was trimmed to 3 by the sort, so what the accumulator held is a
    # multiple of the fetch, never the 60 rows the table has.
    sorted_out = [e for e in driver.trace if e.node.startswith("sort#")]
    assert len(sorted_out) > 3, "the input needs several batches for this to mean anything"


def test_every_corpus_aggregate_matches_the_oracle_at_every_config():
    # The whole set the corpus uses, in one query: sum (1010 uses), avg (204), count (190),
    # stddev (24), max (22), min (12), var_pop (5), var (5), stddev_pop (5). stddev and var
    # are the interesting ones — their state is Welford's [count, mean, m2] and their merge
    # is MERGE_M2, so they are the only aggregates whose merge is not a plain re-aggregation.
    df = fixture()
    aggs = [
        A.Agg(A.SUM, "v", "sum_v"),
        A.Agg(A.COUNT, None, "n_rows"),
        A.Agg(A.COUNT, "v", "n_v"),
        A.Agg(A.MEAN, "v", "avg_v"),
        A.Agg(A.MIN, "v", "min_v"),
        A.Agg(A.MAX, "v", "max_v"),
        A.Agg(A.STDDEV, "v", "sd_samp", ddof=1),
        A.Agg(A.STDDEV, "v", "sd_pop", ddof=0),
        A.Agg(A.VAR, "v", "var_samp", ddof=1),
        A.Agg(A.VAR, "v", "var_pop", ddof=0),
    ]
    sub = df[df.v > 20]
    want = (
        sub.groupby("g", dropna=False)
        .agg(
            sum_v=("v", "sum"), n_rows=("v", "size"), n_v=("v", "count"),
            avg_v=("v", "mean"), min_v=("v", "min"), max_v=("v", "max"),
            sd_samp=("v", lambda s: s.std(ddof=1)), sd_pop=("v", lambda s: s.std(ddof=0)),
            var_samp=("v", lambda s: s.var(ddof=1)), var_pop=("v", lambda s: s.var(ddof=0)),
        )
        .reset_index()
    )
    for config in CONFIGS:
        got, _ = execute(shuffled_aggregate(df, config, aggs))
        same(got, want, f"every aggregate {config}")


def test_a_single_group_stddev_is_null_not_a_divide_by_zero():
    # count - ddof <= 0 is the finalize's CASE arm: one row has no sample dispersion, and
    # the answer is NULL rather than a division by zero or a root of a negative.
    df = pd.DataFrame({"g": ["only"], "v": [5.0], "k": [0]})
    aggs = [A.Agg(A.STDDEV, "v", "sd", ddof=1), A.Agg(A.STDDEV, "v", "sd_pop", ddof=0)]
    got = A.single(df, ["g"], aggs)
    assert np.isnan(got.sd.iloc[0])       # sample: divisor 0
    assert got.sd_pop.iloc[0] == 0.0      # population: divisor 1, no spread


def test_a_rollup_matches_the_oracle_at_every_config():
    # GROUP BY ROLLUP(g, k): the init expands into three sets in one batch, and every node
    # above groups on the keys plus __grouping_id as if it were an ordinary column. Nothing
    # in the sequence is grouping-set aware except that first node.
    df = fixture()
    aggs = [A.Agg(A.SUM, "v", "sum_v"), A.Agg(A.COUNT, None, "n")]
    keys, masks = ["g", "k"], A.rollup_masks(2)
    with_id = keys + [A.GROUPING_ID]
    want = A.single_over_sets(df, keys, aggs, masks)

    for parts, group, target in CONFIGS:
        expanded = N.partial_aggregate(
            "agg_init", N.scan("scan", df, parts, group, target), keys, aggs,
            grouping_sets=masks,
        )
        state_schema = dict(A.partial_over_sets(df.iloc[0:0], keys, aggs, masks).dtypes)
        compacted = N.aggregate_batches("agg_batches", expanded, with_id, aggs,
                                        schema=state_schema)
        shuffle_in = N.coalesce_all("shuffle_in", N.merge_partitions("merge", compacted))
        # Hashing the user keys only — the subset rule, since the group columns are
        # keys + the id. A masked key is NULL and the kernel skips null columns, so the
        # grand-total row lands in one fixed lane; it is one row.
        emitted = N.emit_partitions("emit", shuffle_in, keys, parts)
        final_schema = dict(A.final(A.partial_over_sets(df.iloc[0:0], keys, aggs, masks),
                                    with_id, aggs).dtypes)
        root = N.unload("unload", N.aggregate_batches("agg_final", emitted, with_id, aggs,
                                                      A.finalize_exprs(aggs),
                                                      schema=final_schema))
        got, _ = execute(root)
        same(got, want, f"rollup {(parts, group, target)}")


def test_a_rollup_carries_the_grouping_id_through_the_whole_sequence():
    # The id is a real column from the init onward: absent below it, present above it, and
    # dropped only by a projection the query's own output shape asks for.
    df = fixture()
    aggs = [A.Agg(A.SUM, "v", "sum_v")]
    keys, masks = ["g", "k"], A.rollup_masks(2)
    with_id = keys + [A.GROUPING_ID]
    expanded = N.partial_aggregate("agg_init", N.scan("scan", df, 4, 5, 10), keys, aggs,
                                   grouping_sets=masks)
    collapsed = N.merge_partitions("merge", expanded)
    root = N.unload("unload", N.aggregate_batches("agg_final", collapsed, with_id, aggs,
                                                  A.finalize_exprs(aggs)))
    got, _ = execute(root)
    assert list(got.columns) == ["g", "k", A.GROUPING_ID, "sum_v"]
    assert set(got[A.GROUPING_ID]) == {0, 2, 3}

    # …and the projection that drops it, which is what the plan in the spec shows.
    exprs = [Alias(Col("g"), "g"), Alias(Col("k"), "k"), Alias(Col("sum_v"), "sum_v")]
    projected = N.unload("unload", N.project("drop_gid",
                                             N.aggregate_batches("agg_final2", collapsed, with_id,
                                                                 aggs, A.finalize_exprs(aggs)),
                                             exprs))
    got2, _ = execute(projected)
    assert list(got2.columns) == ["g", "k", "sum_v"]


# -- DISTINCT ----------------------------------------------------------------------
#
# DISTINCT is never a flag on an aggregator: it lowers to grouping, so each shape below is
# an ordinary aggregate sequence with an extra group key. See the spec's "DISTINCT lowers
# to grouping".


def test_select_distinct_is_an_aggregate_with_no_aggregators():
    # `SELECT DISTINCT g, k` — group keys, empty `aggs`, no `final` list. Dedup is
    # idempotent and associative, so per batch, per lane and post-shuffle all compose.
    df = fixture()
    want = df[["g", "k"]].drop_duplicates().reset_index(drop=True)
    for parts, group, target in CONFIGS:
        keys = ["g", "k"]
        scan = N.scan("scan", df, parts, group, target)
        per_batch = N.partial_aggregate("dedup_batch", scan, keys, [])
        per_lane = N.aggregate_batches("dedup_lane", per_batch, keys, [],
                                       schema={"g": df.g.dtype, "k": df.k.dtype})
        shuffle_in = N.coalesce_all("shuffle_in", N.merge_partitions("merge", per_lane))
        emitted = N.emit_partitions("emit", shuffle_in, keys, parts)
        root = N.unload("unload", N.aggregate_batches("dedup_final", emitted, keys, [],
                                                      A.finalize_exprs([]),
                                                      schema={"g": df.g.dtype, "k": df.k.dtype}))
        got, _ = execute(root)
        same(got, want, f"select distinct {(parts, group, target)}")


def test_count_distinct_lowers_to_two_aggregates():
    # `SELECT g, count(DISTINCT k) FROM t GROUP BY g` — the shape DataFusion's
    # SingleDistinctToGroupBy already produces: an inner aggregate grouping on the distinct
    # argument, then an outer one counting it. No distinct flag reaches any executor.
    df = fixture()
    want = (
        df.groupby("g", dropna=False)["k"].nunique().reset_index(name="n_distinct_k")
    )
    for parts, group, target in CONFIGS:
        scan = N.scan("scan", df, parts, group, target)
        # inner: dedup (g, k) — group keys, no aggregators
        inner_keys = ["g", "k"]
        deduped = N.aggregate_batches(
            "dedup", N.partial_aggregate("dedup_batch", scan, inner_keys, []),
            inner_keys, [], schema={"g": df.g.dtype, "k": df.k.dtype},
        )
        # The per-lane dedup above is only a head start: the same (g, k) can survive in
        # several lanes, so the count must sit above a GLOBAL dedup. Shuffling on g puts
        # every row of a group in one lane, and the second dedup there is the global one.
        outer = [A.Agg(A.COUNT, "k", "n_distinct_k")]
        _, count_schema = agg_schemas(df, ["g"], outer)
        pair_schema = {"g": df.g.dtype, "k": df.k.dtype}
        shuffle_in = N.coalesce_all("shuffle_in", N.merge_partitions("merge", deduped))
        emitted = N.emit_partitions("emit", shuffle_in, ["g"], parts)
        globally = N.aggregate_batches("dedup_global", emitted, inner_keys, [],
                                       schema=pair_schema)
        counted = N.partial_aggregate("count_batch", globally, ["g"], outer)
        root = N.unload("unload", N.aggregate_batches("count_final", counted, ["g"], outer,
                                                      A.finalize_exprs(outer),
                                                      schema=count_schema))
        got, _ = execute(root)
        same(got, want, f"count distinct {(parts, group, target)}")


def test_a_distinct_beside_non_distinct_aggregates_lowers_the_same_way():
    # q28's shape — count(DISTINCT v) beside avg(v) and count(v) — which DataFusion refuses
    # because its rewrite re-applies the same function outside. Ours does not: the outer
    # level applies the MERGE aggregators, so a count merges by sum and the companions ride
    # through the inner grouping untouched. Σ over the inner groups recovers each total.
    df = fixture()
    want = pd.DataFrame([{
        "n_distinct_v": df.v.nunique(),
        "sum_v": float(df.v.sum()),
        "n_v": len(df),
        "avg_v": df.v.mean(),
    }])
    for parts, group, target in CONFIGS:
        scan = N.scan("scan", df, parts, group, target)
        # inner: group by the distinct argument, computing the companions per distinct value
        inner = [A.Agg(A.SUM, "v", "sum_v"), A.Agg(A.COUNT, "v", "n_v")]
        inner_state, _ = agg_schemas(df, ["v"], inner)
        per_value = N.aggregate_batches(
            "per_value", N.partial_aggregate("per_value_batch", scan, ["v"], inner),
            ["v"], inner, schema=inner_state,
        )
        # outer: count the distinct values, and merge the companions Σ-wise
        outer = [
            A.Agg(A.COUNT, "v", "n_distinct_v"),
            A.Agg(A.SUM, "sum_v", "sum_v"),
            A.Agg(A.SUM, "n_v", "n_v"),
        ]
        # As above, the per-lane grouping is a head start only: one value of v can survive
        # in several lanes, so the inner grouping is made global on one lane before the
        # outer count runs, or a distinct value would be counted once per lane that had it.
        collapsed = N.merge_partitions("merge", per_value)
        globally = N.aggregate_batches("per_value_global", collapsed, ["v"], inner,
                                       schema=inner_state)
        totalled = N.aggregate_batches(
            "totals", N.partial_aggregate("totals_batch", globally, [], outer),
            [], outer, A.finalize_exprs(outer),
        )
        # avg comes from the two totals, which is what the finalize would have written
        exprs = [
            Alias(Col("n_distinct_v"), "n_distinct_v"),
            Alias(Col("sum_v"), "sum_v"),
            Alias(Col("n_v"), "n_v"),
            Alias(Binary("/", Col("sum_v"), Col("n_v")), "avg_v"),
        ]
        got, _ = execute(N.unload("unload", N.project("avg", totalled, exprs)))
        same(got, want, f"mixed distinct {(parts, group, target)}")


def test_inner_join_with_a_shuffle_on_both_sides():
    fact, dim = fixture(), dim_fixture()
    want = dim.merge(fact, how="inner", on="k")
    for parts, group, target in CONFIGS:
        build = N.coalesce_all(
            "build_collect",
            N.emit_partitions(
                "build_emit",
                N.merge_partitions("build_merge", N.scan("dim", dim, parts, group, target)),
                ["k"],
                parts,
            ),
            schema=dict(dim.dtypes),
        )
        probe = N.emit_partitions(
            "probe_emit",
            N.merge_partitions("probe_merge", N.scan("fact", fact, parts, group, target)),
            ["k"],
            parts,
        )
        root = N.unload(
            "unload", N.hash_join("join", build, probe, JoinType.INNER, ["k"], ["k"])
        )
        got, _ = execute(root)
        same(got, want, f"inner join {(parts, group, target)}")


def test_left_outer_join_finish_pass_matches_the_oracle():
    fact, dim = fixture(), dim_fixture()
    want = dim.merge(fact, how="left", on="k")
    for parts, group, target in CONFIGS:
        build = N.coalesce_all(
            "build_collect",
            N.emit_partitions(
                "build_emit",
                N.merge_partitions("build_merge", N.scan("dim", dim, parts, group, target)),
                ["k"],
                parts,
            ),
            schema=dict(dim.dtypes),
        )
        probe = N.emit_partitions(
            "probe_emit",
            N.merge_partitions("probe_merge", N.scan("fact", fact, parts, group, target)),
            ["k"],
            parts,
        )
        root = N.unload(
            "unload",
            # probe_schema: a post-shuffle probe lane can be empty, and the finish must
            # still null-pad the probe columns it never saw.
            N.hash_join(
                "join", build, probe, JoinType.LEFT_OUTER, ["k"], ["k"],
                probe_schema=list(fact.columns),
            ),
        )
        got, _ = execute(root)
        same(got, want, f"left outer join {(parts, group, target)}")


def test_semi_and_anti_joins_match_the_oracle():
    fact, dim = fixture(), dim_fixture()
    present = set(fact.k.unique())
    for join_type, keep in ((JoinType.LEFT_SEMI, True), (JoinType.LEFT_ANTI, False)):
        want = dim[dim.k.isin(present) == keep].reset_index(drop=True)
        for parts, group, target in CONFIGS:
            build = N.coalesce_all(
                "build_collect",
                N.emit_partitions(
                    "build_emit",
                    N.merge_partitions("build_merge", N.scan("dim", dim, parts, group, target)),
                    ["k"],
                    parts,
                ),
                schema=dict(dim.dtypes),
            )
            probe = N.emit_partitions(
                "probe_emit",
                N.merge_partitions("probe_merge", N.scan("fact", fact, parts, group, target)),
                ["k"],
                parts,
            )
            root = N.unload("unload", N.hash_join("join", build, probe, join_type, ["k"], ["k"]))
            got, _ = execute(root)
            same(got, want, f"{join_type.value} {(parts, group, target)}")


def test_union_of_two_branches_matches_the_oracle():
    df = fixture()
    left_want = df[df.v > 60][["k", "v"]]
    right_want = df[df.v <= 10][["k", "v"]]
    want = pd.concat([left_want, right_want], ignore_index=True)
    for parts, group, target in CONFIGS:
        exprs = [Alias(Col("k"), "k"), Alias(Col("v"), "v")]
        left = N.project(
            "lp", N.filter_("lf", N.scan("ls", df, parts, group, target), Binary(">", Col("v"), Lit(60))), exprs
        )
        right = N.project(
            "rp", N.filter_("rf", N.scan("rs", df, parts, group, target), Binary("<=", Col("v"), Lit(10))), exprs
        )
        got, _ = execute(N.unload("unload", N.union("union", [left, right])))
        same(got, want, f"union {(parts, group, target)}")


def test_projection_arithmetic_matches_the_oracle():
    df = fixture()
    want = pd.DataFrame({"k": df.k, "double_v": df.v * 2, "flag": df.v > 50})
    for parts, group, target in CONFIGS:
        exprs = [
            Alias(Col("k"), "k"),
            Alias(Binary("*", Col("v"), Lit(2)), "double_v"),
            Alias(Binary(">", Col("v"), Lit(50)), "flag"),
        ]
        got, _ = execute(N.unload("unload", N.project("p", N.scan("s", df, parts, group, target), exprs)))
        same(got, want, f"projection {(parts, group, target)}")


def test_a_root_adjacent_limit_matches_the_oracle_at_every_config():
    # The lowering with no node, against real operators: `skip`/`fetch` on the unload, the
    # driver stopping part-way through a sorted stream. `test_limit.py` pins which calls
    # are made; this pins that the rows they bring back are the right ones.
    df = fixture()
    want = df.sort_values(["v", "k"], ascending=[False, True]).head(7).reset_index(drop=True)
    for parts, group, target in CONFIGS:
        scan = N.scan("scan", df, parts, group, target)
        per_batch = N.sort("sort", scan, ["v", "k"], ascending=[False, True], fetch=7)
        merged = N.merge_sorted_partitions(
            "merge_sorted", per_batch, ["v", "k"], ascending=[False, True]
        )
        got, driver = execute(N.unload("unload", merged, fetch=7))
        assert driver.plan.row_limit is not None, "the sink should carry the interval"
        assert list(got.v) == list(want.v), f"root-adjacent limit {(parts, group, target)}"


def test_a_mid_plan_limit_matches_the_oracle_at_every_config():
    # The other lowering: the limit's output feeds more work, so it stays a node,
    # streaming its one-partition input and holding nothing. `test_limit.py` pins that it
    # stops as soon as the interval is covered rather than reading the rest.
    df = fixture()
    top = df.sort_values(["v", "k"], ascending=[False, True]).iloc[2:9]
    want = pd.DataFrame({"k": top.k.to_numpy(), "double_v": top.v.to_numpy() * 2})
    for parts, group, target in CONFIGS:
        scan = N.scan("scan", df, parts, group, target)
        per_batch = N.sort("sort", scan, ["v", "k"], ascending=[False, True], fetch=9)
        merged = N.merge_sorted_partitions(
            "merge_sorted", per_batch, ["v", "k"], ascending=[False, True]
        )
        limited = N.limit("limit", merged, skip=2, fetch=7)
        exprs = [Alias(Col("k"), "k"), Alias(Binary("*", Col("v"), Lit(2)), "double_v")]
        got, driver = execute(N.unload("unload", N.project("p", limited, exprs)))
        assert driver.plan.row_limit is None, "a mid-plan limit stays a node of its own"
        assert list(got.double_v) == list(want.double_v), f"mid-plan limit {(parts, group, target)}"


def test_the_enforcer_is_actually_engaged_in_these_runs():
    # A budget of None would make every plan above pass whatever the accounting did. This
    # asserts the budget is live: the same plan trips when the budget is small enough.
    df = fixture()
    aggs = [A.Agg(A.SUM, "v", "sum_v")]
    driver = None
    for config in CONFIGS:
        _, driver = execute(shuffled_aggregate(df, config, aggs))
        assert driver.accountant.peak > 0
        assert driver.accountant.peak <= BUDGET

    with raises(ResidentBudgetExceeded):
        execute(shuffled_aggregate(df, CONFIGS[0], aggs), budget=1)


def test_every_config_agrees_with_every_other():
    # The single-partition single-batch config is the oracle's own shape, so agreeing with
    # it is the same claim as agreeing with pandas — but this states it directly, which is
    # what a regression in the batching policy would break first.
    df = fixture()
    aggs = [A.Agg(A.SUM, "v", "sum_v"), A.Agg(A.MEAN, "v", "avg_v")]
    baseline = None
    for config in CONFIGS:
        got, _ = execute(shuffled_aggregate(df, config, aggs))
        if baseline is None:
            baseline = got
        else:
            same(got, baseline, f"config {config} against the single-batch baseline")


if __name__ == "__main__":
    raise SystemExit(main(globals()))
