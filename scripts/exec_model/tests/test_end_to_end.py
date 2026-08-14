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

from .harness import main
from ..batch_partitioned_driver import batch_partitioned_driver
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


def execute(root):
    driver = batch_partitioned_driver(Plan.build(root), CpuBackendSelector())
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
        if np.issubdtype(want[column].dtype, np.number):
            assert np.allclose(left.astype(float), right.astype(float), equal_nan=True), (
                f"{label}: column {column} differs"
            )
        else:
            assert list(left) == list(right), f"{label}: column {column} differs"


# -- queries ----------------------------------------------------------------------


def shuffled_aggregate(df, config, aggs, keys=("g",)):
    parts, group, target = config
    keys = list(keys)
    scan = N.scan("scan", df, parts, group, target)
    filtered = N.filter_("filter", scan, Binary(">", Col("v"), Lit(20)))
    partial = N.partial_aggregate("agg_partial", filtered, keys, aggs)
    compacted = N.aggregate_batches("agg_batches", partial, keys, aggs, final=False)
    if parts == 1:
        return N.unload("unload", N.aggregate_batches("agg_final", compacted, keys, aggs, True))
    emitted = N.emit_partitions("emit", N.merge_partitions("merge", compacted), keys, parts)
    return N.unload("unload", N.aggregate_batches("agg_final", emitted, keys, aggs, True))


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
        compacted = N.aggregate_batches("agg_batches", partial, [], aggs, final=False)
        # Keyless needs no shuffle — collapse the lanes and finish once (the v1 shortcut).
        collapsed = N.merge_partitions("merge", compacted)
        root = N.unload("unload", N.aggregate_batches("agg_final", collapsed, [], aggs, True))
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
            schema=list(dim.columns),
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
            schema=list(dim.columns),
        )
        probe = N.emit_partitions(
            "probe_emit",
            N.merge_partitions("probe_merge", N.scan("fact", fact, parts, group, target)),
            ["k"],
            parts,
        )
        root = N.unload(
            "unload", N.hash_join("join", build, probe, JoinType.LEFT_OUTER, ["k"], ["k"])
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
                schema=list(dim.columns),
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
