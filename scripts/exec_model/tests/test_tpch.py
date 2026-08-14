"""Simple plans over real TPC-H tables, with the resident enforcer switched on.

Not the TPC-H queries — hand-built plans of the shapes the mode exists for (filter into a
shuffled aggregate, a small build side against a streamed probe, a top-N across lanes),
run over real column types, real cardinalities and real skew rather than the synthetic
fixtures of `test_end_to_end.py`. Each is checked against a pandas oracle over the same
frames, so the assertions do not depend on which dataset was found.

**Which dataset.** `testdata/tpch.sf1` when it exists, otherwise the committed
`testdata/tpch.minimal`. This file runs in the **cpp-cpu** job, right after
`generate_testdata.sh` produces sf1 — the rest of the prototype suite runs in cost-report,
which has no dataset. The fallback is for local runs where sf1 was never generated: for
the four tables used here the two are the same data (same schema, same row counts —
customer 150000, supplier 10000, nation 25, region 5), so the assertions hold either way.
`part` is deliberately unused: its schema differs between the two.

**The enforcer is on.** Every plan runs under a real budget, so the accounting path is
live and a regression that blew the resident set would trip rather than pass quietly.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .harness import main, raises
from ..batch_partitioned_driver import batch_partitioned_driver
from ..errors import ResidentBudgetExceeded
from ..node import CpuBackendSelector
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import Alias, Binary, Col, Lit
from ..operators.joins import JoinType
from ..plan import Plan

#: Generous against these plans' real peaks (~tens of MB) and far from unbounded, so the
#: enforcer is exercised on every call and a blown resident set fails the test.
BUDGET = 256 * 1024 * 1024

#: Enough rows for real skew and multi-batch lanes; small enough that the row-wise hash in
#: `partition_ids` does not dominate the suite's runtime.
SHUFFLE_ROWS = 20_000

_ROOT = pathlib.Path(__file__).resolve().parents[3] / "testdata"


def tpch_dir() -> pathlib.Path:
    for candidate in ("tpch.sf1", "tpch.minimal"):
        path = _ROOT / candidate
        if (path / "customer.parquet").exists():
            return path
    raise FileNotFoundError(
        f"no TPC-H tables under {_ROOT}; tpch.minimal is committed and should always be here"
    )


def table(name: str, columns: list[str], limit: int | None = None) -> pd.DataFrame:
    """Read columns, casting decimals to float64.

    TPC-H money columns are `decimal128(15, 2)`, and pandas has no decimal dtype — it
    hands back an object column of Python `Decimal`s, which will not multiply by a float
    literal. Casting is the prototype standing in for the decimal support `frame.py`
    already names as a known divergence, and it is a reminder of why the real engine
    carries precision and scale in the flat buffers rather than letting cuDF re-derive
    them (architecture.md's cuDF options table).
    """
    arrow = pq.read_table(tpch_dir() / f"{name}.parquet", columns=columns)
    decimals = [f.name for f in arrow.schema if pa.types.is_decimal(f.type)]
    frame = arrow.to_pandas()
    for column in decimals:
        frame[column] = frame[column].astype("float64")
    return frame.head(limit) if limit is not None else frame


def execute(root, budget: int | None = BUDGET):
    driver = batch_partitioned_driver(Plan.build(root), CpuBackendSelector(), budget)
    driver.run()
    frames = [b.frame for b in driver.results if len(b.frame)]
    got = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return got, driver


def same(got: pd.DataFrame, want: pd.DataFrame, label: str) -> None:
    """Row order is not part of the contract unless a sort says so."""
    assert list(got.columns) == list(want.columns), f"{label}: {list(got.columns)}"
    got = got.sort_values(list(got.columns)).reset_index(drop=True)
    want = want.sort_values(list(want.columns)).reset_index(drop=True)
    assert len(got) == len(want), f"{label}: {len(got)} rows vs {len(want)}"
    for column in want.columns:
        left, right = got[column].to_numpy(), want[column].to_numpy()
        if np.issubdtype(want[column].dtype, np.number):
            assert np.allclose(left.astype(float), right.astype(float), equal_nan=True), (
                f"{label}: column {column}"
            )
        else:
            assert list(left) == list(right), f"{label}: column {column}"


# -- the dataset ------------------------------------------------------------------


def test_the_tables_are_the_shape_both_datasets_share():
    # If this drifts, the fallback stops being equivalent and the tests below start
    # meaning different things depending on which dataset the host has.
    assert len(table("nation", ["n_nationkey"])) == 25
    assert len(table("region", ["r_regionkey"])) == 5
    assert len(table("supplier", ["s_suppkey"])) == 10_000
    assert len(table("customer", ["c_custkey"])) == 150_000


# -- plans ------------------------------------------------------------------------


def test_filter_into_a_shuffled_grouped_aggregate():
    # The motivating pipeline: load, filter, aggregate into few groups. The filter is real
    # (18132 of 20000 rows survive), and the five market segments hash into four lanes as
    # 2/2/1/0 — the low-cardinality imbalance a shuffle actually produces. TPC-H itself is
    # near-uniform across those segments, so this is lane imbalance, not data skew.
    customer = table("customer", ["c_custkey", "c_mktsegment", "c_acctbal"], SHUFFLE_ROWS)
    aggs = [
        A.Agg(A.SUM, "c_acctbal", "total"),
        A.Agg(A.MEAN, "c_acctbal", "avg_bal"),
        A.Agg(A.COUNT, None, "n"),
    ]
    lanes = 4
    scan = N.scan("customer", customer, lanes, 2000, 4000)
    filtered = N.filter_("positive", scan, Binary(">", Col("c_acctbal"), Lit(0.0)))
    partial = N.partial_aggregate("agg_partial", filtered, ["c_mktsegment"], aggs)
    compacted = N.aggregate_batches("agg_batches", partial, ["c_mktsegment"], aggs, False)
    emitted = N.emit_partitions(
        "emit", N.merge_partitions("merge", compacted), ["c_mktsegment"], lanes
    )
    root = N.unload(
        "unload", N.aggregate_batches("agg_final", emitted, ["c_mktsegment"], aggs, True)
    )
    got, driver = execute(root)

    kept = customer[customer.c_acctbal > 0]
    want = (
        kept.groupby("c_mktsegment", dropna=False)
        .agg(total=("c_acctbal", "sum"), avg_bal=("c_acctbal", "mean"), n=("c_acctbal", "size"))
        .reset_index()
    )
    same(got, want, "shuffled aggregate")
    assert driver.accountant.peak > 0
    assert driver.accountant.peak <= BUDGET


def test_a_small_build_side_against_a_streamed_probe():
    # The shape the mode is for: nation (25 rows) collected into one build batch, customer
    # streamed past it in many batches across four lanes.
    nation = table("nation", ["n_nationkey", "n_name"])
    customer = table("customer", ["c_custkey", "c_nationkey"], SHUFFLE_ROWS)

    build = N.coalesce_all("nation_all", N.scan("nation", nation, 1, 25), schema=list(nation.columns))
    probe = N.scan("customer", customer, 1, 2000)
    root = N.unload(
        "unload",
        N.hash_join("join", build, probe, JoinType.INNER, ["n_nationkey"], ["c_nationkey"]),
    )
    got, driver = execute(root)

    want = nation.merge(customer, how="inner", left_on="n_nationkey", right_on="c_nationkey")
    same(got, want, "nation ⋈ customer")
    assert driver.accountant.peak <= BUDGET


def test_a_join_over_two_small_tables():
    nation = table("nation", ["n_nationkey", "n_name", "n_regionkey"])
    region = table("region", ["r_regionkey", "r_name"])

    build = N.coalesce_all("region_all", N.scan("region", region, 1, 5), schema=list(region.columns))
    probe = N.scan("nation", nation, 1, 10)
    root = N.unload(
        "unload",
        N.hash_join("join", build, probe, JoinType.INNER, ["r_regionkey"], ["n_regionkey"]),
    )
    got, _ = execute(root)
    want = region.merge(nation, how="inner", left_on="r_regionkey", right_on="n_regionkey")
    same(got, want, "region ⋈ nation")


def test_a_semi_join_finds_the_nations_that_have_customers():
    # Exercises the finish pass over real data: matches are remembered across probe batches.
    nation = table("nation", ["n_nationkey", "n_name"])
    customer = table("customer", ["c_custkey", "c_nationkey"], SHUFFLE_ROWS)

    build = N.coalesce_all("nation_all", N.scan("nation", nation, 1, 25), schema=list(nation.columns))
    probe = N.scan("customer", customer, 1, 2000)
    root = N.unload(
        "unload",
        N.hash_join("join", build, probe, JoinType.LEFT_SEMI, ["n_nationkey"], ["c_nationkey"]),
    )
    got, _ = execute(root)
    want = nation[nation.n_nationkey.isin(set(customer.c_nationkey))].reset_index(drop=True)
    same(got, want, "semi join")


def test_a_top_n_across_lanes():
    customer = table("customer", ["c_custkey", "c_acctbal"], SHUFFLE_ROWS)
    lanes = 4
    scan = N.scan("customer", customer, lanes, 2000, 4000)
    per_batch = N.sort("sort", scan, ["c_acctbal", "c_custkey"], ascending=[False, True], fetch=20)
    merged = N.merge_sorted_partitions(
        "merge_sorted", per_batch, ["c_acctbal", "c_custkey"], ascending=[False, True], fetch=20
    )
    got, _ = execute(N.unload("unload", merged))

    want = customer.sort_values(["c_acctbal", "c_custkey"], ascending=[False, True]).head(20)
    # A top-N IS order-sensitive, so compare positionally.
    assert list(got.c_custkey) == list(want.c_custkey)


def test_a_keyless_aggregate_over_supplier():
    supplier = table("supplier", ["s_suppkey", "s_acctbal"])
    aggs = [
        A.Agg(A.SUM, "s_acctbal", "total"),
        A.Agg(A.MIN, "s_acctbal", "lo"),
        A.Agg(A.MAX, "s_acctbal", "hi"),
        A.Agg(A.COUNT, None, "n"),
    ]
    scan = N.scan("supplier", supplier, 2, 2500, 5000)
    partial = N.partial_aggregate("agg_partial", scan, [], aggs)
    compacted = N.aggregate_batches("agg_batches", partial, [], aggs, False)
    collapsed = N.merge_partitions("merge", compacted)   # keyless needs no shuffle
    got, _ = execute(N.unload("unload", N.aggregate_batches("agg_final", collapsed, [], aggs, True)))

    want = pd.DataFrame(
        [{
            "total": supplier.s_acctbal.sum(),
            "lo": supplier.s_acctbal.min(),
            "hi": supplier.s_acctbal.max(),
            "n": len(supplier),
        }]
    )
    same(got, want, "keyless aggregate")


def test_a_projection_over_real_column_types():
    customer = table("customer", ["c_custkey", "c_acctbal", "c_mktsegment"], SHUFFLE_ROWS)
    exprs = [
        Alias(Col("c_custkey"), "c_custkey"),
        Alias(Binary("*", Col("c_acctbal"), Lit(2.0)), "double_bal"),
        Alias(Binary(">", Col("c_acctbal"), Lit(5000.0)), "rich"),
    ]
    got, _ = execute(N.unload("unload", N.project("p", N.scan("c", customer, 2, 2000, 4000), exprs)))
    want = pd.DataFrame(
        {
            "c_custkey": customer.c_custkey,
            "double_bal": customer.c_acctbal * 2.0,
            "rich": customer.c_acctbal > 5000.0,
        }
    )
    same(got, want, "projection")


# -- the enforcer -----------------------------------------------------------------


def test_a_tight_budget_fails_a_real_plan_cleanly():
    # The enforcer trips on real data rather than only on synthetic fixtures, and it trips
    # as a clean query failure, not an allocator death.
    customer = table("customer", ["c_custkey", "c_acctbal"], SHUFFLE_ROWS)
    scan = N.scan("customer", customer, 2, 2000, 4000)
    root = N.unload("unload", N.coalesce_all("collect", scan))
    with raises(ResidentBudgetExceeded):
        execute(root, budget=64 * 1024)


def test_the_accumulator_is_what_makes_the_budget_bind():
    """Streaming the whole table fits a budget that collecting it does not.

    Same 150k rows, same scan, same budget — opposite outcomes. That is the claim the mode
    exists to make: with batches only the accumulator's state is mandatory residency, so a
    query fits in a budget the materialized table does not.

    The budget sits between the two measured peaks (1.28 MB streamed, 2.4 MB collected);
    both are asserted so a drift shows up as a specific number rather than as this test
    quietly stopping to discriminate.
    """
    customer = table("customer", ["c_custkey", "c_acctbal"])
    budget = 2 * 1024 * 1024
    scan_config = dict(n_partitions=2, rows_per_group=20_000, target_batch_rows=40_000)

    streamed = N.unload(
        "unload",
        N.filter_("f", N.scan("c", customer, **scan_config), Binary(">", Col("c_acctbal"), Lit(0.0))),
    )
    _, driver = execute(streamed, budget=budget)
    assert driver.accountant.peak < budget

    collected = N.unload("unload", N.coalesce_all("collect", N.scan("c", customer, **scan_config)))
    with raises(ResidentBudgetExceeded):
        execute(collected, budget=budget)

    # And the collected plan does complete once the budget covers the whole table.
    _, driver = execute(collected, budget=BUDGET)
    assert driver.accountant.peak > budget


if __name__ == "__main__":
    raise SystemExit(main(globals()))
