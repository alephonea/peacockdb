"""TPC-H corpus queries: the oracle and the assertions.

The lowerings live in `plans_tpch.py`; the short plan-shape tests, the enforcer tests and
the layout sweeps live in `test_tpch.py`. Split because they are two different suites: these
read whole tables and run on manual dispatch, those run on every push in seconds.

**The oracle is a pandas equivalent written from the SQL**, one per query, unlike the TPC-DS
half which runs the query text through DuckDB. Both are here on purpose: a hand-written
oracle states what the query means in a second language and catches a lowering that answers a
differently-shaped question, and DuckDB catches a *reading* of the SQL that both sides of a
hand-written pair could share. Twenty-two of the first is affordable; seventy-one would not
have been.

Whole tables, the spec's own parameters: sampling was tried and abandoned (see `corpus.py`
for what it does to date-clustered tables).
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

import pathlib

import numpy as np
import pandas as pd

from . import corpus
from .corpus import BUDGET, agg_schemas, execute, in_order, same
from .harness import main, raises
from ..errors import ResidentBudgetExceeded
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import Alias, Binary, Case, Col, Lit
from ..operators.injection import HashMode, LayoutInjector, LayoutPreset
from ..operators.joins import JoinType
from ..plan import Plan
from .plans_tpch import (
    Q10_END, Q10_REPORT, Q10_START, Q11_FRACTION, Q11_NATION, Q12_END, Q12_START, Q14_END,
    Q14_START, Q18_REPORT, Q18_THRESHOLD, Q1_KEYS, Q3_CUTOFF, Q3_KEYS, Q4_END, Q4_START,
    Q5_END, Q5_START, Q7_END, Q7_KEYS, Q7_PAIR, Q7_START, QUERIES,
)
from .plans_tpch_more import (
    Q15_END, Q15_START, Q16_BRAND, Q16_KEYS, Q16_SIZES, Q17_BRAND, Q17_CONTAINER,
    Q17_FRACTION, Q19_BRANDS, Q19_CONTAINERS, Q19_INSTRUCT, Q19_QUANTITIES, Q19_SHIPMODES,
    Q19_SIZES, Q20_END, Q20_FRACTION, Q20_NATION, Q20_START, Q21_NATION, Q21_STATUS,
    Q22_CODES, Q2_REGION, Q2_REPORT, Q2_SIZE, Q8_END, Q8_NATION, Q8_REGION, Q8_START,
    Q8_TYPE, Q9_KEYS,
)

#: Enough rows for real skew and multi-batch lanes; small enough that the row-wise hash in
#: `partition_ids` does not dominate the suite's runtime.
SHUFFLE_ROWS = 20_000


def table(name: str, columns: list[str], limit: int | None = None) -> pd.DataFrame:
    """This file's tables, through the shared reader (`corpus.py`)."""
    return corpus.table("tpch", name, columns, limit)


# -- corpus queries ---------------------------------------------------------------
#
# The plans above are shapes; these are queries. The lowerings live in `plans_tpch.py` —
# a plan is a function of the schemas, so they are built from parquet footers alone when
# `plans.py` renders the goldens — and what lives here is the oracle and the assertions.
#
# **Whole tables, the spec's own parameters.** A sampled corpus was tried and abandoned:
# both benchmarks are written clustered by date, so a row prefix is one quarter of 1992, a
# row-group sample is a set of date windows, and two tables sampled independently join to
# nothing. Every one of those bit. These read lineitem's six million rows and answer the
# query that was asked, which is why they run on manual dispatch (exec-model-corpus.yml)
# rather than on every push, and why `PCK_SHARD` and `PCK_LAYOUT` exist.
#
# Each query runs at every layout in `corpus.LAYOUTS` and its plan is pinned in
# `scripts/exec_model/tpch.plans.txt`.

def test_corpus_q6_forecasting_revenue_change():
    lineitem = corpus.table(
        "tpch", "lineitem", ["l_shipdate", "l_discount", "l_quantity", "l_extendedprice"]
    )
    kept = lineitem[
        (lineitem.l_shipdate >= pd.Timestamp("1994-01-01"))
        & (lineitem.l_shipdate < pd.Timestamp("1995-01-01"))
        & (lineitem.l_discount >= 0.05)
        & (lineitem.l_discount <= 0.07)
        & (lineitem.l_quantity < 24.0)
    ]
    want = pd.DataFrame([{"revenue": (kept.l_extendedprice * kept.l_discount).sum()}])
    assert 0 < len(kept) < len(lineitem) / 40, "q6 keeps about 2% of lineitem"
    for label, got in corpus.run_layouts("tpch", "q6", QUERIES["q6"](corpus.reader("tpch"))):
        same(got, want, label)


# -- q1 ---------------------------------------------------------------------------

def test_corpus_q1_pricing_summary_report():
    columns = ["l_shipdate", "l_returnflag", "l_linestatus", "l_quantity",
               "l_extendedprice", "l_discount", "l_tax"]
    lineitem = corpus.table("tpch", "lineitem", columns)
    priced = lineitem.assign(
        disc_price=lineitem.l_extendedprice * (1 - lineitem.l_discount),
        charge=lineitem.l_extendedprice * (1 - lineitem.l_discount) * (1 + lineitem.l_tax),
    )
    kept = priced[priced.l_shipdate <= pd.Timestamp("1998-09-02")]
    want = (
        kept.groupby(Q1_KEYS, dropna=False)
        .agg(
            sum_qty=("l_quantity", "sum"), sum_base_price=("l_extendedprice", "sum"),
            sum_disc_price=("disc_price", "sum"), sum_charge=("charge", "sum"),
            avg_qty=("l_quantity", "mean"), avg_price=("l_extendedprice", "mean"),
            avg_disc=("l_discount", "mean"), count_order=("l_quantity", "size"),
        )
        .reset_index().sort_values(Q1_KEYS).reset_index(drop=True)
    )
    # The four (returnflag, linestatus) pairs TPC-H has, and every surviving row in exactly
    # one of them — the invariant a grouped aggregate owes.
    assert len(want) == 4 and want.count_order.sum() == len(kept), "q1 lost rows"
    for label, got in corpus.run_layouts("tpch", "q1", QUERIES["q1"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q3 ---------------------------------------------------------------------------

def test_corpus_q3_shipping_priority():
    customer = corpus.table("tpch", "customer", ["c_custkey", "c_mktsegment"])
    orders = corpus.table(
        "tpch", "orders", ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"]
    )
    lineitem = corpus.table(
        "tpch", "lineitem", ["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"]
    )
    oracle = (
        customer[customer.c_mktsegment == "BUILDING"]
        .merge(orders[orders.o_orderdate < Q3_CUTOFF], left_on="c_custkey", right_on="o_custkey")
        .merge(lineitem[lineitem.l_shipdate > Q3_CUTOFF],
               left_on="o_orderkey", right_on="l_orderkey")
    )
    oracle = oracle.assign(revenue=oracle.l_extendedprice * (1 - oracle.l_discount))
    want = (
        oracle.groupby(Q3_KEYS, dropna=False).agg(revenue=("revenue", "sum"))
        .reset_index()
        .sort_values(["revenue", "o_orderdate"], ascending=[False, True])
        .head(10).reset_index(drop=True)
        [["l_orderkey", "revenue", "o_orderdate", "o_shippriority"]]
    )
    assert len(want) == 10, "q3 returns its top ten"
    for label, got in corpus.run_layouts("tpch", "q3", QUERIES["q3"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q12 --------------------------------------------------------------------------

def test_corpus_q12_shipping_modes_and_order_priority():
    orders = corpus.table("tpch", "orders", ["o_orderkey", "o_orderpriority"])
    lineitem = corpus.table(
        "tpch", "lineitem",
        ["l_orderkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"],
    )
    kept = lineitem[
        lineitem.l_shipmode.isin(["MAIL", "SHIP"])
        & (lineitem.l_commitdate < lineitem.l_receiptdate)
        & (lineitem.l_shipdate < lineitem.l_commitdate)
        & (lineitem.l_receiptdate >= Q12_START)
        & (lineitem.l_receiptdate < Q12_END)
    ]
    oracle = orders.merge(kept, left_on="o_orderkey", right_on="l_orderkey")
    priority = oracle.o_orderpriority
    want = (
        oracle.assign(
            high_line=((priority == "1-URGENT") | (priority == "2-HIGH")).astype(int),
            low_line=((priority != "1-URGENT") & (priority != "2-HIGH")).astype(int),
        )
        .groupby("l_shipmode", dropna=False)
        .agg(high_line_count=("high_line", "sum"), low_line_count=("low_line", "sum"))
        .reset_index().sort_values("l_shipmode").reset_index(drop=True)
    )
    assert set(want.l_shipmode) == {"MAIL", "SHIP"}, "q12 reports both ship modes"
    for label, got in corpus.run_layouts("tpch", "q12", QUERIES["q12"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q4 ---------------------------------------------------------------------------

def test_corpus_q4_order_priority_checking():
    orders = corpus.table("tpch", "orders", ["o_orderkey", "o_orderdate", "o_orderpriority"])
    lineitem = corpus.table("tpch", "lineitem", ["l_orderkey", "l_commitdate", "l_receiptdate"])
    quarter = orders[(orders.o_orderdate >= Q4_START) & (orders.o_orderdate < Q4_END)]
    with_late = set(lineitem[lineitem.l_commitdate < lineitem.l_receiptdate].l_orderkey)
    want = (
        quarter[quarter.o_orderkey.isin(with_late)]
        .groupby("o_orderpriority", dropna=False)
        .agg(order_count=("o_orderkey", "size"))
        .reset_index().sort_values("o_orderpriority").reset_index(drop=True)
    )
    assert len(want) == 5, "TPC-H has five order priorities"
    for label, got in corpus.run_layouts("tpch", "q4", QUERIES["q4"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q5 ---------------------------------------------------------------------------

def test_corpus_q5_local_supplier_volume():
    region = corpus.table("tpch", "region", ["r_regionkey", "r_name"])
    nation = corpus.table("tpch", "nation", ["n_nationkey", "n_name", "n_regionkey"])
    supplier = corpus.table("tpch", "supplier", ["s_suppkey", "s_nationkey"])
    customer = corpus.table("tpch", "customer", ["c_custkey", "c_nationkey"])
    orders = corpus.table("tpch", "orders", ["o_orderkey", "o_custkey", "o_orderdate"])
    lineitem = corpus.table(
        "tpch", "lineitem", ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]
    )
    asian = nation.merge(region[region.r_name == "ASIA"],
                         left_on="n_regionkey", right_on="r_regionkey")[["n_nationkey", "n_name"]]
    oracle = (
        asian.merge(supplier, left_on="n_nationkey", right_on="s_nationkey")
        .merge(lineitem, left_on="s_suppkey", right_on="l_suppkey")
        .merge(orders[(orders.o_orderdate >= Q5_START) & (orders.o_orderdate < Q5_END)],
               left_on="l_orderkey", right_on="o_orderkey")
        .merge(customer, left_on="o_custkey", right_on="c_custkey")
    )
    oracle = oracle[oracle.c_nationkey == oracle.s_nationkey]
    want = (
        oracle.assign(revenue=oracle.l_extendedprice * (1 - oracle.l_discount))
        .groupby("n_name", dropna=False).agg(revenue=("revenue", "sum"))
        .reset_index().sort_values("revenue", ascending=False).reset_index(drop=True)
    )
    assert len(want) == 5, "ASIA has five nations"
    for label, got in corpus.run_layouts("tpch", "q5", QUERIES["q5"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q10 --------------------------------------------------------------------------

def test_corpus_q10_returned_item_reporting():
    customer = corpus.table(
        "tpch", "customer",
        ["c_custkey", "c_name", "c_acctbal", "c_nationkey", "c_address", "c_phone", "c_comment"],
    )
    nation = corpus.table("tpch", "nation", ["n_nationkey", "n_name"])
    orders = corpus.table("tpch", "orders", ["o_orderkey", "o_custkey", "o_orderdate"])
    lineitem = corpus.table(
        "tpch", "lineitem",
        ["l_orderkey", "l_extendedprice", "l_discount", "l_returnflag", "l_shipdate"],
    )
    oracle = (
        customer.merge(nation, left_on="c_nationkey", right_on="n_nationkey")
        .merge(orders[(orders.o_orderdate >= Q10_START) & (orders.o_orderdate < Q10_END)],
               left_on="c_custkey", right_on="o_custkey")
        .merge(lineitem[lineitem.l_returnflag == "R"],
               left_on="o_orderkey", right_on="l_orderkey")
    )
    want = (
        oracle.assign(revenue=oracle.l_extendedprice * (1 - oracle.l_discount))
        .groupby(Q10_REPORT, dropna=False).agg(revenue=("revenue", "sum"))
        .reset_index().sort_values("revenue", ascending=False)
        .head(20).reset_index(drop=True)
    )
    assert len(want) == 20, "q10 returns its top twenty"
    for label, got in corpus.run_layouts("tpch", "q10", QUERIES["q10"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q18 --------------------------------------------------------------------------

def test_corpus_q18_large_volume_customer():
    customer = corpus.table("tpch", "customer", ["c_custkey", "c_name"])
    orders = corpus.table(
        "tpch", "orders", ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"]
    )
    lineitem = corpus.table("tpch", "lineitem", ["l_orderkey", "l_quantity"])
    per_order = lineitem.groupby("l_orderkey", dropna=False).l_quantity.sum()
    big = set(per_order[per_order > Q18_THRESHOLD].index)
    joined = customer.merge(orders, left_on="c_custkey", right_on="o_custkey")[Q18_REPORT]
    oracle = joined[joined.o_orderkey.isin(big)].merge(
        lineitem, left_on="o_orderkey", right_on="l_orderkey"
    )
    want = (
        oracle.groupby(Q18_REPORT, dropna=False).agg(total_quantity=("l_quantity", "sum"))
        .reset_index()
        .sort_values(["o_totalprice", "o_orderdate"], ascending=[False, True])
        .head(100).reset_index(drop=True)
    )
    assert 0 < len(want) <= 100, "a handful of orders exceed 300 units"
    for label, got in corpus.run_layouts("tpch", "q18", QUERIES["q18"](corpus.reader("tpch"))):
        in_order(got, want, label)
# -- q7 ---------------------------------------------------------------------------


def test_corpus_q7_volume_shipping():
    supplier = corpus.table("tpch", "supplier", ["s_suppkey", "s_nationkey"])
    customer = corpus.table("tpch", "customer", ["c_custkey", "c_nationkey"])
    nation = corpus.table("tpch", "nation", ["n_nationkey", "n_name"])
    orders = corpus.table("tpch", "orders", ["o_orderkey", "o_custkey"])
    lineitem = corpus.table(
        "tpch", "lineitem",
        ["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"],
    )
    pair = nation[nation.n_name.isin(Q7_PAIR)]
    oracle = (
        pair.rename(columns={"n_name": "supp_nation"})
        .merge(supplier, left_on="n_nationkey", right_on="s_nationkey")
        .merge(lineitem[(lineitem.l_shipdate >= Q7_START) & (lineitem.l_shipdate <= Q7_END)],
               left_on="s_suppkey", right_on="l_suppkey")
        .merge(orders, left_on="l_orderkey", right_on="o_orderkey")
        .merge(customer, left_on="o_custkey", right_on="c_custkey")
        .merge(pair.rename(columns={"n_name": "cust_nation", "n_nationkey": "cust_nationkey"}),
               left_on="c_nationkey", right_on="cust_nationkey")
    )
    oracle = oracle[
        ((oracle.supp_nation == "FRANCE") & (oracle.cust_nation == "GERMANY"))
        | ((oracle.supp_nation == "GERMANY") & (oracle.cust_nation == "FRANCE"))
    ]
    want = (
        oracle.assign(l_year=oracle.l_shipdate.dt.year,
                      volume=oracle.l_extendedprice * (1 - oracle.l_discount))
        .groupby(Q7_KEYS, dropna=False).agg(revenue=("volume", "sum"))
        .reset_index().sort_values(Q7_KEYS).reset_index(drop=True)
    )
    assert len(want) == 4, "two nation pairs over two years"
    for label, got in corpus.run_layouts("tpch", "q7", QUERIES["q7"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q11 --------------------------------------------------------------------------


def test_corpus_q11_important_stock_identification():
    partsupp = corpus.table(
        "tpch", "partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"]
    )
    supplier = corpus.table("tpch", "supplier", ["s_suppkey", "s_nationkey"])
    nation = corpus.table("tpch", "nation", ["n_nationkey", "n_name"])
    german = (
        nation[nation.n_name == Q11_NATION]
        .merge(supplier, left_on="n_nationkey", right_on="s_nationkey")
        .merge(partsupp, left_on="s_suppkey", right_on="ps_suppkey")
    )
    german = german.assign(value=german.ps_supplycost * german.ps_availqty)
    threshold = german.value.sum() * Q11_FRACTION
    per_part = german.groupby("ps_partkey", dropna=False).agg(value=("value", "sum")).reset_index()
    want = (
        per_part[per_part.value > threshold]
        .sort_values("value", ascending=False).reset_index(drop=True)
    )
    assert len(want) > 100, "the threshold leaves the meaningful part of German stock"
    for label, got in corpus.run_layouts("tpch", "q11", QUERIES["q11"](corpus.reader("tpch"))):
        # ORDER BY value DESC and nothing else. Parts 56147 and 151929 both come to
        # 223626.0 exactly — an equal sort key, not float noise — so which of them comes
        # first is not the query's to determine, and every other row agrees. The sort key
        # is compared positionally, the rows as a multiset.
        in_order(got, want, label, order_by=["value"])


# -- q14 --------------------------------------------------------------------------


def test_corpus_q14_promotion_effect():
    lineitem = corpus.table(
        "tpch", "lineitem", ["l_partkey", "l_shipdate", "l_extendedprice", "l_discount"]
    )
    part = corpus.table("tpch", "part", ["p_partkey", "p_type"])
    in_month = lineitem[(lineitem.l_shipdate >= Q14_START) & (lineitem.l_shipdate < Q14_END)]
    oracle = part.merge(in_month, left_on="p_partkey", right_on="l_partkey")
    revenue = oracle.l_extendedprice * (1 - oracle.l_discount)
    promo = revenue.where(oracle.p_type.str.startswith("PROMO"), 0.0)
    want = pd.DataFrame([{"promo_revenue": 100.0 * promo.sum() / revenue.sum()}])
    assert 0 < want.promo_revenue.iloc[0] < 100, "a promotion share is a percentage"
    for label, got in corpus.run_layouts("tpch", "q14", QUERIES["q14"](corpus.reader("tpch"))):
        same(got, want, label)

# -- q2 ---------------------------------------------------------------------------


def test_corpus_q2_minimum_cost_supplier():
    part = corpus.table("tpch", "part", ["p_partkey", "p_mfgr", "p_size", "p_type"])
    supplier = corpus.table("tpch", "supplier", ["s_suppkey", "s_nationkey", "s_acctbal",
                                                 "s_name", "s_address", "s_phone", "s_comment"])
    partsupp = corpus.table("tpch", "partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost"])
    nation = corpus.table("tpch", "nation", ["n_nationkey", "n_name", "n_regionkey"])
    region = corpus.table("tpch", "region", ["r_regionkey", "r_name"])

    european = (
        region[region.r_name == Q2_REGION]
        .merge(nation, left_on="r_regionkey", right_on="n_regionkey")
        .merge(supplier, left_on="n_nationkey", right_on="s_nationkey")
        .merge(partsupp, left_on="s_suppkey", right_on="ps_suppkey")
    )
    mins = european.groupby("ps_partkey", dropna=False).ps_supplycost.min().rename("min_cost")
    wanted = part[(part.p_size == Q2_SIZE) & part.p_type.str.endswith("BRASS")]
    oracle = european.merge(wanted, left_on="ps_partkey", right_on="p_partkey").merge(
        mins, left_on="ps_partkey", right_index=True
    )
    want = (
        oracle[oracle.ps_supplycost == oracle.min_cost][Q2_REPORT]
        .sort_values(["s_acctbal", "n_name", "s_name", "p_partkey"],
                     ascending=[False, True, True, True])
        .head(100).reset_index(drop=True)
    )
    assert len(want) == 100, "q2 returns its top hundred"
    for label, got in corpus.run_layouts("tpch", "q2", QUERIES["q2"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q8 ---------------------------------------------------------------------------


def test_corpus_q8_national_market_share():
    part = corpus.table("tpch", "part", ["p_partkey", "p_type"])
    supplier = corpus.table("tpch", "supplier", ["s_suppkey", "s_nationkey"])
    lineitem = corpus.table(
        "tpch", "lineitem",
        ["l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount"],
    )
    orders = corpus.table("tpch", "orders", ["o_orderkey", "o_custkey", "o_orderdate"])
    customer = corpus.table("tpch", "customer", ["c_custkey", "c_nationkey"])
    nation = corpus.table("tpch", "nation", ["n_nationkey", "n_name", "n_regionkey"])
    region = corpus.table("tpch", "region", ["r_regionkey", "r_name"])

    american = (
        region[region.r_name == Q8_REGION]
        .merge(nation, left_on="r_regionkey", right_on="n_regionkey")
        .merge(customer, left_on="n_nationkey", right_on="c_nationkey")[["c_custkey"]]
    )
    in_range = orders[(orders.o_orderdate >= Q8_START) & (orders.o_orderdate <= Q8_END)]
    oracle = (
        part[part.p_type == Q8_TYPE]
        .merge(lineitem, left_on="p_partkey", right_on="l_partkey")
        .merge(supplier.merge(nation[["n_nationkey", "n_name"]],
                              left_on="s_nationkey", right_on="n_nationkey"),
               left_on="l_suppkey", right_on="s_suppkey")
        .merge(in_range, left_on="l_orderkey", right_on="o_orderkey")
        .merge(american, left_on="o_custkey", right_on="c_custkey")
    )
    volume = oracle.l_extendedprice * (1 - oracle.l_discount)
    oracle = oracle.assign(
        o_year=oracle.o_orderdate.dt.year, volume=volume,
        brazil=volume.where(oracle.n_name == Q8_NATION, 0.0),
    )
    grouped = oracle.groupby("o_year", dropna=False).agg(
        brazil=("brazil", "sum"), total=("volume", "sum")
    ).reset_index()
    want = pd.DataFrame({
        "o_year": grouped.o_year, "mkt_share": grouped.brazil / grouped.total
    }).sort_values("o_year").reset_index(drop=True)
    assert len(want) == 2, "q8 covers 1995 and 1996"
    for label, got in corpus.run_layouts("tpch", "q8", QUERIES["q8"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q9 ---------------------------------------------------------------------------


def test_corpus_q9_product_type_profit_measure():
    part = corpus.table("tpch", "part", ["p_partkey", "p_name"])
    supplier = corpus.table("tpch", "supplier", ["s_suppkey", "s_nationkey"])
    lineitem = corpus.table(
        "tpch", "lineitem",
        ["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount"],
    )
    partsupp = corpus.table("tpch", "partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost"])
    orders = corpus.table("tpch", "orders", ["o_orderkey", "o_orderdate"])
    nation = corpus.table("tpch", "nation", ["n_nationkey", "n_name"])

    oracle = (
        part[part.p_name.str.contains("green", regex=False)]
        .merge(lineitem, left_on="p_partkey", right_on="l_partkey")
        .merge(supplier.merge(nation, left_on="s_nationkey", right_on="n_nationkey"),
               left_on="l_suppkey", right_on="s_suppkey")
        .merge(partsupp, left_on=["l_partkey", "l_suppkey"],
               right_on=["ps_partkey", "ps_suppkey"])
        .merge(orders, left_on="l_orderkey", right_on="o_orderkey")
    )
    amount = (oracle.l_extendedprice * (1 - oracle.l_discount)
              - oracle.ps_supplycost * oracle.l_quantity)
    want = (
        oracle.assign(nation=oracle.n_name, o_year=oracle.o_orderdate.dt.year, amount=amount)
        .groupby(Q9_KEYS, dropna=False).agg(sum_profit=("amount", "sum"))
        .reset_index().sort_values(Q9_KEYS, ascending=[True, False]).reset_index(drop=True)
    )
    assert len(want) > 100, "every nation over seven years"
    for label, got in corpus.run_layouts("tpch", "q9", QUERIES["q9"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q13 --------------------------------------------------------------------------


def test_corpus_q13_customer_distribution():
    customer = corpus.table("tpch", "customer", ["c_custkey"])
    orders = corpus.table("tpch", "orders", ["o_orderkey", "o_custkey", "o_comment"])
    ordinary = orders[~orders.o_comment.str.contains("special.*requests", regex=True)]
    per_customer = (
        customer.merge(ordinary, how="left", left_on="c_custkey", right_on="o_custkey")
        .groupby("c_custkey", dropna=False).agg(c_count=("o_orderkey", "count"))
        .reset_index()
    )
    want = (
        per_customer.groupby("c_count", dropna=False).agg(custdist=("c_custkey", "size"))
        .reset_index()
        .sort_values(["custdist", "c_count"], ascending=[False, False]).reset_index(drop=True)
    )
    assert len(want) > 20, "the distribution has a long tail"
    for label, got in corpus.run_layouts("tpch", "q13", QUERIES["q13"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q15 --------------------------------------------------------------------------


def test_corpus_q15_top_supplier():
    lineitem = corpus.table(
        "tpch", "lineitem", ["l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
    )
    supplier = corpus.table("tpch", "supplier", ["s_suppkey", "s_name", "s_address", "s_phone"])
    in_quarter = lineitem[(lineitem.l_shipdate >= Q15_START) & (lineitem.l_shipdate < Q15_END)]
    revenue = (
        in_quarter.assign(
            revenue=in_quarter.l_extendedprice * (1 - in_quarter.l_discount)
        )
        .groupby("l_suppkey", dropna=False).agg(total_revenue=("revenue", "sum"))
        .reset_index().rename(columns={"l_suppkey": "supplier_no"})
    )
    best = revenue[revenue.total_revenue == revenue.total_revenue.max()]
    want = (
        supplier.merge(best, left_on="s_suppkey", right_on="supplier_no")
        [["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]]
        .sort_values("s_suppkey").reset_index(drop=True)
    )
    assert len(want) >= 1, "some supplier is the best"
    for label, got in corpus.run_layouts("tpch", "q15", QUERIES["q15"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q16 --------------------------------------------------------------------------


def test_corpus_q16_parts_supplier_relationship():
    partsupp = corpus.table("tpch", "partsupp", ["ps_partkey", "ps_suppkey"])
    part = corpus.table("tpch", "part", ["p_partkey", "p_brand", "p_type", "p_size"])
    supplier = corpus.table("tpch", "supplier", ["s_suppkey", "s_comment"])
    complaining = set(
        supplier[supplier.s_comment.str.contains("Customer.*Complaints", regex=True)].s_suppkey
    )
    wanted = part[
        (part.p_brand != Q16_BRAND)
        & ~part.p_type.str.startswith("MEDIUM POLISHED")
        & part.p_size.isin(Q16_SIZES)
    ]
    oracle = wanted.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
    oracle = oracle[~oracle.ps_suppkey.isin(complaining)]
    want = (
        oracle.groupby(Q16_KEYS, dropna=False).agg(supplier_cnt=("ps_suppkey", "nunique"))
        .reset_index()
        .sort_values(["supplier_cnt"] + Q16_KEYS, ascending=[False, True, True, True])
        .reset_index(drop=True)
    )
    assert len(want) > 1000, "q16 groups by brand, type and size"
    for label, got in corpus.run_layouts("tpch", "q16", QUERIES["q16"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q17 --------------------------------------------------------------------------


def test_corpus_q17_small_quantity_order_revenue():
    lineitem = corpus.table("tpch", "lineitem", ["l_partkey", "l_quantity", "l_extendedprice"])
    part = corpus.table("tpch", "part", ["p_partkey", "p_brand", "p_container"])
    thresholds = (
        lineitem.groupby("l_partkey", dropna=False).l_quantity.mean() * Q17_FRACTION
    ).rename("threshold")
    wanted = part[(part.p_brand == Q17_BRAND) & (part.p_container == Q17_CONTAINER)]
    oracle = wanted.merge(lineitem, left_on="p_partkey", right_on="l_partkey").merge(
        thresholds, left_on="l_partkey", right_index=True
    )
    small = oracle[oracle.l_quantity < oracle.threshold]
    want = pd.DataFrame([{"avg_yearly": small.l_extendedprice.sum() / 7.0}])
    assert want.avg_yearly.iloc[0] > 0, "some orders are below a fifth of the average"
    for label, got in corpus.run_layouts("tpch", "q17", QUERIES["q17"](corpus.reader("tpch"))):
        same(got, want, label)


# -- q19 --------------------------------------------------------------------------


def test_corpus_q19_discounted_revenue():
    lineitem = corpus.table(
        "tpch", "lineitem",
        ["l_partkey", "l_quantity", "l_extendedprice", "l_discount", "l_shipmode",
         "l_shipinstruct"],
    )
    part = corpus.table("tpch", "part", ["p_partkey", "p_brand", "p_container", "p_size"])
    joined = part.merge(lineitem, left_on="p_partkey", right_on="l_partkey")
    common = joined.l_shipmode.isin(Q19_SHIPMODES) & (joined.l_shipinstruct == Q19_INSTRUCT)
    keep = False
    for brand, containers, (low, high), size in zip(
        Q19_BRANDS, Q19_CONTAINERS, Q19_QUANTITIES, Q19_SIZES
    ):
        keep = keep | (
            (joined.p_brand == brand)
            & joined.p_container.isin(containers)
            & (joined.l_quantity >= low) & (joined.l_quantity <= high)
            & (joined.p_size >= 1) & (joined.p_size <= size)
            & common
        )
    matching = joined[keep]
    want = pd.DataFrame([{
        "revenue": (matching.l_extendedprice * (1 - matching.l_discount)).sum()
    }])
    assert want.revenue.iloc[0] > 0, "the three branches match something"
    for label, got in corpus.run_layouts("tpch", "q19", QUERIES["q19"](corpus.reader("tpch"))):
        same(got, want, label)


# -- q20 --------------------------------------------------------------------------


def test_corpus_q20_potential_part_promotion():
    supplier = corpus.table(
        "tpch", "supplier", ["s_suppkey", "s_name", "s_address", "s_nationkey"]
    )
    nation = corpus.table("tpch", "nation", ["n_nationkey", "n_name"])
    partsupp = corpus.table("tpch", "partsupp", ["ps_partkey", "ps_suppkey", "ps_availqty"])
    part = corpus.table("tpch", "part", ["p_partkey", "p_name"])
    lineitem = corpus.table(
        "tpch", "lineitem", ["l_partkey", "l_suppkey", "l_quantity", "l_shipdate"]
    )
    forest = part[part.p_name.str.startswith("forest")][["p_partkey"]]
    in_year = lineitem[(lineitem.l_shipdate >= Q20_START) & (lineitem.l_shipdate < Q20_END)]
    thresholds = (
        in_year.groupby(["l_partkey", "l_suppkey"], dropna=False).l_quantity.sum()
        * Q20_FRACTION
    ).rename("threshold").reset_index()
    candidates = (
        forest.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .merge(thresholds, left_on=["ps_partkey", "ps_suppkey"],
               right_on=["l_partkey", "l_suppkey"])
    )
    wanted = set(candidates[candidates.ps_availqty > candidates.threshold].ps_suppkey)
    canadian = nation[nation.n_name == Q20_NATION].merge(
        supplier, left_on="n_nationkey", right_on="s_nationkey"
    )
    want = (
        canadian[canadian.s_suppkey.isin(wanted)][["s_name", "s_address"]]
        .sort_values("s_name").reset_index(drop=True)
    )
    assert len(want) > 0, "some Canadian supplier is over-stocked on forest parts"
    for label, got in corpus.run_layouts("tpch", "q20", QUERIES["q20"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q21 --------------------------------------------------------------------------


def test_corpus_q21_suppliers_who_kept_orders_waiting():
    supplier = corpus.table("tpch", "supplier", ["s_suppkey", "s_name", "s_nationkey"])
    nation = corpus.table("tpch", "nation", ["n_nationkey", "n_name"])
    orders = corpus.table("tpch", "orders", ["o_orderkey", "o_orderstatus"])
    lineitem = corpus.table(
        "tpch", "lineitem", ["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]
    )
    saudi = nation[nation.n_name == Q21_NATION].merge(
        supplier, left_on="n_nationkey", right_on="s_nationkey"
    )[["s_suppkey", "s_name"]]
    late = lineitem[lineitem.l_receiptdate > lineitem.l_commitdate]
    base = (
        saudi.merge(late, left_on="s_suppkey", right_on="l_suppkey")
        .merge(orders[orders.o_orderstatus == Q21_STATUS],
               left_on="l_orderkey", right_on="o_orderkey")
    )[["l_orderkey", "l_suppkey", "s_name"]]

    # EXISTS: another lineitem of the same order, by a different supplier.
    others = lineitem[["l_orderkey", "l_suppkey"]].rename(
        columns={"l_suppkey": "other_suppkey"}
    )
    with_other = base.merge(others, on="l_orderkey")
    exists = set(map(tuple, with_other[with_other.l_suppkey != with_other.other_suppkey]
                     [["l_orderkey", "l_suppkey"]].to_numpy()))
    late_others = late[["l_orderkey", "l_suppkey"]].rename(
        columns={"l_suppkey": "other_suppkey"}
    )
    with_late_other = base.merge(late_others, on="l_orderkey")
    late_exists = set(map(tuple,
                          with_late_other[with_late_other.l_suppkey
                                          != with_late_other.other_suppkey]
                          [["l_orderkey", "l_suppkey"]].to_numpy()))
    keys = list(map(tuple, base[["l_orderkey", "l_suppkey"]].to_numpy()))
    kept = base[[key in exists and key not in late_exists for key in keys]]
    want = (
        kept.groupby("s_name", dropna=False).agg(numwait=("l_orderkey", "size"))
        .reset_index()
        .sort_values(["numwait", "s_name"], ascending=[False, True])
        .head(100).reset_index(drop=True)
    )
    assert len(want) == 100, "q21 returns its top hundred suppliers"
    for label, got in corpus.run_layouts("tpch", "q21", QUERIES["q21"](corpus.reader("tpch"))):
        in_order(got, want, label)


# -- q22 --------------------------------------------------------------------------


def test_corpus_q22_global_sales_opportunity():
    customer = corpus.table("tpch", "customer", ["c_custkey", "c_phone", "c_acctbal"])
    orders = corpus.table("tpch", "orders", ["o_custkey"])
    coded = customer.assign(cntrycode=customer.c_phone.str.slice(0, 2))
    in_codes = coded[coded.cntrycode.isin(Q22_CODES)]
    average = in_codes[in_codes.c_acctbal > 0.0].c_acctbal.mean()
    above = in_codes[in_codes.c_acctbal > average]
    without_orders = above[~above.c_custkey.isin(set(orders.o_custkey))]
    want = (
        without_orders.groupby("cntrycode", dropna=False)
        .agg(numcust=("c_custkey", "size"), totacctbal=("c_acctbal", "sum"))
        .reset_index().sort_values("cntrycode").reset_index(drop=True)
    )
    assert len(want) == len(Q22_CODES), "every country code has customers without orders"
    for label, got in corpus.run_layouts("tpch", "q22", QUERIES["q22"](corpus.reader("tpch"))):
        in_order(got, want, label)

if __name__ == "__main__":
    raise SystemExit(main(globals()))
