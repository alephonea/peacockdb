"""TPC-H query plans, part two: the subqueries and the wider joins.

Split from `plans_tpch.py` at the thousand-line bar (coding-style.md), not at a conceptual
seam — q1 through q19 are in the first file and q2 through q22 in this one because that is
where the line fell. Both halves are the same kind of thing: one hand lowering per query,
over a table provider that never reads a row. `plans_tpch.py` holds the registry that names
both.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

import pandas as pd

from . import corpus
from .corpus import agg_schemas
from .plan_helpers import (
    aggregate_by, aggregate_to_one_row, all_of, date, sorted_output,
)
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import (
    Alias, Binary, Case, Col, DatePart, Like, Lit, Substring,
)
from ..operators.join_types import JoinType

# -- q2 ---------------------------------------------------------------------------

Q2_SIZE, Q2_TYPE, Q2_REGION = 15, "%BRASS", "EUROPE"
Q2_REPORT = ["s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr",
             "s_address", "s_phone", "s_comment"]


def _european_partsupp(tag, region, nation, supplier, partsupp):
    """partsupp ⋈ supplier ⋈ nation ⋈ region='EUROPE'. Built twice: once for the minimum
    cost per part, once for the rows that match it."""
    europe = N.filter_(
        f"{tag}_europe", N.scan(f"{tag}_region", region, 1, 10),
        Binary("==", Col("r_name"), Lit(Q2_REGION)),
    )
    nations = N.project(
        f"{tag}_nations",
        corpus.build_join(f"{tag}_r_n", region, europe, N.scan(f"{tag}_nation", nation, 1, 25),
                          "r_regionkey", "n_regionkey"),
        [Alias(Col("n_nationkey"), "n_nationkey"), Alias(Col("n_name"), "n_name")],
    )
    suppliers = corpus.build_join(
        f"{tag}_n_s", nation, nations, N.scan(f"{tag}_supplier", supplier, 1, 10_000),
        "n_nationkey", "s_nationkey",
    )
    return corpus.build_join(
        f"{tag}_s_ps", supplier, suppliers,
        N.scan(f"{tag}_partsupp", partsupp, 1, 100_000, 100_000), "s_suppkey", "ps_suppkey",
    )


def plan_q2(t):
    """Minimum cost supplier: the correlated `min(ps_supplycost)` per part becomes an
    aggregate joined back on the part key, and the equality against it a filter over the
    joined row."""
    part = t("part", ["p_partkey", "p_mfgr", "p_size", "p_type"])
    supplier = t("supplier", ["s_suppkey", "s_nationkey", "s_acctbal", "s_name",
                              "s_address", "s_phone", "s_comment"])
    partsupp = t("partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost"])
    nation = t("nation", ["n_nationkey", "n_name", "n_regionkey"])
    region = t("region", ["r_regionkey", "r_name"])

    mins = N.project(
        "min_keys",
        aggregate_by(
            "min", _european_partsupp("min", region, nation, supplier, partsupp),
            ["ps_partkey"], [A.Agg(A.MIN, "ps_supplycost", "min_cost")],
            schema_frame=corpus.schema_of(partsupp),
        ),
        [Alias(Col("ps_partkey"), "min_partkey"), Alias(Col("min_cost"), "min_cost")],
    )
    wanted_parts = N.filter_(
        "wanted", N.scan("part", part, 1, 100_000, 100_000),
        Binary("and", Binary("==", Col("p_size"), Lit(Q2_SIZE)), Like(Col("p_type"), Q2_TYPE)),
    )
    with_parts = corpus.build_join(
        "p_ps", part, wanted_parts, _european_partsupp("all", region, nation, supplier, partsupp),
        "p_partkey", "ps_partkey",
    )
    with_mins = N.filter_(
        "at_min",
        corpus.build_join("min_ps", part, mins, with_parts, "min_partkey", "ps_partkey"),
        Binary("==", Col("ps_supplycost"), Col("min_cost")),
    )
    selected = N.project(
        "select", with_mins, [Alias(Col(column), column) for column in Q2_REPORT]
    )
    order = ["s_acctbal", "n_name", "s_name", "p_partkey"]
    return N.unload(
        "unload", sorted_output("sort", selected, order, [False, True, True, True], fetch=100)
    )


# -- q8 ---------------------------------------------------------------------------

Q8_START, Q8_END = pd.Timestamp("1995-01-01"), pd.Timestamp("1996-12-31")
Q8_TYPE, Q8_REGION, Q8_NATION = "ECONOMY ANODIZED STEEL", "AMERICA", "BRAZIL"


def plan_q8(t):
    """National market share: eight tables, `nation` twice again, and a ratio of two sums
    per year — the CASE picks the numerator's rows and the projection divides once, over
    the aggregate's output."""
    part = t("part", ["p_partkey", "p_type"])
    supplier = t("supplier", ["s_suppkey", "s_nationkey"])
    lineitem = t("lineitem", ["l_orderkey", "l_partkey", "l_suppkey",
                              "l_extendedprice", "l_discount"])
    orders = t("orders", ["o_orderkey", "o_custkey", "o_orderdate"])
    customer = t("customer", ["c_custkey", "c_nationkey"])
    nation = t("nation", ["n_nationkey", "n_name", "n_regionkey"])
    region = t("region", ["r_regionkey", "r_name"])

    america = N.filter_(
        "america", N.scan("region", region, 1, 10), Binary("==", Col("r_name"), Lit(Q8_REGION))
    )
    american_nations = N.project(
        "am_nations",
        corpus.build_join("r_n1", region, america, N.scan("nation1", nation, 1, 25),
                          "r_regionkey", "n_regionkey"),
        [Alias(Col("n_nationkey"), "n_nationkey")],
    )
    american_customers = N.project(
        "am_customers",
        corpus.build_join("n1_c", nation, american_nations,
                          N.scan("customer", customer, 1, 50_000, 50_000),
                          "n_nationkey", "c_nationkey"),
        [Alias(Col("c_custkey"), "c_custkey")],
    )
    american_orders = N.project(
        "am_orders",
        corpus.build_join(
            "c_o", customer, american_customers,
            N.filter_("in_range", N.scan("orders", orders, 1, 100_000, 100_000),
                      Binary("and", Binary(">=", Col("o_orderdate"), date("1995-01-01")),
                             Binary("<=", Col("o_orderdate"), date("1996-12-31")))),
            "c_custkey", "o_custkey",
        ),
        [Alias(Col("o_orderkey"), "o_orderkey"),
         Alias(DatePart("year", Col("o_orderdate")), "o_year")],
    )
    supplier_nations = N.project(
        "supp_nations",
        corpus.build_join("n2_s", nation, N.scan("nation2", nation, 1, 25),
                          N.scan("supplier", supplier, 1, 10_000),
                          "n_nationkey", "s_nationkey"),
        [Alias(Col("s_suppkey"), "s_suppkey"), Alias(Col("n_name"), "nation")],
    )
    wanted_parts = N.filter_(
        "wanted", N.scan("part", part, 1, 100_000, 100_000),
        Binary("==", Col("p_type"), Lit(Q8_TYPE)),
    )
    with_parts = corpus.build_join(
        "p_l", part, wanted_parts, N.scan("lineitem", lineitem, 1, 250_000, 250_000),
        "p_partkey", "l_partkey",
    )
    with_suppliers = corpus.build_join(
        "s_l", supplier, supplier_nations, with_parts, "s_suppkey", "l_suppkey"
    )
    joined = corpus.build_join(
        "o_l", orders, american_orders, with_suppliers, "o_orderkey", "l_orderkey"
    )
    volume = Binary("*", Col("l_extendedprice"), Binary("-", Lit(1.0), Col("l_discount")))
    projected = N.project(
        "all_nations", joined,
        [Alias(Col("o_year"), "o_year"), Alias(volume, "volume"),
         Alias(Case(whens=((Binary("==", Col("nation"), Lit(Q8_NATION)), volume),),
                    otherwise=Lit(0.0)), "brazil_volume")],
    )
    final = aggregate_by(
        "agg", projected, ["o_year"],
        [A.Agg(A.SUM, "brazil_volume", "brazil"), A.Agg(A.SUM, "volume", "total")],
        schema_frame=corpus.schema_of(o_year="int64", volume="float64",
                                      brazil_volume="float64"),
    )
    share = N.project(
        "share", final,
        [Alias(Col("o_year"), "o_year"),
         Alias(Binary("/", Col("brazil"), Col("total")), "mkt_share")],
    )
    return N.unload("unload", sorted_output("sort", share, ["o_year"], [True]))


# -- q9 ---------------------------------------------------------------------------

Q9_PATTERN = "%green%"
Q9_KEYS = ["nation", "o_year"]


def plan_q9(t):
    """Product type profit measure: the only corpus query joining on **two** columns at
    once — partsupp is matched on (l_partkey, l_suppkey), which is one `CudfHashJoin` with
    two key pairs rather than a join plus a filter."""
    part = t("part", ["p_partkey", "p_name"])
    supplier = t("supplier", ["s_suppkey", "s_nationkey"])
    lineitem = t("lineitem", ["l_orderkey", "l_partkey", "l_suppkey", "l_quantity",
                              "l_extendedprice", "l_discount"])
    partsupp = t("partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost"])
    orders = t("orders", ["o_orderkey", "o_orderdate"])
    nation = t("nation", ["n_nationkey", "n_name"])

    green_parts = N.filter_(
        "green", N.scan("part", part, 1, 100_000, 100_000), Like(Col("p_name"), Q9_PATTERN)
    )
    suppliers = N.project(
        "supp_nations",
        corpus.build_join("n_s", nation, N.scan("nation", nation, 1, 25),
                          N.scan("supplier", supplier, 1, 10_000),
                          "n_nationkey", "s_nationkey"),
        [Alias(Col("s_suppkey"), "s_suppkey"), Alias(Col("n_name"), "nation")],
    )
    with_parts = corpus.build_join(
        "p_l", part, green_parts, N.scan("lineitem", lineitem, 1, 250_000, 250_000),
        "p_partkey", "l_partkey",
    )
    with_suppliers = corpus.build_join(
        "s_l", supplier, suppliers, with_parts, "s_suppkey", "l_suppkey"
    )
    with_costs = N.hash_join(
        "ps_l",
        N.coalesce_all("ps_all", N.scan("partsupp", partsupp, 1, 200_000, 200_000)),
        with_suppliers, JoinType.INNER,
        ["ps_partkey", "ps_suppkey"], ["l_partkey", "l_suppkey"],
    )
    joined = corpus.build_join(
        "o_l", orders, N.scan("orders", orders, 1, 100_000, 100_000),
        with_costs, "o_orderkey", "l_orderkey",
    )
    amount = Binary(
        "-", Binary("*", Col("l_extendedprice"), Binary("-", Lit(1.0), Col("l_discount"))),
        Binary("*", Col("ps_supplycost"), Col("l_quantity")),
    )
    projected = N.project(
        "profit", joined,
        [Alias(Col("nation"), "nation"),
         Alias(DatePart("year", Col("o_orderdate")), "o_year"), Alias(amount, "amount")],
    )
    final = aggregate_by(
        "agg", projected, Q9_KEYS, [A.Agg(A.SUM, "amount", "sum_profit")],
        schema_frame=corpus.schema_of(nation=nation.n_name.dtype, o_year="int64",
                                      amount="float64").rename(columns={"nation": "nation"}),
    )
    return N.unload("unload", sorted_output("sort", final, Q9_KEYS, [True, False]))


# -- q13 --------------------------------------------------------------------------

Q13_PATTERN = "%special%requests%"


def plan_q13(t):
    """Customer distribution: a left outer join whose ON carries a predicate on the
    nullable side, so the planner pushes it under the probe — which is why this query does
    **not** meet #153. Then a count of counts: two aggregates, the second grouping by the
    first's output."""
    customer = t("customer", ["c_custkey"])
    orders = t("orders", ["o_orderkey", "o_custkey", "o_comment"])

    ordinary = N.filter_(
        "ordinary", N.scan("orders", orders, 1, 100_000, 100_000),
        Like(Col("o_comment"), Q13_PATTERN, negated=True),
    )
    joined = corpus.build_join(
        "c_o", customer, N.scan("customer", customer, 1, 50_000, 50_000),
        ordinary, "c_custkey", "o_custkey", JoinType.LEFT,
    )
    per_customer = aggregate_by(
        "per_customer", joined, ["c_custkey"], [A.Agg(A.COUNT, "o_orderkey", "c_count")],
        schema_frame=corpus.schema_of(customer, orders),
    )
    distribution = aggregate_by(
        "distribution", per_customer, ["c_count"], [A.Agg(A.COUNT, None, "custdist")],
        schema_frame=corpus.schema_of(customer, c_count="int64"),
    )
    return N.unload(
        "unload", sorted_output("sort", distribution, ["custdist", "c_count"], [False, False])
    )


# -- q15 --------------------------------------------------------------------------

Q15_START, Q15_END = pd.Timestamp("1996-01-01"), pd.Timestamp("1996-04-01")


def _revenue0(tag, lineitem):
    """The query's `revenue0` view: revenue per supplier over one quarter. Built twice —
    once to find the maximum, once to match it — which is what inlining a CTE costs."""
    in_quarter = N.filter_(
        f"{tag}_quarter", N.scan(f"{tag}_lineitem", lineitem, 1, 250_000, 250_000),
        Binary("and", Binary(">=", Col("l_shipdate"), date("1996-01-01")),
               Binary("<", Col("l_shipdate"), date("1996-04-01"))),
    )
    revenue = Binary("*", Col("l_extendedprice"), Binary("-", Lit(1.0), Col("l_discount")))
    projected = N.project(
        f"{tag}_revenue", in_quarter,
        [Alias(Col("l_suppkey"), "supplier_no"), Alias(revenue, "revenue")],
    )
    return aggregate_by(
        f"{tag}_agg", projected, ["supplier_no"], [A.Agg(A.SUM, "revenue", "total_revenue")],
        schema_frame=corpus.schema_of(supplier_no=lineitem.l_suppkey.dtype,
                                      revenue="float64"),
    )


def plan_q15(t):
    """Top supplier: an aggregate compared against its own maximum. The maximum is one row,
    so the comparison is again a nested-loop join carrying it as a predicate."""
    lineitem = t("lineitem", ["l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"])
    supplier = t("supplier", ["s_suppkey", "s_name", "s_address", "s_phone"])

    best = N.project(
        "best",
        aggregate_to_one_row(
            "max", _revenue0("max", lineitem),
            [A.Agg(A.MAX, "total_revenue", "max_revenue")],
            corpus.schema_of(total_revenue="float64"),
        ),
        [Alias(Col("max_revenue"), "max_revenue")],
    )
    at_max = N.nested_loop_join(
        "at_max", N.coalesce_all("best_all", best), _revenue0("all", lineitem),
        JoinType.INNER, Binary("==", Col("total_revenue"), Col("max_revenue")),
    )
    joined = corpus.build_join(
        "s_r", supplier, N.scan("supplier", supplier, 1, 10_000), at_max,
        "s_suppkey", "supplier_no",
    )
    selected = N.project(
        "select", joined,
        [Alias(Col("s_suppkey"), "s_suppkey"), Alias(Col("s_name"), "s_name"),
         Alias(Col("s_address"), "s_address"), Alias(Col("s_phone"), "s_phone"),
         Alias(Col("total_revenue"), "total_revenue")],
    )
    return N.unload("unload", sorted_output("sort", selected, ["s_suppkey"], [True]))


# -- q16 --------------------------------------------------------------------------

Q16_BRAND, Q16_TYPE = "Brand#45", "MEDIUM POLISHED%"
Q16_SIZES = (49, 14, 23, 45, 19, 3, 36, 9)
Q16_KEYS = ["p_brand", "p_type", "p_size"]


def plan_q16(t):
    """Parts/supplier relationship: `NOT IN (…)` over the streamed side, which is a
    **RightAnti** in this orientation — the complaining suppliers are the small build side
    and the partsupp stream is what the query emits. Then `count(distinct ps_suppkey)`,
    which lowers to a dedup and a count, never a flag on an aggregator."""
    partsupp = t("partsupp", ["ps_partkey", "ps_suppkey"])
    part = t("part", ["p_partkey", "p_brand", "p_type", "p_size"])
    supplier = t("supplier", ["s_suppkey", "s_comment"])

    sizes = Binary("==", Col("p_size"), Lit(Q16_SIZES[0]))
    for size in Q16_SIZES[1:]:
        sizes = Binary("or", sizes, Binary("==", Col("p_size"), Lit(size)))
    wanted_parts = N.filter_(
        "wanted", N.scan("part", part, 1, 100_000, 100_000),
        all_of(Binary("!=", Col("p_brand"), Lit(Q16_BRAND)),
               Like(Col("p_type"), Q16_TYPE, negated=True), sizes),
    )
    complaining = N.project(
        "complaining",
        N.filter_("complaints", N.scan("supplier", supplier, 1, 10_000),
                  Like(Col("s_comment"), "%Customer%Complaints%")),
        [Alias(Col("s_suppkey"), "s_suppkey")],
    )
    with_parts = corpus.build_join(
        "p_ps", part, wanted_parts, N.scan("partsupp", partsupp, 1, 200_000, 200_000),
        "p_partkey", "ps_partkey",
    )
    without_complaints = corpus.build_join(
        "not_in", supplier, complaining, with_parts, "s_suppkey", "ps_suppkey",
        JoinType.RIGHT_ANTI,
    )
    # count(distinct ps_suppkey): dedup on the group keys plus the distinct argument, then
    # count the survivors per group. The DISTINCT lowering, on corpus data.
    deduped = aggregate_by("dedup", without_complaints, Q16_KEYS + ["ps_suppkey"], [],
                           schema_frame=corpus.schema_of(part, partsupp))
    counted = aggregate_by(
        "count", deduped, Q16_KEYS, [A.Agg(A.COUNT, "ps_suppkey", "supplier_cnt")],
        schema_frame=corpus.schema_of(part, partsupp),
    )
    order = ["supplier_cnt"] + Q16_KEYS
    return N.unload("unload", sorted_output("sort", counted, order, [False, True, True, True]))

# -- q17 --------------------------------------------------------------------------

Q17_BRAND, Q17_CONTAINER, Q17_FRACTION = "Brand#23", "MED BOX", 0.2


def plan_q17(t):
    """Small-quantity-order revenue: a correlated `avg(l_quantity)` per part becomes an
    aggregate over the whole of lineitem, joined back on the part key — and the comparison
    against it a filter over the joined row. The average is over *every* lineitem of that
    part, not only the filtered ones, which is why the aggregate sits on its own scan."""
    lineitem = t("lineitem", ["l_partkey", "l_quantity", "l_extendedprice"])
    part = t("part", ["p_partkey", "p_brand", "p_container"])

    averages = N.project(
        "thresholds",
        aggregate_by(
            "avg", N.scan("lineitem_avg", lineitem, 1, 250_000, 250_000),
            ["l_partkey"], [A.Agg(A.MEAN, "l_quantity", "avg_quantity")],
            schema_frame=corpus.schema_of(lineitem),
        ),
        [Alias(Col("l_partkey"), "avg_partkey"),
         Alias(Binary("*", Lit(Q17_FRACTION), Col("avg_quantity")), "threshold")],
    )
    wanted_parts = N.filter_(
        "wanted", N.scan("part", part, 1, 100_000, 100_000),
        Binary("and", Binary("==", Col("p_brand"), Lit(Q17_BRAND)),
               Binary("==", Col("p_container"), Lit(Q17_CONTAINER))),
    )
    with_parts = corpus.build_join(
        "p_l", part, wanted_parts, N.scan("lineitem", lineitem, 1, 250_000, 250_000),
        "p_partkey", "l_partkey",
    )
    with_thresholds = N.filter_(
        "small",
        corpus.build_join("avg_l", part, averages, with_parts, "avg_partkey", "l_partkey"),
        Binary("<", Col("l_quantity"), Col("threshold")),
    )
    total = aggregate_to_one_row(
        "agg", with_thresholds, [A.Agg(A.SUM, "l_extendedprice", "total")],
        corpus.schema_of(lineitem),
    )
    yearly = N.project(
        "yearly", total, [Alias(Binary("/", Col("total"), Lit(7.0)), "avg_yearly")]
    )
    return N.unload("unload", yearly)


# -- q19 --------------------------------------------------------------------------

Q19_BRANDS = ("Brand#12", "Brand#23", "Brand#34")
Q19_CONTAINERS = (
    ("SM CASE", "SM BOX", "SM PACK", "SM PKG"),
    ("MED BAG", "MED BOX", "MED PKG", "MED PACK"),
    ("LG CASE", "LG BOX", "LG PACK", "LG PKG"),
)
Q19_QUANTITIES = ((1, 11), (10, 20), (20, 30))
Q19_SIZES = (5, 10, 15)
Q19_SHIPMODES = ("AIR", "AIR REG")
Q19_INSTRUCT = "DELIVER IN PERSON"


def _any_of(column, values):
    """`column IN (…)` as the OR-chain `gpu_rule.rs` lowers an IN-list to."""
    predicate = Binary("==", Col(column), Lit(values[0]))
    for value in values[1:]:
        predicate = Binary("or", predicate, Binary("==", Col(column), Lit(value)))
    return predicate


def plan_q19(t):
    """Discounted revenue: three alternative AND-blocks over the joined row, OR'd. Every
    term is a comparison the AST has, so this is one filter node — a shape that says more
    about the expression IR than about the plan."""
    lineitem = t("lineitem", ["l_partkey", "l_quantity", "l_extendedprice", "l_discount",
                              "l_shipmode", "l_shipinstruct"])
    part = t("part", ["p_partkey", "p_brand", "p_container", "p_size"])

    common = Binary("and", _any_of("l_shipmode", Q19_SHIPMODES),
                    Binary("==", Col("l_shipinstruct"), Lit(Q19_INSTRUCT)))
    branches = []
    for brand, containers, (low, high), size in zip(
        Q19_BRANDS, Q19_CONTAINERS, Q19_QUANTITIES, Q19_SIZES
    ):
        branches.append(all_of(
            Binary("==", Col("p_brand"), Lit(brand)),
            _any_of("p_container", containers),
            Binary(">=", Col("l_quantity"), Lit(float(low))),
            Binary("<=", Col("l_quantity"), Lit(float(high))),
            Binary(">=", Col("p_size"), Lit(1)),
            Binary("<=", Col("p_size"), Lit(size)),
            common,
        ))
    predicate = branches[0]
    for branch in branches[1:]:
        predicate = Binary("or", predicate, branch)

    joined = corpus.build_join(
        "p_l", part, N.scan("part", part, 1, 100_000, 100_000),
        N.scan("lineitem", lineitem, 1, 250_000, 250_000), "p_partkey", "l_partkey",
    )
    matching = N.filter_("branches", joined, predicate)
    revenue = Binary("*", Col("l_extendedprice"), Binary("-", Lit(1.0), Col("l_discount")))
    projected = N.project("revenue", matching, [Alias(revenue, "revenue")])
    return N.unload(
        "unload",
        aggregate_to_one_row("agg", projected, [A.Agg(A.SUM, "revenue", "revenue")],
                             corpus.schema_of(revenue="float64")),
    )


# -- q20 --------------------------------------------------------------------------

Q20_PATTERN, Q20_NATION, Q20_FRACTION = "forest%", "CANADA", 0.5
Q20_START, Q20_END = pd.Timestamp("1994-01-01"), pd.Timestamp("1995-01-01")


def plan_q20(t):
    """Potential part promotion: two nested `IN` subqueries and a correlated sum. The inner
    correlation is on **two** columns, so the threshold aggregate is grouped by both and
    joined back on both."""
    supplier = t("supplier", ["s_suppkey", "s_name", "s_address", "s_nationkey"])
    nation = t("nation", ["n_nationkey", "n_name"])
    partsupp = t("partsupp", ["ps_partkey", "ps_suppkey", "ps_availqty"])
    part = t("part", ["p_partkey", "p_name"])
    lineitem = t("lineitem", ["l_partkey", "l_suppkey", "l_quantity", "l_shipdate"])

    forest = N.project(
        "forest",
        N.filter_("forest_parts", N.scan("part", part, 1, 100_000, 100_000),
                  Like(Col("p_name"), Q20_PATTERN)),
        [Alias(Col("p_partkey"), "p_partkey")],
    )
    thresholds = N.project(
        "thresholds",
        aggregate_by(
            "shipped",
            N.filter_("in_year", N.scan("lineitem", lineitem, 1, 250_000, 250_000),
                      Binary("and", Binary(">=", Col("l_shipdate"), date("1994-01-01")),
                             Binary("<", Col("l_shipdate"), date("1995-01-01")))),
            ["l_partkey", "l_suppkey"], [A.Agg(A.SUM, "l_quantity", "shipped_quantity")],
            schema_frame=corpus.schema_of(lineitem),
        ),
        [Alias(Col("l_partkey"), "t_partkey"), Alias(Col("l_suppkey"), "t_suppkey"),
         Alias(Binary("*", Lit(Q20_FRACTION), Col("shipped_quantity")), "threshold")],
    )
    candidates = corpus.build_join(
        "p_ps", part, forest, N.scan("partsupp", partsupp, 1, 200_000, 200_000),
        "p_partkey", "ps_partkey",
    )
    with_thresholds = N.filter_(
        "over",
        N.hash_join(
            "t_ps", N.coalesce_all("thresholds_all", thresholds), candidates,
            JoinType.INNER, ["t_partkey", "t_suppkey"], ["ps_partkey", "ps_suppkey"],
        ),
        Binary(">", Col("ps_availqty"), Col("threshold")),
    )
    wanted_suppliers = N.project(
        "wanted", with_thresholds, [Alias(Col("ps_suppkey"), "ps_suppkey")]
    )
    canadian = N.project(
        "canadian",
        corpus.build_join(
            "n_s", nation,
            N.filter_("canada", N.scan("nation", nation, 1, 25),
                      Binary("==", Col("n_name"), Lit(Q20_NATION))),
            N.scan("supplier", supplier, 1, 10_000), "n_nationkey", "s_nationkey",
        ),
        [Alias(Col("s_suppkey"), "s_suppkey"), Alias(Col("s_name"), "s_name"),
         Alias(Col("s_address"), "s_address")],
    )
    selected = corpus.build_join(
        "s_in", supplier, canadian, wanted_suppliers, "s_suppkey", "ps_suppkey",
        JoinType.LEFT_SEMI,
    )
    projected = N.project(
        "select", selected,
        [Alias(Col("s_name"), "s_name"), Alias(Col("s_address"), "s_address")],
    )
    return N.unload("unload", sorted_output("sort", projected, ["s_name"], [True]))


# -- q21 --------------------------------------------------------------------------

Q21_NATION, Q21_STATUS = "SAUDI ARABIA", "F"


def _other_suppliers(tag, lineitem, late_only):
    """The sibling lineitems of an order, renamed so the residual can name both sides.

    Both sides of these joins are lineitem, and a `TableResult` cannot hold two columns
    called `l_suppkey` — SQL resolves that by alias (l1/l2/l3) and the lowering resolves it
    with a projection.
    """
    scan = N.scan(f"{tag}_lineitem", lineitem, 1, 250_000, 250_000)
    if late_only:
        scan = N.filter_(f"{tag}_late", scan,
                         Binary(">", Col("l_receiptdate"), Col("l_commitdate")))
    return N.project(
        f"{tag}_keys", scan,
        [Alias(Col("l_orderkey"), f"{tag}_orderkey"), Alias(Col("l_suppkey"), f"{tag}_suppkey")],
    )


def plan_q21(t):
    """Suppliers who kept orders waiting: an EXISTS and a NOT EXISTS, both correlated on
    the order key **and** carrying an inequality on the supplier key.

    That inequality is a residual filter on a semi and an anti join, which is the one case
    the capability matrix refuses to stream: #136's finish pass accumulates keys, and a
    keys-only table cannot evaluate a predicate over both sides. So both probe sides are
    collected first — the shape the matrix predicted, on the query it was predicted for."""
    supplier = t("supplier", ["s_suppkey", "s_name", "s_nationkey"])
    nation = t("nation", ["n_nationkey", "n_name"])
    orders = t("orders", ["o_orderkey", "o_orderstatus"])
    lineitem = t("lineitem", ["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"])

    saudi = N.project(
        "saudi",
        corpus.build_join(
            "n_s", nation,
            N.filter_("saudi_nation", N.scan("nation", nation, 1, 25),
                      Binary("==", Col("n_name"), Lit(Q21_NATION))),
            N.scan("supplier", supplier, 1, 10_000), "n_nationkey", "s_nationkey",
        ),
        [Alias(Col("s_suppkey"), "s_suppkey"), Alias(Col("s_name"), "s_name")],
    )
    late = N.filter_(
        "late", N.scan("lineitem", lineitem, 1, 250_000, 250_000),
        Binary(">", Col("l_receiptdate"), Col("l_commitdate")),
    )
    with_suppliers = corpus.build_join("s_l", supplier, saudi, late, "s_suppkey", "l_suppkey")
    base = N.project(
        "base",
        corpus.build_join(
            "o_l", orders,
            N.filter_("finished", N.scan("orders", orders, 1, 100_000, 100_000),
                      Binary("==", Col("o_orderstatus"), Lit(Q21_STATUS))),
            with_suppliers, "o_orderkey", "l_orderkey",
        ),
        [Alias(Col("l_orderkey"), "l_orderkey"), Alias(Col("l_suppkey"), "l_suppkey"),
         Alias(Col("s_name"), "s_name")],
    )
    # A filtered semi/anti takes a single-batch probe, so the planner collects each probe
    # side. Both are projected to two key columns first, which is what keeps that
    # affordable: 6M rows of two integers rather than of the whole lineitem.
    # Built with `hash_join` directly rather than through `corpus.build_join`: the build
    # side here is a subplan, not a table, so there is no frame to take a schema from.
    exists = N.hash_join(
        "exists", N.coalesce_all("base_all", base),
        N.coalesce_all("l2_all", _other_suppliers("l2", lineitem, late_only=False)),
        JoinType.LEFT_SEMI, ["l_orderkey"], ["l2_orderkey"],
        residual=Binary("!=", Col("l_suppkey"), Col("l2_suppkey")),
    )
    not_exists = N.hash_join(
        "not_exists", N.coalesce_all("exists_all", exists),
        N.coalesce_all("l3_all", _other_suppliers("l3", lineitem, late_only=True)),
        JoinType.LEFT_ANTI, ["l_orderkey"], ["l3_orderkey"],
        residual=Binary("!=", Col("l_suppkey"), Col("l3_suppkey")),
    )
    final = aggregate_by(
        "agg", not_exists, ["s_name"], [A.Agg(A.COUNT, None, "numwait")],
        schema_frame=corpus.schema_of(supplier),
    )
    return N.unload(
        "unload", sorted_output("sort", final, ["numwait", "s_name"], [False, True], fetch=100)
    )


# -- q22 --------------------------------------------------------------------------

Q22_CODES = ("13", "31", "23", "29", "30", "18", "17")


def _country_customers(tag, customer):
    """Customers whose phone prefix is one of the seven codes, with the prefix computed —
    `substr` is a `ScalarFunctionExprNode`, evaluated on the column-producing path."""
    prefix = Substring(Col("c_phone"), 1, 2)
    codes = Binary("==", prefix, Lit(Q22_CODES[0]))
    for code in Q22_CODES[1:]:
        codes = Binary("or", codes, Binary("==", prefix, Lit(code)))
    return N.project(
        f"{tag}_codes",
        N.filter_(f"{tag}_in_codes", N.scan(f"{tag}_customer", customer, 1, 50_000, 50_000),
                  codes),
        [Alias(Col("c_custkey"), "c_custkey"), Alias(prefix, "cntrycode"),
         Alias(Col("c_acctbal"), "c_acctbal")],
    )


def plan_q22(t):
    """Global sales opportunity: a phone prefix, a scalar average as a threshold, and a
    NOT EXISTS with no correlation beyond the key — so unlike q21 this anti join has no
    residual and streams its probe, finishing with the build rows nothing matched."""
    customer = t("customer", ["c_custkey", "c_phone", "c_acctbal"])
    orders = t("orders", ["o_custkey"])

    average = N.project(
        "average",
        aggregate_to_one_row(
            "avg",
            N.filter_("positive", _country_customers("avg", customer),
                      Binary(">", Col("c_acctbal"), Lit(0.0))),
            [A.Agg(A.MEAN, "c_acctbal", "avg_acctbal")],
            corpus.schema_of(customer, cntrycode=customer.c_phone.dtype),
        ),
        [Alias(Col("avg_acctbal"), "avg_acctbal")],
    )
    above = N.nested_loop_join(
        "above_average", N.coalesce_all("average_all", average),
        _country_customers("all", customer),
        JoinType.INNER, Binary(">", Col("c_acctbal"), Col("avg_acctbal")),
    )
    selected = N.project(
        "candidates", above,
        [Alias(Col("c_custkey"), "c_custkey"), Alias(Col("cntrycode"), "cntrycode"),
         Alias(Col("c_acctbal"), "c_acctbal")],
    )
    without_orders = N.hash_join(
        "no_orders", N.coalesce_all("candidates_all", selected),
        N.scan("orders", orders, 1, 100_000, 100_000),
        JoinType.LEFT_ANTI, ["c_custkey"], ["o_custkey"],
    )
    final = aggregate_by(
        "agg", without_orders, ["cntrycode"],
        [A.Agg(A.COUNT, None, "numcust"), A.Agg(A.SUM, "c_acctbal", "totacctbal")],
        schema_frame=corpus.schema_of(customer, cntrycode=customer.c_phone.dtype),
    )
    return N.unload("unload", sorted_output("sort", final, ["cntrycode"], [True]))


#: Every TPC-H query the prototype runs, by name. `plans.py` renders these; the tests
#: execute them. Adding a query means adding a builder and a test — the registry is what
#: makes the plan goldens cover the corpus rather than whatever was remembered.
