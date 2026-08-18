"""TPC-H query plans, lowered by hand into the batch-partitioned nodes.

Separated from the tests that run them for two reasons. A plan is a function of the
**schemas**, not the rows — every builder here takes a table provider and never looks at
the data — so `plans.py` can render the plan goldens from parquet footers in milliseconds
where executing the queries takes minutes. And the tests are then the oracles and the
assertions, which is a different thing to read.

Each builder is the lowering the translation layer will have to produce for that query:
where the joins go, which side is the build, where the shuffle is, what the projections
have to materialize before an aggregate can take them as ordinals.
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

# -- corpus queries ---------------------------------------------------------------
#
# The plans above are shapes; these are queries. Each is a TPC-H query text lowered by hand
# into the mode's nodes — the lowering the translation layer will have to produce — and
# checked against a pandas oracle written from the SQL rather than from the plan.
#
# **Whole tables, the spec's own parameters.** A sampled corpus was tried and abandoned:
# both benchmarks are written clustered by date, so a row prefix is one quarter of 1992, a
# row-group sample is a set of date windows, and two tables sampled independently join to
# nothing. Every one of those bit. These read lineitem's six million rows and answer the
# query that was asked, which is why they run on manual dispatch (exec-model-corpus.yml)
# rather than on every push, and why `PCK_SHARD` and `PCK_LAYOUT` exist.
#
# **Each query is a plan builder and a test.** The builder takes a table provider and
# returns a root node, so `plans.py` can render the plan from parquet schemas alone without
# executing anything — a plan is a function of the schemas, and re-running six million rows
# to print a tree would be absurd. The test supplies real tables, computes the oracle, and
# runs the query at every layout `corpus.LAYOUTS` names.
#
# **Where the shuffles are.** Aggregates shuffle on their group keys, which is a handful of
# rows by the time the shuffle sees them. Joins run at one lane with a streamed probe — the
# shape v1 leans on, and the one the join capability work is about.


# -- q6 ---------------------------------------------------------------------------

Q6_WHERE = all_of(
    Binary(">=", Col("l_shipdate"), date("1994-01-01")),
    Binary("<", Col("l_shipdate"), date("1995-01-01")),
    Binary(">=", Col("l_discount"), Lit(0.05)),
    Binary("<=", Col("l_discount"), Lit(0.07)),
    Binary("<", Col("l_quantity"), Lit(24.0)),
)


def plan_q6(t):
    """Forecasting revenue change: the mode's motivating pipeline — load, filter hard,
    aggregate to a single row. Keyless, so the lanes collapse rather than shuffle."""
    lineitem = t("lineitem", ["l_shipdate", "l_discount", "l_quantity", "l_extendedprice"])
    revenue = [Alias(Binary("*", Col("l_extendedprice"), Col("l_discount")), "revenue")]
    scan = N.scan("lineitem", lineitem, 4, 250_000, 500_000)
    projected = N.project("revenue", N.filter_("where", scan, Q6_WHERE), revenue)
    return N.unload(
        "unload",
        aggregate_to_one_row("agg", projected, [A.Agg(A.SUM, "revenue", "revenue")],
                             corpus.schema_of(revenue="float64")),
    )


Q1_KEYS = ["l_returnflag", "l_linestatus"]
Q1_AGGS = [
    A.Agg(A.SUM, "l_quantity", "sum_qty"),
    A.Agg(A.SUM, "l_extendedprice", "sum_base_price"),
    A.Agg(A.SUM, "disc_price", "sum_disc_price"),
    A.Agg(A.SUM, "charge", "sum_charge"),
    A.Agg(A.MEAN, "l_quantity", "avg_qty"),
    A.Agg(A.MEAN, "l_extendedprice", "avg_price"),
    A.Agg(A.MEAN, "l_discount", "avg_disc"),
    A.Agg(A.COUNT, None, "count_order"),
]


def q1_projection():
    """The aggregate's arguments, materialized. `CudfAggregate.aggr_funcs` take ordinals,
    so an aggregate over an expression needs the expression to be a column first."""
    disc_price = Binary("*", Col("l_extendedprice"), Binary("-", Lit(1.0), Col("l_discount")))
    return [
        Alias(Col("l_returnflag"), "l_returnflag"),
        Alias(Col("l_linestatus"), "l_linestatus"),
        Alias(Col("l_quantity"), "l_quantity"),
        Alias(Col("l_extendedprice"), "l_extendedprice"),
        Alias(Col("l_discount"), "l_discount"),
        Alias(disc_price, "disc_price"),
        Alias(Binary("*", disc_price, Binary("+", Lit(1.0), Col("l_tax"))), "charge"),
    ]


def plan_q1(t):
    """Pricing summary report: eight aggregates over two group keys, four of them over
    expressions, and the query's ORDER BY as a per-lane sort plus a k-way merge."""
    columns = ["l_shipdate", "l_returnflag", "l_linestatus", "l_quantity",
               "l_extendedprice", "l_discount", "l_tax"]
    lineitem = t("lineitem", columns)
    lanes = 4
    schemas = agg_schemas(
        lineitem.assign(disc_price=0.0, charge=0.0), Q1_KEYS, Q1_AGGS
    )
    scan = N.scan("lineitem", lineitem, lanes, 250_000, 500_000)
    filtered = N.filter_("where", scan, Binary("<=", Col("l_shipdate"), date("1998-09-02")))
    projected = N.project("exprs", filtered, q1_projection())
    final = aggregate_by("agg", projected, Q1_KEYS, Q1_AGGS, lanes, schemas)
    return N.unload("unload", sorted_output("sort", final, Q1_KEYS, [True, True]))


Q3_CUTOFF = pd.Timestamp("1995-03-15")
Q3_KEYS = ["l_orderkey", "o_orderdate", "o_shippriority"]


def plan_q3(t):
    """Shipping priority: two joins, a grouped aggregate and a top-N. The second join's
    build side is the first join's output, which is why a `GpuCoalesceAllBatches` sits
    between them."""
    customer = t("customer", ["c_custkey", "c_mktsegment"])
    orders = t("orders", ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"])
    lineitem = t("lineitem", ["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"])

    building = N.filter_(
        "building", N.scan("customer", customer, 1, 50_000, 50_000),
        Binary("==", Col("c_mktsegment"), Lit("BUILDING")),
    )
    before = N.filter_(
        "before", N.scan("orders", orders, 1, 100_000, 100_000),
        Binary("<", Col("o_orderdate"), date("1995-03-15")),
    )
    with_orders = corpus.build_join("c_o", customer, building, before, "c_custkey", "o_custkey")
    shipped = N.filter_(
        "shipped", N.scan("lineitem", lineitem, 1, 250_000, 250_000),
        Binary(">", Col("l_shipdate"), date("1995-03-15")),
    )
    joined = N.hash_join(
        "o_l", N.coalesce_all("orders_all", with_orders), shipped,
        JoinType.INNER, ["o_orderkey"], ["l_orderkey"],
    )
    revenue = Binary("*", Col("l_extendedprice"), Binary("-", Lit(1.0), Col("l_discount")))
    projected = N.project(
        "revenue", joined,
        [Alias(Col("l_orderkey"), "l_orderkey"), Alias(Col("o_orderdate"), "o_orderdate"),
         Alias(Col("o_shippriority"), "o_shippriority"), Alias(revenue, "revenue")],
    )
    final = aggregate_by("agg", projected, Q3_KEYS, [A.Agg(A.SUM, "revenue", "revenue")],
                         schema_frame=corpus.schema_of(orders, lineitem, revenue="float64"))
    # The query's select order is not the aggregate's, so a projection restores it.
    selected = N.project(
        "select", final,
        [Alias(Col("l_orderkey"), "l_orderkey"), Alias(Col("revenue"), "revenue"),
         Alias(Col("o_orderdate"), "o_orderdate"), Alias(Col("o_shippriority"), "o_shippriority")],
    )
    top = sorted_output("sort", selected, ["revenue", "o_orderdate"], [False, True], fetch=10)
    return N.unload("unload", top)


Q12_START, Q12_END = pd.Timestamp("1994-01-01"), pd.Timestamp("1995-01-01")
Q12_AGGS = [A.Agg(A.SUM, "high_line", "high_line_count"),
            A.Agg(A.SUM, "low_line", "low_line_count")]


def plan_q12(t):
    """Shipping modes and order priority: an IN-list as the OR-chain `gpu_rule.rs` lowers
    it to, and two CASE expressions the projection materializes for the aggregate."""
    orders = t("orders", ["o_orderkey", "o_orderpriority"])
    lineitem = t(
        "lineitem",
        ["l_orderkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"],
    )
    urgent = Binary("==", Col("o_orderpriority"), Lit("1-URGENT"))
    high = Binary("==", Col("o_orderpriority"), Lit("2-HIGH"))
    predicate = all_of(
        Binary("or", Binary("==", Col("l_shipmode"), Lit("MAIL")),
               Binary("==", Col("l_shipmode"), Lit("SHIP"))),
        Binary("<", Col("l_commitdate"), Col("l_receiptdate")),
        Binary("<", Col("l_shipdate"), Col("l_commitdate")),
        Binary(">=", Col("l_receiptdate"), date("1994-01-01")),
        Binary("<", Col("l_receiptdate"), date("1995-01-01")),
    )
    exprs = [
        Alias(Col("l_shipmode"), "l_shipmode"),
        Alias(Case(whens=((Binary("or", urgent, high), Lit(1)),), otherwise=Lit(0)), "high_line"),
        Alias(
            Case(
                whens=((Binary("and", Binary("!=", Col("o_orderpriority"), Lit("1-URGENT")),
                               Binary("!=", Col("o_orderpriority"), Lit("2-HIGH"))), Lit(1)),),
                otherwise=Lit(0),
            ),
            "low_line",
        ),
    ]
    build = N.scan("orders", orders, 1, 100_000, 100_000)
    probe = N.filter_("where", N.scan("lineitem", lineitem, 1, 250_000, 250_000), predicate)
    joined = corpus.build_join("o_l", orders, build, probe, "o_orderkey", "l_orderkey")
    projected = N.project("case", joined, exprs)
    final = aggregate_by("agg", projected, ["l_shipmode"], Q12_AGGS,
                         schema_frame=corpus.schema_of(lineitem, high_line="int64",
                                                      low_line="int64"))
    return N.unload("unload", sorted_output("sort", final, ["l_shipmode"], [True]))


Q4_START, Q4_END = pd.Timestamp("1993-07-01"), pd.Timestamp("1993-10-01")


def plan_q4(t):
    """Order priority checking: `EXISTS (…)` is a semi join, and a semi join is the shape
    whose whole output comes from the finish pass — the probe calls only record which
    build rows matched. orders is the build side because orders is what the query emits."""
    orders = t("orders", ["o_orderkey", "o_orderdate", "o_orderpriority"])
    lineitem = t("lineitem", ["l_orderkey", "l_commitdate", "l_receiptdate"])
    in_quarter = N.filter_(
        "quarter", N.scan("orders", orders, 1, 100_000, 100_000),
        Binary("and", Binary(">=", Col("o_orderdate"), date("1993-07-01")),
               Binary("<", Col("o_orderdate"), date("1993-10-01"))),
    )
    late = N.filter_(
        "late", N.scan("lineitem", lineitem, 1, 250_000, 250_000),
        Binary("<", Col("l_commitdate"), Col("l_receiptdate")),
    )
    existing = corpus.build_join(
        "exists", orders, in_quarter, late, "o_orderkey", "l_orderkey", JoinType.LEFT_SEMI
    )
    final = aggregate_by("agg", existing, ["o_orderpriority"],
                         [A.Agg(A.COUNT, None, "order_count")],
                         schema_frame=corpus.schema_of(orders))
    return N.unload("unload", sorted_output("sort", final, ["o_orderpriority"], [True]))


Q5_START, Q5_END = pd.Timestamp("1994-01-01"), pd.Timestamp("1995-01-01")


def plan_q5(t):
    """Local supplier volume: six tables, five of them build sides, lineitem streaming past
    the chain. Two things the lowering does that the SQL does not say. Each build side is
    **projected down** to the columns the joins above it need — without that the last join
    has `n_name` on both sides and the output has two columns of that name, which SQL
    resolves by alias and a `TableResult` cannot. And `c_nationkey = s_nationkey` is not a
    join key but a filter over the joined row: both sides are already in the region, and
    this is what makes the customer and the supplier the same nation."""
    region = t("region", ["r_regionkey", "r_name"])
    nation = t("nation", ["n_nationkey", "n_name", "n_regionkey"])
    supplier = t("supplier", ["s_suppkey", "s_nationkey"])
    customer = t("customer", ["c_custkey", "c_nationkey"])
    orders = t("orders", ["o_orderkey", "o_custkey", "o_orderdate"])
    lineitem = t("lineitem", ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"])

    nations_frame = nation.merge(
        region[region.r_name == "ASIA"], left_on="n_regionkey", right_on="r_regionkey"
    )[["n_nationkey", "n_name"]]
    suppliers_frame = nations_frame.merge(
        supplier, left_on="n_nationkey", right_on="s_nationkey"
    )[["s_suppkey", "n_name", "s_nationkey"]]
    customers_frame = nations_frame.merge(
        customer, left_on="n_nationkey", right_on="c_nationkey"
    )[["c_custkey", "c_nationkey"]]
    orders_frame = customers_frame.merge(
        orders, left_on="c_custkey", right_on="o_custkey"
    )[["o_orderkey", "c_nationkey"]]

    def asian_nations(tag):
        """region ⋈ nation, projected. Built twice because a plan is a tree: the supplier
        branch and the customer branch each need their own copy, which is #101's inlined
        CTE in miniature."""
        joined = corpus.build_join(
            f"{tag}_rn", region,
            N.filter_(f"{tag}_asia", N.scan(f"{tag}_region", region, 1, 10),
                      Binary("==", Col("r_name"), Lit("ASIA"))),
            N.scan(f"{tag}_nation", nation, 1, 25), "r_regionkey", "n_regionkey",
        )
        return N.project(
            f"{tag}_nkeys", joined,
            [Alias(Col("n_nationkey"), "n_nationkey"), Alias(Col("n_name"), "n_name")],
        )

    asian_suppliers = N.project(
        "s_keys",
        corpus.build_join("n_s", nations_frame, asian_nations("s"),
                          N.scan("supplier", supplier, 1, 10_000), "n_nationkey", "s_nationkey"),
        [Alias(Col("s_suppkey"), "s_suppkey"), Alias(Col("n_name"), "n_name"),
         Alias(Col("s_nationkey"), "s_nationkey")],
    )
    asian_customers = N.project(
        "c_keys",
        corpus.build_join("n_c", nations_frame, asian_nations("c"),
                          N.scan("customer", customer, 1, 50_000, 50_000),
                          "n_nationkey", "c_nationkey"),
        [Alias(Col("c_custkey"), "c_custkey"), Alias(Col("c_nationkey"), "c_nationkey")],
    )
    asian_orders = N.project(
        "o_keys",
        corpus.build_join(
            "c_o", customers_frame, asian_customers,
            N.filter_("in_year", N.scan("orders", orders, 1, 100_000, 100_000),
                      Binary("and", Binary(">=", Col("o_orderdate"), date("1994-01-01")),
                             Binary("<", Col("o_orderdate"), date("1995-01-01")))),
            "c_custkey", "o_custkey",
        ),
        [Alias(Col("o_orderkey"), "o_orderkey"), Alias(Col("c_nationkey"), "c_nationkey")],
    )
    with_orders = corpus.build_join(
        "o_l", orders_frame, asian_orders,
        N.scan("lineitem", lineitem, 1, 250_000, 250_000), "o_orderkey", "l_orderkey",
    )
    joined = corpus.build_join(
        "s_l", suppliers_frame, asian_suppliers, with_orders, "s_suppkey", "l_suppkey"
    )
    same_nation = N.filter_(
        "same_nation", joined, Binary("==", Col("c_nationkey"), Col("s_nationkey"))
    )
    revenue = Binary("*", Col("l_extendedprice"), Binary("-", Lit(1.0), Col("l_discount")))
    projected = N.project(
        "revenue", same_nation, [Alias(Col("n_name"), "n_name"), Alias(revenue, "revenue")]
    )
    final = aggregate_by("agg", projected, ["n_name"], [A.Agg(A.SUM, "revenue", "revenue")],
                         schema_frame=corpus.schema_of(nation, revenue="float64"))
    return N.unload("unload", sorted_output("sort", final, ["revenue"], [False]))


Q10_START, Q10_END = pd.Timestamp("1993-10-01"), pd.Timestamp("1994-01-01")
Q10_REPORT = ["c_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address", "c_comment"]


def plan_q10(t):
    """Returned item reporting: a seven-column group key, most of it strings carried along
    for the report, and a top-20 by a measure that is not one of them."""
    customer = t(
        "customer",
        ["c_custkey", "c_name", "c_acctbal", "c_nationkey", "c_address", "c_phone", "c_comment"],
    )
    nation = t("nation", ["n_nationkey", "n_name"])
    orders = t("orders", ["o_orderkey", "o_custkey", "o_orderdate"])
    lineitem = t(
        "lineitem",
        ["l_orderkey", "l_extendedprice", "l_discount", "l_returnflag", "l_shipdate"],
    )
    customers_frame = customer.merge(
        nation, left_on="c_nationkey", right_on="n_nationkey"
    )[Q10_REPORT]
    orders_frame = customers_frame.merge(
        orders, left_on="c_custkey", right_on="o_custkey"
    )[Q10_REPORT + ["o_orderkey"]]

    with_nation = N.project(
        "c_keys",
        corpus.build_join("n_c", nation, N.scan("nation", nation, 1, 25),
                          N.scan("customer", customer, 1, 50_000, 50_000),
                          "n_nationkey", "c_nationkey"),
        [Alias(Col(column), column) for column in Q10_REPORT],
    )
    with_orders = N.project(
        "o_keys",
        corpus.build_join(
            "c_o", customers_frame, with_nation,
            N.filter_("quarter", N.scan("orders", orders, 1, 100_000, 100_000),
                      Binary("and", Binary(">=", Col("o_orderdate"), date("1993-10-01")),
                             Binary("<", Col("o_orderdate"), date("1994-01-01")))),
            "c_custkey", "o_custkey",
        ),
        [Alias(Col(column), column) for column in Q10_REPORT + ["o_orderkey"]],
    )
    returned = N.filter_(
        "returned", N.scan("lineitem", lineitem, 1, 250_000, 250_000),
        Binary("==", Col("l_returnflag"), Lit("R")),
    )
    joined = corpus.build_join(
        "o_l", orders_frame, with_orders, returned, "o_orderkey", "l_orderkey"
    )
    revenue = Binary("*", Col("l_extendedprice"), Binary("-", Lit(1.0), Col("l_discount")))
    projected = N.project(
        "revenue", joined,
        [Alias(Col(column), column) for column in Q10_REPORT] + [Alias(revenue, "revenue")],
    )
    final = aggregate_by("agg", projected, Q10_REPORT, [A.Agg(A.SUM, "revenue", "revenue")],
                         schema_frame=corpus.schema_of(customer, nation, revenue="float64"))
    return N.unload("unload", sorted_output("sort", final, ["revenue"], [False], fetch=20))


Q18_THRESHOLD = 300.0
Q18_REPORT = ["c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"]


def plan_q18(t):
    """Large volume customer: `o_orderkey IN (select … group by … having sum > 300)` is an
    aggregate feeding a semi join. lineitem is read twice, once for the inner aggregate and
    once for the outer sum, because a plan is a tree and DataFusion inlines the reference
    too (#101).

    The semi join is a **RightSemi**: the query emits the orders and the orders are the
    streamed side, so the small key set is the build and the join emits its probe rows.
    Collecting 1.5M orders into a build batch instead is 270 MB, and the enforcer says so."""
    customer = t("customer", ["c_custkey", "c_name"])
    orders = t("orders", ["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"])
    lineitem = t("lineitem", ["l_orderkey", "l_quantity"])

    inner_aggs = [A.Agg(A.SUM, "l_quantity", "order_quantity")]
    inner = aggregate_by(
        "inner", N.scan("lineitem_inner", lineitem, 1, 250_000, 250_000),
        ["l_orderkey"], inner_aggs, schema_frame=corpus.schema_of(lineitem),
    )
    big = N.project(
        "big_keys",
        N.filter_("over_threshold", inner,
                  Binary(">", Col("order_quantity"), Lit(Q18_THRESHOLD))),
        [Alias(Col("l_orderkey"), "l_orderkey")],
    )
    big_frame = lineitem[["l_orderkey"]]
    with_customers = N.project(
        "co_keys",
        corpus.build_join("c_o", customer, N.scan("customer", customer, 1, 50_000, 50_000),
                          N.scan("orders", orders, 1, 100_000, 100_000),
                          "c_custkey", "o_custkey"),
        [Alias(Col(column), column) for column in Q18_REPORT],
    )
    selected = corpus.build_join(
        "big_o", big_frame, big, with_customers, "l_orderkey", "o_orderkey", JoinType.RIGHT_SEMI
    )
    joined = corpus.build_join(
        "o_l", customer, selected,
        N.scan("lineitem_outer", lineitem, 1, 250_000, 250_000), "o_orderkey", "l_orderkey",
    )
    final = aggregate_by(
        "agg", joined, Q18_REPORT, [A.Agg(A.SUM, "l_quantity", "total_quantity")],
        schema_frame=corpus.schema_of(customer, orders, lineitem),
    )
    top = sorted_output("sort", final, ["o_totalprice", "o_orderdate"], [False, True], fetch=100)
    return N.unload("unload", top)






# -- q7 ---------------------------------------------------------------------------

Q7_START, Q7_END = pd.Timestamp("1995-01-01"), pd.Timestamp("1996-12-31")
Q7_KEYS = ["supp_nation", "cust_nation", "l_year"]
Q7_PAIR = ("FRANCE", "GERMANY")


def _nation_pair(tag, nation, names=Q7_PAIR):
    """The two nations of interest, as a build side. Scanned per branch because a plan is
    a tree and the query names `nation` twice (n1 and n2)."""
    return N.filter_(
        f"{tag}_pair", N.scan(f"{tag}_nation", nation, 1, 25),
        Binary("or", Binary("==", Col("n_name"), Lit(names[0])),
               Binary("==", Col("n_name"), Lit(names[1]))),
    )


def plan_q7(t):
    """Volume shipping: six tables, `nation` twice, a year extracted from a date, and a
    disjunction over the two nation columns that only exists after the joins."""
    supplier = t("supplier", ["s_suppkey", "s_nationkey"])
    customer = t("customer", ["c_custkey", "c_nationkey"])
    nation = t("nation", ["n_nationkey", "n_name"])
    orders = t("orders", ["o_orderkey", "o_custkey"])
    lineitem = t("lineitem", ["l_orderkey", "l_suppkey", "l_shipdate",
                              "l_extendedprice", "l_discount"])

    suppliers = N.project(
        "supp_keys",
        corpus.build_join("n1_s", nation, _nation_pair("n1", nation),
                          N.scan("supplier", supplier, 1, 10_000),
                          "n_nationkey", "s_nationkey"),
        [Alias(Col("s_suppkey"), "s_suppkey"), Alias(Col("n_name"), "supp_nation")],
    )
    customers = N.project(
        "cust_keys",
        corpus.build_join("n2_c", nation, _nation_pair("n2", nation),
                          N.scan("customer", customer, 1, 50_000, 50_000),
                          "n_nationkey", "c_nationkey"),
        [Alias(Col("c_custkey"), "c_custkey"), Alias(Col("n_name"), "cust_nation")],
    )
    customer_orders = N.project(
        "order_keys",
        corpus.build_join("c_o", customer, customers,
                          N.scan("orders", orders, 1, 100_000, 100_000),
                          "c_custkey", "o_custkey"),
        [Alias(Col("o_orderkey"), "o_orderkey"), Alias(Col("cust_nation"), "cust_nation")],
    )
    shipped = N.filter_(
        "shipped", N.scan("lineitem", lineitem, 1, 250_000, 250_000),
        Binary("and", Binary(">=", Col("l_shipdate"), date("1995-01-01")),
               Binary("<=", Col("l_shipdate"), date("1996-12-31"))),
    )
    with_suppliers = corpus.build_join(
        "s_l", supplier, suppliers, shipped, "s_suppkey", "l_suppkey"
    )
    joined = corpus.build_join(
        "o_l", orders, customer_orders, with_suppliers, "o_orderkey", "l_orderkey"
    )
    pair = N.filter_(
        "pair", joined,
        Binary(
            "or",
            Binary("and", Binary("==", Col("supp_nation"), Lit("FRANCE")),
                   Binary("==", Col("cust_nation"), Lit("GERMANY"))),
            Binary("and", Binary("==", Col("supp_nation"), Lit("GERMANY")),
                   Binary("==", Col("cust_nation"), Lit("FRANCE"))),
        ),
    )
    volume = Binary("*", Col("l_extendedprice"), Binary("-", Lit(1.0), Col("l_discount")))
    projected = N.project(
        "shipping", pair,
        [Alias(Col("supp_nation"), "supp_nation"), Alias(Col("cust_nation"), "cust_nation"),
         Alias(DatePart("year", Col("l_shipdate")), "l_year"), Alias(volume, "volume")],
    )
    final = aggregate_by(
        "agg", projected, Q7_KEYS, [A.Agg(A.SUM, "volume", "revenue")],
        schema_frame=corpus.schema_of(
            supp_nation=nation.n_name.dtype, cust_nation=nation.n_name.dtype,
            l_year="int64", volume="float64"),
    )
    return N.unload("unload", sorted_output("sort", final, Q7_KEYS, [True, True, True]))


# -- q11 --------------------------------------------------------------------------

Q11_NATION = "GERMANY"
Q11_FRACTION = 0.000002


def _german_partsupp(tag, partsupp, supplier, nation):
    """partsupp ⋈ supplier ⋈ nation='GERMANY', with the value column computed.

    Built twice — once for the grouped total, once for the scalar threshold — because a
    plan is a tree and DataFusion inlines the repeated subquery the same way (#101)."""
    german = N.filter_(
        f"{tag}_german", N.scan(f"{tag}_nation", nation, 1, 25),
        Binary("==", Col("n_name"), Lit(Q11_NATION)),
    )
    suppliers = N.project(
        f"{tag}_supp",
        corpus.build_join(f"{tag}_n_s", nation, german,
                          N.scan(f"{tag}_supplier", supplier, 1, 10_000),
                          "n_nationkey", "s_nationkey"),
        [Alias(Col("s_suppkey"), "s_suppkey")],
    )
    joined = corpus.build_join(
        f"{tag}_s_ps", supplier, suppliers,
        N.scan(f"{tag}_partsupp", partsupp, 1, 100_000, 100_000), "s_suppkey", "ps_suppkey",
    )
    return N.project(
        f"{tag}_value", joined,
        [Alias(Col("ps_partkey"), "ps_partkey"),
         Alias(Binary("*", Col("ps_supplycost"), Col("ps_availqty")), "value")],
    )


def plan_q11(t):
    """Important stock identification: a grouped total against a scalar threshold. The
    threshold is one row, so the comparison is a nested-loop join carrying it as a
    predicate — which is what DataFusion plans for a scalar subquery, and why q11 is one of
    the corpus's two `GpuNestedLoopJoin` carriers."""
    partsupp = t("partsupp", ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"])
    supplier = t("supplier", ["s_suppkey", "s_nationkey"])
    nation = t("nation", ["n_nationkey", "n_name"])

    per_part = aggregate_by(
        "agg", _german_partsupp("outer", partsupp, supplier, nation),
        ["ps_partkey"], [A.Agg(A.SUM, "value", "value")],
        schema_frame=corpus.schema_of(partsupp, value="float64"),
    )
    total = aggregate_to_one_row(
        "total", _german_partsupp("inner", partsupp, supplier, nation),
        [A.Agg(A.SUM, "value", "total_value")], corpus.schema_of(value="float64"),
    )
    threshold = N.project(
        "threshold", total,
        [Alias(Binary("*", Col("total_value"), Lit(Q11_FRACTION)), "threshold")],
    )
    # The one-row side is the build; the grouped side streams past it.
    over = N.nested_loop_join(
        "over_threshold", N.coalesce_all("threshold_all", threshold), per_part,
        JoinType.INNER, Binary(">", Col("value"), Col("threshold")),
    )
    selected = N.project(
        "select", over,
        [Alias(Col("ps_partkey"), "ps_partkey"), Alias(Col("value"), "value")],
    )
    return N.unload("unload", sorted_output("sort", selected, ["value"], [False]))


# -- q14 --------------------------------------------------------------------------

Q14_START, Q14_END = pd.Timestamp("1995-09-01"), pd.Timestamp("1995-10-01")


def plan_q14(t):
    """Promotion effect: a ratio of two sums over the same rows, so both are aggregated
    together and the division happens once, in a projection over the single output row."""
    lineitem = t("lineitem", ["l_partkey", "l_shipdate", "l_extendedprice", "l_discount"])
    part = t("part", ["p_partkey", "p_type"])

    in_month = N.filter_(
        "month", N.scan("lineitem", lineitem, 1, 250_000, 250_000),
        Binary("and", Binary(">=", Col("l_shipdate"), date("1995-09-01")),
               Binary("<", Col("l_shipdate"), date("1995-10-01"))),
    )
    joined = corpus.build_join(
        "p_l", part, N.scan("part", part, 1, 100_000, 100_000), in_month,
        "p_partkey", "l_partkey",
    )
    revenue = Binary("*", Col("l_extendedprice"), Binary("-", Lit(1.0), Col("l_discount")))
    projected = N.project(
        "revenue", joined,
        [Alias(Case(whens=((Like(Col("p_type"), "PROMO%"), revenue),), otherwise=Lit(0.0)),
               "promo_revenue"),
         Alias(revenue, "revenue")],
    )
    totals = aggregate_to_one_row(
        "agg", projected,
        [A.Agg(A.SUM, "promo_revenue", "promo"), A.Agg(A.SUM, "revenue", "total")],
        corpus.schema_of(promo_revenue="float64", revenue="float64"),
    )
    ratio = N.project(
        "ratio", totals,
        [Alias(Binary("/", Binary("*", Lit(100.0), Col("promo")), Col("total")),
               "promo_revenue")],
    )
    return N.unload("unload", ratio)


from .plans_tpch_more import (  # noqa: E402  (the registry names both halves)
    plan_q13, plan_q15, plan_q16, plan_q17, plan_q19, plan_q2, plan_q20, plan_q21,
    plan_q22, plan_q8, plan_q9,
)

#: Every TPC-H query the prototype runs, by name.
QUERIES = {
    "q1": plan_q1,
    "q2": plan_q2,
    "q3": plan_q3,
    "q4": plan_q4,
    "q5": plan_q5,
    "q6": plan_q6,
    "q7": plan_q7,
    "q8": plan_q8,
    "q9": plan_q9,
    "q10": plan_q10,
    "q11": plan_q11,
    "q12": plan_q12,
    "q13": plan_q13,
    "q14": plan_q14,
    "q15": plan_q15,
    "q16": plan_q16,
    "q17": plan_q17,
    "q18": plan_q18,
    "q19": plan_q19,
    "q20": plan_q20,
    "q21": plan_q21,
    "q22": plan_q22,
}
