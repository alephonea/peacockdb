"""TPC-DS: the star queries — one fact table streaming past small dimensions.

The shape the batch-partitioned mode is for, and the majority of the benchmark. A fact
table is the probe side of every join in the query, so it is scanned once, in batches, and
never collected; the dimensions are filtered down and collected into one batch each. The
aggregate at the top shuffles on its group keys, by which point it is holding a summary
rather than a table.

What varies between them, and is what each docstring is about: which table is big enough to
be the fact, which predicates can be pushed to a dimension and which span two tables and
cannot, and what a projection has to materialize before the aggregate can take it.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from . import corpus
from .plan_helpers import (
    aggregate_by, aggregate_to_one_row, all_of, any_of, between, date, is_in, rename,
    select, sorted_output,
)
from .plans_tpcds_common import FACT_ROWS, dim, fact, registry, star
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import (
    Alias, Binary, Case, Coalesce, Col, Concat, IsNotNull, Lit, Substring,
)
from ..operators.join_types import JoinType

QUERIES, ORDER_BY, query = registry()


# -- brand and category reports ----------------------------------------------------
#
# Four queries with one lowering: store_sales past a month of date_dim and a slice of item,
# summed by brand or category. They differ in the group keys and the sort, which is exactly
# the kind of near-duplication a benchmark uses to catch a planner that special-cases.


def _brand_report(t, manager, moy, year, columns):
    """store_sales ⋈ date_dim(month) ⋈ item(manager), projected to `columns`."""
    store_sales = fact(t, "store_sales", ["ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_moy", "d_year"],
                   all_of(Binary("==", Col("d_moy"), Lit(moy)),
                          Binary("==", Col("d_year"), Lit(year))))
    item = dim(t, "item", ["i_item_sk", "i_manager_id"] + columns,
               Binary("==", Col("i_manager_id"), Lit(manager)))
    joined = star(
        store_sales[1],
        ("d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
        ("i_ss", item, "i_item_sk", "ss_item_sk"),
    )
    return store_sales, date_dim, item, joined


@query("q55", order_by=("ext_price", "brand_id"))
def plan_q55(t):
    """Brand revenue for one month. The smallest star there is: two dimensions, one sum,
    and a top-100 whose sort key is the aggregate — so the trim cannot happen before it."""
    store_sales, date_dim, item, joined = _brand_report(
        t, 28, 11, 1999, ["i_brand_id", "i_brand"]
    )
    keys = select("keys", joined, "i_brand", "i_brand_id", "ss_ext_sales_price")
    final = aggregate_by(
        "agg", keys, ["i_brand", "i_brand_id"],
        [A.Agg(A.SUM, "ss_ext_sales_price", "ext_price")],
        schema_frame=corpus.schema_of(item[0], store_sales[0]),
    )
    out = rename("out", final,
                 [("i_brand_id", "brand_id"), ("i_brand", "brand"), ("ext_price", "ext_price")])
    return N.unload(
        "unload", sorted_output("sort", out, ["ext_price", "brand_id"], [False, True], fetch=100)
    )


@query("q52", order_by=("d_year", "ext_price", "brand_id"))
def plan_q52(t):
    """q55 with the year in the group key. It is already constant — `d_year = 2000` — so the
    extra key adds a column and no groups, which is the planner's problem and not the
    executor's: the mode groups on what it is given."""
    store_sales, date_dim, item, joined = _brand_report(
        t, 1, 11, 2000, ["i_brand_id", "i_brand"]
    )
    keys = select("keys", joined, "d_year", "i_brand", "i_brand_id", "ss_ext_sales_price")
    final = aggregate_by(
        "agg", keys, ["d_year", "i_brand", "i_brand_id"],
        [A.Agg(A.SUM, "ss_ext_sales_price", "ext_price")],
        schema_frame=corpus.schema_of(date_dim[0], item[0], store_sales[0]),
    )
    out = rename("out", final, [("d_year", "d_year"), ("i_brand_id", "brand_id"),
                                ("i_brand", "brand"), ("ext_price", "ext_price")])
    return N.unload(
        "unload",
        sorted_output("sort", out, ["d_year", "ext_price", "brand_id"], [True, False, True],
                      fetch=100),
    )


@query("q42", order_by=("sum(ss_ext_sales_price)", "d_year", "i_category_id", "i_category"))
def plan_q42(t):
    """The same month by category. The query never aliases its sum, so the output column is
    literally `sum(ss_ext_sales_price)` — the name the oracle checks against, since a
    lowering that renamed it answered a differently-shaped query."""
    store_sales, date_dim, item, joined = _brand_report(
        t, 1, 11, 2000, ["i_category_id", "i_category"]
    )
    keys = select("keys", joined, "d_year", "i_category_id", "i_category", "ss_ext_sales_price")
    final = aggregate_by(
        "agg", keys, ["d_year", "i_category_id", "i_category"],
        [A.Agg(A.SUM, "ss_ext_sales_price", "sum(ss_ext_sales_price)")],
        schema_frame=corpus.schema_of(date_dim[0], item[0], store_sales[0]),
    )
    return N.unload(
        "unload",
        sorted_output("sort", final,
                      ["sum(ss_ext_sales_price)", "d_year", "i_category_id", "i_category"],
                      [False, True, True, True], fetch=100),
    )


@query("q3", order_by=("d_year", "sum_agg", "brand_id"))
def plan_q3(t):
    """Brand revenue by year for one manufacturer: two dimension joins on one streamed fact
    side, a three-key grouped aggregate, and a top-100 whose ORDER BY mixes directions."""
    store_sales = fact(t, "store_sales", ["ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year", "d_moy"],
                   Binary("==", Col("d_moy"), Lit(11)))
    item = dim(t, "item", ["i_item_sk", "i_brand_id", "i_brand", "i_manufact_id"],
               Binary("==", Col("i_manufact_id"), Lit(128)))
    joined = star(
        store_sales[1],
        ("d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
        ("i_ss", item, "i_item_sk", "ss_item_sk"),
    )
    keys = select("keys", joined, "d_year", "i_brand", "i_brand_id", "ss_ext_sales_price")
    final = aggregate_by(
        "agg", keys, ["d_year", "i_brand", "i_brand_id"],
        [A.Agg(A.SUM, "ss_ext_sales_price", "sum_agg")],
        schema_frame=corpus.schema_of(date_dim[0], item[0], store_sales[0]),
    )
    out = rename("out", final, [("d_year", "d_year"), ("i_brand_id", "brand_id"),
                                ("i_brand", "brand"), ("sum_agg", "sum_agg")])
    return N.unload(
        "unload",
        sorted_output("sort", out, ["d_year", "sum_agg", "brand_id"], [True, False, True],
                      fetch=100),
    )


# -- demographic averages ----------------------------------------------------------


_SEGMENT = all_of(
    Binary("==", Col("cd_gender"), Lit("M")),
    Binary("==", Col("cd_marital_status"), Lit("S")),
    Binary("==", Col("cd_education_status"), Lit("College")),
)
#: The promotion predicate q7 and q26 share, verbatim.
_PROMO = Binary("or", Binary("==", Col("p_channel_email"), Lit("N")),
                Binary("==", Col("p_channel_event"), Lit("N")))


def _demographics(t, tag="cd"):
    """customer_demographics, filtered to one segment.

    Declared a fact rather than a dimension: 1.9M rows scanned as one batch is 730 MB of
    python strings and the resident enforcer trips, which is the enforcer being right on
    real data. So it streams through its filter and only the survivors are collected.
    """
    return fact(t, "customer_demographics",
                ["cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status"],
                tag=tag), _SEGMENT


@query("q7", order_by=("i_item_id",))
def plan_q7(t):
    """Promotional averages by item: five tables, four averages. The averages are the
    interesting part — a partial emits [sum, count] per group, the merge adds them, and only
    the finalize divides, so an average is never averaged."""
    store_sales = fact(
        t, "store_sales",
        ["ss_sold_date_sk", "ss_item_sk", "ss_cdemo_sk", "ss_promo_sk",
         "ss_quantity", "ss_list_price", "ss_coupon_amt", "ss_sales_price"],
    )
    (demographics, segment) = _demographics(t)
    demographics = (demographics[0],
                    N.filter_("segment", demographics[1], segment))
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"],
                   Binary("==", Col("d_year"), Lit(2000)))
    item = dim(t, "item", ["i_item_sk", "i_item_id"])
    promotion = dim(t, "promotion", ["p_promo_sk", "p_channel_email", "p_channel_event"],
                    _PROMO)
    joined = star(
        store_sales[1],
        ("d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
        ("cd_ss", demographics, "cd_demo_sk", "ss_cdemo_sk"),
        ("p_ss", promotion, "p_promo_sk", "ss_promo_sk"),
        ("i_ss", item, "i_item_sk", "ss_item_sk"),
    )
    final = aggregate_by(
        "agg", joined, ["i_item_id"],
        [A.Agg(A.MEAN, "ss_quantity", "agg1"), A.Agg(A.MEAN, "ss_list_price", "agg2"),
         A.Agg(A.MEAN, "ss_coupon_amt", "agg3"), A.Agg(A.MEAN, "ss_sales_price", "agg4")],
        schema_frame=corpus.schema_of(item[0], store_sales[0]),
    )
    out = select("out", final, "i_item_id", "agg1", "agg2", "agg3", "agg4")
    return N.unload("unload", sorted_output("sort", out, ["i_item_id"], [True], fetch=100))


@query("q26", order_by=("i_item_id",))
def plan_q26(t):
    """q7 against the catalog channel. Same five tables, same four averages, a different
    fact table and half the rows — the benchmark's own A/B on whether a plan generalizes."""
    catalog_sales = fact(
        t, "catalog_sales",
        ["cs_sold_date_sk", "cs_item_sk", "cs_bill_cdemo_sk", "cs_promo_sk",
         "cs_quantity", "cs_list_price", "cs_coupon_amt", "cs_sales_price"],
    )
    (demographics, segment) = _demographics(t)
    demographics = (demographics[0], N.filter_("segment", demographics[1], segment))
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"],
                   Binary("==", Col("d_year"), Lit(2000)))
    item = dim(t, "item", ["i_item_sk", "i_item_id"])
    promotion = dim(t, "promotion", ["p_promo_sk", "p_channel_email", "p_channel_event"],
                    _PROMO)
    joined = star(
        catalog_sales[1],
        ("d_cs", date_dim, "d_date_sk", "cs_sold_date_sk"),
        ("cd_cs", demographics, "cd_demo_sk", "cs_bill_cdemo_sk"),
        ("p_cs", promotion, "p_promo_sk", "cs_promo_sk"),
        ("i_cs", item, "i_item_sk", "cs_item_sk"),
    )
    final = aggregate_by(
        "agg", joined, ["i_item_id"],
        [A.Agg(A.MEAN, "cs_quantity", "agg1"), A.Agg(A.MEAN, "cs_list_price", "agg2"),
         A.Agg(A.MEAN, "cs_coupon_amt", "agg3"), A.Agg(A.MEAN, "cs_sales_price", "agg4")],
        schema_frame=corpus.schema_of(item[0], catalog_sales[0]),
    )
    out = select("out", final, "i_item_id", "agg1", "agg2", "agg3", "agg4")
    return N.unload("unload", sorted_output("sort", out, ["i_item_id"], [True], fetch=100))


# -- stars with a predicate that spans two tables -----------------------------------


@query("q15", order_by=("ca_zip",))
def plan_q15(t):
    """Catalog revenue by zip. The interesting part is the `OR`: one arm reads the address,
    one reads the sale, so no arm can be pushed to either table's scan and the whole
    disjunction has to be evaluated above the join — the case a pushdown pass must not
    take."""
    catalog_sales = fact(t, "catalog_sales",
                         ["cs_sold_date_sk", "cs_bill_customer_sk", "cs_sales_price"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_qoy", "d_year"],
                   all_of(Binary("==", Col("d_qoy"), Lit(2)),
                          Binary("==", Col("d_year"), Lit(2001))))
    customer = dim(t, "customer", ["c_customer_sk", "c_current_addr_sk"])
    address = dim(t, "customer_address", ["ca_address_sk", "ca_zip", "ca_state"])
    joined = star(
        catalog_sales[1],
        ("d_cs", date_dim, "d_date_sk", "cs_sold_date_sk"),
        ("c_cs", customer, "c_customer_sk", "cs_bill_customer_sk"),
        ("ca_c", address, "ca_address_sk", "c_current_addr_sk"),
    )
    kept = N.filter_(
        "zip_or_state_or_price", joined,
        any_of(is_in(Substring(Col("ca_zip"), 1, 5), Q15_ZIPS),
               is_in(Col("ca_state"), ("CA", "WA", "GA")),
               Binary(">", Col("cs_sales_price"), Lit(500))),
    )
    final = aggregate_by(
        "agg", select("keys", kept, "ca_zip", "cs_sales_price"), ["ca_zip"],
        [A.Agg(A.SUM, "cs_sales_price", "sum(cs_sales_price)")],
        schema_frame=corpus.schema_of(address[0], catalog_sales[0]),
    )
    return N.unload(
        "unload",
        sorted_output("sort", final, ["ca_zip"], [True], fetch=100, nulls_first=True),
    )


Q15_ZIPS = ("85669", "86197", "88274", "83405", "86475", "85392", "85460", "80348", "81792")


@query("q19", order_by=("ext_price", "brand", "brand_id", "i_manufact_id", "i_manufact"))
def plan_q19(t):
    """Sales to customers who do not live near the store. Six tables, and the condition that
    makes it a query rather than a star — the zip codes must differ — reads a column from
    the address and a column from the store, so it is a filter above both joins and never a
    join key."""
    store_sales = fact(t, "store_sales",
                       ["ss_sold_date_sk", "ss_item_sk", "ss_customer_sk", "ss_store_sk",
                        "ss_ext_sales_price"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_moy", "d_year"],
                   all_of(Binary("==", Col("d_moy"), Lit(11)),
                          Binary("==", Col("d_year"), Lit(1998))))
    item = dim(t, "item",
               ["i_item_sk", "i_brand_id", "i_brand", "i_manufact_id", "i_manufact",
                "i_manager_id"],
               Binary("==", Col("i_manager_id"), Lit(8)))
    customer = dim(t, "customer", ["c_customer_sk", "c_current_addr_sk"])
    address = dim(t, "customer_address", ["ca_address_sk", "ca_zip"])
    store = dim(t, "store", ["s_store_sk", "s_zip"])
    joined = star(
        store_sales[1],
        ("d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
        ("i_ss", item, "i_item_sk", "ss_item_sk"),
        ("c_ss", customer, "c_customer_sk", "ss_customer_sk"),
        ("ca_c", address, "ca_address_sk", "c_current_addr_sk"),
        ("s_ss", store, "s_store_sk", "ss_store_sk"),
    )
    elsewhere = N.filter_(
        "different_zip", joined,
        Binary("!=", Substring(Col("ca_zip"), 1, 5), Substring(Col("s_zip"), 1, 5)),
    )
    final = aggregate_by(
        "agg",
        select("keys", elsewhere, "i_brand", "i_brand_id", "i_manufact_id", "i_manufact",
               "ss_ext_sales_price"),
        ["i_brand", "i_brand_id", "i_manufact_id", "i_manufact"],
        [A.Agg(A.SUM, "ss_ext_sales_price", "ext_price")],
        schema_frame=corpus.schema_of(item[0], store_sales[0]),
    )
    out = rename("out", final,
                 [("i_brand_id", "brand_id"), ("i_brand", "brand"),
                  ("i_manufact_id", "i_manufact_id"), ("i_manufact", "i_manufact"),
                  ("ext_price", "ext_price")])
    return N.unload(
        "unload",
        sorted_output("sort", out,
                      ["ext_price", "brand", "brand_id", "i_manufact_id", "i_manufact"],
                      [False, True, True, True, True], fetch=100),
    )


# -- stars whose filter is above the aggregate --------------------------------------


@query("q21", order_by=("w_warehouse_name", "i_item_id"))
def plan_q21(t):
    """Inventory either side of a date, per warehouse and item.

    Two conditional sums over the same rows — the `CASE` is inside the aggregate's input,
    so one pass produces both — and then a ratio test on the *result*, which is a filter
    above the aggregate rather than a HAVING the aggregate could apply. The ratio is NULL
    where nothing was held before, and a NULL comparison is false, so those rows fall out.

    The fact here is `inventory`: 11.7M rows, the largest table in the benchmark.
    """
    inventory = fact(t, "inventory",
                     ["inv_date_sk", "inv_item_sk", "inv_warehouse_sk", "inv_quantity_on_hand"])
    warehouse = dim(t, "warehouse", ["w_warehouse_sk", "w_warehouse_name"])
    item = dim(t, "item", ["i_item_sk", "i_item_id", "i_current_price"],
               between(Col("i_current_price"), Lit(0.99), Lit(1.49)))
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_date"],
                   between(Col("d_date"), date("2000-02-10"), date("2000-04-10")))
    joined = star(
        inventory[1],
        ("d_inv", date_dim, "d_date_sk", "inv_date_sk"),
        ("i_inv", item, "i_item_sk", "inv_item_sk"),
        ("w_inv", warehouse, "w_warehouse_sk", "inv_warehouse_sk"),
    )
    split = date("2000-03-11")
    sides = N.project(
        "sides", joined,
        [Alias(Col("w_warehouse_name"), "w_warehouse_name"), Alias(Col("i_item_id"), "i_item_id"),
         Alias(Case(whens=((Binary("<", Col("d_date"), split), Col("inv_quantity_on_hand")),),
                    otherwise=Lit(0)), "before"),
         Alias(Case(whens=((Binary(">=", Col("d_date"), split), Col("inv_quantity_on_hand")),),
                    otherwise=Lit(0)), "after")],
    )
    totals = aggregate_by(
        "agg", sides, ["w_warehouse_name", "i_item_id"],
        [A.Agg(A.SUM, "before", "inv_before"), A.Agg(A.SUM, "after", "inv_after")],
        schema_frame=corpus.schema_of(warehouse[0], item[0], before="int64", after="int64"),
    )
    ratio = N.project(
        "ratio", totals,
        [Alias(Col("w_warehouse_name"), "w_warehouse_name"), Alias(Col("i_item_id"), "i_item_id"),
         Alias(Col("inv_before"), "inv_before"), Alias(Col("inv_after"), "inv_after"),
         Alias(Case(whens=((Binary(">", Col("inv_before"), Lit(0)),
                            Binary("/", Col("inv_after"), Col("inv_before"))),),
                    otherwise=Lit(float("nan"))), "ratio")],
    )
    steady = N.filter_("two_thirds_to_three_halves", ratio,
                       between(Col("ratio"), Lit(2.0 / 3.0), Lit(3.0 / 2.0)))
    out = select("out", steady, "w_warehouse_name", "i_item_id", "inv_before", "inv_after")
    return N.unload(
        "unload",
        sorted_output("sort", out, ["w_warehouse_name", "i_item_id"], [True, True],
                      fetch=100, nulls_first=True),
    )


# -- three channels joined on the transaction, not the date -------------------------

#: The four columns q25 and q29 both report by.
_Q29_KEYS = ["i_item_id", "i_item_desc", "s_store_id", "s_store_name"]

#: q25's window for the return and for the catalog sale, which are the same one.
_Q25_AUTUMN = (all_of(between(Col("d_moy"), Lit(4), Lit(10)),
                      Binary("==", Col("d_year"), Lit(2001))), ("d_moy", "d_year"))


@query("q25", order_by=tuple(_Q29_KEYS))
def plan_q25(t):
    """A sale, its return, and the catalog sale that followed — reporting money.

    The first of three queries over one join (see `_sale_return_catalog`, below q93, where
    the shape is described): q25 sums profit and loss, q29 sums quantities, q17 takes their
    standard deviations.
    """
    joined, schema = _sale_return_catalog(
        t, "q25",
        (all_of(Binary("==", Col("d_moy"), Lit(4)), Binary("==", Col("d_year"), Lit(2001))),
         ("d_moy", "d_year")),
        _Q25_AUTUMN, _Q25_AUTUMN,
        {"store": ("s_store_id", "s_store_name"), "item": ("i_item_id", "i_item_desc")},
        {"ss": ("ss_net_profit",), "sr": ("sr_net_loss",), "cs": ("cs_net_profit",)},
    )
    final = aggregate_by(
        "agg",
        select("keys", joined, *_Q29_KEYS, "ss_net_profit", "sr_net_loss", "cs_net_profit"),
        _Q29_KEYS,
        [A.Agg(A.SUM, "ss_net_profit", "store_sales_profit"),
         A.Agg(A.SUM, "sr_net_loss", "store_returns_loss"),
         A.Agg(A.SUM, "cs_net_profit", "catalog_sales_profit")],
        schema_frame=schema,
    )
    return N.unload(
        "unload",
        sorted_output("sort", final, _Q29_KEYS, [True] * len(_Q29_KEYS), fetch=100),
    )


@query("q93", order_by=("sumsales", "ss_customer_sk"))
def plan_q93(t):
    """Sales net of returns, for one return reason.

    The query says `store_sales LEFT OUTER JOIN store_returns`, which preserves store_sales
    — 2.9M rows, and a preserved side is the side the join holds. Lowered as a **Right**
    join with store_returns as the build, it preserves the same side while that side stays
    the *probe*, so it streams: Right is probe-local, Left is build-preserving and would
    have collected the fact table. The join then finishes; the `reason` join below it drops
    every unmatched row anyway, which is the query getting an inner join by other means.
    """
    store_sales = fact(t, "store_sales",
                       ["ss_item_sk", "ss_ticket_number", "ss_customer_sk", "ss_quantity",
                        "ss_sales_price"])
    returns = dim(t, "store_returns",
                  ["sr_item_sk", "sr_ticket_number", "sr_reason_sk", "sr_return_quantity"],
                  rows=FACT_ROWS)
    reason = dim(t, "reason", ["r_reason_sk", "r_reason_desc"],
                 Binary("==", Col("r_reason_desc"), Lit("reason 28")))
    with_returns = N.hash_join(
        "sr_ss",
        N.coalesce_all("sr_ss_build", returns[1], schema=dict(returns[0].dtypes)),
        store_sales[1], JoinType.RIGHT,
        ["sr_item_sk", "sr_ticket_number"], ["ss_item_sk", "ss_ticket_number"],
    )
    joined = star(with_returns, ("r_sr", reason, "r_reason_sk", "sr_reason_sk"))
    kept = Binary("*", Binary("-", Col("ss_quantity"), Col("sr_return_quantity")),
                  Col("ss_sales_price"))
    sales = N.project(
        "act_sales", joined,
        [Alias(Col("ss_customer_sk"), "ss_customer_sk"),
         Alias(Case(whens=((IsNotNull(Col("sr_return_quantity")), kept),),
                    otherwise=Binary("*", Col("ss_quantity"), Col("ss_sales_price"))),
               "act_sales")],
    )
    final = aggregate_by(
        "agg", sales, ["ss_customer_sk"], [A.Agg(A.SUM, "act_sales", "sumsales")],
        schema_frame=corpus.schema_of(store_sales[0], act_sales="float64"),
    )
    return N.unload(
        "unload",
        sorted_output("sort", final, ["sumsales", "ss_customer_sk"], [True, True],
                      fetch=100, nulls_first=True),
    )


# -- a join chain with no aggregate at all ------------------------------------------


@query("q84", order_by=("customer_id",))
def plan_q84(t):
    """Customers of one city in one income band. Six tables and no aggregate: the output is
    rows of the join, so the only thing above it is a projection that builds the display
    name — `concat` over `coalesce`, both of which the C++ evaluates on the column-producing
    path (`expr.cpp` ~L738, ~L754) and neither of which `cudf::ast` has.

    Two stages, because there are two things large enough to stream. Customer streams past
    the small dimensions to find who lives in Edgewood in that band; store_returns then
    streams past *that* result, which is what the query's row count is a count of.
    """
    customer = fact(t, "customer",
                    ["c_customer_sk", "c_customer_id", "c_current_addr_sk", "c_current_cdemo_sk",
                     "c_current_hdemo_sk", "c_first_name", "c_last_name"])
    address = dim(t, "customer_address", ["ca_address_sk", "ca_city"],
                  Binary("==", Col("ca_city"), Lit("Edgewood")))
    household = dim(t, "household_demographics", ["hd_demo_sk", "hd_income_band_sk"])
    band = dim(t, "income_band", ["ib_income_band_sk", "ib_lower_bound", "ib_upper_bound"],
               all_of(Binary(">=", Col("ib_lower_bound"), Lit(38128)),
                      Binary("<=", Col("ib_upper_bound"), Lit(38128 + 50000))))
    demographics = fact(t, "customer_demographics", ["cd_demo_sk"])
    eligible = select(
        "eligible",
        star(
            customer[1],
            ("ca_c", address, "ca_address_sk", "c_current_addr_sk"),
            ("hd_c", household, "hd_demo_sk", "c_current_hdemo_sk"),
            ("ib_hd", band, "ib_income_band_sk", "hd_income_band_sk"),
            ("cd_c", demographics, "cd_demo_sk", "c_current_cdemo_sk"),
        ),
        "cd_demo_sk", "c_customer_id", "c_first_name", "c_last_name",
    )
    returns = fact(t, "store_returns", ["sr_cdemo_sk"])
    joined = N.hash_join(
        "sr_cd",
        N.coalesce_all("sr_cd_build", eligible,
                       schema=dict(corpus.schema_of(customer[0], demographics[0]).dtypes)),
        returns[1], JoinType.INNER, ["cd_demo_sk"], ["sr_cdemo_sk"],
    )
    out = N.project(
        "out", joined,
        [Alias(Col("c_customer_id"), "customer_id"),
         Alias(Concat((Concat((Coalesce((Col("c_last_name"), Lit(""))), Lit(", "))),
                       Coalesce((Col("c_first_name"), Lit(""))))),
               "customername")],
    )
    return N.unload(
        "unload",
        sorted_output("sort", out, ["customer_id"], [True], fetch=100, nulls_first=True),
    )


@query("q96", order_by=("count_star()",))
def plan_q96(t):
    """A count over four tables: keyless, so the lanes collapse instead of shuffling and the
    whole answer is one row. The query's ORDER BY and LIMIT are no-ops over one row; the
    plan stops where the answer does."""
    store_sales = fact(t, "store_sales", ["ss_sold_time_sk", "ss_hdemo_sk", "ss_store_sk"])
    time_dim = dim(t, "time_dim", ["t_time_sk", "t_hour", "t_minute"],
                   all_of(Binary("==", Col("t_hour"), Lit(20)),
                          Binary(">=", Col("t_minute"), Lit(30))))
    household = dim(t, "household_demographics", ["hd_demo_sk", "hd_dep_count"],
                    Binary("==", Col("hd_dep_count"), Lit(7)))
    store = dim(t, "store", ["s_store_sk", "s_store_name"],
                Binary("==", Col("s_store_name"), Lit("ese")))
    joined = star(
        store_sales[1],
        ("t_ss", time_dim, "t_time_sk", "ss_sold_time_sk"),
        ("hd_ss", household, "hd_demo_sk", "ss_hdemo_sk"),
        ("s_ss", store, "s_store_sk", "ss_store_sk"),
    )
    return N.unload(
        "unload",
        aggregate_to_one_row("agg", joined, [A.Agg(A.COUNT, None, "count_star()")],
                             corpus.schema_of(store_sales[0])),
    )


# -- distinct rows, expressed as semi joins -----------------------------------------


def _stocked_items(t, tag, price, manufacts, start, end, other, other_key):
    """Items in a price band, held in inventory over a date range, and sold on `other`.

    Both tables that qualify the item contribute existence and nothing else — the query
    ends in a `GROUP BY` over item columns with no aggregate, which is a `DISTINCT` — so
    both are lowered as **semi joins** with the item side as the build. An inner join would
    have been faithful to the text and produced tens of millions of rows for a `DISTINCT`
    to collapse; a semi join emits each build row at most once, which is what the text
    means. Item is the build in both, so both are Left forms and the probe streams.
    """
    item = dim(t, "item",
               ["i_item_sk", "i_item_id", "i_item_desc", "i_current_price", "i_manufact_id"],
               all_of(between(Col("i_current_price"), Lit(price), Lit(price + 30)),
                      is_in(Col("i_manufact_id"), manufacts)),
               tag=f"{tag}_item")
    inventory = fact(t, "inventory",
                     ["inv_date_sk", "inv_item_sk", "inv_quantity_on_hand"],
                     tag=f"{tag}_inventory")
    on_hand = N.filter_(f"{tag}_on_hand", inventory[1],
                        between(Col("inv_quantity_on_hand"), Lit(100), Lit(500)))
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_date"],
                   between(Col("d_date"), date(start), date(end)), tag=f"{tag}_date_dim")
    in_range = star(on_hand, (f"{tag}_d_inv", date_dim, "d_date_sk", "inv_date_sk"))
    stocked = N.hash_join(
        f"{tag}_i_inv",
        N.coalesce_all(f"{tag}_i_inv_build", item[1], schema=dict(item[0].dtypes)),
        in_range, JoinType.LEFT_SEMI, ["i_item_sk"], ["inv_item_sk"],
    )
    sold_on = fact(t, other, [other_key], tag=f"{tag}_{other}")
    sold = N.hash_join(
        f"{tag}_i_{other}",
        N.coalesce_all(f"{tag}_i_{other}_build", stocked, schema=dict(item[0].dtypes)),
        sold_on[1], JoinType.LEFT_SEMI, ["i_item_sk"], [other_key],
    )
    keys = ["i_item_id", "i_item_desc", "i_current_price"]
    distinct = aggregate_by(f"{tag}_distinct", select(f"{tag}_keys", sold, *keys), keys, [],
                            schema_frame=corpus.schema_of(item[0]))
    return N.unload(
        "unload",
        sorted_output("sort", distinct, ["i_item_id"], [True], fetch=100),
    )


@query("q37", order_by=("i_item_id",))
def plan_q37(t):
    """Items in stock and sold from the catalog, in a price band, over two months."""
    return _stocked_items(t, "q37", 68, (677, 940, 694, 808),
                          "2000-02-01", "2000-04-01", "catalog_sales", "cs_item_sk")


@query("q82", order_by=("i_item_id",))
def plan_q82(t):
    """q37 against the store channel, over a different two months. Same lowering, which is
    the point of the pair."""
    return _stocked_items(t, "q82", 62, (129, 270, 821, 423),
                          "2000-05-25", "2000-07-24", "store_sales", "ss_item_sk")


@query("q90", order_by=("am_pm_ratio",))
def plan_q90(t):
    """Morning web orders over evening ones.

    Two counts over the same four tables with one predicate changed, and then a ratio of
    them. Neither count depends on the other, so they are two independent subtrees that
    meet at a **cross join** of one row with one row — which is what a scalar-by-scalar
    expression is, once the scalars are tables. The division is a projection above it, with
    the zero-denominator guard the query writes as a `CASE`.
    """
    def hourly(tag, first_hour):
        web_sales = fact(t, "web_sales",
                         ["ws_sold_time_sk", "ws_ship_hdemo_sk", "ws_web_page_sk"],
                         tag=f"{tag}_web_sales")
        time_dim = dim(t, "time_dim", ["t_time_sk", "t_hour"],
                       between(Col("t_hour"), Lit(first_hour), Lit(first_hour + 1)),
                       tag=f"{tag}_time_dim")
        household = dim(t, "household_demographics", ["hd_demo_sk", "hd_dep_count"],
                        Binary("==", Col("hd_dep_count"), Lit(6)),
                        tag=f"{tag}_household_demographics")
        page = dim(t, "web_page", ["wp_web_page_sk", "wp_char_count"],
                   between(Col("wp_char_count"), Lit(5000), Lit(5200)),
                   tag=f"{tag}_web_page")
        joined = star(
            web_sales[1],
            (f"{tag}_t_ws", time_dim, "t_time_sk", "ws_sold_time_sk"),
            (f"{tag}_hd_ws", household, "hd_demo_sk", "ws_ship_hdemo_sk"),
            (f"{tag}_wp_ws", page, "wp_web_page_sk", "ws_web_page_sk"),
        )
        return aggregate_to_one_row(f"{tag}_agg", joined, [A.Agg(A.COUNT, None, tag)],
                                    corpus.schema_of(web_sales[0]))

    both = N.cross_join("am_by_pm", N.coalesce_all("amc_all", hourly("amc", 8)),
                        hourly("pmc", 19))
    ratio = N.project(
        "ratio", both,
        [Alias(Case(whens=((Binary("==", Col("pmc"), Lit(0)), Lit(float("nan"))),),
                    otherwise=Binary("/", Col("amc"), Col("pmc"))), "am_pm_ratio")],
    )
    return N.unload("unload", sorted_output("sort", ratio, ["am_pm_ratio"], [True], fetch=100))


def _sale_return_catalog(t, tag, d1_predicate, d2_predicate, d3_predicate, dimensions,
                         measures):
    """store_sales ⋈ its return ⋈ the catalog sale that followed — q25, q29 and q17's join.

    Three fact tables joined on the transaction rather than on a dimension: a return matches
    a sale on (customer, item, ticket) and the catalog sale matches the return on (customer,
    item). Both are multi-column equi-joins, so `hash_join` is called directly — `star` only
    ever passes one key. Each of the two smaller facts is narrowed by its own copy of
    date_dim before being collected, which is what keeps them build sides.
    """
    store_sales = fact(t, "store_sales",
                       ["ss_sold_date_sk", "ss_item_sk", "ss_customer_sk", "ss_store_sk",
                        "ss_ticket_number"] + list(measures["ss"]), tag=f"{tag}_store_sales")
    d1 = dim(t, "date_dim", ["d_date_sk"] + list(d1_predicate[1]), d1_predicate[0],
             tag=f"{tag}_d1")
    d2 = dim(t, "date_dim", ["d_date_sk"] + list(d2_predicate[1]), d2_predicate[0],
             tag=f"{tag}_d2")
    # d2 and d3 are separate arguments because q29 narrows them differently: the return has
    # to land in the four months after the sale, the catalog sale only in one of three years.
    d3 = dim(t, "date_dim", ["d_date_sk"] + list(d3_predicate[1]), d3_predicate[0],
             tag=f"{tag}_d3")
    store = dim(t, "store", ["s_store_sk"] + list(dimensions["store"]), tag=f"{tag}_store")
    item = dim(t, "item", ["i_item_sk"] + list(dimensions["item"]), tag=f"{tag}_item")
    returns = fact(t, "store_returns",
                   ["sr_returned_date_sk", "sr_item_sk", "sr_customer_sk", "sr_ticket_number"]
                   + list(measures["sr"]), tag=f"{tag}_store_returns")
    returned = select(
        f"{tag}_returns_keys",
        star(returns[1], (f"{tag}_d2_sr", d2, "d_date_sk", "sr_returned_date_sk")),
        "sr_item_sk", "sr_customer_sk", "sr_ticket_number", *measures["sr"],
    )
    catalog = fact(t, "catalog_sales",
                   ["cs_sold_date_sk", "cs_item_sk", "cs_bill_customer_sk"]
                   + list(measures["cs"]), tag=f"{tag}_catalog_sales")
    catalogued = select(
        f"{tag}_catalog_keys",
        star(catalog[1], (f"{tag}_d3_cs", d3, "d_date_sk", "cs_sold_date_sk")),
        "cs_item_sk", "cs_bill_customer_sk", *measures["cs"],
    )
    sold = star(
        store_sales[1],
        (f"{tag}_d1_ss", d1, "d_date_sk", "ss_sold_date_sk"),
        (f"{tag}_i_ss", item, "i_item_sk", "ss_item_sk"),
        (f"{tag}_s_ss", store, "s_store_sk", "ss_store_sk"),
    )
    with_returns = N.hash_join(
        f"{tag}_sr_ss",
        N.coalesce_all(f"{tag}_sr_ss_build", returned,
                       schema=dict(corpus.schema_of(returns[0]).dtypes)),
        sold, JoinType.INNER,
        ["sr_customer_sk", "sr_item_sk", "sr_ticket_number"],
        ["ss_customer_sk", "ss_item_sk", "ss_ticket_number"],
    )
    joined = N.hash_join(
        f"{tag}_cs_sr",
        N.coalesce_all(f"{tag}_cs_sr_build", catalogued,
                       schema=dict(corpus.schema_of(catalog[0]).dtypes)),
        with_returns, JoinType.INNER,
        ["cs_bill_customer_sk", "cs_item_sk"], ["sr_customer_sk", "sr_item_sk"],
    )
    return joined, corpus.schema_of(item[0], store[0], store_sales[0], returns[0], catalog[0])


@query("q29", order_by=tuple(_Q29_KEYS))
def plan_q29(t):
    """q25's join, reporting quantities rather than money, over three different date
    windows. The lowering is the same tree with different constants, which is the pair's
    purpose in the benchmark."""
    joined, schema = _sale_return_catalog(
        t, "q29",
        (all_of(Binary("==", Col("d_moy"), Lit(9)), Binary("==", Col("d_year"), Lit(1999))),
         ("d_moy", "d_year")),
        (all_of(between(Col("d_moy"), Lit(9), Lit(12)),
                Binary("==", Col("d_year"), Lit(1999))), ("d_moy", "d_year")),
        (is_in(Col("d_year"), (1999, 2000, 2001)), ("d_year",)),
        {"store": ("s_store_id", "s_store_name"), "item": ("i_item_id", "i_item_desc")},
        {"ss": ("ss_quantity",), "sr": ("sr_return_quantity",), "cs": ("cs_quantity",)},
    )
    final = aggregate_by(
        "agg", select("keys", joined, *_Q29_KEYS, "ss_quantity", "sr_return_quantity",
                      "cs_quantity"),
        _Q29_KEYS,
        [A.Agg(A.SUM, "ss_quantity", "store_sales_quantity"),
         A.Agg(A.SUM, "sr_return_quantity", "store_returns_quantity"),
         A.Agg(A.SUM, "cs_quantity", "catalog_sales_quantity")],
        schema_frame=schema,
    )
    return N.unload(
        "unload",
        sorted_output("sort", final, _Q29_KEYS, [True] * len(_Q29_KEYS), fetch=100),
    )


_Q17_KEYS = ["i_item_id", "i_item_desc", "s_state"]
#: `(label, the column each of the three channels measures)`.
_Q17_CHANNELS = (("store_sales", "ss_quantity"), ("store_returns", "sr_return_quantity"),
                 ("catalog_sales", "cs_quantity"))


@query("q17", order_by=tuple(_Q17_KEYS))
def plan_q17(t):
    """q29's join, reporting a coefficient of variation per channel.

    The only query in the corpus that needs `stddev_samp`, and it needs three of them. The
    aggregate carries Welford state — count, mean, M2 — through the partial and the merge,
    and only the finalize takes the square root, which is why a standard deviation survives
    being computed in pieces across lanes. The coefficient of variation is a *ratio of two
    aggregates*, so it is a projection above the aggregate rather than an aggregate.

    Its answer at sf1 is the empty set: the three date windows have no transaction in
    common. That is declared in `test_tpcds.EMPTY_BY_DESIGN` rather than tolerated.
    """
    quarters = ("2001Q1", "2001Q2", "2001Q3")
    joined, schema = _sale_return_catalog(
        t, "q17",
        (Binary("==", Col("d_quarter_name"), Lit("2001Q1")), ("d_quarter_name",)),
        (is_in(Col("d_quarter_name"), quarters), ("d_quarter_name",)),
        (is_in(Col("d_quarter_name"), quarters), ("d_quarter_name",)),
        {"store": ("s_state",), "item": ("i_item_id", "i_item_desc")},
        {"ss": ("ss_quantity",), "sr": ("sr_return_quantity",), "cs": ("cs_quantity",)},
    )
    aggs = []
    for label, column in _Q17_CHANNELS:
        aggs += [A.Agg(A.COUNT, column, f"{label}_quantitycount"),
                 A.Agg(A.MEAN, column, f"{label}_quantityave"),
                 A.Agg(A.STDDEV, column, f"{label}_quantitystdev")]
    final = aggregate_by(
        "agg",
        select("keys", joined, *_Q17_KEYS, *[column for _, column in _Q17_CHANNELS]),
        _Q17_KEYS, aggs, schema_frame=schema,
    )
    reported = [Alias(Col(key), key) for key in _Q17_KEYS]
    for label, _ in _Q17_CHANNELS:
        reported += [
            Alias(Col(f"{label}_quantitycount"), f"{label}_quantitycount"),
            Alias(Col(f"{label}_quantityave"), f"{label}_quantityave"),
            Alias(Col(f"{label}_quantitystdev"), f"{label}_quantitystdev"),
            Alias(Binary("/", Col(f"{label}_quantitystdev"), Col(f"{label}_quantityave")),
                  f"{label}_quantitycov"),
        ]
    out = N.project("out", final, reported)
    return N.unload(
        "unload",
        sorted_output("sort", out, _Q17_KEYS, [True] * len(_Q17_KEYS), fetch=100,
                      nulls_first=True),
    )
