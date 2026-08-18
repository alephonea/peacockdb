"""TPC-DS: the bucketed reports — one pass, many conditional sums.

A shape worth its own module because of what it does to the plan. `sum(CASE WHEN … THEN x
ELSE 0 END)` repeated five or seven times is still **one** aggregate over one scan: the
conditions are evaluated in the projection below it, so the group keys are hashed once and
each group carries seven accumulators. A lowering that made it seven passes, or seven
subqueries unioned, would be answering the same question at seven times the cost — which is
what these queries are in the benchmark to catch.

The cross-joined counts (q88, q90's sibling) are the degenerate case of the same idea, and
the one place where it genuinely *is* several passes: the query writes eight independent
scalar subqueries, so eight subtrees meet at a chain of cross joins.
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
    Alias, Binary, Case, Coalesce, Col, Like, Lit, Lower, Substring,
)
from ..operators.join_types import JoinType

QUERIES, ORDER_BY, query = registry()

#: The five ageing buckets q50, q62 and q99 all report, as `(output name, low, high)` over a
#: day count. `None` is an open end.
_AGEING = (("30 days", None, 30), ("31-60 days", 30, 60), ("61-90 days", 60, 90),
           ("91-120 days", 90, 120), (">120 days", 120, None))


def _bucket_columns(days):
    """One `CASE WHEN days IN bucket THEN 1 ELSE 0 END` per ageing bucket.

    The buckets are disjoint and cover the line, so exactly one of the five is 1 for every
    row — which is why they can share a scan, and why summing them is a histogram.
    """
    columns = []
    for name, low, high in _AGEING:
        tests = []
        if low is not None:
            tests.append(Binary(">", days, Lit(low)))
        if high is not None:
            tests.append(Binary("<=", days, Lit(high)))
        columns.append(Alias(Case(whens=((all_of(*tests), Lit(1)),), otherwise=Lit(0)), name))
    return columns


def _bucket_aggs():
    return [A.Agg(A.SUM, name, name) for name, _, _ in _AGEING]


_BUCKET_NAMES = tuple(name for name, _, _ in _AGEING)


# -- ageing histograms --------------------------------------------------------------


@query("q50", order_by=("s_store_name", "s_company_id", "s_street_number", "s_street_name",
                        "s_street_type", "s_suite_number", "s_city", "s_county", "s_state",
                        "s_zip"))
def plan_q50(t):
    """How long store customers took to return what they bought, by store address.

    The bucket is a difference of two surrogate keys — `sr_returned_date_sk -
    ss_sold_date_sk` — which works because date_dim's keys are consecutive days, and is the
    benchmark relying on a property of the generated data rather than on date arithmetic.
    The two date_dim copies do different jobs: d2 narrows the returns to one month, d1 only
    has to exist, since the sale's date is already in the fact row.

    store_sales is the probe; the returns of one month are small enough to be the build.
    """
    store_sales = fact(t, "store_sales",
                       ["ss_sold_date_sk", "ss_item_sk", "ss_customer_sk", "ss_ticket_number",
                        "ss_store_sk"])
    returns = fact(t, "store_returns",
                   ["sr_returned_date_sk", "sr_item_sk", "sr_customer_sk", "sr_ticket_number"])
    d1 = dim(t, "date_dim", ["d_date_sk"], tag="d1")
    d2 = dim(t, "date_dim", ["d_date_sk", "d_year", "d_moy"],
             all_of(Binary("==", Col("d_year"), Lit(2001)),
                    Binary("==", Col("d_moy"), Lit(8))), tag="d2")
    address = ["s_store_name", "s_company_id", "s_street_number", "s_street_name",
               "s_street_type", "s_suite_number", "s_city", "s_county", "s_state", "s_zip"]
    store = dim(t, "store", ["s_store_sk"] + address)
    in_august = select(
        "returns_keys",
        star(returns[1], ("d2_sr", d2, "d_date_sk", "sr_returned_date_sk")),
        "sr_returned_date_sk", "sr_item_sk", "sr_customer_sk", "sr_ticket_number",
    )
    sold = star(
        store_sales[1],
        ("d1_ss", d1, "d_date_sk", "ss_sold_date_sk"),
        ("s_ss", store, "s_store_sk", "ss_store_sk"),
    )
    returned = N.hash_join(
        "sr_ss",
        N.coalesce_all("sr_ss_build", in_august, schema=dict(corpus.schema_of(returns[0]).dtypes)),
        sold, JoinType.INNER,
        ["sr_customer_sk", "sr_item_sk", "sr_ticket_number"],
        ["ss_customer_sk", "ss_item_sk", "ss_ticket_number"],
    )
    days = Binary("-", Col("sr_returned_date_sk"), Col("ss_sold_date_sk"))
    bucketed = N.project(
        "buckets", returned,
        [Alias(Col(column), column) for column in address] + _bucket_columns(days),
    )
    final = aggregate_by(
        "agg", bucketed, address, _bucket_aggs(),
        schema_frame=corpus.schema_of(store[0], **{name: "int64" for name in _BUCKET_NAMES}),
    )
    out = select("out", final, *address, *_BUCKET_NAMES)
    return N.unload(
        "unload", sorted_output("sort", out, address, [True] * len(address), fetch=100)
    )


def _shipping_lag(t, tag, channel, sold_key, ship_key, warehouse_key, mode_key,
                  site_key, site, site_columns, site_name):
    """How long a channel took to ship, bucketed, by warehouse, ship mode and site.

    q62 and q99 are the same query over web_sales and catalog_sales, and the only thing that
    differs below the projection is which columns the site dimension is called by. The
    warehouse's `substr` is in the group key, so it has to be materialized in a projection
    before the aggregate can group on it — an aggregate takes column ordinals, not
    expressions, which is the constraint the whole lowering is arranged around.
    """
    sales = fact(t, channel, [sold_key, ship_key, warehouse_key, mode_key, site_key],
                 tag=f"{tag}_{channel}")
    warehouse = dim(t, "warehouse", ["w_warehouse_sk", "w_warehouse_name"],
                    tag=f"{tag}_warehouse")
    ship_mode = dim(t, "ship_mode", ["sm_ship_mode_sk", "sm_type"], tag=f"{tag}_ship_mode")
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_month_seq"],
                   between(Col("d_month_seq"), Lit(1200), Lit(1200 + 11)),
                   tag=f"{tag}_date_dim")
    joined = star(
        sales[1],
        (f"{tag}_d_ship", date_dim, "d_date_sk", ship_key),
        (f"{tag}_w", warehouse, "w_warehouse_sk", warehouse_key),
        (f"{tag}_sm", ship_mode, "sm_ship_mode_sk", mode_key),
        (f"{tag}_site", site, site_columns[0], site_key),
    )
    days = Binary("-", Col(ship_key), Col(sold_key))
    bucketed = N.project(
        f"{tag}_buckets", joined,
        [Alias(Substring(Col("w_warehouse_name"), 1, 20), "w_substr"),
         Alias(Col("sm_type"), "sm_type"), site_name] + _bucket_columns(days),
    )
    keys = ["w_substr", "sm_type", site_name.name()]
    return aggregate_by(
        f"{tag}_agg", bucketed, keys, _bucket_aggs(),
        schema_frame=corpus.schema_of(
            warehouse[0], ship_mode[0],
            **{"w_substr": "object", site_name.name(): "object",
               **{name: "int64" for name in _BUCKET_NAMES}},
        ),
    ), keys


@query("q62", order_by=("w_substr", "sm_type", "web_name"))
def plan_q62(t):
    """Web shipping lag over a year, by warehouse, ship mode and site."""
    site = dim(t, "web_site", ["web_site_sk", "web_name"], tag="q62_web_site")
    final, keys = _shipping_lag(
        t, "q62", "web_sales", "ws_sold_date_sk", "ws_ship_date_sk", "ws_warehouse_sk",
        "ws_ship_mode_sk", "ws_web_site_sk", site, ["web_site_sk"],
        Alias(Col("web_name"), "web_name"),
    )
    out = select("out", final, *keys, *_BUCKET_NAMES)
    return N.unload(
        "unload",
        sorted_output("sort", out, keys, [True] * len(keys), fetch=100, nulls_first=True),
    )


@query("q99", order_by=("w_substr", "sm_type", "cc_name_lower"))
def plan_q99(t):
    """The catalog's shipping lag, by call centre. The group key is the call centre's name
    lowercased, so `lower` is materialized in the same projection as the buckets."""
    site = dim(t, "call_center", ["cc_call_center_sk", "cc_name"], tag="q99_call_center")
    final, keys = _shipping_lag(
        t, "q99", "catalog_sales", "cs_sold_date_sk", "cs_ship_date_sk", "cs_warehouse_sk",
        "cs_ship_mode_sk", "cs_call_center_sk", site, ["cc_call_center_sk"],
        Alias(Lower(Col("cc_name")), "cc_name_lower"),
    )
    out = select("out", final, *keys, *_BUCKET_NAMES)
    return N.unload(
        "unload",
        sorted_output("sort", out, keys, [True] * len(keys), fetch=100, nulls_first=True),
    )


# -- conditional sums over a day of the week ----------------------------------------


_DAYS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
_DAY_SALES = tuple(f"{day[:3].lower()}_sales" for day in _DAYS)


@query("q43", order_by=("s_store_name", "s_store_id") + _DAY_SALES)
def plan_q43(t):
    """A store's week: seven conditional sums over one scan of a year of sales.

    `ELSE NULL` rather than `ELSE 0`, which matters — a store that never sold on a Sunday
    reports NULL and not zero, and that is only true because `SUM` over an all-null group is
    NULL. Getting that wrong is invisible in a query that writes `ELSE 0`, which is why
    these two spellings both appear in the benchmark.
    """
    store_sales = fact(t, "store_sales",
                       ["ss_sold_date_sk", "ss_store_sk", "ss_sales_price"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year", "d_day_name"],
                   Binary("==", Col("d_year"), Lit(2000)))
    store = dim(t, "store", ["s_store_sk", "s_store_name", "s_store_id", "s_gmt_offset"],
                Binary("==", Col("s_gmt_offset"), Lit(-5)))
    joined = star(
        store_sales[1],
        ("d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
        ("s_ss", store, "s_store_sk", "ss_store_sk"),
    )
    per_day = N.project(
        "per_day", joined,
        [Alias(Col("s_store_name"), "s_store_name"), Alias(Col("s_store_id"), "s_store_id")]
        + [Alias(Case(whens=((Binary("==", Col("d_day_name"), Lit(day)),
                              Col("ss_sales_price")),),
                      otherwise=Lit(float("nan"))), column)
           for day, column in zip(_DAYS, _DAY_SALES)],
    )
    final = aggregate_by(
        "agg", per_day, ["s_store_name", "s_store_id"],
        [A.Agg(A.SUM, column, column) for column in _DAY_SALES],
        schema_frame=corpus.schema_of(store[0],
                                      **{column: "float64" for column in _DAY_SALES}),
    )
    out = select("out", final, "s_store_name", "s_store_id", *_DAY_SALES)
    return N.unload(
        "unload",
        sorted_output("sort", out, ["s_store_name", "s_store_id"] + list(_DAY_SALES),
                      [True] * (2 + len(_DAY_SALES)), fetch=100),
    )


# -- a sale and the return that may not have happened -------------------------------


@query("q40", order_by=("w_state", "i_item_id"))
def plan_q40(t):
    """Catalog sales either side of a date, net of refunds, by warehouse state.

    `catalog_sales LEFT OUTER JOIN catalog_returns` preserves the fact table, so — as in
    q93 — it is lowered as a **Right** join with the returns as the build: Right preserves
    the probe, which keeps the 1.4M-row side streaming instead of being collected. The
    `coalesce` is what makes the outer join matter, turning the null of an unreturned sale
    into the zero the arithmetic wants.
    """
    catalog_sales = fact(t, "catalog_sales",
                         ["cs_order_number", "cs_item_sk", "cs_warehouse_sk",
                          "cs_sold_date_sk", "cs_sales_price"])
    returns = dim(t, "catalog_returns", ["cr_order_number", "cr_item_sk", "cr_refunded_cash"],
                  rows=FACT_ROWS)
    warehouse = dim(t, "warehouse", ["w_warehouse_sk", "w_state"])
    item = dim(t, "item", ["i_item_sk", "i_item_id", "i_current_price"],
               between(Col("i_current_price"), Lit(0.99), Lit(1.49)))
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_date"],
                   between(Col("d_date"), date("2000-02-10"), date("2000-04-10")))
    refunded = N.hash_join(
        "cr_cs",
        N.coalesce_all("cr_cs_build", returns[1], schema=dict(returns[0].dtypes)),
        catalog_sales[1], JoinType.RIGHT,
        ["cr_order_number", "cr_item_sk"], ["cs_order_number", "cs_item_sk"],
    )
    joined = star(
        refunded,
        ("i_cs", item, "i_item_sk", "cs_item_sk"),
        ("w_cs", warehouse, "w_warehouse_sk", "cs_warehouse_sk"),
        ("d_cs", date_dim, "d_date_sk", "cs_sold_date_sk"),
    )
    split = date("2000-03-11")
    net = Binary("-", Col("cs_sales_price"), Coalesce((Col("cr_refunded_cash"), Lit(0.0))))
    sides = N.project(
        "sides", joined,
        [Alias(Col("w_state"), "w_state"), Alias(Col("i_item_id"), "i_item_id"),
         Alias(Case(whens=((Binary("<", Col("d_date"), split), net),), otherwise=Lit(0.0)),
               "before"),
         Alias(Case(whens=((Binary(">=", Col("d_date"), split), net),), otherwise=Lit(0.0)),
               "after")],
    )
    final = aggregate_by(
        "agg", sides, ["w_state", "i_item_id"],
        [A.Agg(A.SUM, "before", "sales_before"), A.Agg(A.SUM, "after", "sales_after")],
        schema_frame=corpus.schema_of(warehouse[0], item[0], before="float64", after="float64"),
    )
    out = select("out", final, "w_state", "i_item_id", "sales_before", "sales_after")
    return N.unload(
        "unload", sorted_output("sort", out, ["w_state", "i_item_id"], [True, True], fetch=100)
    )


# -- a report whose group key is wider than its output -------------------------------


@query("q91", order_by=("Returns_Loss",))
def plan_q91(t):
    """Catalog returns handled by each call centre, for two customer segments.

    The `GROUP BY` names five columns and the `SELECT` returns four of them: marital status
    and education are grouped on but not projected, so one call centre can appear twice.
    That is legal SQL and a lowering that "tidied" the group key would change the answer,
    which is why the projection above the aggregate drops the two columns rather than the
    aggregate never grouping on them.

    catalog_returns is the fact; `customer` is 100K rows and streams into the build side it
    forms with the demographics, the address and the household.
    """
    returns = fact(t, "catalog_returns",
                   ["cr_call_center_sk", "cr_returned_date_sk", "cr_returning_customer_sk",
                    "cr_net_loss"])
    call_center = dim(t, "call_center",
                      ["cc_call_center_sk", "cc_call_center_id", "cc_name", "cc_manager"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year", "d_moy"],
                   all_of(Binary("==", Col("d_year"), Lit(1998)),
                          Binary("==", Col("d_moy"), Lit(11))))
    customer = fact(t, "customer",
                    ["c_customer_sk", "c_current_cdemo_sk", "c_current_hdemo_sk",
                     "c_current_addr_sk"])
    demographics = fact(t, "customer_demographics",
                        ["cd_demo_sk", "cd_marital_status", "cd_education_status"])
    segments = (demographics[0], N.filter_(
        "segments", demographics[1],
        any_of(all_of(Binary("==", Col("cd_marital_status"), Lit("M")),
                      Binary("==", Col("cd_education_status"), Lit("Unknown"))),
               all_of(Binary("==", Col("cd_marital_status"), Lit("W")),
                      Binary("==", Col("cd_education_status"), Lit("Advanced Degree")))),
    ))
    household = dim(t, "household_demographics", ["hd_demo_sk", "hd_buy_potential"],
                    Like(Col("hd_buy_potential"), "Unknown%"))
    address = dim(t, "customer_address", ["ca_address_sk", "ca_gmt_offset"],
                  Binary("==", Col("ca_gmt_offset"), Lit(-7)))
    eligible = select(
        "eligible",
        star(
            customer[1],
            ("cd_c", segments, "cd_demo_sk", "c_current_cdemo_sk"),
            ("hd_c", household, "hd_demo_sk", "c_current_hdemo_sk"),
            ("ca_c", address, "ca_address_sk", "c_current_addr_sk"),
        ),
        "c_customer_sk", "cd_marital_status", "cd_education_status",
    )
    joined = star(
        returns[1],
        ("cc_cr", call_center, "cc_call_center_sk", "cr_call_center_sk"),
        ("d_cr", date_dim, "d_date_sk", "cr_returned_date_sk"),
    )
    with_customers = N.hash_join(
        "c_cr",
        N.coalesce_all("c_cr_build", eligible,
                       schema=dict(corpus.schema_of(customer[0], demographics[0]).dtypes)),
        joined, JoinType.INNER, ["c_customer_sk"], ["cr_returning_customer_sk"],
    )
    keys = ["cc_call_center_id", "cc_name", "cc_manager", "cd_marital_status",
            "cd_education_status"]
    final = aggregate_by(
        "agg", select("keys", with_customers, *keys, "cr_net_loss"), keys,
        [A.Agg(A.SUM, "cr_net_loss", "Returns_Loss")],
        schema_frame=corpus.schema_of(call_center[0], demographics[0], returns[0]),
    )
    out = rename("out", final,
                 [("cc_call_center_id", "Call_Center"), ("cc_name", "Call_Center_Name"),
                  ("cc_manager", "Manager"), ("Returns_Loss", "Returns_Loss")])
    return N.unload("unload", sorted_output("sort", out, ["Returns_Loss"], [False]))


# -- eight counts and seven cross joins ---------------------------------------------


#: The three (dependants, vehicles) bands q88 accepts, and the half-hours it counts.
_HOUSEHOLD_BANDS = ((4, 6), (2, 4), (0, 2))
_HALF_HOURS = (("h8_30_to_9", 8, False), ("h9_to_9_30", 9, True),
               ("h9_30_to_10", 9, False), ("h10_to_10_30", 10, True),
               ("h10_30_to_11", 10, False), ("h11_to_11_30", 11, True),
               ("h11_30_to_12", 11, False), ("h12_to_12_30", 12, True))


@query("q88")
def plan_q88(t):
    """Store traffic by half-hour, as eight scalar subqueries side by side.

    Written as eight `count(*)`s over the same three joins with one predicate changed, and
    then all eight in one `FROM` — which is a chain of cross joins over eight one-row
    tables. The mode has nothing to say about the cross joins (one row by one row), and
    everything to say about the eight scans: they are eight separate subtrees, so this is
    the query that says what a bucketed report costs when it is *not* written as one pass.
    See q50 and its siblings for the same shape written the other way.
    """
    def half_hour(name, hour, first_half):
        store_sales = fact(t, "store_sales", ["ss_sold_time_sk", "ss_hdemo_sk", "ss_store_sk"],
                           tag=f"{name}_store_sales")
        time_dim = dim(t, "time_dim", ["t_time_sk", "t_hour", "t_minute"],
                       all_of(Binary("==", Col("t_hour"), Lit(hour)),
                              Binary("<", Col("t_minute"), Lit(30)) if first_half
                              else Binary(">=", Col("t_minute"), Lit(30))),
                       tag=f"{name}_time_dim")
        household = dim(t, "household_demographics",
                        ["hd_demo_sk", "hd_dep_count", "hd_vehicle_count"],
                        any_of(*[all_of(Binary("==", Col("hd_dep_count"), Lit(dependants)),
                                        Binary("<=", Col("hd_vehicle_count"), Lit(vehicles)))
                                 for dependants, vehicles in _HOUSEHOLD_BANDS]),
                        tag=f"{name}_household_demographics")
        store = dim(t, "store", ["s_store_sk", "s_store_name"],
                    Binary("==", Col("s_store_name"), Lit("ese")), tag=f"{name}_store")
        joined = star(
            store_sales[1],
            (f"{name}_t_ss", time_dim, "t_time_sk", "ss_sold_time_sk"),
            (f"{name}_hd_ss", household, "hd_demo_sk", "ss_hdemo_sk"),
            (f"{name}_s_ss", store, "s_store_sk", "ss_store_sk"),
        )
        return aggregate_to_one_row(f"{name}_agg", joined, [A.Agg(A.COUNT, None, name)],
                                    corpus.schema_of(store_sales[0]))

    counts = [half_hour(name, hour, first) for name, hour, first in _HALF_HOURS]
    row = counts[0]
    for index, count in enumerate(counts[1:], start=1):
        row = N.cross_join(f"cross{index}",
                           N.coalesce_all(f"cross{index}_build", row), count)
    return N.unload("unload", select("out", row, *[name for name, _, _ in _HALF_HOURS]))


# -- rollups over a dimension's hierarchy -------------------------------------------


def _rollup_report(tag, child, keys, aggs, schema_frame, order_by, ascending, fetch=100):
    """`GROUP BY ROLLUP(k0, k1, …)`: n+1 grouping sets, and the id that keeps them apart.

    The partial emits one row per set per group, tagged with the bitmask of the positions it
    masked, and every phase above it groups on the keys *and* that tag — otherwise a
    rolled-up NULL and a naturally-null key would merge into one group. The tag is dropped
    in the projection at the top: it is machinery, not an answer.
    """
    rolled = aggregate_by(f"{tag}_rollup", child, keys, aggs,
                          grouping_sets=A.rollup_masks(len(keys)),
                          schema_frame=schema_frame)
    out = select(f"{tag}_out", rolled, *keys, *[agg.output for agg in aggs])
    return N.unload(
        "unload",
        sorted_output("sort", out, list(order_by), list(ascending), fetch=fetch,
                      nulls_first=True),
    )


_Q18_KEYS = ["i_item_id", "ca_country", "ca_state", "ca_county"]
#: The seven averages q18 reports, as `(the column averaged, the output name)`.
_Q18_AGGS = (("cs_quantity", "agg1"), ("cs_list_price", "agg2"), ("cs_coupon_amt", "agg3"),
             ("cs_sales_price", "agg4"), ("cs_net_profit", "agg5"), ("c_birth_year", "agg6"),
             ("cd_dep_count", "agg7"))


@query("q18", order_by=("ca_country", "ca_state", "ca_county", "i_item_id"))
def plan_q18(t):
    """Catalog buying by item and by where the buyer lives, rolled up through the geography.

    Two copies of customer_demographics again: one reached from the *sale* (who it was
    billed to) and one from the *customer's* current record, and only the first is filtered.
    The second contributes nothing but a join, which is the query insisting that the
    customer have a demographic record at all.

    `ROLLUP` over four keys is five grouping sets — item within county within state within
    country, and the grand total — and the ORDER BY reads them geography-first, which is not
    the order they are grouped in.
    """
    catalog_sales = fact(t, "catalog_sales",
                         ["cs_sold_date_sk", "cs_item_sk", "cs_bill_cdemo_sk",
                          "cs_bill_customer_sk", "cs_quantity", "cs_list_price",
                          "cs_coupon_amt", "cs_sales_price", "cs_net_profit"])
    cd1 = fact(t, "customer_demographics",
               ["cd_demo_sk", "cd_gender", "cd_education_status", "cd_dep_count"], tag="cd1")
    billed_to = (cd1[0], N.filter_(
        "cd1_segment", cd1[1],
        all_of(Binary("==", Col("cd_gender"), Lit("F")),
               Binary("==", Col("cd_education_status"), Lit("Unknown"))),
    ))
    cd2 = fact(t, "customer_demographics", ["cd_demo_sk"], tag="cd2")
    current = (cd2[0], rename("cd2_renamed", cd2[1], [("cd_demo_sk", "cd2_demo_sk")]))
    customer = dim(t, "customer",
                   ["c_customer_sk", "c_current_cdemo_sk", "c_current_addr_sk",
                    "c_birth_month", "c_birth_year"],
                   is_in(Col("c_birth_month"), (1, 6, 8, 9, 12, 2)), rows=250_000)
    address = dim(t, "customer_address", ["ca_address_sk", "ca_country", "ca_state",
                                          "ca_county"],
                  is_in(Col("ca_state"), ("MS", "IN", "ND", "OK", "NM", "VA")))
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"],
                   Binary("==", Col("d_year"), Lit(1998)))
    item = dim(t, "item", ["i_item_sk", "i_item_id"])
    joined = star(
        catalog_sales[1],
        ("d_cs", date_dim, "d_date_sk", "cs_sold_date_sk"),
        ("i_cs", item, "i_item_sk", "cs_item_sk"),
        ("cd1_cs", billed_to, "cd_demo_sk", "cs_bill_cdemo_sk"),
        ("c_cs", customer, "c_customer_sk", "cs_bill_customer_sk"),
        ("cd2_c", current, "cd2_demo_sk", "c_current_cdemo_sk"),
        ("ca_c", address, "ca_address_sk", "c_current_addr_sk"),
    )
    return _rollup_report(
        "q18", select("keys", joined, *_Q18_KEYS, *[column for column, _ in _Q18_AGGS]),
        _Q18_KEYS, [A.Agg(A.MEAN, column, output) for column, output in _Q18_AGGS],
        corpus.schema_of(item[0], address[0], customer[0], cd1[0], catalog_sales[0]),
        ("ca_country", "ca_state", "ca_county", "i_item_id"), (True, True, True, True),
    )


_Q22_KEYS = ["i_product_name", "i_brand", "i_class", "i_category"]


@query("q22", order_by=("qoh",) + tuple(_Q22_KEYS))
def plan_q22(t):
    """Average stock on hand over a year, rolled up through the product hierarchy.

    The simplest rollup in the corpus and the largest input: 11.7M inventory rows, one join,
    five grouping sets. Worth having because the rollup is over an *average* — the partial
    carries [sum, count] per set per group, the merge adds them, and only the finalize
    divides, so a total's average is computed from its parts rather than by averaging
    averages.
    """
    inventory = fact(t, "inventory", ["inv_date_sk", "inv_item_sk", "inv_quantity_on_hand"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_month_seq"],
                   between(Col("d_month_seq"), Lit(1200), Lit(1200 + 11)))
    item = dim(t, "item", ["i_item_sk"] + _Q22_KEYS)
    joined = star(
        inventory[1],
        ("d_inv", date_dim, "d_date_sk", "inv_date_sk"),
        ("i_inv", item, "i_item_sk", "inv_item_sk"),
    )
    return _rollup_report(
        "q22", select("keys", joined, *_Q22_KEYS, "inv_quantity_on_hand"), _Q22_KEYS,
        [A.Agg(A.MEAN, "inv_quantity_on_hand", "qoh")],
        corpus.schema_of(item[0], inventory[0]),
        ("qoh",) + tuple(_Q22_KEYS), (True, True, True, True, True),
    )
