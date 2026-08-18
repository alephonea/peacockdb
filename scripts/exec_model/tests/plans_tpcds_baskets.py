"""TPC-DS: the basket queries — aggregate a ticket, then join back to the customer.

Five queries with one shape, and it is a shape worth having in the corpus because the
aggregate is *not* at the top. store_sales is summarized per ticket first, which collapses
2.9M rows to tens of thousands, and only then is the customer joined on. That ordering is
the whole point: joining customer first would carry a name and a salutation through every
line item of every basket, and the aggregate would then have to group on them.

So the plan is two stages with the aggregate between them, and the second stage's build side
is the *result* of the first — a subplan rather than a table, which is why these call
`hash_join` directly instead of going through `star`.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from . import corpus
from .plan_helpers import (
    aggregate_by, all_of, any_of, between, is_in, rename, select, sorted_output,
)
from .plans_tpcds_common import dim, fact, registry, star
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import Alias, Binary, Case, Col, Lit, Substring
from ..operators.join_types import JoinType

QUERIES, ORDER_BY, query = registry()

#: The three years every basket query looks at, written in the spec as `1999, 1999+1,
#: 1999+2` and meaning the same thing.
_YEARS = (1999, 2000, 2001)


def _dependants_per_vehicle(threshold):
    """`hd_dep_count / hd_vehicle_count > threshold`, as q34 and q73 write it.

    The `CASE` guarding the division is redundant — `hd_vehicle_count > 0` is already an
    `AND` beside it — but it is what the query says, and a lowering that dropped it would be
    deciding that a division by zero cannot happen rather than being told so.
    """
    return all_of(
        Binary(">", Col("hd_vehicle_count"), Lit(0)),
        Binary(">",
               Case(whens=((Binary(">", Col("hd_vehicle_count"), Lit(0)),
                            Binary("/", Col("hd_dep_count"), Col("hd_vehicle_count"))),),
                    otherwise=Lit(float("nan"))),
               Lit(threshold)),
    )


def _tickets_per_customer(t, tag, day_of_month, counties, buy_potential, ratio):
    """Baskets of one county's stores, counted per ticket. q34 and q73's first stage."""
    store_sales = fact(t, "store_sales",
                       ["ss_sold_date_sk", "ss_store_sk", "ss_hdemo_sk", "ss_ticket_number",
                        "ss_customer_sk"], tag=f"{tag}_store_sales")
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_dom", "d_year"],
                   all_of(day_of_month, is_in(Col("d_year"), _YEARS)),
                   tag=f"{tag}_date_dim")
    store = dim(t, "store", ["s_store_sk", "s_county"],
                is_in(Col("s_county"), counties), tag=f"{tag}_store")
    household = dim(t, "household_demographics",
                    ["hd_demo_sk", "hd_buy_potential", "hd_dep_count", "hd_vehicle_count"],
                    all_of(is_in(Col("hd_buy_potential"), buy_potential),
                           _dependants_per_vehicle(ratio)),
                    tag=f"{tag}_household_demographics")
    joined = star(
        store_sales[1],
        (f"{tag}_d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
        (f"{tag}_s_ss", store, "s_store_sk", "ss_store_sk"),
        (f"{tag}_hd_ss", household, "hd_demo_sk", "ss_hdemo_sk"),
    )
    keys = ["ss_ticket_number", "ss_customer_sk"]
    return store_sales[0], aggregate_by(
        f"{tag}_agg", select(f"{tag}_keys", joined, *keys), keys,
        [A.Agg(A.COUNT, None, "cnt")],
        schema_frame=corpus.schema_of(store_sales[0]),
    )


def _with_customer(t, tag, tickets, ticket_schema, carried):
    """Join the customer onto a per-ticket summary, the summary being the build side.

    The summary is thousands of rows and customer is a hundred thousand, so the summary
    builds and customer streams — the opposite of what the query text's join order suggests
    and the only assignment that keeps a fact-sized table out of a hash table.
    """
    customer = fact(t, "customer",
                    ["c_customer_sk", "c_last_name", "c_first_name", "c_salutation",
                     "c_preferred_cust_flag", "c_current_addr_sk"], tag=f"{tag}_customer")
    joined = N.hash_join(
        f"{tag}_c_dn",
        N.coalesce_all(f"{tag}_c_dn_build", tickets, schema=dict(ticket_schema.dtypes)),
        customer[1], JoinType.INNER, ["ss_customer_sk"], ["c_customer_sk"],
    )
    return customer, joined, carried


# -- counted baskets ----------------------------------------------------------------


@query("q34", order_by=("c_last_name", "c_first_name", "c_salutation",
                        "c_preferred_cust_flag", "ss_ticket_number"))
def plan_q34(t):
    """Customers whose baskets in one county held fifteen to twenty items.

    `cnt BETWEEN 15 AND 20` is a predicate on the aggregate's *output*, so it sits above the
    aggregate and below the join — which is what makes the join's build side small. It is
    also why the count cannot be pushed into the join: the query needs the whole basket
    counted before it knows whether the customer is wanted.
    """
    schema, tickets = _tickets_per_customer(
        t, "q34", any_of(between(Col("d_dom"), Lit(1), Lit(3)),
                         between(Col("d_dom"), Lit(25), Lit(28))),
        ("Williamson County",), (">10000", "Unknown"), 1.2,
    )
    wanted = N.filter_("fifteen_to_twenty", tickets, between(Col("cnt"), Lit(15), Lit(20)))
    _, joined, _ = _with_customer(t, "q34", wanted,
                                  corpus.schema_of(schema, cnt="int64"), None)
    out = select("out", joined, "c_last_name", "c_first_name", "c_salutation",
                 "c_preferred_cust_flag", "ss_ticket_number", "cnt")
    return N.unload(
        "unload",
        sorted_output("sort", out,
                      ["c_last_name", "c_first_name", "c_salutation",
                       "c_preferred_cust_flag", "ss_ticket_number"],
                      [True, True, True, False, True], nulls_first=True),
    )


@query("q73", order_by=("cnt", "c_last_name"))
def plan_q73(t):
    """q34's small-basket twin: four counties, the first two days of a month, one to five
    items. Same lowering with different constants, which is the benchmark checking that a
    plan is chosen from the shape and not from the numbers."""
    schema, tickets = _tickets_per_customer(
        t, "q73", between(Col("d_dom"), Lit(1), Lit(2)),
        ("Orange County", "Bronx County", "Franklin Parish", "Williamson County"),
        ("Unknown", ">10000"), 1,
    )
    wanted = N.filter_("one_to_five", tickets, between(Col("cnt"), Lit(1), Lit(5)))
    _, joined, _ = _with_customer(t, "q73", wanted,
                                  corpus.schema_of(schema, cnt="int64"), None)
    out = select("out", joined, "c_last_name", "c_first_name", "c_salutation",
                 "c_preferred_cust_flag", "ss_ticket_number", "cnt")
    return N.unload(
        "unload", sorted_output("sort", out, ["cnt", "c_last_name"], [False, True])
    )


# -- baskets bought away from home --------------------------------------------------


def _baskets_by_city(t, tag, day_filter, aggs):
    """store_sales summarized per ticket, carrying the city the basket was bought in.

    The city comes from the *sale's* address, and the query compares it with the customer's
    current one — two different rows of customer_address, so the two copies cannot share a
    scan and the first one's column has to be renamed before the second arrives. A batch
    cannot hold two columns called `ca_city`.
    """
    store_sales = fact(t, "store_sales",
                       ["ss_sold_date_sk", "ss_store_sk", "ss_hdemo_sk", "ss_addr_sk",
                        "ss_ticket_number", "ss_customer_sk"] + [agg.column for agg in aggs],
                       tag=f"{tag}_store_sales")
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_dow", "d_dom", "d_year"],
                   all_of(day_filter, is_in(Col("d_year"), _YEARS)), tag=f"{tag}_date_dim")
    store = dim(t, "store", ["s_store_sk", "s_city"],
                is_in(Col("s_city"), ("Fairview", "Midway")), tag=f"{tag}_store")
    household = dim(t, "household_demographics",
                    ["hd_demo_sk", "hd_dep_count", "hd_vehicle_count"],
                    any_of(Binary("==", Col("hd_dep_count"), Lit(4)),
                           Binary("==", Col("hd_vehicle_count"), Lit(3))),
                    tag=f"{tag}_household_demographics")
    address = dim(t, "customer_address", ["ca_address_sk", "ca_city"],
                  tag=f"{tag}_bought_address")
    joined = star(
        store_sales[1],
        (f"{tag}_d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
        (f"{tag}_s_ss", store, "s_store_sk", "ss_store_sk"),
        (f"{tag}_hd_ss", household, "hd_demo_sk", "ss_hdemo_sk"),
        (f"{tag}_ca_ss", address, "ca_address_sk", "ss_addr_sk"),
    )
    keys = ["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "bought_city"]
    bought = rename(
        f"{tag}_bought", joined,
        [(column, column) for column in
         ["ss_ticket_number", "ss_customer_sk", "ss_addr_sk"] + [agg.column for agg in aggs]]
        + [("ca_city", "bought_city")],
    )
    return store_sales[0], aggregate_by(
        f"{tag}_agg", bought, keys, aggs,
        schema_frame=corpus.schema_of(store_sales[0], bought_city="object"),
    )


def _away_from_home(t, tag, schema, tickets, measures):
    """Join the customer and their current address, and keep the baskets bought elsewhere."""
    customer, joined, _ = _with_customer(
        t, tag, tickets,
        corpus.schema_of(schema, bought_city="object",
                         **{name: "float64" for name in measures}), None,
    )
    current = dim(t, "customer_address", ["ca_address_sk", "ca_city"],
                  tag=f"{tag}_current_address")
    with_address = star(joined, (f"{tag}_ca_c", current, "ca_address_sk", "c_current_addr_sk"))
    return N.filter_(f"{tag}_elsewhere", with_address,
                     Binary("!=", Col("ca_city"), Col("bought_city")))


@query("q46", order_by=("c_last_name", "c_first_name", "ca_city", "bought_city",
                        "ss_ticket_number"))
def plan_q46(t):
    """Weekend baskets bought in a city the customer does not live in."""
    measures = [A.Agg(A.SUM, "ss_coupon_amt", "amt"), A.Agg(A.SUM, "ss_net_profit", "profit")]
    schema, tickets = _baskets_by_city(t, "q46", is_in(Col("d_dow"), (6, 0)), measures)
    kept = _away_from_home(t, "q46", schema, tickets, ("amt", "profit"))
    out = select("out", kept, "c_last_name", "c_first_name", "ca_city", "bought_city",
                 "ss_ticket_number", "amt", "profit")
    return N.unload(
        "unload",
        sorted_output("sort", out,
                      ["c_last_name", "c_first_name", "ca_city", "bought_city",
                       "ss_ticket_number"],
                      [True] * 5, fetch=100, nulls_first=True),
    )


@query("q68", order_by=("c_last_name", "ss_ticket_number"))
def plan_q68(t):
    """q46 over the first two days of the month, reporting money rather than profit. The
    ORDER BY names two columns of seven, so most of the output is tied — the compare pins
    the sort keys positionally and the rows as a multiset for exactly this reason."""
    measures = [A.Agg(A.SUM, "ss_ext_sales_price", "extended_price"),
                A.Agg(A.SUM, "ss_ext_list_price", "list_price"),
                A.Agg(A.SUM, "ss_ext_tax", "extended_tax")]
    schema, tickets = _baskets_by_city(t, "q68", between(Col("d_dom"), Lit(1), Lit(2)),
                                       measures)
    kept = _away_from_home(t, "q68", schema, tickets,
                           ("extended_price", "list_price", "extended_tax"))
    out = select("out", kept, "c_last_name", "c_first_name", "ca_city", "bought_city",
                 "ss_ticket_number", "extended_price", "extended_tax", "list_price")
    return N.unload(
        "unload",
        sorted_output("sort", out, ["c_last_name", "ss_ticket_number"], [True, True],
                      fetch=100, nulls_first=True),
    )


@query("q79", order_by=("c_last_name", "c_first_name", 'main."substring"(s_city, 1, 30)',
                        "profit", "ss_ticket_number"))
def plan_q79(t):
    """Monday baskets from mid-sized stores, by store city.

    The city is a `SUBSTRING` in both the `SELECT` and the `ORDER BY` and is never aliased,
    so its output name is whatever the engine calls it — DuckDB says
    `main."substring"(s_city, 1, 30)`, and that string is the column the lowering has to
    produce. A name is part of the answer.
    """
    store_sales = fact(t, "store_sales",
                       ["ss_sold_date_sk", "ss_store_sk", "ss_hdemo_sk", "ss_addr_sk",
                        "ss_ticket_number", "ss_customer_sk", "ss_coupon_amt",
                        "ss_net_profit"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_dow", "d_year"],
                   all_of(Binary("==", Col("d_dow"), Lit(1)), is_in(Col("d_year"), _YEARS)))
    store = dim(t, "store", ["s_store_sk", "s_city", "s_number_employees"],
                between(Col("s_number_employees"), Lit(200), Lit(295)))
    household = dim(t, "household_demographics",
                    ["hd_demo_sk", "hd_dep_count", "hd_vehicle_count"],
                    any_of(Binary("==", Col("hd_dep_count"), Lit(6)),
                           Binary(">", Col("hd_vehicle_count"), Lit(2))))
    joined = star(
        store_sales[1],
        ("d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
        ("s_ss", store, "s_store_sk", "ss_store_sk"),
        ("hd_ss", household, "hd_demo_sk", "ss_hdemo_sk"),
    )
    keys = ["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "s_city"]
    tickets = aggregate_by(
        "agg", select("keys", joined, *keys, "ss_coupon_amt", "ss_net_profit"), keys,
        [A.Agg(A.SUM, "ss_coupon_amt", "amt"), A.Agg(A.SUM, "ss_net_profit", "profit")],
        schema_frame=corpus.schema_of(store_sales[0], store[0]),
    )
    _, with_customer, _ = _with_customer(
        t, "q79", tickets,
        corpus.schema_of(store_sales[0], store[0], amt="float64", profit="float64"), None,
    )
    city = 'main."substring"(s_city, 1, 30)'
    out = N.project(
        "out", with_customer,
        [Alias(Col("c_last_name"), "c_last_name"), Alias(Col("c_first_name"), "c_first_name"),
         Alias(Substring(Col("s_city"), 1, 30), city),
         Alias(Col("ss_ticket_number"), "ss_ticket_number"), Alias(Col("amt"), "amt"),
         Alias(Col("profit"), "profit")],
    )
    return N.unload(
        "unload",
        sorted_output("sort", out,
                      ["c_last_name", "c_first_name", city, "profit", "ss_ticket_number"],
                      [True] * 5, fetch=100, nulls_first=True),
    )
