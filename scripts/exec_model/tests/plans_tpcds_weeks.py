"""TPC-DS: the last handful — nested date subqueries, a self-joined CTE, and q64.

What is left after the families: queries whose lowering is mostly its own. They share one
device, which is why they are together — a date range named indirectly, as *the week a
given day fell in* rather than as two literals:

    d_date IN (SELECT d_date FROM date_dim
               WHERE d_week_seq = (SELECT d_week_seq FROM date_dim WHERE d_date = '…'))

That is two nested uncorrelated subqueries, and both lower to joins: the inner one to a
one-row build side joined on `d_week_seq`, the outer one to nothing at all, because joining
date_dim to its own week is already the set of dates the `IN` describes. A planner that
kept the `IN` as a semi join would get the same answer for more work.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from . import corpus
from .plan_helpers import (
    aggregate_by, aggregate_to_one_row, all_of, any_of, between, date, is_in, rename, select,
    sorted_output,
)
from .plans_tpcds_common import dim, fact, registry, star
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import (
    Alias, Binary, Cast, Col, Lit, Round, Substring, Upper,
)
from ..operators.join_types import JoinType

QUERIES, ORDER_BY, query = registry()


def _weeks_of(t, tag, days):
    """date_dim narrowed to the whole weeks that `days` fall in.

    The inner subquery is a handful of `d_week_seq`s and becomes the build side; date_dim
    streams past it. One join replaces two nested `IN`s.
    """
    seeds = dim(t, "date_dim", ["d_date", "d_week_seq"],
                is_in(Col("d_date"), [date(day).value for day in days]),
                tag=f"{tag}_seed_date_dim")
    weeks = aggregate_by(f"{tag}_weeks", select(f"{tag}_seed_keys", seeds[1], "d_week_seq"),
                         ["d_week_seq"], [],
                         schema_frame=corpus.schema_of(d_week_seq="int64"))
    calendar = dim(t, "date_dim", ["d_date_sk", "d_week_seq"], tag=f"{tag}_date_dim")
    return calendar[0], N.hash_join(
        f"{tag}_in_week",
        N.coalesce_all(f"{tag}_weeks_all", weeks,
                       schema=dict(corpus.schema_of(d_week_seq="int64").dtypes)),
        calendar[1], JoinType.INNER, ["d_week_seq"], ["d_week_seq"],
    )


def _item_revenue(t, tag, channel, prefix, date_key, item_key, measure, output, days):
    """One channel's total per item id, over the weeks `days` name."""
    sales = fact(t, channel, [date_key, item_key, measure], tag=f"{tag}_{channel}")
    item = dim(t, "item", ["i_item_sk", "i_item_id"], tag=f"{tag}_item")
    calendar_frame, calendar = _weeks_of(t, tag, days)
    joined = star(
        sales[1],
        (f"{tag}_d", (corpus.schema_of(calendar_frame), calendar), "d_date_sk", date_key),
        (f"{tag}_i", item, "i_item_sk", item_key),
    )
    totals = aggregate_by(
        f"{tag}_agg", select(f"{tag}_keys", joined, "i_item_id", measure), ["i_item_id"],
        [A.Agg(A.SUM, measure, output)],
        schema_frame=corpus.schema_of(item[0], **{measure: "float64"}),
    )
    return rename(f"{tag}_alias", totals,
                  [("i_item_id", f"{tag}_item_id"), (output, output)])


def _three_way_item_join(t, tag, days, channels, extra=None):
    """Three per-item totals joined on the item id, one row per item present in all three."""
    branches = {}
    for label, channel, prefix, date_key, item_key, measure, output in channels:
        branches[label] = _item_revenue(t, f"{tag}_{label}", channel, prefix, date_key,
                                        item_key, measure, output, days)
    first = channels[0][0]
    joined = branches[first]
    for label, *_, output in channels[1:]:
        joined = N.hash_join(
            f"{tag}_{label}_join",
            N.coalesce_all(f"{tag}_{label}_build", branches[label],
                           schema={f"{tag}_{label}_item_id": "object", output: "float64"}),
            joined, JoinType.INNER, [f"{tag}_{label}_item_id"], [f"{tag}_{first}_item_id"],
        )
    return joined if extra is None else N.filter_(f"{tag}_extra", joined, extra)


_Q58_CHANNELS = (
    ("ss", "store_sales", "ss", "ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price",
     "ss_item_rev"),
    ("cs", "catalog_sales", "cs", "cs_sold_date_sk", "cs_item_sk", "cs_ext_sales_price",
     "cs_item_rev"),
    ("ws", "web_sales", "ws", "ws_sold_date_sk", "ws_item_sk", "ws_ext_sales_price",
     "ws_item_rev"),
)


@query("q58", order_by=("item_id", "ss_item_rev"))
def plan_q58(t):
    """Items that sold about equally well in all three channels in one week.

    "About equally" is six `BETWEEN`s over the three revenues — every pair, both ways — and
    every one of them reads two of the three CTEs, so none can be pushed anywhere. They are
    one filter above the three-way join, which is where a predicate over three relations has
    to live.
    """
    def within_a_tenth(left, right):
        return between(Col(left), Binary("*", Lit(0.9), Col(right)),
                       Binary("*", Lit(1.1), Col(right)))

    pairs = [("ss_item_rev", "cs_item_rev"), ("ss_item_rev", "ws_item_rev"),
             ("cs_item_rev", "ss_item_rev"), ("cs_item_rev", "ws_item_rev"),
             ("ws_item_rev", "ss_item_rev"), ("ws_item_rev", "cs_item_rev")]
    joined = _three_way_item_join(
        t, "q58", ("2000-01-03",), _Q58_CHANNELS,
        extra=all_of(*[within_a_tenth(left, right) for left, right in pairs]),
    )
    total = Binary("+", Binary("+", Col("ss_item_rev"), Col("cs_item_rev")),
                   Col("ws_item_rev"))
    average = Binary("/", total, Lit(3.0))
    out = N.project(
        "out", joined,
        [Alias(Col("q58_ss_item_id"), "item_id"),
         Alias(Col("ss_item_rev"), "ss_item_rev"),
         Alias(Binary("*", Binary("/", Col("ss_item_rev"), average), Lit(100)), "ss_dev"),
         Alias(Col("cs_item_rev"), "cs_item_rev"),
         Alias(Binary("*", Binary("/", Col("cs_item_rev"), average), Lit(100)), "cs_dev"),
         Alias(Col("ws_item_rev"), "ws_item_rev"),
         Alias(Binary("*", Binary("/", Col("ws_item_rev"), average), Lit(100)), "ws_dev"),
         Alias(average, "average")],
    )
    return N.unload(
        "unload",
        sorted_output("sort", out, ["item_id", "ss_item_rev"], [True, True], fetch=100,
                      nulls_first=True),
    )


_Q83_CHANNELS = (
    ("sr", "store_returns", "sr", "sr_returned_date_sk", "sr_item_sk", "sr_return_quantity",
     "sr_item_qty"),
    ("cr", "catalog_returns", "cr", "cr_returned_date_sk", "cr_item_sk", "cr_return_quantity",
     "cr_item_qty"),
    ("wr", "web_returns", "wr", "wr_returned_date_sk", "wr_item_sk", "wr_return_quantity",
     "wr_item_qty"),
)


@query("q83", order_by=("item_id", "sr_item_qty"))
def plan_q83(t):
    """q58 over returns rather than sales, and over three weeks rather than one.

    No `BETWEEN`s here — every item returned in all three channels is reported — so this is
    the same lowering with the filter removed, which is what makes the pair worth having.
    """
    joined = _three_way_item_join(t, "q83",
                                 ("2000-06-30", "2000-09-27", "2000-11-17"), _Q83_CHANNELS)
    total = Binary("+", Binary("+", Col("sr_item_qty"), Col("cr_item_qty")),
                   Col("wr_item_qty"))

    def share(column):
        return Binary("*", Binary("/", Binary("/", Col(column), total), Lit(3.0)), Lit(100))

    out = N.project(
        "out", joined,
        [Alias(Col("q83_sr_item_id"), "item_id"),
         Alias(Col("sr_item_qty"), "sr_item_qty"), Alias(share("sr_item_qty"), "sr_dev"),
         Alias(Col("cr_item_qty"), "cr_item_qty"), Alias(share("cr_item_qty"), "cr_dev"),
         Alias(Col("wr_item_qty"), "wr_item_qty"), Alias(share("wr_item_qty"), "wr_dev"),
         Alias(Binary("/", total, Lit(3.0)), "average")],
    )
    return N.unload(
        "unload",
        sorted_output("sort", out, ["item_id", "sr_item_qty"], [True, True], fetch=100,
                      nulls_first=True),
    )


# -- a CTE summed twice: once filtered, once whole -----------------------------------


_Q24_KEYS = ("c_last_name", "c_first_name", "s_store_name", "ca_state", "s_state", "i_color",
             "i_current_price", "i_manager_id", "i_units", "i_size")


def _q24_ssales(t, copy):
    """q24's `ssales`: what each customer paid at the store whose zip matches their own.

    Two predicates that no scan can hold. `s_zip = ca_zip` relates two *dimensions* — the
    store and the customer's address — so it is a filter above both joins rather than a join
    key, and `c_birth_country <> upper(ca_country)` is the same shape with a function on one
    side. store_sales is the probe throughout; the returns are the build of the one join
    that is on the transaction.
    """
    store_sales = fact(t, "store_sales",
                       ["ss_item_sk", "ss_ticket_number", "ss_customer_sk", "ss_store_sk",
                        "ss_net_paid"], tag=f"{copy}_store_sales")
    returns = dim(t, "store_returns", ["sr_item_sk", "sr_ticket_number"], rows=250_000,
                  tag=f"{copy}_store_returns")
    store = dim(t, "store", ["s_store_sk", "s_store_name", "s_state", "s_zip", "s_market_id"],
                Binary("==", Col("s_market_id"), Lit(8)), tag=f"{copy}_store")
    item = dim(t, "item", ["i_item_sk", "i_color", "i_current_price", "i_manager_id",
                           "i_units", "i_size"], tag=f"{copy}_item")
    customer = dim(t, "customer",
                   ["c_customer_sk", "c_current_addr_sk", "c_first_name", "c_last_name",
                    "c_birth_country"], rows=250_000, tag=f"{copy}_customer")
    address = dim(t, "customer_address", ["ca_address_sk", "ca_state", "ca_zip", "ca_country"],
                  tag=f"{copy}_customer_address")
    returned = N.hash_join(
        f"{copy}_sr_ss",
        N.coalesce_all(f"{copy}_sr_ss_build", returns[1], schema=dict(returns[0].dtypes)),
        store_sales[1], JoinType.INNER,
        ["sr_item_sk", "sr_ticket_number"], ["ss_item_sk", "ss_ticket_number"],
    )
    joined = star(
        returned,
        (f"{copy}_s_ss", store, "s_store_sk", "ss_store_sk"),
        (f"{copy}_i_ss", item, "i_item_sk", "ss_item_sk"),
        (f"{copy}_c_ss", customer, "c_customer_sk", "ss_customer_sk"),
        (f"{copy}_ca_c", address, "ca_address_sk", "c_current_addr_sk"),
    )
    matched = N.filter_(
        f"{copy}_same_zip_other_country", joined,
        all_of(Binary("==", Col("s_zip"), Col("ca_zip")),
               Binary("!=", Col("c_birth_country"), Upper(Col("ca_country")))),
    )
    return aggregate_by(
        f"{copy}_ssales", select(f"{copy}_keys", matched, *_Q24_KEYS, "ss_net_paid"),
        list(_Q24_KEYS), [A.Agg(A.SUM, "ss_net_paid", "netpaid")],
        schema_frame=corpus.schema_of(customer[0], store[0], address[0], item[0],
                                      store_sales[0]),
    )


@query("q24", order_by=("c_last_name", "c_first_name", "s_store_name"))
def plan_q24(t):
    """Who spent well above average on peach-coloured goods, by store.

    The `HAVING` compares each group's total against `0.05 * avg(netpaid)` over the **whole**
    of `ssales` — not over the peach rows — so the CTE is built twice with different filters
    above it, and the two meet at a cross join of one row against the grouped result. Reading
    the subquery as "the average of what we are reporting" would be the natural mistake and a
    different query.
    """
    peach = N.filter_("peach", _q24_ssales(t, "q24_rows"),
                      Binary("==", Col("i_color"), Lit("peach")))
    keys = ["c_last_name", "c_first_name", "s_store_name"]
    paid = aggregate_by(
        "q24_paid", select("q24_paid_keys", peach, *keys, "netpaid"), keys,
        [A.Agg(A.SUM, "netpaid", "paid")],
        schema_frame=corpus.schema_of(**{key: "object" for key in keys}, netpaid="float64"),
    )
    average = aggregate_to_one_row(
        "q24_average",
        select("q24_average_keys", _q24_ssales(t, "q24_avg"), "netpaid"),
        [A.Agg(A.MEAN, "netpaid", "average")], corpus.schema_of(netpaid="float64"),
    )
    threshold = N.project("q24_threshold", average,
                          [Alias(Binary("*", Lit(0.05), Col("average")), "threshold")])
    against = N.cross_join("q24_against", N.coalesce_all("q24_threshold_all", threshold), paid)
    kept = N.filter_("q24_above", against, Binary(">", Col("paid"), Col("threshold")))
    out = select("out", kept, *keys, "paid")
    return N.unload("unload", sorted_output("sort", out, keys, [True, True, True]))


# -- a revenue histogram, bucketed fifty at a time -----------------------------------


@query("q54", order_by=("SEGMENT", "num_customers", "segment_base"))
def plan_q54(t):
    """Customers who bought maternity wear, bucketed by what they spent locally after.

    Three things in one query. The customers come from a `UNION ALL` of two channels, made
    `DISTINCT` — an interleave and a keyless-aggregate-with-keys. The three months after
    December 1998 are named by a **scalar subquery** with arithmetic on it, so the range test
    is a nested-loop join carrying `BETWEEN base+1 AND base+3` as its predicate; there is no
    equi-key to hash on. And the bucket is `cast(round(revenue/50) as int)`, which is where
    the IR needs `round` — half away from zero, as DataFusion and `cudf::round` have it,
    which decides the bucket for every customer whose revenue lands on a boundary.
    """
    buyers = []
    for tag, table, prefix, customer_key in (
        ("cs", "catalog_sales", "cs", "cs_bill_customer_sk"),
        ("ws", "web_sales", "ws", "ws_bill_customer_sk"),
    ):
        sales = fact(t, table, [f"{prefix}_sold_date_sk", f"{prefix}_item_sk", customer_key],
                     tag=f"q54_{tag}_{table}")
        buyers.append(rename(f"q54_{tag}_shape", sales[1],
                             [(f"{prefix}_sold_date_sk", "sold_date_sk"),
                              (f"{prefix}_item_sk", "item_sk"),
                              (customer_key, "customer_sk")]))
    item = dim(t, "item", ["i_item_sk", "i_category", "i_class"],
               all_of(Binary("==", Col("i_category"), Lit("Women")),
                      Binary("==", Col("i_class"), Lit("maternity"))), tag="q54_item")
    december = dim(t, "date_dim", ["d_date_sk", "d_moy", "d_year"],
                   all_of(Binary("==", Col("d_moy"), Lit(12)),
                          Binary("==", Col("d_year"), Lit(1998))), tag="q54_december")
    customer = dim(t, "customer", ["c_customer_sk", "c_current_addr_sk"], rows=250_000,
                   tag="q54_customer")
    bought = star(
        N.interleave("q54_channels", buyers),
        ("q54_d", december, "d_date_sk", "sold_date_sk"),
        ("q54_i", item, "i_item_sk", "item_sk"),
        ("q54_c", customer, "c_customer_sk", "customer_sk"),
    )
    my_customers = aggregate_by(
        "q54_my_customers",
        select("q54_customer_keys", bought, "c_customer_sk", "c_current_addr_sk"),
        ["c_customer_sk", "c_current_addr_sk"], [],
        schema_frame=corpus.schema_of(customer[0]),
    )
    base = aggregate_by(
        "q54_base_month",
        select("q54_base_keys",
               dim(t, "date_dim", ["d_month_seq", "d_moy", "d_year"],
                   all_of(Binary("==", Col("d_moy"), Lit(12)),
                          Binary("==", Col("d_year"), Lit(1998))), tag="q54_base")[1],
               "d_month_seq"),
        ["d_month_seq"], [], schema_frame=corpus.schema_of(d_month_seq="int64"),
    )
    base = rename("q54_base_alias", base, [("d_month_seq", "base_month_seq")])
    calendar = dim(t, "date_dim", ["d_date_sk", "d_month_seq"], tag="q54_calendar")
    in_window = N.nested_loop_join(
        "q54_month_window", N.coalesce_all("q54_base_all", base), calendar[1],
        JoinType.INNER,
        between(Col("d_month_seq"), Binary("+", Col("base_month_seq"), Lit(1)),
                Binary("+", Col("base_month_seq"), Lit(3))),
    )
    store_sales = fact(t, "store_sales",
                       ["ss_sold_date_sk", "ss_customer_sk", "ss_addr_sk",
                        "ss_store_sk", "ss_ext_sales_price"], tag="q54_store_sales")
    address = dim(t, "customer_address", ["ca_address_sk", "ca_county", "ca_state"],
                  tag="q54_customer_address")
    store = dim(t, "store", ["s_store_sk", "s_county", "s_state"], tag="q54_store")
    local = star(
        store_sales[1],
        ("q54_window", (corpus.schema_of(calendar[0], base_month_seq="int64"), in_window),
         "d_date_sk", "ss_sold_date_sk"),
    )
    with_customers = N.hash_join(
        "q54_mine",
        N.coalesce_all("q54_my_customers_all", my_customers,
                       schema=dict(corpus.schema_of(customer[0]).dtypes)),
        local, JoinType.INNER, ["c_customer_sk"], ["ss_customer_sk"],
    )
    with_places = star(
        with_customers,
        ("q54_ca", address, "ca_address_sk", "c_current_addr_sk"),
    )
    with_store = N.hash_join(
        "q54_s",
        N.coalesce_all("q54_store_all", store[1], schema=dict(store[0].dtypes)),
        with_places, JoinType.INNER, ["s_county", "s_state"], ["ca_county", "ca_state"],
    )
    revenue = aggregate_by(
        "q54_revenue",
        select("q54_revenue_keys", with_store, "c_customer_sk", "ss_ext_sales_price"),
        ["c_customer_sk"], [A.Agg(A.SUM, "ss_ext_sales_price", "revenue")],
        schema_frame=corpus.schema_of(customer[0], store_sales[0]),
    )
    segments = N.project(
        "q54_segments", revenue,
        [Alias(Cast(Round(Binary("/", Col("revenue"), Lit(50.0))), "int64"), "SEGMENT")],
    )
    counted = aggregate_by(
        "q54_agg", segments, ["SEGMENT"], [A.Agg(A.COUNT, None, "num_customers")],
        schema_frame=corpus.schema_of(SEGMENT="int64"),
    )
    out = N.project(
        "out", counted,
        [Alias(Col("SEGMENT"), "SEGMENT"), Alias(Col("num_customers"), "num_customers"),
         Alias(Binary("*", Col("SEGMENT"), Lit(50)), "segment_base")],
    )
    return N.unload(
        "unload",
        sorted_output("sort", out, ["SEGMENT", "num_customers", "segment_base"],
                      [True, True, True], fetch=100, nulls_first=True),
    )


# -- a threshold taken from the whole table, applied to a group ----------------------


@query("q23", order_by=("c_last_name", "c_first_name", "sales"))
def plan_q23(t):
    """What the best store customers bought from the catalog and the web in one month.

    Three CTEs, each a different way of being a subquery. `frequent_ss_items` is a grouped
    aggregate with a `HAVING`, joined on the item — and it has more than one row per item, so
    that join fans out and the query means it. `max_store_sales` is a **scalar**: a maximum
    over an aggregate, cross-joined into `best_ss_customer` before its own `HAVING` can
    compare against it. And `best_ss_customer` is then a build side of the two channel
    branches, which are unioned.
    """
    years = (2000, 2001, 2002, 2003)

    def store_sales_in_years(tag, columns):
        sales = fact(t, "store_sales", ["ss_sold_date_sk"] + list(columns),
                     tag=f"{tag}_store_sales")
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_year", "d_date"],
                       is_in(Col("d_year"), years), tag=f"{tag}_date_dim")
        return sales, star(sales[1], (f"{tag}_d", date_dim, "d_date_sk", "ss_sold_date_sk"))

    sales, in_years = store_sales_in_years("q23_freq", ["ss_item_sk"])
    item = dim(t, "item", ["i_item_sk", "i_item_desc"], tag="q23_item")
    described = N.project(
        "q23_itemdesc",
        star(in_years, ("q23_i", item, "i_item_sk", "ss_item_sk")),
        [Alias(Substring(Col("i_item_desc"), 1, 30), "itemdesc"),
         Alias(Col("i_item_sk"), "item_sk"), Alias(Col("d_date"), "solddate")],
    )
    counted = aggregate_by(
        "q23_frequent", described, ["itemdesc", "item_sk", "solddate"],
        [A.Agg(A.COUNT, None, "cnt")],
        schema_frame=corpus.schema_of(itemdesc="object", item_sk="int64",
                                      solddate="datetime64[ns]"),
    )
    frequent = select("q23_frequent_keys",
                      N.filter_("q23_more_than_four", counted, Binary(">", Col("cnt"), Lit(4))),
                      "item_sk")

    def per_customer(tag, years_only):
        """What each customer spent in store. `years_only` is the difference between the two
        uses of it, and it is easy to miss: `max_store_sales` looks at the four years, and
        `best_ss_customer` at **all** of store_sales. Filtering both to the four years made
        every customer's total smaller, so none cleared half the maximum and the query
        answered nothing."""
        columns = ["ss_customer_sk", "ss_quantity", "ss_sales_price"]
        if years_only:
            sales, source = store_sales_in_years(tag, columns)
        else:
            sales = fact(t, "store_sales", columns, tag=f"{tag}_store_sales")
            source = sales[1]
        customer = dim(t, "customer", ["c_customer_sk"], rows=250_000, tag=f"{tag}_customer")
        joined = star(source, (f"{tag}_c", customer, "c_customer_sk", "ss_customer_sk"))
        spent = N.project(
            f"{tag}_spent", joined,
            [Alias(Col("c_customer_sk"), "c_customer_sk"),
             Alias(Binary("*", Col("ss_quantity"), Col("ss_sales_price")), "spent")],
        )
        return aggregate_by(f"{tag}_csales", spent, ["c_customer_sk"],
                            [A.Agg(A.SUM, "spent", "csales")],
                            schema_frame=corpus.schema_of(c_customer_sk="int64",
                                                          spent="float64"))

    biggest = aggregate_to_one_row(
        "q23_cmax", select("q23_cmax_keys", per_customer("q23_max", True), "csales"),
        [A.Agg(A.MAX, "csales", "tpcds_cmax")], corpus.schema_of(csales="float64"),
    )
    threshold = N.project("q23_threshold", biggest,
                          [Alias(Binary("*", Lit(0.5), Col("tpcds_cmax")), "threshold")])
    against = N.cross_join("q23_against", N.coalesce_all("q23_threshold_all", threshold),
                           per_customer("q23_best", False))
    # Renamed, because the branches below join `customer` as well and a batch cannot hold
    # two columns called `c_customer_sk`.
    best = rename("q23_best_keys",
                  N.filter_("q23_best_customers", against,
                            Binary(">", Col("csales"), Col("threshold"))),
                  [("c_customer_sk", "best_customer_sk")])

    branches = []
    for tag, table, prefix, item_key, customer_key, quantity, price in (
        ("cs", "catalog_sales", "cs", "cs_item_sk", "cs_bill_customer_sk", "cs_quantity",
         "cs_list_price"),
        ("ws", "web_sales", "ws", "ws_item_sk", "ws_bill_customer_sk", "ws_quantity",
         "ws_list_price"),
    ):
        channel = fact(t, table, [f"{prefix}_sold_date_sk", item_key, customer_key, quantity,
                                  price], tag=f"q23_{tag}_{table}")
        february = dim(t, "date_dim", ["d_date_sk", "d_year", "d_moy"],
                       all_of(Binary("==", Col("d_year"), Lit(2000)),
                              Binary("==", Col("d_moy"), Lit(2))), tag=f"q23_{tag}_date_dim")
        customer = dim(t, "customer", ["c_customer_sk", "c_first_name", "c_last_name"],
                       rows=250_000, tag=f"q23_{tag}_customer")
        joined = star(channel[1], (f"q23_{tag}_d", february, "d_date_sk",
                                   f"{prefix}_sold_date_sk"))
        with_items = N.hash_join(
            f"q23_{tag}_frequent",
            N.coalesce_all(f"q23_{tag}_frequent_all", frequent,
                           schema={"item_sk": "int64"}),
            joined, JoinType.INNER, ["item_sk"], [item_key],
        )
        with_best = N.hash_join(
            f"q23_{tag}_best",
            N.coalesce_all(f"q23_{tag}_best_all", best,
                           schema={"best_customer_sk": "int64"}),
            with_items, JoinType.INNER, ["best_customer_sk"], [customer_key],
        )
        named = star(with_best, (f"q23_{tag}_c", customer, "c_customer_sk", customer_key))
        spent = N.project(
            f"q23_{tag}_spent", named,
            [Alias(Col("c_last_name"), "c_last_name"), Alias(Col("c_first_name"), "c_first_name"),
             Alias(Binary("*", Col(quantity), Col(price)), "spent")],
        )
        branches.append(aggregate_by(
            f"q23_{tag}_agg", spent, ["c_last_name", "c_first_name"],
            [A.Agg(A.SUM, "spent", "sales")],
            schema_frame=corpus.schema_of(customer[0], spent="float64"),
        ))
    combined = N.union("q23_union", branches)
    return N.unload(
        "unload",
        sorted_output("sort", combined, ["c_last_name", "c_first_name", "sales"],
                      [True, True, True], fetch=100, nulls_first=True),
    )


# -- eighteen tables, and then a self-join ------------------------------------------


#: q64's group key: everything that identifies the product, the store, and the two addresses.
_Q64_KEYS = ("product_name", "item_sk", "store_name", "store_zip",
             "b_street_number", "b_street_name", "b_city", "b_zip",
             "c_street_number", "c_street_name", "c_city", "c_zip",
             "syear", "fsyear", "s2year")
#: The four of those that are not strings.
_Q64_INTEGER_KEYS = frozenset({"item_sk", "syear", "fsyear", "s2year"})


def _q64_cross_sales(t, copy):
    """q64's `cross_sales`: the largest join in the benchmark, eighteen tables wide.

    Every dimension the sale reaches is joined twice — once from the *sale* and once from the
    *customer's current record* — so customer_demographics, household_demographics,
    income_band and customer_address each appear in two copies, and the second copy of each
    has to be renamed before it arrives. That renaming is not cosmetic: `cd1.cd_marital_status
    <> cd2.cd_marital_status` is a predicate over both copies, and a batch that held one
    column called `cd_marital_status` could not express it.

    store_sales is the probe from beginning to end. Everything else is a build side, which
    includes two 1.9M-row copies of customer_demographics — narrowed to two columns first,
    because that is what makes them affordable.
    """
    store_sales = fact(t, "store_sales",
                       ["ss_item_sk", "ss_ticket_number", "ss_store_sk", "ss_sold_date_sk",
                        "ss_customer_sk", "ss_cdemo_sk", "ss_hdemo_sk", "ss_addr_sk",
                        "ss_promo_sk", "ss_wholesale_cost", "ss_list_price",
                        "ss_coupon_amt"],
                       tag=f"{copy}_store_sales")
    returns = dim(t, "store_returns", ["sr_item_sk", "sr_ticket_number"], rows=250_000,
                  tag=f"{copy}_store_returns")
    returned = N.hash_join(
        f"{copy}_sr_ss",
        N.coalesce_all(f"{copy}_sr_ss_build", returns[1], schema=dict(returns[0].dtypes)),
        store_sales[1], JoinType.INNER,
        ["sr_item_sk", "sr_ticket_number"], ["ss_item_sk", "ss_ticket_number"],
    )
    catalog_sales = fact(t, "catalog_sales",
                         ["cs_item_sk", "cs_order_number", "cs_ext_list_price"],
                         tag=f"{copy}_catalog_sales")
    catalog_returns = dim(t, "catalog_returns",
                          ["cr_item_sk", "cr_order_number", "cr_refunded_cash",
                           "cr_reversed_charge", "cr_store_credit"],
                          rows=250_000, tag=f"{copy}_catalog_returns")
    refunded = N.hash_join(
        f"{copy}_cr_cs",
        N.coalesce_all(f"{copy}_cr_cs_build", catalog_returns[1],
                       schema=dict(catalog_returns[0].dtypes)),
        catalog_sales[1], JoinType.INNER,
        ["cr_item_sk", "cr_order_number"], ["cs_item_sk", "cs_order_number"],
    )
    refund = Binary("+", Binary("+", Col("cr_refunded_cash"), Col("cr_reversed_charge")),
                    Col("cr_store_credit"))
    cs_ui = aggregate_by(
        f"{copy}_cs_ui",
        N.project(f"{copy}_cs_ui_measures", refunded,
                  [Alias(Col("cs_item_sk"), "cs_item_sk"),
                   Alias(Col("cs_ext_list_price"), "sale"), Alias(refund, "refund")]),
        ["cs_item_sk"], [A.Agg(A.SUM, "sale", "sale"), A.Agg(A.SUM, "refund", "refund")],
        schema_frame=corpus.schema_of(cs_item_sk="int64", sale="float64", refund="float64"),
    )
    profitable = select(
        f"{copy}_cs_ui_keys",
        N.filter_(f"{copy}_cs_ui_having", cs_ui,
                  Binary(">", Col("sale"), Binary("*", Lit(2.0), Col("refund")))),
        "cs_item_sk",
    )
    item = dim(t, "item", ["i_item_sk", "i_product_name", "i_color", "i_current_price"],
               all_of(is_in(Col("i_color"),
                            ("purple", "burlywood", "indian", "spring", "floral", "medium")),
                      between(Col("i_current_price"), Lit(64), Lit(74)),
                      between(Col("i_current_price"), Lit(65), Lit(79))),
               tag=f"{copy}_item")
    store = dim(t, "store", ["s_store_sk", "s_store_name", "s_zip"], tag=f"{copy}_store")
    d1 = dim(t, "date_dim", ["d_date_sk", "d_year"], tag=f"{copy}_d1")
    promotion = dim(t, "promotion", ["p_promo_sk"], tag=f"{copy}_promotion")
    cd1 = fact(t, "customer_demographics", ["cd_demo_sk", "cd_marital_status"],
               tag=f"{copy}_cd1")
    hd1 = dim(t, "household_demographics", ["hd_demo_sk", "hd_income_band_sk"],
              tag=f"{copy}_hd1")
    ib1 = dim(t, "income_band", ["ib_income_band_sk"], tag=f"{copy}_ib1")
    ad1 = dim(t, "customer_address",
              ["ca_address_sk", "ca_street_number", "ca_street_name", "ca_city", "ca_zip"],
              tag=f"{copy}_ad1")
    customer = dim(t, "customer",
                   ["c_customer_sk", "c_current_cdemo_sk", "c_current_hdemo_sk",
                    "c_current_addr_sk", "c_first_sales_date_sk", "c_first_shipto_date_sk"],
                   rows=250_000, tag=f"{copy}_customer")
    sold = star(
        returned,
        (f"{copy}_s_ss", store, "s_store_sk", "ss_store_sk"),
        (f"{copy}_d1_ss", d1, "d_date_sk", "ss_sold_date_sk"),
        (f"{copy}_i_ss", item, "i_item_sk", "ss_item_sk"),
        (f"{copy}_cd1_ss", cd1, "cd_demo_sk", "ss_cdemo_sk"),
        (f"{copy}_hd1_ss", hd1, "hd_demo_sk", "ss_hdemo_sk"),
        (f"{copy}_ib1_hd1", ib1, "ib_income_band_sk", "hd_income_band_sk"),
        (f"{copy}_ad1_ss", ad1, "ca_address_sk", "ss_addr_sk"),
        (f"{copy}_p_ss", promotion, "p_promo_sk", "ss_promo_sk"),
        (f"{copy}_c_ss", customer, "c_customer_sk", "ss_customer_sk"),
    )
    with_catalog = N.hash_join(
        f"{copy}_cs_ui_ss",
        N.coalesce_all(f"{copy}_cs_ui_all", profitable, schema={"cs_item_sk": "int64"}),
        sold, JoinType.INNER, ["cs_item_sk"], ["ss_item_sk"],
    )
    # The customer's own side of the star: the same four dimensions again, renamed.
    cd2 = (cd1[0], rename(f"{copy}_cd2", fact(t, "customer_demographics",
                                              ["cd_demo_sk", "cd_marital_status"],
                                              tag=f"{copy}_cd2_scan")[1],
                          [("cd_demo_sk", "cd2_demo_sk"),
                           ("cd_marital_status", "cd2_marital_status")]))
    hd2 = (hd1[0], rename(f"{copy}_hd2",
                          dim(t, "household_demographics",
                              ["hd_demo_sk", "hd_income_band_sk"],
                              tag=f"{copy}_hd2_scan")[1],
                          [("hd_demo_sk", "hd2_demo_sk"),
                           ("hd_income_band_sk", "hd2_income_band_sk")]))
    ib2 = (ib1[0], rename(f"{copy}_ib2",
                          dim(t, "income_band", ["ib_income_band_sk"],
                              tag=f"{copy}_ib2_scan")[1],
                          [("ib_income_band_sk", "ib2_income_band_sk")]))
    ad2 = (ad1[0], rename(f"{copy}_ad2",
                          dim(t, "customer_address",
                              ["ca_address_sk", "ca_street_number", "ca_street_name",
                               "ca_city", "ca_zip"], tag=f"{copy}_ad2_scan")[1],
                          [("ca_address_sk", "ca2_address_sk"),
                           ("ca_street_number", "ca2_street_number"),
                           ("ca_street_name", "ca2_street_name"),
                           ("ca_city", "ca2_city"), ("ca_zip", "ca2_zip")]))
    d2 = (None, rename(f"{copy}_d2",
                       dim(t, "date_dim", ["d_date_sk", "d_year"], tag=f"{copy}_d2_scan")[1],
                       [("d_date_sk", "d2_date_sk"), ("d_year", "d2_year")]))
    d3 = (None, rename(f"{copy}_d3",
                       dim(t, "date_dim", ["d_date_sk", "d_year"], tag=f"{copy}_d3_scan")[1],
                       [("d_date_sk", "d3_date_sk"), ("d_year", "d3_year")]))
    year_schema = corpus.schema_of(d2_date_sk="int64", d2_year="int64")
    joined = star(
        with_catalog,
        (f"{copy}_cd2_c", cd2, "cd2_demo_sk", "c_current_cdemo_sk"),
        (f"{copy}_hd2_c", hd2, "hd2_demo_sk", "c_current_hdemo_sk"),
        (f"{copy}_ib2_hd2", ib2, "ib2_income_band_sk", "hd2_income_band_sk"),
        (f"{copy}_ad2_c", ad2, "ca2_address_sk", "c_current_addr_sk"),
        (f"{copy}_d2_c", (year_schema, d2[1]), "d2_date_sk", "c_first_sales_date_sk"),
        (f"{copy}_d3_c", (corpus.schema_of(d3_date_sk="int64", d3_year="int64"), d3[1]),
         "d3_date_sk", "c_first_shipto_date_sk"),
    )
    mixed = N.filter_(f"{copy}_different_status", joined,
                      Binary("!=", Col("cd_marital_status"), Col("cd2_marital_status")))
    named = rename(
        f"{copy}_keys", mixed,
        [("i_product_name", "product_name"), ("i_item_sk", "item_sk"),
         ("s_store_name", "store_name"), ("s_zip", "store_zip"),
         ("ca_street_number", "b_street_number"), ("ca_street_name", "b_street_name"),
         ("ca_city", "b_city"), ("ca_zip", "b_zip"),
         ("ca2_street_number", "c_street_number"), ("ca2_street_name", "c_street_name"),
         ("ca2_city", "c_city"), ("ca2_zip", "c_zip"),
         ("d_year", "syear"), ("d2_year", "fsyear"), ("d3_year", "s2year"),
         ("ss_wholesale_cost", "ss_wholesale_cost"), ("ss_list_price", "ss_list_price"),
         ("ss_coupon_amt", "ss_coupon_amt")],
    )
    return aggregate_by(
        f"{copy}_cross_sales", named, list(_Q64_KEYS),
        [A.Agg(A.COUNT, None, "cnt"), A.Agg(A.SUM, "ss_wholesale_cost", "s1"),
         A.Agg(A.SUM, "ss_list_price", "s2"), A.Agg(A.SUM, "ss_coupon_amt", "s3")],
        schema_frame=corpus.schema_of(
            **{key: "int64" if key in _Q64_INTEGER_KEYS else "object" for key in _Q64_KEYS},
            ss_wholesale_cost="float64", ss_list_price="float64", ss_coupon_amt="float64"),
    )


@query("q64", order_by=("product_name", "store_name", "cnt", "s11", "s12"))
def plan_q64(t):
    """The same product sold at the same store in two consecutive years.

    `cross_sales` is built twice and joined to itself on the item, the store name and the
    store zip, one alias per year. Everything expensive is below that join; the join itself
    is a few thousand rows against a few thousand.
    """
    first = rename(
        "q64_cs1", N.filter_("q64_cs1_year", _q64_cross_sales(t, "q64_a"),
                             Binary("==", Col("syear"), Lit(1999))),
        [(key, key) for key in _Q64_KEYS]
        + [("cnt", "cs1cnt"), ("s1", "s11"), ("s2", "s21"), ("s3", "s31")],
    )
    second = rename(
        "q64_cs2", N.filter_("q64_cs2_year", _q64_cross_sales(t, "q64_b"),
                             Binary("==", Col("syear"), Lit(2000))),
        [("item_sk", "cs2_item_sk"), ("store_name", "cs2_store_name"),
         ("store_zip", "cs2_store_zip"), ("syear", "cs2_syear"), ("cnt", "cs2cnt"),
         ("s1", "s12"), ("s2", "s22"), ("s3", "s32")],
    )
    joined = N.hash_join(
        "q64_cs1_cs2",
        N.coalesce_all("q64_cs2_all", second,
                       schema={"cs2_item_sk": "int64", "cs2_store_name": "object",
                               "cs2_store_zip": "object", "cs2_syear": "int64",
                               "cs2cnt": "int64", "s12": "float64", "s22": "float64",
                               "s32": "float64"}),
        first, JoinType.INNER,
        ["cs2_item_sk", "cs2_store_name", "cs2_store_zip"],
        ["item_sk", "store_name", "store_zip"],
    )
    kept = N.filter_("q64_no_more_than_before", joined,
                     Binary("<=", Col("cs2cnt"), Col("cs1cnt")))
    out = rename(
        "out", kept,
        [("product_name", "product_name"), ("store_name", "store_name"),
         ("store_zip", "store_zip"), ("b_street_number", "b_street_number"),
         ("b_street_name", "b_street_name"), ("b_city", "b_city"), ("b_zip", "b_zip"),
         ("c_street_number", "c_street_number"), ("c_street_name", "c_street_name"),
         ("c_city", "c_city"), ("c_zip", "c_zip"), ("syear", "cs1syear"),
         ("cs1cnt", "cs1cnt"), ("s11", "s11"), ("s21", "s21"), ("s31", "s31"),
         ("s12", "s12"), ("s22", "s22"), ("s32", "s32"), ("cs2_syear", "syear"),
         ("cs2cnt", "cnt")],
    )
    return N.unload(
        "unload",
        sorted_output("sort", out, ["product_name", "store_name", "cnt", "s11", "s12"],
                      [True, True, True, True, True]),
    )
