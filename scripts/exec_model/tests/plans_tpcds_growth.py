"""TPC-DS: year over year — a CTE aggregated once per alias and joined to itself.

`WITH year_total AS (…) SELECT … FROM year_total a, year_total b WHERE a.customer = b.customer
AND a.year = 2001 AND b.year = 2002` is four or six copies of one aggregate, joined on the
customer. A plan is a tree, so each alias is its own subtree — the same thing DataFusion does
with a repeated CTE (#101), and the reason these are the most expensive lowerings in the
corpus.

Two things make them tractable. The year and the sale type each alias filters on are
**constants**, so they push down into that alias's own scan: `t_w_firstyear` never reads
store_sales and never reads 2002. And the aggregate is per customer, so what comes out of
each alias is tens of thousands of rows, small enough that every alias but one is a build
side.

Every column but the join key has to be renamed on the way out — four aliases of one CTE
have four `year_total`s, and a batch holds one column of each name.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from . import corpus
from .plan_helpers import (
    aggregate_by, all_of, between, is_in, rename, select, sorted_output,
)
from .plans_tpcds_common import dim, fact, registry, star
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import Alias, Binary, Case, Coalesce, Col, Lit
from ..operators.join_types import JoinType

QUERIES, ORDER_BY, query = registry()

#: The customer columns `year_total` groups on, in the CTE's own names.
_CUSTOMER = (("c_customer_id", "customer_id"), ("c_first_name", "customer_first_name"),
             ("c_last_name", "customer_last_name"),
             ("c_preferred_cust_flag", "customer_preferred_cust_flag"),
             ("c_birth_country", "customer_birth_country"), ("c_login", "customer_login"),
             ("c_email_address", "customer_email_address"))

#: The three channels' `(prefix, table, the customer key each one bills to)`.
_CHANNELS = {"s": ("ss", "store_sales", "ss_customer_sk"),
             "c": ("cs", "catalog_sales", "cs_bill_customer_sk"),
             "w": ("ws", "web_sales", "ws_bill_customer_sk")}


def _year_total(t, tag, sale_type, year, measure, columns, keys=_CUSTOMER):
    """One alias of `year_total`: one channel, one year, summed per customer.

    The sale type and the year are constants in the outer query, so both are pushed into
    this alias — the type by only building the branch it selects, the year by filtering
    date_dim. That pruning is why four aliases cost four scans and not eight.
    """
    prefix, table, customer_key = _CHANNELS[sale_type]
    sales = fact(t, table, [customer_key, f"{prefix}_sold_date_sk"] + list(columns),
                 tag=f"{tag}_{table}")
    customer = dim(t, "customer", ["c_customer_sk"] + [source for source, _ in keys],
                   rows=250_000, tag=f"{tag}_customer")
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"],
                   Binary("==", Col("d_year"), Lit(year)), tag=f"{tag}_date_dim")
    joined = star(
        sales[1],
        (f"{tag}_d", date_dim, "d_date_sk", f"{prefix}_sold_date_sk"),
        (f"{tag}_c", customer, "c_customer_sk", customer_key),
    )
    named = N.project(
        f"{tag}_measure", joined,
        [Alias(Col(source), target) for source, target in keys]
        + [Alias(measure(prefix), "measure")],
    )
    group = [target for _, target in keys]
    return aggregate_by(
        f"{tag}_total", named, group, [A.Agg(A.SUM, "measure", "year_total")],
        schema_frame=corpus.schema_of(
            **{target: "object" for _, target in keys}, measure="float64"),
    )


def _renamed_alias(tag, node, keep=("customer_id", "year_total")):
    """An alias's two useful columns under names no other alias uses."""
    return rename(f"{tag}_alias", node, [(column, f"{tag}_{column}") for column in keep])


def _ratio(numerator, denominator, otherwise):
    """`CASE WHEN denominator > 0 THEN numerator/denominator ELSE … END`, as written."""
    return Case(whens=((Binary(">", Col(denominator), Lit(0)),
                        Binary("/", Col(numerator), Col(denominator))),),
                otherwise=otherwise)


def _grew_faster(t, tag, first_year, faster, slower, reported, measure, output_columns,
                 otherwise):
    """The customers whose growth in `faster` beat their growth in every channel of `slower`.

    `reported` names the alias whose columns the query returns; it is the probe, and every
    other alias is a build side joined on the customer id. The channels are exactly
    `{faster} | slower`, each of them contributing a first-year and a second-year alias, and
    each first year has to be positive for its ratio to be defined.
    """
    channels = (faster,) + tuple(slower)
    keys = _CUSTOMER if len(output_columns) > 3 else _CUSTOMER[:3]
    aliases = {}
    for sale_type in channels:
        for label, year in (("firstyear", first_year), ("secyear", first_year + 1)):
            name = f"t_{sale_type}_{label}"
            aliases[name] = _year_total(t, f"{tag}_{name}", sale_type, year, measure,
                                        _MEASURE_COLUMNS[sale_type], keys)
    reported_alias = f"t_{reported}_secyear"
    probe = rename(
        f"{tag}_probe",
        select(f"{tag}_reported", aliases[reported_alias], *output_columns, "year_total"),
        [(column, column) for column in output_columns]
        + [("year_total", f"{reported_alias}_year_total")],
    )
    joined = probe
    for name, node in aliases.items():
        if name == reported_alias:
            continue
        joined = N.hash_join(
            f"{tag}_{name}_join",
            N.coalesce_all(f"{tag}_{name}_build", _renamed_alias(name, node),
                           schema={f"{name}_customer_id": "object",
                                   f"{name}_year_total": "float64"}),
            joined, JoinType.INNER, [f"{name}_customer_id"], ["customer_id"],
        )
    positive = all_of(*[Binary(">", Col(f"t_{sale_type}_firstyear_year_total"), Lit(0))
                        for sale_type in channels])
    ahead = _ratio(f"t_{faster}_secyear_year_total", f"t_{faster}_firstyear_year_total",
                   otherwise)
    beats = all_of(*[
        Binary(">", ahead, _ratio(f"t_{sale_type}_secyear_year_total",
                                  f"t_{sale_type}_firstyear_year_total", otherwise))
        for sale_type in slower
    ])
    kept = N.filter_(f"{tag}_grew_faster", joined, Binary("and", positive, beats))
    out = select(f"{tag}_out", kept, *output_columns)
    return N.unload(
        "unload",
        sorted_output("sort", out, list(output_columns), [True] * len(output_columns),
                      fetch=100, nulls_first=True),
    )


#: The fact columns each channel's measure reads. Keyed by sale type, as the query is.
_MEASURE_COLUMNS = {
    "s": ("ss_ext_list_price", "ss_ext_wholesale_cost", "ss_ext_discount_amt",
          "ss_ext_sales_price", "ss_net_paid"),
    "c": ("cs_ext_list_price", "cs_ext_wholesale_cost", "cs_ext_discount_amt",
          "cs_ext_sales_price", "cs_net_paid"),
    "w": ("ws_ext_list_price", "ws_ext_wholesale_cost", "ws_ext_discount_amt",
          "ws_ext_sales_price", "ws_net_paid"),
}

_REPORTED = ("customer_id", "customer_first_name", "customer_last_name",
             "customer_preferred_cust_flag")


@query("q4", order_by=_REPORTED)
def plan_q4(t):
    """Customers whose catalog spending grew faster than both their store and their web
    spending. Six aliases of one CTE over three channels — the largest self-join in the
    corpus, and the reason the aliases prune themselves down to one channel and one year."""
    def measure(prefix):
        return Binary(
            "/",
            Binary("+", Binary("-", Binary("-", Col(f"{prefix}_ext_list_price"),
                                           Col(f"{prefix}_ext_wholesale_cost")),
                               Col(f"{prefix}_ext_discount_amt")),
                   Col(f"{prefix}_ext_sales_price")),
            Lit(2.0),
        )

    return _grew_faster(t, "q4", 2001, "c", ("s", "w"), "s", measure, _REPORTED,
                        otherwise=Lit(float("nan")))


@query("q11", order_by=_REPORTED)
def plan_q11(t):
    """q4 over two channels and a simpler measure, and with `ELSE 0.0` where q4 says `ELSE
    NULL` — a difference that decides what happens to a customer who bought nothing in the
    first year, and the reason both spellings are in the benchmark."""
    def measure(prefix):
        return Binary("-", Col(f"{prefix}_ext_list_price"), Col(f"{prefix}_ext_discount_amt"))

    return _grew_faster(t, "q11", 2001, "w", ("s",), "s", measure, _REPORTED,
                        otherwise=Lit(0.0))


@query("q74", order_by=("customer_id",))
def plan_q74(t):
    """q11 over net paid rather than list price, reporting three columns.

    The CTE groups on three customer columns here rather than seven, which changes nothing
    about the lowering and everything about how wide the build sides are.
    """
    def measure(prefix):
        return Col(f"{prefix}_net_paid")

    return _grew_faster(t, "q74", 2001, "w", ("s",), "s", measure, _REPORTED[:3],
                        otherwise=Lit(float("nan")))


# -- growth by county, quarter over quarter -----------------------------------------


@query("q31", order_by=("ca_county",))
def plan_q31(t):
    """Counties where web sales grew faster than store sales in two consecutive quarters.

    Six aliases again, but of two CTEs rather than one, and keyed on the county rather than
    the customer — so each alias is a few hundred rows and the whole six-way join is
    trivial once the aggregates are done. The quarter and the year each alias wants are
    constants, so both push into that alias's date_dim; without that the plan would scan
    store_sales three times over its whole history instead of over one quarter.
    """
    def quarter(tag, table, prefix, addr_key, measure, qoy):
        sales = fact(t, table, [f"{prefix}_sold_date_sk", addr_key, measure],
                     tag=f"{tag}_{table}")
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_qoy", "d_year"],
                       all_of(Binary("==", Col("d_qoy"), Lit(qoy)),
                              Binary("==", Col("d_year"), Lit(2000))),
                       tag=f"{tag}_date_dim")
        address = dim(t, "customer_address", ["ca_address_sk", "ca_county"],
                      tag=f"{tag}_customer_address")
        joined = star(
            sales[1],
            (f"{tag}_d", date_dim, "d_date_sk", f"{prefix}_sold_date_sk"),
            (f"{tag}_ca", address, "ca_address_sk", addr_key),
        )
        totals = aggregate_by(
            f"{tag}_agg", select(f"{tag}_keys", joined, "ca_county", measure), ["ca_county"],
            [A.Agg(A.SUM, measure, "total")],
            schema_frame=corpus.schema_of(address[0], **{measure: "float64"}),
        )
        return rename(f"{tag}_alias", totals,
                      [("ca_county", f"{tag}_county"), ("total", f"{tag}_total")])

    aliases = {}
    for channel, table, prefix, addr_key, measure in (
        ("ss", "store_sales", "ss", "ss_addr_sk", "ss_ext_sales_price"),
        ("ws", "web_sales", "ws", "ws_bill_addr_sk", "ws_ext_sales_price"),
    ):
        for qoy in (1, 2, 3):
            aliases[f"{channel}{qoy}"] = quarter(f"q31_{channel}{qoy}", table, prefix,
                                                 addr_key, measure, qoy)
    joined = aliases["ss1"]
    for name in ("ss2", "ss3", "ws1", "ws2", "ws3"):
        joined = N.hash_join(
            f"q31_{name}_join",
            N.coalesce_all(f"q31_{name}_build", aliases[name],
                           schema={f"q31_{name}_county": "object",
                                   f"q31_{name}_total": "float64"}),
            joined, JoinType.INNER, [f"q31_{name}_county"], ["q31_ss1_county"],
        )

    def growth(later, earlier):
        return _ratio(f"q31_{later}_total", f"q31_{earlier}_total", Lit(float("nan")))

    kept = N.filter_(
        "q31_grew_faster", joined,
        Binary("and", Binary(">", growth("ws2", "ws1"), growth("ss2", "ss1")),
               Binary(">", growth("ws3", "ws2"), growth("ss3", "ss2"))),
    )
    out = N.project(
        "out", kept,
        [Alias(Col("q31_ss1_county"), "ca_county"), Alias(Lit(2000), "d_year"),
         Alias(Binary("/", Col("q31_ws2_total"), Col("q31_ws1_total")),
               "web_q1_q2_increase"),
         Alias(Binary("/", Col("q31_ss2_total"), Col("q31_ss1_total")),
               "store_q1_q2_increase"),
         Alias(Binary("/", Col("q31_ws3_total"), Col("q31_ws2_total")),
               "web_q2_q3_increase"),
         Alias(Binary("/", Col("q31_ss3_total"), Col("q31_ss2_total")),
               "store_q2_q3_increase")],
    )
    return N.unload("unload", sorted_output("sort", out, ["ca_county"], [True]))


# -- one week against the same week a year earlier ----------------------------------


_DAYS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
_DAY_SALES = tuple(f"{day[:3].lower()}_sales" for day in _DAYS)


@query("q59", order_by=("s_store_name1", "s_store_id1", "d_week_seq1"))
def plan_q59(t):
    """Each store's week against the same week of the year before, day by day.

    `wss` is q43's seven conditional sums, keyed by week and store instead of by store
    alone, and it is aliased twice — one year's worth of weeks and the next. The join
    between them is on the store *and* on `week = week - 52`, so one side carries a shifted
    key computed in a projection: a join takes column ordinals, and `d_week_seq2 - 52` is
    not a column until something makes it one.

    Each alias also joins date_dim on `d_week_seq`, which is a *one-to-many* — a week has
    seven days — and the query means it: the answer carries one row per matching day.
    """
    def weekly(tag, months):
        store_sales = fact(t, "store_sales",
                           ["ss_sold_date_sk", "ss_store_sk", "ss_sales_price"],
                           tag=f"{tag}_store_sales")
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_week_seq", "d_day_name"],
                       tag=f"{tag}_date_dim")
        joined = star(store_sales[1], (f"{tag}_d_ss", date_dim, "d_date_sk",
                                       "ss_sold_date_sk"))
        per_day = N.project(
            f"{tag}_per_day", joined,
            [Alias(Col("d_week_seq"), "d_week_seq"), Alias(Col("ss_store_sk"), "ss_store_sk")]
            + [Alias(Case(whens=((Binary("==", Col("d_day_name"), Lit(day)),
                                  Col("ss_sales_price")),),
                          otherwise=Lit(float("nan"))), column)
               for day, column in zip(_DAYS, _DAY_SALES)],
        )
        wss = aggregate_by(
            f"{tag}_wss", per_day, ["d_week_seq", "ss_store_sk"],
            [A.Agg(A.SUM, column, column) for column in _DAY_SALES],
            schema_frame=corpus.schema_of(date_dim[0], store_sales[0],
                                          **{column: "float64" for column in _DAY_SALES}),
        )
        store = dim(t, "store", ["s_store_sk", "s_store_name", "s_store_id"],
                    tag=f"{tag}_store")
        calendar = dim(t, "date_dim", ["d_date_sk", "d_week_seq", "d_month_seq"],
                       between(Col("d_month_seq"), Lit(months[0]), Lit(months[1])),
                       tag=f"{tag}_calendar")
        in_months = N.hash_join(
            f"{tag}_calendar_join",
            N.coalesce_all(f"{tag}_calendar_all",
                           rename(f"{tag}_calendar_keys", calendar[1],
                                  [("d_week_seq", "calendar_week_seq")]),
                           schema={"d_date_sk": "int64", "calendar_week_seq": "int64",
                                   "d_month_seq": "int64"}),
            wss, JoinType.INNER, ["calendar_week_seq"], ["d_week_seq"],
        )
        return star(in_months, (f"{tag}_s", store, "s_store_sk", "ss_store_sk"))

    this_year = rename(
        "q59_y", weekly("q59_y", (1212, 1212 + 11)),
        [("s_store_name", "s_store_name1"), ("d_week_seq", "d_week_seq1"),
         ("s_store_id", "s_store_id1")]
        + [(column, f"{column}1") for column in _DAY_SALES],
    )
    last_year = N.project(
        "q59_x", weekly("q59_x", (1212 + 12, 1212 + 23)),
        [Alias(Col("s_store_id"), "s_store_id2"),
         Alias(Binary("-", Col("d_week_seq"), Lit(52)), "d_week_seq2_shifted")]
        + [Alias(Col(column), f"{column}2") for column in _DAY_SALES],
    )
    joined = N.hash_join(
        "q59_year_on_year",
        N.coalesce_all("q59_x_all", last_year,
                       schema={"s_store_id2": "object", "d_week_seq2_shifted": "int64",
                               **{f"{column}2": "float64" for column in _DAY_SALES}}),
        this_year, JoinType.INNER,
        ["s_store_id2", "d_week_seq2_shifted"], ["s_store_id1", "d_week_seq1"],
    )
    out = N.project(
        "out", joined,
        [Alias(Col("s_store_name1"), "s_store_name1"), Alias(Col("s_store_id1"), "s_store_id1"),
         Alias(Col("d_week_seq1"), "d_week_seq1")]
        + [Alias(Binary("/", Col(f"{column}1"), Col(f"{column}2")), f"{column}_ratio")
           for column in _DAY_SALES],
    )
    return N.unload(
        "unload",
        sorted_output("sort", out, ["s_store_name1", "s_store_id1", "d_week_seq1"],
                      [True, True, True], fetch=100, nulls_first=True),
    )


# -- a year against the one before, over a set union --------------------------------


_Q75_KEYS = ["d_year", "i_brand_id", "i_class_id", "i_category_id", "i_manufact_id"]
_Q75_CHANNELS = (
    ("cs", "catalog_sales", "catalog_returns", "cs_item_sk", "cs_sold_date_sk",
     "cs_order_number", "cr_item_sk", "cr_order_number", "cs_quantity",
     "cr_return_quantity", "cs_ext_sales_price", "cr_return_amount"),
    ("ss", "store_sales", "store_returns", "ss_item_sk", "ss_sold_date_sk",
     "ss_ticket_number", "sr_item_sk", "sr_ticket_number", "ss_quantity",
     "sr_return_quantity", "ss_ext_sales_price", "sr_return_amt"),
    ("ws", "web_sales", "web_returns", "ws_item_sk", "ws_sold_date_sk",
     "ws_order_number", "wr_item_sk", "wr_order_number", "ws_quantity",
     "wr_return_quantity", "ws_ext_sales_price", "wr_return_amt"),
)


@query("q75", order_by=("sales_cnt_diff", "sales_amt_diff"))
def plan_q75(t):
    """Book lines whose unit sales fell by more than a tenth from one year to the next.

    The one query in the corpus that writes `UNION` rather than `UNION ALL`, and the
    difference is load-bearing: the three channels' *detail rows* are deduplicated before
    they are summed, so two identical (year, brand, class, category, manufacturer, count,
    amount) rows from different channels count once. That lowers to an interleave and a
    grouped aggregate with no aggregates — a `DISTINCT` — under the aggregate that does the
    summing. Getting it wrong would inflate every total.

    Each channel is also a fact left-joined to its returns, which — as everywhere else — is
    a Right join with the returns as the build so the fact keeps streaming.
    """
    details = []
    for (tag, table, returns_table, item_key, date_key, order_key, return_item_key,
         return_order_key, quantity, return_quantity, price, refund) in _Q75_CHANNELS:
        sales = fact(t, table, [item_key, date_key, order_key, quantity, price],
                     tag=f"q75_{tag}_{table}")
        returns = dim(t, returns_table,
                      [return_item_key, return_order_key, return_quantity, refund],
                      rows=250_000, tag=f"q75_{tag}_{returns_table}")
        net = N.hash_join(
            f"q75_{tag}_returns",
            N.coalesce_all(f"q75_{tag}_returns_build", returns[1],
                           schema=dict(returns[0].dtypes)),
            sales[1], JoinType.RIGHT,
            [return_order_key, return_item_key], [order_key, item_key],
        )
        item = dim(t, "item",
                   ["i_item_sk", "i_brand_id", "i_class_id", "i_category_id",
                    "i_manufact_id", "i_category"],
                   Binary("==", Col("i_category"), Lit("Books")), tag=f"q75_{tag}_item")
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"], tag=f"q75_{tag}_date_dim")
        joined = star(
            net,
            (f"q75_{tag}_i", item, "i_item_sk", item_key),
            (f"q75_{tag}_d", date_dim, "d_date_sk", date_key),
        )
        details.append(N.project(
            f"q75_{tag}_detail", joined,
            [Alias(Col(key), key) for key in _Q75_KEYS]
            + [Alias(Binary("-", Col(quantity),
                            Coalesce((Col(return_quantity), Lit(0)))), "sales_cnt"),
               Alias(Binary("-", Col(price), Coalesce((Col(refund), Lit(0.0)))),
                     "sales_amt")],
        ))
    detail_schema = corpus.schema_of(
        **{key: "int64" for key in _Q75_KEYS}, sales_cnt="int64", sales_amt="float64")
    distinct = aggregate_by(
        "q75_distinct", N.interleave("q75_union", details),
        _Q75_KEYS + ["sales_cnt", "sales_amt"], [], schema_frame=detail_schema,
    )

    def year(tag, wanted):
        totals = aggregate_by(
            f"q75_{tag}", N.filter_(f"q75_{tag}_year", distinct,
                                    Binary("==", Col("d_year"), Lit(wanted))),
            _Q75_KEYS, [A.Agg(A.SUM, "sales_cnt", "sales_cnt"),
                        A.Agg(A.SUM, "sales_amt", "sales_amt")],
            schema_frame=detail_schema,
        )
        return rename(f"q75_{tag}_alias", totals,
                      [(key, f"{tag}_{key}") for key in _Q75_KEYS]
                      + [("sales_cnt", f"{tag}_sales_cnt"),
                         ("sales_amt", f"{tag}_sales_amt")])

    item_keys = _Q75_KEYS[1:]
    joined = N.hash_join(
        "q75_prev_curr",
        N.coalesce_all("q75_prev_all", year("prev", 2001),
                       schema={f"prev_{key}": "int64" for key in _Q75_KEYS}
                       | {"prev_sales_cnt": "int64", "prev_sales_amt": "float64"}),
        year("curr", 2002), JoinType.INNER,
        [f"prev_{key}" for key in item_keys], [f"curr_{key}" for key in item_keys],
    )
    fell = N.filter_(
        "q75_fell", joined,
        Binary("<", Binary("/", Col("curr_sales_cnt"), Col("prev_sales_cnt")), Lit(0.9)),
    )
    out = N.project(
        "out", fell,
        [Alias(Col("prev_d_year"), "prev_year"), Alias(Col("curr_d_year"), "year_")]
        + [Alias(Col(f"curr_{key}"), key) for key in item_keys]
        + [Alias(Col("prev_sales_cnt"), "prev_yr_cnt"),
           Alias(Col("curr_sales_cnt"), "curr_yr_cnt"),
           Alias(Binary("-", Col("curr_sales_cnt"), Col("prev_sales_cnt")),
                 "sales_cnt_diff"),
           Alias(Binary("-", Col("curr_sales_amt"), Col("prev_sales_amt")),
                 "sales_amt_diff")],
    )
    return N.unload(
        "unload",
        sorted_output("sort", out, ["sales_cnt_diff", "sales_amt_diff"], [True, True],
                      fetch=100),
    )
