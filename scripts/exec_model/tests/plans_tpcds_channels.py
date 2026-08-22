"""TPC-DS: the same measure over store, catalog and web — union, and what comes after.

The benchmark's recurring reporting shape. One measure is computed per channel over three
near-identical subtrees, the three results are `UNION ALL`ed, and an aggregate on top adds
them up. That makes the union node load-bearing in a way it is nowhere else: it is where
three independent scans of three different fact tables meet, and everything below it can
run without waiting for the others.

Two variants of what sits above the union. The plain ones (q33, q56, q60, q71) re-aggregate
by the same key. The reporting ones (q5, q14, q80) put a `ROLLUP` there, so the aggregate
emits one row per grouping set per group and carries a grouping id — which every phase above
the partial then treats as another key.
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
from .plans_tpcds_common import dim, fact, registry, star
from .plans_tpcds_sets import semi
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import (
    Alias, Binary, Coalesce, Col, Concat, Lit,
)
from ..operators.join_types import JoinType

QUERIES, ORDER_BY, query = registry()

#: What an empty lane's aggregate has to emit for each key these queries group on.
_KEY_DTYPES = {"i_manufact_id": "int64", "i_item_id": "object"}

#: The three channels, as `(tag, table, prefix, the address key each one bills to)`.
CHANNELS = (("ss", "store_sales", "ss", "ss_addr_sk"),
            ("cs", "catalog_sales", "cs", "cs_bill_addr_sk"),
            ("ws", "web_sales", "ws", "ws_bill_addr_sk"))


# -- one month's sales of a slice of the catalogue, by item ------------------------


def _items_like(t, tag, key, predicate):
    """The items sharing an id or a manufacturer with something the predicate selects.

    `i_manufact_id IN (SELECT i_manufact_id FROM item WHERE i_category = …)` is a semi join
    of item with itself on a non-key column. item is the build — it is the dimension the
    star needs anyway — so the semi join narrows the build side before it is ever collected,
    which is the same work a pushdown would do and is expressed as a join because that is
    what the query says.
    """
    items = dim(t, "item", ["i_item_sk", "i_item_id", "i_manufact_id"], tag=f"{tag}_item")
    chosen = rename(
        f"{tag}_chosen",
        dim(t, "item", ["i_item_id", "i_manufact_id", "i_category", "i_color"], predicate,
            tag=f"{tag}_chosen_item")[1],
        [(key, f"chosen_{key}")],
    )
    return items[0], semi(f"{tag}_item_semi", items[1],
                          N.coalesce_all(f"{tag}_chosen_all", chosen),
                          [key], [f"chosen_{key}"], schema=dict(items[0].dtypes))


def _channel_totals(t, tag, key, predicate, year, moy, measure_suffix="_ext_sales_price"):
    """One `total_sales` per key per channel, as three subtrees ready to be unioned."""
    branches = []
    for channel, table, prefix, addr_key in CHANNELS:
        subtag = f"{tag}_{channel}"
        sales = fact(t, table, [f"{prefix}_item_sk", f"{prefix}_sold_date_sk", addr_key,
                                f"{prefix}{measure_suffix}"], tag=f"{subtag}_{table}")
        item_frame, items = _items_like(t, subtag, key, predicate)
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_year", "d_moy"],
                       all_of(Binary("==", Col("d_year"), Lit(year)),
                              Binary("==", Col("d_moy"), Lit(moy))),
                       tag=f"{subtag}_date_dim")
        address = dim(t, "customer_address", ["ca_address_sk", "ca_gmt_offset"],
                      Binary("==", Col("ca_gmt_offset"), Lit(-5)),
                      tag=f"{subtag}_customer_address")
        joined = star(
            sales[1],
            (f"{subtag}_d", date_dim, "d_date_sk", f"{prefix}_sold_date_sk"),
            (f"{subtag}_ca", address, "ca_address_sk", addr_key),
            (f"{subtag}_i", (item_frame, items), "i_item_sk", f"{prefix}_item_sk"),
        )
        branches.append(aggregate_by(
            f"{subtag}_agg",
            select(f"{subtag}_keys", joined, key, f"{prefix}{measure_suffix}"),
            [key], [A.Agg(A.SUM, f"{prefix}{measure_suffix}", "total_sales")],
            schema_frame=corpus.schema_of(item_frame, **{f"{prefix}{measure_suffix}": "float64"}),
        ))
    return branches


def _union_of_channels(t, tag, key, predicate, year, moy, order_by, ascending):
    """The three channels' totals, unioned and re-added."""
    branches = _channel_totals(t, tag, key, predicate, year, moy)
    combined = N.union(f"{tag}_union", branches)
    final = aggregate_by(
        f"{tag}_total", combined, [key], [A.Agg(A.SUM, "total_sales", "total_sales")],
        schema_frame=corpus.schema_of(**{key: _KEY_DTYPES[key], "total_sales": "float64"}),
    )
    return N.unload(
        "unload",
        sorted_output("sort", final, list(order_by), list(ascending), fetch=100,
                      nulls_first=True),
    )


@query("q33", order_by=("total_sales",))
def plan_q33(t):
    """Electronics sold in one month, by manufacturer, across all three channels."""
    return _union_of_channels(
        t, "q33", "i_manufact_id", is_in(Col("i_category"), ("Electronics",)),
        1998, 5, ("total_sales",), (True,),
    )


@query("q56", order_by=("total_sales", "i_item_id"))
def plan_q56(t):
    """q33 keyed on the item id and selected by colour rather than category."""
    return _union_of_channels(
        t, "q56", "i_item_id", is_in(Col("i_color"), ("slate", "blanched", "burnished")),
        2001, 2, ("total_sales", "i_item_id"), (True, True),
    )


@query("q60", order_by=("i_item_id", "total_sales"))
def plan_q60(t):
    """q56 for one category, sorted by item rather than by money — the same lowering, and a
    different top hundred, which is the pair's whole purpose."""
    return _union_of_channels(
        t, "q60", "i_item_id", Binary("==", Col("i_category"), Lit("Music")),
        1998, 9, ("i_item_id", "total_sales"), (True, True),
    )


# -- three channels sold at breakfast and dinner -----------------------------------


@query("q71", order_by=("ext_price", "brand_id", "t_hour"))
def plan_q71(t):
    """One manager's brands, by the half-hour they sold in, across all three channels.

    The union is *below* the joins here rather than above them: the three channels are
    projected to one common four-column shape first, and then item and time_dim are joined
    to the union once. That is the arrangement the mode wants — one join instead of three —
    and it is available exactly because the three branches were made union-compatible first.
    """
    branches = []
    for channel, table, prefix, _ in CHANNELS:
        sales = fact(t, table,
                     [f"{prefix}_ext_sales_price", f"{prefix}_sold_date_sk",
                      f"{prefix}_item_sk", f"{prefix}_sold_time_sk"],
                     tag=f"q71_{channel}_{table}")
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_moy", "d_year"],
                       all_of(Binary("==", Col("d_moy"), Lit(11)),
                              Binary("==", Col("d_year"), Lit(1999))),
                       tag=f"q71_{channel}_date_dim")
        joined = star(sales[1], (f"q71_{channel}_d", date_dim, "d_date_sk",
                                 f"{prefix}_sold_date_sk"))
        branches.append(rename(
            f"q71_{channel}_shape", joined,
            [(f"{prefix}_ext_sales_price", "ext_price"),
             (f"{prefix}_item_sk", "sold_item_sk"),
             (f"{prefix}_sold_time_sk", "time_sk")],
        ))
    # `interleave`, not `union`: a union gives each branch its own lane, and the item and
    # time_dim joins below have one-lane build sides — a join's two sides must agree on lane
    # count. Interleaving keeps the three branches in one lane, which is also what lets the
    # two dimensions be joined once instead of three times.
    combined = N.interleave("q71_union", branches)
    item = dim(t, "item", ["i_item_sk", "i_brand_id", "i_brand", "i_manager_id"],
               Binary("==", Col("i_manager_id"), Lit(1)))
    time_dim = dim(t, "time_dim", ["t_time_sk", "t_hour", "t_minute", "t_meal_time"],
                   is_in(Col("t_meal_time"), ("breakfast", "dinner")))
    with_item = star(
        combined,
        ("q71_i", item, "i_item_sk", "sold_item_sk"),
        ("q71_t", time_dim, "t_time_sk", "time_sk"),
    )
    keys = ["i_brand", "i_brand_id", "t_hour", "t_minute"]
    final = aggregate_by(
        "q71_agg", select("q71_keys", with_item, *keys, "ext_price"), keys,
        [A.Agg(A.SUM, "ext_price", "ext_price")],
        schema_frame=corpus.schema_of(item[0], time_dim[0], ext_price="float64"),
    )
    out = rename("out", final,
                 [("i_brand_id", "brand_id"), ("i_brand", "brand"), ("t_hour", "t_hour"),
                  ("t_minute", "t_minute"), ("ext_price", "ext_price")])
    return N.unload(
        "unload",
        sorted_output("sort", out, ["ext_price", "brand_id", "t_hour"], [False, True, True],
                      nulls_first=True),
    )


# -- the rollup reports --------------------------------------------------------------


#: The measures q5 and q80 both report, in the order their output declares them.
_REPORT = ("sales", "returns_", "profit")
_REPORT_SCHEMA = {"channel": "object", "id": "object",
                  **{measure: "float64" for measure in _REPORT}}


def _labelled(tag, channel_label, prefix, branch, id_column):
    """One channel's totals, labelled and given the composite id the union needs.

    `concat('store', s_store_id)` — a literal and a column — is what makes the three
    branches union-compatible: without it the ids of three different dimensions would
    collide in one column.
    """
    return N.project(
        f"{tag}_labelled", branch,
        [Alias(Lit(channel_label), "channel"),
         Alias(Concat((Lit(prefix), Col(id_column))), "id")]
        + [Alias(Col(measure), measure) for measure in _REPORT],
    )


def _rollup_report(tag, branches):
    """Union the three labelled channels and roll up over (channel, id).

    `ROLLUP(channel, id)` is three grouping sets, so the partial emits one row per set per
    group and tags it with a grouping id; every phase above it groups on the keys *and* that
    id, which is what keeps a rolled-up NULL apart from a natural one. The id is dropped in
    the projection at the top — it is machinery, not an answer.
    """
    combined = N.union(f"{tag}_union", branches)
    rolled = aggregate_by(
        f"{tag}_rollup", combined, ["channel", "id"],
        [A.Agg(A.SUM, measure, measure) for measure in _REPORT],
        grouping_sets=A.rollup_masks(2),
        schema_frame=corpus.schema_of(**_REPORT_SCHEMA),
    )
    out = select("out", rolled, "channel", "id", *_REPORT)
    return N.unload(
        "unload",
        sorted_output("sort", out, ["channel", "id"], [True, True], fetch=100,
                      nulls_first=True),
    )


#: The fortnight q5 reports on, and the month q80 does.
_Q5_RANGE = ("2000-08-23", "2000-09-06")
_Q80_RANGE = ("2000-08-23", "2000-09-22")


@query("q5", order_by=("channel", "id"))
def plan_q5(t):
    """Sales and returns per channel over a fortnight, rolled up.

    Two unions, one above the other. The lower one makes a sale and a return the *same
    shape* — each contributes zeros to the other's measures, which is how the query sums two
    tables in one pass instead of joining them — and the upper one puts the three channels
    side by side under a `ROLLUP`.

    The web branch is the odd one: a return does not carry a web site, so it has to be
    fetched from the sale, and the query writes that as `web_returns LEFT OUTER JOIN
    web_sales`. Preserving the returns while they stream means a **Right** join with the
    sales as the build, as in q40 and q93.
    """
    zero = Lit(0.0)

    def shaped(tag, node, date_key, place_key, sales_price, profit, return_amt, net_loss):
        return N.project(
            f"{tag}_shape", node,
            [Alias(Col(place_key), "place_sk"), Alias(Col(date_key), "date_sk"),
             Alias(sales_price, "sales_price"), Alias(profit, "profit"),
             Alias(return_amt, "return_amt"), Alias(net_loss, "net_loss")],
        )

    def branch(tag, sold, returned, place_table, place_key, place_id, place_columns):
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_date"],
                       between(Col("d_date"), date(_Q5_RANGE[0]), date(_Q5_RANGE[1])),
                       tag=f"{tag}_date_dim")
        place = dim(t, place_table, place_columns, tag=f"{tag}_{place_table}")
        # Interleaved rather than unioned, for the same reason as q71: the date and place
        # joins above have one-lane build sides.
        both = N.interleave(f"{tag}_sales_returns", [sold, returned])
        joined = star(both,
                      (f"{tag}_d", date_dim, "d_date_sk", "date_sk"),
                      (f"{tag}_place", place, place_key, "place_sk"))
        totals = aggregate_by(
            f"{tag}_agg",
            select(f"{tag}_keys", joined, place_id, "sales_price", "profit", "return_amt",
                   "net_loss"),
            [place_id],
            [A.Agg(A.SUM, "sales_price", "sales"), A.Agg(A.SUM, "profit", "profit"),
             A.Agg(A.SUM, "return_amt", "returns_"),
             A.Agg(A.SUM, "net_loss", "profit_loss")],
            schema_frame=corpus.schema_of(place[0], sales_price="float64", profit="float64",
                                          return_amt="float64", net_loss="float64"),
        )
        return N.project(
            f"{tag}_net", totals,
            [Alias(Col(place_id), place_id), Alias(Col("sales"), "sales"),
             Alias(Col("returns_"), "returns_"),
             Alias(Binary("-", Col("profit"), Col("profit_loss")), "profit")],
        )

    store_branch = branch(
        "q5_store",
        shaped("q5_ss", fact(t, "store_sales",
                             ["ss_store_sk", "ss_sold_date_sk", "ss_ext_sales_price",
                              "ss_net_profit"], tag="q5_store_sales")[1],
               "ss_sold_date_sk", "ss_store_sk", Col("ss_ext_sales_price"),
               Col("ss_net_profit"), zero, zero),
        shaped("q5_sr", fact(t, "store_returns",
                             ["sr_store_sk", "sr_returned_date_sk", "sr_return_amt",
                              "sr_net_loss"], tag="q5_store_returns")[1],
               "sr_returned_date_sk", "sr_store_sk", zero, zero, Col("sr_return_amt"),
               Col("sr_net_loss")),
        "store", "s_store_sk", "s_store_id", ["s_store_sk", "s_store_id"],
    )
    catalog_branch = branch(
        "q5_catalog",
        shaped("q5_cs", fact(t, "catalog_sales",
                             ["cs_catalog_page_sk", "cs_sold_date_sk", "cs_ext_sales_price",
                              "cs_net_profit"], tag="q5_catalog_sales")[1],
               "cs_sold_date_sk", "cs_catalog_page_sk", Col("cs_ext_sales_price"),
               Col("cs_net_profit"), zero, zero),
        shaped("q5_cr", fact(t, "catalog_returns",
                             ["cr_catalog_page_sk", "cr_returned_date_sk", "cr_return_amount",
                              "cr_net_loss"], tag="q5_catalog_returns")[1],
               "cr_returned_date_sk", "cr_catalog_page_sk", zero, zero,
               Col("cr_return_amount"), Col("cr_net_loss")),
        "catalog_page", "cp_catalog_page_sk", "cp_catalog_page_id",
        ["cp_catalog_page_sk", "cp_catalog_page_id"],
    )
    web_sales = dim(t, "web_sales", ["ws_item_sk", "ws_order_number", "ws_web_site_sk"],
                    rows=250_000, tag="q5_web_sales_build")
    web_returns = fact(t, "web_returns",
                       ["wr_item_sk", "wr_order_number", "wr_returned_date_sk",
                        "wr_return_amt", "wr_net_loss"], tag="q5_web_returns")
    returns_with_site = N.hash_join(
        "q5_ws_wr",
        N.coalesce_all("q5_ws_wr_build", web_sales[1], schema=dict(web_sales[0].dtypes)),
        web_returns[1], JoinType.RIGHT,
        ["ws_item_sk", "ws_order_number"], ["wr_item_sk", "wr_order_number"],
    )
    web_branch = branch(
        "q5_web",
        shaped("q5_ws", fact(t, "web_sales",
                             ["ws_web_site_sk", "ws_sold_date_sk", "ws_ext_sales_price",
                              "ws_net_profit"], tag="q5_web_sales")[1],
               "ws_sold_date_sk", "ws_web_site_sk", Col("ws_ext_sales_price"),
               Col("ws_net_profit"), zero, zero),
        shaped("q5_wr", returns_with_site, "wr_returned_date_sk", "ws_web_site_sk",
               zero, zero, Col("wr_return_amt"), Col("wr_net_loss")),
        "web_site", "web_site_sk", "web_site_id", ["web_site_sk", "web_site_id"],
    )
    return _rollup_report("q5", [
        _labelled("q5_store", "store channel", "store", store_branch, "s_store_id"),
        _labelled("q5_catalog", "catalog channel", "catalog_page", catalog_branch,
                  "cp_catalog_page_id"),
        _labelled("q5_web", "web channel", "web_site", web_branch, "web_site_id"),
    ])


@query("q80", order_by=("channel", "id"))
def plan_q80(t):
    """A month of promoted, expensive goods per channel, net of returns, rolled up.

    Where q5 unions a sale with a return, this one **joins** them — `LEFT OUTER JOIN` on the
    transaction, so a sale that was never returned contributes a null that `coalesce` turns
    into a zero. Again the fact table is the preserved side and again that means a Right
    join with the returns as the build.
    """
    def branch(tag, table, prefix, returns_table, returns_prefix, transaction_keys,
               place_table, place_key, place_fact_key, place_id, refund, loss):
        sales = fact(t, table,
                     [f"{prefix}_item_sk", f"{prefix}_sold_date_sk", place_fact_key,
                      f"{prefix}_promo_sk", f"{prefix}_ext_sales_price",
                      f"{prefix}_net_profit"] + [transaction_keys[0]],
                     tag=f"{tag}_{table}")
        returns = dim(t, returns_table,
                      [f"{returns_prefix}_item_sk", transaction_keys[1], refund, loss],
                      rows=250_000, tag=f"{tag}_{returns_table}")
        net = N.hash_join(
            f"{tag}_returns",
            N.coalesce_all(f"{tag}_returns_build", returns[1], schema=dict(returns[0].dtypes)),
            sales[1], JoinType.RIGHT,
            [f"{returns_prefix}_item_sk", transaction_keys[1]],
            [f"{prefix}_item_sk", transaction_keys[0]],
        )
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_date"],
                       between(Col("d_date"), date(_Q80_RANGE[0]), date(_Q80_RANGE[1])),
                       tag=f"{tag}_date_dim")
        place = dim(t, place_table, [place_key, place_id], tag=f"{tag}_{place_table}")
        item = dim(t, "item", ["i_item_sk", "i_current_price"],
                   Binary(">", Col("i_current_price"), Lit(50)), tag=f"{tag}_item")
        promotion = dim(t, "promotion", ["p_promo_sk", "p_channel_tv"],
                        Binary("==", Col("p_channel_tv"), Lit("N")), tag=f"{tag}_promotion")
        joined = star(
            net,
            (f"{tag}_d", date_dim, "d_date_sk", f"{prefix}_sold_date_sk"),
            (f"{tag}_place", place, place_key, place_fact_key),
            (f"{tag}_i", item, "i_item_sk", f"{prefix}_item_sk"),
            (f"{tag}_p", promotion, "p_promo_sk", f"{prefix}_promo_sk"),
        )
        measured = N.project(
            f"{tag}_measures", joined,
            [Alias(Col(place_id), place_id),
             Alias(Col(f"{prefix}_ext_sales_price"), "sales"),
             Alias(Coalesce((Col(refund), Lit(0.0))), "returns_"),
             Alias(Binary("-", Col(f"{prefix}_net_profit"),
                          Coalesce((Col(loss), Lit(0.0)))), "profit")],
        )
        return aggregate_by(
            f"{tag}_agg", measured, [place_id],
            [A.Agg(A.SUM, measure, measure) for measure in _REPORT],
            schema_frame=corpus.schema_of(place[0], **{m: "float64" for m in _REPORT}),
        )

    store_branch = branch(
        "q80_store", "store_sales", "ss", "store_returns", "sr",
        ("ss_ticket_number", "sr_ticket_number"), "store", "s_store_sk", "ss_store_sk",
        "s_store_id", "sr_return_amt", "sr_net_loss",
    )
    catalog_branch = branch(
        "q80_catalog", "catalog_sales", "cs", "catalog_returns", "cr",
        ("cs_order_number", "cr_order_number"), "catalog_page", "cp_catalog_page_sk",
        "cs_catalog_page_sk", "cp_catalog_page_id", "cr_return_amount", "cr_net_loss",
    )
    web_branch = branch(
        "q80_web", "web_sales", "ws", "web_returns", "wr",
        ("ws_order_number", "wr_order_number"), "web_site", "web_site_sk", "ws_web_site_sk",
        "web_site_id", "wr_return_amt", "wr_net_loss",
    )
    return _rollup_report("q80", [
        _labelled("q80_store", "store channel", "store", store_branch, "s_store_id"),
        _labelled("q80_catalog", "catalog channel", "catalog_page", catalog_branch,
                  "cp_catalog_page_id"),
        _labelled("q80_web", "web channel", "web_site", web_branch, "web_site_id"),
    ])


# -- the triple every channel sold, and the average every channel beat ---------------


_Q14_TRIPLE = ("i_brand_id", "i_class_id", "i_category_id")
_Q14_CHANNELS = (("store", "store_sales", "ss"), ("catalog", "catalog_sales", "cs"),
                 ("web", "web_sales", "ws"))


def _q14_triples(t, tag, channel, prefix):
    """The distinct (brand, class, category) triples one channel sold over three years."""
    sales = fact(t, channel, [f"{prefix}_item_sk", f"{prefix}_sold_date_sk"],
                 tag=f"{tag}_{channel}")
    item = dim(t, "item", ["i_item_sk"] + list(_Q14_TRIPLE), tag=f"{tag}_item")
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"],
                   between(Col("d_year"), Lit(1999), Lit(2001)), tag=f"{tag}_date_dim")
    joined = star(
        sales[1],
        (f"{tag}_d", date_dim, "d_date_sk", f"{prefix}_sold_date_sk"),
        (f"{tag}_i", item, "i_item_sk", f"{prefix}_item_sk"),
    )
    return aggregate_by(f"{tag}_distinct", select(f"{tag}_keys", joined, *_Q14_TRIPLE),
                        list(_Q14_TRIPLE), [],
                        schema_frame=corpus.schema_of(**{key: "int64"
                                                         for key in _Q14_TRIPLE}))


def _q14_cross_items(t, copy):
    """item, narrowed to the triples all three channels sold — two INTERSECTs and a join.

    Each `INTERSECT` is a semi join over distinct sides, and the last join back to `item`
    turns a set of triples into a set of item keys. Built once per channel branch, because a
    plan is a tree and this subtree is named three times: three copies of nine fact scans is
    what the query costs when nothing is shared, and the cost model sees all of it.
    """
    triples = {label: _q14_triples(t, f"{copy}_{label}", channel, prefix)
               for label, channel, prefix in _Q14_CHANNELS}
    schema = dict(corpus.schema_of(**{key: "int64" for key in _Q14_TRIPLE}).dtypes)
    surviving = triples["store"]
    for label in ("catalog", "web"):
        renamed = rename(f"{copy}_{label}_triple", triples[label],
                         [(key, f"{label}_{key}") for key in _Q14_TRIPLE])
        surviving = semi(f"{copy}_{label}_intersect", surviving,
                         N.coalesce_all(f"{copy}_{label}_all", renamed),
                         list(_Q14_TRIPLE), [f"{label}_{key}" for key in _Q14_TRIPLE],
                         schema=schema)
    item = dim(t, "item", ["i_item_sk"] + list(_Q14_TRIPLE), tag=f"{copy}_cross_item")
    return item[0], semi(f"{copy}_cross_items", item[1],
                         N.coalesce_all(f"{copy}_surviving_all", surviving),
                         list(_Q14_TRIPLE), list(_Q14_TRIPLE),
                         schema=dict(item[0].dtypes))


@query("q14", order_by=("channel", "i_brand_id", "i_class_id", "i_category_id"))
def plan_q14(t):
    """November 2001, by brand, for the products every channel sells and that beat the
    three-year average.

    Everything this module is about, in one query. `cross_items` is two `INTERSECT`s, so two
    semi joins. `avg_sales` is a scalar over a three-channel `UNION ALL`, and it is
    cross-joined **once**, above the union rather than into each branch: the `HAVING`
    compares each group's own total against a constant, so filtering after the union is the
    same query and two thirds cheaper. And the top is a four-key `ROLLUP` over the labelled
    channels.
    """
    totals = []
    for label, channel, prefix in _Q14_CHANNELS:
        sales = fact(t, channel,
                     [f"{prefix}_item_sk", f"{prefix}_sold_date_sk", f"{prefix}_quantity",
                      f"{prefix}_list_price"], tag=f"q14_{label}_{channel}")
        item_frame, cross_items = _q14_cross_items(t, f"q14_{label}")
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_year", "d_moy"],
                       all_of(Binary("==", Col("d_year"), Lit(2001)),
                              Binary("==", Col("d_moy"), Lit(11))),
                       tag=f"q14_{label}_date_dim")
        joined = star(
            sales[1],
            (f"q14_{label}_d", date_dim, "d_date_sk", f"{prefix}_sold_date_sk"),
            (f"q14_{label}_i", (item_frame, cross_items), "i_item_sk", f"{prefix}_item_sk"),
        )
        measured = N.project(
            f"q14_{label}_measure", joined,
            [Alias(Col(key), key) for key in _Q14_TRIPLE]
            + [Alias(Binary("*", Col(f"{prefix}_quantity"), Col(f"{prefix}_list_price")),
                     "sold")],
        )
        grouped = aggregate_by(
            f"q14_{label}_agg", measured, list(_Q14_TRIPLE),
            [A.Agg(A.SUM, "sold", "sales"), A.Agg(A.COUNT, None, "number_sales")],
            schema_frame=corpus.schema_of(item_frame, sold="float64"),
        )
        totals.append(N.project(
            f"q14_{label}_labelled", grouped,
            [Alias(Lit(label), "channel")]
            + [Alias(Col(key), key) for key in _Q14_TRIPLE]
            + [Alias(Col("sales"), "sales"),
               Alias(Col("number_sales"), "number_sales")],
        ))
    average = _q14_average(t)
    # Interleaved rather than unioned: a union gives each branch its own lane and the cross
    # join above has a one-lane build, which a join refuses. Same reason as q71 and q5.
    against = N.cross_join("q14_against", N.coalesce_all("q14_average_all", average),
                           N.interleave("q14_union", totals))
    beating = N.filter_("q14_above_average", against,
                        Binary(">", Col("sales"), Col("average_sales")))
    keys = ["channel"] + list(_Q14_TRIPLE)
    rolled = aggregate_by(
        "q14_rollup", select("q14_rollup_keys", beating, *keys, "sales", "number_sales"),
        keys, [A.Agg(A.SUM, "sales", "sum_sales"),
               A.Agg(A.SUM, "number_sales", "sum_number_sales")],
        grouping_sets=A.rollup_masks(len(keys)),
        schema_frame=corpus.schema_of(channel="object",
                                      **{key: "int64" for key in _Q14_TRIPLE},
                                      sales="float64", number_sales="int64"),
    )
    out = select("out", rolled, *keys, "sum_sales", "sum_number_sales")
    return N.unload(
        "unload",
        sorted_output("sort", out, keys, [True] * len(keys), fetch=100, nulls_first=True),
    )


def _q14_average(t):
    """`avg(quantity * list_price)` over all three channels' three years — one number."""
    branches = []
    for label, channel, prefix in _Q14_CHANNELS:
        sales = fact(t, channel,
                     [f"{prefix}_sold_date_sk", f"{prefix}_quantity", f"{prefix}_list_price"],
                     tag=f"q14_avg_{label}_{channel}")
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"],
                       between(Col("d_year"), Lit(1999), Lit(2001)),
                       tag=f"q14_avg_{label}_date_dim")
        joined = star(sales[1], (f"q14_avg_{label}_d", date_dim, "d_date_sk",
                                 f"{prefix}_sold_date_sk"))
        branches.append(N.project(
            f"q14_avg_{label}_measure", joined,
            [Alias(Binary("*", Col(f"{prefix}_quantity"), Col(f"{prefix}_list_price")),
                   "sold")],
        ))
    return aggregate_to_one_row(
        "q14_average", N.interleave("q14_avg_union", branches),
        [A.Agg(A.MEAN, "sold", "average_sales")], corpus.schema_of(sold="float64"),
    )
