"""TPC-DS: EXISTS, NOT EXISTS and IN — the queries that lower to semi, anti and mark joins.

This is the module the join capability work is about. A correlated `EXISTS` is a semi join,
a `NOT EXISTS` is an anti join, and an `EXISTS` that appears inside an `OR` is neither: it
has to produce a *boolean per row* rather than filter rows, which is a **mark** join. All
three preserve the build side and emit at the finish, so all three make the build side the
thing the query keeps and the probe the thing that streams — the opposite assignment from a
star, and the one that decides whether these queries fit in memory.

Two of them (q16, q94) also carry a residual: `EXISTS (… AND cs1.warehouse <> cs2.warehouse)`
is a semi join with a non-equi predicate, which the matrix says needs a single-batch probe,
because #136's finish pass sees accumulated keys and a keys-only table cannot evaluate a
predicate over both sides. That is the same shape as TPC-H q21, on the queries it was
predicted for.

`count(DISTINCT x)` appears here too and is not a join at all: it lowers to two aggregates,
the first grouping by `x` and the second counting the groups.
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
    Alias, Binary, Case, Col, IsNotNull, Lit, Not, Substring,
)
from ..operators.join_types import JoinType

QUERIES, ORDER_BY, query = registry()


def semi(name, build, probe, build_keys, probe_keys, schema=None, residual=None):
    """`EXISTS` — the build rows that matched, each at most once."""
    return N.hash_join(name, N.coalesce_all(f"{name}_build", build, schema=schema), probe,
                       JoinType.LEFT_SEMI, build_keys, probe_keys, residual=residual)


def anti(name, build, probe, build_keys, probe_keys, schema=None, residual=None):
    """`NOT EXISTS` — the build rows that never matched."""
    return N.hash_join(name, N.coalesce_all(f"{name}_build", build, schema=schema), probe,
                       JoinType.LEFT_ANTI, build_keys, probe_keys, residual=residual)


def mark(name, build, probe, build_keys, probe_keys, schema=None):
    """`EXISTS` inside an `OR` — every build row, plus a boolean saying whether it matched.

    The join emits its answer in a column called `mark`, so a query with two of them has to
    rename the first before the second arrives: a batch cannot hold two columns of one name,
    and this is the case that says so with a boolean rather than with data.
    """
    return N.hash_join(name, N.coalesce_all(f"{name}_build", build, schema=schema), probe,
                       JoinType.LEFT_MARK, build_keys, probe_keys)


# -- orders shipped from two warehouses and never returned --------------------------


def _split_shipment(t, tag, channel, prefix, returns_table, return_key, state, place_column,
                    place_table, place_dim_key, place_fact_key, place_value, quarter,
                    measures):
    """q16 and q94: one channel's orders that shipped from two warehouses and came back from
    none.

    The `EXISTS` is a self-join of the fact table on the order number with `warehouse <>
    warehouse` as a residual — a filtered semi join, which the capability matrix refuses to
    stream, so its probe side is collected. Projecting that probe to two columns first is
    what keeps it affordable: 1.4M rows of two integers rather than of the whole table.

    `count(DISTINCT order_number)` is the two-aggregate form — group by the order number,
    then count the groups — and the two sums ride along in the first of them, since summing
    per order and then summing those is the same total.
    """
    order = f"{prefix}_order_number"
    warehouse = f"{prefix}_warehouse_sk"
    sales = fact(t, channel,
                 [order, warehouse, f"{prefix}_ship_date_sk", f"{prefix}_ship_addr_sk",
                  place_fact_key] + list(measures), tag=f"{tag}_{channel}")
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_date"],
                   between(Col("d_date"), date(quarter[0]), date(quarter[1])),
                   tag=f"{tag}_date_dim")
    address = dim(t, "customer_address", ["ca_address_sk", "ca_state"],
                  Binary("==", Col("ca_state"), Lit(state)), tag=f"{tag}_customer_address")
    place = dim(t, place_table, [place_dim_key, place_column],
                Binary("==", Col(place_column), Lit(place_value)), tag=f"{tag}_{place_table}")
    shipped = star(
        sales[1],
        (f"{tag}_d", date_dim, "d_date_sk", f"{prefix}_ship_date_sk"),
        (f"{tag}_ca", address, "ca_address_sk", f"{prefix}_ship_addr_sk"),
        (f"{tag}_place", place, place_dim_key, place_fact_key),
    )
    candidates = select(f"{tag}_candidates", shipped, order, warehouse, *measures)
    siblings = fact(t, channel, [order, warehouse], tag=f"{tag}_sibling_{channel}")
    other_warehouse = rename(
        f"{tag}_siblings", siblings[1],
        [(order, "other_order_number"), (warehouse, "other_warehouse_sk")],
    )
    candidate_schema = dict(corpus.schema_of(sales[0]).dtypes)
    split = semi(
        f"{tag}_exists", candidates,
        N.coalesce_all(f"{tag}_siblings_all", other_warehouse),
        [order], ["other_order_number"], schema=candidate_schema,
        residual=Binary("!=", Col(warehouse), Col("other_warehouse_sk")),
    )
    returns = fact(t, returns_table, [return_key], tag=f"{tag}_{returns_table}")
    kept = anti(
        f"{tag}_not_exists", split, N.coalesce_all(f"{tag}_returns_all", returns[1]),
        [order], [return_key], schema=candidate_schema,
    )
    per_order = aggregate_by(
        f"{tag}_per_order", select(f"{tag}_order_keys", kept, order, *measures), [order],
        [A.Agg(A.SUM, column, column) for column in measures],
        schema_frame=corpus.schema_of(sales[0]),
    )
    return N.unload(
        "unload",
        aggregate_to_one_row(
            f"{tag}_agg", per_order,
            [A.Agg(A.COUNT, None, "order count"),
             A.Agg(A.SUM, measures[0], "total shipping cost"),
             A.Agg(A.SUM, measures[1], "total net profit")],
            corpus.schema_of(sales[0]),
        ),
    )


@query("q16", order_by=("order count",))
def plan_q16(t):
    """Catalog orders into Georgia handled by one call centre, split across warehouses."""
    return _split_shipment(
        t, "q16", "catalog_sales", "cs", "catalog_returns", "cr_order_number", "GA",
        "cc_county", "call_center", "cc_call_center_sk", "cs_call_center_sk",
        "Williamson County",
        ("2002-02-01", "2002-04-02"), ("cs_ext_ship_cost", "cs_net_profit"),
    )


@query("q94", order_by=("order count",))
def plan_q94(t):
    """q16 over the web channel, into Illinois, for one company."""
    return _split_shipment(
        t, "q94", "web_sales", "ws", "web_returns", "wr_order_number", "IL",
        "web_company_name", "web_site", "web_site_sk", "ws_web_site_sk", "pri",
        ("1999-02-01", "1999-04-02"), ("ws_ext_ship_cost", "ws_net_profit"),
    )


@query("q95", order_by=("order count",))
def plan_q95(t):
    """q94 written with `IN` over a CTE instead of `EXISTS`.

    The CTE `ws_wh` is the self-join on its own, materialized, and both `IN`s read it — so
    it is built twice, and the plan says so. The second `IN` is `web_returns` intersected
    with `ws_wh`, which is why the anti join of q94 becomes a *semi* join here: the query
    keeps the orders that *were* returned, and the two spellings are not the same query.
    """
    order, warehouse = "ws_order_number", "ws_warehouse_sk"

    def ws_wh(copy):
        """Order numbers shipped from more than one warehouse."""
        left = fact(t, "web_sales", [order, warehouse], tag=f"{copy}_ws1")
        right = rename(f"{copy}_ws2",
                       fact(t, "web_sales", [order, warehouse], tag=f"{copy}_ws2_scan")[1],
                       [(order, "wh2_order_number"), (warehouse, "wh2_warehouse_sk")])
        paired = N.hash_join(
            f"{copy}_pairs", N.coalesce_all(f"{copy}_pairs_build", right), left[1],
            JoinType.INNER, ["wh2_order_number"], [order],
            residual=Binary("!=", Col(warehouse), Col("wh2_warehouse_sk")),
        )
        return aggregate_by(f"{copy}_distinct", select(f"{copy}_orders", paired, order),
                            [order], [], schema_frame=corpus.schema_of(left[0]))

    measures = ("ws_ext_ship_cost", "ws_net_profit")
    sales = fact(t, "web_sales",
                 [order, "ws_ship_date_sk", "ws_ship_addr_sk", "ws_web_site_sk"]
                 + list(measures))
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_date"],
                   between(Col("d_date"), date("1999-02-01"), date("1999-04-02")))
    address = dim(t, "customer_address", ["ca_address_sk", "ca_state"],
                  Binary("==", Col("ca_state"), Lit("IL")))
    site = dim(t, "web_site", ["web_site_sk", "web_company_name"],
               Binary("==", Col("web_company_name"), Lit("pri")))
    candidates = select(
        "candidates",
        star(sales[1],
             ("d_ws", date_dim, "d_date_sk", "ws_ship_date_sk"),
             ("ca_ws", address, "ca_address_sk", "ws_ship_addr_sk"),
             ("site_ws", site, "web_site_sk", "ws_web_site_sk")),
        order, *measures,
    )
    schema = dict(corpus.schema_of(sales[0]).dtypes)
    split = semi("in_ws_wh", candidates,
                 N.coalesce_all("ws_wh_a_all", rename("ws_wh_a_keys", ws_wh("a"),
                                                      [(order, "wh_order_number")])),
                 [order], ["wh_order_number"], schema=schema)
    returned = semi(
        "returned_and_split",
        N.coalesce_all("returned_build",
                       rename("wr_keys",
                              fact(t, "web_returns", ["wr_order_number"])[1],
                              [("wr_order_number", "wr_order_number")])),
        N.coalesce_all("ws_wh_b_all", rename("ws_wh_b_keys", ws_wh("b"),
                                             [(order, "wh_order_number")])),
        ["wr_order_number"], ["wh_order_number"],
    )
    kept = semi("in_returned", split, N.coalesce_all("returned_all", returned),
                [order], ["wr_order_number"], schema=schema)
    per_order = aggregate_by(
        "per_order", select("order_keys", kept, order, *measures), [order],
        [A.Agg(A.SUM, column, column) for column in measures],
        schema_frame=corpus.schema_of(sales[0]),
    )
    return N.unload(
        "unload",
        aggregate_to_one_row(
            "agg", per_order,
            [A.Agg(A.COUNT, None, "order count"),
             A.Agg(A.SUM, measures[0], "total shipping cost"),
             A.Agg(A.SUM, measures[1], "total net profit")],
            corpus.schema_of(sales[0]),
        ),
    )


# -- customers who bought here but not there ----------------------------------------


#: The date window each of the three existence queries looks through.
_Q10_WINDOW = (all_of(Binary("==", Col("d_year"), Lit(2002)),
                      between(Col("d_moy"), Lit(1), Lit(4))), ("d_year", "d_moy"))
_Q69_WINDOW = (all_of(Binary("==", Col("d_year"), Lit(2001)),
                      between(Col("d_moy"), Lit(4), Lit(6))), ("d_year", "d_moy"))
#: q35 says `d_qoy < 4`, which is nine months and not three — reading it as a month range
#: silently answered a different query.
_Q35_WINDOW = (all_of(Binary("==", Col("d_year"), Lit(2002)),
                      Binary("<", Col("d_qoy"), Lit(4))), ("d_year", "d_qoy"))

_DEMOGRAPHIC_COLUMNS = ("cd_gender", "cd_marital_status", "cd_education_status",
                        "cd_purchase_estimate", "cd_credit_rating", "cd_dep_count",
                        "cd_dep_employed_count", "cd_dep_college_count")


def _buyers(t, tag, channel, prefix, customer_key, window):
    """One channel's customers over a window of date_dim, as a probe side of one column.

    `window` is `(predicate, the date_dim columns it reads)` — q10 and q69 name a range of
    months, q35 names the first three quarters, and the difference is only what the scan
    filters on.
    """
    sales = fact(t, channel, [customer_key, f"{prefix}_sold_date_sk"], tag=f"{tag}_{channel}")
    date_dim = dim(t, "date_dim", ["d_date_sk"] + list(window[1]), window[0],
                   tag=f"{tag}_{channel}_date_dim")
    joined = star(sales[1], (f"{tag}_{channel}_d", date_dim, "d_date_sk",
                             f"{prefix}_sold_date_sk"))
    return select(f"{tag}_{channel}_buyers", joined, customer_key)


def _demographic_counts(t, tag, place_column, places, columns, combine):
    """q10 and q69: count the customers of a region by every demographic attribute at once.

    The customer side is the build of all three existence joins — a hundred thousand rows,
    narrowed by the region first — and each channel's customer keys stream past it. Which
    join each `EXISTS` becomes is `combine`'s business, and it is the whole difference
    between the two queries.
    """
    customer = fact(t, "customer",
                    ["c_customer_sk", "c_current_addr_sk", "c_current_cdemo_sk"],
                    tag=f"{tag}_customer")
    address = dim(t, "customer_address", ["ca_address_sk", place_column],
                  is_in(Col(place_column), places), tag=f"{tag}_customer_address")
    demographics = fact(t, "customer_demographics",
                        ["cd_demo_sk"] + list(columns), tag=f"{tag}_customer_demographics")
    local = star(
        customer[1],
        (f"{tag}_ca", address, "ca_address_sk", "c_current_addr_sk"),
        (f"{tag}_cd", demographics, "cd_demo_sk", "c_current_cdemo_sk"),
    )
    candidates = select(f"{tag}_candidates", local, "c_customer_sk", *columns)
    schema = dict(corpus.schema_of(customer[0], demographics[0]).dtypes)
    kept = combine(candidates, schema)
    counts = [A.Agg(A.COUNT, None, f"cnt{index}") for index in range(1, 7)]
    return aggregate_by(
        f"{tag}_agg", select(f"{tag}_keys", kept, *columns), list(columns), counts,
        schema_frame=corpus.schema_of(demographics[0]),
    )


@query("q10", order_by=_DEMOGRAPHIC_COLUMNS)
def plan_q10(t):
    """Customers of five counties who bought in store, and on the web **or** in the catalog.

    The `OR` is what makes this query worth having. Its two arms are correlated `EXISTS`es,
    and an `EXISTS` under an `OR` cannot filter — it has to answer per row, so both become
    **mark** joins and the disjunction is a filter over their two booleans. The `AND`ed one
    above them is an ordinary semi join, and the contrast between the two spellings in one
    query is the point.
    """
    def combine(candidates, schema):
        bought = semi("q10_in_store", candidates,
                      N.coalesce_all("q10_store_all",
                                     _buyers(t, "q10", "store_sales", "ss", "ss_customer_sk",
                                             _Q10_WINDOW)),
                      ["c_customer_sk"], ["ss_customer_sk"], schema=schema)
        on_web = mark("q10_on_web", bought,
                      N.coalesce_all("q10_web_all",
                                     _buyers(t, "q10", "web_sales", "ws",
                                             "ws_bill_customer_sk", _Q10_WINDOW)),
                      ["c_customer_sk"], ["ws_bill_customer_sk"], schema=schema)
        # The second mark join would produce a second `mark`, so the first is renamed here.
        renamed = rename("q10_web_mark", on_web,
                         [("c_customer_sk", "c_customer_sk")]
                         + [(column, column) for column in _DEMOGRAPHIC_COLUMNS]
                         + [("mark", "web_mark")])
        in_catalog = mark("q10_in_catalog", renamed,
                          N.coalesce_all("q10_catalog_all",
                                         _buyers(t, "q10", "catalog_sales", "cs",
                                                 "cs_ship_customer_sk", _Q10_WINDOW)),
                          ["c_customer_sk"], ["cs_ship_customer_sk"],
                          schema=dict(corpus.schema_of(
                              t("customer", ["c_customer_sk"]),
                              t("customer_demographics", list(_DEMOGRAPHIC_COLUMNS)),
                              web_mark="bool").dtypes))
        return N.filter_("q10_web_or_catalog", in_catalog,
                         Binary("or", Col("web_mark"), Col("mark")))

    final = _demographic_counts(t, "q10", "ca_county",
                                ("Rush County", "Toole County", "Jefferson County",
                                 "Dona Ana County", "La Porte County"),
                                _DEMOGRAPHIC_COLUMNS, combine)
    out = select("out", final, "cd_gender", "cd_marital_status", "cd_education_status", "cnt1",
                 "cd_purchase_estimate", "cnt2", "cd_credit_rating", "cnt3", "cd_dep_count",
                 "cnt4", "cd_dep_employed_count", "cnt5", "cd_dep_college_count", "cnt6")
    return N.unload(
        "unload",
        sorted_output("sort", out, list(_DEMOGRAPHIC_COLUMNS),
                      [True] * len(_DEMOGRAPHIC_COLUMNS), fetch=100),
    )


_Q69_COLUMNS = _DEMOGRAPHIC_COLUMNS[:5]


@query("q69", order_by=_Q69_COLUMNS)
def plan_q69(t):
    """q10's opposite: bought in store and **neither** on the web nor in the catalog.

    Three `AND`ed existence tests, so all three filter and none needs a mark — a semi join
    and two anti joins, chained, each one narrowing the build side the next one holds.
    """
    def combine(candidates, schema):
        bought = semi("q69_in_store", candidates,
                      N.coalesce_all("q69_store_all",
                                     _buyers(t, "q69", "store_sales", "ss", "ss_customer_sk",
                                             _Q69_WINDOW)),
                      ["c_customer_sk"], ["ss_customer_sk"], schema=schema)
        not_web = anti("q69_not_web", bought,
                       N.coalesce_all("q69_web_all",
                                      _buyers(t, "q69", "web_sales", "ws",
                                              "ws_bill_customer_sk", _Q69_WINDOW)),
                       ["c_customer_sk"], ["ws_bill_customer_sk"], schema=schema)
        return anti("q69_not_catalog", not_web,
                    N.coalesce_all("q69_catalog_all",
                                   _buyers(t, "q69", "catalog_sales", "cs",
                                           "cs_ship_customer_sk", _Q69_WINDOW)),
                    ["c_customer_sk"], ["cs_ship_customer_sk"], schema=schema)

    final = _demographic_counts(t, "q69", "ca_state", ("KY", "GA", "NM"),
                                _Q69_COLUMNS, combine)
    out = select("out", final, "cd_gender", "cd_marital_status", "cd_education_status", "cnt1",
                 "cd_purchase_estimate", "cnt2", "cd_credit_rating", "cnt3")
    return N.unload(
        "unload",
        sorted_output("sort", out, list(_Q69_COLUMNS), [True] * len(_Q69_COLUMNS), fetch=100),
    )


# -- a full outer join, which is the only way to count both sides at once ------------


@query("q97")
def plan_q97(t):
    """How many (customer, item) pairs bought in store only, in the catalog only, or both.

    The only query in the corpus that needs a **full** outer join, and it needs one because
    the three counts are over three different halves of the same relationship — a left join
    would answer two of them and lose the third. Full is build-preserving: unmatched probe
    rows come out as the probe streams, and unmatched build rows at the finish, which is the
    two-sided version of #136's pass.

    Both sides are `DISTINCT` pairs, which is a grouped aggregate with no aggregates, and
    that is what makes them small enough to join at all.
    """
    def pairs(tag, channel, prefix, customer_key, item_key, names):
        sales = fact(t, channel, [customer_key, item_key, f"{prefix}_sold_date_sk"],
                     tag=f"{tag}_{channel}")
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_month_seq"],
                       between(Col("d_month_seq"), Lit(1200), Lit(1200 + 11)),
                       tag=f"{tag}_date_dim")
        joined = star(sales[1], (f"{tag}_d", date_dim, "d_date_sk", f"{prefix}_sold_date_sk"))
        keyed = rename(f"{tag}_keys", joined,
                       [(customer_key, names[0]), (item_key, names[1])])
        return aggregate_by(f"{tag}_distinct", keyed, list(names), [],
                            schema_frame=corpus.schema_of(**{name: "int64" for name in names}))

    store = pairs("ssci", "store_sales", "ss", "ss_customer_sk", "ss_item_sk",
                  ("ss_customer_sk", "ss_item_sk"))
    catalog = pairs("csci", "catalog_sales", "cs", "cs_bill_customer_sk", "cs_item_sk",
                    ("cs_customer_sk", "cs_item_sk"))
    both = N.hash_join(
        "ssci_csci",
        N.coalesce_all("ssci_csci_build", catalog,
                       schema=dict(corpus.schema_of(cs_customer_sk="int64",
                                                    cs_item_sk="int64").dtypes)),
        store, JoinType.FULL,
        ["cs_customer_sk", "cs_item_sk"], ["ss_customer_sk", "ss_item_sk"],
        probe_schema=["ss_customer_sk", "ss_item_sk"],
    )
    in_store, in_catalog = IsNotNull(Col("ss_customer_sk")), IsNotNull(Col("cs_customer_sk"))
    sides = N.project(
        "sides", both,
        [Alias(Case(whens=((all_of(in_store, Not(in_catalog)), Lit(1)),), otherwise=Lit(0)),
               "store_only"),
         Alias(Case(whens=((all_of(Not(in_store), in_catalog), Lit(1)),), otherwise=Lit(0)),
               "catalog_only"),
         Alias(Case(whens=((all_of(in_store, in_catalog), Lit(1)),), otherwise=Lit(0)),
               "store_and_catalog")],
    )
    return N.unload(
        "unload",
        aggregate_to_one_row(
            "agg", sides,
            [A.Agg(A.SUM, "store_only", "store_only"),
             A.Agg(A.SUM, "catalog_only", "catalog_only"),
             A.Agg(A.SUM, "store_and_catalog", "store_and_catalog")],
            corpus.schema_of(store_only="int64", catalog_only="int64",
                             store_and_catalog="int64"),
        ),
    )


# -- an IN subquery inside an OR -----------------------------------------------------


_Q45_ZIPS = ("85669", "86197", "88274", "83405", "86475", "85392", "85460", "80348", "81792")
_Q45_ITEMS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


@query("q45", order_by=("ca_zip", "ca_city"))
def plan_q45(t):
    """Web revenue by zip, for nine zips **or** ten items' product ids.

    The `IN (SELECT …)` is under an `OR`, so — as in q10 — it cannot filter and becomes a
    **mark** join. The subquery is uncorrelated and tiny (ten item keys, a handful of
    distinct `i_item_id`s), which makes it the probe side and the quarter's web sales the
    build: the mark join preserves its build, and the build is what the query is reporting.
    """
    web_sales = fact(t, "web_sales",
                     ["ws_sold_date_sk", "ws_bill_customer_sk", "ws_item_sk", "ws_sales_price"])
    customer = dim(t, "customer", ["c_customer_sk", "c_current_addr_sk"])
    address = dim(t, "customer_address", ["ca_address_sk", "ca_zip", "ca_city"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_qoy", "d_year"],
                   all_of(Binary("==", Col("d_qoy"), Lit(2)),
                          Binary("==", Col("d_year"), Lit(2001))))
    item = dim(t, "item", ["i_item_sk", "i_item_id"])
    joined = star(
        web_sales[1],
        ("d_ws", date_dim, "d_date_sk", "ws_sold_date_sk"),
        ("c_ws", customer, "c_customer_sk", "ws_bill_customer_sk"),
        ("ca_c", address, "ca_address_sk", "c_current_addr_sk"),
        ("i_ws", item, "i_item_sk", "ws_item_sk"),
    )
    candidates = select("candidates", joined, "ca_zip", "ca_city", "i_item_id", "ws_sales_price")
    chosen = rename(
        "chosen_items",
        dim(t, "item", ["i_item_sk", "i_item_id"], is_in(Col("i_item_sk"), _Q45_ITEMS),
            tag="chosen_item")[1],
        [("i_item_id", "chosen_item_id")],
    )
    marked = mark(
        "item_in_list", candidates, N.coalesce_all("chosen_all", chosen),
        ["i_item_id"], ["chosen_item_id"],
        schema=dict(corpus.schema_of(address[0], item[0], web_sales[0]).dtypes),
    )
    kept = N.filter_(
        "zip_or_item", marked,
        Binary("or", is_in(Substring(Col("ca_zip"), 1, 5), _Q45_ZIPS), Col("mark")),
    )
    final = aggregate_by(
        "agg", select("keys", kept, "ca_zip", "ca_city", "ws_sales_price"),
        ["ca_zip", "ca_city"], [A.Agg(A.SUM, "ws_sales_price", "sum(ws_sales_price)")],
        schema_frame=corpus.schema_of(address[0], web_sales[0]),
    )
    return N.unload(
        "unload",
        sorted_output("sort", final, ["ca_zip", "ca_city"], [True, True], fetch=100),
    )


_Q35_KEYS = ("ca_state", "cd_gender", "cd_marital_status", "cd_dep_count",
             "cd_dep_employed_count", "cd_dep_college_count")
#: q35 reports count/min/max/avg of each dependant count. The third set is never aliased, so
#: its columns are named after the expressions themselves.
_Q35_MEASURES = (("cd_dep_count", "cnt1", "min1", "max1", "avg1"),
                 ("cd_dep_employed_count", "cnt2", "min2", "max2", "avg2"),
                 ("cd_dep_college_count", "cnt3", "min(cd_dep_college_count)",
                  "max(cd_dep_college_count)", "avg(cd_dep_college_count)"))


@query("q35", order_by=_Q35_KEYS)
def plan_q35(t):
    """q10's existence tests, over 2002's first three quarters, reporting four statistics of
    each dependant count.

    Same three joins — a semi join and two marks under an `OR` — and a different aggregate:
    twelve accumulators over six group keys, three of which are also the columns being
    summarized. A column can be both a group key and an aggregate input, and this is the
    query that says so.
    """
    def combine(candidates, schema):
        bought = semi("q35_in_store", candidates,
                      N.coalesce_all("q35_store_all",
                                     _buyers(t, "q35", "store_sales", "ss", "ss_customer_sk",
                                             _Q35_WINDOW)),
                      ["c_customer_sk"], ["ss_customer_sk"], schema=schema)
        on_web = mark("q35_on_web", bought,
                      N.coalesce_all("q35_web_all",
                                     _buyers(t, "q35", "web_sales", "ws",
                                             "ws_bill_customer_sk", _Q35_WINDOW)),
                      ["c_customer_sk"], ["ws_bill_customer_sk"], schema=schema)
        renamed = rename("q35_web_mark", on_web,
                         [("c_customer_sk", "c_customer_sk"), ("ca_state", "ca_state")]
                         + [(column, column) for column in _Q35_KEYS[1:]]
                         + [("mark", "web_mark")])
        in_catalog = mark("q35_in_catalog", renamed,
                          N.coalesce_all("q35_catalog_all",
                                         _buyers(t, "q35", "catalog_sales", "cs",
                                                 "cs_ship_customer_sk", _Q35_WINDOW)),
                          ["c_customer_sk"], ["cs_ship_customer_sk"],
                          schema=dict(corpus.schema_of(
                              t("customer", ["c_customer_sk"]),
                              t("customer_address", ["ca_state"]),
                              t("customer_demographics", list(_Q35_KEYS[1:])),
                              web_mark="bool").dtypes))
        return N.filter_("q35_web_or_catalog", in_catalog,
                         Binary("or", Col("web_mark"), Col("mark")))

    customer = fact(t, "customer", ["c_customer_sk", "c_current_addr_sk",
                                    "c_current_cdemo_sk"], tag="q35_customer")
    address = dim(t, "customer_address", ["ca_address_sk", "ca_state"],
                  tag="q35_customer_address")
    demographics = fact(t, "customer_demographics",
                        ["cd_demo_sk"] + list(_Q35_KEYS[1:]), tag="q35_customer_demographics")
    local = star(
        customer[1],
        ("q35_ca", address, "ca_address_sk", "c_current_addr_sk"),
        ("q35_cd", demographics, "cd_demo_sk", "c_current_cdemo_sk"),
    )
    candidates = select("q35_candidates", local, "c_customer_sk", *_Q35_KEYS)
    kept = combine(candidates, dict(corpus.schema_of(customer[0], address[0],
                                                     demographics[0]).dtypes))
    aggs = []
    for column, count, low, high, average in _Q35_MEASURES:
        aggs += [A.Agg(A.COUNT, None, count), A.Agg(A.MIN, column, low),
                 A.Agg(A.MAX, column, high), A.Agg(A.MEAN, column, average)]
    final = aggregate_by(
        "q35_agg", select("q35_keys", kept, *_Q35_KEYS), list(_Q35_KEYS), aggs,
        schema_frame=corpus.schema_of(address[0], demographics[0]),
    )
    reported = ["ca_state", "cd_gender", "cd_marital_status"]
    for column, count, low, high, average in _Q35_MEASURES:
        reported += [column, count, low, high, average]
    out = select("out", final, *reported)
    return N.unload(
        "unload",
        sorted_output("sort", out, list(_Q35_KEYS), [True] * len(_Q35_KEYS), fetch=100,
                      nulls_first=True),
    )


#: The four hundred zip codes q8 names literally, taken from the query text.
_Q8_ZIPS = (
    "24128", "76232", "65084", "87816", "83926", "77556", "20548", "26231", "43848", "15126",
    "91137", "61265", "98294", "25782", "17920", "18426", "98235", "40081", "84093", "28577",
    "55565", "17183", "54601", "67897", "22752", "86284", "18376", "38607", "45200", "21756",
    "29741", "96765", "23932", "89360", "29839", "25989", "28898", "91068", "72550", "10390",
    "18845", "47770", "82636", "41367", "76638", "86198", "81312", "37126", "39192", "88424",
    "72175", "81426", "53672", "10445", "42666", "66864", "66708", "41248", "48583", "82276",
    "18842", "78890", "49448", "14089", "38122", "34425", "79077", "19849", "43285", "39861",
    "66162", "77610", "13695", "99543", "83444", "83041", "12305", "57665", "68341", "25003",
    "57834", "62878", "49130", "81096", "18840", "27700", "23470", "50412", "21195", "16021",
    "76107", "71954", "68309", "18119", "98359", "64544", "10336", "86379", "27068", "39736",
    "98569", "28915", "24206", "56529", "57647", "54917", "42961", "91110", "63981", "14922",
    "36420", "23006", "67467", "32754", "30903", "20260", "31671", "51798", "72325", "85816",
    "68621", "13955", "36446", "41766", "68806", "16725", "15146", "22744", "35850", "88086",
    "51649", "18270", "52867", "39972", "96976", "63792", "11376", "94898", "13595", "10516",
    "90225", "58943", "39371", "94945", "28587", "96576", "57855", "28488", "26105", "83933",
    "25858", "34322", "44438", "73171", "30122", "34102", "22685", "71256", "78451", "54364",
    "13354", "45375", "40558", "56458", "28286", "45266", "47305", "69399", "83921", "26233",
    "11101", "15371", "69913", "35942", "15882", "25631", "24610", "44165", "99076", "33786",
    "70738", "26653", "14328", "72305", "62496", "22152", "10144", "64147", "48425", "14663",
    "21076", "18799", "30450", "63089", "81019", "68893", "24996", "51200", "51211", "45692",
    "92712", "70466", "79994", "22437", "25280", "38935", "71791", "73134", "56571", "14060",
    "19505", "72425", "56575", "74351", "68786", "51650", "20004", "18383", "76614", "11634",
    "18906", "15765", "41368", "73241", "76698", "78567", "97189", "28545", "76231", "75691",
    "22246", "51061", "90578", "56691", "68014", "51103", "94167", "57047", "14867", "73520",
    "15734", "63435", "25733", "35474", "24676", "94627", "53535", "17879", "15559", "53268",
    "59166", "11928", "59402", "33282", "45721", "43933", "68101", "33515", "36634", "71286",
    "19736", "58058", "55253", "67473", "41918", "19515", "36495", "19430", "22351", "77191",
    "91393", "49156", "50298", "87501", "18652", "53179", "18767", "63193", "23968", "65164",
    "68880", "21286", "72823", "58470", "67301", "13394", "31016", "70372", "67030", "40604",
    "24317", "45748", "39127", "26065", "77721", "31029", "31880", "60576", "24671", "45549",
    "13376", "50016", "33123", "19769", "22927", "97789", "46081", "72151", "15723", "46136",
    "51949", "68100", "96888", "64528", "14171", "79777", "28709", "11489", "25103", "32213",
    "78668", "22245", "15798", "27156", "37930", "62971", "21337", "51622", "67853", "10567",
    "38415", "15455", "58263", "42029", "60279", "37125", "56240", "88190", "50308", "26859",
    "64457", "89091", "82136", "62377", "36233", "63837", "58078", "17043", "30010", "60099",
    "28810", "98025", "29178", "87343", "73273", "30469", "64034", "39516", "86057", "21309",
    "90257", "67875", "40162", "11356", "73650", "61810", "72013", "30431", "22461", "19512",
    "13375", "55307", "30625", "83849", "68908", "26689", "96451", "38193", "46820", "88885",
    "84935", "69035", "83144", "47537", "56616", "94983", "48033", "69952", "25486", "61547",
    "27385", "61860", "58048", "56910", "16807", "17871", "35258", "31387", "35458", "35576",
)


# -- an INTERSECT, and a join on a two-character prefix ------------------------------


@query("q8", order_by=("s_store_name",))
def plan_q8(t):
    """Store profit in one quarter, for stores whose zip shares a prefix with a chosen zip.

    Two things worth having. The `INTERSECT` is a **semi join over distinct sides** — the
    four hundred literal zips on one side, the zips with more than ten preferred customers
    on the other — which is the set operation the mode does not have a node for and does not
    need one for.

    And the join to the store is on `substr(zip, 1, 2)`, two characters, so it is an
    equi-join on a computed column with a fanout: one sale matches every chosen zip whose
    first two digits agree, and the query means that — the sum counts a sale once per
    matching zip.
    """
    listed = dim(t, "customer_address", ["ca_zip"],
                 is_in(Substring(Col("ca_zip"), 1, 5), _Q8_ZIPS), tag="q8_listed_address")
    listed_zips = aggregate_by(
        "q8_listed", N.project("q8_listed_prefix", listed[1],
                               [Alias(Substring(Col("ca_zip"), 1, 5), "ca_zip")]),
        ["ca_zip"], [], schema_frame=corpus.schema_of(ca_zip="object"),
    )
    address = fact(t, "customer_address", ["ca_address_sk", "ca_zip"],
                   tag="q8_preferred_address")
    customer = dim(t, "customer", ["c_customer_sk", "c_current_addr_sk",
                                   "c_preferred_cust_flag"],
                   Binary("==", Col("c_preferred_cust_flag"), Lit("Y")), rows=250_000,
                   tag="q8_customer")
    preferred = star(address[1], ("q8_c_ca", customer, "c_current_addr_sk", "ca_address_sk"))
    counted = aggregate_by(
        "q8_preferred",
        N.project("q8_preferred_prefix", preferred,
                  [Alias(Substring(Col("ca_zip"), 1, 5), "popular_zip")]),
        ["popular_zip"], [A.Agg(A.COUNT, None, "cnt")],
        schema_frame=corpus.schema_of(popular_zip="object"),
    )
    popular = N.filter_("q8_more_than_ten", counted, Binary(">", Col("cnt"), Lit(10)))
    chosen = semi("q8_intersect", listed_zips, N.coalesce_all("q8_popular_all", popular),
                  ["ca_zip"], ["popular_zip"],
                  schema=dict(corpus.schema_of(ca_zip="object").dtypes))
    prefixes = N.project("q8_prefix", chosen,
                         [Alias(Substring(Col("ca_zip"), 1, 2), "zip_prefix")])
    store_sales = fact(t, "store_sales", ["ss_store_sk", "ss_sold_date_sk", "ss_net_profit"],
                       tag="q8_store_sales")
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_qoy", "d_year"],
                   all_of(Binary("==", Col("d_qoy"), Lit(2)),
                          Binary("==", Col("d_year"), Lit(1998))), tag="q8_date_dim")
    store = dim(t, "store", ["s_store_sk", "s_store_name", "s_zip"], tag="q8_store")
    sold = N.project(
        "q8_sold",
        star(store_sales[1],
             ("q8_d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
             ("q8_s_ss", store, "s_store_sk", "ss_store_sk")),
        [Alias(Col("s_store_name"), "s_store_name"),
         Alias(Substring(Col("s_zip"), 1, 2), "store_prefix"),
         Alias(Col("ss_net_profit"), "ss_net_profit")],
    )
    matched = N.hash_join(
        "q8_zip_prefix",
        N.coalesce_all("q8_prefix_all", prefixes,
                       schema=dict(corpus.schema_of(zip_prefix="object").dtypes)),
        sold, JoinType.INNER, ["zip_prefix"], ["store_prefix"],
    )
    final = aggregate_by(
        "q8_agg", select("q8_keys", matched, "s_store_name", "ss_net_profit"),
        ["s_store_name"], [A.Agg(A.SUM, "ss_net_profit", "sum(ss_net_profit)")],
        schema_frame=corpus.schema_of(store[0], store_sales[0]),
    )
    return N.unload(
        "unload", sorted_output("sort", final, ["s_store_name"], [True], fetch=100)
    )
