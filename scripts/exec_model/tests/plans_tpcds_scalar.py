"""TPC-DS: the correlated subqueries — an aggregate joined back to what it summarized.

`x > (SELECT avg(x) FROM … WHERE key = outer.key)` has no operator. What it has is a
lowering, and it is the same one every time: aggregate the inner relation by the correlation
key, then **join that back** to the outer stream on the same key and compare two columns of
one row. The subquery becomes a build side of a few rows, and the outer relation never stops
streaming.

That is also why several of these read a table twice. The correlated relation appears once
as the thing being summarized and once as the thing being filtered, and a plan is a tree, so
the subtree is built twice — which is what DataFusion does with a repeated CTE too (#101).
The prototype's cost model sees both, honestly.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from . import corpus
from .plan_helpers import (
    aggregate_by, all_of, any_of, between, date, is_in, rename, select, sorted_output,
)
from .plans_tpcds_common import dim, fact, registry, star
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import Alias, Binary, Col, Lit
from ..operators.join_types import JoinType

QUERIES, ORDER_BY, query = registry()


def _threshold(name, summary, key, column, factor, output="threshold"):
    """`avg(column) * factor` per `key`, as a build side of one row per key.

    The shape the whole module turns on. It is built from the same subtree the outer query
    streams, and it is small — one row per correlation key — so it is the build side and the
    comparison above the join is two columns of one row.
    """
    averaged = aggregate_by(f"{name}_avg", summary, [key], [A.Agg(A.MEAN, column, "average")],
                            schema_frame=corpus.schema_of(**{key: "int64", column: "float64"}))
    return N.project(
        f"{name}_threshold", averaged,
        [Alias(Col(key), key), Alias(Binary("*", Col("average"), Lit(factor)), output)],
    )


# -- returns above the local average ------------------------------------------------


def _customer_total_return(t, tag, returns_table, keys, date_year, amount, by_state):
    """A year of one channel's returns, totalled per (customer, store) or (customer, state).

    q1, q30 and q81's `customer_total_return` CTE. Built twice per query — once to be
    averaged and once to be filtered — because a plan is a tree; the tag keeps the two
    copies' node names apart.
    """
    customer_key, place_key = keys
    # sr_ / wr_ / cr_ — taken from the key rather than from the table name, which does not
    # spell it (`store_returns` columns are `sr_`, not `st_`).
    prefix = customer_key.split("_")[0]
    returns = fact(t, returns_table,
                   [customer_key, place_key, f"{prefix}_returned_date_sk", amount],
                   tag=f"{tag}_{returns_table}")
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"],
                   Binary("==", Col("d_year"), Lit(date_year)), tag=f"{tag}_date_dim")
    joined = star(returns[1], (f"{tag}_d", date_dim, "d_date_sk",
                               f"{prefix}_returned_date_sk"))
    if by_state:
        address = dim(t, "customer_address", ["ca_address_sk", "ca_state"],
                      tag=f"{tag}_customer_address")
        joined = star(joined, (f"{tag}_ca", address, "ca_address_sk", place_key))
        place = "ca_state"
    else:
        place = place_key
    grouped = rename(f"{tag}_ctr_keys", joined,
                     [(customer_key, "ctr_customer_sk"), (place, "ctr_place"), (amount, amount)])
    return aggregate_by(
        f"{tag}_ctr", grouped, ["ctr_customer_sk", "ctr_place"],
        [A.Agg(A.SUM, amount, "ctr_total_return")],
        schema_frame=corpus.schema_of(returns[0],
                                      ctr_customer_sk="int64",
                                      ctr_place="object" if by_state else "int64"),
    )


def _above_local_average(t, tag, returns_table, keys, year, amount, by_state, carried):
    """The rows of `customer_total_return` above 1.2× their own place's average."""
    def ctr(copy):
        return _customer_total_return(t, f"{tag}{copy}", returns_table, keys, year, amount,
                                      by_state)

    threshold = _threshold(tag, ctr("_avg"), "ctr_place", "ctr_total_return", 1.2)
    against = N.hash_join(
        f"{tag}_ctr_threshold",
        N.coalesce_all(f"{tag}_ctr_threshold_build", threshold), ctr("_rows"),
        JoinType.INNER, ["ctr_place"], ["ctr_place"],
    )
    above = N.filter_(f"{tag}_above_average", against,
                      Binary(">", Col("ctr_total_return"), Col("threshold")))
    return select(f"{tag}_above_keys", above, "ctr_customer_sk", "ctr_place",
                  "ctr_total_return", *carried)


@query("q1", order_by=("c_customer_id",))
def plan_q1(t):
    """Customers whose returns to a Tennessee store beat that store's average by 20%.

    The correlation key is the store, of which there are twelve, so the subquery's whole
    result is twelve rows — the clearest case in the benchmark of a subquery that becomes a
    build side rather than a re-execution.
    """
    above = _above_local_average(
        t, "q1", "store_returns", ("sr_customer_sk", "sr_store_sk"), 2000, "sr_return_amt",
        by_state=False, carried=(),
    )
    store = dim(t, "store", ["s_store_sk", "s_state"],
                Binary("==", Col("s_state"), Lit("TN")))
    in_tn = star(above, ("s_ctr", store, "s_store_sk", "ctr_place"))
    customer = fact(t, "customer", ["c_customer_sk", "c_customer_id"])
    joined = N.hash_join(
        "c_ctr",
        N.coalesce_all("c_ctr_build", in_tn,
                       schema=dict(corpus.schema_of(
                           store[0], ctr_customer_sk="int64", ctr_place="int64",
                           ctr_total_return="float64").dtypes)),
        customer[1], JoinType.INNER, ["ctr_customer_sk"], ["c_customer_sk"],
    )
    out = select("out", joined, "c_customer_id")
    return N.unload("unload", sorted_output("sort", out, ["c_customer_id"], [True], fetch=100))


_Q30_COLUMNS = ("c_customer_id", "c_salutation", "c_first_name", "c_last_name",
                "c_preferred_cust_flag", "c_birth_day", "c_birth_month", "c_birth_year",
                "c_birth_country", "c_login", "c_email_address", "c_last_review_date_sk")


@query("q30", order_by=_Q30_COLUMNS + ("ctr_total_return",))
def plan_q30(t):
    """q1 by state rather than by store, over web returns, reporting the whole customer row.

    Note which `ca_state` is which: the CTE groups by the state the goods were returned
    *from*, and the outer query filters on the state the customer lives in now. They are
    different rows of customer_address reached by different keys, which is why the CTE
    renames its copy to `ctr_place` before anything else can collide with it.
    """
    above = _above_local_average(
        t, "q30", "web_returns", ("wr_returning_customer_sk", "wr_returning_addr_sk"),
        2002, "wr_return_amt", by_state=True, carried=(),
    )
    customer = fact(t, "customer", ["c_customer_sk", "c_current_addr_sk"] + list(_Q30_COLUMNS))
    address = dim(t, "customer_address", ["ca_address_sk", "ca_state"],
                  Binary("==", Col("ca_state"), Lit("GA")), tag="q30_current_address")
    in_ga = star(customer[1], ("ca_c", address, "ca_address_sk", "c_current_addr_sk"))
    joined = N.hash_join(
        "c_ctr",
        N.coalesce_all("c_ctr_build", above,
                       schema=dict(corpus.schema_of(
                           ctr_customer_sk="int64", ctr_place="object",
                           ctr_total_return="float64").dtypes)),
        in_ga, JoinType.INNER, ["ctr_customer_sk"], ["c_customer_sk"],
    )
    out = select("out", joined, *_Q30_COLUMNS, "ctr_total_return")
    return N.unload(
        "unload",
        sorted_output("sort", out, list(_Q30_COLUMNS) + ["ctr_total_return"],
                      [True] * (len(_Q30_COLUMNS) + 1), fetch=100, nulls_first=True),
    )


_Q81_ADDRESS = ("ca_street_number", "ca_street_name", "ca_street_type", "ca_suite_number",
                "ca_city", "ca_county", "ca_state", "ca_zip", "ca_country", "ca_gmt_offset",
                "ca_location_type")
_Q81_COLUMNS = ("c_customer_id", "c_salutation", "c_first_name", "c_last_name") + _Q81_ADDRESS


@query("q81", order_by=_Q81_COLUMNS + ("ctr_total_return",))
def plan_q81(t):
    """q30 over catalog returns, reporting the address rather than the person. Same
    lowering; the difference is that the outer query keeps eleven address columns, so the
    address is joined for its data and not only for its predicate."""
    above = _above_local_average(
        t, "q81", "catalog_returns", ("cr_returning_customer_sk", "cr_returning_addr_sk"),
        2000, "cr_return_amt_inc_tax", by_state=True, carried=(),
    )
    customer = fact(t, "customer",
                    ["c_customer_sk", "c_current_addr_sk", "c_customer_id", "c_salutation",
                     "c_first_name", "c_last_name"])
    address = dim(t, "customer_address", ["ca_address_sk"] + list(_Q81_ADDRESS),
                  Binary("==", Col("ca_state"), Lit("GA")), tag="q81_current_address")
    in_ga = star(customer[1], ("ca_c", address, "ca_address_sk", "c_current_addr_sk"))
    joined = N.hash_join(
        "c_ctr",
        N.coalesce_all("c_ctr_build", above,
                       schema=dict(corpus.schema_of(
                           ctr_customer_sk="int64", ctr_place="object",
                           ctr_total_return="float64").dtypes)),
        in_ga, JoinType.INNER, ["ctr_customer_sk"], ["c_customer_sk"],
    )
    out = select("out", joined, *_Q81_COLUMNS, "ctr_total_return")
    return N.unload(
        "unload",
        sorted_output("sort", out, list(_Q81_COLUMNS) + ["ctr_total_return"],
                      [True] * (len(_Q81_COLUMNS) + 1), fetch=100),
    )


# -- a scalar that is one row, and one that is one row per category ------------------


@query("q6", order_by=("cnt", "state"))
def plan_q6(t):
    """Where the customers are who bought something expensive for its category.

    Two subqueries of different kinds. The month is **uncorrelated** — one value for the
    whole query — so it lowers to a one-row build side joined to date_dim on `d_month_seq`,
    which is how a scalar subquery becomes a join when there is nothing to correlate on.
    The price is **correlated on the category**, so it lowers to the module's usual shape: an
    average per category, joined back to item.

    `HAVING count(*) >= 10` is a filter above the aggregate, where a HAVING always ends up.
    """
    month = aggregate_by(
        "month", select("month_keys",
                        dim(t, "date_dim", ["d_date_sk", "d_year", "d_moy", "d_month_seq"],
                            all_of(Binary("==", Col("d_year"), Lit(2001)),
                                   Binary("==", Col("d_moy"), Lit(1))),
                            tag="month_date_dim")[1],
                        "d_month_seq"),
        ["d_month_seq"], [],
        schema_frame=corpus.schema_of(d_month_seq="int64"),
    )
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_month_seq"])
    in_month = N.hash_join(
        "d_month",
        N.coalesce_all("d_month_build", month,
                       schema=dict(corpus.schema_of(d_month_seq="int64").dtypes)),
        date_dim[1], JoinType.INNER, ["d_month_seq"], ["d_month_seq"],
    )
    item = dim(t, "item", ["i_item_sk", "i_category", "i_current_price"], tag="item_rows")
    by_category = aggregate_by(
        "category_avg",
        select("category_keys",
               dim(t, "item", ["i_category", "i_current_price"], tag="item_avg")[1],
               "i_category", "i_current_price"),
        ["i_category"], [A.Agg(A.MEAN, "i_current_price", "average")],
        schema_frame=corpus.schema_of(i_category="object", i_current_price="float64"),
    )
    threshold = N.project(
        "category_threshold", by_category,
        [Alias(Col("i_category"), "i_category"),
         Alias(Binary("*", Col("average"), Lit(1.2)), "threshold")],
    )
    priced = N.filter_(
        "expensive",
        N.hash_join("i_threshold",
                    N.coalesce_all("i_threshold_build", threshold), item[1],
                    JoinType.INNER, ["i_category"], ["i_category"]),
        Binary(">", Col("i_current_price"), Col("threshold")),
    )
    store_sales = fact(t, "store_sales", ["ss_sold_date_sk", "ss_customer_sk", "ss_item_sk"])
    customer = dim(t, "customer", ["c_customer_sk", "c_current_addr_sk"])
    address = dim(t, "customer_address", ["ca_address_sk", "ca_state"])
    joined = star(
        store_sales[1],
        ("d_ss", (corpus.schema_of(date_dim[0], d_month_seq="int64"), in_month),
         "d_date_sk", "ss_sold_date_sk"),
        ("i_ss", (corpus.schema_of(item[0], threshold="float64"), priced),
         "i_item_sk", "ss_item_sk"),
        ("c_ss", customer, "c_customer_sk", "ss_customer_sk"),
        ("ca_c", address, "ca_address_sk", "c_current_addr_sk"),
    )
    counted = aggregate_by(
        "agg", rename("state_keys", joined, [("ca_state", "state")]), ["state"],
        [A.Agg(A.COUNT, None, "cnt")], schema_frame=corpus.schema_of(state="object"),
    )
    popular = N.filter_("at_least_ten", counted, Binary(">=", Col("cnt"), Lit(10)))
    out = select("out", popular, "state", "cnt")
    return N.unload(
        "unload",
        sorted_output("sort", out, ["cnt", "state"], [True, True], fetch=100,
                      nulls_first=True),
    )


# -- discounts above the item's own average -----------------------------------------


def _excess_discount(t, tag, channel, item_key, sold_key, discount, manufact, output):
    """One channel's discounts that beat 1.3× the item's own average over the same quarter.

    The subquery's `WHERE cs_item_sk = i_item_sk` correlates it to the outer item, so the
    average is per item — 18,000 rows, which is a build side. The date range is repeated in
    both halves and is *not* the correlation, which is easy to misread: the average is over
    the quarter, not over all time.
    """
    quarter = (date("2000-01-27"), date("2000-04-26"))

    def in_quarter(copy, columns):
        sales = fact(t, channel, columns, tag=f"{tag}_{copy}_{channel}")
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_date"],
                       between(Col("d_date"), *quarter), tag=f"{tag}_{copy}_date_dim")
        return sales, star(sales[1], (f"{tag}_{copy}_d", date_dim, "d_date_sk", sold_key))

    _, averaged_over = in_quarter("avg", [item_key, sold_key, discount])
    threshold = _threshold(tag, select(f"{tag}_avg_keys", averaged_over, item_key, discount),
                           item_key, discount, 1.3)
    sales, stream = in_quarter("rows", [item_key, sold_key, discount])
    item = dim(t, "item", ["i_item_sk", "i_manufact_id"],
               Binary("==", Col("i_manufact_id"), Lit(manufact)), tag=f"{tag}_item")
    of_manufacturer = star(stream, (f"{tag}_i", item, "i_item_sk", item_key))
    against = N.hash_join(
        f"{tag}_threshold_join",
        N.coalesce_all(f"{tag}_threshold_build", threshold), of_manufacturer,
        JoinType.INNER, [item_key], [item_key],
    )
    excess = N.filter_(f"{tag}_excess", against,
                       Binary(">", Col(discount), Col("threshold")))
    from .plan_helpers import aggregate_to_one_row

    return N.unload(
        "unload",
        aggregate_to_one_row(f"{tag}_agg", select(f"{tag}_keys", excess, discount),
                             [A.Agg(A.SUM, discount, output)],
                             corpus.schema_of(sales[0])),
    )


@query("q32")
def plan_q32(t):
    """Catalog discounts on one manufacturer's items that beat the item's own average."""
    return _excess_discount(t, "q32", "catalog_sales", "cs_item_sk", "cs_sold_date_sk",
                            "cs_ext_discount_amt", 977, "excess discount amount")


@query("q92", order_by=("Excess Discount Amount",))
def plan_q92(t):
    """q32 over the web channel. The only difference that reaches the plan is the table."""
    return _excess_discount(t, "q92", "web_sales", "ws_item_sk", "ws_sold_date_sk",
                            "ws_ext_discount_amt", 350, "Excess Discount Amount")


# -- an average of an aggregate -----------------------------------------------------


@query("q65", order_by=("s_store_name", "i_item_desc"))
def plan_q65(t):
    """Items selling at under a tenth of their store's average item revenue.

    Two aggregates stacked: revenue per (store, item), and then the **average of that** per
    store — so the second one's input is the first one's output and not the fact table. The
    inner aggregate is then needed again as rows, so it is built twice, which is what a
    repeated CTE costs when a plan is a tree.
    """
    def revenue(copy):
        store_sales = fact(t, "store_sales",
                           ["ss_sold_date_sk", "ss_store_sk", "ss_item_sk", "ss_sales_price"],
                           tag=f"{copy}_store_sales")
        date_dim = dim(t, "date_dim", ["d_date_sk", "d_month_seq"],
                       between(Col("d_month_seq"), Lit(1176), Lit(1176 + 11)),
                       tag=f"{copy}_date_dim")
        joined = star(store_sales[1], (f"{copy}_d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"))
        return aggregate_by(
            f"{copy}_revenue",
            select(f"{copy}_keys", joined, "ss_store_sk", "ss_item_sk", "ss_sales_price"),
            ["ss_store_sk", "ss_item_sk"], [A.Agg(A.SUM, "ss_sales_price", "revenue")],
            schema_frame=corpus.schema_of(store_sales[0]),
        )

    per_store = aggregate_by(
        "store_average", select("average_keys", revenue("sa"), "ss_store_sk", "revenue"),
        ["ss_store_sk"], [A.Agg(A.MEAN, "revenue", "ave")],
        schema_frame=corpus.schema_of(ss_store_sk="int64", revenue="float64"),
    )
    against = N.hash_join(
        "sb_sc", N.coalesce_all("sb_sc_build", per_store), revenue("sc"),
        JoinType.INNER, ["ss_store_sk"], ["ss_store_sk"],
    )
    slow = N.filter_("under_a_tenth", against,
                     Binary("<=", Col("revenue"), Binary("*", Lit(0.1), Col("ave"))))
    store = dim(t, "store", ["s_store_sk", "s_store_name"])
    item = dim(t, "item", ["i_item_sk", "i_item_desc", "i_current_price", "i_wholesale_cost",
                           "i_brand"])
    described = star(
        slow,
        ("s_sc", store, "s_store_sk", "ss_store_sk"),
        ("i_sc", item, "i_item_sk", "ss_item_sk"),
    )
    out = select("out", described, "s_store_name", "i_item_desc", "revenue", "i_current_price",
                 "i_wholesale_cost", "i_brand")
    return N.unload(
        "unload",
        sorted_output("sort", out, ["s_store_name", "i_item_desc"], [True, True], fetch=100,
                      nulls_first=True),
    )


# -- a correlated count, which is an existence test ----------------------------------


#: q41's eight arms, each `(category, colours, units, sizes)`. The query writes them as two
#: groups of four inside one `OR`, which is the same set.
_Q41_ARMS = (
    ("Women", ("powder", "khaki"), ("Ounce", "Oz"), ("medium", "extra large")),
    ("Women", ("brown", "honeydew"), ("Bunch", "Ton"), ("N/A", "small")),
    ("Men", ("floral", "deep"), ("N/A", "Dozen"), ("petite", "petite")),
    ("Men", ("light", "cornflower"), ("Box", "Pound"), ("medium", "extra large")),
    ("Women", ("midnight", "snow"), ("Pallet", "Gross"), ("medium", "extra large")),
    ("Women", ("cyan", "papaya"), ("Cup", "Dram"), ("N/A", "small")),
    ("Men", ("orange", "frosted"), ("Each", "Tbl"), ("petite", "petite")),
    ("Men", ("forest", "ghost"), ("Lb", "Bundle"), ("medium", "extra large")),
)


@query("q41", order_by=("i_product_name",))
def plan_q41(t):
    """Products from manufacturers who make at least one item matching any of eight
    descriptions.

    The subquery is a `count(*)` correlated on the manufacturer and compared `> 0`, which is
    an existence test written the long way — so it lowers to a **semi join** on
    `i_manufact` and no count is ever computed. A group exists exactly when its count is
    positive, and the finish pass that a Left semi join runs is exactly the "did this build
    row ever match" question.

    Both sides are `item`, read twice with different predicates, so the manufacturers'
    side is renamed before the join.
    """
    matching = dim(t, "item",
                   ["i_manufact", "i_category", "i_color", "i_units", "i_size"],
                   any_of(*[all_of(Binary("==", Col("i_category"), Lit(category)),
                                   is_in(Col("i_color"), colours),
                                   is_in(Col("i_units"), units),
                                   is_in(Col("i_size"), sizes))
                            for category, colours, units, sizes in _Q41_ARMS]),
                   tag="described_item")
    manufacturers = rename("manufacturers", matching[1], [("i_manufact", "matched_manufact")])
    i1 = dim(t, "item", ["i_manufact", "i_manufact_id", "i_product_name"],
             between(Col("i_manufact_id"), Lit(738), Lit(738 + 40)), tag="i1")
    made = N.hash_join(
        "i1_manufact",
        N.coalesce_all("i1_manufact_build", i1[1], schema=dict(i1[0].dtypes)),
        manufacturers, JoinType.LEFT_SEMI, ["i_manufact"], ["matched_manufact"],
    )
    distinct = aggregate_by("distinct", select("names", made, "i_product_name"),
                            ["i_product_name"], [],
                            schema_frame=corpus.schema_of(i1[0]))
    return N.unload(
        "unload", sorted_output("sort", distinct, ["i_product_name"], [True], fetch=100)
    )
