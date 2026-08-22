"""TPC-DS: the banded queries — a disjunction that no scan can hold.

q13, q48 and q85 are one query written three times. Each pairs three *demographic* bands
with three *geographic* ones, and every arm of every disjunction reads one column from a
dimension and one from the fact row it qualifies — a marital status and a sale price, a
state and a net profit. So neither disjunction can be pushed to either table's scan, and
both have to be evaluated above the joins.

What *can* be pushed is the union of each disjunction's dimension arms: the three (marital
status, education) pairs, and the nine states. That is the pushdown a planner is entitled
to, it is what these lowerings do, and it is not the same thing as the predicate — the full
disjunction is still applied above the join, because dropping it would let a sale in Ohio
match Oregon's profit band.

The answer is one row: no group key, no ORDER BY, three averages and a sum.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from . import corpus
from .plan_helpers import (
    aggregate_by, aggregate_to_one_row, all_of, any_of, between, is_in, select, sorted_output,
)
from .plans_tpcds_common import dim, fact, registry, star
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import Alias, Binary, Col, Lit, Substring
from ..operators.join_types import JoinType

QUERIES, ORDER_BY, query = registry()

#: The nine states, in the three groups each of which carries its own profit band.
_REGIONS_Q13 = (("TX", "OH", "TX", 100, 200), ("OR", "NM", "KY", 150, 300),
                ("VA", "TX", "MS", 50, 250))
_REGIONS_Q48 = (("CO", "OH", "TX", 0, 2000), ("OR", "MN", "KY", 150, 3000),
                ("VA", "CA", "MS", 50, 25000))
_REGIONS_Q85 = (("IN", "OH", "NJ", 100, 200), ("WI", "CT", "KY", 150, 300),
                ("LA", "IA", "AR", 50, 250))


def _states(regions):
    """Every state named by any arm, for the scan that can be narrowed."""
    return [state for region in regions for state in region[:3]]


def _region_predicate(regions, profit):
    """`(state IN … AND profit BETWEEN …) OR …` — the part no scan can hold."""
    return any_of(*[all_of(is_in(Col("ca_state"), states),
                           between(Col(profit), Lit(low), Lit(high)))
                    for *states, low, high in regions])


def _address(t, regions, tag="customer_address"):
    return dim(t, "customer_address", ["ca_address_sk", "ca_state", "ca_country"],
               all_of(Binary("==", Col("ca_country"), Lit("United States")),
                      is_in(Col("ca_state"), _states(regions))), tag=tag)


def _segment_predicate(bands, price, status="cd_marital_status", education="cd_education_status",
                       extra=None):
    """`(status = … AND education = … AND price BETWEEN …) OR …`, plus any per-arm extra."""
    arms = []
    for index, (marital, degree, low, high) in enumerate(bands):
        tests = [Binary("==", Col(status), Lit(marital)),
                 Binary("==", Col(education), Lit(degree)),
                 between(Col(price), Lit(low), Lit(high))]
        if extra is not None:
            tests.append(extra(index))
        arms.append(all_of(*tests))
    return any_of(*arms)


def _segments(t, bands, tag="customer_demographics"):
    """customer_demographics narrowed to the (marital status, education) pairs any arm names.

    1.9M rows, so it is declared a fact and streams through its filter: scanned as one batch
    it is 730 MB of python strings and the resident enforcer trips, which is the enforcer
    being right on real data.
    """
    demographics = fact(t, "customer_demographics",
                        ["cd_demo_sk", "cd_marital_status", "cd_education_status"], tag=tag)
    return (demographics[0], N.filter_(
        f"{tag}_any_segment", demographics[1],
        any_of(*[all_of(Binary("==", Col("cd_marital_status"), Lit(marital)),
                        Binary("==", Col("cd_education_status"), Lit(degree)))
                 for marital, degree, *_ in bands]),
    ))


# -- store sales in a band ----------------------------------------------------------


_Q13_BANDS = (("M", "Advanced Degree", 100.00, 150.00, 3),
              ("S", "College", 50.00, 100.00, 1),
              ("W", "2 yr Degree", 150.00, 200.00, 1))


@query("q13")
def plan_q13(t):
    """Averages over the sales that fall in one of three demographic and three geographic
    bands. The demographic arms also name a household's dependant count, so each arm reads
    three tables — which is the same reason as ever for the predicate being above the join,
    with one more table in it."""
    store_sales = fact(t, "store_sales",
                       ["ss_store_sk", "ss_sold_date_sk", "ss_cdemo_sk", "ss_hdemo_sk",
                        "ss_addr_sk", "ss_sales_price", "ss_net_profit", "ss_quantity",
                        "ss_ext_sales_price", "ss_ext_wholesale_cost"])
    store = dim(t, "store", ["s_store_sk"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"],
                   Binary("==", Col("d_year"), Lit(2001)))
    bands = tuple((marital, degree, low, high) for marital, degree, low, high, _ in _Q13_BANDS)
    household = dim(t, "household_demographics", ["hd_demo_sk", "hd_dep_count"],
                    is_in(Col("hd_dep_count"),
                          sorted({dependants for *_, dependants in _Q13_BANDS})))
    joined = star(
        store_sales[1],
        ("s_ss", store, "s_store_sk", "ss_store_sk"),
        ("d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
        ("cd_ss", _segments(t, bands), "cd_demo_sk", "ss_cdemo_sk"),
        ("hd_ss", household, "hd_demo_sk", "ss_hdemo_sk"),
        ("ca_ss", _address(t, _REGIONS_Q13), "ca_address_sk", "ss_addr_sk"),
    )
    kept = N.filter_(
        "bands_and_regions", joined,
        Binary("and",
               _segment_predicate(
                   bands, "ss_sales_price",
                   extra=lambda index: Binary("==", Col("hd_dep_count"),
                                              Lit(_Q13_BANDS[index][4]))),
               _region_predicate(_REGIONS_Q13, "ss_net_profit")),
    )
    return N.unload(
        "unload",
        aggregate_to_one_row(
            "agg",
            select("keys", kept, "ss_quantity", "ss_ext_sales_price", "ss_ext_wholesale_cost"),
            [A.Agg(A.MEAN, "ss_quantity", "avg1"),
             A.Agg(A.MEAN, "ss_ext_sales_price", "avg2"),
             A.Agg(A.MEAN, "ss_ext_wholesale_cost", "avg3"),
             A.Agg(A.SUM, "ss_ext_wholesale_cost", "sum(ss_ext_wholesale_cost)")],
            corpus.schema_of(store_sales[0]),
        ),
    )


_Q48_BANDS = (("M", "4 yr Degree", 100.00, 150.00), ("D", "2 yr Degree", 50.00, 100.00),
              ("S", "College", 150.00, 200.00))


@query("q48")
def plan_q48(t):
    """q13 without the household, and reporting a quantity rather than four measures."""
    store_sales = fact(t, "store_sales",
                       ["ss_store_sk", "ss_sold_date_sk", "ss_cdemo_sk", "ss_addr_sk",
                        "ss_sales_price", "ss_net_profit", "ss_quantity"])
    store = dim(t, "store", ["s_store_sk"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"],
                   Binary("==", Col("d_year"), Lit(2000)))
    joined = star(
        store_sales[1],
        ("s_ss", store, "s_store_sk", "ss_store_sk"),
        ("d_ss", date_dim, "d_date_sk", "ss_sold_date_sk"),
        ("cd_ss", _segments(t, _Q48_BANDS), "cd_demo_sk", "ss_cdemo_sk"),
        ("ca_ss", _address(t, _REGIONS_Q48), "ca_address_sk", "ss_addr_sk"),
    )
    kept = N.filter_(
        "bands_and_regions", joined,
        Binary("and", _segment_predicate(_Q48_BANDS, "ss_sales_price"),
               _region_predicate(_REGIONS_Q48, "ss_net_profit")),
    )
    return N.unload(
        "unload",
        aggregate_to_one_row("agg", select("keys", kept, "ss_quantity"),
                             [A.Agg(A.SUM, "ss_quantity", "sum(ss_quantity)")],
                             corpus.schema_of(store_sales[0])),
    )


# -- web returns in a band, by reason ------------------------------------------------


_Q85_BANDS = (("M", "Advanced Degree", 100.00, 150.00), ("S", "College", 50.00, 100.00),
              ("W", "2 yr Degree", 150.00, 200.00))
_Q85_REASON = 'main."substring"(r_reason_desc, 1, 20)'


@query("q85", order_by=(_Q85_REASON, "avg1", "avg2", "avg(wr_fee)"))
def plan_q85(t):
    """Why web orders came back, for customers in a band, averaged per reason.

    Two copies of customer_demographics: the arms require the refunding and the returning
    customer to have the *same* marital status and education, so one arm is three equalities
    and two of them compare a column with a column. A batch cannot hold two `cd_gender`s, so
    the second copy is renamed on the way in — the same rename the join capability work
    needed for a self-join, applied to a dimension.

    web_returns is the fact; web_sales joins to it on (item, order), and is small enough at
    719K rows to be the build side of that one.
    """
    returns = fact(t, "web_returns",
                   ["wr_item_sk", "wr_order_number", "wr_refunded_cdemo_sk",
                    "wr_returning_cdemo_sk", "wr_refunded_addr_sk", "wr_reason_sk",
                    "wr_refunded_cash", "wr_fee"])
    sales = dim(t, "web_sales",
                ["ws_item_sk", "ws_order_number", "ws_web_page_sk", "ws_sold_date_sk",
                 "ws_sales_price", "ws_net_profit", "ws_quantity"], rows=250_000)
    page = dim(t, "web_page", ["wp_web_page_sk"])
    date_dim = dim(t, "date_dim", ["d_date_sk", "d_year"],
                   Binary("==", Col("d_year"), Lit(2000)))
    sold_in_2000 = select(
        "sales_keys",
        star(sales[1], ("d_ws", date_dim, "d_date_sk", "ws_sold_date_sk"),
             ("wp_ws", page, "wp_web_page_sk", "ws_web_page_sk")),
        "ws_item_sk", "ws_order_number", "ws_sales_price", "ws_net_profit", "ws_quantity",
    )
    with_sales = N.hash_join(
        "ws_wr",
        N.coalesce_all("ws_wr_build", sold_in_2000, schema=dict(sales[0].dtypes)),
        returns[1], JoinType.INNER,
        ["ws_item_sk", "ws_order_number"], ["wr_item_sk", "wr_order_number"],
    )
    cd1 = _segments(t, _Q85_BANDS, tag="cd1")
    cd2 = _segments(t, _Q85_BANDS, tag="cd2")
    cd2 = (cd2[0], N.project(
        "cd2_renamed", cd2[1],
        [Alias(Col("cd_demo_sk"), "cd2_demo_sk"),
         Alias(Col("cd_marital_status"), "cd2_marital_status"),
         Alias(Col("cd_education_status"), "cd2_education_status")],
    ))
    reason = dim(t, "reason", ["r_reason_sk", "r_reason_desc"])
    joined = star(
        with_sales,
        ("cd1_wr", cd1, "cd_demo_sk", "wr_refunded_cdemo_sk"),
        ("cd2_wr", cd2, "cd2_demo_sk", "wr_returning_cdemo_sk"),
        ("ca_wr", _address(t, _REGIONS_Q85), "ca_address_sk", "wr_refunded_addr_sk"),
        ("r_wr", reason, "r_reason_sk", "wr_reason_sk"),
    )
    same_segment = any_of(*[
        all_of(Binary("==", Col("cd_marital_status"), Lit(marital)),
               Binary("==", Col("cd_marital_status"), Col("cd2_marital_status")),
               Binary("==", Col("cd_education_status"), Lit(degree)),
               Binary("==", Col("cd_education_status"), Col("cd2_education_status")),
               between(Col("ws_sales_price"), Lit(low), Lit(high)))
        for marital, degree, low, high in _Q85_BANDS
    ])
    kept = N.filter_("bands_and_regions", joined,
                     Binary("and", same_segment,
                            _region_predicate(_REGIONS_Q85, "ws_net_profit")))
    final = aggregate_by(
        "agg", select("keys", kept, "r_reason_desc", "ws_quantity", "wr_refunded_cash",
                      "wr_fee"),
        ["r_reason_desc"],
        [A.Agg(A.MEAN, "ws_quantity", "avg1"), A.Agg(A.MEAN, "wr_refunded_cash", "avg2"),
         A.Agg(A.MEAN, "wr_fee", "avg(wr_fee)")],
        schema_frame=corpus.schema_of(reason[0], sales[0], returns[0]),
    )
    out = N.project(
        "out", final,
        [Alias(Substring(Col("r_reason_desc"), 1, 20), _Q85_REASON),
         Alias(Col("avg1"), "avg1"), Alias(Col("avg2"), "avg2"),
         Alias(Col("avg(wr_fee)"), "avg(wr_fee)")],
    )
    return N.unload(
        "unload",
        sorted_output("sort", out, [_Q85_REASON, "avg1", "avg2", "avg(wr_fee)"],
                      [True] * 4, fetch=100),
    )
