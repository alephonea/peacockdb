"""TPC-DS query plans, lowered by hand into the batch-partitioned nodes — the registry.

The corpus is every TPC-DS query the engine already runs in `full_table` mode
(`testdata/cost-registry.csv`, `ftc_tp1 = enabled`) that does not need a window function:
78 minus 7, so 71. Window functions are the one thing missing here that is missing on
purpose — the mode has no `GpuWindow` node, so a lowering would have to invent one, and a
lowering that invents a node proves nothing about the surface it is supposed to be testing.

The queries themselves live in the family modules, grouped by what their lowering has to
do rather than by number:

  `plans_tpcds_star.py`      one fact table streaming past small dimensions
  `plans_tpcds_reports.py`   one pass, many conditional sums — the bucketed reports
  `plans_tpcds_baskets.py`   aggregate the ticket first, join the customer after
  `plans_tpcds_bands.py`     disjunctions that pair a dimension with the fact's own money
  `plans_tpcds_scalar.py`    a correlated subquery, lowered to an aggregate and a join
  `plans_tpcds_sets.py`      IN / EXISTS / INTERSECT, lowered to semi and anti joins
  `plans_tpcds_channels.py`  the same measure over store, catalog and web, unioned
  `plans_tpcds_growth.py`    a CTE aggregated once per alias and joined to itself
  `plans_tpcds_weeks.py`     a date range named indirectly, and what else is left

Every builder takes a table provider and never reads a row, so `plans.py` renders all of
them from parquet footers in a fraction of a second; see `plans_tpch.py` for why that split
exists. `ORDER_BY` records, per query, which output columns its ORDER BY names — the oracle
compares those positionally and the rest as a multiset, which is what SQL determines.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from . import (
    plans_tpcds_bands, plans_tpcds_baskets, plans_tpcds_channels, plans_tpcds_growth,
    plans_tpcds_reports, plans_tpcds_scalar, plans_tpcds_sets, plans_tpcds_star,
    plans_tpcds_weeks,
)

#: The family modules, in the order their queries are listed.
FAMILIES = (plans_tpcds_star, plans_tpcds_reports, plans_tpcds_baskets,
            plans_tpcds_scalar, plans_tpcds_bands, plans_tpcds_sets,
            plans_tpcds_channels, plans_tpcds_growth, plans_tpcds_weeks)

#: Every TPC-DS query the prototype runs, by name.
QUERIES: dict = {}
#: Per query, the output columns its ORDER BY names, in order.
ORDER_BY: dict = {}

for _family in FAMILIES:
    _clash = set(QUERIES) & set(_family.QUERIES)
    assert not _clash, f"{_family.__name__} redefines {sorted(_clash)}"
    QUERIES.update(_family.QUERIES)
    ORDER_BY.update(_family.ORDER_BY)

assert set(QUERIES) == set(ORDER_BY), "every query declares its ORDER BY, even as ()"
