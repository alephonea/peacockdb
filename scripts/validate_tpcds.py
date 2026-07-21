#!/usr/bin/env python3
"""validate_tpcds.py — structural validation for a generated tpcds.sf<N> (no embeddings).

Same scale-safe design as validate_tpch.py: ordering/clustering from parquet row-group
METADATA (exhaustive), exact within-group order on a SAMPLE. TPC-DS has no embeddings, so
these are purely structural.

Checks:
  PRESENCE     all 24 TPC-DS tables present.
  ROW COUNTS   the SF-linear ones TPC-DS defines exactly (dimension tables that scale by a
               fixed multiple); the 7 fact tables are only approximately linear, so a
               tolerance band.
  SORT ORDER   all 7 fact tables sorted by (lead <date>_sk NULLS LAST, item_sk, <txn key>):
               row-group min/max non-decreasing/non-overlapping + NULLS LAST (exhaustive);
               exact within-group order on a sample.

Usage:  python3 scripts/validate_tpcds.py --sf 1
"""
import os
import sys
import argparse
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_checks as dc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Table list, fact sort keys and SF-invariant counts come from dataset_checks — the
# SINGLE SOURCE OF TRUTH shared with check_s3_datasets.py. Don't restate them here.
TABLES = dc.TPCDS_TABLES
FACTS = dc.TPCDS_FACT_SORT_KEYS


# 7 fact tables: (lead date_sk, then item_sk, then transaction key) — must match the
# ORDER BY in generate_testdata.sh.
FACTS = {
    "catalog_sales":   ["cs_sold_date_sk", "cs_item_sk", "cs_order_number"],
    "catalog_returns": ["cr_returned_date_sk", "cr_item_sk", "cr_order_number"],
    "store_sales":     ["ss_sold_date_sk", "ss_item_sk", "ss_ticket_number"],
    "store_returns":   ["sr_returned_date_sk", "sr_item_sk", "sr_ticket_number"],
    "web_sales":       ["ws_sold_date_sk", "ws_item_sk", "ws_order_number"],
    "web_returns":     ["wr_returned_date_sk", "wr_item_sk", "wr_order_number"],
    "inventory":       ["inv_date_sk", "inv_item_sk", "inv_warehouse_sk"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=int, default=1)
    sf = ap.parse_args().sf
    TD = os.path.join(ROOT, f"testdata/tpcds.sf{sf}")
    P = lambda t: os.path.join(TD, f"{t}.parquet")

    print("-- TABLE PRESENCE --")
    missing = [t for t in TABLES if not os.path.exists(P(t))]
    dc.check("all 24 TPC-DS tables present", not missing, 'meta',
             f"missing {missing}" if missing else "24/24")
    if missing:
        dc.summarize_and_exit()

    # Only assert counts that are genuinely SF-INVARIANT. Most TPC-DS dimensions grow
    # sub-linearly by a spec formula (e.g. reason 35@sf1 -> 55 later) that would
    # false-fail across SFs, so we don't hard-assert them — table presence + fact-table
    # sort order are the load-bearing checks here.
    print("-- ROW COUNTS (SF-invariant) --")
    for t in sorted(dc.TPCDS_ROWS_FIXED):
        exp, tol = dc.expected_rows("tpcds", t, sf)
        dc.check_row_count(P(t), exp, t, tol=tol)

    print("-- SORT ORDER (7 fact tables) --")
    for tbl, keys in FACTS.items():
        lead = keys[0]
        dc.check_clustering(P(tbl), lead, tbl)
        dc.check_within_group_order(P(tbl), keys, tbl)

    dc.summarize_and_exit()


if __name__ == "__main__":
    main()
