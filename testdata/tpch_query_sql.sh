#!/usr/bin/env bash
# tpch_query_sql.sh — THE benchmark query text, written ONCE and sourced by everything
# that needs it.
#
# WHY THIS FILE EXISTS: testdata/gen_duckdb_goldens.sh produces the goldens the bare-cudf
# GPU tests assert against, and benchmarks/duckdb_minimal.sh times DuckDB on the same
# queries. If those two kept their own copies of the SQL they would drift, and the moment
# they did the benchmark would be measuring something the goldens do not describe — a
# number with no correctness attached. Both source this file, so a query can only be
# changed in one place.
#
# WHAT IS AND IS NOT IN HERE: each function emits a SELECT and NOTHING ELSE. No CREATE
# VIEW, no read_parquet, no PRAGMA. The caller decides what the table names resolve to —
# parquet views (goldens) or native DuckDB tables (benchmark) — which is precisely the
# difference between the two consumers and the only thing they are allowed to disagree on.
#
# DETERMINISM: every query with more than one row has an ORDER BY that is a TOTAL order
# (the sort keys are unique per output row). A query ordered by a non-unique key would
# leave ties to the engine and make the golden flaky rather than wrong — which is worse,
# because it passes until it doesn't.
#
# Vector queries take (D, literal) as arguments, resolved by the caller from the COMMITTED
# testdata/tpch-vec-queries/query_params.jsonl. Nothing here hardcodes a query vector.
#
# Usage:  source "$(dirname "$0")/tpch_query_sql.sh"
#         sql=$(sql_q6)
#         sql=$(sql_q11v "$D" "$LIT")

# ---------------------------------------------------------------------------
# Plain TPC-H (no vector predicate)
# ---------------------------------------------------------------------------

# Q6: sum(l_extendedprice * l_discount) over a date/discount/quantity filter. One row.
# Predicate constants are written as exact decimal literals (0.05 / 0.07) rather than
# TPC-H's '.06 - 0.01' / '.06 + 0.01': identical values, but written so the golden and
# the cudf-side scalars are visibly the same numbers.
sql_q6() { cat <<'SQL'
  SELECT sum(l_extendedprice * l_discount) AS revenue
  FROM lineitem
  WHERE l_shipdate >= DATE '1994-01-01'
    AND l_shipdate <  DATE '1995-01-01'
    AND l_discount BETWEEN 0.05 AND 0.07
    AND l_quantity < 24;
SQL
}

# Q1: pricing summary. GROUP BY (l_returnflag, l_linestatus) with 8 aggregates, ordered
# by the group keys. 4 rows.
# ORDER BY is on the two group keys, which are UNIQUE per row here, so the row order is
# total and cannot depend on tie-breaking — the GPU side sorts by the same two keys and
# the rows line up positionally.
# Column types matter for the comparison on the other side, so they are pinned here:
#   sum_qty / sum_base_price / sum_disc_price / sum_charge  DECIMAL (exact)
#   avg_qty / avg_price / avg_disc                          DOUBLE  (DuckDB's AVG over a
#                                                           DECIMAL returns DOUBLE)
#   count_order                                             BIGINT  (exact)
# The AVG columns are the ONLY ones needing a tolerance; see test_tpch.cpp.
sql_q1() { cat <<'SQL'
  SELECT l_returnflag, l_linestatus,
         sum(l_quantity)                                       AS sum_qty,
         sum(l_extendedprice)                                  AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount))               AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity)                                       AS avg_qty,
         avg(l_extendedprice)                                  AS avg_price,
         avg(l_discount)                                       AS avg_disc,
         count(*)                                              AS count_order
  FROM lineitem
  WHERE l_shipdate <= DATE '1998-09-02'
  GROUP BY l_returnflag, l_linestatus
  ORDER BY l_returnflag, l_linestatus;
SQL
}

# Q3: shipping priority. customer JOIN orders JOIN lineitem, filters, groupby, sort, top 10.
#
# TIE-BREAK: TPC-H Q3 specifies ORDER BY revenue DESC, o_orderdate — which is NOT a total
# order, so two rows with equal (revenue, o_orderdate) could come back either way round and
# a LIMIT 10 could even include different rows run to run. l_orderkey is appended as a
# final tie-breaker (it is unique per group here) so the ordering is TOTAL and the golden
# is reproducible. The cudf side sorts by the same three keys. Without this the comparison
# would be flaky rather than wrong — the worst kind of test.
sql_q3() { cat <<'SQL'
  SELECT l_orderkey,
         sum(l_extendedprice * (1 - l_discount)) AS revenue,
         o_orderdate,
         o_shippriority
  FROM customer, orders, lineitem
  WHERE c_mktsegment = 'BUILDING'
    AND c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate < DATE '1995-03-15'
    AND l_shipdate  > DATE '1995-03-15'
  GROUP BY l_orderkey, o_orderdate, o_shippriority
  ORDER BY revenue DESC, o_orderdate, l_orderkey
  LIMIT 10;
SQL
}

# Q8: national market share. SEVEN distinct tables (nation appears twice as n1/n2).
#
# FOUR COLUMNS, NOT THE SPEC'S TWO — deliberate. TPC-H Q8 outputs (o_year, mkt_share), but
# mkt_share is a DIVISION of two sums and DuckDB returns DOUBLE for it (verified:
# typeof(DECIMAL/DECIMAL) = DOUBLE), so it can only be compared within a tolerance. The two
# sums themselves are DECIMAL(38,4) and exact. Emitting them separately lets the test
# compare both EXACTLY and confine the tolerance to the ratio alone; comparing only
# mkt_share would let two compensating errors in the sums cancel inside the division and
# pass unnoticed.
sql_q8() { cat <<'SQL'
  SELECT o_year,
         sum(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) AS brazil_volume,
         sum(volume)                                             AS total_volume,
         sum(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / sum(volume) AS mkt_share
  FROM (
    SELECT extract(year FROM o_orderdate)        AS o_year,
           l_extendedprice * (1 - l_discount)    AS volume,
           n2.n_name                             AS nation
    FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
    WHERE p_partkey = l_partkey
      AND s_suppkey = l_suppkey
      AND l_orderkey = o_orderkey
      AND o_custkey = c_custkey
      AND c_nationkey = n1.n_nationkey
      AND n1.n_regionkey = r_regionkey
      AND r_name = 'AMERICA'
      AND s_nationkey = n2.n_nationkey
      AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
      AND p_type = 'ECONOMY ANODIZED STEEL'
  ) AS all_nations
  GROUP BY o_year
  ORDER BY o_year;
SQL
}

# ---------------------------------------------------------------------------
# TPC-H+V — the vector queries.  $1 = D (distance threshold), $2 = the query vector as a
# DuckDB list literal.  Both come from the COMMITTED query_params.jsonl.
#
# array_distance(), not the pgvector <-> operator: DuckDB has no <-> and the committed
# testdata/tpch-vec-queries/*.sql uses it only as notation. array_distance returns TRUE L2
# (the root), which is why the cudf side builds its cuVS index with L2SqrtExpanded rather
# than the default L2Expanded — see cpp/tests/gpu/test_tpchv.cpp.
#
# The ::FLOAT[N] casts are not decoration: the parquet column is a LIST and array_distance
# requires fixed-size ARRAY on both sides. N differs by column — 96 for the image
# embeddings, 100 for the text ones — and getting it wrong is an error, not a wrong answer.
# ---------------------------------------------------------------------------

# COUNT GOLDEN, per embedding column: how many rows of the BASE table fall under D, before
# any join. This is the corroboration that matters for a vector query — it pins the row set
# the distance predicate selects, independent of everything the joins and aggregates do
# afterwards. A final result can coincidentally match while the search returned the wrong
# neighbours; the count cannot.
#
# There are two of these, one per embedding column, NOT one per query: the count is a
# property of (column, probe), not of the query built on top of it. q12v/q10v/q9v all
# filter part on p_text_embedding, so they share sql_ptext_count and each uses the golden
# for its own probe id.
sql_psimage_count() { local D="$1" lit="$2"; cat <<SQL
    SELECT count(*) FROM partsupp
    WHERE array_distance(ps_image_embedding::FLOAT[96], ${lit}::FLOAT[96]) < ${D};
SQL
}

sql_ptext_count() { local D="$1" lit="$2"; cat <<SQL
    SELECT count(*) FROM part
    WHERE array_distance(p_text_embedding::FLOAT[100], ${lit}::FLOAT[100]) < ${D};
SQL
}

# q11v — national market value over partsupp, restricted to a vector neighbourhood.
# The HAVING subquery deliberately does NOT carry the vector predicate: its threshold is
# computed over every German partsupp row. So the join is needed twice over — once
# unfiltered for the threshold, once vector-filtered for the groups.
# ORDER BY adds ps_partkey after value: TPC-H q11 orders by value alone, which is not a
# total order once two parts share a value. ps_partkey is unique per group, so appending it
# makes the row order reproducible instead of leaving ties to the engine.
sql_q11v() { local D="$1" lit="$2"; cat <<SQL
    SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value
    FROM partsupp, supplier, nation
    WHERE ps_suppkey = s_suppkey
      AND s_nationkey = n_nationkey
      AND n_name = 'GERMANY'
      AND array_distance(ps_image_embedding::FLOAT[96], ${lit}::FLOAT[96]) < ${D}
    GROUP BY ps_partkey
    HAVING sum(ps_supplycost * ps_availqty) > (
      SELECT sum(ps_supplycost * ps_availqty) * 0.000002
      FROM partsupp, supplier, nation
      WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
    )
    ORDER BY value DESC, ps_partkey;
SQL
}

# q12v — shipping-mode SLA counts, restricted to a vector neighbourhood on part.
# THREE tables (orders, lineitem, part) and NO decimal anywhere: both outputs are integer
# counters built from CASE, so the whole result is exactly comparable.
# The interesting shapes for the GPU side are the COLUMN-TO-COLUMN date comparisons
# (l_commitdate < l_receiptdate, l_shipdate < l_commitdate) — every other query in this
# suite compares a date to a LITERAL — and the string group key.
# The upper date bound is written as DATE '1995-01-01' rather than TPC-H's
# "DATE '1994-01-01' + INTERVAL '1' YEAR": identical value, written so the golden and the
# cudf-side scalar are visibly the same number (same convention as Q6's 0.05/0.07).
# ORDER BY l_shipmode is total — it is the group key and unique per row.
sql_q12v() { local D="$1" lit="$2"; cat <<SQL
    SELECT l_shipmode,
           sum(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH'
                    THEN 1 ELSE 0 END) AS high_line_count,
           sum(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH'
                    THEN 1 ELSE 0 END) AS low_line_count
    FROM orders, lineitem, part
    WHERE o_orderkey = l_orderkey
      AND l_partkey = p_partkey
      AND l_shipmode IN ('MAIL', 'SHIP')
      AND l_commitdate < l_receiptdate
      AND l_shipdate < l_commitdate
      AND l_receiptdate >= DATE '1994-01-01'
      AND l_receiptdate <  DATE '1995-01-01'
      AND array_distance(p_text_embedding::FLOAT[100], ${lit}::FLOAT[100]) < ${D}
    GROUP BY l_shipmode
    ORDER BY l_shipmode;
SQL
}

# q10v — returned-item revenue by customer, restricted to a vector neighbourhood on part.
# FIVE tables, and a GROUP BY over SEVEN columns of which five are strings (c_comment runs
# to ~72 characters). Nothing else in this suite groups by strings at all.
#
# TIE-BREAK: TPC-H Q10 specifies ORDER BY revenue DESC alone, which is not a total order,
# so a LIMIT 20 could return different rows run to run. c_custkey is appended — it is
# unique per group — making the order total and the golden reproducible.
#
# NOTE FOR THE CSV GOLDEN: c_address and c_comment contain commas, so DuckDB quotes those
# fields. The golden reader parses RFC4180 quoting; it did not before this query existed.
sql_q10v() { local D="$1" lit="$2"; cat <<SQL
    SELECT c_custkey, c_name,
           sum(l_extendedprice * (1 - l_discount)) AS revenue,
           c_acctbal, n_name, c_address, c_phone, c_comment
    FROM customer, orders, lineitem, nation, part
    WHERE c_custkey = o_custkey
      AND l_orderkey = o_orderkey
      AND l_partkey = p_partkey
      AND o_orderdate >= DATE '1993-10-01'
      AND o_orderdate <  DATE '1994-01-01'
      AND l_returnflag = 'R'
      AND c_nationkey = n_nationkey
      AND array_distance(p_text_embedding::FLOAT[100], ${lit}::FLOAT[100]) < ${D}
    GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
    ORDER BY revenue DESC, c_custkey
    LIMIT 20;
SQL
}

# q9v — product-line profit by nation and year, restricted to a vector neighbourhood.
# SIX tables, and the only COMPOSITE JOIN in the suite: partsupp is joined to lineitem on
# BOTH (ps_partkey, ps_suppkey). Also the only substring predicate (LIKE '%green%', not a
# prefix) and the only expression that SUBTRACTS two decimal products rather than
# accumulating one, which forces scale reconciliation.
# ORDER BY (nation, o_year) is total — together they are the group key.
# The upper date bound of the year range is implicit here; there is no date filter in Q9.
sql_q9v() { local D="$1" lit="$2"; cat <<SQL
    SELECT nation, o_year, sum(amount) AS sum_profit
    FROM (
      SELECT n_name                                                             AS nation,
             extract(year FROM o_orderdate)                                     AS o_year,
             l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity    AS amount
      FROM part, supplier, lineitem, partsupp, orders, nation
      WHERE s_suppkey = l_suppkey
        AND ps_suppkey = l_suppkey
        AND ps_partkey = l_partkey
        AND p_partkey = l_partkey
        AND o_orderkey = l_orderkey
        AND s_nationkey = n_nationkey
        AND p_name LIKE '%green%'
        AND array_distance(p_text_embedding::FLOAT[100], ${lit}::FLOAT[100]) < ${D}
    ) AS profit
    GROUP BY nation, o_year
    ORDER BY nation, o_year DESC;
SQL
}
