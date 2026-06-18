#!/bin/bash
#
# Generate TPC-H or TPC-DS test data as Parquet files using DuckDB.
#
# Usage:
#   ./testdata/generate_testdata.sh                    # generate tpch.sf1
#   ./testdata/generate_testdata.sh --sf 10            # generate tpch.sf10
#   ./testdata/generate_testdata.sh --bench tpcds      # generate tpcds.sf1
#   ./testdata/generate_testdata.sh --bench tpcds --sf 10
#
# Requires duckdb in PATH, or set DUCKDB=/path/to/duckdb.

set -euo pipefail

SF=1
BENCH=""   # required — pass --bench tpch|tpcds explicitly
while [ $# -gt 0 ]; do
  case "$1" in
    --sf) SF="$2"; shift ;;
    --bench) BENCH="$2"; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

case "$BENCH" in
  tpch|tpcds) ;;
  "") echo "error: --bench is required (tpch or tpcds)"; exit 1 ;;
  *) echo "error: --bench must be tpch or tpcds (got: $BENCH)"; exit 1 ;;
esac

DUCKDB=${DUCKDB:-$(which duckdb 2>/dev/null)} || { echo "error: duckdb not found in PATH"; exit 1; }

# Pinned DuckDB version. The TPC-DS `dsdgen` extension's column types drift
# across DuckDB releases (e.g. INT32 vs INT64 surrogate keys), which would
# shift `row_width` and `target_batch_size` across every plan canonical file
# — silently invalidating the goldens. Bump here + regenerate goldens.
#
# We compare only the `vX.Y.Z` token — `duckdb --version` also prints a build
# short-SHA (e.g. `v1.5.4 <sha>` from upstream vs conda-forge's rebuild). The
# SHA differs by packaging and isn't part of dsdgen's semantics; pinning
# on it would falsely reject the same upstream release packaged differently.
# Set EXPECTED_DUCKDB= (empty) to skip the check.
EXPECTED_DUCKDB=${EXPECTED_DUCKDB-"v1.5.4"}
if [ -n "$EXPECTED_DUCKDB" ]; then
  ACTUAL_DUCKDB_FULL=$("$DUCKDB" --version)
  ACTUAL_DUCKDB=$(echo "$ACTUAL_DUCKDB_FULL" | awk '{print $1}')
  if [ "$ACTUAL_DUCKDB" != "$EXPECTED_DUCKDB" ]; then
    echo "error: duckdb version mismatch" >&2
    echo "  expected: $EXPECTED_DUCKDB" >&2
    echo "  actual:   $ACTUAL_DUCKDB  (full: $ACTUAL_DUCKDB_FULL)" >&2
    echo "  duckdb:   $DUCKDB" >&2
    echo "Plan-canonical row_width / target_batch_size depend on the exact" >&2
    echo "schema dsdgen emits. Install duckdb-cli=1.5.4 (e.g. \`conda install" >&2
    echo "-c conda-forge duckdb-cli=1.5.4\`), or set EXPECTED_DUCKDB= to skip." >&2
    exit 1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${SCRIPT_DIR}/${BENCH}.sf${SF}"

if [ -d "$OUTDIR" ]; then
  echo "Directory $OUTDIR already exists, skipping generation."
  echo "Run testdata/clean_testdata.sh first to regenerate."
  exit 0
fi

mkdir -p "$OUTDIR"

if [ "$BENCH" = "tpch" ]; then
  echo "Generating TPC-H SF=${SF} into ${OUTDIR}..."
  # threads=1: a parallel COPY writes row groups non-deterministically even with a
  # total-order ORDER BY, producing different bytes run-to-run (verified). Single
  # thread is required for byte-identical parquet across hosts. The fact tables are
  # ORDER BY'd on their lead date column (NULLS LAST) + PK tiebreak so each parquet
  # row group's date min/max is tight — this clusters by date to match the
  # Hortonworks bin_partitioned layout and lets row-group pruning take effect.
  $DUCKDB :memory: <<SQL
PRAGMA threads=1;
INSTALL tpch;
LOAD tpch;
CALL dbgen(sf=${SF});

COPY nation    TO '${OUTDIR}/nation.parquet'    (FORMAT parquet);
COPY region    TO '${OUTDIR}/region.parquet'    (FORMAT parquet);
COPY supplier  TO '${OUTDIR}/supplier.parquet'  (FORMAT parquet);
COPY customer  TO '${OUTDIR}/customer.parquet'  (FORMAT parquet);
COPY part      TO '${OUTDIR}/part.parquet'      (FORMAT parquet);
COPY partsupp  TO '${OUTDIR}/partsupp.parquet'  (FORMAT parquet);
COPY (SELECT * FROM orders   ORDER BY o_orderdate NULLS LAST, o_orderkey)               TO '${OUTDIR}/orders.parquet'   (FORMAT parquet);
COPY (SELECT * FROM lineitem ORDER BY l_shipdate NULLS LAST, l_orderkey, l_linenumber)  TO '${OUTDIR}/lineitem.parquet' (FORMAT parquet);
SQL
else
  # TPC-DS: 24 tables. Discover them from the duckdb extension rather than
  # hard-coding so we don't drift if the extension changes.
  echo "Generating TPC-DS SF=${SF} into ${OUTDIR}..."
  # threads=1 for byte-identical output (parallel COPY is non-deterministic even
  # with ORDER BY — verified). The 7 fact tables are ORDER BY'd on their lead
  # date_sk (NULLS LAST; ~4% NULLs) + PK tiebreak for a deterministic TOTAL order
  # that clusters by date (matches Hortonworks bin_partitioned, enables row-group
  # pruning). Dimension tables stay in generation order (small; no scan benefit).
  $DUCKDB :memory: <<SQL
PRAGMA threads=1;
INSTALL tpcds;
LOAD tpcds;
CALL dsdgen(sf=${SF});

COPY (SELECT * FROM call_center)            TO '${OUTDIR}/call_center.parquet'            (FORMAT parquet);
COPY (SELECT * FROM catalog_page)           TO '${OUTDIR}/catalog_page.parquet'           (FORMAT parquet);
COPY (SELECT * FROM catalog_returns ORDER BY cr_returned_date_sk NULLS LAST, cr_item_sk, cr_order_number)  TO '${OUTDIR}/catalog_returns.parquet'  (FORMAT parquet);
COPY (SELECT * FROM catalog_sales   ORDER BY cs_sold_date_sk     NULLS LAST, cs_item_sk, cs_order_number)  TO '${OUTDIR}/catalog_sales.parquet'    (FORMAT parquet);
COPY (SELECT * FROM customer)               TO '${OUTDIR}/customer.parquet'               (FORMAT parquet);
COPY (SELECT * FROM customer_address)       TO '${OUTDIR}/customer_address.parquet'       (FORMAT parquet);
COPY (SELECT * FROM customer_demographics)  TO '${OUTDIR}/customer_demographics.parquet'  (FORMAT parquet);
COPY (SELECT * FROM date_dim)               TO '${OUTDIR}/date_dim.parquet'               (FORMAT parquet);
COPY (SELECT * FROM household_demographics) TO '${OUTDIR}/household_demographics.parquet' (FORMAT parquet);
COPY (SELECT * FROM income_band)            TO '${OUTDIR}/income_band.parquet'            (FORMAT parquet);
COPY (SELECT * FROM inventory        ORDER BY inv_date_sk        NULLS LAST, inv_item_sk, inv_warehouse_sk) TO '${OUTDIR}/inventory.parquet'       (FORMAT parquet);
COPY (SELECT * FROM item)                   TO '${OUTDIR}/item.parquet'                   (FORMAT parquet);
COPY (SELECT * FROM promotion)              TO '${OUTDIR}/promotion.parquet'              (FORMAT parquet);
COPY (SELECT * FROM reason)                 TO '${OUTDIR}/reason.parquet'                 (FORMAT parquet);
COPY (SELECT * FROM ship_mode)              TO '${OUTDIR}/ship_mode.parquet'              (FORMAT parquet);
COPY (SELECT * FROM store)                  TO '${OUTDIR}/store.parquet'                  (FORMAT parquet);
COPY (SELECT * FROM store_returns   ORDER BY sr_returned_date_sk NULLS LAST, sr_item_sk, sr_ticket_number) TO '${OUTDIR}/store_returns.parquet'   (FORMAT parquet);
COPY (SELECT * FROM store_sales     ORDER BY ss_sold_date_sk     NULLS LAST, ss_item_sk, ss_ticket_number) TO '${OUTDIR}/store_sales.parquet'     (FORMAT parquet);
COPY (SELECT * FROM time_dim)               TO '${OUTDIR}/time_dim.parquet'               (FORMAT parquet);
COPY (SELECT * FROM warehouse)              TO '${OUTDIR}/warehouse.parquet'              (FORMAT parquet);
COPY (SELECT * FROM web_page)               TO '${OUTDIR}/web_page.parquet'               (FORMAT parquet);
COPY (SELECT * FROM web_returns     ORDER BY wr_returned_date_sk NULLS LAST, wr_item_sk, wr_order_number)  TO '${OUTDIR}/web_returns.parquet'     (FORMAT parquet);
COPY (SELECT * FROM web_sales       ORDER BY ws_sold_date_sk     NULLS LAST, ws_item_sk, ws_order_number)  TO '${OUTDIR}/web_sales.parquet'       (FORMAT parquet);
COPY (SELECT * FROM web_site)               TO '${OUTDIR}/web_site.parquet'               (FORMAT parquet);
SQL
fi

echo "Done. Files in ${OUTDIR}:"
ls -lh "$OUTDIR"
