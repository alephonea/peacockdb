#!/bin/bash
set -e

DUCKDB=${DUCKDB:-$(which duckdb 2>/dev/null)} || { echo "error: duckdb not found in PATH"; exit 1; }
OUTDIR=$(dirname "$0")/tpchsf1

mkdir -p "$OUTDIR"

$DUCKDB :memory: <<SQL
INSTALL tpch;
LOAD tpch;
CALL dbgen(sf=1);

COPY nation    TO '${OUTDIR}/nation.parquet'    (FORMAT parquet);
COPY region    TO '${OUTDIR}/region.parquet'    (FORMAT parquet);
COPY supplier  TO '${OUTDIR}/supplier.parquet'  (FORMAT parquet);
COPY customer  TO '${OUTDIR}/customer.parquet'  (FORMAT parquet);
COPY part      TO '${OUTDIR}/part.parquet'      (FORMAT parquet);
SQL

echo "Done. Files in $OUTDIR:"
ls -lh "$OUTDIR"
