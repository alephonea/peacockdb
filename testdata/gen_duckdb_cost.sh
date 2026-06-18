#!/usr/bin/env bash
# Generate the DuckDB cost goldens (Task 5).
#
# GENERATION/EXTRACTION SPLIT (#70):
#   generate (default; needs duckdb + data): run each TPC-H/TPC-DS query under
#     JSON profiling (actual execution — EXPLAIN ANALYZE semantics) and PERSIST a
#     lean normalized subset of the profiling tree to
#     testdata/duckdb-profiles/{tpch,tpcds}/<q>.json (committed), then extract the
#     golden from it.
#   --extract-only (pure; no duckdb, no data, runs anywhere): re-extract every
#     golden from the persisted JSON. Use this for annotation / cost-model tweaks
#     — NO query re-execution needed.
#
# Golden line (written by duckdb_cost.py — see it for the model + unit tests),
# annotations first then cost fields, mirroring .cpu.txt:
#   <OP>: [<annotation>, ]output_bytes=<n>, output_rows=<n>, materialized=<n>[, bytes_read_est=<n>, rows_read=<n>]
# materialized = this node's pipeline-breaker contribution (0 for streaming);
# duckdb_cost = Σ materialized. bytes_read_est/rows_read on TABLE_SCAN only
# (bytes_read_est is DERIVED, not measured). Annotation = DuckDB's own extra_info
# (table/projections/filters, join_type/conditions, groups/aggregates, order_by/top).
#
# PRAGMA threads=1 below is LOAD-BEARING: rows_read (operator_rows_scanned) scales
# with the thread count, so single-threaded keeps the goldens reproducible across
# machines (output_bytes/output_rows are already thread-independent).
#
# Run once (preferably on verda, which has duckdb 1.2.2 installed):
#   DUCKDB=~/.local/bin/duckdb testdata/gen_duckdb_cost.sh [TESTDATA_DIR]
#   testdata/gen_duckdb_cost.sh --extract-only [TESTDATA_DIR]   # no duckdb needed
#
# Requires duckdb 1.2.2 (generate only; via DUCKDB=) and python3.
set -euo pipefail

EXTRACT_ONLY=0
if [ "${1:-}" = "--extract-only" ]; then EXTRACT_ONLY=1; shift; fi

# duckdb + version pin are only needed for the generate phase. --extract-only is
# pure (reads persisted JSON), so it skips them and runs without duckdb/data.
DUCKDB=${DUCKDB:-$(command -v duckdb 2>/dev/null || true)}
if [ "$EXTRACT_ONLY" = 0 ]; then
  [ -x "$DUCKDB" ] || { echo "error: duckdb not found; set DUCKDB=/path/to/duckdb" >&2; exit 1; }
  # Pin to the same engine as generate_testdata.sh / CI (vX.Y.Z token compare).
  # Goldens are only reproducible against this exact engine + pinned testdata.
  EXPECTED_DUCKDB=${EXPECTED_DUCKDB-"v1.2.2"}
  if [ -n "$EXPECTED_DUCKDB" ]; then
    ACTUAL_FULL=$("$DUCKDB" --version)
    ACTUAL=$(echo "$ACTUAL_FULL" | awk '{print $1}')
    if [ "$ACTUAL" != "$EXPECTED_DUCKDB" ]; then
      echo "error: duckdb version mismatch" >&2
      echo "  expected: $EXPECTED_DUCKDB" >&2
      echo "  actual:   $ACTUAL  (full: $ACTUAL_FULL)" >&2
      echo "  duckdb:   $DUCKDB" >&2
      exit 1
    fi
  fi
fi

TESTDATA=${1:-"$(cd "$(dirname "$0")" && pwd)"}

# All cost-extraction logic lives in the standalone, unit-tested duckdb_cost.py
# module (testdata/duckdb_cost.py). This script only runs the duckdb CLI under
# JSON profiling and hands each profile to the module, which writes the per-node
# cost-tree golden and prints the scalar duckdb_cost (Σ materialized).
COST_PY="$(cd "$(dirname "$0")" && pwd)/duckdb_cost.py"
[ -f "$COST_PY" ] || { echo "error: $COST_PY not found" >&2; exit 1; }

# EXTRACTION phase: persisted JSON -> golden, for every <q>.json in profiles/.
# Pure (no duckdb/data) — re-run for annotation/model tweaks.
extract_dataset() {
  local label=$1 profiles=$2 canon=$3
  if [ ! -d "$profiles" ]; then echo "[$label] skip: no profiles dir $profiles" >&2; return; fi
  mkdir -p "$canon"
  local ok=0
  for pj in "$profiles"/q*.json; do
    [ -f "$pj" ] || continue
    local q; q=$(basename "$pj" .json)
    local cost; cost=$(python3 "$COST_PY" extract "$pj" "$canon/$q.duckdb_cost.txt")
    echo "[$label] $q: duckdb_cost=$cost"
    ok=$((ok + 1))
  done
  echo "[$label] extracted=$ok"
}

# GENERATION phase: run each query under JSON profiling, persist the normalized
# subset to profiles/<q>.json, then extract its golden.
gen_dataset() {
  local label=$1 data=$2 queries=$3 canon=$4 profiles=$5 lo=$6 hi=$7
  if [ ! -d "$data" ]; then echo "[$label] skip: no data dir $data" >&2; return; fi
  local db="$data/.duckdb_cache/$label.db"
  mkdir -p "$data/.duckdb_cache" "$canon" "$profiles"

  # Rebuild the cached native DB if missing or if any parquet is newer than it.
  local rebuild=0
  if [ ! -f "$db" ]; then
    rebuild=1
  elif [ -n "$(find "$data" -maxdepth 1 -name '*.parquet' -newer "$db" -print -quit)" ]; then
    rebuild=1
  fi
  if [ "$rebuild" = 1 ]; then
    echo "[$label] importing parquet -> $db"
    rm -f "$db" "$db.wal"
    for pq in "$data"/*.parquet; do
      local t; t=$(basename "$pq" .parquet)
      "$DUCKDB" "$db" -c "CREATE OR REPLACE TABLE \"$t\" AS SELECT * FROM read_parquet('$pq');"
    done
  fi

  local ok=0 fail=0
  for n in $(seq "$lo" "$hi"); do
    local q="q$n" sql="$queries/q$n.sql"
    [ -f "$sql" ] || continue
    local prof="/tmp/peacock_duckdb_${label}_${q}.json"
    rm -f "$prof"
    if "$DUCKDB" "$db" >/dev/null 2>/tmp/peacock_duckdb_err <<SQL
PRAGMA threads=1;
SET enable_profiling='json';
SET profiling_output='$prof';
$(cat "$sql")
SQL
    then
      if [ -s "$prof" ]; then
        python3 "$COST_PY" normalize "$prof" "$profiles/$q.json"   # persist lean subset
        local cost; cost=$(python3 "$COST_PY" extract "$profiles/$q.json" "$canon/$q.duckdb_cost.txt")
        echo "[$label] $q: duckdb_cost=$cost"
        ok=$((ok + 1))
      else
        echo "[$label] $q: FAILED (no profile written)" >&2
        fail=$((fail + 1))
      fi
    else
      echo "[$label] $q: FAILED -- $(tail -1 /tmp/peacock_duckdb_err)" >&2
      fail=$((fail + 1))
    fi
  done
  echo "[$label] generated=$ok failed=$fail"
}

TPCH_PROFILES="$TESTDATA/duckdb-profiles/tpch"
TPCDS_PROFILES="$TESTDATA/duckdb-profiles/tpcds"

if [ "$EXTRACT_ONLY" = 1 ]; then
  extract_dataset tpch  "$TPCH_PROFILES"  "$TESTDATA/plans.sf1"
  extract_dataset tpcds "$TPCDS_PROFILES" "$TESTDATA/plans-tpcds.sf1"
else
  gen_dataset tpch  "$TESTDATA/tpch.sf1"  "$TESTDATA/tpch-queries"  "$TESTDATA/plans.sf1"       "$TPCH_PROFILES"  1 22
  gen_dataset tpcds "$TESTDATA/tpcds.sf1" "$TESTDATA/tpcds-queries" "$TESTDATA/plans-tpcds.sf1" "$TPCDS_PROFILES" 1 99
fi
echo "done."
