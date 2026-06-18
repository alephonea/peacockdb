#!/usr/bin/env bash
# Generate the DuckDB cost goldens (Task 5/#10).
#
# TWO-PASS GENERATION (both JSON profiling; join_filter_pushdown is the only diff):
#   pass 1 (JFP OFF): deterministic cost tree. DuckDB's join_filter_pushdown installs
#     OPTIONAL min/max dynamic filters on probe scans, applied opportunistically
#     (build-vs-probe race) -> a scan's operator_cardinality is NONDETERMINISTIC even
#     at threads=1. Disabling it makes the cost-tree cardinalities reproducible (scans
#     report their stable post-STATIC-filter count); the query RESULT is unchanged.
#     Persisted lean to testdata/duckdb-profiles/{tpch,tpcds}/<q>.json (committed).
#   pass 2 (JFP ON): the runtime dynamic filters appear in the JSON profile's per-scan
#     extra_info "Dynamic Filters". We keep ONLY the deterministic min/max RANGE bounds
#     (the row counts they reduce are the flaky race — discarded). Persisted BOUNDS-ONLY
#     to testdata/duckdb-dynfilters/{tpch,tpcds}/<q>.json (committed).
#
# duckdb_cost.py extract COMBINES the two: cost tree + duckdb_cost from pass 1; the
# storage-pruning section from pass-1 static filters ∩ pass-2 dynamic bounds applied to
# the parquet row-group min/max. duckdb_cost NEVER uses pass-2's flaky numbers.
#
#   GEN (needs duckdb 1.5.4 + parquet): --gen  -> both passes + extract.
#   DEFAULT / --extract-only (needs parquet for the pruning section, no duckdb):
#     re-extract goldens from the committed pass-1 profiles + pass-2 bounds.
#
# threads=1 throughout (reproducibility).
#
# Run on verda (duckdb 1.5.4 + parquet + pyarrow):
#   DUCKDB=~/.local/bin/duckdb testdata/gen_duckdb_cost.sh --gen [TESTDATA_DIR]
#   testdata/gen_duckdb_cost.sh --extract-only [TESTDATA_DIR]
set -euo pipefail

GEN=0
case "${1:-}" in
  --gen)          GEN=1; shift ;;
  --extract-only) GEN=0; shift ;;
  --*)            echo "error: unknown flag $1 (use --gen or --extract-only)" >&2; exit 1 ;;
esac

DUCKDB=${DUCKDB:-$(command -v duckdb 2>/dev/null || true)}
if [ "$GEN" = 1 ]; then
  [ -x "$DUCKDB" ] || { echo "error: duckdb not found; set DUCKDB=/path/to/duckdb" >&2; exit 1; }
  # Pin to the same engine as generate_testdata.sh / CI (vX.Y.Z token compare).
  EXPECTED_DUCKDB=${EXPECTED_DUCKDB-"v1.5.4"}
  if [ -n "$EXPECTED_DUCKDB" ]; then
    ACTUAL_FULL=$("$DUCKDB" --version)
    ACTUAL=$(echo "$ACTUAL_FULL" | awk '{print $1}')
    if [ "$ACTUAL" != "$EXPECTED_DUCKDB" ]; then
      echo "error: duckdb version mismatch (expected $EXPECTED_DUCKDB, got $ACTUAL_FULL)" >&2
      exit 1
    fi
  fi
fi

TESTDATA=${1:-"$(cd "$(dirname "$0")" && pwd)"}
COST_PY="$(cd "$(dirname "$0")" && pwd)/duckdb_cost.py"
[ -f "$COST_PY" ] || { echo "error: $COST_PY not found" >&2; exit 1; }

# Run one query under JSON profiling. jfp=off disables join_filter_pushdown.
profile() {  # db sqlfile jfp(on|off) out
  local db="$1" sqlfile="$2" jfp="$3" out="$4" dis=""
  [ "$jfp" = off ] && dis="SET disabled_optimizers='join_filter_pushdown';"
  rm -f "$out"
  "$DUCKDB" "$db" >/dev/null 2>/tmp/peacock_duckdb_err <<SQL
PRAGMA threads=1;
$dis
SET enable_profiling='json';
SET profiling_output='$out';
$(cat "$sqlfile")
SQL
}

# extract one golden by combining the committed inputs (+ parquet for pruning).
extract_one() {  # profile dynf canon data q label
  local profile="$1" dynf="$2" canon="$3" data="$4" q="$5" label="$6"
  local args=("$profile" "$canon/$q.duckdb_cost.txt")
  [ -f "$dynf/$q.json" ] && args+=(--dynfilters "$dynf/$q.json")
  [ -d "$data" ] && args+=(--data-dir "$data")
  local cost; cost=$(python3 "$COST_PY" extract "${args[@]}")
  echo "[$label] $q: duckdb_cost=$cost"
}

# EXTRACT-ONLY: re-render goldens from committed pass-1 profiles + pass-2 bounds.
extract_dataset() {
  local label=$1 profiles=$2 dynf=$3 canon=$4 data=$5
  [ -d "$profiles" ] || { echo "[$label] skip: no profiles $profiles" >&2; return; }
  mkdir -p "$canon"
  local ok=0
  for pj in "$profiles"/q*.json; do
    [ -f "$pj" ] || continue
    local q; q=$(basename "$pj" .json)
    extract_one "$pj" "$dynf" "$canon" "$data" "$q" "$label"
    ok=$((ok + 1))
  done
  echo "[$label] extracted=$ok"
}

# GEN: two profiling passes per query -> pass-1 profile + pass-2 bounds, then extract.
gen_dataset() {
  local label=$1 data=$2 queries=$3 canon=$4 profiles=$5 dynf=$6 lo=$7 hi=$8
  [ -d "$data" ] || { echo "[$label] skip: no data dir $data" >&2; return; }
  local db="$data/.duckdb_cache/$label.db"
  mkdir -p "$data/.duckdb_cache" "$canon" "$profiles" "$dynf"

  local rebuild=0
  if [ ! -f "$db" ]; then rebuild=1
  elif [ -n "$(find "$data" -maxdepth 1 -name '*.parquet' -newer "$db" -print -quit)" ]; then rebuild=1; fi
  if [ "$rebuild" = 1 ]; then
    echo "[$label] importing parquet -> $db"; rm -f "$db" "$db.wal"
    for pq in "$data"/*.parquet; do
      local t; t=$(basename "$pq" .parquet)
      "$DUCKDB" "$db" -c "CREATE OR REPLACE TABLE \"$t\" AS SELECT * FROM read_parquet('$pq');"
    done
  fi

  local ok=0 fail=0
  for n in $(seq "$lo" "$hi"); do
    local q="q$n" sql="$queries/q$n.sql"
    [ -f "$sql" ] || continue
    local pon="/tmp/pk_${label}_${q}_on.json" poff="/tmp/pk_${label}_${q}_off.json"
    profile "$db" "$sql" on  "$pon"   || true
    profile "$db" "$sql" off "$poff"  || true
    if [ ! -s "$poff" ]; then
      echo "[$label] $q: FAILED -- $(tail -1 /tmp/peacock_duckdb_err 2>/dev/null)" >&2
      fail=$((fail + 1)); continue
    fi
    # pass 2 -> bounds-only (deterministic dynamic-filter ranges)
    [ -s "$pon" ] && python3 "$COST_PY" dynfilters "$pon" "$dynf/$q.json"
    # pass 1 -> lean cost profile
    python3 "$COST_PY" normalize "$poff" "$profiles/$q.json"
    extract_one "$profiles/$q.json" "$dynf" "$canon" "$data" "$q" "$label"
    ok=$((ok + 1))
  done
  echo "[$label] generated=$ok failed=$fail"
}

TPCH_PROFILES="$TESTDATA/duckdb-profiles/tpch";   TPCDS_PROFILES="$TESTDATA/duckdb-profiles/tpcds"
TPCH_DYNF="$TESTDATA/duckdb-dynfilters/tpch";      TPCDS_DYNF="$TESTDATA/duckdb-dynfilters/tpcds"
TPCH_GOLD="$TESTDATA/goldens/tpch.sf1";            TPCDS_GOLD="$TESTDATA/goldens/tpcds.sf1"

if [ "$GEN" = 1 ]; then
  gen_dataset tpch  "$TESTDATA/tpch.sf1"  "$TESTDATA/tpch-queries"  "$TPCH_GOLD"  "$TPCH_PROFILES"  "$TPCH_DYNF"  1 22
  gen_dataset tpcds "$TESTDATA/tpcds.sf1" "$TESTDATA/tpcds-queries" "$TPCDS_GOLD" "$TPCDS_PROFILES" "$TPCDS_DYNF" 1 99
else
  extract_dataset tpch  "$TPCH_PROFILES"  "$TPCH_DYNF"  "$TPCH_GOLD"  "$TESTDATA/tpch.sf1"
  extract_dataset tpcds "$TPCDS_PROFILES" "$TPCDS_DYNF" "$TPCDS_GOLD" "$TESTDATA/tpcds.sf1"
fi
echo "done."
