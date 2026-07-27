#!/usr/bin/env bash
# duckdb_minimal.sh — time DuckDB on the TPC-H / TPC-H+V queries WE HAVE GOLDENS FOR,
# over DuckDB's OWN storage rather than parquet.
#
# WHY THE GOLDENS ARE THE POINT: the query text comes from testdata/tpch_query_sql.sh, the
# same file testdata/gen_duckdb_goldens.sh uses to produce the goldens the GPU tests assert
# against. Neither file has its own copy. So a number produced here is attached to a result
# that something else independently checks — and by default this script re-verifies each
# query against its golden before timing it, so "fast" and "right" cannot drift apart.
#
# WHAT IS MEASURED, and what that requires:
#   * NATIVE DUCKDB STORAGE, not read_parquet. The tables are imported once into a .duckdb
#     file; the timing loop never touches the parquet. Otherwise the numbers would be
#     mostly parquet decode.
#   * WARM PAGE CACHE. The db file is warmed and residency is VERIFIED before timing, not
#     assumed — if the file is bigger than RAM the warm-up SILENTLY fails and every number
#     quietly becomes disk-bound. That case is detected and reported per query rather than
#     papered over.
#   * SECOND-SMALLEST of 6 runs. Discards the single luckiest run without letting a slow
#     outlier dominate.
#
# HONESTY MACHINERY — read this before trusting a number:
#   Per query the report gives an approximate WORKING SET (the on-disk size of just the
#   columns that query reads) and the BYTES ACTUALLY READ FROM THE BLOCK DEVICE during the
#   reported run. The second is a direct measurement, not an inference: ~0 bytes read means
#   the run was genuinely served from RAM; a number close to the working set means it was
#   not, whatever the warm-up claimed. DuckDB is columnar, so on a box where the whole db
#   does not fit, some queries are still genuinely warm and others are not — a blanket
#   "cold" verdict would be honest and useless.
#
# Usage:
#   benchmarks/duckdb_minimal.sh [--dir DIR] [--sf N] [--runs N] [--duckdb PATH]
#                                [--bucket NAME] [--endpoint URL] [--region NAME]
#                                [--threads N] [--skip-download] [--skip-verify]
#                                [--force-import] [--s3-direct]
#
#   --s3-direct import each table straight from s3://BUCKET/ over DuckDB httpfs, with NO
#               local parquet intermediate — for a host that fits the ~38 GB db but not
#               parquet+db (39.5+38 GB). Lands only the db. Row-count verification reads
#               from S3 too. The local download+import path is unchanged for other hosts.
#   --dir       where parquet (local mode) and the .duckdb file live. Default
#               /media/data/peacock-bench. NOTHING is hardcoded: point it at the target box.
#   --sf        scale factor; selects the bucket (tpch-sf<N>) and the golden set.
#   --threads   DuckDB threads for the TIMED runs. Default: DuckDB's own default (all
#               cores) — this is a benchmark, not a determinism exercise. The VERIFY pass
#               always uses threads=1, matching how the goldens were generated.
#
# Re-running is cheap: the S3 download and the import are both skip-if-present, so a second
# invocation goes straight to warming and timing.
set -uo pipefail

# --------------------------------------------------------------------------- args
DIR=/media/data/peacock-bench
SF=40
RUNS=6
DUCKDB="${DUCKDB:-duckdb}"
BUCKET=""
ENDPOINT="${PEACOCK_S3_ENDPOINT:-https://storage.eu-north1.nebius.cloud:443}"
REGION="${PEACOCK_S3_REGION:-eu-north-1}"
THREADS=""
SKIP_DOWNLOAD=0
SKIP_VERIFY=0
FORCE_IMPORT=0
S3_DIRECT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --sf) SF="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --duckdb) DUCKDB="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    --force-import) FORCE_IMPORT=1; shift ;;
    # S3-DIRECT: import straight from S3 via DuckDB httpfs, NO local parquet intermediate.
    # For hosts where parquet (39.5 GB) + native db (~38 GB) does not fit but the db alone
    # does (e.g. the 76 GB Nebius VM). Everything downstream — warm-up, residency, 6-run/
    # 2nd-min, golden verify — is UNCHANGED; only the table SOURCE differs.
    --s3-direct) S3_DIRECT=1; SKIP_DOWNLOAD=1; shift ;;
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
  esac
done
[ -n "$BUCKET" ] || BUCKET="tpch-sf${SF}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck source=testdata/tpch_query_sql.sh
. "${ROOT}/testdata/tpch_query_sql.sh"

PARQUET_DIR="${DIR}/tpch.sf${SF}"
DB="${DIR}/tpch.sf${SF}.duckdb"
GOLDEN_DIR="${ROOT}/testdata/goldens/tpch.sf${SF}"
PARAMS="${ROOT}/testdata/tpch-vec-queries/query_params.jsonl"
TABLES="customer lineitem nation orders part partsupp region supplier"

fail() { echo; echo "ERROR: $*" >&2; exit 1; }
hr()   { printf '%s\n' "-----------------------------------------------------------------------------"; }
gb()   { awk -v b="$1" 'BEGIN{printf "%.2f", b/1073741824}'; }

command -v "$DUCKDB" >/dev/null 2>&1 || fail "duckdb not found ($DUCKDB); pass --duckdb PATH"
command -v python3 >/dev/null 2>&1 || fail "python3 is required (query params, sizing)"
[ -f "$PARAMS" ] || fail "missing $PARAMS — the vector queries resolve q and D from it"
[ "$S3_DIRECT" -eq 1 ] || mkdir -p "$PARQUET_DIR" || fail "cannot create $PARQUET_DIR"

# --------------------------------------------------------------------------- table source
# Where read_parquet reads each table from. LOCAL mode: the downloaded parquet on --dir.
# S3-DIRECT mode: straight from the bucket over httpfs, no local copy.
if [ "$S3_DIRECT" -eq 1 ]; then
  src() { echo "read_parquet('s3://${BUCKET}/${1}.parquet')"; }
else
  src() { echo "read_parquet('${PARQUET_DIR}/${1}.parquet')"; }
fi

# httpfs + S3 secret prelude, prepended to EVERY duckdb invocation in s3-direct mode (each
# -c is a fresh process, so the secret must be re-created each time). ENDPOINT here is the
# BARE host — no scheme, no port — which is the form Nebius requires; derive it from the
# --endpoint value by stripping https:// and any :port. URL_STYLE 'path' + credential_chain
# (reads ~/.aws) is the exact combination proven to work against Nebius.
S3_HOST="${ENDPOINT#http://}"; S3_HOST="${S3_HOST#https://}"; S3_HOST="${S3_HOST%%:*}"; S3_HOST="${S3_HOST%/}"
if [ "$S3_DIRECT" -eq 1 ]; then
  DUCK_S3="INSTALL httpfs; LOAD httpfs;
    CREATE OR REPLACE SECRET neb (TYPE s3, PROVIDER credential_chain,
      ENDPOINT '${S3_HOST}', REGION '${REGION}', URL_STYLE 'path', USE_SSL true);"
else
  DUCK_S3=""
fi

echo "=== duckdb_minimal.sh — TPC-H sf${SF} on native DuckDB storage ==="
echo "target dir : $DIR"
echo "parquet    : $PARQUET_DIR"
echo "database   : $DB"
echo "duckdb     : $DUCKDB ($("$DUCKDB" --version 2>/dev/null | head -1))"
echo

# --------------------------------------------------------------------------- storage facts
# The backing device matters more than usual here: if the db does not fit in RAM the
# non-fitting queries are reading from this device on every run, so whether it is an SSD or
# a spinning disk changes how the numbers should be read.
DF_SRC="$(df --output=source "$DIR" 2>/dev/null | tail -1)"
DEV_NAME="$(basename "$(readlink -f "$DF_SRC" 2>/dev/null || echo "$DF_SRC")")"
# /sys/block/ has an entry per WHOLE device, not per partition: a partition like vda1 or
# nvme0n1p3 keeps its stat under the PARENT (vda / nvme0n1). Ask lsblk for the parent first;
# if that fails, fall back to the device's own stat, and only then give up. Without this the
# per-run device-bytes-read measurement silently disappears on any partition-backed dir.
STAT_FILE="/sys/block/${DEV_NAME}/stat"
if [ ! -r "$STAT_FILE" ] && command -v lsblk >/dev/null 2>&1; then
  PARENT="$(lsblk -no PKNAME "$DF_SRC" 2>/dev/null | awk 'NF{print; exit}')"
  [ -n "$PARENT" ] && [ -r "/sys/block/${PARENT}/stat" ] && { STAT_FILE="/sys/block/${PARENT}/stat"; DEV_NAME="$PARENT"; }
fi
[ -r "$STAT_FILE" ] || STAT_FILE=""
echo "filesystem : $DF_SRC  (block device: ${DEV_NAME})"
if command -v lsblk >/dev/null 2>&1; then
  echo "device tree:"
  lsblk -no NAME,ROTA,SIZE,TYPE,MODEL "$DF_SRC" 2>/dev/null | sed 's/^/  /' || true
  echo "  (ROTA=0 means non-rotational / SSD, ROTA=1 means spinning)"
fi
[ -n "$STAT_FILE" ] || echo "  NOTE: $STAT_FILE unreadable — per-run disk-read measurement disabled."

# sectors read from the backing device, cumulative. Field 3 of /sys/block/*/stat.
# System-wide for that device, not per-process: on a busy box this over-reports. It is
# still a DIRECT measurement of whether a run went to disk, which no timing can give.
sectors_read() {
  [ -n "$STAT_FILE" ] || { echo 0; return; }
  awk '{print $3}' "$STAT_FILE" 2>/dev/null || echo 0
}

MEM_TOTAL_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
MEM_AVAIL_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
MEM_TOTAL=$((MEM_TOTAL_KB * 1024))
MEM_AVAIL=$((MEM_AVAIL_KB * 1024))
echo "memory     : $(gb $MEM_TOTAL) GiB total, $(gb $MEM_AVAIL) GiB available"
echo

# --------------------------------------------------------------------------- preflight
# Fail with NUMBERS before spending an hour downloading 42 GB into a filesystem that cannot
# hold it. Disk is fatal; RAM is not — a db larger than RAM still produces meaningful
# per-query numbers for the queries whose working set fits, which is the whole reason the
# report is per query.
echo "=== preflight ==="
[ "$S3_DIRECT" -eq 1 ] && echo "mode       : S3-DIRECT (import straight from s3://${BUCKET}/, no local parquet)"
REMOTE_BYTES=0
# List the bucket when we either download OR import straight from it — both need the remote
# sizes: local mode to size the download, s3-direct to estimate the db and to verify row
# counts against the real source.
if [ "$SKIP_DOWNLOAD" -eq 0 ] || [ "$S3_DIRECT" -eq 1 ]; then
  command -v aws >/dev/null 2>&1 || fail "aws CLI not found (needed to size/verify the bucket)"
  REMOTE_LIST="$(aws --endpoint-url "$ENDPOINT" --region "$REGION" s3 ls "s3://${BUCKET}/" 2>&1)" \
    || fail "cannot list s3://${BUCKET}/ — is the bucket right and are credentials configured?\n${REMOTE_LIST}"
  REMOTE_BYTES=$(printf '%s\n' "$REMOTE_LIST" | awk '/\.parquet$/{s+=$3} END{print s+0}')
  REMOTE_FILES=$(printf '%s\n' "$REMOTE_LIST" | grep -c '\.parquet$' || true)
  [ "${REMOTE_BYTES:-0}" -gt 0 ] || fail "s3://${BUCKET}/ lists no parquet files"
  echo "remote     : ${REMOTE_FILES} parquet, $(gb $REMOTE_BYTES) GiB in s3://${BUCKET}/"
fi
LOCAL_BYTES=0
if [ "$S3_DIRECT" -eq 0 ]; then
  LOCAL_BYTES=$(find "$PARQUET_DIR" -maxdepth 1 -name '*.parquet' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
  echo "local      : $(gb $LOCAL_BYTES) GiB of parquet already present"
fi

# The db ends up roughly the size of the parquet: the embedding columns are near-random
# floats and compress in neither format, and they dominate the dataset.
EST_DB=${REMOTE_BYTES:-0}
[ "$EST_DB" -gt 0 ] || EST_DB=$LOCAL_BYTES
# An existing db is already on disk and is reused/overwritten in place (skip-if-present, or
# CREATE OR REPLACE into the same file), so it does NOT need to be found as NEW free space.
# Without this credit a re-run fails preflight the moment the db exists: free has dropped by
# the db size while NEED would still ask for the whole db again.
EXISTING_DB=0; [ -f "$DB" ] && EXISTING_DB=$(stat -c %s "$DB" 2>/dev/null || echo 0)
if [ "$S3_DIRECT" -eq 1 ]; then
  # s3-direct lands ONLY the db on local disk — no parquet — so the disk need is the db alone.
  NEED=$(( EST_DB + EST_DB / 10 - EXISTING_DB ))
  note="db only; no local parquet"
else
  NEED=$(( (REMOTE_BYTES > LOCAL_BYTES ? REMOTE_BYTES - LOCAL_BYTES : 0) + EST_DB ))
  NEED=$(( NEED + NEED / 10 - EXISTING_DB ))   # 10% headroom for the import's temporaries
  note="remaining download + db + 10%"
fi
[ "$NEED" -lt 0 ] && NEED=0
[ "$EXISTING_DB" -gt 0 ] && note="${note}; crediting existing $(gb $EXISTING_DB) GiB db"
FREE=$(df -B1 --output=avail "$DIR" | tail -1)
echo "disk       : $(gb $FREE) GiB free, need ~$(gb $NEED) GiB (${note})"
if [ "$FREE" -lt "$NEED" ]; then
  fail "insufficient disk in $DIR: $(gb $FREE) GiB free, ~$(gb $NEED) GiB required.
       Point --dir at a filesystem with room, or free $(gb $((NEED - FREE))) GiB."
fi
if [ "$MEM_TOTAL" -lt "$EST_DB" ]; then
  echo
  echo "  !! RAM ($(gb $MEM_TOTAL) GiB) IS SMALLER THAN THE ESTIMATED DATABASE ($(gb $EST_DB) GiB)."
  echo "  !! The page-cache warm-up CANNOT make the whole file resident on this machine."
  echo "  !! This is not fatal: DuckDB is columnar, so a query touching a few narrow"
  echo "  !! columns can still be fully warm. The per-query report below says which."
fi
echo

# --------------------------------------------------------------------------- download
# Resumable / skip-if-present. --size-only so an already-downloaded file is not re-fetched
# because of a timestamp; every file's size is then checked against the bucket manifest,
# which upload_datasets.sh writes LAST as its completion sentinel.
if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
  echo "=== download (skip-if-present) ==="
  t0=$(date +%s)
  aws --endpoint-url "$ENDPOINT" --region "$REGION" s3 sync "s3://${BUCKET}/" "$PARQUET_DIR/" \
      --exclude "*" --include "*.parquet" --size-only \
    || fail "aws s3 sync failed"
  t1=$(date +%s)
  echo "download wall time: $((t1 - t0)) s"

  for t in $TABLES; do
    [ -f "${PARQUET_DIR}/${t}.parquet" ] || fail "missing ${PARQUET_DIR}/${t}.parquet after sync"
  done
  # size check against the bucket listing: a truncated object would otherwise import as a
  # short table and every number below would be measured on the wrong data
  printf '%s\n' "$REMOTE_LIST" | awk '/\.parquet$/{print $4, $3}' | while read -r name size; do
    local_size=$(stat -c %s "${PARQUET_DIR}/${name}" 2>/dev/null || echo 0)
    [ "$local_size" = "$size" ] || { echo "ERROR: ${name} is ${local_size} B locally, ${size} B in the bucket" >&2; exit 1; }
  done || fail "downloaded parquet does not match the bucket sizes"
  echo "all ${SF} parquet files present and byte-size-matched against the bucket"
  echo
fi

# --------------------------------------------------------------------------- import
# Skip-if-present, and the check is on CONTENT not existence: a db whose row counts do not
# match the parquet is a half-finished import, which would otherwise be silently benchmarked.
# Keep ONLY the "table,count" data rows. The s3-direct prelude (INSTALL/LOAD/CREATE SECRET)
# emits its own result rows (a bare `true` from CREATE SECRET) under -csv, which would make
# parquet_row_counts one line longer than db_row_counts and fail the diff on a database that
# is actually correct. Filtering to rows that look like "<name>,<number>" makes the compare
# immune to any prelude chatter on either side.
_counts_only() { grep -E '^[a-z_]+,[0-9]+$'; }

db_row_counts() {
  local sql=""
  for t in $TABLES; do sql="${sql}SELECT '${t}', count(*) FROM ${t};"; done
  "$DUCKDB" "$DB" -csv -noheader -c "$sql" 2>/dev/null | _counts_only
}
# Row counts of the SOURCE — local parquet, or S3 straight from the bucket in s3-direct mode.
# In s3-direct this reads over httpfs (DUCK_S3 prelude), so a re-run still verifies the db
# against the REAL source rather than falling through to a local path that does not exist.
parquet_row_counts() {
  local sql=""
  for t in $TABLES; do
    sql="${sql}SELECT '${t}', count(*) FROM $(src "$t");"
  done
  "$DUCKDB" :memory: -csv -noheader -c "${DUCK_S3} ${sql}" 2>/dev/null | _counts_only
}

IMPORT_SECS=0
NEED_IMPORT=1
if [ -f "$DB" ] && [ "$FORCE_IMPORT" -eq 0 ]; then
  echo "=== import: existing database found, checking completeness ==="
  if diff <(db_row_counts) <(parquet_row_counts) >/dev/null 2>&1; then
    echo "row counts match the parquet for all 8 tables — skipping import"
    NEED_IMPORT=0
  else
    echo "row counts DO NOT match the parquet (or the db is unreadable) — re-importing"
    rm -f "$DB" "${DB}.wal"
  fi
  echo
fi

# DUCKDB SAFETY PRELUDE — issued in the SAME invocation as every heavy statement, BEFORE it.
# This is the fix for the crash: a laptop wedged here importing 27 GB of near-incompressible
# partsupp embeddings because DuckDB defaults memory_limit to 80% of TOTAL ram and had
# nowhere to spill, so it tried to build the table in RAM and drove the machine into the
# ground. generate_testdata.sh already learned this: a big DuckDB operation needs BOTH an
# explicit memory_limit AND a temp_directory, or it exhausts RAM. Neither alone suffices —
# memory_limit without a temp_directory makes the statement FAIL instead of spill.
#
#   memory_limit : a fraction of MemAVAILABLE (what the OS can actually give right now), not
#                  of total and not DuckDB's 80% default — the page cache and everything else
#                  need headroom. 60% of available, floored at 1 GiB so a tiny box still runs.
#   temp_directory: on --dir, i.e. the big filesystem the db already lives on, so spilled
#                  data has the same room the db does (177 GB here) rather than a small /tmp.
# On the large VM this binds nothing (plenty of RAM, no spill); on a small box it spills to
# disk and SURVIVES. That is the acceptance bar: impossible to crash any host, whatever the
# RAM-vs-db ratio.
DUCK_TMP="${DIR}/duckdb_tmp"
mkdir -p "$DUCK_TMP" || fail "cannot create temp dir $DUCK_TMP"
IMPORT_MEM_MB=$(( (MEM_AVAIL_KB / 1024) * 6 / 10 ))
[ "$IMPORT_MEM_MB" -ge 1024 ] || IMPORT_MEM_MB=1024
DUCK_SAFE="SET memory_limit='${IMPORT_MEM_MB}MB'; SET temp_directory='${DUCK_TMP}';"

if [ "$NEED_IMPORT" -eq 1 ]; then
  echo "=== import $([ "$S3_DIRECT" -eq 1 ] && echo 's3://'"$BUCKET"' (httpfs)' || echo 'parquet') -> native DuckDB storage ==="
  echo "memory_limit=${IMPORT_MEM_MB}MB (60% of $(gb $MEM_AVAIL) GiB available), spill dir=${DUCK_TMP}"
  [ "$FORCE_IMPORT" -eq 1 ] && rm -f "$DB" "${DB}.wal"
  t0=$(date +%s)
  for t in $TABLES; do
    printf '  %-10s ' "$t"
    ts=$(date +%s)
    # DUCK_S3 (httpfs secret, empty in local mode) then DUCK_SAFE, in the SAME -c invocation
    # as the CREATE, so the memory bound is in force while the table is built — this is
    # exactly the line whose absence crashed the box. src() reads local parquet or s3://.
    "$DUCKDB" "$DB" -c "${DUCK_S3} ${DUCK_SAFE} CREATE OR REPLACE TABLE ${t} AS SELECT * FROM $(src "$t");" \
      >/dev/null || fail "import of ${t} failed (memory_limit=${IMPORT_MEM_MB}MB, temp=${DUCK_TMP})"
    echo "$(( $(date +%s) - ts )) s"
  done
  # CHECKPOINT forces everything out of the WAL into the db file, so the file we are about
  # to warm actually contains the data. Without it a fresh import can leave a large WAL and
  # the residency numbers below would describe the wrong file. It can spike memory too, so it
  # gets the same bound. (No DUCK_S3 needed — this touches only the local db.)
  "$DUCKDB" "$DB" -c "${DUCK_SAFE} CHECKPOINT;" >/dev/null || fail "checkpoint failed"
  IMPORT_SECS=$(( $(date +%s) - t0 ))
  echo "import wall time: ${IMPORT_SECS} s"
  diff <(db_row_counts) <(parquet_row_counts) >/dev/null 2>&1 \
    || fail "post-import row counts do not match the parquet"
  echo "row counts verified against the parquet"
  echo
fi

DB_BYTES=$(stat -c %s "$DB")
echo "=== database ==="
echo "file       : $DB  ($(gb $DB_BYTES) GiB)"
echo "per-table on-disk size (from DuckDB's own storage metadata):"
# pragma_storage_info reports one row per column SEGMENT with the block it lives in.
# Distinct blocks per table x block size is DuckDB's own accounting of what that table
# occupies — approximate (a block can be shared) but derived from the file, not guessed.
BLOCK_SIZE=$("$DUCKDB" "$DB" -csv -noheader -c "SELECT block_size FROM pragma_database_size();" 2>/dev/null | head -1)
case "$BLOCK_SIZE" in ''|*[!0-9]*) BLOCK_SIZE=262144 ;; esac
TABLE_SIZE_SQL=""
for t in $TABLES; do
  TABLE_SIZE_SQL="${TABLE_SIZE_SQL}SELECT '${t}' AS t, count(DISTINCT block_id) AS blocks FROM pragma_storage_info('${t}') WHERE block_id IS NOT NULL UNION ALL "
done
TABLE_SIZE_SQL="SELECT t, blocks FROM (${TABLE_SIZE_SQL%UNION ALL }) ORDER BY blocks DESC;"
"$DUCKDB" "$DB" -csv -noheader -c "$TABLE_SIZE_SQL" 2>/dev/null | while IFS=, read -r t blocks; do
  printf '  %-10s %8.2f GiB\n' "$t" "$(awk -v b="$blocks" -v s="$BLOCK_SIZE" 'BEGIN{printf "%.4f", b*s/1073741824}')"
done
echo

# per-column bytes, used below to size each query's working set
COL_SIZES="$(mktemp)"
trap 'rm -f "$COL_SIZES" "$COL_SIZES".* 2>/dev/null' EXIT
COL_SQL=""
for t in $TABLES; do
  COL_SQL="${COL_SQL}SELECT '${t}' AS tbl, column_name, count(DISTINCT block_id) * ${BLOCK_SIZE} AS bytes FROM pragma_storage_info('${t}') WHERE block_id IS NOT NULL GROUP BY column_name UNION ALL "
done
"$DUCKDB" "$DB" -csv -noheader -c "SELECT * FROM (${COL_SQL%UNION ALL });" > "$COL_SIZES" 2>/dev/null || true
[ -s "$COL_SIZES" ] || echo "  NOTE: per-column sizes unavailable from pragma_storage_info; working sets will read 'n/a'."

# --------------------------------------------------------------------------- page cache
# THIS MUST BE REAL, NOT A GESTURE — AND IMPOSSIBLE TO CRASH ANY HOST BY CONSTRUCTION.
#
# Two invariants, both load-bearing after this session crashed a box:
#  1. RECLAIMABLE ONLY. The warm-up ONLY reads file bytes into the page cache. It NEVER
#     locks pages — no mlock, no `vmtouch -l`, no `vmtouch -t` (whose read volume is
#     unbounded). Page-cache pages are clean and reclaimable, so the kernel evicts them
#     under pressure; a reclaimable read cannot OOM a machine. Locking would reintroduce
#     exactly the non-reclaimable-memory failure the import fix just removed.
#  2. BOUNDED. It warms at most a safe fraction of what the OS can currently give
#     (50% of MemAvailable). On a box smaller than the db it warms what fits, REPORTS the
#     shortfall, and does NOT attempt the rest — the "degrade honestly" path. The residency
#     figure and the per-query device-bytes-read below then tell the true story regardless.
#
# The warm is driven by `dd` reading a bounded prefix (count-limited), NOT by vmtouch —
# vmtouch is used only to MEASURE residency afterwards (read-only, no locking). This keeps
# the volume pulled into cache capped even where vmtouch is installed.
echo "=== page cache warm-up ==="
SAFE_WARM=$(( (MEM_AVAIL_KB * 1024) / 2 ))               # 50% of MemAvailable, in bytes
WARM_TARGET=$DB_BYTES
[ "$WARM_TARGET" -le "$SAFE_WARM" ] || WARM_TARGET=$SAFE_WARM
if [ "$WARM_TARGET" -lt "$DB_BYTES" ]; then
  echo "  !! db ($(gb $DB_BYTES) GiB) EXCEEDS the safe warm cap ($(gb $SAFE_WARM) GiB = 50% of"
  echo "  !! $(gb $((MEM_AVAIL_KB*1024))) GiB available). Warming the first $(gb $WARM_TARGET) GiB ONLY;"
  echo "  !! the rest stays cold and its queries are I/O-bound (shown per query below). This is"
  echo "  !! the honest-degrade path — not a full warm, and not a crash."
fi
WARM_BS=$((8 * 1024 * 1024))
WARM_COUNT=$(( (WARM_TARGET + WARM_BS - 1) / WARM_BS ))
# capture Cached before the read so the no-vmtouch fallback can measure the delta
C0=$(awk '/^Cached:/{print $2*1024}' /proc/meminfo)
# bounded, reclaimable read: at most WARM_COUNT blocks, never the whole file if it is huge
dd if="$DB" of=/dev/null bs="$WARM_BS" count="$WARM_COUNT" status=none 2>/dev/null || true
if command -v vmtouch >/dev/null 2>&1; then
  WARM_METHOD="bounded dd read + vmtouch residency measure"
  # -v REPORTS residency; it does not touch or lock. "Resident Pages: 1234/5678 ..." —
  # take the page counts, not the human-readable sizes.
  RESIDENT_PAGES=$(vmtouch -v "$DB" 2>/dev/null | awk '/Resident Pages/{split($3,a,"/"); print a[1]; exit}')
  PAGE_SIZE=$(getconf PAGE_SIZE 2>/dev/null || echo 4096)
  case "$RESIDENT_PAGES" in ''|*[!0-9]*) RESIDENT_PAGES=0 ;; esac
  RESIDENT_BYTES=$((RESIDENT_PAGES * PAGE_SIZE))
else
  # FALLBACK, no vmtouch: report ABSOLUTE residency, not the cache-delta. The dd above just
  # read the whole warm-target into the page cache, so immediately after it the db's pages
  # ARE cached (there is no memory pressure here to evict them). The kernel's Cached figure
  # therefore includes them; capping Cached at the db size gives a sound lower bound on how
  # much of the db is resident. (The naive delta C1-C0 is WRONG for an already-warm file: if
  # the file was cached before the dd, the delta is ~0 while residency is ~100% — which is
  # exactly what a freshly-imported db looks like. The per-query device-read below is still
  # the authoritative warm/cold signal.)
  WARM_METHOD="bounded dd read + absolute-Cached (vmtouch not installed)"
  C1=$(awk '/^Cached:/{print $2*1024}' /proc/meminfo)
  RESIDENT_BYTES=$(( C1 < WARM_TARGET ? C1 : WARM_TARGET ))
fi
RESIDENT_PCT=$(awk -v r="$RESIDENT_BYTES" -v t="$DB_BYTES" 'BEGIN{printf "%.1f", (t>0? 100.0*r/t : 0)}')
echo "method     : $WARM_METHOD"
echo "resident   : $(gb $RESIDENT_BYTES) GiB of $(gb $DB_BYTES) GiB  (${RESIDENT_PCT}%)"
FULLY_WARM=0
if awk -v p="$RESIDENT_PCT" 'BEGIN{exit !(p >= 99.0)}'; then
  FULLY_WARM=1
  echo "the whole database is resident — every query below is genuinely warm"
else
  echo
  echo "  !! THE WARM-UP DID NOT FULLY SUCCEED. Only ${RESIDENT_PCT}% of the database is"
  echo "  !! resident, so any query whose working set does not fit in the page cache is"
  echo "  !! reading from disk on every run. The per-query table below reports the bytes"
  echo "  !! each reported run actually pulled from the block device, so this is visible"
  echo "  !! per query rather than being averaged into a single misleading verdict."
fi
echo

# --------------------------------------------------------------------------- query set
# id <TAB> label <TAB> golden-file <TAB> sql. The vector queries get D and the query vector
# from the COMMITTED query_params.jsonl — the same source gen_duckdb_goldens.sh uses, so a
# timed query and its golden can never be built from different parameters.
# final trap wins over the earlier COL_SIZES-only one: clean every temp this script owns,
# including the DuckDB spill dir (DUCK_TMP is always set by the import prelude above)
QUERIES="$(mktemp)"
trap 'rm -f "$COL_SIZES" "$COL_SIZES".* "$QUERIES" 2>/dev/null; rm -rf "$DUCK_TMP" 2>/dev/null' EXIT
# QUERIES is a tab-delimited, one-record-per-LINE file, but the SQL from tpch_query_sql.sh is
# MULTI-LINE. Written verbatim, each SQL continuation line would become its own bogus record
# (qid = a stray SQL fragment, empty golden) and the real query would run only its first line
# -> "QUERY FAILED". SQL is whitespace-insensitive, so collapse every run of whitespace
# (newlines included) to a single space before storing. One physical line per query, exact
# same SQL semantics.
emit() {
  local sql; sql="$(printf '%s' "$3" | tr '\n' ' ' | tr -s ' ')"
  printf '%s\t%s\t%s\n' "$1" "$2" "$sql" >> "$QUERIES"
}
emit q6 "duckdb_q6.csv" "$(sql_q6)"
emit q1 "duckdb_q1.csv" "$(sql_q1)"
emit q3 "duckdb_q3.csv" "$(sql_q3)"
emit q8 "duckdb_q8.csv" "$(sql_q8)"

# probe id -> query, mirroring gen_duckdb_goldens.sh's mapping (one probe per k tier)
while IFS=$'\t' read -r vid vD vlit; do
  case "$vid" in
    img_*) emit "q11v/${vid}" "duckdb_q11v_${vid}.csv" "$(sql_q11v "$vD" "$vlit")" ;;
    txt_000) emit "q12v/${vid}" "duckdb_q12v_${vid}.csv" "$(sql_q12v "$vD" "$vlit")" ;;
    txt_017) emit "q10v/${vid}" "duckdb_q10v_${vid}.csv" "$(sql_q10v "$vD" "$vlit")" ;;
    txt_034) emit "q9v/${vid}"  "duckdb_q9v_${vid}.csv"  "$(sql_q9v  "$vD" "$vlit")" ;;
  esac
done < <(python3 - "$PARAMS" <<'PY'
import json, sys
WANT = ["img_000", "img_017", "img_034", "txt_000", "txt_017", "txt_034"]
rows = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["id"] in WANT:
        rows[r["id"]] = (r["D"], "[" + ",".join(repr(float(x)) for x in r["q"]) + "]")
missing = [w for w in WANT if w not in rows]
if missing:
    sys.exit(f"error: query_params.jsonl is missing probes: {missing}")
for w in WANT:
    D, lit = rows[w]
    print(f"{w}\t{D!r}\t{lit}")
PY
) || fail "could not resolve the vector query parameters"

# --------------------------------------------------------------------------- helpers
# WORKING SET: the on-disk size of exactly the columns a query reads. Derived by matching
# every column name in the database against the query text, so there is no second list of
# "which columns does q9v touch" to drift out of step with the SQL. TPC-H column names are
# table-prefixed and none is a substring of another, which is what makes this reliable
# here; it over-approximates if a name appears in a query without being read, which none
# of these do.
working_set_bytes() {
  local sql="$1"
  [ -s "$COL_SIZES" ] || { echo ""; return; }
  python3 - "$COL_SIZES" <<PY
import csv, re, sys
sql = """${sql}"""
total = 0
for tbl, col, b in csv.reader(open(sys.argv[1])):
    if re.search(r'\b' + re.escape(col) + r'\b', sql):
        total += int(b)
print(total)
PY
}

# run one query once; print "<elapsed_ms> <device_bytes_read>"
run_once() {
  local sql="$1" pragma="$2"
  local s0 s1 t0 t1
  s0=$(sectors_read)
  t0=$(date +%s%N)
  "$DUCKDB" "$DB" -c "${pragma} ${sql}" > /dev/null 2>&1 || return 1
  t1=$(date +%s%N)
  s1=$(sectors_read)
  echo "$(( (t1 - t0) / 1000000 )) $(( (s1 - s0) * 512 ))"
}

# SPILL TARGET for the query runs too — a temp_directory but deliberately NO memory_limit.
# The point is the same safety-by-construction bar as the import: a big query (q11v scans
# 12 GB of ps_image_embedding) must SPILL rather than OOM on a small box. No memory_limit,
# because capping it would distort the very timings this script exists to measure, and on
# the VM the working set fits so nothing spills and this costs nothing. It only engages on a
# box too small for a query — where the number is I/O-bound and honestly reported anyway.
TIMED_PRAGMA="SET temp_directory='${DUCK_TMP}';"
[ -n "$THREADS" ] && TIMED_PRAGMA="${TIMED_PRAGMA} PRAGMA threads=${THREADS};"

# --------------------------------------------------------------------------- verify
# Same SQL, same data, threads=1 — exactly how the goldens were generated. If this fails,
# the timing below is measuring a query that produces the wrong answer, and the number is
# worthless however good it looks.
declare -A VERDICT=()
if [ "$SKIP_VERIFY" -eq 0 ]; then
  echo "=== verify against committed goldens (threads=1) ==="
  if [ ! -d "$GOLDEN_DIR" ]; then
    echo "  golden dir not found: $GOLDEN_DIR — SKIPPING verification. NOTHING WAS CHECKED."
  else
    while IFS=$'\t' read -r qid golden sql; do
      gpath="${GOLDEN_DIR}/${golden}"
      if [ ! -f "$gpath" ]; then
        VERDICT[$qid]="NO GOLDEN"; printf '  %-12s NO GOLDEN (%s)\n' "$qid" "$golden"; continue
      fi
      out="${COL_SIZES}.out"
      if ! "$DUCKDB" "$DB" -csv -noheader -c "SET temp_directory='${DUCK_TMP}'; PRAGMA threads=1; ${sql}" > "$out" 2>/dev/null; then
        VERDICT[$qid]="QUERY FAILED"; printf '  %-12s QUERY FAILED\n' "$qid"; continue
      fi
      if diff -q "$out" "$gpath" >/dev/null 2>&1; then
        VERDICT[$qid]="ok"; printf '  %-12s ok\n' "$qid"
      else
        VERDICT[$qid]="MISMATCH"
        printf '  %-12s MISMATCH vs %s\n' "$qid" "$golden"
        diff "$gpath" "$out" | head -6 | sed 's/^/      /'
      fi
    done < "$QUERIES"
  fi
  echo
fi

# --------------------------------------------------------------------------- timing
echo "=== timing: ${RUNS} runs per query, reporting the SECOND-SMALLEST ==="
[ -n "$THREADS" ] && echo "threads    : ${THREADS}" || echo "threads    : DuckDB default (all cores)"
echo "run 1 of each query warms whatever of its working set fits; runs 2-${RUNS} are"
echo "therefore 'as warm as this machine can get' for the queries that fit, and remain"
echo "I/O-bound for the ones that do not."
echo
RESULTS="${COL_SIZES}.res"
: > "$RESULTS"
while IFS=$'\t' read -r qid golden sql; do
  printf '  %-12s ' "$qid"
  ws=$(working_set_bytes "$sql")
  times=""; reads=""
  failed=0
  for i in $(seq 1 "$RUNS"); do
    if ! out=$(run_once "$sql" "$TIMED_PRAGMA"); then failed=1; break; fi
    set -- $out
    times="${times}${1} "
    reads="${reads}${2} "
    printf '.'
  done
  if [ "$failed" -eq 1 ]; then
    printf ' FAILED\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$qid" "${ws:-}" "FAILED" "" "" "" >> "$RESULTS"
    continue
  fi
  # second-smallest wall time, and the device read of the run that produced it
  read -r best2 best2_read <<EOF
$(python3 - <<PY
t = [int(x) for x in "$times".split()]
r = [int(x) for x in "$reads".split()]
order = sorted(range(len(t)), key=lambda i: t[i])
i = order[1] if len(order) > 1 else order[0]
print(t[i], r[i])
PY
)
EOF
  printf ' %6d ms\n' "$best2"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$qid" "${ws:-}" "$best2" "$best2_read" "$times" "${VERDICT[$qid]:-not verified}" >> "$RESULTS"
done < "$QUERIES"
echo

# --------------------------------------------------------------------------- report
echo "=== results ==="
hr
printf '%-12s %10s %9s %10s %9s %-8s %s\n' \
  "query" "2nd-min" "workset" "disk read" "warm?" "golden" "all runs (ms)"
hr
while IFS=$'\t' read -r qid ws best2 best2_read times verdict; do
  if [ "$best2" = "FAILED" ]; then
    printf '%-12s %10s %9s %10s %9s %-8s %s\n' "$qid" "FAILED" "" "" "" "${verdict}" ""
    continue
  fi
  ws_h="n/a"; [ -n "$ws" ] && ws_h="$(gb "$ws") G"
  rd_h="$(gb "$best2_read") G"
  # WARM VERDICT — from the measurement, not the intention. A run that pulled essentially
  # nothing from the block device was served from RAM whatever the global warm-up managed.
  if [ "$best2_read" -lt 16777216 ]; then warm="yes"
  elif [ -n "$ws" ] && [ "$ws" -gt 0 ] && awk -v r="$best2_read" -v w="$ws" 'BEGIN{exit !(r < 0.2*w)}'; then warm="partial"
  else warm="NO"; fi
  printf '%-12s %8d ms %9s %10s %9s %-8s %s\n' \
    "$qid" "$best2" "$ws_h" "$rd_h" "$warm" "$verdict" "$times"
done < "$RESULTS"
hr
echo "2nd-min   : second-smallest of ${RUNS} wall times"
echo "workset   : approximate on-disk size of the columns this query reads"
echo "disk read : bytes pulled from ${DEV_NAME} during the reported run (system-wide counter)"
echo "warm?     : yes = <16 MiB read from disk; partial = <20% of the working set; NO = otherwise"
echo "golden    : result compared against testdata/goldens/tpch.sf${SF}/"
echo
echo "database $(gb $DB_BYTES) GiB, $(gb $RESIDENT_BYTES) GiB resident (${RESIDENT_PCT}%) via ${WARM_METHOD}"
[ "$IMPORT_SECS" -gt 0 ] && echo "import wall time this run: ${IMPORT_SECS} s"
[ "$FULLY_WARM" -eq 1 ] || echo "NOT FULLY WARM — see the per-query 'warm?' column before quoting any number."
