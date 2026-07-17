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
REAL_EMB=0 # --real-embeddings: populate part/partsupp embeddings from fetched
           # open datasets (DEEP1B image + GloVe text). Default OFF = synthetic
           # FLOAT[8] embeddings (no fetch), so CI/H200 keep working unchanged.
while [ $# -gt 0 ]; do
  case "$1" in
    --sf) SF="$2"; shift ;;
    --bench) BENCH="$2"; shift ;;
    --real-embeddings) REAL_EMB=1 ;;
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
  # --- Vector augmentation (TPC-H+V) ---------------------------------------
  # part/partsupp carry embedding columns so the engine can be exercised on
  # vector-search queries over a familiar schema. The augmentation scheme (which
  # tables/columns carry embeddings) follows:
  #   Exqutor: Extended Query Optimizer for Vector-augmented Analytical Queries
  #   arXiv:2512.09695. The paper's repo is CC BY-NC 4.0; NONE of its code is used.
  # Two modes, chosen by --real-embeddings; both APPEND columns AFTER every stock
  # column so existing column indices (and projection-pushdown goldens) don't shift:
  #   default  synthetic FLOAT[8], seeded by DuckDB hash() of the row key.
  #   --real   image = DEEP1B (CC BY 4.0) 96-dim, ordinal per partsupp scan row;
  #            text  = mean of GloVe 100d (PDDL) word-vectors over the row's tokens.
  # Both are DETERMINISTIC (threads=1 + pinned DuckDB). Synthetic (hash-based) reproduces
  # byte-for-byte on any pinned-duckdb host. --real is generated on THIS gen box and
  # SHIPPED as parquet — verda/shad-gpu never regenerate it — so its determinism is
  # run-to-run on the gen box, not "reproducible on verda".
  CACHE="${SCRIPT_DIR}/embeddings-cache"
  if [ "$REAL_EMB" = 1 ]; then
    N=$(( 800000 * SF ))                       # partsupp row count = image-vector count
    DEEP_FBIN="${CACHE}/deep_base.sf${SF}.fbin"
    GLOVE_TXT="${CACHE}/glove.6B.100d.txt"
    for f in "$DEEP_FBIN" "$GLOVE_TXT"; do
      [ -f "$f" ] || { echo "error: missing $f — run testdata/fetch_embeddings.sh --sf ${SF} first" >&2; exit 1; }
    done
    # HARD CONTRACT: the saved slice keeps the original fbin header (num_vectors reads
    # 1e9) but holds only N vectors. Assert size == 8 + N*96*4 and read EXACTLY N rows;
    # never trust the header count or we read past EOF.
    EXPECT_BYTES=$(( 8 + N * 96 * 4 ))
    ACT_BYTES=$(stat -c%s "$DEEP_FBIN")
    [ "$ACT_BYTES" = "$EXPECT_BYTES" ] || { echo "error: $DEEP_FBIN size $ACT_BYTES != 8+${N}*96*4=$EXPECT_BYTES" >&2; exit 1; }
    DEEP_PARQUET="${CACHE}/deep_base.sf${SF}.image.parquet"   # transient (gitignored cache, never shipped)
    echo "  converting DEEP1B slice ($N vectors) -> $DEEP_PARQUET ..."
    python3 - "$DEEP_FBIN" "$N" "$DEEP_PARQUET" <<'PY'
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, os, sys
path, N, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
DIM = 96
assert os.path.getsize(path) == 8 + N*DIM*4          # hard contract; never trust header
with open(path, 'rb') as f:
    _nv, nd = np.frombuffer(f.read(8), dtype='<i4'); assert nd == DIM, nd
    data = np.frombuffer(f.read(N*DIM*4), dtype='<f4').reshape(N, DIM)   # read EXACTLY N
arr = pa.FixedSizeListArray.from_arrays(pa.array(data.reshape(-1), type=pa.float32()), DIM)
pq.write_table(pa.table({'idx': pa.array(np.arange(N, dtype='<i8')), 'image_embedding': arr}), out)
PY
    GLOVE_COLS=$(python3 -c "print('{'+\"'column0':'VARCHAR',\"+','.join(\"'column%d':'FLOAT'\"%i for i in range(1,101))+'}')")
    GLOVE_LIST=$(python3 -c "print(','.join('column%d'%i for i in range(1,101)))")
    # 100 per-dimension SUM aggregates. Summing each dim in a streaming/spillable hash
    # aggregate (rather than list()-ing every word-vector then folding) keeps peak memory
    # bounded on long text like ps_comment; threads=1 fixes the reduction order.
    SUMG=$(python3 -c "print(','.join('sum(g.vec[%d])'%i for i in range(1,101)))")
    # GloVe vocab as word -> FLOAT[100]. Read whole rows via sep=' ' (word + 100 floats).
    EMB_PREAMBLE="CREATE TEMP TABLE glove AS
  SELECT column0 AS word, [${GLOVE_LIST}]::FLOAT[100] AS vec
  FROM read_csv('${GLOVE_TXT}', sep=' ', header=false, quote='', escape='', auto_detect=false, columns=${GLOVE_COLS});"
    # part.p_text_embedding = mean GloVe over lower(p_name || ' ' || p_type). list_reduce
    # folds vectors in a FIXED order (list ordered by word,tok_ord) -> deterministic sum.
    # Empty-vocab row -> zero vector.
    PART_COPY="COPY (
  WITH toks AS (
    SELECT p_partkey, tok FROM part,
      unnest(regexp_split_to_array(lower(p_name || ' ' || p_type), '[^a-z0-9]+')) AS u(tok)
  ),
  matched AS (
    SELECT t.p_partkey, [${SUMG}] AS sums, count(*)::FLOAT AS n
    FROM toks t JOIN glove g ON g.word = t.tok WHERE t.tok <> '' GROUP BY t.p_partkey
  )
  SELECT part.*,
    CASE WHEN m.p_partkey IS NULL THEN [0.0 FOR _ IN range(100)]::FLOAT[100]
         ELSE list_transform(m.sums, lambda s: s / m.n)::FLOAT[100] END AS p_text_embedding
  FROM part LEFT JOIN matched m ON m.p_partkey = part.p_partkey
) TO '${OUTDIR}/part.parquet' (FORMAT parquet);"
    # partsupp: ps_image_embedding = DEEP1B[ordinal in scan order]; ps_text_embedding =
    # mean GloVe over ps_comment; ps_tag = same deterministic categorical as synthetic.
    PARTSUPP_COPY="COPY (
  WITH ps_rn AS (SELECT *, (row_number() OVER () - 1) AS _rn FROM partsupp),
  toks AS (
    SELECT ps_partkey, ps_suppkey, tok FROM partsupp,
      unnest(regexp_split_to_array(lower(ps_comment), '[^a-z0-9]+')) AS u(tok)
  ),
  matched AS (
    SELECT t.ps_partkey, t.ps_suppkey, [${SUMG}] AS sums, count(*)::FLOAT AS n
    FROM toks t JOIN glove g ON g.word = t.tok WHERE t.tok <> '' GROUP BY t.ps_partkey, t.ps_suppkey
  )
  SELECT ps_rn.* EXCLUDE (_rn),
    deep.image_embedding::FLOAT[96] AS ps_image_embedding,
    CASE WHEN m.ps_partkey IS NULL THEN [0.0 FOR _ IN range(100)]::FLOAT[100]
         ELSE list_transform(m.sums, lambda s: s / m.n)::FLOAT[100] END AS ps_text_embedding,
    (['electronics','apparel','home','toys','sports','automotive','grocery','books'])[ ((hash(ps_rn.ps_partkey::VARCHAR || '_' || ps_rn.ps_suppkey::VARCHAR || ':tag') % 8) + 1)::BIGINT ] AS ps_tag
  FROM ps_rn
  JOIN read_parquet('${DEEP_PARQUET}') deep ON deep.idx = ps_rn._rn
  LEFT JOIN matched m ON m.ps_partkey = ps_rn.ps_partkey AND m.ps_suppkey = ps_rn.ps_suppkey
) TO '${OUTDIR}/partsupp.parquet' (FORMAT parquet);"
  else
    EMB_PREAMBLE=""
    PART_COPY="COPY (
  SELECT part.*,
    [ (hash(p_partkey::VARCHAR || ':p_text:' || i::VARCHAR) % 100000)::FLOAT / 100000.0 FOR i IN range(8) ]::FLOAT[8] AS p_text_embedding
  FROM part
) TO '${OUTDIR}/part.parquet' (FORMAT parquet);"
    PARTSUPP_COPY="COPY (
  SELECT partsupp.*,
    [ (hash(ps_partkey::VARCHAR || '_' || ps_suppkey::VARCHAR || ':ps_image:' || i::VARCHAR) % 100000)::FLOAT / 100000.0 FOR i IN range(8) ]::FLOAT[8] AS ps_image_embedding,
    [ (hash(ps_partkey::VARCHAR || '_' || ps_suppkey::VARCHAR || ':ps_text:'  || i::VARCHAR) % 100000)::FLOAT / 100000.0 FOR i IN range(8) ]::FLOAT[8] AS ps_text_embedding,
    (['electronics','apparel','home','toys','sports','automotive','grocery','books'])[ ((hash(ps_partkey::VARCHAR || '_' || ps_suppkey::VARCHAR || ':tag') % 8) + 1)::BIGINT ] AS ps_tag
  FROM partsupp
) TO '${OUTDIR}/partsupp.parquet' (FORMAT parquet);"
  fi

  $DUCKDB :memory: <<SQL
PRAGMA threads=1;
INSTALL tpch;
LOAD tpch;
CALL dbgen(sf=${SF});

COPY nation    TO '${OUTDIR}/nation.parquet'    (FORMAT parquet);
COPY region    TO '${OUTDIR}/region.parquet'    (FORMAT parquet);
COPY supplier  TO '${OUTDIR}/supplier.parquet'  (FORMAT parquet);
COPY customer  TO '${OUTDIR}/customer.parquet'  (FORMAT parquet);
${EMB_PREAMBLE}
${PART_COPY}
${PARTSUPP_COPY}
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
