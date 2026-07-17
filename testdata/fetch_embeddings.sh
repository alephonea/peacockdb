#!/bin/bash
#
# Fetch open-dataset embedding sources for `generate_testdata.sh --real-embeddings`.
#
# LOCAL-ONLY. Downloads ~1.1GB into testdata/embeddings-cache/ and checksum-verifies
# every artifact. The augmented parquet is generated locally and SHIPPED to the test
# hosts as parquet — CI, verda and shad-gpu MUST NEVER run this (see the hard guard
# below). Re-runs are skip-if-present: an artifact whose SHA256 already matches is not
# re-downloaded.
#
# Sources (pinned, immutable objects):
#   DEEP1B  image vectors, 96-dim float32, CC BY 4.0
#           storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/
#   GloVe6B text vectors, 100-dim, PDDL / public domain
#           downloads.cs.stanford.edu/nlp/data/glove.6B.zip
#
# Usage:
#   ./testdata/fetch_embeddings.sh            # SF=1 (default): first 800k DEEP base vectors
#   ./testdata/fetch_embeddings.sh --sf 10    # first 8M DEEP base vectors
#
set -euo pipefail

SF=1
while [ $# -gt 0 ]; do
  case "$1" in
    --sf) SF="$2"; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

# HARD GUARD — local-only. CI and the remote test hosts receive generated parquet via
# rsync; they must never reach out to the dataset origins (network policy + determinism:
# the bytes are pinned here once and shipped). Bail loudly if a CI or known remote env
# is detected.
if [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ] || [ -n "${PEACOCK_NO_FETCH:-}" ]; then
  echo "error: fetch_embeddings.sh is LOCAL-ONLY (CI/remote env detected via CI/GITHUB_ACTIONS/PEACOCK_NO_FETCH)." >&2
  echo "       Embedding sources are fetched on a dev box; hosts receive the generated parquet." >&2
  exit 1
fi
case "$(hostname 2>/dev/null)" in
  llm-gpu0h200*|*shad-gpu*)
    echo "error: fetch_embeddings.sh must not run on the remote GPU host ($(hostname))." >&2
    exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE="${SCRIPT_DIR}/embeddings-cache"
mkdir -p "$CACHE"

DEEP=https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP
GLOVE_URL=https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip

# fbin layout: int32 num_vectors, int32 num_dims, then num_vectors*num_dims row-major
# float32. DEEP1B base is 1e9 x 96. ps_image_embedding maps ORDINALLY to base[row], so
# we only need the first N = 800000*SF vectors (the partsupp row count). Range-fetch the
# 8-byte header + N*96*4 bytes rather than the full 384GB object. The saved slice keeps
# the original header (num_vectors still reads 1e9); the generator reads exactly N rows.
DEEP_DIMS=96
N=$(( 800000 * SF ))
BASE_BYTES=$(( 8 + N * DEEP_DIMS * 4 ))

# Pinned SHA256, computed on the first pull from the immutable sources. Locking these
# guarantees byte-identical embedding inputs on every later run / dev box.
SHA_BASE_SF1=3e0dbaede4d3a989f53b19e6046e69dccd38654a92bb7387decece28b13b2e9e
SHA_QUERY=8438fc763f14e0f9741fda15b3e11215aef089ce17d1ad47d53b52a7c9fda5bb
SHA_GT=9429e662db0f1a24a78a279f41ab1a20405e372178875e79f342d3a57ced9970
SHA_GLOVE_ZIP=617afb2fe6cbd085c235baf7a465b96f4112bd7f7ccb2b2cbd649fed9cbcf2fb

BASE_FILE="$CACHE/deep_base.sf${SF}.fbin"
QUERY_FILE="$CACHE/deep_query.public.10K.fbin"
GT_FILE="$CACHE/deep_groundtruth.public.10K.ibin"
GLOVE_ZIP="$CACHE/glove.6B.zip"
GLOVE_100D="$CACHE/glove.6B.100d.txt"

# verify <file> <sha256>  — empty sha means "cannot verify" (returns non-zero).
verify() { [ -n "$2" ] && [ -f "$1" ] && echo "$2  $1" | sha256sum -c --status; }

# fetch <url> <outfile> <sha> [extra curl args...]  — skip if already verified.
fetch() {
  local url="$1" out="$2" sha="$3"; shift 3
  if verify "$out" "$sha"; then echo "  cached+verified: $(basename "$out")"; return 0; fi
  echo "  fetching $(basename "$out") ..."
  curl -fSL --retry 5 --retry-delay 3 "$@" "$url" -o "$out"
  if [ -n "$sha" ]; then
    verify "$out" "$sha" || { echo "error: checksum mismatch for $out" >&2; exit 1; }
    echo "  ok (sha256 verified): $(basename "$out")"
  else
    echo "  ok (no pinned sha for SF=$SF; source is immutable): $(basename "$out")"
  fi
}

echo "==> DEEP1B base slice: first $N vectors (SF=$SF), $BASE_BYTES bytes"
# Source-authenticity gate that works for ANY SF with ONE pinned constant: because
# base.1B.fbin is immutable and every slice is a byte-PREFIX of the next, the first
# 800k-vector prefix (header + 800000*96 float32 = 307200008 bytes) is identical across
# all SF. So we assert head -c 307200008 == the pinned SF=1 hash regardless of SF — this
# authenticates the whole file (for SF=1 the prefix IS the file). SF1_PREFIX_BYTES is a
# fixed constant, NOT derived from $SF.
SF1_PREFIX_BYTES=307200008   # 8 + 800000*96*4
base_prefix_ok() {
  [ -f "$BASE_FILE" ] && [ "$(stat -c%s "$BASE_FILE")" -ge "$SF1_PREFIX_BYTES" ] \
    && [ "$(head -c "$SF1_PREFIX_BYTES" "$BASE_FILE" | sha256sum | awk '{print $1}')" = "$SHA_BASE_SF1" ]
}
if base_prefix_ok && [ "$(stat -c%s "$BASE_FILE")" -eq "$BASE_BYTES" ]; then
  echo "  cached+verified: $(basename "$BASE_FILE")"
else
  echo "  fetching $(basename "$BASE_FILE") ..."
  curl -fSL --retry 5 --retry-delay 3 -r "0-$((BASE_BYTES-1))" "$DEEP/base.1B.fbin" -o "$BASE_FILE"
  base_prefix_ok || { echo "error: DEEP base prefix-gate sha256 mismatch (want $SHA_BASE_SF1 over first $SF1_PREFIX_BYTES bytes)" >&2; exit 1; }
  echo "  ok (sha256 prefix-gate verified): $(basename "$BASE_FILE")"
fi

echo "==> DEEP1B query set + groundtruth (full)"
fetch "$DEEP/query.public.10K.fbin"       "$QUERY_FILE" "$SHA_QUERY" -C -
fetch "$DEEP/groundtruth.public.10K.ibin" "$GT_FILE"    "$SHA_GT"    -C -

echo "==> GloVe 6B (zip once, extract 100d)"
fetch "$GLOVE_URL" "$GLOVE_ZIP" "$SHA_GLOVE_ZIP" -C -
if [ ! -f "$GLOVE_100D" ]; then
  echo "  extracting glove.6B.100d.txt ..."
  unzip -o "$GLOVE_ZIP" glove.6B.100d.txt -d "$CACHE" >/dev/null
fi
echo "  glove.6B.100d.txt: $(wc -l < "$GLOVE_100D") tokens"

echo "Done. Embedding sources in $CACHE:"
ls -lh "$CACHE"
