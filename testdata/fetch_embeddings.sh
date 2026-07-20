#!/bin/bash
#
# Fetch open-dataset embedding sources for `generate_testdata.sh --embeddings external`.
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

# HARD GUARD — never fetch from an AUTOMATED run. CI must never reach out to the
# dataset origins (network policy + determinism: the bytes are pinned once and the
# generated parquet is distributed).
#
# The guard is on the ENVIRONMENT, not the host, and that distinction matters: the
# GPU host is BOTH the CI runner AND the only machine with the disk to generate the
# large scale factors (sf200 needs ~380GB parquet + ~62GB of sources; dev boxes have
# nowhere near that). So a MANUAL run on that host is legitimate and allowed, while a
# CI run on the very same machine is still blocked — CI sets CI/GITHUB_ACTIONS, and
# any other automation can set PEACOCK_NO_FETCH to opt out explicitly.
if [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ] || [ -n "${PEACOCK_NO_FETCH:-}" ]; then
  echo "error: fetch_embeddings.sh must not run from CI/automation" >&2
  echo "       (detected via CI / GITHUB_ACTIONS / PEACOCK_NO_FETCH)." >&2
  echo "       Automated jobs consume generated parquet; they never fetch sources." >&2
  exit 1
fi

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
  # Same stall guard as the base slice: some origins are unreachable from some hosts
  # (the GPU box cannot reach downloads.cs.stanford.edu at all — the connection opens
  # and then transfers nothing), and without this curl sits there until --retry gives
  # up minutes later. Abort a dead connection fast instead.
  # If an origin is blocked from a host, copy the file in from a machine that can reach
  # it: the pinned SHA256 below is what makes that safe.
  curl -fSL --retry 5 --retry-delay 3 --speed-limit "${FETCH_MIN_BPS:-262144}" --speed-time 120 \
       "$@" "$url" -o "$out"
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
base_prefix_ok() {   # $1 = file
  [ -f "$1" ] && [ "$(stat -c%s "$1")" -ge "$SF1_PREFIX_BYTES" ] \
    && [ "$(head -c "$SF1_PREFIX_BYTES" "$1" | sha256sum | awk '{print $1}')" = "$SHA_BASE_SF1" ]
}

# Every slice is a byte-PREFIX of every larger one (immutable source, always
# range-fetched from offset 0), so an already-downloaded BIGGER slice serves a smaller
# SF as-is: sf40 reads the first 12GB of the sf200 file instead of pulling its own
# 12GB copy. Link to it rather than duplicating ~12-61GB on disk.
larger_slice_for() {   # $1 = bytes needed -> prints a usable file, or returns 1
  local need=$1 f sz
  for f in "$CACHE"/deep_base.sf*.fbin; do
    [ -e "$f" ] || continue
    [ "$f" -ef "$BASE_FILE" ] 2>/dev/null && continue
    sz=$(stat -Lc%s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$need" ] && base_prefix_ok "$f"; then echo "$f"; return 0; fi
  done
  return 1
}

# PRE-FLIGHT FREE SPACE. The guard above is environment-based, so a manual run can now
# happen on ANY machine — including hosts with a small root filesystem, where a 61GB pull
# would fill / and destabilise the box rather than merely failing. Refuse before the
# first byte, counting only what is actually still missing.
need=0
have_base=$(stat -Lc%s "$BASE_FILE" 2>/dev/null || echo 0)
if [ "$have_base" -lt "$BASE_BYTES" ] && ! larger_slice_for "$BASE_BYTES" >/dev/null 2>&1; then
  need=$(( need + BASE_BYTES - have_base ))
fi
[ -f "$QUERY_FILE" ] || need=$(( need + 3840008 ))
[ -f "$GT_FILE" ]    || need=$(( need + 4000008 ))
[ -f "$GLOVE_100D" ] || need=$(( need + 862182613 + 347116733 ))   # zip + extracted 100d
need=$(( need + need / 20 + 1073741824 ))                          # 5% + 1GiB margin
avail=$(df -PB1 "$CACHE" | awk 'NR==2 {print $4}')
if [ "$need" -gt "${avail:-0}" ]; then
  echo "error: not enough free space for the fetch on $(df -P "$CACHE" | awk 'NR==2{print $6}')" >&2
  echo "       need ~$(( need / 1048576 )) MiB (incl. margin), have $(( ${avail:-0} / 1048576 )) MiB" >&2
  echo "       SF=$SF wants a $(( BASE_BYTES / 1048576 )) MiB DEEP slice; use a smaller --sf or a bigger volume." >&2
  exit 1
fi
echo "  pre-flight: need ~$(( need / 1048576 )) MiB, have $(( ${avail:-0} / 1048576 )) MiB free"

if base_prefix_ok "$BASE_FILE" && [ "$(stat -Lc%s "$BASE_FILE")" -ge "$BASE_BYTES" ]; then
  echo "  cached+verified: $(basename "$BASE_FILE") ($(stat -Lc%s "$BASE_FILE") bytes, need $BASE_BYTES)"
elif donor=$(larger_slice_for "$BASE_BYTES"); then
  ln -sfn "$donor" "$BASE_FILE"
  echo "  reusing prefix of $(basename "$donor") ($(stat -Lc%s "$donor") bytes >= $BASE_BYTES) -> $(basename "$BASE_FILE")"
else
  # RESUMABLE ranged download. curl's --retry RESTARTS a transfer rather than resuming
  # it, and -o truncates the target first; at sf1 (307MB) that is invisible, but at
  # sf200 (61GB) over a flaky link a single hiccup would cost the entire pull. So drive
  # the range ourselves: append only the bytes still missing and loop until complete.
  # (-C - cannot be combined with an explicit -r range.)
  [ -L "$BASE_FILE" ] && rm -f "$BASE_FILE"          # never append into a reused donor
  have=$(stat -c%s "$BASE_FILE" 2>/dev/null || echo 0)
  if [ "$have" -gt "$BASE_BYTES" ]; then rm -f "$BASE_FILE"; have=0; fi   # stale bigger file
  stall=0
  while [ "$have" -lt "$BASE_BYTES" ]; do
    echo "  fetching bytes ${have}..$((BASE_BYTES-1))  ($(( (BASE_BYTES - have) / 1048576 )) MiB remaining)"
    # --speed-limit/--speed-time turn a dead-but-open connection into a fast abort +
    # resume instead of an indefinite hang. 256KB/s over 120s: low enough not to thrash
    # on a genuinely slow-but-progressing link, high enough to kill a stalled one in 2min.
    curl -fL --retry 5 --retry-delay 3 --speed-limit "${FETCH_MIN_BPS:-262144}" --speed-time 120 \
         -r "${have}-$((BASE_BYTES-1))" "$DEEP/base.1B.fbin" >> "$BASE_FILE" || true
    new=$(stat -c%s "$BASE_FILE" 2>/dev/null || echo 0)
    if [ "$new" -le "$have" ]; then
      stall=$((stall + 1))
      if [ "$stall" -ge 10 ]; then
        echo "error: no download progress after 10 attempts (stuck at $new/$BASE_BYTES bytes)" >&2
        exit 1
      fi
      echo "  no progress; retrying in 10s (attempt $stall/10)"; sleep 10
    else
      stall=0
    fi
    have=$new
  done
  base_prefix_ok "$BASE_FILE" || { echo "error: DEEP base prefix-gate sha256 mismatch (want $SHA_BASE_SF1 over first $SF1_PREFIX_BYTES bytes)" >&2; exit 1; }
  echo "  ok (sha256 prefix-gate verified): $(basename "$BASE_FILE")"
fi
df -h "$CACHE" | tail -1 | awk '{print "  cache disk: "$4" free ("$5" used)"}'

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
