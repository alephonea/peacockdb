#!/usr/bin/env bash
# Run cargo for cudf-feature (FFI) builds with the per-root target dir, so ad-hoc
# invocations never land in ./target and thrash the rust-only cache (feature flags +
# cudf_ROOT changes bust fingerprints; see llm-wiki/build-test.md).
#   CUDF_ROOT=<rapids env> scripts/cargo-cudf.sh test -p peacockdb-core --test test_gpu --no-run
set -euo pipefail
: "${CUDF_ROOT:?set CUDF_ROOT to the rapids env to build against}"
_root="$(basename "$CUDF_ROOT")"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target-cudf-$_root}"

# CC/CXX must MATCH what the build-test script for this same cuDF root exports, and
# for the same reason the target dir is per-root: cc-rs emits rerun-if-env-changed on
# CC/CXX, so entering one target dir with a different C compiler re-runs every native
# build script (zstd-sys, bzip2-sys, lzma-sys, psm, blake3) and drags the whole
# DataFusion stack rebuild along behind it. Leaving these unset here made this script
# and build-test-shadgpu.sh thrash each other's cache on every alternation.
#
# Keyed on the SAME basename as the target dir — one key picks both, so they cannot
# drift apart. Exhaustive on purpose: an unknown root must state its gcc rather than
# default to one that silently invalidates the dir it lands in.
case "$_root" in
  rapids-cuda-12.2) GCC_VERSION="${GCC_VERSION:-12}" ;;  # cuDF 25.02 — build-test-shadgpu.sh
  rapids)           GCC_VERSION="${GCC_VERSION:-14}" ;;  # cuDF 26.02 — build-test.sh
  *)
    if [ -z "${GCC_VERSION:-}" ]; then
      echo "cargo-cudf.sh: unknown cuDF root '$_root' — set GCC_VERSION to the gcc the" \
           "build-test script uses for it (cuDF 25.02 -> 12, 26.02 -> 14)" >&2
      exit 1
    fi
    ;;
esac
export CC="${CC:-/usr/bin/gcc-${GCC_VERSION}}"
export CXX="${CXX:-/usr/bin/g++-${GCC_VERSION}}"

exec cargo "$@"
