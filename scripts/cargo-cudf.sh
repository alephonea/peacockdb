#!/usr/bin/env bash
# Run cargo for cudf-feature (FFI) builds with the per-root target dir, so ad-hoc
# invocations never land in ./target and thrash the rust-only cache (feature flags +
# cudf_ROOT changes bust fingerprints; see llm-wiki/build-test.md).
#   CUDF_ROOT=<rapids env> scripts/cargo-cudf.sh test -p peacockdb-core --test test_gpu --no-run
set -euo pipefail
: "${CUDF_ROOT:?set CUDF_ROOT to the rapids env to build against}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target-cudf-$(basename "$CUDF_ROOT")}"
exec cargo "$@"
