#!/bin/bash

set -e

CUDF_ROOT=/home/dmitry/data/miniforge3/envs/rapids-cuda-12.2
export CUDF_ROOT

# nvcc 12.2 (the conda env's CUDA toolkit) hard-rejects gcc>12 in
# host_config.h. Ubuntu's default cc/c++ is gcc-14, so pin gcc-12 for
# both the C++ build (via --gcc-version below) and the cargo cmake
# invocation (via CC/CXX, which the `cmake` crate honors).
#   sudo apt install gcc-12 g++-12
GCC_VERSION=12
export CC=/usr/bin/gcc-${GCC_VERSION}
export CXX=/usr/bin/g++-${GCC_VERSION}

# Isolate the cudf (default-feature, C++/FFI-linked) build in its OWN target dir so
# it never competes with `--features rust-only` builds sharing ./target. Toggling
# rust-only re-enables arrow's `ffi` feature → a different arrow/DataFusion
# fingerprint, so a shared target dir recompiles that subgraph on every flip.
#
# FURTHER: key the dir off the cuDF ROOT. A DIFFERENT cudf_ROOT (e.g. the local
# rapids-26.02 used by build-test.sh vs this script's rapids-cuda-12.2/25.02) changes
# the FFI build's resolved Arrow/cudf and busts the C-build-script fingerprints, so
# sharing ONE target-cudf across cudf versions recompiles the whole DataFusion stack
# on every switch (build-test.sh ⇄ build-test-shadgpu.sh / local pre-flight checks).
# One dir per cudf root = each version stays permanently warm. Override w/ CARGO_TARGET_DIR.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target-cudf-$(basename "$CUDF_ROOT")}"

# The cudf build recompiles the whole DataFusion stack at opt-3 (#85); the FIRST
# build in a fresh target-cudf is a full cold rebuild. On a memory-constrained host
# a full-parallelism cold build exhausts RAM+swap and gets OOM-killed (seen on a
# 15GiB box: two kills before throttling). Cap parallel cargo jobs on low-memory
# hosts; override with CARGO_BUILD_JOBS.
if [ -z "${CARGO_BUILD_JOBS:-}" ]; then
  _mem_gib=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 32)
  if [ "${_mem_gib:-32}" -lt 20 ]; then
    export CARGO_BUILD_JOBS=3
    echo "==> low-memory host (${_mem_gib}GiB RAM): throttling CARGO_BUILD_JOBS=3 to avoid OOM"
  fi
fi

# Rust integration tests that link libpeacock_gpu.so and need to run on the GPU host.
# After build, each binary is staged under cpp/install/rust-tests/<name> so the
# existing rsync step picks them up alongside the C++ binaries.
RUST_TESTS=(test_gpu test_inc2_conformance)
RUST_TESTS_STAGING=cpp/install/rust-tests

BUILD=0
RSYNC=0
PATCH=0
RUN=0

if [ $# -eq 0 ]; then
  echo "Usage: $0 [--build] [--rsync] [--patch] [--run] [--all]"
  exit 1
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=1 ;;
    --rsync) RSYNC=1 ;;
    --patch) PATCH=1 ;;
    --run)   RUN=1 ;;
    --all)   BUILD=1; RSYNC=1; PATCH=1; RUN=1 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

# rsync over the flaky/bursty shad-gpu link. The connection stalls often, so
# make each transfer self-healing instead of an all-or-nothing shot:
#   --partial --inplace  keep partially-sent files and update them in place, so
#                        a retry resumes the same file (no temp-rename restart).
#   --timeout=90         abort a stalled connection instead of hanging forever,
#                        so the loop can reconnect and resume.
# Retries until rsync reports success (each attempt is bounded by --timeout),
# capped so a genuinely-down host eventually fails instead of looping forever.
# Caller passes mode flags (e.g. -r / -a) and the src/dst; -P/--partial/--inplace
# /--timeout are added here.
resilient_rsync() {
  local attempt=1 max_attempts=100 rc=0
  while :; do
    rsync -P --partial --inplace --timeout=90 "$@" && return 0
    rc=$?
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "rsync: giving up after $attempt attempts (last rc=$rc)" >&2
      return "$rc"
    fi
    echo "rsync: attempt $attempt stalled/failed (rc=$rc); resuming in 5s..." >&2
    attempt=$((attempt + 1))
    sleep 5
  done
}

if [ "$BUILD" -eq 1 ]; then
  ./scripts/build.sh --cudf_ROOT "$CUDF_ROOT" --gcc-version "$GCC_VERSION" --configure
  ./scripts/build.sh --cudf_ROOT "$CUDF_ROOT" --gcc-version "$GCC_VERSION" --build
  ./scripts/build.sh --cudf_ROOT "$CUDF_ROOT" --gcc-version "$GCC_VERSION" --install

  # The FFI crate builds its own libpeacock_gpu.so via the `cmake` crate in
  # cargo's OUT_DIR, and cmake caches the resolved cudf_DIR/Arrow there. If the
  # previous build used a different cuDF root (e.g. the CPU build-test.sh path's
  # rapids-26.02), that stale cache makes the link pick the wrong Arrow and fail
  # (`ld returned 1`). So clean the FFI crate to force a cmake reconfigure — but
  # ONLY when the cuDF root actually changed since the last build. Cleaning
  # unconditionally rebuilds the whole cmake sub-tree (flatbuffers + gtest +
  # libpeacock_gpu.so) from scratch on every --build, which dominates wall-clock
  # when the root is unchanged across iterations. A stamp under CARGO_TARGET_DIR
  # records the root the FFI was last configured against; it survives
  # `cargo clean -p peacockdb-ffi` (which removes only that crate's artifacts).
  # Set PEACOCK_FFI_CLEAN=1 to force the clean regardless.
  ffi_root_stamp="$CARGO_TARGET_DIR/.peacock-ffi-cudf-root"
  if [ "${PEACOCK_FFI_CLEAN:-0}" = "1" ] \
     || [ ! -f "$ffi_root_stamp" ] \
     || [ "$(cat "$ffi_root_stamp" 2>/dev/null)" != "$CUDF_ROOT" ]; then
    echo "--- peacockdb-ffi: cuDF root changed or clean forced; cleaning to reconfigure cmake"
    cargo clean -p peacockdb-ffi
    mkdir -p "$CARGO_TARGET_DIR"
    printf '%s\n' "$CUDF_ROOT" > "$ffi_root_stamp"
  else
    echo "--- peacockdb-ffi: cuDF root unchanged ($CUDF_ROOT); skipping clean (reuse cmake _deps)"
  fi

  mkdir -p "$RUST_TESTS_STAGING"
  for t in "${RUST_TESTS[@]}"; do
    # cargo test --no-run prints a json artifact line per built target; the
    # integration test we want has .target.name == $t and a non-null .executable.
    exec_path=$(cargo test --no-run -p peacockdb-core --test "$t" \
        --message-format=json \
      | python3 -c '
import json, sys
name = sys.argv[1]
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    if m.get("executable") and (m.get("target") or {}).get("name") == name:
        print(m["executable"]); break
' "$t")
    if [ -z "$exec_path" ] || [ ! -f "$exec_path" ]; then
      echo "ERROR: failed to locate built binary for $t"; exit 1
    fi
    cp -f "$exec_path" "$RUST_TESTS_STAGING/$t"
    echo "--- Staged rust test: $RUST_TESTS_STAGING/$t"
  done
fi

if [ "$RSYNC" -eq 1 ]; then
  # Always strip the rust test binaries before shipping. Unstripped debug builds
  # are ~565MB each and choke the (sometimes very slow / bursty) link to
  # shad-gpu; stripped they are ~155MB. --strip-debug drops only the debug
  # sections, keeping the dynamic symbol table the glibc patchelf step needs.
  for t in "${RUST_TESTS[@]}"; do
    [ -f "$RUST_TESTS_STAGING/$t" ] && strip --strip-debug "$RUST_TESTS_STAGING/$t"
  done
  resilient_rsync -r cpp/install/* shad-gpu:/home/info/peacockdb/cpp/install/
  # Ship the result/cost/cpu GOLDENS the rust GPU tests assert against. Without this
  # the remote keeps whatever goldens a PREVIOUS run left, so a locally-regenerated
  # golden (e.g. a join subtree flipping 1→8 partitions) silently compares the fresh
  # GPU output against a STALE golden → false-RED. --delete keeps the remote tree an
  # exact mirror so removed/renamed goldens don't linger. (Cheap: goldens are small.)
  ssh shad-gpu "mkdir -p /home/info/peacockdb/testdata/goldens"
  resilient_rsync -r --delete testdata/goldens/ shad-gpu:/home/info/peacockdb/testdata/goldens/
  # Ship our setup-glibc.sh (with patch_rust_dir) so --patch uses the
  # version that knows about cpp/install/rust-tests/.
  ssh shad-gpu "mkdir -p /home/info/peacockdb/scripts"
  resilient_rsync -a scripts/setup-glibc.sh shad-gpu:/home/info/peacockdb/scripts/
fi

if [ "$PATCH" -eq 1 ]; then
  ssh shad-gpu "/home/info/peacockdb/scripts/setup-glibc.sh --repo-dir /home/info/peacockdb --patch"
fi

if [ "$RUN" -eq 1 ]; then
  # Optional knobs (set in the caller's env, not via flags):
  #   PEACOCK_GPU_DEBUG=1    enable PCK_TRACE + per-node cudaStreamSynchronize
  #                          in plan_executor.cpp (localizes async errors).
  #   PCK_TEST_FILTER=<sub>  cargo-test name filter forwarded to the rust
  #                          binary (e.g. gpu_tpch_sf1_q13_H200). Empty = run all.
  #   PCK_RUN_CPP=0          skip peacock_plan_tests (default: run them).
  : "${PEACOCK_GPU_DEBUG:=}"
  : "${PCK_TEST_FILTER:=}"
  : "${PCK_RUN_CPP:=1}"

  # Note the heredoc uses no quoting on the EOF marker, so $VARS expand
  # *locally* before being sent to the remote shell. Escape with \$ for
  # any var that should be expanded remotely (e.g. \$LD_LIBRARY_PATH).
  # Deliberately NO 'set -e' around the test runs, matching the CI gpu-tests job:
  # run EVERY binary even when an earlier one fails, OR each exit code into rc, and
  # fail at the end. Under set -e a single crashing test aborted the whole remote
  # script — a SIGSEGV in test_gpu silently cost us all 10 test_inc2_conformance
  # results, which read as "not run" but looked like "fine". One flaky test must not
  # be able to hide every later binary's result.
  ssh shad-gpu bash <<EOF
    export PEACOCK_TESTDATA_DIR=/home/info/peacockdb/testdata
    export PEACOCK_GPU_DEBUG='$PEACOCK_GPU_DEBUG'
    # cpp/install/lib first so libpeacock_gpu.so resolves for the rust test
    # binary (its baked-in rpath points at the build host's cargo target).
    export LD_LIBRARY_PATH=/home/info/peacockdb/cpp/install/lib:/usr/local/cuda-12.5/compat:/home/info/glibc-2.35/lib:\$HOME/miniforge3/envs/rapids-cuda-12.2/lib:\$LD_LIBRARY_PATH

    rc=0

    if [ '$PCK_RUN_CPP' = '1' ]; then
      echo "==> peacock_plan_tests (C++)"
      /home/info/peacockdb/cpp/install/bin/peacock_plan_tests || rc=1
    fi

    echo "==> rust GPU integration tests (filter='$PCK_TEST_FILTER')"
    for t in /home/info/peacockdb/cpp/install/rust-tests/*; do
      [ -x "\$t" ] || continue
      echo "--- \$(basename "\$t")"
      # --test-threads=1: GPU/RMM context is process-wide, parallel tests OOM.
      "\$t" --nocapture --test-threads=1 '$PCK_TEST_FILTER'
      status=\$?
      if [ "\$status" -ne 0 ]; then
        # 139 = SIGSEGV. Name it explicitly: a bare non-zero code here has already
        # been mistaken for an assertion failure.
        echo "!!! \$(basename "\$t") FAILED (exit \$status)"
        rc=1
      fi
    done

    if [ "\$rc" -ne 0 ]; then
      echo "==> GPU test run FAILED (see '!!!' lines above)"
    else
      echo "==> GPU test run OK"
    fi
    exit "\$rc"
EOF
fi
