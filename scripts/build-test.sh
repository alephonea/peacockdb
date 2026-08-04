#!/bin/bash

set -e

# Build the test suite locally and run it on a remote host. Mirrors
# build-test-shadgpu.sh (build here, ship binaries, run there). Two suites,
# selected by --cpu (default) or --gpu:
#
#   --cpu        C++ peacock_cpu_tests + every Rust target that builds with cmake
#                (a SUPERSET of --rust-only, plus test_ffi).
#   --rust-only   every Rust target that builds WITHOUT cmake — see build-test.md's
#                "What rust-only means". Golden regen + cpu/plan verify.
#   --gpu        C++ peacock_plan_tests + the GPU-runtime Rust targets, run
#                one-at-a-time on the GPU.
# The three sets are DERIVED from the sources, not listed here — see the suite
# selection below. A mode that builds more must not run less.
#
# The remote host is NOT hardcoded — pass it with --host. We build locally
# against a cuDF that matches the remote's ABI (default: a local cudf-26.02
# env), ship cpp/install (the lib + C++ test binaries + staged Rust binaries),
# and run them against the remote's cuDF runtime. There is no --patch step
# (the remote is a modern-glibc host, unlike the shad-gpu path).
#
# NOTE: the Rust CPU test crates bake their testdata path from CARGO_MANIFEST_DIR
# (cargo canonicalizes symlinks), so a binary built here looks for testdata at
# this box's absolute repo path (e.g. /media/data/peacockdb/testdata). Until they
# honor PEACOCK_TESTDATA_DIR (issue #49), the remote must expose that same path —
# e.g. a symlink /media/data/peacockdb -> <remote repo>. The GPU test crate DOES
# honor PEACOCK_TESTDATA_DIR, which --gpu sets, so it needs no such symlink.

# ---- defaults (override via flags) -----------------------------------------
HOST=""                                                       # ssh destination, e.g. dmitry@86.38.182.185 (required)
REMOTE_DIR="/home/dmitry/peacockdb"                           # repo dir on the remote (holds testdata + receives cpp/install)
LOCAL_CUDF_ROOT="/home/dmitry/data/miniforge3/envs/rapids"    # local cuDF (26.02) to build against
REMOTE_CUDF_ROOT="/home/dmitry/miniforge3/envs/rapids-26.02"  # cuDF runtime libs on the remote
GCC_VERSION=14                                                # gcc-N for the C++/cmake build (cuDF 26.02 / CUDA 12.x accepts 14)
MODE=cpu                                                      # cpu | gpu (set via --cpu / --gpu)
RUST_ONLY=0                                                   # --rust-only: build cpu/plan test bins with --features rust-only (NO C++/FFI); for golden regen + cpu/plan verify on verda

# Dedicated 26.02 C++ build dir, separate from the default cpp/build (which is
# kept at 25.02 on purpose). Using a distinct dir also avoids the find_package
# stale-cache trap: cmake caches the resolved cudf_DIR, so reconfiguring a dir
# that first found a 25.02 env would keep using it even with a new cudf_ROOT.
BUILD_DIR=cpp/build26
INSTALL_DIR="$BUILD_DIR/install"
CUDA_ARCHITECTURES="80;90"
RUST_TESTS_STAGING="$INSTALL_DIR/rust-tests"

BUILD=0
RSYNC=0
RUN=0
UPDATE_CANON=0   # --update-canonical: regen goldens during the remote run (UPDATE_CANONICAL=1)
FETCH_GOLDENS=0  # --fetch-goldens: pull the remote goldens back into the local repo after the run
PUSH_TESTDATA="" # --push-testdata KIND[,KIND]: rsync local testdata kinds -> remote
PULL_TESTDATA="" # --pull-testdata KIND[,KIND]: rsync remote testdata kinds -> local
# Per-kind testdata sync. KIND in {parquet,goldens,duckdb-profiles,queries}; each
# maps to one or more dirs under testdata/. Used to move datasets generated once on
# verda to local/shad-gpu (parquet), or pull regenerated goldens back, without
# touching the (separately shipped) test binaries.

usage() {
  echo "Usage: $0 --host <ssh-dest> [--cpu|--gpu|--rust-only] [--remote-dir <path>] [--local-cudf-root <path>] [--remote-cudf-root <path>] [--gcc-version <n>] [--build] [--rsync] [--run] [--all] [--update-canonical] [--fetch-goldens] [--push-testdata KIND[,KIND]] [--pull-testdata KIND[,KIND]]"
  echo "  --rust-only: cpu/plan goldens via --features rust-only (no C++/FFI); regen with --update-canonical, fetch with --fetch-goldens, verify by re-running without --update-canonical"
  echo "  testdata KIND: parquet | goldens | duckdb-profiles | duckdb-dynfilters | queries"
  exit 1
}

# Map a testdata KIND to its repo-relative dir(s) under testdata/.
testdata_dirs_for_kind() {
  case "$1" in
    parquet)          echo "tpch.sf1 tpcds.sf1 tpch.minimal" ;;
    goldens)          echo "goldens" ;;
    duckdb-profiles)  echo "duckdb-profiles" ;;
    duckdb-dynfilters) echo "duckdb-dynfilters" ;;
    queries)          echo "tpch-queries tpcds-queries tpch-vec-queries" ;;
    *) echo "error: unknown testdata kind '$1' (parquet|goldens|duckdb-profiles|duckdb-dynfilters|queries)" >&2; exit 1 ;;
  esac
}

if [ $# -eq 0 ]; then usage; fi

while [ $# -gt 0 ]; do
  case "$1" in
    --host)             HOST="$2"; shift ;;
    --remote-dir)       REMOTE_DIR="$2"; shift ;;
    --local-cudf-root)  LOCAL_CUDF_ROOT="$2"; shift ;;
    --remote-cudf-root) REMOTE_CUDF_ROOT="$2"; shift ;;
    --gcc-version)      GCC_VERSION="$2"; shift ;;
    --cpu)              MODE=cpu ;;
    --gpu)              MODE=gpu ;;
    --rust-only)        MODE=cpu; RUST_ONLY=1 ;;
    --build)            BUILD=1 ;;
    --rsync)            RSYNC=1 ;;
    --run)              RUN=1 ;;
    --all)              BUILD=1; RSYNC=1; RUN=1 ;;
    --update-canonical) UPDATE_CANON=1 ;;
    --fetch-goldens)    FETCH_GOLDENS=1 ;;
    --push-testdata)    PUSH_TESTDATA="$2"; shift ;;
    --pull-testdata)    PULL_TESTDATA="$2"; shift ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
  shift
done

# Suite selection, DERIVED — each entry is <package>:<test-name>; each binary is staged
# under <install>/rust-tests/<name> so the rsync step ships it.
#
# Targets are classified ONCE, by what they REQUIRE, and each mode takes everything it
# can support. Three hand-written lists is what this replaces: they drifted apart, and
# the drift was always in the direction of running less than the mode could (--gpu
# silently skipped test_inc2_conformance, the murmur3 conformance gate; the default
# branch omitted four targets that build fine with the full feature set).
#
# A mode that BUILDS more must not RUN less. So:
#   needs_cmake   file-gated on not(rust-only): cannot compile without libpeacock_gpu.
#   rust_only     everything else — no cmake, no CUDA, CPU by construction.
#   default       rust_only + needs_cmake, minus GPU-runtime-only targets.
#   gpu           the GPU-runtime set.
#
# The rust-only membership test is the FEATURE'S OWN DEFINITION (see build-test.md's
# "What rust-only means"): a file-level `#![cfg(not(feature = "rust-only"))]` is exactly
# the marker that says "this cannot exist without the FFI". Derived from the sources, so
# adding a test file classifies itself.
rust_only_targets() {
  local pkg dir f base
  for pkg in peacockdb-core peacockdb-ffi; do
    dir="$pkg/tests"
    [ -d "$dir" ] || continue
    for f in "$dir"/*.rs; do
      [ -e "$f" ] || continue
      base=$(basename "$f" .rs)
      # File-level gate => needs cmake; excluded from the rust-only set.
      grep -qF '#![cfg(not(feature = "rust-only"))]' "$f" && continue
      # EXCLUSIONS, each for a measured reason rather than taste:
      #   test_ffi  compiles under rust-only but yields ZERO tests (measured) — the
      #             whole file is behind one cfg. Staging it ships a binary that runs
      #             nothing, which reads as coverage.
      #   diag_flip_audit  a manual diagnostic with no assertions (see its module doc
      #             and its test_ci_coverage exemption). It cannot fail, so it cannot
      #             contribute to a verify run, and a regen run must not depend on it.
      case "$base" in
        test_ffi|diag_flip_audit) continue ;;
      esac
      printf '%s:%s\n' "$pkg" "$base"
    done
  done
}

# Targets that need cmake to compile at all, in dependency-name order.
needs_cmake_targets() {
  local pkg dir f base
  for pkg in peacockdb-core peacockdb-ffi; do
    dir="$pkg/tests"
    [ -d "$dir" ] || continue
    for f in "$dir"/*.rs; do
      [ -e "$f" ] || continue
      base=$(basename "$f" .rs)
      grep -qF '#![cfg(not(feature = "rust-only"))]' "$f" && printf '%s:%s\n' "$pkg" "$base"
    done
  done
  # test_ffi is not file-gated but is empty without the FFI, so it belongs to the
  # cmake side rather than the rust-only one.
  printf '%s\n' "peacockdb-ffi:test_ffi"
}

# Targets that must NOT be swept into a regen run. --update-canonical exports
# UPDATE_CANONICAL=1 for EVERY staged binary, so membership here is the only thing
# standing between a bulk regen and a golden that is supposed to move deliberately.
#
#   test_plan_bytes  its digests ARE the wire-format guard: a diff means the FlatBuffer
#                    layout moved, and the C++ side reads those bytes. Its own header
#                    says "regenerate deliberately". Auto-regenerating it would convert
#                    the guard into a rubber stamp — it would rewrite the evidence
#                    instead of failing. It stays in the VERIFY set, just not the regen.
#
# Deliberately NOT excluded: test_cost_model. .cost.txt is derived from .cpu.txt, which
# a regen DOES rewrite, so leaving it out is what makes a regen self-inconsistent —
# today's list regenerates .cpu.txt and leaves every .cost.txt stale, so test_cost_model
# goes red immediately after a "successful" regen. Including it fixes that.
regen_excluded() {
  case "$1" in
    *:test_plan_bytes) return 0 ;;
    *) return 1 ;;
  esac
}

if [ "$MODE" = "gpu" ]; then
  # GPU-runtime set. Kept in step with build-test-shadgpu.sh:RUST_TESTS and
  # pipeline.yml's gpu-tests staging array — three runners had three lists and this
  # one was short by test_inc2_conformance, the GPU<->comet bit-exact murmur3 gate.
  RUST_TESTS=(
    peacockdb-core:test_gpu_full_table
    peacockdb-core:test_gpu_partitioned
    peacockdb-core:test_inc2_conformance
  )
  CPP_TEST_BIN=peacock_plan_tests
elif [ "$RUST_ONLY" -eq 1 ]; then
  # Golden regen / cpu+plan verify: no C++, no FFI.
  mapfile -t RUST_TESTS < <(rust_only_targets)
  CPP_TEST_BIN=""
else
  # Superset: everything rust-only can run, plus what only cmake makes buildable.
  # test_gpu_* are excluded — they compile here but need a GPU at RUNTIME, which is
  # what --gpu is for.
  mapfile -t RUST_TESTS < <(
    rust_only_targets
    needs_cmake_targets | grep -v ':test_gpu_'
  )
  CPP_TEST_BIN=peacock_cpu_tests
fi

# Drop the regen-excluded targets when this run is a regen.
if [ "$UPDATE_CANON" -eq 1 ]; then
  _kept=()
  for spec in "${RUST_TESTS[@]}"; do
    if regen_excluded "$spec"; then
      echo "==> --update-canonical: skipping $spec (must be regenerated deliberately, not swept)"
    else
      _kept+=("$spec")
    fi
  done
  RUST_TESTS=("${_kept[@]}")
fi

# --fetch-goldens is shorthand for --pull-testdata goldens.
if [ "$FETCH_GOLDENS" -eq 1 ]; then
  PULL_TESTDATA="${PULL_TESTDATA:+$PULL_TESTDATA,}goldens"
fi

if { [ "$RSYNC" -eq 1 ] || [ "$RUN" -eq 1 ] || [ -n "$PUSH_TESTDATA" ] || [ -n "$PULL_TESTDATA" ]; } && [ -z "$HOST" ]; then
  echo "error: --host is required for --rsync/--run/--push-testdata/--pull-testdata (e.g. --host dmitry@86.38.182.185)" >&2
  exit 1
fi

# Push named testdata kinds local -> remote (before any --run that consumes them).
# --delete keeps the remote subtree exact (drops files removed locally).
if [ -n "$PUSH_TESTDATA" ]; then
  IFS=',' read -ra _kinds <<< "$PUSH_TESTDATA"
  for kind in "${_kinds[@]}"; do
    for d in $(testdata_dirs_for_kind "$kind"); do
      [ -d "testdata/$d" ] || { echo "--push-testdata: skip missing testdata/$d"; continue; }
      echo "==> push testdata/$d -> $HOST:$REMOTE_DIR/testdata/$d"
      ssh "$HOST" "mkdir -p '$REMOTE_DIR/testdata/$d'"
      rsync -a --delete "testdata/$d/" "$HOST:$REMOTE_DIR/testdata/$d/"
    done
  done
fi

if [ "$BUILD" -eq 1 ]; then
  CARGO_FEATURES=""
  if [ "$RUST_ONLY" -eq 1 ]; then
    # No C++/FFI — build the test binaries with --features rust-only (the part that
    # compiles locally without the cuDF toolchain). Goldens are rust-only artifacts.
    # Uses the default ./target so it stays warm alongside plain `cargo test`.
    echo "==> build rust-only test binaries (no C++/FFI)"
    CARGO_FEATURES="--features rust-only"
    mkdir -p "$RUST_TESTS_STAGING"
  else
    # cudf (default-feature) build: isolate it in its OWN target dir so it doesn't
    # recompile the arrow/DataFusion subgraph every time it alternates with a
    # rust-only build sharing ./target (rust-only toggles arrow's `ffi` feature →
    # different fingerprint). PLUS key the dir off the cuDF root: a different cudf_ROOT
    # (this script's local rapids-26.02 vs build-test-shadgpu.sh's rapids-cuda-12.2)
    # busts the FFI/arrow fingerprints, so sharing one target-cudf across versions
    # recompiles the whole DataFusion stack on every switch. See build-test-shadgpu.sh.
    export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target-cudf-$(basename "$LOCAL_CUDF_ROOT")}"
    # Low-memory throttle (same rationale as build-test-shadgpu.sh): a cold opt-3
    # cudf build OOMs a 15GiB host at full parallelism. Override with CARGO_BUILD_JOBS.
    if [ -z "${CARGO_BUILD_JOBS:-}" ]; then
      _mem_gib=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 32)
      if [ "${_mem_gib:-32}" -lt 20 ]; then
        export CARGO_BUILD_JOBS=3
        echo "==> low-memory host (${_mem_gib}GiB RAM): throttling CARGO_BUILD_JOBS=3"
      fi
    fi
    echo "==> build C++ in $BUILD_DIR against cuDF at $LOCAL_CUDF_ROOT (gcc-$GCC_VERSION)"
    # cuDF env first on PATH so nvcc/cmake/ninja resolve from the rapids env.
    export PATH="$LOCAL_CUDF_ROOT/bin:$PATH"
    export CC=/usr/bin/gcc-${GCC_VERSION}
    export CXX=/usr/bin/g++-${GCC_VERSION}
    export CUDACXX="$LOCAL_CUDF_ROOT/bin/nvcc"
    export LDFLAGS="-Wl,-rpath-link,$LOCAL_CUDF_ROOT/lib"
    # Drive cmake directly (build.sh hardcodes cpp/build) so we land in cpp/build26
    # and leave the 25.02 cpp/build untouched.
    # ccache when available (host compilers only — ccache+nvcc is unreliable).
    ccache_flags=""
    if command -v ccache >/dev/null 2>&1; then
      ccache_flags="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
      echo "==> ccache found; using it as the C/C++ compiler launcher"
    fi
    cmake -S cpp -B "$BUILD_DIR" -G Ninja \
      $ccache_flags \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      "-DCMAKE_JOB_POOLS=link_pool=1" \
      -DCMAKE_JOB_POOL_LINK=link_pool \
      -Dcudf_ROOT="$LOCAL_CUDF_ROOT"
    # link_pool=1 above serializes links (Ninja): at most 1 binary links at a time —
    # parallel links OOM this host class; compiles still run at full parallelism.
    cmake --build "$BUILD_DIR" --parallel "$(nproc)"
    cmake --install "$BUILD_DIR"

    echo "==> stage Rust $MODE test binaries"
    mkdir -p "$RUST_TESTS_STAGING"
    export CUDF_ROOT="$LOCAL_CUDF_ROOT"
    # The FFI crate builds its own libpeacock_gpu via the cmake crate in cargo's
    # OUT_DIR, which caches the resolved cudf_DIR. Clean ONLY when the cuDF root
    # actually changed (stamp under CARGO_TARGET_DIR, same scheme as
    # build-test-shadgpu.sh) — an unconditional clean rebuilds flatbuffers+gtest+
    # libpeacock_gpu on every --build. PEACOCK_FFI_CLEAN=1 forces it.
    ffi_root_stamp="$CARGO_TARGET_DIR/.peacock-ffi-cudf-root"
    if [ "${PEACOCK_FFI_CLEAN:-0}" = "1" ] \
       || [ ! -f "$ffi_root_stamp" ] \
       || [ "$(cat "$ffi_root_stamp" 2>/dev/null)" != "$CUDF_ROOT" ]; then
      echo "--- peacockdb-ffi: cuDF root changed or clean forced; cleaning to reconfigure cmake"
      cargo clean -p peacockdb-ffi
      mkdir -p "$CARGO_TARGET_DIR"
      printf '%s\n' "$CUDF_ROOT" > "$ffi_root_stamp"
    else
      echo "--- peacockdb-ffi: cuDF root unchanged ($CUDF_ROOT); skipping clean"
    fi
  fi
  for spec in "${RUST_TESTS[@]}"; do
    pkg="${spec%%:*}"
    t="${spec##*:}"
    # cargo test --no-run prints a json artifact line per built target; the
    # integration test we want has .target.name == $t and a non-null .executable.
    exec_path=$(cargo test --no-run $CARGO_FEATURES -p "$pkg" --test "$t" \
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
      echo "ERROR: failed to locate built binary for $pkg:$t"; exit 1
    fi
    cp -f "$exec_path" "$RUST_TESTS_STAGING/$t"
    echo "--- Staged rust test: $RUST_TESTS_STAGING/$t"
  done
fi

if [ "$RSYNC" -eq 1 ]; then
  # Strip the rust test binaries before shipping: unstripped debug builds link
  # the whole DataFusion/Arrow stack and are huge. --strip-debug drops only the
  # debug sections, keeping the dynamic symbol table intact.
  for spec in "${RUST_TESTS[@]}"; do
    t="${spec##*:}"
    [ -f "$RUST_TESTS_STAGING/$t" ] && strip --strip-debug "$RUST_TESTS_STAGING/$t"
  done
  echo "==> rsync $INSTALL_DIR to $HOST:$REMOTE_DIR/cpp/install"
  ssh "$HOST" "mkdir -p '$REMOTE_DIR/cpp/install'"
  rsync -r -P "$INSTALL_DIR"/* "$HOST:$REMOTE_DIR/cpp/install/"

  if [ -d testdata/goldens ]; then
    # Ship the committed goldens (testdata/goldens/<dataset>.sf<N>/) so they match
    # the just-built binaries — version-controlled fixtures, run-independent of the
    # remote's checked-out commit. Heavy parquet datasets are generated on the
    # remote, untouched.
    # (For an --update-canonical run the remote regenerates these in place.)
    #
    # EVERY mode ships them; there is no exclusion here, and both exclusions this
    # line used to carry were bugs of the same shape:
    #   --rust-only IS the golden/plan verify mode, so it needs them most.
    #   --gpu was skipped on the claim that "the GPU suite uses no goldens". False:
    #     assert_gpu_query verifies the per-node cost tree against the .cpu.txt on
    #     EVERY run, and the final result against the .result.txt in the three
    #     golden_* modes.
    # In both cases the run verified the new binaries against whatever stale goldens
    # the remote happened to have — a device-label skew alone produced 110/110
    # "canonical file not found".
    echo "==> rsync goldens testdata/goldens"
    rsync -r --delete testdata/goldens/ "$HOST:$REMOTE_DIR/testdata/goldens/"
  fi
fi

if [ "$RUN" -eq 1 ]; then
  # PCK_TEST_FILTER=<sub>  name filter forwarded to each test binary.
  : "${PCK_TEST_FILTER:=}"

  # GPU tests share one process-wide cuDF/RMM pool, so they must run
  # sequentially (--test-threads=1) and locate testdata via PEACOCK_TESTDATA_DIR
  # (the GPU crate honors it). CPU tests have neither constraint.
  if [ "$MODE" = "gpu" ]; then
    THREADS_ARG="--test-threads=1"
    TESTDATA_ENV="export PEACOCK_TESTDATA_DIR=$REMOTE_DIR/testdata"
  elif [ "$RUST_ONLY" -eq 1 ]; then
    THREADS_ARG=""
    # rust-only test crates honor PEACOCK_TESTDATA_DIR -> point at the remote testdata.
    TESTDATA_ENV="export PEACOCK_TESTDATA_DIR=$REMOTE_DIR/testdata"
  else
    THREADS_ARG=""
    TESTDATA_ENV=":"
  fi

  # rust-only binaries link neither libpeacock_gpu nor cuDF, so skip LD_LIBRARY_PATH.
  if [ "$RUST_ONLY" -eq 1 ]; then
    LD_ENV=":"
  else
    LD_ENV="export LD_LIBRARY_PATH=$REMOTE_DIR/cpp/install/lib:$REMOTE_CUDF_ROOT/lib:\$LD_LIBRARY_PATH"
  fi

  # --update-canonical: the test binaries regenerate their goldens in-place
  # (UPDATE_CANONICAL=1) on the remote instead of asserting against them. The
  # regenerated goldens are pulled back to the local repo after the run.
  if [ "$UPDATE_CANON" -eq 1 ]; then
    UPDATE_CANON_ENV="export UPDATE_CANONICAL=1"
  else
    UPDATE_CANON_ENV=":"
  fi

  # Run only this mode's binaries by explicit name — globbing rust-tests/* would
  # also pick up stale binaries left by a previous run of the other mode (the
  # rsync doesn't --delete), e.g. test_cpu_full_table lingering during a --gpu run.
  RUST_TEST_NAMES=""
  for spec in "${RUST_TESTS[@]}"; do RUST_TEST_NAMES="$RUST_TEST_NAMES ${spec##*:}"; done

  echo "==> $MODE tests on $HOST"
  # Unquoted heredoc: $VARS expand locally; escape with \$ for remote expansion.
  ssh "$HOST" bash <<EOF
    # Deliberately no 'set -e': run every test binary even when an earlier one
    # fails (so e.g. a failing C++ test doesn't skip the Rust tests), then fail
    # at the end if anything failed. Each result is OR'd into rc.
    # cpp/install/lib first so libpeacock_gpu.so resolves for the test binaries
    # (their baked rpath points at this build host); then the remote's cuDF libs.
    $LD_ENV
    $TESTDATA_ENV
    $UPDATE_CANON_ENV

    rc=0

    if [ -n "$CPP_TEST_BIN" ]; then
      echo "==> $CPP_TEST_BIN (C++)"
      "$REMOTE_DIR/cpp/install/bin/$CPP_TEST_BIN" || rc=1
    fi

    echo "==> Rust $MODE integration tests (filter='$PCK_TEST_FILTER')"
    for name in $RUST_TEST_NAMES; do
      t="$REMOTE_DIR/cpp/install/rust-tests/\$name"
      [ -x "\$t" ] || { echo "--- \$name: missing, skipping"; continue; }
      echo "--- \$name"
      "\$t" --nocapture $THREADS_ARG '$PCK_TEST_FILTER' || rc=1
    done

    exit \$rc
EOF
fi

# Pull named testdata kinds remote -> local. Runs last, so an --update-canonical
# --run --fetch-goldens pulls the regenerated goldens only after the run succeeded
# (set -e). For goldens we pull only *.txt (the cost/plan goldens); other kinds
# (e.g. parquet generated once on verda) sync in full. Opt-in so a plain run never
# overwrites local data.
if [ -n "$PULL_TESTDATA" ]; then
  IFS=',' read -ra _kinds <<< "$PULL_TESTDATA"
  for kind in "${_kinds[@]}"; do
    for d in $(testdata_dirs_for_kind "$kind"); do
      echo "==> pull $HOST:$REMOTE_DIR/testdata/$d -> testdata/$d"
      mkdir -p "testdata/$d"
      if [ "$kind" = "goldens" ]; then
        rsync -r --include='*/' --include='*.txt' --exclude='*' \
          "$HOST:$REMOTE_DIR/testdata/$d/" "testdata/$d/"
      else
        rsync -a --delete "$HOST:$REMOTE_DIR/testdata/$d/" "testdata/$d/"
      fi
    done
  done
fi
