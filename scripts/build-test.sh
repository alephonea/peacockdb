#!/bin/bash

set -e

# Build the test suite locally and run it on a remote host. Mirrors
# build-test-shadgpu.sh (build here, ship binaries, run there). Two suites,
# selected by --cpu (default) or --gpu:
#
#   --cpu  C++ peacock_cpu_tests + the Rust CPU integration tests
#          (test_plan_serialiser, test_query_plan, test_cpu_executor, test_ffi).
#   --gpu  C++ peacock_plan_tests + the Rust GPU integration test
#          (test_gpu_executor, TPC-H + TPC-DS), run one-at-a-time on the GPU.
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

usage() {
  echo "Usage: $0 --host <ssh-dest> [--cpu|--gpu] [--remote-dir <path>] [--local-cudf-root <path>] [--remote-cudf-root <path>] [--gcc-version <n>] [--build] [--rsync] [--run] [--all] [--update-canonical] [--fetch-goldens]"
  exit 1
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
    --build)            BUILD=1 ;;
    --rsync)            RSYNC=1 ;;
    --run)              RUN=1 ;;
    --all)              BUILD=1; RSYNC=1; RUN=1 ;;
    --update-canonical) UPDATE_CANON=1 ;;
    --fetch-goldens)    FETCH_GOLDENS=1 ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
  shift
done

# Suite selection. Each entry is <package>:<test-name>; each binary is staged
# under <install>/rust-tests/<name> so the rsync step ships it.
if [ "$MODE" = "gpu" ]; then
  RUST_TESTS=(peacockdb-core:test_gpu_executor)
  CPP_TEST_BIN=peacock_plan_tests
else
  RUST_TESTS=(
    peacockdb-core:test_plan_serialiser
    peacockdb-core:test_query_plan
    peacockdb-core:test_cpu_executor
    peacockdb-ffi:test_ffi
  )
  CPP_TEST_BIN=peacock_cpu_tests
fi

if { [ "$RSYNC" -eq 1 ] || [ "$RUN" -eq 1 ]; } && [ -z "$HOST" ]; then
  echo "error: --host is required for --rsync/--run (e.g. --host dmitry@86.38.182.185)" >&2
  exit 1
fi

if [ "$BUILD" -eq 1 ]; then
  echo "==> build C++ in $BUILD_DIR against cuDF at $LOCAL_CUDF_ROOT (gcc-$GCC_VERSION)"
  # cuDF env first on PATH so nvcc/cmake/ninja resolve from the rapids env.
  export PATH="$LOCAL_CUDF_ROOT/bin:$PATH"
  export CC=/usr/bin/gcc-${GCC_VERSION}
  export CXX=/usr/bin/g++-${GCC_VERSION}
  export CUDACXX="$LOCAL_CUDF_ROOT/bin/nvcc"
  export LDFLAGS="-Wl,-rpath-link,$LOCAL_CUDF_ROOT/lib"
  # Drive cmake directly (build.sh hardcodes cpp/build) so we land in cpp/build26
  # and leave the 25.02 cpp/build untouched.
  cmake -S cpp -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -Dcudf_ROOT="$LOCAL_CUDF_ROOT"
  cmake --build "$BUILD_DIR" --parallel "$(nproc)"
  cmake --install "$BUILD_DIR"

  echo "==> stage Rust $MODE test binaries"
  mkdir -p "$RUST_TESTS_STAGING"
  export CUDF_ROOT="$LOCAL_CUDF_ROOT"
  # The FFI crate builds its own libpeacock_gpu via the cmake crate in cargo's
  # OUT_DIR, which carries the same stale cudf_DIR risk as the C++ build dir.
  # Clean it so it reconfigures against the 26.02 root selected above.
  cargo clean -p peacockdb-ffi
  for spec in "${RUST_TESTS[@]}"; do
    pkg="${spec%%:*}"
    t="${spec##*:}"
    # cargo test --no-run prints a json artifact line per built target; the
    # integration test we want has .target.name == $t and a non-null .executable.
    exec_path=$(cargo test --no-run -p "$pkg" --test "$t" \
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

  if [ "$MODE" = "cpu" ] && [ -d testdata/goldens ]; then
    # Ship the committed goldens (testdata/goldens/<dataset>.sf<N>/) so they match
    # the just-built binaries — version-controlled fixtures, run-independent of the
    # remote's checked-out commit. Heavy parquet datasets are generated on the
    # remote, untouched. The GPU suite uses no goldens, so skip in --gpu mode.
    # (For an --update-canonical run the remote regenerates these in place.)
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
  else
    THREADS_ARG=""
    TESTDATA_ENV=":"
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
  # rsync doesn't --delete), e.g. test_cpu_executor lingering during a --gpu run.
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
    export LD_LIBRARY_PATH="$REMOTE_DIR/cpp/install/lib:$REMOTE_CUDF_ROOT/lib:\$LD_LIBRARY_PATH"
    $TESTDATA_ENV
    $UPDATE_CANON_ENV

    rc=0

    echo "==> $CPP_TEST_BIN (C++)"
    "$REMOTE_DIR/cpp/install/bin/$CPP_TEST_BIN" || rc=1

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

# Pull the remote goldens back into the local repo — only with --fetch-goldens
# (typically paired with --update-canonical, which regenerates *.txt via
# test_query_plan and *.cpu.txt via test_cpu_executor on the remote). Opt-in so a
# plain verify run never overwrites local goldens. set -e means we only reach
# here if the run succeeded.
if [ "$RUN" -eq 1 ] && [ "$FETCH_GOLDENS" -eq 1 ] && [ "$MODE" = "cpu" ]; then
  echo "==> fetching goldens back from $HOST"
  rsync -r --include='*/' --include='*.txt' --exclude='*' \
    "$HOST:$REMOTE_DIR/testdata/goldens/" testdata/goldens/
fi
