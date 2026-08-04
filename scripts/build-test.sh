#!/bin/bash

# pipefail is load-bearing here, not boilerplate: the staging step pipes
# `cargo test --no-run … | python3`, and without it the pipeline takes python's
# status, so a cargo failure is invisible until the emptiness check downstream.
set -euo pipefail

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
MODE=cpu                                                      # cpu | gpu (set via --gpu); cpu is the default
RUST_ONLY=0                                                   # --rust-only: build cpu/plan test bins with --features rust-only (NO C++/FFI); for golden regen + cpu/plan verify on verda
MODE_FLAG=""                                                  # which mode flag was actually passed, for the conflict message

# Dedicated 26.02 C++ build dir, separate from the default cpp/build (which is
# kept at 25.02 on purpose). Using a distinct dir also avoids the find_package
# stale-cache trap: cmake caches the resolved cudf_DIR, so reconfiguring a dir
# that first found a 25.02 env would keep using it even with a new cudf_ROOT.
BUILD_DIR=cpp/build26
INSTALL_DIR="$BUILD_DIR/install"
CUDA_ARCHITECTURES="80;90"
RUST_TESTS_STAGING="$INSTALL_DIR/rust-tests"

BUILD=0
PUSH_BINARIES=0  # --push-binaries: ship cpp/install (lib + C++ bins + staged rust bins) + goldens
RUN=0
UPDATE_CANON=0   # --update-canonical: regen goldens during the remote run (UPDATE_CANONICAL=1)

# Per-kind testdata sync, one flag per kind in each direction. The parser IS the
# validator: an unknown kind is an unknown flag, caught by the `*) usage` arm before
# any side effect. The old --push-testdata KIND[,KIND] form validated inside a $(…),
# where `exit 1` ends only the subshell — the error printed and the script carried on.
PUSH_KINDS=()
PULL_KINDS=()

usage() {
  cat >&2 <<'USAGE'
Usage: build-test.sh --host <ssh-dest> [mode] <action...> [options]

  mode      (default cpu)
    --rust-only    every Rust target that builds WITHOUT cmake. Golden regen + verify.
    --gpu          the GPU-runtime targets; needs a GPU on the remote.
    (no --cpu flag: cpu IS the default, and a flag whose only effect is the default
     is noise. --gpu and --rust-only are mutually exclusive and rejected, not ordered.)

  MODE LADDER: rust-only ⊂ cpu ⊂ gpu-capable. A mode that BUILDS more never RUNS less —
  the three suites are derived from the sources, not listed, so they cannot drift apart.

  actions (at least one required)
    --build              build C++ (unless --rust-only) + stage the Rust test binaries
    --push-binaries      ship cpp/install + the goldens the binaries assert against
    --run                run the suite on the remote
    --all                --build --push-binaries --run
    --push-<kind>        parquet | queries | goldens | duckdb-profiles | duckdb-dynfilters
    --pull-<kind>        parquet | goldens | duckdb-profiles | duckdb-dynfilters
                         (no --pull-queries: a --pull-<kind> exists only where the
                          remote can PRODUCE that kind. Queries are AUTHORED — nothing
                          on a remote generates a .sql — so pulling would overwrite
                          committed source with whatever a scratch host holds. The
                          other four are generated artifacts: parquet on verda,
                          goldens by --update-canonical, both duckdb sets by their
                          extractors.)
    --update-canonical   regen goldens during --run instead of asserting
    (--push-binaries always ships goldens: binaries without the fixtures they assert
     against is what produced 110/110 "canonical file not found". --push-goldens is the
     subset operation — refresh fixtures without rebuilding or reshipping binaries.)

  options
    --remote-dir <path> --local-cudf-root <path> --remote-cudf-root <path>
    --gcc-version <n>

  DELIBERATELY ABSENT, so these do not get filed as gaps:
    no --pull-binaries      binaries flow one way: built locally, shipped to the remote.
    no embeddings-cache     a per-host INTERMEDIATE for the tpch vector datasets
                            (fetch_embeddings.sh, ~1.8 GB, gitignored). Regenerate it
                            where you need it; syncing it would ship a derived artifact.
USAGE
  exit 1
}

# The ONE goldens push. It existed twice — the --push-binaries block used
# `rsync -r --delete`, --push-goldens used `rsync -a --delete` — same intent, different
# metadata handling, no shared code.
#
# PUSH MIRRORS (--delete); PULL IS ADDITIVE. The asymmetry is deliberate and must not be
# "fixed": the remote is a PARTIAL mirror. testdata/goldens/ contains tpch.sf40/ (16
# CSVs) and sf40 lives on shad-gpu, so mirroring downward from verda would DELETE
# fixtures that host never had — out of a git working tree.
#
# Accepted consequence: a regen that deletes a .result.txt (result over 256 KB, see
# maybe_write_result_golden) cannot propagate that deletion back through an additive
# pull. The deletion is announced on stderr and reaches the operator through the ssh
# heredoc; that is the handling, not a --delete armed for one rare case.
# #113: ship every COMMITTED testdata fixture, swept from git rather than named by hand.
#
# The hand list is the defect, not the missing file. build-test.sh shipped goldens and
# nothing else, so widening the rust-only suite surfaced two absent fixtures at once
# (cost-registry.csv, cost_model.conf) — and load_csv's own panic text had already
# predicted exactly this, naming the two OTHER provisioning paths that had been fixed
# one at a time. Three paths, three lists, each patched only when something broke.
#
# `--cached --others --exclude-standard` is #113's prescription and both halves matter:
#   --others         picks up a NEW fixture that is not committed yet; plain ls-files
#                    would miss it and reintroduce the same class.
#   --exclude-standard honours .gitignore, which is what keeps the ~1.7 GB of GENERATED
#                    parquet (tpch.sf1, tpcds.sf1) out — those are produced on the
#                    remote, not shipped to it.
# Do NOT add --exclude=*.parquet: tpch.minimal commits 5 parquet files the tests need.
#
# goldens are excluded here and pushed by sync_goldens instead — they are the one kind
# that must MIRROR (--delete) so a regen's baseline is the committed set exactly.
sync_fixtures() {
  local list
  list=$(mktemp)
  git ls-files --cached --others --exclude-standard testdata \
    | grep -v '^testdata/goldens/' > "$list"
  local n
  n=$(wc -l < "$list")
  [ "$n" -gt 0 ] || { echo "error: no tracked testdata fixtures found — git sweep is wrong" >&2; rm -f "$list"; exit 1; }
  echo "==> push $n committed testdata fixtures -> $HOST:$REMOTE_DIR/"
  rsync -a --files-from="$list" ./ "$HOST:$REMOTE_DIR/"
  rm -f "$list"
}

sync_goldens() {
  case "$1" in
    push)
      echo "==> push goldens -> $HOST:$REMOTE_DIR/testdata/goldens"
      ssh "$HOST" "mkdir -p '$REMOTE_DIR/testdata/goldens'"
      rsync -a --delete testdata/goldens/ "$HOST:$REMOTE_DIR/testdata/goldens/"
      ;;
    pull)
      echo "==> pull $HOST:$REMOTE_DIR/testdata/goldens -> testdata/goldens (additive)"
      mkdir -p testdata/goldens
      rsync -a "$HOST:$REMOTE_DIR/testdata/goldens/" testdata/goldens/
      ;;
    *) echo "sync_goldens: bad direction '$1'" >&2; exit 1 ;;
  esac
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

# A value-taking flag must be given a value: `--host` at the end of the line would
# otherwise consume nothing and leave HOST empty, which reads downstream as "no --host".
need_value() {
  [ -n "${2:-}" ] || { echo "error: $1 requires a value" >&2; exit 1; }
}

# --gpu and --rust-only are mutually exclusive, and REJECTED rather than resolved by
# order. Resolving by order made the same two flags mean different things depending on
# how they were typed, and one order was half-applied: MODE=gpu with RUST_ONLY still
# live left LD_LIBRARY_PATH unset for the GPU binaries, so they could not resolve
# libpeacock_gpu and failed as if the product were broken.
set_mode() {
  if [ -n "$MODE_FLAG" ] && [ "$MODE_FLAG" != "$1" ]; then
    echo "error: $MODE_FLAG and $1 are mutually exclusive — they select different" >&2
    echo "       build configurations (rust-only builds without cmake; gpu needs the" >&2
    echo "       linked executor and a GPU at runtime). Pass one." >&2
    exit 1
  fi
  MODE_FLAG="$1"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --host)             need_value "$1" "${2:-}"; HOST="$2"; shift ;;
    --remote-dir)       need_value "$1" "${2:-}"; REMOTE_DIR="$2"; shift ;;
    --local-cudf-root)  need_value "$1" "${2:-}"; LOCAL_CUDF_ROOT="$2"; shift ;;
    --remote-cudf-root) need_value "$1" "${2:-}"; REMOTE_CUDF_ROOT="$2"; shift ;;
    --gcc-version)      need_value "$1" "${2:-}"; GCC_VERSION="$2"; shift ;;
    --gpu)              set_mode "$1"; MODE=gpu ;;
    --rust-only)        set_mode "$1"; MODE=cpu; RUST_ONLY=1 ;;
    --build)            BUILD=1 ;;
    --push-binaries)    PUSH_BINARIES=1 ;;
    --run)              RUN=1 ;;
    --all)              BUILD=1; PUSH_BINARIES=1; RUN=1 ;;
    --update-canonical) UPDATE_CANON=1 ;;
    --push-parquet)           PUSH_KINDS+=(parquet) ;;
    --push-queries)           PUSH_KINDS+=(queries) ;;
    --push-goldens)           PUSH_KINDS+=(goldens) ;;
    --push-duckdb-profiles)   PUSH_KINDS+=(duckdb-profiles) ;;
    --push-duckdb-dynfilters) PUSH_KINDS+=(duckdb-dynfilters) ;;
    --pull-parquet)           PULL_KINDS+=(parquet) ;;
    --pull-goldens)           PULL_KINDS+=(goldens) ;;
    --pull-duckdb-profiles)   PULL_KINDS+=(duckdb-profiles) ;;
    --pull-duckdb-dynfilters) PULL_KINDS+=(duckdb-dynfilters) ;;
    *) echo "Unknown flag: $1" >&2; usage ;;
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
      # SECOND AXIS: what does the target verify — the RUNTIME, or the REPO?
      # A test that reads the checkout (.github/workflows, the tests/ tree) cannot be
      # verified from a shipped binary: none of that travels with it. Running
      # test_ci_coverage on verda compared today's binary against a pipeline.yml left
      # there by some earlier push — verda is not even a git checkout — so it failed
      # for reasons that say nothing about the remote. Measured: it is the ONLY target
      # that reaches outside testdata, so this excludes exactly one thing today and
      # classifies the next one automatically.
      # It still runs locally and in CI, which is where a repo check belongs.
      if grep -qE 'repo_root|\.github/workflows' "$f"; then
        continue
      fi
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

# ---- validation: EVERYTHING below runs before the first side effect ----------
# A typo must fail before anything is built, shipped or deleted — not halfway through.

# An invocation that does nothing and exits 0 is indistinguishable from a successful
# one. `--host x` alone used to be exactly that.
if [ "$BUILD" -eq 0 ] && [ "$PUSH_BINARIES" -eq 0 ] && [ "$RUN" -eq 0 ] \
   && [ ${#PUSH_KINDS[@]} -eq 0 ] && [ ${#PULL_KINDS[@]} -eq 0 ]; then
  echo "error: no action requested — nothing would happen and the script would exit 0." >&2
  echo "       Pass at least one of --build / --push-binaries / --run / --all /" >&2
  echo "       --push-<kind> / --pull-<kind>." >&2
  usage
fi

if { [ "$PUSH_BINARIES" -eq 1 ] || [ "$RUN" -eq 1 ] || [ ${#PUSH_KINDS[@]} -gt 0 ] \
     || [ ${#PULL_KINDS[@]} -gt 0 ]; } && [ -z "$HOST" ]; then
  echo "error: --host is required for --push-binaries/--run/--push-<kind>/--pull-<kind>" >&2
  echo "       (e.g. --host dmitry@86.38.182.185)" >&2
  exit 1
fi

# A derived suite that comes back EMPTY must be an error. `mapfile` from a helper that
# prints nothing yields a zero-length array, every `for` body over it vanishes, and the
# script exits 0 having run no tests — a derivation typo would report success.
if [ ${#RUST_TESTS[@]} -eq 0 ]; then
  echo "error: the derived Rust suite is EMPTY for mode '${MODE_FLAG:---cpu}'." >&2
  echo "       This is a bug in the derivation (see rust_only_targets/needs_cmake_targets)," >&2
  echo "       not a valid configuration: a run with no targets would exit 0 having" >&2
  echo "       verified nothing." >&2
  exit 1
fi

# Push named testdata kinds local -> remote (before any --run that consumes them).
# --delete keeps the remote subtree exact (drops files removed locally).
if [ ${#PUSH_KINDS[@]} -gt 0 ]; then
  for kind in "${PUSH_KINDS[@]}"; do
    for d in $(testdata_dirs_for_kind "$kind"); do
      [ -d "testdata/$d" ] || { echo "--push-$kind: skip missing testdata/$d"; continue; }
      if [ "$kind" = "goldens" ]; then
        sync_goldens push
      else
        echo "==> push testdata/$d -> $HOST:$REMOTE_DIR/testdata/$d"
        ssh "$HOST" "mkdir -p '$REMOTE_DIR/testdata/$d'"
        rsync -a --delete "testdata/$d/" "$HOST:$REMOTE_DIR/testdata/$d/"
      fi
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
    # Stage from EMPTY, as build-test-shadgpu.sh does: a binary left by a previous
    # mode is otherwise shipped alongside this mode's, and only the run-by-explicit-name
    # loop keeps it from executing. Renaming a target makes the orphan permanent.
    rm -rf "$RUST_TESTS_STAGING"
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
    # Stage from EMPTY here too — see the rust-only branch above.
    rm -rf "$RUST_TESTS_STAGING"
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

if [ "$PUSH_BINARIES" -eq 1 ]; then
  # Strip the rust test binaries before shipping: unstripped debug builds link
  # the whole DataFusion/Arrow stack and are huge. --strip-debug drops only the
  # debug sections, keeping the dynamic symbol table intact.
  for spec in "${RUST_TESTS[@]}"; do
    t="${spec##*:}"
    [ -f "$RUST_TESTS_STAGING/$t" ] && strip --strip-debug "$RUST_TESTS_STAGING/$t"
  done
  echo "==> rsync $INSTALL_DIR to $HOST:$REMOTE_DIR/cpp/install"
  ssh "$HOST" "mkdir -p '$REMOTE_DIR/cpp/install'"
  # --delete, and the source is "$INSTALL_DIR/" NOT "$INSTALL_DIR"/*: with a glob rsync
  # receives several sources and --delete does not mean what it looks like.
  # Without it remote orphans are permanent — verda was carrying test_cpu_executor and
  # test_cpu_h200 (this branch's PRE-RENAME binaries) plus three targets deleted in
  # June. Clearing the local staging dir stops shipping NEW orphans; only --delete
  # removes the ones already there. It matters most where the runner GLOBS
  # rust-tests/* (build-test-shadgpu.sh, pipeline.yml) — there an orphan RUNS.
  rsync -r -P --delete "$INSTALL_DIR/" "$HOST:$REMOTE_DIR/cpp/install/"

  if [ -d testdata/goldens ]; then
    # Ship the committed goldens (testdata/goldens/<dataset>.sf<N>/) so they match
    # the just-built binaries — version-controlled fixtures, run-independent of the
    # remote's checked-out commit. Heavy parquet datasets are generated on the
    # remote, untouched.
    # TWO reasons, and the regen one is not a footnote:
    #   VERIFY  the binaries assert against exactly these files.
    #   REGEN   the push establishes the BASELINE. --delete makes the remote tree equal
    #           the local committed set, so what comes back is
    #           (local-committed ∪ regenerated) rather than (remote-leftovers ∪
    #           regenerated). Without it a regen inherits whatever a previous run left.
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
    sync_goldens push
  fi
  # Fixtures the binaries READ but which are not goldens (cost-registry.csv,
  # cost_model.conf, the query .sql sets, tpch.minimal). Swept from git — see
  # sync_fixtures.
  sync_fixtures
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
if [ ${#PULL_KINDS[@]} -gt 0 ]; then
  for kind in "${PULL_KINDS[@]}"; do
    for d in $(testdata_dirs_for_kind "$kind"); do
      if [ "$kind" = "goldens" ]; then
        # No *.txt filter any more. Its real job was keeping plan_bytes.sha256 out of
        # the round trip, and that is now the TEST's job (test_plan_bytes refuses to
        # regenerate without PEACOCK_REWRITE_PLAN_BYTES). Two mechanisms for one
        # invariant, with the weaker one in the wrong layer, is how they drift. The
        # filter also silently dropped the 16 sf40 CSVs.
        sync_goldens pull
      else
        echo "==> pull $HOST:$REMOTE_DIR/testdata/$d -> testdata/$d (additive)"
        mkdir -p "testdata/$d"
        rsync -a "$HOST:$REMOTE_DIR/testdata/$d/" "testdata/$d/"
      fi
    done
  done
fi
