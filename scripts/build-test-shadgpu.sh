#!/bin/bash
#
# Build for the GPU host, ship, patch, and either gate on it or measure it.
#
# The gate (--run) and the benchmark run (--run-benchmarks) are one script because they
# share a toolchain, target dir, push and patch. They must never share an exit code: one
# number cannot mean both "correctness passed" and "measurement completed", and OR-ing
# them makes a benchmark-infrastructure failure read as a test regression. Hence the
# validation block, and two staging dirs:
#
#   cpp/install/rust-tests/        swept by --run's glob      -> gate
#   cpp/install/rust-benchmarks/   not swept                  -> measurement
#
# --run enforces that rather than documenting it: a benchmark binary under rust-tests/
# turns the run red.
#
# --build / --build-benchmarks need a cuDF toolchain, in practice scripts/docker-build.sh.
# Every later phase needs this workstation's ssh keys and is refused in the container,
# where the failure would surface as an ssh error deep inside a phase.
#
# USAGE
#   ./scripts/build-test-shadgpu.sh --all                # gate: build+push+patch+run
#   scripts/docker-build.sh --no-image -- ./scripts/build-test-shadgpu.sh --build-benchmarks
#   ./scripts/build-test-shadgpu.sh --push-binaries --patch --run-benchmarks --pull-benchmarks
#
# Both runs take tens of minutes, hence a detached form for each:
#   ./scripts/build-test-shadgpu.sh --push-binaries --patch --run-benchmarks-detached
#   ./scripts/build-test-shadgpu.sh --benchmark-status    # going? finished? log tail
#   ./scripts/build-test-shadgpu.sh --pull-benchmarks     # once it reports finished
#
#   PCK_TEST_FILTER=bench_tpch_sf1_q1 ./scripts/build-test-shadgpu.sh --run-benchmarks
#
# BENCHMARK OUTPUT
#   testdata/benchmark-results/<dataset>.sf<sf>/<mode>.benchmark.txt
# one file per (dataset, mode), holding a section per query timed at that mode. A run
# with no filter times every declared mode, and each mode's sections land in its own
# file — the modes do not share one. Written on the GPU host and copied back by
# --pull-benchmarks; llm-wiki/build-test.md has the file format.
#
#   testdata/calibration/records.tsv        (git-ignored)
# The same run also emits calibration rows, one per cuDF CALL. Unconditionally
# rather than behind a flag: the rows are derived from the run that wrote the tree
# above, and a flag someone has to remember is a way for the two to silently
# disagree about which measurement they describe. Truncated at the start of every
# run -- one file per run is what a fit reads, and appending across runs would mix
# build profiles and allocators under one header. One file for every mode: `mode`
# is a column, and what must not mix is the CONDITIONS, which the `# run:` heading
# holds and record.rs refuses to merge across.
#
#   testdata/calibration/records-hbm.tsv    (PCK_BENCH_HBM only)
#   testdata/calibration/capture-hbm.sqlite
# The HBM pass's pair, and deliberately not the files above. Its times are measured
# under GPU memory counters that cost the query ~7%, so it is read for TRAFFIC and
# never for microseconds; nsys_hbm.py joins the two into hbm.tsv, whose tuple then
# joins onto the clean records.tsv. That pass also writes no .benchmark.txt at all.

# pipefail so a failing cargo in stage_cargo_test_binary's pipeline reports as a build
# failure, not a missing binary. The remote scripts do not inherit it: see launch_remote.
set -euo pipefail

# Toolchain pinning, CARGO_TARGET_DIR, REMOTE/REMOTE_REPO, resilient_rsync,
# stage_cargo_test_binary.
. "$(dirname "${BASH_SOURCE[0]}")/lib/shadgpu-env.sh"

# Rust integration tests that link libpeacock_gpu.so and must run on the GPU host.
RUST_TESTS=(test_gpu_full_table test_gpu_partitioned test_inc2_conformance test_gpu_abi test_gpu_recipe_walk test_gpu_executors test_gpu_bp_corpus test_node_timing)
RUST_TESTS_STAGING=cpp/install/rust-tests

# The measurement target and its own staging dir. setup-glibc.sh patches both.
BENCH_TARGET=peacock_gpu_benchmarks
BENCH_STAGING=cpp/install/rust-benchmarks
# Relative to testdata/ on both sides, so one name drives the remote export and the pull.
BENCH_RECORD_REL=calibration/records.tsv
# The Nsight capture of a PCK_BENCH_NSYS run, and its sqlite export. Same convention.
# Only the export travels: the .nsys-rep is the profiler's own format and is tens of
# times the size, and nothing on this side reads it.
BENCH_CAPTURE_REL=calibration/capture
# What the trace capture is turned into, here rather than on the host: the reader is a
# local script reading local goldens, and the capture comes home anyway.
BENCH_CALLS_REL=calibration/calls.tsv
# Goldens the recipes cross-check reads. sf1 because that is where plan goldens live —
# a plan's recipes are its shape and do not depend on how much data it reads.
BENCH_PLANS_DIR=testdata/goldens/tpch.sf1
# The HBM pass writes BOTH of these, and neither may be the clean pass's. Its capture is
# a different nsys mode; its record holds the same rows measured ~7% slow, so it exists
# to be read for TRAFFIC and never for time.
BENCH_HBM_CAPTURE_REL=calibration/capture-hbm
BENCH_HBM_RECORD_REL=calibration/records-hbm.tsv
# opt-3: the default test profile leaves workspace crates at opt-level 1 and so measures
# a host overhead that is not the engine's. See `[profile.benchmarks]` in Cargo.toml.
BENCH_PROFILE=benchmarks

# Runner, log, exit code and run id of a detached run. Outside cpp/install/, which
# --push-binaries mirrors with --delete.
REMOTE_STATE=$REMOTE_REPO/.run-state

BUILD=0
BUILD_BENCH=0
RSYNC=0
PATCH=0
RUN=0
RUN_ATTACHED=0
RUN_DETACH=0
RUN_STATUS=0
RUN_BENCH=0
RUN_BENCH_ATTACHED=0
BENCH_DETACH=0
BENCH_STATUS=0
PULL_BENCH=0
# Both spellings of each run set RUN/RUN_BENCH; the *_ATTACHED flags exist so the
# contradiction between them is still visible after both have set it.

usage() {
  cat >&2 <<'USAGE'
Usage: build-test-shadgpu.sh [flags]

  --build                     C++ build+install, stage the correctness rust tests
  --build-benchmarks          C++ build+install, stage peacock_gpu_benchmarks
  --push-binaries             mirror cpp/install/ to the host + goldens + registry
  --patch                     glibc-patch the shipped binaries on the host
  --run                       the correctness gate
  --run-detached              setsid on the host; poll with --run-status
  --run-status                read-only: still going / finished / log tail
  --run-benchmarks            attached measurement run
  --run-benchmarks-detached   setsid on the host; poll with --benchmark-status
  --benchmark-status          read-only: still going / finished / log tail
  --pull-benchmarks           fetch testdata/benchmark-results/, the calibration record
                              and the sqlite export of a capture, if there is one

Knobs read from the environment, not flags:
  PCK_TEST_FILTER=<sub>       cargo-test name filter forwarded to the rust binaries
  PCK_BENCH_NSYS=1            --run-benchmarks captures under nsys (see the note at
                              PCK_BENCH_NSYS in this file)
  PCK_BENCH_HBM=1             a SECOND pass under GPU memory counters, its own capture
                              and its own record. PCK_BENCH_HBM_FILTER narrows it (it
                              falls back to PCK_TEST_FILTER); unset means every case
  PCK_BENCH_HBM_SET=<set>     nsys metric set for the device (default gh100)
  PCK_BENCH_HBM_FREQ=<Hz>     counter sampling rate (default 20000)
  --all                       = --build --push-binaries --patch --run

--all deliberately does NOT imply the benchmark phases: that is what keeps a
measurement out of the merge gate.

A status flag exits 0 only when the latest run of that phase finished with 0.
USAGE
  exit 1
}

[ $# -eq 0 ] && usage

while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=1 ;;
    --build-benchmarks) BUILD_BENCH=1 ;;
    --push-binaries) RSYNC=1 ;;
    --patch) PATCH=1 ;;
    --run) RUN=1; RUN_ATTACHED=1 ;;
    --run-detached) RUN=1; RUN_DETACH=1 ;;
    --run-status) RUN_STATUS=1 ;;
    --run-benchmarks) RUN_BENCH=1; RUN_BENCH_ATTACHED=1 ;;
    --run-benchmarks-detached) RUN_BENCH=1; BENCH_DETACH=1 ;;
    --benchmark-status) BENCH_STATUS=1 ;;
    --pull-benchmarks) PULL_BENCH=1 ;;
    --all) BUILD=1; RSYNC=1; PATCH=1; RUN=1; RUN_ATTACHED=1 ;;
    *) echo "Unknown flag: $1" >&2; usage ;;
  esac
  shift
done

# --- validation: every contradiction named, none resolved by argument order ---
# Before the first side effect: half a deploy followed by "you cannot do that" is worse
# than either outcome alone.
die() { echo "$*" >&2; exit 1; }

if [ "$RUN" -eq 1 ] && [ "$RUN_BENCH" -eq 1 ]; then
  die "a gate run with a benchmark run: one exit code cannot mean both 'correctness
     passed' and 'measurement completed'. Run them as two invocations."
fi
if [ "$RUN_ATTACHED" -eq 1 ] && [ "$RUN_DETACH" -eq 1 ]; then
  die "--run with --run-detached: pick who owns the process."
fi
if [ "$RUN_BENCH_ATTACHED" -eq 1 ] && [ "$BENCH_DETACH" -eq 1 ]; then
  die "--run-benchmarks with --run-benchmarks-detached: pick who owns the process."
fi
if [ "$PULL_BENCH" -eq 1 ] && [ "$BENCH_DETACH" -eq 1 ]; then
  # Reject rather than silently downgrade: the run has not finished, so a pull here
  # returns a partial tree that looks like a completed measurement.
  die "--pull-benchmarks with --run-benchmarks-detached: the run has not finished yet.
     Poll with --benchmark-status, then --pull-benchmarks."
fi
if [ -f /.dockerenv ] \
   && [ $((RSYNC + PATCH + RUN + RUN_STATUS + RUN_BENCH + BENCH_STATUS + PULL_BENCH)) -gt 0 ]; then
  die "only --build / --build-benchmarks work inside the builder container;
     the remaining phases need this workstation's ssh access to $REMOTE."
fi

# --- build --------------------------------------------------------------------
# The C++ half is not optional for either target: the staged binary resolves
# libpeacock_gpu.so from cpp/install/lib, and the per-node timing lives in that library.
# A fresh binary against a stale .so fails to link on the host, or — if the symbol
# happens to resolve — reports zeros for every node.
if [ "$BUILD" -eq 1 ] || [ "$BUILD_BENCH" -eq 1 ]; then
  ./scripts/build.sh --cudf_ROOT "$CUDF_ROOT" --gcc-version "$GCC_VERSION" --configure
  ./scripts/build.sh --cudf_ROOT "$CUDF_ROOT" --gcc-version "$GCC_VERSION" --build
  ./scripts/build.sh --cudf_ROOT "$CUDF_ROOT" --gcc-version "$GCC_VERSION" --install

  # peacockdb-ffi builds its own libpeacock_gpu.so through the `cmake` crate, which
  # caches the resolved cudf_DIR/Arrow in OUT_DIR. A cache from a different cuDF
  # root makes the link pick the wrong Arrow (`ld returned 1`), so clean the crate
  # to force a reconfigure — but only when the root actually changed, since a clean
  # rebuilds the whole cmake sub-tree (flatbuffers + gtest + libpeacock_gpu.so) and
  # dominates wall-clock otherwise. The stamp survives `cargo clean -p peacockdb-ffi`,
  # which removes only that crate's artifacts. PEACOCK_FFI_CLEAN=1 forces.
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
fi

if [ "$BUILD" -eq 1 ]; then
  # Stage from empty, and clear only rust-tests/ — --build-benchmarks owns the
  # sibling. The remote runner globs the directory rather than reading RUST_TESTS,
  # so a binary left here by an earlier build is shipped and executed even when its
  # target no longer exists: a renamed target runs on against goldens renamed out
  # from under it and fails as if the change were broken. rsync --delete cleans the
  # host, not this.
  rm -rf "$RUST_TESTS_STAGING"
  mkdir -p "$RUST_TESTS_STAGING"
  for t in "${RUST_TESTS[@]}"; do
    stage_cargo_test_binary "$t" "$RUST_TESTS_STAGING"
  done
fi

if [ "$BUILD_BENCH" -eq 1 ]; then
  # The first build under $BENCH_PROFILE is a cold compile of the whole DataFusion
  # stack plus a third libpeacock_gpu.so — peacockdb-ffi's OUT_DIR lives inside the
  # profile directory. One-time per profile; the correctness caches are untouched.
  rm -rf "$BENCH_STAGING"
  stage_cargo_test_binary "$BENCH_TARGET" "$BENCH_STAGING" --profile "$BENCH_PROFILE"
fi

# --- push ---------------------------------------------------------------------
if [ "$RSYNC" -eq 1 ]; then
  # Unstripped binaries are ~565MB each against ~155MB stripped, and the link to the
  # host is slow and bursty. --strip-debug keeps the dynamic symbol table patchelf
  # needs.
  for t in "${RUST_TESTS[@]}"; do
    [ -f "$RUST_TESTS_STAGING/$t" ] && strip --strip-debug "$RUST_TESTS_STAGING/$t"
  done
  [ -f "$BENCH_STAGING/$BENCH_TARGET" ] && strip --strip-debug "$BENCH_STAGING/$BENCH_TARGET"

  # Source is cpp/install/ and NOT cpp/install/* : with a glob rsync gets several
  # sources and --delete stops meaning what it looks like. It removes host orphans,
  # which for a directory the remote runner globs is the difference between a stale
  # binary sitting there and a stale binary executing. It cuts the other way too: the
  # mirror covers rust-benchmarks/, so a gate push from a checkout that never ran
  # --build-benchmarks deletes the benchmark binary off the host.
  #
  # -a, not -r. cpp/install/lib is a vendored tree full of soname chains
  # (libglog.so.2 -> libglog.so.0.7.1, the left name being the DT_SONAME the linker
  # asks for), and `rsync -r` skips symlinks — the host would get the target and not
  # the name, and every binary would die at start on "loading shared libraries".
  resilient_rsync -a --delete cpp/install/ "$REMOTE:$REMOTE_REPO/cpp/install/"
  # The goldens the rust GPU tests assert against. Without this the host keeps
  # whatever a previous run left, so a locally-regenerated golden is compared
  # against a stale one and goes false-red. testdata/benchmark-results/ is
  # deliberately not mirrored: it is written on the host and travels back through
  # --pull-benchmarks, and a --delete push from a box that never ran the benchmarks
  # would erase the measurement history.
  ssh "$REMOTE" "mkdir -p $REMOTE_REPO/testdata/goldens"
  resilient_rsync -r --delete testdata/goldens/ "$REMOTE:$REMOTE_REPO/testdata/goldens/"
  # Everything else under testdata/ that the binaries READ rather than assert against:
  # cost-registry.csv, cost_model.conf, the query .sql sets, tpch.minimal. Swept from
  # git rather than named, because the hand-maintained list this replaces had gone
  # stale five times. The fifth was the expensive one: cost_model.conf was added to
  # build-test.sh's sweep and to nothing else, so the host kept the pre-taxonomy-split
  # file and a calibration run came home tagged with categories that no longer
  # exist -- wrong data rather than a red test.
  #
  # Additive, unlike the goldens push above: --delete here would erase whatever else
  # provisions this host, and nothing under testdata/ that we do not track is ours to
  # remove. Generated datasets are not swept -- they are git-ignored, which is what
  # --exclude-standard turns on, and they live on the host at tens of GiB.
  # Two tracked trees are excluded. goldens/ is owned by the --delete mirror above,
  # and re-adding it here additively would be the mirror's opposite. benchmark-results/
  # is WRITTEN on the host and travels back through --pull-benchmarks, so pushing a
  # checkout's copy over it replaces the host's measurements with whatever this box
  # last pulled.
  fixtures=$(mktemp)
  git ls-files --cached --others --exclude-standard testdata \
    | grep -vE '^testdata/(goldens|benchmark-results)/' > "$fixtures"
  [ -s "$fixtures" ] || die "no tracked testdata fixtures found -- the git sweep is wrong"
  echo "==> push $(wc -l < "$fixtures") committed testdata fixtures -> $REMOTE"
  resilient_rsync -a --files-from="$fixtures" ./ "$REMOTE:$REMOTE_REPO/"
  rm -f "$fixtures"
  # Our setup-glibc.sh, so --patch uses the version that knows both rust dirs.
  ssh "$REMOTE" "mkdir -p $REMOTE_REPO/scripts"
  resilient_rsync -a scripts/setup-glibc.sh "$REMOTE:$REMOTE_REPO/scripts/"
fi

if [ "$PATCH" -eq 1 ]; then
  ssh "$REMOTE" "$REMOTE_REPO/scripts/setup-glibc.sh --repo-dir $REMOTE_REPO --patch"
fi

# --- the shared launcher ------------------------------------------------------
# One launcher for both phases: the gate and the measurement differ in what the
# remote script does, never in how it is started.
#
# The script is installed on the host and executed from there, so attached and
# detached run byte-identical remote code; only the process owner differs. Detached
# hands it to setsid — a session with no controlling terminal, so the SIGHUP after a
# dropped ssh never reaches it — with stdin from /dev/null, which the process would
# otherwise block or die on.
#
# Each launch writes a fresh run id and removes the previous exit code. The runner
# writes "<id> <rc>" last, and a status call reports a result only when that id is
# the latest launch's — otherwise an older run's completion reads as this one's.
remote_state_paths() {
  phase_runner=$REMOTE_STATE/$1.sh
  phase_log=$REMOTE_STATE/$1.log
  phase_rc=$REMOTE_STATE/$1.rc
  phase_id=$REMOTE_STATE/$1.id
}

# launch_remote <phase> <detach>, remote script on stdin.
# Attached: returns the run's exit code. Detached: returns 0 only once the run is
# confirmed alive, or the run's code if it has already finished.
launch_remote() {
  local phase=$1 detach=$2 run_id
  remote_state_paths "$phase"
  run_id="$(date +%Y%m%dT%H%M%S)-$$"

  ssh "$REMOTE" "mkdir -p $REMOTE_STATE && cat > $phase_runner && chmod +x $phase_runner"

  if [ "$detach" -eq 0 ]; then
    ssh "$REMOTE" bash <<EOF
      printf '%s\n' '$run_id' > $phase_id
      rm -f $phase_rc
      bash $phase_runner 2>&1 | tee $phase_log
      status=\${PIPESTATUS[0]}
      printf '%s %s\n' '$run_id' "\$status" > $phase_rc
      exit "\$status"
EOF
    return
  fi

  # pgrep matches the wrapper rather than the binary: it is alive for the whole run,
  # including between binaries, and it is what the runner's own path identifies.
  # The pattern reaches the host through a heredoc, so pgrep cannot match the shell
  # carrying it.
  ssh "$REMOTE" bash <<EOF
    printf '%s\n' '$run_id' > $phase_id
    rm -f $phase_rc
    setsid nohup bash -c 'bash $phase_runner > $phase_log 2>&1; printf "%s %s\n" "$run_id" "\$?" > $phase_rc' \
      < /dev/null > /dev/null 2>&1 &
    sleep 3
    if [ -f $phase_rc ]; then
      rc=\$(cut -d' ' -f2 $phase_rc)
      echo "==> $phase run finished within 3s, exit code \$rc"
      tail -20 $phase_log
      exit "\$rc"
    fi
    if pgrep -f $phase_runner > /dev/null; then
      echo "==> detached $phase run going on $REMOTE (pid \$(pgrep -f $phase_runner | head -1))"
      exit 0
    fi
    echo "!!! detached $phase run is neither alive nor finished — it never started"
    tail -20 $phase_log 2>/dev/null || echo "(no log)"
    exit 1
EOF
}

# report_status <phase>. Read-only, safe to call as often as you like.
# Exits 0 only when the latest launch finished with 0. Running, died, and an exit
# code belonging to an earlier run are all non-zero: the detached form exists to
# carry the run's code across the ssh session, and a status that always returns 0
# drops it on arrival.
report_status() {
  local phase=$1 extra=
  remote_state_paths "$phase"
  [ "$phase" = benchmark ] && extra="echo \"==> records on host: \$(find $REMOTE_REPO/testdata/benchmark-results -name '*.benchmark.txt' 2>/dev/null | wc -l)\""

  ssh "$REMOTE" bash <<EOF
    id=\$(cat $phase_id 2>/dev/null || true)
    $extra
    if [ -z "\$id" ]; then
      echo "!!! no $phase run has been launched on $REMOTE"
      exit 1
    fi
    echo "--- tail of $phase_log"
    tail -15 $phase_log 2>/dev/null || echo "(no log yet)"
    if [ -f $phase_rc ]; then
      rc_id=\$(cut -d' ' -f1 $phase_rc)
      rc=\$(cut -d' ' -f2 $phase_rc)
      if [ "\$rc_id" = "\$id" ]; then
        echo "==> run \$id FINISHED, exit code \$rc"
        exit "\$rc"
      fi
      echo "!!! the newest exit code is run \$rc_id's, not the current run \$id's"
    fi
    if pgrep -f $phase_runner > /dev/null; then
      echo "==> run \$id STILL GOING (pid \$(pgrep -f $phase_runner | head -1))"
      exit 1
    fi
    echo "!!! run \$id left no exit code and has no process — it died (host reboot,"
    echo "    OOM-killer, manual kill). Anything it wrote before that is intact."
    exit 1
EOF
}

# Filters are human-typed and reach the remote script as a single-quoted literal,
# so quote them for the shell rather than assuming they contain no apostrophe.
: "${PCK_TEST_FILTER:=}"
filter_q=$(printf '%q' "$PCK_TEST_FILTER")

# Capture knob for --run-benchmarks. Off unless set, and deliberately not a flag: a
# capture is a different measurement, not a variant of the run — nsys serializes what it
# traces, so the times in the .benchmark.txt of a captured run are not comparable with
# any other. Something a caller has to type is the right shape for that.
: "${PCK_BENCH_NSYS:=}"
: "${PCK_BENCH_HBM:=}"
# Which case the HBM pass runs. Falls back to the run's own filter, so the common form
# is one variable: PCK_TEST_FILTER=<case> PCK_BENCH_HBM=1.
: "${PCK_BENCH_HBM_FILTER:=${PCK_TEST_FILTER:-}}"
# Device index, metric set and sampling rate for the counters. The set is a property of
# the ARCHITECTURE — nsys numbers its metrics differently per set, which is why
# nsys_hbm.py looks its ids up by name rather than assuming them.
: "${PCK_BENCH_HBM_DEVICE:=0}"
: "${PCK_BENCH_HBM_SET:=gh100}"
: "${PCK_BENCH_HBM_FREQ:=20000}"
# After the default above, which reads PCK_TEST_FILTER.
hbm_filter_q=$(printf '%q' "$PCK_BENCH_HBM_FILTER")

# --- run: the correctness gate ------------------------------------------------
# Knobs, set in the caller's env rather than as flags:
#   PEACOCK_GPU_DEBUG=1    PCK_TRACE + a per-node cudaStreamSynchronize in
#                          src/expr.cpp, which localizes async errors
#   PCK_TEST_FILTER=<sub>  cargo-test name filter forwarded to the rust binaries
#   PCK_RUN_CPP=0          skip the C++ suites (default: run them)
#
# The heredoc marker is unquoted, so $VARS expand locally before the text is sent;
# escape with \$ anything the remote shell should expand.
remote_gate_script() {
  : "${PEACOCK_GPU_DEBUG:=}"
  : "${PCK_RUN_CPP:=1}"
  cat <<EOF
    # Superset env, mirroring CI: every binary gets every variable it might need and
    # ignores the rest. Without PEACOCK_TPCH_{SF40_DIR,GOLDEN_DIR,VEC_PARAMS} the
    # sf40 suites fall back to a relative golden path and fail as a mis-provisioned
    # run. That dataset lives outside the repo and is read in place.
    export PEACOCK_TESTDATA_DIR=$REMOTE_REPO/testdata
    export PEACOCK_TPCH_SF40_DIR=/home/info/peacock-datasets/testdata/tpch.sf40
    export PEACOCK_TPCH_GOLDEN_DIR=$REMOTE_REPO/testdata/goldens/tpch.sf40
    export PEACOCK_TPCH_VEC_PARAMS=$REMOTE_REPO/testdata/tpch-vec-queries/query_params.jsonl
    export PEACOCK_GPU_DEBUG='$PEACOCK_GPU_DEBUG'
    # cpp/install/lib first, so libpeacock_gpu.so resolves for the rust binaries:
    # their baked-in rpath points at the build host's cargo target. The benchmark
    # runner deliberately does not export its equivalent — see the reason there.
    export LD_LIBRARY_PATH=$REMOTE_REPO/cpp/install/lib:/usr/local/cuda-12.5/compat:/home/info/glibc-2.35/lib:\$HOME/miniforge3/envs/rapids-cuda-12.2/lib:\$LD_LIBRARY_PATH

    # Deliberately no \`set -e\`, matching CI: run every binary even after one fails and
    # OR the codes into rc. Under set -e a SIGSEGV in one GPU binary cost us every
    # later result, which read as "not run" but looked like "fine".
    rc=0

    # The staging separation, as something that can go red: a measurement binary run
    # as a gate asserts nothing, exits 0, and reads green having verified nothing.
    if ls $REMOTE_REPO/cpp/install/rust-tests/*benchmark* >/dev/null 2>&1; then
      echo "!!! a benchmark binary is staged in rust-tests/ — it would be run as a gate"
      rc=1
    fi

    # Glob peacock_*_tests, matching CI: a hardcoded name meant three of the four
    # binaries never ran locally, so a "C++ green" sign-off covered one of them. The
    # two guards are CI's as well — a suite that skips everything exits 0, and a glob
    # that matches nothing makes every C++ test vanish silently.
    if [ '$PCK_RUN_CPP' = '1' ]; then
      ran_any=0
      for t in $REMOTE_REPO/cpp/install/bin/peacock_*_tests; do
        [ -x "\$t" ] || continue
        tname=\${t##*/}
        # The multi-GPU suites are EXCLUDE_FROM_ALL and need two visible GPUs. If
        # someone builds them locally they land in install/bin, where this glob would
        # sweep them into the gate and they would fail for want of a second GPU.
        case "\$tname" in peacock_multi_gpu_*) echo "==> \$tname (skipped: multi-GPU is manual-only)"; continue ;; esac
        echo "==> \$tname (C++)"
        tlog=/tmp/\$tname.log
        "\$t" > "\$tlog" 2>&1
        trc=\$?
        [ "\$trc" -eq 0 ] || { echo "!!! \$tname FAILED (exit \$trc)"; rc=1; }
        tzero=0
        while IFS= read -r line; do
          printf '%s\n' "\$line"
          case "\$line" in *"[  PASSED  ] 0 tests"*) tzero=1 ;; esac
        done < "\$tlog"
        if [ "\$tzero" -eq 1 ]; then
          echo "!!! \$tname ran 0 tests (all skipped) — nothing was verified"
          rc=1
        fi
        ran_any=\$((ran_any + 1))
      done
      if [ "\$ran_any" -eq 0 ]; then
        echo "!!! no peacock_*_tests binaries found — every C++ test vanished"
        rc=1
      fi
      echo "==> ran \$ran_any C++ test binaries"
    fi

    echo "==> rust GPU integration tests (filter=$filter_q)"
    rust_ran=0
    for t in $REMOTE_REPO/cpp/install/rust-tests/*; do
      [ -x "\$t" ] || continue
      tname=\${t##*/}
      echo "--- \$tname"
      rlog=/tmp/\$tname.rustlog
      # --test-threads=1: the GPU/RMM context is process-wide, parallel tests OOM.
      "\$t" --nocapture --test-threads=1 $filter_q > "\$rlog" 2>&1
      status=\$?
      # Zero tests is a fault only when nothing was filtered out: with a filter set,
      # every other binary legitimately matches nothing, and a red banner for a run
      # that did exactly what was asked is how people learn to ignore the banner.
      rzero=0
      while IFS= read -r line; do
        printf '%s\n' "\$line"
        case "\$line" in *"test result:"*" 0 passed"*) rzero=1 ;; esac
      done < "\$rlog"
      if [ "\$status" -ne 0 ]; then
        # 139 is SIGSEGV; a bare non-zero code here has already been mistaken for an
        # assertion failure.
        echo "!!! \$tname FAILED (exit \$status)"
        rc=1
      elif [ "\$rzero" -eq 1 ] && [ -z $filter_q ]; then
        echo "!!! \$tname ran 0 tests (filter $filter_q matched nothing?) — nothing was verified"
        rc=1
      fi
      rust_ran=\$((rust_ran + 1))
    done
    if [ "\$rust_ran" -eq 0 ]; then
      echo "!!! no rust test binaries found in cpp/install/rust-tests — every rust GPU test vanished"
      rc=1
    fi
    echo "==> ran \$rust_ran rust test binaries"

    if [ "\$rc" -ne 0 ]; then
      echo "==> GPU test run FAILED (see '!!!' lines above)"
    else
      echo "==> GPU test run OK"
    fi
    exit "\$rc"
EOF
}

# --- run: the benchmark measurement -------------------------------------------
# PEACOCK_GPU_DEBUG is deliberately not forwarded here, unlike in the gate: it adds
# a cudaStreamSynchronize after every operator, which changes exactly the thing
# being measured, and the numbers would not be comparable with any other run.
remote_bench_script() {
  cat <<EOF
    export PEACOCK_TESTDATA_DIR=$REMOTE_REPO/testdata
    export PEACOCK_TPCH_SF40_DIR=/home/info/peacock-datasets/testdata/tpch.sf40
    export PEACOCK_TPCH_VEC_PARAMS=$REMOTE_REPO/testdata/tpch-vec-queries/query_params.jsonl

    # The sf40 dataset is read in place, outside the repo, and the C++ suites reach it
    # through the variable above. The rust side has no such variable: it resolves data
    # as <testdata>/<dataset>.sf<sf> and nothing else, so a benchmark case at sf40 needs
    # that name to exist. A symlink is what makes one convention cover both -- against
    # adding a second way to name a dataset path, which is how a run ends up reading one
    # dataset while reporting another. testdata/.gitignore already hides /tpch.sf*/.
    #
    # Only here, not in the gate: the gate's sf40 work is the C++ binaries, which use the
    # variable. Never replaces what it finds -- a real directory under that name is
    # someone else's provisioning of this shared host, and the run stops instead.
    sf40_link=\$PEACOCK_TESTDATA_DIR/tpch.sf40
    if [ -L "\$sf40_link" ]; then
      have=\$(readlink "\$sf40_link")
      if [ "\$have" != "\$PEACOCK_TPCH_SF40_DIR" ]; then
        echo "!!! \$sf40_link -> \$have, expected \$PEACOCK_TPCH_SF40_DIR"
        exit 1
      fi
    elif [ -e "\$sf40_link" ]; then
      echo "!!! \$sf40_link exists and is not a symlink -- not touching it"
      exit 1
    elif [ ! -d "\$PEACOCK_TPCH_SF40_DIR" ]; then
      echo "!!! no sf40 dataset at \$PEACOCK_TPCH_SF40_DIR"
      exit 1
    else
      ln -s "\$PEACOCK_TPCH_SF40_DIR" "\$sf40_link"
      echo "==> linked \$sf40_link -> \$PEACOCK_TPCH_SF40_DIR"
    fi

    # Applied per-command on the benchmark binary alone rather than exported: this
    # path carries glibc-2.35, and exporting it makes the host's own coreutils load
    # the newer libc under the old loader and SIGSEGV — the mkdir/find/wc below would
    # die and the run would report a bogus exit code having actually succeeded.
    # (setup-glibc.sh warns about this at the end of --patch.)
    bench_ld=$REMOTE_REPO/cpp/install/lib:/usr/local/cuda-12.5/compat:/home/info/glibc-2.35/lib:\$HOME/miniforge3/envs/rapids-cuda-12.2/lib

    bin=$REMOTE_REPO/$BENCH_STAGING/$BENCH_TARGET
    if [ ! -x "\$bin" ]; then
      echo "!!! benchmark binary not found at \$bin"
      echo "    Build it with --build-benchmarks and ship it with --push-binaries."
      echo "    (A --push-binaries from a checkout that never built benchmarks mirrors"
      echo "     it away again — see the --delete note in build-test-shadgpu.sh.)"
      exit 1
    fi

    results=\$PEACOCK_TESTDATA_DIR/benchmark-results
    mkdir -p "\$results"

    # Calibration rows alongside the tree. Removed rather than appended to:
    # record.rs writes the header only into a fresh file, so a leftover from an
    # earlier run would swallow this one's rows under the earlier one's heading.
    export PEACOCK_RECORD_PATH=\$PEACOCK_TESTDATA_DIR/$BENCH_RECORD_REL
    mkdir -p "\$(dirname "\$PEACOCK_RECORD_PATH")"
    rm -f "\$PEACOCK_RECORD_PATH"
    # What this run wrote, not what is on the host: the tree accumulates across runs,
    # so a total can only go red on a first-ever run and a filter that matches nothing
    # would read green having measured nothing. mktemp gives the comparison point.
    stamp=\$(mktemp)

    # PCK_BENCH_NSYS: trace the run instead of just measuring it. PEACOCK_NVTX turns on
    # the node/partition ranges the capture is joined on -- without it the trace has
    # libcudf's calls and no way to say which node they belong to.
    #
    # --trace=nvtx,cuda and nothing else: no --sample (the CPU profiler's SIGPROF
    # interrupts the very host spans the ranges measure) and no --gpu-metrics-device
    # (that is the hbm measurement, it needs a frequency chosen against the device and
    # it doubles the file for a column this join does not read).
    #
    # Exported to sqlite here, on the machine that wrote it: nsys export needs the same
    # nsys that captured, and only the export is small enough to want on the wire.
    # Filters as variables of the REMOTE shell, assigned from text this heredoc expands.
    # `printf %q ""` is two quote CHARACTERS, and the difference between them being shell
    # syntax and being data decides everything: assigned here, the remote shell reads
    # them and the variable is empty; passed as a string through a function argument they
    # survive as an argument `''`, which libtest matches against every test name and
    # filters all nine out. That is exactly how the main pass of the first end-to-end run
    # measured nothing and said "0 passed" where a green run belonged.
    main_filter=$filter_q
    hbm_filter=$hbm_filter_q

    # One invocation of the binary, with nsys wrapped around it or not.
    #
    # Output goes to the terminal AND to \$blog, which the checks below read: libtest's
    # own "test result: N passed" is the only honest answer to "did the filter match
    # anything", and counting files that appeared is not the same question.
    #
    # \`env\` between nsys and the binary, not LD_LIBRARY_PATH in front of nsys: the path
    # carries glibc-2.35, and nsys is a host binary that would load it under the host's
    # own loader and die — the same trap the bench_ld comment above describes. env
    # inherits an untouched environment and sets the variable for its child alone.
    #
    # --test-threads=1 is not optional: cuDF/RMM share one process-wide pool and one
    # default stream, so concurrent cases would measure each other's contention.
    bench_run() {
      local label=\$1 filter=\$2; shift 2
      blog=/tmp/$BENCH_TARGET.\$label.log
      if [ \$# -eq 0 ]; then
        LD_LIBRARY_PATH="\$bench_ld:\${LD_LIBRARY_PATH:-}" \\
          "\$bin" --nocapture --test-threads=1 \$filter 2>&1 | tee "\$blog"
      else
        "\$@" env LD_LIBRARY_PATH="\$bench_ld:\${LD_LIBRARY_PATH:-}" \\
          "\$bin" --nocapture --test-threads=1 \$filter 2>&1 | tee "\$blog"
      fi
      return \${PIPESTATUS[0]}
    }

    # Red on "the filter matched nothing", NOT on "no .benchmark.txt appeared". The
    # binary carries non-device tests of its own -- the record's switch, the section
    # merge -- and a filter naming one of them runs, passes, and writes no tree file.
    # Failing that is a red banner for a run that did exactly what was asked, which is
    # how people learn to ignore the banner (the gate's \`rzero\` says the same).
    #
    # Per pass and immediately after it, not once at the end: two passes share nothing
    # but the binary, and a check reading whichever log was written last would have let
    # a main pass that ran nothing through on the strength of the HBM pass's one test.
    # It did, once.
    ran_check() {
      local label=\$1 filter=\$2
      local n
      n=\$(sed -n 's/^test result:.* \([0-9][0-9]*\) passed.*/\1/p' \\
            "/tmp/$BENCH_TARGET.\$label.log" | awk '{n += \$1} END {print n + 0}')
      if [ "\$n" -eq 0 ]; then
        echo "!!! the \$label pass ran no tests (filter '\$filter' matched nothing?)"
        exit 1
      fi
      echo "==> the \$label pass ran \$n tests"
    }

    # Exported even after a failed run: a capture of the executions that did happen is
    # still the only copy of them, and re-running to get one costs the whole run again.
    # Exported here, on the machine that wrote it — nsys export needs the same nsys that
    # captured, and only the export is small enough to want on the wire.
    export_capture() {
      nsys export --type=sqlite --force-overwrite=true -o "\$1.sqlite" "\$1.nsys-rep" || true
      ls -l "\$1.nsys-rep" "\$1.sqlite" 2>/dev/null || true
    }

    fresh_capture() {
      mkdir -p "\$(dirname "\$1")"
      rm -f "\$1.nsys-rep" "\$1.sqlite"
    }

    capture=""
    if [ -n "$PCK_BENCH_NSYS" ]; then
      capture=\$PEACOCK_TESTDATA_DIR/$BENCH_CAPTURE_REL
      fresh_capture "\$capture"
      export PEACOCK_NVTX=1
      echo "==> capturing to \$capture.nsys-rep"
    fi

    echo "==> $BENCH_TARGET (filter=$filter_q)"
    if [ -n "\$capture" ]; then
      # --trace=nvtx,cuda and nothing else: no --sample (the CPU profiler's SIGPROF
      # interrupts the very host spans the ranges measure) and no --gpu-metrics-device
      # — that is the SECOND pass below, and combining them would change what is being
      # measured twice over.
      bench_run main "\$main_filter" \\
        nsys profile --trace=nvtx,cuda --sample=none --cpuctxsw=none \\
                     --force-overwrite=true -o "\$capture"
      status=\$?
      export_capture "\$capture"
    else
      bench_run main "\$main_filter"
      status=\$?
    fi
    if [ "\$status" -ne 0 ]; then
      echo "!!! $BENCH_TARGET FAILED (exit \$status)"
      exit "\$status"
    fi
    ran_check main "\$main_filter"
    written=\$(find "\$results" -name '*.benchmark.txt' -newer "\$stamp" | wc -l)
    total=\$(find "\$results" -name '*.benchmark.txt' | wc -l)
    rm -f "\$stamp"
    echo "==> benchmark records written by this run: \$written (on host: \$total)"
    echo "==> calibration rows: \$(grep -vc '^#' "\$PEACOCK_RECORD_PATH" 2>/dev/null || echo 0)"
    # Not a failure: see ran_check. The main pass having run tests is already established.
    if [ "\$written" -eq 0 ]; then
      echo "==> no .benchmark.txt written: none of the tests that ran times a case"
    fi

    # ── the HBM pass ───────────────────────────────────────────────────────────
    # A SECOND run of the same cases under GPU memory counters. Separate rather than one
    # capture with more flags, and it is not a preference:
    #
    #   - the counters cost what they measure. A capture with them on runs the query ~7%
    #     slow and a heavy scan ~11%, so its times are not the times, and its record is
    #     written to its OWN file for that reason alone. Nothing downstream should ever
    #     read a microsecond out of it.
    #   - the traffic is what it carries, and traffic does not care that the run was
    #     slower. It joins onto the clean run's rows by the tuple, which is the whole
    #     reason the record carries a tuple rather than a node number.
    #
    # ANY NUMBER OF CASES. The harness wraps each in a named NVTX range, so the capture
    # says which query a call was in and \`nsys_hbm.py\` reads it rather than being told.
    # PCK_BENCH_HBM_FILTER therefore narrows the pass for TIME — a capture under memory
    # counters is minutes and gigabytes at sf40 — and no longer for correctness.
    if [ -n "$PCK_BENCH_HBM" ]; then
      hbm_capture=\$PEACOCK_TESTDATA_DIR/$BENCH_HBM_CAPTURE_REL
      fresh_capture "\$hbm_capture"
      export PEACOCK_NVTX=1
      export PEACOCK_RECORD_PATH=\$PEACOCK_TESTDATA_DIR/$BENCH_HBM_RECORD_REL
      rm -f "\$PEACOCK_RECORD_PATH"
      # The tree file is the clean pass's. This pass would overwrite each section with
      # times taken under the counters — the one number in it that is knowingly wrong.
      export PEACOCK_BENCHMARK_RESULTS_RO=1
      echo "==> HBM pass (filter=\$hbm_filter) to \$hbm_capture.nsys-rep"
      # --gpu-metrics-frequency is part of the measurement, not a display detail: too
      # high and the device cannot sustain the sampling, which nsys_hbm.py refuses by
      # the >100%-of-peak check rather than reporting quietly wrong bytes.
      bench_run hbm "\$hbm_filter" \\
        nsys profile --trace=nvtx,cuda --sample=none --cpuctxsw=none \\
                     --gpu-metrics-device=$PCK_BENCH_HBM_DEVICE \\
                     --gpu-metrics-set=$PCK_BENCH_HBM_SET \\
                     --gpu-metrics-frequency=$PCK_BENCH_HBM_FREQ \\
                     --force-overwrite=true -o "\$hbm_capture"
      hbm_status=\$?
      export_capture "\$hbm_capture"
      echo "==> HBM calibration rows: \$(grep -vc '^#' "\$PEACOCK_RECORD_PATH" 2>/dev/null || echo 0)"
      if [ "\$hbm_status" -ne 0 ]; then
        echo "!!! $BENCH_TARGET FAILED under GPU metrics (exit \$hbm_status)"
        exit "\$hbm_status"
      fi
      ran_check hbm "\$hbm_filter"
    fi

EOF
}

run_rc=0
if [ "$RUN" -eq 1 ]; then
  remote_gate_script | launch_remote gate "$RUN_DETACH" || run_rc=$?
fi
if [ "$RUN_BENCH" -eq 1 ]; then
  # `|| run_rc=$?` rather than letting set -e abort: --pull-benchmarks must still
  # run, or every record written before the failure is left on the host.
  remote_bench_script | launch_remote benchmark "$BENCH_DETACH" || run_rc=$?
fi
if [ "$run_rc" -ne 0 ]; then
  if [ "$RUN_BENCH" -eq 1 ]; then
    echo "==> benchmark run FAILED (rc=$run_rc). Records written before the failure are" >&2
    echo "    still on the host; recover them with: $0 --pull-benchmarks" >&2
  fi
  if [ "$PULL_BENCH" -ne 1 ]; then exit "$run_rc"; fi
  # With a pull requested, pull first and fail afterwards.
  trap 'exit '"$run_rc" EXIT
fi

status_rc=0
if [ "$RUN_STATUS" -eq 1 ]; then
  report_status gate || status_rc=$?
fi
if [ "$BENCH_STATUS" -eq 1 ]; then
  report_status benchmark || status_rc=$?
fi

if [ "$PULL_BENCH" -eq 1 ]; then
  # The detached workflow is two invocations, and this is the second one: pulling
  # mid-run brings home a partial tree that looks like a finished measurement. Three
  # states, and only the first is a refusal — a run that died left its records intact,
  # and collecting them is the documented recovery, so that one says so and pulls.
  remote_state_paths benchmark
  pull_state=$(ssh "$REMOTE" bash <<EOF
    id=\$(cat $phase_id 2>/dev/null || true)
    if [ -z "\$id" ] || grep -q "^\$id " $phase_rc 2>/dev/null; then
      echo settled
    elif pgrep -f $phase_runner > /dev/null; then
      echo running
    else
      echo died
    fi
EOF
  )
  case "$pull_state" in
    running)
      die "a benchmark run is still going on $REMOTE; --pull-benchmarks now would bring
     home a partial tree. Poll with --benchmark-status." ;;
    died)
      echo "!!! the last benchmark run on $REMOTE left no exit code — it died partway." >&2
      echo "    Pulling anyway: what it wrote before that is intact, but the tree is a" >&2
      echo "    partial run's output, not a completed measurement." >&2 ;;
  esac
  mkdir -p testdata/benchmark-results
  # No --delete, unlike every push: a filtered run rewrites only the cases it ran,
  # and mirroring would wipe every record of the others. Nothing prunes the host
  # tree either, so a renamed case's record lives there until someone removes it and
  # rides home on every later pull.
  resilient_rsync -r "$REMOTE:$REMOTE_REPO/testdata/benchmark-results/" testdata/benchmark-results/
  echo "==> fetched $(find testdata/benchmark-results -name '*.benchmark.txt' | wc -l) benchmark records"
  # The four files a run can leave beside the tree: two records and two sqlite exports,
  # one pair per pass. Missing is not an error for any of them -- --pull-benchmarks is
  # also the recovery path for a run from before they existed, for one that died before
  # its first record, and for the ordinary case of a run with no capture at all.
  #
  # Tested over ssh rather than by letting the transfer fail: resilient_rsync retries a
  # missing source a hundred times before giving up, and eight minutes of backoff is not
  # how "there is no record" should read.
  #
  # Not the .nsys-rep beside each export -- see BENCH_CAPTURE_REL. Left on the host until
  # the next capture overwrites it, so a pull following an ordinary run brings home the
  # PREVIOUS capture; the file is dated by its mtime and nothing joins it to a record
  # automatically.
  pull_one() {                    # pull_one <relative path> <what it is>
    local rel=$1 what=$2
    if ! ssh "$REMOTE" test -f "$REMOTE_REPO/testdata/$rel"; then
      echo "==> $what: nothing on the host"
      return 0
    fi
    mkdir -p "testdata/$(dirname "$rel")"
    resilient_rsync "$REMOTE:$REMOTE_REPO/testdata/$rel" "testdata/$rel"
    case "$rel" in
      *.tsv) echo "==> $what: $(grep -vc '^#' "testdata/$rel") rows" ;;
      *)     echo "==> $what: $(du -h "testdata/$rel" | cut -f1)" ;;
    esac
  }
  pull_one "$BENCH_RECORD_REL"            "the calibration record"
  pull_one "$BENCH_CAPTURE_REL.sqlite"    "the trace capture"
  pull_one "$BENCH_HBM_RECORD_REL"        "the HBM pass's record"
  pull_one "$BENCH_HBM_CAPTURE_REL.sqlite" "the HBM capture"

  # The trace capture read down to what a call splits into inside libcudf, which is a
  # level `records.tsv` cannot hold: one ABI call is several libcudf calls, and a hash
  # join's build, probe and gather cost differently — a coefficient fitted against their
  # sum is wrong for all three.
  #
  # Derived here rather than on the host and rerun on every pull: it reads a capture that
  # is already home and goldens that are already local, costs seconds, and a derived file
  # older than the capture beside it is exactly the kind of stale that goes unnoticed.
  if [ -f "testdata/$BENCH_CAPTURE_REL.sqlite" ]; then
    if python3 scripts/calibration/nsys_calls.py \
         --capture "testdata/$BENCH_CAPTURE_REL.sqlite" \
         --plans-dir "$BENCH_PLANS_DIR" \
         --out "testdata/$BENCH_CALLS_REL"; then
      echo "==> the call breakdown: $(grep -vc '^#' "testdata/$BENCH_CALLS_REL") rows"
    else
      # Not fatal: the records and the tree are the measurement, and this is a reading of
      # it. A capture from before the harness pushed case ranges refuses here and says so.
      echo "!!! the call breakdown was not produced (see above); the rest of the pull stands"
    fi
  fi

  # What came home that nothing here writes any more.
  #
  # The host tree accumulates and the pull has no --delete (see above), so a file whose
  # naming scheme is gone rides home on EVERY later pull. That is how 128 legacy
  # `<query>.<label>.benchmark.txt` files, deleted from the working tree, came back
  # without a word and stayed for weeks: the deletion was never staged, and the next
  # pull recreated them byte for byte.
  #
  # Told apart by shape, which is exact for this tree: a current file is
  # `<mode>.benchmark.txt` — one dot — and every legacy one carries the query in the name
  # too. Reported rather than deleted: what to keep is not this script's call, and a pull
  # that quietly removed measurements would be the same silence from the other side.
  stale=$(find testdata/benchmark-results -name '*.*.benchmark.txt' | wc -l)
  if [ "$stale" -gt 0 ]; then
    echo "==> $stale file(s) here match no mode this build writes — a naming scheme that"
    echo "    is gone. They live on the host and come home on every pull; deleting them"
    echo "    locally does not stick until the host's copies go too:"
    find testdata/benchmark-results -name '*.*.benchmark.txt' | head -3 | sed 's/^/      /'
    echo "      ssh $REMOTE \"find $REMOTE_REPO/testdata/benchmark-results -name '*.*.benchmark.txt' -delete\""
  fi

  # What is here now, not what this pull moved. The two differ whenever a filter ran, and
  # the question a caller has after one invocation is "do I have what the plots need" —
  # which is about the tree, not about the transfer.
  echo "==> what is here now:"
  for f in $(find testdata/benchmark-results -name '*.benchmark.txt' | sort); do
    echo "      $f ($(grep -c '^== ' "$f") queries)"
  done
  for rel in "$BENCH_RECORD_REL" "$BENCH_HBM_RECORD_REL" "$BENCH_CALLS_REL"; do
    [ -f "testdata/$rel" ] && echo "      testdata/$rel ($(($(grep -vc '^#' "testdata/$rel") - 1)) rows)"
  done
  for rel in "$BENCH_CAPTURE_REL.sqlite" "$BENCH_HBM_CAPTURE_REL.sqlite"; do
    [ -f "testdata/$rel" ] && echo "      testdata/$rel ($(du -h "testdata/$rel" | cut -f1))"
  done
fi

exit "$status_rc"
