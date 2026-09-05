//! Benchmark run over the GPU suite: every query the correctness gate verifies, timed.
//!
//! Same case list, different question. `test_gpu_full_table` / `test_gpu_partitioned`
//! ask "is the answer right"; this target asks "how long did each plan node take", and
//! asserts nothing at all — all three `include!` the same `common/gpu_cases.inc`, so
//! the sets cannot drift apart.
//!
//! Each case runs `BENCH_WARMUP_RUNS` discarded executions and then
//! `BENCH_MEASURED_RUNS` measured ones, and the run with the SECOND-SMALLEST
//! end-to-end time is written to
//!
//!   testdata/benchmark-results/<dataset>.sf<sf>/<query>.<label>.benchmark.txt
//!
//! as the plan tree with `time_us` per node, then `build_profile` (which release profile
//! the harness was compiled under — see [`common::benchmark::BUILD_PROFILE`]),
//! `allocator` (the rmm pool the times were taken under — see
//! `executors::backend::gpu_node_executor::install_rmm_pool`), `nodes_total_us` and
//! `total_us`.
//!
//! The first two are conditions of the run rather than variables of it: the records only
//! compare with each other, and with the C++ gtest binaries' numbers, because every one of
//! them is a release build measuring over a pool. `run_gpu_benchmark` asserts both before
//! it measures anything, so the two lines say WHICH profile and which pool sizes, not
//! whether there was one. See it too for why one whole run is picked rather than a per-node
//! minimum, and `executors::backend::gpu_node_executor::set_node_timing` for why measuring
//! at all requires synchronizing the CUDA stream.
//!
//! `<label>` is the `<mode>-<tp>-<tier>` component the `.cpu.txt` goldens already
//! carry, and it is in the file name because 16 of the cases (tpch q1/q3/q5/…, tpcds
//! q17, the shuffle_* set) appear at BOTH `full_table-tp1-standard` and
//! `partitioned-tp8-standard`: same query, different plan, different time.
//!
//! NOT run by CI (see `INTENTIONALLY_NOT_IN_CI` in test_ci_coverage.rs): it needs a
//! GPU, it takes tens of minutes, and its output is a measurement, not a gate.
//! Build and run it with `scripts/build-test-shadgpu.sh --build-benchmarks` and
//! `--run-benchmarks` (which is also what compiles it under `[profile.benchmarks]`
//! rather than the opt-level-1 default), or on the host directly:
//!
//!   ./peacock_gpu_benchmarks --nocapture --test-threads=1
//!
//! `--test-threads=1` is not optional — cuDF/RMM share one process-wide pool, and
//! concurrent queries would time each other's contention.
#![cfg(not(feature = "rust-only"))]
// The golden labels and the derived fn names follow the suite convention.
#![allow(non_snake_case)]
mod common;

use common::exec_mode::ExecMode;

/// This target's reading of a case-list entry: time it, write the record, assert
/// nothing.
///
/// Unlike the two correctness targets, BOTH arms expand — this is the one place the
/// whole list runs. Each arm names its own [`ExecMode`], for the same reason the
/// correctness macros do: the mode is stated by the call site, never recovered by
/// parsing the label. `$mode` (the result-comparison mode) is accepted and ignored —
/// it says how the gate should CHECK the answer, which is not this target's business.
macro_rules! gpu_case {
    ($dataset:ident, $sf:literal, $query:ident, full_table_tp1_standard, $mode:ident) => {
        bench_case!($dataset, $sf, $query, full_table_tp1_standard, ExecMode::FullTable);
    };
    ($dataset:ident, $sf:literal, $query:ident, partitioned_tp8_standard, $mode:ident) => {
        bench_case!($dataset, $sf, $query, partitioned_tp8_standard, ExecMode::Partitioned);
    };
}

/// The body both arms share. Split out so the two arms differ ONLY in the label and
/// the mode they pair it with — the pairing being the thing that has to be right.
macro_rules! bench_case {
    ($dataset:ident, $sf:literal, $query:ident, $label:ident, $mode:expr) => {
        paste::paste! {
            #[tokio::test]
            async fn [<bench_ $dataset _sf $sf _ $query _ $label>]() {
                common::benchmark::run_gpu_benchmark(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    stringify!($label),
                    $mode,
                )
                .await;
            }
        }
    };
}

include!("common/gpu_cases.inc");
