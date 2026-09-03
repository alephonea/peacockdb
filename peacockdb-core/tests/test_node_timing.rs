//! Does the instrument change what it measures, and does it measure what it claims?
//!
//! Every record under `testdata/benchmark-results/` assumes a node's reported time is
//! the time it would have taken unobserved. Events record without draining the stream,
//! so they can satisfy that — whether they do is measured here rather than assumed.
//!
//! One query, because this is about the instrument and not the corpus:
//!   1. Is `Off` actually off?
//!   2. Does `Events` cost anything measurable? (the assertion)
//!   3. Do the events land where they claim?
#![cfg(not(feature = "rust-only"))]
mod common;

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use peacockdb_core::cpu_executor::NodeMemoryStats;
use peacockdb_core::gpu_executor::{install_rmm_pool, set_node_timing, GpuExecutor, NodeTiming};

use common::exec_mode::{golden_label, ExecMode};
use common::{data_dir_for, device_config, queries_dir_for};

/// q3 for its SHAPE, not its answer (`test_gpu_full_table` owns that): scan → filter →
/// two joins → aggregate → sort → limit covers enough operator families that a missing
/// `mark_device_start` shows up as a too-small Σ device_us.
const DATASET: &str = "tpch";
const SF: &str = "1";
const QUERY: &str = "q3";
/// tp1: one output partition per node, so check 4b's wall-clock bound is as tight as
/// it gets.
const DEVICE: &str = "tp1-standard";
const MODE: ExecMode = ExecMode::FullTable;

/// Each round runs all three modes, so the totals come from interleaved samples. In
/// blocks, host drift would land entirely on whichever mode ran last — which is the
/// difference check 2 reports.
const ROUNDS: usize = 7;

/// The bound, and the only one: a guessed constant far above the real cost of two
/// `cudaEventRecord`s. A tighter bound calibrated against a draining mode's cost stood
/// beside it and went with that mode — on this plan it never engaged, since draining cost
/// +0.1% here, below its own significance threshold. Measured: events cost +0.0% against
/// unobserved, so the margin to this limit is twentyfold. Not tightened for that reason:
/// on a shared host a bound tight enough to fail on noise gets muted, which is worse.
const EVENTS_GROSS_LIMIT: f64 = 1.20;

type Run = (u64, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>);

/// Second-smallest wall clock, matching the benchmark harness: the minimum is the sample
/// most likely to have caught a favourable scheduling accident, the rest are dragged up
/// by whatever else the host was doing.
fn second_smallest(mut runs: Vec<Run>) -> Run {
    runs.sort_by_key(|(us, _, _)| *us);
    runs.swap_remove(1)
}

fn sum(stats: &[NodeMemoryStats], f: fn(&NodeMemoryStats) -> u64) -> u64 {
    stats.iter().map(f).sum()
}

#[tokio::test]
async fn events_are_free_and_land_where_they_claim() {
    const _: () = assert!(ROUNDS >= 2, "a second minimum needs >= 2 rounds");

    // Before the executor, so before any cuDF allocation: rmm uses whatever resource is
    // current at the time of the call. Without it the three walls would differ by
    // allocator behaviour as much as by instrument.
    let allocator = install_rmm_pool();

    let (partitions, budget) = device_config(DEVICE);
    let data_dir = data_dir_for(DATASET, SF);
    let sql_path = queries_dir_for(DATASET).join(format!("{QUERY}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));

    let gpu = GpuExecutor::new_mode(&data_dir, partitions, budget, MODE.partition_mode())
        .await
        .unwrap();

    // The switch is process-global. A guard rather than a trailing reset: everything
    // below unwraps, and an unwind would leave every later test in this binary measured.
    struct Loan;
    impl Drop for Loan {
        fn drop(&mut self) {
            set_node_timing(NodeTiming::Off);
        }
    }
    let _loan = Loan;

    // Discarded: the first execution pays for the page cache, CUDA module load and JIT,
    // and allocator growth — none of which belongs to whichever mode runs first.
    set_node_timing(NodeTiming::Off);
    gpu.execute_instrumented(&sql).await.unwrap();

    let modes = [NodeTiming::Off, NodeTiming::Events];
    let mut runs: Vec<Vec<Run>> = modes.iter().map(|_| Vec::with_capacity(ROUNDS)).collect();
    for _ in 0..ROUNDS {
        for (slot, mode) in runs.iter_mut().zip(modes) {
            set_node_timing(mode);
            let t0 = std::time::Instant::now();
            let (_batches, plan, stats) = gpu.execute_instrumented(&sql).await.unwrap();
            // `execute_instrumented` materializes the root off the device, so the clock
            // is read after the device has finished. Under Events nothing else would
            // guarantee that: the node walk returns while the stream may still run.
            slot.push((t0.elapsed().as_micros() as u64, plan, stats));
        }
    }
    set_node_timing(NodeTiming::Off);

    let mut picked = runs.into_iter().map(second_smallest);
    let (off_us, _, off_stats) = picked.next().unwrap();
    let (events_us, plan, ev_stats) = picked.next().unwrap();

    let ev_setup = sum(&ev_stats, |s| s.host_setup_us);
    let ev_device = sum(&ev_stats, |s| s.device_us);
    let over = |us: u64| 100.0 * (us as f64 / off_us as f64 - 1.0);
    eprintln!(
        "node-timing {DATASET}/{QUERY} [{}] alloc=[{allocator}] nodes={}\n  \
         off    wall={off_us}us\n  \
         events wall={events_us}us ({:+.1}%)  setup={ev_setup} submit={} device={ev_device}",
        golden_label(MODE, DEVICE),
        ev_stats.len(),
        over(events_us),
        sum(&ev_stats, |s| s.host_submit_us),
    );

    // 1. Off is off. All three modes share one global, and a leak would put a stream
    // drain into every correctness run in the process — nothing would fail, the suite
    // would just quietly get slower.
    for s in &off_stats {
        assert_eq!(
            (s.host_setup_us, s.host_submit_us, s.device_us),
            (0, 0, 0),
            "NodeTiming::Off left a time on {}",
            s.node_name,
        );
    }

    // 2. The assertion this target exists for, against unobserved.
    assert!(
        events_us as f64 <= off_us as f64 * EVENTS_GROSS_LIMIT,
        "events mode cost {events_us}us against {off_us}us unobserved ({:+.1}%, limit \
         +{:.1}%) — a synchronization has gotten back inside the timed region, and \
         every benchmark record taken this way describes a schedule the engine does \
         not actually run",
        over(events_us),
        100.0 * (EVENTS_GROSS_LIMIT - 1.0),
    );
    // 3a. Zero means no region created its pair, or none reached `mark_device_start`.
    assert!(ev_device > 0, "events mode recorded no device time at all");

    // 3b. Placement. Regions record on cuDF's single default stream in host program
    // order, so their intervals are disjoint and must fit inside the wall clock.
    // Exceeding it means a pair spans work that is not its region's — what a mark left
    // at region ENTRY produces, since the start event is only reached after the host
    // prologue and the pair then swallows the next node's launches.
    assert!(
        ev_device <= events_us,
        "Σ device_us = {ev_device}us exceeds the {events_us}us wall clock: the event \
         pairs are not disjoint, so at least one spans work outside its own region",
    );

    // 3c. The split is not degenerate. `host_setup_us` is the peacockdb-only prologue
    // the cost model fits as its own constant, and the reason the region is cut in two;
    // zero means the marks sit at region entry and the prologue is billed as device work.
    assert!(
        ev_setup > 0,
        "Σ host_setup_us is zero: no node reported any pre-device host time, so the \
         `mark_device_start` calls are at region entry rather than at the first device \
         touch, and const_peacock cannot be fitted from records taken this way",
    );

    // 3d. Same tree in both modes — every comparison above is between whole-plan sums,
    // so a mode that changed the plan would make them meaningless, not just wrong.
    assert_eq!(ev_stats.len(), off_stats.len(), "off and events ran different plans");
    assert_eq!(ev_stats.len(), plan_nodes(&plan), "one stat per plan node");
}

fn plan_nodes(plan: &Arc<dyn ExecutionPlan>) -> usize {
    1 + plan.children().iter().map(|c| plan_nodes(c)).sum::<usize>()
}
