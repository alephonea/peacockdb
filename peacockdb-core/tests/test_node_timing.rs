//! Does the instrument change what it measures, and does it measure what it claims?
//! (#153)
//!
//! Every record under `testdata/benchmark-results/` assumes a node's reported time is
//! the time it would have taken unobserved. `Sync` cannot satisfy that by construction
//! — it drains the stream at every region boundary — which is why the events mode
//! exists; how much that buys is measured here rather than assumed.
//!
//! One query, because this is about the instrument and not the corpus:
//!   1. Is `Off` actually off?
//!   2. What does `Sync` cost? (reported, not asserted; it also calibrates 3)
//!   3. Does `Events` cost anything measurable? (the assertion)
//!   4. Do the events land where they claim?
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

/// Absolute backstop, far above the real cost of two `cudaEventRecord`s. Covers the
/// case the calibrated bound cannot: a plan that was serial anyway, where sync costs
/// nothing and there is no in-run scale to derive a threshold from. On a shared host a
/// bound tight enough to fail on noise gets muted, which is worse than a loose one.
const EVENTS_GROSS_LIMIT: f64 = 1.20;

/// How far above unobserved `sync` must land to count as a usable scale rather than
/// noise. Below it the two methods are indistinguishable on this plan and only
/// [`EVENTS_GROSS_LIMIT`] applies.
const SYNC_IS_SIGNIFICANT: f64 = 1.05;

type Run = (u64, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>);

/// Second-smallest wall clock, matching `run_gpu_benchmark`: the minimum is the sample
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
    // below unwraps, and an unwind would leave every later test in this binary measured
    // — or, under Sync, draining the stream at every region boundary.
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

    let modes = [NodeTiming::Off, NodeTiming::Events, NodeTiming::Sync];
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
    let (sync_us, _, sync_stats) = picked.next().unwrap();

    let ev_setup = sum(&ev_stats, |s| s.host_setup_us);
    let ev_device = sum(&ev_stats, |s| s.device_us);
    let over = |us: u64| 100.0 * (us as f64 / off_us as f64 - 1.0);
    eprintln!(
        "node-timing {DATASET}/{QUERY} [{}] alloc=[{allocator}] nodes={}\n  \
         off    wall={off_us}us\n  \
         events wall={events_us}us ({:+.1}%)  setup={ev_setup} submit={} device={ev_device}\n  \
         sync   wall={sync_us}us ({:+.1}%)  setup={} submit={} device={}",
        golden_label(MODE, DEVICE),
        ev_stats.len(),
        over(events_us),
        sum(&ev_stats, |s| s.host_submit_us),
        over(sync_us),
        sum(&sync_stats, |s| s.host_setup_us),
        sum(&sync_stats, |s| s.host_submit_us),
        sum(&sync_stats, |s| s.device_us),
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

    // 2. Sync's cost is printed above, not asserted: it is a property of the plan (how
    // much cuDF could have pipelined), not of this code. A floor would fail on a plan
    // that is serial anyway, a ceiling on exactly the plans most worth reporting.

    // 3. The assertion this target exists for, on two bounds. The gross one is a guessed
    // constant and catches only a gross regression. The calibrated one is the real
    // check: when sync is measurably above off, that gap is what a synchronization back
    // inside the region costs on this plan, host and run, so the midpoint discriminates
    // at the scale of the actual failure and survives a slow host, which moves all three
    // together.
    assert!(
        events_us as f64 <= off_us as f64 * EVENTS_GROSS_LIMIT,
        "events mode cost {events_us}us against {off_us}us unobserved ({:+.1}%, limit \
         +{:.1}%) — a synchronization has gotten back inside the timed region, and \
         every benchmark record taken this way describes a schedule the engine does \
         not actually run",
        over(events_us),
        100.0 * (EVENTS_GROSS_LIMIT - 1.0),
    );
    if sync_us as f64 >= off_us as f64 * SYNC_IS_SIGNIFICANT {
        let midpoint = (off_us + sync_us) / 2;
        assert!(
            events_us <= midpoint,
            "events wall {events_us}us sits on the SYNC side of the {midpoint}us \
             midpoint between unobserved ({off_us}us) and per-region draining \
             ({sync_us}us): the events mode is paying most of what the sync mode \
             pays, which is what it exists not to do",
        );
    } else {
        eprintln!(
            "  note: sync cost only {:+.1}% here, below the {:.0}% significance \
             threshold — no calibrated bound, only the {:.0}% gross one",
            over(sync_us),
            100.0 * (SYNC_IS_SIGNIFICANT - 1.0),
            100.0 * (EVENTS_GROSS_LIMIT - 1.0),
        );
    }

    // 4a. Zero means no region created its pair, or none reached `mark_device_start`.
    assert!(ev_device > 0, "events mode recorded no device time at all");

    // 4b. Placement. Regions record on cuDF's single default stream in host program
    // order, so their intervals are disjoint and must fit inside the wall clock.
    // Exceeding it means a pair spans work that is not its region's — what a mark left
    // at region ENTRY produces, since the start event is only reached after the host
    // prologue and the pair then swallows the next node's launches.
    assert!(
        ev_device <= events_us,
        "Σ device_us = {ev_device}us exceeds the {events_us}us wall clock: the event \
         pairs are not disjoint, so at least one spans work outside its own region",
    );

    // 4c. The split is not degenerate. `host_setup_us` is the peacockdb-only prologue
    // the cost model fits as its own constant, and the reason the region is cut in two;
    // zero means the marks sit at region entry and the prologue is billed as device work.
    assert!(
        ev_setup > 0,
        "Σ host_setup_us is zero: no node reported any pre-device host time, so the \
         `mark_device_start` calls are at region entry rather than at the first device \
         touch, and const_peacock cannot be fitted from records taken this way",
    );

    // 4d. Same tree in all three modes — every comparison above is between whole-plan
    // sums, so a mode that changed the plan would make them meaningless, not just wrong.
    assert_eq!(ev_stats.len(), off_stats.len(), "off and events ran different plans");
    assert_eq!(ev_stats.len(), sync_stats.len(), "sync and events ran different plans");
    assert_eq!(ev_stats.len(), plan_nodes(&plan), "one stat per plan node");
}

fn plan_nodes(plan: &Arc<dyn ExecutionPlan>) -> usize {
    1 + plan.children().iter().map(|c| plan_nodes(c)).sum::<usize>()
}
