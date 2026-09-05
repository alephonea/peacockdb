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

use std::time::Instant;

use peacockdb_core::batch_partitioned::GpuNode;
use peacockdb_core::batch_partitioned::driver::{Region, batch_partitioned_driver};
use peacockdb_core::batch_partitioned::gpu_backend::backend::GpuBackend;
use peacockdb_core::gpu_executor::{NodeTiming, install_rmm_pool, set_node_timing};

use common::bp_mode::mode_named;
use common::corpus::plan_at;
use common::gpu_session::{Session, region_cap};

/// q19 for its SHAPE, not its answer (`test_gpu_bp_corpus` owns that): scan → filter →
/// join → aggregate covers enough operator families that a missing `mark_device_start`
/// shows up as a too-small Σ device_us. The richest device-enabled query at sf1 — q3 has
/// more node kinds but does not run on the device, and this needs regions, not a plan.
const DATASET: &str = "tpch";
const SF: &str = "1";
const QUERY: &str = "q19";
/// One batch per partition, so check 3b's wall-clock bound is as tight as it gets.
const MODE: &str = "bp_tp1_single";

/// Each round runs both modes, so the totals come from interleaved samples. In blocks,
/// host drift would land entirely on whichever mode ran last — which is the difference
/// check 2 reports.
const ROUNDS: usize = 7;

/// The bound, and the only one: a guessed constant far above the real cost of two
/// `cudaEventRecord`s. Measured at +0.0% against unobserved, so the margin is twentyfold.
/// Not tightened for that reason: on a shared host a bound tight enough to fail on noise
/// gets muted, which is worse than a loose one that never does.
const EVENTS_GROSS_LIMIT: f64 = 1.20;

/// One execution.
struct Run {
    total_us: u64,
    /// ABI calls the driver made, counted by the driver rather than by the journal —
    /// the journal is armed only in a measured mode, so it is the one figure of the two
    /// that both modes report.
    calls: usize,
    /// The run's region bound: recorded calls times the widest output any one has. At
    /// tp1 that width is 1, so the regions this run produced must number exactly this.
    region_cap: usize,
    regions: Vec<Region>,
}

/// Second-smallest wall clock, matching the benchmark harness: the minimum is the sample
/// most likely to have caught a favourable scheduling accident, the rest are dragged up
/// by whatever else the host was doing.
fn second_smallest(mut runs: Vec<Run>) -> Run {
    runs.sort_by_key(|run| run.total_us);
    runs.swap_remove(1)
}

/// One execution, timed end to end, with what the device recorded for it.
fn run_once(tree: &dyn GpuNode, what: &str) -> Run {
    let mut session = Session::open(tree, what);
    let ctx = session.context();
    let started = Instant::now();
    let report = batch_partitioned_driver::<GpuBackend>(tree, &ctx, None)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    // Read after the run, not inside it: `GpuUnload` copies the root off the device, so
    // the driver has returned only once the device is actually finished. Under events
    // nothing else would guarantee that — the walk returns while the stream may still run.
    let total_us = started.elapsed().as_micros() as u64;
    let cap = region_cap(&report);
    Run {
        total_us,
        calls: report.calls,
        region_cap: cap,
        regions: session.regions(cap, what),
    }
}

fn sum(regions: &[Region], f: fn(&Region) -> u64) -> u64 {
    regions.iter().map(f).sum()
}

#[tokio::test]
async fn events_are_free_and_land_where_they_claim() {
    const _: () = assert!(ROUNDS >= 2, "a second minimum needs >= 2 rounds");

    // Before anything allocates: rmm uses whatever resource is current at the time of the
    // call, and without it the two walls would differ by allocator behaviour as much as
    // by instrument.
    let allocator = install_rmm_pool();

    let mode = mode_named(MODE);
    let what = format!("{DATASET}/{QUERY} at {}", mode.name);
    let (_ctx, tree) = plan_at(DATASET, SF, QUERY, mode).await;

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
    run_once(tree.as_ref(), &what);

    let modes = [NodeTiming::Off, NodeTiming::Events];
    let mut runs: Vec<Vec<Run>> = modes.iter().map(|_| Vec::with_capacity(ROUNDS)).collect();
    for _ in 0..ROUNDS {
        for (slot, mode) in runs.iter_mut().zip(modes) {
            set_node_timing(mode);
            slot.push(run_once(tree.as_ref(), &what));
        }
    }
    set_node_timing(NodeTiming::Off);

    let mut picked = runs.into_iter().map(second_smallest);
    let off = picked.next().unwrap();
    let events = picked.next().unwrap();
    let (off_us, events_us) = (off.total_us, events.total_us);
    let (off_regions, ev_regions) = (&off.regions, &events.regions);

    let ev_setup = sum(ev_regions, |r| r.host_setup_us);
    let ev_device = sum(ev_regions, |r| r.device_us);
    let over = |us: u64| 100.0 * (us as f64 / off_us as f64 - 1.0);
    eprintln!(
        "node-timing {DATASET}/{QUERY} [{}] alloc=[{allocator}] regions={}\n  \
         off    wall={off_us}us\n  \
         events wall={events_us}us ({:+.1}%)  setup={ev_setup} submit={} device={ev_device}",
        mode.name,
        ev_regions.len(),
        over(events_us),
        sum(ev_regions, |r| r.host_submit_us),
    );

    // 1. Off is off. A region is opened only in a measured mode, so a leak shows up as
    // regions existing at all — and it would put an event pair into every correctness run
    // in the process, where nothing would fail and the suite would just get slower.
    assert!(
        off_regions.is_empty(),
        "NodeTiming::Off recorded {} regions",
        off_regions.len(),
    );

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
    // at region ENTRY produces, since the pair then swallows the next call's launches.
    assert!(
        ev_device <= events_us,
        "Σ device_us = {ev_device}us exceeds the {events_us}us wall clock: the event \
         pairs are not disjoint, so at least one spans work outside its own region",
    );

    // 3c. The split is not degenerate. `host_setup_us` is the peacockdb-only prologue the
    // cost model fits as its own constant, and the reason the region is cut in two; zero
    // means the marks sit at region entry and the prologue is billed as device work.
    assert!(
        ev_setup > 0,
        "Σ host_setup_us is zero: no region reported any pre-device host time, so the \
         `mark_device_start` calls are at region entry rather than at the first device \
         touch, and const_peacock cannot be fitted from records taken this way",
    );

    // 3d. Nothing went unmeasured. The sums above are over whatever regions came back,
    // so a run that recorded half its calls would satisfy every check so far while
    // describing half a query. At tp1 a call has one output partition, so the two counts
    // are equal rather than merely ordered.
    assert_eq!(
        ev_regions.len(),
        events.region_cap,
        "{} regions for {} recorded ABI calls: a call ran without opening one",
        ev_regions.len(),
        events.region_cap,
    );
    assert_eq!(off.calls, events.calls, "off and events ran different plans");
}
