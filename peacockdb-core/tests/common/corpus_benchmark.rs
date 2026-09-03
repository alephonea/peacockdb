//! One benchmark case at one batch-partitioned mode: plan it, run it, keep what it took.
//!
//! The timing counterpart of [`corpus_gpu`](super::corpus_gpu), and deliberately not part
//! of it: that module holds a device run to what the cpu wrote, and this one compares
//! nothing at all. Sharing a module would put an assertion path and a measurement path
//! behind one door.

use std::path::PathBuf;
use std::time::Instant;

use peacockdb_core::batch_partitioned::GpuNode;
use peacockdb_core::gpu_executor::{
    NodeTiming, RmmPool, install_rmm_pool, nvtx_range, set_node_timing, set_nvtx_ranges,
};
use peacockdb_core::batch_partitioned::driver::{
    Measurements, Region, RunReport, batch_partitioned_driver, join_regions, node_measured,
    nodes_as_recorded,
};
use peacockdb_core::batch_partitioned::gpu_backend::backend::GpuBackend;
use peacockdb_core::batch_partitioned::plan_text::render_timings;
use peacockdb_core::batch_partitioned::recipe::attach_recipes;

use super::bp_mode::mode_named;
use super::corpus::plan_at;
use super::corpus_golden::{Regeneration, SKIPPED, merge_section};
use super::record::{RunMeta, append_records, declared_steps, record_rows, rows_match_the_recipes};
use super::gpu_session::Session;
use super::registry::stem;
use super::testdata_root;

/// One benchmark declaration, submitted where it is written.
///
/// Separate from `RegistryEntry`, which is keyed to a column of the coverage CSV: the
/// benchmark list is its own on purpose, because the sf worth timing and the sf worth
/// checking differ. What this answers is narrower — which queries a (dataset, mode) file
/// is supposed to hold a section for.
pub struct BenchmarkCase {
    pub dataset: &'static str,
    pub sf: &'static str,
    /// Underscore form, as written in the macro (`q6`).
    pub query: &'static str,
    /// Underscore form, as written in the macro (`bp_tp1_single`), or `NOT_TIMED`.
    pub mode: &'static str,
}

/// The `mode` a declaration carries when it names no mode at all — a query written down
/// and deliberately not timed. A record rather than an omission, so it reaches every file
/// of its dataset as a marker instead of being absent for an unstated reason.
pub const NOT_TIMED: &str = "none";

inventory::collect!(BenchmarkCase);

/// The sections a `(dataset, sf, mode)` file is declared to hold: the queries timed at
/// this mode, and a marker for each one declared and timed at none.
///
/// Sorted numerically within the letter prefix, so `q2` precedes `q10`. `inventory` gives
/// no order of its own — it is link order — and a file whose sections move between builds
/// would diff against itself.
pub fn declared_for(dataset: &str, sf: &str, mode: &str) -> Vec<(String, Option<String>)> {
    let mut declared: Vec<(String, Option<String>)> = inventory::iter::<BenchmarkCase>
        .into_iter()
        .filter(|case| case.dataset == dataset && case.sf == sf)
        .filter(|case| case.mode == mode || case.mode == NOT_TIMED)
        .map(|case| {
            let marker = (case.mode == NOT_TIMED)
                .then(|| format!("{SKIPPED}declared with no mode to time it at"));
            (stem(case.query), marker)
        })
        .collect();
    declared.sort_by(|(a, _), (b, _)| query_order(a).cmp(&query_order(b)));
    declared.dedup_by(|(a, _), (b, _)| a == b);
    declared
}

/// `testdata/benchmark-results/<dataset>.sf<sf>/<mode>.benchmark.txt` — one file per
/// (dataset, mode), holding a section per query it timed.
///
/// One file rather than one per query because a mode's queries are read together: what a
/// reader compares is this mode against that one, and a directory of one-query files makes
/// that a directory listing rather than a diff.
/// Set ⇒ a case measures and records but leaves the `.benchmark.txt` tree alone.
///
/// For the HBM pass, whose times are distorted by the counters it exists to read. Named
/// on the harness rather than decided by it: a run that must not publish its times is a
/// property of how it was launched, and the case cannot see the nsys command line.
pub const RESULTS_READ_ONLY_ENV: &str = "PEACOCK_BENCHMARK_RESULTS_RO";

pub fn results_file(dataset: &str, sf: &str, mode: &str) -> PathBuf {
    testdata_root()
        .join(format!("benchmark-results/{dataset}.sf{sf}"))
        .join(format!("{}.benchmark.txt", file_stem_of(mode)))
}

/// A mode identifier as the file stem for it: `bp_tp4_sized` names
/// `tp4_sized.benchmark.txt`.
///
/// The identifier carries `bp_` because that is what a mode is called throughout the code
/// — `BpMode::name` is `bp-tp4-sized`, and the case list writes the underscore form of it.
/// The stem drops it because the file has no need of it: what the prefix distinguishes is
/// this execution model from the legacy one, and every file under `benchmark-results/` is
/// this model's. It would be a distinction against a sibling that does not exist here, and
/// will not exist anywhere once the legacy path is removed.
fn file_stem_of(mode: &str) -> &str {
    mode.strip_prefix("bp_").unwrap_or(mode)
}

/// Merge this query's section into its file, under the lock the goldens take.
///
/// Reused rather than reimplemented: `merge_section` already locks the file, reads inside
/// the critical section and publishes by rename, and those three are the whole of what
/// makes several cases writing one path safe.
///
/// Always `Sections`, never `Whole`. This file is written rather than asserted — what a
/// benchmark produces is a measurement, and correctness belongs to `corpus_query!` — and
/// the binary can be run under a name filter at any moment with nothing in the run to say
/// so. Pruning what this run did not produce would delete a measurement nobody asked to
/// lose; clearing a stale section is a deliberate act, not a side effect of a filtered run.
pub fn write_section(dataset: &str, sf: &str, mode: &str, query: &str, body: &str) {
    // The one run whose tree is knowingly wrong: an HBM pass measures under GPU memory
    // counters, which cost the query ~7% and a heavy scan ~11%. It wants the traffic and
    // the record's coordinates, and it must not leave those times behind in the committed
    // tree. Refusing to write is the whole guard — its record goes to its own file.
    if std::env::var_os(RESULTS_READ_ONLY_ENV).is_some() {
        return;
    }
    merge_section(
        &results_file(dataset, sf, mode),
        &declared_for(dataset, sf, mode),
        query,
        body,
        Regeneration::Sections,
    );
}

/// A query name as `(prefix, number, whole)`, so `q2` sorts before `q10` and a name with
/// no number in it still has a total order.
fn query_order(query: &str) -> (String, u32, String) {
    let digits = query.trim_start_matches(|c: char| !c.is_ascii_digit());
    let head = &query[..query.len() - digits.len()];
    let number: String = digits.chars().take_while(char::is_ascii_digit).collect();
    (
        head.to_string(),
        number.parse().unwrap_or(0),
        query.to_string(),
    )
}

/// How this harness was compiled, as the record states it.
///
/// Built by `--build-benchmarks` this reads `benchmarks opt-level=3`; from a plain
/// `cargo test` it would read `debug opt-level=1`, which measures a different host
/// overhead — see `[profile.benchmarks]` in the workspace Cargo.toml. A run under that
/// build is refused rather than recorded, so this line says WHICH release build measured,
/// not whether one did.
pub const BUILD_PROFILE: &str =
    concat!(env!("PEACOCK_BUILD_PROFILE"), " opt-level=", env!("PEACOCK_BUILD_OPT_LEVEL"));

/// The extra-data section: what the whole run cost, and under what conditions.
///
/// Named `--- run ---` after the `--- recipes ---` and `--- memory ---` the plan goldens
/// carry, so a reader who has seen one file knows where a section ends in the other.
///
/// **Two totals, deliberately named apart.** `run_us` is the whole execution end to end;
/// `device_us` is the sum of the tree's `total_us`. They do NOT agree, and the difference
/// is the point: it is what the run spent outside the calls — the driver's own scheduling
/// and the host prologue between them. One name for both would read as a discrepancy.
fn run_section(chosen: &Run, times: &Measurements, spread: &[u64]) -> String {
    let device_us: u64 = (0..times.nodes())
        .filter_map(|node| node_measured(times, node))
        .map(|time| time.device_us)
        .sum();
    let spread: Vec<String> = spread.iter().map(u64::to_string).collect();
    format!(
        "--- run ---\n\
         run_us={}\n\
         device_us={device_us}\n\
         runs=[{}]\n\
         build_profile={BUILD_PROFILE}\n\
         allocator={}\n",
        chosen.total_us,
        spread.join(","),
        install_rmm_pool(),
    )
}

/// Discarded runs. The first execution pays for the page cache, CUDA module load and JIT,
/// and allocator growth — the host's recent history rather than the plan. The pool removes
/// most of the third before this runs, which is a reason to keep the warm-up: what is left
/// is the part that varies.
const BENCH_WARMUP_RUNS: usize = 1;

/// Measured runs per query. The reported run is the second-smallest by end-to-end time:
/// the minimum is the run most likely to have caught a favourable scheduling accident, the
/// rest are dragged up by whatever else the shared host was doing. Must be >= 2.
const BENCH_MEASURED_RUNS: usize = 10;

/// One measured execution.
struct Run {
    total_us: u64,
    report: RunReport,
    /// What the device answered with, drained before the session that recorded it closed.
    regions: Vec<Region>,
}

/// Time `query` at `mode`.
///
/// Takes the mode as its macro spelling (`bp_tp4_sized`) rather than a `&BpMode`, so a
/// case-list line and a call site read alike; `mode_named` resolves it and panics naming
/// the five when it is not one of them.
///
/// Planning happens once, outside the runs: `plan_at` registers the dataset's tables,
/// which reads every file's parquet metadata, and repeating that would time the catalog
/// rather than the query. So `total_us` here is EXECUTION only — narrower than the legacy
/// record's, which timed parse and plan too. What the written record carries is decided
/// with its format, not here.
pub async fn benchmark_case(dataset: &str, sf: &str, query: &str, mode: &str) {
    const _: () = assert!(BENCH_MEASURED_RUNS >= 2, "a second minimum needs >= 2 runs");

    let mode = mode_named(mode);
    let what = format!("{dataset}/{query} at {} on a device", mode.name);

    // Conditions of a benchmark run, not choices: rmm's default makes every cuDF
    // intermediate a cudaMalloc/cudaFree round trip charged to the node that allocated it,
    // and a draining measurement reports a schedule the engine does not run. Installing
    // the pool first because rmm takes whatever resource is current when it allocates.
    let _ = install_rmm_pool();
    set_node_timing(NodeTiming::Events);

    let (_ctx, tree) = plan_at(dataset, sf, query, mode).await;

    // OFF across the warm-up, and set here rather than assumed: the switch is
    // process-global and this binary runs its cases in one process, so a case that only
    // ever turned it ON would have every LATER case warm up under the previous one's
    // setting — ranges emitted with no case range open, which is a call belonging to no
    // query. The capture's containment check catches exactly that, and did.
    set_nvtx_ranges(false);
    for _ in 0..BENCH_WARMUP_RUNS {
        run_once(tree.as_ref(), &what);
    }
    // After the warm-up, not before it. Ranges are for a capture and cost device work of
    // their own, so they stay behind the variable the capture sets — and the warm-up is
    // not written to the record, so ranging it would leave a capture with one more
    // execution than the file it is joined against. Deriving that off in the reader
    // means teaching it a Rust constant.
    if std::env::var_os("PEACOCK_NVTX").is_some() {
        set_nvtx_ranges(true);
    }
    // Around the measured runs, naming the case: a capture holding several of them cannot
    // say from a node range's name which query it was in — the name carries `seq`, and seq
    // numbering restarts with every plan, so q6 and q19 both open with `0.0 CudfScan`.
    // Containment answers it, and `nsys_hbm.py` reads the case off this rather than being
    // told it on a command line, which is a thing a human can get wrong in silence.
    //
    // Held to the end of the function: dropping it here would close the range before the
    // runs it is meant to contain.
    let _case = nvtx_range(&format!("{dataset}.sf{sf} {query} {}", mode.name));

    let mut runs = Vec::with_capacity(BENCH_MEASURED_RUNS);
    for _ in 0..BENCH_MEASURED_RUNS {
        let started = Instant::now();
        let (report, regions) = run_once(tree.as_ref(), &what);
        // Read after the run rather than inside it: `GpuUnload` copies the root off the
        // device, so the driver has returned only once the device is actually finished.
        // Under events nothing else would guarantee that — the node walk returns while
        // the stream may still be running.
        runs.push(Run {
            total_us: started.elapsed().as_micros() as u64,
            report,
            regions,
        });
    }

    let times: Vec<u64> = runs.iter().map(|run| run.total_us).collect();
    // Before the pick, and every run rather than the chosen one: the record's rows are the
    // fit's samples, and a spread of ten is what tells a coefficient from an accident. The
    // file beside it reports one run instead — the two answer different questions.
    //
    // In the order they ran, which `second_smallest` is about to destroy: a repeat of
    // `call_index` 0 is where one execution's rows end, and sorting first would interleave
    // executions with no way back.
    let nodes = nodes_as_recorded(tree.as_ref()).unwrap_or_else(|e| panic!("{what}: {e}"));
    let allocator = install_rmm_pool().to_string();
    let meta = RunMeta {
        dataset,
        sf,
        query,
        mode: mode.name,
        timing_mode: "events",
        build_profile: BUILD_PROFILE,
        allocator: &allocator,
    };
    // Attached once more here rather than reached for through the session: `Session::open`
    // hands its plan to the driver and the driver consumes it. `attach_recipes` reads the
    // finished tree and builds a buffer, which is host work outside every measured run.
    let recipes = attach_recipes(tree.as_ref()).unwrap_or_else(|e| panic!("{what}: {e}"));
    let declared = declared_steps(&recipes);
    // One append for the case rather than one per run: the file is opened, its heading
    // checked and its lock taken each time, and ten of that says nothing ten times.
    let rows: Vec<String> = runs
        .iter()
        .enumerate()
        .flat_map(|(at, run)| {
            let rows = record_rows(&nodes, &run.report, &measured_of(run, &what), &meta, at);
            // Per run, before they are flattened: the check reads one execution's
            // `call_index` sequences, and ten concatenated executions repeat every one.
            rows_match_the_recipes(&rows, &declared)
                .unwrap_or_else(|e| panic!("{what}: the record disagrees with the plan: {e}"));
            rows
        })
        .collect();
    append_records(&rows, &meta);

    let chosen = second_smallest(runs);
    let costed = measured_of(&chosen, &what);
    // Three terms, never their sum: under events the host submission and the device
    // execution overlap, so adding them describes no interval.
    let body = format!(
        "{}{}",
        render_timings(tree.as_ref(), &costed),
        run_section(&chosen, &costed, &times)
    );
    write_section(dataset, sf, mode.ident().as_str(), query, &body);

    let per_node: Vec<String> = (0..costed.nodes())
        .map(|node| match node_measured(&costed, node) {
            // Three states again: not measured, measured but addressing no seq (an unload
            // exports through a door that opens no region), and measured with regions.
            None => "unmeasured".to_string(),
            Some(t) if t.regions == 0 => "no regions".to_string(),
            Some(t) => format!("{}/{}/{}", t.host_setup_us, t.host_submit_us, t.device_us),
        })
        .collect();
    // Printed until there is a file to write it to. Every time, not just the chosen one: a
    // second minimum says nothing about the spread it was picked out of.
    println!(
        "{what}: {}us of {times:?}, {} nodes, {} regions, per-node setup/submit/device {per_node:?}",
        chosen.total_us,
        chosen.report.emitted.len(),
        chosen.regions.len()
    );
}

/// One run's calls costed, refusing a region no recorded call named.
///
/// A region nobody claimed means the two sides disagree about what ran — a defect in the
/// join, not a number to report around.
fn measured_of(run: &Run, what: &str) -> Measurements {
    let (costed, unclaimed) = join_regions(&run.report, &run.regions);
    assert!(
        unclaimed.is_empty(),
        "{what}: {} of {} regions match no recorded call — the first is {:?}",
        unclaimed.len(),
        run.regions.len(),
        unclaimed[0]
    );
    costed
}

/// The run worth reporting: second-smallest by end-to-end time.
///
/// The minimum is the run most likely to have caught a favourable scheduling accident,
/// and the rest are dragged up by whatever else the shared host was doing.
///
/// A whole run rather than a per-node minimum across runs: the latter gives a tree
/// belonging to no single execution, which can sum to less than any of them.
fn second_smallest(mut runs: Vec<Run>) -> Run {
    runs.sort_by_key(|run| run.total_us);
    runs.swap_remove(1)
}

/// One device session and one run over it.
///
/// The session is per run rather than per case: `attach_recipes` and `begin_plan` are what
/// a query costs on this side of the FFI, and holding one across runs would time the
/// second differently from the first. Executor construction rides along and is a plain
/// `new` of a small struct — the thing the legacy harness kept outside its loop was a
/// DataFusion `SessionContext`, which here is outside it already.
///
/// No budget, so the accountant records without ever tripping — a benchmark that refuses
/// to finish reports nothing, and the mode's budget has already done its work at plan
/// time, where it sized the batches.
fn run_once(tree: &dyn GpuNode, what: &str) -> (RunReport, Vec<Region>) {
    // In the measured path rather than beside the install, which is the whole point: the
    // install was once lost with the file that held it, and a check standing next to what
    // it guards goes the same way. `install_rmm_pool` is idempotent, so asking here is
    // asking what the resource IS.
    assert!(
        matches!(install_rmm_pool(), RmmPool::Pool { .. }),
        "{what} would measure over rmm's default resource, where every cuDF intermediate is \
         a cudaMalloc/cudaFree round trip charged to the node that allocated it — the \
         numbers would describe the allocator, not the plan"
    );
    // Beside it because it is the same kind of statement: the record's `build_profile` line
    // says WHICH release build measured, and a line saying that is a claim until something
    // refuses the build that would make it false. A plain `cargo test` compiles this at
    // opt-level 1, where the host prologue is a different quantity entirely.
    assert!(
        !cfg!(debug_assertions),
        "{what} would measure a debug build ({BUILD_PROFILE}); build it with \
         `scripts/build-test-shadgpu.sh --build-benchmarks`, which compiles under \
         [profile.benchmarks]"
    );
    let mut session = Session::open(tree, what);
    let ctx = session.context();
    let report = batch_partitioned_driver::<GpuBackend>(tree, &ctx, None)
        .unwrap_or_else(|e| panic!("{what}: {e}"));

    // Checked on every run, not on the reported one: a leak surfaces later as a case
    // timing a device that is still holding batches, and by then it names the wrong query.
    assert_eq!(report.in_flight_bytes, 0, "{what} ended holding batches");
    assert_eq!(
        report.holds, report.releases,
        "{what} held {} batches and released {}",
        report.holds, report.releases
    );
    // Drained here rather than by the caller: the events die with the session, and the
    // session is this function's. The cap is what the run can have produced — every
    // recorded call, times the widest output a single call has, which is a scatter's lanes.
    let calls: usize = report
        .abi_calls
        .iter()
        .flatten()
        .flatten()
        .filter_map(|made| made.recorded())
        .map(|made| made.len())
        .sum();
    let widest = report.lanes_of.iter().copied().max().unwrap_or(1).max(1);
    let regions = session.regions(calls * widest, what);
    (report, regions)
}
