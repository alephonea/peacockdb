//! The `peacock_gpu_benchmarks` harness: where a record is written, how it is
//! formatted, and the run shape the numbers in it come from.
//!
//! Split out of `common/mod.rs`, which every test target compiles — including the
//! rust-only ones, which have no FFI to call. So the gating here is per item rather
//! than per file: the path builder is pure and stays available; the formatter and the
//! run itself are cut out under `rust-only`, because both now read the conditions of
//! the run off the process instead of taking them as arguments.

use std::path::PathBuf;

use super::testdata_root;

// Everything but the path builder is cut out under rust-only: the run needs the FFI, and
// so now does the formatter, which reads the conditions of the run off the process rather
// than taking them as arguments.
#[cfg(not(feature = "rust-only"))]
use std::sync::Arc;

#[cfg(not(feature = "rust-only"))]
use datafusion::physical_plan::ExecutionPlan;
#[cfg(not(feature = "rust-only"))]
use peacockdb_core::cpu_executor::NodeMemoryStats;
#[cfg(not(feature = "rust-only"))]
use peacockdb_core::gpu_executor::{install_rmm_pool, RmmPool};

#[cfg(not(feature = "rust-only"))]
use super::exec_mode::{golden_label, gpu_label_device, ExecMode};
#[cfg(not(feature = "rust-only"))]
use super::{data_dir_for, device_config, queries_dir_for, OneLine};

/// Per-node timing snapshot written by the `peacock_gpu_benchmarks` target.
///
/// Deliberately NOT under `goldens/`: every tree there is mirrored to the GPU host
/// with `rsync --delete` by `--push-binaries`, which for a measurement history that
/// is written ON the host and travels the other way would mean erasing it from any
/// box that has not run the benchmarks itself.
///
/// `label` is the same `<mode>-<tp>-<tier>` component the `.cpu.txt` goldens carry
/// ([`exec_mode::golden_label`]) — the mode belongs in the name for the same reason
/// it does there: the same query at the same device produces a different plan, and
/// therefore different times, under full-table vs partitioned execution.
pub fn benchmark_result(dataset: &str, sf: &str, query: &str, label: &str) -> PathBuf {
    testdata_root()
        .join(format!("benchmark-results/{dataset}.sf{sf}"))
        .join(format!("{query}.{label}.benchmark.txt"))
}

// --- benchmark tree ---------------------------------------------------------
/// Format per-node EXECUTION TIMES as the same pre-order tree `cpu_stats_str`
/// renders costs into, then the two totals.
///
/// Same shape, same node labels (each node's own `Display`), same "sub-lines only
/// when N>1" rule — so a `.benchmark.txt` can be read side by side with its
/// `.cpu.txt` sibling, line for line. Only the trailing field differs: `time_us`
/// instead of `partitions=/output_rows=/output_bytes=`. Rows and bytes are
/// deliberately NOT repeated here; they are deterministic and already asserted next
/// door, and duplicating them would put non-reproducible and reproducible data in
/// one file.
///
/// `total_us` is the caller's END-TO-END wall clock for the query, which is strictly
/// larger than the node sum: it also covers SQL parsing, physical planning, plan
/// serialization, `begin_plan`, and the final `materialize` back across the FFI.
/// Both are written because the gap between them IS the per-query overhead.
///
/// `sync_floor_us` is what the measurement costs when there is nothing to measure
/// (see `measure_timing_floor_us`). It is written because every node time above
/// silently includes one, so without it a reader cannot separate "this node is cheap"
/// from "this node is under the method's resolution".
///
/// `build_profile` and `allocator` are read here rather than passed in, because they
/// are conditions of the run and not choices of the caller: every record is written
/// from a release build measuring over a pooled rmm resource, and `run_gpu_benchmark`
/// asserts both before it measures anything. The two lines say WHICH release profile
/// and which pool sizes, not whether there was one.
#[cfg(not(feature = "rust-only"))]
pub fn bench_stats_str(
    plan: &Arc<dyn ExecutionPlan>,
    stats: &[NodeMemoryStats],
    total_us: u64,
    sync_floor_us: u64,
) -> String {
    struct Node<'a> {
        stat: &'a NodeMemoryStats,
        plan: &'a Arc<dyn ExecutionPlan>,
        children: Vec<Node<'a>>,
    }
    fn collect<'a>(
        plan: &'a Arc<dyn ExecutionPlan>,
        stats: &'a [NodeMemoryStats],
        idx: &mut usize,
    ) -> Node<'a> {
        let children: Vec<Node<'a>> =
            plan.children().iter().map(|c| collect(c, stats, idx)).collect();
        let stat = &stats[*idx];
        *idx += 1;
        Node { stat, plan, children }
    }
    fn walk(node: &Node, indent: usize, lines: &mut Vec<String>) {
        lines.push(format!(
            "{}{}, time_us={}",
            " ".repeat(indent),
            OneLine(node.plan.as_ref()),
            node.stat.time_us,
        ));
        // Per-partition sub-lines only when N>1, matching the .cpu.txt convention.
        // For the Hash repartition the shared concat+scatter is charged to p0, so
        // these still sum to the node line above.
        if node.stat.part_stats.len() > 1 {
            let sub = " ".repeat(indent + 2);
            for (k, ps) in node.stat.part_stats.iter().enumerate() {
                lines.push(format!("{sub}p{k}: time_us={}", ps.time_us));
            }
        }
        for child in &node.children {
            walk(child, indent + 2, lines);
        }
    }
    let root = collect(plan, stats, &mut 0);
    let mut lines = Vec::new();
    walk(&root, 0, &mut lines);
    let nodes_total_us: u64 = stats.iter().map(|s| s.time_us).sum();
    // How many node lines sit at or under the floor — i.e. how much of the tree above
    // this file cannot actually resolve. One integer, because "sync_floor_us=6" alone
    // still leaves the reader counting by hand, and the answer changes the reading of
    // the whole record: 2/40 unresolved is a profile, 35/40 is a measurement that
    // mostly measured its own instrument.
    //
    // The floor scales with the partition count. sync_floor_us is ONE empty timed region,
    // but NodeMemoryStats::time_us is Σ over the node's output partitions, so a tp8 node
    // carries eight floors, not one. Comparing the sum against a single floor makes every
    // partitioned node look resolved when it is not — the very failure this line exists to
    // report, silently worst on exactly the widest plans.
    let at_floor = stats
        .iter()
        .filter(|s| s.time_us <= sync_floor_us * s.part_stats.len().max(1) as u64)
        .count();
    lines.push(String::new());
    lines.push(
        "# sync_floor_us = cost of the measurement itself (empty timed region: clock + \
         cudaStreamSynchronize on an idle stream). Each node time INCLUDES one PER OUTPUT \
         PARTITION, since a node line is the Σ over its partitions; a node at or below its \
         own floor (sync_floor_us x partitions) is unresolved, not cheap. Do not subtract it."
            .to_string(),
    );
    lines.push(
        "# nodes_total_us = Σ of the per-node times above; total_us = the whole query, \
         end to end (parse + plan + serialize + node walk + materialize)"
            .to_string(),
    );
    lines.push(
        "# build_profile = how the harness itself was compiled. Always a release build; \
         total_us MINUS nodes_total_us is that Rust, and node times are device work and \
         barely move."
            .to_string(),
    );
    lines.push(
        "# allocator = the rmm device resource the node times were taken under. Always a \
         pool: with rmm's default every cuDF intermediate is a cudaMalloc/cudaFree round \
         trip charged to the node that allocated it, which moves the PROFILE and not only \
         the scale. The sizes vary with free memory when the pool was built."
            .to_string(),
    );
    // Written for every tree, including trees with no repartition in them. A line
    // emitted only where it applies cannot be read: absence would mean either "this
    // plan has no scatter" or "this record predates the line", which is the same
    // ambiguity the disclosure exists to remove.
    lines.push(
        "# shared_work_charged_to = which p<k> sub-line carries work a node does once \
         for all of its output partitions. A hash repartition concatenates its input \
         and scatters it in a single operation, and that time is billed to p0 — so p0 \
         standing far above its siblings is the accounting, not skew. Sub-lines sum to \
         their node line either way. Written whether or not this plan has a repartition."
            .to_string(),
    );
    lines.push(format!("build_profile={BUILD_PROFILE}"));
    lines.push(format!("allocator={}", install_rmm_pool()));
    lines.push("shared_work_charged_to=p0".to_string());
    lines.push(format!("sync_floor_us={sync_floor_us}"));
    lines.push(format!("nodes_at_or_below_floor={at_floor}/{}", stats.len()));
    lines.push(format!("nodes_total_us={nodes_total_us}"));
    lines.push(format!("total_us={total_us}"));
    lines.join("\n")
}

// --- benchmark mode ---------------------------------------------------------
/// How this binary was compiled, as `<profile-dir> opt-level=<n>`, baked in by
/// `build.rs` and written into every record.
///
/// The one condition of a run a reader cannot recover from the numbers themselves.
/// Built by `--build-benchmarks` this reads `benchmarks opt-level=3`; from a plain
/// `cargo test` it would read `debug opt-level=1`, which measures a different host
/// overhead — see `[profile.benchmarks]` in the workspace Cargo.toml — and
/// `run_gpu_benchmark` refuses that build rather than recording it.
pub const BUILD_PROFILE: &str =
    concat!(env!("PEACOCK_BUILD_PROFILE"), " opt-level=", env!("PEACOCK_BUILD_OPT_LEVEL"));

/// Runs before the measured runs and is thrown away. The first execution of a query
/// pays for things that are not the query: the OS page cache for its parquet files,
/// CUDA module loading and JIT for the kernels this plan reaches, and any growth the
/// device allocator still has to do. Reporting those would say more about the host's
/// recent history than about the plan.
///
/// `install_rmm_pool` reserves most of free memory up front, so on a quiet host the
/// third of those is nearly gone before this runs — which is a reason to keep the
/// warm-up, not to drop it: what remains is the part that varies.
#[cfg(not(feature = "rust-only"))]
pub const BENCH_WARMUP_RUNS: usize = 1;

/// Measured runs per query. The reported tree is the run with the SECOND-smallest
/// end-to-end time — the fastest run is discarded as the one most likely to have
/// caught a favourable scheduling accident, and every other run is dragged upward by
/// whatever else the (shared) GPU host was doing. Must be >= 2.
#[cfg(not(feature = "rust-only"))]
pub const BENCH_MEASURED_RUNS: usize = 10;

/// Empty timed regions sampled to establish the resolution floor. Each costs one
/// `cudaStreamSynchronize` of an idle stream — microseconds — so a large sample is
/// cheap next to a single query, and a stable floor is worth more than the ~1ms it
/// costs: it is the number that decides whether a small node time means anything.
#[cfg(not(feature = "rust-only"))]
pub const BENCH_FLOOR_SAMPLES: u32 = 200;

/// Time one case from `common/gpu_cases.inc` and write `testdata/benchmark-results/…`.
///
/// Deliberately asserts NOTHING about the result or the cost tree. `test_gpu_full_table`
/// / `test_gpu_partitioned` already own correctness for exactly this case list (all
/// three targets `include!` the same `common/gpu_cases.inc`), and re-checking inside
/// the timing loop would put golden I/O and a full result comparison inside the region
/// being measured.
///
/// Shape of a run:
///   1. ONE `GpuExecutor` for all 11 executions. `new_mode` builds a DataFusion
///      SessionContext *and* calls `peacock_executor_create`; re-doing that per
///      iteration would benchmark executor construction.
///   2. `BENCH_WARMUP_RUNS` discarded, then `BENCH_MEASURED_RUNS` measured.
///   3. Pick the second-smallest run BY TOTAL and report THAT run's node times.
///      Taking a per-node minimum across runs instead would produce a tree whose
///      nodes belong to no single execution and can sum to less than any of them.
///
/// Per-node times come from the C++ session with [`set_node_timing`] on — see its
/// doc for why they would otherwise measure kernel submission rather than execution.
#[cfg(not(feature = "rust-only"))]
pub async fn run_gpu_benchmark(
    dataset: &str,
    sf: &str,
    query: &str,
    gpu_label: &str,
    mode: ExecMode,
) {
    use peacockdb_core::gpu_executor::{measure_timing_floor_us, set_node_timing, GpuExecutor};

    const _: () = assert!(BENCH_MEASURED_RUNS >= 2, "a second minimum needs >= 2 runs");

    // FIRST, before the executor and therefore before any cuDF allocation: rmm hands out
    // memory through whatever resource is current at the moment of the call, so a pool
    // installed after the fact would leave the early intermediates on the default one and
    // make the record's own allocator= line a half-truth.
    //
    // Called per case rather than once behind a OnceLock because the C++ side is already
    // idempotent — a second call returns the first one's outcome and builds nothing. One
    // guard, on the side that owns the resource, rather than two that can disagree.
    let allocator = install_rmm_pool();
    assert!(
        matches!(allocator, RmmPool::Pool { .. }),
        "the benchmarks measure over a pooled rmm resource, always — this run has \
         {allocator}. Refuse rather than write it: with rmm's default every cuDF \
         intermediate is a cudaMalloc/cudaFree round trip charged to the node that \
         allocated it, so the record would compare with nothing already in \
         testdata/benchmark-results/."
    );
    // Release-family rather than a profile NAME, so a plain `--release` build is not
    // rejected for the wrong reason; the name still goes into the record, because which
    // one it was is worth knowing. Checked before the runs: a debug harness measures a
    // different host overhead in total_us MINUS nodes_total_us, and learning that at write
    // time costs the whole run.
    assert!(
        env!("PEACOCK_BUILD_OPT_LEVEL") == "3" && !cfg!(debug_assertions),
        "the benchmarks are measured from a release build, always — this one is \
         {BUILD_PROFILE}. Build it with scripts/build-test-shadgpu.sh --build-benchmarks."
    );

    let data_dir = data_dir_for(dataset, sf);
    let sql_path = queries_dir_for(dataset).join(format!("{query}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));
    // Same split-and-validate as `assert_gpu_query`: the mode comes from the macro arm,
    // never from parsing the label, and a crossed pair panics here. A benchmark that
    // silently timed the other mode would produce a record that looks comparable to the
    // correctness run's plan and is not.
    let device = gpu_label_device(mode, gpu_label);
    let (partitions, budget) = device_config(&device);
    let label = golden_label(mode, &device);

    // Real per-node numbers require draining the stream at each measurement
    // boundary; without it the host clock sees submission, not execution.
    //
    // The switch is process-global, so turning it on is a loan. A guard rather than a
    // trailing call: everything below unwraps, and an unwind past a trailing call
    // would leave every later user in the process paying a stream sync per node.
    // Restored to off, the default, because the FFI exposes no way to read it back —
    // `measure_timing_floor_us` saves and restores because it runs with the switch
    // already in a known state on the C++ side.
    struct NodeTimingLoan;
    impl Drop for NodeTimingLoan {
        fn drop(&mut self) {
            peacockdb_core::gpu_executor::set_node_timing(false);
        }
    }
    set_node_timing(true);
    let _timing = NodeTimingLoan;

    let gpu = GpuExecutor::new_mode(&data_dir, partitions, budget, mode.partition_mode())
        .await
        .unwrap();

    for _ in 0..BENCH_WARMUP_RUNS {
        gpu.execute_instrumented(&sql).await.unwrap();
    }

    // AFTER the warm-up and BEFORE the measured runs, on purpose. The floor has to be
    // sampled under the same conditions the node times are: CUDA context up, modules
    // loaded, the device allocator settled. Sampling it before the warm-up would measure a
    // colder machine and understate the floor — the one direction that matters, since an
    // understated floor makes unresolvable nodes look resolved.
    let sync_floor_us = measure_timing_floor_us(BENCH_FLOOR_SAMPLES);

    let mut runs: Vec<(u64, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>)> =
        Vec::with_capacity(BENCH_MEASURED_RUNS);
    for _ in 0..BENCH_MEASURED_RUNS {
        let t0 = std::time::Instant::now();
        let (_batches, plan, stats) = gpu.execute_instrumented(&sql).await.unwrap();
        // The last node's time_us already includes its own stream drain, so the
        // walk is complete before the clock is read; `materialize` is inside.
        let total_us = t0.elapsed().as_micros() as u64;
        runs.push((total_us, plan, stats));
    }

    runs.sort_by_key(|(total_us, _, _)| *total_us);
    let (total_us, plan, stats) = &runs[1]; // second minimum by total

    let out = benchmark_result(dataset, sf, query, &label);
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    std::fs::write(
        &out,
        format!("{}\n", bench_stats_str(plan, stats, *total_us, sync_floor_us)),
    )
    .unwrap();
    eprintln!(
        "bench {dataset}/{query} [{label}]: total_us={total_us} \
         (min={} max={}) floor={sync_floor_us}us alloc=[{allocator}] -> {}",
        runs[0].0,
        runs[runs.len() - 1].0,
        out.display(),
    );
}
