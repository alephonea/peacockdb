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
/// Not under `goldens/`: `--push-binaries` mirrors that tree to the GPU host with
/// `rsync --delete`, which for records written ON the host and travelling the other way
/// would erase them from any box that has not run the benchmarks itself.
///
/// `label` is the same `<mode>-<tp>-<tier>` the `.cpu.txt` goldens carry
/// ([`exec_mode::golden_label`]); the mode belongs in the name because the same query at
/// the same device gets a different plan, and different times, full-table vs partitioned.
pub fn benchmark_result(dataset: &str, sf: &str, query: &str, label: &str) -> PathBuf {
    testdata_root()
        .join(format!("benchmark-results/{dataset}.sf{sf}"))
        .join(format!("{query}.{label}.benchmark.txt"))
}

// --- benchmark tree ---------------------------------------------------------
/// Format per-node execution times as the same pre-order tree `cpu_stats_str` renders
/// costs into, then the totals and the `#` trailer that documents every field.
///
/// Same shape, node labels and "sub-lines only when N>1" rule as the `.cpu.txt` sibling,
/// so the two read side by side line for line; only the trailing fields differ. Rows and
/// bytes are not repeated here — they are deterministic and asserted next door, and
/// mixing them in would put reproducible and non-reproducible data in one file.
///
/// Three timing terms rather than one `time_us`, because a single number cannot be
/// fitted across the peacockdb and bare-cuDF datasets (#153): `setup_us` is a prologue
/// only peacockdb pays, so folding it in attributes decode overhead to the device and
/// makes every coefficient wrong by an amount that varies with plan shape.
///
/// `sync_floor_us` is what the measurement costs when there is nothing to measure
/// (see `measure_timing_floor_us`). It is written because every node time above
/// silently includes one, so without it a reader cannot separate "this node is cheap"
/// from "this node is under the method's resolution".
///
/// `timing_mode` is the caller's, and names the instrument the three terms came from:
/// records taken under different ones are not comparable. `build_profile` and `allocator`
/// are read here rather than passed in, because they are conditions of the run and not
/// choices of the caller: every record is written from a release build measuring over a
/// pooled rmm resource, and `run_gpu_benchmark` asserts both before it measures anything.
/// Those two lines say WHICH release profile and which pool sizes, not whether there was
/// one.
#[cfg(not(feature = "rust-only"))]
pub fn bench_stats_str(
    plan: &Arc<dyn ExecutionPlan>,
    stats: &[NodeMemoryStats],
    total_us: u64,
    timing_mode: &str,
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
            "{}{}, setup_us={} submit_us={} device_us={}",
            " ".repeat(indent),
            OneLine(node.plan.as_ref()),
            node.stat.host_setup_us,
            node.stat.host_submit_us,
            node.stat.device_us,
        ));
        // Per-partition sub-lines only when N>1, matching the .cpu.txt convention.
        // For the Hash repartition the shared concat+scatter is charged to p0, so
        // these still sum to the node line above.
        if node.stat.part_stats.len() > 1 {
            let sub = " ".repeat(indent + 2);
            for (k, ps) in node.stat.part_stats.iter().enumerate() {
                lines.push(format!(
                    "{sub}p{k}: setup_us={} submit_us={} device_us={}",
                    ps.host_setup_us, ps.host_submit_us, ps.device_us,
                ));
            }
        }
        for child in &node.children {
            walk(child, indent + 2, lines);
        }
    }
    let root = collect(plan, stats, &mut 0);
    let mut lines = Vec::new();
    walk(&root, 0, &mut lines);
    let nodes_setup_us: u64 = stats.iter().map(|s| s.host_setup_us).sum();
    let nodes_submit_us: u64 = stats.iter().map(|s| s.host_submit_us).sum();
    let nodes_device_us: u64 = stats.iter().map(|s| s.device_us).sum();
    // Host side only: the host ran while the device did, so adding nodes_device_us would
    // double-count two concurrent spans of the same clock.
    let nodes_total_us: u64 = nodes_setup_us + nodes_submit_us;
    // How much of the tree the record cannot resolve. 2/40 unresolved is a profile,
    // 35/40 is a measurement that mostly measured its own instrument.
    //
    // Scaled by partition count: sync_floor_us is ONE empty region, but a node line is Σ
    // over the node's output partitions, so a tp8 node carries eight floors. Against a
    // single floor every partitioned node would look resolved when it is not — worst on
    // exactly the widest plans.
    //
    // Against all three terms so the number stays continuous across modes.
    let at_floor = stats
        .iter()
        .filter(|s| {
            s.host_setup_us + s.host_submit_us + s.device_us
                <= sync_floor_us * s.part_stats.len().max(1) as u64
        })
        .count();
    lines.push(String::new());
    lines.push(
        "# timing_mode = how the numbers were taken, and what they MEAN. sync: the region \
         ends in a cudaStreamSynchronize, so submit_us contains the device execution and \
         device_us is 0. events: no explicit drain, and CUDA events bracket the device \
         work as device_us. submit_us is NOT launch cost in either mode — cuDF and rmm \
         synchronize internally, so the host waits for most of what it submits and \
         submit_us tracks device_us closely. Only total_us is comparable across the two."
            .to_string(),
    );
    lines.push(
        "# setup_us = host time before the node's first device touch (flatbuffer decode, \
         handle lookups, AST build) — the prologue bare cuDF has no analogue for, which is \
         why it is fitted as its own constant. submit_us / device_us: see timing_mode."
            .to_string(),
    );
    lines.push(
        "# sync_floor_us = cost of the SYNC measurement itself (empty timed region: clock + \
         cudaStreamSynchronize on an idle stream). Measured in both modes: under sync it is \
         the resolution floor every node time INCLUDES, once PER OUTPUT PARTITION; under \
         events it is the per-region stall that is no longer paid. Do not subtract it."
            .to_string(),
    );
    lines.push(
        "# nodes_total_us = Σ setup_us + Σ submit_us, the HOST side of the walk. \
         nodes_device_us is deliberately NOT added to it — the host ran while the device \
         did, so the two are concurrent spans of the same clock. Device regions are on \
         one stream in program order, so nodes_device_us is disjoint and <= total_us; \
         the gap is stream idle, i.e. the device waiting through host prologue. \
         total_us = the whole query, end to end (parse + plan + serialize + node walk + \
         materialize)."
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
    // Written even for trees with no repartition: emitted only where it applies, its
    // absence would mean either "no scatter here" or "this record predates the line" —
    // the same ambiguity the disclosure exists to remove.
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
    lines.push(format!("timing_mode={timing_mode}"));
    lines.push(format!("sync_floor_us={sync_floor_us}"));
    lines.push(format!("nodes_at_or_below_floor={at_floor}/{}", stats.len()));
    lines.push(format!("nodes_setup_us={nodes_setup_us}"));
    lines.push(format!("nodes_submit_us={nodes_submit_us}"));
    lines.push(format!("nodes_device_us={nodes_device_us}"));
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

/// Discarded runs. The first execution pays for the page cache, CUDA module load and JIT,
/// and allocator growth — the host's recent history rather than the plan. The pool
/// removes most of the third before this runs, which is a reason to keep the warm-up:
/// what is left is the part that varies.
#[cfg(not(feature = "rust-only"))]
pub const BENCH_WARMUP_RUNS: usize = 1;

/// Measured runs per query. The reported tree is the second-smallest by end-to-end time:
/// the minimum is the run most likely to have caught a favourable scheduling accident,
/// the rest are dragged up by whatever else the shared host was doing. Must be >= 2.
#[cfg(not(feature = "rust-only"))]
pub const BENCH_MEASURED_RUNS: usize = 10;

/// Empty timed regions sampled for the resolution floor. Each is one
/// `cudaStreamSynchronize` on an idle stream, so the whole sample costs ~1ms next to a
/// query — cheap for the number that decides whether a small node time means anything.
#[cfg(not(feature = "rust-only"))]
pub const BENCH_FLOOR_SAMPLES: u32 = 200;

/// Time one case from `common/gpu_cases.inc` and write `testdata/benchmark-results/…`.
///
/// Asserts nothing: `test_gpu_full_table` / `test_gpu_partitioned` already own
/// correctness for this exact case list (all three `include!` `common/gpu_cases.inc`),
/// and re-checking would put golden I/O and a result comparison inside the timed region.
///
/// Shape of a run:
///   1. One `GpuExecutor` for all executions — `new_mode` builds a SessionContext *and*
///      calls `peacock_executor_create`, so rebuilding it per iteration would benchmark
///      executor construction.
///   2. `BENCH_WARMUP_RUNS` discarded, then `BENCH_MEASURED_RUNS` measured.
///   3. Second-smallest run by total, reporting that run's node times. A per-node minimum
///      across runs would give a tree belonging to no single execution, which can sum to
///      less than any of them.
///
/// Times come from the C++ session under [`NodeTiming::Events`]. `Sync` drains the stream
/// at every region boundary, measuring a schedule the shipping engine never runs; it is
/// still sampled once as `sync_floor_us`, so a record says what events bought.
#[cfg(not(feature = "rust-only"))]
pub async fn run_gpu_benchmark(
    dataset: &str,
    sf: &str,
    query: &str,
    gpu_label: &str,
    mode: ExecMode,
) {
    use peacockdb_core::gpu_executor::{
        measure_timing_floor_us, set_node_timing, GpuExecutor, NodeTiming,
    };

    const _: () = assert!(BENCH_MEASURED_RUNS >= 2, "a second minimum needs >= 2 runs");

    // Before the executor, so before any cuDF allocation: rmm uses whatever resource is
    // current at the moment of the call, and a pool installed later would leave the early
    // intermediates on the default one, making the record's own allocator= a half-truth.
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
    // never from the label, and a crossed pair panics here rather than producing a record
    // that looks comparable to the correctness run's plan and is not.
    let device = gpu_label_device(mode, gpu_label);
    let (partitions, budget) = device_config(&device);
    let label = golden_label(mode, &device);

    // The switch is process-global. A guard rather than a trailing reset: everything
    // below unwraps, and an unwind would leave every later user in the process still
    // allocating an event pair per node. Restored to Off rather than to its previous
    // value because the FFI exposes no way to read it back.
    const TIMING_MODE: &str = "events";
    struct NodeTimingLoan;
    impl Drop for NodeTimingLoan {
        fn drop(&mut self) {
            peacockdb_core::gpu_executor::set_node_timing(NodeTiming::Off);
        }
    }
    set_node_timing(NodeTiming::Events);
    let _timing = NodeTimingLoan;

    // NVTX ranges for an Nsight capture, env-gated like the calibration record: nothing
    // reads them without a profiler attached, and the committed tree stays measured
    // without them. No loan, unlike the timing switch above: the env var scopes this to
    // a whole capture process, where every later case wants the ranges too.
    if std::env::var_os("PEACOCK_NVTX").is_some() {
        peacockdb_core::gpu_executor::set_nvtx_ranges(true);
    }

    let gpu = GpuExecutor::new_mode(&data_dir, partitions, budget, mode.partition_mode())
        .await
        .unwrap();

    for _ in 0..BENCH_WARMUP_RUNS {
        gpu.execute_instrumented(&sql).await.unwrap();
    }

    // After the warm-up, so the floor is sampled under the same conditions as the node
    // times: context up, modules loaded, allocator settled. Sampling it cold would
    // understate the floor — the direction that matters, since it makes unresolvable
    // nodes look resolved.
    let sync_floor_us = measure_timing_floor_us(BENCH_FLOOR_SAMPLES);

    let mut runs: Vec<(u64, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>)> =
        Vec::with_capacity(BENCH_MEASURED_RUNS);
    for _ in 0..BENCH_MEASURED_RUNS {
        let t0 = std::time::Instant::now();
        let (_batches, plan, stats) = gpu.execute_instrumented(&sql).await.unwrap();
        // `materialize` is inside, and it copies the root off the device — so the
        // clock is read after the device has actually finished, in either mode. Under
        // Events nothing else would guarantee that: the node walk returns while the
        // stream may still be running.
        let total_us = t0.elapsed().as_micros() as u64;
        runs.push((total_us, plan, stats));
    }

    runs.sort_by_key(|(total_us, _, _)| *total_us);
    let (total_us, plan, stats) = &runs[1]; // second minimum by total

    let out = benchmark_result(dataset, sf, query, &label);
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    std::fs::write(
        &out,
        format!(
            "{}\n",
            bench_stats_str(plan, stats, *total_us, TIMING_MODE, sync_floor_us)
        ),
    )
    .unwrap();
    // The same run, as calibration rows (#153). Off unless PEACOCK_RECORD_PATH is
    // set: the record is for a collection run, not for the committed .benchmark.txt,
    // and the two must not start depending on each other. Sourced from the same
    // second-minimum run the record above reports, so the two files never disagree.
    super::record::append_records(
        plan,
        stats,
        &super::record::RunMeta {
            source: "peacockdb",
            dataset,
            sf,
            query,
            label: &label,
            timing_mode: TIMING_MODE,
            build_profile: BUILD_PROFILE,
            allocator: &allocator.to_string(),
        },
    );

    eprintln!(
        "bench {dataset}/{query} [{label}]: total_us={total_us} \
         (min={} max={}) floor={sync_floor_us}us alloc=[{allocator}] -> {}",
        runs[0].0,
        runs[runs.len() - 1].0,
        out.display(),
    );
}
