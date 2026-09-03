//! Bespoke (non-macro) CPU-executor tests. The parameterized result/cost suite
//! lives in test_cpu_full_table.rs / test_cpu_partitioned.rs; shared helpers live
//! in common/mod.rs.
#[macro_use]
mod common;

use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::physical_plan::ExecutionPlan;

use peacockdb_core::create_context_with_tables_mode;
use peacockdb_core::cpu_executor::NodeMemoryStats;
use peacockdb_core::executors::full_table_cpu_executor::execute_full_table_instrumented;

use common::exec_mode::{CpuOracle, ExecMode, ResultGolden};
use common::{
    all_node_names, data_dir_for, device_config, fmt_plan, has_gpu_node, make_ctx,
    plan_is_partitioned_executable, queries_dir_for, scan_batch_sizes, FULL_BUDGET,
    BATCH_STRESS_BUDGET,
};

/// Lock the routing predicate — the gate is AGG-KIND (state-mergeability), not the
/// mere presence of a repartition:
///   - q6 (global additive agg, no shuffle)                    → partitioned-executable
///   - shuffle_additive (GROUP BY + Hash shuffle, SUM/COUNT)   → partitioned-executable
///   - q1 (GROUP BY + Hash shuffle, AVG state = sum/count)     → partitioned-executable
///   - shuffle_stddev (GROUP BY + Hash shuffle, STDDEV M2)     → partitioned-executable
/// All are evaluated at tp8-standard so they carry a scan map; the ONLY discriminator
/// is whether every Final-aggregate is state-mergeable. Fails LOUDLY if the predicate
/// is narrowed to reject a legitimate mergeable shuffle (sum/count/min/max/avg/stddev/var)
/// — an UNKNOWN aggregate still defaults to NON-mergeable (whitelist fail-safe).
#[tokio::test]
async fn routing_predicate_gates_on_agg_kind() {
    async fn plan_for(query: &str, device: &str) -> Arc<dyn ExecutionPlan> {
        let (parts, budget) = device_config(device);
        // Real-partitioning mode so the scan carries its map (the predicate keys on
        // has_scan_map); the enum — not the budget — is the gate. Stated here rather
        // than derived from `device`: this builds an EXECUTION context.
        let ctx = create_context_with_tables_mode(
            &data_dir_for("tpch", "1"),
            parts,
            budget,
            ExecMode::Partitioned.partition_mode(),
        )
        .await
        .unwrap();
        let sql =
            std::fs::read_to_string(queries_dir_for("tpch").join(format!("{query}.sql"))).unwrap();
        ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap()
    }
    let q6 = plan_for("q6", "tp8-standard").await;
    assert!(plan_is_partitioned_executable(&q6), "q6 (additive global agg) must route to the partitioned executor");
    let shuffle = plan_for("shuffle-additive", "tp8-standard").await;
    assert!(
        plan_is_partitioned_executable(&shuffle),
        "shuffle_additive (SUM/COUNT over a Hash shuffle) must route to the partitioned executor"
    );
    let q1 = plan_for("q1", "tp8-standard").await;
    assert!(
        plan_is_partitioned_executable(&q1),
        "q1 (AVG = additive sum/count state) must route to the partitioned executor (Inc4)"
    );
    let stddev = plan_for("shuffle-stddev", "tp8-standard").await;
    assert!(
        plan_is_partitioned_executable(&stddev),
        "shuffle_stddev (STDDEV = Welford count/mean/m2 state) must route to the partitioned executor (Inc5)"
    );
}

/// A STDDEV Final-agg query driven through the partitioned CpuNodeExecutor at a
/// real-partitioning device (tp8-standard) MERGES CORRECTLY — its Welford
/// [count, mean, m2] state combines across the 8 hash partitions (each group lands
/// wholly in one bucket, so the per-bucket Final is exact). Asserts the #13 result
/// matches DataFusion (the oracle). An UNKNOWN aggregate still stays full-table
/// (whitelist fail-safe).
#[tokio::test]
async fn inc5_stddev_final_agg_at_tp8_partitioned_matches_datafusion() {
    // DataFusionApproximate (see CpuOracle for the general rationale): here the
    // reassociated sum is the Welford M2 across the 8 partitions — observed rel diff
    // ~3e-14, so exact-string compare cannot be used.
    // Write, not Skip: the golden_approx_std consumer at
    // shuffle-stddev/partitioned-tp8-standard is enabled again (#103 was an
    // uninitialized OutBuild field, not the merge), so this golden is read rather than
    // orphaned — which is the thing the `ResultGolden` gate is there to prevent.
    common::assert_cpu_results_match_datafusion(
        "tpch",
        "1",
        "shuffle-stddev",
        "tp8-standard",
        CpuOracle::DataFusionApproximate,
        ExecMode::Partitioned,
        ResultGolden::Write,
    )
    .await;
}

/// Every Gpu* node must be stripped before CPU execution; no stat may name one.
#[tokio::test]
async fn test_execution_strips_gpu_nodes() {
    let ctx = make_ctx(FULL_BUDGET).await;
    let plan = ctx
        .sql("SELECT count(*) FROM nation WHERE n_regionkey >= 0")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    assert!(has_gpu_node(&plan), "expected GPU nodes in plan, got: {:?}", all_node_names(&plan));

    let mut stats: Vec<NodeMemoryStats> = vec![];
    execute_full_table_instrumented(plan, ctx.task_ctx(), &mut stats).await.unwrap();

    assert!(!stats.is_empty(), "no nodes were executed");
    let gpu_names: Vec<&str> =
        stats.iter().filter(|s| s.node_name.starts_with("Gpu")).map(|s| s.node_name.as_str()).collect();
    assert!(gpu_names.is_empty(), "GPU nodes not stripped: {gpu_names:?}");
}

#[tokio::test]
async fn test_memory_boundary_preserved_tight_budget() {
    let query = "SELECT count(*) FROM customer WHERE c_custkey > 0";

    let ctx_full = make_ctx(FULL_BUDGET).await;
    let plan_full = ctx_full.sql(query).await.unwrap().create_physical_plan().await.unwrap();

    let ctx_tight = make_ctx(BATCH_STRESS_BUDGET).await;
    let plan_tight = ctx_tight.sql(query).await.unwrap().create_physical_plan().await.unwrap();

    eprintln!("\n=== FULL BUDGET ({} GiB) plan ===\n{}", FULL_BUDGET / (1024 * 1024 * 1024), fmt_plan(&plan_full));
    eprintln!("=== TIGHT BUDGET ({} KiB) plan ===\n{}", BATCH_STRESS_BUDGET / 1024, fmt_plan(&plan_tight));

    let tight_scan_sizes = scan_batch_sizes(&plan_tight);
    assert!(
        !tight_scan_sizes.is_empty(),
        "expected GpuScanExec in tight plan; node names: {:?}",
        all_node_names(&plan_tight)
    );
    let gpu_batch_size = *tight_scan_sizes.iter().max().unwrap();

    let full_scan_sizes = scan_batch_sizes(&plan_full);
    let full_batch_size = *full_scan_sizes.iter().max().unwrap();

    eprintln!("GpuScanExec batch_size — full budget: {full_batch_size}, tight budget: {gpu_batch_size}");
    assert!(
        gpu_batch_size < full_batch_size,
        "tight budget batch_size ({gpu_batch_size}) should be smaller than full budget ({full_batch_size})"
    );

    let mut stats: Vec<NodeMemoryStats> = vec![];
    let batches =
        execute_full_table_instrumented(plan_tight, ctx_tight.task_ctx(), &mut stats).await.unwrap();

    let count = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
    assert_eq!(count, 150_000, "customer table must have 150 000 rows");

    let scan_stats: Vec<&NodeMemoryStats> =
        stats.iter().filter(|s| s.node_name == "ParquetExec").collect();
    assert!(!scan_stats.is_empty(), "expected ParquetExec in stats");

    eprintln!("Per-node stats (post-order):");
    for s in &stats {
        eprintln!(
            "  {}: rows={}, max_batch={}, alloc={}B, out={}B",
            s.node_name, s.row_count, s.max_batch_rows, s.allocated_bytes, s.output_bytes
        );
    }

    for s in &scan_stats {
        assert!(
            s.max_batch_rows <= gpu_batch_size,
            "ParquetExec batch {} rows exceeds gpu_batch_size={}",
            s.max_batch_rows,
            gpu_batch_size
        );
    }

    let gpu_names: Vec<&str> =
        stats.iter().filter(|s| s.node_name.starts_with("Gpu")).map(|s| s.node_name.as_str()).collect();
    assert!(gpu_names.is_empty(), "GPU nodes in stats: {gpu_names:?}");
}

#[tokio::test]
async fn test_instrumented_stats_are_populated() {
    let ctx = make_ctx(FULL_BUDGET).await;
    let plan = ctx
        .sql("SELECT n_name, n_regionkey FROM nation WHERE n_regionkey = 1")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    let mut stats: Vec<NodeMemoryStats> = vec![];
    let batches =
        execute_full_table_instrumented(plan, ctx.task_ctx(), &mut stats).await.unwrap();

    let final_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let root_stat = stats.last().unwrap();
    assert_eq!(root_stat.row_count, final_rows, "root node row_count in stats does not match actual output");
    assert!(root_stat.allocated_bytes > 0, "root node allocated_bytes should be > 0");
    assert!(root_stat.output_bytes > 0, "root node output_bytes should be > 0");
    assert!(
        root_stat.allocated_bytes >= root_stat.output_bytes,
        "allocated_bytes ({}) must be >= output_bytes ({})",
        root_stat.allocated_bytes,
        root_stat.output_bytes
    );
}

// --- the calibration record -------------------------------------------------


/// The benchmark path must not touch the CPU tier — neither what it PRODUCES nor what it
/// RUNS.
///
/// Two different prohibitions, and the second is the one this list used to miss.
///
/// **Its goldens.** The measurement suite runs at sf40, where there are no `.cpu.txt`
/// statistics and no `.result.txt` results, and where producing them would mean executing
/// the query on the CPU over 42 GB of parquet. Reading one makes every case fail on a
/// missing file, on the GPU host, long after the change.
///
/// **Its execution.** `CpuBackend` is a working backend: a case that drove it would not
/// fail at all. It would produce a whole tree of plausible microseconds measured on the
/// wrong machine, and nothing downstream — not the record, not the fit, not a plot —
/// carries a field that says which backend ran. That is worse than a missing file, and
/// it is why the two names sit in one list.
///
/// Deliberately NOT `CpuBatch`: it lives on the GPU path legitimately, because an unload
/// hands back a host batch. Forbidding it would break correct code and teach the next
/// person to edit this list instead of their change.
///
/// Checked here, in the CPU tier, on a machine with no GPU, at the moment the call is
/// added — rather than as a device failure hours later.
///
/// By source text rather than by types because the thing being forbidden is a CALL, and
/// no type says "this function was not called". Line comments are stripped first: the
/// point is calls, and a comment explaining why the benchmark does not read a golden must
/// not itself be the failure. Two known limits — a `//` inside a string literal hides the
/// rest of that line from the search, and a rename of any of these names silently empties
/// the check. It is a tripwire, not a proof.
#[test]
fn the_benchmark_path_reads_no_cpu_side_golden() {
    // Every accessor in common/ that names a file the CPU tier writes, the oracle
    // comparison mode, and the CPU backend itself. The first five are about READING what
    // the CPU produced; the last is about RUNNING on it, which fails in the opposite way
    // — silently, with numbers.
    const FORBIDDEN: &[&str] = &[
        "cpu_golden",
        "result_golden",
        "cost_golden",
        "plan_golden",
        "CpuOracle",
        "CpuBackend",
    ];
    // The files the benchmark path is its own. `corpus.rs` is not among them: the path
    // borrows `plan_at` from it, and the rest of that file is the corpus tier, which reads
    // goldens for a living. A whole-file check cannot separate the two.
    const SOURCES: &[&str] = &[
        "tests/peacock_gpu_benchmarks.rs",
        "tests/common/corpus_benchmark.rs",
        "tests/common/gpu_session.rs",
    ];

    for rel in SOURCES {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "cannot read {}: {e}. This guard names the benchmark path by path; if \
                 the file moved, point it at the new one rather than dropping it.",
                path.display()
            )
        });
        let code: String = text
            .lines()
            .map(|l| l.split_once("//").map_or(l, |(before, _)| before))
            .collect::<Vec<_>>()
            .join("\n");
        for name in FORBIDDEN {
            assert!(
                !code.contains(name),
                "{rel} mentions `{name}`. The benchmark suite runs at sf40 and must not \
                 touch the CPU tier. A golden it reads does not exist there and cannot \
                 be produced, so every case fails on a missing file — on the GPU host, \
                 long after the change. `CpuBackend` is worse: it WORKS, and the run \
                 would report a tree of plausible microseconds measured on the wrong \
                 machine, with no column anywhere saying which backend produced them. \
                 If the benchmark genuinely needs a CPU-side input, that is a decision \
                 about the sf40 suite, not an edit to this list."
            );
        }
    }
}
