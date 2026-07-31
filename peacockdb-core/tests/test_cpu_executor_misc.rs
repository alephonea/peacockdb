//! Bespoke (non-macro) CPU-executor tests. The parameterized result/cost suite
//! lives in test_cpu_executor.rs; shared helpers live in common/mod.rs.
#[macro_use]
mod common;

use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::physical_plan::ExecutionPlan;

use peacockdb_core::create_context_with_tables_mode;
use peacockdb_core::cpu_executor::{execute_node_by_node_instrumented, NodeMemoryStats};

use common::{
    all_node_names, data_dir_for, device_config, fmt_plan, has_gpu_node, make_ctx,
    partition_mode, plan_is_node13_executable, queries_dir_for, scan_batch_sizes, FULL_BUDGET,
    BATCH_STRESS_BUDGET,
};

/// Lock the routing predicate (reviewer regression guard) — the gate is AGG-KIND
/// (state-mergeability), not the mere presence of a repartition:
///   - q6 (global additive agg, no shuffle)                    → node13-executable
///   - shuffle_additive (GROUP BY + Hash shuffle, SUM/COUNT)   → node13-executable
///   - q1 (GROUP BY + Hash shuffle, AVG state = sum/count)     → node13-executable (Inc4)
///   - shuffle_stddev (GROUP BY + Hash shuffle, STDDEV)        → node13-executable (Inc5, M2)
/// All are evaluated at tp8-standard so they carry a scan map; the ONLY discriminator
/// is whether every Final-aggregate is state-mergeable. Fails LOUDLY if the predicate
/// is narrowed to reject a legitimate mergeable shuffle (sum/count/min/max/avg/stddev/var)
/// — an UNKNOWN aggregate still defaults to NON-mergeable (whitelist fail-safe).
#[tokio::test]
async fn routing_predicate_gates_on_agg_kind() {
    async fn plan_for(query: &str, device: &str) -> Arc<dyn ExecutionPlan> {
        let (parts, budget) = device_config(device);
        // Real-partitioning mode at tp8-standard so the scan carries its map (the
        // predicate keys on has_scan_map); the enum — not the budget — is the gate.
        let ctx = create_context_with_tables_mode(
            &data_dir_for("tpch", "1"),
            parts,
            budget,
            partition_mode(device),
        )
        .await
        .unwrap();
        let sql =
            std::fs::read_to_string(queries_dir_for("tpch").join(format!("{query}.sql"))).unwrap();
        ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap()
    }
    let q6 = plan_for("q6", "tp8-standard").await;
    assert!(plan_is_node13_executable(&q6), "q6 (additive global agg) must route to #13");
    let shuffle = plan_for("shuffle-additive", "tp8-standard").await;
    assert!(
        plan_is_node13_executable(&shuffle),
        "shuffle_additive (SUM/COUNT over a Hash shuffle) must route to #13"
    );
    let q1 = plan_for("q1", "tp8-standard").await;
    assert!(
        plan_is_node13_executable(&q1),
        "q1 (AVG = additive sum/count state) must route to #13 (Inc4)"
    );
    let stddev = plan_for("shuffle-stddev", "tp8-standard").await;
    assert!(
        plan_is_node13_executable(&stddev),
        "shuffle_stddev (STDDEV = Welford count/mean/m2 state) must route to #13 (Inc5)"
    );
}

/// I-3 (reviewer), INVERTED for Inc5: a STDDEV Final-agg query driven through the #13
/// CpuNodeExecutor at a real-partitioning device (tp8-standard) now MERGES CORRECTLY —
/// its Welford [count, mean, m2] state combines across the 8 hash partitions (each group
/// lands wholly in one bucket, so the per-bucket Final is exact). This asserts the #13
/// result matches DataFusion (the oracle); it panicked pre-Inc5 (non-mergeable guard),
/// so passing here proves the boundary moved. STDDEV/VAR is the last additive family;
/// an UNKNOWN aggregate still stays on #11 (whitelist fail-safe).
#[tokio::test]
async fn inc5_stddev_final_agg_at_tp8_node13_matches_datafusion() {
    // Relative tolerance 1e-12 (the q14/q39 convention): the SOLE divergence from the
    // DataFusion oracle is float summation reassociation of the Welford M2 across the 8
    // partitions (~1 ULP; observed rel diff ~3e-14). Exact-string compare can't tolerate
    // it — stddev/var results are inherently approx (unlike Inc4's exact sum/count avg).
    // gen=true: this IS the tp8-standard .result.txt generator, consumed by the
    // golden_approx gpu_test! for shuffle_stddev at tp8-standard (Inc5-iii).
    common::assert_cpu_results_match_datafusion(
        "tpch", "1", "shuffle-stddev", "tp8-standard", Some(1e-12), true, true,
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
    execute_node_by_node_instrumented(plan, ctx.task_ctx(), &mut stats).await.unwrap();

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
        execute_node_by_node_instrumented(plan_tight, ctx_tight.task_ctx(), &mut stats).await.unwrap();

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
        execute_node_by_node_instrumented(plan, ctx.task_ctx(), &mut stats).await.unwrap();

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
