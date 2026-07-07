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
    TIGHT_BUDGET,
};

/// Lock the routing predicate (reviewer regression guard) — the gate is AGG-KIND
/// (state-mergeability), not the mere presence of a repartition:
///   - q6 (global additive agg, no shuffle)                    → node13-executable
///   - shuffle_additive (GROUP BY + Hash shuffle, SUM/COUNT)   → node13-executable
///   - q1 (GROUP BY + Hash shuffle, AVG state = sum/count)     → node13-executable (Inc4)
///   - shuffle_stddev (GROUP BY + Hash shuffle, STDDEV)        → NOT (compound moments = Inc5)
/// All are evaluated at tp8-mem120gib so they carry a scan map; the ONLY discriminator
/// is whether every Final-aggregate is state-mergeable. Fails LOUDLY if the predicate
/// is broadened to admit STDDEV/VAR (which #13 can't merge yet) or narrowed to reject
/// a legitimate mergeable shuffle (sum/count/min/max/avg).
#[tokio::test]
async fn routing_predicate_gates_on_agg_kind() {
    async fn plan_for(query: &str, device: &str) -> Arc<dyn ExecutionPlan> {
        let (parts, budget) = device_config(device);
        // Real-partitioning mode at tp8-mem120gib so the scan carries its map (the
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
    let q6 = plan_for("q6", "tp8-mem120gib").await;
    assert!(plan_is_node13_executable(&q6), "q6 (additive global agg) must route to #13");
    let shuffle = plan_for("shuffle-additive", "tp8-mem120gib").await;
    assert!(
        plan_is_node13_executable(&shuffle),
        "shuffle_additive (SUM/COUNT over a Hash shuffle) must route to #13"
    );
    let q1 = plan_for("q1", "tp8-mem120gib").await;
    assert!(
        plan_is_node13_executable(&q1),
        "q1 (AVG = additive sum/count state) must route to #13 (Inc4)"
    );
    let stddev = plan_for("shuffle-stddev", "tp8-mem120gib").await;
    assert!(
        !plan_is_node13_executable(&stddev),
        "shuffle_stddev (STDDEV compound moments) must stay on #11 until Inc5"
    );
}

/// I-3 (reviewer): LOCK the Inc5 boundary. Forcing a STDDEV Final-agg query through
/// the #13 CpuNodeExecutor at a real-partitioning device (tp8-mem120gib) must PANIC on
/// the node13-executable safety guard — #13 cannot merge STDDEV/VAR compound-moment
/// state across the 8 hash partitions until Inc5. Guards against a future relaxation
/// silently admitting a non-mergeable Final-agg and mis-merging it. (AVG now PASSES —
/// Inc4 made its sum/count state mergeable; the negative case moved to STDDEV.)
#[tokio::test]
#[should_panic(expected = "node13-executable")]
async fn inc5_stddev_final_agg_at_tp8_node13_panics() {
    common::assert_cpu_results_match_datafusion(
        "tpch", "1", "shuffle-stddev", "tp8-mem120gib", None, true, false,
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

    let ctx_tight = make_ctx(TIGHT_BUDGET).await;
    let plan_tight = ctx_tight.sql(query).await.unwrap().create_physical_plan().await.unwrap();

    eprintln!("\n=== FULL BUDGET ({} GiB) plan ===\n{}", FULL_BUDGET / (1024 * 1024 * 1024), fmt_plan(&plan_full));
    eprintln!("=== TIGHT BUDGET ({} KiB) plan ===\n{}", TIGHT_BUDGET / 1024, fmt_plan(&plan_tight));

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
