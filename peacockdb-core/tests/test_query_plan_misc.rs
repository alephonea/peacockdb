//! Bespoke (non-macro) plan/executor tests. The parameterized suite lives in
//! test_query_plan.rs; shared helpers live in common/mod.rs.
#[macro_use]
mod common;

use datafusion::arrow::array::Int64Array;

use peacockdb_core::create_context_with_tables;
use peacockdb_core::gpu_rule::{analyze_memory, row_width};
use peacockdb_core::CpuExecutor;

use common::{
    assert_plan_matches_canonical, count, find_node, scan_batch_sizes, test_ctx,
    testdata_minimal_dir, TEST_TARGET_PARTITIONS,
};

// ── CpuExecutor integration tests ────────────────────────────────────────
#[tokio::test]
async fn test_cpu_executor_simple_query() {
    let exec = CpuExecutor::new(&testdata_minimal_dir(), 1, 2 * 1024 * 1024 * 1024).await.unwrap();
    let batches = exec
        .execute("SELECT count(*) FROM nation WHERE n_regionkey >= 0")
        .await
        .unwrap();
    let count = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
    assert_eq!(count, 25);
}

#[tokio::test]
async fn test_cpu_executor_instrumented() {
    let exec = CpuExecutor::new(&testdata_minimal_dir(), 1, 2 * 1024 * 1024 * 1024).await.unwrap();
    let (batches, _plan, stats) = exec
        .execute_instrumented("SELECT count(*) FROM nation WHERE n_regionkey >= 0")
        .await
        .unwrap();
    let count = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
    assert_eq!(count, 25);
    for s in &stats {
        assert!(!s.node_name.starts_with("Gpu"), "GPU node '{}' leaked into stats", s.node_name);
    }
    assert!(!stats.is_empty());
}

// ── Basic correctness ────────────────────────────────────────────────────
#[tokio::test]
async fn test_nation_row_count() {
    let ctx = test_ctx(&testdata_minimal_dir()).await.unwrap();
    assert_eq!(count(&ctx, "SELECT count(*) FROM nation").await, 25);
}

#[tokio::test]
async fn test_region_nation_join() {
    let ctx = test_ctx(&testdata_minimal_dir()).await.unwrap();
    let n = count(
        &ctx,
        "SELECT count(*) FROM nation JOIN region ON nation.n_regionkey = region.r_regionkey",
    )
    .await;
    assert_eq!(n, 25);
}

// ── GPU plan node tests ──────────────────────────────────────────────────
#[tokio::test]
async fn test_gpu_nodes_filter_agg() {
    let ctx = test_ctx(&testdata_minimal_dir()).await.unwrap();
    let query = "SELECT count(*) FROM customer WHERE c_acctbal > 0";
    let plan = ctx.sql(query).await.unwrap().create_physical_plan().await.unwrap();
    assert_plan_matches_canonical(&plan, "filter_agg");
    let n = count(&ctx, query).await;
    assert!(n > 0 && n <= 150_000, "unexpected count {n}");
}

#[tokio::test]
async fn test_gpu_nodes_join_sort() {
    let ctx = test_ctx(&testdata_minimal_dir()).await.unwrap();
    let query = "
        SELECT n.n_name, r.r_name
        FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey
        ORDER BY n.n_name";
    let plan = ctx.sql(query).await.unwrap().create_physical_plan().await.unwrap();
    assert_plan_matches_canonical(&plan, "join_sort");
    let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 25);
}

#[tokio::test]
async fn test_gpu_nodes_group_join_sort() {
    let ctx = test_ctx(&testdata_minimal_dir()).await.unwrap();
    let query = "
        SELECT r.r_name, count(*) AS nation_count
        FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey
        GROUP BY r.r_name
        ORDER BY nation_count DESC, r.r_name";
    let plan = ctx.sql(query).await.unwrap().create_physical_plan().await.unwrap();
    assert_plan_matches_canonical(&plan, "group_join_sort");
    let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 5);
    let counts = batches[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    for i in 0..counts.len() {
        assert_eq!(counts.value(i), 5, "region {} has {} nations, expected 5", i, counts.value(i));
    }
}

// ── Memory cost tests ────────────────────────────────────────────────────
#[tokio::test]
async fn test_memory_cost_leaf_and_filter() {
    let ctx = test_ctx(&testdata_minimal_dir()).await.unwrap();

    let scan_plan = ctx
        .sql("SELECT n_nationkey FROM nation")
        .await.unwrap()
        .create_physical_plan().await.unwrap();
    let scan_node = find_node(&scan_plan, "GpuScanExec").expect("expected GpuScanExec");
    let scan_rw = row_width(&scan_node.schema());
    let scan_mem = analyze_memory(&scan_node);
    assert_eq!(scan_mem.input_row_bytes, 0, "leaf has no input");
    assert_eq!(scan_mem.output_row_bytes, scan_rw, "leaf output = row_width");
    assert_eq!(scan_mem.subtree_max_row_bytes, scan_rw, "leaf peak = row_width");

    let filter_plan = ctx
        .sql("SELECT n_nationkey FROM nation WHERE n_nationkey > 0")
        .await.unwrap()
        .create_physical_plan().await.unwrap();
    let filter_node = find_node(&filter_plan, "GpuFilterExec").expect("expected GpuFilterExec");
    let filter_rw = row_width(&filter_node.schema());
    let child_rw = row_width(&filter_node.children()[0].schema());
    let filter_mem = analyze_memory(&filter_node);
    assert_eq!(filter_mem.input_row_bytes, child_rw, "filter input = child row_width");
    assert_eq!(filter_mem.output_row_bytes, filter_rw, "filter output = row_width (sel=1.0)");
    assert!(
        filter_mem.subtree_max_row_bytes >= filter_mem.input_row_bytes + filter_mem.output_row_bytes,
        "peak ({}) must be >= input + output ({})",
        filter_mem.subtree_max_row_bytes,
        filter_mem.input_row_bytes + filter_mem.output_row_bytes,
    );
}

// ── Memory budget tests ──────────────────────────────────────────────────
#[tokio::test]
async fn test_memory_budget_reduces_batch_size() {
    let ctx = create_context_with_tables(&testdata_minimal_dir(), TEST_TARGET_PARTITIONS, 10 * 1024)
        .await
        .unwrap();
    let query = "
        SELECT n.n_name, r.r_name
        FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey
        ORDER BY n.n_name";
    let plan = ctx.sql(query).await.unwrap().create_physical_plan().await.unwrap();
    let sizes = scan_batch_sizes(&plan);
    assert!(!sizes.is_empty(), "expected GpuScanExec nodes in plan");
    for &bs in &sizes {
        assert!(bs < 8192, "expected batch_size < 8192 with 10KiB budget, got {bs}");
        assert!(bs >= 1, "batch_size must be at least 1");
    }
    let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 25);
}
