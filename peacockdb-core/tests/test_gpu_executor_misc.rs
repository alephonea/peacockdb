//! Bespoke (non-macro) GPU smoke tests (scan/filter/aggregate/join/sort/lifecycle).
//! The parameterized per-query suite lives in test_gpu.rs; shared helpers
//! live in common/mod.rs. Gated off under rust-only (no GPU executor there).
#![cfg(not(feature = "rust-only"))]
#![allow(non_snake_case)] // consistent with the H200-device fn names in the macro suite
#[macro_use]
mod common;

use peacockdb_core::gpu_executor::GpuExecutor;

use common::{testdata_dir, total_rows, GPU_BUDGET};

// All-at-once GPU executor (peacock_execute): final-result-only fast path, slated for
// retirement once the node-by-node full_table/partitioned executors cover these.
// See ticket #110 (llm-wiki/tickets.md).
#[tokio::test]
async fn test_all_at_once_executor_scan_nation() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    let batches = exec.execute("SELECT * FROM nation").await.unwrap();
    println!("nation rows from GPU: {}", total_rows(&batches));
}

#[tokio::test]
async fn test_all_at_once_executor_filter_nation() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    let batches = exec
        .execute("SELECT n_name FROM nation WHERE CAST(n_nationkey AS BIGINT) > 5")
        .await
        .unwrap();
    println!("filtered nation rows from GPU: {}", total_rows(&batches));
}

#[tokio::test]
async fn test_all_at_once_executor_aggregate_count() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    // COUNT(*) alone triggers DataFusion's PlaceholderRowExec (metadata-only
    // row count, no scan). Use SUM to force a real GPU scan + aggregate.
    let batches = exec.execute("SELECT SUM(n_nationkey) FROM nation").await.unwrap();
    println!("aggregate result rows from GPU: {}", total_rows(&batches));
}

#[tokio::test]
async fn test_all_at_once_executor_join_nation_region() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    let batches = exec
        .execute(
            "SELECT n.n_name, r.r_name \
             FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey",
        )
        .await
        .unwrap();
    println!("join result rows from GPU: {}", total_rows(&batches));
}

#[tokio::test]
async fn test_all_at_once_executor_sort_nation() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    let batches = exec.execute("SELECT n_name FROM nation ORDER BY n_name ASC").await.unwrap();
    println!("sorted nation rows from GPU: {}", total_rows(&batches));
}

#[tokio::test]
async fn test_gpu_executor_create_destroy() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    drop(exec);
}
