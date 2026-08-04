//! #13 validation: the unified node-executor walk with the CPU
//! backend must produce byte-identical per-node stats AND results to the existing
//! recursive `execute_node_by_node`. Pure rust-only (no GPU toolchain).

use std::path::PathBuf;

use peacockdb_core::cpu_executor::NodeMemoryStats;
use peacockdb_core::executors::full_table_cpu_executor::execute_full_table_instrumented;
use peacockdb_core::node_executor::{execute_node_by_node, CpuNodeExecutor};

fn key(stats: &[NodeMemoryStats]) -> Vec<(usize, usize)> {
    stats.iter().map(|s| (s.row_count, s.output_bytes)).collect()
}

#[tokio::test]
async fn cpu_node_executor_matches_recursive() {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal");
    // SinglePartition: tpch.minimal is a single row group, so the scan can't split
    // N-way (empty map) — this test validates that the #13 multi-handle walk equals
    // the recursive executor for the single-partition case (its true invariant; #13
    // Σ-over-partitions intentionally DIVERGES from the coalesced recursive baseline
    // once a real N-way shuffle exists, so a RealMultiPartition query can't be
    // compared here). tp8 so the plan still has the two-phase-agg + repartition shape;
    // the un-lowered Hash repartition falls through to the coalesced path on both
    // executors. Real N-way #13-vs-GPU is covered by the shuffle_additive goldens.
    let budget = 120 * 1024 * 1024 * 1024;
    let ctx = peacockdb_core::create_context_with_tables(&data, 8, budget).await.unwrap();
    let sql = "SELECT n.n_name, count(*) AS c \
               FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey \
               GROUP BY n.n_name ORDER BY n.n_name";
    let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();

    // Reference: existing recursive node-by-node executor.
    let mut ref_stats: Vec<NodeMemoryStats> = vec![];
    let ref_batches =
        execute_full_table_instrumented(plan.clone(), ctx.task_ctx(), &mut ref_stats)
            .await
            .unwrap();

    // Unified walk with the CPU backend.
    let mut backend = CpuNodeExecutor::new(ctx.task_ctx());
    let (walk_batches, walk_stats) = execute_node_by_node(&plan, &mut backend).await.unwrap();

    assert_eq!(
        key(&ref_stats),
        key(&walk_stats),
        "per-node (rows, output_bytes) must match the recursive executor"
    );

    let to_str = |b: &[datafusion::arrow::record_batch::RecordBatch]| {
        datafusion::arrow::util::pretty::pretty_format_batches(b).unwrap().to_string()
    };
    assert_eq!(to_str(&ref_batches), to_str(&walk_batches), "results must match");
}
