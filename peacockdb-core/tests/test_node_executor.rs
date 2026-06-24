//! Phase-1 (Task #13) validation: the unified node-executor walk with the CPU
//! backend must produce byte-identical per-node stats AND results to the existing
//! recursive `execute_node_by_node`. Pure rust-only (no GPU toolchain).

use std::path::PathBuf;

use peacockdb_core::cpu_executor::{execute_node_by_node_instrumented, NodeMemoryStats};
use peacockdb_core::node_executor::{execute_node_by_node, CpuNodeExecutor};

fn key(stats: &[NodeMemoryStats]) -> Vec<(usize, usize)> {
    stats.iter().map(|s| (s.row_count, s.output_bytes)).collect()
}

#[tokio::test]
async fn cpu_node_executor_matches_recursive() {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal");
    let budget = 2 * 1024 * 1024 * 1024;
    // tp8 so the plan has the repartition + two-phase aggregate + SPM shape.
    let ctx = peacockdb_core::create_context_with_tables(&data, 8, budget).await.unwrap();
    let sql = "SELECT n.n_name, count(*) AS c \
               FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey \
               GROUP BY n.n_name ORDER BY n.n_name";
    let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();

    // Reference: existing recursive node-by-node executor.
    let mut ref_stats: Vec<NodeMemoryStats> = vec![];
    let ref_batches =
        execute_node_by_node_instrumented(plan.clone(), ctx.task_ctx(), &mut ref_stats)
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
