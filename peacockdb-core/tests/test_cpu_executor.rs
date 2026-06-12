use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::physical_plan::ExecutionPlan;

use peacockdb_core::cpu_executor::{
    execute_node_by_node_instrumented, NodeMemoryStats,
};
use peacockdb_core::{create_context_with_tables, build_session_state, register_tables_for};

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.sf1")
}

fn queries_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch-queries")
}

fn tpcds_testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpcds.sf1")
}

fn tpcds_queries_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpcds-queries")
}

fn canondata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/plans.sf1")
}

fn tpcds_canondata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/plans-tpcds.sf1")
}

fn has_gpu_node(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.name().starts_with("Gpu") || plan.children().iter().any(|c| has_gpu_node(c))
}

fn all_node_names(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
    let mut names = vec![plan.name().to_string()];
    for child in plan.children() {
        names.extend(all_node_names(child));
    }
    names
}

fn scan_batch_sizes(plan: &Arc<dyn ExecutionPlan>) -> Vec<usize> {
    use peacockdb_core::gpu_rule::GpuScanExec;
    let mut sizes = vec![];
    if let Some(scan) = plan.as_any().downcast_ref::<GpuScanExec>() {
        sizes.push(scan.gpu_batch_size);
    }
    for child in plan.children() {
        sizes.extend(scan_batch_sizes(child));
    }
    sizes
}

fn fmt_plan(plan: &Arc<dyn ExecutionPlan>) -> String {
    use datafusion::physical_plan::display::DisplayableExecutionPlan;
    DisplayableExecutionPlan::new(plan.as_ref())
        .indent(true)
        .to_string()
}

async fn make_ctx(budget: usize) -> datafusion::execution::context::SessionContext {
    // 1 partition on purpose: the per-node cost canon (`.cpu.txt`) records
    // output_bytes, which is batch-boundary sensitive and only reproducible
    // single-stream. Multi-partition determinism is tracked in #53.
    create_context_with_tables(&testdata_dir(), 1, budget)
        .await
        .unwrap()
}

const FULL_BUDGET: usize = 2 * 1024 * 1024 * 1024;
const TIGHT_BUDGET: usize = 10 * 1024;

// 
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

    assert!(
        has_gpu_node(&plan),
        "expected GPU nodes in plan, got: {:?}",
        all_node_names(&plan)
    );

    let mut stats: Vec<NodeMemoryStats> = vec![];
    execute_node_by_node_instrumented(plan, ctx.task_ctx(), &mut stats)
        .await
        .unwrap();

    assert!(!stats.is_empty(), "no nodes were executed");
    let gpu_names: Vec<&str> = stats
        .iter()
        .filter(|s| s.node_name.starts_with("Gpu"))
        .map(|s| s.node_name.as_str())
        .collect();
    assert!(gpu_names.is_empty(), "GPU nodes not stripped: {gpu_names:?}");
}

/// Render RecordBatches as a pretty table and sort the data rows so that
/// result comparison is order-independent (queries without ORDER BY may
/// return rows in any order depending on the executor path).
fn batches_to_sorted_str(batches: &[RecordBatch]) -> String {
    let formatted = pretty_format_batches(batches).unwrap().to_string();
    let lines: Vec<&str> = formatted.lines().collect();
    // Layout: border / header / border / ...data rows... / border
    if lines.len() > 4 {
        let mut data = lines[3..lines.len() - 1].to_vec();
        data.sort_unstable();
        let mut out = lines[..3].to_vec();
        out.extend(data);
        out.push(lines[lines.len() - 1]);
        out.join("\n")
    } else {
        formatted
    }
}

/// Format per-node CPU execution stats as a pre-order tree. Stats are collected
/// post-order during execution; the plan tree supplies the indentation.
fn cpu_stats_str(plan: &Arc<dyn ExecutionPlan>, stats: &[NodeMemoryStats]) -> String {
    struct Node<'a> {
        stat: &'a NodeMemoryStats,
        children: Vec<Node<'a>>,
    }

    fn collect<'a>(plan: &Arc<dyn ExecutionPlan>, stats: &'a [NodeMemoryStats], idx: &mut usize) -> Node<'a> {
        let children: Vec<Node<'a>> = plan.children().iter()
            .map(|c| collect(c, stats, idx))
            .collect();
        let stat = &stats[*idx];
        *idx += 1;
        Node { stat, children }
    }

    fn walk(node: &Node, indent: usize, lines: &mut Vec<String>) {
        lines.push(format!(
            "{}{}: output_bytes={}",
            " ".repeat(indent),
            node.stat.node_name,
            node.stat.output_bytes,
        ));
        for child in &node.children {
            walk(child, indent + 2, lines);
        }
    }

    let root = collect(plan, stats, &mut 0);
    let mut lines = Vec::new();
    walk(&root, 0, &mut lines);
    lines.join("\n")
}

/// Compare the per-node CPU cost tree to the `.cpu.txt` canonical in `dir`, or
/// regenerate it when `UPDATE_CANONICAL` is set.
fn assert_cpu_cost_canonical(
    plan: &Arc<dyn ExecutionPlan>,
    stats: &[NodeMemoryStats],
    name: &str,
    dir: &std::path::Path,
) {
    let canonical_path = dir.join(format!("{name}.cpu.txt"));
    let actual = cpu_stats_str(plan, stats);

    if std::env::var("UPDATE_CANONICAL").is_ok() {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(&canonical_path, &actual).unwrap();
        eprintln!("Updated CPU canonical: {}", canonical_path.display());
        return;
    }

    let canonical = std::fs::read_to_string(&canonical_path).unwrap_or_else(|_| {
        panic!(
            "CPU canonical file not found: {}\nRun with UPDATE_CANONICAL=1 to generate it.",
            canonical_path.display()
        )
    });
    assert_eq!(
        actual,
        canonical.trim_end(),
        "CPU cost tree for '{name}' does not match {}",
        canonical_path.display()
    );
}

/// Run `name.sql` through both plain DataFusion and the CPU executor, then
/// assert that the result sets are equal (order-independent) and that the
/// per-node CPU cost tree matches the `.cpu.txt` canonical.
async fn assert_cpu_results_match_datafusion(name: &str) {
    let data_dir = testdata_dir();

    let sql_path = queries_dir().join(format!("{name}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));
    let mut df_ctx = build_session_state(1);
    df_ctx = register_tables_for(df_ctx, &data_dir).await.unwrap();
    // Ground truth: plain DataFusion without GPU rules.
    let expected = df_ctx.sql(&sql).await.unwrap().collect().await.unwrap();

    // CPU executor: GPU-annotated plan executed node-by-node on CPU, capturing
    // per-node memory stats for cost canonization.
    let cpu_ctx = make_ctx(FULL_BUDGET).await;
    let plan = cpu_ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();
    let mut stats: Vec<NodeMemoryStats> = vec![];
    let actual = execute_node_by_node_instrumented(plan.clone(), cpu_ctx.task_ctx(), &mut stats)
        .await
        .unwrap();

    assert_eq!(
        batches_to_sorted_str(&actual),
        batches_to_sorted_str(&expected),
        "CPU executor result for '{name}' differs from plain DataFusion"
    );

    assert_cpu_cost_canonical(&plan, &stats, name, &canondata_dir());
}

macro_rules! cpu_result_test {
    ($func_name:ident, $query_name:literal) => {
        #[tokio::test]
        async fn $func_name() {
            assert_cpu_results_match_datafusion($query_name).await;
        }
    };
}

cpu_result_test!(test_cpu_hash_join, "hash-join");
cpu_result_test!(test_cpu_left_join, "left-join");
cpu_result_test!(test_cpu_mixed_join, "mixed-join");

cpu_result_test!(test_cpu_scan_limit, "scan-limit");
cpu_result_test!(test_cpu_filter_project, "filter-project");
cpu_result_test!(test_cpu_aggregate_groupby, "aggregate-groupby");
cpu_result_test!(test_cpu_semi_join, "semi-join");
cpu_result_test!(test_cpu_anti_join, "anti-join");
cpu_result_test!(test_cpu_nested_loop_join, "nested-loop-join");
cpu_result_test!(test_cpu_cross_join, "cross-join");
cpu_result_test!(test_cpu_tpch_q1, "q1");
cpu_result_test!(test_cpu_tpch_q2, "q2");
cpu_result_test!(test_cpu_tpch_q3, "q3");
cpu_result_test!(test_cpu_tpch_q4, "q4");
cpu_result_test!(test_cpu_tpch_q5, "q5");
cpu_result_test!(test_cpu_tpch_q6, "q6");
cpu_result_test!(test_cpu_tpch_q7, "q7");
cpu_result_test!(test_cpu_tpch_q8, "q8");
cpu_result_test!(test_cpu_tpch_q9, "q9");
cpu_result_test!(test_cpu_tpch_q10, "q10");
cpu_result_test!(test_cpu_tpch_q11, "q11");
cpu_result_test!(test_cpu_tpch_q12, "q12");
cpu_result_test!(test_cpu_tpch_q13, "q13");
cpu_result_test!(test_cpu_tpch_q14, "q14");
cpu_result_test!(test_cpu_tpch_q15, "q15");  // view inlined as a CTE (see q15.sql)
cpu_result_test!(test_cpu_tpch_q16, "q16");
cpu_result_test!(test_cpu_tpch_q17, "q17");
cpu_result_test!(test_cpu_tpch_q18, "q18");
cpu_result_test!(test_cpu_tpch_q19, "q19");
cpu_result_test!(test_cpu_tpch_q20, "q20");
cpu_result_test!(test_cpu_tpch_q21, "q21");
cpu_result_test!(test_cpu_tpch_q22, "q22");

async fn assert_cpu_results_match_datafusion_tpcds(name: &str) {
    let data_dir = tpcds_testdata_dir();

    let sql_path = tpcds_queries_dir().join(format!("{name}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));
    let mut df_ctx = build_session_state(1);
    df_ctx = register_tables_for(df_ctx, &data_dir).await.unwrap();
    let expected = df_ctx.sql(&sql).await.unwrap().collect().await.unwrap();

    // 1 partition on purpose — see make_ctx / #53 (cost canon needs single-stream
    // execution for reproducible per-node output_bytes).
    let cpu_ctx = create_context_with_tables(&data_dir, 1, FULL_BUDGET)
        .await
        .unwrap();
    let plan = cpu_ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();
    let mut stats: Vec<NodeMemoryStats> = vec![];
    let actual = execute_node_by_node_instrumented(plan.clone(), cpu_ctx.task_ctx(), &mut stats)
        .await
        .unwrap();

    assert_eq!(
        batches_to_sorted_str(&actual),
        batches_to_sorted_str(&expected),
        "CPU executor result for TPC-DS '{name}' differs from plain DataFusion"
    );

    assert_cpu_cost_canonical(&plan, &stats, name, &tpcds_canondata_dir());
}

macro_rules! tpcds_cpu_result_test {
    ($func_name:ident, $query_name:literal) => {
        #[tokio::test]
        async fn $func_name() {
            assert_cpu_results_match_datafusion_tpcds($query_name).await;
        }
    };
}

tpcds_cpu_result_test!(test_cpu_tpcds_q1,  "q1");
tpcds_cpu_result_test!(test_cpu_tpcds_q2,  "q2");
tpcds_cpu_result_test!(test_cpu_tpcds_q3,  "q3");
tpcds_cpu_result_test!(test_cpu_tpcds_q4,  "q4");
tpcds_cpu_result_test!(test_cpu_tpcds_q5,  "q5");
tpcds_cpu_result_test!(test_cpu_tpcds_q6,  "q6");
tpcds_cpu_result_test!(test_cpu_tpcds_q7,  "q7");
tpcds_cpu_result_test!(test_cpu_tpcds_q8,  "q8");
tpcds_cpu_result_test!(test_cpu_tpcds_q9,  "q9");
tpcds_cpu_result_test!(test_cpu_tpcds_q10, "q10");
tpcds_cpu_result_test!(test_cpu_tpcds_q11, "q11");
tpcds_cpu_result_test!(test_cpu_tpcds_q12, "q12");
tpcds_cpu_result_test!(test_cpu_tpcds_q13, "q13");
tpcds_cpu_result_test!(test_cpu_tpcds_q14, "q14");
tpcds_cpu_result_test!(test_cpu_tpcds_q15, "q15");
tpcds_cpu_result_test!(test_cpu_tpcds_q16, "q16");
tpcds_cpu_result_test!(test_cpu_tpcds_q17, "q17");
tpcds_cpu_result_test!(test_cpu_tpcds_q18, "q18");
tpcds_cpu_result_test!(test_cpu_tpcds_q19, "q19");
tpcds_cpu_result_test!(test_cpu_tpcds_q20, "q20");
tpcds_cpu_result_test!(test_cpu_tpcds_q21, "q21");
tpcds_cpu_result_test!(test_cpu_tpcds_q22, "q22");
tpcds_cpu_result_test!(test_cpu_tpcds_q23, "q23");
tpcds_cpu_result_test!(test_cpu_tpcds_q24, "q24");
tpcds_cpu_result_test!(test_cpu_tpcds_q25, "q25");
tpcds_cpu_result_test!(test_cpu_tpcds_q26, "q26");
// Skipped (issue #14, DataFusion 45 limitation — SanityCheckPlan rejects
// SortPreservingMergeExec ordering for ROLLUP):
// tpcds_cpu_result_test!(test_cpu_tpcds_q27, "q27");
tpcds_cpu_result_test!(test_cpu_tpcds_q28, "q28");
tpcds_cpu_result_test!(test_cpu_tpcds_q29, "q29");
tpcds_cpu_result_test!(test_cpu_tpcds_q30, "q30");
tpcds_cpu_result_test!(test_cpu_tpcds_q31, "q31");
tpcds_cpu_result_test!(test_cpu_tpcds_q32, "q32");
tpcds_cpu_result_test!(test_cpu_tpcds_q33, "q33");
tpcds_cpu_result_test!(test_cpu_tpcds_q34, "q34");
tpcds_cpu_result_test!(test_cpu_tpcds_q35, "q35");
tpcds_cpu_result_test!(test_cpu_tpcds_q36, "q36");
tpcds_cpu_result_test!(test_cpu_tpcds_q37, "q37");
tpcds_cpu_result_test!(test_cpu_tpcds_q38, "q38");
tpcds_cpu_result_test!(test_cpu_tpcds_q39, "q39");
tpcds_cpu_result_test!(test_cpu_tpcds_q40, "q40");
tpcds_cpu_result_test!(test_cpu_tpcds_q41, "q41");
tpcds_cpu_result_test!(test_cpu_tpcds_q42, "q42");
tpcds_cpu_result_test!(test_cpu_tpcds_q43, "q43");
tpcds_cpu_result_test!(test_cpu_tpcds_q44, "q44");
tpcds_cpu_result_test!(test_cpu_tpcds_q45, "q45");
tpcds_cpu_result_test!(test_cpu_tpcds_q46, "q46");
tpcds_cpu_result_test!(test_cpu_tpcds_q47, "q47");
tpcds_cpu_result_test!(test_cpu_tpcds_q48, "q48");
tpcds_cpu_result_test!(test_cpu_tpcds_q49, "q49");
tpcds_cpu_result_test!(test_cpu_tpcds_q50, "q50");
tpcds_cpu_result_test!(test_cpu_tpcds_q51, "q51");
tpcds_cpu_result_test!(test_cpu_tpcds_q52, "q52");
tpcds_cpu_result_test!(test_cpu_tpcds_q53, "q53");
tpcds_cpu_result_test!(test_cpu_tpcds_q54, "q54");
tpcds_cpu_result_test!(test_cpu_tpcds_q55, "q55");
tpcds_cpu_result_test!(test_cpu_tpcds_q56, "q56");
tpcds_cpu_result_test!(test_cpu_tpcds_q57, "q57");
tpcds_cpu_result_test!(test_cpu_tpcds_q58, "q58");
tpcds_cpu_result_test!(test_cpu_tpcds_q59, "q59");
tpcds_cpu_result_test!(test_cpu_tpcds_q60, "q60");
tpcds_cpu_result_test!(test_cpu_tpcds_q61, "q61");
tpcds_cpu_result_test!(test_cpu_tpcds_q62, "q62");
tpcds_cpu_result_test!(test_cpu_tpcds_q63, "q63");
tpcds_cpu_result_test!(test_cpu_tpcds_q64, "q64");
tpcds_cpu_result_test!(test_cpu_tpcds_q65, "q65");
tpcds_cpu_result_test!(test_cpu_tpcds_q66, "q66");
tpcds_cpu_result_test!(test_cpu_tpcds_q67, "q67");
tpcds_cpu_result_test!(test_cpu_tpcds_q68, "q68");
tpcds_cpu_result_test!(test_cpu_tpcds_q69, "q69");
// Skipped (issue #14, DataFusion 45 limitation — GROUPING() aggregate has no
// physical-plan support):
// tpcds_cpu_result_test!(test_cpu_tpcds_q70, "q70");
tpcds_cpu_result_test!(test_cpu_tpcds_q71, "q71");
// Skipped (issue #14, DataFusion 45 limitation — Date32 + Int64 type-coercion
// not supported):
// tpcds_cpu_result_test!(test_cpu_tpcds_q72, "q72");
tpcds_cpu_result_test!(test_cpu_tpcds_q73, "q73");
tpcds_cpu_result_test!(test_cpu_tpcds_q74, "q74");
tpcds_cpu_result_test!(test_cpu_tpcds_q75, "q75");
tpcds_cpu_result_test!(test_cpu_tpcds_q76, "q76");
tpcds_cpu_result_test!(test_cpu_tpcds_q77, "q77");
tpcds_cpu_result_test!(test_cpu_tpcds_q78, "q78");
tpcds_cpu_result_test!(test_cpu_tpcds_q79, "q79");
tpcds_cpu_result_test!(test_cpu_tpcds_q80, "q80");
tpcds_cpu_result_test!(test_cpu_tpcds_q81, "q81");
tpcds_cpu_result_test!(test_cpu_tpcds_q82, "q82");
tpcds_cpu_result_test!(test_cpu_tpcds_q83, "q83");
tpcds_cpu_result_test!(test_cpu_tpcds_q84, "q84");
tpcds_cpu_result_test!(test_cpu_tpcds_q85, "q85");
// Skipped (issue #14, DataFusion 45 limitation — GROUPING() aggregate has no
// physical-plan support):
// tpcds_cpu_result_test!(test_cpu_tpcds_q86, "q86");
tpcds_cpu_result_test!(test_cpu_tpcds_q87, "q87");
tpcds_cpu_result_test!(test_cpu_tpcds_q88, "q88");
tpcds_cpu_result_test!(test_cpu_tpcds_q89, "q89");
tpcds_cpu_result_test!(test_cpu_tpcds_q90, "q90");
tpcds_cpu_result_test!(test_cpu_tpcds_q91, "q91");
tpcds_cpu_result_test!(test_cpu_tpcds_q92, "q92");
tpcds_cpu_result_test!(test_cpu_tpcds_q93, "q93");
tpcds_cpu_result_test!(test_cpu_tpcds_q94, "q94");
tpcds_cpu_result_test!(test_cpu_tpcds_q95, "q95");
tpcds_cpu_result_test!(test_cpu_tpcds_q96, "q96");
tpcds_cpu_result_test!(test_cpu_tpcds_q97, "q97");
tpcds_cpu_result_test!(test_cpu_tpcds_q98, "q98");
tpcds_cpu_result_test!(test_cpu_tpcds_q99, "q99");

#[tokio::test]
async fn test_memory_boundary_preserved_tight_budget() {
    let query = "SELECT count(*) FROM customer WHERE c_custkey > 0";

    let ctx_full = make_ctx(FULL_BUDGET).await;
    let plan_full = ctx_full.sql(query).await.unwrap()
        .create_physical_plan().await.unwrap();

    let ctx_tight = make_ctx(TIGHT_BUDGET).await;
    let plan_tight = ctx_tight.sql(query).await.unwrap()
        .create_physical_plan().await.unwrap();

    eprintln!("\n=== FULL BUDGET ({} GiB) plan ===\n{}", FULL_BUDGET / (1024*1024*1024), fmt_plan(&plan_full));
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

    eprintln!(
        "GpuScanExec batch_size — full budget: {full_batch_size}, tight budget: {gpu_batch_size}"
    );
    assert!(
        gpu_batch_size < full_batch_size,
        "tight budget batch_size ({gpu_batch_size}) should be smaller than full budget ({full_batch_size})"
    );

    let mut stats: Vec<NodeMemoryStats> = vec![];
    let batches =
        execute_node_by_node_instrumented(plan_tight, ctx_tight.task_ctx(), &mut stats)
            .await
            .unwrap();

    let count = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
    assert_eq!(count, 150_000, "customer table must have 150 000 rows");

    let scan_stats: Vec<&NodeMemoryStats> = stats
        .iter()
        .filter(|s| s.node_name == "ParquetExec")
        .collect();
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
            s.max_batch_rows, gpu_batch_size
        );
    }

    let gpu_names: Vec<&str> = stats
        .iter()
        .filter(|s| s.node_name.starts_with("Gpu"))
        .map(|s| s.node_name.as_str())
        .collect();
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
    assert_eq!(
        root_stat.row_count, final_rows,
        "root node row_count in stats does not match actual output"
    );
    assert!(
        root_stat.allocated_bytes > 0,
        "root node allocated_bytes should be > 0"
    );
    assert!(
        root_stat.output_bytes > 0,
        "root node output_bytes should be > 0"
    );
    assert!(
        root_stat.allocated_bytes >= root_stat.output_bytes,
        "allocated_bytes ({}) must be >= output_bytes ({})",
        root_stat.allocated_bytes,
        root_stat.output_bytes
    );
}