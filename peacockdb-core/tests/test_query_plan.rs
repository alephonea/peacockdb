//! Parameterized tests that canonize GPU execution plans for TPC-H queries.
//!
//! Each test reads a SQL file from `testdata/tpch-queries/<name>.sql`, plans it
//! against the SF-1 dataset, strips GPU wrappers, and compares the result against
//! the canonical plan stored in `tests/canondata/<name>.txt`.
//!
//! # Canonizing
//!
//! To write (or overwrite) canonical files from the current actual output, run with
//! `CANONIZE=1` or pass `-Z` to the test binary:
//!
//!   CANONIZE=1 cargo test --test test_queries
//!
//! Each canonical file contains the normalized, GPU-stripped physical plan for one
//! TPC-H query. ParquetExec lines are normalized to `ParquetExec: table=<stem>` so
//! the files are path-independent.

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::ExecutionPlan;

use peacockdb_core::build_session_state_with_gpu_rules;
use peacockdb_core::register_tables_for;
use peacockdb_core::create_context_with_tables;
use peacockdb_core::gpu_rule::{analyze_memory, row_width, GpuScanExec};
use peacockdb_core::plan_serializer;
use peacockdb_core::CpuExecutor;

const TARGET_PARTITIONS: usize = 8;
const TEST_TARGET_PARTITIONS: usize = 8;
const TEST_GPU_MEMORY_BUDGET: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.sf1")
}

fn testdata_minimal_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal")
}

fn queries_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch-queries")
}

fn canondata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/plans.sf1")
}

fn plans_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/plans")
}

fn tpcds_testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpcds.sf1")
}

fn tpcds_queries_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpcds-queries")
}

fn tpcds_canondata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/plans-tpcds.sf1")
}

fn test_ctx(data_dir: &std::path::Path) -> impl std::future::Future<Output = datafusion::error::Result<SessionContext>> + '_ {
    create_context_with_tables(data_dir, TEST_TARGET_PARTITIONS, TEST_GPU_MEMORY_BUDGET)
}

async fn count(ctx: &SessionContext, query: &str) -> i64 {
    let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

/// Render the plan to a string, normalizing ParquetExec lines to be path-independent.
fn plan_str(plan: &Arc<dyn ExecutionPlan>) -> String {
    let raw = DisplayableExecutionPlan::new(plan.as_ref())
        .indent(false)
        .to_string();
    raw.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            if line.trim_start().starts_with("ParquetExec:") {
                let indent = line.len() - line.trim_start().len();
                let table = line
                    .find(".parquet")
                    .and_then(|end| line[..end].rfind('/').map(|sep| &line[sep + 1..end]))
                    .unwrap_or("unknown");
                format!("{}ParquetExec: table={table}", &line[..indent])
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn memory_str(plan: &Arc<dyn ExecutionPlan>) -> String {
    fn total_estimate_cost(plan: &Arc<dyn ExecutionPlan>) -> usize {
        let mem = analyze_memory(plan);
        let node_cost = mem.input_row_bytes + mem.output_row_bytes;
        node_cost + plan.children().iter().map(|c| total_estimate_cost(c)).sum::<usize>()
    }

    fn walk(plan: &Arc<dyn ExecutionPlan>, indent: usize, lines: &mut Vec<String>) {
        let mem = analyze_memory(plan);
        let rw = row_width(&plan.schema());
        let estimate_cost = mem.input_row_bytes + mem.output_row_bytes;
        lines.push(format!(
            "{}{}: row_width={}, subtree_max_row_bytes={}, estimate_input_bytes={}, estimate_output_bytes={}, estimate_cost={}",
            " ".repeat(indent),
            plan.name(),
            rw,
            mem.subtree_max_row_bytes,
            mem.input_row_bytes,
            mem.output_row_bytes,
            estimate_cost,
        ));
        for child in plan.children() {
            walk(child, indent + 2, lines);
        }
    }
    let mut lines = Vec::new();
    walk(plan, 0, &mut lines);
    format!("total_estimate_cost={}\n{}", total_estimate_cost(plan), lines.join("\n"))
}

fn assert_plan_matches_canonical_at(plan: &Arc<dyn ExecutionPlan>, name: &str, dir: &std::path::Path) {
    let canonical_path = dir.join(format!("{name}.txt"));
    let actual = format!("{}\n--- memory ---\n{}", plan_str(plan), memory_str(plan));

    if std::env::var("UPDATE_CANONICAL").is_ok() {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(&canonical_path, &actual).unwrap();
        eprintln!("Updated canonical plan: {}", canonical_path.display());
        return;
    }

    let canonical = std::fs::read_to_string(&canonical_path).unwrap_or_else(|_| {
        panic!(
            "canonical file not found: {}\nRun with UPDATE_CANONICAL=1 to generate it.",
            canonical_path.display()
        )
    });
    assert_eq!(
        actual,
        canonical.trim_end(),
        "plan for '{name}' does not match {}",
        canonical_path.display()
    );

    // Flatbuffer roundtrip — skip if the plan contains unsupported nodes/expressions.
    match plan_serializer::serialize_plan(plan) {
        Ok(bytes) => {
            match plan_serializer::deserialize_plan(&bytes) {
                Ok(reconstructed) => {
                    // Two complementary oracles. Serialization is a verbatim
                    // encoding of the plan (all GPU lowering happens earlier, in
                    // GpuExecutionRule), so the reconstructed plan must match the
                    // original on both:
                    //  - text-equal on `plan_str` — a readable check that the node
                    //    tree and the fields DisplayAs prints are reproduced; gives
                    //    a legible diff when it breaks.
                    //  - bytes-equal on the re-serialized IR — the complete,
                    //    Display-independent oracle: it surfaces every serialized
                    //    field (decimal precision/scale, schema field types,
                    //    join-filter inner exprs) and FlatBuffers offset ordering,
                    //    none of which DisplayAs is guaranteed to print.
                    assert_eq!(
                        plan_str(&reconstructed),
                        plan_str(plan),
                        "flatbuffer roundtrip (plan_str) mismatch for '{name}'"
                    );
                    match plan_serializer::serialize_plan(&reconstructed) {
                        Ok(reserialized) => assert_eq!(
                            reserialized, bytes,
                            "flatbuffer roundtrip (bytes) mismatch for '{name}'"
                        ),
                        Err(e) => panic!(
                            "re-serialize of reconstructed plan failed for '{name}': {e}"
                        ),
                    }
                }
                Err(e) if e.contains("not supported") || e.contains("unsupported") => {
                    eprintln!("Skipping flatbuffer roundtrip for '{name}': {e}");
                }
                Err(e) => panic!("flatbuffer deserialization failed for '{name}': {e}"),
            }
        }
        Err(e) if e.contains("unsupported") => {
            eprintln!("Skipping flatbuffer roundtrip for '{name}': {e}");
        }
        Err(e) => panic!("flatbuffer serialization failed for '{name}': {e}"),
    }
}

fn assert_plan_matches_canonical(plan: &Arc<dyn ExecutionPlan>, name: &str) {
    assert_plan_matches_canonical_at(plan, name, &plans_dir());
}

/// Plan `name` against `data_dir`/`queries_dir` and compare it to the canonical
/// plan in `canon_dir`. `gen_hint` is the generate_testdata.sh invocation shown
/// if the dataset is missing. Shared by the TPC-H and TPC-DS test macros.
async fn run_query_test_at(
    name: &str,
    data_dir: &std::path::Path,
    queries_dir: &std::path::Path,
    canon_dir: &std::path::Path,
    gen_hint: &str,
) {
    if !data_dir.exists() {
        panic!(
            "SF-1 dataset not found at {}. Run {gen_hint} first.",
            data_dir.display()
        );
    }

    let sql_path = queries_dir.join(format!("{name}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));

    let gpu_ctx = register_tables_for(
        build_session_state_with_gpu_rules(TARGET_PARTITIONS, TEST_GPU_MEMORY_BUDGET),
        data_dir,
    )
    .await
    .unwrap();

    let plan = gpu_ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();
    assert_plan_matches_canonical_at(&plan, name, canon_dir);
}

/// Define a TPC-H plan-canonical test (testdata/tpch-queries → testdata/plans.sf1).
macro_rules! query_plan_test {
    ($func_name:ident, $query_name:literal) => {
        #[tokio::test]
        async fn $func_name() {
            run_query_test_at(
                $query_name,
                &testdata_dir(),
                &queries_dir(),
                &canondata_dir(),
                "testdata/generate_testdata.sh",
            )
            .await;
        }
    };
}

/// Define a TPC-DS plan-canonical test (testdata/tpcds-queries → testdata/plans-tpcds.sf1).
macro_rules! query_plan_test_tpcds {
    ($func_name:ident, $query_name:literal) => {
        #[tokio::test]
        async fn $func_name() {
            run_query_test_at(
                $query_name,
                &tpcds_testdata_dir(),
                &tpcds_queries_dir(),
                &tpcds_canondata_dir(),
                "testdata/generate_testdata.sh --bench tpcds",
            )
            .await;
        }
    };
}

query_plan_test!(test_scan_limit, "scan-limit");
query_plan_test!(test_filter_project, "filter-project");
query_plan_test!(test_aggregate_groupby, "aggregate-groupby");
query_plan_test!(test_hash_join, "hash-join");
query_plan_test!(test_left_join, "left-join");
query_plan_test!(test_semi_join, "semi-join");
query_plan_test!(test_anti_join, "anti-join");
query_plan_test!(test_nested_loop_join, "nested-loop-join");
query_plan_test!(test_mixed_join, "mixed-join");
query_plan_test!(test_cross_join, "cross-join");


query_plan_test!(plan_tpch_q1,  "q1");
query_plan_test!(plan_tpch_q2,  "q2");
query_plan_test!(plan_tpch_q3,  "q3");
query_plan_test!(plan_tpch_q4,  "q4");
query_plan_test!(plan_tpch_q5,  "q5");
query_plan_test!(plan_tpch_q6,  "q6");
query_plan_test!(plan_tpch_q7,  "q7");
query_plan_test!(plan_tpch_q8,  "q8");
query_plan_test!(plan_tpch_q9,  "q9");
query_plan_test!(plan_tpch_q10, "q10");
query_plan_test!(plan_tpch_q11, "q11");
query_plan_test!(plan_tpch_q12, "q12");
query_plan_test!(plan_tpch_q13, "q13");
query_plan_test!(plan_tpch_q14, "q14");
query_plan_test!(plan_tpch_q15, "q15");
query_plan_test!(plan_tpch_q16, "q16");
query_plan_test!(plan_tpch_q17, "q17");
query_plan_test!(plan_tpch_q18, "q18");
query_plan_test!(plan_tpch_q19, "q19");
query_plan_test!(plan_tpch_q20, "q20");
query_plan_test!(plan_tpch_q21, "q21");
query_plan_test!(plan_tpch_q22, "q22");

// ── CpuExecutor integration tests ────────────────────────────────────────

/// Full end-to-end example showing the idiomatic usage:
///   1. CpuExecutor::new  — builds a SessionContext with GPU rules
///   2. exec.execute(sql) — SQL → GPU plan → CPU execution → RecordBatches
#[tokio::test]
async fn test_cpu_executor_simple_query() {
    let exec = CpuExecutor::new(&testdata_minimal_dir(), 1, 2 * 1024 * 1024 * 1024)
        .await
        .unwrap();

    let batches = exec
        .execute("SELECT count(*) FROM nation WHERE n_regionkey >= 0")
        .await
        .unwrap();

    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);

    assert_eq!(count, 25);
}

/// execute_instrumented returns both results and per-node stats in one call.
#[tokio::test]
async fn test_cpu_executor_instrumented() {
    let exec = CpuExecutor::new(&testdata_minimal_dir(), 1, 2 * 1024 * 1024 * 1024)
        .await
        .unwrap();

    let (batches, _plan, stats) = exec
        .execute_instrumented("SELECT count(*) FROM nation WHERE n_regionkey >= 0")
        .await
        .unwrap();

    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 25);

    // Every stat entry must name a CPU node, never a GPU wrapper.
    for s in &stats {
        assert!(
            !s.node_name.starts_with("Gpu"),
            "GPU node '{}' leaked into stats",
            s.node_name
        );
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

/// Filter + aggregate: SELECT count(*) FROM customer WHERE c_acctbal > 0
/// Expected GPU nodes: GpuAggregateExec (partial + final), GpuFilterExec
#[tokio::test]
async fn test_gpu_nodes_filter_agg() {
    let ctx = test_ctx(&testdata_minimal_dir()).await.unwrap();
    let query = "SELECT count(*) FROM customer WHERE c_acctbal > 0";

    let plan = ctx.sql(query).await.unwrap().create_physical_plan().await.unwrap();
    assert_plan_matches_canonical(&plan, "filter_agg");

    let n = count(&ctx, query).await;
    assert!(n > 0 && n <= 150_000, "unexpected count {n}");
}

/// Hash join + sort: nations joined with their region, sorted by name.
/// Expected GPU nodes: GpuSortExec, GpuHashJoinExec
#[tokio::test]
async fn test_gpu_nodes_join_sort() {
    let ctx = test_ctx(&testdata_minimal_dir()).await.unwrap();
    let query = "
        SELECT n.n_name, r.r_name
        FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey
        ORDER BY n.n_name";

    let plan = ctx.sql(query).await.unwrap().create_physical_plan().await.unwrap();
    assert_plan_matches_canonical(&plan, "join_sort");

    // Result: 25 rows (every nation has exactly one region)
    let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 25);
}

/// Group by + join + sort: nations per region, sorted descending by count.
/// Expected GPU nodes: GpuSortExec, GpuAggregateExec, GpuHashJoinExec
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

    // Result: 5 regions, each with exactly 5 nations.
    let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 5);
    let counts = batches[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    for i in 0..counts.len() {
        assert_eq!(counts.value(i), 5, "region {} has {} nations, expected 5", i, counts.value(i));
    }
}

// ── Memory cost tests ────────────────────────────────────────────────────

fn find_node(plan: &Arc<dyn ExecutionPlan>, name: &str) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.name() == name {
        return Some(plan.clone());
    }
    plan.children().iter().find_map(|c| find_node(c, name))
}

#[tokio::test]
async fn test_memory_cost_leaf_and_filter() {
    let ctx = test_ctx(&testdata_minimal_dir()).await.unwrap();

    // Leaf (GpuScanExec): no input, output = row_width, peak = row_width.
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

    // Filter (GpuFilterExec) with trivial selectivity (1.0):
    //   input  = child output width  (ratio 1.0 × child_row_width)
    //   output = filter_row_width    (ratio 1.0 × sel 1.0 × output_width)
    //   peak   = max(child_peak, input + output)
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

/// Find all GpuScanExec nodes and return their batch sizes.
fn scan_batch_sizes(plan: &Arc<dyn ExecutionPlan>) -> Vec<usize> {
    let mut sizes = Vec::new();
    if let Some(scan) = plan.as_any().downcast_ref::<GpuScanExec>() {
        sizes.push(scan.gpu_batch_size);
    }
    for child in plan.children() {
        sizes.extend(scan_batch_sizes(child));
    }
    sizes
}

/// With a tight GPU memory budget, the batch size should be reduced below
/// the default 8192. Results must still be correct.
#[tokio::test]
async fn test_memory_budget_reduces_batch_size() {
    // 10 KiB budget → should force a very small batch size.
    let ctx =
        create_context_with_tables(&testdata_minimal_dir(), TEST_TARGET_PARTITIONS, 10 * 1024).await.unwrap();
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

    // Results must still be correct despite smaller batches.
    let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 25);
}

// ── TPC-DS plan-canonical tests ──────────────────────────────────────────
// Reads testdata/tpcds-queries/q<N>.sql, compares against testdata/plans-tpcds.sf1.

query_plan_test_tpcds!(plan_tpcds_q1, "q1");
query_plan_test_tpcds!(plan_tpcds_q2, "q2");
query_plan_test_tpcds!(plan_tpcds_q3, "q3");
query_plan_test_tpcds!(plan_tpcds_q4, "q4");
query_plan_test_tpcds!(plan_tpcds_q5, "q5");
query_plan_test_tpcds!(plan_tpcds_q6, "q6");
query_plan_test_tpcds!(plan_tpcds_q7, "q7");
query_plan_test_tpcds!(plan_tpcds_q8, "q8");
query_plan_test_tpcds!(plan_tpcds_q9, "q9");
query_plan_test_tpcds!(plan_tpcds_q10, "q10");
query_plan_test_tpcds!(plan_tpcds_q11, "q11");
query_plan_test_tpcds!(plan_tpcds_q12, "q12");
query_plan_test_tpcds!(plan_tpcds_q13, "q13");
query_plan_test_tpcds!(plan_tpcds_q14, "q14");
query_plan_test_tpcds!(plan_tpcds_q15, "q15");
query_plan_test_tpcds!(plan_tpcds_q16, "q16");
query_plan_test_tpcds!(plan_tpcds_q17, "q17");
query_plan_test_tpcds!(plan_tpcds_q18, "q18");
query_plan_test_tpcds!(plan_tpcds_q19, "q19");
query_plan_test_tpcds!(plan_tpcds_q20, "q20");
query_plan_test_tpcds!(plan_tpcds_q21, "q21");
query_plan_test_tpcds!(plan_tpcds_q22, "q22");
query_plan_test_tpcds!(plan_tpcds_q23, "q23");
query_plan_test_tpcds!(plan_tpcds_q24, "q24");
query_plan_test_tpcds!(plan_tpcds_q25, "q25");
query_plan_test_tpcds!(plan_tpcds_q26, "q26");
// q27: DataFusion 45 SanityCheckPlan rejects the SortPreservingMergeExec ordering
// emitted for ROLLUP. Re-enable once upstream is fixed.
// query_plan_test_tpcds!(plan_tpcds_q27, "q27");
query_plan_test_tpcds!(plan_tpcds_q28, "q28");
query_plan_test_tpcds!(plan_tpcds_q29, "q29");
query_plan_test_tpcds!(plan_tpcds_q30, "q30");
query_plan_test_tpcds!(plan_tpcds_q31, "q31");
query_plan_test_tpcds!(plan_tpcds_q32, "q32");
query_plan_test_tpcds!(plan_tpcds_q33, "q33");
query_plan_test_tpcds!(plan_tpcds_q34, "q34");
query_plan_test_tpcds!(plan_tpcds_q35, "q35");
query_plan_test_tpcds!(plan_tpcds_q36, "q36");
query_plan_test_tpcds!(plan_tpcds_q37, "q37");
query_plan_test_tpcds!(plan_tpcds_q38, "q38");
query_plan_test_tpcds!(plan_tpcds_q39, "q39");
query_plan_test_tpcds!(plan_tpcds_q40, "q40");
query_plan_test_tpcds!(plan_tpcds_q41, "q41");
query_plan_test_tpcds!(plan_tpcds_q42, "q42");
query_plan_test_tpcds!(plan_tpcds_q43, "q43");
query_plan_test_tpcds!(plan_tpcds_q44, "q44");
query_plan_test_tpcds!(plan_tpcds_q45, "q45");
query_plan_test_tpcds!(plan_tpcds_q46, "q46");
query_plan_test_tpcds!(plan_tpcds_q47, "q47");
query_plan_test_tpcds!(plan_tpcds_q48, "q48");
query_plan_test_tpcds!(plan_tpcds_q49, "q49");
query_plan_test_tpcds!(plan_tpcds_q50, "q50");
query_plan_test_tpcds!(plan_tpcds_q51, "q51");
query_plan_test_tpcds!(plan_tpcds_q52, "q52");
query_plan_test_tpcds!(plan_tpcds_q53, "q53");
query_plan_test_tpcds!(plan_tpcds_q54, "q54");
query_plan_test_tpcds!(plan_tpcds_q55, "q55");
query_plan_test_tpcds!(plan_tpcds_q56, "q56");
query_plan_test_tpcds!(plan_tpcds_q57, "q57");
query_plan_test_tpcds!(plan_tpcds_q58, "q58");
query_plan_test_tpcds!(plan_tpcds_q59, "q59");
query_plan_test_tpcds!(plan_tpcds_q60, "q60");
query_plan_test_tpcds!(plan_tpcds_q61, "q61");
query_plan_test_tpcds!(plan_tpcds_q62, "q62");
query_plan_test_tpcds!(plan_tpcds_q63, "q63");
query_plan_test_tpcds!(plan_tpcds_q64, "q64");
query_plan_test_tpcds!(plan_tpcds_q65, "q65");
query_plan_test_tpcds!(plan_tpcds_q66, "q66");
query_plan_test_tpcds!(plan_tpcds_q67, "q67");
query_plan_test_tpcds!(plan_tpcds_q68, "q68");
query_plan_test_tpcds!(plan_tpcds_q69, "q69");
// q70: DataFusion 45 doesn't physical-plan the GROUPING() aggregate.
// query_plan_test_tpcds!(plan_tpcds_q70, "q70");
query_plan_test_tpcds!(plan_tpcds_q71, "q71");
// q72: DataFusion 45 type-coercion can't handle Date32 + Int64 arithmetic.
// query_plan_test_tpcds!(plan_tpcds_q72, "q72");
query_plan_test_tpcds!(plan_tpcds_q73, "q73");
query_plan_test_tpcds!(plan_tpcds_q74, "q74");
query_plan_test_tpcds!(plan_tpcds_q75, "q75");
query_plan_test_tpcds!(plan_tpcds_q76, "q76");
query_plan_test_tpcds!(plan_tpcds_q77, "q77");
query_plan_test_tpcds!(plan_tpcds_q78, "q78");
query_plan_test_tpcds!(plan_tpcds_q79, "q79");
query_plan_test_tpcds!(plan_tpcds_q80, "q80");
query_plan_test_tpcds!(plan_tpcds_q81, "q81");
query_plan_test_tpcds!(plan_tpcds_q82, "q82");
query_plan_test_tpcds!(plan_tpcds_q83, "q83");
query_plan_test_tpcds!(plan_tpcds_q84, "q84");
query_plan_test_tpcds!(plan_tpcds_q85, "q85");
// q86: DataFusion 45 doesn't physical-plan the GROUPING() aggregate.
// query_plan_test_tpcds!(plan_tpcds_q86, "q86");
query_plan_test_tpcds!(plan_tpcds_q87, "q87");
query_plan_test_tpcds!(plan_tpcds_q88, "q88");
query_plan_test_tpcds!(plan_tpcds_q89, "q89");
query_plan_test_tpcds!(plan_tpcds_q90, "q90");
query_plan_test_tpcds!(plan_tpcds_q91, "q91");
query_plan_test_tpcds!(plan_tpcds_q92, "q92");
query_plan_test_tpcds!(plan_tpcds_q93, "q93");
query_plan_test_tpcds!(plan_tpcds_q94, "q94");
query_plan_test_tpcds!(plan_tpcds_q95, "q95");
query_plan_test_tpcds!(plan_tpcds_q96, "q96");
query_plan_test_tpcds!(plan_tpcds_q97, "q97");
query_plan_test_tpcds!(plan_tpcds_q98, "q98");
query_plan_test_tpcds!(plan_tpcds_q99, "q99");
