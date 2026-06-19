//! Shared test harness for the plan / cpu / gpu suites.
//!
//! All helper functions and the unified test macros live here; the suite files
//! (test_query_plan.rs, test_cpu_executor.rs, test_gpu_executor.rs) contain only
//! macro invocations. Each integration-test crate includes this via
//! `#[macro_use] mod common;`. Every suite uses a subset, so dead code is fine.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::{DisplayFormatType, ExecutionPlan};

use peacockdb_core::cpu_executor::{execute_node_by_node_instrumented_enforced, NodeMemoryStats};
use peacockdb_core::gpu_rule::{analyze_memory, row_width};
use peacockdb_core::plan_serializer;
use peacockdb_core::{
    build_session_state, build_session_state_with_gpu_rules, create_context_with_tables,
    register_tables_for,
};

// --- run configs encoded in the device label -------------------------------
pub const TARGET_PARTITIONS: usize = 8; // plan tests
pub const TEST_GPU_MEMORY_BUDGET: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
pub const FULL_BUDGET: usize = 2 * 1024 * 1024 * 1024;
pub const TIGHT_BUDGET: usize = 10 * 1024;
pub const GPU_BUDGET: usize = 2 * 1024 * 1024 * 1024;

// --- parameterized testdata layout -----------------------------------------
//   data    = <root>/<dataset>.sf<sf>/        (parquet)
//   queries = <root>/<dataset>-queries/<query>.sql
//   goldens = <root>/goldens/<dataset>.sf<sf>/<query>.<device>.{plan.txt,cpu.txt}
// PEACOCK_TESTDATA_DIR overrides the compile-time root so a binary built on one
// machine can run on another (e.g. shad-gpu).
pub fn testdata_root() -> PathBuf {
    if let Some(d) = std::env::var_os("PEACOCK_TESTDATA_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata")
}

pub fn data_dir_for(dataset: &str, sf: &str) -> PathBuf {
    testdata_root().join(format!("{dataset}.sf{sf}"))
}

pub fn queries_dir_for(dataset: &str) -> PathBuf {
    testdata_root().join(format!("{dataset}-queries"))
}

pub fn golden_dir_for(dataset: &str, sf: &str) -> PathBuf {
    testdata_root().join(format!("goldens/{dataset}.sf{sf}"))
}

pub fn plan_golden(dataset: &str, sf: &str, query: &str, device: &str) -> PathBuf {
    golden_dir_for(dataset, sf).join(format!("{query}.{device}.plan.txt"))
}

pub fn cpu_golden(dataset: &str, sf: &str, query: &str, device: &str) -> PathBuf {
    golden_dir_for(dataset, sf).join(format!("{query}.{device}.cpu.txt"))
}

pub fn testdata_dir() -> PathBuf {
    testdata_root().join("tpch.sf1")
}

/// Decode a device label like `tp8-mem2gib` into `(target_partitions, budget_bytes)`.
/// The device dimension is authoritative: the golden path/name AND the run config
/// both derive from it, so a mislabeled test (e.g. a cpu test tagged tp8-…) runs the
/// config it claims rather than silently diverging from its label. (H200 is GPU-only
/// and carries no plan/cpu config, so it never reaches here.)
pub fn device_config(device: &str) -> (usize, usize) {
    let (tp, mem) = device
        .split_once('-')
        .unwrap_or_else(|| panic!("device label '{device}' must look like 'tp<N>-mem<N>gib'"));
    let partitions: usize = tp
        .strip_prefix("tp")
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("device '{device}': bad partition count in '{tp}'"));
    let gib: usize = mem
        .strip_prefix("mem")
        .and_then(|m| m.strip_suffix("gib"))
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("device '{device}': bad budget in '{mem}' (expected mem<N>gib)"));
    (partitions, gib * 1024 * 1024 * 1024)
}

pub fn testdata_minimal_dir() -> PathBuf {
    testdata_root().join("tpch.minimal")
}

// --- plan rendering ---------------------------------------------------------
/// Render the plan to a string, normalizing ParquetExec lines to be path-independent.
pub fn plan_str(plan: &Arc<dyn ExecutionPlan>) -> String {
    let raw = DisplayableExecutionPlan::new(plan.as_ref()).indent(false).to_string();
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

pub fn memory_str(plan: &Arc<dyn ExecutionPlan>) -> String {
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

/// One node's one-line Display (its `DisplayAs::fmt_as`, no children).
pub struct OneLine<'a>(pub &'a dyn ExecutionPlan);
impl std::fmt::Display for OneLine<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_as(DisplayFormatType::Default, f)
    }
}

// --- result formatting ------------------------------------------------------
/// Pretty-print batches with data rows sorted, for order-independent compares.
pub fn batches_to_sorted_str(batches: &[RecordBatch]) -> String {
    let formatted = pretty_format_batches(batches).unwrap().to_string();
    let lines: Vec<&str> = formatted.lines().collect();
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

pub fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

// --- plan-canonical assertion ----------------------------------------------
pub fn assert_plan_matches_canonical_at(plan: &Arc<dyn ExecutionPlan>, canonical_path: &Path) {
    let actual = format!("{}\n--- memory ---\n{}", plan_str(plan), memory_str(plan));

    if std::env::var("UPDATE_CANONICAL").is_ok() {
        std::fs::create_dir_all(canonical_path.parent().unwrap()).unwrap();
        std::fs::write(canonical_path, &actual).unwrap();
        eprintln!("Updated canonical plan: {}", canonical_path.display());
        return;
    }

    let name = canonical_path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
    let canonical = std::fs::read_to_string(canonical_path).unwrap_or_else(|_| {
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
        Ok(bytes) => match plan_serializer::deserialize_plan(&bytes) {
            Ok(reconstructed) => {
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
                    Err(e) => panic!("re-serialize of reconstructed plan failed for '{name}': {e}"),
                }
            }
            Err(e) if e.contains("not supported") || e.contains("unsupported") => {
                eprintln!("Skipping flatbuffer roundtrip for '{name}': {e}");
            }
            Err(e) => panic!("flatbuffer deserialization failed for '{name}': {e}"),
        },
        Err(e) if e.contains("unsupported") => {
            eprintln!("Skipping flatbuffer roundtrip for '{name}': {e}");
        }
        Err(e) => panic!("flatbuffer serialization failed for '{name}': {e}"),
    }
}

/// Synthetic plan tests (programmatic plans) at the plan device (8 part / 2 GiB).
pub fn assert_plan_matches_canonical(plan: &Arc<dyn ExecutionPlan>, name: &str) {
    assert_plan_matches_canonical_at(plan, &plan_golden("tpch", "1", name, "tp8-mem2gib"));
}

/// Plan `<dataset>-queries/<query>.sql` and compare to the plan golden.
pub async fn run_query_test_at(dataset: &str, sf: &str, query: &str, device: &str) {
    let data_dir = data_dir_for(dataset, sf);
    if !data_dir.exists() {
        panic!(
            "dataset not found at {}. Run testdata/generate_testdata.sh{} first.",
            data_dir.display(),
            if dataset == "tpcds" { " --bench tpcds" } else { "" }
        );
    }
    let sql_path = queries_dir_for(dataset).join(format!("{query}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));
    let (partitions, budget) = device_config(device);
    let gpu_ctx = register_tables_for(
        build_session_state_with_gpu_rules(partitions, budget),
        &data_dir,
    )
    .await
    .unwrap();
    let plan = gpu_ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();
    assert_plan_matches_canonical_at(&plan, &plan_golden(dataset, sf, query, device));
}

// --- cpu cost tree + assertion ---------------------------------------------
/// Format per-node CPU execution stats as a pre-order tree. Node label + rich
/// annotations come from each node's own Display (same source as the .txt plan
/// goldens); only the trailing cost fields are .cpu.txt-specific.
pub fn cpu_stats_str(plan: &Arc<dyn ExecutionPlan>, stats: &[NodeMemoryStats]) -> String {
    struct Node<'a> {
        stat: &'a NodeMemoryStats,
        plan: &'a Arc<dyn ExecutionPlan>,
        children: Vec<Node<'a>>,
    }
    fn collect<'a>(plan: &'a Arc<dyn ExecutionPlan>, stats: &'a [NodeMemoryStats], idx: &mut usize) -> Node<'a> {
        let children: Vec<Node<'a>> = plan.children().iter().map(|c| collect(c, stats, idx)).collect();
        let stat = &stats[*idx];
        *idx += 1;
        Node { stat, plan, children }
    }
    fn walk(node: &Node, indent: usize, lines: &mut Vec<String>) {
        lines.push(format!(
            "{}{}, output_bytes={}, output_rows={}",
            " ".repeat(indent),
            OneLine(node.plan.as_ref()),
            node.stat.output_bytes,
            node.stat.row_count,
        ));
        for child in &node.children {
            walk(child, indent + 2, lines);
        }
    }
    let root = collect(plan, stats, &mut 0);
    let mut lines = Vec::new();
    walk(&root, 0, &mut lines);
    // Explicit total footer, symmetric with the duckdb golden's `duckdb_cost=`.
    let total: usize = stats.iter().map(|s| s.output_bytes).sum();
    lines.push(format!("peacockdb_cost={total}"));
    lines.join("\n")
}

pub fn assert_cpu_cost_canonical(plan: &Arc<dyn ExecutionPlan>, stats: &[NodeMemoryStats], canonical_path: &Path) {
    let name = canonical_path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
    let actual = cpu_stats_str(plan, stats);

    if std::env::var("UPDATE_CANONICAL").is_ok() {
        std::fs::create_dir_all(canonical_path.parent().unwrap()).unwrap();
        std::fs::write(canonical_path, &actual).unwrap();
        eprintln!("Updated CPU canonical: {}", canonical_path.display());
        return;
    }
    let canonical = std::fs::read_to_string(canonical_path).unwrap_or_else(|_| {
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

/// Order-independent result comparison with an OPTIONAL relative tolerance on
/// `Float64` columns.
///
/// - `rel_tol = None`: exact sorted-string equality (the default).
/// - `rel_tol = Some(tol)`: rows are grouped by their NON-float columns
///   (formatted) and every `Float64` cell must agree within `tol` relative error.
///   Used only where the sole divergence from the DataFusion oracle is float
///   summation reassociation across partitions (~1 ULP), which the node-by-node
///   executor incurs at tp>1 and exact-string compare can't tolerate.
fn assert_results_match(
    expected: &[RecordBatch],
    actual: &[RecordBatch],
    rel_tol: Option<f64>,
    query: &str,
) {
    let Some(tol) = rel_tol else {
        assert_eq!(
            batches_to_sorted_str(actual),
            batches_to_sorted_str(expected),
            "CPU executor result for {query} differs from plain DataFusion"
        );
        return;
    };

    use std::collections::HashMap;

    use datafusion::arrow::array::{Array, Float64Array};
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};

    // key (non-float columns, formatted) -> list of the row's Float64 cells.
    fn index(batches: &[RecordBatch]) -> HashMap<String, Vec<Vec<f64>>> {
        let mut m: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
        let opts = FormatOptions::default();
        for b in batches {
            let s = b.schema();
            let floats: Vec<usize> = (0..s.fields().len())
                .filter(|&i| s.field(i).data_type() == &DataType::Float64)
                .collect();
            for r in 0..b.num_rows() {
                let mut key = String::new();
                for c in 0..s.fields().len() {
                    if floats.contains(&c) {
                        continue;
                    }
                    let f = ArrayFormatter::try_new(b.column(c), &opts).unwrap();
                    key.push_str(&f.value(r).to_string());
                    key.push('\u{1}');
                }
                let vals = floats
                    .iter()
                    .map(|&c| {
                        let a = b.column(c).as_any().downcast_ref::<Float64Array>().unwrap();
                        if a.is_null(r) { f64::NAN } else { a.value(r) }
                    })
                    .collect();
                m.entry(key).or_default().push(vals);
            }
        }
        m
    }

    // Stable order for the float-tuples within one key group (NaN treated as equal).
    fn tuple_cmp(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
        for (p, q) in a.iter().zip(b) {
            match p.partial_cmp(q) {
                Some(std::cmp::Ordering::Equal) | None => continue,
                Some(o) => return o,
            }
        }
        std::cmp::Ordering::Equal
    }

    let (mut em, am) = (index(expected), index(actual));
    assert_eq!(
        em.len(),
        am.len(),
        "approx compare: distinct non-float row keys differ for {query} (expected {}, actual {})",
        em.len(),
        am.len()
    );
    for (key, mut avs) in am {
        let mut evs = em
            .remove(&key)
            .unwrap_or_else(|| panic!("approx compare: actual row key absent from expected for {query}"));
        assert_eq!(
            evs.len(),
            avs.len(),
            "approx compare: row multiplicity differs for a key in {query}"
        );
        evs.sort_by(|a, b| tuple_cmp(a, b));
        avs.sort_by(|a, b| tuple_cmp(a, b));
        for (ev, av) in evs.iter().zip(&avs) {
            for (e, a) in ev.iter().zip(av) {
                if e.is_nan() && a.is_nan() {
                    continue;
                }
                let d = (e - a).abs();
                let rel = if *e != 0.0 { d / e.abs() } else { d };
                assert!(
                    rel <= tol,
                    "approx compare: float cell rel diff {rel:.3e} > tol {tol:.0e} for {query} (expected={e}, actual={a})"
                );
            }
        }
    }
}

/// Run a query through plain DataFusion (ground truth) and the CPU executor;
/// assert results match (order-independent) and the cpu cost tree matches golden.
///
/// `rel_tol` is `None` for the exact sorted-string comparison (the default for
/// nearly all queries). A `Some(tol)` is passed ONLY via `cpu_result_approx_test!`
/// for the handful of queries (q39, q14) whose only divergence from the oracle is
/// float summation reassociation (~1 ULP) — see [`assert_results_match`].
pub async fn assert_cpu_results_match_datafusion(
    dataset: &str,
    sf: &str,
    query: &str,
    device: &str,
    rel_tol: Option<f64>,
) {
    let data_dir = data_dir_for(dataset, sf);
    let sql_path = queries_dir_for(dataset).join(format!("{query}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));
    let mut df_ctx = build_session_state(1);
    df_ctx = register_tables_for(df_ctx, &data_dir).await.unwrap();
    let expected = df_ctx.sql(&sql).await.unwrap().collect().await.unwrap();

    // Partitions+budget come from the device label. output_bytes is accounted
    // per-node from total rows + schema (cpu_executor), so the cost golden is
    // reproducible at any partition count; most cpu devices are tp8 (matching the
    // plan device), but LIMIT-without-total-order queries are canonized at tp1
    // (their result row set isn't partition-invariant — see test_cpu_executor.rs).
    let (partitions, budget) = device_config(device);
    let cpu_ctx = create_context_with_tables(&data_dir, partitions, budget).await.unwrap();
    let plan = cpu_ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();
    let mut stats: Vec<NodeMemoryStats> = vec![];
    // Run WITH strict resident control at the device budget. At tp8-mem2gib the
    // peak resident (~135 MB at SF1) is far under 2 GiB, so it never trips and these
    // results are unchanged — this also continuously guards that enforcement is a
    // pure ADDED check that doesn't perturb the 127 real-test outcomes.
    let actual =
        execute_node_by_node_instrumented_enforced(plan.clone(), cpu_ctx.task_ctx(), budget, &mut stats)
            .await
            .unwrap();

    assert_results_match(&expected, &actual, rel_tol, &format!("{dataset}/{query}"));
    assert_cpu_cost_canonical(&plan, &stats, &cpu_golden(dataset, sf, query, device));
}

// --- resident-memory OOM (Part 2) ------------------------------------------
/// Plan + execute `query` at `target_partitions = 8` under STRICT resident control
/// (in-engine, mid-run) at the raw `budget`; return whether execution completed
/// (`Ok`) or was OOM-killed (`Err`). Resident size is the per-node `output_bytes`
/// logical basis (Part-1 metric), independent of the batch size the budget induces.
async fn run_cpu_enforced(query: &str, dataset: &str, sf: &str, budget: usize) -> Result<(), String> {
    let data_dir = data_dir_for(dataset, sf);
    let sql_path = queries_dir_for(dataset).join(format!("{query}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));
    let ctx = create_context_with_tables(&data_dir, TARGET_PARTITIONS, budget).await.unwrap();
    let plan = ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();
    let mut stats: Vec<NodeMemoryStats> = vec![];
    execute_node_by_node_instrumented_enforced(plan, ctx.task_ctx(), budget, &mut stats)
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Assert the query is OOM-killed mid-run (`ResourcesExhausted`) under strict
/// resident control at `budget`. A query that flips pass→OOM moves to this
/// assertion — never disabled.
pub async fn assert_cpu_oom(dataset: &str, sf: &str, query: &str, budget: usize) {
    let err = run_cpu_enforced(query, dataset, sf, budget).await.expect_err(&format!(
        "{dataset}/{query}: expected resident OOM at budget {budget} bytes, but it completed"
    ));
    assert!(
        err.contains("resident GPU memory budget exceeded"),
        "{dataset}/{query}: expected a ResourcesExhausted resident error, got: {err}"
    );
}

/// Assert the query COMPLETES within the resident budget (boundary-passing case:
/// proves the OOM boundary is real, not "everything errors").
pub async fn assert_cpu_fits(dataset: &str, sf: &str, query: &str, budget: usize) {
    let res = run_cpu_enforced(query, dataset, sf, budget).await;
    assert!(
        res.is_ok(),
        "{dataset}/{query}: expected to FIT resident budget {budget} bytes, but it OOM'd: {res:?}"
    );
}

// --- plan/cpu bespoke-test helpers -----------------------------------------
pub async fn make_ctx(budget: usize) -> SessionContext {
    create_context_with_tables(&testdata_dir(), 1, budget).await.unwrap()
}

pub fn test_ctx(data_dir: &Path) -> impl std::future::Future<Output = datafusion::error::Result<SessionContext>> + '_ {
    create_context_with_tables(data_dir, TEST_TARGET_PARTITIONS, TEST_GPU_MEMORY_BUDGET)
}
pub const TEST_TARGET_PARTITIONS: usize = 8;

pub fn has_gpu_node(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.name().starts_with("Gpu") || plan.children().iter().any(|c| has_gpu_node(c))
}

pub fn all_node_names(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
    let mut names = vec![plan.name().to_string()];
    for child in plan.children() {
        names.extend(all_node_names(child));
    }
    names
}

pub fn scan_batch_sizes(plan: &Arc<dyn ExecutionPlan>) -> Vec<usize> {
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

pub fn fmt_plan(plan: &Arc<dyn ExecutionPlan>) -> String {
    DisplayableExecutionPlan::new(plan.as_ref()).indent(true).to_string()
}

pub async fn count(ctx: &SessionContext, query: &str) -> i64 {
    let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .unwrap()
        .value(0)
}

pub fn find_node(plan: &Arc<dyn ExecutionPlan>, name: &str) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.name() == name {
        return Some(plan.clone());
    }
    plan.children().iter().find_map(|c| find_node(c, name))
}

// --- gpu result harness (needs the GPU executor) ---------------------------
#[cfg(not(feature = "rust-only"))]
pub async fn assert_gpu_results_match_cpu(data_dir: &Path, queries_dir: &Path, name: &str) {
    use peacockdb_core::gpu_executor::GpuExecutor;
    use peacockdb_core::CpuExecutor;

    let sql_path = queries_dir.join(format!("{name}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));

    let cpu = CpuExecutor::new(data_dir, 1, GPU_BUDGET).await.unwrap();
    let expected = cpu.execute(&sql).await.unwrap();
    let gpu = GpuExecutor::new(data_dir, 1, GPU_BUDGET).await.unwrap();
    let actual = gpu.execute(&sql).await.unwrap();

    // Empty result sets: compare schema only when both carry a batch (see the
    // empty-vs-zero-row representation note; q17 hits this).
    if total_rows(&expected) == 0 && total_rows(&actual) == 0 {
        if let (Some(e), Some(a)) = (expected.first(), actual.first()) {
            assert_eq!(
                e.schema().fields(),
                a.schema().fields(),
                "GPU executor schema for '{name}' differs from peacock CPU executor (both empty)"
            );
        }
        return;
    }
    assert_eq!(
        batches_to_sorted_str(&actual),
        batches_to_sorted_str(&expected),
        "GPU executor result for '{name}' differs from peacock CPU executor"
    );
}

// --- unified test macros ----------------------------------------------------
/// `query_plan_test!(dataset, sf, query, device)` — all idents/literals; the fn
/// name is derived via paste!, hyphenated path parts come from `_` -> `-`.
#[macro_export]
macro_rules! query_plan_test {
    ($dataset:ident, $sf:literal, $query:ident, $device:ident) => {
        paste::paste! {
            #[tokio::test]
            async fn [<plan_ $dataset _sf $sf _ $query _ $device>]() {
                $crate::common::run_query_test_at(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    &stringify!($device).replace('_', "-"),
                )
                .await;
            }
        }
    };
}

/// `cpu_result_test!(dataset, sf, query, device)` — EXACT result compare.
#[macro_export]
macro_rules! cpu_result_test {
    ($dataset:ident, $sf:literal, $query:ident, $device:ident) => {
        paste::paste! {
            #[tokio::test]
            async fn [<cpu_ $dataset _sf $sf _ $query _ $device>]() {
                $crate::common::assert_cpu_results_match_datafusion(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    &stringify!($device).replace('_', "-"),
                    None,
                )
                .await;
            }
        }
    };
}

/// `cpu_result_approx_test!(dataset, sf, query, device)` — result compare with a
/// relative tolerance of 1e-12 on Float64 columns. ONLY for queries whose sole
/// divergence from the DataFusion oracle is float summation reassociation (~1 ULP)
/// at tp>1. The output_bytes cost golden is still exact (float value doesn't
/// change byte width).
#[macro_export]
macro_rules! cpu_result_approx_test {
    ($dataset:ident, $sf:literal, $query:ident, $device:ident) => {
        paste::paste! {
            #[tokio::test]
            async fn [<cpu_ $dataset _sf $sf _ $query _ $device>]() {
                $crate::common::assert_cpu_results_match_datafusion(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    &stringify!($device).replace('_', "-"),
                    Some(1e-12),
                )
                .await;
            }
        }
    };
}

/// `cpu_result_error_test!(dataset, sf, query, budget)` — strict resident control
/// (Part 2): asserts the query OOMs (`ResourcesExhausted`) at the raw `budget`.
/// Used ONLY by the tight-budget OOM set; a query that flips pass→OOM moves here,
/// it is never disabled. `budget` is raw bytes (no device label).
#[macro_export]
macro_rules! cpu_result_error_test {
    ($dataset:ident, $sf:literal, $query:ident, $budget:expr) => {
        paste::paste! {
            #[tokio::test]
            async fn [<cpu_oom_ $dataset _sf $sf _ $query>]() {
                $crate::common::assert_cpu_oom(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    $budget,
                )
                .await;
            }
        }
    };
}

/// `cpu_result_fits_test!(dataset, sf, query, budget)` — boundary-passing case:
/// asserts the query FITS the same tight `budget` (proves the OOM boundary is real).
#[macro_export]
macro_rules! cpu_result_fits_test {
    ($dataset:ident, $sf:literal, $query:ident, $budget:expr) => {
        paste::paste! {
            #[tokio::test]
            async fn [<cpu_fits_ $dataset _sf $sf _ $query>]() {
                $crate::common::assert_cpu_fits(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    $budget,
                )
                .await;
            }
        }
    };
}

/// `gpu_result_test!(dataset, sf, query, device)` — device H200, no golden.
#[macro_export]
macro_rules! gpu_result_test {
    ($dataset:ident, $sf:literal, $query:ident, $device:ident) => {
        paste::paste! {
            #[tokio::test]
            async fn [<gpu_ $dataset _sf $sf _ $query _ $device>]() {
                $crate::common::assert_gpu_results_match_cpu(
                    &$crate::common::data_dir_for(stringify!($dataset), stringify!($sf)),
                    &$crate::common::queries_dir_for(stringify!($dataset)),
                    &stringify!($query).replace('_', "-"),
                )
                .await;
            }
        }
    };
}
