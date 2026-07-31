//! Shared test harness for the plan / cpu / gpu suites.
//!
//! All helper functions and the unified test macros live here; the suite files
//! (test_query_plan.rs, test_cpu_executor.rs, test_gpu.rs) contain only
//! macro invocations. Each integration-test crate includes this via
//! `#[macro_use] mod common;`. Every suite uses a subset, so dead code is fine.
#![allow(dead_code)]

pub mod cost_model;
pub mod registry;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::{DisplayFormatType, ExecutionPlan};

use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};

use peacockdb_core::config::{MemoryLimit, TargetPartitions};
use peacockdb_core::cpu_executor::NodeMemoryStats;
use peacockdb_core::executors::full_table_cpu_executor::execute_full_table_instrumented_enforced;
use peacockdb_core::gpu_rule::{
    analyze_memory, row_width, GpuAggregateExec, GpuRepartitionExec, GpuScanExec,
};
use peacockdb_core::node_executor::{execute_node_by_node, CpuNodeExecutor};
use peacockdb_core::plan_serializer;
use peacockdb_core::{
    build_session_state, build_session_state_with_gpu_rules_mode, create_context_with_tables,
    create_context_with_tables_mode, register_tables_for, PartitionMode,
};

/// Per-device default [`PartitionMode`] — the map + Hash-repartition-lowering
/// discriminator (dmitry, replacing the old 16 GiB budget threshold). tp8-standard
/// is the real-8-way device; every other label (tp1-*, tp8-mini) is single-
/// partition. The enum — NOT the budget — is now the sole discriminator, so a
/// memory-constrained genuine-8-way device (GitHub #91) would just add its label
/// here → `RealMultiPartition`, no budget change needed.
pub fn partition_mode(device: &str) -> PartitionMode {
    match device {
        "tp8-standard" => PartitionMode::RealMultiPartition,
        _ => PartitionMode::SinglePartition,
    }
}

// --- run configs encoded in the device label -------------------------------
// All single-sourced from `config` — a tier's value must exist in exactly one place,
// or retuning one silently desynchronizes the tests from the executors.
pub const TARGET_PARTITIONS: usize = peacockdb_core::config::TARGET_PARTITIONS; // plan tests
pub const TEST_GPU_MEMORY_BUDGET: usize = MemoryLimit::Mini.bytes();
pub const FULL_BUDGET: usize = MemoryLimit::Mini.bytes();
pub const BATCH_STRESS_BUDGET: usize = peacockdb_core::config::BATCH_STRESS_BUDGET;
pub const GPU_BUDGET: usize = MemoryLimit::Mini.bytes();

/// Max rendered size for a committed `.result.txt` golden. Above this the golden is
/// NOT written (full-result text doesn't scale — e.g. tpch anti-join renders ~240
/// MB / 1.2M rows and trips the repo's push size guard). Large-result queries fall
/// back to the live CPU oracle in the merged GPU test (Inc0.5, dmitry's size rule).
pub const RESULT_GOLDEN_MAX_BYTES: usize = 256 * 1024;

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

/// Total-cost + per-category breakdown golden (#83), derived from the `.cpu.txt`
/// per-node tree via `cost_model::cost_text_from_cpu`; verified by `test_cost_model.rs`.
pub fn cost_golden(dataset: &str, sf: &str, query: &str, device: &str) -> PathBuf {
    golden_dir_for(dataset, sf).join(format!("{query}.{device}.cost.txt"))
}

/// Frozen final-result snapshot (`batches_to_sorted_str`), generated ONLY from the
/// CPU oracle under UPDATE_CANONICAL (never the GPU), so the merged GPU test can
/// assert the final result without a live CPU run (Inc0.5).
pub fn result_golden(dataset: &str, sf: &str, query: &str, device: &str) -> PathBuf {
    golden_dir_for(dataset, sf).join(format!("{query}.{device}.result.txt"))
}

pub fn testdata_dir() -> PathBuf {
    testdata_root().join("tpch.sf1")
}

/// Decode a device label like `tp8-mini` into `(target_partitions, budget_bytes)`.
/// The device dimension is authoritative: the golden path/name AND the run config
/// both derive from it, so a mislabeled test (e.g. a cpu test tagged tp8-…) runs the
/// config it claims rather than silently diverging from its label. (H200 is GPU-only
/// and carries no plan/cpu config, so it never reaches here.)
pub fn device_config(device: &str) -> (usize, usize) {
    let (tp, mem) = device
        .split_once('-')
        .unwrap_or_else(|| panic!("device label '{device}' must look like 'tp<N>-<memtier>'"));
    let partitions = TargetPartitions::from_label(tp)
        .unwrap_or_else(|| panic!("device '{device}': bad partition count in '{tp}'"));
    let budget = MemoryLimit::from_label(mem)
        .unwrap_or_else(|| panic!("device '{device}': unknown memory tier '{mem}'"));
    (partitions.hint(), budget.bytes())
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
    assert_plan_matches_canonical_at(plan, &plan_golden("tpch", "1", name, "tp8-mini"));
}

/// Build the GPU physical plan for `<dataset>-queries/<query>.sql` at `device`'s
/// partition config + [`PartitionMode`] (via [`partition_mode`]). Shared by the
/// plan-canonical test and bespoke serializer tests that need the lowered plan.
pub async fn plan_for(dataset: &str, sf: &str, query: &str, device: &str) -> Arc<dyn ExecutionPlan> {
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
        build_session_state_with_gpu_rules_mode(partitions, budget, partition_mode(device)),
        &data_dir,
    )
    .await
    .unwrap();
    gpu_ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap()
}

/// Plan `<dataset>-queries/<query>.sql` and compare to the plan golden.
pub async fn run_query_test_at(dataset: &str, sf: &str, query: &str, device: &str) {
    let plan = plan_for(dataset, sf, query, device).await;
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
        // partitions=N on every node line (N = OUTPUT partition count; empty
        // part_stats ⇒ N=1). Field order: partitions, output_rows, output_bytes.
        let n = node.stat.part_stats.len().max(1);
        lines.push(format!(
            "{}{}, partitions={}, output_rows={}, output_bytes={}",
            " ".repeat(indent),
            OneLine(node.plan.as_ref()),
            n,
            node.stat.row_count,
            node.stat.output_bytes,
        ));
        // Per-partition sub-lines only when N>1 (tp1 + tp8-mini stay compact).
        if node.stat.part_stats.len() > 1 {
            let sub = " ".repeat(indent + 2);
            for (k, ps) in node.stat.part_stats.iter().enumerate() {
                if !ps.row_groups.is_empty() {
                    // Scan partition: the row groups it reads.
                    let rgs =
                        ps.row_groups.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ");
                    lines.push(format!(
                        "{sub}p{k}: row_groups=[{rgs}] out_rows={} out_bytes={}",
                        ps.out_rows, ps.out_bytes,
                    ));
                } else if node.plan.as_any().is::<GpuRepartitionExec>() {
                    // Hash repartition (Inc2): the child (a lowered GpuCoalescePartitions)
                    // is a SINGLE partition, so there is no child.out_rows[k] to source
                    // an input count from — the shuffle redistributes one table into N.
                    // Per dmitry, render in_rows = out_rows[k] (this output partition's
                    // rows are exactly what flows on); out_rows[k] is the load-bearing
                    // murmur3-fidelity number the GPU must reproduce.
                    lines.push(format!(
                        "{sub}p{k}: in_rows={} out_rows={} out_bytes={}",
                        ps.out_rows, ps.out_rows, ps.out_bytes,
                    ));
                } else {
                    // Non-scan partition: in_rows = the (single) child's out_rows[k]
                    // (count-preserving map op; the child has the same N).
                    let in_rows = node
                        .children
                        .first()
                        .and_then(|c| c.stat.part_stats.get(k))
                        .map(|cp| cp.out_rows)
                        .unwrap_or(0);
                    lines.push(format!(
                        "{sub}p{k}: in_rows={in_rows} out_rows={} out_bytes={}",
                        ps.out_rows, ps.out_bytes,
                    ));
                }
            }
        }
        for child in &node.children {
            walk(child, indent + 2, lines);
        }
    }
    let root = collect(plan, stats, &mut 0);
    let mut lines = Vec::new();
    walk(&root, 0, &mut lines);
    // The total cost (and its per-category breakdown) now lives in the sibling
    // `.cost.txt` golden, derived purely from this tree's text — see
    // `cost_model::cost_text_from_cpu` and `test_cost_model.rs`.
    lines.join("\n")
}

/// CPU-oracle cost-golden assert: WRITES under UPDATE_CANONICAL, else verifies.
/// Only the CPU path may use this (it can write). The GPU path uses the read-only
/// `assert_cost_golden_verify` so a UPDATE_CANONICAL on the GPU host can never
/// overwrite cost goldens from GPU stats (Inc0.5 req#1, extended to cost).
pub fn assert_cpu_cost_canonical(plan: &Arc<dyn ExecutionPlan>, stats: &[NodeMemoryStats], canonical_path: &Path) {
    if std::env::var("UPDATE_CANONICAL").is_ok() {
        let actual = cpu_stats_str(plan, stats);
        std::fs::create_dir_all(canonical_path.parent().unwrap()).unwrap();
        std::fs::write(canonical_path, &actual).unwrap();
        eprintln!("Updated CPU canonical: {}", canonical_path.display());
        return;
    }
    assert_cost_golden_verify(plan, stats, canonical_path);
}

/// Read-only cost-golden verify (NEVER writes, ignores UPDATE_CANONICAL). Used by
/// the GPU path. Fail-closed: a missing golden PANICS.
pub fn assert_cost_golden_verify(plan: &Arc<dyn ExecutionPlan>, stats: &[NodeMemoryStats], canonical_path: &Path) {
    let name = canonical_path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
    let actual = cpu_stats_str(plan, stats);
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
            "result for {query} differs from oracle (exact compare)"
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

/// Write the final-result golden (`batches_to_sorted_str`) under UPDATE_CANONICAL.
/// Called ONLY on the CPU oracle path (Inc0.5 req #1: goldens never come from GPU).
/// A no-op when UPDATE_CANONICAL is unset.
pub fn maybe_write_result_golden(batches: &[RecordBatch], golden_path: &Path) {
    if std::env::var("UPDATE_CANONICAL").is_err() {
        return;
    }
    let s = batches_to_sorted_str(batches);
    if s.len() >= RESULT_GOLDEN_MAX_BYTES {
        // Too large to commit → write NO golden and clear any stale one. The GPU
        // test detects the absent golden and falls back to the live CPU oracle.
        let _ = std::fs::remove_file(golden_path);
        eprintln!(
            "Result too large ({} bytes >= {}); no golden (GPU test uses live oracle): {}",
            s.len(),
            RESULT_GOLDEN_MAX_BYTES,
            golden_path.display()
        );
        return;
    }
    std::fs::create_dir_all(golden_path.parent().unwrap()).unwrap();
    std::fs::write(golden_path, s).unwrap();
    eprintln!("Updated result golden: {}", golden_path.display());
}

/// Float-tolerant comparison of two `batches_to_sorted_str` renderings. The data
/// rows are grouped by their NON-numeric cells (so a ULP difference in a numeric
/// cell can't reorder the sorted lines and break pairing — same idea as
/// `assert_results_match`'s float path), and every numeric cell must agree within
/// `tol` relative error. Used for the result-golden approx path (q14/q39).
fn assert_sorted_str_approx(golden: &str, actual: &str, tol: f64, query: &str) {
    use std::collections::HashMap;

    fn split_cells(line: &str) -> Vec<String> {
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() < 2 {
            return vec![line.trim().to_string()];
        }
        parts[1..parts.len() - 1].iter().map(|c| c.trim().to_string()).collect()
    }
    // (header lines [0..3], data rows as cells). Header/border must match exactly.
    fn parse(s: &str) -> (Vec<String>, Vec<Vec<String>>) {
        let lines: Vec<&str> = s.lines().collect();
        if lines.len() <= 4 {
            return (lines.iter().map(|l| l.to_string()).collect(), vec![]);
        }
        let header = lines[..3].iter().map(|l| l.to_string()).collect();
        let data = lines[3..lines.len() - 1].iter().map(|l| split_cells(l)).collect();
        (header, data)
    }
    // key = non-numeric cells joined; vals = the numeric cells (as f64) per row.
    fn index(rows: &[Vec<String>]) -> HashMap<String, Vec<Vec<f64>>> {
        let mut m: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
        for row in rows {
            let mut key = String::new();
            let mut nums = Vec::new();
            for cell in row {
                match cell.parse::<f64>() {
                    Ok(v) => nums.push(v),
                    Err(_) => {
                        key.push_str(cell);
                        key.push('\u{1}');
                    }
                }
            }
            m.entry(key).or_default().push(nums);
        }
        m
    }
    fn tuple_cmp(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
        for (p, q) in a.iter().zip(b) {
            match p.partial_cmp(q) {
                Some(std::cmp::Ordering::Equal) | None => continue,
                Some(o) => return o,
            }
        }
        std::cmp::Ordering::Equal
    }

    let (gh, gd) = parse(golden);
    let (ah, ad) = parse(actual);
    assert_eq!(gh, ah, "result header/schema for {query} differs from golden");
    let (mut gm, am) = (index(&gd), index(&ad));
    assert_eq!(
        gm.len(),
        am.len(),
        "approx result: distinct non-numeric row keys differ for {query} (golden {}, actual {})",
        gm.len(),
        am.len()
    );
    for (key, mut avs) in am {
        let mut evs = gm
            .remove(&key)
            .unwrap_or_else(|| panic!("approx result: actual row key absent from golden for {query}"));
        assert_eq!(evs.len(), avs.len(), "approx result: row multiplicity differs for a key in {query}");
        evs.sort_by(|a, b| tuple_cmp(a, b));
        avs.sort_by(|a, b| tuple_cmp(a, b));
        for (ev, av) in evs.iter().zip(&avs) {
            assert_eq!(ev.len(), av.len(), "approx result: numeric-cell count differs for {query}");
            for (e, a) in ev.iter().zip(av) {
                let d = (e - a).abs();
                let rel = if *e != 0.0 { d / e.abs() } else { d };
                assert!(
                    rel <= tol,
                    "approx result: cell rel diff {rel:.3e} > tol {tol:.0e} for {query} (golden={e}, actual={a})"
                );
            }
        }
    }
}

/// Assert `actual` matches the frozen result golden at `golden_path` (fail-closed:
/// a missing golden PANICS, mirroring `assert_cpu_cost_canonical`). `rel_tol`
/// `None` = exact sorted-string equality; `Some(tol)` = float-tolerant (q14/q39).
/// Never writes (the GPU side never updates canon — Inc0.5 req #1).
pub fn assert_result_golden(
    actual: &[RecordBatch],
    golden_path: &Path,
    rel_tol: Option<f64>,
    query: &str,
) {
    let golden = std::fs::read_to_string(golden_path).unwrap_or_else(|_| {
        panic!(
            "result golden not found: {}\nGenerate it on the CPU oracle with UPDATE_CANONICAL=1.",
            golden_path.display()
        )
    });
    let golden = golden.trim_end();
    // Empty-result robustness: `[]` (no batch) and `[empty_batch_with_schema]` do
    // NOT render identically via batches_to_sorted_str, and the GPU vs the CPU
    // oracle may pick different empty representations (q17). Per-node cost already
    // verified the structure, so when the GPU produced 0 rows, accept iff the
    // golden also has no data rows — otherwise it's a real divergence.
    if total_rows(actual) == 0 {
        let golden_data_rows = golden.lines().count().saturating_sub(4);
        assert!(
            golden_data_rows == 0,
            "result for {query}: GPU produced 0 rows but golden has {golden_data_rows} data row(s) at {}",
            golden_path.display()
        );
        return;
    }
    let actual_str = batches_to_sorted_str(actual);
    match rel_tol {
        None => assert_eq!(
            actual_str,
            golden.trim_end(),
            "result for {query} does not match golden {}",
            golden_path.display()
        ),
        Some(tol) => assert_sorted_str_approx(golden.trim_end(), &actual_str, tol, query),
    }
}

/// Run a query through plain DataFusion (ground truth) and the CPU executor;
/// assert results match (order-independent) and the cpu cost tree matches golden.
///
/// `rel_tol` is `None` for the exact sorted-string comparison (the default for
/// nearly all queries). A `Some(tol)` is passed ONLY via `cpu_result_approx_test!`
/// for the handful of queries (q39, q14) whose only divergence from the oracle is
/// float summation reassociation (~1 ULP) — see [`assert_results_match`].
/// True if the plan's scan carries a non-empty RG→partition map (tp>1 multi-
/// partition). The map is attached by `GpuMemoryBudgetRule` for EVERY
/// target_partitions>1, so this alone is NOT enough to route to #13.
fn has_scan_map(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if let Some(s) = plan.as_any().downcast_ref::<GpuScanExec>() {
        if !s.batches_map().is_empty() {
            return true;
        }
    }
    plan.children().iter().any(|c| has_scan_map(c))
}

/// True if the plan contains a `GpuRepartitionExec` (Hash/RoundRobin shuffle).
fn has_repartition(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.as_any().is::<GpuRepartitionExec>()
        || plan.children().iter().any(|c| has_repartition(c))
}

/// Whether an aggregate's two-phase STATE is mergeable across hash partitions by a
/// per-bucket Final re-aggregation. SUM/COUNT/MIN/MAX merge trivially (Σ / extremum);
/// AVG merges because its state (sum, count) IS additive — the per-bucket Final does
/// Σsum/Σcount = correct mean, no mean-of-means (Inc4, #25). STDDEV/VAR merge via the
/// Welford [count, mean, m2] state + cuDF MERGE_M2 across buckets (Inc5, #25). Whitelist
/// (not blacklist) so any unrecognized aggregate defaults to NON-mergeable → the query
/// stays on the #11 single-partition path (correct) rather than silently mis-merging on #13.
fn state_mergeable_agg(fun_name: &str) -> bool {
    matches!(
        fun_name.to_ascii_lowercase().as_str(),
        "sum" | "count" | "min" | "max" | "avg" | "mean"
            | "stddev" | "stddev_samp" | "stddev_pop"
            | "var" | "var_samp" | "var_pop" | "variance"
    )
}

/// True iff every multi-partition FINAL-stage aggregate in the plan is state-mergeable
/// (see [`state_mergeable_agg`]). Partial-stage aggregates are unconstrained (their
/// state is merged downstream); only the Final merge is what #13 must be able to
/// combine across the 8 hash partitions.
fn all_final_aggs_state_mergeable(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let here = plan
        .as_any()
        .downcast_ref::<GpuAggregateExec>()
        .and_then(|g| g.inner().as_any().downcast_ref::<AggregateExec>())
        .map(|agg| match agg.mode() {
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                agg.aggr_expr().iter().all(|a| state_mergeable_agg(a.fun().name()))
            }
            _ => true,
        })
        .unwrap_or(true);
    here && plan.children().iter().all(|c| all_final_aggs_state_mergeable(c))
}

/// True iff the plan can be driven by the #13 CpuNodeExecutor: it has a multi-
/// partition scan map AND every multi-partition Final-aggregate is state-mergeable
/// (SUM/COUNT/MIN/MAX/AVG — see [`state_mergeable_agg`]). Inc2 lowers a Hash
/// `GpuRepartitionExec` into GpuCoalescePartitions(M→1) + GpuRepartition(1→N) and
/// hash-partitions via Spark-murmur3, so every group lands wholly in one bucket; the
/// per-bucket Final then merges that group's state — Σ for sum/count, Σsum/Σcount for
/// AVG (Inc4, #25). A STDDEV/VAR Final-agg still can't be merged (compound moments,
/// Inc5) → those queries stay on the #11 instrumented-enforced path (correct single-
/// partition goldens). Gate on AGG-KIND, NOT on the mere presence of a
/// GpuRepartitionExec (which would wrongly admit a STDDEV/VAR shuffle).
pub(crate) fn plan_is_node13_executable(plan: &Arc<dyn ExecutionPlan>) -> bool {
    has_scan_map(plan) && all_final_aggs_state_mergeable(plan)
}

pub async fn assert_cpu_results_match_datafusion(
    dataset: &str,
    sf: &str,
    query: &str,
    device: &str,
    rel_tol: Option<f64>,
    use_node13: bool,
    gen_result_golden: bool,
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
    let cpu_ctx =
        create_context_with_tables_mode(&data_dir, partitions, budget, partition_mode(device))
            .await
            .unwrap();
    let plan = cpu_ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();

    // Executor choice is EXPLICIT per-test (`use_node13`), NOT inferred from the plan:
    // the SAME plan (e.g. q6 at 8 partitions) backs BOTH the tp8-mini golden — the
    // #11 instrumented-enforced executor, which streams each node single-partition-
    // coalesced regardless of target_partitions — AND the tp8-standard golden — the
    // #13 CpuNodeExecutor, which maintains N partitions across nodes (partial-agg =
    // Σ-over-partitions, CoalescePartitions concat N→1), matching the real 8-way GPU.
    // So a plan-only predicate can't tell them apart; only the device/test does.
    // `plan_is_node13_executable` is asserted here purely as a SAFETY guard: a query
    // opted into #13 MUST have a scan map and only additive Final-aggregates (a Hash
    // repartition is lowered + Spark-murmur3 partitioned in Inc2; AVG/STDDEV/VAR merge
    // waits for Inc4/Inc5). #11 also backs the resident-OOM tests. Both yield
    // post-order stats + a coalesced result, so the assertions below are identical.
    let (actual, stats): (Vec<RecordBatch>, Vec<NodeMemoryStats>) = if use_node13 {
        assert!(
            plan_is_node13_executable(&plan),
            "{dataset}/{query} @ {device}: #13 requested but plan is not node13-executable \
             (missing scan map, or a Final-agg uses AVG/STDDEV/VAR — non-additive merge is Inc4/Inc5)"
        );
        let mut backend = CpuNodeExecutor::new(cpu_ctx.task_ctx());
        execute_node_by_node(&plan, &mut backend).await.unwrap()
    } else {
        let mut stats: Vec<NodeMemoryStats> = vec![];
        let actual = execute_full_table_instrumented_enforced(
            plan.clone(),
            cpu_ctx.task_ctx(),
            budget,
            &mut stats,
        )
        .await
        .unwrap();
        (actual, stats)
    };

    assert_results_match(&expected, &actual, rel_tol, &format!("{dataset}/{query}"));
    assert_cpu_cost_canonical(&plan, &stats, &cpu_golden(dataset, sf, query, device));
    // Snapshot the (DataFusion-validated) result for the merged GPU test to verify
    // against, so the GPU run needs no live CPU oracle (Inc0.5). UPDATE_CANONICAL only.
    //
    // GATED on `gen_result_golden`: write the `.result.txt` ONLY when a golden-
    // asserting `gpu_test!` (GoldenExact/GoldenApprox) actually consumes this
    // (query, device). INVARIANT: `gen_result_golden` must be TRUE exactly for the
    // (query, device) pairs that have such a consumer — TRUE for tp1-standard
    // golden_exact/approx + the tp8-standard real-partitioning goldens (q6,
    // shuffle_additive); FALSE for tp8-mini (no gpu_test! consumer) and for
    // oracle-mode queries (>256KB result → GPU uses the live oracle, no golden).
    // false-when-should-be-true = missing golden = the GPU test fails loud (safe);
    // true-when-should-be-false = an orphan golden written but never read (the
    // silent case this gate exists to prevent).
    if gen_result_golden {
        maybe_write_result_golden(&actual, &result_golden(dataset, sf, query, device));
    }
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
    execute_full_table_instrumented_enforced(plan, ctx.task_ctx(), budget, &mut stats)
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

/// GPU node-by-node verification (Task #13): run the query through the GPU
/// node-executor interface and assert its per-node stats (exact row counts + the
/// rows+schema-derived cost) match the CPU-emulated `.cpu.txt` golden for `device`.
/// The golden is produced by the CPU path; this just compares the GPU run against
/// it (never UPDATE_CANONICAL from GPU). Use the H200/tp1 device (tp1-standard):
/// at tp1 the plan is single-partition, so GPU and CPU emulation share node
/// structure + row counts exactly (no partial-aggregate divergence).
#[cfg(not(feature = "rust-only"))]
pub async fn assert_gpu_nodes_match_golden(dataset: &str, sf: &str, query: &str, device: &str) {
    use peacockdb_core::gpu_executor::GpuExecutor;

    let data_dir = data_dir_for(dataset, sf);
    let sql_path = queries_dir_for(dataset).join(format!("{query}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));
    let (partitions, budget) = device_config(device);
    let gpu = GpuExecutor::new_mode(&data_dir, partitions, budget, partition_mode(device))
        .await
        .unwrap();
    let (_batches, plan, stats) = gpu.execute_instrumented(&sql).await.unwrap();
    assert_cost_golden_verify(&plan, &stats, &cpu_golden(dataset, sf, query, device));
}

/// Per-query result-assertion mode for the merged GPU test (`gpu_test!`) — chosen
/// EXPLICITLY at each call site so a reader sees, per query, golden vs live oracle.
/// `GoldenExact`  = static result-golden, exact compare (fail-closed: missing panics).
/// `GoldenApprox` = static result-golden, 1e-12 float-tolerant (q14/q39 — avg/sum
///                  summation reassociation across partitions, ~1 ULP).
/// `GoldenApproxStddev` = static result-golden, 1e-11 float-tolerant. STDDEV/VAR ONLY:
///                  cuDF's variance algorithm (Σ(x-x̄)² then sqrt) accumulates more
///                  float error than sum/avg, and cuDF make_std/MERGE_M2 vs DataFusion's
///                  Welford diverge ~2e-12 (observed 1.996e-12) — beyond the 1e-12
///                  convention. 1e-11 = ~5× headroom, still 11 significant digits. The
///                  CPU-side #13 approx tests stay at 1e-12 (CPU#13 == DataFusion at
///                  ~3e-14); this looser tol is GPU-cuDF-specific. See #94-adjacent.
/// `Oracle`       = live CPU-oracle compare, NO golden — for results too large to
///                  commit as text (>= RESULT_GOLDEN_MAX_BYTES, e.g. anti-join's
///                  ~240MB/1.2M rows). R4 preserved: still result-validated, live.
/// `Skip`         = per-node only (non-deterministic LIMIT; tp8-only escape).
#[cfg(not(feature = "rust-only"))]
#[derive(Clone, Copy)]
pub enum GpuResultMode {
    GoldenExact,
    GoldenApprox,
    GoldenApproxStddev,
    Oracle,
    Skip,
}

/// Map a mode keyword (from `gpu_test!`) to a `GpuResultMode`.
#[cfg(not(feature = "rust-only"))]
pub fn gpu_result_mode(s: &str) -> GpuResultMode {
    match s {
        "golden_exact" => GpuResultMode::GoldenExact,
        "golden_approx" => GpuResultMode::GoldenApprox,
        "golden_approx_std" => GpuResultMode::GoldenApproxStddev,
        "oracle" => GpuResultMode::Oracle,
        "skip" => GpuResultMode::Skip,
        other => panic!(
            "gpu_test!: unknown result mode '{other}' (expected golden_exact|golden_approx|golden_approx_std|oracle|skip)"
        ),
    }
}

/// Merged per-query GPU verification (Task #13 Phase 2, C2): a SINGLE GPU run
/// (the node-by-node executor, which also materializes the final result) asserts
/// BOTH (a) per-node exact rows + rows/schema cost vs the `.cpu.txt` golden
/// (ALWAYS), AND (b) the final RESULT vs the peacock CPU oracle (per `result_mode`).
/// Replaces the separate node-only + result-only GPU tests with one execution.
#[cfg(not(feature = "rust-only"))]
pub async fn assert_gpu_query(
    dataset: &str,
    sf: &str,
    query: &str,
    device: &str,
    result_mode: GpuResultMode,
) {
    use peacockdb_core::gpu_executor::GpuExecutor;
    use peacockdb_core::CpuExecutor;

    let data_dir = data_dir_for(dataset, sf);
    let sql_path = queries_dir_for(dataset).join(format!("{query}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));
    let (partitions, budget) = device_config(device);
    let qlabel = format!("{dataset}/{query}");

    // ONE GPU execution → final batches + plan + per-node stats.
    let gpu = GpuExecutor::new_mode(&data_dir, partitions, budget, partition_mode(device))
        .await
        .unwrap();
    let (actual, plan, stats) = gpu.execute_instrumented(&sql).await.unwrap();

    // (a) per-node rows + rows/schema cost vs the golden — ALWAYS (fail-closed,
    //     READ-ONLY: the GPU side must never write/overwrite a cost golden).
    assert_cost_golden_verify(&plan, &stats, &cpu_golden(dataset, sf, query, device));

    // (b) final result — dispatch on the explicitly-declared mode.
    match result_mode {
        GpuResultMode::Skip => {}
        GpuResultMode::GoldenExact => assert_result_golden(
            &actual,
            &result_golden(dataset, sf, query, device),
            None,
            &qlabel,
        ),
        GpuResultMode::GoldenApprox => assert_result_golden(
            &actual,
            &result_golden(dataset, sf, query, device),
            Some(1e-12),
            &qlabel,
        ),
        // STDDEV/VAR: 1e-11 (vs the 1e-12 convention) — cuDF's variance algorithm
        // diverges from DataFusion's Welford by ~2e-12, more than sum/avg. See the
        // GpuResultMode::GoldenApproxStddev doc.
        GpuResultMode::GoldenApproxStddev => assert_result_golden(
            &actual,
            &result_golden(dataset, sf, query, device),
            Some(1e-11),
            &qlabel,
        ),
        GpuResultMode::Oracle => {
            // Result too large to commit as a golden → validate against a LIVE CPU
            // oracle run (exact). Still result-validated (R4), just not frozen.
            let cpu = CpuExecutor::new_mode(&data_dir, partitions, budget, partition_mode(device))
                .await
                .unwrap();
            let expected = cpu.execute(&sql).await.unwrap();
            if total_rows(&expected) == 0 && total_rows(&actual) == 0 {
                if let (Some(e), Some(a)) = (expected.first(), actual.first()) {
                    assert_eq!(
                        e.schema().fields(),
                        a.schema().fields(),
                        "GPU result schema for {qlabel} differs from peacock CPU (both empty)"
                    );
                }
                return;
            }
            assert_results_match(&expected, &actual, None, &qlabel);
        }
    }
}

// --- unified test macros ----------------------------------------------------
/// Submit one [`common::registry::RegistryEntry`] for a test-macro invocation.
///
/// Called by every unified macro below, so the registry records what the suite
/// actually declares. `inventory` collects per linked binary, which is why the
/// verify step is per-binary — see `common/registry.rs`.
#[macro_export]
macro_rules! register_test {
    ($kind:literal, $dataset:ident, $sf:literal, $query:ident, $device:ident, $state:expr) => {
        ::inventory::submit! {
            $crate::common::registry::RegistryEntry {
                kind: $kind,
                dataset: stringify!($dataset),
                sf: stringify!($sf),
                query: stringify!($query),
                device: stringify!($device),
                state: $state,
            }
        }
    };
}

/// Map a `gpu_test!` mode keyword to its registry state, at COMPILE time.
///
/// `skip` means the GPU runs but its result is not validated (`~` in the widget);
/// every other mode validates and counts as `enabled`. Two macro arms rather than a
/// runtime match, because `inventory::submit!` needs a const field.
#[macro_export]
macro_rules! gpu_mode_state {
    (skip) => {
        "skip"
    };
    ($other:ident) => {
        "enabled"
    };
}

/// `query_plan_test!(dataset, sf, query, device)` — all idents/literals; the fn
/// name is derived via paste!, hyphenated path parts come from `_` -> `-`.
#[macro_export]
macro_rules! query_plan_test {
    ($dataset:ident, $sf:literal, $query:ident, $device:ident) => {
        $crate::register_test!("plan", $dataset, $sf, $query, $device, "enabled");
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

/// `cpu_result_test!(dataset, sf, query, device, gen_result_golden)` — EXACT result
/// compare, #11 (instrumented-enforced, single-partition-coalesced) executor.
/// `gen_result_golden` (bool literal): write the `.result.txt` golden under
/// UPDATE_CANONICAL only when a golden-asserting `gpu_test!` consumes it (TRUE for
/// tp1-standard golden_exact; FALSE for tp8-mini and oracle-mode).
#[macro_export]
macro_rules! cpu_result_test {
    ($dataset:ident, $sf:literal, $query:ident, $device:ident, $gen:literal) => {
        $crate::register_test!("ftc", $dataset, $sf, $query, $device, "enabled");
        paste::paste! {
            #[tokio::test]
            async fn [<cpu_ $dataset _sf $sf _ $query _ $device>]() {
                $crate::common::assert_cpu_results_match_datafusion(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    &stringify!($device).replace('_', "-"),
                    None,
                    false, // #11 executor
                    $gen,
                )
                .await;
            }
        }
    };
}

/// `cpu_node13_result_test!(dataset, sf, query, device, gen_result_golden)` — EXACT
/// result compare, driven by the #13 multi-handle CpuNodeExecutor (real N-partition,
/// Σ-over-partitions cost). EXPLICIT opt-in (not plan-inferred): a query routed here
/// MUST be node13-executable (scan map + only additive Final-aggregates; asserted at
/// runtime). Used for the H200/tp8 device (tp8-standard); AVG/STDDEV/VAR queries
/// wait for Inc4/Inc5. The SAME plan at tp8-mini stays on #11 via `cpu_result_test!`.
#[macro_export]
macro_rules! cpu_node13_result_test {
    ($dataset:ident, $sf:literal, $query:ident, $device:ident, $gen:literal) => {
        $crate::register_test!("node13", $dataset, $sf, $query, $device, "enabled");
        paste::paste! {
            #[tokio::test]
            async fn [<cpu_ $dataset _sf $sf _ $query _ $device>]() {
                $crate::common::assert_cpu_results_match_datafusion(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    &stringify!($device).replace('_', "-"),
                    None,
                    true, // #13 CpuNodeExecutor
                    $gen,
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
    ($dataset:ident, $sf:literal, $query:ident, $device:ident, $gen:literal) => {
        $crate::register_test!("ftc", $dataset, $sf, $query, $device, "enabled");
        paste::paste! {
            #[tokio::test]
            async fn [<cpu_ $dataset _sf $sf _ $query _ $device>]() {
                $crate::common::assert_cpu_results_match_datafusion(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    &stringify!($device).replace('_', "-"),
                    Some(1e-12),
                    false, // #11 executor
                    $gen,
                )
                .await;
            }
        }
    };
}

/// `cpu_node13_result_approx_test!(dataset, sf, query, device, gen)` — like
/// [`cpu_node13_result_test!`] (the #13 real-N-partition CpuNodeExecutor) but with a
/// 1e-12 relative tolerance on Float64 columns. Required for STDDEV/VAR queries (Inc5):
/// the Welford M2 state, merged across the 8 hash partitions, reassociates float
/// summation (~1 ULP; ~3e-14 rel) vs the DataFusion single-pass oracle, so exact-string
/// compare can't be used. The output_bytes cost golden stays exact (float byte width
/// is unchanged). `gen` writes the `.result.txt` iff a golden `gpu_test!` consumes it.
#[macro_export]
macro_rules! cpu_node13_result_approx_test {
    ($dataset:ident, $sf:literal, $query:ident, $device:ident, $gen:literal) => {
        $crate::register_test!("node13", $dataset, $sf, $query, $device, "enabled");
        paste::paste! {
            #[tokio::test]
            async fn [<cpu_ $dataset _sf $sf _ $query _ $device>]() {
                $crate::common::assert_cpu_results_match_datafusion(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    &stringify!($device).replace('_', "-"),
                    Some(1e-12),
                    true, // #13 CpuNodeExecutor
                    $gen,
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

/// `gpu_test!(dataset, sf, query, device, mode)` — the MERGED GPU test (Phase 2
/// C2): one GPU run asserts per-node rows+cost vs the `.cpu.txt` golden AND the
/// final result. `mode` ∈ { golden_exact | golden_approx | oracle | skip } (see
/// `GpuResultMode`). Derived fn name `gpu_<ds>_sf<sf>_<query>_<device>` matches the
/// former gpu_result_test names so CI/--exact filters keep working. (Replaces the
/// old gpu_node_test! + gpu_result_test! macros, removed in Inc0.)
#[macro_export]
macro_rules! gpu_test {
    ($dataset:ident, $sf:literal, $query:ident, $device:ident, $mode:ident) => {
        #[cfg(not(feature = "rust-only"))]
        $crate::register_test!("gpu", $dataset, $sf, $query, $device,
                               $crate::gpu_mode_state!($mode));
        paste::paste! {
            #[cfg(not(feature = "rust-only"))]
            #[tokio::test]
            async fn [<gpu_ $dataset _sf $sf _ $query _ $device>]() {
                $crate::common::assert_gpu_query(
                    stringify!($dataset),
                    stringify!($sf),
                    &stringify!($query).replace('_', "-"),
                    &stringify!($device).replace('_', "-"),
                    $crate::common::gpu_result_mode(stringify!($mode)),
                )
                .await;
            }
        }
    };
}
