//! Flip-audit utility (#13 corpus rollout): decides whether a query is real-8-way
//! FLIP-able or which gate it hits, from its gpu-rules physical plan at tp8. For each
//! query it reports:
//!   - join modes/types + key column types,
//!   - Final-agg mergeability + distinct   → count-distinct/ROLLUP = #11 gate,
//!   - scan-map presence                   → node13-executable prerequisite,
//!   - CollectLeft joins                    → #97-b broadcast defer,
//!   - EVERY repartition key's cuDF TYPE (join AND group-by keys) vs the murmur3
//!     kernel-supported set {STRING, INT8/16/32/64, DATE32, dict-string}. This is the
//!     check CPU-conformance CANNOT make (comet handles all types) and is an IMPERFECT
//!     proxy: it reads the DataFusion type, which can differ from what reaches the GPU
//!     kernel (e.g. cudf::extract_year → INT16 while DataFusion shows Int32; a DATE
//!     group key → cuDF TIMESTAMP_DAYS). Treat a `badRepartKeys` flag as advisory; the
//!     kernel's type_id-printing throw is the on-GPU ground truth.
//! Edit the query list in `diag_flip_audit()` for the bucket under audit.
//! Run: cargo test --features rust-only --test diag_flip_audit -- --nocapture --test-threads=1
use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
use peacockdb_core::gpu_rule::{GpuAggregateExec, GpuHashJoinExec, GpuRepartitionExec, GpuScanExec};
use peacockdb_core::{build_session_state_with_gpu_rules_mode, register_tables_for, PartitionMode};

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata")
}

fn mergeable(f: &str) -> bool {
    matches!(f.to_ascii_lowercase().as_str(),
        "sum"|"count"|"min"|"max"|"avg"|"mean"|"stddev"|"stddev_samp"|"stddev_pop"
        |"var"|"var_samp"|"var_pop"|"variance")
}

// Inc6 GPU spark_hash_partition kernel supports STRING + INT32 + INT64 (+ composite/
// null). Everything else (Date32/Timestamp/Decimal/Float/...) is unsupported → a
// repartition on such a key fails the GPU kernel even though comet (CPU) handles it.
fn murmur3_supported(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Int32 | DataType::Int64)
}

async fn audit(dataset: &str, query: &str) {
    let ctx = build_session_state_with_gpu_rules_mode(8, 120*1024*1024*1024, PartitionMode::RealMultiPartition);
    let ctx = register_tables_for(ctx, &testdata().join(format!("{dataset}.sf1"))).await.unwrap();
    let sql = std::fs::read_to_string(testdata().join(format!("{dataset}-queries/{query}.sql"))).unwrap();
    let plan = match ctx.sql(&sql).await {
        Ok(df) => match df.create_physical_plan().await {
            Ok(p) => p, Err(e) => { eprintln!("[AUDIT {query}] PLAN ERROR: {e}"); return; }
        },
        Err(e) => { eprintln!("[AUDIT {query}] SQL ERROR: {e}"); return; }
    };
    let mut collect_left = false;
    let mut decimal_key = false;
    let mut non_mergeable: Vec<String> = Vec::new();
    let mut distinct = false;
    let mut has_scan_map = false;
    let mut joins: Vec<String> = Vec::new();
    let mut bad_repart: Vec<String> = Vec::new();

    fn walk(p: &Arc<dyn ExecutionPlan>, cl: &mut bool, dk: &mut bool, nm: &mut Vec<String>,
            dist: &mut bool, sm: &mut bool, joins: &mut Vec<String>, br: &mut Vec<String>) {
        if let Some(s) = p.as_any().downcast_ref::<GpuScanExec>() {
            if !s.batches_map().is_empty() { *sm = true; }
        }
        // EVERY repartition key (join-key AND group-by-key Hash repartitions) must be a
        // murmur3-supported type on the GPU — CPU comet handles all types, so this is
        // the check CPU-conformance can't make. (The gap that let q7/q8/q9 through.)
        let rp = p.as_any().downcast_ref::<GpuRepartitionExec>()
            .and_then(|g| g.inner().as_any().downcast_ref::<RepartitionExec>())
            .or_else(|| p.as_any().downcast_ref::<RepartitionExec>());
        if let Some(r) = rp {
            if let Partitioning::Hash(exprs, _n) = r.partitioning() {
                let in_schema = r.input().schema();
                for e in exprs {
                    if let Ok(t) = e.data_type(&in_schema) {
                        if !murmur3_supported(&t) { br.push(format!("{e}:{t:?}")); }
                    }
                }
            }
        }
        let join = p.as_any().downcast_ref::<GpuHashJoinExec>()
            .and_then(|g| g.inner().as_any().downcast_ref::<HashJoinExec>())
            .or_else(|| p.as_any().downcast_ref::<HashJoinExec>());
        if let Some(j) = join {
            let m = format!("{:?}", j.partition_mode());
            if m == "CollectLeft" { *cl = true; }
            let ls = j.left().schema(); let rs = j.right().schema();
            let mut kt = Vec::new();
            for (l, r) in j.on() {
                let lt = l.data_type(&ls).map(|t| format!("{t:?}")).unwrap_or_default();
                let rt = r.data_type(&rs).map(|t| format!("{t:?}")).unwrap_or_default();
                if lt.contains("Decimal") || rt.contains("Decimal") { *dk = true; }
                kt.push(format!("{lt}={rt}"));
            }
            joins.push(format!("{}:{:?}[{}]", m, j.join_type(), kt.join(",")));
        }
        let agg = p.as_any().downcast_ref::<GpuAggregateExec>()
            .and_then(|g| g.inner().as_any().downcast_ref::<AggregateExec>())
            .or_else(|| p.as_any().downcast_ref::<AggregateExec>());
        if let Some(a) = agg {
            let mode = format!("{:?}", a.mode());
            if mode.starts_with("Final") || mode == "Single" || mode == "SinglePartitioned" {
                for e in a.aggr_expr() {
                    if !mergeable(e.fun().name()) { nm.push(e.fun().name().to_string()); }
                    if e.is_distinct() { *dist = true; }
                }
            }
        }
        for c in p.children() { walk(c, cl, dk, nm, dist, sm, joins, br); }
    }
    walk(&plan, &mut collect_left, &mut decimal_key, &mut non_mergeable, &mut distinct,
         &mut has_scan_map, &mut joins, &mut bad_repart);

    let node13 = has_scan_map && non_mergeable.is_empty() && !distinct;
    let verdict = if !node13 { if distinct || !non_mergeable.is_empty() { "GATE(agg→#11)" } else { "GATE(no-scan-map)" } }
                  else if !bad_repart.is_empty() { "GATE(kernel-key-type)" }
                  else if decimal_key { "GATE(decimal→#95)" }
                  else if collect_left { "GATE(CollectLeft→defer)" }
                  else { "FLIP" };
    eprintln!("[AUDIT {query}] {verdict} | scan_map={has_scan_map} collectLeft={collect_left} decimalKey={decimal_key} distinct={distinct} nonMergeable={non_mergeable:?} badRepartKeys={bad_repart:?}");
    for j in &joins { eprintln!("    join {j}"); }
}

#[tokio::test]
async fn diag_flip_audit() {
    // Next-phase audit bucket (edit as the sweep proceeds). q2/q10/q18/q22 are known
    // DECIMAL-key gates (#95); q16 is count-distinct (#11).
    for q in ["q14", "q15", "q20", "q21"] { audit("tpch", q).await; }
}
