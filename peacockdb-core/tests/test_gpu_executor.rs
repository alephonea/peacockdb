//! GPU-execution tests for the TPC-H and TPC-DS suites. Each query is run
//! through the GPU executor and the result set is compared against peacock's
//! CPU executor (order-independent). TPC-H tests use `gpu_result_test!`, TPC-DS
//! tests use `gpu_result_test_tpcds!`; both share the same comparison harness,
//! differing only in the data/query directories.
//!
//! Many TPC-DS queries do not yet execute on the GPU. Those are commented out
//! below and tracked in the TPC-DS GPU-execution ticket (issue #29), grouped
//! into failure buckets; they are re-enabled as the underlying gaps are fixed.

#[cfg(not(feature = "rust-only"))]
mod gpu_executor_tests {
use std::path::{Path, PathBuf};

use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use peacockdb_core::gpu_executor::GpuExecutor;
use peacockdb_core::CpuExecutor;

// Root of the testdata tree. PEACOCK_TESTDATA_DIR overrides the
// compile-time path so the binary can be built on one machine and run
// on another (e.g. shad-gpu), where CARGO_MANIFEST_DIR doesn't exist.
fn testdata_root() -> PathBuf {
    if let Some(d) = std::env::var_os("PEACOCK_TESTDATA_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata")
}

fn testdata_dir() -> PathBuf {
    testdata_root().join("tpch.sf1")
}

fn queries_dir() -> PathBuf {
    testdata_root().join("tpch-queries")
}

fn tpcds_data_dir() -> PathBuf {
    testdata_root().join("tpcds.sf1")
}

fn tpcds_queries_dir() -> PathBuf {
    testdata_root().join("tpcds-queries")
}

const GPU_BUDGET: usize = 2 * 1024 * 1024 * 1024;

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
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

/// Run `name.sql` through both the GPU executor and the peacock CPU
/// executor (`execute_node_by_node` via `CpuExecutor`), then assert that
/// the result sets are equal (order-independent).
async fn assert_gpu_results_match_cpu(data_dir: &Path, queries_dir: &Path, name: &str) {
    let sql_path = queries_dir.join(format!("{name}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));

    // Ground truth: peacock's CPU executor (GPU-annotated plan run on CPU
    // via execute_node_by_node).
    let cpu = CpuExecutor::new(data_dir, 1, GPU_BUDGET).await.unwrap();
    let expected = cpu.execute(&sql).await.unwrap();

    // Subject: GPU executor.
    let gpu = GpuExecutor::new(data_dir, 1, GPU_BUDGET).await.unwrap();
    let actual = gpu.execute(&sql).await.unwrap();

    // Empty result sets render differently depending on whether the executor
    // emits a single zero-row batch (which carries the schema) or an empty
    // batch list (which carries none); `pretty_format_batches` then produces a
    // header-bearing table for one and a bare "++" for the other, so the string
    // compare fails on that formatting detail alone, not on any data.
    //
    // This asymmetry is the normal case here, not a corner case: the GPU
    // executor decodes a zero-row query from its Arrow IPC stream as an *empty
    // Vec* (schema message, zero batch messages -> `vec![]`, no `.first()`),
    // while the peacock CPU executor yields a single zero-row batch that *does*
    // carry the schema. q17 is exactly this shape. We therefore compare schema
    // fields only when *both* sides carry a batch; when exactly one does (the
    // q17 case) there is no second schema to compare against, so we accept the
    // match rather than fail on the empty-vs-zero-row representation gap. When
    // both sides carry a batch a real schema divergence is still caught.
    if total_rows(&expected) == 0 && total_rows(&actual) == 0 {
        if let (Some(e), Some(a)) = (expected.first(), actual.first()) {
            assert_eq!(
                e.schema().fields(),
                a.schema().fields(),
                "GPU executor schema for '{name}' differs from peacock CPU \
                 executor (both result sets are empty)"
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

/// Define a TPC-H GPU-vs-CPU result test (reads `testdata/tpch-queries`).
macro_rules! gpu_result_test {
    ($func_name:ident, $query_name:literal) => {
        #[tokio::test]
        async fn $func_name() {
            assert_gpu_results_match_cpu(&testdata_dir(), &queries_dir(), $query_name).await;
        }
    };
}

/// Define a TPC-DS GPU-vs-CPU result test (reads `testdata/tpcds-queries`).
macro_rules! gpu_result_test_tpcds {
    ($func_name:ident, $query_name:literal) => {
        #[tokio::test]
        async fn $func_name() {
            assert_gpu_results_match_cpu(&tpcds_data_dir(), &tpcds_queries_dir(), $query_name)
                .await;
        }
    };
}

gpu_result_test!(test_gpu_scan_limit, "scan-limit");
gpu_result_test!(test_gpu_filter_project, "filter-project");
gpu_result_test!(test_gpu_aggregate_groupby, "aggregate-groupby");
gpu_result_test!(test_gpu_semi_join, "semi-join");
gpu_result_test!(test_gpu_anti_join, "anti-join");
// CrossJoinExec / NestedLoopJoinExec now supported (issue #12):
gpu_result_test!(test_gpu_nested_loop_join, "nested-loop-join");
gpu_result_test!(test_gpu_cross_join, "cross-join");
gpu_result_test!(test_gpu_tpch_q1, "q1");
gpu_result_test!(test_gpu_tpch_q2, "q2");
gpu_result_test!(test_gpu_tpch_q3, "q3");
gpu_result_test!(test_gpu_tpch_q4, "q4");
gpu_result_test!(test_gpu_tpch_q5, "q5");
gpu_result_test!(test_gpu_tpch_q6, "q6");
gpu_result_test!(test_gpu_tpch_q7, "q7");
gpu_result_test!(test_gpu_tpch_q8, "q8");
gpu_result_test!(test_gpu_tpch_q9, "q9");
gpu_result_test!(test_gpu_tpch_q10, "q10");
// TPC-H q11: correlated HAVING subquery → NestedLoopJoinExec (now supported).
gpu_result_test!(test_gpu_tpch_q11, "q11");
gpu_result_test!(test_gpu_tpch_q12, "q12");
gpu_result_test!(test_gpu_tpch_q13, "q13");
gpu_result_test!(test_gpu_tpch_q14, "q14");
gpu_result_test!(test_gpu_tpch_q15, "q15"); // view inlined as a CTE (see q15.sql)
gpu_result_test!(test_gpu_tpch_q16, "q16");
gpu_result_test!(test_gpu_tpch_q17, "q17");
gpu_result_test!(test_gpu_tpch_q18, "q18");
gpu_result_test!(test_gpu_tpch_q19, "q19");
gpu_result_test!(test_gpu_tpch_q20, "q20");
gpu_result_test!(test_gpu_tpch_q21, "q21");
// TPC-H q22: correlated subquery → NestedLoopJoinExec (now supported).
gpu_result_test!(test_gpu_tpch_q22, "q22");

#[tokio::test]
async fn test_gpu_scan_nation() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    let batches = exec.execute("SELECT * FROM nation").await.unwrap();
    println!("nation rows from GPU: {}", total_rows(&batches));
}

#[tokio::test]
async fn test_gpu_filter_nation() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    let batches = exec
        .execute("SELECT n_name FROM nation WHERE CAST(n_nationkey AS BIGINT) > 5")
        .await
        .unwrap();
    println!("filtered nation rows from GPU: {}", total_rows(&batches));
}

#[tokio::test]
async fn test_gpu_aggregate_count() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    // COUNT(*) alone triggers DataFusion's PlaceholderRowExec (metadata-only
    // row count, no scan). Use SUM to force a real GPU scan + aggregate.
    let batches = exec
        .execute("SELECT SUM(n_nationkey) FROM nation")
        .await
        .unwrap();
    println!("aggregate result rows from GPU: {}", total_rows(&batches));
}

#[tokio::test]
async fn test_gpu_join_nation_region() {
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
async fn test_gpu_sort_nation() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    let batches = exec
        .execute("SELECT n_name FROM nation ORDER BY n_name ASC")
        .await
        .unwrap();
    println!("sorted nation rows from GPU: {}", total_rows(&batches));
}

#[tokio::test]
async fn test_gpu_executor_create_destroy() {
    let exec = GpuExecutor::new(&testdata_dir(), 1, GPU_BUDGET).await.unwrap();
    drop(exec);
}

// =====================================================================
// TPC-DS suite. Compared against the peacock CPU executor on the H200
// (SF-1). Disabled queries are commented out below, grouped by failure
// bucket; the 4 that don't physical-plan are noted inline. See the TPC-DS
// GPU-execution tracking ticket (issue #29) for the analysis.
// =====================================================================

gpu_result_test_tpcds!(test_gpu_tpcds_q1, "q1");
gpu_result_test_tpcds!(test_gpu_tpcds_q3, "q3");
gpu_result_test_tpcds!(test_gpu_tpcds_q19, "q19");
gpu_result_test_tpcds!(test_gpu_tpcds_q25, "q25");
gpu_result_test_tpcds!(test_gpu_tpcds_q29, "q29");
gpu_result_test_tpcds!(test_gpu_tpcds_q30, "q30");
gpu_result_test_tpcds!(test_gpu_tpcds_q31, "q31");
gpu_result_test_tpcds!(test_gpu_tpcds_q34, "q34");
gpu_result_test_tpcds!(test_gpu_tpcds_q37, "q37");
gpu_result_test_tpcds!(test_gpu_tpcds_q42, "q42");
gpu_result_test_tpcds!(test_gpu_tpcds_q43, "q43");
gpu_result_test_tpcds!(test_gpu_tpcds_q46, "q46");
gpu_result_test_tpcds!(test_gpu_tpcds_q48, "q48");
gpu_result_test_tpcds!(test_gpu_tpcds_q52, "q52");
gpu_result_test_tpcds!(test_gpu_tpcds_q55, "q55");
gpu_result_test_tpcds!(test_gpu_tpcds_q58, "q58");
gpu_result_test_tpcds!(test_gpu_tpcds_q59, "q59");
gpu_result_test_tpcds!(test_gpu_tpcds_q65, "q65");
gpu_result_test_tpcds!(test_gpu_tpcds_q68, "q68");
gpu_result_test_tpcds!(test_gpu_tpcds_q69, "q69");
gpu_result_test_tpcds!(test_gpu_tpcds_q73, "q73");
gpu_result_test_tpcds!(test_gpu_tpcds_q82, "q82");
gpu_result_test_tpcds!(test_gpu_tpcds_q83, "q83");
gpu_result_test_tpcds!(test_gpu_tpcds_q85, "q85");
gpu_result_test_tpcds!(test_gpu_tpcds_q91, "q91");
// Bucket A (UnionExec / InterleaveExec → GpuUnion):
gpu_result_test_tpcds!(test_gpu_tpcds_q33, "q33");
gpu_result_test_tpcds!(test_gpu_tpcds_q56, "q56");
gpu_result_test_tpcds!(test_gpu_tpcds_q60, "q60");
gpu_result_test_tpcds!(test_gpu_tpcds_q71, "q71");
// Bucket B (GlobalLimitExec → GpuLimit):
gpu_result_test_tpcds!(test_gpu_tpcds_q16, "q16");
gpu_result_test_tpcds!(test_gpu_tpcds_q32, "q32");
gpu_result_test_tpcds!(test_gpu_tpcds_q92, "q92");
gpu_result_test_tpcds!(test_gpu_tpcds_q94, "q94");
gpu_result_test_tpcds!(test_gpu_tpcds_q95, "q95");

// =====================================================================
// Skipped — failing on the GPU. Buckets per issue #29 (surface analysis;
// a query may move buckets once its first blocker is fixed). Re-enable a
// line as its bucket is addressed.
// =====================================================================

// q5: the Bucket A union-concatenate blocker is now fixed (the executor casts
// each branch to the union's declared output type via GpuUnion.output_schema).
// But q5 then hits a SECOND, separate blocker — its final GROUP BY ROLLUP
// (channel, id) needs grouping-sets aggregation, which the GPU path does not
// implement yet (same gap that parks q14/q18/q22/q36/q67/q70/q80/q86). Re-enable
// once grouping-sets lands.
// gpu_result_test_tpcds!(test_gpu_tpcds_q5, "q5");

// --- Bucket C: window functions (BoundedWindowAggExec) ---
// gpu_result_test_tpcds!(test_gpu_tpcds_q49, "q49");

// --- Bucket D: joins ---
// CrossJoinExec (now supported):
gpu_result_test_tpcds!(test_gpu_tpcds_q23, "q23");
// q77: CrossJoin works, but GPU returns 40 rows vs 45 (an upstream join/aggregate
// drops rows; cross join cannot drop rows) — distinct correctness blocker, not
// the join (issue #47).
// gpu_result_test_tpcds!(test_gpu_tpcds_q77, "q77");
// q28: CrossJoin works, but blocked by a downstream GpuAggregate cuDF failure
// ("Reduction operators other than min/max are not supported for non-arithmetic
// types") — distinct blocker, not a join issue (issue #44).
// gpu_result_test_tpcds!(test_gpu_tpcds_q28, "q28");
// q14: NestedLoopJoin now supported, but q14 also uses GROUP BY ROLLUP — blocked
// by the grouping-sets gap (issue #40).
// gpu_result_test_tpcds!(test_gpu_tpcds_q14, "q14");
// q80: Right join now supported, but q80 also uses GROUP BY ROLLUP — blocked by
// the grouping-sets gap (issue #40).
// gpu_result_test_tpcds!(test_gpu_tpcds_q80, "q80");

// --- Bucket E: aggregate gaps ---
// q2: two blockers. (1) The Partial GpuAggregate sums a CASE over a string
// equality (sum(CASE WHEN d_day_name='Sunday' THEN sales_price END)) — a
// GpuAggregate binaryop "Unsupported operator for these types". (2) Even past
// that, the final projection uses round() (round(.../...,2)), an unsupported
// scalar function (issue #43, same as q54/q78). Not a pure aggregate gap.
// gpu_result_test_tpcds!(test_gpu_tpcds_q2, "q2");

// --- Bucket F: projection / scalar-expr gaps ---
// q96: empty GpuProject feeding count(*) (row-count placeholder).
// q97: IsNotNull in AST.
gpu_result_test_tpcds!(test_gpu_tpcds_q96, "q96");
gpu_result_test_tpcds!(test_gpu_tpcds_q97, "q97");
// --- Bucket E: aggregate gaps ---
// q66: sum(decimal/int) is a two-phase decimal aggregate. DataFusion casts the
// divisor to Decimal128 (__common_expr_1) and evaluates the division only in the
// partial aggregate; our GpuAggregate re-evaluates it against the final-phase
// input (int group key + partial-sum state), so the division operand types don't
// line up (CUDF cast failure). Needs partial/final aggregate handling.
// gpu_result_test_tpcds!(test_gpu_tpcds_q66, "q66");
// --- Bucket I: set operations / multi-input dedup (result divergence) ---
// Execute on GPU but diverge from CPU; root cause beyond projection scope.
// q38: INTERSECT (×3) of DISTINCT sets feeding count(*).
// gpu_result_test_tpcds!(test_gpu_tpcds_q38, "q38");
// q75: UNION (distinct) + COALESCE + decimal-cast self-join; the UNION-DISTINCT
// dedup goes through the GROUP BY path, now null_policy::INCLUDE (Bucket H), so
// it is now GPU-green (zero code).
gpu_result_test_tpcds!(test_gpu_tpcds_q75, "q75");
// q76: UNION ALL + IS NULL filters + grouped count(*) — still diverges on GPU
// (executes, wrong result) even after Bucket H. Parked under Bucket I (#59).
// gpu_result_test_tpcds!(test_gpu_tpcds_q76, "q76");

// --- Bucket C: window functions (GpuWindow node) ---
// Whole-partition aggregate windows now execute on GPU via cudf::grouped_rolling_window.
gpu_result_test_tpcds!(test_gpu_tpcds_q12, "q12");
// q20: the LIMIT-100 top-N NULL-ordering tiebreak is now reconciled by GpuSort
// honoring per-column null_precedence (issue #42).
gpu_result_test_tpcds!(test_gpu_tpcds_q20, "q20");
// q98: same whole-partition-sum pattern as q12; computes correctly (GPU-green).
// It can OOM under shared-H200 memory pressure (a resource flake, not a
// correctness gap) until the analyze_memory GPU-memory modeling lands; that
// flakiness risk is knowingly accepted to keep it enabled.
gpu_result_test_tpcds!(test_gpu_tpcds_q98, "q98");
// rank() windows (StandardWindowExpr) not yet supported by GpuWindow:
// gpu_result_test_tpcds!(test_gpu_tpcds_q36, "q36");
// gpu_result_test_tpcds!(test_gpu_tpcds_q44, "q44");
// gpu_result_test_tpcds!(test_gpu_tpcds_q47, "q47");
// gpu_result_test_tpcds!(test_gpu_tpcds_q57, "q57");
// gpu_result_test_tpcds!(test_gpu_tpcds_q67, "q67");
// Window + Bucket F filter gaps now resolved (q51 → IsNotNull; q53/q63/q89 → abs).
// q89 additionally needs expression sort keys (sum_sales - avg_monthly_sales),
// now materialised via build_column in execute_sort.
gpu_result_test_tpcds!(test_gpu_tpcds_q51, "q51");
gpu_result_test_tpcds!(test_gpu_tpcds_q53, "q53");
gpu_result_test_tpcds!(test_gpu_tpcds_q63, "q63");
gpu_result_test_tpcds!(test_gpu_tpcds_q89, "q89");

// --- Bucket D: joins not yet supported ---
// LeftMark join (now supported):
gpu_result_test_tpcds!(test_gpu_tpcds_q10, "q10");
gpu_result_test_tpcds!(test_gpu_tpcds_q45, "q45");
// q35: final ORDER BY ca_state NULLS FIRST now honored (GpuSort threads
// per-column null_precedence from SortExpr.nulls_first) — issue #42.
gpu_result_test_tpcds!(test_gpu_tpcds_q35, "q35");
// NestedLoopJoinExec (now supported, incl. Inner + Left):
// q9: NestedLoopJoin works (executes past the join), but blocked by a downstream
// GpuAggregate cuDF failure ("Reduction operators other than min/max are not
// supported for non-arithmetic types") — same blocker as q28, not a join issue
// (issue #44).
// gpu_result_test_tpcds!(test_gpu_tpcds_q9, "q9");
// q24: string->string key-cast (s_zip = ca_zip) now a no-op in the CastExpr
// column path (#45).
gpu_result_test_tpcds!(test_gpu_tpcds_q24, "q24");
// q54: scalar fn `round` now implemented in the GpuProject column path (#43).
gpu_result_test_tpcds!(test_gpu_tpcds_q54, "q54");
// CrossJoinExec (now supported):
gpu_result_test_tpcds!(test_gpu_tpcds_q88, "q88");
gpu_result_test_tpcds!(test_gpu_tpcds_q90, "q90");
// q61: CrossJoin works (the cross-joined `total` matches CPU), but the
// `promotions` sum subtree produces a wrong value on GPU — distinct upstream
// correctness blocker, not the join (issue #46).
// gpu_result_test_tpcds!(test_gpu_tpcds_q61, "q61");
// Right join (now supported):
gpu_result_test_tpcds!(test_gpu_tpcds_q40, "q40");
gpu_result_test_tpcds!(test_gpu_tpcds_q93, "q93");
// q78: scalar fn `round` now executes (q54 confirms round is correct), but q78's
// result still diverges. The divergence is NOT the rounding — it is upstream:
// q78 is a 3-CTE anti-join (LEFT JOIN ... WHERE *_order_number IS NULL) feeding
// two more LEFT JOINs and a `ORDER BY ... DESC ... LIMIT 100` top-N over the
// rounded ratio. Left disabled — tracked in issue #60; #43 (round) itself is done.
// gpu_result_test_tpcds!(test_gpu_tpcds_q78, "q78");

// --- Bucket E: aggregate gaps ---
// q13: global avg of an integer column. The compound (mean) reduce now outputs
// FLOAT64 instead of echoing the integer input type, fixing the cuDF
// reductions/compound.cuh failure. GPU-green.
gpu_result_test_tpcds!(test_gpu_tpcds_q13, "q13");
// q17, q39: stddev is now mapped (cuDF STD aggregation, sample ddof=1; two-phase
// Partial/Final handled like AVG — Final is a singleton identity). q17 returns
// 0 rows on the SF1 testdata (the store/return/catalog cross-channel join is
// empty), so it never materialises a stddev value to diverge on; with the
// empty-result handling in `assert_gpu_results_match_cpu` it is GPU-green.
gpu_result_test_tpcds!(test_gpu_tpcds_q17, "q17");
// q39 stays disabled: it returns rows, and the result-set comparison here is
// *exact* (pretty-printed string equality), but cuDF's STD and DataFusion's
// Welford-based stddev_samp differ in the last float ULP (e.g. cov
// 1.0561770587198125 vs 1.0561770587198123). That ULP both fails the string
// compare directly and flips the ~53 rows whose cov straddles the `cov > 1`
// filter boundary. Re-enable once the GPU harness gains float-tolerant
// comparison for stddev/variance columns (proposed ticket).
// gpu_result_test_tpcds!(test_gpu_tpcds_q39, "q39");
// q18, q22: GROUP BY ROLLUP — the executor ignores the grouping_sets mask, so
// these need grouping-sets aggregation (issue #40), not an aggregate-function
// gap. Keep disabled until grouping sets land.
// gpu_result_test_tpcds!(test_gpu_tpcds_q18, "q18");
// gpu_result_test_tpcds!(test_gpu_tpcds_q22, "q22");

// --- Bucket F: projection / scalar-expr gaps ---
gpu_result_test_tpcds!(test_gpu_tpcds_q41, "q41"); // Boolean AST literal
gpu_result_test_tpcds!(test_gpu_tpcds_q84, "q84"); // scalar fn: concat
// q99: lower() + days-bucket; fixed with the Bucket H cuDF SQL-semantics fixes.
gpu_result_test_tpcds!(test_gpu_tpcds_q99, "q99");

// --- Bucket I: set operations (result divergence) ---
// q87: EXCEPT (×2) of DISTINCT sets feeding count(*) — diverges.
// gpu_result_test_tpcds!(test_gpu_tpcds_q87, "q87");

// --- Bucket G: FlatBuffer verification failed (large plans → raised verifier max_depth) ---
gpu_result_test_tpcds!(test_gpu_tpcds_q8, "q8");
// q64: large plan; fixed with the Bucket H cuDF SQL-semantics fixes.
gpu_result_test_tpcds!(test_gpu_tpcds_q64, "q64");

// --- Bucket H: result divergence — fixed by four cuDF SQL-semantics fixes in
// plan_executor.cpp (issue #29): NULL_LOGICAL_AND/OR (3-valued logic),
// groupby null_policy::INCLUDE (NULL is its own group), and join
// null_equality::UNEQUAL (NULL keys don't match). ---
gpu_result_test_tpcds!(test_gpu_tpcds_q4, "q4");
gpu_result_test_tpcds!(test_gpu_tpcds_q6, "q6");
gpu_result_test_tpcds!(test_gpu_tpcds_q7, "q7");
gpu_result_test_tpcds!(test_gpu_tpcds_q11, "q11");
gpu_result_test_tpcds!(test_gpu_tpcds_q15, "q15");
gpu_result_test_tpcds!(test_gpu_tpcds_q21, "q21");
gpu_result_test_tpcds!(test_gpu_tpcds_q26, "q26");
gpu_result_test_tpcds!(test_gpu_tpcds_q50, "q50");
gpu_result_test_tpcds!(test_gpu_tpcds_q62, "q62");
gpu_result_test_tpcds!(test_gpu_tpcds_q74, "q74");
gpu_result_test_tpcds!(test_gpu_tpcds_q79, "q79");
gpu_result_test_tpcds!(test_gpu_tpcds_q81, "q81");

// --- Does not physical-plan (also skipped in the plan tests, not a GPU gap) ---
// q27: ROLLUP ordering rejected by SanityCheckPlan.
// q70, q86: GROUPING() aggregate not physical-planned.
// q72: Date32 + Int64 type coercion.
}
