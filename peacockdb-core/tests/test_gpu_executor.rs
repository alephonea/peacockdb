//! Parameterized GPU-execution tests for TPC-H and TPC-DS. Each
//! `gpu_result_test!(dataset, sf, query, H200)` runs <dataset>-queries/<query>.sql
//! through the GPU executor and compares the result set against peacock's CPU
//! executor (order-independent). No golden — it's a live GPU-vs-CPU equality check.
//! Helpers + the macro live in common/mod.rs; bespoke smoke tests in
//! test_gpu_executor_misc.rs. Disabled TPC-DS queries are grouped at the bottom by
//! failure bucket (issue #29), each with its per-query blocker and ticket ref.
#![cfg(not(feature = "rust-only"))]
// Device label 'H200' is uppercase per dmitry's spec and lives in the derived fn
// names (gpu_<ds>_sf1_qN_H200); keep it rather than lowercasing to satisfy lints.
#![allow(non_snake_case)]
#[macro_use]
mod common;

// ── TPC-H ─────────────────────────────────────────────────────────────────
gpu_result_test!(tpch, 1, scan_limit, H200);
gpu_result_test!(tpch, 1, filter_project, H200);
gpu_result_test!(tpch, 1, aggregate_groupby, H200);
gpu_result_test!(tpch, 1, semi_join, H200);
gpu_result_test!(tpch, 1, anti_join, H200);
gpu_result_test!(tpch, 1, nested_loop_join, H200);
gpu_result_test!(tpch, 1, cross_join, H200);
gpu_result_test!(tpch, 1, q1, H200);
gpu_result_test!(tpch, 1, q2, H200);
gpu_result_test!(tpch, 1, q3, H200);
gpu_result_test!(tpch, 1, q4, H200);
gpu_result_test!(tpch, 1, q5, H200);
gpu_result_test!(tpch, 1, q6, H200);
gpu_result_test!(tpch, 1, q7, H200);
gpu_result_test!(tpch, 1, q8, H200);
gpu_result_test!(tpch, 1, q9, H200);
gpu_result_test!(tpch, 1, q10, H200);
gpu_result_test!(tpch, 1, q11, H200);
gpu_result_test!(tpch, 1, q12, H200);
gpu_result_test!(tpch, 1, q13, H200);
gpu_result_test!(tpch, 1, q14, H200);
gpu_result_test!(tpch, 1, q15, H200); // view inlined as a CTE (see q15.sql)
gpu_result_test!(tpch, 1, q16, H200);
gpu_result_test!(tpch, 1, q17, H200);
gpu_result_test!(tpch, 1, q18, H200);
gpu_result_test!(tpch, 1, q19, H200);
gpu_result_test!(tpch, 1, q20, H200);
gpu_result_test!(tpch, 1, q21, H200);
gpu_result_test!(tpch, 1, q22, H200);

// ── TPC-DS (enabled) ────────────────────────────────────────────────────────
gpu_result_test!(tpcds, 1, q1, H200);
gpu_result_test!(tpcds, 1, q3, H200);
gpu_result_test!(tpcds, 1, q19, H200);
gpu_result_test!(tpcds, 1, q25, H200);
gpu_result_test!(tpcds, 1, q29, H200);
gpu_result_test!(tpcds, 1, q30, H200);
gpu_result_test!(tpcds, 1, q31, H200);
gpu_result_test!(tpcds, 1, q34, H200);
gpu_result_test!(tpcds, 1, q37, H200);
gpu_result_test!(tpcds, 1, q42, H200);
gpu_result_test!(tpcds, 1, q43, H200);
gpu_result_test!(tpcds, 1, q46, H200);
gpu_result_test!(tpcds, 1, q48, H200);
gpu_result_test!(tpcds, 1, q52, H200);
gpu_result_test!(tpcds, 1, q55, H200);
gpu_result_test!(tpcds, 1, q58, H200);
gpu_result_test!(tpcds, 1, q59, H200);
gpu_result_test!(tpcds, 1, q65, H200);
gpu_result_test!(tpcds, 1, q68, H200);
gpu_result_test!(tpcds, 1, q69, H200);
gpu_result_test!(tpcds, 1, q73, H200);
gpu_result_test!(tpcds, 1, q82, H200);
gpu_result_test!(tpcds, 1, q83, H200);
gpu_result_test!(tpcds, 1, q85, H200);
gpu_result_test!(tpcds, 1, q91, H200);
gpu_result_test!(tpcds, 1, q33, H200);
gpu_result_test!(tpcds, 1, q56, H200);
gpu_result_test!(tpcds, 1, q60, H200);
gpu_result_test!(tpcds, 1, q71, H200);
gpu_result_test!(tpcds, 1, q16, H200);
gpu_result_test!(tpcds, 1, q32, H200);
gpu_result_test!(tpcds, 1, q92, H200);
gpu_result_test!(tpcds, 1, q94, H200);
gpu_result_test!(tpcds, 1, q95, H200);
gpu_result_test!(tpcds, 1, q5, H200);
gpu_result_test!(tpcds, 1, q23, H200);
gpu_result_test!(tpcds, 1, q14, H200);
gpu_result_test!(tpcds, 1, q80, H200);
gpu_result_test!(tpcds, 1, q96, H200);
gpu_result_test!(tpcds, 1, q97, H200);
gpu_result_test!(tpcds, 1, q75, H200);
gpu_result_test!(tpcds, 1, q12, H200);
gpu_result_test!(tpcds, 1, q20, H200);
gpu_result_test!(tpcds, 1, q98, H200);
gpu_result_test!(tpcds, 1, q51, H200);
gpu_result_test!(tpcds, 1, q53, H200);
gpu_result_test!(tpcds, 1, q63, H200);
gpu_result_test!(tpcds, 1, q89, H200);
gpu_result_test!(tpcds, 1, q10, H200);
gpu_result_test!(tpcds, 1, q45, H200);
gpu_result_test!(tpcds, 1, q35, H200);
gpu_result_test!(tpcds, 1, q24, H200);
gpu_result_test!(tpcds, 1, q54, H200);
gpu_result_test!(tpcds, 1, q88, H200);
gpu_result_test!(tpcds, 1, q90, H200);
gpu_result_test!(tpcds, 1, q40, H200);
gpu_result_test!(tpcds, 1, q93, H200);
gpu_result_test!(tpcds, 1, q13, H200);
gpu_result_test!(tpcds, 1, q17, H200);
gpu_result_test!(tpcds, 1, q18, H200);
gpu_result_test!(tpcds, 1, q22, H200);
gpu_result_test!(tpcds, 1, q41, H200); // Boolean AST literal
gpu_result_test!(tpcds, 1, q84, H200); // scalar fn: concat
gpu_result_test!(tpcds, 1, q99, H200);
gpu_result_test!(tpcds, 1, q8, H200);
gpu_result_test!(tpcds, 1, q64, H200);
gpu_result_test!(tpcds, 1, q4, H200);
gpu_result_test!(tpcds, 1, q6, H200);
gpu_result_test!(tpcds, 1, q7, H200);
gpu_result_test!(tpcds, 1, q11, H200);
gpu_result_test!(tpcds, 1, q15, H200);
gpu_result_test!(tpcds, 1, q21, H200);
gpu_result_test!(tpcds, 1, q26, H200);
gpu_result_test!(tpcds, 1, q50, H200);
gpu_result_test!(tpcds, 1, q62, H200);
gpu_result_test!(tpcds, 1, q74, H200);
gpu_result_test!(tpcds, 1, q79, H200);
gpu_result_test!(tpcds, 1, q81, H200);

// ============================================================================
// TPC-DS disabled on the GPU — grouped by failure bucket (issue #29). A query can
// move buckets once its first blocker is fixed. Rationale + ticket refs restored
// verbatim from df99bf6; re-enable a line as its bucket is addressed.
// ============================================================================

// --- Bucket C: window functions ---
// q49: BoundedWindowAggExec not yet supported by GpuWindow.
// gpu_result_test!(tpcds, 1, q49, H200);
// rank() windows (StandardWindowExpr) not yet supported by GpuWindow:
// gpu_result_test!(tpcds, 1, q36, H200);
// gpu_result_test!(tpcds, 1, q44, H200);
// gpu_result_test!(tpcds, 1, q47, H200);
// gpu_result_test!(tpcds, 1, q57, H200);
// gpu_result_test!(tpcds, 1, q67, H200);

// --- Bucket D: joins (upstream correctness divergence, not the join) ---
// q77: CrossJoin works, but GPU returns 40 rows vs 45 (an upstream join/aggregate
// drops rows; cross join cannot drop rows) — distinct correctness blocker, not the
// join (issue #47).
// gpu_result_test!(tpcds, 1, q77, H200);
// q61: CrossJoin works (the cross-joined `total` matches CPU), but the `promotions`
// sum subtree produces a wrong value on GPU — distinct upstream correctness blocker,
// not the join (issue #46).
// gpu_result_test!(tpcds, 1, q61, H200);
// q9: the #44 decimal-reduce blocker is gone, but q9 now hits a different one — a
// GpuProject cuDF failure (copying/copy.cu:367) building its top-level CASE of 15
// scalar-subquery comparisons (copy_if_else over a 1-row scalar vs an empty branch).
// Distinct blocker, not count-distinct; parked under #63.
// gpu_result_test!(tpcds, 1, q9, H200);
// q78: scalar fn `round` now executes (q54 confirms round is correct), but q78's
// result still diverges. The divergence is NOT the rounding — it is upstream: q78 is
// a 3-CTE anti-join (LEFT JOIN ... WHERE *_order_number IS NULL) feeding two more
// LEFT JOINs and a `ORDER BY ... DESC ... LIMIT 100` top-N over the rounded ratio.
// Tracked in issue #60; #43 (round) itself is done.
// gpu_result_test!(tpcds, 1, q78, H200);

// --- Bucket E: aggregate gaps ---
// q2: two blockers. (1) The Partial GpuAggregate sums a CASE over a string equality
// (sum(CASE WHEN d_day_name='Sunday' THEN sales_price END)) — a GpuAggregate
// binaryop "Unsupported operator for these types". (2) Even past that, the final
// projection uses round() (round(.../...,2)), an unsupported scalar function
// (issue #43, same as q54/q78). Not a pure aggregate gap.
// gpu_result_test!(tpcds, 1, q2, H200);
// q28: the #44 decimal-reduce blocker is gone (global avg over decimal works now),
// but q28 executes and DIVERGES: it uses count(DISTINCT ss_list_price) alongside
// non-distinct avg/count in the same aggregate, and make_agg ignores the DISTINCT
// flag (counts all rows). Mixed distinct + non-distinct aggregates in one group-by —
// parked under #62.
// gpu_result_test!(tpcds, 1, q28, H200);
// q66: sum(decimal/int) is a two-phase decimal aggregate. DataFusion casts the
// divisor to Decimal128 (__common_expr_1) and evaluates the division only in the
// partial aggregate; our GpuAggregate re-evaluates it against the final-phase input
// (int group key + partial-sum state), so the division operand types don't line up
// (CUDF cast failure). Needs partial/final aggregate handling (issue #55).
// gpu_result_test!(tpcds, 1, q66, H200);
// q39: stddev is mapped (cuDF STD, sample ddof=1), but q39 returns rows and the
// comparison here is exact (pretty-printed string equality); cuDF's STD and
// DataFusion's Welford-based stddev_samp differ in the last float ULP (e.g. cov
// 1.0561770587198125 vs ...123). That ULP fails the string compare and flips the
// ~53 rows whose cov straddles the `cov > 1` filter boundary. Re-enable once the GPU
// harness gains float-tolerant comparison for stddev/variance (proposed ticket).
// gpu_result_test!(tpcds, 1, q39, H200);

// --- Bucket I: set operations / multi-input dedup (result divergence) ---
// q38: INTERSECT (×3) of DISTINCT sets feeding count(*).
// gpu_result_test!(tpcds, 1, q38, H200);
// q76: UNION ALL + IS NULL filters + grouped count(*) — still diverges on GPU
// (executes, wrong result) even after Bucket H. Parked under Bucket I (#59).
// gpu_result_test!(tpcds, 1, q76, H200);
// q87: EXCEPT (×2) of DISTINCT sets feeding count(*) — diverges.
// gpu_result_test!(tpcds, 1, q87, H200);
