//! Parameterized CPU-executor result + cost tests for TPC-H and TPC-DS. Each
//! `cpu_result_test!(dataset, sf, query, device)` runs <dataset>-queries/<query>.sql
//! through plain DataFusion (ground truth) and the CPU executor, asserting the
//! results match and the cost tree matches testdata/goldens/<dataset>.sf<sf>/<query>.<device>.cpu.txt.
//! Helpers + the macro live in common/mod.rs; bespoke tests in test_cpu_executor_misc.rs.
#[macro_use]
mod common;

// ── TPC-H ─────────────────────────────────────────────────────────────────
cpu_result_test!(tpch, 1, hash_join, tp1_mem2gib);
cpu_result_test!(tpch, 1, left_join, tp1_mem2gib);
cpu_result_test!(tpch, 1, mixed_join, tp1_mem2gib);
cpu_result_test!(tpch, 1, scan_limit, tp1_mem2gib);
cpu_result_test!(tpch, 1, filter_project, tp1_mem2gib);
cpu_result_test!(tpch, 1, aggregate_groupby, tp1_mem2gib);
cpu_result_test!(tpch, 1, semi_join, tp1_mem2gib);
cpu_result_test!(tpch, 1, anti_join, tp1_mem2gib);
cpu_result_test!(tpch, 1, nested_loop_join, tp1_mem2gib);
cpu_result_test!(tpch, 1, cross_join, tp1_mem2gib);
cpu_result_test!(tpch, 1, q1, tp1_mem2gib);
cpu_result_test!(tpch, 1, q2, tp1_mem2gib);
cpu_result_test!(tpch, 1, q3, tp1_mem2gib);
cpu_result_test!(tpch, 1, q4, tp1_mem2gib);
cpu_result_test!(tpch, 1, q5, tp1_mem2gib);
cpu_result_test!(tpch, 1, q6, tp1_mem2gib);
cpu_result_test!(tpch, 1, q7, tp1_mem2gib);
cpu_result_test!(tpch, 1, q8, tp1_mem2gib);
cpu_result_test!(tpch, 1, q9, tp1_mem2gib);
cpu_result_test!(tpch, 1, q10, tp1_mem2gib);
cpu_result_test!(tpch, 1, q11, tp1_mem2gib);
cpu_result_test!(tpch, 1, q12, tp1_mem2gib);
cpu_result_test!(tpch, 1, q13, tp1_mem2gib);
cpu_result_test!(tpch, 1, q14, tp1_mem2gib);
cpu_result_test!(tpch, 1, q15, tp1_mem2gib);  // view inlined as a CTE (see q15.sql)
cpu_result_test!(tpch, 1, q16, tp1_mem2gib);
cpu_result_test!(tpch, 1, q17, tp1_mem2gib);
cpu_result_test!(tpch, 1, q18, tp1_mem2gib);
cpu_result_test!(tpch, 1, q19, tp1_mem2gib);
cpu_result_test!(tpch, 1, q20, tp1_mem2gib);
cpu_result_test!(tpch, 1, q21, tp1_mem2gib);
cpu_result_test!(tpch, 1, q22, tp1_mem2gib);

// ── TPC-DS ────────────────────────────────────────────────────────────────
cpu_result_test!(tpcds, 1, q1, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q2, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q3, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q4, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q5, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q6, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q7, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q8, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q9, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q10, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q11, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q12, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q13, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q14, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q15, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q16, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q17, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q18, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q19, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q20, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q21, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q22, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q23, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q24, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q25, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q26, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q28, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q29, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q30, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q31, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q32, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q33, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q34, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q35, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q36, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q37, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q38, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q39, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q40, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q41, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q42, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q43, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q44, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q45, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q46, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q47, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q48, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q49, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q50, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q51, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q52, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q53, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q54, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q55, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q56, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q57, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q58, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q59, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q60, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q61, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q62, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q63, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q64, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q65, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q66, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q67, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q68, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q69, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q71, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q73, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q74, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q75, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q76, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q77, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q78, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q79, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q80, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q81, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q82, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q83, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q84, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q85, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q87, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q88, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q89, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q90, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q91, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q92, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q93, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q94, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q95, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q96, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q97, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q98, tp1_mem2gib);
cpu_result_test!(tpcds, 1, q99, tp1_mem2gib);

// ── TPC-DS disabled — blocked on DataFusion 46+ upgrade (issue #23) ──────────
// These four don't physical-plan under DataFusion 45, so they're also disabled
// in the plan and gpu suites. Re-enable once the DataFusion 46+ upgrade (#23,
// which names these exact queries) lands.
//
// q27: SanityCheckPlan rejects the SortPreservingMergeExec ordering for ROLLUP.
// cpu_result_test!(tpcds, 1, q27, tp1_mem2gib);
// q70: GROUPING() aggregate has no physical-plan support.
// cpu_result_test!(tpcds, 1, q70, tp1_mem2gib);
// q72: Date32 + Int64 type-coercion not supported.
// cpu_result_test!(tpcds, 1, q72, tp1_mem2gib);
// q86: GROUPING() aggregate has no physical-plan support.
// cpu_result_test!(tpcds, 1, q86, tp1_mem2gib);
