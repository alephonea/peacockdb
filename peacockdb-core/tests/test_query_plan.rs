//! Parameterized GPU plan-canonical tests for TPC-H and TPC-DS. Each
//! `query_plan_test!(dataset, sf, query, device)` plans <dataset>-queries/<query>.sql
//! and compares it to testdata/goldens/<dataset>.sf<sf>/<query>.<device>.plan.txt.
//! Helpers + the macro live in common/mod.rs; bespoke tests in test_query_plan_misc.rs.
#[macro_use]
mod common;

// ── TPC-H ─────────────────────────────────────────────────────────────────
query_plan_test!(tpch, 1, scan_limit, tp8_mem2gib);
query_plan_test!(tpch, 1, filter_project, tp8_mem2gib);
query_plan_test!(tpch, 1, aggregate_groupby, tp8_mem2gib);
query_plan_test!(tpch, 1, hash_join, tp8_mem2gib);
query_plan_test!(tpch, 1, left_join, tp8_mem2gib);
query_plan_test!(tpch, 1, semi_join, tp8_mem2gib);
query_plan_test!(tpch, 1, anti_join, tp8_mem2gib);
query_plan_test!(tpch, 1, nested_loop_join, tp8_mem2gib);
query_plan_test!(tpch, 1, mixed_join, tp8_mem2gib);
query_plan_test!(tpch, 1, cross_join, tp8_mem2gib);
query_plan_test!(tpch, 1, q1, tp8_mem2gib);
query_plan_test!(tpch, 1, q2, tp8_mem2gib);
query_plan_test!(tpch, 1, q3, tp8_mem2gib);
query_plan_test!(tpch, 1, q4, tp8_mem2gib);
query_plan_test!(tpch, 1, q5, tp8_mem2gib);
query_plan_test!(tpch, 1, q6, tp8_mem2gib);
query_plan_test!(tpch, 1, q7, tp8_mem2gib);
query_plan_test!(tpch, 1, q8, tp8_mem2gib);
query_plan_test!(tpch, 1, q9, tp8_mem2gib);
query_plan_test!(tpch, 1, q10, tp8_mem2gib);
query_plan_test!(tpch, 1, q11, tp8_mem2gib);
query_plan_test!(tpch, 1, q12, tp8_mem2gib);
query_plan_test!(tpch, 1, q13, tp8_mem2gib);
query_plan_test!(tpch, 1, q14, tp8_mem2gib);
query_plan_test!(tpch, 1, q15, tp8_mem2gib);
query_plan_test!(tpch, 1, q16, tp8_mem2gib);
query_plan_test!(tpch, 1, q17, tp8_mem2gib);
query_plan_test!(tpch, 1, q18, tp8_mem2gib);
query_plan_test!(tpch, 1, q19, tp8_mem2gib);
query_plan_test!(tpch, 1, q20, tp8_mem2gib);
query_plan_test!(tpch, 1, q21, tp8_mem2gib);
query_plan_test!(tpch, 1, q22, tp8_mem2gib);

// ── TPC-DS ────────────────────────────────────────────────────────────────
query_plan_test!(tpcds, 1, q1, tp8_mem2gib);
query_plan_test!(tpcds, 1, q2, tp8_mem2gib);
query_plan_test!(tpcds, 1, q3, tp8_mem2gib);
query_plan_test!(tpcds, 1, q4, tp8_mem2gib);
query_plan_test!(tpcds, 1, q5, tp8_mem2gib);
query_plan_test!(tpcds, 1, q6, tp8_mem2gib);
query_plan_test!(tpcds, 1, q7, tp8_mem2gib);
query_plan_test!(tpcds, 1, q8, tp8_mem2gib);
query_plan_test!(tpcds, 1, q9, tp8_mem2gib);
query_plan_test!(tpcds, 1, q10, tp8_mem2gib);
query_plan_test!(tpcds, 1, q11, tp8_mem2gib);
query_plan_test!(tpcds, 1, q12, tp8_mem2gib);
query_plan_test!(tpcds, 1, q13, tp8_mem2gib);
query_plan_test!(tpcds, 1, q14, tp8_mem2gib);
query_plan_test!(tpcds, 1, q15, tp8_mem2gib);
query_plan_test!(tpcds, 1, q16, tp8_mem2gib);
query_plan_test!(tpcds, 1, q17, tp8_mem2gib);
query_plan_test!(tpcds, 1, q18, tp8_mem2gib);
query_plan_test!(tpcds, 1, q19, tp8_mem2gib);
query_plan_test!(tpcds, 1, q20, tp8_mem2gib);
query_plan_test!(tpcds, 1, q21, tp8_mem2gib);
query_plan_test!(tpcds, 1, q22, tp8_mem2gib);
query_plan_test!(tpcds, 1, q23, tp8_mem2gib);
query_plan_test!(tpcds, 1, q24, tp8_mem2gib);
query_plan_test!(tpcds, 1, q25, tp8_mem2gib);
query_plan_test!(tpcds, 1, q26, tp8_mem2gib);
query_plan_test!(tpcds, 1, q28, tp8_mem2gib);
query_plan_test!(tpcds, 1, q29, tp8_mem2gib);
query_plan_test!(tpcds, 1, q30, tp8_mem2gib);
query_plan_test!(tpcds, 1, q31, tp8_mem2gib);
query_plan_test!(tpcds, 1, q32, tp8_mem2gib);
query_plan_test!(tpcds, 1, q33, tp8_mem2gib);
query_plan_test!(tpcds, 1, q34, tp8_mem2gib);
query_plan_test!(tpcds, 1, q35, tp8_mem2gib);
query_plan_test!(tpcds, 1, q36, tp8_mem2gib);
query_plan_test!(tpcds, 1, q37, tp8_mem2gib);
query_plan_test!(tpcds, 1, q38, tp8_mem2gib);
query_plan_test!(tpcds, 1, q39, tp8_mem2gib);
query_plan_test!(tpcds, 1, q40, tp8_mem2gib);
query_plan_test!(tpcds, 1, q41, tp8_mem2gib);
query_plan_test!(tpcds, 1, q42, tp8_mem2gib);
query_plan_test!(tpcds, 1, q43, tp8_mem2gib);
query_plan_test!(tpcds, 1, q44, tp8_mem2gib);
query_plan_test!(tpcds, 1, q45, tp8_mem2gib);
query_plan_test!(tpcds, 1, q46, tp8_mem2gib);
query_plan_test!(tpcds, 1, q47, tp8_mem2gib);
query_plan_test!(tpcds, 1, q48, tp8_mem2gib);
query_plan_test!(tpcds, 1, q49, tp8_mem2gib);
query_plan_test!(tpcds, 1, q50, tp8_mem2gib);
query_plan_test!(tpcds, 1, q51, tp8_mem2gib);
query_plan_test!(tpcds, 1, q52, tp8_mem2gib);
query_plan_test!(tpcds, 1, q53, tp8_mem2gib);
query_plan_test!(tpcds, 1, q54, tp8_mem2gib);
query_plan_test!(tpcds, 1, q55, tp8_mem2gib);
query_plan_test!(tpcds, 1, q56, tp8_mem2gib);
query_plan_test!(tpcds, 1, q57, tp8_mem2gib);
query_plan_test!(tpcds, 1, q58, tp8_mem2gib);
query_plan_test!(tpcds, 1, q59, tp8_mem2gib);
query_plan_test!(tpcds, 1, q60, tp8_mem2gib);
query_plan_test!(tpcds, 1, q61, tp8_mem2gib);
query_plan_test!(tpcds, 1, q62, tp8_mem2gib);
query_plan_test!(tpcds, 1, q63, tp8_mem2gib);
query_plan_test!(tpcds, 1, q64, tp8_mem2gib);
query_plan_test!(tpcds, 1, q65, tp8_mem2gib);
query_plan_test!(tpcds, 1, q66, tp8_mem2gib);
query_plan_test!(tpcds, 1, q67, tp8_mem2gib);
query_plan_test!(tpcds, 1, q68, tp8_mem2gib);
query_plan_test!(tpcds, 1, q69, tp8_mem2gib);
query_plan_test!(tpcds, 1, q71, tp8_mem2gib);
query_plan_test!(tpcds, 1, q73, tp8_mem2gib);
query_plan_test!(tpcds, 1, q74, tp8_mem2gib);
query_plan_test!(tpcds, 1, q75, tp8_mem2gib);
query_plan_test!(tpcds, 1, q76, tp8_mem2gib);
query_plan_test!(tpcds, 1, q77, tp8_mem2gib);
query_plan_test!(tpcds, 1, q78, tp8_mem2gib);
query_plan_test!(tpcds, 1, q79, tp8_mem2gib);
query_plan_test!(tpcds, 1, q80, tp8_mem2gib);
query_plan_test!(tpcds, 1, q81, tp8_mem2gib);
query_plan_test!(tpcds, 1, q82, tp8_mem2gib);
query_plan_test!(tpcds, 1, q83, tp8_mem2gib);
query_plan_test!(tpcds, 1, q84, tp8_mem2gib);
query_plan_test!(tpcds, 1, q85, tp8_mem2gib);
query_plan_test!(tpcds, 1, q87, tp8_mem2gib);
query_plan_test!(tpcds, 1, q88, tp8_mem2gib);
query_plan_test!(tpcds, 1, q89, tp8_mem2gib);
query_plan_test!(tpcds, 1, q90, tp8_mem2gib);
query_plan_test!(tpcds, 1, q91, tp8_mem2gib);
query_plan_test!(tpcds, 1, q92, tp8_mem2gib);
query_plan_test!(tpcds, 1, q93, tp8_mem2gib);
query_plan_test!(tpcds, 1, q94, tp8_mem2gib);
query_plan_test!(tpcds, 1, q95, tp8_mem2gib);
query_plan_test!(tpcds, 1, q96, tp8_mem2gib);
query_plan_test!(tpcds, 1, q97, tp8_mem2gib);
query_plan_test!(tpcds, 1, q98, tp8_mem2gib);
query_plan_test!(tpcds, 1, q99, tp8_mem2gib);

// ── TPC-DS disabled — blocked on DataFusion 46+ upgrade (issue #23) ──────────
// These four don't physical-plan under DataFusion 45, so they're also disabled
// in the cpu and gpu suites. Re-enable once the DataFusion 46+ upgrade (#23,
// which names these exact queries) lands.
//
// q27: DataFusion 45 SanityCheckPlan rejects the SortPreservingMergeExec ordering
// emitted for ROLLUP. Re-enable once upstream is fixed.
// query_plan_test!(tpcds, 1, q27, tp8_mem2gib);
// q70: DataFusion 45 doesn't physical-plan the GROUPING() aggregate.
// query_plan_test!(tpcds, 1, q70, tp8_mem2gib);
// q72: DataFusion 45 type-coercion can't handle Date32 + Int64 arithmetic.
// query_plan_test!(tpcds, 1, q72, tp8_mem2gib);
// q86: DataFusion 45 doesn't physical-plan the GROUPING() aggregate.
// query_plan_test!(tpcds, 1, q86, tp8_mem2gib);
