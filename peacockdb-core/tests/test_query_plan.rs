//! Parameterized GPU plan-canonical tests for TPC-H and TPC-DS. Each
//! `query_plan_test!(dataset, sf, query, device)` plans <dataset>-queries/<query>.sql
//! and compares it to testdata/goldens/<dataset>.sf<sf>/<query>.<device>.plan.txt.
//! Helpers + the macro live in common/mod.rs; bespoke tests in test_query_plan_misc.rs.
#[macro_use]
mod common;

// ── TPC-H ─────────────────────────────────────────────────────────────────
query_plan_test!(tpch, 1, scan_limit, tp8_mini);
query_plan_test!(tpch, 1, filter_project, tp8_mini);
query_plan_test!(tpch, 1, aggregate_groupby, tp8_mini);
query_plan_test!(tpch, 1, hash_join, tp8_mini);
query_plan_test!(tpch, 1, left_join, tp8_mini);
query_plan_test!(tpch, 1, semi_join, tp8_mini);
query_plan_test!(tpch, 1, anti_join, tp8_mini);
query_plan_test!(tpch, 1, nested_loop_join, tp8_mini);
query_plan_test!(tpch, 1, nested_loop_left_join, tp8_mini);
query_plan_test!(tpch, 1, mixed_join, tp8_mini);
query_plan_test!(tpch, 1, cross_join, tp8_mini);
query_plan_test!(tpch, 1, q1, tp8_mini);
query_plan_test!(tpch, 1, q2, tp8_mini);
query_plan_test!(tpch, 1, q3, tp8_mini);
query_plan_test!(tpch, 1, q4, tp8_mini);
query_plan_test!(tpch, 1, q5, tp8_mini);
query_plan_test!(tpch, 1, q6, tp8_mini);
query_plan_test!(tpch, 1, q7, tp8_mini);
query_plan_test!(tpch, 1, q8, tp8_mini);
query_plan_test!(tpch, 1, q9, tp8_mini);
query_plan_test!(tpch, 1, q10, tp8_mini);
query_plan_test!(tpch, 1, q11, tp8_mini);
query_plan_test!(tpch, 1, q12, tp8_mini);
query_plan_test!(tpch, 1, q13, tp8_mini);
query_plan_test!(tpch, 1, q14, tp8_mini);
query_plan_test!(tpch, 1, q15, tp8_mini);
query_plan_test!(tpch, 1, q16, tp8_mini);
query_plan_test!(tpch, 1, q17, tp8_mini);
query_plan_test!(tpch, 1, q18, tp8_mini);
query_plan_test!(tpch, 1, q19, tp8_mini);
query_plan_test!(tpch, 1, q20, tp8_mini);
query_plan_test!(tpch, 1, q21, tp8_mini);
query_plan_test!(tpch, 1, q22, tp8_mini);
// The ONLY tp8-standard plan test — the real-partitioning device where
// the scan carries an RG→batch→partition map AND the Hash repartition is lowered into
// GpuCoalescePartitions(8→1) + GpuRepartition(1→8). Exercises the full lowered plan's
// flatbuffer roundtrip (plan_str + bytes) so the map + lowering survive serialize/
// deserialize. tp8-mini entries stay dormant (no map, no lowering).
query_plan_test!(tpch, 1, shuffle_additive, tp8_standard);

// ── TPC-DS ────────────────────────────────────────────────────────────────
query_plan_test!(tpcds, 1, q1, tp8_mini);
query_plan_test!(tpcds, 1, q2, tp8_mini);
query_plan_test!(tpcds, 1, q3, tp8_mini);
query_plan_test!(tpcds, 1, q4, tp8_mini);
query_plan_test!(tpcds, 1, q5, tp8_mini);
query_plan_test!(tpcds, 1, q6, tp8_mini);
query_plan_test!(tpcds, 1, q7, tp8_mini);
query_plan_test!(tpcds, 1, q8, tp8_mini);
query_plan_test!(tpcds, 1, q9, tp8_mini);
query_plan_test!(tpcds, 1, q10, tp8_mini);
query_plan_test!(tpcds, 1, q11, tp8_mini);
query_plan_test!(tpcds, 1, q12, tp8_mini);
query_plan_test!(tpcds, 1, q13, tp8_mini);
query_plan_test!(tpcds, 1, q14, tp8_mini);
query_plan_test!(tpcds, 1, q15, tp8_mini);
query_plan_test!(tpcds, 1, q16, tp8_mini);
query_plan_test!(tpcds, 1, q17, tp8_mini);
query_plan_test!(tpcds, 1, q18, tp8_mini);
query_plan_test!(tpcds, 1, q19, tp8_mini);
query_plan_test!(tpcds, 1, q20, tp8_mini);
query_plan_test!(tpcds, 1, q21, tp8_mini);
query_plan_test!(tpcds, 1, q22, tp8_mini);
query_plan_test!(tpcds, 1, q23, tp8_mini);
query_plan_test!(tpcds, 1, q24, tp8_mini);
query_plan_test!(tpcds, 1, q25, tp8_mini);
query_plan_test!(tpcds, 1, q26, tp8_mini);
query_plan_test!(tpcds, 1, q28, tp8_mini);
query_plan_test!(tpcds, 1, q29, tp8_mini);
query_plan_test!(tpcds, 1, q30, tp8_mini);
query_plan_test!(tpcds, 1, q31, tp8_mini);
query_plan_test!(tpcds, 1, q32, tp8_mini);
query_plan_test!(tpcds, 1, q33, tp8_mini);
query_plan_test!(tpcds, 1, q34, tp8_mini);
query_plan_test!(tpcds, 1, q35, tp8_mini);
query_plan_test!(tpcds, 1, q36, tp8_mini);
query_plan_test!(tpcds, 1, q37, tp8_mini);
query_plan_test!(tpcds, 1, q38, tp8_mini);
query_plan_test!(tpcds, 1, q39, tp8_mini);
query_plan_test!(tpcds, 1, q40, tp8_mini);
query_plan_test!(tpcds, 1, q41, tp8_mini);
query_plan_test!(tpcds, 1, q42, tp8_mini);
query_plan_test!(tpcds, 1, q43, tp8_mini);
query_plan_test!(tpcds, 1, q44, tp8_mini);
query_plan_test!(tpcds, 1, q45, tp8_mini);
query_plan_test!(tpcds, 1, q46, tp8_mini);
query_plan_test!(tpcds, 1, q47, tp8_mini);
query_plan_test!(tpcds, 1, q48, tp8_mini);
query_plan_test!(tpcds, 1, q49, tp8_mini);
query_plan_test!(tpcds, 1, q50, tp8_mini);
query_plan_test!(tpcds, 1, q51, tp8_mini);
query_plan_test!(tpcds, 1, q52, tp8_mini);
query_plan_test!(tpcds, 1, q53, tp8_mini);
query_plan_test!(tpcds, 1, q54, tp8_mini);
query_plan_test!(tpcds, 1, q55, tp8_mini);
query_plan_test!(tpcds, 1, q56, tp8_mini);
query_plan_test!(tpcds, 1, q57, tp8_mini);
query_plan_test!(tpcds, 1, q58, tp8_mini);
query_plan_test!(tpcds, 1, q59, tp8_mini);
query_plan_test!(tpcds, 1, q60, tp8_mini);
query_plan_test!(tpcds, 1, q61, tp8_mini);
query_plan_test!(tpcds, 1, q62, tp8_mini);
query_plan_test!(tpcds, 1, q63, tp8_mini);
query_plan_test!(tpcds, 1, q64, tp8_mini);
query_plan_test!(tpcds, 1, q65, tp8_mini);
query_plan_test!(tpcds, 1, q66, tp8_mini);
query_plan_test!(tpcds, 1, q67, tp8_mini);
query_plan_test!(tpcds, 1, q68, tp8_mini);
query_plan_test!(tpcds, 1, q69, tp8_mini);
query_plan_test!(tpcds, 1, q71, tp8_mini);
query_plan_test!(tpcds, 1, q73, tp8_mini);
query_plan_test!(tpcds, 1, q74, tp8_mini);
query_plan_test!(tpcds, 1, q75, tp8_mini);
query_plan_test!(tpcds, 1, q76, tp8_mini);
query_plan_test!(tpcds, 1, q77, tp8_mini);
query_plan_test!(tpcds, 1, q78, tp8_mini);
query_plan_test!(tpcds, 1, q79, tp8_mini);
query_plan_test!(tpcds, 1, q80, tp8_mini);
query_plan_test!(tpcds, 1, q81, tp8_mini);
query_plan_test!(tpcds, 1, q82, tp8_mini);
query_plan_test!(tpcds, 1, q83, tp8_mini);
query_plan_test!(tpcds, 1, q84, tp8_mini);
query_plan_test!(tpcds, 1, q85, tp8_mini);
query_plan_test!(tpcds, 1, q87, tp8_mini);
query_plan_test!(tpcds, 1, q88, tp8_mini);
query_plan_test!(tpcds, 1, q89, tp8_mini);
query_plan_test!(tpcds, 1, q90, tp8_mini);
query_plan_test!(tpcds, 1, q91, tp8_mini);
query_plan_test!(tpcds, 1, q92, tp8_mini);
query_plan_test!(tpcds, 1, q93, tp8_mini);
query_plan_test!(tpcds, 1, q94, tp8_mini);
query_plan_test!(tpcds, 1, q95, tp8_mini);
query_plan_test!(tpcds, 1, q96, tp8_mini);
query_plan_test!(tpcds, 1, q97, tp8_mini);
query_plan_test!(tpcds, 1, q98, tp8_mini);
query_plan_test!(tpcds, 1, q99, tp8_mini);

// ── TPC-DS disabled — blocked on DataFusion 46+ upgrade (issue #23) ──────────
// These four don't physical-plan under DataFusion 45, so they're also disabled
// in the cpu and gpu suites. Re-enable once the DataFusion 46+ upgrade (#23,
// which names these exact queries) lands.
//
// q27: DataFusion 45 SanityCheckPlan rejects the SortPreservingMergeExec ordering
// emitted for ROLLUP. Re-enable once upstream is fixed.
// query_plan_test!(tpcds, 1, q27, tp8_mini);
// q70: DataFusion 45 doesn't physical-plan the GROUPING() aggregate.
// query_plan_test!(tpcds, 1, q70, tp8_mini);
// q72: DataFusion 45 type-coercion can't handle Date32 + Int64 arithmetic.
// query_plan_test!(tpcds, 1, q72, tp8_mini);
// q86: DataFusion 45 doesn't physical-plan the GROUPING() aggregate.
// query_plan_test!(tpcds, 1, q86, tp8_mini);

// ── registry verification ───────────────────────────────────────────────────
/// This binary owns the `plan` column of testdata/cost-registry.csv. `inventory`
/// collects per binary, so each suite verifies its own columns; together the five
/// suites cover all six. See common/registry.rs.
#[test]
fn registry_matches_csv_plan_column() {
    common::registry::assert_registry_matches_csv(&["plan"], &[]);
}

/// The cross-mode golden invariant: a GPU mode marked `enabled` in the CSV requires
/// its same-device `.cpu.txt` golden, which the GPU test asserts per-node rows+cost
/// against. It lives HERE, in the CPU tier, rather than in the GPU suites: it inspects
/// only committed files, so putting it behind the GPU build would make it unable to
/// fail on the CPU tiers where most CI runs happen.
#[test]
fn registry_cross_mode_golden_invariant() {
    common::registry::assert_cross_mode_golden_invariant();
}
