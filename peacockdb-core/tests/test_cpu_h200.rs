//! tp1 CPU-emulated cost goldens (#13).
//!
//! Device `tp1-standard` = single-partition execution at the standard budget.
//! These `.cpu.txt` goldens are produced by the CPU oracle and are the
//! verification target for the merged GPU test (`test_gpu.rs`): at tp1 the plan
//! is single-partition, so GPU and CPU emulation share node structure + per-node
//! row counts exactly. At tp1 there is no float reassociation (single partition),
//! so exact compare holds even for avg/stddev.
#[macro_use]
mod common;

// ── TPC-H (the GPU-supported set; mirrors test_gpu.rs) ─────────────────
cpu_result_test!(tpch, 1, scan_limit, tp1_standard, true);
cpu_result_test!(tpch, 1, filter_project, tp1_standard, false);
cpu_result_test!(tpch, 1, aggregate_groupby, tp1_standard, true);
cpu_result_test!(tpch, 1, semi_join, tp1_standard, false);
cpu_result_test!(tpch, 1, anti_join, tp1_standard, false);
cpu_result_test!(tpch, 1, nested_loop_join, tp1_standard, true);
cpu_result_test!(tpch, 1, cross_join, tp1_standard, true);
cpu_result_test!(tpch, 1, q1, tp1_standard, true);
cpu_result_test!(tpch, 1, q2, tp1_standard, true);
cpu_result_test!(tpch, 1, q3, tp1_standard, true);
cpu_result_test!(tpch, 1, q4, tp1_standard, true);
cpu_result_test!(tpch, 1, q5, tp1_standard, true);
cpu_result_test!(tpch, 1, q6, tp1_standard, true);
cpu_result_test!(tpch, 1, q7, tp1_standard, true);
cpu_result_test!(tpch, 1, q8, tp1_standard, true);
cpu_result_test!(tpch, 1, q9, tp1_standard, true);
cpu_result_test!(tpch, 1, q10, tp1_standard, true);
cpu_result_test!(tpch, 1, q11, tp1_standard, false);
cpu_result_test!(tpch, 1, q12, tp1_standard, true);
cpu_result_test!(tpch, 1, q13, tp1_standard, true);
cpu_result_test!(tpch, 1, q14, tp1_standard, true);
cpu_result_test!(tpch, 1, q15, tp1_standard, true);
cpu_result_test!(tpch, 1, q16, tp1_standard, false);
cpu_result_test!(tpch, 1, q17, tp1_standard, true);
cpu_result_test!(tpch, 1, q18, tp1_standard, true);
cpu_result_test!(tpch, 1, q19, tp1_standard, true);
cpu_result_test!(tpch, 1, q20, tp1_standard, true);
cpu_result_test!(tpch, 1, q21, tp1_standard, true);
cpu_result_test!(tpch, 1, q22, tp1_standard, true);
// shuffle_additive at tp1: single-partition baseline (no shuffle); the GPU verifies
// its result against this golden. The real 8-way variant is tp8-standard.
cpu_result_test!(tpch, 1, shuffle_additive, tp1_standard, true);
// shuffle_additive_avg at tp1: single-partition baseline (DF mode=Single, plain mean);
// the GPU verifies its result against this golden. Real 8-way variant is tp8-standard.
cpu_result_test!(tpch, 1, shuffle_additive_avg, tp1_standard, true);
// shuffle_stddev at tp1: single-partition baseline (DF mode=Partial+Final over 1 part;
// stddev/var = make_std/make_variance singleton, NOT the M2-merge path). The GPU verifies
// its result here; real 8-way M2 merge is tp8-standard. Approx (stddev/var are float).
cpu_result_approx_test!(tpch, 1, shuffle_stddev, tp1_standard, true);

// ── TPC-DS (GPU-operational set) ───────────────────────────────────────────
cpu_result_test!(tpcds, 1, q1, tp1_standard, true);
cpu_result_test!(tpcds, 1, q3, tp1_standard, true);
cpu_result_test!(tpcds, 1, q4, tp1_standard, true);
cpu_result_test!(tpcds, 1, q5, tp1_standard, true);
cpu_result_test!(tpcds, 1, q6, tp1_standard, true);
cpu_result_test!(tpcds, 1, q7, tp1_standard, true);
cpu_result_test!(tpcds, 1, q8, tp1_standard, true);
cpu_result_test!(tpcds, 1, q10, tp1_standard, true);
cpu_result_test!(tpcds, 1, q11, tp1_standard, true);
cpu_result_test!(tpcds, 1, q12, tp1_standard, true);
cpu_result_test!(tpcds, 1, q13, tp1_standard, true);
cpu_result_test!(tpcds, 1, q14, tp1_standard, true);
cpu_result_test!(tpcds, 1, q15, tp1_standard, true);
cpu_result_test!(tpcds, 1, q16, tp1_standard, true);
cpu_result_test!(tpcds, 1, q17, tp1_standard, true);
cpu_result_test!(tpcds, 1, q18, tp1_standard, true);
cpu_result_test!(tpcds, 1, q19, tp1_standard, true);
cpu_result_test!(tpcds, 1, q20, tp1_standard, true);
cpu_result_test!(tpcds, 1, q21, tp1_standard, true);
cpu_result_test!(tpcds, 1, q22, tp1_standard, true);
cpu_result_test!(tpcds, 1, q23, tp1_standard, true);
cpu_result_test!(tpcds, 1, q24, tp1_standard, true);
cpu_result_test!(tpcds, 1, q25, tp1_standard, true);
cpu_result_test!(tpcds, 1, q26, tp1_standard, true);
cpu_result_test!(tpcds, 1, q29, tp1_standard, true);
cpu_result_test!(tpcds, 1, q30, tp1_standard, true);
cpu_result_test!(tpcds, 1, q31, tp1_standard, true);
cpu_result_test!(tpcds, 1, q32, tp1_standard, true);
cpu_result_test!(tpcds, 1, q33, tp1_standard, true);
cpu_result_test!(tpcds, 1, q34, tp1_standard, true);
cpu_result_test!(tpcds, 1, q35, tp1_standard, true);
cpu_result_test!(tpcds, 1, q37, tp1_standard, true);
cpu_result_test!(tpcds, 1, q40, tp1_standard, true);
cpu_result_test!(tpcds, 1, q41, tp1_standard, true);
cpu_result_test!(tpcds, 1, q42, tp1_standard, true);
cpu_result_test!(tpcds, 1, q43, tp1_standard, true);
cpu_result_test!(tpcds, 1, q45, tp1_standard, true);
cpu_result_test!(tpcds, 1, q46, tp1_standard, true);
cpu_result_test!(tpcds, 1, q48, tp1_standard, true);
cpu_result_test!(tpcds, 1, q50, tp1_standard, true);
cpu_result_test!(tpcds, 1, q51, tp1_standard, true);
cpu_result_test!(tpcds, 1, q52, tp1_standard, true);
cpu_result_test!(tpcds, 1, q53, tp1_standard, true);
cpu_result_test!(tpcds, 1, q54, tp1_standard, true);
cpu_result_test!(tpcds, 1, q55, tp1_standard, true);
cpu_result_test!(tpcds, 1, q56, tp1_standard, true);
cpu_result_test!(tpcds, 1, q58, tp1_standard, true);
cpu_result_test!(tpcds, 1, q59, tp1_standard, true);
cpu_result_test!(tpcds, 1, q60, tp1_standard, true);
cpu_result_test!(tpcds, 1, q62, tp1_standard, true);
cpu_result_test!(tpcds, 1, q63, tp1_standard, true);
cpu_result_test!(tpcds, 1, q64, tp1_standard, true);
cpu_result_test!(tpcds, 1, q65, tp1_standard, true);
cpu_result_test!(tpcds, 1, q68, tp1_standard, true);
cpu_result_test!(tpcds, 1, q69, tp1_standard, true);
cpu_result_test!(tpcds, 1, q71, tp1_standard, true);
cpu_result_test!(tpcds, 1, q73, tp1_standard, true);
cpu_result_test!(tpcds, 1, q74, tp1_standard, true);
cpu_result_test!(tpcds, 1, q75, tp1_standard, true);
cpu_result_test!(tpcds, 1, q79, tp1_standard, true);
cpu_result_test!(tpcds, 1, q80, tp1_standard, true);
cpu_result_test!(tpcds, 1, q81, tp1_standard, true);
cpu_result_test!(tpcds, 1, q82, tp1_standard, true);
cpu_result_test!(tpcds, 1, q83, tp1_standard, true);
cpu_result_test!(tpcds, 1, q84, tp1_standard, true);
cpu_result_test!(tpcds, 1, q85, tp1_standard, true);
cpu_result_test!(tpcds, 1, q88, tp1_standard, true);
cpu_result_test!(tpcds, 1, q89, tp1_standard, true);
cpu_result_test!(tpcds, 1, q90, tp1_standard, true);
cpu_result_test!(tpcds, 1, q91, tp1_standard, true);
cpu_result_test!(tpcds, 1, q92, tp1_standard, true);
cpu_result_test!(tpcds, 1, q93, tp1_standard, true);
cpu_result_test!(tpcds, 1, q94, tp1_standard, true);
cpu_result_test!(tpcds, 1, q95, tp1_standard, true);
cpu_result_test!(tpcds, 1, q96, tp1_standard, true);
cpu_result_test!(tpcds, 1, q97, tp1_standard, true);
cpu_result_test!(tpcds, 1, q98, tp1_standard, false);
cpu_result_test!(tpcds, 1, q99, tp1_standard, true);

// ── registry verification ───────────────────────────────────────────────────
/// Owns the `ftc_tp1` column outright — every tp1 query is registered here at
/// tp1-standard, including scan_limit (which is ALSO registered at tp1-mini in
/// test_cpu_executor.rs; both map to ftc_tp1, so no cross-binary exception is
/// needed).
#[test]
fn registry_matches_csv_ftc_tp1_column() {
    common::registry::assert_registry_matches_csv(&["ftc_tp1"], &[]);
}
