//! Full-table CPU execution (`cpu_full_table_result_test!`).
//!
//! The instrumented-enforced executor streams each node single-partition-coalesced
//! regardless of `target_partitions`, so every device label here runs the SAME
//! executor — the mode comes from the macro name, not from the label. Each test runs
//! <dataset>-queries/<query>.sql through plain DataFusion (ground truth) and the CPU
//! executor, asserting the results match and the cost tree matches
//! testdata/goldens/<dataset>.sf<sf>/<query>.full_table-<tp>-<tier>.cpu.txt.
//! Helpers + the macros live in common/mod.rs; bespoke tests in
//! test_cpu_executor_misc.rs. The real 8-way variants live in
//! test_cpu_partitioned.rs.
#[macro_use]
mod common;

// ══ full_table-tp8-mini / full_table-tp1-mini ═══════════════════════════════
cpu_full_table_result_test!(tpch, 1, hash_join, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, left_join, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, mixed_join, tp8_mini, data_fusion_exact, no_result_golden);
// LIMIT without a total order (no/under-specified ORDER BY → ties at the LIMIT
// boundary) has no partition-invariant row set: at tp>1 both the result rows and
// the per-node output_bytes vary run-to-run. These queries are therefore canonized
// at tp1 (deterministic single-stream) for the result-assert AND the cost golden.
// Their .plan.txt (plan-shape) goldens stay tp8 — only the cpu device moves.
cpu_full_table_result_test!(tpch, 1, scan_limit, tp1_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, filter_project, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, aggregate_groupby, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, semi_join, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, anti_join, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, nested_loop_join, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, cross_join, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q1, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q2, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q3, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q4, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q5, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q6, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q7, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q8, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q9, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q10, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q11, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q12, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q13, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q14, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q15, tp8_mini, data_fusion_exact, no_result_golden);  // view inlined as a CTE (see q15.sql)
cpu_full_table_result_test!(tpch, 1, q16, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q17, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q18, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q19, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q20, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q21, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q22, tp8_mini, data_fusion_exact, no_result_golden);
// shuffle_additive (GROUP BY rf,ls; count/sum only) — the hash-shuffle proof
// query. At tp8-mini it stays full-table (single-partition-coalesced, partitions=1);
// the real 8-way golden is the tp8-standard entry in test_cpu_partitioned.rs.
cpu_full_table_result_test!(tpch, 1, shuffle_additive, tp8_mini, data_fusion_exact, no_result_golden);
// shuffle_additive_avg (GROUP BY rf,ls; count, avg, sum — avg BEFORE sum) — the
// AVG proof query. tp8-mini stays full-table; the real 8-way golden is in
// test_cpu_partitioned.rs.
cpu_full_table_result_test!(tpch, 1, shuffle_additive_avg, tp8_mini, data_fusion_exact, no_result_golden);
// shuffle_stddev (GROUP BY rf,ls; stddev_samp/pop + var_samp/pop) — the STDDEV/VAR
// proof. tp8-mini stays full-table (SinglePartition make_std/make_variance singleton);
// the real 8-way Welford-M2-merge golden is at tp8-standard. Approx: the M2
// summation reassociates across partitions (~1 ULP), so float compare is tolerant.
cpu_full_table_result_test!(tpch, 1, shuffle_stddev, tp8_mini, data_fusion_approximate, no_result_golden);

// ── TPC-DS ────────────────────────────────────────────────────────────────
cpu_full_table_result_test!(tpcds, 1, q1, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q2, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q3, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q4, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q5, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q6, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q7, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q8, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q9, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q10, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q11, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q12, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q13, tp8_mini, data_fusion_exact, no_result_golden);
// q14/q39: float (avg/stddev) summation reassociates across partitions at tp8 →
// ~1 ULP result drift vs the single-partition DataFusion oracle. Tolerant result
// compare (rel≤1e-12); output_bytes golden stays exact (float value doesn't change
// byte width), so these remain tp8.
cpu_full_table_result_test!(tpcds, 1, q14, tp8_mini, data_fusion_approximate, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q15, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q16, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q17, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q18, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q19, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q20, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q21, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q22, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q23, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q24, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q25, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q26, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q28, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q29, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q30, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q31, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q32, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q33, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q34, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q35, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q36, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q37, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q38, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q39, tp8_mini, data_fusion_approximate, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q40, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q41, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q42, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q43, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q44, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q45, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q46, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q47, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q48, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q49, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q50, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q51, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q52, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q53, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q54, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q55, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q56, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q57, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q58, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q59, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q60, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q61, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q62, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q63, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q64, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q65, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q66, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q67, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q68, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q69, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q71, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q73, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q74, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q75, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q76, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q77, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q78, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q79, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q80, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q81, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q82, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q83, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q84, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q85, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q87, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q88, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q89, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q90, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q91, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q92, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q93, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q94, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q95, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q96, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q97, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q98, tp8_mini, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q99, tp8_mini, data_fusion_exact, no_result_golden);

// ── TPC-DS disabled — blocked on DataFusion 46+ upgrade (issue #23) ──────────
// These four don't physical-plan under DataFusion 45, so they're also disabled
// in the plan and gpu suites. Re-enable once the DataFusion 46+ upgrade (#23,
// which names these exact queries) lands.
//
// q27: SanityCheckPlan rejects the SortPreservingMergeExec ordering for ROLLUP.
// cpu_full_table_result_test!(tpcds, 1, q27, tp8_mini, data_fusion_exact, no_result_golden);
// q70: GROUPING() aggregate has no physical-plan support.
// cpu_full_table_result_test!(tpcds, 1, q70, tp8_mini, data_fusion_exact, no_result_golden);
// q72: Date32 + Int64 type-coercion not supported.
// cpu_full_table_result_test!(tpcds, 1, q72, tp8_mini, data_fusion_exact, no_result_golden);
// q86: GROUPING() aggregate has no physical-plan support.
// cpu_full_table_result_test!(tpcds, 1, q86, tp8_mini, data_fusion_exact, no_result_golden);

// ══ full_table-tp1-standard: single-partition at the standard budget ════════
// These .cpu.txt goldens are the verification target for the merged GPU test
// (test_gpu_full_table.rs): at tp1 the plan is single-partition, so GPU and CPU
// emulation share node structure + per-node row counts exactly. At tp1 there is
// no float reassociation, so exact compare holds even for avg/stddev.
// ── TPC-H (the GPU-supported set; mirrors test_gpu_full_table.rs) ──────
cpu_full_table_result_test!(tpch, 1, scan_limit, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, filter_project, tp1_standard, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, aggregate_groupby, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, semi_join, tp1_standard, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, anti_join, tp1_standard, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, nested_loop_join, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, cross_join, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q1, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q2, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q3, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q4, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q5, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q6, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q7, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q8, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q9, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q10, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q11, tp1_standard, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q12, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q13, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q14, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q15, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q16, tp1_standard, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpch, 1, q17, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q18, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q19, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q20, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q21, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpch, 1, q22, tp1_standard, data_fusion_exact, result_golden);
// shuffle_additive at tp1: single-partition baseline (no shuffle); the GPU verifies
// its result against this golden. The real 8-way variant is tp8-standard.
cpu_full_table_result_test!(tpch, 1, shuffle_additive, tp1_standard, data_fusion_exact, result_golden);
// shuffle_additive_avg at tp1: single-partition baseline (DF mode=Single, plain mean);
// the GPU verifies its result against this golden. Real 8-way variant is tp8-standard.
cpu_full_table_result_test!(tpch, 1, shuffle_additive_avg, tp1_standard, data_fusion_exact, result_golden);
// shuffle_stddev at tp1: single-partition baseline (DF mode=Partial+Final over 1 part;
// stddev/var = make_std/make_variance singleton, NOT the M2-merge path). The GPU verifies
// its result here; real 8-way M2 merge is tp8-standard. Approx (stddev/var are float).
cpu_full_table_result_test!(tpch, 1, shuffle_stddev, tp1_standard, data_fusion_approximate, result_golden);

// ── TPC-DS (GPU-operational set) ───────────────────────────────────────────
cpu_full_table_result_test!(tpcds, 1, q1, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q3, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q4, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q5, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q6, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q7, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q8, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q10, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q11, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q12, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q13, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q14, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q15, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q16, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q17, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q18, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q19, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q20, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q21, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q22, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q23, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q24, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q25, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q26, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q29, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q30, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q31, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q32, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q33, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q34, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q35, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q37, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q40, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q41, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q42, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q43, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q45, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q46, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q48, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q50, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q51, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q52, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q53, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q54, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q55, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q56, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q58, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q59, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q60, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q62, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q63, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q64, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q65, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q68, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q69, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q71, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q73, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q74, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q75, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q79, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q80, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q81, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q82, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q83, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q84, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q85, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q88, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q89, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q90, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q91, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q92, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q93, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q94, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q95, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q96, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q97, tp1_standard, data_fusion_exact, result_golden);
cpu_full_table_result_test!(tpcds, 1, q98, tp1_standard, data_fusion_exact, no_result_golden);
cpu_full_table_result_test!(tpcds, 1, q99, tp1_standard, data_fusion_exact, result_golden);

// ── registry verification ───────────────────────────────────────────────────
/// Owns `ftc_tp1` and `ftc_tp8` outright: every full-table invocation — tp8-mini,
/// tp1-mini (scan_limit) and tp1-standard — lives in this binary, so both
/// directions of the check are complete here.
#[test]
fn registry_matches_csv_full_table_columns() {
    common::registry::assert_registry_matches_csv(&["ftc_tp1", "ftc_tp8"], &[]);
}
