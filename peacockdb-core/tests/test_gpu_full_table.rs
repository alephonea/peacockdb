//! Merged per-query GPU verification, full-table (single-partition) execution.
//!
//! ONE GPU run per query asserts BOTH (a) per-node exact rows + rows/schema-derived
//! cost vs the `full_table-tp1-standard.cpu.txt` golden AND (b) the final RESULT.
//! Helpers + the `gpu_full_table_test!` macro live in common/mod.rs; the real 8-way
//! half of the suite is test_gpu_partitioned.rs.
//!
//! `gpu_full_table_test!(dataset, sf, query, label, mode)` — the device argument is
//! the combined golden label (`full_table_tp1_standard`), so the golden filename is
//! reconstructible from the call site with no lookup, and a label whose mode prefix
//! disagrees with the macro cannot pass silently. The result mode is EXPLICIT per
//! call site so golden-vs-live-oracle is visible at a glance:
//!   golden_exact  = static result golden, exact compare (fail-closed: missing panics);
//!   golden_approx = static result golden, 1e-12 float-tolerant (q14/q39);
//!   oracle        = live CPU-oracle compare, NO golden — for results too large to
//!                   commit as text (>= 256KB; e.g. anti-join ~240MB/1.2M rows);
//!   skip          = per-node only (non-deterministic LIMIT).
//! per-node cost is ALWAYS asserted (read-only) regardless of mode.
#![cfg(not(feature = "rust-only"))]
#[macro_use]
mod common;

// ── TPC-H (the GPU-supported set) ───────────────────────────────────────────
gpu_full_table_test!(tpch, 1, scan_limit, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, filter_project, full_table_tp1_standard, oracle);    // >256KB result
gpu_full_table_test!(tpch, 1, aggregate_groupby, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, semi_join, full_table_tp1_standard, oracle);        // >256KB result
gpu_full_table_test!(tpch, 1, anti_join, full_table_tp1_standard, oracle);        // >256KB result
gpu_full_table_test!(tpch, 1, nested_loop_join, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, cross_join, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q1, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q2, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q3, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q4, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q5, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q6, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q7, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q8, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q9, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q10, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q11, full_table_tp1_standard, oracle);              // >256KB result
gpu_full_table_test!(tpch, 1, q12, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q13, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q14, full_table_tp1_standard, golden_approx);
gpu_full_table_test!(tpch, 1, q15, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q16, full_table_tp1_standard, oracle);              // >256KB result
gpu_full_table_test!(tpch, 1, q17, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q18, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q19, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q20, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q21, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, q22, full_table_tp1_standard, golden_exact);

gpu_full_table_test!(tpch, 1, shuffle_additive, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpch, 1, shuffle_additive_avg, full_table_tp1_standard, golden_exact);
// STDDEV/VAR at tp1 (single-partition make_std/make_variance). golden_approx_std
// (1e-11): cuDF's variance algo diverges from DataFusion's Welford by ~2e-12 —
// more than the 1e-12 sum/avg convention. See GpuResultMode doc.
gpu_full_table_test!(tpch, 1, shuffle_stddev, full_table_tp1_standard, golden_approx_std);

// ── TPC-DS (GPU-operational set) ────────────────────────────────────────────
gpu_full_table_test!(tpcds, 1, q1, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q3, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q4, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q5, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q6, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q7, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q8, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q10, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q11, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q12, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q13, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q14, full_table_tp1_standard, golden_approx);
gpu_full_table_test!(tpcds, 1, q15, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q16, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q17, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q18, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q19, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q20, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q21, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q22, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q23, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q24, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q25, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q26, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q29, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q30, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q31, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q32, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q33, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q34, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q35, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q37, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q40, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q41, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q42, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q43, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q45, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q46, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q48, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q50, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q51, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q52, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q53, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q54, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q55, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q56, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q58, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q59, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q60, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q62, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q63, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q64, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q65, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q68, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q69, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q71, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q73, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q74, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q75, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q79, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q80, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q81, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q82, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q83, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q84, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q85, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q88, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q89, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q90, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q91, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q92, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q93, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q94, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q95, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q96, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q97, full_table_tp1_standard, golden_exact);
gpu_full_table_test!(tpcds, 1, q98, full_table_tp1_standard, oracle);             // >256KB result
gpu_full_table_test!(tpcds, 1, q99, full_table_tp1_standard, golden_exact);

// ── registry verification ───────────────────────────────────────────────────
/// Owns `full_table_gpu`. Cfg'd off under rust-only, where gpu_full_table_test!
/// emits neither test nor registration — an uncfg'd reverse check would then see an
/// empty inventory and fail spuriously.
#[cfg(not(feature = "rust-only"))]
#[test]
fn registry_matches_csv_full_table_gpu_column() {
    common::registry::assert_registry_matches_csv(&["full_table_gpu"], &[]);
}

// NOTE: the cross-mode golden invariant deliberately does NOT live here. It reads
// only the CSV and the goldens on disk — no GPU, no inventory — so gating it behind
// this binary's `not(rust-only)` build would mean it could only ever fail on a host
// that has a GPU toolchain. That is the same "guard that cannot go red where it
// matters" hole test_ci_coverage exists to close. It runs in the CPU tier instead,
// in test_query_plan.rs.
