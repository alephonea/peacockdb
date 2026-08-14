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

/// This target's reading of a case-list entry: the full correctness check at tp1.
///
/// Two arms, one per golden label. The label is matched as a LITERAL token, so a row
/// carrying neither known label fails to compile rather than expanding to nothing —
/// see the header of `common/gpu_cases.inc`.
macro_rules! gpu_case {
    ($dataset:ident, $sf:literal, $query:ident, full_table_tp1_standard, $mode:ident) => {
        gpu_full_table_test!($dataset, $sf, $query, full_table_tp1_standard, $mode);
    };
    // The 8-way rows belong to test_gpu_partitioned; drop them here.
    ($dataset:ident, $sf:literal, $query:ident, partitioned_tp8_standard, $mode:ident) => {};
}

include!("common/gpu_cases.inc");


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
