//! Part 2 — strict resident "GPU"-memory control.
//!
//! At the real test budget (tp8-mem2gib) NOTHING OOMs: SF1 data is small and the
//! peak concurrently-resident data set across the whole corpus is ~135 MB (path-sum
//! model), far under 2 GiB. So the OOM path is exercised here with a TIGHT raw
//! budget (no device-label change), mirroring `test_memory_boundary_preserved_tight_budget`.
//!
//! Budget = 100 MiB, chosen in the wide gap between the top query (tpcds q78 ≈
//! 135.5 MB) and the next (tpch q7 ≈ 90.7 MB) so both sides clear ~10% margin:
//!   - q78 OOMs (+29% over budget),
//!   - q7 (−13.5%) and q18 (−16%) FIT (boundary-passing cases — the boundary is real).
//! Resident size uses the Part-1 per-node `output_bytes` logical basis, so it's
//! independent of the batch size the budget induces.
#[macro_use]
mod common;

const TIGHT_BUDGET: usize = 100 * 1024 * 1024; // 100 MiB

// Flips pass→OOM under strict resident control (asserted, never disabled).
cpu_result_error_test!(tpcds, 1, q78, TIGHT_BUDGET);

// Boundary-passing cases: just under the budget, must still FIT.
cpu_result_fits_test!(tpch, 1, q7, TIGHT_BUDGET);
cpu_result_fits_test!(tpch, 1, q18, TIGHT_BUDGET);
