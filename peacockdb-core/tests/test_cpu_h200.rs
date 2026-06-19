//! H200/tp1 CPU-emulated cost goldens (Task #13, Phase 1).
//!
//! Device `tp1-mem120gib` = single-partition execution at the H200's 120 GiB
//! budget. These `.cpu.txt` goldens are produced by the CPU oracle and are the
//! verification target for the GPU node-by-node tests (`test_gpu_node.rs`): at
//! tp1 the plan is single-partition, so GPU and CPU emulation share node
//! structure + per-node row counts exactly.
//!
//! Rolled out incrementally (tpch bucket first, then tpcds) — see the coordinator
//! staging plan. Same float-tolerance policy as #11 where needed.
#[macro_use]
mod common;

// ── TPC-H (initial validating bucket: aggregate + joins) ───────────────────
cpu_result_test!(tpch, 1, aggregate_groupby, tp1_mem120gib);
cpu_result_test!(tpch, 1, q1, tp1_mem120gib);
cpu_result_test!(tpch, 1, q3, tp1_mem120gib);
cpu_result_test!(tpch, 1, q5, tp1_mem120gib);
cpu_result_test!(tpch, 1, q6, tp1_mem120gib);
