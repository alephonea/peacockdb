//! GPU node-by-node verification (Task #13, Phase 1) — runs on shad-gpu.
//!
//! Each test runs the query through the GPU node-executor interface
//! (`GpuExecutor::execute_instrumented`) and asserts its per-node stats (exact row
//! counts + the rows+schema-derived cost) match the H200/tp1 CPU-emulated golden
//! produced by `test_cpu_h200.rs`. Verifies CPU-emulated == GPU at tp1, where the
//! single-partition plan makes node structure + row counts identical.
//!
//! Mirrors the `test_cpu_h200.rs` bucket; expands incrementally (tpch → tpcds).
#[macro_use]
mod common;

// ── TPC-H (matches test_cpu_h200.rs initial bucket) ────────────────────────
gpu_node_test!(tpch, 1, aggregate_groupby, tp1_mem120gib);
gpu_node_test!(tpch, 1, q1, tp1_mem120gib);
gpu_node_test!(tpch, 1, q3, tp1_mem120gib);
gpu_node_test!(tpch, 1, q5, tp1_mem120gib);
gpu_node_test!(tpch, 1, q6, tp1_mem120gib);
