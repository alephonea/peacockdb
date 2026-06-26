//! Inc2 GPU↔CPU murmur3 hash-partition conformance (the linchpin gate).
//!
//! The #13 CpuNodeExecutor and the GPU both must assign each row to the SAME
//! shuffle partition, else the hash-repartition golden (per-node rows + result)
//! diverges. Both sides use a Spark-compatible murmur3:
//!   - CPU twin: comet `create_murmur3_hashes` (buffer pre-seeded to 42), then pmod.
//!   - GPU: cuDF `spark_murmurhash3_x86_32(keys, seed=42)`, then pmod (added in a
//!     later step; this file starts with the CPU side + the dep-confirm).
//!
//! BOUNDARY (reviewer I-2): the sample below is STRING-KEY-ONLY (q1's char keys).
//! Non-string shuffle keys are UNPROVEN until this conformance set is extended.

// Step-i dep-confirm: the low-level Spark murmur3 API is PUBLIC (not UDF-only).
use datafusion_comet_spark_expr::hash_funcs::murmur3::{
    create_murmur3_hashes, spark_compatible_murmur3_hash,
};

use datafusion::arrow::array::{ArrayRef, StringArray};
use std::sync::Arc;

/// Spark `pmod` (positive modulo), NOT raw `%` — negative hashes must wrap to
/// `[0, n)` identically on both sides (spec: pmod, not %).
fn pmod(h: i32, n: i32) -> i32 {
    ((h % n) + n) % n
}

/// CPU partition assignment for `key_cols` over `n` partitions, via the REAL comet
/// helper (Spark seed 42, iterative left-to-right column seeding, exact per-type +
/// null encoding). This is the SAME computation the #13 CpuNodeExecutor will use.
fn cpu_partition_ids(key_cols: &[ArrayRef], n: i32) -> Vec<i32> {
    let rows = key_cols[0].len();
    let mut buf = vec![42u32; rows]; // Spark HashPartitioning seed = 42
    create_murmur3_hashes(key_cols, &mut buf).unwrap();
    buf.iter().map(|&h| pmod(h as i32, n)).collect()
}

#[test]
fn step_i_comet_murmur3_public_api_compiles_and_runs() {
    // Single-value low-level hash (UTF-8 bytes), seed 42 — the q1 chars.
    let h_a = spark_compatible_murmur3_hash(b"A", 42);
    let h_n = spark_compatible_murmur3_hash(b"N", 42);
    eprintln!("spark_compatible_murmur3_hash('A',42)={h_a}  ('N',42)={h_n}");
    assert_ne!(h_a, h_n, "distinct keys should (almost surely) hash differently");

    // Array-level multi-row (single string column) — the production path.
    let keys: ArrayRef = Arc::new(StringArray::from(vec!["A", "N", "R", "F", "O"]));
    let ids = cpu_partition_ids(&[keys], 8);
    eprintln!("cpu_partition_ids(['A','N','R','F','O'], 8) = {ids:?}");
    assert_eq!(ids.len(), 5);
    assert!(ids.iter().all(|&p| (0..8).contains(&p)), "all ids in [0,8)");
}

#[test]
fn pmod_handles_negative_hashes() {
    // pmod must differ from raw % for negative hashes (the classic mismatch source).
    assert_eq!(pmod(-1, 8), 7);
    assert_eq!(pmod(-9, 8), 7);
    assert_eq!(pmod(7, 8), 7);
    assert_ne!(-1 % 8, pmod(-1, 8), "raw % would give -1, pmod gives 7");
}
