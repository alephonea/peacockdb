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

/// CPU reference partition-ids for the FULL proof-query key shape: 2 columns
/// (l_returnflag, l_linestatus) incl a NULL in each column. These are the values
/// the GPU cuDF standard murmur3 probe must reproduce to clear the conformance
/// gate. Printed for the probe comparison (coordinator reads GPU-vs-these).
#[test]
fn cpu_reference_2col_partition_ids_for_probe() {
    // q1-style groups + NULLs (the null-handling is a classic divergence source).
    let returnflag: ArrayRef = Arc::new(StringArray::from(vec![
        Some("A"), Some("N"), Some("N"), Some("R"), None, Some("A"),
    ]));
    let linestatus: ArrayRef = Arc::new(StringArray::from(vec![
        Some("F"), Some("F"), Some("O"), Some("F"), Some("F"), None,
    ]));
    let n = 8;
    let rows = returnflag.len();
    let mut buf = vec![42u32; rows];
    create_murmur3_hashes(&[returnflag, linestatus], &mut buf).unwrap();
    let ids: Vec<i32> = buf.iter().map(|&h| pmod(h as i32, n)).collect();
    eprintln!("PROBE CPU(comet) 2-col hashes(seed42) = {buf:?}");
    eprintln!("PROBE CPU(comet) 2-col partition_ids(pmod {n}) = {ids:?}");
    eprintln!("  rows: (A,F)(N,F)(N,O)(R,F)(NULL,F)(A,NULL)");
}

/// PERMANENT I-1 conformance gate (runs in the GPU-remote job): the REAL GPU path
/// (peacock::partitioning::spark_partition_ids via the FFI hook) and the REAL comet
/// CPU helper, in ONE process, over the SAME bytes — asserted bit-exact. NOT
/// hardcoded reference values. Requires a GPU + the cudf-linked build.
#[cfg(not(feature = "rust-only"))]
#[test]
fn gpu_spark_partition_ids_match_comet_live() {
    use datafusion::arrow::array::{Array, StructArray};
    use datafusion::arrow::datatypes::{DataType, Field};
    use datafusion::arrow::ffi::{to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
    use std::ffi::c_void;

    // Full proof-query key shape: 2 columns (l_returnflag, l_linestatus) + a NULL
    // in each — exercises multi-key left-to-right seeding + Spark null-skip.
    let rf: ArrayRef = Arc::new(StringArray::from(vec![
        Some("A"), Some("N"), Some("N"), Some("R"), None, Some("A"),
    ]));
    let ls: ArrayRef = Arc::new(StringArray::from(vec![
        Some("F"), Some("F"), Some("O"), Some("F"), Some("F"), None,
    ]));
    let n_parts: i32 = 8;
    let rows = rf.len();
    let comet = cpu_partition_ids(&[rf.clone(), ls.clone()], n_parts);

    // Export the key columns as a struct array (= the table) over the Arrow C-Data
    // interface; cuDF reads the struct as a table.
    let struct_arr = StructArray::from(vec![
        (Arc::new(Field::new("rf", DataType::Utf8, true)), rf),
        (Arc::new(Field::new("ls", DataType::Utf8, true)), ls),
    ]);
    let (ffi_arr, ffi_schema) = to_ffi(&struct_arr.to_data()).unwrap();

    let key_cols: [u32; 2] = [0, 1];
    let mut out = vec![0i32; rows];
    let mut got_n: u64 = 0;
    let rc = unsafe {
        peacockdb_ffi::raw::peacock_spark_partition_ids(
            &ffi_schema as *const FFI_ArrowSchema as *const c_void,
            &ffi_arr as *const FFI_ArrowArray as *const c_void,
            key_cols.as_ptr(),
            key_cols.len() as u64,
            n_parts as u32,
            42,
            out.as_mut_ptr(),
            out.len() as u64,
            &mut got_n,
        )
    };
    assert_eq!(rc, 0, "peacock_spark_partition_ids FFI returned an error");
    assert_eq!(got_n as usize, rows, "FFI returned wrong row count");
    eprintln!("LIVE comet CPU partition_ids = {comet:?}");
    eprintln!("LIVE GPU  FFI partition_ids = {out:?}");
    assert_eq!(
        out, comet,
        "GPU Spark-murmur3 partition-ids must match the comet CPU twin bit-exact"
    );
}

#[test]
fn pmod_handles_negative_hashes() {
    // pmod must differ from raw % for negative hashes (the classic mismatch source).
    assert_eq!(pmod(-1, 8), 7);
    assert_eq!(pmod(-9, 8), 7);
    assert_eq!(pmod(7, 8), 7);
    assert_ne!(-1 % 8, pmod(-1, 8), "raw % would give -1, pmod gives 7");
}
