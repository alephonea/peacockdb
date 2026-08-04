//! GPU↔CPU murmur3 hash-partition conformance (the linchpin gate).
//!
//! The #13 CpuNodeExecutor and the GPU both must assign each row to the SAME
//! shuffle partition, else the hash-repartition golden (per-node rows + result)
//! diverges. Both sides use a Spark-compatible murmur3:
//!   - CPU twin: comet `create_murmur3_hashes` (buffer pre-seeded to 42), then pmod.
//!   - GPU: our OWN Spark-murmur3 CUDA kernel `peacock::partitioning::spark_partition_ids`
//!     (seed=42, per-column left-to-right running-seed, Spark null-skip, UTF-8 bytes),
//!     then pmod. cuDF ships only STANDARD murmur3 (proven ≠ Spark by the probe), so we
//!     own the hash kernel (Route B) and reuse cudf::partition for the scatter.
//! The `*_match_comet_live` gates drive the REAL GPU kernel via the FFI hook and
//! assert it is bit-exact against the comet CPU twin in one process. Covered key
//! shapes: string, int32, int64, int16, date32, composite int, composite mixed —
//! a key type NOT covered here is unproven on the GPU.

// The low-level Spark murmur3 API is public (not UDF-only).
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

/// Conformance harness: drive the REAL GPU path (peacock::partitioning::
/// spark_partition_ids via the FFI hook) and the REAL comet CPU helper, in ONE
/// process, over the SAME `cols` — assert bit-exact. NOT hardcoded reference values.
/// Each `(Field, ArrayRef)` is a key column; they seed-chain left-to-right (composite
/// keys). Requires a GPU + the cudf-linked build.
#[cfg(not(feature = "rust-only"))]
fn assert_gpu_matches_comet_live(cols: Vec<(datafusion::arrow::datatypes::Field, ArrayRef)>, n_parts: i32) {
    use datafusion::arrow::array::{Array, StructArray};
    use datafusion::arrow::ffi::{to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
    use std::ffi::c_void;

    let arrays: Vec<ArrayRef> = cols.iter().map(|(_, a)| Arc::clone(a)).collect();
    let rows = arrays[0].len();
    let comet = cpu_partition_ids(&arrays, n_parts);

    // Export the key columns as a struct array (= the table) over the Arrow C-Data
    // interface; cuDF reads the struct as a table.
    let struct_arr = StructArray::from(
        cols.into_iter().map(|(f, a)| (Arc::new(f), a)).collect::<Vec<_>>(),
    );
    let (ffi_arr, ffi_schema) = to_ffi(&struct_arr.to_data()).unwrap();

    let key_cols: Vec<u32> = (0..arrays.len() as u32).collect();
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

/// PERMANENT gate (STRING keys): 2 string columns (l_returnflag, l_linestatus)
/// + a NULL in each (multi-key seeding + null-skip).
#[cfg(not(feature = "rust-only"))]
#[test]
fn gpu_spark_partition_ids_match_comet_live() {
    use datafusion::arrow::datatypes::{DataType, Field};
    let rf: ArrayRef = Arc::new(StringArray::from(vec![
        Some("A"), Some("N"), Some("N"), Some("R"), None, Some("A"),
    ]));
    let ls: ArrayRef = Arc::new(StringArray::from(vec![
        Some("F"), Some("F"), Some("O"), Some("F"), Some("F"), None,
    ]));
    assert_gpu_matches_comet_live(
        vec![
            (Field::new("rf", DataType::Utf8, true), rf),
            (Field::new("ls", DataType::Utf8, true), ls),
        ],
        8,
    );
}

/// INT32 key conformance — edge values (0, -1, i32::MAX/MIN) + a NULL.
/// int32 = one 4-byte LE block, no tail; exercises the generic fixed-width kernel
/// and the negative/extreme two's-complement encodings + Spark null-skip.
#[cfg(not(feature = "rust-only"))]
#[test]
fn gpu_spark_partition_ids_int32_match_comet_live() {
    use datafusion::arrow::array::Int32Array;
    use datafusion::arrow::datatypes::{DataType, Field};
    let k: ArrayRef = Arc::new(Int32Array::from(vec![
        Some(1), Some(0), Some(-1), Some(i32::MAX), Some(i32::MIN), None, Some(42),
    ]));
    assert_gpu_matches_comet_live(vec![(Field::new("k", DataType::Int32, true), k)], 8);
}

/// INT64 key conformance — the dominant surrogate-key (*_sk) case.
/// int64 = two 4-byte LE blocks (low then high), no tail. Edge values + NULL.
#[cfg(not(feature = "rust-only"))]
#[test]
fn gpu_spark_partition_ids_int64_match_comet_live() {
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field};
    let k: ArrayRef = Arc::new(Int64Array::from(vec![
        Some(1), Some(0), Some(-1), Some(i64::MAX), Some(i64::MIN), None, Some(1234567890123),
    ]));
    assert_gpu_matches_comet_live(vec![(Field::new("k", DataType::Int64, true), k)], 8);
}

/// (#18) INT16 key conformance — the GROUP-BY year case (cudf::extract_year emits
/// INT16, so a year-grouped query repartitions on an INT16 key). Spark widens short→int
/// (4-byte hash); the GPU casts INT16→INT32 before the fixed kernel, so this proves the
/// widened hash is bit-exact vs comet. Edge values + NULL.
#[cfg(not(feature = "rust-only"))]
#[test]
fn gpu_spark_partition_ids_int16_match_comet_live() {
    use datafusion::arrow::array::Int16Array;
    use datafusion::arrow::datatypes::{DataType, Field};
    let k: ArrayRef = Arc::new(Int16Array::from(vec![
        Some(1), Some(0), Some(-1), Some(i16::MAX), Some(i16::MIN), None, Some(1998),
    ]));
    assert_gpu_matches_comet_live(vec![(Field::new("k", DataType::Int16, true), k)], 8);
}

/// (#18) DATE32 key conformance — the GROUP-BY date case (q3 groups by o_orderdate;
/// cuDF stores it as TIMESTAMP_DAYS = int32 days-since-epoch). Spark hashes DATE as the
/// int32 day count (4-byte); the GPU bit-casts TIMESTAMP_DAYS→INT32, so this proves the
/// days hash is bit-exact vs comet. Epoch, real dates, pre-epoch negative, NULL.
#[cfg(not(feature = "rust-only"))]
#[test]
fn gpu_spark_partition_ids_date32_match_comet_live() {
    use datafusion::arrow::array::Date32Array;
    use datafusion::arrow::datatypes::{DataType, Field};
    let k: ArrayRef = Arc::new(Date32Array::from(vec![
        Some(0), Some(9203), Some(-1), Some(i32::MAX), None, Some(10000),
    ]));
    assert_gpu_matches_comet_live(vec![(Field::new("k", DataType::Date32, true), k)], 8);
}

/// COMPOSITE all-INT key conformance — the q17 join-key shape
/// (ss_customer_sk, ss_item_sk, ss_ticket_number are all int surrogate keys).
/// Proves the seed-chain across multiple int columns, incl per-column NULLs (a null
/// in ONE column of a row still folds the other columns — Spark skips only that col).
#[cfg(not(feature = "rust-only"))]
#[test]
fn gpu_spark_partition_ids_composite_int_match_comet_live() {
    use datafusion::arrow::array::{Int32Array, Int64Array};
    use datafusion::arrow::datatypes::{DataType, Field};
    let a: ArrayRef = Arc::new(Int32Array::from(vec![
        Some(1), Some(2), Some(-1), None, Some(i32::MIN), Some(7),
    ]));
    let b: ArrayRef = Arc::new(Int64Array::from(vec![
        Some(100), None, Some(-100), Some(i64::MAX), Some(0), Some(7),
    ]));
    assert_gpu_matches_comet_live(
        vec![
            (Field::new("a", DataType::Int32, true), a),
            (Field::new("b", DataType::Int64, true), b),
        ],
        8,
    );
}

/// COMPOSITE MIXED-type key conformance (int64 + string, nulls in each) —
/// proves the running seed chains correctly across type-heterogeneous columns (the
/// general case: an int join/group key interleaved with a string dimension key).
#[cfg(not(feature = "rust-only"))]
#[test]
fn gpu_spark_partition_ids_composite_mixed_match_comet_live() {
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field};
    let ints: ArrayRef = Arc::new(Int64Array::from(vec![
        Some(10), Some(-5), None, Some(i64::MAX), Some(0), Some(999),
    ]));
    let strs: ArrayRef = Arc::new(StringArray::from(vec![
        Some("A"), Some("BB"), Some("C"), None, Some(""), Some("ticket"),
    ]));
    assert_gpu_matches_comet_live(
        vec![
            (Field::new("i", DataType::Int64, true), ints),
            (Field::new("s", DataType::Utf8, true), strs),
        ],
        8,
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
