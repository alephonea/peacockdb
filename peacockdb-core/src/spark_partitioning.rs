//! Which lane a row belongs to, Spark-murmur3 — the one rule both CPU paths and the GPU
//! kernel answer with.
//!
//! `peacock::partitioning::spark_hash_partition` is the device's copy and the live
//! conformance gate proves the two agree bit for bit, so a second spelling on this side
//! would be a divergence nothing else could see: the rows would still be joined, in the
//! wrong lanes, and every per-partition count would drift from its golden.
//!
//! Deliberately not DataFusion's `RepartitionExec`, whose ahash lands the same key in a
//! different partition.

use std::sync::Arc;

use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::physical_expr::PhysicalExpr;
use datafusion_comet_spark_expr::hash_funcs::murmur3::create_murmur3_hashes;

/// Spark's `HashPartitioning` seed; comet and the GPU kernel both initialize to it.
const SEED: u32 = 42;

/// Spark `pmod` (positive modulo): a signed murmur3 hash into `[0, n)`. Must match the
/// GPU kernel's exactly — negative hashes wrap the same way — or per-partition row counts
/// diverge from the golden.
fn pmod(hash: i32, n: i32) -> i32 {
    ((hash % n) + n) % n
}

/// The rows of `batch` that belong to each of `lanes` lanes, in row order.
pub fn rows_per_lane(
    batch: &RecordBatch,
    hash_exprs: &[Arc<dyn PhysicalExpr>],
    lanes: usize,
) -> DfResult<Vec<Vec<u32>>> {
    let rows = batch.num_rows();
    let keys = hash_keys(batch, hash_exprs)?;
    let mut hashes = vec![SEED; rows];
    if rows > 0 {
        create_murmur3_hashes(&keys, &mut hashes)
            .map_err(|error| DataFusionError::External(format!("comet murmur3: {error}").into()))?;
    }
    let n = lanes as i32;
    let mut per_lane: Vec<Vec<u32>> = vec![Vec::new(); lanes];
    for (row, hash) in hashes.iter().enumerate() {
        per_lane[pmod(*hash as i32, n) as usize].push(row as u32);
    }
    Ok(per_lane)
}

/// The key columns, in the layout comet's hasher accepts. DataFusion 45's parquet reader
/// emits the Arrow view layouts and comet rejects them; the cast is to the same bytes the
/// device hashes, so the assignment is unchanged.
fn hash_keys(batch: &RecordBatch, hash_exprs: &[Arc<dyn PhysicalExpr>]) -> DfResult<Vec<ArrayRef>> {
    hash_exprs
        .iter()
        .map(|expr| {
            let array = expr
                .evaluate(batch)
                .and_then(|v| v.into_array(batch.num_rows()))?;
            match array.data_type() {
                DataType::Utf8View => cast(&array, &DataType::Utf8).map_err(DataFusionError::from),
                DataType::BinaryView => {
                    cast(&array, &DataType::Binary).map_err(DataFusionError::from)
                }
                _ => Ok(array),
            }
        })
        .collect()
}
