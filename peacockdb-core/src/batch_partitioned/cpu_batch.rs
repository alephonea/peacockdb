//! [`CpuBatch`]: the CPU backend's batch, and what leaves the device at the unload.

use datafusion::arrow::array::RecordBatch;

use super::batch::Batch;
use crate::memory::{batch_varlen_content_bytes, logical_size_from_schema};

/// Not `Clone`, deliberately — symmetry with `GpuBatch`, whose handle cannot be.
#[derive(Debug)]
pub struct CpuBatch {
    batch: RecordBatch,
}

impl CpuBatch {
    pub fn new(batch: RecordBatch) -> Self {
        Self { batch }
    }

    pub fn record_batch(&self) -> &RecordBatch {
        &self.batch
    }

    pub fn into_record_batch(self) -> RecordBatch {
        self.batch
    }
}

impl Batch for CpuBatch {
    fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    /// The plan's declared width times the rows, plus what the var-length columns
    /// actually hold — the same formula the device is priced by, and by every other
    /// component in the tree. Arrow's `get_array_memory_size` is what was ALLOCATED:
    /// 64-byte aligned buffers and an absent validity bitmap where the device counts one,
    /// so the two engines put different numbers on one node of one query.
    fn byte_size(&self) -> usize {
        logical_size_from_schema(
            &self.batch.schema(),
            self.batch.num_rows(),
            batch_varlen_content_bytes(&self.batch),
        )
    }
}
