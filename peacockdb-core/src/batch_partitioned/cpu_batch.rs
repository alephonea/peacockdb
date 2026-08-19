//! [`CpuBatch`]: the CPU backend's batch, and what leaves the device at the unload.

use datafusion::arrow::array::RecordBatch;

use super::batch::Batch;

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

    fn byte_size(&self) -> usize {
        self.batch.get_array_memory_size()
    }
}
