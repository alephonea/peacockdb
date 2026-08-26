//! `GpuEmitPartitions` on the CPU: one batch in, N out, some of them empty.
//!
//! The hash is [`crate::spark_partitioning`]'s, which is the device kernel's — so a key
//! lands in the same lane on either backend, and the per-lane counts a golden records are
//! the same numbers. A scatter rather than a `RepartitionExec` because that node has N
//! output partitions and the per-node relay drives one.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch, UInt32Array};
use datafusion::arrow::compute::take;
use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;

use crate::spark_partitioning::rows_per_lane;

use super::super::cpu_batch::CpuBatch;
use super::super::error::PlanError;
use super::super::executor::{BackendError, CallResult, CallStats};
use super::super::nodes::GpuEmitPartitions;

pub struct CpuEmitter {
    hash_keys: Vec<Arc<dyn PhysicalExpr>>,
    lanes: usize,
    schema: SchemaRef,
}

impl CpuEmitter {
    pub fn new(
        node: &GpuEmitPartitions,
        lanes: usize,
        input: &ArrowSchema,
    ) -> Result<Self, PlanError> {
        let mut hash_keys: Vec<Arc<dyn PhysicalExpr>> = Vec::with_capacity(node.hash_keys.len());
        for key in &node.hash_keys {
            let field = input.fields().get(*key as usize).ok_or_else(|| {
                PlanError::Invalid(format!(
                    "the hash key at {key} is past the {} columns its input has",
                    input.fields().len()
                ))
            })?;
            hash_keys.push(Arc::new(Column::new(field.name(), *key as usize)));
        }
        if lanes == 0 {
            return Err(PlanError::Invalid(
                "a scatter into no lanes emits nothing a plan could read".to_string(),
            ));
        }
        Ok(Self {
            hash_keys,
            lanes,
            schema: Arc::new(input.clone()),
        })
    }

    /// Exactly N batches, in lane order, empty where the hash sent nothing. The count is
    /// the contract: a driver reads output `p` as lane `p`'s, so a skipped empty would
    /// shift every lane above it.
    pub fn emit(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        let batch = batch.into_record_batch();
        let per_lane = rows_per_lane(&batch, &self.hash_keys, self.lanes).map_err(|error| {
            BackendError::new(format!("assigning the scatter's lanes: {error}"))
        })?;
        let mut lanes = Vec::with_capacity(self.lanes);
        for rows in per_lane {
            lanes.push(CpuBatch::new(self.rows_of(&batch, &rows)?));
        }
        Ok((lanes, CallStats::default()))
    }

    fn rows_of(&self, batch: &RecordBatch, rows: &[u32]) -> Result<RecordBatch, BackendError> {
        if rows.is_empty() {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }
        let indices = UInt32Array::from(rows.to_vec());
        let columns = batch
            .columns()
            .iter()
            .map(|column| take(column, &indices, None))
            .collect::<Result<Vec<ArrayRef>, _>>()
            .map_err(|error| BackendError::new(format!("gathering a lane's rows: {error}")))?;
        RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|error| BackendError::new(format!("a lane's batch: {error}")))
    }
}
