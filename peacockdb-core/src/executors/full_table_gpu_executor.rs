//! full_table_gpu: the node-by-node GPU mode at tp1.
//!
//! THIN CONFIG WRAPPER BY DESIGN — this is not a missing code split. Both GPU
//! node-by-node modes run the SAME [`GpuNodeExecutor`] backend through the SAME
//! [`execute_node_by_node`] driver; they differ ONLY in the [`PartitionMode`] they
//! construct with (SinglePartition) and the target-partition count (tp1).
//! Layer-2 shape: mode class = (config) + (pick backend) + (run driver).

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result as DfResult;
use datafusion::physical_plan::ExecutionPlan;
use std::path::Path;
use std::sync::Arc;

use crate::config::MemoryLimit;
use crate::PartitionMode;

use super::all_at_once_gpu_executor::GpuExecutor;
use super::executor::{Executor, InstrumentedExecutor, NodeMemoryStats};

pub struct FullTableGpuExecutor {
    inner: GpuExecutor,
}

impl FullTableGpuExecutor {
    /// tp1, `PartitionMode::SinglePartition` — both implied by the class, not params.
    pub async fn new(data_dir: &Path, mem: MemoryLimit) -> DfResult<Self> {
        let inner = GpuExecutor::new_mode(
            data_dir,
            1,
            mem.bytes(),
            PartitionMode::SinglePartition,
        )
        .await?;
        Ok(Self { inner })
    }
}

impl Executor for FullTableGpuExecutor {
    async fn execute(&self, sql: &str) -> DfResult<Vec<RecordBatch>> {
        Ok(self.inner.execute_instrumented(sql).await?.0)
    }
}

impl InstrumentedExecutor for FullTableGpuExecutor {
    async fn execute_instrumented(
        &self,
        sql: &str,
    ) -> DfResult<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>)> {
        self.inner.execute_instrumented(sql).await
    }
}
