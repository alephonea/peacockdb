//! partitioned_gpu: the node-by-node GPU mode at tp8.
//!
//! THIN CONFIG WRAPPER BY DESIGN — this is not a missing code split. Both GPU
//! node-by-node modes run the SAME [`GpuNodeExecutor`] backend through the SAME
//! [`execute_node_by_node`] driver; they differ ONLY in the [`PartitionMode`] they
//! construct with (`RealMultiPartition`) and the target-partition count (tp8).
//! Layer-2 shape: mode class = (config) + (pick backend) + (run driver).
//!
//! [`GpuNodeExecutor`]: super::backend::gpu_node_executor::GpuNodeExecutor
//! [`execute_node_by_node`]: super::node_by_node::execute_node_by_node

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result as DfResult;
use datafusion::physical_plan::ExecutionPlan;
use std::path::Path;
use std::sync::Arc;

use crate::config::{MemoryLimit, TargetPartitions};
use crate::PartitionMode;

use super::all_at_once_gpu_executor::GpuExecutor;
use super::executor::{Executor, InstrumentedExecutor, NodeMemoryStats};

pub struct PartitionedGpuExecutor {
    inner: GpuExecutor,
}

impl PartitionedGpuExecutor {
    /// tp8, `PartitionMode::RealMultiPartition` — both implied by the class, not params.
    pub async fn new(data_dir: &Path, mem: MemoryLimit) -> DfResult<Self> {
        let inner = GpuExecutor::new_mode(
            data_dir,
            TargetPartitions::Multi.hint(),
            mem.bytes(),
            PartitionMode::RealMultiPartition,
        )
        .await?;
        Ok(Self { inner })
    }
}

impl Executor for PartitionedGpuExecutor {
    async fn execute(&self, sql: &str) -> DfResult<Vec<RecordBatch>> {
        Ok(self.inner.execute_instrumented(sql).await?.0)
    }
}

impl InstrumentedExecutor for PartitionedGpuExecutor {
    async fn execute_instrumented(
        &self,
        sql: &str,
    ) -> DfResult<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>)> {
        self.inner.execute_instrumented(sql).await
    }
}
