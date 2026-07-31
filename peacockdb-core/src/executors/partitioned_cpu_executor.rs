//! partitioned_cpu (#13): real N-way partitioning on the DataFusion backend.
//!
//! Config + trait wiring only. The partitioning machinery itself (scan map,
//! Spark-murmur3 hash repartition, partitioned join arity) is a backend internal
//! and lives in [`super::backend::cpu_node_executor`]; this class just constructs
//! that backend at `RealMultiPartition` and runs the shared driver.

use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;

use crate::config::{MemoryLimit, TargetPartitions};
use crate::create_context_with_tables_mode;
use crate::PartitionMode;

use super::backend::cpu_node_executor::CpuNodeExecutor;
use super::executor::{Executor, InstrumentedExecutor, NodeMemoryStats};
use super::node_by_node::execute_node_by_node;

/// tp8, `RealMultiPartition`. Partition mode is implied by the class, not a param.
pub struct PartitionedCpuExecutor {
    ctx: SessionContext,
}

impl PartitionedCpuExecutor {
    pub async fn new(data_dir: &std::path::Path, mem: MemoryLimit) -> Result<Self> {
        let ctx = create_context_with_tables_mode(
            data_dir,
            TargetPartitions::Multi.hint(),
            mem.bytes(),
            PartitionMode::RealMultiPartition,
        )
        .await?;
        Ok(Self { ctx })
    }

    async fn run(
        &self,
        sql: &str,
    ) -> Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>)> {
        let plan = self.ctx.sql(sql).await?.create_physical_plan().await?;
        let mut backend = CpuNodeExecutor::new(self.ctx.task_ctx());
        let (batches, stats) = execute_node_by_node(&plan, &mut backend).await?;
        Ok((batches, plan, stats))
    }
}

impl Executor for PartitionedCpuExecutor {
    async fn execute(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        Ok(self.run(sql).await?.0)
    }
}

impl InstrumentedExecutor for PartitionedCpuExecutor {
    async fn execute_instrumented(
        &self,
        sql: &str,
    ) -> Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>)> {
        self.run(sql).await
    }
}
