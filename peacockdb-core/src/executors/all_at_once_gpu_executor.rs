//! all_at_once_gpu: the final-result-only GPU fast path (one `peacock_execute` FFI
//! call), plus [`GpuExecutor`], the resource holder both GPU modes are built on.
//!
//! [`GpuExecutor`] owns the `*mut PeacockExecutor` and is deliberately the ONLY
//! place that pointer lives; the node-by-node GPU backend borrows it. Keeping the
//! owner, the `unsafe impl Send`, and `Drop` in one file keeps that contract
//! reviewable — see [`super::backend::gpu_node_executor`].
//!
//! [`AllAtOnceGpuExecutor`] implements [`Executor`] ONLY: it makes a single FFI
//! call and never sees individual nodes, so it has no per-node stats to report.
//! Slated for retirement once the node-by-node modes cover it —
//! https://github.com/asymptote-tech/peacockdb/issues/110

use std::path::Path;

use crate::config::MemoryLimit;

use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::context::SessionContext;

use crate::executors::executor::{Executor, NodeMemoryStats};
use crate::executors::backend::gpu_node_executor::GpuNodeExecutor;
use crate::executors::node_by_node::execute_node_by_node;
use crate::{create_context_with_tables_mode, plan_serializer::serialize_plan_mode, PartitionMode};

use peacockdb_ffi::raw::{
    peacock_execute, peacock_executor_create, peacock_executor_destroy, peacock_last_error,
    peacock_result_free, PeacockExecutor,
};

/// TODO(#110): `GpuExecutor` is currently the shared impl behind ALL THREE GPU mode
/// classes (all_at_once, full_table, partitioned) — deliberate for Inc2/Inc3, since it
/// is the sole owner of the `*mut PeacockExecutor` and the facades keep old paths
/// resolving. Revisit when #110 retires the all-at-once path: at that point `execute`
/// (the one-shot FFI call) goes away and what remains is purely the node-by-node
/// resource holder, which probably wants a name that says so.
///
/// Executes SQL queries on the GPU via the C++ peacock_gpu library.
///
/// Lifecycle: `new()` registers tables and creates the C executor; `execute()`
/// serializes the GPU-annotated plan to FlatBuffers, calls `peacock_execute`,
/// and deserializes the Arrow IPC result. `Drop` destroys the C executor.
pub struct GpuExecutor {
    ctx: SessionContext,
    executor: *mut PeacockExecutor,
    /// Threaded into the plan serializer so aggregate nodes carry the
    /// `mergeable_agg_state` flag (RealMultiPartition ⇒ 3-col Welford stddev/var
    /// state; see [`serialize_plan_mode`]).
    partition_mode: PartitionMode,
}

// SAFETY: GpuExecutor has exclusive ownership of the PeacockExecutor pointer.
unsafe impl Send for GpuExecutor {}

impl GpuExecutor {
    pub async fn new(
        data_dir: &Path,
        target_partitions: usize,
        gpu_memory_budget: usize,
    ) -> DfResult<Self> {
        Self::new_mode(data_dir, target_partitions, gpu_memory_budget, PartitionMode::SinglePartition)
            .await
    }

    /// Like [`GpuExecutor::new`] but at an explicit [`PartitionMode`]. The
    /// real-partitioning GPU verify (tp8-standard) passes
    /// [`PartitionMode::RealMultiPartition`] so the scan map + Hash-repartition
    /// lowering match the CPU-emulated golden it verifies against.
    pub async fn new_mode(
        data_dir: &Path,
        target_partitions: usize,
        gpu_memory_budget: usize,
        partition_mode: PartitionMode,
    ) -> DfResult<Self> {
        let ctx = create_context_with_tables_mode(
            data_dir,
            target_partitions,
            gpu_memory_budget,
            partition_mode,
        )
        .await?;

        let mut executor: *mut PeacockExecutor = std::ptr::null_mut();
        let rc =
            unsafe { peacock_executor_create(gpu_memory_budget as u64, &mut executor) };
        if rc != 0 {
            return Err(DataFusionError::External(
                format!("peacock_executor_create failed with code {rc}").into(),
            ));
        }

        Ok(Self { ctx, executor, partition_mode })
    }

    /// Execute `sql` on the GPU.
    ///
    /// Steps:
    /// 1. Build a GPU-annotated physical plan via the DataFusion session.
    /// 2. Serialize the plan to FlatBuffers with `serialize_plan`.
    /// 3. Call `peacock_execute` — the C++ engine runs the plan on the GPU.
    /// 4. Deserialize the Arrow IPC stream result back to `Vec<RecordBatch>`.
    ///
    /// The empty-buffer branch (`out_result_len == 0`) is defensive: today the
    /// C side always emits a full Arrow IPC stream (schema + zero or more
    /// batches) on success, so a zero-row query still produces a non-empty
    /// buffer that decodes to an empty `Vec<RecordBatch>`.
    pub async fn execute(&self, sql: &str) -> DfResult<Vec<RecordBatch>> {
        let plan = self.ctx.sql(sql).await?.create_physical_plan().await?;
        let plan_bytes = serialize_plan_mode(&plan, self.partition_mode)
            .map_err(|e| DataFusionError::External(e.into()))?;

        let mut out_ptr: *mut u8 = std::ptr::null_mut();
        let mut out_len: u64 = 0;

        let rc = unsafe {
            peacock_execute(
                self.executor,
                plan_bytes.as_ptr(),
                plan_bytes.len() as u64,
                &mut out_ptr,
                &mut out_len,
            )
        };

        if rc != 0 {
            let msg = unsafe {
                let ptr = peacock_last_error(self.executor);
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            };
            return Err(DataFusionError::External(
                format!("peacock_execute failed (code {rc}): {msg}").into(),
            ));
        }

        if out_len == 0 || out_ptr.is_null() {
            return Ok(vec![]);
        }

        let ipc_bytes = unsafe { std::slice::from_raw_parts(out_ptr, out_len as usize) };
        let batches = read_ipc_stream(ipc_bytes)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        unsafe { peacock_result_free(out_ptr) };

        Ok(batches)
    }

    /// Execute `sql` on the GPU NODE-BY-NODE through the unified node-executor
    /// interface, returning the result batches plus per-node [`NodeMemoryStats`]
    /// (post-order). Intermediates stay GPU-resident; only the root crosses out.
    /// This is the instrumented path the GPU equivalence tests compare against the
    /// CPU golden; [`execute`] remains the all-at-once production fast path.
    pub async fn execute_instrumented(
        &self,
        sql: &str,
    ) -> DfResult<(
        Vec<RecordBatch>,
        std::sync::Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        Vec<NodeMemoryStats>,
    )> {
        let plan = self.ctx.sql(sql).await?.create_physical_plan().await?;
        let plan_bytes = serialize_plan_mode(&plan, self.partition_mode)
            .map_err(|e| DataFusionError::External(e.into()))?;
        let mut backend = GpuNodeExecutor::new(self.executor, &plan_bytes)?;
        let (batches, stats) = execute_node_by_node(&plan, &mut backend).await?;
        Ok((batches, plan, stats))
    }
}

impl Drop for GpuExecutor {
    fn drop(&mut self) {
        if !self.executor.is_null() {
            unsafe { peacock_executor_destroy(self.executor) };
        }
    }
}

fn read_ipc_stream(bytes: &[u8]) -> Result<Vec<RecordBatch>, arrow::error::ArrowError> {
    StreamReader::try_new(std::io::Cursor::new(bytes), None)?.collect()
}

/// The all-at-once mode class. `Executor` only — no per-node stats exist on this path.
pub struct AllAtOnceGpuExecutor {
    inner: GpuExecutor,
}

impl AllAtOnceGpuExecutor {
    /// tp1, `SinglePartition` — implied by the class.
    pub async fn new(data_dir: &Path, mem: MemoryLimit) -> DfResult<Self> {
        Ok(Self { inner: GpuExecutor::new(data_dir, 1, mem.bytes()).await? })
    }
}

impl Executor for AllAtOnceGpuExecutor {
    async fn execute(&self, sql: &str) -> DfResult<Vec<RecordBatch>> {
        self.inner.execute(sql).await
    }
}
