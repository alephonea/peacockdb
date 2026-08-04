//! GPU backend for the node-by-node driver: the C++/cuDF FFI.
//!
//! Handles are GPU-resident tables; per-node stats are reconstructed in Rust from
//! the FFI's `{rows, var-len content}` via the single-source
//! [`crate::cpu_executor::logical_size_from_schema`], so CPU and GPU costs are
//! identical by construction whenever per-node row counts match.
//!
//! SAFETY: the raw `*mut PeacockExecutor` here is BORROWED — it is owned by the
//! [`crate::executors::all_at_once_gpu_executor::GpuExecutor`] that handed it over,
//! which must outlive this backend. That contract is documented, not enforced by
//! lifetimes; keep the two in view of each other when changing either.


use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result as DfResult;
use datafusion::physical_plan::ExecutionPlan;

use crate::executors::executor::{NodeMemoryStats, PartitionStat};
use crate::executors::node_by_node::NodeExecutor;

use arrow::ipc::reader::StreamReader;
use datafusion::error::DataFusionError;

use peacockdb_ffi::raw::{
    peacock_executor_begin_plan, peacock_executor_end_plan, peacock_executor_execute_node,
    peacock_handle_release, peacock_last_error, peacock_result_free, peacock_result_from_handle,
    PeacockExecutor, PeacockNodeStats,
};

use crate::cpu_executor::logical_size_from_schema;

/// GPU backend: intermediates stay GPU-resident behind handles in the C++
/// `NodeSession`; the executor pointer is BORROWED (owned by `GpuExecutor`).
/// On drop, `peacock_executor_end_plan` frees the session + all remaining
/// resident handles — the VRAM-safety net for mid-walk errors.
pub struct GpuNodeExecutor {
    executor: *mut PeacockExecutor,
}

impl GpuNodeExecutor {
    /// Load the serialized plan into the C++ session (indexes post-order).
    pub fn new(executor: *mut PeacockExecutor, plan_bytes: &[u8]) -> DfResult<Self> {
        let mut node_count: u64 = 0;
        let rc = unsafe {
            peacock_executor_begin_plan(
                executor,
                plan_bytes.as_ptr(),
                plan_bytes.len() as u64,
                &mut node_count,
            )
        };
        if rc != 0 {
            return Err(last_error(executor, "peacock_executor_begin_plan"));
        }
        Ok(Self { executor })
    }
}

fn last_error(executor: *mut PeacockExecutor, ctx: &str) -> DataFusionError {
    let msg = unsafe {
        std::ffi::CStr::from_ptr(peacock_last_error(executor))
            .to_string_lossy()
            .into_owned()
    };
    DataFusionError::External(format!("{ctx} failed: {msg}").into())
}

impl NodeExecutor for GpuNodeExecutor {
    async fn execute_node(
        &mut self,
        seq: usize,
        node: &Arc<dyn ExecutionPlan>,
        input_handles: &[Vec<u64>],
    ) -> DfResult<(Vec<u64>, NodeMemoryStats)> {
        // Flatten the per-child partition handles + per-child counts.
        let counts: Vec<u64> = input_handles.iter().map(|c| c.len() as u64).collect();
        let flat: Vec<u64> = input_handles.iter().flatten().copied().collect();
        // Output partition count is bounded by target_partitions; a fixed
        // caller buffer avoids an FFI allocation/free for the handle array.
        const OUT_CAP: usize = 64;
        let mut out_buf = [0u64; OUT_CAP];
        let mut out_count: u64 = 0;
        // Per-partition stats (parallel to out_handles); see the FFI contract.
        let mut out_stats = [PeacockNodeStats::default(); OUT_CAP];
        let rc = unsafe {
            peacock_executor_execute_node(
                self.executor,
                seq as u64,
                flat.as_ptr(),
                counts.as_ptr(),
                counts.len() as u64,
                out_buf.as_mut_ptr(),
                OUT_CAP as u64,
                &mut out_count,
                out_stats.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(last_error(self.executor, "peacock_executor_execute_node"));
        }
        let n = out_count as usize;
        let out_handles: Vec<u64> = out_buf[..n].to_vec();
        // Cost = Σ-over-partitions of the PER-PARTITION ColAccum overhead (each
        // partition charged its own bitmap/offset +1 fixed terms) + the var-len
        // content C++ measured — matching the #13 CpuNodeExecutor's Σ-over-
        // partition golden, NOT ColAccum(Σ rows). Rust owns the byte formula
        // (logical_size_from_schema), single-sourced → no CPU/GPU drift.
        let schema = node.schema();
        // Scan's per-partition row groups come from the SAME RG→batch→partition
        // map the C++ side replays via set_row_groups — so the GPU's per-partition
        // sub-lines match the #13 CPU golden by construction (the golden is
        // GPU-VERIFIED, not just CPU-printed).
        let scan_map = node
            .as_any()
            .downcast_ref::<crate::gpu_rule::GpuScanExec>()
            .map(|s| s.batches_map())
            .unwrap_or(&[]);
        let mut rows = 0usize;
        let mut output_bytes = 0usize;
        let mut max_batch_rows = 0usize;
        let mut part_stats: Vec<PartitionStat> = Vec::with_capacity(n);
        for (k, st) in out_stats[..n].iter().enumerate() {
            let rp = st.rows as usize;
            let bp = logical_size_from_schema(&schema, rp, st.varlen_content_bytes as usize);
            rows += rp;
            output_bytes += bp;
            max_batch_rows = max_batch_rows.max(rp);
            part_stats.push(PartitionStat {
                out_rows: rp,
                out_bytes: bp,
                row_groups: scan_map.get(k).map(|e| e.row_groups.clone()).unwrap_or_default(),
            });
        }
        let stat = NodeMemoryStats {
            node_name: node.name().to_string(),
            allocated_bytes: 0, // not modeled on GPU (VRAM layout not compared)
            output_bytes,
            row_count: rows,
            max_batch_rows,
            // Only N>1 carries sub-lines (matches the CPU golden's N==1 ⇒ none).
            part_stats: if n > 1 { part_stats } else { Vec::new() },
        };
        Ok((out_handles, stat))
    }

    async fn materialize(&mut self, handles: &[u64]) -> DfResult<Vec<RecordBatch>> {
        let mut out = Vec::new();
        for &handle in handles {
            let mut out_ptr: *mut u8 = std::ptr::null_mut();
            let mut out_len: u64 = 0;
            let rc = unsafe {
                peacock_result_from_handle(self.executor, handle, &mut out_ptr, &mut out_len)
            };
            if rc != 0 {
                return Err(last_error(self.executor, "peacock_result_from_handle"));
            }
            if out_len == 0 || out_ptr.is_null() {
                continue;
            }
            let ipc = unsafe { std::slice::from_raw_parts(out_ptr, out_len as usize) };
            let batches = StreamReader::try_new(std::io::Cursor::new(ipc), None)
                .and_then(|r| r.collect::<Result<Vec<_>, _>>())
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            unsafe { peacock_result_free(out_ptr) };
            out.extend(batches);
        }
        Ok(out)
    }

    fn release(&mut self, handles: &[u64]) {
        for &handle in handles {
            unsafe { peacock_handle_release(self.executor, handle) };
        }
    }
}
impl Drop for GpuNodeExecutor {
    fn drop(&mut self) {
        unsafe { peacock_executor_end_plan(self.executor) };
    }
}
