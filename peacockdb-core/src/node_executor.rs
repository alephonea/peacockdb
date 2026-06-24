//! Unified CPU/GPU node-execution interface (Task #13, Phase 1).
//!
//! One backend-agnostic orchestrator drives a physical plan ONE node at a time:
//! each node is executed given handles to its already-computed child outputs, and
//! returns a handle to its own output plus per-node [`NodeMemoryStats`].
//! Intermediates stay resident in the backend (GPU VRAM / CPU registry); results
//! cross out only once, at the root (`materialize`).
//!
//! - [`CpuNodeExecutor`] = the DataFusion oracle (handles = `Vec<RecordBatch>` in
//!   a local registry); stats via the Part-1 `ColAccum` over the actual batches.
//! - [`GpuNodeExecutor`] = the C++/cuDF FFI (handles = GPU-resident tables); stats
//!   reconstructed in Rust from the FFI's `{rows, var-len content}` via the
//!   single-source [`crate::cpu_executor::logical_size_from_schema`] — so CPU and
//!   GPU costs are identical by construction whenever per-node row counts match.
//!
//! The orchestrator and the C++ session walk nodes in the SAME canonical
//! post-order (children left-to-right, then the node), so child handles line up.

use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result as DfResult;
use datafusion::physical_plan::ExecutionPlan;

use crate::cpu_executor::NodeMemoryStats;

/// A backend that executes individual plan nodes, holding intermediate outputs by
/// opaque handle. Used generically (static dispatch) by [`execute_node_by_node`],
/// never as `dyn`, so the missing auto-trait bounds the lint warns about don't apply.
#[allow(async_fn_in_trait)]
pub trait NodeExecutor {
    /// Execute the node at post-order `seq`, given the PARTITIONED outputs of its
    /// children: `input_handles[c]` is child `c`'s output partition handles (in
    /// child order). Returns this node's output PARTITION handles + its stats
    /// (stats are Σ-over-partitions, matching the CPU oracle's execute_stream
    /// coalesce). Ordinary ops map over partitions (same count out);
    /// CoalescePartitions concats M→1; the scan emits N per its map (Phase 2).
    async fn execute_node(
        &mut self,
        seq: usize,
        node: &Arc<dyn ExecutionPlan>,
        input_handles: &[Vec<u64>],
    ) -> DfResult<(Vec<u64>, NodeMemoryStats)>;

    /// Materialize the output behind these partition `handles` into record batches
    /// (root only; partitions concatenated in order).
    async fn materialize(&mut self, handles: &[u64]) -> DfResult<Vec<RecordBatch>>;

    /// Release resident partition handles (idempotent).
    fn release(&mut self, handles: &[u64]);
}

/// Flatten a plan into canonical post-order (children left-to-right, then node),
/// recording each node's children's post-order positions. Matches the C++
/// `NodeSession` indexing so handles align across the FFI.
fn post_order(root: &Arc<dyn ExecutionPlan>) -> Vec<(Arc<dyn ExecutionPlan>, Vec<usize>)> {
    let mut out: Vec<(Arc<dyn ExecutionPlan>, Vec<usize>)> = Vec::new();
    fn visit(
        node: &Arc<dyn ExecutionPlan>,
        out: &mut Vec<(Arc<dyn ExecutionPlan>, Vec<usize>)>,
    ) -> usize {
        let child_idxs: Vec<usize> = node.children().iter().map(|c| visit(c, out)).collect();
        out.push((Arc::clone(node), child_idxs));
        out.len() - 1
    }
    visit(root, &mut out);
    out
}

/// Drive a plan through a [`NodeExecutor`] node-by-node (post-order), returning
/// the root's materialized batches and the per-node stats (post-order).
pub async fn execute_node_by_node<E: NodeExecutor>(
    root: &Arc<dyn ExecutionPlan>,
    backend: &mut E,
) -> DfResult<(Vec<RecordBatch>, Vec<NodeMemoryStats>)> {
    let nodes = post_order(root);
    // Each node's output is a SET of partition handles (multi-handle model).
    let mut handles: Vec<Vec<u64>> = vec![Vec::new(); nodes.len()];
    let mut stats: Vec<NodeMemoryStats> = Vec::with_capacity(nodes.len());

    for (seq, (node, child_idxs)) in nodes.iter().enumerate() {
        let input_handles: Vec<Vec<u64>> = child_idxs.iter().map(|&i| handles[i].clone()).collect();
        let (out_handles, stat) = backend.execute_node(seq, node, &input_handles).await?;
        handles[seq] = out_handles;
        stats.push(stat);
    }

    let root_handles = handles.last().expect("plan has at least one node").clone();
    let batches = backend.materialize(&root_handles).await?;
    backend.release(&root_handles);
    Ok((batches, stats))
}

// ---------------------------------------------------------------------------
// CPU backend (DataFusion oracle) — available without the GPU toolchain.
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use datafusion::execution::TaskContext;

use crate::cpu_executor::execute_single_node;

/// CPU backend: handles are `Vec<RecordBatch>` held in a local registry; each
/// node runs through the same DataFusion machinery as the recursive executor, so
/// its stats are byte-identical to `execute_node_by_node`.
pub struct CpuNodeExecutor {
    task_ctx: Arc<TaskContext>,
    registry: HashMap<u64, Vec<RecordBatch>>,
    next_handle: u64,
}

impl CpuNodeExecutor {
    pub fn new(task_ctx: Arc<TaskContext>) -> Self {
        Self { task_ctx, registry: HashMap::new(), next_handle: 1 }
    }
}

impl NodeExecutor for CpuNodeExecutor {
    async fn execute_node(
        &mut self,
        _seq: usize,
        node: &Arc<dyn ExecutionPlan>,
        input_handles: &[Vec<u64>],
    ) -> DfResult<(Vec<u64>, NodeMemoryStats)> {
        // Step (ii): single-partition behavior under the multi-handle interface —
        // each child's partition handles are concatenated into one input and the
        // node runs once → one output partition. (Step iii adds the map-based
        // N-partition scan + CoalescePartitions M→1 concat; the Σ-over-partitions
        // stat then falls out of running per-partition and summing.)
        let inputs: Vec<Vec<RecordBatch>> = input_handles
            .iter()
            .map(|child_parts| {
                let mut batches = Vec::new();
                for h in child_parts {
                    batches.extend(self.registry.remove(h).unwrap_or_default());
                }
                batches
            })
            .collect();
        let (batches, stat) = execute_single_node(node, inputs, self.task_ctx.clone()).await?;
        let handle = self.next_handle;
        self.next_handle += 1;
        self.registry.insert(handle, batches);
        Ok((vec![handle], stat))
    }

    async fn materialize(&mut self, handles: &[u64]) -> DfResult<Vec<RecordBatch>> {
        let mut out = Vec::new();
        for h in handles {
            out.extend(self.registry.remove(h).unwrap_or_default());
        }
        Ok(out)
    }

    fn release(&mut self, handles: &[u64]) {
        for h in handles {
            self.registry.remove(h);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU backend (C++/cuDF FFI) — only when the GPU executor is linked.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "rust-only"))]
pub use gpu::GpuNodeExecutor;

#[cfg(not(feature = "rust-only"))]
mod gpu {
    use super::*;

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
            let mut st = PeacockNodeStats::default();
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
                    &mut st,
                )
            };
            if rc != 0 {
                return Err(last_error(self.executor, "peacock_executor_execute_node"));
            }
            let out_handles: Vec<u64> = out_buf[..out_count as usize].to_vec();
            let rows = st.rows as usize;
            // Single-source cost: Rust applies the ColAccum overhead from the
            // node's output schema + (Σ-over-partitions) rows, plus the var-len
            // content C++ measured. (Per-partition overhead accounting for >1
            // partition is Inc1 step iii.)
            let schema = node.schema();
            let output_bytes =
                logical_size_from_schema(&schema, rows, st.varlen_content_bytes as usize);
            let stat = NodeMemoryStats {
                node_name: node.name().to_string(),
                allocated_bytes: 0, // not modeled on GPU (VRAM layout not compared)
                output_bytes,
                row_count: rows,
                max_batch_rows: rows,
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
}
