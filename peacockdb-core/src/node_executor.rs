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

use datafusion::datasource::physical_plan::parquet::{ParquetAccessPlan, RowGroupAccess};
use datafusion::datasource::physical_plan::ParquetExec;
use datafusion::execution::TaskContext;

use datafusion::error::DataFusionError;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

use crate::cpu_executor::execute_single_node;
use crate::gpu_rowgroup_prune::all_row_groups;
use crate::gpu_rule::{GpuCoalescePartitionsExec, GpuScanExec, GpuSortPreservingMergeExec};

/// Partition-collapsing nodes: N input partitions → 1 output (the CPU oracle
/// realizes this by concatenating all child partitions into a single input). Every
/// other single-child node maps over its input partitions (count preserved).
fn collapses_partitions(node: &Arc<dyn ExecutionPlan>) -> bool {
    let any = node.as_any();
    any.is::<GpuCoalescePartitionsExec>()
        || any.is::<GpuSortPreservingMergeExec>()
        || any.is::<CoalescePartitionsExec>()
        || any.is::<SortPreservingMergeExec>()
}

/// Σ-over-partitions accumulation of per-partition [`NodeMemoryStats`]: row counts
/// and (per-partition `ColAccum`) byte sizes ADD; `max_batch_rows` takes the max.
/// This is exactly the GPU's per-node accounting (each partition charged its own
/// overhead), so the tp8 cost golden generated here matches the real 8-way GPU.
fn merge_stats(acc: &mut Option<NodeMemoryStats>, s: NodeMemoryStats) {
    match acc {
        None => *acc = Some(s),
        Some(a) => {
            a.allocated_bytes += s.allocated_bytes;
            a.output_bytes += s.output_bytes;
            a.row_count += s.row_count;
            a.max_batch_rows = a.max_batch_rows.max(s.max_batch_rows);
        }
    }
}

/// Execute a `GpuScanExec` carrying a non-empty RG→batch→partition map, returning
/// one materialized partition per map entry plus the Σ-over-partitions stats.
///
/// Each partition restricts the underlying `ParquetExec` to exactly that entry's
/// row groups via a [`ParquetAccessPlan`] on the file's `extensions` — the SAME
/// row-group→partition assignment the GPU replays (cuDF `set_row_groups`), so
/// per-partition row counts match by construction. The scan's predicate + parquet
/// options are preserved, so each partition's rows are identical to what the full
/// DataFusion scan would yield for those groups (peacock RG-prunes but does NOT
/// push row filters into the scan, so all rows of the selected groups are read).
async fn cpu_scan_partitions(
    scan: &GpuScanExec,
    task_ctx: Arc<TaskContext>,
) -> DfResult<(Vec<Vec<RecordBatch>>, NodeMemoryStats)> {
    let parquet = scan.inner().as_any().downcast_ref::<ParquetExec>().ok_or_else(|| {
        DataFusionError::Internal("cpu_scan_partitions: GpuScanExec inner is not a ParquetExec".into())
    })?;
    // Access-plan length MUST equal the file's true row-group count or DataFusion
    // rejects the plan.
    let total_rgs = all_row_groups(parquet).map(|v| v.len()).ok_or_else(|| {
        DataFusionError::Internal(
            "cpu_scan_partitions: cannot determine row-group count for partitioned scan".into(),
        )
    })?;
    let base_file = parquet
        .base_config()
        .file_groups
        .iter()
        .flatten()
        .next()
        .cloned()
        .ok_or_else(|| DataFusionError::Internal("cpu_scan_partitions: scan has no source file".into()))?;

    let mut out_parts: Vec<Vec<RecordBatch>> = Vec::with_capacity(scan.batches_map().len());
    let mut acc: Option<NodeMemoryStats> = None;
    for entry in scan.batches_map() {
        // Scan ONLY this partition's row groups; skip every other group.
        let mut access = ParquetAccessPlan::new(vec![RowGroupAccess::Skip; total_rgs]);
        for &rg in &entry.row_groups {
            access.scan(rg as usize);
        }
        let mut file = base_file.clone();
        // The base file may carry a byte RANGE (DataFusion's tp8 split); clear it so
        // the access plan ALONE decides which row groups this partition reads.
        file.range = None;
        file.extensions = Some(Arc::new(access) as Arc<dyn std::any::Any + Send + Sync>);

        let mut config = parquet.base_config().clone();
        config.file_groups = vec![vec![file]];
        // Preserve the scan predicate per-partition. It only ROW-GROUP-prunes here
        // (pushdown_filters is off → no per-row filtering at the scan), and the map's
        // RGs are already the predicate survivors, so it removes nothing — the
        // partition reads exactly its mapped RGs, matching the GPU's set_row_groups.
        // CPU and GPU agree as long as no SUB-row-group (page) pruning diverges; today
        // it can't (our DuckDB parquet carries no page index / bloom filters).
        let mut builder = ParquetExec::builder(config)
            .with_table_parquet_options(parquet.table_parquet_options().clone());
        if let Some(pred) = parquet.predicate() {
            builder = builder.with_predicate(pred.clone());
        }
        let part_parquet: Arc<dyn ExecutionPlan> = Arc::new(builder.build());
        // Re-wrap so execute_single_node applies the same gpu_batch_size override
        // and records the node as "ParquetExec" (matching the recursive oracle).
        let part_scan: Arc<dyn ExecutionPlan> =
            Arc::new(GpuScanExec::new(part_parquet, scan.gpu_batch_size));
        let (batches, stat) = execute_single_node(&part_scan, vec![], task_ctx.clone()).await?;
        out_parts.push(batches);
        merge_stats(&mut acc, stat);
    }
    let stat = acc.unwrap_or_else(|| NodeMemoryStats {
        node_name: "ParquetExec".to_string(),
        allocated_bytes: 0,
        output_bytes: 0,
        row_count: 0,
        max_batch_rows: 0,
    });
    Ok((out_parts, stat))
}

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

    fn store(&mut self, batches: Vec<RecordBatch>) -> u64 {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.registry.insert(handle, batches);
        handle
    }
}

impl NodeExecutor for CpuNodeExecutor {
    async fn execute_node(
        &mut self,
        _seq: usize,
        node: &Arc<dyn ExecutionPlan>,
        input_handles: &[Vec<u64>],
    ) -> DfResult<(Vec<u64>, NodeMemoryStats)> {
        // (iii) SCAN with an explicit RG→batch→partition map → N partition handles.
        if let Some(scan) = node.as_any().downcast_ref::<GpuScanExec>() {
            if !scan.batches_map().is_empty() {
                let (parts, stat) = cpu_scan_partitions(scan, self.task_ctx.clone()).await?;
                let handles: Vec<u64> = parts.into_iter().map(|b| self.store(b)).collect();
                return Ok((handles, stat));
            }
        }

        // Materialize each child's partition batches (consuming registry handles).
        let child_parts: Vec<Vec<Vec<RecordBatch>>> = input_handles
            .iter()
            .map(|child| child.iter().map(|h| self.registry.remove(h).unwrap_or_default()).collect())
            .collect();

        // Ordinary single-child op with the multi-partition map active → MAP the
        // node over each input partition (count preserved); the Σ-over-partitions
        // stat falls out of summing per-partition runs. Partition-collapsing ops
        // (CoalescePartitions / SortPreservingMerge) and any multi-/zero-child node
        // fall through to the concat-into-one path — which also covers tp1 (single
        // partition) byte-identically (one partition in → one run → one out).
        if !collapses_partitions(node) && child_parts.len() == 1 && !child_parts[0].is_empty() {
            let mut handles = Vec::with_capacity(child_parts[0].len());
            let mut acc: Option<NodeMemoryStats> = None;
            for part in &child_parts[0] {
                let (batches, stat) =
                    execute_single_node(node, vec![part.clone()], self.task_ctx.clone()).await?;
                handles.push(self.store(batches));
                merge_stats(&mut acc, stat);
            }
            return Ok((handles, acc.expect("non-empty child has at least one partition")));
        }

        // Concat-into-one: each child's partitions concatenated into a single input,
        // the node runs once → one output partition.
        let inputs: Vec<Vec<RecordBatch>> =
            child_parts.into_iter().map(|child| child.into_iter().flatten().collect()).collect();
        let (batches, stat) = execute_single_node(node, inputs, self.task_ctx.clone()).await?;
        let handle = self.store(batches);
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
            let mut rows = 0usize;
            let mut output_bytes = 0usize;
            let mut max_batch_rows = 0usize;
            for st in &out_stats[..n] {
                let rp = st.rows as usize;
                rows += rp;
                output_bytes +=
                    logical_size_from_schema(&schema, rp, st.varlen_content_bytes as usize);
                max_batch_rows = max_batch_rows.max(rp);
            }
            let stat = NodeMemoryStats {
                node_name: node.name().to_string(),
                allocated_bytes: 0, // not modeled on GPU (VRAM layout not compared)
                output_bytes,
                row_count: rows,
                max_batch_rows,
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
