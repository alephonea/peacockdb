//! The backend-agnostic node-by-node driver.
//!
//! One orchestrator drives a physical plan ONE node at a time: each node is
//! executed given handles to its already-computed child outputs, and returns a
//! handle to its own output plus per-node [`NodeMemoryStats`]. Intermediates stay
//! resident in the backend (GPU VRAM / CPU registry); results cross out only once,
//! at the root (`materialize`).
//!
//! Nothing here is CPU- or GPU-specific: the backends differ only in their
//! [`NodeExecutor`] impl (see [`super::backend`]). The orchestrator and the C++
//! session walk nodes in the SAME canonical post-order (children left-to-right,
//! then the node), so child handles line up.
//!
//! This is the canonical `execute_node_by_node`; the recursive/streaming CPU family
//! lives on [`super::full_table_cpu_executor::FullTableCpuExecutor`].

use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result as DfResult;
use datafusion::physical_plan::ExecutionPlan;

use super::executor::NodeMemoryStats;
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
    /// CoalescePartitions concats M→1; the scan emits N per its map.
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

    /// Drain the per-region DEVICE times recorded during this walk, as
    /// `(seq, partition, device_us)`.
    ///
    /// Separate from [`Self::execute_node`] because under the events mode the answer
    /// does not exist yet when a node returns — no sync has happened, and forcing one
    /// would defeat the mode. Called once, after [`Self::materialize`]. Only the GPU
    /// backend with events on has anything to report, hence the default.
    async fn collect_device_times(&mut self) -> DfResult<Vec<(usize, usize, u64)>> {
        Ok(Vec::new())
    }
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

    // After materialize (the root's device work is part of the walk) and before release
    // (which tears down the session owning the events).
    for (seq, partition, device_us) in backend.collect_device_times().await? {
        let Some(stat) = stats.get_mut(seq) else { continue };
        stat.device_us += device_us;
        // `part_stats` is emptied at N==1 by the golden convention, so a missing entry
        // is normal — the node total above already has it.
        if let Some(ps) = stat.part_stats.get_mut(partition) {
            ps.device_us += device_us;
        }
    }

    backend.release(&root_handles);
    Ok((batches, stats))
}
