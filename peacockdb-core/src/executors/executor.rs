//! The executor contract: what every execution mode offers a caller, plus the
//! per-node stats types the drivers and backends report through.
//!
//! Five mode classes implement these (see [`super`]): full_table_cpu,
//! partitioned_cpu, full_table_gpu, partitioned_gpu, all_at_once_gpu. All but
//! all_at_once_gpu can also report per-node stats, so only they implement
//! [`InstrumentedExecutor`] — the all-at-once GPU path makes a single FFI call and
//! never sees individual nodes.

use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result as DfResult;
use datafusion::physical_plan::ExecutionPlan;

/// Run SQL to completion and hand back the result batches.
#[allow(async_fn_in_trait)]
pub trait Executor {
    async fn execute(&self, sql: &str) -> DfResult<Vec<RecordBatch>>;
}

/// An [`Executor`] that can additionally report the GPU-annotated plan it ran and
/// per-node [`NodeMemoryStats`] in post-order, so the stats line up with the tree.
#[allow(async_fn_in_trait)]
pub trait InstrumentedExecutor: Executor {
    async fn execute_instrumented(
        &self,
        sql: &str,
    ) -> DfResult<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>)>;
}

// ---------------------------------------------------------------------------
// Node-by-node CPU execution
// ---------------------------------------------------------------------------

/// Per-OUTPUT-partition breakdown of a node's stats. Empty
/// on a node's [`NodeMemoryStats`] means a single output partition (N=1): the
/// `.cpu.txt` golden renders `partitions=1` with NO per-partition sub-lines. When
/// the node emits N>1 output partitions (the real-partitioning device), there is
/// one entry per output partition, in partition order, and the golden renders a
/// `p{k}: …` sub-line per entry. `row_groups` is populated for the SCAN only (the
/// groups that output partition reads, matching the GPU's `set_row_groups`); for
/// every other node it is empty and the golden's `in_rows` is derived from the
/// child's per-partition `out_rows`.
#[derive(Clone, Default)]
pub struct PartitionStat {
    /// This output partition's row count.
    pub out_rows: usize,
    /// This output partition's logical byte size (its own per-partition `ColAccum`).
    pub out_bytes: usize,
    /// Scan only: the row groups this partition reads (empty for non-scan nodes).
    pub row_groups: Vec<u32>,
}

/// Per-node memory stats collected via the `on_node` callback.
#[derive(Clone, Default)]
pub struct NodeMemoryStats {
    /// Name of the CPU node that was executed (GPU wrapper already stripped).
    pub node_name: String,
    /// Sum of `get_array_memory_size()` across all output batches (allocated upper bound).
    pub allocated_bytes: usize,
    /// Logical byte size of all batches produced by this node (Σ over partitions).
    pub output_bytes: usize,
    /// Total number of output rows across all batches (Σ over partitions).
    pub row_count: usize,
    /// Largest single batch (in rows) produced by this node.
    /// Compare against `GpuScanExec.gpu_batch_size` to verify the memory contract.
    pub max_batch_rows: usize,
    /// Per-output-partition breakdown (empty ⇒ N=1, no sub-lines). See [`PartitionStat`].
    pub part_stats: Vec<PartitionStat>,
}
