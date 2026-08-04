//! The `Operator` contract: what every GPU-wrapper plan node exposes so the rest of
//! the engine can treat them uniformly instead of downcasting at each call site.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

/// How an operator maps INPUT partitions to OUTPUT partitions. This is the property
/// the node-by-node backends actually branch on, so encoding it here replaces a
/// downcast ladder with a single lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionTopology {
    /// The scan: emits N partitions from its row-group map (or 1 with no map).
    ScanEmit,
    /// 1:1 per partition — filter, project, and friends.
    Map,
    /// M -> 1 (CoalescePartitions).
    Collapse,
    /// M -> 1 preserving order (SortPreservingMerge).
    KWayMerge,
    /// 1 -> N by Spark-murmur3 hash.
    RepartitionHash,
    /// Two inputs -> one output.
    Join,
}

/// Implemented by every `Gpu*Exec` wrapper.
pub trait Operator: ExecutionPlan {
    /// The wrapped DataFusion node.
    fn inner(&self) -> &Arc<dyn ExecutionPlan>;

    fn partition_topology(&self) -> PartitionTopology;

    /// Whether the recursive full-table CPU driver replaces this wrapper with its
    /// inner node before executing.
    ///
    /// NOT uniformly true, and the asymmetry is LOAD-BEARING — see the per-operator
    /// impls and `strip_target` in [`super`]. Five operators deliberately return
    /// false; flipping them changes execution substitution and the reported
    /// `NodeMemoryStats.node_name`, which is a behavior change, not a cleanup.
    fn strips_to_inner(&self) -> bool {
        true
    }
}
