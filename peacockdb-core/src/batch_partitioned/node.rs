//! [`GpuNode`] — what a plan node offers the driver and the validator.
//!
//! The tree is heterogeneous and planning is not hot, so this is the one place trait
//! objects are used; everything on the per-batch path is static (see `backend.rs`).

use std::any::Any;

use super::error::PlanError;
use super::layout::NodeKind;

/// A limit's `skip`/`fetch`, carried by whichever node owns the interval: a mid-plan
/// `GpuLimit`, or the `GpuUnload` that absorbed a root-adjacent one. Intervals nest — each
/// counts the stream its own node is handed — and the spec's limit lowering rule says why
/// only the non-adjacent form ever arrives that way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowInterval {
    pub skip: u64,
    pub fetch: Option<u64>,
}

impl RowInterval {
    /// The row after the last one wanted, counting from the start of this node's stream.
    /// `None` is a pure offset: no prefix determines the answer, so it never satisfies.
    pub fn stop(&self) -> Option<u64> {
        self.fetch.map(|fetch| self.skip + fetch)
    }

    /// True once no further row could change the answer — what `is_satisfied` asks.
    pub fn satisfied_by(&self, seen: u64) -> bool {
        self.stop().is_some_and(|stop| seen >= stop)
    }

    /// Which rows of the next batch are wanted, or `None` to release it uncalled. `seen`
    /// is how many rows of the stream have already gone past this node, which is a count
    /// across lanes and therefore the driver's rather than an executor's.
    pub fn range_of(&self, seen: u64, n_rows: u64) -> Option<super::executor::RowRange> {
        let start = self.skip.saturating_sub(seen);
        let stop = match self.stop() {
            Some(stop) => n_rows.min(stop.saturating_sub(seen)),
            None => n_rows,
        };
        // `then`, not `then_some`: the subtraction is the answer only when it is in range,
        // and an eager argument underflows on every batch of the skip prefix.
        (start < stop).then(|| super::executor::RowRange {
            offset: start,
            length: stop - start,
        })
    }
}

pub trait GpuNode: std::fmt::Debug {
    /// Layout and schema live inside the kind.
    fn kind(&self) -> &NodeKind;

    /// What a plan line and a validation message call this node. The registry is the
    /// mapping, so a node kind is named in one place; a node outside it — a hand-built
    /// one under test — says its own name rather than reaching a registry it is not in.
    fn name(&self) -> &'static str {
        super::nodes::node_name(self.as_any())
    }

    fn children(&self) -> Vec<&dyn GpuNode>;

    /// Checks children's schemas, partition topology, key distribution, sortedness and
    /// batch layout against this node's requirements, and captured column indices
    /// against child schemas. Runs before the generic structural rules, because a node
    /// can name the fix where a generic rule can only say what is wrong.
    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError>;

    /// `Some` only where a limit's interval landed — see the limit lowering rule.
    fn row_interval(&self) -> Option<RowInterval> {
        None
    }

    /// The one downcast point. Consumers that need a node's own parameters — the
    /// renderer, a backend's executor match, the serializer — go through
    /// [`nodes::as_node_ref`](super::nodes::as_node_ref) rather than downcasting here.
    fn as_any(&self) -> &dyn Any;
}
