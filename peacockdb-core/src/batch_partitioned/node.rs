//! [`GpuNode`] — what a plan node offers the driver and the validator.
//!
//! The tree is heterogeneous and planning is not hot, so this is the one place trait
//! objects are used; everything on the per-batch path is static (see `backend.rs`).

use super::error::PlanError;
use super::layout::NodeKind;

/// A limit's `skip`/`fetch`, carried by whichever node owns the interval: a mid-plan
/// `GpuLimit`, or the `GpuUnload` that absorbed a root-adjacent one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowInterval {
    pub skip: u64,
    pub fetch: Option<u64>,
}

pub trait GpuNode {
    /// Layout and schema live inside the kind.
    fn kind(&self) -> &NodeKind;

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
}
