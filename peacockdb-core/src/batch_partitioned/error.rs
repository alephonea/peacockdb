//! What planning and validation return. Both are plan-time: this mode refuses a shape
//! it cannot run where the planner can see it, rather than throwing mid-query.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// A shape this mode does not implement — window functions (#143), a mixed
    /// distinct (#62), a value-form CASE (#57). Names the shape, not the node's fix.
    Unsupported(String),
    /// A plan that violates what a node requires of its children. Names the fix, since
    /// the node knows it: "the planner inserts `GpuMergePartitions` below it".
    Invalid(String),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "unsupported in batch-partitioned mode: {what}"),
            Self::Invalid(what) => write!(f, "invalid batch-partitioned plan: {what}"),
        }
    }
}

impl std::error::Error for PlanError {}
