//! What planning and validation return. Both are plan-time: this mode refuses a shape
//! it cannot run where the planner can see it, rather than throwing mid-query.

use datafusion::error::DataFusionError;
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

/// Which check refused the call. It crosses the boundary as data because a caller may one
/// day answer the two differently — a pre-call refusal is a call that never ran, where a
/// post-call one is work already done. Today both end the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum When {
    PreCall,
    PostCall,
}

impl When {
    /// The words the message uses. Kept beside the variant so the field and the sentence
    /// cannot drift: the field is for code and the sentence is for a person, and neither
    /// replaces the other.
    pub fn describe(&self) -> &'static str {
        match self {
            Self::PreCall => "before the call",
            Self::PostCall => "after it",
        }
    }
}

/// What can go wrong while a plan runs, as against [`PlanError`], which is what can go
/// wrong before it does. A trip is a clean query failure, and what it promises is narrow:
/// a check that sees an accounted total over budget fails the query. The peak is an
/// observation taken elsewhere and is not what is checked, so it can exceed a budget that
/// never tripped — see the Memory accounting section of the task spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// One kind of failure with a phase, rather than two kinds: every caller that does not
    /// care about the phase — the DataFusion conversion, `Display` — keeps one arm, and a
    /// pair of variants mapping to the same thing is a pair that drifts.
    BudgetExceeded { when: When, message: String },
    /// A protocol violation no type can reach: a build side that produced zero or two
    /// batches, an `emit` returning other than N, a step after finishing.
    Protocol(String),
    /// A call failed. The session is gone with it, so the query is over.
    CallFailed(String),
    /// The backend has no executor for this node, or the set it returned drives a
    /// different category than the node is.
    Backend(PlanError),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetExceeded { message, .. } => write!(f, "{message}"),
            Self::Protocol(what) => write!(f, "batch-partitioned protocol violation: {what}"),
            Self::CallFailed(what) => write!(f, "batch-partitioned call failed: {what}"),
            Self::Backend(inner) => write!(f, "{inner}"),
        }
    }
}

impl std::error::Error for RunError {}

impl From<RunError> for DataFusionError {
    fn from(error: RunError) -> Self {
        match error {
            // The same class the legacy enforcer raises, so a caller distinguishing an
            // over-budget query from a broken one keeps doing it by the same match.
            RunError::BudgetExceeded { message, .. } => {
                DataFusionError::ResourcesExhausted(message)
            }
            RunError::Protocol(_) | RunError::CallFailed(_) => {
                DataFusionError::Execution(error.to_string())
            }
            RunError::Backend(_) => DataFusionError::Plan(error.to_string()),
        }
    }
}
