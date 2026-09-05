//! The two drivers, the schedule they follow, and the accounting they fail a query on.
//!
//! [`partitioned`] owns the tree, the queues, the schedule and the three cross-lane
//! categories; [`single_partition`] is one lane of one lane-scoped node, deciding which
//! call that lane's input state calls for and making exactly one. Both are generic over
//! [`Backend`](super::backend::Backend), so each backend monomorphizes and nothing on the
//! per-batch path is boxed. The design is `llm-wiki/tasks/batch_partitioned_executor.md`,
//! sections Drivers and Memory accounting.

mod accounting;
pub(crate) mod index;
pub use index::{nodes_as_recorded, post_order_of_every_node};
#[cfg(test)]
mod mock;
mod partitioned;
#[cfg(test)]
mod plans;
mod scheduler;
mod single_partition;
mod measurements;

#[cfg(test)]
mod tests;

pub use accounting::Underestimate;
pub use partitioned::{RunReport, batch_partitioned_driver};
pub use measurements::{Measured, Measurements, Region, join_regions, node_measured};

use crate::batch_partitioned::error::RunError;
use accounting::Trip;

/// What an executor was asked to do. An enum rather than a label: this is written once per
/// call, and a per-call format is a cost the trace does not need to impose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallKind {
    NextBatch,
    SourceExhausted,
    Exec,
    Unload,
    /// An unload of part of a batch — the interval straddles one of its ends.
    UnloadRange,
    /// A batch outside the interval, released where it stood. No call was made, which is
    /// the whole saving and the thing a test on the rows returned cannot see.
    ReleaseUnwanted,
    Accumulate,
    MarkDone,
    SetBuild,
    /// A join lane whose build side ended with no batch — its scatter gave it no build
    /// rows. No call was made and none will be: what the lane owed was nothing.
    NoBuild,
    Probe,
    Finish,
    EndOfInput,
    Emit,
    EmitDone,
    /// One lane's batch delivered to a cross-lane accumulator.
    LaneEvent,
    /// One lane's end delivered to a cross-lane accumulator.
    LaneDone,
    Forward,
    ForwardDone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceEvent {
    pub step: u32,
    pub node: u32,
    pub lane: u32,
    pub call: CallKind,
    pub outputs: u32,
}

/// A trip carries no name, and the driver is what can supply one — so every step returns
/// this and the driver renders it at the one place a query ends.
#[derive(Debug)]
pub(crate) enum StepError {
    Run(RunError),
    Trip(Trip),
}

impl From<RunError> for StepError {
    fn from(error: RunError) -> Self {
        Self::Run(error)
    }
}

impl From<Trip> for StepError {
    fn from(trip: Trip) -> Self {
        Self::Trip(trip)
    }
}
