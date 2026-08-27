//! The executor contracts, one per node category.
//!
//! Every state transition emits in the same call, so there is no wrong interleaving to
//! construct and output timing is a pure function of the call sequence. Every method
//! that ends a protocol consumes `self`, which makes four run-time guards the prototype
//! needed into compile errors: probing before `set_build`, a second `set_build`, probing
//! after `finish_and_fetch`, and accumulating after `mark_done_and_fetch`. The source's
//! consuming step removes a fifth — the driver's own exhaustion flag.

use super::backend::Backend;
use super::cpu_batch::CpuBatch;

/// Why a call failed: a message and no kind, because there is one response to all of them.
/// The driver adds the node and the lane and fails the query — a retry with a smaller batch
/// is #142's adaptive future and not this design.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError {
    pub message: String,
}

impl BackendError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BackendError {}

/// What a call gives back. Every one can fail, and a failure ends the query: the C++ side
/// resets the session and every resident table with it, so there is nothing to resume from.
pub type CallResult<T> = Result<(T, CallStats), BackendError>;

/// `scratch_bytes` is the measured transient; `None` when the run is not instrumented.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CallStats {
    pub scratch_bytes: Option<usize>,
}

pub trait Executor {
    /// State held between calls.
    fn resident_bytes(&self) -> usize;

    /// Pre-call model. May consult `self`, so an accumulator includes its state. Calls
    /// with no input batch — `mark_done_and_fetch`, `finish_and_fetch` — are modeled
    /// with `n_rows = 0, n_bytes = 0`.
    fn scratch_bytes(&self, n_rows: u64, n_bytes: usize) -> usize;
}

pub trait ExecExecutor<B: Backend>: Executor {
    fn exec(&mut self, batch: B::Batch) -> CallResult<B::Batch>;
}

pub trait BatchAccumulatorExecutor<B: Backend>: Executor {
    fn accumulate_and_fetch(&mut self, batch: B::Batch) -> CallResult<Vec<B::Batch>>;
    fn mark_done_and_fetch(self) -> CallResult<Vec<B::Batch>>;
}

/// Parameterized by the BATCH rather than by the backend: an event carries a batch, and
/// binding it to a whole backend would make the type unnameable until one exists.
pub enum LaneEvent<B: super::batch::Batch> {
    Batch(B),
    Done,
}

pub trait PartitionAccumulatorExecutor<B: Backend>: Executor {
    /// One call per lane event — the shape round-robin driving actually produces. The
    /// call delivering the last lane's `Done` is the emitting call.
    fn accumulate_and_fetch(
        &mut self,
        partition: usize,
        event: LaneEvent<B::Batch>,
    ) -> CallResult<Vec<B::Batch>>;
}

pub trait PartitionEmitterExecutor<B: Backend>: Executor {
    /// Exactly N outputs, some of them empty; N is a plan value, so the count is checked
    /// once inside the returned type rather than at each call site.
    fn emit(&mut self, batch: B::Batch) -> CallResult<Vec<B::Batch>>;
}

/// A typestate: build -> probe -> done, each transition consuming the last state.
pub trait JoinExecutor<B: Backend>: Executor {
    type Probing: ProbingJoin<B>;
    fn set_build(self, batch: B::Batch) -> CallResult<Self::Probing>;
}

pub trait ProbingJoin<B: Backend>: Executor {
    fn probe_and_fetch(&mut self, batch: B::Batch) -> CallResult<Vec<B::Batch>>;
    fn finish_and_fetch(self) -> CallResult<Vec<B::Batch>>;
}

/// Exhaustion consumes the source, so the driver's slot IS its liveness.
pub enum SourceStep<B: Backend> {
    Batch {
        batch: B::Batch,
        stats: CallStats,
        source: B::Source,
    },
    Exhausted,
}

pub trait SourceExecutor<B: Backend>: Executor {
    fn next_batch(self) -> Result<SourceStep<B>, BackendError>;
}

/// `length: u64::MAX` means to the end. Straight through to the fetch's row range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRange {
    pub offset: u64,
    pub length: u64,
}

impl RowRange {
    /// Every row, which is what a node with no interval above it asks for.
    pub const WHOLE: Self = Self {
        offset: 0,
        length: u64::MAX,
    };

    /// Whether this names the whole of a batch that size — in which case the call needs no
    /// range at all, and the trace should not read as a trimmed one.
    pub fn covers(&self, n_rows: u64) -> bool {
        self.offset == 0 && self.length >= n_rows
    }

    /// The rows of a batch this range actually names, as `(offset, length)`. The twin of
    /// C++'s `clamp_row_range` (`node_session.cpp`), which the export and the slice share
    /// so that the two cannot disagree — this is the same rule for the backend that never
    /// crosses the ABI, and the two answering differently would be a divergence no test
    /// of either one alone could see.
    pub fn clamp(&self, n_rows: u64) -> (u64, u64) {
        let offset = self.offset.min(n_rows);
        (offset, self.length.min(n_rows - offset))
    }
}

/// Unload is its own category because it is the one operator whose output is not
/// `B::Batch`: this is where data leaves the device, and the type says so. The row range
/// is a call argument because the count a root-adjacent limit derives from is cross-lane,
/// and an unload instance is per lane — only the driver holds that count.
pub trait UnloadExecutor<B: Backend>: Executor {
    fn unload(&mut self, batch: B::Batch, rows: RowRange) -> CallResult<CpuBatch>;
}

#[cfg(test)]
mod tests {
    use super::RowRange;

    /// The to-the-end sentinel is the case the clamp is written around: subtracting the
    /// offset from the row count rather than adding it to the length is what keeps
    /// `u64::MAX` from wrapping.
    #[test]
    fn a_range_to_the_end_takes_every_row_after_its_offset() {
        assert_eq!(RowRange::WHOLE.clamp(4), (0, 4));
        assert_eq!(
            RowRange {
                offset: 3,
                length: u64::MAX,
            }
            .clamp(4),
            (3, 1)
        );
    }

    /// A fetch legitimately overruns the batch it straddles, and an offset past the end
    /// names no rows rather than a negative count.
    #[test]
    fn a_range_past_the_end_clamps_to_what_is_there() {
        assert_eq!(
            RowRange {
                offset: 1,
                length: 100
            }
            .clamp(4),
            (1, 3)
        );
        assert_eq!(
            RowRange {
                offset: 9,
                length: 1
            }
            .clamp(4),
            (4, 0)
        );
    }
}
