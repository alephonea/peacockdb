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
    fn exec(&mut self, batch: B::Batch) -> (B::Batch, CallStats);
}

pub trait BatchAccumulatorExecutor<B: Backend>: Executor {
    fn accumulate_and_fetch(&mut self, batch: B::Batch) -> (Vec<B::Batch>, CallStats);
    fn mark_done_and_fetch(self) -> (Vec<B::Batch>, CallStats);
}

pub enum LaneEvent<B: Backend> {
    Batch(B::Batch),
    Done,
}

pub trait PartitionAccumulatorExecutor<B: Backend>: Executor {
    /// One call per lane event — the shape round-robin driving actually produces. The
    /// call delivering the last lane's `Done` is the emitting call.
    fn accumulate_and_fetch(
        &mut self,
        partition: usize,
        event: LaneEvent<B>,
    ) -> (Vec<B::Batch>, CallStats);
}

pub trait PartitionEmitterExecutor<B: Backend>: Executor {
    /// Exactly N outputs, some of them empty; N is a plan value, so the count is checked
    /// once inside the returned type rather than at each call site.
    fn emit(&mut self, batch: B::Batch) -> (Vec<B::Batch>, CallStats);
}

/// A typestate: build -> probe -> done, each transition consuming the last state.
pub trait JoinExecutor<B: Backend>: Executor {
    type Probing: ProbingJoin<B>;
    fn set_build(self, batch: B::Batch) -> (Self::Probing, CallStats);
}

pub trait ProbingJoin<B: Backend>: Executor {
    fn probe_and_fetch(&mut self, batch: B::Batch) -> (Vec<B::Batch>, CallStats);
    fn finish_and_fetch(self) -> (Vec<B::Batch>, CallStats);
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
    fn next_batch(self) -> SourceStep<B>;
}

/// `length: u64::MAX` means to the end. Straight through to the fetch's row range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRange {
    pub offset: u64,
    pub length: u64,
}

/// Unload is its own category because it is the one operator whose output is not
/// `B::Batch`: this is where data leaves the device, and the type says so. The row range
/// is a call argument because the count a root-adjacent limit derives from is cross-lane,
/// and an unload instance is per lane — only the driver holds that count.
pub trait UnloadExecutor<B: Backend>: Executor {
    fn unload(&mut self, batch: B::Batch, rows: RowRange) -> (CpuBatch, CallStats);
}
