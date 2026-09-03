//! `batch_single_partition_driver` — one lane of one lane-scoped node.
//!
//! The four lane-scoped categories plus unload get one executor instance per (node, lane),
//! and this is that instance's state machine: it decides which call the lane's current
//! input state calls for, makes exactly one, and reports whether the lane will ever
//! produce again. Everything cross-partition is the partitioned driver's.
//!
//! Deciding and doing are separate on purpose. Runnability asks the same question the
//! driver later acts on, and a predicate that borrowed the input queues could not be asked
//! while the executor is held mutably — so [`LaneDriver::select`] reads an availability
//! summary and the caller hands over the batch it named.

use std::mem;

use super::accounting::{Held, ResidentAccountant, Slot};
use super::{CallKind, StepError};
use crate::batch_partitioned::backend::{Backend, NodeExecutors};
use crate::batch_partitioned::cpu_batch::CpuBatch;
use crate::batch_partitioned::error::{PlanError, RunError};
use crate::batch_partitioned::executor::{
    AbiCalls, BackendError, BatchAccumulatorExecutor, CallStats, ExecExecutor, JoinExecutor,
    ProbingJoin, RowRange, SourceExecutor, SourceStep, UnloadExecutor,
};
use crate::batch_partitioned::node::GpuNode;
use crate::batch_partitioned::nodes::ExecutorCategory;

pub(crate) const BUILD_SLOT: usize = 0;
pub(crate) const PROBE_SLOT: usize = 1;

/// What one lane's input edges look like right now. `done` means no further batch can ever
/// arrive on that slot: the producer finished and its queue for this lane is empty.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Avail {
    pub has: [bool; 2],
    pub done: [bool; 2],
}

/// Which call this lane's input state calls for. The unload's row range is not in here: it
/// comes from a count across lanes that only the partitioned driver holds, so it is an
/// argument of the call rather than part of the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneCall {
    NextBatch,
    Exec,
    Unload,
    Accumulate,
    MarkDone,
    SetBuild,
    /// The build side ended with no batch. The executor says whether the lane owes
    /// nothing — six of the nine join types do — and the lane drains from there.
    NoBuild,
    /// A probe batch for a lane that has no build side: released rather than probed, since
    /// what it could have matched is not there.
    DropProbe,
    Probe,
    Finish,
    /// No batch can arrive again, and this category has nothing to emit at the end — the
    /// lane simply ends, with no call made.
    EndOfInput,
}

impl LaneCall {
    /// Which input slot the call consumes, in the node's own child order — so a probe
    /// reads the right child and a build the left. One answer, because the driver pops
    /// the batch and the lane consumes it, and the two reading it differently is a join
    /// probing its own build side.
    pub(crate) fn consumes(&self) -> Option<usize> {
        match self {
            Self::Exec | Self::Unload | Self::Accumulate => Some(0),
            Self::SetBuild => Some(BUILD_SLOT),
            Self::Probe | Self::DropProbe => Some(PROBE_SLOT),
            Self::NextBatch | Self::MarkDone | Self::Finish | Self::NoBuild | Self::EndOfInput => {
                None
            }
        }
    }
}

/// Where the executor comes from, so `run` can build one on first use without the driver
/// deciding when that is.
pub(crate) struct LaneSite<'a, B: Backend> {
    pub ctx: &'a B::Context,
    pub node: &'a dyn GpuNode,
    pub category: ExecutorCategory,
    /// The node's children-first position, which is what a recipe is addressed by.
    pub post_order: usize,
    pub lane: usize,
    pub slot: Slot,
}

pub(crate) enum LaneOutputs<B: Backend> {
    Device(Vec<Held<B::Batch>>),
    /// Only an unload's, which is the one operator whose output is not `B::Batch`.
    Host(Vec<Held<CpuBatch>>),
}

pub(crate) struct LaneOutcome<B: Backend> {
    pub outputs: LaneOutputs<B>,
    pub finished: bool,
    pub call: CallKind,
    /// The ABI calls this one made, or `None` where no backend executor was reached at
    /// all — an end-of-input, a lane that owed no build. Carried up rather than recorded
    /// where it is taken: only the driver knows which node and which lane this was.
    ///
    /// `None` rather than an unmeasured `AbiCalls`, because the two say different things
    /// and one of them must not take a place in the per-batch record: a driver-only step
    /// that took a slot there would shift every batch after it by one.
    pub calls: Option<AbiCalls>,
}

impl<B: Backend> LaneOutcome<B> {
    fn made(mut self, calls: AbiCalls) -> Self {
        self.calls = Some(calls);
        self
    }
}

/// The executor, in whichever state it is. `Unbuilt` is a lane never entered — for a join
/// that is still its build phase, since nothing has been set.
enum LaneState<B: Backend> {
    Unbuilt,
    Source(B::Source),
    Exec(B::Exec),
    BatchAcc(B::BatchAcc),
    Build(B::Join),
    Probe(<B::Join as JoinExecutor<B>>::Probing),
    /// A join whose build side was empty and which owes nothing: its probe batches still
    /// arrive, and the lane reads them only to let them go.
    Draining,
    Unload(B::Unload),
    Finished,
}

pub(crate) struct LaneDriver<B: Backend> {
    state: LaneState<B>,
}

impl<B: Backend> Default for LaneDriver<B> {
    fn default() -> Self {
        Self {
            state: LaneState::Unbuilt,
        }
    }
}

impl<B: Backend> LaneDriver<B> {
    pub(crate) fn is_finished(&self) -> bool {
        matches!(self.state, LaneState::Finished)
    }

    /// True while this lane of a join has yet to leave its build phase. A lane never
    /// entered has not been, so it counts.
    pub(crate) fn awaits_build(&self) -> bool {
        matches!(self.state, LaneState::Unbuilt | LaneState::Build(_))
    }

    /// Whether this lane can make progress. It answers the same question as `select` and
    /// is checked against it over the whole cross product in the tests, because two
    /// readings of one rule are two rules.
    pub(crate) fn can_step(&self, category: ExecutorCategory, avail: &Avail) -> bool {
        if self.is_finished() {
            return false;
        }
        match category {
            // A source is its own input.
            ExecutorCategory::Source => true,
            ExecutorCategory::Join => {
                let slot = if self.awaits_build() {
                    BUILD_SLOT
                } else {
                    PROBE_SLOT
                };
                avail.has[slot] || avail.done[slot]
            }
            _ => avail.has[0] || avail.done[0],
        }
    }

    /// What to call, for a lane `can_step` has already accepted.
    pub(crate) fn select(
        &self,
        category: ExecutorCategory,
        avail: &Avail,
    ) -> Result<LaneCall, RunError> {
        debug_assert!(
            self.can_step(category, avail),
            "select on a lane that cannot step"
        );
        Ok(match category {
            ExecutorCategory::Source => LaneCall::NextBatch,
            ExecutorCategory::Exec if avail.has[0] => LaneCall::Exec,
            ExecutorCategory::Unload if avail.has[0] => LaneCall::Unload,
            ExecutorCategory::Exec | ExecutorCategory::Unload => LaneCall::EndOfInput,
            ExecutorCategory::BatchAccumulator if avail.has[0] => LaneCall::Accumulate,
            ExecutorCategory::BatchAccumulator => LaneCall::MarkDone,
            // A lane whose scatter gave its build side no rows: routine for a small table
            // over many lanes, and the executor is asked what the type owes rather than
            // the driver assuming the plan is at fault.
            ExecutorCategory::Join if self.awaits_build() && !avail.has[BUILD_SLOT] => {
                LaneCall::NoBuild
            }
            ExecutorCategory::Join if self.awaits_build() => LaneCall::SetBuild,
            ExecutorCategory::Join if matches!(self.state, LaneState::Draining) => {
                match avail.has[PROBE_SLOT] {
                    true => LaneCall::DropProbe,
                    false => LaneCall::EndOfInput,
                }
            }
            ExecutorCategory::Join if avail.has[BUILD_SLOT] => {
                return Err(RunError::Protocol(
                    "a join's build side produced a second batch".to_string(),
                ));
            }
            ExecutorCategory::Join if avail.has[PROBE_SLOT] => LaneCall::Probe,
            ExecutorCategory::Join => LaneCall::Finish,
            other => {
                return Err(RunError::Protocol(format!(
                    "{other:?} is not lane-scoped and has no lane call"
                )));
            }
        })
    }

    /// Make exactly one call. The input arrives already accounted — it is alive on the
    /// device while the call runs, so the pre-check counts it — and outputs leave here
    /// held, so the caller only queues them.
    pub(crate) fn run(
        &mut self,
        site: &LaneSite<'_, B>,
        call: LaneCall,
        input: Option<Held<B::Batch>>,
        rows: RowRange,
        acct: &mut ResidentAccountant,
    ) -> Result<LaneOutcome<B>, StepError> {
        if self.is_finished() {
            return Err(RunError::Protocol(format!(
                "{}: a lane stepped after it finished",
                site.node.name()
            ))
            .into());
        }
        if matches!(self.state, LaneState::Unbuilt) {
            self.build(site)?;
        }
        let (n_rows, n_bytes) = input
            .as_ref()
            .map_or((0, 0), |held| (held.rows(), held.bytes));
        match call {
            LaneCall::EndOfInput => {
                self.state = LaneState::Finished;
                acct.forget(site.slot);
                Ok(self.outcome(LaneOutputs::Device(Vec::new()), true, CallKind::EndOfInput))
            }
            LaneCall::NextBatch => self.source_step(site, acct),
            LaneCall::Exec => {
                let batch = self.expect_input(input, site)?;
                let LaneState::Exec(executor) = &mut self.state else {
                    return Err(self.wrong_state(site, "exec"));
                };
                let modelled = acct.begin_call(site.slot, executor, n_rows, n_bytes)?;
                let (out, stats) = executor
                    .exec(batch.batch)
                    .map_err(|e| failed(site, acct, Some(n_bytes), e))?;
                let out = Held::of(out);
                acct.release(batch.bytes)?;
                acct.hold(out.bytes);
                acct.end_call(site.slot, executor, &stats, modelled)?;
                Ok(self
                    .outcome(LaneOutputs::Device(vec![out]), false, CallKind::Exec)
                    .made(stats.calls))
            }
            LaneCall::Unload => {
                let batch = self.expect_input(input, site)?;
                let LaneState::Unload(executor) = &mut self.state else {
                    return Err(self.wrong_state(site, "unload"));
                };
                let modelled = acct.begin_call(site.slot, executor, n_rows, n_bytes)?;
                let (out, stats) = executor
                    .unload(batch.batch, rows)
                    .map_err(|e| failed(site, acct, Some(n_bytes), e))?;
                let out = Held::of(out);
                acct.release(batch.bytes)?;
                acct.hold(out.bytes);
                acct.end_call(site.slot, executor, &stats, modelled)?;
                let kind = if rows == RowRange::WHOLE {
                    CallKind::Unload
                } else {
                    CallKind::UnloadRange
                };
                Ok(self.outcome(LaneOutputs::Host(vec![out]), false, kind).made(stats.calls))
            }
            LaneCall::Accumulate => {
                let batch = self.expect_input(input, site)?;
                let LaneState::BatchAcc(executor) = &mut self.state else {
                    return Err(self.wrong_state(site, "accumulate_and_fetch"));
                };
                let modelled = acct.begin_call(site.slot, executor, n_rows, n_bytes)?;
                let (out, stats) = executor
                    .accumulate_and_fetch(batch.batch)
                    .map_err(|e| failed(site, acct, Some(n_bytes), e))?;
                let out = hold_all(out, acct, Some(batch.bytes))?;
                acct.end_call(site.slot, executor, &stats, modelled)?;
                Ok(self
                    .outcome(LaneOutputs::Device(out), false, CallKind::Accumulate)
                    .made(stats.calls))
            }
            LaneCall::MarkDone => {
                let LaneState::BatchAcc(executor) =
                    mem::replace(&mut self.state, LaneState::Finished)
                else {
                    return Err(self.wrong_state(site, "mark_done_and_fetch"));
                };
                let modelled = acct.begin_call(site.slot, &executor, 0, 0)?;
                let (out, stats) = executor
                    .mark_done_and_fetch()
                    .map_err(|e| failed(site, acct, None, e))?;
                let out = hold_all(out, acct, None)?;
                acct.end_consuming_call(site.slot, &stats, modelled)?;
                Ok(self
                    .outcome(LaneOutputs::Device(out), true, CallKind::MarkDone)
                    .made(stats.calls))
            }
            LaneCall::SetBuild => {
                let batch = self.expect_input(input, site)?;
                let LaneState::Build(executor) = mem::replace(&mut self.state, LaneState::Unbuilt)
                else {
                    return Err(self.wrong_state(site, "set_build"));
                };
                let modelled = acct.begin_call(site.slot, &executor, n_rows, n_bytes)?;
                let (probing, stats) = executor
                    .set_build(batch.batch)
                    .map_err(|e| failed(site, acct, Some(n_bytes), e))?;
                acct.release(batch.bytes)?;
                // The successor reports for the same slot: what the build side became is
                // this instance's residency now.
                acct.end_call(site.slot, &probing, &stats, modelled)?;
                self.state = LaneState::Probe(probing);
                Ok(self
                    .outcome(LaneOutputs::Device(Vec::new()), false, CallKind::SetBuild)
                    .made(stats.calls))
            }
            LaneCall::NoBuild => {
                let LaneState::Build(executor) = mem::replace(&mut self.state, LaneState::Draining)
                else {
                    return Err(self.wrong_state(site, "without_build"));
                };
                executor
                    .without_build()
                    .map_err(|e| failed(site, acct, None, e))?;
                acct.forget(site.slot);
                // Not finished: the probe side is still producing for this lane, and what
                // it produces has to be read to be let go of.
                Ok(self.outcome(LaneOutputs::Device(Vec::new()), false, CallKind::NoBuild))
            }
            LaneCall::DropProbe => {
                let batch = self.expect_input(input, site)?;
                acct.release(batch.bytes)?;
                Ok(self.outcome(
                    LaneOutputs::Device(Vec::new()),
                    false,
                    CallKind::ReleaseUnwanted,
                ))
            }
            LaneCall::Probe => {
                let batch = self.expect_input(input, site)?;
                let LaneState::Probe(executor) = &mut self.state else {
                    return Err(self.wrong_state(site, "probe_and_fetch"));
                };
                let modelled = acct.begin_call(site.slot, executor, n_rows, n_bytes)?;
                let (out, stats) = executor
                    .probe_and_fetch(batch.batch)
                    .map_err(|e| failed(site, acct, Some(n_bytes), e))?;
                let out = hold_all(out, acct, Some(batch.bytes))?;
                acct.end_call(site.slot, executor, &stats, modelled)?;
                Ok(self.outcome(LaneOutputs::Device(out), false, CallKind::Probe).made(stats.calls))
            }
            LaneCall::Finish => {
                let LaneState::Probe(executor) = mem::replace(&mut self.state, LaneState::Finished)
                else {
                    return Err(self.wrong_state(site, "finish_and_fetch"));
                };
                let modelled = acct.begin_call(site.slot, &executor, 0, 0)?;
                let (out, stats) = executor
                    .finish_and_fetch()
                    .map_err(|e| failed(site, acct, None, e))?;
                let out = hold_all(out, acct, None)?;
                acct.end_consuming_call(site.slot, &stats, modelled)?;
                Ok(self.outcome(LaneOutputs::Device(out), true, CallKind::Finish).made(stats.calls))
            }
        }
    }

    fn source_step(
        &mut self,
        site: &LaneSite<'_, B>,
        acct: &mut ResidentAccountant,
    ) -> Result<LaneOutcome<B>, StepError> {
        let LaneState::Source(source) = mem::replace(&mut self.state, LaneState::Finished) else {
            return Err(self.wrong_state(site, "next_batch"));
        };
        let modelled = acct.begin_call(site.slot, &source, 0, 0)?;
        match source
            .next_batch()
            .map_err(|e| failed(site, acct, None, e))?
        {
            SourceStep::Batch {
                batch,
                stats,
                source,
            } => {
                let out = Held::of(batch);
                acct.hold(out.bytes);
                acct.end_call(site.slot, &source, &stats, modelled)?;
                self.state = LaneState::Source(source);
                Ok(self
                    .outcome(LaneOutputs::Device(vec![out]), false, CallKind::NextBatch)
                    .made(stats.calls))
            }
            // Exhaustion consumed the source, so the slot's liveness is the state itself.
            SourceStep::Exhausted => {
                acct.end_consuming_call(site.slot, &CallStats::default(), modelled)?;
                Ok(self.outcome(
                    LaneOutputs::Device(Vec::new()),
                    true,
                    CallKind::SourceExhausted,
                ))
            }
        }
    }

    fn build(&mut self, site: &LaneSite<'_, B>) -> Result<(), RunError> {
        let executors = B::executors_for(site.ctx, site.node, site.post_order, site.lane)
            .map_err(RunError::Backend)?;
        if executors.category() != site.category {
            return Err(RunError::Backend(PlanError::Invalid(format!(
                "{}: the backend built a {:?} executor for a {:?} node",
                site.node.name(),
                executors.category(),
                site.category
            ))));
        }
        self.state = match executors {
            NodeExecutors::Source(source) => LaneState::Source(source),
            NodeExecutors::Exec(exec) => LaneState::Exec(exec),
            NodeExecutors::BatchAccumulator(acc) => LaneState::BatchAcc(acc),
            NodeExecutors::Join(join) => LaneState::Build(join),
            NodeExecutors::Unload(unload) => LaneState::Unload(unload),
            // The category check above passed, so these cannot be reached from a
            // lane-scoped node; a backend that returns one is caught there.
            NodeExecutors::PartitionAccumulator(_)
            | NodeExecutors::PartitionEmitter(_)
            | NodeExecutors::BatchForwarder(_) => {
                return Err(RunError::Backend(PlanError::Invalid(format!(
                    "{}: a cross-lane executor for a lane-scoped node",
                    site.node.name()
                ))));
            }
        };
        Ok(())
    }

    fn expect_input(
        &self,
        input: Option<Held<B::Batch>>,
        site: &LaneSite<'_, B>,
    ) -> Result<Held<B::Batch>, StepError> {
        input.ok_or_else(|| {
            RunError::Protocol(format!(
                "{}: the call needs an input batch and was handed none",
                site.node.name()
            ))
            .into()
        })
    }

    fn wrong_state(&self, site: &LaneSite<'_, B>, call: &str) -> StepError {
        RunError::Protocol(format!(
            "{}: {call} on a lane holding another kind of executor",
            site.node.name()
        ))
        .into()
    }

    fn outcome(&self, outputs: LaneOutputs<B>, finished: bool, call: CallKind) -> LaneOutcome<B> {
        LaneOutcome {
            outputs,
            finished,
            call,
            calls: None,
        }
    }
}

/// The query is over, and the site is what a message can name that a backend cannot. The
/// input went into the call that failed, so its bytes leave the accounted set here rather
/// than after the call that would have released them — the totals still reconcile, and the
/// release error is dropped because a failure is already on its way out.
fn failed<B: Backend>(
    site: &LaneSite<'_, B>,
    acct: &mut ResidentAccountant,
    consumed: Option<usize>,
    error: BackendError,
) -> StepError {
    if let Some(bytes) = consumed {
        let _ = acct.release(bytes);
    }
    StepError::Run(RunError::CallFailed(format!(
        "{} lane {}: {error}",
        site.node.name(),
        site.lane
    )))
}

/// Account a call's outputs and release its input, in the order the spec states: the input
/// is alive while the call runs, so it leaves the resident set after it. `consumed` is
/// `None` for the calls that take no input — zero bytes would count a release that never
/// happened, which is what the hold and release counts exist to notice.
fn hold_all<T: crate::batch_partitioned::batch::Batch>(
    outputs: Vec<T>,
    acct: &mut ResidentAccountant,
    consumed: Option<usize>,
) -> Result<Vec<Held<T>>, RunError> {
    if let Some(bytes) = consumed {
        acct.release(bytes)?;
    }
    Ok(outputs
        .into_iter()
        .map(|batch| {
            let held = Held::of(batch);
            acct.hold(held.bytes);
            held
        })
        .collect())
}

#[cfg(test)]
mod tests;
