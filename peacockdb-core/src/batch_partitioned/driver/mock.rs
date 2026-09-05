//! A third `Backend` instantiation, alongside the CPU and GPU ones: scripted counts,
//! sizes, skew patterns and accumulator behaviour, and no operator that computes anything.
//!
//! What the driver tests assert is calls — pull counts, queue bounds, release — so an
//! operator here only has to produce the right number of batches of the right size at the
//! right moment. Sources are scripted per table name and per lane; every other category
//! takes one rule for the whole plan, which is what keeps a test's setup readable.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

use crate::batch_partitioned::backend::{Backend, NodeExecutors};
use crate::batch_partitioned::batch::Batch;
use crate::batch_partitioned::cpu_batch::CpuBatch;
use crate::batch_partitioned::error::PlanError;
use crate::batch_partitioned::executor::{
    AbiCalls, BackendError, BatchAccumulatorExecutor, CallResult, CallStats, ExecExecutor,
    Executor, JoinExecutor, LaneEvent, PartitionAccumulatorExecutor, PartitionEmitterExecutor,
    ProbingJoin, RowRange, SourceExecutor, SourceStep, UnloadExecutor,
};
use crate::batch_partitioned::forwarder::forwarder_for;
use crate::batch_partitioned::node::GpuNode;
use crate::batch_partitioned::nodes::{ExecutorCategory, NodeRef, as_node_ref, category_of};

/// Rows and bytes, which is all any assertion here reads off a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Spec {
    pub rows: usize,
    pub bytes: usize,
}

pub(super) fn spec(rows: usize, bytes: usize) -> Spec {
    Spec { rows, bytes }
}

#[derive(Debug)]
pub(super) struct MockBatch {
    pub rows: usize,
    pub bytes: usize,
}

impl Batch for MockBatch {
    fn num_rows(&self) -> usize {
        self.rows
    }
    fn byte_size(&self) -> usize {
        self.bytes
    }
}

/// What each category does. One rule per category for the whole plan: a test that needs
/// two different emitters is a test about two plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExecRule {
    /// One batch out per batch in, unchanged.
    Identity,
    /// One batch out per batch in, holding no rows — a filter that passed nothing.
    Empty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AccRule {
    /// Holds every batch and emits one at done, its residency growing as it holds.
    CoalesceAll,
    /// Emits each batch as it arrives and holds nothing — what a mid-plan limit does.
    Streaming,
    /// Emits this many batches at done, holding nothing. Zero and two are what a join's
    /// build side must never do.
    EmitAtDone(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EmitRule {
    /// Rows spread evenly over the lanes.
    RoundRobin,
    /// One output short of the lane count. N is a plan value, so the driver checks it.
    WrongCount,
    /// Every row into one lane, the rest emitted empty — hash skew at its worst.
    ToLane(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct JoinRule {
    /// Whether a lane with no build batch owes its probe side rather than nothing — the
    /// three join types that preserve unmatched probe rows, which no executor can answer
    /// without a build table.
    pub empty_build_owes_its_probe: bool,
    /// Rows the finish pass emits. Zero is a join that needs no finish.
    pub finish_rows: usize,
    /// Bytes the build side keeps resident until the join is done.
    pub build_residency: usize,
}

/// Which call the backend fails at. One per script: a query stops at the first failure, so
/// a second one is unreachable by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FailAt {
    SourceStep,
    Exec,
    Emit,
    Probe,
    Finish,
    MarkDone,
}

#[derive(Debug, Clone)]
pub(super) struct Script {
    /// Per table, per lane, the batches that lane's loader emits.
    pub sources: Vec<(String, Vec<Vec<Spec>>)>,
    pub exec: ExecRule,
    pub accumulate: AccRule,
    pub emit: EmitRule,
    pub join: JoinRule,
    /// What every call reports as measured scratch. `None` is an uninstrumented run.
    pub measured_scratch: Option<usize>,
    /// What every executor models per input byte, before its own state.
    pub scratch_per_byte: usize,
    /// Whether an accumulator prices the output it is about to build. A knob rather than
    /// the default: a mock that always prices honestly cannot express a model that stays
    /// silent, and a silent model is what lets a peak cross a budget uncaught.
    pub accumulator_prices_emission: bool,
    /// Where the backend fails, if it does.
    pub fails_at: Option<FailAt>,
    /// How many executor sets have been built. A lane builds on its first step, not
    /// before, and an eager build is invisible without a count.
    pub built: Arc<AtomicUsize>,
    /// Every build, as the node's address and the post-order it was handed. The count
    /// above says a set was built; this says which node the driver thought it was
    /// building for, which is the only place that number is observable at all. An address
    /// rather than a name, because a plan holds several nodes of a kind and a name would
    /// pair a build with the wrong one of them.
    pub built_at: Arc<Mutex<Vec<(usize, usize)>>>,
    /// Build an exec executor for every node, whatever its category — a backend wired to
    /// the wrong trait, which the driver has to catch where the set was built.
    pub miswired: bool,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            sources: Vec::new(),
            exec: ExecRule::Identity,
            accumulate: AccRule::CoalesceAll,
            emit: EmitRule::RoundRobin,
            join: JoinRule {
                empty_build_owes_its_probe: false,
                finish_rows: 0,
                build_residency: 0,
            },
            measured_scratch: None,
            scratch_per_byte: 0,
            accumulator_prices_emission: false,
            fails_at: None,
            miswired: false,
            built: Arc::new(AtomicUsize::new(0)),
            built_at: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Script {
    pub(super) fn source(mut self, table: &str, lanes: Vec<Vec<Spec>>) -> Self {
        self.sources.push((table.to_string(), lanes));
        self
    }

    pub(super) fn with_exec(mut self, exec: ExecRule) -> Self {
        self.exec = exec;
        self
    }

    /// The rule for accumulators that are not limits. A `GpuLimit` streams whatever is set
    /// here, because that is what a limit is — see `executors_for`.
    pub(super) fn with_accumulator(mut self, accumulate: AccRule) -> Self {
        self.accumulate = accumulate;
        self
    }

    pub(super) fn with_emit(mut self, emit: EmitRule) -> Self {
        self.emit = emit;
        self
    }

    pub(super) fn with_join(mut self, join: JoinRule) -> Self {
        self.join = join;
        self
    }

    /// The accumulator includes what it holds in its pre-call model, which is what the
    /// spec permits by letting `scratch_bytes` consult `&self`.
    pub(super) fn pricing_the_emission(mut self) -> Self {
        self.accumulator_prices_emission = true;
        self
    }

    pub(super) fn failing_at(mut self, at: FailAt) -> Self {
        self.fails_at = Some(at);
        self
    }

    pub(super) fn miswired(mut self) -> Self {
        self.miswired = true;
        self
    }

    pub(super) fn measuring(mut self, scratch: usize) -> Self {
        self.measured_scratch = Some(scratch);
        self
    }

    pub(super) fn modelling(mut self, per_byte: usize) -> Self {
        self.scratch_per_byte = per_byte;
        self
    }

    fn batches_for(&self, table: &str, lane: usize) -> Vec<Spec> {
        self.sources
            .iter()
            .find(|(name, _)| name == table)
            .and_then(|(_, lanes)| lanes.get(lane).cloned())
            .unwrap_or_default()
    }

    /// The failure a real backend reports: a message, and nothing to resume from.
    fn check(&self, at: FailAt) -> Result<(), BackendError> {
        match self.fails_at {
            Some(scripted) if scripted == at => Err(BackendError::new(format!(
                "the device gave up during {at:?}"
            ))),
            _ => Ok(()),
        }
    }

    fn stats(&self) -> CallStats {
        CallStats {
            scratch_bytes: self.measured_scratch,
            // A mock addresses no seq: what it drives is a script, not a recipe.
            calls: AbiCalls::default(),
        }
    }
}

pub(super) struct Mock;

pub(super) struct MockSource {
    remaining: Vec<Spec>,
    script: Script,
}

pub(super) struct MockExec {
    script: Script,
}

pub(super) struct MockAcc {
    script: Script,
    held_rows: usize,
    held_bytes: usize,
}

pub(super) struct MockPartAcc {
    script: Script,
    held_rows: usize,
    held_bytes: usize,
    lanes_done: usize,
    lanes: usize,
}

pub(super) struct MockEmitter {
    script: Script,
    lanes: usize,
}

pub(super) struct MockJoin {
    script: Script,
}

pub(super) struct MockProbing {
    script: Script,
    build_rows: usize,
}

pub(super) struct MockUnload {
    script: Script,
}

/// Every executor models a share of the batch it is about to be handed, and nothing else.
/// A real one may add its own state — the accountant's unit tests cover that — but keeping
/// state out here is what lets a test trip the post-check without the pre-check firing
/// first on the same figure.
fn model(script: &Script, _resident: usize, n_bytes: usize) -> usize {
    n_bytes * script.scratch_per_byte
}

impl Executor for MockSource {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        model(&self.script, 0, n_bytes)
    }
}

impl Executor for MockExec {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        model(&self.script, 0, n_bytes)
    }
}

impl Executor for MockAcc {
    fn resident_bytes(&self) -> usize {
        self.held_bytes
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        // What it holds is what the emitting call has to build a copy of, so a model that
        // says so refuses the call rather than letting the two live at once uncounted.
        let pending = if self.script.accumulator_prices_emission {
            self.held_bytes
        } else {
            0
        };
        pending + model(&self.script, self.held_bytes, n_bytes)
    }
}

impl Executor for MockPartAcc {
    fn resident_bytes(&self) -> usize {
        self.held_bytes
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        model(&self.script, self.held_bytes, n_bytes)
    }
}

impl Executor for MockEmitter {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        model(&self.script, 0, n_bytes)
    }
}

impl Executor for MockJoin {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        model(&self.script, 0, n_bytes)
    }
}

impl Executor for MockProbing {
    fn resident_bytes(&self) -> usize {
        self.script.join.build_residency
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        model(&self.script, self.script.join.build_residency, n_bytes)
    }
}

impl Executor for MockUnload {
    fn resident_bytes(&self) -> usize {
        0
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        model(&self.script, 0, n_bytes)
    }
}

impl SourceExecutor<Mock> for MockSource {
    fn next_batch(mut self) -> Result<SourceStep<Mock>, BackendError> {
        self.script.check(FailAt::SourceStep)?;
        if self.remaining.is_empty() {
            return Ok(SourceStep::Exhausted);
        }
        let spec = self.remaining.remove(0);
        let stats = self.script.stats();
        Ok(SourceStep::Batch {
            batch: MockBatch {
                rows: spec.rows,
                bytes: spec.bytes,
            },
            stats,
            source: self,
        })
    }
}

impl ExecExecutor<Mock> for MockExec {
    fn exec(&mut self, batch: MockBatch) -> CallResult<MockBatch> {
        self.script.check(FailAt::Exec)?;
        let out = match self.script.exec {
            ExecRule::Identity => batch,
            // Zero rows is not zero bytes: an empty lane still owes a typed batch.
            ExecRule::Empty => MockBatch { rows: 0, bytes: 8 },
        };
        Ok((out, self.script.stats()))
    }
}

impl BatchAccumulatorExecutor<Mock> for MockAcc {
    fn accumulate_and_fetch(&mut self, batch: MockBatch) -> CallResult<Vec<MockBatch>> {
        let stats = self.script.stats();
        Ok(match self.script.accumulate {
            AccRule::Streaming => (vec![batch], stats),
            _ => {
                self.held_rows += batch.rows;
                self.held_bytes += batch.bytes;
                (Vec::new(), stats)
            }
        })
    }

    fn mark_done_and_fetch(self) -> CallResult<Vec<MockBatch>> {
        self.script.check(FailAt::MarkDone)?;
        let stats = self.script.stats();
        let out = match self.script.accumulate {
            AccRule::Streaming => Vec::new(),
            AccRule::CoalesceAll => vec![MockBatch {
                rows: self.held_rows,
                bytes: self.held_bytes.max(8),
            }],
            AccRule::EmitAtDone(n) => (0..n)
                .map(|_| MockBatch {
                    rows: self.held_rows,
                    bytes: self.held_bytes.max(8),
                })
                .collect(),
        };
        Ok((out, stats))
    }
}

impl PartitionAccumulatorExecutor<Mock> for MockPartAcc {
    fn accumulate_and_fetch(
        &mut self,
        _partition: usize,
        event: LaneEvent<MockBatch>,
    ) -> CallResult<Vec<MockBatch>> {
        let stats = self.script.stats();
        Ok(match event {
            LaneEvent::Batch(batch) => {
                self.held_rows += batch.rows;
                self.held_bytes += batch.bytes;
                (Vec::new(), stats)
            }
            LaneEvent::Done => {
                self.lanes_done += 1;
                if self.lanes_done < self.lanes {
                    return Ok((Vec::new(), stats));
                }
                let out = vec![MockBatch {
                    rows: self.held_rows,
                    bytes: self.held_bytes.max(8),
                }];
                self.held_rows = 0;
                self.held_bytes = 0;
                (out, stats)
            }
        })
    }
}

impl PartitionEmitterExecutor<Mock> for MockEmitter {
    fn emit(&mut self, batch: MockBatch) -> CallResult<Vec<MockBatch>> {
        self.script.check(FailAt::Emit)?;
        let stats = self.script.stats();
        let share = |rows: usize, bytes: usize| MockBatch { rows, bytes };
        let out = match self.script.emit {
            EmitRule::RoundRobin => (0..self.lanes)
                .map(|lane| {
                    let rows =
                        batch.rows / self.lanes + usize::from(lane < batch.rows % self.lanes);
                    share(
                        rows,
                        if rows == 0 {
                            8
                        } else {
                            batch.bytes / self.lanes.max(1)
                        },
                    )
                })
                .collect(),
            EmitRule::WrongCount => (0..self.lanes.saturating_sub(1))
                .map(|_| share(batch.rows, batch.bytes))
                .collect(),
            EmitRule::ToLane(hot) => (0..self.lanes)
                .map(|lane| {
                    if lane == hot {
                        share(batch.rows, batch.bytes)
                    } else {
                        share(0, 8)
                    }
                })
                .collect(),
        };
        Ok((out, stats))
    }
}

impl JoinExecutor<Mock> for MockJoin {
    type Probing = MockProbing;

    fn set_build(self, batch: MockBatch) -> CallResult<MockProbing> {
        let stats = self.script.stats();
        Ok((
            MockProbing {
                script: self.script,
                build_rows: batch.rows,
            },
            stats,
        ))
    }

    /// A script says what a join owes with no build side, since a mock join has no type.
    fn without_build(self) -> Result<(), BackendError> {
        match self.script.join.empty_build_owes_its_probe {
            false => Ok(()),
            true => Err(BackendError::new("this lane's build side is empty")),
        }
    }
}

impl ProbingJoin<Mock> for MockProbing {
    fn probe_and_fetch(&mut self, batch: MockBatch) -> CallResult<Vec<MockBatch>> {
        self.script.check(FailAt::Probe)?;
        let stats = self.script.stats();
        Ok((
            vec![MockBatch {
                rows: batch.rows.min(self.build_rows),
                bytes: batch.bytes,
            }],
            stats,
        ))
    }

    fn finish_and_fetch(self) -> CallResult<Vec<MockBatch>> {
        self.script.check(FailAt::Finish)?;
        let stats = self.script.stats();
        if self.script.join.finish_rows == 0 {
            return Ok((Vec::new(), stats));
        }
        Ok((
            vec![MockBatch {
                rows: self.script.join.finish_rows,
                bytes: self.script.join.finish_rows * 8,
            }],
            stats,
        ))
    }
}

impl UnloadExecutor<Mock> for MockUnload {
    fn unload(&mut self, batch: MockBatch, rows: RowRange) -> CallResult<CpuBatch> {
        let start = (rows.offset as usize).min(batch.rows);
        let taken = if rows.length == u64::MAX {
            batch.rows - start
        } else {
            (rows.length as usize).min(batch.rows - start)
        };
        Ok((cpu_batch(taken), self.script.stats()))
    }
}

/// A node's identity as a number, which is what a recording can hold and compare. The
/// vtable half of a trait-object pointer is not stable across casts, so only the data
/// half is taken.
pub(super) fn address_of(node: &dyn GpuNode) -> usize {
    node as *const dyn GpuNode as *const () as usize
}

pub(super) fn cpu_batch(rows: usize) -> CpuBatch {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "k",
        DataType::Int64,
        true,
    )]));
    let column = Arc::new(Int64Array::from(vec![0i64; rows]));
    CpuBatch::new(RecordBatch::try_new(schema, vec![column]).expect("a mock batch"))
}

impl Backend for Mock {
    type Context = Script;
    type Batch = MockBatch;
    type Source = MockSource;
    type Exec = MockExec;
    type BatchAcc = MockAcc;
    type PartAcc = MockPartAcc;
    type Emitter = MockEmitter;
    type Join = MockJoin;
    type Unload = MockUnload;

    fn executors_for(
        script: &Script,
        node: &dyn GpuNode,
        post_order: usize,
        lane: usize,
    ) -> Result<NodeExecutors<Self>, PlanError> {
        script.built.fetch_add(1, Ordering::Relaxed);
        script
            .built_at
            .lock()
            .expect("the recording mutex")
            .push((address_of(node), post_order));
        let script = script.clone();
        if script.miswired {
            return Ok(NodeExecutors::Exec(MockExec { script }));
        }
        Ok(match category_of(node) {
            ExecutorCategory::Source => {
                let NodeRef::LoadParquet(load) = as_node_ref(node) else {
                    return Err(PlanError::Invalid(
                        "a source that is not a load".to_string(),
                    ));
                };
                NodeExecutors::Source(MockSource {
                    remaining: script.batches_for(&load.table, lane),
                    script,
                })
            }
            ExecutorCategory::Exec => NodeExecutors::Exec(MockExec { script }),
            ExecutorCategory::BatchAccumulator => {
                // Per node, not per category: a limit streams and holds nothing whatever
                // the script says, which is what lets a test put a coalescing parent above
                // a streaming limit — the shape a script with one rule for the category
                // cannot express.
                let mut script = script;
                if matches!(as_node_ref(node), NodeRef::Limit(_)) {
                    script.accumulate = AccRule::Streaming;
                }
                NodeExecutors::BatchAccumulator(MockAcc {
                    script,
                    held_rows: 0,
                    held_bytes: 0,
                })
            }
            ExecutorCategory::PartitionAccumulator => {
                let lanes = node
                    .children()
                    .first()
                    .and_then(|child| child.kind().layout())
                    .map_or(1, |layout| layout.n);
                NodeExecutors::PartitionAccumulator(MockPartAcc {
                    script,
                    held_rows: 0,
                    held_bytes: 0,
                    lanes_done: 0,
                    lanes,
                })
            }
            ExecutorCategory::PartitionEmitter => {
                let lanes = node.kind().layout().map_or(1, |layout| layout.n);
                NodeExecutors::PartitionEmitter(MockEmitter { script, lanes })
            }
            ExecutorCategory::Join => NodeExecutors::Join(MockJoin { script }),
            ExecutorCategory::Unload => NodeExecutors::Unload(MockUnload { script }),
            ExecutorCategory::BatchForwarder => NodeExecutors::BatchForwarder(forwarder_for(node)),
        })
    }
}
