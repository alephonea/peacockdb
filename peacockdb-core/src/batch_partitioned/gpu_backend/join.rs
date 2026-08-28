//! The joins on a device: the calls the recipe names, and the one it names that the frozen
//! surface cannot make.
//!
//! `execute_node` erases every input handle it reads, and there is no copy symbol —
//! `slice_handle` consumes its input too, so it moves rather than copies. Two recipes ask
//! for one anyway, and both are refused by name rather than run against a dead handle: a
//! streamed probe's second batch, whose build side the first call erased, and a Left or
//! Full join's probe batch, which its key project and its join both read. #152 records
//! them and #145 retires both.

use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef};
use datafusion::common::JoinType;

use peacockdb_ffi::raw::PeacockExecutor;

use super::super::batch::Batch;
use super::super::error::PlanError;
use super::super::executor::{BackendError, CallResult, CallStats};
use super::super::gpu_batch::GpuBatch;
use super::super::nodes::join::empty_build_answers_nothing;
use super::super::recipe::{CallPattern, FbKind, Input, ProjectRole, Recipe, Seq};
use super::{execute_node, produced};

/// One call of a join's recipe: the seq, and the inputs it names in order. Each named
/// input is one child slot — the C++ reads its output count off the first slot, so two
/// handles in one would ask for two outputs and be refused for a buffer it never needed.
#[derive(Clone)]
struct JoinCall {
    seq: Seq,
    kind: FbKind,
    inputs: Vec<Input>,
}

/// A join before its build side arrives.
pub struct GpuJoin {
    executor: *mut PeacockExecutor,
    /// The node's own type, and `None` for the two joins that have none — cross and
    /// nested-loop. What a finish over no keys owes is decided by this rather than read
    /// back off the call list: two different nodes publish a LeftAnti at done, and one of
    /// them owes padded rows.
    join_type: Option<JoinType>,
    per_probe: Vec<JoinCall>,
    at_done: Vec<JoinCall>,
    keys_schema: Option<SchemaRef>,
    output: SchemaRef,
}

impl GpuJoin {
    /// `keys` is the schema of the probe keys a finishing join accumulates — the key
    /// project's output, which is the node's key columns and nothing else.
    pub fn new(
        executor: *mut PeacockExecutor,
        recipe: &Recipe,
        join_type: Option<JoinType>,
        keys: Option<&ArrowSchema>,
        output: &ArrowSchema,
    ) -> Result<Self, PlanError> {
        let mut per_probe = Vec::new();
        let mut at_done = Vec::new();
        for call in &recipe.calls {
            let (seq, kind) = call.target.ok_or_else(|| {
                PlanError::Invalid(format!("a join's call takes a seq, and {call:?} does not"))
            })?;
            if call.inputs.is_empty() {
                return Err(PlanError::Invalid(format!(
                    "{call:?} names no input to read"
                )));
            }
            let placed = JoinCall {
                seq,
                kind,
                inputs: call.inputs.clone(),
            };
            match call.when {
                CallPattern::PerProbeBatch => per_probe.push(placed),
                CallPattern::AtDone => at_done.push(placed),
                other => {
                    return Err(PlanError::Invalid(format!(
                        "a join calls per probe batch and at done, and {call:?} is {other:?}"
                    )));
                }
            }
        }
        if per_probe.is_empty() {
            return Err(PlanError::Invalid(
                "a join makes at least one call per probe batch".to_string(),
            ));
        }
        Ok(Self {
            executor,
            join_type,
            per_probe,
            at_done,
            keys_schema: keys.map(|schema| Arc::new(schema.clone()) as SchemaRef),
            output: Arc::new(output.clone()),
        })
    }

    /// This lane's build side finished with no batch — its scatter gave it no build rows.
    /// The type decides what it owes, and the rule is the one the CPU reads too.
    pub fn without_build(self) -> Result<(), BackendError> {
        // Cross and nested-loop joins carry no type here and owe nothing either: every row
        // they emit is built from a build row, the Left form's padding included.
        let owes_nothing = self.join_type.map_or(true, empty_build_answers_nothing);
        if owes_nothing {
            return Ok(());
        }
        Err(BackendError::new(
            "this lane's build side is empty, and what this join owes is its probe side — \
             which takes a call over a build table that does not exist (#175)",
        ))
    }

    /// The build side, which is one batch per lane. It is held rather than consumed: which
    /// call takes it, and whether it survives that call, is what the recipe says.
    pub fn set_build(self, batch: GpuBatch) -> CallResult<GpuProbingJoin> {
        Ok((
            GpuProbingJoin {
                join: self,
                build: Some(batch),
                accumulated: Vec::new(),
                probes: 0,
            },
            CallStats::default(),
        ))
    }
}

/// A join with its build side set, taking probe batches.
pub struct GpuProbingJoin {
    join: GpuJoin,
    /// `None` once a call has consumed it, which is the last probe call for a join that
    /// hands it over and the finish call for one that does not.
    build: Option<GpuBatch>,
    /// The probe keys each batch contributed, held until the finish concatenates them.
    accumulated: Vec<GpuBatch>,
    probes: u64,
}

impl GpuProbingJoin {
    /// The build side, from `set_build` until the call that consumes it — `None` after,
    /// because the surface has no copy and the recipe says which call takes it.
    pub fn build_bytes(&self) -> usize {
        self.build.as_ref().map_or(0, GpuBatch::byte_size)
    }

    /// The probe keys a finishing type keeps until its finish pass runs (#136).
    pub fn accumulated_bytes(&self) -> usize {
        self.accumulated.iter().map(GpuBatch::byte_size).sum()
    }

    /// Whether a probe call reads the build side at all. False for the build-side semi
    /// family, whose probe call is the key project alone — and the accounting has to know,
    /// because a transient charged for a read that never happens refuses work that fits.
    pub fn probe_reads_build(&self) -> bool {
        self.join
            .per_probe
            .iter()
            .any(|call| call.inputs.iter().any(Input::is_build_side))
    }

    pub fn probe_and_fetch(&mut self, batch: GpuBatch) -> CallResult<Vec<GpuBatch>> {
        self.probes += 1;
        let mut out = Vec::new();
        let mut batch = Some(batch);
        let mut prior: Option<GpuBatch> = None;
        for call in self.join.per_probe.clone() {
            let kind = call.kind;
            let produced = self.make(call, &mut batch, &mut prior)?;
            match kind {
                // The key project's output is the lane's growing key table, not an answer.
                FbKind::Project(ProjectRole::ProbeKeys) => self.accumulated.push(produced),
                _ => prior = Some(produced),
            }
        }
        out.extend(prior);
        Ok((out, CallStats::default()))
    }

    /// The question a streamed probe could not answer, in the calls the recipe names: the
    /// keys concatenated, the finish join against the build side, and the pad where the
    /// node's output is the joined schema.
    ///
    /// A lane whose probe was empty takes the same three calls rather than a case of its own:
    /// the concat of no keys answers with an empty table of the key schema (#173), and the
    /// finish against it produces exactly what the cpu backend produces by concatenating no
    /// batches against the same schema. Which is why there is no switch on `join_type` here —
    /// what a semi, a mark or an outer owes an empty probe is what its own finish computes.
    pub fn finish_and_fetch(mut self) -> CallResult<Vec<GpuBatch>> {
        if self.join.at_done.is_empty() {
            return Ok((Vec::new(), CallStats::default()));
        }
        let mut prior: Option<GpuBatch> = None;
        let mut none = None;
        for call in self.join.at_done.clone() {
            prior = Some(self.make(call, &mut none, &mut prior)?);
        }
        Ok((prior.into_iter().collect(), CallStats::default()))
    }

    /// One call, its named inputs resolved to the handles this join is holding.
    fn make(
        &mut self,
        call: JoinCall,
        batch: &mut Option<GpuBatch>,
        prior: &mut Option<GpuBatch>,
    ) -> Result<GpuBatch, BackendError> {
        let mut slots: Vec<Vec<u64>> = Vec::with_capacity(call.inputs.len());
        for input in &call.inputs {
            slots.push(match input {
                Input::Batch => vec![self.take(batch.take(), "the probe batch")?],
                Input::BatchCopy => vec![self.copy_of(batch)?],
                Input::BuildSide => {
                    let build = self.build.take();
                    vec![self.take(build, "the build side")?]
                }
                Input::BuildSideCopy => {
                    let build = self.build.take();
                    vec![self.build_copy(build)?]
                }
                Input::AccumulatedKeys => hand_over(std::mem::take(&mut self.accumulated)),
                Input::PriorOutput => vec![self.take(prior.take(), "the call before it")?],
                other => {
                    return Err(BackendError::new(format!(
                        "a join's call reads {other:?}, which no join recipe names"
                    )));
                }
            });
        }
        let (handle, stats) = execute_node(self.join.executor, call.seq, call.kind, &slots)?;
        let schema = self.schema_of(call.kind);
        Ok(produced(self.join.executor, handle, stats, schema))
    }

    /// What the call produced is priced by: the key project's output is the keys, and
    /// everything else is the node's own row.
    fn schema_of(&self, kind: FbKind) -> &SchemaRef {
        match (kind, &self.join.keys_schema) {
            (FbKind::Project(ProjectRole::ProbeKeys), Some(keys)) => keys,
            _ => &self.join.output,
        }
    }

    fn take(&self, batch: Option<GpuBatch>, what: &str) -> Result<u64, BackendError> {
        batch
            .map(|batch| batch.consume().1)
            .ok_or_else(|| BackendError::new(format!("{what} was already consumed")))
    }

    /// A copy of the probe batch, which the ABI cannot make either. A recipe names one
    /// only where a later call in the same probe still needs that batch — the key project
    /// of a Left or Full join, with the per-call join reading the batch after it — so
    /// there is no arrangement of these two calls that leaves both a batch to read.
    fn copy_of(&self, _batch: &mut Option<GpuBatch>) -> Result<u64, BackendError> {
        Err(BackendError::new(
            "this join's recipe copies its probe batch — the key project keeps the keys \
             and the join below it reads the same batch — and the ABI has no copy, so \
             neither call can run without erasing the other's input (#152)",
        ))
    }

    /// The build side, where the recipe asked for a copy of it. The first probe batch can
    /// have the original; a second has nothing to be given, because the call that read it
    /// erased it.
    fn build_copy(&self, build: Option<GpuBatch>) -> Result<u64, BackendError> {
        match build {
            Some(build) => Ok(build.consume().1),
            None => Err(BackendError::new(format!(
                "this join's recipe copies its build side per probe batch and the ABI has \
                 no copy: probe batch {} has no build side left, since the call for batch \
                 1 erased it (#152)",
                self.probes
            ))),
        }
    }
}

fn hand_over(batches: Vec<GpuBatch>) -> Vec<u64> {
    batches.into_iter().map(|batch| batch.consume().1).collect()
}
