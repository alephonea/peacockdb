//! `GpuEmitPartitions` on a device: the repartition arm, one call per batch, N handles out.
//!
//! The scatter is the one call whose output count is a plan value rather than one, and the
//! count is the contract: a driver reads output `p` as lane `p`'s, so a lane the hash sent
//! nothing to still comes back as a handle to an empty table.

use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef};

use peacockdb_ffi::raw::PeacockExecutor;

use super::super::error::PlanError;
use super::super::executor::{BackendError, CallResult, CallStats};
use super::super::gpu_batch::GpuBatch;
use super::super::recipe::{CallPattern, FbKind, Input, Recipe, Seq};
use super::{execute_node_many, produced};

pub struct GpuEmitter {
    executor: *mut PeacockExecutor,
    seq: Seq,
    kind: FbKind,
    lanes: usize,
    schema: SchemaRef,
}

impl GpuEmitter {
    pub fn new(
        executor: *mut PeacockExecutor,
        recipe: &Recipe,
        schema: &ArrowSchema,
    ) -> Result<Self, PlanError> {
        let [call] = recipe.calls.as_slice() else {
            return Err(PlanError::Invalid(format!(
                "a scatter makes one call per batch, and this recipe is `{recipe}`"
            )));
        };
        if call.when != CallPattern::PerBatch || call.inputs.as_slice() != [Input::Batch] {
            return Err(PlanError::Invalid(format!(
                "a scatter calls per batch over the batch, and this one is {call:?}"
            )));
        }
        let (seq, kind) = call
            .target
            .ok_or_else(|| PlanError::Invalid(format!("{call:?} addresses no seq")))?;
        // The lane count rides the call's own kind, which is where the recipe repeats it
        // from the node it addresses — so the executor and the fb node cannot disagree.
        let FbKind::Repartition { lanes } = kind else {
            return Err(PlanError::Invalid(format!(
                "a scatter addresses a repartition, and this call addresses {kind}"
            )));
        };
        Ok(Self {
            executor,
            seq,
            kind,
            lanes: lanes as usize,
            schema: Arc::new(schema.clone()),
        })
    }

    pub fn emit(&mut self, batch: GpuBatch) -> CallResult<Vec<GpuBatch>> {
        let (_, handle) = batch.consume();
        let produced_lanes =
            execute_node_many(self.executor, self.seq, self.kind, &[vec![handle]], self.lanes)?;
        if produced_lanes.len() != self.lanes {
            return Err(BackendError::new(format!(
                "the scatter answered with {} handles where the plan declares {} lanes — a \
                 driver reads output p as lane p's, so a missing empty shifts every lane \
                 above it",
                produced_lanes.len(),
                self.lanes
            )));
        }
        Ok((
            produced_lanes
                .into_iter()
                .map(|(handle, stats)| produced(self.executor, handle, stats, &self.schema))
                .collect(),
            CallStats::default(),
        ))
    }
}
