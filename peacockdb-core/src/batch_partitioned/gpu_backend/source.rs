//! `GpuLoadParquet` on a device: one `execute_scan_rowgroups` per batch, over the row
//! groups the mapping gave this lane.
//!
//! The additive symbol is what makes the node a source at all — the frozen `execute_node`
//! reads every row group the scan node carries, in one call — so the row groups a batch
//! reads are an argument here rather than a field of the node the seq addresses.

use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef};

use peacockdb_ffi::raw::{
    PeacockExecutor, PeacockNodeStats, peacock_executor_execute_scan_rowgroups,
};

use super::super::error::PlanError;
use super::super::executor::{BackendError, CallStats};
use super::super::gpu_batch::GpuBatch;
use super::super::nodes::GpuLoadParquet;
use super::super::recipe::{AbiSymbol, CallPattern, Input, Recipe, Seq};
use super::{last_error, produced};

/// A lane's reads, in the order the mapping named them.
pub struct GpuSource {
    executor: *mut PeacockExecutor,
    seq: Seq,
    /// The row groups per batch this lane still owes, front first.
    batches: std::collections::VecDeque<Vec<u32>>,
    schema: SchemaRef,
}

impl GpuSource {
    pub fn new(
        executor: *mut PeacockExecutor,
        recipe: &Recipe,
        node: &GpuLoadParquet,
        lane: usize,
        schema: &ArrowSchema,
    ) -> Result<Self, PlanError> {
        let [call] = recipe.calls.as_slice() else {
            return Err(PlanError::Invalid(format!(
                "a source makes one call per batch, and this recipe is `{recipe}`"
            )));
        };
        if call.symbol != AbiSymbol::ExecuteScanRowGroups
            || call.when != CallPattern::PerBatch
            || call.inputs.as_slice() != [Input::RowGroups]
        {
            return Err(PlanError::Invalid(format!(
                "a source reads the row groups of one batch per call, and this one is {call:?}"
            )));
        }
        let (seq, _) = call
            .target
            .ok_or_else(|| PlanError::Invalid(format!("{call:?} addresses no seq")))?;
        let batches = node.partition_groups.get(lane).ok_or_else(|| {
            PlanError::Invalid(format!(
                "lane {lane} of a scan the partitioner mapped into {} lanes",
                node.partition_groups.len()
            ))
        })?;
        Ok(Self {
            executor,
            seq,
            batches: batches.iter().cloned().collect(),
            schema: Arc::new(schema.clone()),
        })
    }

    /// The next batch, or `None` where the mapping gave this lane nothing more.
    pub fn read_next(&mut self) -> Result<Option<(GpuBatch, CallStats)>, BackendError> {
        let Some(groups) = self.batches.pop_front() else {
            return Ok(None);
        };
        let mut handle = 0u64;
        let mut stats = PeacockNodeStats::default();
        let rc = unsafe {
            peacock_executor_execute_scan_rowgroups(
                self.executor,
                self.seq as u64,
                groups.as_ptr(),
                groups.len() as u64,
                &mut handle,
                &mut stats,
            )
        };
        if rc != 0 {
            return Err(BackendError::new(format!(
                "execute_scan_rowgroups(#{}, {groups:?}): {}",
                self.seq,
                last_error(self.executor)
            )));
        }
        Ok(Some((
            produced(self.executor, handle, stats, &self.schema),
            CallStats::default(),
        )))
    }
}
