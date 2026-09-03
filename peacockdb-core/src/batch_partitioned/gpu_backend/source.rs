//! `GpuLoadParquet` on a device: one `execute_scan_rowgroups` per batch, over the row
//! groups the mapping gave this lane.
//!
//! The additive symbol is what makes the node a source at all — the frozen `execute_node`
//! reads every row group the scan node carries, in one call — so the row groups a batch
//! reads are an argument here rather than a field of the node the seq addresses.

use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef};

use peacockdb_ffi::raw::{PeacockNodeStats, peacock_executor_execute_scan_rowgroups};

use crate::executors::node_timing_on;

use super::super::error::PlanError;
use super::super::executor::{AbiCalls, BackendError, CallStats};
use super::super::gpu_batch::GpuBatch;
use super::super::nodes::GpuLoadParquet;
use super::super::recipe::{AbiSymbol, CallPattern, FbKind, Input, Recipe, Seq};
use super::{Device, last_error, produced};

/// A lane's reads, in the order the mapping named them.
pub struct GpuSource {
    dev: Device,
    seq: Seq,
    kind: FbKind,
    /// The row groups per batch this lane still owes, front first.
    batches: std::collections::VecDeque<Vec<u32>>,
    schema: SchemaRef,
}

impl GpuSource {
    pub fn new(
        dev: Device,
        recipe: &Recipe,
        node: &GpuLoadParquet,
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
        let (seq, kind) = call
            .target
            .ok_or_else(|| PlanError::Invalid(format!("{call:?} addresses no seq")))?;
        let lane = dev.lane;
        let batches = node.partition_groups.get(lane).ok_or_else(|| {
            PlanError::Invalid(format!(
                "lane {lane} of a scan the partitioner mapped into {} lanes",
                node.partition_groups.len()
            ))
        })?;
        Ok(Self {
            dev,
            seq,
            kind,
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
                self.dev.executor,
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
                last_error(self.dev.executor)
            )));
        }
        // A scan takes no batch, so its input is nothing rather than unknown — the same
        // zero the driver models it with.
        let mut calls = AbiCalls::armed(node_timing_on());
        calls.record(self.seq, self.kind, 0, Some(0));
        Ok(Some((
            produced(self.dev.executor, handle, stats, &self.schema),
            CallStats {
                scratch_bytes: None,
                calls,
            },
        )))
    }
}
