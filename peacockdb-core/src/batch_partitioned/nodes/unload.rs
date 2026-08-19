//! `GpuUnload`: the boundary crossing, and the only node whose output is a CPU batch.

use std::any::Any;

use super::super::error::PlanError;
use super::super::layout::NodeKind;
use super::super::node::{GpuNode, RowInterval};

/// Carries a root-adjacent limit's `skip`/`fetch`, because the interval belongs to the
/// crossing: it is a statement about which rows are worth moving over PCIe, and trimming
/// after the transfer ships an unbounded prefix to drop it.
#[derive(Debug)]
pub struct GpuUnload {
    kind: NodeKind,
    pub interval: Option<RowInterval>,
    input: Box<dyn GpuNode>,
}

impl GpuUnload {
    pub fn new(input: Box<dyn GpuNode>, interval: Option<RowInterval>) -> Self {
        Self {
            kind: NodeKind::Sink,
            interval,
            input,
        }
    }
}

impl GpuNode for GpuUnload {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        vec![self.input.as_ref()]
    }

    /// Nothing: an unload takes any lane count and any batch layout, and its interval is
    /// counted across lanes by the driver rather than requiring a merge below it.
    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        Ok(())
    }

    fn row_interval(&self) -> Option<RowInterval> {
        self.interval
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
