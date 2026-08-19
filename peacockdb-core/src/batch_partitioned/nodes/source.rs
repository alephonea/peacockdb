//! `GpuLoadParquet`: the one source node.

use super::super::error::PlanError;
use super::super::layout::{BatchLayout, KeyDistribution, NodeKind, PartitionLayout, SortOrder};
use super::super::node::GpuNode;
use super::super::schema::Schema;
use std::any::Any;

/// Reads its lane's row groups out of the partitioner's mapping. The mapping is stored
/// verbatim — partitions outermost, batches within, row groups innermost — because the
/// loader executes it, the golden prints it and validation counts lanes off it.
#[derive(Debug)]
pub struct GpuLoadParquet {
    kind: NodeKind,
    pub table: String,
    pub files: Vec<String>,
    pub projection: Vec<u32>,
    pub partition_groups: Vec<Vec<Vec<u32>>>,
    /// A limit pushed into the scan by DataFusion, not one this mode derived.
    pub limit: Option<usize>,
}

impl GpuLoadParquet {
    pub fn new(
        table: String,
        files: Vec<String>,
        projection: Vec<u32>,
        partition_groups: Vec<Vec<Vec<u32>>>,
        limit: Option<usize>,
        schema: Schema,
    ) -> Self {
        // Batching off still declares MultipleBatches: no downstream phase may assume a
        // lane is one batch, and only an accumulator may make that declaration.
        let layout = PartitionLayout {
            n: partition_groups.len(),
            key_distribution: KeyDistribution::NotSpecified,
            sort_order: SortOrder::NotSpecified,
            batch_layout: BatchLayout::MultipleBatches,
        };
        Self {
            kind: NodeKind::Source { layout, schema },
            table,
            files,
            projection,
            partition_groups,
            limit,
        }
    }
}

impl GpuNode for GpuLoadParquet {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        Vec::new()
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        if self.partition_groups.is_empty() {
            return Err(PlanError::Invalid(format!(
                "{}: the partitioner returned no lanes",
                self.table
            )));
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
