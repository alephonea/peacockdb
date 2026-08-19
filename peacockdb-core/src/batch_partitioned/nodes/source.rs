//! `GpuLoadParquet`: the one source node.

use super::super::error::PlanError;
use super::super::layout::{BatchLayout, KeyDistribution, NodeKind, PartitionLayout, SortOrder};
use super::super::node::GpuNode;
use super::super::partitioner::RowGroupMeta;
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
    /// The row groups the mapping addresses, with their rows and their parquet bytes over
    /// the projected columns — the only real numbers a plan-time model has, and what lets
    /// the estimator price the batches this mapping actually produces rather than the ones
    /// a budget would have afforded.
    pub survivors: Vec<RowGroupMeta>,
    /// A limit pushed into the scan by DataFusion, not one this mode derived.
    pub limit: Option<usize>,
}

impl GpuLoadParquet {
    pub fn new(
        table: String,
        files: Vec<String>,
        projection: Vec<u32>,
        partition_groups: Vec<Vec<Vec<u32>>>,
        survivors: &[RowGroupMeta],
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
            survivors: survivors.to_vec(),
            limit,
        }
    }
}

impl GpuLoadParquet {
    pub fn rows(&self) -> u64 {
        self.survivors.iter().map(|group| group.rows).sum()
    }

    pub fn bytes(&self) -> u64 {
        self.survivors.iter().map(|group| group.bytes).sum()
    }

    /// The largest batch the mapping produces. One batch per lane, one per row group and a
    /// budgeted size are three different answers to this, which is why the model reads it
    /// off the mapping rather than off the budget.
    pub fn largest_batch_bytes(&self) -> u64 {
        // Every entry in the mapping came from these survivors, so a miss would be a
        // mapping addressing a row group this scan does not read.
        let bytes_of = |index: &u32| {
            self.survivors
                .iter()
                .find(|group| group.index == *index)
                .expect("the mapping addresses a surviving row group")
                .bytes
        };
        self.partition_groups
            .iter()
            .flatten()
            .map(|batch| batch.iter().map(bytes_of).sum::<u64>())
            .max()
            .unwrap_or(0)
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
