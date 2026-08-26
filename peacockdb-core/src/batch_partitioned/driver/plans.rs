//! Plans built from the real node types, since the driver reads what a node declares —
//! lane counts, categories, intervals. Only the executors are mock.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::JoinType;

use crate::batch_partitioned::expr::{Expr, NamedExpr};
use crate::batch_partitioned::layout::ColumnOrder;
use crate::batch_partitioned::node::{GpuNode, RowInterval};
use crate::batch_partitioned::nodes::{
    GpuCoalesceAllBatches, GpuEmitPartitions, GpuFilter, GpuInterleave, GpuJoin, GpuLimit,
    GpuLoadParquet, GpuMergePartitions, GpuMergeSortedPartitions, GpuProject, GpuSort, GpuUnion,
    GpuUnload,
};
use crate::batch_partitioned::parquet_meta::ScanMetadata;
use crate::batch_partitioned::partitioner::RowGroupMeta;
use crate::batch_partitioned::schema::Schema;

pub(super) fn schema() -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
        "k",
        DataType::Int64,
        true,
    )])))
}

/// A loader with `lanes` lanes. The mapping is what the driver reads the lane count off;
/// how many batches each lane emits is the mock's script.
pub(super) fn source(table: &str, lanes: usize) -> Box<dyn GpuNode> {
    let groups: Vec<RowGroupMeta> = (0..lanes as u32)
        .map(|index| RowGroupMeta {
            index,
            rows: 100,
            bytes: 800,
        })
        .collect();
    let scan = ScanMetadata {
        groups: groups.clone(),
        can_be_null: vec![false],
    };
    let partition_groups = (0..lanes as u32).map(|lane| vec![vec![lane]]).collect();
    Box::new(GpuLoadParquet::new(
        table.to_string(),
        vec![format!("/{table}.parquet")],
        vec![0],
        partition_groups,
        &scan,
        None,
        schema(),
    ))
}

pub(super) fn filter(input: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
    Box::new(GpuFilter::new(input, Expr::column(0, "k"), None, schema()))
}

pub(super) fn project(input: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
    Box::new(GpuProject::new(
        input,
        vec![NamedExpr::new(Expr::column(0, "k"), "k")],
        schema(),
    ))
}

pub(super) fn sort(input: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
    Box::new(GpuSort::new(input, vec![key()], None))
}

pub(super) fn coalesce_all(input: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
    Box::new(GpuCoalesceAllBatches::new(input))
}

pub(super) fn limit(input: Box<dyn GpuNode>, skip: u64, fetch: Option<u64>) -> Box<dyn GpuNode> {
    Box::new(GpuLimit::new(input, RowInterval { skip, fetch }))
}

pub(super) fn emit(input: Box<dyn GpuNode>, lanes: usize) -> Box<dyn GpuNode> {
    Box::new(GpuEmitPartitions::new(input, vec![0], lanes))
}

pub(super) fn merge(input: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
    Box::new(GpuMergePartitions::new(input))
}

pub(super) fn merge_sorted(input: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
    Box::new(GpuMergeSortedPartitions::new(input, vec![key()], None))
}

pub(super) fn union(branches: Vec<Box<dyn GpuNode>>) -> Box<dyn GpuNode> {
    Box::new(GpuUnion::new(branches, schema()))
}

pub(super) fn interleave(branches: Vec<Box<dyn GpuNode>>) -> Box<dyn GpuNode> {
    Box::new(GpuInterleave::new(branches, schema()))
}

/// The build side is always the left child, which is the orientation the schedule turns
/// into "the build subtree drains first".
pub(super) fn join(build: Box<dyn GpuNode>, probe: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
    Box::new(GpuJoin::new(
        build,
        probe,
        JoinType::Inner,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        None,
        schema(),
    ))
}

pub(super) fn unload(input: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
    Box::new(GpuUnload::new(input, None))
}

pub(super) fn unload_limited(
    input: Box<dyn GpuNode>,
    skip: u64,
    fetch: Option<u64>,
) -> Box<dyn GpuNode> {
    Box::new(GpuUnload::new(input, Some(RowInterval { skip, fetch })))
}

fn key() -> ColumnOrder {
    ColumnOrder {
        column: 0,
        ascending: true,
        nulls_first: false,
    }
}
