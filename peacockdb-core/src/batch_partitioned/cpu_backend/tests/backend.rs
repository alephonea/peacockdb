//! What `executors_for` builds, and what the executors it built report holding.
//!
//! The category claim is the one the driver cannot make for itself: it checks the category
//! it was handed against the node's, so a backend that builds the wrong executor is caught
//! at run time and named — but only for a node some plan actually reaches. Every node kind
//! is asked here instead.

use super::*;
use crate::batch_partitioned::backend::Backend;
use crate::batch_partitioned::cpu_backend::backend::CpuBackend;
use crate::batch_partitioned::cpu_backend::join::CpuJoin;
use crate::batch_partitioned::executor::Executor;
use crate::batch_partitioned::layout::{ColumnOrder, PartitionLayout};
use crate::batch_partitioned::node::RowInterval;
use crate::batch_partitioned::nodes::join::{JoinFilterColumn, JoinSide, NestedLoopJoinType};
use crate::batch_partitioned::nodes::{
    ExecutorCategory, GpuAccumulateBatchesAndSort, GpuAggregate, GpuAggregateBatches,
    GpuCoalesceAllBatches, GpuCrossJoin, GpuEmitPartitions, GpuFilter, GpuInterleave, GpuJoin,
    GpuLimit, GpuMergePartitions, GpuMergeSortedPartitions, GpuNestedLoopJoin, GpuProject, GpuSort,
    GpuUnion, GpuUnload, NodeRef, as_node_ref, category_of,
};
use datafusion::common::JoinType;

/// A stub input with a layout stated, since a join's two sides differ in exactly that.
fn given(columns: &[(&str, DataType)], batches: BatchLayout) -> Box<dyn GpuNode> {
    Box::new(Given {
        kind: NodeKind::Intermediate {
            layout: PartitionLayout {
                batch_layout: batches,
                ..PartitionLayout::new(1)
            },
            schema: schema_of(columns),
        },
    })
}

fn streaming() -> Box<dyn GpuNode> {
    given(&GROUPED, BatchLayout::MultipleBatches)
}

fn one_batch() -> Box<dyn GpuNode> {
    given(&GROUPED, BatchLayout::SingleBatch)
}

fn order() -> ColumnOrder {
    ColumnOrder {
        column: 1,
        ascending: true,
        nulls_first: false,
    }
}

/// One node of every kind that reaches a backend. The scan is not here: it opens its file
/// when it is built, so it is proved in `source.rs`, over a parquet that exists.
fn every_kind() -> Vec<Box<dyn GpuNode>> {
    let merged: Box<dyn GpuNode> = Box::new(GpuMergeSortedPartitions::new(
        given(&GROUPED, BatchLayout::MultipleBatches),
        vec![order()],
        None,
    ));
    vec![
        Box::new(GpuFilter::new(
            streaming(),
            greater_than_value(0),
            None,
            schema_of(&GROUPED),
        )),
        Box::new(GpuProject::new(
            streaming(),
            vec![NamedExpr {
                expr: Expr::column(0, "k"),
                name: "k".to_string(),
            }],
            schema_of(&[GROUPED[0].clone()]),
        )),
        Box::new(GpuSort::new(streaming(), vec![order()], None)),
        Box::new(GpuAggregate::new(
            streaming(),
            summing(),
            summed_state(),
            summed_state(),
        )),
        Box::new(GpuAggregateBatches::new(
            given_schema(summed_state()),
            merging(),
            summed_state(),
            summed_state(),
        )),
        Box::new(GpuCoalesceAllBatches::new(streaming())),
        Box::new(GpuAccumulateBatchesAndSort::new(
            streaming(),
            vec![order()],
            None,
        )),
        Box::new(GpuLimit::new(
            streaming(),
            RowInterval {
                skip: 0,
                fetch: Some(1),
            },
        )),
        merged,
        Box::new(GpuEmitPartitions::new(streaming(), vec![0], 2)),
        Box::new(GpuJoin::new(
            one_batch(),
            streaming(),
            JoinType::Inner,
            vec![(0, 0)],
            None,
            Vec::new(),
            false,
            None,
            schema_of(&[GROUPED, GROUPED].concat()),
        )),
        Box::new(GpuCrossJoin::new(
            one_batch(),
            one_batch(),
            None,
            schema_of(&[GROUPED, GROUPED].concat()),
        )),
        Box::new(GpuNestedLoopJoin::new(
            one_batch(),
            one_batch(),
            NestedLoopJoinType::Inner,
            Expr::binary(
                Expr::column(0, "v"),
                BinaryOp::Gt,
                Expr::column(1, "v"),
                DataType::Boolean,
            ),
            vec![
                JoinFilterColumn {
                    side: JoinSide::Build,
                    index: 1,
                },
                JoinFilterColumn {
                    side: JoinSide::Probe,
                    index: 1,
                },
            ],
            None,
            schema_of(&[GROUPED, GROUPED].concat()),
        )),
        Box::new(GpuMergePartitions::new(streaming())),
        Box::new(GpuUnion::new(
            vec![streaming(), streaming()],
            schema_of(&GROUPED),
        )),
        Box::new(GpuInterleave::new(
            vec![streaming(), streaming()],
            schema_of(&GROUPED),
        )),
        Box::new(GpuUnload::new(streaming(), None)),
    ]
}

/// A stub input over a state schema, which is what a merge reads.
fn given_schema(schema: Schema) -> Box<dyn GpuNode> {
    Box::new(Given {
        kind: NodeKind::Intermediate {
            layout: PartitionLayout {
                batch_layout: BatchLayout::MultipleBatches,
                ..PartitionLayout::new(1)
            },
            schema,
        },
    })
}

/// One sum over the value column, grouped by the key — the smallest body there is, since
/// what is under test is which executor gets built.
fn summing() -> AggregateBody {
    sum_over(1, "v")
}

/// The same sum a column later: a merge reads the state its child emitted, where column 1
/// is that sum rather than the value it came from.
fn merging() -> AggregateBody {
    sum_over(1, "sum(v)")
}

fn sum_over(column: u32, name: &str) -> AggregateBody {
    AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![AggCall {
            func: PlanAgg::Sum,
            args: vec![Expr::column(column, name)],
            outputs: vec![Field::new("sum(v)", DataType::Int64, true)],
        }],
        finalize: None,
    }
}

fn summed_state() -> Schema {
    state_of(
        &[("k", DataType::Utf8), ("sum(v)", DataType::Int64)],
        1,
        None,
    )
}

fn greater_than_value(bound: i64) -> Expr {
    Expr::binary(
        Expr::column(1, "v"),
        BinaryOp::Gt,
        Expr::Literal(ScalarValue::Int64(Some(bound))),
        DataType::Boolean,
    )
}

/// A number per node kind, from an exhaustive match, so the hand-written list below is
/// answerable: a variant added to `NodeRef` fails to compile here rather than going
/// quietly untested.
fn node_kind(node: &dyn GpuNode) -> u8 {
    match as_node_ref(node) {
        NodeRef::LoadParquet(_) => LOAD_PARQUET,
        NodeRef::Filter(_) => 1,
        NodeRef::Project(_) => 2,
        NodeRef::Sort(_) => 3,
        NodeRef::Aggregate(_) => 4,
        NodeRef::CoalesceAllBatches(_) => 5,
        NodeRef::AccumulateBatchesAndSort(_) => 6,
        NodeRef::AggregateBatches(_) => 7,
        NodeRef::Limit(_) => 8,
        NodeRef::MergeSortedPartitions(_) => 9,
        NodeRef::EmitPartitions(_) => 10,
        NodeRef::Join(_) => 11,
        NodeRef::CrossJoin(_) => 12,
        NodeRef::NestedLoopJoin(_) => 13,
        NodeRef::MergePartitions(_) => 14,
        NodeRef::Union(_) => 15,
        NodeRef::Interleave(_) => 16,
        NodeRef::Unload(_) => 17,
    }
}

const LOAD_PARQUET: u8 = 0;
const NODE_KINDS: u8 = 18;

fn category_built(node: &dyn GpuNode) -> ExecutorCategory {
    CpuBackend::executors_for(&ctx(), node, 0, 0)
        .expect("this backend implements every node kind")
        .category()
}

/// The aggregate here is the one node kind the list holds twice — `GpuAggregate` and
/// `GpuAggregateBatches` are different kinds — so the cover is over kinds, not over the
/// list's length.
#[test]
fn the_hand_written_list_holds_every_node_kind_but_the_scan() {
    let covered: std::collections::BTreeSet<u8> = every_kind()
        .iter()
        .map(|node| node_kind(node.as_ref()))
        .collect();
    let expected: std::collections::BTreeSet<u8> = (0..NODE_KINDS)
        .filter(|kind| *kind != LOAD_PARQUET)
        .collect();
    assert_eq!(
        covered, expected,
        "a node kind exists that this file does not build; the scan is the one exception, \
         and it is built in source.rs over a parquet that exists"
    );
}

#[test]
fn every_node_kind_builds_the_executor_its_category_names() {
    for node in every_kind() {
        assert_eq!(
            category_built(node.as_ref()),
            category_of(node.as_ref()),
            "{} built an executor of another category",
            node.name()
        );
    }
}

/// The enforcer reads `scratch_bytes` BEFORE the call, so a transient charged for a read
/// that never happens refuses a query that fits. The build-side semi family's probe call
/// is the key project alone — it does not touch the build side — and this charged it
/// anyway until the review found it.
#[test]
fn a_build_side_semi_joins_probe_is_not_charged_the_build_side() {
    let build = grouped(vec![Some("a"), Some("b")], vec![Some(1), Some(2)]);
    let build_bytes = build.record_batch().get_array_memory_size();
    assert!(build_bytes > 0, "a build side with rows costs something");

    let semi = CpuJoin::hash(
        &semi_join(JoinType::LeftSemi),
        &schema_of(&GROUPED).fields,
        &schema_of(&GROUPED).fields,
        ctx(),
    )
    .expect("a semi join builds");
    let (probing, _) = semi.set_build(build).expect("the build side is set");
    assert_eq!(
        probing.scratch_bytes(2, 64),
        64,
        "a probe call that only projects keys transiently costs the batch"
    );
    assert_eq!(
        probing.resident_bytes(),
        build_bytes,
        "and the build side is still held until the finish consumes it"
    );
}

/// An Inner join, whose probe call does read the build side, so the charge is right there.
#[test]
fn an_inner_joins_probe_is_charged_the_build_side_it_reads() {
    let build = grouped(vec![Some("a"), Some("b")], vec![Some(1), Some(2)]);
    let build_bytes = build.record_batch().get_array_memory_size();
    let inner = CpuJoin::hash(
        &semi_join(JoinType::Inner),
        &schema_of(&GROUPED).fields,
        &schema_of(&GROUPED).fields,
        ctx(),
    )
    .expect("an inner join builds");
    let (probing, _) = inner.set_build(build).expect("the build side is set");
    assert_eq!(probing.scratch_bytes(2, 64), build_bytes + 64);
}

fn semi_join(join_type: JoinType) -> GpuJoin {
    let output = match join_type {
        JoinType::LeftSemi => schema_of(&GROUPED),
        _ => schema_of(&[GROUPED, GROUPED].concat()),
    };
    GpuJoin::new(
        one_batch(),
        streaming(),
        join_type,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        None,
        output,
    )
}
