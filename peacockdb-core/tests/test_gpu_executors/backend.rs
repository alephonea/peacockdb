//! `GpuBackend` as a `Backend`: the caller the impl did not have.
//!
//! The CPU twin of this asserts the same thing over every node kind, and the gap here was
//! a missing caller rather than a missing check — `executors_for` typechecked and nothing
//! constructed it. A session already holds the two fields a `GpuContext` carries, so the
//! claim is buildable without a driver: for each kind this backend implements, the
//! executor built is the one the node's category names, found at the post-order the driver
//! would have handed in.

use peacockdb_core::batch_partitioned::nodes::join::joined_schema;
use super::*;

use datafusion::common::JoinType;
use peacockdb_core::batch_partitioned::backend::Backend;
use peacockdb_core::batch_partitioned::gpu_backend::backend::{GpuBackend, GpuContext};
use peacockdb_core::batch_partitioned::nodes::{
    ExecutorCategory, GpuAccumulateBatchesAndSort, GpuEmitPartitions, GpuJoin, GpuUnload,
    category_of,
};
use peacockdb_core::batch_partitioned::recipe::attach_recipes;

/// The tree children-first, which is the numbering a recipe is addressed by and what the
/// driver hands `executors_for`.
fn children_first<'a>(node: &'a dyn GpuNode, into: &mut Vec<&'a dyn GpuNode>) {
    for child in node.children() {
        children_first(child, into);
    }
    into.push(node);
}

/// Six categories in one tree: a source, an exec, a batch accumulator, an emitter, a join
/// and an unload. The emitter and the join are the arms that do the work — the join's
/// reads `per_call_join_type` and derives a key schema off the probe — so a tree without
/// them would leave `executors_for` half-called, which is the shape this file exists to
/// end.
fn tree() -> Box<dyn GpuNode> {
    let filtered = GpuFilter::new(
        source_per_row_group(),
        Expr::binary(
            Expr::column(1, "v"),
            BinaryOp::Gt,
            Expr::Literal(ScalarValue::Int64(Some(0))),
            DataType::Boolean,
        ),
        None,
        Schema::new(Arc::new(columns())),
    );
    // Both sides scatter into two lanes, so the join is one the planner could have made:
    // it joins lane-wise, and a build side of one lane against a probe of two is a shape
    // no plan produces.
    let build = GpuAccumulateBatchesAndSort::new(
        Box::new(GpuEmitPartitions::new(Box::new(filtered), vec![0], 2)),
        vec![ColumnOrder {
            column: 1,
            ascending: true,
            nulls_first: false,
        }],
        None,
    );
    // A LeftSemi: its probe call is the key project alone, so the join arm takes the
    // branch that derives a key schema rather than the one that does not.
    let joined = GpuJoin::new(
        Box::new(build),
        Box::new(GpuEmitPartitions::new(source(), vec![0], 2)),
        JoinType::LeftSemi,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        None,
        Schema::new(Arc::new(columns())),
        joined_schema(&columns(), &columns(), JoinType::LeftSemi),
    );
    Box::new(GpuUnload::new(Box::new(joined), None))
}

#[test]
fn each_node_builds_the_executor_its_category_names() {
    let tree = tree();
    let session = Session::open(tree.as_ref());
    let ctx = GpuContext {
        executor: session.executor,
        recipes: attach_recipes(tree.as_ref()).expect("every node's payload is writable"),
    };
    let mut nodes = Vec::new();
    children_first(tree.as_ref(), &mut nodes);
    // Sorted before it is deduped, since `dedup` only drops neighbours and the walk hands
    // these back interleaved.
    let mut built: Vec<String> = nodes
        .iter()
        .map(|node| format!("{:?}", category_of(*node)))
        .collect();
    built.sort();
    built.dedup();
    assert_eq!(
        built.len(),
        6,
        "six categories in the tree, so six arms of executors_for are called: {built:?}"
    );

    for (post_order, node) in nodes.iter().enumerate() {
        let built = GpuBackend::executors_for(&ctx, *node, post_order, 0)
            .unwrap_or_else(|error| panic!("{}: {error}", node.name()))
            .category();
        assert_eq!(
            built,
            category_of(*node),
            "{} built an executor of another category",
            node.name()
        );
    }
}

/// The post-order is an address, so handing the wrong one finds another node's recipe —
/// and the shapes differ enough that building fails rather than running the wrong calls.
/// This is the claim `executors_for` taking a number rather than deriving one rests on.
#[test]
fn a_node_built_at_another_nodes_post_order_is_refused() {
    let tree = tree();
    let session = Session::open(tree.as_ref());
    let ctx = GpuContext {
        executor: session.executor,
        recipes: attach_recipes(tree.as_ref()).expect("every node's payload is writable"),
    };
    let mut nodes = Vec::new();
    children_first(tree.as_ref(), &mut nodes);
    // A source's recipe is one call taking row groups, which is not what an accumulator's
    // two calls look like — so the accumulator built at the scan's position is refused.
    let sort = nodes
        .iter()
        .find(|node| category_of(**node) == ExecutorCategory::BatchAccumulator)
        .expect("the tree carries an accumulator");
    assert!(
        GpuBackend::executors_for(&ctx, *sort, 0, 0).is_err(),
        "the scan's recipe built an accumulator without complaint"
    );
}
