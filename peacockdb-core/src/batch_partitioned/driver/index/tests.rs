//! What the index derives, against a plan whose shape is written down here.
//!
//! Every one of these is read far from where it is computed — a subtree range decides
//! which nodes a join holds, `ready_lanes` decides when a cross-lane node is runnable, and
//! `slot_base` decides whose residency a call is charged to — so a wrong one shows as a
//! deadlock or a mis-billed byte rather than as a wrong index.

use super::*;
use crate::batch_partitioned::driver::plans::*;

/// unload <- join <- [coalesce <- merge_sorted <- source(build, 2),
///                    emit(4) <- source(probe, 1)]
///
/// One of everything the counts below distinguish: a join with two children, a partition
/// accumulator with two input lanes, an emitter reading one lane and filling four, and a
/// lane-scoped node on each side of it.
fn plan() -> Box<dyn GpuNode> {
    unload(join(
        coalesce_all(merge_sorted(sort(source("build", 2)))),
        emit(source("probe", 1), 4),
    ))
}

/// The plan above in pre-order, which is the numbering every assertion here is written in.
const PRE_ORDER: [&str; 8] = [
    "GpuUnload",
    "GpuJoin",
    "GpuCoalesceAllBatches",
    "GpuMergeSortedPartitions",
    "GpuSort",
    "GpuLoadParquet",
    "GpuEmitPartitions",
    "GpuLoadParquet",
];

fn indexed(root: &dyn GpuNode) -> PlanIndex<'_> {
    PlanIndex::build(root).expect("the plan indexes")
}

#[test]
fn nodes_are_numbered_pre_order_and_each_knows_its_parent() {
    let plan = plan();
    let index = indexed(plan.as_ref());
    let names: Vec<&str> = index.nodes.iter().map(|node| node.node.name()).collect();
    assert_eq!(names, PRE_ORDER);
    let parents: Vec<Option<usize>> = index.nodes.iter().map(|node| node.parent).collect();
    assert_eq!(
        parents,
        vec![
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(1),
            Some(6)
        ],
        "the root has no parent and every other node is its own child's"
    );
    let children: Vec<Vec<usize>> = index
        .nodes
        .iter()
        .map(|node| node.children.clone())
        .collect();
    assert_eq!(
        children,
        vec![
            vec![1],
            vec![2, 6],
            vec![3],
            vec![4],
            vec![5],
            Vec::new(),
            vec![7],
            Vec::new(),
        ],
        "the join's build side is its first child, which is what makes BUILD_SLOT zero"
    );
}

/// The property the numbering exists for: a hold is expressed as a range, so a subtree has
/// to be contiguous. It is what stops a join's hold from covering nodes outside its probe.
#[test]
fn a_subtree_is_a_contiguous_range_of_the_numbering() {
    let plan = plan();
    let index = indexed(plan.as_ref());
    assert_eq!(
        index.shape.subtree,
        vec![
            (0, 8),
            (1, 8),
            (2, 6),
            (3, 6),
            (4, 6),
            (5, 6),
            (6, 8),
            (7, 8),
        ],
        "each node's range starts at itself and ends past its last descendant"
    );
    let join = &index.shape.joins;
    assert_eq!(join.len(), 1);
    assert_eq!(
        join[0].probe,
        (6, 8),
        "a join holds its PROBE subtree, which is its second child's range and not its \
         first's"
    );
}

/// Three counts, three categories, and they differ for reasons a reader cannot see from
/// the lane count alone.
#[test]
fn the_counts_a_category_changes_are_the_ones_it_should() {
    let plan = plan();
    let index = indexed(plan.as_ref());
    let of = |name: &str| {
        index
            .nodes
            .iter()
            .position(|node| node.node.name() == name)
            .expect("a node of that kind")
    };
    let merge = &index.nodes[of("GpuMergeSortedPartitions")];
    assert_eq!(
        (merge.lanes, merge.input_lanes, merge.ready_lanes),
        (1, 2, 2),
        "a partition accumulator emits one lane, owes a Done per input lane, and becomes \
         ready one input lane at a time"
    );
    let emitter = &index.nodes[of("GpuEmitPartitions")];
    assert_eq!(
        (emitter.lanes, emitter.input_lanes, emitter.ready_lanes),
        (4, 0, 1),
        "an emitter fills four lanes from the one it reads, and it is that one that makes \
         it runnable"
    );
    let sort = &index.nodes[of("GpuSort")];
    assert_eq!(
        (sort.lanes, sort.input_lanes, sort.ready_lanes),
        (2, 0, 2),
        "a lane-scoped node is ready per lane, and owes no Done events at all"
    );
}

/// One slot per lane where an executor is per lane, one for the node where it is not —
/// which is what makes a cross-lane node's residency a single figure rather than N.
#[test]
fn a_slot_is_per_lane_only_where_the_executor_is() {
    let plan = plan();
    let index = indexed(plan.as_ref());
    let bases: Vec<usize> = index.nodes.iter().map(|node| node.slot_base).collect();
    // Four lanes for the unload and the join, which take the emitter's; one for the
    // coalesce; one apiece for the merge and the emitter, which are cross-lane; and two
    // for the sort and its source, which are the build side's.
    assert_eq!(bases, vec![0, 4, 8, 9, 10, 12, 14, 15]);
    assert_eq!(index.slots, 16);
    let merge = 3;
    assert_eq!(
        (index.slot(merge, 0), index.slot(merge, 1)),
        (9, 9),
        "every lane of a cross-lane node bills the same slot"
    );
    let sort = 4;
    assert_eq!(
        (index.slot(sort, 0), index.slot(sort, 1)),
        (10, 11),
        "and a lane-scoped one bills its own"
    );
}
