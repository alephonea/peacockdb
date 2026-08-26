//! The tree indexed once: heights, pre-order numbering, lane counts, and the ranges every
//! hold is expressed over.
//!
//! Node indices are pre-order, which buys two things the driver leans on: a subtree is a
//! contiguous range, and "order" — the leftmost tie-break — is the index itself. Children
//! are snapshotted here because [`GpuNode::children`] builds a `Vec` per call, and the
//! schedule asks for them on every step.

use super::scheduler::{JoinShape, PlanShape};
use crate::batch_partitioned::error::PlanError;
use crate::batch_partitioned::node::{GpuNode, RowInterval};
use crate::batch_partitioned::nodes::{ExecutorCategory, category_of};

pub(crate) struct IndexedNode<'a> {
    pub node: &'a dyn GpuNode,
    pub category: ExecutorCategory,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
    pub lanes: usize,
    /// This node's position in a children-first walk, which is what `RecipePlan` indexes
    /// by and what a backend hands to `executors_for`. Pre-order is what the schedule
    /// needs and post-order is what the recipes are addressed by; the two are computed
    /// here together so nothing has to reconcile them later.
    pub post_order: usize,
    /// The child's lane count, which is how many `Done` events a partition accumulator
    /// owes. Zero for every other category.
    pub input_lanes: usize,
    pub interval: Option<RowInterval>,
    /// How many independently-ready units the schedule tracks for this node. Output lanes
    /// for most, but a cross-lane accumulator becomes ready one input lane at a time and
    /// an emitter reads a single one.
    pub ready_lanes: usize,
    /// Where this node's accounting slots start: one per lane when it is lane-scoped,
    /// one for the node otherwise.
    pub slot_base: usize,
}

pub(crate) struct PlanIndex<'a> {
    pub nodes: Vec<IndexedNode<'a>>,
    pub shape: PlanShape,
    pub slots: usize,
}

impl<'a> PlanIndex<'a> {
    pub(crate) fn build(root: &'a dyn GpuNode) -> Result<Self, PlanError> {
        let mut nodes = Vec::new();
        let mut heights = Vec::new();
        let mut post_order = 0;
        walk(root, 0, None, &mut nodes, &mut heights, &mut post_order);

        for node in &mut nodes {
            node.ready_lanes = match node.category {
                ExecutorCategory::PartitionAccumulator => node.input_lanes,
                ExecutorCategory::PartitionEmitter => 1,
                _ => node.lanes,
            };
        }
        let subtree = subtree_ranges(&nodes);
        let joins = nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.category == ExecutorCategory::Join)
            .map(|(index, node)| JoinShape {
                node: index,
                probe: subtree[node.children[PROBE_CHILD]],
                lanes: node.lanes,
            })
            .collect();

        let mut slots = 0;
        for node in &mut nodes {
            node.slot_base = slots;
            slots += if node.category.is_lane_scoped() {
                node.lanes
            } else {
                1
            };
        }

        Ok(Self {
            shape: PlanShape {
                heights,
                lanes: nodes.iter().map(|node| node.ready_lanes).collect(),
                subtree,
                joins,
            },
            nodes,
            slots,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn slot(&self, node: usize, lane: usize) -> usize {
        let indexed = &self.nodes[node];
        indexed.slot_base
            + if indexed.category.is_lane_scoped() {
                lane
            } else {
                0
            }
    }
}

pub(crate) const ROOT: usize = 0;
pub(crate) const PROBE_CHILD: usize = 1;

fn walk<'a>(
    node: &'a dyn GpuNode,
    height: u32,
    parent: Option<usize>,
    nodes: &mut Vec<IndexedNode<'a>>,
    heights: &mut Vec<u32>,
    post_order: &mut usize,
) -> usize {
    let index = nodes.len();
    let category = category_of(node);
    nodes.push(IndexedNode {
        node,
        category,
        children: Vec::new(),
        parent,
        lanes: 0,
        input_lanes: 0,
        interval: node.row_interval(),
        ready_lanes: 0,
        slot_base: 0,
        post_order: 0,
    });
    heights.push(height);
    let children: Vec<usize> = node
        .children()
        .into_iter()
        .map(|child| walk(child, height + 1, Some(index), nodes, heights, post_order))
        .collect();
    // A sink declares no layout of its own, so it runs the lanes its input hands it.
    let lanes = match node.kind().layout() {
        Some(layout) => layout.n,
        None => nodes[children[0]].lanes,
    };
    let input_lanes = if category == ExecutorCategory::PartitionAccumulator {
        nodes[children[0]].lanes
    } else {
        0
    };
    nodes[index].children = children;
    nodes[index].lanes = lanes;
    nodes[index].input_lanes = input_lanes;
    // Children first, so this is the position `attach_recipes` gave the same node — the
    // address a recipe is looked up by, and the one the FFI's seqs are numbered in. Taken
    // from the walk the driver already makes rather than derived a third time.
    nodes[index].post_order = *post_order;
    *post_order += 1;
    index
}

/// `[start, end)` per node. Pre-order makes every subtree contiguous, so the end is the
/// largest end among the children.
fn subtree_ranges(nodes: &[IndexedNode<'_>]) -> Vec<(usize, usize)> {
    let mut end: Vec<usize> = (0..nodes.len()).map(|index| index + 1).collect();
    for index in (0..nodes.len()).rev() {
        for child in &nodes[index].children {
            end[index] = end[index].max(end[*child]);
        }
    }
    (0..nodes.len()).map(|index| (index, end[index])).collect()
}

/// Each node's children-first position, in the driver's own pre-order.
///
/// Exposed for one guard: that these are the positions `attach_recipes` gave the same
/// nodes. Two numberings computed by two walks in two files — one at plan time, one at run
/// time — agree by construction only until someone changes a walk, and nothing else
/// compares them.
pub fn post_order_of_every_node(root: &dyn GpuNode) -> Result<Vec<usize>, PlanError> {
    Ok(PlanIndex::build(root)?
        .nodes
        .iter()
        .map(|node| node.post_order)
        .collect())
}

#[cfg(test)]
mod tests;
