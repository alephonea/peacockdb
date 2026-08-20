//! The recipe plan: which frozen-ABI calls each plan node makes, and against which seq.
//!
//! The C++ side never sees this mode's tree. It is sent a plan in the legacy vocabulary
//! whose nodes exist to be addressed — a menu of parameterized kernels — and the driver
//! then calls the ABI as often as its own schedule wants. This module is that mapping:
//! per node, the calls in order, each naming the seq it addresses and the handles it is
//! passed. `llm-wiki/tasks/batch_partitioned_executor.md` has the table it implements.
//!
//! A node that makes no ABI call gets no recipe, and the absence is a statement about the
//! node rather than a gap: a forwarder routes batches and touches no device.

mod join;
mod types;

use super::node::GpuNode;
use super::nodes::{
    GpuAccumulateBatchesAndSort, GpuAggregateBatches, GpuCoalesceAllBatches, GpuCrossJoin,
    GpuEmitPartitions, GpuLimit, GpuLoadParquet, GpuMergeSortedPartitions, GpuSort, GpuUnload,
    NodeRef, as_node_ref,
};
use super::schema::Schema;

pub use types::{AbiSymbol, Call, CallPattern, FbKind, Input, ProjectRole, Recipe, Seq, Seqs};

/// Every node's recipe, indexed by the post-order position the estimator and the memory
/// golden already number by, so a reader lines the three up without a second convention.
///
/// That index is a position in the TREE and a [`Seq`] is an address in the recipe plan:
/// they part company at the first node with two calls, and a driver holding both at once
/// has to keep them apart.
#[derive(Debug, Default)]
pub struct RecipePlan {
    recipes: Vec<Option<Recipe>>,
}

impl RecipePlan {
    /// By post-order position in the tree, not by seq. `None` where that node makes no ABI
    /// call at all, which is what a forwarder does.
    pub fn get(&self, node: usize) -> Option<&Recipe> {
        self.recipes.get(node).and_then(|recipe| recipe.as_ref())
    }

    /// Nodes walked — the length the memory model's per-node vector has, and what a
    /// consumer checks its own tree against before reading a `None` as an answer.
    pub fn nodes(&self) -> usize {
        self.recipes.len()
    }
}

/// Hang a recipe on every node that drives the ABI. It runs over the finished tree: a
/// recipe is a statement about a node, and a node is not finished until the plan is.
pub fn attach_recipes(root: &dyn GpuNode) -> RecipePlan {
    let mut plan = RecipePlan::default();
    let mut seqs = Seqs::default();
    attach_node(root, &mut seqs, &mut plan);
    plan
}

fn attach_node(node: &dyn GpuNode, seqs: &mut Seqs, plan: &mut RecipePlan) {
    for child in node.children() {
        attach_node(child, seqs, plan);
    }
    let recipe = recipe_of(node, seqs);
    plan.recipes.push(recipe);
}

/// The schemas a node declares it consumes, in child order. Read here and handed to the
/// recipe function rather than reached for by it: what the function may not do is walk
/// the tree, and a node's own declared input arriving as an argument is not walking.
fn input_schemas(node: &dyn GpuNode) -> Vec<&Schema> {
    node.children()
        .into_iter()
        .map(|child| child.kind().schema().expect("a sink cannot be an input"))
        .collect()
}

/// The mapping, one arm per row of the table. Every arm reads its own node and the
/// schemas above, and nothing else: the table is a per-node claim, and a function that
/// could reach a child would let it stop being one.
///
/// Seqs ascend with call order within a node, so a recipe read left to right is also the
/// recipe plan read top to bottom.
fn recipe_of(node: &dyn GpuNode, seqs: &mut Seqs) -> Option<Recipe> {
    let inputs = input_schemas(node);
    let inputs = inputs.as_slice();
    match as_node_ref(node) {
        NodeRef::LoadParquet(node) => scan(node, inputs, seqs),
        // One table row, and so one function: the three are the generic map arm, and
        // which kernel runs is the node's own kind.
        NodeRef::Filter(_) => map_arm(FbKind::Filter, inputs, seqs),
        NodeRef::Project(_) => map_arm(FbKind::PlainProject, inputs, seqs),
        NodeRef::Aggregate(_) => map_arm(FbKind::Aggregate, inputs, seqs),
        NodeRef::Sort(node) => sort(node, inputs, seqs),
        NodeRef::AccumulateBatchesAndSort(node) => accumulate_and_sort(node, inputs, seqs),
        NodeRef::CoalesceAllBatches(node) => coalesce_all_batches(node, inputs, seqs),
        NodeRef::AggregateBatches(node) => aggregate_batches(node, inputs, seqs),
        NodeRef::MergeSortedPartitions(node) => merge_sorted_partitions(node, inputs, seqs),
        NodeRef::EmitPartitions(node) => emit_partitions(node, inputs, seqs),
        NodeRef::Join(node) => join::hash_join(node, inputs, seqs),
        NodeRef::CrossJoin(node) => cross_join(node, inputs, seqs),
        NodeRef::NestedLoopJoin(node) => join::nested_loop_join(node, inputs, seqs),
        NodeRef::Limit(node) => limit(node, inputs, seqs),
        NodeRef::Unload(node) => unload(node, inputs, seqs),
        // The three that route rather than compute: not one call between them.
        NodeRef::MergePartitions(_) | NodeRef::Union(_) | NodeRef::Interleave(_) => None,
    }
}

/// The additive entry point, and the reason it exists: the node's own row-group list is
/// overridden per call, so one seq serves every batch the mapping cut the scan into.
fn scan(_node: &GpuLoadParquet, _inputs: &[&Schema], seqs: &mut Seqs) -> Option<Recipe> {
    Some(Recipe::of(vec![Call::seq(
        seqs.allocate(),
        FbKind::Scan,
        vec![Input::RowGroups],
        CallPattern::PerBatch,
    )]))
}

fn map_arm(kind: FbKind, _inputs: &[&Schema], seqs: &mut Seqs) -> Option<Recipe> {
    Some(Recipe::of(vec![Call::seq(
        seqs.allocate(),
        kind,
        vec![Input::Batch],
        CallPattern::PerBatch,
    )]))
}

/// The map arm again; a top-N's `fetch` rides the node, so it trims each batch and the
/// accumulator above it trims the merged result.
fn sort(_node: &GpuSort, _inputs: &[&Schema], seqs: &mut Seqs) -> Option<Recipe> {
    map_arm(FbKind::Sort, _inputs, seqs)
}

/// Sorted as the batches arrive, merged once at done: the merge arm takes whatever k
/// handles the call hands it, so the lane's batch count is a runtime number.
fn accumulate_and_sort(
    _node: &GpuAccumulateBatchesAndSort,
    _inputs: &[&Schema],
    seqs: &mut Seqs,
) -> Option<Recipe> {
    Some(Recipe::of(vec![
        Call::seq(
            seqs.allocate(),
            FbKind::Sort,
            vec![Input::Batch],
            CallPattern::PerBatch,
        ),
        Call::seq(
            seqs.allocate(),
            FbKind::SortPreservingMerge,
            vec![Input::LaneBatches],
            CallPattern::AtDone,
        ),
    ]))
}

fn coalesce_all_batches(
    _node: &GpuCoalesceAllBatches,
    _inputs: &[&Schema],
    seqs: &mut Seqs,
) -> Option<Recipe> {
    Some(Recipe::of(vec![Call::seq(
        seqs.allocate(),
        FbKind::CoalescePartitions,
        vec![Input::LaneBatches],
        CallPattern::AtDone,
    )]))
}

/// A compaction runs exactly what done runs, which is what makes the doubling threshold a
/// scheduling decision rather than a second computation.
fn aggregate_batches(
    _node: &GpuAggregateBatches,
    _inputs: &[&Schema],
    seqs: &mut Seqs,
) -> Option<Recipe> {
    Some(Recipe::of(vec![
        Call::seq(
            seqs.allocate(),
            FbKind::CoalescePartitions,
            vec![Input::LaneBatches],
            CallPattern::PerCompaction,
        ),
        Call::seq(
            seqs.allocate(),
            FbKind::Aggregate,
            vec![Input::PriorOutput],
            CallPattern::PerCompaction,
        ),
    ]))
}

fn merge_sorted_partitions(
    _node: &GpuMergeSortedPartitions,
    _inputs: &[&Schema],
    seqs: &mut Seqs,
) -> Option<Recipe> {
    Some(Recipe::of(vec![Call::seq(
        seqs.allocate(),
        FbKind::SortPreservingMerge,
        vec![Input::AllLanes],
        CallPattern::AtDone,
    )]))
}

/// The lane count is the one thing a recipe repeats from the plan line, because it is a
/// field of the node the call addresses rather than a fact about this node's output.
fn emit_partitions(
    node: &GpuEmitPartitions,
    _inputs: &[&Schema],
    seqs: &mut Seqs,
) -> Option<Recipe> {
    let lanes = node.kind().layout().expect("an emitter is not a sink").n as u32;
    Some(Recipe::of(vec![Call::seq(
        seqs.allocate(),
        FbKind::Repartition { lanes },
        vec![Input::Batch],
        CallPattern::PerBatch,
    )]))
}

fn cross_join(_node: &GpuCrossJoin, _inputs: &[&Schema], seqs: &mut Seqs) -> Option<Recipe> {
    Some(Recipe::of(vec![Call::seq(
        seqs.allocate(),
        FbKind::CrossJoin,
        vec![Input::BuildSideCopy, Input::Batch],
        CallPattern::PerProbeBatch,
    )]))
}

/// No seq: skip and fetch are frozen per node, and the right bound for a batch is a
/// runtime value no frozen node can carry. Only the two straddling batches are sliced —
/// the rest are forwarded or released where they stand.
fn limit(_node: &GpuLimit, _inputs: &[&Schema], _seqs: &mut Seqs) -> Option<Recipe> {
    Some(Recipe::of(vec![Call::bare(
        AbiSymbol::SliceHandle,
        vec![Input::Batch, Input::RowRange],
        CallPattern::PerStraddlingBatch,
    )]))
}

/// Also no seq, and for the same reason: the row range the sink exports is counted across
/// lanes at run time. A batch outside a root-adjacent interval is released without a call.
fn unload(_node: &GpuUnload, _inputs: &[&Schema], _seqs: &mut Seqs) -> Option<Recipe> {
    Some(Recipe::of(vec![Call::bare(
        AbiSymbol::ResultFromHandle,
        vec![Input::Batch, Input::RowRange],
        CallPattern::PerHandle,
    )]))
}

#[cfg(test)]
mod tests;
