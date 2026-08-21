//! The recipe plan: the buffer the C++ is handed, and per node the ABI calls it drives.
//!
//! The C++ side never sees this mode's tree. It is sent a plan in the legacy vocabulary
//! whose nodes exist to be addressed — a menu of parameterized kernels — and the driver
//! then calls the ABI as often as its own schedule wants. One walk builds that buffer and
//! records, per plan node, the calls it makes and the seqs they address;
//! `llm-wiki/tasks/batch_partitioned_executor.md` has the table it implements.
//!
//! A node that makes no ABI call gets no recipe, and the absence is a statement about the
//! node rather than a gap: a forwarder routes batches and touches no device.

mod aggregate_writer;
mod expr_writer;
mod join;
mod node_writer;
mod types;
mod writer;

use super::error::PlanError;
use super::node::GpuNode;
use super::nodes::{
    GpuAccumulateBatchesAndSort, GpuAggregate, GpuAggregateBatches, GpuCoalesceAllBatches,
    GpuCrossJoin, GpuEmitPartitions, GpuFilter, GpuLimit, GpuLoadParquet, GpuMergeSortedPartitions,
    GpuProject, GpuSort, GpuUnload, NodeRef, as_node_ref,
};
use super::schema::Schema;
use aggregate_writer::Phase;
use writer::Writer;

pub use types::{AbiSymbol, Call, CallPattern, FbKind, Input, ProjectRole, Recipe, Seq};

/// Every node's recipe, indexed by the post-order position the estimator and the memory
/// golden already number by, plus the bytes those recipes address.
///
/// That index is a position in the TREE and a [`Seq`] is an address in the recipe plan:
/// they part company at the first node with two calls, and a driver holding both at once
/// has to keep them apart.
#[derive(Debug, Default)]
pub struct RecipePlan {
    recipes: Vec<Option<Recipe>>,
    bytes: Vec<u8>,
    unwritable: Vec<(Seq, String)>,
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

    /// The serialized recipe plan: what `peacock_executor_begin_plan` is given, and what
    /// every seq in every recipe indexes into. Check [`RecipePlan::unwritable`] before
    /// running it — a plan with a payload the wire could not carry is structurally whole
    /// and semantically short of one node.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Why the payload at a seq is missing, where one is. Empty for every plan whose
    /// expressions the wire can carry, which is all but #168's today.
    pub fn unwritable(&self) -> &[(Seq, String)] {
        &self.unwritable
    }

    /// What that seq could not write, for the payload rendering.
    pub fn unwritable_at(&self, seq: Seq) -> Option<&str> {
        self.unwritable
            .iter()
            .find(|(at, _)| *at == seq)
            .map(|(_, why)| why.as_str())
    }
}

/// Build the recipe plan for a finished tree. It runs after planning because a recipe is
/// a statement about a node, and a node is not finished until the plan is.
pub fn attach_recipes(root: &dyn GpuNode) -> Result<RecipePlan, PlanError> {
    let mut writer = Writer::new();
    let mut recipes = Vec::new();
    walk(root, &mut writer, &mut recipes)?;
    let (bytes, unwritable) = writer.finish()?;
    Ok(RecipePlan {
        recipes,
        bytes,
        unwritable,
    })
}

fn walk(
    node: &dyn GpuNode,
    writer: &mut Writer,
    recipes: &mut Vec<Option<Recipe>>,
) -> Result<(), PlanError> {
    let mark = writer.mark();
    for child in node.children() {
        walk(child, writer, recipes)?;
    }
    let inputs = input_schemas(node);
    let recipe = emit(node, inputs.as_slice(), writer)?;
    recipes.push(recipe);
    // Whatever this node did not consume is gathered here, so nothing it was handed
    // becomes unreachable: an orphan is never indexed and every seq above it would shift.
    writer.reduce(mark)
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
/// schemas above, and nothing else.
///
/// Seqs ascend with the order the nodes are built, which is the order the C++ indexes
/// them — `writer.rs` has the three rules that keep those the same sequence.
fn emit(
    node: &dyn GpuNode,
    inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    match as_node_ref(node) {
        NodeRef::LoadParquet(load) => scan(load, node, writer),
        NodeRef::Filter(filter_node) => filter(filter_node, inputs, writer),
        NodeRef::Project(project_node) => project(project_node, inputs, writer),
        NodeRef::Sort(sort_node) => sort(sort_node, inputs, writer),
        NodeRef::Aggregate(aggregate_node) => aggregate(aggregate_node, inputs, writer),
        NodeRef::AccumulateBatchesAndSort(accumulator) => {
            accumulate_and_sort(accumulator, inputs, writer)
        }
        NodeRef::CoalesceAllBatches(coalesce) => coalesce_all_batches(coalesce, inputs, writer),
        NodeRef::AggregateBatches(merge) => aggregate_batches(merge, inputs, writer),
        NodeRef::MergeSortedPartitions(merge) => merge_sorted_partitions(merge, inputs, writer),
        NodeRef::EmitPartitions(emitter) => emit_partitions(emitter, node, inputs, writer),
        NodeRef::Join(join_node) => join::hash_join(join_node, inputs, writer),
        NodeRef::CrossJoin(cross) => cross_join(cross, inputs, writer),
        NodeRef::NestedLoopJoin(nested) => join::nested_loop_join(nested, inputs, writer),
        NodeRef::Limit(limit_node) => limit(limit_node, inputs, writer),
        NodeRef::Unload(unload_node) => unload(unload_node, inputs, writer),
        // The three that route rather than compute: not one call between them.
        NodeRef::MergePartitions(_) | NodeRef::Union(_) | NodeRef::Interleave(_) => Ok(None),
    }
}

/// The additive entry point, and the reason it exists: the node's own row-group list is
/// overridden per call, so one seq serves every batch the mapping cut the scan into.
fn scan(
    load: &GpuLoadParquet,
    node: &dyn GpuNode,
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let output = node.kind().schema().expect("a source declares its columns");
    let seq = writer.node(0, |b, _| node_writer::scan(b, load, output))?;
    Ok(Some(Recipe::of(vec![Call::seq(
        seq,
        FbKind::Scan,
        vec![Input::RowGroups],
        CallPattern::PerBatch,
    )])))
}

fn filter(
    node: &GpuFilter,
    _inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let seq = writer.node(1, |b, kids| node_writer::filter(b, node, kids))?;
    Ok(Some(Recipe::of(vec![Call::seq(
        seq,
        FbKind::Filter,
        vec![Input::Batch],
        CallPattern::PerBatch,
    )])))
}

fn project(
    node: &GpuProject,
    _inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let seq = writer.node(1, |b, kids| node_writer::project(b, node, kids))?;
    Ok(Some(Recipe::of(vec![Call::seq(
        seq,
        FbKind::PlainProject,
        vec![Input::Batch],
        CallPattern::PerBatch,
    )])))
}

/// The map arm; a top-N's `fetch` rides the node, so it trims each batch and whatever
/// accumulates above it trims the merged result.
fn sort(
    node: &GpuSort,
    inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let input = inputs[0];
    let seq = writer.node(1, |b, kids| node_writer::sort(b, node, input, kids))?;
    Ok(Some(Recipe::of(vec![Call::seq(
        seq,
        FbKind::Sort,
        vec![Input::Batch],
        CallPattern::PerBatch,
    )])))
}

/// State from raw values, per batch.
fn aggregate(
    aggregate_node: &GpuAggregate,
    inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let input = inputs[0];
    let body = &aggregate_node.body;
    let state = aggregate_node.intermediate();
    let seq = writer.node(1, |b, kids| {
        aggregate_writer::aggregate(b, body, Phase::Init, input, state, kids)
    })?;
    Ok(Some(Recipe::of(vec![Call::seq(
        seq,
        FbKind::Aggregate { merge: false },
        vec![Input::Batch],
        CallPattern::PerBatch,
    )])))
}

/// Sorted as the batches arrive, merged once at done: the merge arm takes whatever k
/// handles the call hands it, so the lane's batch count is a runtime number.
fn accumulate_and_sort(
    node: &GpuAccumulateBatchesAndSort,
    inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let input = inputs[0];
    let per_batch = writer.node(1, |b, kids| {
        node_writer::accumulating_sort(b, node, input, kids)
    })?;
    let at_done = writer.node(1, |b, kids| {
        node_writer::merge_sorted(b, &node.keys, node.fetch, input, kids)
    })?;
    Ok(Some(Recipe::of(vec![
        Call::seq(
            per_batch,
            FbKind::Sort,
            vec![Input::Batch],
            CallPattern::PerBatch,
        ),
        Call::seq(
            at_done,
            FbKind::SortPreservingMerge,
            vec![Input::LaneBatches],
            CallPattern::AtDone,
        ),
    ])))
}

fn coalesce_all_batches(
    _node: &GpuCoalesceAllBatches,
    _inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let seq = writer.node(1, |b, kids| Ok(node_writer::coalesce_partitions(b, kids)))?;
    Ok(Some(Recipe::of(vec![Call::seq(
        seq,
        FbKind::CoalescePartitions,
        vec![Input::LaneBatches],
        CallPattern::AtDone,
    )])))
}

/// A compaction runs exactly what done runs, which is what makes the doubling threshold a
/// scheduling decision rather than a second computation. Where it finalizes, the finalize
/// rides a project of its own — ours, so both engines evaluate the same expression.
fn aggregate_batches(
    merge: &GpuAggregateBatches,
    inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let input = inputs[0];
    let body = &merge.body;
    let state = merge.intermediate();
    let concat = writer.node(1, |b, kids| Ok(node_writer::coalesce_partitions(b, kids)))?;
    let merged = writer.node(1, |b, kids| {
        aggregate_writer::aggregate(b, body, Phase::Merge, input, state, kids)
    })?;
    let mut calls = vec![
        Call::seq(
            concat,
            FbKind::CoalescePartitions,
            vec![Input::LaneBatches],
            CallPattern::PerCompaction,
        ),
        Call::seq(
            merged,
            FbKind::Aggregate { merge: true },
            vec![Input::PriorOutput],
            CallPattern::PerCompaction,
        ),
    ];
    if let Some(finalize) = &body.finalize {
        let seq = writer.node(1, |b, kids| {
            let mut exprs = Vec::with_capacity(finalize.len());
            for column in finalize {
                exprs.push(expr_writer::write_expr(b, &column.expr)?);
            }
            let names = finalize
                .iter()
                .map(|column| b.create_string(&column.name))
                .collect();
            Ok(node_writer::project_payload(b, exprs, names, kids[0]))
        })?;
        calls.push(Call::seq(
            seq,
            FbKind::Project(ProjectRole::Finalize),
            vec![Input::PriorOutput],
            CallPattern::AtDone,
        ));
    }
    Ok(Some(Recipe::of(calls)))
}

fn merge_sorted_partitions(
    node: &GpuMergeSortedPartitions,
    inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let input = inputs[0];
    let seq = writer.node(1, |b, kids| {
        node_writer::merge_partitions(b, node, input, kids)
    })?;
    Ok(Some(Recipe::of(vec![Call::seq(
        seq,
        FbKind::SortPreservingMerge,
        vec![Input::AllLanes],
        CallPattern::AtDone,
    )])))
}

/// The lane count is the one thing a recipe repeats from the plan line, because it is a
/// field of the node the call addresses rather than a fact about this node's output.
fn emit_partitions(
    emitter: &GpuEmitPartitions,
    node: &dyn GpuNode,
    inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let input = inputs[0];
    let lanes = node.kind().layout().expect("an emitter is not a sink").n as u32;
    let seq = writer.node(1, |b, kids| {
        node_writer::repartition(b, emitter, input, lanes, kids)
    })?;
    Ok(Some(Recipe::of(vec![Call::seq(
        seq,
        FbKind::Repartition { lanes },
        vec![Input::Batch],
        CallPattern::PerBatch,
    )])))
}

fn cross_join(
    _node: &GpuCrossJoin,
    _inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let seq = writer.node(2, |b, kids| Ok(join::cross_join_payload(b, kids)))?;
    Ok(Some(Recipe::of(vec![Call::seq(
        seq,
        FbKind::CrossJoin,
        vec![Input::BuildSideCopy, Input::Batch],
        CallPattern::PerProbeBatch,
    )])))
}

/// No seq: skip and fetch are frozen per node, and the right bound for a batch is a
/// runtime value no frozen node can carry. Only the two straddling batches are sliced —
/// the rest are forwarded or released where they stand.
fn limit(
    _node: &GpuLimit,
    _inputs: &[&Schema],
    _writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    Ok(Some(Recipe::of(vec![Call::bare(
        AbiSymbol::SliceHandle,
        vec![Input::Batch, Input::RowRange],
        CallPattern::PerStraddlingBatch,
    )])))
}

/// Also no seq, and for the same reason: the row range the sink exports is counted across
/// lanes at run time. A batch outside a root-adjacent interval is released without a call.
fn unload(
    _node: &GpuUnload,
    _inputs: &[&Schema],
    _writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    Ok(Some(Recipe::of(vec![Call::bare(
        AbiSymbol::ResultFromHandle,
        vec![Input::Batch, Input::RowRange],
        CallPattern::PerHandle,
    )])))
}

#[cfg(test)]
mod tests;
