//! Reading the recipe plan back: the node at a seq, by the same post-order the C++
//! indexes with.
//!
//! Beside the writer rather than beside the renderer, because it is the other half of one
//! claim — a seq means the same node on both sides — and the check that it holds
//! ([`check_seq_kinds`]) is not a rendering concern.

use crate::generated::gpu_plan_generated::peacock::plan as fb;

use super::super::error::PlanError;
use super::RecipePlan;

/// The node at `seq`, by the post-order the C++ indexes with — children in
/// `node_children` order, then the node.
pub fn node_at<'a>(plan: &fb::GpuPlan<'a>, seq: u32) -> Option<fb::PlanNode<'a>> {
    let mut position = 0;
    let root = plan.root()?;
    find(root, seq, &mut position)
}

fn find<'a>(node: fb::PlanNode<'a>, seq: u32, position: &mut u32) -> Option<fb::PlanNode<'a>> {
    for child in children(&node) {
        if let Some(found) = find(child, seq, position) {
            return Some(found);
        }
    }
    let at = *position;
    *position += 1;
    (at == seq).then_some(node)
}

/// The child order `NodeSession::node_children` walks, which is what makes a seq mean the
/// same node on both sides.
fn children<'a>(node: &fb::PlanNode<'a>) -> Vec<fb::PlanNode<'a>> {
    match node.node_type() {
        fb::PlanNodeKind::CudfFilter => one(node.node_as_cudf_filter().and_then(|n| n.input())),
        fb::PlanNodeKind::CudfProject => one(node.node_as_cudf_project().and_then(|n| n.input())),
        fb::PlanNodeKind::CudfAggregate => {
            one(node.node_as_cudf_aggregate().and_then(|n| n.input()))
        }
        fb::PlanNodeKind::CudfSort => one(node.node_as_cudf_sort().and_then(|n| n.input())),
        fb::PlanNodeKind::CudfCoalesceBatches => {
            one(node.node_as_cudf_coalesce_batches().and_then(|n| n.input()))
        }
        fb::PlanNodeKind::CudfCoalescePartitions => one(node
            .node_as_cudf_coalesce_partitions()
            .and_then(|n| n.input())),
        fb::PlanNodeKind::CudfRepartition => {
            one(node.node_as_cudf_repartition().and_then(|n| n.input()))
        }
        fb::PlanNodeKind::CudfSortPreservingMerge => one(node
            .node_as_cudf_sort_preserving_merge()
            .and_then(|n| n.input())),
        fb::PlanNodeKind::CudfLimit => one(node.node_as_cudf_limit().and_then(|n| n.input())),
        fb::PlanNodeKind::CudfWindow => one(node.node_as_cudf_window().and_then(|n| n.input())),
        fb::PlanNodeKind::CudfHashJoin => node
            .node_as_cudf_hash_join()
            .map(|n| pair(n.left(), n.right()))
            .unwrap_or_default(),
        fb::PlanNodeKind::CudfCrossJoin => node
            .node_as_cudf_cross_join()
            .map(|n| pair(n.left(), n.right()))
            .unwrap_or_default(),
        fb::PlanNodeKind::CudfNestedLoopJoin => node
            .node_as_cudf_nested_loop_join()
            .map(|n| pair(n.left(), n.right()))
            .unwrap_or_default(),
        fb::PlanNodeKind::CudfUnion => node
            .node_as_cudf_union()
            .and_then(|n| n.inputs())
            .map(|inputs| inputs.iter().collect())
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn one<'a>(child: Option<fb::PlanNode<'a>>) -> Vec<fb::PlanNode<'a>> {
    child.into_iter().collect()
}

fn pair<'a>(
    left: Option<fb::PlanNode<'a>>,
    right: Option<fb::PlanNode<'a>>,
) -> Vec<fb::PlanNode<'a>> {
    left.into_iter().chain(right).collect()
}


/// Every published seq resolves to a node, and to a node of the kind its recipe claims.
///
/// The two halves are separate failures. A seq that resolves to nothing means the tree is
/// shorter than the numbering — something the walk built was left unreachable, and every
/// seq above the gap addresses the wrong node. A seq that resolves to the wrong kind means
/// the claim and the buffer disagree about what a call runs, which no golden would catch:
/// both are rendered from the same recipe.
pub fn check_seq_kinds(plan: &RecipePlan) -> Result<(), PlanError> {
    // The depth the C++ verifier allows, since a recipe plan is a chain and is as deep as
    // it is long (#169).
    let options = flatbuffers::VerifierOptions {
        max_depth: 1024,
        ..Default::default()
    };
    let buffer = flatbuffers::root_with_opts::<fb::GpuPlan>(&options, plan.bytes())
        .map_err(|e| PlanError::Invalid(format!("the recipe plan does not verify: {e}")))?;
    for node in 0..plan.nodes() {
        let Some(recipe) = plan.get(node) else {
            continue;
        };
        for call in &recipe.calls {
            let Some((seq, kind)) = call.target else {
                continue;
            };
            let Some(written) = node_at(&buffer, seq) else {
                return Err(PlanError::Invalid(format!(
                    "seq #{seq} is published as {kind} and the plan has no node there — the \
                     numbering is longer than the tree, so every seq above it addresses the \
                     wrong node"
                )));
            };
            if written.node_type() != kind.wire_kind() {
                return Err(PlanError::Invalid(format!(
                    "seq #{seq} is published as {kind} and holds {:?}",
                    written.node_type()
                )));
            }
        }
    }
    Ok(())
}

/// How deep the recipe plan is. A chain of nodes is as deep as it is long, and the C++
/// verifier refuses a plan past its depth limit at `begin_plan` — the whole query, before
/// any call, which is #169. Read here rather than guessed so a test can watch the headroom.
pub fn depth(plan: &RecipePlan) -> Result<usize, PlanError> {
    let options = flatbuffers::VerifierOptions {
        max_depth: 1024,
        ..Default::default()
    };
    let buffer = flatbuffers::root_with_opts::<fb::GpuPlan>(&options, plan.bytes())
        .map_err(|e| PlanError::Invalid(format!("the recipe plan does not verify: {e}")))?;
    let root = buffer
        .root()
        .ok_or_else(|| PlanError::Invalid("the recipe plan has no root".to_string()))?;
    Ok(depth_of(root))
}

fn depth_of(node: fb::PlanNode<'_>) -> usize {
    1 + children(&node).into_iter().map(depth_of).max().unwrap_or(0)
}
