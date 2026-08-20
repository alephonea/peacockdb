//! The joins, which is where the mapping has structure: a streamed probe cannot know
//! which build rows matched, so the types that owe their build side a row keep the probe
//! keys per batch and answer the question once, at done (#136).
//!
//! Which of the four shapes a join takes is `nodes::join::capability` — the same function
//! the planner and the estimator read, so the recipe cannot claim a pass they did not
//! plan for.

use datafusion::common::JoinType;

use super::super::nodes::join::NestedLoopJoinType;
use super::super::nodes::{GpuJoin, GpuNestedLoopJoin};
use super::super::schema::Schema;
use super::types::{Call, CallPattern, FbKind, Input, ProjectRole, Recipe, Seqs};

pub(super) fn hash_join(node: &GpuJoin, inputs: &[&Schema], seqs: &mut Seqs) -> Option<Recipe> {
    let capability = node
        .capability()
        .expect("a planned join has a capability — the planner refuses the shapes without one");

    if !capability.probe_streams {
        // The legacy call: one node over a probe side the planner made single-batch, so
        // the build handle is handed over rather than copied.
        return Some(Recipe::of(vec![Call::seq(
            seqs.allocate(),
            FbKind::HashJoin {
                join_type: node.join_type,
            },
            vec![Input::BuildSide, Input::Batch],
            CallPattern::PerProbeBatch,
        )]));
    }
    if !capability.needs_finish {
        return Some(Recipe::of(vec![Call::seq(
            seqs.allocate(),
            FbKind::HashJoin {
                join_type: node.join_type,
            },
            vec![Input::BuildSideCopy, Input::Batch],
            CallPattern::PerProbeBatch,
        )]));
    }

    let mut calls = Vec::new();
    // The build-side semi family makes no per-call join at all: its probe call is the key
    // project alone, so the build side is untouched until the finish consumes it. Every
    // other finishing type still emits this batch's matches, and the join below consumes
    // the batch — hence the copy the keys come off.
    let per_call = per_call_join_type(node.join_type);
    calls.push(Call::seq(
        seqs.allocate(),
        FbKind::Project(ProjectRole::ProbeKeys),
        vec![if per_call.is_some() {
            Input::BatchCopy
        } else {
            Input::Batch
        }],
        CallPattern::PerProbeBatch,
    ));
    if let Some(join_type) = per_call {
        calls.push(Call::seq(
            seqs.allocate(),
            FbKind::HashJoin { join_type },
            vec![Input::BuildSideCopy, Input::Batch],
            CallPattern::PerProbeBatch,
        ));
    }

    calls.push(Call::seq(
        seqs.allocate(),
        FbKind::CoalescePartitions,
        vec![Input::AccumulatedKeys],
        CallPattern::AtDone,
    ));
    calls.push(Call::seq(
        seqs.allocate(),
        FbKind::HashJoin {
            join_type: finish_join_type(node.join_type),
        },
        vec![Input::BuildSide, Input::PriorOutput],
        CallPattern::AtDone,
    ));
    if per_call.is_some() {
        calls.push(Call::seq(
            seqs.allocate(),
            FbKind::Project(ProjectRole::NullPad {
                nulls: padded_columns(node, inputs),
            }),
            vec![Input::PriorOutput],
            CallPattern::AtDone,
        ));
    }
    Some(Recipe::of(calls))
}

/// What the per-probe-batch join emits, which is not what the node is: a Left emits this
/// batch's matches and waits for the finish, and a Full also emits the probe rows this
/// batch had no match for — batch-local, because the build side was complete before the
/// first call.
///
/// `None` is the build-side semi family, whose probe call is only the key project.
fn per_call_join_type(join_type: JoinType) -> Option<JoinType> {
    match join_type {
        JoinType::Left => Some(JoinType::Inner),
        JoinType::Full => Some(JoinType::Right),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => None,
        other => unreachable!("{other:?} needs no finish pass"),
    }
}

/// The join the finish pass runs against the accumulated keys. Left and Full ask which
/// build rows nothing ever matched; the semi family asks its own question, and asks it
/// with the node's own NULL semantics, so the pass substitutes for a legacy single call
/// rather than improving on it (#59, #80).
fn finish_join_type(join_type: JoinType) -> JoinType {
    match join_type {
        JoinType::Left | JoinType::Full => JoinType::LeftAnti,
        semi @ (JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark) => semi,
        other => unreachable!("{other:?} needs no finish pass"),
    }
}

/// How many typed NULLs the pad project appends: the probe columns the join's projection
/// keeps. The anti join emits build columns only, and the node's output is the joined
/// schema, so the difference is exactly those.
fn padded_columns(node: &GpuJoin, inputs: &[&Schema]) -> usize {
    let [build, probe] = inputs else {
        unreachable!("a join declares two inputs")
    };
    let build_width = build.fields.fields().len() as u32;
    match &node.projection {
        Some(kept) => kept.iter().filter(|column| **column >= build_width).count(),
        None => probe.fields.fields().len(),
    }
}

/// Inner streams; Left takes a single-batch probe, since the finish trick accumulates keys
/// and a predicate join has none. So Left's one call may consume the build outright.
pub(super) fn nested_loop_join(
    node: &GpuNestedLoopJoin,
    _inputs: &[&Schema],
    seqs: &mut Seqs,
) -> Option<Recipe> {
    let build = match node.join_type {
        NestedLoopJoinType::Inner => Input::BuildSideCopy,
        NestedLoopJoinType::Left => Input::BuildSide,
    };
    Some(Recipe::of(vec![Call::seq(
        seqs.allocate(),
        FbKind::NestedLoopJoin,
        vec![build, Input::Batch],
        CallPattern::PerProbeBatch,
    )]))
}
