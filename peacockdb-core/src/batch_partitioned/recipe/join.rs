//! The joins, which is where the mapping has structure: a streamed probe cannot know
//! which build rows matched, so the types that owe their build side a row keep the probe
//! keys per batch and answer the question once, at done (#136).
//!
//! Which of the four shapes a join takes is `nodes::join::capability` — the same function
//! the planner and the estimator read, so the recipe cannot claim a pass they did not
//! plan for.

use datafusion::common::JoinType;
use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::generated::gpu_plan_generated::peacock::plan as fb;

use super::super::error::PlanError;
use super::super::expr::Expr;
use super::super::nodes::join::{
    JoinFilterColumn, JoinSide, NestedLoopJoinType, finish_join_type, per_call_join_type,
};
use super::super::nodes::{GpuJoin, GpuNestedLoopJoin};
use super::super::schema::Schema;
use super::expr_writer::write_expr;
use super::node_writer;
use super::types::{Call, CallPattern, FbKind, Input, ProjectRole, Recipe};
use super::writer::{Payload, Writer};

pub(super) fn hash_join(
    node: &GpuJoin,
    inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let capability = node
        .capability()
        .expect("a planned join has a capability — the planner refuses the shapes without one");
    let [build, probe] = inputs else {
        unreachable!("a join declares two inputs")
    };

    if capability.answers_in_one_call() {
        // One node over the whole probe side, or one per batch where every emitted row is
        // decided by (build, this batch). The handles differ — a single-batch probe hands
        // the build over, a streamed one copies it (#152) — and the plan does not.
        let handed_over = !capability.probe_streams;
        let join_type = node.join_type;
        let seq = writer.node(2, |b, kids| {
            probe_join(b, node, join_type, build, probe, kids)
        })?;
        return Ok(Some(Recipe::of(vec![Call::seq(
            seq,
            FbKind::HashJoin { join_type },
            vec![
                if handed_over {
                    Input::BuildSide
                } else {
                    Input::BuildSideCopy
                },
                Input::Batch,
            ],
            CallPattern::PerProbeBatch,
        )])));
    }

    let mut calls = Vec::new();
    // The build-side semi family makes no per-call join at all: its probe call is the key
    // project alone, so the build side is untouched until the finish consumes it. Every
    // other finishing type still emits this batch's matches, and the join below consumes
    // the batch — hence the copy the keys come off.
    let per_call = per_call_join_type(node.join_type);
    let keys = writer.node(1, |b, kids| key_project(b, node, probe, kids))?;
    calls.push(Call::seq(
        keys,
        FbKind::Project(ProjectRole::ProbeKeys),
        vec![if per_call.is_some() {
            Input::BatchCopy
        } else {
            Input::Batch
        }],
        CallPattern::PerProbeBatch,
    ));
    if let Some(join_type) = per_call {
        let seq = writer.node(2, |b, kids| {
            probe_join(b, node, join_type, build, probe, kids)
        })?;
        calls.push(Call::seq(
            seq,
            FbKind::HashJoin { join_type },
            vec![Input::BuildSideCopy, Input::Batch],
            CallPattern::PerProbeBatch,
        ));
    }

    let concat = writer.node(1, |b, kids| Ok(node_writer::coalesce_partitions(b, kids)))?;
    calls.push(Call::seq(
        concat,
        FbKind::CoalescePartitions,
        vec![Input::AccumulatedKeys],
        CallPattern::AtDone,
    ));
    let finish_type = finish_join_type(node.join_type);
    let finish = writer.node(2, |b, kids| {
        finish_join(b, node, finish_type, build, probe, kids)
    })?;
    calls.push(Call::seq(
        finish,
        FbKind::HashJoin {
            join_type: finish_type,
        },
        vec![Input::BuildSide, Input::PriorOutput],
        CallPattern::AtDone,
    ));
    if per_call.is_some() {
        let nulls = padded_columns(node, build, probe);
        let seq = writer.node(1, |b, kids| pad_project(b, node, build, probe, kids))?;
        calls.push(Call::seq(
            seq,
            FbKind::Project(ProjectRole::NullPad { nulls }),
            vec![Input::PriorOutput],
            CallPattern::AtDone,
        ));
    }
    Ok(Some(Recipe::of(calls)))
}

/// The join a probe batch runs: the node's own keys and residual, and the type the call
/// emits — the node's for a probe-local join, the per-call one for an outer.
fn probe_join<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuJoin,
    join_type: JoinType,
    build: &Schema,
    probe: &Schema,
    kids: &[WIPOffset<fb::PlanNode<'a>>],
) -> Result<Payload, PlanError> {
    let mut keys = Vec::with_capacity(node.keys.len());
    for (build_ordinal, probe_ordinal) in &node.keys {
        let left = column(b, *build_ordinal, build)?;
        let right = column(b, *probe_ordinal, probe)?;
        keys.push(fb::JoinKey::create(
            b,
            &fb::JoinKeyArgs {
                left: Some(left),
                right: Some(right),
            },
        ));
    }
    let keys = b.create_vector(&keys);
    let filter = match &node.filter {
        Some(expr) => Some(write_expr(b, expr)?),
        None => None,
    };
    let filter_columns = (!node.filter_columns.is_empty()).then(|| {
        let columns: Vec<fb::JoinFilterColumn> =
            node.filter_columns.iter().map(filter_column).collect();
        b.create_vector(&columns)
    });
    let projection = node
        .projection
        .as_ref()
        .map(|columns| b.create_vector(columns));
    let join = fb::CudfHashJoin::create(
        b,
        &fb::CudfHashJoinArgs {
            join_type: wire_join_type(join_type),
            keys: Some(keys),
            filter,
            filter_columns,
            left: Some(kids[0]),
            right: Some(kids[1]),
            projection,
            null_equals_null: node.null_equals_null,
        },
    );
    Ok(Payload {
        kind: fb::PlanNodeKind::CudfHashJoin,
        value: join.as_union_value(),
    })
}

/// The finish join, against the accumulated probe keys: their ordinals are `0..k`, since
/// the key project emitted the key columns and nothing else. No residual and no
/// projection — a build-preserving type asks which build rows matched, and the pad
/// project is what makes the output the joined schema.
fn finish_join<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuJoin,
    join_type: JoinType,
    build: &Schema,
    probe: &Schema,
    kids: &[WIPOffset<fb::PlanNode<'a>>],
) -> Result<Payload, PlanError> {
    let mut keys = Vec::with_capacity(node.keys.len());
    for (position, (build_ordinal, probe_ordinal)) in node.keys.iter().enumerate() {
        let left = column(b, *build_ordinal, build)?;
        // The name the key project gave that column, which is the probe's: this join is
        // its only reader, and two names for one column is how the two sides drift.
        let name = node_writer::field_at(probe, *probe_ordinal).name().clone();
        let right = write_expr(b, &Expr::column(position as u32, &name))?;
        keys.push(fb::JoinKey::create(
            b,
            &fb::JoinKeyArgs {
                left: Some(left),
                right: Some(right),
            },
        ));
    }
    let keys = b.create_vector(&keys);
    let join = fb::CudfHashJoin::create(
        b,
        &fb::CudfHashJoinArgs {
            join_type: wire_join_type(join_type),
            keys: Some(keys),
            left: Some(kids[0]),
            right: Some(kids[1]),
            null_equals_null: node.null_equals_null,
            ..Default::default()
        },
    );
    Ok(Payload {
        kind: fb::PlanNodeKind::CudfHashJoin,
        value: join.as_union_value(),
    })
}

/// This batch's contribution to the accumulation: its key columns and nothing else, which
/// is what makes the concat at done cheap (#136).
fn key_project<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuJoin,
    probe: &Schema,
    kids: &[WIPOffset<fb::PlanNode<'a>>],
) -> Result<Payload, PlanError> {
    let mut exprs = Vec::with_capacity(node.keys.len());
    let mut names = Vec::with_capacity(node.keys.len());
    for (_, probe_ordinal) in &node.keys {
        exprs.push(column(b, *probe_ordinal, probe)?);
        names.push(b.create_string(node_writer::field_at(probe, *probe_ordinal).name()));
    }
    Ok(node_writer::project_payload(b, exprs, names, kids[0]))
}

/// The node's declared row, out of an anti join that emitted build columns only: each
/// column the projection keeps, in its order — a build one read where the anti join left
/// it, a probe one as a typed NULL. Walking the projection rather than the build side is
/// what keeps this the node's shape: the anti join emits every build column whatever the
/// projection says.
fn pad_project<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuJoin,
    build: &Schema,
    probe: &Schema,
    kids: &[WIPOffset<fb::PlanNode<'a>>],
) -> Result<Payload, PlanError> {
    let build_width = build.fields.fields().len() as u32;
    let kept: Vec<u32> = match &node.projection {
        Some(columns) => columns.clone(),
        None => (0..build_width + probe.fields.fields().len() as u32).collect(),
    };
    let mut exprs = Vec::with_capacity(kept.len());
    let mut names = Vec::with_capacity(kept.len());
    for ordinal in kept {
        if ordinal < build_width {
            exprs.push(column(b, ordinal, build)?);
            names.push(b.create_string(node_writer::field_at(build, ordinal).name()));
            continue;
        }
        let field = node_writer::field_at(probe, ordinal - build_width);
        exprs.push(node_writer::null_literal(b, field)?);
        names.push(b.create_string(field.name()));
    }
    Ok(node_writer::project_payload(b, exprs, names, kids[0]))
}

pub(super) fn cross_join_payload<'a>(
    b: &mut FlatBufferBuilder<'a>,
    kids: &[WIPOffset<fb::PlanNode<'a>>],
) -> Payload {
    let cross = fb::CudfCrossJoin::create(
        b,
        &fb::CudfCrossJoinArgs {
            left: Some(kids[0]),
            right: Some(kids[1]),
        },
    );
    Payload {
        kind: fb::PlanNodeKind::CudfCrossJoin,
        value: cross.as_union_value(),
    }
}

/// Inner streams; Left takes a single-batch probe, since the finish trick accumulates
/// keys and a predicate join has none. So Left's one call may consume the build outright.
pub(super) fn nested_loop_join(
    node: &GpuNestedLoopJoin,
    _inputs: &[&Schema],
    writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError> {
    let build = match node.join_type {
        NestedLoopJoinType::Inner => Input::BuildSideCopy,
        NestedLoopJoinType::Left => Input::BuildSide,
    };
    let seq = writer.node(2, |b, kids| {
        let filter = write_expr(b, &node.filter)?;
        let columns: Vec<fb::JoinFilterColumn> =
            node.filter_columns.iter().map(filter_column).collect();
        let columns = b.create_vector(&columns);
        let projection = node
            .projection
            .as_ref()
            .map(|columns| b.create_vector(columns));
        let join = fb::CudfNestedLoopJoin::create(
            b,
            &fb::CudfNestedLoopJoinArgs {
                join_type: match node.join_type {
                    NestedLoopJoinType::Inner => fb::JoinType::Inner,
                    NestedLoopJoinType::Left => fb::JoinType::Left,
                },
                filter: Some(filter),
                filter_columns: Some(columns),
                left: Some(kids[0]),
                right: Some(kids[1]),
                projection,
            },
        );
        Ok(Payload {
            kind: fb::PlanNodeKind::CudfNestedLoopJoin,
            value: join.as_union_value(),
        })
    })?;
    Ok(Some(Recipe::of(vec![Call::seq(
        seq,
        FbKind::NestedLoopJoin,
        vec![build, Input::Batch],
        CallPattern::PerProbeBatch,
    )])))
}

fn column<'a>(
    b: &mut FlatBufferBuilder<'a>,
    ordinal: u32,
    schema: &Schema,
) -> Result<WIPOffset<fb::Expr<'a>>, PlanError> {
    let field = node_writer::field_at(schema, ordinal);
    write_expr(b, &Expr::column(ordinal, field.name()))
}

fn filter_column(column: &JoinFilterColumn) -> fb::JoinFilterColumn {
    fb::JoinFilterColumn::new(
        column.index,
        match column.side {
            JoinSide::Build => fb::JoinSide::Left,
            JoinSide::Probe => fb::JoinSide::Right,
        },
    )
}

fn wire_join_type(join_type: JoinType) -> fb::JoinType {
    match join_type {
        JoinType::Inner => fb::JoinType::Inner,
        JoinType::Left => fb::JoinType::Left,
        JoinType::Right => fb::JoinType::Right,
        JoinType::Full => fb::JoinType::Full,
        JoinType::LeftSemi => fb::JoinType::LeftSemi,
        JoinType::RightSemi => fb::JoinType::RightSemi,
        JoinType::LeftAnti => fb::JoinType::LeftAnti,
        JoinType::RightAnti => fb::JoinType::RightAnti,
        JoinType::LeftMark => fb::JoinType::LeftMark,
    }
}

/// How many typed NULLs the pad project appends: the probe columns the join's projection
/// keeps.
fn padded_columns(node: &GpuJoin, build: &Schema, probe: &Schema) -> usize {
    let build_width = build.fields.fields().len() as u32;
    match &node.projection {
        Some(kept) => kept.iter().filter(|column| **column >= build_width).count(),
        None => probe.fields.fields().len(),
    }
}
