//! The aggregate payload, which is the one kind whose vocabulary differs on the two
//! sides.
//!
//! We declare aggregators — `count`, `mean`, `m2`, `merge_m2` — and the wire declares SQL
//! aggregates plus a mode, leaving the state layout to the executor. Four of ours map by
//! name; the Welford triple does not, since no `m2` name exists there. It is rebuilt into
//! the one SQL aggregate it decomposes, which is what the schema's `agg_state`
//! annotations are for: they say which output columns belong to which aggregate, and with
//! what `ddof`. The same reconstruction serves the merge, where `merge_m2` is one call
//! against three state columns and the wire spells it as the SQL aggregate plus `Merge`.

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::generated::gpu_plan_generated::peacock::plan as fb;
use crate::plan_serializer::serialize_schema;

use super::super::aggregates::{AggCall, AggFunc, PlanAgg};
use super::super::error::PlanError;
use super::super::nodes::aggregate::AggregateBody;
use super::super::schema::Schema;
use super::expr_writer::write_expr;
use super::writer::Payload;

/// Which side of the decomposition a node runs: state from raw values, or state merged
/// from state. `Partial` and `Merge` on the wire — never `Final`, which also finalizes,
/// and in this mode a finalize is a project of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Init,
    Merge,
}

pub(super) fn aggregate<'a>(
    b: &mut FlatBufferBuilder<'a>,
    body: &AggregateBody,
    phase: Phase,
    input: &Schema,
    state: &Schema,
    kids: &[WIPOffset<fb::PlanNode<'a>>],
) -> Result<Payload, PlanError> {
    let mut group_exprs = Vec::with_capacity(body.group_by.len());
    for key in &body.group_by {
        group_exprs.push(write_expr(b, key)?);
    }
    let group_names: Vec<WIPOffset<&str>> = state
        .fields
        .fields()
        .iter()
        .take(body.group_by.len())
        .map(|field| b.create_string(field.name()))
        .collect();

    let (funcs, mergeable) = state_funcs(b, body, state)?;

    let group_exprs = b.create_vector(&group_exprs);
    let group_names = b.create_vector(&group_names);
    let funcs = b.create_vector(&funcs);
    let input_schema = serialize_schema(b, &input.fields);
    let aggregate = fb::CudfAggregate::create(
        b,
        &fb::CudfAggregateArgs {
            mode: match phase {
                Phase::Init => fb::AggregateMode::Partial,
                Phase::Merge => fb::AggregateMode::Merge,
            },
            group_exprs: Some(group_exprs),
            group_names: Some(group_names),
            aggr_funcs: Some(funcs),
            input: Some(kids[0]),
            aggr_input_schema: Some(input_schema),
            mergeable_agg_state: mergeable,
            ..Default::default()
        },
    );
    Ok(Payload {
        kind: fb::PlanNodeKind::CudfAggregate,
        value: aggregate.as_union_value(),
    })
}

/// The aggregators as the wire declares them, in the order their state columns appear.
///
/// Order is load-bearing and not cosmetic: the executor reads a state-shaped input
/// positionally, walking a cursor by each aggregate's state width, so an aggregate listed
/// out of order reads another one's columns. Everything but the Welford triple is one of
/// ours to one SQL name; the triple is folded into the `stddev`/`var` it decomposes, at
/// the position of the first of its three, because the wire has no `m2` and the executor
/// produces or merges all three columns under that one name.
fn state_funcs<'a>(
    b: &mut FlatBufferBuilder<'a>,
    body: &AggregateBody,
    state: &Schema,
) -> Result<(Vec<WIPOffset<fb::AggregateFuncNode<'a>>>, bool), PlanError> {
    let welford = welford_owners(body, state);
    let mut funcs = Vec::new();
    let mut emitted: Vec<&str> = Vec::new();

    for (position, call) in body.aggs.iter().enumerate() {
        match welford.get(position).copied().flatten() {
            Some(state) => {
                // The second and third aggregators of a triple add nothing: the SQL name
                // covers all three columns.
                if emitted.contains(&state.output.as_str()) {
                    continue;
                }
                emitted.push(&state.output);
                funcs.push(named_func(
                    b,
                    sql_name(state.func, state.ddof),
                    call,
                    &state.output,
                )?);
            }
            None => {
                let alias = call
                    .outputs
                    .first()
                    .map(|field| field.name().clone())
                    .unwrap_or_default();
                funcs.push(named_func(b, wire_name(call.func)?, call, &alias)?);
            }
        }
    }
    Ok((funcs, !emitted.is_empty()))
}

/// Per aggregator, the Welford state it belongs to, if any.
///
/// An aggregator owns as many state columns as it emits — one each at the init, three for
/// a merge's `merge_m2` — so which aggregator a state position names is a walk over those
/// widths, never `position - group_width`.
fn welford_owners<'a>(
    body: &AggregateBody,
    state: &'a Schema,
) -> Vec<Option<&'a super::super::schema::AggStateColumns>> {
    let mut owners = Vec::new();
    for (position, call) in body.aggs.iter().enumerate() {
        for _ in 0..call.outputs.len().max(1) {
            owners.push(position);
        }
    }
    let group_width = body.group_by.len();
    let mut per_agg = vec![None; body.aggs.len()];
    for columns in &state.agg_state {
        if !matches!(columns.func, AggFunc::Stddev | AggFunc::Var) {
            continue;
        }
        for position in &columns.positions {
            let Some(state_position) = (*position as usize).checked_sub(group_width) else {
                continue;
            };
            if let Some(agg) = owners.get(state_position) {
                per_agg[*agg] = Some(columns);
            }
        }
    }
    per_agg
}

fn named_func<'a>(
    b: &mut FlatBufferBuilder<'a>,
    name: &str,
    call: &AggCall,
    alias: &str,
) -> Result<WIPOffset<fb::AggregateFuncNode<'a>>, PlanError> {
    let mut args = Vec::with_capacity(call.args.len());
    for arg in &call.args {
        args.push(write_expr(b, arg)?);
    }
    let args = b.create_vector(&args);
    let name = b.create_string(name);
    let alias = b.create_string(alias);
    Ok(fb::AggregateFuncNode::create(
        b,
        &fb::AggregateFuncNodeArgs {
            name: Some(name),
            args: Some(args),
            distinct: false,
            alias: Some(alias),
            ..Default::default()
        },
    ))
}

/// What the executor calls this aggregate, which is DataFusion's own name plus the `ddof`
/// spelled into it: `stddev` is the sample form and `stddev_pop` the population one.
fn sql_name(func: AggFunc, ddof: u32) -> &'static str {
    match (func, ddof) {
        (AggFunc::Stddev, 0) => "stddev_pop",
        (AggFunc::Stddev, _) => "stddev",
        (AggFunc::Var, 0) => "var_pop",
        (AggFunc::Var, _) => "var",
        (AggFunc::Sum, _) => "sum",
        (AggFunc::Min, _) => "min",
        (AggFunc::Max, _) => "max",
        (AggFunc::Count, _) => "count",
        (AggFunc::Avg, _) => "avg",
    }
}

/// One of ours by the name the executor knows it by. `m2` and `merge_m2` have none of
/// their own: both are folded into the SQL aggregate whose state they are, above.
fn wire_name(agg: PlanAgg) -> Result<&'static str, PlanError> {
    Ok(match agg {
        PlanAgg::Sum => "sum",
        PlanAgg::Min => "min",
        PlanAgg::Max => "max",
        PlanAgg::Count => "count",
        PlanAgg::Mean => "mean",
        PlanAgg::M2 => {
            return Err(PlanError::Unsupported(
                "m2 alone has no name on the wire — it is written as part of the stddev or \
                 var it decomposes"
                    .to_string(),
            ));
        }
        PlanAgg::MergeM2 => {
            return Err(PlanError::Unsupported(
                "merge_m2 alone has no name on the wire either — it is written as the \
                 stddev or var whose three state columns it merges"
                    .to_string(),
            ));
        }
    })
}
