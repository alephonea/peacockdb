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

use super::super::aggregates::AggCall;
use super::super::error::PlanError;
use super::super::nodes;
use super::super::nodes::aggregate::AggregateBody;
use super::super::schema::Schema;
use super::expr_writer::write_expr;
use super::node_writer;
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
    let group_names: Vec<WIPOffset<&str>> = (0..body.group_by.len())
        .map(|position| b.create_string(group_name_at(state, position)))
        .collect();

    // ROLLUP and CUBE, which only an init node expands: the per-position NULL placeholder
    // and the masks that say which positions a set drops. `aggregate.cpp` discriminates on
    // `null_exprs` rather than on the masks, so writing one without the other would run a
    // plain group-by and answer with one set where the query asked for five.
    let mut null_exprs = Vec::with_capacity(body.null_exprs.len());
    for expr in &body.null_exprs {
        null_exprs.push(write_expr(b, expr)?);
    }
    let null_names: Vec<WIPOffset<&str>> = (0..null_exprs.len())
        .map(|position| b.create_string(group_name_at(state, position)))
        .collect();
    let masks: Vec<WIPOffset<fb::GroupingSetMask>> = body
        .grouping_sets
        .iter()
        .map(|mask| {
            let values = b.create_vector(mask);
            fb::GroupingSetMask::create(
                b,
                &fb::GroupingSetMaskArgs {
                    values: Some(values),
                },
            )
        })
        .collect();

    let (funcs, mergeable) = state_funcs(b, body, state)?;

    let group_exprs = b.create_vector(&group_exprs);
    let group_names = b.create_vector(&group_names);
    let null_exprs = b.create_vector(&null_exprs);
    let null_names = b.create_vector(&null_names);
    let grouping_sets = b.create_vector(&masks);
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
            null_exprs: Some(null_exprs),
            null_names: Some(null_names),
            grouping_sets: Some(grouping_sets),
            aggr_input_schema: Some(input_schema),
            mergeable_agg_state: mergeable,
        },
    );
    Ok(Payload {
        kind: fb::PlanNodeKind::CudfAggregate,
        value: aggregate.as_union_value(),
    })
}

/// The state's key columns lead it, so a group position is a position in it. Shared by the
/// group names and the NULL-placeholder names, which the wire keeps parallel.
fn group_name_at(state: &Schema, position: usize) -> &str {
    node_writer::field_at(state, position as u32).name()
}

/// The finalize as the project it becomes — the columns are
/// [`nodes::aggregate::finalize_columns`](super::super::nodes::aggregate::finalize_columns);
/// this writes what that names.
pub(super) fn finalize_project<'a>(
    b: &mut FlatBufferBuilder<'a>,
    body: &AggregateBody,
    state: &Schema,
    output: &Schema,
    kids: &[WIPOffset<fb::PlanNode<'a>>],
) -> Result<Payload, PlanError> {
    let columns = nodes::aggregate::finalize_columns(body, state, output)?;
    let mut exprs = Vec::with_capacity(columns.len());
    for column in &columns {
        exprs.push(write_expr(b, &column.expr)?);
    }
    // Every expression before every name, which is the order the payload golden was
    // canonized in: a builder writes back to front, so interleaving the two would move
    // bytes that mean the same thing.
    let names = columns
        .iter()
        .map(|column| b.create_string(&column.name))
        .collect();
    Ok(node_writer::project_payload(b, exprs, names, kids[0]))
}

/// The aggregators as the wire declares them, in the order their state columns appear —
/// which aggregate is which, and what it is called, is
/// [`nodes::aggregate::state_funcs`](super::super::nodes::aggregate::state_funcs); this
/// writes what that names.
fn state_funcs<'a>(
    b: &mut FlatBufferBuilder<'a>,
    body: &AggregateBody,
    state: &Schema,
) -> Result<(Vec<WIPOffset<fb::AggregateFuncNode<'a>>>, bool), PlanError> {
    let declared = nodes::aggregate::state_funcs(body, state)?;
    let mut funcs = Vec::with_capacity(declared.len());
    let mut folded = false;
    for func in &declared {
        folded |= func.welford;
        funcs.push(named_func(b, func.name, func.call, &func.alias)?);
    }
    Ok((funcs, folded))
}

/// `out_decimal_precision`/`out_decimal_scale` stay at zero, and that is a decision. They
/// are the legacy writer's channel for `avg`'s declared decimal type (`operators/aggregate.rs`,
/// read by `aggregate.cpp` at :216 and :711), but this mode never sends an `avg` to a device:
/// decomposition splits it into sum and count, so the scale rides on the finalize divide's own
/// `out_decimal_precision`, which `expr_writer` sets and
/// `an_average_finalizes_to_the_digits_the_oracle_computes` proves against the oracle's digits.
/// No shape this mode plans sends a name `is_avg` matches: inside the Welford triple the func is
/// named by `sql_name`, and `wire_name`'s `"mean"` is reachable only from the arm below it, which
/// decomposition leaves no `Mean` aggregator to enter. Written out rather than defaulted so a
/// field added to the table has to be answered here.
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
            out_decimal_precision: 0,
            out_decimal_scale: 0,
        },
    ))
}
