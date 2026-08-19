//! Which columns can be NULL, and the one join shape that has to be refused because of it.
//!
//! Three of the join types hardcode `cudf::null_equality::EQUAL` whatever the plan's flag
//! says (`cpp/src/operators/join.cpp`), so a NULL key matches a NULL key there. That is
//! what a set operation wants and what SQL's `NOT IN` does not, and the difference only
//! shows where NULLs can meet — which is why the refusal reads the data rather than the
//! declared types: every column in both benchmarks is declared nullable, keys included.
//!
//! Where the analysis cannot decide, a column is possibly-NULL. A false positive costs a
//! refusal a reader can see; a false negative is the wrong answer this exists to prevent.

use datafusion::common::JoinType;

use super::error::PlanError;
use super::expr::Expr;
use super::node::GpuNode;
use super::nodes::{NodeRef, as_node_ref};

/// Refuses an anti or mark join whose NULLs can meet under SQL semantics. Everything else
/// plans: semi honours the flag, and `null_equals_null=true` is asking for the equality the
/// executor hardcodes.
pub fn refuse_null_unsafe_joins(root: &dyn GpuNode) -> Result<(), PlanError> {
    for child in root.children() {
        refuse_null_unsafe_joins(child)?;
    }
    let NodeRef::Join(join) = as_node_ref(root) else {
        return Ok(());
    };
    if !hardcodes_null_equality(join.join_type) || join.null_equals_null {
        return Ok(());
    }
    let (build, probe) = (root.children()[0], root.children()[1]);
    let (build_nulls, probe_nulls) = (can_be_null(build), can_be_null(probe));
    let meeting = join.keys.iter().find(|(b, p)| {
        nullable_at(&build_nulls, *b) && nullable_at(&probe_nulls, *p)
    });
    match meeting {
        Some((b, p)) => Err(PlanError::Unsupported(format!(
            "{:?} join on a key that can be NULL on both sides (build @{b}, probe @{p}) with \
             null_equals_null=false: the executor matches NULL to NULL there whatever the \
             flag says, which is set semantics rather than SQL's (#59, #80)",
            join.join_type
        ))),
        None => Ok(()),
    }
}

/// The three where the executor's `null_equality` is not the plan's to choose.
fn hardcodes_null_equality(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::LeftAnti | JoinType::RightAnti | JoinType::LeftMark
    )
}

fn nullable_at(columns: &[bool], ordinal: u32) -> bool {
    columns.get(ordinal as usize).copied().unwrap_or(true)
}

/// Per output column of this node, whether it can be NULL.
pub fn can_be_null(node: &dyn GpuNode) -> Vec<bool> {
    let width = node
        .kind()
        .schema()
        .map_or(0, |schema| schema.fields.fields().len());
    let child = |index: usize| -> Vec<bool> {
        node.children()
            .get(index)
            .map(|child| can_be_null(*child))
            .unwrap_or_default()
    };

    match as_node_ref(node) {
        // The leaf: what the surviving row groups actually hold.
        NodeRef::LoadParquet(load) => load.can_be_null.clone(),

        // A projection re-numbers columns and can compute new ones, so each is its own
        // question; a filter's projection is the same question over a subset.
        NodeRef::Project(project) => project
            .exprs
            .iter()
            .map(|named| expr_can_be_null(&named.expr, &child(0)))
            .collect(),
        NodeRef::Filter(filter) => {
            let input = child(0);
            match &filter.projection {
                Some(kept) => kept.iter().map(|c| nullable_at(&input, *c)).collect(),
                None => input,
            }
        }

        // Rows move, are dropped or are re-ordered; a column that could not be NULL below
        // still cannot be above.
        NodeRef::Sort(_)
        | NodeRef::AccumulateBatchesAndSort(_)
        | NodeRef::MergeSortedPartitions(_)
        | NodeRef::CoalesceAllBatches(_)
        | NodeRef::MergePartitions(_)
        | NodeRef::EmitPartitions(_)
        | NodeRef::Limit(_)
        | NodeRef::Unload(_) => child(0),

        // Branches disagree by column, so a column is nullable if it is in any branch.
        NodeRef::Union(_) | NodeRef::Interleave(_) => {
            let mut columns = vec![false; width];
            for index in 0..node.children().len() {
                for (position, nullable) in child(index).iter().enumerate() {
                    if position < columns.len() {
                        columns[position] |= *nullable;
                    }
                }
            }
            columns
        }

        // A group key cannot become NULL by being grouped on — except where a grouping set
        // substitutes one deliberately, which the plan already carries as null_exprs. An
        // aggregate's own output can be NULL: a sum over no rows is.
        NodeRef::Aggregate(aggregate) => aggregate_can_be_null(&aggregate.body, &child(0), width),
        NodeRef::AggregateBatches(aggregate) => {
            aggregate_can_be_null(&aggregate.body, &child(0), width)
        }

        // An outer join null-pads the side it does not preserve. The projection is over the
        // joined table, so the padding is read there and then narrowed.
        NodeRef::Join(join) => {
            let joined = joined_can_be_null(join.join_type, &child(0), &child(1));
            match &join.projection {
                Some(kept) => kept.iter().map(|c| nullable_at(&joined, *c)).collect(),
                None => joined,
            }
        }
        NodeRef::CrossJoin(_) => [child(0), child(1)].concat(),
        NodeRef::NestedLoopJoin(join) => {
            use super::nodes::join::NestedLoopJoinType;
            let (build, probe) = (child(0), child(1));
            match join.join_type {
                NestedLoopJoinType::Inner => [build, probe].concat(),
                // Left keeps its build rows and pads the probe.
                NestedLoopJoinType::Left => {
                    [build, vec![true; probe.len()]].concat()
                }
            }
        }
    }
}

/// The joined table, before any projection: build columns then probe columns, with the
/// unpreserved side padded. The semi family emits one side only.
fn joined_can_be_null(join_type: JoinType, build: &[bool], probe: &[bool]) -> Vec<bool> {
    let padded = |columns: &[bool]| vec![true; columns.len()];
    match join_type {
        JoinType::Inner => [build.to_vec(), probe.to_vec()].concat(),
        JoinType::Left => [build.to_vec(), padded(probe)].concat(),
        JoinType::Right => [padded(build), probe.to_vec()].concat(),
        JoinType::Full => [padded(build), padded(probe)].concat(),
        JoinType::LeftSemi | JoinType::LeftAnti => build.to_vec(),
        JoinType::RightSemi | JoinType::RightAnti => probe.to_vec(),
        // The mark column is a boolean the join computes and never NULL.
        JoinType::LeftMark => [build.to_vec(), vec![false]].concat(),
    }
}

fn aggregate_can_be_null(
    body: &super::nodes::AggregateBody,
    input: &[bool],
    width: usize,
) -> Vec<bool> {
    let mut columns: Vec<bool> = body
        .group_by
        .iter()
        .map(|key| expr_can_be_null(key, input))
        .collect();
    // A grouping set substitutes NULL for the keys it excludes, so any key some set drops
    // can be NULL in the output — and the grouping id itself never is.
    for mask in &body.grouping_sets {
        for (position, dropped) in mask.iter().enumerate() {
            if *dropped && position < columns.len() {
                columns[position] = true;
            }
        }
    }
    if !body.grouping_sets.is_empty() {
        columns.push(false);
    }
    // Everything the aggregators and the finalize expressions produce: a sum over no rows
    // is NULL, and a stddev under its ddof is NULL by construction.
    columns.resize(width.max(columns.len()), true);
    columns.truncate(width);
    columns
}

/// An expression can be NULL unless every path to a value is known not to be. A bare column
/// asks its input; an operator asks its operands; anything this does not model says yes.
fn expr_can_be_null(expr: &Expr, input: &[bool]) -> bool {
    match expr {
        Expr::Column(reference) => nullable_at(input, reference.index),
        Expr::Literal(value) => value.is_null(),
        Expr::Binary { left, right, .. } => {
            expr_can_be_null(left, input) || expr_can_be_null(right, input)
        }
        Expr::Cast { expr, .. } => expr_can_be_null(expr, input),
        // A predicate answers true or false about a NULL rather than becoming one.
        Expr::Unary { op, arg } => match op {
            super::expr::UnaryOp::IsNull | super::expr::UnaryOp::IsNotNull => false,
            _ => expr_can_be_null(arg, input),
        },
        // No ELSE is an implicit NULL, and every branch is a value the CASE can return.
        Expr::Case { when_then, else_expr, .. } => match else_expr {
            None => true,
            Some(otherwise) => {
                expr_can_be_null(otherwise, input)
                    || when_then
                        .iter()
                        .any(|(_, then)| expr_can_be_null(then, input))
            }
        },
        Expr::Like { expr, pattern, .. } => {
            expr_can_be_null(expr, input) || expr_can_be_null(pattern, input)
        }
        // coalesce is the shape that makes a general rule wrong here, so this does not
        // guess: a function's result is possibly-NULL.
        Expr::ScalarFunction { .. } => true,
    }
}
