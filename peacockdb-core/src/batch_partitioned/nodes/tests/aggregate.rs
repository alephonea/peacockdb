//! The two rules an aggregate body states about its own state columns: which aggregator
//! owns which of them, and what the finalize project emits. Both are read twice — by the
//! recipe writer for the wire and by the CPU backend for DataFusion — so a case here is
//! about the rule rather than about either reader.

use super::*;
use crate::batch_partitioned::aggregates::decomposition;
use crate::batch_partitioned::nodes::aggregate::{finalize_columns, key_width, state_funcs};
use crate::batch_partitioned::schema::AggStateColumns;
use datafusion::common::ScalarValue;

/// A state schema whose annotation says where the Welford triple sits: the columns, the
/// keys, and the one aggregate the last three of them belong to.
fn welford_state(columns: &[&str], keys: usize, output: &str) -> Schema {
    let fields: Vec<Field> = columns
        .iter()
        .map(|name| Field::new(*name, DataType::Float64, true))
        .collect();
    Schema {
        fields: Arc::new(ArrowSchema::new(fields)),
        group_keys: (0..keys as u32).collect(),
        agg_state: vec![AggStateColumns {
            output: output.to_string(),
            func: AggFunc::Stddev,
            ddof: 1,
            positions: (keys as u32..keys as u32 + 3).collect(),
        }],
    }
}

/// The three aggregators a stddev decomposes into, each over the same value expression —
/// which is what makes them one SQL aggregate on the far side.
fn welford_aggs(column: u32, name: &str) -> Vec<AggCall> {
    decomposition(AggFunc::Stddev)
        .state
        .iter()
        .map(|(suffix, func)| AggCall {
            func: *func,
            args: vec![Expr::column(column, "v")],
            outputs: vec![Field::new(
                format!("{name}{suffix}"),
                DataType::Float64,
                true,
            )],
        })
        .collect()
}

/// The plain case, and the one every other reading is measured against.
#[test]
fn a_welford_triple_is_one_aggregate_under_the_name_sql_wrote() {
    let body = AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: welford_aggs(1, "stddev(v)"),
        finalize: None,
    };
    let state = welford_state(
        &["k", "stddev(v)$count", "stddev(v)$mean", "stddev(v)$m2"],
        1,
        "stddev(v)",
    );
    let funcs = state_funcs(&body, &state).expect("the aggregators are nameable");
    let named: Vec<(&str, &str)> = funcs
        .iter()
        .map(|func| (func.name, func.alias.as_str()))
        .collect();
    assert_eq!(
        named,
        vec![("stddev", "stddev(v)")],
        "three aggregators, one SQL name, one alias"
    );
}

/// The grouping-set init emits `__grouping_id` beside its keys, so the state is one column
/// wider than the group list. Reading the group list as the key width shifts every state
/// position by one: the triple is then found under the aggregator after the one that owns
/// it, its first aggregator is left unclaimed, and the wire is told about a `count` that
/// nothing on the far side has a state column for.
#[test]
fn a_welford_triple_beside_a_grouping_id_is_still_one_aggregate() {
    let body = AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: vec![vec![false], vec![true]],
        null_exprs: vec![Expr::Literal(ScalarValue::Int64(None))],
        aggs: welford_aggs(1, "stddev(v)"),
        finalize: None,
    };
    let state = welford_state(
        &[
            "k",
            "__grouping_id",
            "stddev(v)$count",
            "stddev(v)$mean",
            "stddev(v)$m2",
        ],
        2,
        "stddev(v)",
    );
    assert_eq!(
        key_width(&body),
        2,
        "the keys are the group list and the id"
    );
    let funcs = state_funcs(&body, &state).expect("the aggregators are nameable");
    let named: Vec<&str> = funcs.iter().map(|func| func.name).collect();
    assert_eq!(
        named,
        vec!["stddev"],
        "the triple is one aggregate wherever it sits in the state"
    );
}

/// A finalize over a grouping-set state: the id is a key, so the project carries it
/// through like any other. Counting the group list alone leaves the row a column short,
/// which the width check catches — a refusal where the query has an answer.
#[test]
fn a_finalize_over_a_grouping_id_carries_it_through_as_a_key() {
    let body = AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: vec![vec![false], vec![true]],
        null_exprs: vec![Expr::Literal(ScalarValue::Int64(None))],
        aggs: vec![AggCall {
            func: PlanAgg::Sum,
            args: vec![Expr::column(1, "v")],
            outputs: vec![Field::new("sum(v)", DataType::Int64, true)],
        }],
        finalize: Some(vec![NamedExpr::new(Expr::column(2, "sum(v)"), "sum(v)")]),
    };
    let state = columns(&["k", "__grouping_id", "sum(v)"]);
    let output = columns(&["k", "__grouping_id", "sum(v)"]);
    let columns = finalize_columns(&body, &state, &output).expect("the row is projectable");
    let named: Vec<(&str, &Expr)> = columns
        .iter()
        .map(|column| (column.name.as_str(), &column.expr))
        .collect();
    assert_eq!(
        named.iter().map(|(name, _)| *name).collect::<Vec<&str>>(),
        vec!["k", "__grouping_id", "sum(v)"],
        "the project is the whole row, keys included"
    );
    assert!(
        matches!(named[1].1, Expr::Column(reference) if reference.index == 1),
        "the id is read from the state where it sits: {:?}",
        named[1].1
    );
}

fn columns(names: &[&str]) -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(
        names
            .iter()
            .map(|name| Field::new(*name, DataType::Int64, true))
            .collect::<Vec<Field>>(),
    )))
}
