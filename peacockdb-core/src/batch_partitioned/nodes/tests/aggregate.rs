//! The two rules an aggregate body states about its own state columns: which aggregator
//! owns which of them, and what the finalize project emits. Both are read twice — by the
//! recipe writer for the wire and by the CPU backend for DataFusion — so a case here is
//! about the rule rather than about either reader.

use super::*;
use crate::batch_partitioned::aggregates::{AggSpec, decomposition, finalize};
use crate::batch_partitioned::expr_physical::physical_expr;
use crate::batch_partitioned::nodes::aggregate::{finalize_columns, key_width, state_funcs};
use crate::batch_partitioned::schema::AggStateColumns;
use datafusion::arrow::array::{Array, Float64Array, RecordBatch, UInt64Array};
use datafusion::common::ScalarValue;
use datafusion::execution::context::SessionContext;

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

/// DataFusion's own Welford state types, which is what makes the finalize's typing a
/// question at all: the count is unsigned and the output it feeds is a float.
fn welford_fields(name: &str) -> Vec<Field> {
    vec![
        Field::new(format!("{name}$count"), DataType::UInt64, true),
        Field::new(format!("{name}$mean"), DataType::Float64, true),
        Field::new(format!("{name}$m2"), DataType::Float64, true),
    ]
}

/// The `count - ddof` both arms rest on, read off the finalize rather than rebuilt, plus
/// the zero it is compared against.
fn denominator_and_zero(expr: &Expr) -> (&Expr, &ScalarValue) {
    let Expr::Case { when_then, .. } = expr else {
        panic!("a dispersion finalize is a CASE over its denominator: {expr:?}");
    };
    let Expr::Binary {
        left,
        op: BinaryOp::LtEq,
        right,
        ..
    } = &when_then[0].0
    else {
        panic!(
            "the NULL arm compares the denominator: {:?}",
            when_then[0].0
        );
    };
    let Expr::Literal(zero) = right.as_ref() else {
        panic!("the comparand is a literal: {right:?}");
    };
    (left.as_ref(), zero)
}

/// Every operand in the output's type. The count is cast before anything is subtracted
/// from it, and the two literals are floats: an Int64 ddof puts the subtraction in the
/// count's own unsigned type, where a count below ddof wraps instead of going negative.
#[test]
fn a_dispersion_finalize_subtracts_in_the_type_it_outputs() {
    for func in [AggFunc::Stddev, AggFunc::Var] {
        let state = welford_fields("d(v)");
        let expr = finalize(AggSpec { func, ddof: 1 }, &state, 0, &DataType::Float64);
        let (denominator, zero) = denominator_and_zero(&expr);
        assert_eq!(
            zero,
            &ScalarValue::Float64(Some(0.0)),
            "{func:?} compares its denominator against an Int64 zero"
        );
        let Expr::Binary {
            left,
            op: BinaryOp::Minus,
            right,
            out_type,
        } = denominator
        else {
            panic!("{func:?}'s denominator is count - ddof: {denominator:?}");
        };
        assert!(
            matches!(
                left.as_ref(),
                Expr::Cast {
                    target: DataType::Float64,
                    ..
                }
            ),
            "{func:?} subtracts from the count in the count's type: {left:?}"
        );
        assert_eq!(
            right.as_ref(),
            &Expr::Literal(ScalarValue::Float64(Some(1.0))),
            "{func:?}'s ddof is not typed to the output"
        );
        assert_eq!(&DataType::Float64, out_type, "{func:?}'s denominator type");
    }
}

/// What the typing buys, over a state no corpus query produces: a group with fewer rows
/// than its degrees of freedom has no dispersion to report and owes NULL. Subtracting in
/// the count's own unsigned type never reaches that arm — cuDF wraps past it and answers a
/// value, arrow refuses the mixed subtraction outright.
#[test]
fn a_group_below_its_degrees_of_freedom_finalizes_to_null() {
    let state = welford_fields("d(v)");
    let schema = ArrowSchema::new(state.clone());
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 1, 3])),
            Arc::new(Float64Array::from(vec![0.0, 10.0, 10.0])),
            Arc::new(Float64Array::from(vec![0.0, 0.0, 8.0])),
        ],
    )
    .expect("a Welford state of three groups");
    let registry = SessionContext::new().task_ctx();
    for (func, three_rows) in [(AggFunc::Stddev, 2.0), (AggFunc::Var, 4.0)] {
        let expr = finalize(AggSpec { func, ddof: 1 }, &state, 0, &DataType::Float64);
        let values = physical_expr(&expr, &schema, registry.as_ref())
            .expect("the finalize lowers to DataFusion")
            .evaluate(&batch)
            .expect("and evaluates over the state")
            .into_array(batch.num_rows())
            .expect("as one value per group");
        let values = values
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("a dispersion finalizes to its declared float");
        assert!(
            values.is_null(0),
            "{func:?} over an empty group answers {}",
            values.value(0)
        );
        assert!(
            values.is_null(1),
            "{func:?} over one row at ddof 1 answers {}",
            values.value(1)
        );
        assert_eq!(
            values.value(2),
            three_rows,
            "{func:?} over three rows with m2 8"
        );
    }
}

fn columns(names: &[&str]) -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(
        names
            .iter()
            .map(|name| Field::new(*name, DataType::Int64, true))
            .collect::<Vec<Field>>(),
    )))
}
