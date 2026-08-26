//! Both directions over one batch: DataFusion's expression, this mode's, and DataFusion's
//! again must all read the same column out of the same rows.
//!
//! Comparing the lowered expression to the original by shape would pin the spelling rather
//! than the meaning — `Sqrt` deliberately comes back as a function call, not a unary — so
//! what is asserted is the array each produces.

use super::*;
use crate::batch_partitioned::expr_translate::translate_expr;
use datafusion::arrow::array::{
    Array, ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray,
};
use datafusion::common::ScalarValue;
use datafusion::execution::context::SessionContext;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::ColumnarValue;

fn batch() -> RecordBatch {
    let n: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), Some(4), None, Some(9)]));
    let d: ArrayRef = Arc::new(Float64Array::from(vec![
        Some(1.0),
        Some(16.0),
        Some(0.25),
        None,
    ]));
    let s: ArrayRef = Arc::new(StringArray::from(vec![
        Some("ab"),
        Some("cd"),
        None,
        Some("ae"),
    ]));
    RecordBatch::try_from_iter_with_nullable([("n", n, true), ("d", d, true), ("s", s, true)])
        .expect("a batch of three columns")
}

fn column(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new(name, index))
}

fn registry() -> Arc<TaskContext> {
    SessionContext::new().task_ctx()
}

fn values(expr: &Arc<dyn PhysicalExpr>, batch: &RecordBatch) -> ArrayRef {
    match expr.evaluate(batch).expect("it evaluates") {
        ColumnarValue::Array(array) => array,
        ColumnarValue::Scalar(scalar) => scalar
            .to_array_of_size(batch.num_rows())
            .expect("a scalar widens"),
    }
}

/// The round trip a CPU executor makes: DataFusion's expression in, this mode's in the
/// middle, DataFusion's back out, and the same rows at both ends.
fn agree(what: &str, original: Arc<dyn PhysicalExpr>) {
    let batch = batch();
    let schema = batch.schema();
    let ours = translate_expr(&original, &schema).expect("this mode reads it");
    let back = physical_expr(&ours, &schema, registry().as_ref()).expect("and writes it back");
    assert_eq!(
        &values(&original, &batch),
        &values(&back, &batch),
        "{what} does not survive the round trip: {ours:?}"
    );
}

#[test]
fn a_column_comes_back_at_the_ordinal_it_went_in_at() {
    agree("a column", column("s", 2));
}

#[test]
fn a_comparison_and_the_boolean_over_it_survive() {
    let predicate = Arc::new(BinaryExpr::new(
        Arc::new(BinaryExpr::new(
            column("n", 0),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
        )),
        Operator::And,
        Arc::new(IsNotNullExpr::new(column("s", 2))),
    ));
    agree("a conjunction", predicate);
}

#[test]
fn arithmetic_keeps_the_type_datafusion_derived() {
    let sum = Arc::new(BinaryExpr::new(
        column("d", 1),
        Operator::Multiply,
        Arc::new(Literal::new(ScalarValue::Float64(Some(2.5)))),
    ));
    agree("a product", sum);
}

#[test]
fn a_cast_and_a_negation_survive() {
    let expr = Arc::new(NegativeExpr::new(Arc::new(CastExpr::new(
        column("n", 0),
        DataType::Float64,
        None,
    ))));
    agree("a negated cast", expr);
}

#[test]
fn a_like_keeps_its_negation_and_its_case_folding() {
    for (negated, case_insensitive) in [(false, false), (true, false), (false, true)] {
        let expr = Arc::new(LikeExpr::new(
            negated,
            case_insensitive,
            column("s", 2),
            Arc::new(Literal::new(ScalarValue::Utf8(Some("a%".to_string())))),
        ));
        agree(&format!("like(negated={negated}, ci={case_insensitive})"), expr);
    }
}

#[test]
fn a_case_survives_in_both_its_forms() {
    let searched = CaseExpr::try_new(
        None,
        vec![(
            Arc::new(BinaryExpr::new(
                column("n", 0),
                Operator::Gt,
                Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
            )) as Arc<dyn PhysicalExpr>,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("big".to_string())))) as Arc<dyn PhysicalExpr>,
        )],
        Some(Arc::new(Literal::new(ScalarValue::Utf8(Some(
            "small".to_string(),
        ))))),
    )
    .expect("a searched case");
    agree("a searched case", Arc::new(searched));

    let valued = CaseExpr::try_new(
        Some(column("n", 0)),
        vec![(
            Arc::new(Literal::new(ScalarValue::Int32(Some(4)))) as Arc<dyn PhysicalExpr>,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("four".to_string())))) as Arc<dyn PhysicalExpr>,
        )],
        None,
    )
    .expect("a valued case");
    agree("a valued case", Arc::new(valued));
}

#[test]
fn a_scalar_function_resolves_through_the_session() {
    let upper = SessionContext::new()
        .state()
        .scalar_functions()
        .get("upper")
        .expect("upper is a session function")
        .clone();
    let expr = Arc::new(ScalarFunctionExpr::new(
        "upper",
        upper,
        vec![column("s", 2)],
        DataType::Utf8,
    ));
    agree("upper", expr);
}

/// `Sqrt` is the one expression with no DataFusion unary behind it: it exists because a
/// stddev's finalize needs it. Lowering has to find the session's own `sqrt` rather than
/// compute a second one, so this asserts the digits and not the shape.
#[test]
fn sqrt_lowers_to_the_session_function_and_computes_the_root() {
    let batch = batch();
    let ours = Expr::unary(UnaryOp::Sqrt, Expr::column(1, "d"));
    let lowered =
        physical_expr(&ours, &batch.schema(), registry().as_ref()).expect("sqrt lowers");
    let got = values(&lowered, &batch);
    let got = got
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("a float column");
    assert_eq!(
        (got.value(0), got.value(1), got.value(2), got.is_null(3)),
        (1.0, 4.0, 0.5, true),
        "sqrt did not answer the roots of [1, 16, 0.25, null]"
    );
}

/// The name beside the ordinal is the check, not decoration: a rebase that moved a column
/// and left the name behind reads a different column, and every value it returns is the
/// right type. Nothing downstream can see that, which is why it is refused here.
#[test]
fn a_column_whose_name_disagrees_with_its_position_is_refused() {
    let batch = batch();
    let refused = physical_expr(
        &Expr::column(0, "d"),
        &batch.schema(),
        registry().as_ref(),
    )
    .expect_err("`d` is not column 0");
    let message = format!("{refused}");
    assert!(
        message.contains("column 0 is `d` here and `n` in the input"),
        "the refusal has to name both spellings: {message}"
    );
}

#[test]
fn a_column_past_the_end_of_the_input_is_refused() {
    let batch = batch();
    let refused = physical_expr(
        &Expr::column(7, "missing"),
        &batch.schema(),
        registry().as_ref(),
    )
    .expect_err("there is no column 7");
    let message = format!("{refused}");
    assert!(
        message.contains("at 7") && message.contains("3 columns"),
        "the refusal has to name the position and the width: {message}"
    );
}

#[test]
fn a_function_the_session_does_not_have_is_refused_by_name() {
    let batch = batch();
    let ours = Expr::ScalarFunction {
        name: "no_such_function".to_string(),
        args: vec![Expr::column(0, "n")],
        return_type: DataType::Int32,
        nullable: true,
    };
    let refused = physical_expr(&ours, &batch.schema(), registry().as_ref())
        .expect_err("the session has no such function");
    assert!(
        format!("{refused}").contains("no_such_function"),
        "the refusal has to name the function: {refused}"
    );
}

/// One case per field of the mode's own list, so a variant added to `Expr` without a
/// lowering arm is a compile error here rather than a refusal at run time.
#[test]
fn every_expression_kind_has_a_lowering() {
    let sample = Expr::Literal(ScalarValue::Int32(Some(1)));
    let all = [
        Expr::Column(crate::batch_partitioned::expr::ColumnRef {
            index: 0,
            name: "n".to_string(),
        }),
        Expr::Literal(ScalarValue::Int32(Some(1))),
        Expr::binary(
            sample.clone(),
            BinaryOp::Plus,
            sample.clone(),
            DataType::Int32,
        ),
        Expr::unary(UnaryOp::IsNull, sample.clone()),
        Expr::Cast {
            expr: Box::new(sample.clone()),
            target: DataType::Int64,
        },
        Expr::Like {
            expr: Box::new(Expr::column(2, "s")),
            pattern: Box::new(Expr::Literal(ScalarValue::Utf8(Some("a%".to_string())))),
            negated: false,
            case_insensitive: false,
        },
        Expr::Case {
            comparand: None,
            when_then: vec![(
                Expr::Literal(ScalarValue::Boolean(Some(true))),
                sample.clone(),
            )],
            else_expr: None,
        },
        Expr::ScalarFunction {
            name: "upper".to_string(),
            args: vec![Expr::column(2, "s")],
            return_type: DataType::Utf8,
            nullable: true,
        },
    ];
    let batch = batch();
    for expr in all {
        physical_expr(&expr, &batch.schema(), registry().as_ref())
            .unwrap_or_else(|error| panic!("{expr:?} has no lowering: {error}"));
    }
}

#[test]
fn a_projection_carries_its_output_names() {
    let batch = batch();
    let named = [
        NamedExpr::new(Expr::column(0, "n"), "the_number"),
        NamedExpr::new(
            Expr::binary(
                Expr::column(0, "n"),
                BinaryOp::Plus,
                Expr::Literal(ScalarValue::Int32(Some(1))),
                DataType::Int32,
            ),
            "one_more",
        ),
    ];
    let lowered = physical_projection(&named, &batch.schema(), registry().as_ref())
        .expect("the projection lowers");
    let names: Vec<&str> = lowered.iter().map(|(_, name)| name.as_str()).collect();
    assert_eq!(names, vec!["the_number", "one_more"]);
}
