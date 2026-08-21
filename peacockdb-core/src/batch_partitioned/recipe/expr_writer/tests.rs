//! Every variant, every operator, and the literals the corpus actually produces.
//!
//! Read back through the generated accessors rather than compared as bytes: what matters
//! is that the C++ reading this buffer finds the expression we meant, and a byte compare
//! would pass on two spellings of the wrong thing as readily as on the right one.
//!
//! Decimals come first among the literals because that is where a wrong write is
//! invisible: a precision or a scale dropped here reads as a plausible number on a
//! device and as nothing at all in a plan.

use super::*;
use datafusion::common::ScalarValue;

/// Round-trip one expression: written as the buffer's root, then read back.
fn written(expr: &Expr) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let offset = write_expr(&mut b, expr).expect("the writer takes this expression");
    b.finish(offset, None);
    b.finished_data().to_vec()
}

fn read(bytes: &[u8]) -> fb::Expr<'_> {
    flatbuffers::root::<fb::Expr>(bytes).expect("a verifiable Expr")
}

fn column(index: u32, name: &str) -> Expr {
    Expr::column(index, name)
}

#[test]
fn a_column_carries_its_ordinal_and_the_name_beside_it() {
    let bytes = written(&column(3, "l_quantity"));
    let expr = read(&bytes);
    assert_eq!(expr.node_type(), fb::ExprNode::ColumnRef);
    let reference = expr.node_as_column_ref().expect("a ColumnRef");
    assert_eq!(reference.index(), 3);
    assert_eq!(reference.name(), Some("l_quantity"));
}

#[test]
fn a_decimal_literal_keeps_its_precision_scale_and_both_halves() {
    // 1234567890123456789012345678 needs the high word, so a writer that dropped it
    // would still produce a plausible small number.
    let value: i128 = 1_234_567_890_123_456_789_012_345_678;
    let bytes = written(&Expr::Literal(ScalarValue::Decimal128(
        Some(value),
        38,
        10,
    )));
    let scalar = read(&bytes)
        .node_as_literal_expr()
        .expect("a LiteralExpr")
        .value()
        .expect("a value");
    assert_eq!(scalar.type_(), fb::DataType::Decimal128);
    assert_eq!(scalar.decimal_precision(), 38);
    assert_eq!(scalar.decimal_scale(), 10);
    let recovered = ((scalar.decimal_hi() as i128) << 64) | scalar.decimal_lo() as i128;
    assert_eq!(recovered, value);
    assert!(!scalar.is_null());
}

#[test]
fn a_negative_decimal_survives_the_split_into_two_words() {
    let value: i128 = -55_555_555_555;
    let bytes = written(&Expr::Literal(ScalarValue::Decimal128(Some(value), 15, 2)));
    let scalar = read(&bytes)
        .node_as_literal_expr()
        .unwrap()
        .value()
        .unwrap();
    let recovered = ((scalar.decimal_hi() as i128) << 64) | scalar.decimal_lo() as i128;
    assert_eq!(recovered, value, "the sign lives in the high word");
}

#[test]
fn a_typed_null_says_which_type_it_is_null_of() {
    // A decimal NULL is what a stddev's finalize substitutes, so the precision and scale
    // have to survive a value that has no digits.
    let bytes = written(&Expr::Literal(ScalarValue::Decimal128(None, 20, 4)));
    let scalar = read(&bytes)
        .node_as_literal_expr()
        .unwrap()
        .value()
        .unwrap();
    assert!(scalar.is_null());
    assert_eq!(scalar.type_(), fb::DataType::Decimal128);
    assert_eq!((scalar.decimal_precision(), scalar.decimal_scale()), (20, 4));
}

#[test]
fn the_literal_kinds_the_corpus_produces_all_write() {
    let cases: Vec<(ScalarValue, fb::DataType)> = vec![
        (ScalarValue::Boolean(Some(true)), fb::DataType::Boolean),
        (ScalarValue::Int32(Some(-7)), fb::DataType::Int32),
        (ScalarValue::Int64(Some(1 << 40)), fb::DataType::Int64),
        (ScalarValue::Float64(Some(0.5)), fb::DataType::Float64),
        (
            ScalarValue::Utf8(Some("BUILDING".to_string())),
            fb::DataType::Utf8,
        ),
        (ScalarValue::Date32(Some(9131)), fb::DataType::Date32),
        (ScalarValue::Null, fb::DataType::Null),
    ];
    for (value, expected) in cases {
        let bytes = written(&Expr::Literal(value.clone()));
        let scalar = read(&bytes)
            .node_as_literal_expr()
            .unwrap()
            .value()
            .unwrap();
        assert_eq!(scalar.type_(), expected, "{value:?}");
    }
    let bytes = written(&Expr::Literal(ScalarValue::Utf8(Some("A".to_string()))));
    let scalar = read(&bytes)
        .node_as_literal_expr()
        .unwrap()
        .value()
        .unwrap();
    assert_eq!(scalar.string_val(), Some("A"));
}

#[test]
fn every_binary_operator_maps_to_its_own_wire_op() {
    let cases = [
        (BinaryOp::Eq, fb::BinaryOp::Eq),
        (BinaryOp::NotEq, fb::BinaryOp::NotEq),
        (BinaryOp::Lt, fb::BinaryOp::Lt),
        (BinaryOp::LtEq, fb::BinaryOp::LtEq),
        (BinaryOp::Gt, fb::BinaryOp::Gt),
        (BinaryOp::GtEq, fb::BinaryOp::GtEq),
        (BinaryOp::Plus, fb::BinaryOp::Plus),
        (BinaryOp::Minus, fb::BinaryOp::Minus),
        (BinaryOp::Multiply, fb::BinaryOp::Multiply),
        (BinaryOp::Divide, fb::BinaryOp::Divide),
        (BinaryOp::Modulo, fb::BinaryOp::Modulo),
        (BinaryOp::And, fb::BinaryOp::And),
        (BinaryOp::Or, fb::BinaryOp::Or),
        (BinaryOp::BitwiseAnd, fb::BinaryOp::BitwiseAnd),
        (BinaryOp::BitwiseOr, fb::BinaryOp::BitwiseOr),
        (BinaryOp::BitwiseXor, fb::BinaryOp::BitwiseXor),
        (BinaryOp::BitwiseShiftLeft, fb::BinaryOp::BitwiseShiftLeft),
        (BinaryOp::BitwiseShiftRight, fb::BinaryOp::BitwiseShiftRight),
        (BinaryOp::StringConcat, fb::BinaryOp::StringConcat),
        (BinaryOp::IsDistinctFrom, fb::BinaryOp::IsDistinctFrom),
        (BinaryOp::IsNotDistinctFrom, fb::BinaryOp::IsNotDistinctFrom),
    ];
    // Every arm of the IR enum appears above: a new operator added to one side and not
    // the other is what this count catches.
    assert_eq!(cases.len(), 21, "the IR has 21 binary operators");
    for (ours, theirs) in cases {
        let bytes = written(&Expr::binary(
            column(0, "a"),
            ours,
            column(1, "b"),
            DataType::Boolean,
        ));
        let binary = read(&bytes)
            .node_as_binary_expr_node()
            .expect("a BinaryExprNode");
        assert_eq!(binary.op(), theirs, "{ours:?}");
        assert_eq!(
            binary.left().unwrap().node_as_column_ref().unwrap().index(),
            0
        );
        assert_eq!(
            binary.right().unwrap().node_as_column_ref().unwrap().index(),
            1
        );
    }
}

#[test]
fn a_binary_op_carries_the_declared_decimal_output_and_zeroes_where_there_is_none() {
    // The case the field exists for: cuDF's divide scale is s_left - s_right and
    // DataFusion's is not, so the declared type is how the C++ lands on the right one.
    let bytes = written(&Expr::binary(
        column(0, "a"),
        BinaryOp::Divide,
        column(1, "b"),
        DataType::Decimal128(25, 6),
    ));
    let binary = read(&bytes).node_as_binary_expr_node().unwrap();
    assert_eq!(binary.out_decimal_precision(), 25);
    assert_eq!(binary.out_decimal_scale(), 6);

    let bytes = written(&Expr::binary(
        column(0, "a"),
        BinaryOp::Plus,
        column(1, "b"),
        DataType::Int64,
    ));
    let binary = read(&bytes).node_as_binary_expr_node().unwrap();
    assert_eq!(
        (binary.out_decimal_precision(), binary.out_decimal_scale()),
        (0, 0),
        "zero precision is how the wire says `not a decimal`"
    );
}

#[test]
fn every_unary_operator_maps_to_its_own_wire_op() {
    for (ours, theirs) in [
        (UnaryOp::Not, fb::UnaryOp::Not),
        (UnaryOp::IsNull, fb::UnaryOp::IsNull),
        (UnaryOp::IsNotNull, fb::UnaryOp::IsNotNull),
        (UnaryOp::Negative, fb::UnaryOp::Negative),
        // The fifth, appended to the schema for a stddev's finalize.
        (UnaryOp::Sqrt, fb::UnaryOp::Sqrt),
    ] {
        let bytes = written(&Expr::unary(ours, column(0, "a")));
        let unary = read(&bytes).node_as_unary_expr_node().expect("a UnaryExprNode");
        assert_eq!(unary.op(), theirs, "{ours:?}");
        assert_eq!(unary.arg().unwrap().node_as_column_ref().unwrap().index(), 0);
    }

}

#[test]
fn a_cast_carries_its_target_and_a_decimal_targets_scale() {
    let bytes = written(&Expr::Cast {
        expr: Box::new(column(0, "n")),
        target: DataType::Decimal128(18, 0),
    });
    let cast = read(&bytes).node_as_cast_expr_node().expect("a CastExprNode");
    assert_eq!(cast.target_type(), fb::DataType::Decimal128);
    assert_eq!((cast.decimal_precision(), cast.decimal_scale()), (18, 0));

    let bytes = written(&Expr::Cast {
        expr: Box::new(column(0, "n")),
        target: DataType::Float64,
    });
    let cast = read(&bytes).node_as_cast_expr_node().unwrap();
    assert_eq!(cast.target_type(), fb::DataType::Float64);
    assert_eq!((cast.decimal_precision(), cast.decimal_scale()), (0, 0));
}

#[test]
fn a_like_carries_both_flags_independently() {
    for negated in [false, true] {
        for case_insensitive in [false, true] {
            let bytes = written(&Expr::Like {
                expr: Box::new(column(0, "p_type")),
                pattern: Box::new(Expr::Literal(ScalarValue::Utf8(Some("%BRASS".to_string())))),
                negated,
                case_insensitive,
            });
            let like = read(&bytes).node_as_like_expr_node().expect("a LikeExprNode");
            assert_eq!(like.negated(), negated);
            assert_eq!(like.case_insensitive(), case_insensitive);
            assert_eq!(
                like.pattern().unwrap().node_type(),
                fb::ExprNode::LiteralExpr
            );
        }
    }
}

#[test]
fn both_case_forms_write_and_the_search_form_leaves_no_comparand() {
    let when_then = vec![(
        Expr::binary(
            column(0, "n"),
            BinaryOp::LtEq,
            Expr::Literal(ScalarValue::Int64(Some(0))),
            DataType::Boolean,
        ),
        Expr::Literal(ScalarValue::Int64(None)),
    )];

    let bytes = written(&Expr::Case {
        comparand: None,
        when_then: when_then.clone(),
        else_expr: Some(Box::new(column(1, "v"))),
    });
    let case = read(&bytes).node_as_case_expr_node().expect("a CaseExprNode");
    assert!(case.expr().is_none(), "the search form has no comparand");
    assert_eq!(case.when_thens().unwrap().len(), 1);
    assert!(case.else_expr().is_some());

    let bytes = written(&Expr::Case {
        comparand: Some(Box::new(column(2, "k"))),
        when_then,
        else_expr: None,
    });
    let case = read(&bytes).node_as_case_expr_node().unwrap();
    assert_eq!(
        case.expr().unwrap().node_as_column_ref().unwrap().index(),
        2
    );
    assert!(case.else_expr().is_none(), "an absent ELSE stays absent");
}

#[test]
fn a_case_with_several_arms_keeps_them_in_order() {
    // Order is the semantics: the first true WHEN wins, so a writer that reordered them
    // would answer a different question with the same columns.
    let arm = |k: i64| {
        (
            Expr::binary(
                column(0, "k"),
                BinaryOp::Eq,
                Expr::Literal(ScalarValue::Int64(Some(k))),
                DataType::Boolean,
            ),
            Expr::Literal(ScalarValue::Int64(Some(k * 10))),
        )
    };
    let bytes = written(&Expr::Case {
        comparand: None,
        when_then: vec![arm(1), arm(2), arm(3)],
        else_expr: None,
    });
    let case = read(&bytes).node_as_case_expr_node().unwrap();
    let arms = case.when_thens().unwrap();
    assert_eq!(arms.len(), 3);
    for (position, expected) in [10, 20, 30].into_iter().enumerate() {
        let then = arms
            .get(position)
            .then()
            .unwrap()
            .node_as_literal_expr()
            .unwrap()
            .value()
            .unwrap();
        assert_eq!(then.int_val(), expected);
    }
}

#[test]
fn a_scalar_function_carries_its_name_return_type_and_nullability() {
    let bytes = written(&Expr::ScalarFunction {
        name: "date_part".to_string(),
        args: vec![
            Expr::Literal(ScalarValue::Utf8(Some("YEAR".to_string()))),
            column(0, "o_orderdate"),
        ],
        return_type: DataType::Float64,
        nullable: false,
    });
    let function = read(&bytes)
        .node_as_scalar_function_expr_node()
        .expect("a ScalarFunctionExprNode");
    assert_eq!(function.name(), Some("date_part"));
    assert_eq!(function.return_type(), fb::DataType::Float64);
    assert!(!function.nullable(), "DataFusion derived this per function");
    assert_eq!(function.args().unwrap().len(), 2);
    assert_eq!(
        function.args().unwrap().get(1).node_type(),
        fb::ExprNode::ColumnRef,
        "argument order is the call"
    );
}

#[test]
fn a_scalar_function_returning_a_decimal_carries_its_precision_and_scale() {
    let bytes = written(&Expr::ScalarFunction {
        name: "coalesce".to_string(),
        args: vec![column(0, "a")],
        return_type: DataType::Decimal128(22, 3),
        nullable: true,
    });
    let function = read(&bytes).node_as_scalar_function_expr_node().unwrap();
    assert_eq!(
        (
            function.return_decimal_precision(),
            function.return_decimal_scale()
        ),
        (22, 3),
        "the DataType enum cannot carry them, so a sum over this would recompute wrong"
    );
}

#[test]
fn nesting_survives_to_the_depth_a_finalize_reaches() {
    // The shape of an avg's finalize, which is the deepest thing the planner builds:
    // a divide of two casts, under a CASE, under a NOT.
    let quotient = Expr::binary(
        Expr::Cast {
            expr: Box::new(column(0, "sum")),
            target: DataType::Decimal128(25, 6),
        },
        BinaryOp::Divide,
        Expr::Cast {
            expr: Box::new(column(1, "count")),
            target: DataType::Decimal128(25, 0),
        },
        DataType::Decimal128(25, 6),
    );
    let expr = Expr::unary(
        UnaryOp::Not,
        Expr::Case {
            comparand: None,
            when_then: vec![(column(2, "flag"), quotient)],
            else_expr: None,
        },
    );
    let bytes = written(&expr);
    let case = read(&bytes)
        .node_as_unary_expr_node()
        .unwrap()
        .arg()
        .unwrap()
        .node_as_case_expr_node()
        .unwrap();
    let divide = case
        .when_thens()
        .unwrap()
        .get(0)
        .then()
        .unwrap()
        .node_as_binary_expr_node()
        .unwrap();
    assert_eq!(divide.op(), fb::BinaryOp::Divide);
    assert_eq!(divide.out_decimal_scale(), 6);
    let denominator = divide.right().unwrap().node_as_cast_expr_node().unwrap();
    assert_eq!(
        denominator.decimal_scale(),
        0,
        "the denominator is cast to an integer-valued decimal so cuDF's divide scale lands \
         where DataFusion declared it"
    );
}

/// A stddev's finalize, whole: the shape the fifth operator was appended for.
#[test]
fn a_stddev_finalize_writes_with_its_sqrt_under_the_case() {
    let expr = Expr::Case {
        comparand: None,
        when_then: vec![(
            column(0, "flag"),
            Expr::unary(UnaryOp::Sqrt, column(1, "m2")),
        )],
        else_expr: None,
    };
    let bytes = written(&expr);
    let then = read(&bytes)
        .node_as_case_expr_node()
        .unwrap()
        .when_thens()
        .unwrap()
        .get(0)
        .then()
        .unwrap();
    assert_eq!(
        then.node_as_unary_expr_node().unwrap().op(),
        fb::UnaryOp::Sqrt
    );
}
