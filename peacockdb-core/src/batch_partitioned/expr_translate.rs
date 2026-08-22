//! DataFusion `PhysicalExpr` → the mode's [`Expr`].
//!
//! One conscious decision per expression kind; an unrecognized kind is a plan-time error
//! naming it, never a silent pass-through. Types are read back off DataFusion — a binary
//! op's declared output type in particular, since cuDF's own fixed-point result scale
//! differs from DataFusion's.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr,
    Literal, NegativeExpr, NotExpr,
};
use datafusion::physical_plan::PhysicalExpr;

use super::error::PlanError;
use super::expr::{BinaryOp, ColumnRef, Expr, UnaryOp};

pub fn translate_expr(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Result<Expr, PlanError> {
    let any = expr.as_any();

    if let Some(col) = any.downcast_ref::<Column>() {
        return Ok(Expr::Column(ColumnRef {
            index: col.index() as u32,
            name: col.name().to_string(),
        }));
    }
    if let Some(lit) = any.downcast_ref::<Literal>() {
        return Ok(Expr::Literal(lit.value().clone()));
    }
    if let Some(bin) = any.downcast_ref::<BinaryExpr>() {
        let out_type = bin.data_type(input_schema).map_err(|e| {
            PlanError::Invalid(format!("binary expression has no output type: {e}"))
        })?;
        return Ok(Expr::binary(
            translate_expr(bin.left(), input_schema)?,
            translate_operator(bin.op())?,
            translate_expr(bin.right(), input_schema)?,
            out_type,
        ));
    }
    if let Some(not) = any.downcast_ref::<NotExpr>() {
        return Ok(Expr::unary(
            UnaryOp::Not,
            translate_expr(not.arg(), input_schema)?,
        ));
    }
    if let Some(is_null) = any.downcast_ref::<IsNullExpr>() {
        return Ok(Expr::unary(
            UnaryOp::IsNull,
            translate_expr(is_null.arg(), input_schema)?,
        ));
    }
    if let Some(is_not_null) = any.downcast_ref::<IsNotNullExpr>() {
        return Ok(Expr::unary(
            UnaryOp::IsNotNull,
            translate_expr(is_not_null.arg(), input_schema)?,
        ));
    }
    if let Some(neg) = any.downcast_ref::<NegativeExpr>() {
        return Ok(Expr::unary(
            UnaryOp::Negative,
            translate_expr(neg.arg(), input_schema)?,
        ));
    }
    if let Some(cast) = any.downcast_ref::<CastExpr>() {
        return Ok(Expr::Cast {
            expr: Box::new(translate_expr(cast.expr(), input_schema)?),
            target: cast.cast_type().clone(),
        });
    }
    if let Some(like) = any.downcast_ref::<LikeExpr>() {
        return Ok(Expr::Like {
            expr: Box::new(translate_expr(like.expr(), input_schema)?),
            pattern: Box::new(translate_expr(like.pattern(), input_schema)?),
            negated: like.negated(),
            case_insensitive: like.case_insensitive(),
        });
    }
    if let Some(case) = any.downcast_ref::<CaseExpr>() {
        let comparand = match case.expr() {
            Some(e) => Some(Box::new(translate_expr(e, input_schema)?)),
            None => None,
        };
        let mut when_then = Vec::with_capacity(case.when_then_expr().len());
        for (when, then) in case.when_then_expr() {
            when_then.push((
                translate_expr(when, input_schema)?,
                translate_expr(then, input_schema)?,
            ));
        }
        let else_expr = match case.else_expr() {
            Some(e) => Some(Box::new(translate_expr(e, input_schema)?)),
            None => None,
        };
        return Ok(Expr::Case {
            comparand,
            when_then,
            else_expr,
        });
    }
    if let Some(in_list) = any.downcast_ref::<InListExpr>() {
        return expand_in_list(in_list, input_schema);
    }
    if let Some(func) = any.downcast_ref::<ScalarFunctionExpr>() {
        let mut args = Vec::with_capacity(func.args().len());
        for arg in func.args() {
            args.push(translate_expr(arg, input_schema)?);
        }
        return Ok(Expr::ScalarFunction {
            name: func.name().to_string(),
            args,
            return_type: func.return_type().clone(),
            nullable: func.nullable(),
        });
    }

    Err(PlanError::Unsupported(format!("expression {expr} (#162)")))
}

/// `x IN (a, b)` becomes `(x = a) OR (x = b)`, and `NOT IN` its negation: cuDF's AST has
/// no IN opcode, so the lowering has to happen somewhere and the IR is where this mode
/// can state it. Legacy lowers the same shape in its wrapper rule.
fn expand_in_list(in_list: &InListExpr, input_schema: &Schema) -> Result<Expr, PlanError> {
    if in_list.list().is_empty() {
        return Err(PlanError::Unsupported("IN with an empty list".to_string()));
    }
    let target = translate_expr(in_list.expr(), input_schema)?;
    let mut chain: Option<Expr> = None;
    for item in in_list.list() {
        let eq = Expr::binary(
            target.clone(),
            BinaryOp::Eq,
            translate_expr(item, input_schema)?,
            DataType::Boolean,
        );
        chain = Some(match chain {
            None => eq,
            Some(acc) => Expr::binary(acc, BinaryOp::Or, eq, DataType::Boolean),
        });
    }
    let chain = chain.expect("the list is not empty");
    Ok(if in_list.negated() {
        Expr::unary(UnaryOp::Not, chain)
    } else {
        chain
    })
}

fn translate_operator(op: &Operator) -> Result<BinaryOp, PlanError> {
    Ok(match op {
        Operator::Eq => BinaryOp::Eq,
        Operator::NotEq => BinaryOp::NotEq,
        Operator::Lt => BinaryOp::Lt,
        Operator::LtEq => BinaryOp::LtEq,
        Operator::Gt => BinaryOp::Gt,
        Operator::GtEq => BinaryOp::GtEq,
        Operator::Plus => BinaryOp::Plus,
        Operator::Minus => BinaryOp::Minus,
        Operator::Multiply => BinaryOp::Multiply,
        Operator::Divide => BinaryOp::Divide,
        Operator::Modulo => BinaryOp::Modulo,
        Operator::And => BinaryOp::And,
        Operator::Or => BinaryOp::Or,
        Operator::BitwiseAnd => BinaryOp::BitwiseAnd,
        Operator::BitwiseOr => BinaryOp::BitwiseOr,
        Operator::BitwiseXor => BinaryOp::BitwiseXor,
        Operator::BitwiseShiftLeft => BinaryOp::BitwiseShiftLeft,
        Operator::BitwiseShiftRight => BinaryOp::BitwiseShiftRight,
        Operator::StringConcat => BinaryOp::StringConcat,
        Operator::IsDistinctFrom => BinaryOp::IsDistinctFrom,
        Operator::IsNotDistinctFrom => BinaryOp::IsNotDistinctFrom,
        other => return Err(PlanError::Unsupported(format!("binary operator {other} (#162)"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::Field;
    use datafusion::common::ScalarValue;
    use datafusion::functions::string::upper;
    use datafusion::physical_expr::expressions::TryCastExpr;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("price", DataType::Decimal128(20, 2), true),
            Field::new("s", DataType::Utf8, true),
        ])
    }

    fn col(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new(name, index))
    }

    fn lit(value: ScalarValue) -> Arc<dyn PhysicalExpr> {
        Arc::new(Literal::new(value))
    }

    fn translate(expr: Arc<dyn PhysicalExpr>) -> Result<Expr, PlanError> {
        translate_expr(&expr, &schema())
    }

    #[test]
    fn a_column_keeps_its_ordinal_and_its_name() {
        assert_eq!(translate(col("a", 0)).unwrap(), Expr::column(0, "a"));
    }

    #[test]
    fn a_literal_keeps_datafusions_scalar() {
        let value = ScalarValue::Decimal128(Some(1234), 20, 2);
        assert_eq!(translate(lit(value.clone())).unwrap(), Expr::Literal(value));
    }

    #[test]
    fn a_binary_op_carries_the_type_datafusion_declared() {
        let expr = Arc::new(BinaryExpr::new(
            col("price", 1),
            Operator::Divide,
            lit(ScalarValue::Decimal128(Some(2), 20, 2)),
        ));
        let Expr::Binary { op, out_type, .. } = translate(expr).unwrap() else {
            panic!("expected a binary expression");
        };
        assert_eq!(op, BinaryOp::Divide);
        // cuDF's own rule for a decimal divide gives scale s_l - s_r, which is 0 here;
        // what travels is the scale DataFusion derived.
        assert_eq!(out_type, DataType::Decimal128(26, 6));
    }

    #[test]
    fn the_four_datafusion_unaries_map_one_for_one() {
        let not = Arc::new(NotExpr::new(lit(ScalarValue::Boolean(Some(true)))));
        let is_null = Arc::new(IsNullExpr::new(col("a", 0)));
        let is_not_null = Arc::new(IsNotNullExpr::new(col("a", 0)));
        let negative = Arc::new(NegativeExpr::new(col("a", 0)));

        assert!(matches!(
            translate(not).unwrap(),
            Expr::Unary {
                op: UnaryOp::Not,
                ..
            }
        ));
        assert!(matches!(
            translate(is_null).unwrap(),
            Expr::Unary {
                op: UnaryOp::IsNull,
                ..
            }
        ));
        assert!(matches!(
            translate(is_not_null).unwrap(),
            Expr::Unary {
                op: UnaryOp::IsNotNull,
                ..
            }
        ));
        assert!(matches!(
            translate(negative).unwrap(),
            Expr::Unary {
                op: UnaryOp::Negative,
                ..
            }
        ));
    }

    #[test]
    fn a_cast_keeps_its_target_precision_and_scale() {
        let expr = Arc::new(CastExpr::new(
            col("a", 0),
            DataType::Decimal128(38, 6),
            None,
        ));
        assert_eq!(
            translate(expr).unwrap(),
            Expr::Cast {
                expr: Box::new(Expr::column(0, "a")),
                target: DataType::Decimal128(38, 6),
            }
        );
    }

    #[test]
    fn a_like_keeps_negation_and_case_sensitivity() {
        let expr = Arc::new(LikeExpr::new(
            true,
            true,
            col("s", 2),
            lit(ScalarValue::Utf8(Some("%x%".to_string()))),
        ));
        let Expr::Like {
            negated,
            case_insensitive,
            ..
        } = translate(expr).unwrap()
        else {
            panic!("expected a like expression");
        };
        assert!(negated && case_insensitive);
    }

    #[test]
    fn both_case_forms_translate_and_keep_their_branches() {
        let search = Arc::new(
            CaseExpr::try_new(
                None,
                vec![(
                    Arc::new(IsNullExpr::new(col("a", 0))) as Arc<dyn PhysicalExpr>,
                    lit(ScalarValue::Int64(Some(0))),
                )],
                Some(col("a", 0)),
            )
            .unwrap(),
        );
        let Expr::Case {
            comparand,
            when_then,
            else_expr,
        } = translate(search).unwrap()
        else {
            panic!("expected a case expression");
        };
        assert!(comparand.is_none() && when_then.len() == 1 && else_expr.is_some());

        let value = Arc::new(
            CaseExpr::try_new(
                Some(col("a", 0)),
                vec![(
                    lit(ScalarValue::Int64(Some(1))),
                    lit(ScalarValue::Int64(Some(2))),
                )],
                None,
            )
            .unwrap(),
        );
        let Expr::Case { comparand, .. } = translate(value).unwrap() else {
            panic!("expected a case expression");
        };
        assert_eq!(comparand.as_deref(), Some(&Expr::column(0, "a")));
    }

    #[test]
    fn an_in_list_lowers_to_an_or_chain() {
        let expr = Arc::new(InListExpr::new(
            col("a", 0),
            vec![
                lit(ScalarValue::Int64(Some(1))),
                lit(ScalarValue::Int64(Some(2))),
            ],
            false,
            None,
        ));
        let eq = |value: i64| {
            Expr::binary(
                Expr::column(0, "a"),
                BinaryOp::Eq,
                Expr::Literal(ScalarValue::Int64(Some(value))),
                DataType::Boolean,
            )
        };
        assert_eq!(
            translate(expr).unwrap(),
            Expr::binary(eq(1), BinaryOp::Or, eq(2), DataType::Boolean)
        );
    }

    #[test]
    fn a_negated_in_list_wraps_the_chain_in_a_not() {
        let expr = Arc::new(InListExpr::new(
            col("a", 0),
            vec![lit(ScalarValue::Int64(Some(1)))],
            true,
            None,
        ));
        assert!(matches!(
            translate(expr).unwrap(),
            Expr::Unary {
                op: UnaryOp::Not,
                ..
            }
        ));
    }

    #[test]
    fn an_empty_in_list_is_refused_rather_than_lowered_to_nothing() {
        let expr = Arc::new(InListExpr::new(col("a", 0), vec![], false, None));
        assert!(matches!(
            translate(expr).unwrap_err(),
            PlanError::Unsupported(_)
        ));
    }

    #[test]
    fn a_scalar_function_keeps_its_name_and_return_type() {
        let expr = Arc::new(ScalarFunctionExpr::new(
            "upper",
            upper(),
            vec![col("s", 2)],
            DataType::Utf8,
        ));
        let Expr::ScalarFunction {
            name,
            args,
            return_type,
            ..
        } = translate(expr).unwrap()
        else {
            panic!("expected a scalar function");
        };
        assert_eq!(
            (name.as_str(), args.len(), return_type),
            ("upper", 1, DataType::Utf8)
        );
    }

    #[test]
    fn an_unrecognized_expression_kind_is_refused_and_named() {
        let expr = Arc::new(TryCastExpr::new(col("a", 0), DataType::Int32));
        let err = translate(expr).unwrap_err();
        assert!(
            matches!(&err, PlanError::Unsupported(what) if what.contains("TRY_CAST")),
            "{err}"
        );
    }

    #[test]
    fn an_unrecognized_binary_operator_is_refused_and_named() {
        let expr = Arc::new(BinaryExpr::new(
            col("s", 2),
            Operator::RegexMatch,
            lit(ScalarValue::Utf8(Some("^x".to_string()))),
        ));
        let err = translate(expr).unwrap_err();
        assert!(
            matches!(&err, PlanError::Unsupported(what) if what.contains('~')),
            "{err}"
        );
    }
}
