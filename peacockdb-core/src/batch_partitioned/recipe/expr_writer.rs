//! This mode's [`Expr`] into the wire's `fb::Expr`.
//!
//! Separate from `plan_serializer.rs`'s writer, which takes an `Arc<dyn PhysicalExpr>`
//! and downcasts: that file's job is a DataFusion plan, and a second input shape is how
//! it would stop having one. The scalar and type helpers are shared, so a literal or a
//! decimal is written by exactly one piece of code whichever plan asked for it.
//!
//! Every variant writes, `Sqrt` included: the schema gained it so that a finalizing
//! aggregate carries its own finalize rather than leaving the arithmetic to the C++.

use datafusion::arrow::datatypes::DataType;
use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::generated::gpu_plan_generated::peacock::plan as fb;
use crate::plan_serializer::{convert_data_type, serialize_scalar_value};

use super::super::error::PlanError;
use super::super::expr::{BinaryOp, Expr, UnaryOp};

/// Write one expression, children first, as FlatBuffers requires.
pub(crate) fn write_expr<'a>(
    b: &mut FlatBufferBuilder<'a>,
    expr: &Expr,
) -> Result<WIPOffset<fb::Expr<'a>>, PlanError> {
    let (node_type, node) = match expr {
        Expr::Column(reference) => {
            let name = b.create_string(&reference.name);
            let column = fb::ColumnRef::create(
                b,
                &fb::ColumnRefArgs {
                    index: reference.index,
                    name: Some(name),
                },
            );
            (fb::ExprNode::ColumnRef, column.as_union_value())
        }
        Expr::Literal(value) => {
            // The ticket rides the message because it is what a reader of a golden gets:
            // the wire's `ScalarValue` has no interval, and one corpus residual adds one to
            // a column, so folding never reaches it (#168).
            let scalar = serialize_scalar_value(b, value)
                .map_err(|why| PlanError::Unsupported(format!("{why} (#168)")))?;
            let literal = fb::LiteralExpr::create(
                b,
                &fb::LiteralExprArgs {
                    value: Some(scalar),
                },
            );
            (fb::ExprNode::LiteralExpr, literal.as_union_value())
        }
        Expr::Binary {
            left,
            op,
            right,
            out_type,
        } => {
            let left = write_expr(b, left)?;
            let right = write_expr(b, right)?;
            // The declared decimal output travels with the op: cuDF derives its own
            // fixed-point result scale, division most visibly, and the C++ reproduces
            // DataFusion's from these two rather than re-deriving the coercion.
            let (out_decimal_precision, out_decimal_scale) = decimal_parts(out_type);
            let binary = fb::BinaryExprNode::create(
                b,
                &fb::BinaryExprNodeArgs {
                    left: Some(left),
                    op: binary_op(*op),
                    right: Some(right),
                    out_decimal_precision,
                    out_decimal_scale,
                },
            );
            (fb::ExprNode::BinaryExprNode, binary.as_union_value())
        }
        Expr::Unary { op, arg } => {
            let arg = write_expr(b, arg)?;
            let unary = fb::UnaryExprNode::create(
                b,
                &fb::UnaryExprNodeArgs {
                    op: unary_op(*op),
                    arg: Some(arg),
                },
            );
            (fb::ExprNode::UnaryExprNode, unary.as_union_value())
        }
        Expr::Cast { expr, target } => {
            let inner = write_expr(b, expr)?;
            let (decimal_precision, decimal_scale) = decimal_parts(target);
            let cast = fb::CastExprNode::create(
                b,
                &fb::CastExprNodeArgs {
                    expr: Some(inner),
                    target_type: data_type(target)?,
                    decimal_precision,
                    decimal_scale,
                },
            );
            (fb::ExprNode::CastExprNode, cast.as_union_value())
        }
        Expr::Like {
            expr,
            pattern,
            negated,
            case_insensitive,
        } => {
            let inner = write_expr(b, expr)?;
            let pattern = write_expr(b, pattern)?;
            let like = fb::LikeExprNode::create(
                b,
                &fb::LikeExprNodeArgs {
                    expr: Some(inner),
                    pattern: Some(pattern),
                    negated: *negated,
                    case_insensitive: *case_insensitive,
                },
            );
            (fb::ExprNode::LikeExprNode, like.as_union_value())
        }
        Expr::Case {
            comparand,
            when_then,
            else_expr,
        } => {
            let comparand = comparand
                .as_ref()
                .map(|expr| write_expr(b, expr))
                .transpose()?;
            let mut pairs = Vec::with_capacity(when_then.len());
            for (when, then) in when_then {
                let when = write_expr(b, when)?;
                let then = write_expr(b, then)?;
                pairs.push(fb::CaseWhenThen::create(
                    b,
                    &fb::CaseWhenThenArgs {
                        when: Some(when),
                        then: Some(then),
                    },
                ));
            }
            let pairs = b.create_vector(&pairs);
            let otherwise = else_expr
                .as_ref()
                .map(|expr| write_expr(b, expr))
                .transpose()?;
            let case = fb::CaseExprNode::create(
                b,
                &fb::CaseExprNodeArgs {
                    expr: comparand,
                    when_thens: Some(pairs),
                    else_expr: otherwise,
                },
            );
            (fb::ExprNode::CaseExprNode, case.as_union_value())
        }
        Expr::ScalarFunction {
            name,
            args,
            return_type,
            nullable,
        } => {
            let mut written = Vec::with_capacity(args.len());
            for arg in args {
                written.push(write_expr(b, arg)?);
            }
            let args = b.create_vector(&written);
            let name = b.create_string(name);
            let (return_decimal_precision, return_decimal_scale) = decimal_parts(return_type);
            let function = fb::ScalarFunctionExprNode::create(
                b,
                &fb::ScalarFunctionExprNodeArgs {
                    name: Some(name),
                    args: Some(args),
                    return_type: data_type(return_type)?,
                    return_decimal_precision,
                    return_decimal_scale,
                    nullable: *nullable,
                },
            );
            (
                fb::ExprNode::ScalarFunctionExprNode,
                function.as_union_value(),
            )
        }
    };
    Ok(fb::Expr::create(
        b,
        &fb::ExprArgs {
            node_type,
            node: Some(node),
        },
    ))
}

/// Precision and scale, or `(0, 0)` — which the wire reads as "not a decimal", the same
/// convention `plan_serializer` writes.
fn decimal_parts(data_type: &DataType) -> (u8, i8) {
    match data_type {
        DataType::Decimal128(precision, scale) => (*precision, *scale),
        _ => (0, 0),
    }
}

fn data_type(data_type: &DataType) -> Result<fb::DataType, PlanError> {
    convert_data_type(data_type).map_err(PlanError::Unsupported)
}

fn binary_op(op: BinaryOp) -> fb::BinaryOp {
    match op {
        BinaryOp::Eq => fb::BinaryOp::Eq,
        BinaryOp::NotEq => fb::BinaryOp::NotEq,
        BinaryOp::Lt => fb::BinaryOp::Lt,
        BinaryOp::LtEq => fb::BinaryOp::LtEq,
        BinaryOp::Gt => fb::BinaryOp::Gt,
        BinaryOp::GtEq => fb::BinaryOp::GtEq,
        BinaryOp::Plus => fb::BinaryOp::Plus,
        BinaryOp::Minus => fb::BinaryOp::Minus,
        BinaryOp::Multiply => fb::BinaryOp::Multiply,
        BinaryOp::Divide => fb::BinaryOp::Divide,
        BinaryOp::Modulo => fb::BinaryOp::Modulo,
        BinaryOp::And => fb::BinaryOp::And,
        BinaryOp::Or => fb::BinaryOp::Or,
        BinaryOp::BitwiseAnd => fb::BinaryOp::BitwiseAnd,
        BinaryOp::BitwiseOr => fb::BinaryOp::BitwiseOr,
        BinaryOp::BitwiseXor => fb::BinaryOp::BitwiseXor,
        BinaryOp::BitwiseShiftLeft => fb::BinaryOp::BitwiseShiftLeft,
        BinaryOp::BitwiseShiftRight => fb::BinaryOp::BitwiseShiftRight,
        BinaryOp::StringConcat => fb::BinaryOp::StringConcat,
        BinaryOp::IsDistinctFrom => fb::BinaryOp::IsDistinctFrom,
        BinaryOp::IsNotDistinctFrom => fb::BinaryOp::IsNotDistinctFrom,
    }
}

fn unary_op(op: UnaryOp) -> fb::UnaryOp {
    match op {
        UnaryOp::Not => fb::UnaryOp::Not,
        UnaryOp::IsNull => fb::UnaryOp::IsNull,
        UnaryOp::IsNotNull => fb::UnaryOp::IsNotNull,
        UnaryOp::Negative => fb::UnaryOp::Negative,
        UnaryOp::Sqrt => fb::UnaryOp::Sqrt,
    }
}

#[cfg(test)]
mod tests;
