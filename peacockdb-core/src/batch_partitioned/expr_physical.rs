//! The mode's [`Expr`] → DataFusion `PhysicalExpr`, the inverse of [`expr_translate`].
//!
//! The CPU backend relays to DataFusion, so it needs the expressions back in DataFusion's
//! own vocabulary. Going back is not free: a column reference is an ordinal into a child
//! whose column order this mode decided, so the name that rides beside it is checked
//! against the schema at that position rather than trusted — the mismatch this catches is
//! the one a rebase gets wrong, and it is silent on the device.
//!
//! [`expr_translate`]: super::expr_translate

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Schema as ArrowSchema};
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, IsNotNullExpr, IsNullExpr, LikeExpr, Literal,
    NegativeExpr, NotExpr,
};
use datafusion::physical_plan::PhysicalExpr;

use super::error::PlanError;
use super::expr::{BinaryOp, Expr, NamedExpr, UnaryOp};

/// `Sqrt` has no DataFusion unary, so it resolves through the registry like any other
/// function — the same `sqrt` a query would have used, rather than a second implementation.
const SQRT: &str = "sqrt";

pub fn physical_expr(
    expr: &Expr,
    input: &ArrowSchema,
    registry: &dyn FunctionRegistry,
) -> Result<Arc<dyn PhysicalExpr>, PlanError> {
    match expr {
        Expr::Column(column) => {
            let index = column.index as usize;
            let field = input.fields().get(index).ok_or_else(|| {
                PlanError::Invalid(format!(
                    "column `{}` is at {index}, and the input has {} columns",
                    column.name,
                    input.fields().len()
                ))
            })?;
            if field.name() != &column.name {
                return Err(PlanError::Invalid(format!(
                    "column {index} is `{}` here and `{}` in the input",
                    column.name,
                    field.name()
                )));
            }
            Ok(Arc::new(Column::new(&column.name, index)))
        }
        Expr::Literal(value) => Ok(Arc::new(Literal::new(value.clone()))),
        Expr::Binary { left, op, right, .. } => Ok(Arc::new(BinaryExpr::new(
            physical_expr(left, input, registry)?,
            operator(*op),
            physical_expr(right, input, registry)?,
        ))),
        Expr::Unary { op, arg } => {
            let arg = physical_expr(arg, input, registry)?;
            Ok(match op {
                UnaryOp::Not => Arc::new(NotExpr::new(arg)),
                UnaryOp::IsNull => Arc::new(IsNullExpr::new(arg)),
                UnaryOp::IsNotNull => Arc::new(IsNotNullExpr::new(arg)),
                UnaryOp::Negative => Arc::new(NegativeExpr::new(arg)),
                UnaryOp::Sqrt => scalar_function(SQRT, vec![arg], DataType::Float64, registry)?,
            })
        }
        Expr::Cast { expr, target } => Ok(Arc::new(CastExpr::new(
            physical_expr(expr, input, registry)?,
            target.clone(),
            None,
        ))),
        Expr::Like {
            expr,
            pattern,
            negated,
            case_insensitive,
        } => Ok(Arc::new(LikeExpr::new(
            *negated,
            *case_insensitive,
            physical_expr(expr, input, registry)?,
            physical_expr(pattern, input, registry)?,
        ))),
        Expr::Case {
            comparand,
            when_then,
            else_expr,
        } => {
            let comparand = comparand
                .as_ref()
                .map(|expr| physical_expr(expr, input, registry))
                .transpose()?;
            let mut arms = Vec::with_capacity(when_then.len());
            for (when, then) in when_then {
                arms.push((
                    physical_expr(when, input, registry)?,
                    physical_expr(then, input, registry)?,
                ));
            }
            let otherwise = else_expr
                .as_ref()
                .map(|expr| physical_expr(expr, input, registry))
                .transpose()?;
            CaseExpr::try_new(comparand, arms, otherwise)
                .map(|case| Arc::new(case) as Arc<dyn PhysicalExpr>)
                .map_err(|error| PlanError::Invalid(format!("case: {error}")))
        }
        Expr::ScalarFunction {
            name,
            args,
            return_type,
            ..
        } => {
            let mut lowered = Vec::with_capacity(args.len());
            for arg in args {
                lowered.push(physical_expr(arg, input, registry)?);
            }
            scalar_function(name, lowered, return_type.clone(), registry)
        }
    }
}

/// A project list, in the shape `ProjectionExec` takes.
pub fn physical_projection(
    exprs: &[NamedExpr],
    input: &ArrowSchema,
    registry: &dyn FunctionRegistry,
) -> Result<Vec<(Arc<dyn PhysicalExpr>, String)>, PlanError> {
    exprs
        .iter()
        .map(|named| {
            Ok((
                physical_expr(&named.expr, input, registry)?,
                named.name.clone(),
            ))
        })
        .collect()
}

fn scalar_function(
    name: &str,
    args: Vec<Arc<dyn PhysicalExpr>>,
    return_type: DataType,
    registry: &dyn FunctionRegistry,
) -> Result<Arc<dyn PhysicalExpr>, PlanError> {
    let udf = registry.udf(name).map_err(|error| {
        PlanError::Unsupported(format!("`{name}` is not in this session's functions: {error}"))
    })?;
    Ok(Arc::new(ScalarFunctionExpr::new(
        name,
        udf,
        args,
        return_type,
    )))
}

fn operator(op: BinaryOp) -> Operator {
    match op {
        BinaryOp::Eq => Operator::Eq,
        BinaryOp::NotEq => Operator::NotEq,
        BinaryOp::Lt => Operator::Lt,
        BinaryOp::LtEq => Operator::LtEq,
        BinaryOp::Gt => Operator::Gt,
        BinaryOp::GtEq => Operator::GtEq,
        BinaryOp::Plus => Operator::Plus,
        BinaryOp::Minus => Operator::Minus,
        BinaryOp::Multiply => Operator::Multiply,
        BinaryOp::Divide => Operator::Divide,
        BinaryOp::Modulo => Operator::Modulo,
        BinaryOp::And => Operator::And,
        BinaryOp::Or => Operator::Or,
        BinaryOp::BitwiseAnd => Operator::BitwiseAnd,
        BinaryOp::BitwiseOr => Operator::BitwiseOr,
        BinaryOp::BitwiseXor => Operator::BitwiseXor,
        BinaryOp::BitwiseShiftLeft => Operator::BitwiseShiftLeft,
        BinaryOp::BitwiseShiftRight => Operator::BitwiseShiftRight,
        BinaryOp::StringConcat => Operator::StringConcat,
        BinaryOp::IsDistinctFrom => Operator::IsDistinctFrom,
        BinaryOp::IsNotDistinctFrom => Operator::IsNotDistinctFrom,
    }
}

#[cfg(test)]
mod tests;
