//! The aggregate vocabulary and the decomposition registry.
//!
//! An aggregate node carries no phase: it declares aggregators over its input and,
//! where it finishes the aggregate, one expression per output column. The split into
//! init / merge / finalize is this table, and adding an aggregate is a row here rather
//! than an arm in C++. State *types* come from DataFusion's `state_fields()`, so the
//! split cannot drift from the one DataFusion planned; the state *names* are ours,
//! since the golden and every later reference read them.

use datafusion::arrow::datatypes::{DataType, Field};

use super::error::PlanError;
use super::expr::{BinaryOp, Expr, UnaryOp};

/// What sql asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Sum,
    Min,
    Max,
    Count,
    Avg,
    Stddev,
    Var,
}

/// What a node runs. `Avg` is never one — decomposing it is the point — and `MergeM2`
/// is never an `AggFunc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanAgg {
    Sum,
    Min,
    Max,
    Count,
    Mean,
    M2,
    MergeM2,
}

/// How a state merges. `Combined` exists only because `merge_m2` is not a per-column
/// reduction: it needs the count-weighted mean and the cross term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Merge {
    PerColumn(&'static [PlanAgg]),
    Combined(PlanAgg),
}

/// `state` pairs each column's name suffix with the aggregator producing it, so a column
/// and its aggregator cannot desync. `merge` is listed rather than derived: the rule
/// would be "the same aggregator, except count merges by sum", and that exception is the
/// whole content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decomposition {
    pub state: &'static [(&'static str, PlanAgg)],
    pub merge: Merge,
}

/// One aggregate as sql wrote it: the function, and the `ddof` that separates the sample
/// forms from the population ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AggSpec {
    pub func: AggFunc,
    pub ddof: u32,
}

pub fn resolve(name: &str) -> Result<AggSpec, PlanError> {
    let (func, ddof) = match name {
        "sum" => (AggFunc::Sum, 0),
        "min" => (AggFunc::Min, 0),
        "max" => (AggFunc::Max, 0),
        "count" => (AggFunc::Count, 0),
        "avg" => (AggFunc::Avg, 0),
        "stddev" | "stddev_samp" => (AggFunc::Stddev, 1),
        "stddev_pop" => (AggFunc::Stddev, 0),
        "var" | "var_samp" | "variance" => (AggFunc::Var, 1),
        "var_pop" => (AggFunc::Var, 0),
        other => return Err(PlanError::Unsupported(format!("aggregate {other}"))),
    };
    Ok(AggSpec { func, ddof })
}

pub fn decomposition(func: AggFunc) -> Decomposition {
    const WELFORD: Decomposition = Decomposition {
        state: &[
            ("$count", PlanAgg::Count),
            ("$mean", PlanAgg::Mean),
            ("$m2", PlanAgg::M2),
        ],
        merge: Merge::Combined(PlanAgg::MergeM2),
    };
    match func {
        AggFunc::Sum => Decomposition {
            state: &[("", PlanAgg::Sum)],
            merge: Merge::PerColumn(&[PlanAgg::Sum]),
        },
        AggFunc::Min => Decomposition {
            state: &[("", PlanAgg::Min)],
            merge: Merge::PerColumn(&[PlanAgg::Min]),
        },
        AggFunc::Max => Decomposition {
            state: &[("", PlanAgg::Max)],
            merge: Merge::PerColumn(&[PlanAgg::Max]),
        },
        // A count merges by SUM. The one place where naming the merge separately from
        // the init is the difference between a right and a wrong answer.
        AggFunc::Count => Decomposition {
            state: &[("", PlanAgg::Count)],
            merge: Merge::PerColumn(&[PlanAgg::Sum]),
        },
        AggFunc::Avg => Decomposition {
            state: &[("$sum", PlanAgg::Sum), ("$count", PlanAgg::Count)],
            merge: Merge::PerColumn(&[PlanAgg::Sum, PlanAgg::Sum]),
        },
        AggFunc::Stddev | AggFunc::Var => WELFORD,
    }
}

/// One aggregator call: what it runs, over which expressions, and the columns it
/// produces. `merge_m2` is the reason `outputs` is a list — it returns its three state
/// columns together.
#[derive(Debug, Clone, PartialEq)]
pub struct AggCall {
    pub func: PlanAgg,
    pub args: Vec<Expr>,
    pub outputs: Vec<Field>,
}

/// The expression that turns merged state into the aggregate's output column. A rename
/// for the five simple aggregates, a divide for `avg`, and a `CASE` over a `sqrt` for
/// the Welford pair — all of them ordinary IR, which is what replaces the hardwired
/// `avg_div` and `std_finalize` arms.
pub fn finalize(spec: AggSpec, state: &[Field], state_at: u32, out_type: &DataType) -> Expr {
    let column = |offset: usize| Expr::column(state_at + offset as u32, state[offset].name());
    match spec.func {
        AggFunc::Sum | AggFunc::Min | AggFunc::Max | AggFunc::Count => column(0),
        AggFunc::Avg => {
            // The denominator is cast to an exact integer-valued decimal, so cuDF's own
            // divide scale (s_left - s_right) lands on the scale DataFusion declared.
            let (num_type, den_type) = match out_type {
                DataType::Decimal128(p, s) => {
                    (DataType::Decimal128(*p, *s), DataType::Decimal128(*p, 0))
                }
                other => (other.clone(), other.clone()),
            };
            Expr::binary(
                Expr::Cast {
                    expr: Box::new(column(0)),
                    target: num_type,
                },
                BinaryOp::Divide,
                Expr::Cast {
                    expr: Box::new(column(1)),
                    target: den_type,
                },
                out_type.clone(),
            )
        }
        AggFunc::Stddev | AggFunc::Var => {
            let ddof = Expr::Literal(datafusion::common::ScalarValue::Int64(Some(
                spec.ddof as i64,
            )));
            let denominator = Expr::binary(
                column(0),
                BinaryOp::Minus,
                ddof.clone(),
                state[0].data_type().clone(),
            );
            let quotient = Expr::binary(
                column(2),
                BinaryOp::Divide,
                denominator.clone(),
                out_type.clone(),
            );
            let value = match spec.func {
                AggFunc::Stddev => Expr::unary(UnaryOp::Sqrt, quotient),
                _ => quotient,
            };
            // A group with count <= ddof has no dispersion to report, so it is NULL
            // rather than a division by zero or a root of a negative.
            Expr::Case {
                comparand: None,
                when_then: vec![(
                    Expr::binary(
                        denominator,
                        BinaryOp::LtEq,
                        Expr::Literal(datafusion::common::ScalarValue::Int64(Some(0))),
                        DataType::Boolean,
                    ),
                    Expr::Literal(null_of(out_type)),
                )],
                else_expr: Some(Box::new(value)),
            }
        }
    }
}

fn null_of(data_type: &DataType) -> datafusion::common::ScalarValue {
    datafusion::common::ScalarValue::try_from(data_type)
        .expect("an aggregate output type has a null scalar")
}
