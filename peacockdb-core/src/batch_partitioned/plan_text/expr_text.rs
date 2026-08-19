//! Rendering one expression as text. The ordinal is authoritative and a name is printed
//! beside it, so `name@ordinal` is the form everywhere — and a literal is quoted on the
//! same rule as a name, since a rendered value can hold the comma the list is cut on.

use std::fmt::Write as _;

use datafusion::common::ScalarValue;

use super::super::expr::{BinaryOp, ColumnRef, Expr, UnaryOp};
use super::super::nodes::join::{JoinFilterColumn, JoinSide};
use super::super::schema::Schema;
use super::{name_at, quoted, type_text};

pub fn expr_text(expr: &Expr) -> String {
    // An ordinary reference indexes the node's input, which is the line below it.
    expr_text_resolved(expr, &|reference| {
        format!("{}@{}", quoted(&reference.name), reference.index)
    })
}

/// The same rendering with the column form supplied, because a join filter's ordinals
/// index a table of the filter's own that appears on no line.
fn expr_text_resolved(expr: &Expr, column: &dyn Fn(&ColumnRef) -> String) -> String {
    let nested = |expr: &Expr| nested_expr_resolved(expr, column);
    let plain = |expr: &Expr| expr_text_resolved(expr, column);
    match expr {
        Expr::Column(reference) => column(reference),
        Expr::Literal(value) => quoted(&literal_text(value)),
        Expr::Binary {
            left, op, right, ..
        } => format!("{} {} {}", nested(left), binary_op_text(*op), nested(right)),
        Expr::Unary { op, arg } => match op {
            UnaryOp::Not => format!("NOT {}", nested(arg)),
            UnaryOp::IsNull => format!("{} IS NULL", nested(arg)),
            UnaryOp::IsNotNull => format!("{} IS NOT NULL", nested(arg)),
            UnaryOp::Negative => format!("-{}", nested(arg)),
            UnaryOp::Sqrt => format!("sqrt({})", plain(arg)),
        },
        Expr::Cast { expr, target } => {
            format!("CAST({} AS {})", plain(expr), type_text(target))
        }
        Expr::Like {
            expr,
            pattern,
            negated,
            case_insensitive,
        } => format!(
            "{} {}{} {}",
            nested(expr),
            if *negated { "NOT " } else { "" },
            if *case_insensitive { "ILIKE" } else { "LIKE" },
            nested(pattern)
        ),
        Expr::Case {
            comparand,
            when_then,
            else_expr,
        } => {
            let mut text = "CASE".to_string();
            if let Some(comparand) = comparand {
                let _ = write!(text, " {}", plain(comparand));
            }
            for (when, then) in when_then {
                let _ = write!(text, " WHEN {} THEN {}", plain(when), plain(then));
            }
            if let Some(otherwise) = else_expr {
                let _ = write!(text, " ELSE {}", plain(otherwise));
            }
            text + " END"
        }
        Expr::ScalarFunction { name, args, .. } => format!(
            "{name}({})",
            args.iter().map(plain).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// A sub-expression that is itself an operator is parenthesized, so precedence is read off
/// the line rather than assumed.
fn nested_expr_resolved(expr: &Expr, column: &dyn Fn(&ColumnRef) -> String) -> String {
    match expr {
        Expr::Binary { .. } | Expr::Case { .. } => {
            format!("({})", expr_text_resolved(expr, column))
        }
        other => expr_text_resolved(other, column),
    }
}

/// A join filter's reference, resolved onto the side it came from and that side's own
/// ordinal — `k@build:0` is column 0 of the build child, whose schema is on its own line.
/// Left as its filter-schema ordinal only where the map is short, which validation refuses.
pub(super) fn join_filter_text(
    filter: &Expr,
    columns: &[JoinFilterColumn],
    build: Option<&Schema>,
    probe: Option<&Schema>,
) -> String {
    expr_text_resolved(
        filter,
        &|reference| match columns.get(reference.index as usize) {
            Some(mapped) => {
                let (side, schema) = match mapped.side {
                    JoinSide::Build => ("build", build),
                    JoinSide::Probe => ("probe", probe),
                };
                format!("{}@{side}:{}", name_at(schema, mapped.index), mapped.index)
            }
            None => format!("{}@{}", quoted(&reference.name), reference.index),
        },
    )
}

/// A decimal scalar prints as its value, not as the triple its `Display` gives: a plan
/// reader compares a literal against the column beside it.
/// The value alone, without the type wrapper `ScalarValue`'s own Display prints for some
/// variants. Quoted by its caller on the same rule as a name: a string literal can hold a
/// comma — tpcds q66 concatenates one — and an unquotable token in a comma-separated list
/// is a golden a reader cannot tokenize.
fn literal_text(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Decimal128(Some(unscaled), _, scale) => decimal_text(*unscaled, *scale),
        ScalarValue::Decimal256(Some(unscaled), _, scale) => decimal_text(
            unscaled.to_string().parse::<i128>().unwrap_or_default(),
            *scale,
        ),
        ScalarValue::IntervalYearMonth(Some(months)) => interval_text(*months, 0, 0),
        ScalarValue::IntervalDayTime(Some(interval)) => {
            interval_text(0, interval.days, interval.milliseconds as i64 * 1_000_000)
        }
        ScalarValue::IntervalMonthDayNano(Some(interval)) => {
            interval_text(interval.months, interval.days, interval.nanoseconds)
        }
        other => other.to_string(),
    }
}

/// The parts that are not zero, in the units they are declared in — `90 days` where the
/// struct form spends sixty characters saying the same thing. All-zero prints as `0 days`,
/// since an interval of nothing is still an interval.
fn interval_text(months: i32, days: i32, nanoseconds: i64) -> String {
    let parts: Vec<String> = [
        (months as i64, "mons"),
        (days as i64, "days"),
        (nanoseconds, "nanos"),
    ]
    .into_iter()
    .filter(|(value, _)| *value != 0)
    .map(|(value, unit)| format!("{value} {unit}"))
    .collect();
    if parts.is_empty() {
        "0 days".to_string()
    } else {
        parts.join(" ")
    }
}

fn decimal_text(unscaled: i128, scale: i8) -> String {
    if scale <= 0 {
        return unscaled.to_string();
    }
    let divisor = 10i128.pow(scale as u32);
    let (sign, magnitude) = if unscaled < 0 {
        ("-", -unscaled)
    } else {
        ("", unscaled)
    };
    format!(
        "{sign}{}.{:0width$}",
        magnitude / divisor,
        magnitude % divisor,
        width = scale as usize
    )
}

fn binary_op_text(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Eq => "=",
        BinaryOp::NotEq => "!=",
        BinaryOp::Lt => "<",
        BinaryOp::LtEq => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::GtEq => ">=",
        BinaryOp::Plus => "+",
        BinaryOp::Minus => "-",
        BinaryOp::Multiply => "*",
        BinaryOp::Divide => "/",
        BinaryOp::Modulo => "%",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::BitwiseAnd => "&",
        BinaryOp::BitwiseOr => "|",
        BinaryOp::BitwiseXor => "^",
        BinaryOp::BitwiseShiftLeft => "<<",
        BinaryOp::BitwiseShiftRight => ">>",
        BinaryOp::StringConcat => "||",
        BinaryOp::IsDistinctFrom => "IS DISTINCT FROM",
        BinaryOp::IsNotDistinctFrom => "IS NOT DISTINCT FROM",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::DataType;
    #[test]
    fn an_interval_literal_prints_the_parts_that_are_not_zero() {
        use datafusion::arrow::datatypes::IntervalMonthDayNano;
        let ninety_days = Expr::Literal(ScalarValue::IntervalMonthDayNano(Some(
            IntervalMonthDayNano::new(0, 90, 0),
        )));
        // Quoted on the same rule as a name: an interval reads as several tokens, and a
        // plan line is a comma-separated list.
        assert_eq!(expr_text(&ninety_days), "`90 days`");
        let mixed = Expr::Literal(ScalarValue::IntervalMonthDayNano(Some(
            IntervalMonthDayNano::new(2, 1, 500),
        )));
        assert_eq!(expr_text(&mixed), "`2 mons 1 days 500 nanos`");
        let nothing = Expr::Literal(ScalarValue::IntervalMonthDayNano(Some(
            IntervalMonthDayNano::new(0, 0, 0),
        )));
        assert_eq!(expr_text(&nothing), "`0 days`");
    }

    #[test]
    fn a_decimal_literal_prints_as_a_value_rather_than_its_parts() {
        let money = Expr::Literal(ScalarValue::Decimal128(Some(-123_456), 15, 2));
        assert_eq!(expr_text(&money), "-1234.56");
        let whole = Expr::Literal(ScalarValue::Decimal128(Some(7), 15, 0));
        assert_eq!(expr_text(&whole), "7");
    }

    #[test]
    fn every_expression_form_renders_readably() {
        use crate::batch_partitioned::expr::{BinaryOp, UnaryOp};
        let column = Expr::column(2, "s");
        let cast = Expr::Cast {
            expr: Box::new(Expr::column(0, "a")),
            target: DataType::Decimal128(38, 6),
        };
        assert_eq!(expr_text(&cast), "CAST(a@0 AS Decimal128(38,6))");

        let like = Expr::Like {
            expr: Box::new(column.clone()),
            pattern: Box::new(Expr::Literal(ScalarValue::Utf8(Some("%x%".to_string())))),
            negated: true,
            case_insensitive: false,
        };
        assert_eq!(expr_text(&like), "s@2 NOT LIKE %x%");

        // A nested operator is parenthesized, so precedence is read off the line.
        let nested = Expr::binary(
            Expr::binary(
                Expr::column(0, "a"),
                BinaryOp::Plus,
                Expr::column(1, "b"),
                DataType::Int64,
            ),
            BinaryOp::Gt,
            Expr::Literal(ScalarValue::Int64(Some(3))),
            DataType::Boolean,
        );
        assert_eq!(expr_text(&nested), "(a@0 + b@1) > 3");

        let guard = Expr::Case {
            comparand: None,
            when_then: vec![(
                Expr::unary(UnaryOp::IsNull, Expr::column(0, "a")),
                Expr::Literal(ScalarValue::Int64(None)),
            )],
            else_expr: Some(Box::new(Expr::unary(UnaryOp::Sqrt, Expr::column(0, "a")))),
        };
        assert_eq!(
            expr_text(&guard),
            "CASE WHEN a@0 IS NULL THEN NULL ELSE sqrt(a@0) END"
        );

        let call = Expr::ScalarFunction {
            name: "date_part".to_string(),
            args: vec![
                Expr::Literal(ScalarValue::Utf8(Some("year".to_string()))),
                column,
            ],
            return_type: DataType::Int32,
            nullable: true,
        };
        assert_eq!(expr_text(&call), "date_part(year, s@2)");
    }
}
