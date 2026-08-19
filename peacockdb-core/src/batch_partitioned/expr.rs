//! The mode's expression IR.
//!
//! Types and literals are DataFusion's own — the coercions and the decimal precision and
//! scale it derived are exactly what must not be re-derived (#55/#56/#63 are what
//! re-deriving them costs). What is not reused is the shape: a column reference is an
//! ordinal into a child whose column order this mode decides, so every reference is
//! rebased at each node the translation layer inserts.

use datafusion::arrow::datatypes::DataType;
use datafusion::common::ScalarValue;

/// The name rides beside the ordinal so a plan can be checked against the schema at that
/// position rather than trusting it — #135's class, caught at plan time here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnRef {
    pub index: u32,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    And,
    Or,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseShiftLeft,
    BitwiseShiftRight,
    StringConcat,
    IsDistinctFrom,
    IsNotDistinctFrom,
}

/// `Sqrt` is not a DataFusion unary: it is what a stddev's finalize expression needs, and
/// cuDF's `unary_operator::SQRT` is what the hardwired finalize already calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    IsNull,
    IsNotNull,
    Negative,
    Sqrt,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(ColumnRef),
    Literal(ScalarValue),
    /// `out_type` is DataFusion's declared output type. cuDF derives its own fixed-point
    /// result scale — division most visibly — so the declared one travels with the op.
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
        out_type: DataType,
    },
    Unary {
        op: UnaryOp,
        arg: Box<Expr>,
    },
    Cast {
        expr: Box<Expr>,
        target: DataType,
    },
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        negated: bool,
        case_insensitive: bool,
    },
    /// Search form leaves `comparand` `None`; the value form sets it.
    Case {
        comparand: Option<Box<Expr>>,
        when_then: Vec<(Expr, Expr)>,
        else_expr: Option<Box<Expr>>,
    },
    ScalarFunction {
        name: String,
        args: Vec<Expr>,
        return_type: DataType,
        nullable: bool,
    },
}

impl Expr {
    pub fn column(index: u32, name: &str) -> Self {
        Self::Column(ColumnRef {
            index,
            name: name.to_string(),
        })
    }

    pub fn binary(left: Expr, op: BinaryOp, right: Expr, out_type: DataType) -> Self {
        Self::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
            out_type,
        }
    }

    pub fn unary(op: UnaryOp, arg: Expr) -> Self {
        Self::Unary {
            op,
            arg: Box::new(arg),
        }
    }
}

/// An expression with the name its column takes in the node's output — a project list
/// entry, or one column of an aggregate's `final` list.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedExpr {
    pub expr: Expr,
    pub name: String,
}

impl NamedExpr {
    pub fn new(expr: Expr, name: &str) -> Self {
        Self {
            expr,
            name: name.to_string(),
        }
    }
}
