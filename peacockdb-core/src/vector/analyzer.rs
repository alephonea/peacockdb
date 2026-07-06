//! `VectorTopKAnalyzerRule` — collapse `Limit(fetch=k, Sort([l2_distance(v,q) ASC]))`
//! into a [`VectorTopK`] extension node. Runs as an ANALYZER (before projection
//! pushdown could prune the embedding column out of the sort key). Only a single
//! ascending `l2_distance` sort key directly under a plain `LIMIT k` (no OFFSET)
//! is rewritten; multi-key sorts, descending, or non-vector keys are left as an
//! ordinary Sort+Limit (still correct, just not accelerated).

use datafusion::arrow::array::{Array, Float16Array};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan};
use datafusion::optimizer::AnalyzerRule;

use super::logical::VectorTopK;

pub const L2_DISTANCE_UDF: &str = "l2_distance";

#[derive(Debug, Default)]
pub struct VectorTopKAnalyzerRule;

impl AnalyzerRule for VectorTopKAnalyzerRule {
    fn name(&self) -> &str {
        "vector_top_k"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_down(rewrite_limit_sort).map(|t| t.data)
    }
}

fn rewrite_limit_sort(node: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Limit(limit) = &node else {
        return Ok(Transformed::no(node));
    };
    // Plain LIMIT k with no OFFSET.
    if let Some(skip) = &limit.skip {
        if literal_usize(skip) != Some(0) {
            return Ok(Transformed::no(node));
        }
    }
    let Some(fetch) = limit.fetch.as_deref().and_then(literal_usize) else {
        return Ok(Transformed::no(node));
    };
    let LogicalPlan::Sort(sort) = limit.input.as_ref() else {
        return Ok(Transformed::no(node));
    };
    // A single ascending sort key (L2: smaller distance == nearer).
    if sort.expr.len() != 1 {
        return Ok(Transformed::no(node));
    }
    let sort_expr = &sort.expr[0];
    if !sort_expr.asc {
        return Ok(Transformed::no(node));
    }
    let Some((query, dim)) = as_l2_distance(&sort_expr.expr) else {
        return Ok(Transformed::no(node));
    };

    let vtopk = VectorTopK::new(
        sort.input.as_ref().clone(),
        sort_expr.expr.clone(),
        fetch,
        query,
        dim,
    );
    Ok(Transformed::yes(LogicalPlan::Extension(Extension {
        node: std::sync::Arc::new(vtopk),
    })))
}

/// `Some((query_bytes, dim))` iff `expr` is an `l2_distance(_, _)` call. The query
/// bytes/dim come from whichever argument is a FixedSizeList<Float16> literal
/// (little-endian element bytes); a non-literal query yields empty bytes (the CPU
/// exec still scores via the expr — the bytes only feed the serialized IR).
fn as_l2_distance(expr: &Expr) -> Option<(Vec<u8>, u32)> {
    let Expr::ScalarFunction(ScalarFunction { func, args }) = expr else {
        return None;
    };
    if func.name() != L2_DISTANCE_UDF || args.len() != 2 {
        return None;
    }
    let query = args.iter().find_map(|a| match a {
        Expr::Literal(sv) => encode_fp16_query(sv),
        _ => None,
    });
    Some(query.unwrap_or_default())
}

/// A `FixedSizeList<Float16, dim>` literal → (little-endian f16 bytes, dim).
fn encode_fp16_query(sv: &ScalarValue) -> Option<(Vec<u8>, u32)> {
    let ScalarValue::FixedSizeList(arr) = sv else {
        return None;
    };
    let vals = arr.values().as_any().downcast_ref::<Float16Array>()?;
    let mut bytes = Vec::with_capacity(vals.len() * 2);
    for i in 0..vals.len() {
        bytes.extend_from_slice(&vals.value(i).to_le_bytes());
    }
    Some((bytes, arr.value_length() as u32))
}

/// A non-negative integer literal (Int64/UInt64) as `usize`.
fn literal_usize(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Literal(ScalarValue::Int64(Some(n))) if *n >= 0 => Some(*n as usize),
        Expr::Literal(ScalarValue::UInt64(Some(n))) => Some(*n as usize),
        _ => None,
    }
}
