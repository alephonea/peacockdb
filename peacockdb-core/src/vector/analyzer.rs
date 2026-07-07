//! `VectorTopKAnalyzerRule` — collapse `Limit(fetch=k, Sort([l2_distance(v,q) ASC]))`
//! into a [`VectorTopK`] extension node. Runs as an ANALYZER (before projection
//! pushdown could prune the embedding column out of the sort key). Only a single
//! ascending `l2_distance` sort key directly under a plain `LIMIT k` (no OFFSET)
//! is rewritten; multi-key sorts, descending, or non-vector keys are left as an
//! ordinary Sort+Limit (still correct, just not accelerated).

use std::sync::Arc;

use datafusion::arrow::array::{Array, Float16Array, RecordBatch, RecordBatchOptions};
use datafusion::arrow::datatypes::Schema;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{DFSchema, Result, ScalarValue};
use datafusion::execution::context::ExecutionProps;
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan};
use datafusion::optimizer::AnalyzerRule;
use datafusion::physical_expr::create_physical_expr;

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
    // The top-k shape reaches the analyzer as either:
    //  - `Limit(fetch=k) -> [Projection]* -> Sort(l2_distance)` — SQL
    //    `SELECT id .. ORDER BY .. LIMIT k` interposes a projection between the
    //    Limit and the Sort (the SELECT list, above the sort's own id+embedding
    //    projection); DataFrame `.sort().limit()` has no interposed projection.
    //  - `Sort(.., fetch=Some(k))` — a fetch already fused into the Sort node.
    // The fetch (k) lives on whichever of Limit/Sort carries it; the Sort holding
    // the l2_distance key is found by descending through any interposed Projections.
    match &node {
        LogicalPlan::Limit(limit) => {
            // Plain LIMIT k with no OFFSET.
            if let Some(skip) = &limit.skip {
                if literal_usize(skip) != Some(0) {
                    return Ok(Transformed::no(node));
                }
            }
            let Some(fetch) = limit.fetch.as_deref().and_then(literal_usize) else {
                return Ok(Transformed::no(node));
            };
            match rewrite_topk_under(limit.input.as_ref(), fetch) {
                Some(new_input) => {
                    let mut limit = limit.clone();
                    limit.input = Arc::new(new_input);
                    Ok(Transformed::yes(LogicalPlan::Limit(limit)))
                }
                None => Ok(Transformed::no(node)),
            }
        }
        LogicalPlan::Sort(sort) => match sort.fetch.and_then(|f| vector_top_k_from_sort(sort, f)) {
            Some(vtopk) => Ok(Transformed::yes(vtopk)),
            None => Ok(Transformed::no(node)),
        },
        _ => Ok(Transformed::no(node)),
    }
}

/// Descend a `Limit`'s input through any interposed `Projection`s to the `Sort`
/// carrying the l2_distance key, and replace that `Sort` with a `VectorTopK{k}`,
/// rebuilding the projections above it. `None` if no such Sort is found.
fn rewrite_topk_under(plan: &LogicalPlan, k: usize) -> Option<LogicalPlan> {
    match plan {
        LogicalPlan::Sort(sort) => vector_top_k_from_sort(sort, k),
        LogicalPlan::Projection(proj) => {
            let new_input = rewrite_topk_under(proj.input.as_ref(), k)?;
            // The rewritten input (VectorTopK) has the same schema as the Sort it
            // replaced, so this projection still type-checks.
            Some(LogicalPlan::Projection(
                datafusion::logical_expr::Projection::try_new(
                    proj.expr.clone(),
                    Arc::new(new_input),
                )
                .ok()?,
            ))
        }
        _ => None,
    }
}

/// If `sort` is a single ascending `l2_distance(embedding, <const query>)` key,
/// the `VectorTopK{k=fetch}` extension plan that replaces it; `None` otherwise
/// (multi-key, descending, non-vector, or non-constant query — left as Sort+Limit).
fn vector_top_k_from_sort(
    sort: &datafusion::logical_expr::Sort,
    fetch: usize,
) -> Option<LogicalPlan> {
    if sort.expr.len() != 1 {
        return None;
    }
    let sort_expr = &sort.expr[0];
    // L2: smaller distance == nearer, so only an ascending key is a top-k.
    if !sort_expr.asc {
        return None;
    }
    let (query, dim) = as_l2_distance(&sort_expr.expr)?;
    let vtopk = VectorTopK::new(
        sort.input.as_ref().clone(),
        sort_expr.expr.clone(),
        fetch,
        query,
        dim,
    );
    Some(LogicalPlan::Extension(Extension {
        node: Arc::new(vtopk),
    }))
}

/// `Some((query_bytes, dim))` iff `expr` is an `l2_distance(_, _)` whose query
/// argument resolves to a constant `FixedSizeList<Float16>` — either directly (a
/// literal) or by folding a constant expression such as `to_vector(1,2,3,4)`. The
/// embedding-column argument does not fold to a constant, which is how the two are
/// told apart. `None` (not an l2_distance call, or the query isn't constant) leaves
/// the plan as a plain Sort+Limit.
fn as_l2_distance(expr: &Expr) -> Option<(Vec<u8>, u32)> {
    let Expr::ScalarFunction(ScalarFunction { func, args }) = expr else {
        return None;
    };
    if func.name() != L2_DISTANCE_UDF || args.len() != 2 {
        return None;
    }
    // Query is conventionally the 2nd arg; accept the 1st too for robustness.
    resolve_const_vector(&args[1]).or_else(|| resolve_const_vector(&args[0]))
}

/// Resolve `expr` to a constant `FixedSizeList<Float16>` → (little-endian f16
/// bytes, dim). A literal is read directly; anything else is const-folded (built
/// as a physical expr over an empty schema and evaluated). Returns `None` if it
/// isn't constant (e.g. references a column) or doesn't fold to an fp16 vector.
fn resolve_const_vector(expr: &Expr) -> Option<(Vec<u8>, u32)> {
    if let Expr::Literal(sv) = expr {
        return encode_fp16_query(sv);
    }
    let df_schema = DFSchema::empty();
    let phys = create_physical_expr(expr, &df_schema, &ExecutionProps::new()).ok()?;
    // A single-row, zero-column batch is enough to evaluate a constant expression.
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )
    .ok()?;
    let arr = phys.evaluate(&batch).ok()?.into_array(1).ok()?;
    let sv = ScalarValue::try_from_array(&arr, 0).ok()?;
    encode_fp16_query(&sv)
}

/// A `FixedSizeList<Float16, dim>` scalar → (little-endian f16 bytes, dim).
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
