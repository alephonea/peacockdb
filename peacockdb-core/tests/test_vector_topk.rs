//! VectorTopK end-to-end: the analyzer collapses `ORDER BY l2_distance(v,q) LIMIT
//! k` into a VectorTopK that lowers to a CPU GpuVectorSearchExec, returning the
//! exact top-k (checked against an in-test brute-force reference); the optimizer
//! pushes a pre-filter below the node; and non-matching sorts are left alone.
//! rust-only (pure DataFusion CPU — build_session_state has no GPU rules).

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, BooleanArray, FixedSizeListArray, FixedSizeListBuilder, Float16Builder, Int32Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{col, lit, Extension, Filter, LogicalPlan};
use datafusion::optimizer::{OptimizerContext, OptimizerRule};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use half::f16;

use peacockdb_core::vector::{l2_distance_udf, PushFilterIntoVectorTopK, VectorTopK};

const DIM: usize = 4;

// 8 rows x dim 4; distances to the query below are all distinct (no top-k ties).
fn rows() -> Vec<[f32; DIM]> {
    vec![
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 5.0],
        [9.0, 9.0, 9.0, 9.0],
        [2.0, 2.0, 2.0, 2.0],
        [5.0, 6.0, 7.0, 8.0],
        [1.0, 1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0, 3.0],
    ]
}
fn query() -> [f32; DIM] {
    [1.0, 2.0, 3.0, 4.0]
}
// flag: true on even ids (0,2,4,6) — the pre-filter set.
fn flag(id: usize) -> bool {
    id % 2 == 0
}

fn to_f16(r: &[f32]) -> Vec<f16> {
    r.iter().map(|&x| f16::from_f32(x)).collect()
}

fn ref_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

/// The k nearest ids (ascending distance), optionally restricted to flagged rows.
fn brute_top_k(k: usize, only_flagged: bool) -> Vec<i32> {
    let q = query();
    let mut scored: Vec<(f32, i32)> = rows()
        .iter()
        .enumerate()
        .filter(|(id, _)| !only_flagged || flag(*id))
        .map(|(id, r)| (ref_l2(r, &q), id as i32))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

fn fsl_column(data: &[[f32; DIM]]) -> Arc<dyn Array> {
    let mut b = FixedSizeListBuilder::new(Float16Builder::new(), DIM as i32);
    for r in data {
        for &v in &to_f16(r) {
            b.values().append_value(v);
        }
        b.append(true);
    }
    Arc::new(b.finish())
}

/// A FixedSizeList<Float16, DIM> literal holding the query vector.
fn query_literal() -> ScalarValue {
    let vals = Float16Builder::new();
    let mut b = FixedSizeListBuilder::new(vals, DIM as i32);
    for &v in &to_f16(&query()) {
        b.values().append_value(v);
    }
    b.append(true);
    let arr: FixedSizeListArray = b.finish();
    ScalarValue::FixedSizeList(Arc::new(arr))
}

async fn ctx_with_table() -> SessionContext {
    let vec_ty = DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float16, true)), DIM as i32);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", vec_ty, false),
        Field::new("flag", DataType::Boolean, false),
    ]));
    let data = rows();
    let ids: Int32Array = (0..data.len() as i32).collect();
    let flags: BooleanArray = (0..data.len()).map(|i| Some(flag(i))).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), fsl_column(&data), Arc::new(flags)],
    )
    .unwrap();

    let ctx = peacockdb_core::build_session_state(1);
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.register_table("t", Arc::new(table)).unwrap();
    ctx
}

fn contains_exec(plan: &Arc<dyn ExecutionPlan>, name: &str) -> bool {
    plan.name() == name || plan.children().iter().any(|c| contains_exec(c, name))
}

/// The distance expr `l2_distance(v, <query literal>)`.
fn distance_expr() -> datafusion::logical_expr::Expr {
    l2_distance_udf().call(vec![col("v"), lit(query_literal())])
}

#[tokio::test]
async fn analyzer_rewrites_l2_sort_limit_to_vector_topk() {
    let ctx = ctx_with_table().await;
    let plan = ctx
        .table("t")
        .await
        .unwrap()
        .sort(vec![distance_expr().sort(true, false)])
        .unwrap()
        .limit(0, Some(3))
        .unwrap()
        .select(vec![col("id")])
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    assert!(
        contains_exec(&plan, "GpuVectorSearchExec"),
        "l2_distance sort + limit must lower to GpuVectorSearchExec, got:\n{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );
}

#[tokio::test]
async fn multi_key_and_non_vector_sorts_are_not_rewritten() {
    let ctx = ctx_with_table().await;

    // Multi-key sort (distance, then id) — not a single vector key.
    let multi = ctx
        .table("t")
        .await
        .unwrap()
        .sort(vec![distance_expr().sort(true, false), col("id").sort(true, false)])
        .unwrap()
        .limit(0, Some(3))
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    assert!(!contains_exec(&multi, "GpuVectorSearchExec"));
    assert!(contains_exec(&multi, "SortExec"));

    // Non-vector single sort key.
    let plain = ctx
        .table("t")
        .await
        .unwrap()
        .sort(vec![col("id").sort(true, false)])
        .unwrap()
        .limit(0, Some(3))
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    assert!(!contains_exec(&plain, "GpuVectorSearchExec"));
}

#[tokio::test]
async fn filter_pushed_below_vector_topk() {
    let ctx = ctx_with_table().await;
    let input_lp = ctx.table("t").await.unwrap().into_unoptimized_plan();

    // Filter(VectorTopK(scan)) — the shape the rule normalizes.
    let vtopk = VectorTopK::new(input_lp, distance_expr(), 3, vec![], DIM as u32);
    let ext = LogicalPlan::Extension(Extension { node: Arc::new(vtopk) });
    let filtered = LogicalPlan::Filter(Filter::try_new(col("flag"), Arc::new(ext)).unwrap());

    let rule = PushFilterIntoVectorTopK;
    let out = rule.rewrite(filtered, &OptimizerContext::new()).unwrap().data;

    // Expect VectorTopK(Filter(scan)).
    let LogicalPlan::Extension(e) = &out else {
        panic!("expected Extension at root, got {out:?}");
    };
    let v = e.node.as_any().downcast_ref::<VectorTopK>().expect("VectorTopK");
    assert!(
        matches!(v.input(), LogicalPlan::Filter(_)),
        "filter must be pushed below VectorTopK, input was {:?}",
        v.input()
    );
}

async fn run_top_k_ids(ctx: &SessionContext, prefilter: bool) -> Vec<i32> {
    let mut df = ctx.table("t").await.unwrap();
    if prefilter {
        df = df.filter(col("flag")).unwrap();
    }
    let batches = df
        .sort(vec![distance_expr().sort(true, false)])
        .unwrap()
        .limit(0, Some(3))
        .unwrap()
        .select(vec![col("id")])
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut ids = Vec::new();
    for b in &batches {
        let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            ids.push(col.value(i));
        }
    }
    ids
}

#[tokio::test]
async fn e2e_top_k_matches_brute_force_reference() {
    let ctx = ctx_with_table().await;
    let ids = run_top_k_ids(&ctx, false).await;
    assert_eq!(ids, brute_top_k(3, false), "top-3 nearest ids, nearest-first");
}

#[tokio::test]
async fn e2e_top_k_with_prefilter_matches_reference() {
    let ctx = ctx_with_table().await;
    let ids = run_top_k_ids(&ctx, true).await;
    assert_eq!(
        ids,
        brute_top_k(3, true),
        "top-3 nearest among flagged rows (WHERE pre-filter below the top-k)"
    );
}
