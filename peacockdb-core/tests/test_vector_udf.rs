//! End-to-end: register an in-memory fp16 vector table and run
//! `SELECT id, l2_distance(v, q) FROM t` through the SQL planner, asserting the
//! distances match a brute-force Rust reference computed here. Dependency-free
//! (no NumPy/Faiss oracle — that's a later ticket).
//!
//! The query vector is carried as a second column `q` (broadcast to every row)
//! rather than a SQL `VECTOR(n)` literal: DataFusion's sqlparser parses `VECTOR(n)`
//! as an unsupported `Custom` type, so that sugar is deferred (see the ticket 1
//! report). An explicit Arrow `FixedSizeList<Float16, n>` fixture exercises the
//! same UDF invoke path.

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, FixedSizeListBuilder, Float16Builder, Float32Array, Int32Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use half::f16;

use peacockdb_core::vector::{vector_dtype_for_dim, VectorScalar};

const DIM: usize = 4;

fn to_f16(row: &[f32]) -> Vec<f16> {
    row.iter().map(|&x| f16::from_f32(x)).collect()
}

/// Brute-force reference distance (plain Rust), the oracle the SQL result is checked against.
fn ref_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

fn build_fsl(rows: &[Vec<f16>]) -> Arc<dyn Array> {
    let mut b = FixedSizeListBuilder::new(Float16Builder::new(), DIM as i32);
    for row in rows {
        for &v in row {
            b.values().append_value(v);
        }
        b.append(true);
    }
    Arc::new(b.finish())
}

#[tokio::test]
async fn l2_distance_over_in_memory_fp16_table_matches_brute_force() {
    // 8 rows x dim 4, values chosen so the fp16 round-trip is exact (small integers).
    let data: Vec<[f32; DIM]> = vec![
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [5.0, 0.0, 12.0, 0.0],
        [8.0, 8.0, 8.0, 8.0],
    ];
    let query: [f32; DIM] = [1.0, 2.0, 3.0, 4.0];

    let vecs: Vec<Vec<f16>> = data.iter().map(|r| to_f16(r)).collect();
    let queries: Vec<Vec<f16>> = (0..data.len()).map(|_| to_f16(&query)).collect();

    let vector_ty = vector_dtype_for_dim(DIM as u32, VectorScalar::F16);
    assert!(matches!(vector_ty, DataType::FixedSizeList(_, 4)));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", vector_ty.clone(), false),
        Field::new("q", vector_ty, false),
    ]));
    let ids: Int32Array = (0..data.len() as i32).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), build_fsl(&vecs), build_fsl(&queries)],
    )
    .unwrap();

    // Pure CPU: build_session_state has no GPU rules, so this runs on vanilla
    // DataFusion (and stays green under --features rust-only, no FFI).
    let ctx = peacockdb_core::build_session_state(1);
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.register_table("t", Arc::new(table)).unwrap();

    let df = ctx
        .sql("SELECT id, l2_distance(v, q) AS d FROM t ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    let mut got: Vec<(i32, f32)> = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let ds = b.column(1).as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(ds.data_type(), &DataType::Float32, "l2_distance returns Float32");
        for i in 0..b.num_rows() {
            got.push((ids.value(i), ds.value(i)));
        }
    }

    assert_eq!(got.len(), data.len());
    for (id, d) in got {
        let expected = ref_l2(&data[id as usize], &query);
        assert!(
            (d - expected).abs() < 1e-2,
            "row {id}: sql l2_distance={d}, brute-force={expected}"
        );
    }
}
