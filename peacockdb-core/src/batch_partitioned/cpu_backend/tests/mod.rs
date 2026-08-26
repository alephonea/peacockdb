//! One hand-built node per executor, one hand-written expected result.
//!
//! The plans are constructed rather than planned, so a test names the shape it means; the
//! oracle is written down rather than computed by anything under test.

use super::*;
use crate::batch_partitioned::aggregates::{AggCall, AggFunc, PlanAgg};
use crate::batch_partitioned::expr::{BinaryOp, Expr, NamedExpr};
use crate::batch_partitioned::layout::{BatchLayout, NodeKind, PartitionLayout};
use crate::batch_partitioned::node::GpuNode;
use crate::batch_partitioned::nodes::aggregate::AggregateBody;
use crate::batch_partitioned::schema::{AggStateColumns, Schema};
use datafusion::arrow::array::{Array, ArrayRef, Int32Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::ScalarValue;
use datafusion::execution::context::SessionContext;
use std::any::Any;

/// A child that declares a schema and nothing else: an executor is handed its node and its
/// input batches, so a stub is what keeps a case about the operator.
#[derive(Debug)]
struct Given {
    kind: NodeKind,
}

impl Given {
    fn of(columns: &[(&str, DataType)]) -> Box<dyn GpuNode> {
        Self::of_schema(schema_of(columns))
    }

    /// The same stub over a schema already built — a merge's input is the state its child
    /// emitted, which is a `Schema` with annotations rather than a column list.
    fn of_schema(schema: Schema) -> Box<dyn GpuNode> {
        Box::new(Given {
            kind: NodeKind::Intermediate {
                layout: PartitionLayout {
                    batch_layout: BatchLayout::MultipleBatches,
                    ..PartitionLayout::new(1)
                },
                schema,
            },
        })
    }
}

impl GpuNode for Given {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }
    fn children(&self) -> Vec<&dyn GpuNode> {
        Vec::new()
    }
    fn validate_schemas_and_partitions(&self) -> Result<(), crate::batch_partitioned::PlanError> {
        Ok(())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn schema_of(columns: &[(&str, DataType)]) -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(
        columns
            .iter()
            .map(|(name, kind)| Field::new(*name, kind.clone(), true))
            .collect::<Vec<_>>(),
    )))
}

const COLUMNS: [(&str, DataType); 2] = [("n", DataType::Int32), ("s", DataType::Utf8)];

fn input() -> ArrowSchema {
    schema_of(&COLUMNS).fields.as_ref().clone()
}

fn batch(numbers: Vec<Option<i32>>, strings: Vec<Option<&str>>) -> CpuBatch {
    let n: ArrayRef = Arc::new(Int32Array::from(numbers));
    let s: ArrayRef = Arc::new(StringArray::from(strings));
    CpuBatch::new(
        RecordBatch::try_new(Arc::new(input()), vec![n, s]).expect("the columns fit the schema"),
    )
}

fn ctx() -> Arc<TaskContext> {
    SessionContext::new().task_ctx()
}

/// What came back, as a list per column, so an expected result reads as the rows a person
/// would write down.
fn rows(batch: &CpuBatch) -> (Vec<Option<i32>>, Vec<Option<String>>) {
    let batch = batch.record_batch();
    let n = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("an int column");
    let s = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("a string column");
    (
        (0..batch.num_rows())
            .map(|i| (!n.is_null(i)).then(|| n.value(i)))
            .collect(),
        (0..batch.num_rows())
            .map(|i| (!s.is_null(i)).then(|| s.value(i).to_string()))
            .collect(),
    )
}

fn greater_than(bound: i32) -> Expr {
    Expr::binary(
        Expr::column(0, "n"),
        BinaryOp::Gt,
        Expr::Literal(ScalarValue::Int32(Some(bound))),
        DataType::Boolean,
    )
}

const GROUPED: [(&str, DataType); 2] = [("k", DataType::Utf8), ("v", DataType::Int64)];

fn grouped(keys: Vec<Option<&str>>, values: Vec<Option<i64>>) -> CpuBatch {
    let k: ArrayRef = Arc::new(StringArray::from(keys));
    let v: ArrayRef = Arc::new(Int64Array::from(values));
    CpuBatch::new(
        RecordBatch::try_new(
            Arc::new(schema_of(&GROUPED).fields.as_ref().clone()),
            vec![k, v],
        )
        .expect("the columns fit the schema"),
    )
}

/// A state schema in the shape an aggregate declares one: the keys lead it, and the
/// annotation says which columns a Welford triple owns.
fn state_of(columns: &[(&str, DataType)], keys: usize, welford: Option<&str>) -> Schema {
    Schema {
        fields: Arc::new(schema_of(columns).fields.as_ref().clone()),
        group_keys: (0..keys as u32).collect(),
        agg_state: welford
            .map(|output| {
                vec![AggStateColumns {
                    output: output.to_string(),
                    func: AggFunc::Stddev,
                    ddof: 1,
                    positions: (keys as u32..keys as u32 + 3).collect(),
                }]
            })
            .unwrap_or_default(),
    }
}

fn agg(func: PlanAgg, output: &str, kind: DataType) -> AggCall {
    AggCall {
        func,
        args: vec![Expr::column(1, "v")],
        outputs: vec![Field::new(output, kind, true)],
    }
}

fn by_key(batch: &CpuBatch) -> Vec<(String, Vec<ScalarValue>)> {
    let batch = batch.record_batch();
    let keys = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("the key column");
    let mut rows: Vec<(String, Vec<ScalarValue>)> = (0..batch.num_rows())
        .map(|row| {
            let rest: Vec<ScalarValue> = (1..batch.num_columns())
                .map(|column| {
                    ScalarValue::try_from_array(batch.column(column), row)
                        .expect("a value at every position")
                })
                .collect();
            (keys.value(row).to_string(), rest)
        })
        .collect();
    // A hash aggregate answers in whatever order its table holds; every claim here is
    // about which group got which value.
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    rows
}

/// The key, or `total` where the set dropped it, paired with what that row summed.
fn out_rows(batch: &CpuBatch) -> Vec<(String, ScalarValue)> {
    let record = batch.record_batch();
    let keys = record
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("the key column");
    (0..record.num_rows())
        .map(|row| {
            let key = if keys.is_null(row) {
                "total".to_string()
            } else {
                keys.value(row).to_string()
            };
            (
                key,
                ScalarValue::try_from_array(record.column(2), row).expect("a sum"),
            )
        })
        .collect()
}

mod accumulate;
mod exec;
