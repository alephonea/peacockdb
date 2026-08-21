//! Which columns the planner believes can be NULL, one test per rule, over hand-built nodes.
//!
//! The analysis is a safety property: an anti or mark join is refused when NULLs can meet on
//! both sides, so a rule that answers "cannot be NULL" too readily permits a join we decided
//! is wrong, silently and with no plan to look at. The refusal itself, the leaf that reads
//! parquet statistics and the outer join's padding are covered against real files in
//! `test_planner_join_capability.rs`; what is here is every other rule, asserted on
//! `can_be_null` directly, because a hand-built input can say "this column is not nullable"
//! and a corpus fixture cannot — every column in both benchmarks is declared nullable.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::ScalarValue;

use peacockdb_core::batch_partitioned::expr::{BinaryOp, Expr, NamedExpr};
use peacockdb_core::batch_partitioned::node::{GpuNode, RowInterval};
use peacockdb_core::batch_partitioned::nodes::{
    AggregateBody, GpuAggregate, GpuFilter, GpuLimit, GpuLoadParquet, GpuMergePartitions,
    GpuProject, GpuSort, GpuUnion,
};
use peacockdb_core::batch_partitioned::nulls::can_be_null;
use peacockdb_core::batch_partitioned::parquet_meta::ScanMetadata;
use peacockdb_core::batch_partitioned::partitioner::RowGroupMeta;
use peacockdb_core::batch_partitioned::schema::Schema;

/// A source declaring exactly the nullability asked for. The leaf reads this off parquet
/// statistics; here it is stated, which is the only way to get a NOT-nullable column at all.
fn source(nullable: &[bool]) -> Box<dyn GpuNode> {
    let fields: Vec<Field> = nullable
        .iter()
        .enumerate()
        .map(|(index, _)| Field::new(format!("c{index}"), DataType::Int64, true))
        .collect();
    let schema = Schema::new(Arc::new(ArrowSchema::new(fields)));
    let scan = ScanMetadata {
        file: "/t.parquet".to_string(),
        groups: vec![RowGroupMeta {
            index: 0,
            rows: 100,
            bytes: 800,
        }],
        can_be_null: nullable.to_vec(),
    };
    Box::new(GpuLoadParquet::new(
        "t".to_string(),
        (0..nullable.len() as u32).collect(),
        vec![vec![vec![0]]],
        &scan,
        None,
        schema,
    ))
}

/// One column named `out`, which is what every expression test projects into.
fn one_column() -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
        "out",
        DataType::Int64,
        true,
    )])))
}

/// Whether the analysis says an expression over `[nullable c0, non-nullable c1]` can be NULL.
fn projected(expr: Expr) -> bool {
    let project = GpuProject::new(
        source(&[true, false]),
        vec![NamedExpr::new(expr, "out")],
        one_column(),
    );
    can_be_null(&project)[0]
}

fn column(index: u32) -> Expr {
    Expr::column(index, &format!("c{index}"))
}

// ── introducing ────────────────────────────────────────────────────────────────

#[test]
fn a_case_with_no_else_can_be_null() {
    // The implicit ELSE returns NULL, and no branch of the CASE says otherwise.
    let branch = (column(1), column(1));
    let no_else = Expr::Case {
        comparand: None,
        when_then: vec![branch.clone()],
        else_expr: None,
    };
    assert!(projected(no_else), "a CASE with no ELSE returns NULL");

    let with_else = Expr::Case {
        comparand: None,
        when_then: vec![branch],
        else_expr: Some(Box::new(column(1))),
    };
    assert!(
        !projected(with_else),
        "every arm reads a non-nullable column"
    );
}

#[test]
fn an_expression_is_null_exactly_when_an_operand_can_be() {
    let over_nullable = Expr::binary(column(0), BinaryOp::Plus, column(1), DataType::Int64);
    let over_neither = Expr::binary(column(1), BinaryOp::Plus, column(1), DataType::Int64);
    assert!(projected(over_nullable), "c0 can be NULL, so the sum can");
    assert!(!projected(over_neither), "neither operand can be NULL");
}

#[test]
fn a_null_literal_can_be_null_and_a_value_cannot() {
    assert!(projected(Expr::Literal(ScalarValue::Int64(None))));
    assert!(!projected(Expr::Literal(ScalarValue::Int64(Some(1)))));
}

#[test]
fn a_scalar_function_can_be_null_even_over_operands_that_cannot() {
    // Deliberately conservative: coalesce is the shape a general rule gets wrong, so no
    // function's result is claimed non-nullable. Nothing in the corpus would notice if this
    // were reversed, which is why it is asserted here.
    let call = Expr::ScalarFunction {
        name: "coalesce".to_string(),
        args: vec![column(1), column(1)],
        return_type: DataType::Int64,
        nullable: false,
    };
    assert!(projected(call));
}

#[test]
fn a_union_column_is_null_where_any_branch_says_so() {
    let schema = Schema::new(Arc::new(ArrowSchema::new(vec![
        Field::new("c0", DataType::Int64, true),
        Field::new("c1", DataType::Int64, true),
    ])));
    let union = GpuUnion::new(
        vec![source(&[false, false]), source(&[true, false])],
        schema,
    );
    assert_eq!(can_be_null(&union), vec![true, false]);
}

#[test]
fn a_grouping_set_makes_the_keys_it_drops_null() {
    // The rollup substitutes NULL for a key the set excludes, so a column that is not
    // nullable below the aggregate is nullable above it — and `__grouping_id` never is.
    let keys = vec![column(1), column(1)];
    let body = AggregateBody {
        group_by: keys,
        grouping_sets: vec![vec![false, false], vec![true, false]],
        null_exprs: Vec::new(),
        aggs: Vec::new(),
        finalize: None,
    };
    let schema = Schema::new(Arc::new(ArrowSchema::new(vec![
        Field::new("k0", DataType::Int64, true),
        Field::new("k1", DataType::Int64, true),
        Field::new("__grouping_id", DataType::UInt8, false),
    ])));
    let aggregate = GpuAggregate::new(source(&[true, false]), body, schema.clone(), schema);
    assert_eq!(can_be_null(&aggregate), vec![true, false, false]);
}

// ── preserving ─────────────────────────────────────────────────────────────────

#[test]
fn moving_and_dropping_rows_leaves_nullability_alone() {
    // A filter, a sort, a limit and a merge change which rows are where and nothing about
    // what a column holds. Nothing pinned this side of the analysis before.
    let filter = GpuFilter::new(
        source(&[true, false]),
        Expr::binary(
            column(1),
            BinaryOp::Gt,
            Expr::Literal(ScalarValue::Int64(Some(0))),
            DataType::Boolean,
        ),
        None,
        Schema::new(Arc::new(ArrowSchema::new(vec![
            Field::new("c0", DataType::Int64, true),
            Field::new("c1", DataType::Int64, true),
        ]))),
    );
    let sort = GpuSort::new(Box::new(filter), Vec::new(), None);
    let limit = GpuLimit::new(
        Box::new(sort),
        RowInterval {
            skip: 0,
            fetch: Some(10),
        },
    );
    let merged = GpuMergePartitions::new(Box::new(limit));
    assert_eq!(can_be_null(&merged), vec![true, false]);
}

#[test]
fn a_filter_that_projects_answers_for_the_columns_it_keeps() {
    let filter = GpuFilter::new(
        source(&[true, false]),
        Expr::Literal(ScalarValue::Boolean(Some(true))),
        Some(vec![1]),
        one_column(),
    );
    assert_eq!(can_be_null(&filter), vec![false]);
}
