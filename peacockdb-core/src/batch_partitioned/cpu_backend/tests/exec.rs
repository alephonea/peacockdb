//! The exec executors: a filter, a project, a per-batch sort, an aggregate with and
//! without its finalize, and the unload's row range.

use super::*;

#[test]
fn a_filter_answers_with_the_rows_its_predicate_keeps() {
    let node = GpuFilter::new(
        Given::of(&COLUMNS),
        greater_than(2),
        None,
        schema_of(&COLUMNS),
    );
    let mut exec = CpuExec::filter(&node, &input(), ctx()).expect("the filter builds");
    let (out, _) = exec
        .exec(batch(
            vec![Some(1), Some(3), None, Some(5)],
            vec![Some("a"), Some("b"), Some("c"), Some("d")],
        ))
        .expect("it runs");
    assert_eq!(
        rows(&out),
        (
            vec![Some(3), Some(5)],
            vec![Some("b".to_string()), Some("d".to_string())]
        ),
        "a null is not greater than 2, and neither is 1"
    );
}

/// The contract is one batch out per batch in. A filter that keeps nothing still owes a
/// batch, because the node above counts calls and not rows.
#[test]
fn a_filter_that_keeps_nothing_still_answers_with_a_batch() {
    let node = GpuFilter::new(
        Given::of(&COLUMNS),
        greater_than(100),
        None,
        schema_of(&COLUMNS),
    );
    let mut exec = CpuExec::filter(&node, &input(), ctx()).expect("the filter builds");
    let (out, _) = exec
        .exec(batch(vec![Some(1), Some(2)], vec![Some("a"), Some("b")]))
        .expect("it runs");
    assert_eq!(out.record_batch().num_rows(), 0);
    assert_eq!(
        out.record_batch().schema().fields().len(),
        2,
        "an empty answer still has the node's schema"
    );
}

/// DataFusion's filter projects as well as filtering, and dropping that would leave the
/// node declaring its child's columns while emitting fewer.
#[test]
fn a_filter_that_projects_emits_the_columns_it_named() {
    let kept = schema_of(&[("s", DataType::Utf8)]);
    let node = GpuFilter::new(Given::of(&COLUMNS), greater_than(1), Some(vec![1]), kept);
    let mut exec = CpuExec::filter(&node, &input(), ctx()).expect("the filter builds");
    let (out, _) = exec
        .exec(batch(
            vec![Some(1), Some(4)],
            vec![Some("dropped"), Some("kept")],
        ))
        .expect("it runs");
    let out = out.record_batch();
    assert_eq!(out.schema().field(0).name(), "s");
    assert_eq!(out.schema().fields().len(), 1);
    assert_eq!(out.num_rows(), 1);
}

#[test]
fn a_project_evaluates_its_expressions_under_the_names_it_declares() {
    let out_schema = schema_of(&[("twice", DataType::Int32), ("s", DataType::Utf8)]);
    let node = GpuProject::new(
        Given::of(&COLUMNS),
        vec![
            NamedExpr::new(
                Expr::binary(
                    Expr::column(0, "n"),
                    BinaryOp::Multiply,
                    Expr::Literal(ScalarValue::Int32(Some(2))),
                    DataType::Int32,
                ),
                "twice",
            ),
            NamedExpr::new(Expr::column(1, "s"), "s"),
        ],
        out_schema,
    );
    let mut exec = CpuExec::project(&node, &input(), ctx()).expect("the project builds");
    let (out, _) = exec
        .exec(batch(vec![Some(3), None], vec![Some("a"), Some("b")]))
        .expect("it runs");
    assert_eq!(
        out.record_batch().schema().field(0).name(),
        "twice",
        "the output name is the project's, not the input's"
    );
    assert_eq!(
        rows(&out),
        (
            vec![Some(6), None],
            vec![Some("a".to_string()), Some("b".to_string())]
        ),
        "null times two is null"
    );
}

#[test]
fn a_sort_orders_the_batch_it_was_given() {
    let node = GpuSort::new(
        Given::of(&COLUMNS),
        vec![ColumnOrder {
            column: 0,
            ascending: true,
            nulls_first: false,
        }],
        None,
    );
    let mut exec = CpuExec::sort(&node, &input(), ctx()).expect("the sort builds");
    let (out, _) = exec
        .exec(batch(
            vec![Some(3), None, Some(1)],
            vec![Some("c"), Some("null"), Some("a")],
        ))
        .expect("it runs");
    assert_eq!(
        rows(&out).0,
        vec![Some(1), Some(3), None],
        "ascending with nulls last"
    );
}

#[test]
fn a_sort_with_a_fetch_keeps_the_top_of_its_own_batch() {
    let node = GpuSort::new(
        Given::of(&COLUMNS),
        vec![ColumnOrder {
            column: 0,
            ascending: false,
            nulls_first: false,
        }],
        Some(2),
    );
    let mut exec = CpuExec::sort(&node, &input(), ctx()).expect("the sort builds");
    let (out, _) = exec
        .exec(batch(
            vec![Some(3), Some(9), Some(1), Some(7)],
            vec![Some("a"), Some("b"), Some("c"), Some("d")],
        ))
        .expect("it runs");
    assert_eq!(
        rows(&out).0,
        vec![Some(9), Some(7)],
        "the two largest, in order"
    );
}

#[test]
fn an_unload_of_the_whole_batch_hands_it_over_untouched() {
    let (out, _) = CpuUnload
        .unload(
            batch(vec![Some(1), Some(2)], vec![Some("a"), Some("b")]),
            RowRange::WHOLE,
        )
        .expect("it runs");
    assert_eq!(out.record_batch().num_rows(), 2);
}

/// The straddling batch: the driver hands a range because the rows wanted are a window
/// inside this batch, and shipping the batch to trim it afterwards is what the range
/// exists to avoid.
#[test]
fn an_unload_with_a_range_answers_with_the_rows_it_names() {
    let (out, _) = CpuUnload
        .unload(
            batch(
                vec![Some(1), Some(2), Some(3), Some(4)],
                vec![Some("a"), Some("b"), Some("c"), Some("d")],
            ),
            RowRange {
                offset: 1,
                length: 2,
            },
        )
        .expect("it runs");
    assert_eq!(rows(&out).0, vec![Some(2), Some(3)]);
}

#[test]
fn an_unload_whose_range_runs_past_the_end_stops_at_the_end() {
    let (out, _) = CpuUnload
        .unload(
            batch(vec![Some(1), Some(2)], vec![Some("a"), Some("b")]),
            RowRange {
                offset: 1,
                length: 100,
            },
        )
        .expect("it runs");
    assert_eq!(rows(&out).0, vec![Some(2)]);
}

#[test]
fn an_unload_whose_offset_is_past_the_end_answers_empty() {
    let (out, _) = CpuUnload
        .unload(
            batch(vec![Some(1)], vec![Some("a")]),
            RowRange {
                offset: 9,
                length: 1,
            },
        )
        .expect("it runs");
    assert_eq!(out.record_batch().num_rows(), 0);
}

#[test]
fn a_partial_aggregate_emits_its_state_under_the_names_the_node_declared() {
    let state = state_of(
        &[("k", DataType::Utf8), ("sum(v)", DataType::Int64)],
        1,
        None,
    );
    let node = GpuAggregate::new(
        Given::of(&GROUPED),
        AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![agg(PlanAgg::Sum, "sum(v)", DataType::Int64)],
            finalize: None,
        },
        state.clone(),
        state,
    );
    let mut exec = CpuExec::aggregate(&node, &schema_of(&GROUPED).fields, ctx())
        .expect("the aggregate builds");
    let (out, _) = exec
        .exec(grouped(
            vec![Some("a"), Some("b"), Some("a")],
            vec![Some(1), Some(2), Some(3)],
        ))
        .expect("it runs");
    assert_eq!(
        out.record_batch().schema().field(1).name(),
        "sum(v)",
        "DataFusion names a state column after its accumulator; the node's name is the one \
         every reference above resolves against"
    );
    assert_eq!(
        by_key(&out),
        vec![
            ("a".to_string(), vec![ScalarValue::Int64(Some(4))]),
            ("b".to_string(), vec![ScalarValue::Int64(Some(2))])
        ]
    );
}

/// The single-node shortcut: state and finalize on one node, which is two operators here
/// and two calls on the device. The finalize is this mode's own expression, evaluated by
/// DataFusion rather than by a `Single`-mode accumulator — the same expression the device
/// is sent.
#[test]
fn an_aggregate_that_finalizes_divides_the_state_it_just_built() {
    let state = state_of(
        &[
            ("k", DataType::Utf8),
            ("avg(v)$sum", DataType::Int64),
            ("avg(v)$count", DataType::Int64),
        ],
        1,
        None,
    );
    let output = state_of(
        &[("k", DataType::Utf8), ("avg(v)", DataType::Float64)],
        1,
        None,
    );
    let average = Expr::binary(
        Expr::Cast {
            expr: Box::new(Expr::column(1, "avg(v)$sum")),
            target: DataType::Float64,
        },
        BinaryOp::Divide,
        Expr::Cast {
            expr: Box::new(Expr::column(2, "avg(v)$count")),
            target: DataType::Float64,
        },
        DataType::Float64,
    );
    let node = GpuAggregate::new(
        Given::of(&GROUPED),
        AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![
                agg(PlanAgg::Sum, "avg(v)$sum", DataType::Int64),
                agg(PlanAgg::Count, "avg(v)$count", DataType::Int64),
            ],
            finalize: Some(vec![NamedExpr::new(average, "avg(v)")]),
        },
        state,
        output,
    );
    let mut exec = CpuExec::aggregate(&node, &schema_of(&GROUPED).fields, ctx())
        .expect("the aggregate builds");
    let (out, _) = exec
        .exec(grouped(
            vec![Some("a"), Some("b"), Some("a"), Some("b")],
            vec![Some(1), Some(2), Some(3), Some(5)],
        ))
        .expect("it runs");
    assert_eq!(
        out.record_batch().schema().fields().len(),
        2,
        "the finalized row is the keys and the finalized columns, and nothing of the state"
    );
    assert_eq!(
        by_key(&out),
        vec![
            ("a".to_string(), vec![ScalarValue::Float64(Some(2.0))]),
            ("b".to_string(), vec![ScalarValue::Float64(Some(3.5))])
        ]
    );
}

/// The Welford triple is three of this mode's aggregators and one of DataFusion's, so what
/// proves the fold is that one aggregate filled three declared columns — a triple sent as
/// three separate aggregators would fill three columns of its own and disagree with the
/// declared state's width.
#[test]
fn a_welford_triple_is_one_aggregate_filling_three_declared_columns() {
    let state = state_of(
        &[
            ("k", DataType::Utf8),
            ("stddev(v)$count", DataType::UInt64),
            ("stddev(v)$mean", DataType::Float64),
            ("stddev(v)$m2", DataType::Float64),
        ],
        1,
        Some("stddev(v)"),
    );
    let node = GpuAggregate::new(
        Given::of(&GROUPED),
        AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![
                agg(PlanAgg::Count, "stddev(v)$count", DataType::UInt64),
                agg(PlanAgg::Mean, "stddev(v)$mean", DataType::Float64),
                agg(PlanAgg::M2, "stddev(v)$m2", DataType::Float64),
            ],
            finalize: None,
        },
        state.clone(),
        state,
    );
    let mut exec = CpuExec::aggregate(&node, &schema_of(&GROUPED).fields, ctx())
        .expect("the aggregate builds");
    let (out, _) = exec
        .exec(grouped(
            vec![Some("a"), Some("a"), Some("a"), Some("b")],
            vec![Some(2), Some(4), Some(6), Some(9)],
        ))
        .expect("it runs");
    let schema = out.record_batch().schema();
    let names: Vec<&str> = schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect();
    assert_eq!(
        names,
        vec!["k", "stddev(v)$count", "stddev(v)$mean", "stddev(v)$m2"]
    );
    assert_eq!(
        by_key(&out),
        vec![
            (
                "a".to_string(),
                vec![
                    ScalarValue::UInt64(Some(3)),
                    ScalarValue::Float64(Some(4.0)),
                    ScalarValue::Float64(Some(8.0))
                ]
            ),
            (
                "b".to_string(),
                vec![
                    ScalarValue::UInt64(Some(1)),
                    ScalarValue::Float64(Some(9.0)),
                    ScalarValue::Float64(Some(0.0))
                ]
            )
        ],
        "count, mean and the sum of squared deviations from it: 4 + 0 + 4 for [2, 4, 6]"
    );
}

/// The state columns are relabelled by position, so a declared state that is not the one
/// DataFusion's accumulators produce would rename a column rather than fail. Caught where
/// the executor is built, not at the first batch.
#[test]
fn a_declared_state_of_another_type_than_the_accumulators_produce_is_refused() {
    let state = state_of(
        &[("k", DataType::Utf8), ("sum(v)", DataType::Utf8)],
        1,
        None,
    );
    let node = GpuAggregate::new(
        Given::of(&GROUPED),
        AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![agg(PlanAgg::Sum, "sum(v)", DataType::Utf8)],
            finalize: None,
        },
        state.clone(),
        state,
    );
    let refused = match CpuExec::aggregate(&node, &schema_of(&GROUPED).fields, ctx()) {
        Err(refused) => refused,
        Ok(_) => panic!("a sum does not produce a string"),
    };
    let message = format!("{refused}");
    assert!(
        message.contains("column 1 is Utf8") && message.contains("Int64"),
        "the refusal has to name the position and both types: {message}"
    );
}

/// ROLLUP: the one shape whose state is wider than its group list, because the init emits
/// `__grouping_id` beside the keys and every node above groups on it. The width matters
/// here — the state columns are found by walking past the keys — so a case that plans two
/// sets is what tells a right width from a wrong one.
#[test]
fn a_grouping_set_aggregate_emits_the_id_beside_its_keys() {
    let state = state_of(
        &[
            ("k", DataType::Utf8),
            ("__grouping_id", DataType::UInt8),
            ("sum(v)", DataType::Int64),
        ],
        2,
        None,
    );
    let node = GpuAggregate::new(
        Given::of(&GROUPED),
        AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            grouping_sets: vec![vec![false], vec![true]],
            null_exprs: vec![Expr::Literal(ScalarValue::Utf8(None))],
            aggs: vec![agg(PlanAgg::Sum, "sum(v)", DataType::Int64)],
            finalize: None,
        },
        state.clone(),
        state,
    );
    let mut exec = CpuExec::aggregate(&node, &schema_of(&GROUPED).fields, ctx())
        .expect("the aggregate builds");
    let (out, _) = exec
        .exec(grouped(
            vec![
                Some("a"),
                Some("b"),
                Some("a"),
                Some("b"),
                Some("a"),
                Some("b"),
            ],
            vec![Some(2), Some(1), Some(4), Some(3), Some(6), Some(5)],
        ))
        .expect("it runs");
    let mut rows: Vec<(String, ScalarValue)> = out_rows(&out);
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(
        rows,
        vec![
            ("a".to_string(), ScalarValue::Int64(Some(12))),
            ("b".to_string(), ScalarValue::Int64(Some(9))),
            ("total".to_string(), ScalarValue::Int64(Some(21))),
        ],
        "one row per key in the first set, and one for the whole table in the second"
    );
    assert_eq!(
        out.record_batch().schema().field(1).name(),
        "__grouping_id",
        "the id is a column of the state, at the position the node declared it"
    );
}
