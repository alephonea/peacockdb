//! The exec executors: filter, project, the per-batch sort, the aggregate with and
//! without its finalize, and the export's row range.

use super::*;

#[test]
fn a_filter_answers_with_the_rows_its_predicate_keeps() {
    let out = columns();
    let node = GpuFilter::new(
        source(),
        greater_than(3),
        None,
        Schema::new(Arc::new(out.clone())),
    );
    assert_eq!(
        rows(&one_node(Box::new(node), &out)),
        vec![
            vec![string("a"), ScalarValue::Int64(Some(4))],
            vec![string("a"), ScalarValue::Int64(Some(6))],
            vec![string("b"), ScalarValue::Int64(Some(5))],
        ],
        "the three rows above 3, in the order the scan read them"
    );
}

/// A filter that keeps nothing still answers with a batch, because the node above counts
/// calls and not rows — and on this side that batch is a resident table of zero rows.
#[test]
fn a_filter_that_keeps_nothing_still_answers_with_a_batch() {
    let out = columns();
    let node = GpuFilter::new(
        source(),
        greater_than(100),
        None,
        Schema::new(Arc::new(out.clone())),
    );
    let answer = one_node(Box::new(node), &out);
    assert_eq!(answer.record_batch().num_rows(), 0);
    assert_eq!(
        answer.record_batch().schema().fields().len(),
        2,
        "an empty answer still has the node's columns"
    );
}

#[test]
fn a_project_evaluates_its_expressions_under_the_names_it_declares() {
    let out = schema_of(&[("twice", DataType::Int64)]);
    let node = GpuProject::new(
        source(),
        vec![NamedExpr::new(
            Expr::binary(
                Expr::column(1, "v"),
                BinaryOp::Multiply,
                Expr::Literal(ScalarValue::Int64(Some(2))),
                DataType::Int64,
            ),
            "twice",
        )],
        Schema::new(Arc::new(out.clone())),
    );
    let answer = one_node(Box::new(node), &out);
    assert_eq!(
        answer.record_batch().schema().field(0).name(),
        "twice",
        "the output name is the project's, not the input's"
    );
    assert_eq!(
        rows(&answer)
            .into_iter()
            .map(|row| row[0].clone())
            .collect::<Vec<ScalarValue>>(),
        VALUES
            .iter()
            .map(|v| ScalarValue::Int64(Some(v * 2)))
            .collect::<Vec<ScalarValue>>()
    );
}

#[test]
fn a_sort_orders_the_batch_it_was_given() {
    let out = columns();
    let node = GpuSort::new(
        source(),
        vec![ColumnOrder {
            column: 1,
            ascending: true,
            nulls_first: false,
        }],
        None,
    );
    assert_eq!(
        rows(&one_node(Box::new(node), &out))
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        (1..=6)
            .map(|v| ScalarValue::Int64(Some(v)))
            .collect::<Vec<ScalarValue>>()
    );
}

/// The per-batch sort's `fetch` is a top-N inside the batch it was handed. Ordering a whole
/// stream is `GpuAccumulateBatchesAndSort`, a different node.
#[test]
fn a_sort_with_a_fetch_keeps_the_top_of_its_own_batch() {
    let out = columns();
    let node = GpuSort::new(
        source(),
        vec![ColumnOrder {
            column: 1,
            ascending: false,
            nulls_first: false,
        }],
        Some(2),
    );
    assert_eq!(
        rows(&one_node(Box::new(node), &out))
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        vec![ScalarValue::Int64(Some(6)), ScalarValue::Int64(Some(5))],
        "the two largest, in order"
    );
}

#[test]
fn a_partial_aggregate_emits_the_state_its_node_declared() {
    let out = schema_of(&[("k", DataType::Utf8), ("sum(v)", DataType::Int64)]);
    let state = Schema::new(Arc::new(out.clone()));
    let node = GpuAggregate::new(source(), summing("sum(v)"), state.clone(), state);
    assert_eq!(
        by_key(&one_node(Box::new(node), &out)),
        vec![
            vec![string("a"), ScalarValue::Int64(Some(12))],
            vec![string("b"), ScalarValue::Int64(Some(9))],
        ],
        "2 + 4 + 6 under a, 1 + 3 + 5 under b"
    );
}

/// The single-node shortcut: state and finalize on one node, which is two calls on one
/// executor. The finalize is this mode's own expression — the same one the CPU backend
/// hands DataFusion — rather than an aggregate mode that also finalizes.
#[test]
fn an_aggregate_that_finalizes_runs_both_of_its_calls() {
    let state_columns = schema_of(&[
        ("k", DataType::Utf8),
        ("avg(v)$sum", DataType::Int64),
        ("avg(v)$count", DataType::Int64),
    ]);
    let out = schema_of(&[("k", DataType::Utf8), ("avg(v)", DataType::Float64)]);
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
    let value = |output: &str, func| AggCall {
        func,
        args: vec![Expr::column(1, "v")],
        outputs: vec![Field::new(output, DataType::Int64, true)],
    };
    let node = GpuAggregate::new(
        source(),
        AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![
                value("avg(v)$sum", PlanAgg::Sum),
                value("avg(v)$count", PlanAgg::Count),
            ],
            finalize: Some(vec![NamedExpr::new(average, "avg(v)")]),
        },
        Schema::new(Arc::new(state_columns)),
        Schema::new(Arc::new(out.clone())),
    );
    assert_eq!(
        by_key(&one_node(Box::new(node), &out)),
        vec![
            vec![string("a"), ScalarValue::Float64(Some(4.0))],
            vec![string("b"), ScalarValue::Float64(Some(3.0))],
        ],
        "12/3 and 9/3, and the state columns are gone from the row"
    );
}

/// The whole handle, which is what every case above asks for after its node has run. Named
/// on its own so that an export which trimmed says so, rather than showing up as whichever
/// operator the failing test was about.
#[test]
fn an_export_of_the_whole_handle_brings_back_every_row() {
    let session = Session::open(source().as_ref());
    let batch = session.scan(&ROW_GROUPS);
    let (answer, _) = session
        .export(&columns())
        .unload(batch, RowRange::WHOLE)
        .expect("the rows cross the boundary");
    assert_eq!(
        rows(&answer)
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        VALUES
            .iter()
            .map(|v| ScalarValue::Int64(Some(*v)))
            .collect::<Vec<ScalarValue>>(),
        "six rows, in the order the scan read them"
    );
}

/// The range is what a straddling batch costs: the rows wanted cross the boundary rather
/// than the batch they sit in.
#[test]
fn an_export_with_a_range_answers_with_the_rows_it_names() {
    let session = Session::open(source().as_ref());
    let batch = session.scan(&ROW_GROUPS);
    let (answer, _) = session
        .export(&columns())
        .unload(
            batch,
            RowRange {
                offset: 1,
                length: 2,
            },
        )
        .expect("the rows cross the boundary");
    assert_eq!(
        rows(&answer)
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(4))]
    );
}

/// A limit's fetch legitimately overruns the batch it straddles, so a range past the end
/// clamps rather than failing.
#[test]
fn an_export_whose_range_runs_past_the_end_stops_at_the_end() {
    let session = Session::open(source().as_ref());
    let batch = session.scan(&ROW_GROUPS);
    let (answer, _) = session
        .export(&columns())
        .unload(
            batch,
            RowRange {
                offset: 4,
                length: 100,
            },
        )
        .expect("the rows cross the boundary");
    assert_eq!(answer.record_batch().num_rows(), 2);
}

/// A range naming no rows of a table that has them exports nothing at all, and the answer
/// is still a batch of the sink's columns rather than a missing one.
#[test]
fn an_export_whose_offset_is_past_the_end_answers_empty() {
    let session = Session::open(source().as_ref());
    let batch = session.scan(&ROW_GROUPS);
    let (answer, _) = session
        .export(&columns())
        .unload(
            batch,
            RowRange {
                offset: 99,
                length: 1,
            },
        )
        .expect("the export succeeds");
    assert_eq!(answer.record_batch().num_rows(), 0);
    assert_eq!(answer.record_batch().schema().fields().len(), 2);
}

/// An exec executor drives a straight line of per-batch calls. A recipe that waits for
/// done belongs to an accumulator, and building one here would call it once per batch —
/// a wrong answer rather than an error, since every call succeeds.
#[test]
fn a_recipe_whose_calls_wait_for_done_is_refused_by_an_exec_executor() {
    let keys = vec![ColumnOrder {
        column: 1,
        ascending: true,
        nulls_first: false,
    }];
    let tree: Box<dyn GpuNode> = Box::new(
        peacockdb_core::batch_partitioned::nodes::GpuAccumulateBatchesAndSort::new(
            source(),
            keys,
            None,
        ),
    );
    let recipes = attach_recipes(tree.as_ref()).expect("the payloads are writable");
    let refused = match GpuExec::new(
        std::ptr::null_mut(),
        recipes.get(1).expect("the accumulator makes ABI calls"),
        &columns(),
    ) {
        Err(refused) => refused,
        Ok(_) => panic!("an accumulator's recipe is not an exec node's"),
    };
    let message = format!("{refused}");
    assert!(
        message.contains("AtDone"),
        "the refusal has to name the pattern it found: {message}"
    );
}
