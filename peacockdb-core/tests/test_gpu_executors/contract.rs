//! The device half of the contract in `executor_cases.inc`: the same table, the same
//! expectations, driven through this backend.
//!
//! What it adds over either side's own tests is the only thing neither can say — that two
//! implementations of one contract answer alike. A case here that a backend gets wrong is
//! wrong against the other engine, not against a fixture of its own.

use datafusion::common::JoinType;
use super::*;

use peacockdb_core::batch_partitioned::gpu_backend::accumulate::GpuAccumulator;
use peacockdb_core::batch_partitioned::gpu_backend::emit::GpuEmitter;
use peacockdb_core::batch_partitioned::gpu_backend::join::GpuJoin as GpuJoinExec;
use peacockdb_core::batch_partitioned::nodes::GpuJoin;
use peacockdb_core::batch_partitioned::nodes::join::joined_schema;
use peacockdb_core::batch_partitioned::nodes::{
    GpuAccumulateBatchesAndSort, GpuAggregateBatches, GpuCoalesceAllBatches, GpuEmitPartitions,
};

/// `k|v` per row, sorted, as the table writes its answers.
fn rendered(batch: &CpuBatch) -> Vec<String> {
    let batch = batch.record_batch();
    (0..batch.num_rows())
        .map(|row| {
            (0..batch.num_columns())
                .map(|column| {
                    match ScalarValue::try_from_array(batch.column(column), row)
                        .expect("a value at every position")
                    {
                        ScalarValue::Utf8(Some(text)) => text,
                        other => other.to_string(),
                    }
                })
                .collect::<Vec<String>>()
                .join("|")
        })
        .collect()
}

/// The state a `sum(v)` decomposes into, with `keys` leading it.
fn summed(keys: &[(&str, DataType)]) -> (Schema, AggregateBody, AggregateBody) {
    let state = Schema::new(Arc::new(schema_of(
        &[keys, &[("sum(v)", DataType::Int64)]].concat(),
    )));
    let group_by: Vec<Expr> = keys
        .iter()
        .enumerate()
        .map(|(ordinal, (name, _))| Expr::column(ordinal as u32, name))
        .collect();
    let sum = |column: u32| AggCall {
        func: PlanAgg::Sum,
        args: vec![Expr::column(column, "v")],
        outputs: vec![Field::new("sum(v)", DataType::Int64, true)],
    };
    (
        state,
        AggregateBody {
            group_by: group_by.clone(),
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![sum(1)],
            finalize: None,
        },
        AggregateBody {
            group_by,
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![sum(keys.len() as u32)],
            finalize: None,
        },
    )
}

/// One case through this backend, its lane arriving as the fixture's three row groups. The
/// rows keep the order the node emitted them in where that order is the answer.
fn answer(shape: Shape) -> Vec<String> {
    let mut rows = emitted(shape);
    if !shape.order_is_the_answer() {
        rows.sort();
    }
    rows
}

fn emitted(shape: Shape) -> Vec<String> {
    match shape {
        Shape::Filter { above } => {
            let out = columns();
            let node = GpuFilter::new(
                source_per_row_group(),
                Expr::binary(
                    Expr::column(1, "v"),
                    BinaryOp::Gt,
                    Expr::Literal(ScalarValue::Int64(Some(above))),
                    DataType::Boolean,
                ),
                None,
                Schema::new(Arc::new(out.clone())),
            );
            per_batch(Box::new(node), &out)
        }
        Shape::Double => {
            let out = columns();
            let node = GpuProject::new(
                source_per_row_group(),
                vec![
                    NamedExpr::new(Expr::column(0, "k"), "k"),
                    NamedExpr::new(
                        Expr::binary(
                            Expr::column(1, "v"),
                            BinaryOp::Multiply,
                            Expr::Literal(ScalarValue::Int64(Some(2))),
                            DataType::Int64,
                        ),
                        "v",
                    ),
                ],
                Schema::new(Arc::new(out.clone())),
            );
            per_batch(Box::new(node), &out)
        }
        Shape::SortLane { fetch } => {
            let out = columns();
            let node = GpuAccumulateBatchesAndSort::new(
                source_per_row_group(),
                vec![ColumnOrder {
                    column: 1,
                    ascending: true,
                    nulls_first: false,
                }],
                fetch,
            );
            let tree: Box<dyn GpuNode> = Box::new(node);
            let session = Session::open(tree.as_ref());
            let accumulator = GpuAccumulator::sorted(session.executor, session.recipe(1), &out)
                .expect("the sort builds");
            at_done(&session, accumulator, &out)
        }
        Shape::CoalesceLane => {
            let out = columns();
            let tree: Box<dyn GpuNode> =
                Box::new(GpuCoalesceAllBatches::new(source_per_row_group()));
            let session = Session::open(tree.as_ref());
            let accumulator = GpuAccumulator::coalesce(session.executor, session.recipe(1), &out)
                .expect("the coalesce builds");
            at_done(&session, accumulator, &out)
        }
        Shape::SumByKey { finalize } => {
            let keys = [("k", DataType::Utf8)];
            let (state, init, mut merge) = summed(&keys);
            let output = match finalize {
                true => {
                    merge.finalize = Some(vec![NamedExpr::new(Expr::column(1, "sum(v)"), "total")]);
                    Schema::new(Arc::new(schema_of(&[
                        ("k", DataType::Utf8),
                        ("total", DataType::Int64),
                    ])))
                }
                false => state.clone(),
            };
            merged(source_per_row_group(), state, init, merge, output)
        }
        Shape::SumByKeyAndGroupingId => {
            // The gid is a plan value here: a project puts it beside the key, which is the
            // shape a grouping-set init emits and the width the merge reads past.
            let keys = [("k", DataType::Utf8), ("__grouping_id", DataType::Int64)];
            let (state, _, merge) = summed(&keys);
            let with_id = GpuProject::new(
                source_per_row_group(),
                vec![
                    NamedExpr::new(Expr::column(0, "k"), "k"),
                    NamedExpr::new(Expr::Literal(ScalarValue::Int64(Some(0))), "__grouping_id"),
                    NamedExpr::new(Expr::column(1, "v"), "v"),
                ],
                Schema::new(Arc::new(schema_of(&[
                    ("k", DataType::Utf8),
                    ("__grouping_id", DataType::Int64),
                    ("v", DataType::Int64),
                ]))),
            );
            let widened = Schema::new(Arc::new(schema_of(&[
                ("k", DataType::Utf8),
                ("__grouping_id", DataType::Int64),
                ("v", DataType::Int64),
            ])));
            let init = AggregateBody {
                group_by: vec![Expr::column(0, "k"), Expr::column(1, "__grouping_id")],
                grouping_sets: Vec::new(),
                null_exprs: Vec::new(),
                aggs: vec![AggCall {
                    func: PlanAgg::Sum,
                    args: vec![Expr::column(2, "v")],
                    outputs: vec![Field::new("sum(v)", DataType::Int64, true)],
                }],
                finalize: None,
            };
            let answered = merged_over(
                Box::new(with_id),
                &[widened.fields.as_ref().clone()],
                state.clone(),
                init,
                merge,
                state,
            );
            answered
                .into_iter()
                .map(|row| {
                    let cells: Vec<&str> = row.split('|').collect();
                    format!("{}|{}", cells[0], cells[2])
                })
                .collect()
        }
        Shape::FinishWithNoProbe { join_type } => {
            // The build side set and done called with no probe batch. Before #173 the device
            // refused here; now the concat of no keys answers with an empty table of the key
            // schema and the finish computes against it, which is what the cpu backend does
            // with an empty concat of its own. The rows are the contract's, not this file's.
            let out = joined_schema(&columns(), &columns(), join_type);
            let tree = finishing_join(join_type, &out);
            let session = Session::open(tree.as_ref());
            let keys = schema_of(&[("k", DataType::Utf8)]);
            let join = GpuJoinExec::new(
                session.executor,
                session.recipe(2),
                Some(join_type),
                Some(&keys),
                &out.fields,
            )
            .expect("the join builds");
            let (probing, _): (_, _) = join
                .set_build(session.scan(&ROW_GROUPS))
                .expect("the build side is set");
            let (finished, _) = probing.finish_and_fetch().expect("the finish runs");
            let mut answered = Vec::new();
            for batch in finished {
                let (back, _) = session
                    .export(&out.fields)
                    .unload(batch, RowRange::WHOLE)
                    .expect("the rows cross the boundary");
                answered.extend(rendered(&back));
            }
            answered
        }
        Shape::ScatterLanes { lanes } => {
            let out = columns();
            let tree: Box<dyn GpuNode> = Box::new(GpuEmitPartitions::new(
                source_per_row_group(),
                vec![0],
                lanes,
            ));
            let session = Session::open(tree.as_ref());
            let mut emitter = GpuEmitter::new(session.executor, session.recipe(1), &out)
                .expect("the scatter builds");
            let mut answered = Vec::new();
            for group in ROW_GROUPS {
                let (per_lane, _) = emitter.emit(session.scan(&[group])).expect("it runs");
                for (lane, batch) in per_lane.into_iter().enumerate() {
                    let (back, _) = session
                        .export(&out)
                        .unload(batch, RowRange::WHOLE)
                        .expect("the lane crosses the boundary");
                    for row in rendered(&back) {
                        answered.push(format!("{lane}|{row}"));
                    }
                }
            }
            answered
        }
    }
}

/// A join of `join_type` over two copies of the fixture, keyed on `k`, declaring `out`. Both
/// sides the same shape because what is under test is what the finish owes rather than what
/// it matches — the probe never produces a batch.
fn finishing_join(join_type: JoinType, out: &Schema) -> Box<dyn GpuNode> {
    Box::new(GpuJoin::new(
        source(),
        source_per_row_group(),
        join_type,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        None,
        out.clone(),
        out.clone(),
    ))
}

fn per_batch(tree: Box<dyn GpuNode>, out: &ArrowSchema) -> Vec<String> {
    let session = Session::open(tree.as_ref());
    let mut exec = session.exec(1, out);
    let mut answered = Vec::new();
    for group in ROW_GROUPS {
        let (produced, _) = exec.exec(session.scan(&[group])).expect("the node runs");
        let (back, _) = session
            .export(out)
            .unload(produced, RowRange::WHOLE)
            .expect("the rows cross the boundary");
        answered.extend(rendered(&back));
    }
    answered
}

fn at_done(session: &Session, accumulator: GpuAccumulator, out: &ArrowSchema) -> Vec<String> {
    let mut accumulator = accumulator;
    for group in ROW_GROUPS {
        accumulator
            .accumulate_and_fetch(session.scan(&[group]))
            .expect("the arrival is accepted");
    }
    let (emitted, _) = accumulator.mark_done_and_fetch().expect("done is accepted");
    let mut answered = Vec::new();
    for batch in emitted {
        let (back, _) = session
            .export(out)
            .unload(batch, RowRange::WHOLE)
            .expect("the rows cross the boundary");
        answered.extend(rendered(&back));
    }
    answered
}

fn merged(
    source: Box<dyn GpuNode>,
    state: Schema,
    init: AggregateBody,
    merge: AggregateBody,
    output: Schema,
) -> Vec<String> {
    merged_over(source, &[], state, init, merge, output)
}

/// The two nodes the planner stacks, driven by hand: an init per batch and a merge at done.
fn merged_over(
    source: Box<dyn GpuNode>,
    between: &[ArrowSchema],
    state: Schema,
    init: AggregateBody,
    merge: AggregateBody,
    output: Schema,
) -> Vec<String> {
    let init_node = GpuAggregate::new(source, init, state.clone(), state.clone());
    let tree: Box<dyn GpuNode> = Box::new(GpuAggregateBatches::new(
        Box::new(init_node),
        merge,
        state.clone(),
        output.clone(),
    ));
    let session = Session::open(tree.as_ref());
    let state_columns = state.fields.as_ref().clone();
    let out = output.fields.as_ref().clone();
    let merge_index = session.recipes_len() - 1;
    let mut partial = session.exec(merge_index - 1, &state_columns);
    let mut accumulator = GpuAccumulator::aggregate(
        session.executor,
        session.recipe(merge_index),
        &state_columns,
        &out,
        1 << 30,
    )
    .expect("the merge builds");
    for group in ROW_GROUPS {
        // Every exec node the tree puts between the scan and the init runs first: the
        // batch the init reads is the one its own child produced, not the scan's.
        let mut batch = session.scan(&[group]);
        for (offset, schema) in between.iter().enumerate() {
            batch = session
                .exec(offset + 1, schema)
                .exec(batch)
                .expect("a node between the scan and the init runs")
                .0;
        }
        let (partials, _) = partial.exec(batch).expect("the init runs");
        accumulator
            .accumulate_and_fetch(partials)
            .expect("the arrival is accepted");
    }
    let (emitted, _) = accumulator.mark_done_and_fetch().expect("done is accepted");
    let mut answered = Vec::new();
    for batch in emitted {
        let (back, _) = session
            .export(&out)
            .unload(batch, RowRange::WHOLE)
            .expect("the rows cross the boundary");
        answered.extend(rendered(&back));
    }
    answered
}

#[test]
fn every_case_answers_what_the_contract_says() {
    for case in CASES {
        assert_eq!(
            answer(case.shape),
            case.expect
                .iter()
                .map(|row| row.to_string())
                .collect::<Vec<String>>(),
            "{}",
            case.name
        );
    }
}
