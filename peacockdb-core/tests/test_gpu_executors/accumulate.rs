//! The accumulators on a device: three batches into each, and what it emits at done.
//!
//! The source is mapped one batch per row group here, so a lane genuinely has several
//! batches to accumulate — the one shape the exec cases never see.

use super::*;

use peacockdb_core::batch_partitioned::executor::LaneEvent;
use peacockdb_core::batch_partitioned::gpu_backend::accumulate::{
    GpuAccumulator, GpuPartitionAccumulator,
};
use peacockdb_core::batch_partitioned::node::RowInterval;
use peacockdb_core::batch_partitioned::nodes::{
    GpuAccumulateBatchesAndSort, GpuAggregateBatches, GpuCoalesceAllBatches, GpuLimit,
    GpuMergeSortedPartitions,
};

/// Held bytes at which a compaction runs. Large enough that nothing here compacts before
/// done, since what these cases are about is the answer rather than the schedule.
const NO_COMPACTION: usize = 1 << 30;

/// One lane of three batches through one accumulator, and what it answered with.
fn accumulate(session: &Session, accumulator: GpuAccumulator, out: &ArrowSchema) -> Vec<CpuBatch> {
    let mut accumulator = accumulator;
    let mut emitted = Vec::new();
    for group in ROW_GROUPS {
        let (produced, _) = accumulator
            .accumulate_and_fetch(session.scan(&[group]))
            .expect("the arrival is accepted");
        emitted.extend(produced);
    }
    let (last, _) = accumulator.mark_done_and_fetch().expect("done is accepted");
    emitted.extend(last);
    emitted
        .into_iter()
        .map(|batch| {
            session
                .export(out)
                .unload(batch, RowRange::WHOLE)
                .expect("the rows cross the boundary")
                .0
        })
        .collect()
}

#[test]
fn a_coalesce_answers_with_one_batch_holding_every_row_of_the_lane() {
    let tree: Box<dyn GpuNode> = Box::new(GpuCoalesceAllBatches::new(source_per_row_group()));
    let session = Session::open(tree.as_ref());
    let out = columns();
    let accumulator = GpuAccumulator::coalesce(session.executor, session.recipe(1), &out)
        .expect("the coalesce builds");
    let answered = accumulate(&session, accumulator, &out);
    assert_eq!(answered.len(), 1, "a coalesce emits nothing until done");
    assert_eq!(
        rows(&answered[0])
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        VALUES
            .iter()
            .map(|v| ScalarValue::Int64(Some(*v)))
            .collect::<Vec<ScalarValue>>(),
        "the three batches in the order they arrived"
    );
}

fn accumulating_sort(ascending: bool, fetch: Option<usize>) -> Box<dyn GpuNode> {
    Box::new(GpuAccumulateBatchesAndSort::new(
        source_per_row_group(),
        vec![ColumnOrder {
            column: 1,
            ascending,
            nulls_first: false,
        }],
        fetch,
    ))
}

/// The stream, not each batch: the runs are sorted as they arrive and merged at done.
#[test]
fn an_accumulating_sort_orders_the_whole_lane() {
    let tree = accumulating_sort(true, None);
    let session = Session::open(tree.as_ref());
    let out = columns();
    let accumulator =
        GpuAccumulator::sorted(session.executor, session.recipe(1), &out).expect("the sort builds");
    let answered = accumulate(&session, accumulator, &out);
    assert_eq!(
        rows(&answered[0])
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        (1..=6)
            .map(|v| ScalarValue::Int64(Some(v)))
            .collect::<Vec<ScalarValue>>()
    );
}

/// The fetch rides the merge, so it is a top-N over the lane rather than over each batch.
#[test]
fn an_accumulating_sort_with_a_fetch_keeps_the_top_of_the_lane() {
    let tree = accumulating_sort(false, Some(2));
    let session = Session::open(tree.as_ref());
    let out = columns();
    let accumulator =
        GpuAccumulator::sorted(session.executor, session.recipe(1), &out).expect("the sort builds");
    let answered = accumulate(&session, accumulator, &out);
    assert_eq!(
        rows(&answered[0])
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        vec![ScalarValue::Int64(Some(6)), ScalarValue::Int64(Some(5))],
        "the two largest of all six"
    );
}

/// A merge reads state, so the init that builds it runs first: the same two nodes the
/// planner stacks, driven by hand. Each lane batch becomes a partial, and the merge folds
/// the three of them.
#[test]
fn a_merge_folds_the_partials_its_init_produced() {
    let state = Schema::new(Arc::new(schema_of(&[
        ("k", DataType::Utf8),
        ("sum(v)", DataType::Int64),
    ])));
    let summing = |column: u32| AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![AggCall {
            func: PlanAgg::Sum,
            args: vec![Expr::column(
                column,
                if column == 1 { "v" } else { "sum(v)" },
            )],
            outputs: vec![Field::new("sum(v)", DataType::Int64, true)],
        }],
        finalize: None,
    };
    let init = GpuAggregate::new(
        source_per_row_group(),
        summing(1),
        state.clone(),
        state.clone(),
    );
    let tree: Box<dyn GpuNode> = Box::new(GpuAggregateBatches::new(
        Box::new(init),
        summing(1),
        state.clone(),
        state.clone(),
    ));
    let session = Session::open(tree.as_ref());
    let out = schema_of(&[("k", DataType::Utf8), ("sum(v)", DataType::Int64)]);
    let mut partial = session.exec(1, &out);
    let mut merge = GpuAccumulator::aggregate(
        session.executor,
        session.recipe(2),
        &out,
        &out,
        NO_COMPACTION,
    )
    .expect("the merge builds");
    for group in ROW_GROUPS {
        let (state, _) = partial
            .exec(session.scan(&[group]))
            .expect("the init runs on this batch");
        merge
            .accumulate_and_fetch(state)
            .expect("the arrival is accepted");
    }
    let (emitted, _) = merge.mark_done_and_fetch().expect("done is accepted");
    let [merged] = <[GpuBatch; 1]>::try_from(emitted).expect("a merge emits one batch");
    let (answer, _) = session
        .export(&out)
        .unload(merged, RowRange::WHOLE)
        .expect("the rows cross the boundary");
    assert_eq!(
        by_key(&answer),
        vec![
            vec![string("a"), ScalarValue::Int64(Some(12))],
            vec![string("b"), ScalarValue::Int64(Some(9))],
        ],
        "2 + 4 + 6 under a and 1 + 3 + 5 under b, across three partials"
    );
}

/// The three things a limit does to a batch: the first two rows are before the interval,
/// the middle batch is wholly inside it, and the third straddles its end.
#[test]
fn a_limit_drops_forwards_and_slices_by_where_the_batch_falls() {
    let tree: Box<dyn GpuNode> = Box::new(GpuLimit::new(
        source_per_row_group(),
        RowInterval {
            skip: 2,
            fetch: Some(3),
        },
    ));
    let session = Session::open(tree.as_ref());
    let out = columns();
    let accumulator = GpuAccumulator::limit(
        session.executor,
        session.recipe(1),
        RowInterval {
            skip: 2,
            fetch: Some(3),
        },
        &out,
    )
    .expect("the limit builds");
    let answered = accumulate(&session, accumulator, &out);
    let kept: Vec<Vec<ScalarValue>> = answered
        .iter()
        .map(|batch| rows(batch).into_iter().map(|row| row[1].clone()).collect())
        .collect();
    assert_eq!(
        kept,
        vec![
            vec![ScalarValue::Int64(Some(4)), ScalarValue::Int64(Some(3))],
            vec![ScalarValue::Int64(Some(6))],
        ],
        "the second batch whole and one row of the third; nothing for the first"
    );
}

/// A lane that received nothing emits nothing on this backend, and the reason is an ABI
/// fact rather than a choice: the collapse arm does answer a call with no handles, but the
/// table it hands back has no columns, so it is not the batch of the node's schema that a
/// SingleBatch output owes downstream. Exporting one is what goes wrong — the stream
/// decodes to a batch of no columns and reading the first one is out of bounds.
#[test]
fn a_lane_that_received_nothing_emits_no_batch_on_this_backend() {
    let tree: Box<dyn GpuNode> = Box::new(GpuCoalesceAllBatches::new(source_per_row_group()));
    let session = Session::open(tree.as_ref());
    let out = columns();
    let accumulator = GpuAccumulator::coalesce(session.executor, session.recipe(1), &out)
        .expect("the coalesce builds");
    let (emitted, _) = accumulator
        .mark_done_and_fetch()
        .expect("done is accepted whatever arrived");
    assert!(
        emitted.is_empty(),
        "the empty batch a SingleBatch output owes is the driver's to supply"
    );
}

/// Two lanes, each sorted by the per-batch sort above it, merged into one at the last
/// lane's done. The handles enter the merge in lane order, which is what a k-way merge
/// over partitions means.
#[test]
fn every_lanes_sorted_run_is_merged_at_the_last_done() {
    let keys = vec![ColumnOrder {
        column: 1,
        ascending: true,
        nulls_first: false,
    }];
    let sorted = GpuSort::new(source_two_lanes(), keys.clone(), None);
    let tree: Box<dyn GpuNode> =
        Box::new(GpuMergeSortedPartitions::new(Box::new(sorted), keys, None));
    let session = Session::open(tree.as_ref());
    let out = columns();
    let mut per_batch = session.exec(1, &out);
    let mut merge =
        GpuPartitionAccumulator::merge_sorted(session.executor, session.recipe(2), 2, &out)
            .expect("the merge builds");
    let lanes: [(usize, &[u32]); 3] = [(0, &[0]), (1, &[1]), (1, &[2])];
    for (lane, groups) in lanes {
        let (run, _) = per_batch
            .exec(session.scan(groups))
            .expect("the sort runs on this batch");
        merge
            .accumulate_and_fetch(lane, LaneEvent::Batch(run))
            .expect("the arrival is accepted");
    }
    let mut emitted = Vec::new();
    for lane in [0, 1] {
        let (produced, _) = merge
            .accumulate_and_fetch(lane, LaneEvent::Done)
            .expect("the done is accepted");
        emitted.extend(produced);
    }
    let [merged] = <[GpuBatch; 1]>::try_from(emitted).expect("one batch, at the last done");
    let (answer, _) = session
        .export(&out)
        .unload(merged, RowRange::WHOLE)
        .expect("the rows cross the boundary");
    assert_eq!(
        rows(&answer)
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        (1..=6)
            .map(|v| ScalarValue::Int64(Some(v)))
            .collect::<Vec<ScalarValue>>(),
        "both lanes' rows in one order"
    );
}
