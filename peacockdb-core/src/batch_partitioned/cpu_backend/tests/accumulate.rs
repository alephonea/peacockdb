//! The accumulators: what each one holds, and what it emits once its input is complete.
//!
//! Every state batch here is written down rather than produced by an init aggregate — a
//! merge that reads state has to be provable without the node below it — and every
//! expected answer is hand-computed.

use super::*;
use datafusion::execution::context::SessionContext;
use crate::batch_partitioned::cpu_backend::accumulate::{CpuAccumulator, CpuPartitionAccumulator};
use crate::batch_partitioned::executor::LaneEvent;
use crate::batch_partitioned::node::RowInterval;
use crate::batch_partitioned::nodes::{
    GpuAccumulateBatchesAndSort, GpuAggregateBatches, GpuCoalesceAllBatches, GpuLimit,
    GpuMergeSortedPartitions,
};
use datafusion::arrow::array::Float64Array;
use datafusion::arrow::datatypes::UInt64Type;

/// A lane's worth of arrivals through one accumulator, and what it answered with at done.
fn drive(mut accumulator: CpuAccumulator, arrivals: Vec<CpuBatch>) -> Vec<CpuBatch> {
    let mut out = Vec::new();
    for batch in arrivals {
        let (produced, _) = accumulator
            .accumulate_and_fetch(batch)
            .expect("the arrival is accepted");
        out.extend(produced);
    }
    let (last, _) = accumulator.mark_done_and_fetch().expect("done is accepted");
    out.extend(last);
    out
}

fn numbers(values: Vec<i64>) -> CpuBatch {
    grouped(
        values.iter().map(|_| Some("a")).collect(),
        values.into_iter().map(Some).collect(),
    )
}

fn values_of(batch: &CpuBatch) -> Vec<i64> {
    let record = batch.record_batch();
    let column = record
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("the value column");
    (0..record.num_rows())
        .map(|row| column.value(row))
        .collect()
}

fn keys_of(batch: &CpuBatch) -> Vec<String> {
    let record = batch.record_batch();
    let column = record
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("the key column");
    (0..record.num_rows())
        .map(|row| column.value(row).to_string())
        .collect()
}

fn coalesce() -> CpuAccumulator {
    let node = GpuCoalesceAllBatches::new(Given::of(&GROUPED));
    CpuAccumulator::coalesce(&node, &schema_of(&GROUPED).fields)
}

#[test]
fn a_coalesce_answers_with_one_batch_holding_every_row_it_was_given() {
    let out = drive(
        coalesce(),
        vec![numbers(vec![1, 2]), numbers(vec![3]), numbers(vec![4, 5])],
    );
    assert_eq!(out.len(), 1, "a coalesce emits nothing until done");
    assert_eq!(values_of(&out[0]), vec![1, 2, 3, 4, 5]);
}

/// An empty lane emits nothing, on this backend as on the device — the collapse of no
/// handles is a refusal there (#173), so a batch invented here would be a row the other
/// engine cannot produce.
#[test]
fn a_coalesce_that_received_nothing_emits_nothing() {
    assert!(drive(coalesce(), Vec::new()).is_empty());
}

/// A session whose sort cannot stay in place, so `ExternalSorter` takes the arm that
/// spawns instead of the one that concatenates and sorts once. Stated rather than reached
/// by making the input large: a threshold in bytes and a fixture in rows are two facts
/// that drift apart, and the test would go quiet rather than red.
fn spilling_sort(ascending: bool, fetch: Option<usize>) -> CpuAccumulator {
    let config = datafusion::prelude::SessionConfig::new()
        .set_u64("datafusion.execution.sort_in_place_threshold_bytes", 0);
    sort_with(ascending, fetch, SessionContext::new_with_config(config).task_ctx())
}

fn accumulating_sort(ascending: bool, fetch: Option<usize>) -> CpuAccumulator {
    sort_with(ascending, fetch, ctx())
}

fn sort_with(ascending: bool, fetch: Option<usize>, ctx: Arc<TaskContext>) -> CpuAccumulator {
    let node = GpuAccumulateBatchesAndSort::new(
        Given::of(&GROUPED),
        vec![ColumnOrder {
            column: 1,
            ascending,
            nulls_first: false,
        }],
        fetch,
    );
    CpuAccumulator::sorted(&node, &schema_of(&GROUPED).fields, ctx).expect("the sort builds")
}

/// The whole stream, not each batch: batches that are each sorted and not sorted against
/// each other is exactly the shape that reads as ordered and is not.
#[test]
fn an_accumulating_sort_orders_the_stream_and_not_each_batch() {
    let out = drive(
        accumulating_sort(true, None),
        vec![
            numbers(vec![2, 5]),
            numbers(vec![1, 6]),
            numbers(vec![3, 4]),
        ],
    );
    assert_eq!(out.len(), 1);
    assert_eq!(values_of(&out[0]), vec![1, 2, 3, 4, 5, 6]);
}

/// The fetch is a top-N over the stream. Per batch it would keep N from each and answer
/// with N times the batch count.
#[test]
fn an_accumulating_sort_with_a_fetch_keeps_the_top_of_the_stream() {
    let out = drive(
        accumulating_sort(false, Some(2)),
        vec![
            numbers(vec![2, 5]),
            numbers(vec![1, 6]),
            numbers(vec![3, 4]),
        ],
    );
    assert_eq!(values_of(&out[0]), vec![6, 5], "the two largest of all six");
}

/// Long enough to leave the insertion sort behind. Rust sorts under twenty elements in
/// place, which is stable whether or not the sort claims to be — so a shorter case pins
/// nothing about the sort it names.
const PAST_THE_CUTOFF: usize = 40;

/// Ten distinct keys over forty rows, so every key is tied four ways.
fn tied_rows() -> Vec<CpuBatch> {
    (0..PAST_THE_CUTOFF)
        .map(|row| grouped(vec![Some("k")], vec![Some((row % 10) as i64)]))
        .collect()
}

fn ordered_values(batch: &CpuBatch) -> Vec<i64> {
    values_of(batch)
}

/// What a sort here does promise: the keys come out ordered, every row survives, and the
/// same input answers the same way twice. What it does not promise is tie order — neither
/// DataFusion's sort nor cuDF's is stable — so a plan that needs one asks for a key that
/// decides it.
#[test]
fn an_accumulating_sort_orders_its_keys_and_answers_the_same_way_twice() {
    let once = drive(accumulating_sort(true, None), tied_rows());
    let again = drive(accumulating_sort(true, None), tied_rows());
    let values = ordered_values(&once[0]);
    assert_eq!(values.len(), PAST_THE_CUTOFF, "every row survives the sort");
    assert!(
        values.windows(2).all(|pair| pair[0] <= pair[1]),
        "the keys are ordered: {values:?}"
    );
    assert_eq!(values, ordered_values(&again[0]), "and the same run twice");
}

/// The same under a fetch, which is the half that used to keep a heap: the rows kept are
/// the ones whose keys win, and which of four tied rows was kept does not change between
/// runs.
#[test]
fn a_fetch_keeps_the_rows_whose_keys_win_and_keeps_the_same_ones() {
    let kept = 6;
    let once = drive(accumulating_sort(true, Some(kept)), tied_rows());
    let again = drive(accumulating_sort(true, Some(kept)), tied_rows());
    let values = ordered_values(&once[0]);
    assert_eq!(values.len(), kept);
    assert!(
        values.iter().all(|value| *value <= 1),
        "six of the forty rows, and the keys 0 and 1 are the six smallest: {values:?}"
    );
    assert_eq!(values, ordered_values(&again[0]));
}

/// `[k, sum(v), count(v)]` — the state an avg decomposes into, which is what a merge reads.
fn avg_state() -> Schema {
    state_of(
        &[
            ("k", DataType::Utf8),
            ("avg(v)$sum", DataType::Int64),
            ("avg(v)$count", DataType::Int64),
        ],
        1,
        None,
    )
}

fn avg_state_batch(rows: Vec<(&str, i64, i64)>) -> CpuBatch {
    let keys: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|(k, _, _)| Some(*k)).collect::<Vec<_>>(),
    ));
    let sums: ArrayRef = Arc::new(Int64Array::from(
        rows.iter().map(|(_, s, _)| Some(*s)).collect::<Vec<_>>(),
    ));
    let counts: ArrayRef = Arc::new(Int64Array::from(
        rows.iter().map(|(_, _, c)| Some(*c)).collect::<Vec<_>>(),
    ));
    CpuBatch::new(
        RecordBatch::try_new(avg_state().fields.clone(), vec![keys, sums, counts])
            .expect("the columns fit the state"),
    )
}

/// The merge body an avg's state takes: a count merges by sum, which is the one place
/// naming the merge separately from the init is the difference between a right and a
/// wrong answer.
fn avg_merge_body(finalize: Option<Vec<NamedExpr>>) -> AggregateBody {
    let summing = |column: u32, output: &str| AggCall {
        func: PlanAgg::Sum,
        args: vec![Expr::column(column, output)],
        outputs: vec![Field::new(output, DataType::Int64, true)],
    };
    AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![summing(1, "avg(v)$sum"), summing(2, "avg(v)$count")],
        finalize,
    }
}

fn merging(body: AggregateBody, output: Schema, compact_bytes: usize) -> CpuAccumulator {
    let state = avg_state();
    let node =
        GpuAggregateBatches::new(Given::of_schema(state.clone()), body, state.clone(), output);
    CpuAccumulator::aggregate(&node, &state.fields, ctx(), compact_bytes).expect("the merge builds")
}

#[test]
fn a_merge_folds_state_into_state_and_emits_at_done() {
    let out = drive(
        merging(avg_merge_body(None), avg_state(), 1 << 20),
        vec![
            avg_state_batch(vec![("a", 12, 3), ("b", 5, 1)]),
            avg_state_batch(vec![("a", 10, 1)]),
        ],
    );
    assert_eq!(out.len(), 1, "a merge emits nothing until done");
    assert_eq!(
        by_key(&out[0]),
        vec![
            (
                "a".to_string(),
                vec![ScalarValue::Int64(Some(22)), ScalarValue::Int64(Some(4))]
            ),
            (
                "b".to_string(),
                vec![ScalarValue::Int64(Some(5)), ScalarValue::Int64(Some(1))]
            ),
        ],
        "the sums and the counts add, and the state keeps its shape"
    );
}

#[test]
fn a_merge_that_finalizes_emits_the_finalized_row() {
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
    let body = avg_merge_body(Some(vec![NamedExpr::new(average, "avg(v)")]));
    let out = drive(
        merging(body, output, 1 << 20),
        vec![
            avg_state_batch(vec![("a", 12, 3)]),
            avg_state_batch(vec![("a", 10, 1)]),
        ],
    );
    assert_eq!(
        by_key(&out[0]),
        vec![("a".to_string(), vec![ScalarValue::Float64(Some(5.5))])],
        "22 over 4, and the state columns are gone from the row"
    );
}

/// The same for a merge, and the reason is not only the device: a grouped merge over no
/// arrivals owes no groups. The shape that would owe a row — a global aggregate's identity
/// row, count 0 rather than no row — has one lane and no scatter above it, so its lane is
/// never the empty one.
#[test]
fn a_merge_that_received_nothing_emits_nothing() {
    let output = state_of(
        &[("k", DataType::Utf8), ("avg(v)", DataType::Float64)],
        1,
        None,
    );
    let average = NamedExpr::new(Expr::column(1, "avg(v)$sum"), "avg(v)");
    let emitted = drive(
        merging(avg_merge_body(Some(vec![average])), output, 1 << 20),
        Vec::new(),
    );
    assert!(emitted.is_empty());
}

/// `[k, count, mean, m2]` — the Welford triple, in DataFusion's own state order and types.
fn welford_state() -> Schema {
    state_of(
        &[
            ("k", DataType::Utf8),
            ("stddev(v)$count", DataType::UInt64),
            ("stddev(v)$mean", DataType::Float64),
            ("stddev(v)$m2", DataType::Float64),
        ],
        1,
        Some("stddev(v)"),
    )
}

fn welford_batch(key: &str, count: u64, mean: f64, m2: f64) -> CpuBatch {
    let keys: ArrayRef = Arc::new(StringArray::from(vec![Some(key)]));
    let counts: ArrayRef = Arc::new(
        vec![Some(count)]
            .into_iter()
            .collect::<datafusion::arrow::array::PrimitiveArray<UInt64Type>>(),
    );
    let means: ArrayRef = Arc::new(Float64Array::from(vec![Some(mean)]));
    let m2s: ArrayRef = Arc::new(Float64Array::from(vec![Some(m2)]));
    CpuBatch::new(
        RecordBatch::try_new(
            welford_state().fields.clone(),
            vec![keys, counts, means, m2s],
        )
        .expect("the columns fit the state"),
    )
}

/// The one aggregate whose merge is not a per-column reduction: combining two partials
/// needs the count-weighted mean and the cross term. [2, 4, 6] and [10] merge to a count
/// of 4, a mean of 5.5 and an m2 of 35 — which is the sum of squared deviations from 5.5,
/// computed by hand and not by another accumulator.
#[test]
fn a_welford_triple_merges_as_one_aggregate() {
    let body = AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![AggCall {
            func: PlanAgg::MergeM2,
            args: vec![
                Expr::column(1, "stddev(v)$count"),
                Expr::column(2, "stddev(v)$mean"),
                Expr::column(3, "stddev(v)$m2"),
            ],
            outputs: welford_state().fields.fields()[1..]
                .iter()
                .map(|field| field.as_ref().clone())
                .collect(),
        }],
        finalize: None,
    };
    let state = welford_state();
    let node = GpuAggregateBatches::new(
        Given::of_schema(state.clone()),
        body,
        state.clone(),
        state.clone(),
    );
    let accumulator =
        CpuAccumulator::aggregate(&node, &state.fields, ctx(), 1 << 20).expect("the merge builds");
    let out = drive(
        accumulator,
        vec![
            welford_batch("a", 3, 4.0, 8.0),
            welford_batch("a", 1, 10.0, 0.0),
        ],
    );
    assert_eq!(
        by_key(&out[0]),
        vec![(
            "a".to_string(),
            vec![
                ScalarValue::UInt64(Some(4)),
                ScalarValue::Float64(Some(5.5)),
                ScalarValue::Float64(Some(35.0)),
            ]
        )],
        "the count-weighted mean and the cross term, not an average of the two means"
    );
}

/// Rows per arrival. Large enough that arrow's own accounting separates a state of one
/// row from a state of thousands, which is the difference the threshold reads.
const ROWS: usize = 50;
const ARRIVALS: usize = 40;

/// A state batch of `ROWS` rows under keys the caller names.
fn state_under(keys: Vec<String>) -> CpuBatch {
    let names: ArrayRef = Arc::new(StringArray::from(
        keys.iter()
            .map(|key| Some(key.as_str()))
            .collect::<Vec<_>>(),
    ));
    let ones: ArrayRef = Arc::new(Int64Array::from(vec![Some(1i64); keys.len()]));
    CpuBatch::new(
        RecordBatch::try_new(avg_state().fields.clone(), vec![names, ones.clone(), ones])
            .expect("the columns fit the state"),
    )
}

/// Every key distinct, so nothing ever merges and the state grows by `ROWS` each time.
fn disjoint_arrivals() -> Vec<CpuBatch> {
    (0..ARRIVALS)
        .map(|arrival| {
            state_under(
                (0..ROWS)
                    .map(|row| format!("key-{arrival}-{row}"))
                    .collect(),
            )
        })
        .collect()
}

/// Every row under one key, so every compaction leaves the state a single row.
fn shared_arrivals() -> Vec<CpuBatch> {
    (0..ARRIVALS)
        .map(|_| state_under(vec!["key".to_string(); ROWS]))
        .collect()
}

fn compactions_over(arrivals: Vec<CpuBatch>, compact_bytes: usize) -> usize {
    let mut accumulator = merging(avg_merge_body(None), avg_state(), compact_bytes);
    for batch in arrivals {
        accumulator
            .accumulate_and_fetch(batch)
            .expect("the arrival is accepted");
    }
    match &accumulator {
        CpuAccumulator::Aggregate(state) => state.compactions(),
        _ => panic!("a merge is an Aggregate accumulator"),
    }
}

/// The doubling, shown by the two regimes it exists to separate. Where the keys are
/// disjoint a compaction shrinks nothing, so the threshold moves to twice what it left and
/// the compactions land at geometrically growing sizes. Where one key repeats the state
/// stays a row wide, the threshold never moves, and an arrival compacts whenever the
/// pending arrivals fill it.
///
/// Asserted as the comparison rather than as two numbers: without the doubling the disjoint
/// run compacts as often as the shared one, which is the mutation this catches.
#[test]
fn disjoint_keys_raise_the_threshold_where_a_repeating_key_does_not() {
    let one_arrival = state_under(vec!["key".to_string(); ROWS])
        .record_batch()
        .get_array_memory_size();
    let disjoint = compactions_over(disjoint_arrivals(), one_arrival);
    let shared = compactions_over(shared_arrivals(), one_arrival);
    assert!(
        disjoint < shared,
        "{ARRIVALS} disjoint arrivals compacted {disjoint} times and {ARRIVALS} sharing a \
         key compacted {shared}: the threshold is not moving where a compaction shrinks \
         nothing"
    );
    assert!(
        disjoint <= 10,
        "the disjoint run compacted {disjoint} times over {ARRIVALS} arrivals, which is \
         per-arrival rather than geometric"
    );
}

fn limiting(skip: u64, fetch: Option<u64>) -> CpuAccumulator {
    let node = GpuLimit::new(Given::of(&GROUPED), RowInterval { skip, fetch });
    CpuAccumulator::limit(&node)
}

/// The three things a limit does to a batch, in one stream: the first is wholly before the
/// interval, the second straddles its start, the third is wholly inside, the fourth
/// straddles its end and the fifth is wholly past it.
#[test]
fn a_limit_drops_forwards_and_slices_by_where_the_batch_falls() {
    let out = drive(
        limiting(3, Some(6)),
        vec![
            numbers(vec![1, 2, 3]),
            numbers(vec![4, 5, 6]),
            numbers(vec![7, 8]),
            numbers(vec![9, 10, 11]),
            numbers(vec![12, 13]),
        ],
    );
    let kept: Vec<Vec<i64>> = out.iter().map(values_of).collect();
    assert_eq!(
        kept,
        vec![vec![4, 5, 6], vec![7, 8], vec![9]],
        "rows 3..9 of the stream, and no batch emitted for the two outside it"
    );
}

/// A pure offset never satisfies, so every batch past the skip is forwarded whole.
#[test]
fn a_limit_with_no_fetch_forwards_everything_past_its_offset() {
    let out = drive(
        limiting(2, None),
        vec![numbers(vec![1, 2, 3]), numbers(vec![4, 5])],
    );
    let kept: Vec<Vec<i64>> = out.iter().map(values_of).collect();
    assert_eq!(kept, vec![vec![3], vec![4, 5]]);
}

#[test]
fn a_limit_emits_nothing_at_done_because_it_held_nothing() {
    let mut accumulator = limiting(0, Some(2));
    let (produced, _) = accumulator
        .accumulate_and_fetch(numbers(vec![1, 2, 3]))
        .expect("the arrival is accepted");
    assert_eq!(produced.len(), 1);
    let (last, _) = accumulator.mark_done_and_fetch().expect("done is accepted");
    assert!(last.is_empty(), "a limit that streams has nothing to flush");
}

fn merge_sorted(lanes: usize, fetch: Option<usize>) -> CpuPartitionAccumulator {
    let node = GpuMergeSortedPartitions::new(
        Given::of(&GROUPED),
        vec![ColumnOrder {
            column: 1,
            ascending: true,
            nulls_first: false,
        }],
        fetch,
    );
    CpuPartitionAccumulator::merge_sorted(&node, lanes, &schema_of(&GROUPED).fields, ctx())
        .expect("the merge builds")
}

/// One call per lane event, and the call carrying the last `Done` is the emitting one.
fn drive_lanes(
    mut accumulator: CpuPartitionAccumulator,
    events: Vec<(usize, Option<CpuBatch>)>,
) -> Vec<CpuBatch> {
    let mut out = Vec::new();
    for (lane, batch) in events {
        let event = match batch {
            Some(batch) => LaneEvent::Batch(batch),
            None => LaneEvent::Done,
        };
        let (produced, _) = accumulator
            .accumulate_and_fetch(lane, event)
            .expect("the event is accepted");
        out.extend(produced);
    }
    out
}

#[test]
fn every_lanes_run_is_merged_into_one_batch_at_the_last_done() {
    let out = drive_lanes(
        merge_sorted(2, None),
        vec![
            (0, Some(numbers(vec![1, 4]))),
            (1, Some(numbers(vec![2, 3]))),
            (0, None),
            (1, None),
        ],
    );
    assert_eq!(out.len(), 1, "one batch, and only at the last done");
    assert_eq!(values_of(&out[0]), vec![1, 2, 3, 4]);
}

/// Until every lane has reported, the answer could still change: a lane yet to send its
/// rows may hold the smallest of them.
#[test]
fn nothing_is_emitted_while_a_lane_could_still_send() {
    let out = drive_lanes(
        merge_sorted(3, None),
        vec![
            (0, Some(numbers(vec![1]))),
            (0, None),
            (1, None),
            (2, Some(numbers(vec![2]))),
        ],
    );
    assert!(out.is_empty(), "lane 2 has not reported done");
}

/// The merge across lanes, past the cutoff: every lane's rows arrive, the keys come out
/// ordered, and the answer does not move between runs. Which of two rows tied on the keys
/// comes first is not promised — a k-way merge over unstable per-lane sorts has no order
/// to preserve.
#[test]
fn the_merge_orders_every_lanes_rows_and_answers_the_same_way_twice() {
    let events = || {
        let mut events: Vec<(usize, Option<CpuBatch>)> = (0..PAST_THE_CUTOFF)
            .map(|row| {
                (
                    row % 2,
                    Some(grouped(vec![Some("k")], vec![Some((row % 10) as i64)])),
                )
            })
            .collect();
        events.push((0, None));
        events.push((1, None));
        events
    };
    let once = drive_lanes(merge_sorted(2, None), events());
    let again = drive_lanes(merge_sorted(2, None), events());
    let values = values_of(&once[0]);
    assert_eq!(values.len(), PAST_THE_CUTOFF, "both lanes' rows arrive");
    assert!(
        values.windows(2).all(|pair| pair[0] <= pair[1]),
        "the keys are ordered across lanes: {values:?}"
    );
    assert_eq!(values, values_of(&again[0]));
}

#[test]
fn a_fetch_over_the_merge_keeps_the_top_of_every_lane_together() {
    let out = drive_lanes(
        merge_sorted(2, Some(3)),
        vec![
            (0, Some(numbers(vec![1, 6]))),
            (1, Some(numbers(vec![2, 5]))),
            (0, None),
            (1, None),
        ],
    );
    assert_eq!(values_of(&out[0]), vec![1, 2, 5]);
}

/// A global aggregate's lane can be empty — a mid-plan limit that dropped every batch of
/// the one lane is how — and what it owes then is its identity row rather than no row.
/// `count(*)` over nothing is 0, and a query asking for it gets an answer.
#[test]
fn a_global_aggregate_that_received_nothing_still_owes_its_identity_row() {
    let counted = Schema::new(Arc::new(
        schema_of(&[("count(v)", DataType::Int64)])
            .fields
            .as_ref()
            .clone(),
    ));
    let body = AggregateBody {
        group_by: Vec::new(),
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![AggCall {
            func: PlanAgg::Sum,
            args: vec![Expr::column(0, "count(v)")],
            outputs: vec![Field::new("count(v)", DataType::Int64, true)],
        }],
        finalize: None,
    };
    let node = GpuAggregateBatches::new(
        Given::of_schema(counted.clone()),
        body,
        counted.clone(),
        counted.clone(),
    );
    let accumulator = CpuAccumulator::aggregate(&node, &counted.fields, ctx(), 1 << 20)
        .expect("the merge builds");
    let emitted = drive(accumulator, Vec::new());
    assert_eq!(
        emitted.len(),
        1,
        "a global aggregate answers whatever arrived"
    );
    assert_eq!(
        emitted[0].record_batch().num_rows(),
        1,
        "and its answer is one row, not an empty batch"
    );
}

/// The one case that runs under a runtime with threads to spare, because every other test
/// here is a plain `#[test]` where DataFusion never spawns.
///
/// A call blocks a thread on one node's stream, and a sort past its in-place threshold
/// spawns onto the runtime from under that block. On a single worker that is a deadlock;
/// this is the shape a driver has to leave room for, and the header that says so was on no
/// test's path until now.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_accumulating_sort_answers_from_under_a_blocked_worker() {
    const ROWS: usize = 4096;
    let arrivals: Vec<CpuBatch> = (0..8)
        .map(|batch| {
            let values: Vec<Option<i64>> = (0..ROWS / 8)
                .map(|row| Some(((batch * ROWS / 8 + row) % 997) as i64))
                .collect();
            grouped(values.iter().map(|_| Some("k")).collect(), values)
        })
        .collect();
    let answered =
        tokio::task::spawn_blocking(move || drive(accumulating_sort(true, None), arrivals))
            .await
            .expect("the sort finishes rather than deadlocking under its own spawn");
    let values = values_of(&answered[0]);
    assert_eq!(values.len(), ROWS);
    assert!(
        values.windows(2).all(|pair| pair[0] <= pair[1]),
        "and the answer is ordered"
    );
}
