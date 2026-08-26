//! Both limit lowerings, asserted on the calls that were and were not made. A test on the
//! rows returned passes just as well when the whole input was read and trimmed at the end,
//! which is the thing this lowering exists to avoid.

use super::super::mock::{Script, spec};
use super::super::plans::*;
use super::*;

/// Six batches of ten rows on one lane, over a chain the interval sits on top of.
fn chain(skip: u64, fetch: Option<u64>) -> Box<dyn GpuNode> {
    unload_limited(filter(source("part", 1)), skip, fetch)
}

fn six_batches() -> Script {
    Script::default().source("part", vec![vec![spec(10, 80); 6]])
}

#[test]
fn a_root_adjacent_limit_stops_the_run_early() {
    let report = run(chain(0, Some(15)).as_ref(), &six_batches());
    assert!(report.early_exit);
    assert_eq!(
        count(&report, CallKind::NextBatch),
        2,
        "the scan stopped being scheduled once the interval was satisfied"
    );
    assert_eq!(rows_returned(&report), 15);
    assert_accounted(&report);
}

#[test]
fn without_the_limit_the_same_plan_drains_everything() {
    let report = run(unload(filter(source("part", 1))).as_ref(), &six_batches());
    assert!(!report.early_exit);
    assert_eq!(count(&report, CallKind::NextBatch), 6);
    assert_eq!(rows_returned(&report), 60);
}

#[test]
fn the_skip_prefix_is_never_unloaded() {
    let report = run(chain(25, Some(10)).as_ref(), &six_batches());
    assert_eq!(
        count(&report, CallKind::ReleaseUnwanted),
        2,
        "the two batches before row 25 were released where they stood"
    );
    assert_eq!(
        report.rows_skipped.iter().sum::<u64>(),
        20,
        "and their rows never crossed the boundary"
    );
    assert_eq!(rows_returned(&report), 10);
    assert_accounted(&report);
}

#[test]
fn only_the_straddling_batches_are_narrowed() {
    // Rows 25..35 of ten-row batches: batch 2 straddles the start, batch 3 the end.
    let report = run(chain(25, Some(10)).as_ref(), &six_batches());
    assert_eq!(count(&report, CallKind::UnloadRange), 2);
    assert_eq!(
        count(&report, CallKind::Unload),
        0,
        "no batch of this interval is wanted whole"
    );
}

#[test]
fn a_batch_wholly_inside_the_interval_is_unloaded_whole() {
    // Rows 10..40: batches 1, 2 and 3 are wanted entire, and no range rides the call.
    let report = run(chain(10, Some(30)).as_ref(), &six_batches());
    assert_eq!(count(&report, CallKind::Unload), 3);
    assert_eq!(count(&report, CallKind::UnloadRange), 0);
    assert_eq!(count(&report, CallKind::ReleaseUnwanted), 1);
    assert_eq!(rows_returned(&report), 30);
}

#[test]
fn the_count_is_across_lanes_and_not_per_lane() {
    // Three lanes of one four-row batch each under a fetch of five. Counted per lane every
    // lane would pass its own four rows and the query would return twelve.
    let plan = unload_limited(filter(source("part", 3)), 0, Some(5));
    let script = Script::default().source(
        "part",
        vec![vec![spec(4, 32)], vec![spec(4, 32)], vec![spec(4, 32)]],
    );
    let report = run(plan.as_ref(), &script);
    assert_eq!(rows_returned(&report), 5);
    assert_eq!(
        count(&report, CallKind::Unload),
        1,
        "lane 0 was wanted whole"
    );
    assert_eq!(
        count(&report, CallKind::UnloadRange),
        1,
        "lane 1 straddles the end"
    );
    assert_eq!(
        count(&report, CallKind::ReleaseUnwanted),
        1,
        "lane 2 is past it"
    );
    assert_accounted(&report);
}

#[test]
fn a_zero_fetch_unloads_nothing_at_all_and_the_plan_still_completes() {
    let report = run(chain(0, Some(0)).as_ref(), &six_batches());
    assert!(report.early_exit, "satisfied before a single step");
    assert_eq!(count(&report, CallKind::Unload), 0);
    assert_eq!(count(&report, CallKind::UnloadRange), 0);
    assert_eq!(rows_returned(&report), 0);
    assert_eq!(report.steps, 0, "nothing was ever runnable");
    assert_eq!(report.in_flight_bytes, 0);
}

#[test]
fn an_offset_with_no_fetch_never_exits_early() {
    let report = run(chain(25, None).as_ref(), &six_batches());
    assert!(
        !report.early_exit,
        "no prefix determines a pure offset, so it can only drop and trim"
    );
    assert_eq!(
        count(&report, CallKind::NextBatch),
        6,
        "every batch was read"
    );
    assert_eq!(count(&report, CallKind::ReleaseUnwanted), 2);
    assert_eq!(rows_returned(&report), 35);
    assert_accounted(&report);
}

#[test]
fn a_skip_past_the_end_returns_nothing_and_unloads_nothing() {
    let report = run(chain(1000, Some(10)).as_ref(), &six_batches());
    assert_eq!(
        count(&report, CallKind::Unload) + count(&report, CallKind::UnloadRange),
        0
    );
    assert_eq!(count(&report, CallKind::ReleaseUnwanted), 6);
    assert_eq!(rows_returned(&report), 0);
    assert_eq!(report.in_flight_bytes, 0);
}

#[test]
fn the_early_exit_reaches_through_a_shuffle() {
    let plan = unload_limited(merge(emit(source("part", 1), 4)), 0, Some(6));
    let script = Script::default().source("part", vec![vec![spec(8, 64); 5]]);
    let report = run(plan.as_ref(), &script);
    assert!(report.early_exit);
    assert!(
        count(&report, CallKind::NextBatch) < 5,
        "the hold reached the scan through the merge and the emit"
    );
    assert_eq!(
        report.in_flight_bytes, 0,
        "every batch still queued when the run stopped was released"
    );
}

#[test]
fn a_mid_plan_limit_stops_its_own_subtree_and_holds_nothing() {
    // The mid-plan node is an accumulator by category and streams: the driver counts the
    // rows going past it, and satisfaction stops the scan the same way it does at a sink.
    let plan = unload(filter(limit(merge(source("part", 1)), 0, Some(12))));
    let script = Script::default().source("part", vec![vec![spec(10, 80); 6]]);
    let report = run(plan.as_ref(), &script);
    assert!(report.early_exit);
    assert_eq!(
        count(&report, CallKind::NextBatch),
        2,
        "the scan stopped once twelve rows had gone past the limit"
    );
    let limit_node = 2;
    assert!(
        report.peak_queued[limit_node] <= 1,
        "a limit holds nothing: it forwards or releases each batch as it arrives"
    );
}

#[test]
fn a_satisfied_limit_reports_done_so_the_node_above_it_can_finish() {
    // `SELECT count(*) FROM (SELECT ... LIMIT n)`: an accumulator above the limit only
    // emits when it is told its input ended, and satisfaction is what ends it — no later
    // batch is coming. A limit that is held without being marked done leaves the parent
    // waiting for a lane that has in fact finished, and the query answers nothing.
    let plan = unload(coalesce_all(limit(merge(source("part", 1)), 0, Some(12))));
    let script = Script::default().source("part", vec![vec![spec(10, 80); 6]]);
    let report = run(plan.as_ref(), &script);
    assert!(report.early_exit);
    assert_eq!(
        count(&report, CallKind::MarkDone),
        1,
        "the accumulator above the limit was never told its input had ended"
    );
    // 20 rather than 12 because the mock's limit forwards whole batches: trimming a
    // mid-plan interval is the executor's job, so the number here means "not nothing".
    assert_eq!(
        rows_returned(&report),
        20,
        "the count reached the root instead of the query answering nothing"
    );
}

#[test]
fn a_limit_whose_only_consumer_is_the_sink_is_refused_before_the_run() {
    // The planner's canonical form, checked on the way in: a mock plan meets the same
    // refusal a planned one would, rather than the driver quietly running a tree the
    // planner does not emit.
    let plan = unload(limit(source("part", 1), 0, Some(5)));
    let script = Script::default().source("part", vec![vec![spec(10, 80)]]);
    match run_with(plan.as_ref(), &script, None) {
        Err(RunError::Backend(error)) => assert!(
            error.to_string().contains("a limit feeding only the sink"),
            "the refusal says the wrong thing: {error}"
        ),
        other => panic!("expected the planner's refusal, got {other:?}"),
    }
}
