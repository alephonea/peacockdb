//! The schedule and the two holds, asserted on calls rather than on rows.

use super::super::mock::{AccRule, EmitRule, ExecRule, JoinRule, Script, spec};
use super::super::plans::*;
use super::*;

fn one_lane_chain() -> Box<dyn GpuNode> {
    unload(filter(source("part", 1)))
}

#[test]
fn a_batch_is_carried_to_the_root_before_the_next_one_is_produced() {
    let script = Script::default().source("part", vec![vec![spec(10, 80), spec(10, 80)]]);
    let report = run(one_lane_chain().as_ref(), &script);
    let path: Vec<CallKind> = report.trace.iter().map(|event| event.call).collect();
    assert_eq!(
        path,
        vec![
            CallKind::NextBatch,
            CallKind::Exec,
            CallKind::Unload,
            CallKind::NextBatch,
            CallKind::Exec,
            CallKind::Unload,
            CallKind::SourceExhausted,
            CallKind::EndOfInput,
            CallKind::EndOfInput,
        ],
        "min-height selection walks each batch to the root before the next is produced"
    );
    assert_eq!(rows_returned(&report), 20);
    assert_accounted(&report);
}

#[test]
fn running_a_node_runs_every_one_of_its_partitions() {
    let script = Script::default().source(
        "part",
        vec![vec![spec(10, 80)], vec![spec(10, 80)], vec![spec(10, 80)]],
    );
    let report = run(unload(filter(source("part", 3))).as_ref(), &script);
    let first_step: Vec<u32> = report
        .trace
        .iter()
        .filter(|event| event.step == 1)
        .map(|event| event.lane)
        .collect();
    assert_eq!(
        first_step,
        vec![0, 1, 2],
        "one step is every lane of one node"
    );
}

#[test]
fn queues_stay_bounded_by_one_batch_per_lane_without_a_cap() {
    let batches = vec![spec(10, 80); 6];
    let script = Script::default().source("part", vec![batches.clone(), batches]);
    let report = run(unload(project(filter(source("part", 2)))).as_ref(), &script);
    for (node, queued) in report.peak_queued.iter().enumerate() {
        assert!(
            *queued <= 2,
            "node {node} held {queued} batches over two lanes"
        );
    }
    assert_accounted(&report);
}

#[test]
fn a_lane_that_runs_dry_does_not_stall_the_others() {
    // Lane 1 has nothing at all, which is what a scan whose lane got no row group looks
    // like: it is simply never runnable, and needs no mechanism of its own.
    let script = Script::default().source("part", vec![vec![spec(10, 80)], vec![]]);
    let report = run(unload(filter(source("part", 2))).as_ref(), &script);
    assert_eq!(rows_returned(&report), 10);
    assert_accounted(&report);
}

#[test]
fn an_operator_emitting_empty_batches_is_carried_through() {
    let script = Script::default()
        .source("part", vec![vec![spec(10, 80), spec(10, 80)]])
        .with_exec(ExecRule::Empty);
    let report = run(one_lane_chain().as_ref(), &script);
    assert_eq!(
        count(&report, CallKind::Unload),
        2,
        "an empty batch is still a batch"
    );
    assert_eq!(rows_returned(&report), 0);
    assert_accounted(&report);
}

// -- the join hold -----------------------------------------------------------------

/// unload <- join <- [coalesce <- source(build), filter <- source(probe)]
fn join_plan(lanes: usize) -> Box<dyn GpuNode> {
    unload(join(
        coalesce_all(source("build", lanes)),
        filter(source("probe", lanes)),
    ))
}

fn join_script() -> Script {
    Script::default()
        .source("build", vec![vec![spec(4, 32)]])
        .source("probe", vec![vec![spec(10, 80), spec(10, 80)]])
}

#[test]
fn probe_side_queues_stay_empty_until_the_build_is_set() {
    let plan = join_plan(1);
    let script = join_script();
    let mut driver = driver(plan.as_ref(), &script);
    // Node numbering is pre-order: 0 unload, 1 join, 2 coalesce, 3 build source,
    // 4 filter, 5 probe source.
    let (probe_filter, probe_source) = (4, 5);
    let mut set_build_seen = false;
    while driver.step().expect("no protocol violation") {
        if !set_build_seen {
            // Empty, not merely bounded: nothing would drain a probe batch yet.
            assert_eq!(
                (
                    driver.queue_len(probe_filter, 0),
                    driver.queue_len(probe_source, 0)
                ),
                (0, 0),
                "the probe subtree ran while the join was still building"
            );
        }
        set_build_seen |= driver.last_call() == Some(CallKind::SetBuild);
    }
    assert!(set_build_seen, "the join never reached its probe phase");
}

#[test]
fn the_hold_covers_the_whole_probe_subtree_and_lifts_together() {
    let plan = join_plan(1);
    let script = join_script();
    let report = run(plan.as_ref(), &script);
    let step_of = |kind: CallKind| {
        report
            .trace
            .iter()
            .find(|event| event.call == kind)
            .map(|event| event.step)
    };
    let set_build = step_of(CallKind::SetBuild).expect("the build side was set");
    let first_probe_pull = report
        .trace
        .iter()
        .find(|event| event.node == 5 && event.call == CallKind::NextBatch)
        .map(|event| event.step)
        .expect("the probe source ran");
    assert!(
        first_probe_pull > set_build,
        "the probe source ran at step {first_probe_pull}, before set_build at {set_build}"
    );
    assert_accounted(&report);
}

#[test]
fn a_join_inside_another_joins_build_subtree_is_not_held() {
    // The outer join's build side is itself a join, and a build subtree is never held —
    // which is what keeps the whole arrangement from deadlocking.
    let inner = join(coalesce_all(source("a", 1)), filter(source("b", 1)));
    let plan = unload(join(coalesce_all(inner), filter(source("c", 1))));
    let script = Script::default()
        .source("a", vec![vec![spec(4, 32)]])
        .source("b", vec![vec![spec(6, 48)]])
        .source("c", vec![vec![spec(8, 64)]]);
    let report = run(plan.as_ref(), &script);
    assert_eq!(
        count(&report, CallKind::SetBuild),
        2,
        "both joins reached their probe phase"
    );
    assert_accounted(&report);
}

#[test]
fn nested_joins_resolve_outermost_first_without_deadlocking() {
    // The outer join's probe side holds an inner join: the outer build drains, the outer
    // hold lifts, and only then does the inner one get to run at all.
    let inner = join(coalesce_all(source("a", 1)), filter(source("b", 1)));
    let plan = unload(join(coalesce_all(source("c", 1)), inner));
    let script = Script::default()
        .source("a", vec![vec![spec(4, 32)]])
        .source("b", vec![vec![spec(6, 48)]])
        .source("c", vec![vec![spec(8, 64)]]);
    let report = run(plan.as_ref(), &script);
    let builds: Vec<u32> = report
        .trace
        .iter()
        .filter(|event| event.call == CallKind::SetBuild)
        .map(|event| event.node)
        .collect();
    assert_eq!(
        builds,
        vec![1, 4],
        "the outer join is set before the inner one"
    );
    assert_accounted(&report);
}

#[test]
fn a_build_side_that_produced_no_batch_is_an_error() {
    let plan = join_plan(1);
    let script = join_script().with_accumulator(AccRule::EmitAtDone(0));
    protocol_error(
        run_with(plan.as_ref(), &script, None),
        "build side finished without producing a batch",
    );
}

#[test]
fn a_build_side_that_produced_two_batches_is_an_error() {
    let plan = join_plan(1);
    let script = join_script().with_accumulator(AccRule::EmitAtDone(2));
    protocol_error(
        run_with(plan.as_ref(), &script, None),
        "build side produced a second batch",
    );
}

// -- shuffles and routing ----------------------------------------------------------

#[test]
fn a_skewed_shuffle_drops_the_empty_lanes_at_the_emit() {
    let plan = unload(merge(emit(source("part", 1), 4)));
    let script = Script::default()
        .source("part", vec![vec![spec(12, 96), spec(12, 96)]])
        .with_emit(EmitRule::ToLane(2));
    let report = run(plan.as_ref(), &script);
    let emits = calls(&report, CallKind::Emit);
    assert!(
        emits.iter().all(|event| event.outputs == 1),
        "three of four lanes were empty and never queued"
    );
    assert_eq!(
        rows_returned(&report),
        24,
        "the hot lane carried everything"
    );
    assert_accounted(&report);
}

#[test]
fn a_merge_serves_its_one_lane_from_every_lane_of_its_child() {
    let plan = unload(merge(emit(source("part", 1), 3)));
    let script = Script::default().source("part", vec![vec![spec(9, 72)]]);
    let report = run(plan.as_ref(), &script);
    assert_eq!(
        count(&report, CallKind::Forward),
        3,
        "one batch per lane, forwarded"
    );
    assert_eq!(rows_returned(&report), 9);
}

#[test]
fn a_union_relabels_lanes_and_forwards_every_batch() {
    let plan = unload(merge(union(vec![
        filter(source("a", 2)),
        filter(source("b", 1)),
    ])));
    let script = Script::default()
        .source("a", vec![vec![spec(3, 24)], vec![spec(4, 32)]])
        .source("b", vec![vec![spec(5, 40)]]);
    let report = run(plan.as_ref(), &script);
    assert_eq!(
        rows_returned(&report),
        12,
        "every branch's rows reached the root"
    );
    assert_accounted(&report);
}

#[test]
fn a_union_with_one_branch_exhausted_keeps_serving_the_other() {
    let plan = unload(merge(union(vec![
        filter(source("a", 1)),
        filter(source("b", 1)),
    ])));
    let script = Script::default()
        .source("a", vec![vec![spec(3, 24)]])
        .source("b", vec![vec![spec(4, 32), spec(4, 32), spec(4, 32)]]);
    let report = run(plan.as_ref(), &script);
    assert_eq!(rows_returned(&report), 15);
    assert_accounted(&report);
}

#[test]
fn an_interleave_rotates_its_children_within_each_lane() {
    let plan = unload(merge(interleave(vec![
        filter(source("a", 2)),
        filter(source("b", 2)),
    ])));
    let script = Script::default()
        .source("a", vec![vec![spec(1, 8)], vec![spec(2, 16)]])
        .source("b", vec![vec![spec(4, 32)], vec![spec(8, 64)]]);
    let report = run(plan.as_ref(), &script);
    assert_eq!(
        rows_returned(&report),
        15,
        "lane p takes lane p of every child"
    );
    assert_accounted(&report);
}

#[test]
fn a_cross_lane_accumulator_emits_once_every_lane_has_ended() {
    let plan = unload(merge_sorted(sort(source("part", 3))));
    let script = Script::default().source(
        "part",
        vec![vec![spec(2, 16)], vec![spec(3, 24)], vec![spec(4, 32)]],
    );
    let report = run(plan.as_ref(), &script);
    let emitting: Vec<&TraceEvent> = calls(&report, CallKind::LaneDone)
        .into_iter()
        .filter(|event| event.outputs > 0)
        .collect();
    assert_eq!(emitting.len(), 1, "only the last lane's end emits");
    assert_eq!(rows_returned(&report), 9);
    assert_accounted(&report);
}

// -- determinism -------------------------------------------------------------------

#[test]
fn two_runs_of_one_plan_produce_identical_traces() {
    let plan = unload(merge(emit(filter(source("part", 1)), 4)));
    let script =
        Script::default().source("part", vec![vec![spec(8, 64), spec(8, 64), spec(7, 56)]]);
    let first = run(plan.as_ref(), &script);
    let second = run(plan.as_ref(), &script);
    assert_eq!(first.trace, second.trace);
    assert_eq!(first.peak_bytes, second.peak_bytes);
}

// -- wiring ------------------------------------------------------------------------

#[test]
fn a_backend_that_builds_the_wrong_executor_is_caught_where_it_was_built() {
    let script = Script::default()
        .source("part", vec![vec![spec(10, 80)]])
        .miswired();
    match run_with(one_lane_chain().as_ref(), &script, None) {
        Err(RunError::Backend(error)) => assert!(
            error.to_string().contains("executor for a"),
            "the error does not name the disagreement: {error}"
        ),
        other => panic!("expected a backend error, got {other:?}"),
    }
}

#[test]
fn an_emit_returning_the_wrong_lane_count_is_an_error() {
    let plan = unload(merge(emit(source("part", 1), 4)));
    let script = Script::default()
        .source("part", vec![vec![spec(8, 64)]])
        .with_emit(EmitRule::WrongCount);
    protocol_error(run_with(plan.as_ref(), &script, None), "emit returned");
}

#[test]
fn a_join_that_needs_a_finish_pass_emits_at_the_end() {
    let plan = join_plan(1);
    let script = join_script().with_join(JoinRule {
        finish_rows: 3,
        build_residency: 64,
    });
    let report = run(plan.as_ref(), &script);
    let finish = calls(&report, CallKind::Finish);
    assert_eq!(finish.len(), 1);
    assert_eq!(
        finish[0].outputs, 1,
        "the unmatched build rows leave at finish"
    );
    assert_accounted(&report);
}

#[test]
fn a_run_that_completed_normally_left_nothing_anywhere() {
    // The property `assert_drained` enforces, stated: every lane of the root ended, and no
    // queue in the tree still holds a batch. A run that ends any other way without a limit
    // having been satisfied has lost work rather than finished.
    let plan = unload(merge(emit(filter(source("part", 1)), 4)));
    let script = Script::default().source("part", vec![vec![spec(9, 72); 3]]);
    let report = run(plan.as_ref(), &script);
    assert!(!report.early_exit);
    assert!(
        report.peak_queued.iter().any(|queued| *queued > 0),
        "a plan that never queued anything proves nothing about draining"
    );
    assert_eq!(report.holds, report.releases);
    assert_eq!(report.in_flight_bytes, 0);
    assert_eq!(rows_returned(&report), 27);
}

#[test]
fn a_run_stopped_part_way_is_reported_as_not_drained() {
    // Driven by hand and abandoned, which is the only way to reach the condition: the
    // driver's own schedule does not leave work behind. Without the check the loss is
    // silent, since the batches are simply dropped with the driver.
    let plan = unload(filter(source("part", 1)));
    let script = Script::default().source("part", vec![vec![spec(10, 80); 4]]);
    let mut driver = driver(plan.as_ref(), &script);
    driver.step().expect("the source runs");
    match driver.finish() {
        Err(StepError::Run(RunError::Protocol(said))) => assert!(
            said.contains("lanes that never ended"),
            "the message does not name the failure: {said}"
        ),
        other => panic!("a half-run plan reported as drained: {other:?}"),
    }
}

#[test]
fn an_early_exit_leaks_nothing_even_though_it_leaves_work_behind() {
    // The early-exit path deliberately ends with lanes not done and queues non-empty, so
    // drainage is the wrong invariant; what has to hold is that every batch held was given
    // back. Counted rather than netted: a total back at zero is also what releasing
    // something never held would leave.
    let plan = unload_limited(merge(emit(filter(source("part", 1)), 4)), 0, Some(5));
    let script = Script::default().source("part", vec![vec![spec(9, 72); 4]]);
    let report = run(plan.as_ref(), &script);
    assert!(report.early_exit, "the limit is what ends this run");
    assert_eq!(
        report.holds, report.releases,
        "{} batches were held and {} released",
        report.holds, report.releases
    );
    assert_eq!(report.in_flight_bytes, 0);
}
