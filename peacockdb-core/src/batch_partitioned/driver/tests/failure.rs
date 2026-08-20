//! A call that fails ends the query. There is one response to every backend failure —
//! stop — so what these assert is that it stops, that the message says where, and that the
//! accounting still reconciles on the way out.

use super::super::mock::{FailAt, JoinRule, Script, spec};
use super::super::plans::*;
use super::*;

fn source_batches(lanes: usize) -> Script {
    Script::default().source("part", vec![vec![spec(10, 80); 3]; lanes])
}

/// Fails the query, and says which node and lane it was calling.
fn fails_at(plan: &dyn GpuNode, script: &Script, node: &str, lane: usize) {
    match run_with(plan, script, None) {
        Err(RunError::CallFailed(said)) => {
            assert!(said.contains(node), "the message names no node: {said}");
            assert!(
                said.contains(&format!("lane {lane}")),
                "the message names no lane: {said}"
            );
            assert!(
                said.contains("gave up"),
                "the backend's own words are lost: {said}"
            );
        }
        other => panic!("expected a failed call, got {other:?}"),
    }
}

#[test]
fn a_source_that_fails_ends_the_query() {
    let plan = unload(filter(source("part", 1)));
    fails_at(
        plan.as_ref(),
        &source_batches(1).failing_at(FailAt::SourceStep),
        "GpuLoadParquet",
        0,
    );
}

#[test]
fn a_mid_chain_exec_that_fails_ends_the_query() {
    let plan = unload(project(filter(source("part", 1))));
    fails_at(
        plan.as_ref(),
        &source_batches(1).failing_at(FailAt::Exec),
        "GpuFilter",
        0,
    );
}

#[test]
fn an_emit_that_fails_ends_the_query() {
    let plan = unload(merge(emit(source("part", 1), 4)));
    fails_at(
        plan.as_ref(),
        &source_batches(1).failing_at(FailAt::Emit),
        "GpuEmitPartitions",
        0,
    );
}

fn join_plan() -> Box<dyn GpuNode> {
    unload(join(
        coalesce_all(source("build", 1)),
        filter(source("probe", 1)),
    ))
}

fn join_script() -> Script {
    Script::default()
        .source("build", vec![vec![spec(4, 32)]])
        .source("probe", vec![vec![spec(10, 80), spec(10, 80)]])
}

#[test]
fn a_join_probe_that_fails_ends_the_query() {
    fails_at(
        join_plan().as_ref(),
        &join_script().failing_at(FailAt::Probe),
        "GpuJoin",
        0,
    );
}

#[test]
fn a_call_that_consumes_its_executor_can_fail_too() {
    // `finish_and_fetch` takes the executor by value, so on failure there is nothing left
    // to hold a slot open — which is the same thing the query being over already means.
    let script = join_script()
        .failing_at(FailAt::Finish)
        .with_join(JoinRule {
            finish_rows: 3,
            build_residency: 64,
        });
    fails_at(join_plan().as_ref(), &script, "GpuJoin", 0);
}

#[test]
fn an_accumulators_final_call_can_fail_too() {
    let plan = unload(coalesce_all(source("part", 1)));
    fails_at(
        plan.as_ref(),
        &source_batches(1).failing_at(FailAt::MarkDone),
        "GpuCoalesceAllBatches",
        0,
    );
}

#[test]
fn the_message_names_the_lane_that_failed_and_not_the_first_one() {
    // Lane 0 has nothing, so its exec goes straight to EndOfInput and lane 1 makes the
    // first call. Which variable reaches the message is a live choice in the driver — the
    // emitter passes a literal 0 where the accumulator passes its lane — and a message
    // hard-coded to lane 0 would pass every other case here.
    let plan = unload(filter(source("part", 2)));
    let script = Script::default()
        .source("part", vec![vec![], vec![spec(10, 80)]])
        .failing_at(FailAt::Exec);
    fails_at(plan.as_ref(), &script, "GpuFilter", 1);
}

#[test]
fn a_failure_stops_the_schedule_where_it_stood() {
    // Three batches on each of two lanes, failing at the first exec: what must not happen
    // is the rest of the plan being driven anyway.
    let plan = unload(filter(source("part", 2)));
    let script = source_batches(2).failing_at(FailAt::Exec);
    let mut driver = driver(plan.as_ref(), &script);
    let mut steps = 0;
    let error = loop {
        match driver.step() {
            Ok(true) => steps += 1,
            Ok(false) => panic!("the run ended without failing"),
            Err(error) => break error,
        }
    };
    assert!(matches!(error, StepError::Run(RunError::CallFailed(_))));
    assert_eq!(steps, 1, "only the source ran before the filter failed");
    assert_eq!(
        driver.last_call(),
        Some(CallKind::NextBatch),
        "a call that failed recorded itself, or another node ran after it"
    );
}

#[test]
fn a_failed_query_gives_back_everything_it_held() {
    // The handles are not touched again, but the totals still reconcile: the release path
    // is the same one the early exit uses, so held equals released on every way out.
    // Both paths, because they release in different places: a lane call's input goes back
    // where the call failed, a cross-lane one before the result is even looked at.
    for at in [FailAt::Exec, FailAt::Emit] {
        let plan = unload(merge(emit(filter(source("part", 1)), 4)));
        let script = source_batches(1).failing_at(at);
        let mut driver = driver(plan.as_ref(), &script);
        while let Ok(true) = driver.step() {}
        driver.release_all().expect("what it held goes back");
        let (holds, releases) = driver.hops();
        assert!(holds > 0, "a query that held nothing proves nothing");
        assert_eq!(holds, releases, "{at:?}: {holds} held, {releases} released");
    }
}
