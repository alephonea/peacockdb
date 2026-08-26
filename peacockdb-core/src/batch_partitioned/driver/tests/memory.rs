//! The enforcer through the drivers: what trips, what is merely recorded, and that a
//! finished run gave everything back.

use super::super::mock::{JoinRule, Script, spec};
use super::super::plans::*;
use super::*;
use crate::batch_partitioned::error::When;

fn chain(lanes: usize) -> Box<dyn GpuNode> {
    unload(filter(source("part", lanes)))
}

fn batches(lanes: usize, per_lane: usize) -> Script {
    Script::default().source("part", vec![vec![spec(10, 80); per_lane]; lanes])
}

#[test]
fn a_generous_budget_completes_and_records_a_peak() {
    let report = run_with(chain(1).as_ref(), &batches(1, 4), Some(10_000)).expect("it fits");
    assert!(
        report.peak_bytes > 0 && report.peak_bytes <= 10_000,
        "peak {} is outside 0 < peak <= budget",
        report.peak_bytes
    );
    assert_eq!(report.in_flight_bytes, 0);
}

#[test]
fn a_tight_budget_fails_the_query_cleanly() {
    // The model is a byte of scratch per input byte, so the first real call asks for
    // twice what one batch costs.
    let script = batches(1, 4).modelling(1);
    let plan = chain(1);
    match run_with(plan.as_ref(), &script, Some(100)) {
        Err(RunError::BudgetExceeded { message, .. }) => {
            assert!(
                message.contains("GpuFilter"),
                "the message names no node: {message}"
            );
            assert!(message.contains("budget 100"), "and no budget: {message}");
        }
        other => panic!("expected a clean budget failure, got {other:?}"),
    }
}

#[test]
fn a_consumed_input_stays_accounted_through_its_call() {
    // One 800-byte batch under a budget of 1000, modelling a byte of scratch per input
    // byte. Counting the input while the call runs asks for 1600 and cannot fit; pricing
    // the call as if the input were already gone asks for 800 and would.
    let one_batch = Script::default().source("part", vec![vec![spec(10, 800)]]);
    assert!(
        run_with(
            chain(1).as_ref(),
            &one_batch.clone().modelling(1),
            Some(1000)
        )
        .is_err(),
        "the input was released before the call it feeds was priced"
    );
    run_with(chain(1).as_ref(), &one_batch, Some(1000)).expect("the same run without a model fits");
}

#[test]
fn the_pre_check_trips_before_the_call_runs() {
    // A run-level trip carries a phase, a node and a lane, so all three are asserted here
    // as well as the thing that distinguishes the two checks: the call never ran. A
    // refused call and a failed one both return nothing, so the rows cannot tell them
    // apart and the trace is what does.
    let script = batches(1, 4).modelling(1);
    let plan = chain(1);
    match run_with(plan.as_ref(), &script, Some(100)) {
        Err(RunError::BudgetExceeded { when, message }) => {
            assert_eq!(when, When::PreCall);
            assert!(message.contains("GpuFilter"), "{message}");
            assert!(message.contains("lane 0"), "{message}");
        }
        other => panic!("expected a pre-call trip, got {other:?}"),
    }
    let mut driver = driver_with(plan.as_ref(), &script, Some(100));
    assert_eq!(
        last_call_before_failing(&mut driver),
        Some(CallKind::NextBatch),
        "the refused call recorded itself, so it ran"
    );
}

#[test]
fn an_accumulators_residency_is_visible_while_it_holds_rows() {
    let plan = unload(coalesce_all(source("part", 1)));
    let script = batches(1, 4);
    let report = run(plan.as_ref(), &script);
    assert!(
        report.peak_bytes >= 4 * 80,
        "the accumulator held four batches and reported {}",
        report.peak_bytes
    );
    assert_eq!(report.in_flight_bytes, 0);
}

#[test]
fn the_post_check_trips_on_residency_the_model_did_not_predict() {
    // Nothing is modelled, so every pre-check passes; the accumulator keeps what it is
    // handed, and the check after the call is what notices.
    let plan = unload(coalesce_all(source("part", 1)));
    let script = batches(1, 6);
    match run_with(plan.as_ref(), &script, Some(200)) {
        Err(RunError::BudgetExceeded { when, .. }) => {
            assert_eq!(when, When::PostCall, "the trip should be the post-call one")
        }
        other => panic!("expected a post-call trip, got {other:?}"),
    }
}

#[test]
fn a_memory_bound_is_asserted_at_more_than_one_partitioning() {
    // The same total rows over one lane and over four. A bound that only ever ran at one
    // layout is a bound about one shape of arrival: only a streamed lane accumulates.
    let one = run(chain(1).as_ref(), &batches(1, 4));
    let four = run(chain(4).as_ref(), &batches(4, 1));
    for report in [&one, &four] {
        assert_eq!(report.in_flight_bytes, 0);
        assert!(report.peak_bytes > 0);
    }
    assert!(
        four.peak_bytes >= one.peak_bytes,
        "four lanes are live at once, so the peak cannot be lower: {} against {}",
        four.peak_bytes,
        one.peak_bytes
    );
}

#[test]
fn an_under_predicting_model_is_recorded_rather_than_failing() {
    let script = batches(1, 2).measuring(4096);
    let report =
        run_with(chain(1).as_ref(), &script, Some(10_000)).expect("a diagnostic, not a failure");
    assert!(
        !report.underestimates.is_empty(),
        "a call that used 4 KiB against a model of nothing is an under-estimate"
    );
    assert_eq!(report.underestimates[0].measured, 4096);
}

#[test]
fn an_uninstrumented_run_records_no_underestimates() {
    let report = run(chain(1).as_ref(), &batches(1, 2));
    assert!(
        report.underestimates.is_empty(),
        "None means the run was not instrumented, not a model of zero"
    );
}

#[test]
fn a_join_that_holds_its_build_side_reports_it_while_it_probes() {
    let plan = unload(join(
        coalesce_all(source("build", 1)),
        filter(source("probe", 1)),
    ));
    let script = Script::default()
        .source("build", vec![vec![spec(4, 512)]])
        .source("probe", vec![vec![spec(10, 80), spec(10, 80)]])
        .with_join(JoinRule {
            finish_rows: 0,
            build_residency: 4096,
        });
    let report = run(plan.as_ref(), &script);
    assert!(
        report.peak_bytes >= 4096,
        "the build side is resident across every probe call, and was not counted"
    );
    assert_eq!(report.in_flight_bytes, 0);
}
