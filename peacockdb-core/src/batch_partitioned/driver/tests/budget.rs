//! What the accountant decides at the boundary, through a real run.
//!
//! Every budget here is derived from the peak an unbudgeted run of the same plan reported,
//! never written down: a byte count in a test is a magic number that rots the first time a
//! mock's sizes move, and it can make a test pass for a reason nobody meant.

use super::super::mock::{Script, spec};
use super::super::plans::*;
use super::*;
use crate::batch_partitioned::error::When;

/// The test's own input, so the sizes below are data rather than expectations.
const BATCH_ROWS: usize = 10;
const BATCH_BYTES: usize = 800;

fn chain() -> Box<dyn GpuNode> {
    unload(filter(source("part", 1)))
}

fn batches(n: usize) -> Script {
    Script::default().source("part", vec![vec![spec(BATCH_ROWS, BATCH_BYTES); n]])
}

/// What the same plan peaks at when the accountant only watches.
fn unbudgeted_peak(plan: &dyn GpuNode, script: &Script) -> usize {
    let report = run(plan, script);
    assert!(report.peak_bytes > 0, "an unbudgeted run observed nothing");
    report.peak_bytes
}

/// The phase as data and the sentence as words: a caller branches on the first, a person
/// reads the second, and both are pinned.
fn budget_error(plan: &dyn GpuNode, script: &Script, budget: usize) -> (When, String) {
    match run_with(plan, script, Some(budget)) {
        Err(RunError::BudgetExceeded { when, message }) => (when, message),
        other => panic!("expected a budget failure at {budget}, got {other:?}"),
    }
}

#[test]
fn a_budget_equal_to_the_peak_completes() {
    // `check` trips on `value > budget`, so equality is the boundary rather than a near
    // miss — and it is the off-by-one a user would find rather than a test.
    let (plan, script) = (chain(), batches(4));
    let peak = unbudgeted_peak(plan.as_ref(), &script);
    let report = run_with(plan.as_ref(), &script, Some(peak)).expect("the peak itself fits");
    assert_eq!(
        report.peak_bytes, peak,
        "the budget changed what the run did"
    );
    assert_eq!(report.in_flight_bytes, 0);
}

#[test]
fn a_budget_one_byte_below_the_peak_trips() {
    let (plan, script) = (chain(), batches(4));
    let peak = unbudgeted_peak(plan.as_ref(), &script);
    let (_, said) = budget_error(plan.as_ref(), &script, peak - 1);
    // The peak of this plan is one scan batch in flight, so the scan is where it is
    // reached: the filter is 1:1 and releases its input as it produces.
    assert!(said.contains("GpuLoadParquet"), "{said}");
    assert!(
        said.contains(&format!("budget {} bytes", peak - 1)),
        "{said}"
    );
}

/// The plan the finding is about: everything the source produced, held by one node until it
/// emits a copy of it.
fn accumulating() -> Box<dyn GpuNode> {
    unload(coalesce_all(source("part", 1)))
}

#[test]
fn a_model_that_prices_its_emission_refuses_the_call_that_would_not_fit() {
    // The accumulator holds 3200 and is about to build another 3200 out of it. A budget
    // under the two together is refused BEFORE the call, which is what the pre-check is
    // for — the alternative is both live at once and nothing having said no.
    let (plan, script) = (accumulating(), batches(4).pricing_the_emission());
    let peak = unbudgeted_peak(plan.as_ref(), &batches(4));
    let budget = peak - 1;

    let (when, said) = budget_error(plan.as_ref(), &script, budget);
    assert_eq!(when, When::PreCall);
    // The one place the sentence itself is pinned, so the human-facing half cannot drift
    // away from the field beside it.
    assert_eq!(
        said,
        format!(
            "resident GPU memory budget exceeded at GpuCoalesceAllBatches lane 0, before \
             the call: {peak} bytes > budget {budget} bytes"
        )
    );
    assert!(
        said.contains("GpuCoalesceAllBatches"),
        "the trip names where the run was, not what could not fit: {said}"
    );

    // The refused call is the accumulator's `mark_done_and_fetch`, and a call records
    // itself only after it returns — so the last thing in the trace is what ran before it,
    // the source reporting itself exhausted, and never the MarkDone.
    let mut driver = driver_with(plan.as_ref(), &script, Some(budget));
    assert_eq!(
        last_call_before_failing(&mut driver),
        Some(CallKind::SourceExhausted),
        "the trace ends in the refused MarkDone, so the call ran"
    );
}

#[test]
fn a_silent_model_lets_the_peak_pass_the_budget_it_was_never_checked_against() {
    // The design rather than a state of affairs: a model that prices nothing gets no
    // protection, and always will not. The post-check runs after the emitting call's state
    // has been forgotten, so it reads 3200 where the instant before held 6400 — the peak is
    // honest and the check never sees it. The model above is what closes it.
    let (plan, script) = (accumulating(), batches(4));
    let peak = unbudgeted_peak(plan.as_ref(), &script);
    let report = run_with(plan.as_ref(), &script, Some(peak - 1)).expect("nothing refuses it");
    assert_eq!(
        report.peak_bytes, peak,
        "the run reported a peak above the budget it completed under"
    );
}

#[test]
fn an_accumulators_held_rows_are_what_crosses_the_budget() {
    // Above any single batch and below what the accumulator builds out of them, so the
    // trip cannot be a batch in flight: it is the state, and the node named says so. It
    // takes the model to reach it — see the two tests above for why.
    let plan = accumulating();
    let script = batches(4).pricing_the_emission();
    let peak = unbudgeted_peak(plan.as_ref(), &batches(4));
    let budget = peak / 2;
    assert!(
        budget > BATCH_BYTES,
        "the budget has to clear a single batch, or this proves nothing"
    );
    let (_, said) = budget_error(plan.as_ref(), &script, budget);
    assert!(said.contains("GpuCoalesceAllBatches"), "{said}");
}

#[test]
fn a_limit_above_an_accumulator_saves_nothing() {
    // A limit only saves what it stops being read, and by the time an accumulator emits it
    // has already built its whole state — so an interval at the sink above one arrives too
    // late to have prevented anything. Where a limit sits is what decides whether it is
    // worth having, and this is the case that says so.
    let full = accumulating();
    let limited = unload_limited(coalesce_all(source("part", 1)), 0, Some(BATCH_ROWS as u64));
    let script = batches(6);

    assert_eq!(
        unbudgeted_peak(limited.as_ref(), &script),
        unbudgeted_peak(full.as_ref(), &script),
        "an interval above an accumulator changed the peak, which would be new"
    );

    // Below the scan the same interval does save, and what it saves is reading: the scan
    // stops being scheduled. Measured in pulls rather than in bytes, because a 1:1 chain
    // peaks at one batch however much of it is read — the peak is the wrong instrument
    // here and the read count is the right one.
    let early = unload_limited(filter(source("part", 1)), 0, Some(BATCH_ROWS as u64));
    let plain = unload(filter(source("part", 1)));
    let limited_reads = count(&run(early.as_ref(), &script), CallKind::NextBatch);
    let control_reads = count(&run(plain.as_ref(), &script), CallKind::NextBatch);
    assert_eq!(
        limited_reads, 1,
        "the limit read more than the one batch it needed"
    );
    assert_eq!(
        control_reads, 6,
        "the control: the same plan with no interval reads every batch there is"
    );
}
