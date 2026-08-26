//! Flow, backpressure, limits and accounting, over the mock backend.
//!
//! Every case asserts on calls — pull counts, queue bounds, batch release, the trace —
//! rather than on the rows that came back: a limit test that checks the rows passes just
//! as well when the whole input was read and thrown away.

mod budget;
mod failure;
mod flow;
mod limit;
mod memory;
mod stress;
mod wiring;

use super::mock::{Mock, Script};
use super::partitioned::{Driver, RunReport};
use super::{CallKind, StepError, TraceEvent};
use crate::batch_partitioned::error::RunError;
use crate::batch_partitioned::node::GpuNode;

/// Run to completion under no budget, which is what most flow cases want.
fn run(root: &dyn GpuNode, script: &Script) -> RunReport {
    run_with(root, script, None).expect("the plan runs")
}

fn run_with(
    root: &dyn GpuNode,
    script: &Script,
    budget: Option<usize>,
) -> Result<RunReport, RunError> {
    // Construction can refuse — the planner's canonical form is checked there — so this
    // propagates rather than unwrapping, which is what the refusal case asserts on.
    Driver::<Mock>::new(root, script, budget)?.run(10_000)
}

fn driver<'a>(root: &'a dyn GpuNode, script: &'a Script) -> Driver<'a, Mock> {
    driver_with(root, script, None)
}

fn driver_with<'a>(
    root: &'a dyn GpuNode,
    script: &'a Script,
    budget: Option<usize>,
) -> Driver<'a, Mock> {
    let mut driver = Driver::<Mock>::new(root, script, budget).expect("the plan indexes");
    driver.seed();
    driver
}

/// Drive to the failure, and report what the driver had recorded when it stopped — which
/// is how a refused call is told from one that ran and failed.
fn last_call_before_failing(driver: &mut Driver<'_, Mock>) -> Option<CallKind> {
    loop {
        match driver.step() {
            Ok(true) => continue,
            Ok(false) => panic!("the run ended without failing"),
            Err(_) => return driver.last_call(),
        }
    }
}

fn calls(report: &RunReport, kind: CallKind) -> Vec<&TraceEvent> {
    report
        .trace
        .iter()
        .filter(|event| event.call == kind)
        .collect()
}

fn count(report: &RunReport, kind: CallKind) -> usize {
    calls(report, kind).len()
}

fn rows_returned(report: &RunReport) -> usize {
    report
        .batches
        .iter()
        .map(|batch| batch.record_batch().num_rows())
        .sum()
}

/// Every run that finished has to have released what it held, and a peak of zero means the
/// accountant watched nothing.
fn assert_accounted(report: &RunReport) {
    assert_eq!(
        report.in_flight_bytes, 0,
        "a batch was held and never released"
    );
    // The bytes balancing is not the counts balancing: a release of nothing subtracts
    // nothing and still counts, which is how a cross-lane accumulator's lane-done events
    // made these disagree with the invariant the report states.
    assert_eq!(
        report.holds, report.releases,
        "{} batches held and {} released",
        report.holds, report.releases
    );
    assert!(
        report.peak_bytes > 0,
        "a run that peaked at zero observed nothing"
    );
}

fn protocol_error(result: Result<RunReport, RunError>, mentions: &str) {
    match result {
        Err(RunError::Protocol(said)) => assert!(
            said.contains(mentions),
            "the protocol error says the wrong thing: {said}"
        ),
        other => panic!("expected a protocol error naming {mentions:?}, got {other:?}"),
    }
}
