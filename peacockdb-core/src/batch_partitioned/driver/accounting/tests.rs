//! The formula, the cache, and the two checks. Every case here is a unit case: the
//! accountant takes figures, so nothing below builds a plan or a backend.

use super::*;
use crate::batch_partitioned::executor::AbiCalls;

/// State and model as two numbers the test sets, which is all the accountant reads.
struct Held {
    resident: usize,
    scratch: usize,
}

impl Executor for Held {
    fn resident_bytes(&self) -> usize {
        self.resident
    }
    fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
        self.scratch + n_bytes
    }
}

fn slot(index: usize) -> Slot {
    Slot {
        index,
        node: index as u32,
        lane: 0,
    }
}

fn measured(bytes: usize) -> CallStats {
    CallStats {
        scratch_bytes: Some(bytes),
        calls: AbiCalls::default(),
    }
}

#[test]
fn resident_is_in_flight_plus_executor_state() {
    let mut acct = ResidentAccountant::new(2, None);
    acct.hold(100);
    let held = Held {
        resident: 40,
        scratch: 0,
    };
    let modelled = acct.begin_call(slot(0), &held, 0, 0).unwrap();
    acct.end_call(slot(0), &held, &CallStats::default(), modelled)
        .unwrap();
    assert_eq!((acct.in_flight(), acct.resident()), (100, 140));
}

#[test]
fn the_cached_executor_total_tracks_a_live_sum() {
    let mut acct = ResidentAccountant::new(2, None);
    for (index, resident) in [(0, 10), (1, 30)] {
        let held = Held {
            resident,
            scratch: 0,
        };
        acct.end_call(slot(index), &held, &CallStats::default(), 0)
            .unwrap();
    }
    assert_eq!(acct.resident(), 40);
    // One instance grows: the total moves by its delta, not by a re-sum.
    let grown = Held {
        resident: 55,
        scratch: 0,
    };
    acct.end_call(slot(0), &grown, &CallStats::default(), 0)
        .unwrap();
    assert_eq!(acct.resident(), 85);
}

#[test]
fn forgetting_an_executor_removes_its_contribution() {
    let mut acct = ResidentAccountant::new(1, None);
    let held = Held {
        resident: 64,
        scratch: 0,
    };
    acct.end_call(slot(0), &held, &CallStats::default(), 0)
        .unwrap();
    assert_eq!(acct.resident(), 64);
    acct.forget(slot(0));
    assert_eq!(acct.resident(), 0);
    acct.forget(slot(0));
    assert_eq!(acct.resident(), 0, "forgetting twice is not a second debit");
}

#[test]
fn a_consuming_call_stops_the_slot_contributing() {
    let mut acct = ResidentAccountant::new(1, None);
    let held = Held {
        resident: 64,
        scratch: 0,
    };
    acct.end_call(slot(0), &held, &CallStats::default(), 0)
        .unwrap();
    // mark_done_and_fetch and friends consume the executor, so there is nothing left to
    // ask — the slot goes to zero rather than keeping its last figure.
    acct.end_consuming_call(slot(0), &CallStats::default(), 0)
        .unwrap();
    assert_eq!(acct.resident(), 0);
}

#[test]
fn releasing_a_batch_that_was_never_held_is_an_error() {
    let mut acct = ResidentAccountant::new(1, None);
    acct.hold(10);
    match acct.release(11) {
        Err(RunError::Protocol(said)) => assert!(said.contains("without having been held")),
        other => panic!("expected a protocol error, got {other:?}"),
    }
}

#[test]
fn the_pre_check_trips_before_the_call_runs() {
    let mut acct = ResidentAccountant::new(1, Some(100));
    acct.hold(90);
    let hungry = Held {
        resident: 0,
        scratch: 20,
    };
    let trip = acct
        .begin_call(slot(0), &hungry, 0, 0)
        .expect_err("90 held plus 20 modelled is over 100");
    assert_eq!((trip.when, trip.bytes), (When::PreCall, 110));
    assert_eq!(acct.calls(), 0, "a call the budget refused never ran");
}

#[test]
fn the_post_check_trips_on_residency_the_model_did_not_predict() {
    let mut acct = ResidentAccountant::new(1, Some(100));
    let liar = Held {
        resident: 0,
        scratch: 0,
    };
    let modelled = acct.begin_call(slot(0), &liar, 0, 0).expect("nothing yet");
    // The call ran and kept far more than it said it would need.
    let kept = Held {
        resident: 150,
        scratch: 0,
    };
    let trip = acct
        .end_call(slot(0), &kept, &CallStats::default(), modelled)
        .expect_err("150 resident is over 100");
    assert_eq!((trip.when, trip.bytes), (When::PostCall, 150));
}

#[test]
fn no_budget_means_it_accounts_without_tripping() {
    let mut acct = ResidentAccountant::new(1, None);
    acct.hold(1 << 40);
    let huge = Held {
        resident: 1 << 40,
        scratch: 1 << 40,
    };
    let modelled = acct.begin_call(slot(0), &huge, 0, 0).expect("no budget");
    acct.end_call(slot(0), &huge, &CallStats::default(), modelled)
        .expect("no budget");
    assert_eq!(acct.resident(), 2 << 40);
}

#[test]
fn the_peak_is_the_high_water_mark_not_the_final_value() {
    let mut acct = ResidentAccountant::new(1, None);
    acct.hold(500);
    acct.release(500).unwrap();
    assert_eq!((acct.in_flight(), acct.peak()), (0, 500));
}

#[test]
fn an_under_predicting_model_is_recorded_with_its_magnitude() {
    let mut acct = ResidentAccountant::new(1, None);
    let executor = Held {
        resident: 0,
        scratch: 10,
    };
    let modelled = acct.begin_call(slot(0), &executor, 0, 0).unwrap();
    acct.end_call(slot(0), &executor, &measured(25), modelled)
        .unwrap();
    let recorded = acct.underestimates();
    assert_eq!(recorded.len(), 1);
    assert_eq!((recorded[0].modelled, recorded[0].measured), (10, 25));
    assert_eq!(recorded[0].ratio(), 2.5);
}

#[test]
fn model_accuracy_is_recorded_rather_than_enforced() {
    let mut acct = ResidentAccountant::new(1, Some(1000));
    let executor = Held {
        resident: 0,
        scratch: 10,
    };
    let modelled = acct.begin_call(slot(0), &executor, 0, 0).unwrap();
    // Under by 100x and the query still runs: the enforcer's contract is the accounted
    // peak against the budget, not that the model was right.
    acct.end_call(slot(0), &executor, &measured(1000), modelled)
        .expect("an under-estimate is a diagnostic, not a failure");
    assert_eq!(acct.underestimates().len(), 1);
}

#[test]
fn a_model_that_held_records_nothing() {
    let mut acct = ResidentAccountant::new(1, None);
    let executor = Held {
        resident: 0,
        scratch: 100,
    };
    let modelled = acct.begin_call(slot(0), &executor, 0, 0).unwrap();
    acct.end_call(slot(0), &executor, &measured(80), modelled)
        .unwrap();
    assert!(acct.underestimates().is_empty());
}

#[test]
fn an_absent_measurement_is_not_recorded_as_an_underestimate() {
    let mut acct = ResidentAccountant::new(1, None);
    let executor = Held {
        resident: 0,
        scratch: 0,
    };
    let modelled = acct.begin_call(slot(0), &executor, 0, 0).unwrap();
    // None is "this run was not instrumented", which is not a model of zero.
    acct.end_call(slot(0), &executor, &CallStats::default(), modelled)
        .unwrap();
    assert!(acct.underestimates().is_empty());
}

#[test]
fn the_model_may_consult_the_batch_it_is_about_to_be_handed() {
    let mut acct = ResidentAccountant::new(1, Some(100));
    let scales = Held {
        resident: 0,
        scratch: 10,
    };
    assert_eq!(acct.begin_call(slot(0), &scales, 7, 40).unwrap(), 50);
    assert!(
        acct.begin_call(slot(0), &scales, 7, 95).is_err(),
        "the same executor over a bigger batch is what crosses the budget"
    );
}
