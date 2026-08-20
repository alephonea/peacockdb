//! The lane state machine on its own: one (node, lane), one call at a time, with no tree
//! and no schedule around it. Reaching these through a whole plan works but says nothing
//! about which of the two is wrong when one breaks.

use std::sync::atomic::Ordering;

use super::*;
use crate::batch_partitioned::batch::Batch;
use crate::batch_partitioned::driver::accounting::ResidentAccountant;
use crate::batch_partitioned::driver::mock::{JoinRule, Mock, Script, spec};
use crate::batch_partitioned::driver::plans;

fn accountant() -> ResidentAccountant {
    ResidentAccountant::new(1, None)
}

fn slot() -> Slot {
    Slot {
        index: 0,
        node: 0,
        lane: 0,
    }
}

/// A site over a real node of the right kind, since `executors_for` reads the node.
fn site<'a>(script: &'a Script, node: &'a dyn GpuNode) -> LaneSite<'a, Mock> {
    LaneSite {
        ctx: script,
        category: crate::batch_partitioned::nodes::category_of(node),
        node,
        lane: 0,
        slot: slot(),
    }
}

fn batch(
    rows: usize,
    bytes: usize,
    acct: &mut ResidentAccountant,
) -> Held<<Mock as Backend>::Batch> {
    let held = Held::of(crate::batch_partitioned::driver::mock::MockBatch { rows, bytes });
    acct.hold(held.bytes);
    held
}

fn has_input() -> Avail {
    Avail {
        has: [true, true],
        done: [false, false],
    }
}

fn input_ended() -> Avail {
    Avail {
        has: [false, false],
        done: [true, true],
    }
}

fn rows_of(outputs: &LaneOutputs<Mock>) -> usize {
    match outputs {
        LaneOutputs::Device(batches) => batches.iter().map(|held| held.batch.rows).sum(),
        LaneOutputs::Host(batches) => batches.iter().map(|held| held.batch.num_rows()).sum(),
    }
}

#[test]
fn a_source_runs_until_it_reports_exhausted() {
    let script = Script::default().source("part", vec![vec![spec(4, 32), spec(6, 48)]]);
    let node = plans::source("part", 1);
    let site = site(&script, node.as_ref());
    let mut lane = LaneDriver::<Mock>::default();
    let mut acct = accountant();

    for expected in [4, 6] {
        let outcome = lane
            .run(&site, LaneCall::NextBatch, None, RowRange::WHOLE, &mut acct)
            .expect("a batch");
        assert_eq!(rows_of(&outcome.outputs), expected);
        assert!(!outcome.finished);
    }
    let outcome = lane
        .run(&site, LaneCall::NextBatch, None, RowRange::WHOLE, &mut acct)
        .expect("exhaustion is not an error");
    assert_eq!(outcome.call, CallKind::SourceExhausted);
    assert!(outcome.finished && lane.is_finished());
}

#[test]
fn stepping_a_lane_after_it_finished_is_an_error() {
    let script = Script::default().source("part", vec![vec![]]);
    let node = plans::source("part", 1);
    let site = site(&script, node.as_ref());
    let mut lane = LaneDriver::<Mock>::default();
    let mut acct = accountant();

    lane.run(&site, LaneCall::NextBatch, None, RowRange::WHOLE, &mut acct)
        .expect("an empty source is exhausted at once");
    match lane.run(&site, LaneCall::NextBatch, None, RowRange::WHOLE, &mut acct) {
        Err(StepError::Run(RunError::Protocol(said))) => {
            assert!(said.contains("stepped after it finished"), "{said}")
        }
        _ => panic!("a finished lane accepted another call"),
    }
}

#[test]
fn the_executor_is_built_on_the_first_step_and_not_before() {
    // A lane that never runs must cost nothing: at four lanes over two row groups, two of
    // them have no work and building eagerly would allocate for both.
    let script = Script::default().source("part", vec![vec![spec(4, 32)]]);
    let node = plans::source("part", 1);
    let site = site(&script, node.as_ref());
    let mut lane = LaneDriver::<Mock>::default();
    let mut acct = accountant();

    assert_eq!(
        script.built.load(Ordering::Relaxed),
        0,
        "built before a step"
    );
    lane.run(&site, LaneCall::NextBatch, None, RowRange::WHOLE, &mut acct)
        .expect("a batch");
    assert_eq!(script.built.load(Ordering::Relaxed), 1);
    lane.run(&site, LaneCall::NextBatch, None, RowRange::WHOLE, &mut acct)
        .expect("exhausted");
    assert_eq!(script.built.load(Ordering::Relaxed), 1, "rebuilt mid-lane");
}

#[test]
fn an_exec_is_one_call_per_batch_and_then_ends_the_lane() {
    let script = Script::default();
    let node = plans::filter(plans::source("part", 1));
    let site = site(&script, node.as_ref());
    let mut lane = LaneDriver::<Mock>::default();
    let mut acct = accountant();

    assert_eq!(
        lane.select(site.category, &has_input()).unwrap(),
        LaneCall::Exec
    );
    let input = batch(7, 56, &mut acct);
    let outcome = lane
        .run(
            &site,
            LaneCall::Exec,
            Some(input),
            RowRange::WHOLE,
            &mut acct,
        )
        .expect("one out per one in");
    assert_eq!(rows_of(&outcome.outputs), 7);
    assert!(!outcome.finished);

    // Its input ended, and an exec has nothing to emit at the end — so the lane ends with
    // no call at all rather than a call that produces nothing.
    assert_eq!(
        lane.select(site.category, &input_ended()).unwrap(),
        LaneCall::EndOfInput
    );
    let outcome = lane
        .run(
            &site,
            LaneCall::EndOfInput,
            None,
            RowRange::WHOLE,
            &mut acct,
        )
        .expect("the lane ends");
    assert!(outcome.finished && rows_of(&outcome.outputs) == 0);
}

#[test]
fn an_accumulator_emits_only_when_it_is_told_its_input_ended() {
    let script = Script::default();
    let node = plans::coalesce_all(plans::source("part", 1));
    let site = site(&script, node.as_ref());
    let mut lane = LaneDriver::<Mock>::default();
    let mut acct = accountant();

    for rows in [3, 4] {
        let input = batch(rows, rows * 8, &mut acct);
        let outcome = lane
            .run(
                &site,
                LaneCall::Accumulate,
                Some(input),
                RowRange::WHOLE,
                &mut acct,
            )
            .expect("accumulated");
        assert_eq!(
            rows_of(&outcome.outputs),
            0,
            "an accumulator emits at done only"
        );
    }
    assert_eq!(
        lane.select(site.category, &input_ended()).unwrap(),
        LaneCall::MarkDone
    );
    let outcome = lane
        .run(&site, LaneCall::MarkDone, None, RowRange::WHOLE, &mut acct)
        .expect("emitted at done");
    assert_eq!(rows_of(&outcome.outputs), 7, "everything it held, once");
    assert!(outcome.finished);
}

/// A join lane driven to its probe phase, with the build batch already delivered.
fn probing(script: &Script, node: &dyn GpuNode, acct: &mut ResidentAccountant) -> LaneDriver<Mock> {
    let mut lane = LaneDriver::<Mock>::default();
    let build = batch(5, 40, acct);
    lane.run(
        &site(script, node),
        LaneCall::SetBuild,
        Some(build),
        RowRange::WHOLE,
        acct,
    )
    .expect("the build side is set");
    lane
}

fn join_node() -> Box<dyn GpuNode> {
    plans::join(
        plans::coalesce_all(plans::source("build", 1)),
        plans::source("probe", 1),
    )
}

#[test]
fn a_join_sets_its_build_side_before_it_probes() {
    let script = Script::default();
    let node = join_node();
    let mut acct = accountant();
    let fresh = LaneDriver::<Mock>::default();
    let category = site(&script, node.as_ref()).category;

    assert!(fresh.awaits_build());
    assert_eq!(
        fresh.select(category, &has_input()).unwrap(),
        LaneCall::SetBuild
    );

    let lane = probing(&script, node.as_ref(), &mut acct);
    assert!(!lane.awaits_build(), "set_build did not move the lane on");
    // Only the probe slot: a build side with a second batch is the violation below.
    let probe_only = Avail {
        has: [false, true],
        done: [true, false],
    };
    assert_eq!(lane.select(category, &probe_only).unwrap(), LaneCall::Probe);
}

#[test]
fn a_join_emits_its_unmatched_rows_at_finish() {
    let script = Script::default().with_join(JoinRule {
        finish_rows: 3,
        build_residency: 0,
    });
    let node = join_node();
    let mut acct = accountant();
    let mut lane = probing(&script, node.as_ref(), &mut acct);
    let site = site(&script, node.as_ref());

    let probe = batch(9, 72, &mut acct);
    let outcome = lane
        .run(
            &site,
            LaneCall::Probe,
            Some(probe),
            RowRange::WHOLE,
            &mut acct,
        )
        .expect("probed");
    assert_eq!(rows_of(&outcome.outputs), 5, "capped by the build side");
    assert!(!outcome.finished);

    assert_eq!(
        lane.select(site.category, &input_ended()).unwrap(),
        LaneCall::Finish
    );
    let outcome = lane
        .run(&site, LaneCall::Finish, None, RowRange::WHOLE, &mut acct)
        .expect("finished");
    assert_eq!(
        rows_of(&outcome.outputs),
        3,
        "the build rows nothing matched"
    );
    assert!(outcome.finished);
}

#[test]
fn a_build_side_that_never_produced_is_refused_at_the_decision() {
    let script = Script::default();
    let node = join_node();
    let category = site(&script, node.as_ref()).category;
    let ended = Avail {
        has: [false, true],
        done: [true, false],
    };
    match LaneDriver::<Mock>::default().select(category, &ended) {
        Err(RunError::Protocol(said)) => {
            assert!(said.contains("without producing a batch"), "{said}")
        }
        other => panic!("a join with no build batch chose {other:?}"),
    }
}

#[test]
fn a_second_build_batch_is_refused_at_the_decision() {
    let script = Script::default();
    let node = join_node();
    let mut acct = accountant();
    let lane = probing(&script, node.as_ref(), &mut acct);
    match lane.select(site(&script, node.as_ref()).category, &has_input()) {
        Err(RunError::Protocol(said)) => assert!(said.contains("second batch"), "{said}"),
        other => panic!("a join probing with a second build batch chose {other:?}"),
    }
}

#[test]
fn a_decision_only_ever_consumes_a_slot_that_has_a_batch() {
    // Derived rather than transcribed: restating `can_step`'s disjunction here would keep
    // this green through any change made in both places. What no arm of `select` may do is
    // name a slot with nothing in it — which is exactly what a probe reading the build slot
    // did, and what that bug did not turn red.
    let script = Script::default();
    let nodes: Vec<Box<dyn GpuNode>> = vec![
        plans::source("part", 1),
        plans::filter(plans::source("part", 1)),
        plans::coalesce_all(plans::source("part", 1)),
        join_node(),
        plans::unload(plans::source("part", 1)),
    ];
    // A guard has to prove it was reached: the loop skips every state `can_step` refuses,
    // so a category narrowed to never step would empty its own 32 states in silence.
    let mut stepped = vec![0usize; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        let category = site(&script, node.as_ref()).category;
        for bits in 0..16u8 {
            let avail = Avail {
                has: [bits & 1 != 0, bits & 2 != 0],
                done: [bits & 4 != 0, bits & 8 != 0],
            };
            for probe_phase in [false, true] {
                let mut acct = accountant();
                let lane = match (probe_phase, category) {
                    (true, ExecutorCategory::Join) => probing(&script, node.as_ref(), &mut acct),
                    (true, _) => continue,
                    (false, _) => LaneDriver::<Mock>::default(),
                };
                if !lane.can_step(category, &avail) {
                    continue;
                }
                stepped[index] += 1;
                // Either a call or a named violation — never a silence the driver would
                // have to invent an answer for.
                let decision = lane.select(category, &avail);
                match decision {
                    Ok(call) => {
                        if let Some(slot) = call.consumes() {
                            assert!(
                                avail.has[slot],
                                "{category:?} chose {call:?}, which consumes slot {slot}, \
                                 over {avail:?}"
                            );
                        }
                    }
                    Err(RunError::Protocol(_)) => {}
                    other => panic!("{category:?} over {avail:?} produced {other:?}"),
                }
            }
        }
    }
    for (index, count) in stepped.iter().enumerate() {
        assert!(
            *count > 0,
            "{:?} never said yes to any of its states, so nothing about it was asserted",
            site(&script, nodes[index].as_ref()).category
        );
    }
}
