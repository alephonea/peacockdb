//! The portable half of the prototype's stress surface: the knobs that assert flow rather
//! than answers.
//!
//! Under mocks "the same answer at every shape" degrades to "every row that entered was
//! delivered", which is still worth asserting — a routing bug breaks it. The oracle
//! comparison the injector exists for needs real operators and belongs to T14.

use super::super::mock::{EmitRule, Script, spec};
use super::super::plans::*;
use super::*;

/// One shape of the same query: how many lanes the scan has, how many batches each holds,
/// and where the scatter puts the rows.
struct Shape {
    what: &'static str,
    lanes: usize,
    batches: usize,
    emit: EmitRule,
}

fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            what: "one lane, one batch",
            lanes: 1,
            batches: 1,
            emit: EmitRule::RoundRobin,
        },
        Shape {
            what: "one lane, many batches",
            lanes: 1,
            batches: 5,
            emit: EmitRule::RoundRobin,
        },
        Shape {
            what: "four lanes, one batch each",
            lanes: 4,
            batches: 1,
            emit: EmitRule::RoundRobin,
        },
        Shape {
            what: "four lanes, many batches",
            lanes: 4,
            batches: 3,
            emit: EmitRule::RoundRobin,
        },
        Shape {
            what: "every key into one lane",
            lanes: 4,
            batches: 3,
            emit: EmitRule::ToLane(1),
        },
    ]
}

fn script_for(shape: &Shape) -> Script {
    Script::default()
        .source("part", vec![vec![spec(8, 64); shape.batches]; shape.lanes])
        .with_emit(shape.emit)
}

/// scan -> merge -> scatter -> merge -> unload: a shuffle whose lane count is the shape's,
/// so every layout runs the same plan over the same rows.
fn shuffled(lanes: usize) -> Box<dyn GpuNode> {
    unload(merge(emit(merge(source("part", lanes)), 4)))
}

#[test]
fn every_shape_delivers_every_row_it_was_given() {
    for shape in shapes() {
        let script = script_for(&shape);
        let report = run(shuffled(shape.lanes).as_ref(), &script);
        assert_eq!(
            rows_returned(&report),
            8 * shape.batches * shape.lanes,
            "rows were lost or invented at {}",
            shape.what
        );
        assert_eq!(
            report.in_flight_bytes, 0,
            "a batch leaked at {}",
            shape.what
        );
        assert_eq!(
            report.holds, report.releases,
            "held and released disagree at {}",
            shape.what
        );
    }
}

#[test]
fn every_shape_holds_the_queue_bound() {
    for shape in shapes() {
        let script = script_for(&shape);
        let plan = shuffled(shape.lanes);
        let report = run(plan.as_ref(), &script);
        for (node, queued) in report.peak_queued.iter().enumerate() {
            // One batch per lane per producing node, with no cap anywhere: the parent's
            // height is strictly lower, so it drains before the producer runs again.
            let lanes = report.lanes_of[node];
            assert!(
                *queued <= lanes,
                "node {node} held {queued} batches over {lanes} lanes at {}",
                shape.what
            );
        }
    }
}

/// A node the plan did not ask for, in two kinds. The pass-through moves every height and
/// every rank without touching a batch; the rebatcher also changes the shape batches arrive
/// in, which is the property the prototype injects #139's node for — and this mode has one
/// today, since a coalesce is N-to-1.
fn pass_through(input: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
    // A pure offset of zero forwards every batch and never satisfies, by the lowering rule.
    limit(input, 0, None)
}

fn injected_plan(
    lanes: usize,
    inject: &dyn Fn(Box<dyn GpuNode>) -> Box<dyn GpuNode>,
) -> Box<dyn GpuNode> {
    unload(merge(emit(inject(merge(inject(source("part", lanes)))), 4)))
}

#[test]
fn an_extra_level_in_the_tree_changes_no_delivery_and_no_bound() {
    for shape in shapes() {
        let script = script_for(&shape);
        let plain = run(shuffled(shape.lanes).as_ref(), &script);
        let carried = run(injected_plan(shape.lanes, &pass_through).as_ref(), &script);
        assert_eq!(
            rows_returned(&carried),
            rows_returned(&plain),
            "an extra level changed what was delivered at {}",
            shape.what
        );
        for (node, queued) in carried.peak_queued.iter().enumerate() {
            assert!(*queued <= carried.lanes_of[node], "at {}", shape.what);
        }
        assert_eq!(carried.holds, carried.releases);
        assert_eq!(carried.in_flight_bytes, 0);
    }
}

#[test]
fn a_node_that_collapses_the_batches_below_it_changes_no_delivery() {
    // Nothing downstream may depend on the shape batches arrive in. A coalesce above every
    // source turns many batches into one, so every consumer sees a different arrival
    // pattern for the same rows.
    for shape in shapes() {
        let script = script_for(&shape);
        let plain = run(shuffled(shape.lanes).as_ref(), &script);
        let plan = injected_plan(shape.lanes, &coalesce_all);
        let rebatched = run(plan.as_ref(), &script);
        assert_eq!(
            rows_returned(&rebatched),
            rows_returned(&plain),
            "collapsing the batches changed what was delivered at {}",
            shape.what
        );
        for (node, queued) in rebatched.peak_queued.iter().enumerate() {
            assert!(*queued <= rebatched.lanes_of[node], "at {}", shape.what);
        }
        assert_eq!(rebatched.holds, rebatched.releases);
        assert_eq!(rebatched.in_flight_bytes, 0);
    }
}

#[test]
fn nested_shuffles_hold_every_bound_at_once() {
    // scatter over merge over scatter: two shuffles live at the same time, which is a
    // different bound from two joins — nothing here is held, so every queue has to be
    // drained by height alone.
    let plan = unload(merge(emit(merge(emit(merge(source("part", 2)), 4)), 3)));
    let script = Script::default().source("part", vec![vec![spec(12, 96); 2]; 2]);
    let report = run(plan.as_ref(), &script);
    assert_eq!(rows_returned(&report), 48);
    for (node, queued) in report.peak_queued.iter().enumerate() {
        assert!(
            *queued <= report.lanes_of[node],
            "node {node} held {queued} over {} lanes",
            report.lanes_of[node]
        );
    }
    assert_accounted(&report);
}
