//! What each node emitted and what it consumed — the two records the corpus goldens read.
//!
//! The law they rest on is `consumed + abandoned == emitted`, per the child's own lane: a
//! row a node emitted was either taken by its parent or left standing by an early exit.
//! An equality on every run, with no exception keyed on node kind — the two that once
//! looked like exceptions were measured and are not, and each is pinned below.

use super::super::mock::{AccRule, EmitRule, Script, spec};
use super::super::plans::*;
use super::*;

/// Node numbering is pre-order, so the root is 0 and a chain descends from it.
const UNLOAD: usize = 0;

fn rows_of(report: &RunReport, node: usize, lane: usize) -> Vec<u64> {
    report.emitted[node][lane]
        .iter()
        .map(|batch| batch.rows)
        .collect()
}

fn bytes_of(report: &RunReport, node: usize, lane: usize) -> Vec<usize> {
    report.emitted[node][lane]
        .iter()
        .map(|batch| batch.bytes)
        .collect()
}

fn emitted_rows(report: &RunReport, node: usize) -> u64 {
    report.emitted[node]
        .iter()
        .flatten()
        .map(|batch| batch.rows)
        .sum()
}

#[test]
fn every_batch_a_node_emitted_is_in_the_report_in_order() {
    let script = Script::default().source("part", vec![vec![spec(10, 80), spec(7, 56)]]);
    let plan = unload(filter(source("part", 1)));
    let report = run(plan.as_ref(), &script);
    assert_eq!(
        rows_of(&report, 2, 0),
        vec![10, 7],
        "the source's two batches"
    );
    assert_eq!(bytes_of(&report, 2, 0), vec![80, 56]);
    assert_eq!(
        rows_of(&report, 1, 0),
        vec![10, 7],
        "the filter passes both"
    );
    assert_eq!(
        rows_of(&report, UNLOAD, 0),
        vec![10, 7],
        "and each crosses to the host as its own batch"
    );
    assert_accounted(&report);
}

#[test]
fn a_lane_that_produced_nothing_carries_an_empty_list_rather_than_no_lane() {
    let script = Script::default().source("part", vec![vec![spec(10, 80)], vec![]]);
    let plan = unload(filter(source("part", 2)));
    let report = run(plan.as_ref(), &script);
    assert_eq!(report.emitted[2].len(), 2, "both lanes are in the record");
    assert_eq!(rows_of(&report, 2, 1), Vec::<u64>::new());
    assert_eq!(report.consumed[1][0][1], 0);
    assert_conserved(plan.as_ref(), &report);
}

#[test]
fn a_map_node_consumed_exactly_what_its_child_emitted_lane_for_lane() {
    let batches = vec![spec(10, 80); 3];
    let script =
        Script::default().source("part", vec![batches.clone(), vec![spec(4, 32)], batches]);
    let plan = unload(project(filter(source("part", 3))));
    let report = run(plan.as_ref(), &script);
    // 0 unload, 1 project, 2 filter, 3 source.
    assert_conserved(plan.as_ref(), &report);
    assert_eq!(report.consumed[2][0], vec![30, 4, 30]);
    assert_accounted(&report);
}

/// An emitter is the node the identity is worth having on: it reads one lane and writes
/// four, so consumed indexed by its own lanes could not be compared to anything.
#[test]
fn an_emitter_redistributes_and_its_consumed_still_names_the_lane_it_read() {
    let script = Script::default()
        .source("part", vec![vec![spec(12, 96), spec(8, 64)]])
        .with_emit(EmitRule::RoundRobin);
    let plan = unload(merge(emit(source("part", 1), 4)));
    let report = run(plan.as_ref(), &script);
    // 0 unload, 1 merge, 2 emit, 3 source.
    assert_eq!(report.consumed[2][0], vec![20], "one lane in, both batches");
    assert_eq!(
        emitted_rows(&report, 2),
        20,
        "and the same rows out over four"
    );
    assert_eq!(report.emitted[2].len(), 4);
    assert_conserved(plan.as_ref(), &report);
    assert_accounted(&report);
}

/// The scatter drops an output holding no rows before it reaches a queue, so the count is
/// what flowed rather than what was returned.
#[test]
fn a_scatter_output_with_no_rows_is_not_a_batch() {
    let script = Script::default()
        .source("part", vec![vec![spec(12, 96), spec(8, 64)]])
        .with_emit(EmitRule::ToLane(0));
    let plan = unload(merge(emit(source("part", 1), 4)));
    let report = run(plan.as_ref(), &script);
    assert_eq!(rows_of(&report, 2, 0), vec![12, 8], "every row took lane 0");
    for lane in 1..4 {
        assert_eq!(
            rows_of(&report, 2, lane),
            Vec::<u64>::new(),
            "lane {lane} was emitted empty and dropped"
        );
    }
    assert_eq!(emitted_rows(&report, 2), 20);
    assert_conserved(plan.as_ref(), &report);
}

/// The cross-lane accumulator: four lanes in, one out, and `consumed` indexed by the lane
/// each batch came from.
#[test]
fn a_partition_accumulator_names_the_lane_each_batch_came_from() {
    let script = Script::default().source(
        "part",
        vec![
            vec![spec(10, 80)],
            vec![spec(3, 24)],
            vec![],
            vec![spec(7, 56), spec(1, 8)],
        ],
    );
    let plan = unload(merge(source("part", 4)));
    let report = run(plan.as_ref(), &script);
    // 0 unload, 1 merge, 2 source.
    assert_eq!(report.consumed[1][0], vec![10, 3, 0, 8]);
    assert_conserved(plan.as_ref(), &report);
    assert_accounted(&report);
}

/// A satisfied limit does not break the identity at the node that carries it: every batch
/// that reached it was consumed, and it stopped by not being scheduled again rather than by
/// leaving one behind.
#[test]
fn a_node_whose_limit_was_satisfied_consumed_every_batch_that_reached_it() {
    let script = Script::default().source("part", vec![vec![spec(10, 80); 6]]);
    let plan = unload_limited(filter(source("part", 1)), 0, Some(15));
    let report = run(plan.as_ref(), &script);
    assert!(!report.satisfied.is_empty());
    let produced: u64 = rows_of(&report, 1, 0).iter().sum();
    assert_eq!(produced, 20, "the filter had passed two batches");
    assert_eq!(
        report.consumed[UNLOAD][0][0], produced,
        "both of which the unload consumed, taking 15 rows of them"
    );
    assert_eq!(emitted_rows(&report, UNLOAD), 15);
    assert_conserved(plan.as_ref(), &report);
    assert_accounted(&report);
}

/// Where the inequality actually is. An early exit stops the schedule, and whatever sits in
/// a queue at that moment is released rather than consumed — so the node that falls short is
/// wherever the stop caught one, which is not the node carrying the limit. Measured here,
/// one four-row batch per lane under `LIMIT 5`:
///
///   lane 0: source emitted [4], merge consumed 4
///   lane 1: source emitted [4], merge consumed 4
///   lane 2: source emitted [4], merge consumed 0     <- the shortfall
///   merge emitted [4, 4],       unload consumed 8    <- the limit's own node, at equality
#[test]
fn an_early_exit_leaves_a_batch_unconsumed_below_the_node_that_stopped() {
    let script = Script::default().source("part", vec![vec![spec(4, 32)]; 3]);
    let plan = unload_limited(merge(source("part", 3)), 0, Some(5));
    let report = run(plan.as_ref(), &script);
    // 0 unload, 1 merge, 2 source.
    assert!(!report.satisfied.is_empty());
    assert_eq!(rows_of(&report, 2, 2), vec![4], "lane 2 produced its batch");
    assert_eq!(
        report.consumed[1][0][2], 0,
        "and the merge above it never read it"
    );
    assert_eq!(
        report.abandoned[2],
        vec![0, 0, 4],
        "which is where those rows are"
    );
    assert_conserved(plan.as_ref(), &report);
    assert_eq!(
        report.consumed[UNLOAD][0][0],
        emitted_rows(&report, 1),
        "while the limit's own node is at equality with its child"
    );
    assert_accounted(&report);
}

/// A join whose build side ended empty owes nothing, and it reads its probe side anyway to
/// let go of it — a dropped batch is a consumed one, so the identity holds here too.
#[test]
fn a_join_with_no_build_side_still_consumes_the_probe_batches_it_drops() {
    let script = Script::default()
        .source("build", vec![vec![spec(4, 32)]])
        .source("probe", vec![vec![spec(10, 80), spec(10, 80)]])
        .with_accumulator(AccRule::EmitAtDone(0));
    let plan = unload(join(
        coalesce_all(source("build", 1)),
        filter(source("probe", 1)),
    ));
    let report = run(plan.as_ref(), &script);
    // 0 unload, 1 join, 2 coalesce, 3 build source, 4 filter, 5 probe source.
    assert_eq!(rows_returned(&report), 0, "no build rows, no joined rows");
    assert_eq!(
        report.consumed[1][1][0], 20,
        "both probe batches were read to be released"
    );
    assert_conserved(plan.as_ref(), &report);
    assert_accounted(&report);
}

/// A skipped prefix is consumed and never emitted — the batch is released where it stands,
/// so it is in `consumed` and in no `emitted` list.
#[test]
fn a_released_skip_prefix_counts_as_consumed_and_emits_nothing() {
    let script = Script::default().source("part", vec![vec![spec(10, 80); 6]]);
    let plan = unload_limited(filter(source("part", 1)), 25, Some(10));
    let report = run(plan.as_ref(), &script);
    assert_eq!(
        report.consumed[UNLOAD][0][0], 40,
        "four batches reached the unload"
    );
    assert_eq!(emitted_rows(&report, UNLOAD), 10, "and ten rows left it");
    assert_eq!(report.rows_skipped.iter().sum::<u64>(), 20);
    assert_conserved(plan.as_ref(), &report);
}

/// The call record is indexed by the lanes that DRIVE a node, and the report says how
/// many those are — so a reader never has to guess whether a lane index means the input
/// side or the output one.
#[test]
fn the_call_record_is_indexed_by_the_driving_lanes_the_report_names() {
    let script = Script::default().source("part", vec![vec![spec(10, 80), spec(7, 56)]]);
    let plan = unload(filter(source("part", 1)));
    let report = run(plan.as_ref(), &script);
    assert_eq!(report.abi_calls.len(), report.driving_lanes.len());
    for node in 0..report.abi_calls.len() {
        assert_eq!(
            report.abi_calls[node].len(),
            report.driving_lanes[node],
            "node {node} has one call list per driving lane"
        );
        assert_eq!(
            report.driving_lanes[node], report.lanes_of[node],
            "a chain of map nodes drives on the lanes it emits into"
        );
    }
}

/// The two counts part company at exactly two nodes, and this is what makes naming them
/// separately worth the field: a scatter is driven on one lane and emits into four, a
/// cross-lane merge is driven on four and emits on one.
#[test]
fn a_scatter_and_a_cross_lane_merge_drive_on_other_lanes_than_they_emit() {
    let script = Script::default()
        .source("part", vec![vec![spec(12, 96)]])
        .with_emit(EmitRule::RoundRobin);
    let plan = unload(merge_sorted(emit(source("part", 1), 4)));
    let report = run(plan.as_ref(), &script);
    let differ: Vec<usize> = (0..report.abi_calls.len())
        .filter(|node| report.driving_lanes[*node] != report.lanes_of[*node])
        .collect();
    assert_eq!(differ.len(), 2, "the scatter and the merge, and nothing else");
    for node in differ {
        assert_eq!(
            report.abi_calls[node].len(),
            report.driving_lanes[node],
            "node {node} is recorded on the lanes it was driven on"
        );
    }
}

/// One entry per call that reached an executor, and none for a step the driver answered
/// itself. Two batches through a filter is two calls plus the done, and the end-of-input
/// that follows is the driver's own — a fourth entry would shift every batch after it.
#[test]
fn a_lane_records_its_backend_calls_and_not_the_drivers_own() {
    let script = Script::default().source("part", vec![vec![spec(10, 80), spec(7, 56)]]);
    let plan = unload(filter(source("part", 1)));
    let report = run(plan.as_ref(), &script);
    let filter_node = UNLOAD + 1;
    assert_eq!(
        report.abi_calls[filter_node][0].len(),
        rows_of(&report, filter_node, 0).len(),
        "a filter emits one batch per call and makes no call at done"
    );
}

/// A backend that addresses no seq reports nothing, and that is not the same statement as
/// a backend that made no calls — the mock is the first kind, and a reader that cannot
/// tell them apart renders a silent backend as a fast one.
#[test]
fn a_backend_that_names_no_seq_leaves_every_entry_unmeasured() {
    let script = Script::default().source("part", vec![vec![spec(10, 80), spec(7, 56)]]);
    let plan = unload(filter(source("part", 1)));
    let report = run(plan.as_ref(), &script);
    let mut entries = 0;
    for lanes in &report.abi_calls {
        for calls in lanes.iter().flatten() {
            assert!(calls.recorded().is_none(), "the mock measures nothing");
            entries += 1;
        }
    }
    assert!(entries > 0, "the run made calls to leave entries for");
}
