//! The execution golden's text, over the mock backend so every number in it is chosen.
//!
//! The device batches carry the script's sizes; the unload's are the host's, priced from the
//! plan's schema like everything else — ten Int64 rows are 80 bytes and a two-byte bitmap.
//!
//! Whole-text asserts rather than substring ones: what the renderer is for is a file a
//! comparator reads line for line, and a test that checks a line is there does not check
//! that nothing else is.

use super::super::mock::{EmitRule, Script, spec};
use super::super::plans::*;
use super::*;
use crate::batch_partitioned::driver::{join_regions, nodes_as_recorded};
use crate::batch_partitioned::driver::Region;
use crate::batch_partitioned::plan_text::{render_plan, render_run, render_timings};

#[test]
fn a_chain_renders_the_tree_and_what_each_node_produced() {
    let script = Script::default().source("part", vec![vec![spec(10, 80), spec(7, 56)]]);
    let plan = unload(filter(source("part", 1)));
    let report = run(plan.as_ref(), &script);
    assert_eq!(
        render_run(plan.as_ref(), &report),
        "\
early_exit=none
GpuUnload: output_rows=17, output_bytes=139
  in_rows=[[17]] batch_rows=[[10,7]] batch_bytes=[[82,57]]
  GpuFilter: predicate=k@0, lanes=1, batches=multiple, output_rows=17, output_bytes=136
    in_rows=[[17]] batch_rows=[[10,7]] batch_bytes=[[80,56]]
    GpuLoadParquet: table=part, projections=[k@0], partition_groups=[[[0]]], lanes=1, \
batches=multiple, output_rows=17, output_bytes=136
      in_rows=[] batch_rows=[[10,7]] batch_bytes=[[80,56]]
"
    );
}

/// Lanes outermost and batches within, which is what makes element `i` of lane `j` one
/// batch in all three lists — and `in_rows` indexed by the child's lane, which on an
/// emitter is the only indexing that can be compared to anything.
#[test]
fn an_emitter_renders_four_lanes_out_and_the_one_lane_it_read() {
    let script = Script::default()
        .source("part", vec![vec![spec(12, 96), spec(8, 64)]])
        .with_emit(EmitRule::RoundRobin);
    let plan = unload(merge(emit(source("part", 1), 4)));
    let report = run(plan.as_ref(), &script);
    assert_eq!(
        render_run(plan.as_ref(), &report),
        "\
early_exit=none
GpuUnload: output_rows=20, output_bytes=168
  in_rows=[[20]] batch_rows=[[3,3,3,3,2,2,2,2]] batch_bytes=[[25,25,25,25,17,17,17,17]]
  GpuMergePartitions: lanes=1, batches=multiple, output_rows=20, output_bytes=160
    in_rows=[[5,5,5,5]] batch_rows=[[3,3,3,3,2,2,2,2]] batch_bytes=[[24,24,24,24,16,16,16,16]]
    GpuEmitPartitions: hash=[k@0], lanes=4, batches=multiple, hashed_on=[k@0], output_rows=20, \
output_bytes=160
      in_rows=[[20]] batch_rows=[[3,2],[3,2],[3,2],[3,2]] batch_bytes=[[24,16],[24,16],[24,16],[24,16]]
      GpuLoadParquet: table=part, projections=[k@0], partition_groups=[[[0]]], lanes=1, \
batches=multiple, output_rows=20, output_bytes=160
        in_rows=[] batch_rows=[[12,8]] batch_bytes=[[96,64]]
"
    );
}

/// A lane that produced nothing is an empty list at its own position, not a lane the file
/// leaves out: the position is what pairs a lane with its sibling numbers.
#[test]
fn a_lane_that_produced_nothing_keeps_its_position() {
    let script = Script::default().source("part", vec![vec![spec(10, 80)], vec![]]);
    let plan = unload(merge(filter(source("part", 2))));
    let report = run(plan.as_ref(), &script);
    assert_eq!(
        render_run(plan.as_ref(), &report),
        "\
early_exit=none
GpuUnload: output_rows=10, output_bytes=82
  in_rows=[[10]] batch_rows=[[10]] batch_bytes=[[82]]
  GpuMergePartitions: lanes=1, batches=multiple, output_rows=10, output_bytes=80
    in_rows=[[10,0]] batch_rows=[[10]] batch_bytes=[[80]]
    GpuFilter: predicate=k@0, lanes=2, batches=multiple, output_rows=10, output_bytes=80
      in_rows=[[10,0]] batch_rows=[[10],[]] batch_bytes=[[80],[]]
      GpuLoadParquet: table=part, projections=[k@0], partition_groups=[[[0]],[[1]]], lanes=2, \
batches=multiple, output_rows=10, output_bytes=80
        in_rows=[] batch_rows=[[10],[]] batch_bytes=[[80],[]]
"
    );
}

/// The declared schema is a plan fact and belongs to the plan golden alone — recorded once
/// per plan rather than repeated in every query's execution section.
#[test]
fn the_execution_text_carries_no_declared_schema() {
    let script = Script::default().source("part", vec![vec![spec(10, 80)]]);
    let plan = unload(filter(source("part", 1)));
    let report = run(plan.as_ref(), &script);
    assert!(render_plan(plan.as_ref()).contains("schema=[k:Int64]"));
    assert!(!render_run(plan.as_ref(), &report).contains("schema="));
}

/// One node, one entry, however many batches it emitted — the worst source in the corpus
/// has 96 in a lane, and a line per batch would make a node unfindable.
#[test]
fn a_node_stays_one_entry_however_many_batches_it_emitted() {
    let script = Script::default().source("part", vec![vec![spec(1, 8); 20]]);
    let plan = unload(filter(source("part", 1)));
    let report = run(plan.as_ref(), &script);
    let text = render_run(plan.as_ref(), &report);
    assert_eq!(
        text.lines().count(),
        7,
        "the marker, then three nodes at two lines each:\n{text}"
    );
    assert!(text.contains(&format!("batch_rows=[[{}]]", ["1"; 20].join(","))));
}

/// `abandoned` is what makes the arithmetic a law rather than an inequality: the merge
/// consumed 4+4+0 of the loader's three lanes, and the 4 rows the early exit stranded are
/// on the loader's own line. It prints only where a run left something behind.
#[test]
fn an_early_exit_renders_the_rows_it_left_standing() {
    let script = Script::default().source("part", vec![vec![spec(4, 32)]; 3]);
    let plan = unload_limited(merge(source("part", 3)), 0, Some(5));
    let report = run(plan.as_ref(), &script);
    assert_eq!(
        render_run(plan.as_ref(), &report),
        "\
early_exit=GpuUnload@2
GpuUnload: skip=0, fetch=5, output_rows=5, output_bytes=42
  in_rows=[[8]] batch_rows=[[4,1]] batch_bytes=[[33,9]]
  GpuMergePartitions: lanes=1, batches=multiple, output_rows=8, output_bytes=64
    in_rows=[[4,4,0]] batch_rows=[[4,4]] batch_bytes=[[32,32]]
    GpuLoadParquet: table=part, projections=[k@0], partition_groups=[[[0]],[[1]],[[2]]], \
lanes=3, batches=multiple, output_rows=12, output_bytes=96
      in_rows=[] batch_rows=[[4],[4],[4]] batch_bytes=[[32],[32],[32]] abandoned=[0,0,4]
"
    );
}

/// `rows_skipped` is the saving a limit buys, and it sits with `abandoned` because both are
/// early-exit quantities: rows released without an unload call, on the node that released
/// them. Twenty of the forty rows that reached the unload were before the interval began.
#[test]
fn a_skipped_prefix_is_named_on_the_node_that_released_it() {
    let script = Script::default().source("part", vec![vec![spec(10, 80); 6]]);
    let plan = unload_limited(filter(source("part", 1)), 25, Some(10));
    let report = run(plan.as_ref(), &script);
    let text = render_run(plan.as_ref(), &report);
    assert!(
        text.contains("early_exit=GpuUnload@2\n"),
        "the marker names the limit that stopped it:\n{text}"
    );
    assert!(
        text.contains(" rows_skipped=20\n"),
        "and the unload says what it threw away:\n{text}"
    );
    assert_eq!(
        text.matches("rows_skipped").count(),
        1,
        "no other node skipped anything:\n{text}"
    );
}

/// A run that drained mentions it nowhere, which is what keeps the common section as it
/// was: five queries in the corpus stop early, and every other node of every other query
/// would otherwise carry a list of zeroes.
#[test]
fn a_run_that_drained_renders_no_abandoned_at_all() {
    let script = Script::default().source("part", vec![vec![spec(4, 32)]; 3]);
    let plan = unload(merge(source("part", 3)));
    let report = run(plan.as_ref(), &script);
    let text = render_run(plan.as_ref(), &report);
    assert!(!text.contains("abandoned"), "{text}");
    assert!(!text.contains("rows_skipped"), "{text}");
}

/// The same tree the execution golden renders, annotated with what it cost instead of what
/// it produced — and rows and bytes deliberately absent, since the file beside this one
/// already carries them.
///
/// The mock addresses no seq, so nothing here is measured and every entry is `-`. That is
/// the case worth pinning first: a renderer that printed `0` for it would report a backend
/// that measures nothing as one that costs nothing.
///
/// Two shapes worth reading off this: `GpuUnload` carries no colon, because the colon
/// separates a node from its properties and this file gives it none; and the source has one
/// entry per batch and none for the step that found the queue empty, which made no call to
/// record.
#[test]
fn a_timing_record_renders_the_tree_and_what_each_node_cost() {
    let script = Script::default().source("part", vec![vec![spec(10, 80), spec(7, 56)]]);
    let plan = unload(filter(source("part", 1)));
    let report = run(plan.as_ref(), &script);
    let (times, unclaimed) = join_regions(&report, &[]);
    assert!(unclaimed.is_empty(), "no regions, so none go unclaimed");
    assert_eq!(
        render_timings(plan.as_ref(), &times),
        "\
GpuUnload
  time_us=[[-,-]] total_us=-
  GpuFilter: predicate=k@0, lanes=1, batches=multiple
    time_us=[[-,-]] total_us=-
    GpuLoadParquet: table=part, projections=[k@0], partition_groups=[[[0]]], lanes=1, \
batches=multiple
      time_us=[[-,-]] total_us=-
"
    );
}

/// The two node indexes stay apart: `node_seq` is the tree's post-order, and the report is
/// walked in the driver's pre-order.
///
/// Asserted on a chain where the two disagree at every node — a three-node chain has the
/// root at pre-order 0 and post-order 2 — because a writer that emitted the walk index
/// instead would produce numbers that look exactly as plausible.
#[test]
fn the_recorded_node_index_is_the_post_order_not_the_walk_order() {
    let plan = unload(filter(source("part", 1)));
    let recorded = nodes_as_recorded(plan.as_ref()).expect("the tree indexes");
    let walk: Vec<usize> = (0..recorded.len()).collect();
    let post: Vec<usize> = recorded.iter().map(|(_, post)| *post).collect();
    assert_eq!(
        recorded.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        ["GpuUnload", "GpuFilter", "GpuLoadParquet"],
        "pre-order names the root first"
    );
    assert_eq!(post, [2, 1, 0], "post-order names the leaf first");
    assert_ne!(walk, post, "the two orders are what a writer must not confuse");
}

/// A driver call addressing two seqs is measured as two, and the entry is their sum.
///
/// The distinction is the whole reason the join keeps both granularities. Costing an entry
/// and handing the total to each of its seqs reports a merge that produced one row as
/// having produced everything its entry did — which is what it did before this test.
#[test]
fn each_seq_of_one_driver_call_keeps_its_own_measurement() {
    let script = Script::default().source("part", vec![vec![spec(10, 80)]]);
    let plan = unload(filter(source("part", 1)));
    let report = run(plan.as_ref(), &script);
    let regions = [
        region(7, 0, 40, 4),
        region(8, 0, 60, 1),
    ];
    let (measured, unclaimed) = join_regions(&report, &regions);
    assert_eq!(unclaimed.len(), 2, "the mock addresses no seq, so neither is claimed");

    let first = measured.call(7, 0).expect("the device answered for it");
    let second = measured.call(8, 0).expect("and for the other");
    assert_eq!((first.device_us, first.out_rows), (40, 4));
    assert_eq!((second.device_us, second.out_rows), (60, 1), "not the pair's total");
}

/// A region with the numbers a case wants to assert on, built here because the mock backend
/// addresses no seq and so produces none of its own.
fn region(seq: u32, call_index: u64, device_us: u64, out_rows: u64) -> Region {
    Region {
        seq,
        partition: 0,
        call_index,
        host_setup_us: 0,
        host_submit_us: device_us,
        device_us,
        out_rows,
        logical_bytes: out_rows * 8,
    }
}
