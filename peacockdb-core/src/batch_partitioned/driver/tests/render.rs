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
use crate::batch_partitioned::plan_text::{render_plan, render_run};

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
