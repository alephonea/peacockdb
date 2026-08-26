//! The mock, against its own script.
//!
//! Every assertion in this directory is measured against this backend, so a mock that
//! miscounts is a thousand lines of tests agreeing with the wrong answer and staying
//! green. What is checked here is only that the instrument reads what it was set to —
//! a few cases, not a suite.

use super::super::mock::{AccRule, EmitRule, Mock, Script, spec};
use super::super::plans::*;
use super::*;

use crate::batch_partitioned::backend::{Backend, NodeExecutors};
use crate::batch_partitioned::batch::Batch;
use crate::batch_partitioned::driver::mock::MockBatch;
use crate::batch_partitioned::executor::{
    BatchAccumulatorExecutor, PartitionEmitterExecutor, SourceExecutor, SourceStep,
};

/// The node `depth` steps below the root, following first children — which is how a case
/// names the one node it is about without the plan builders returning handles.
fn descend(root: &dyn GpuNode, depth: usize) -> &dyn GpuNode {
    let mut node = root;
    for _ in 0..depth {
        node = *node.children().first().expect("a node above a leaf");
    }
    node
}

fn mock_batch(rows: usize, bytes: usize) -> MockBatch {
    MockBatch { rows, bytes }
}

/// Every batch one lane's source produces, as the rows and bytes it reported.
fn drained(script: &Script, node: &dyn GpuNode, lane: usize) -> Vec<(usize, usize)> {
    let NodeExecutors::Source(mut source) =
        Mock::executors_for(script, node, 0, lane).expect("a loader builds a source")
    else {
        panic!("a loader is a source");
    };
    let mut seen = Vec::new();
    loop {
        match source.next_batch().expect("the read succeeds") {
            SourceStep::Batch {
                batch,
                source: next,
                ..
            } => {
                seen.push((batch.num_rows(), batch.byte_size()));
                source = next;
            }
            SourceStep::Exhausted => return seen,
        }
    }
}

#[test]
fn a_source_produces_the_batches_its_script_gave_that_lane() {
    let script = Script::default().source(
        "part",
        vec![
            vec![spec(10, 80), spec(3, 24)],
            vec![spec(7, 56)],
            Vec::new(),
        ],
    );
    let node = source("part", 3);
    assert_eq!(drained(&script, node.as_ref(), 0), vec![(10, 80), (3, 24)]);
    assert_eq!(drained(&script, node.as_ref(), 1), vec![(7, 56)]);
    assert_eq!(
        drained(&script, node.as_ref(), 2),
        Vec::new(),
        "a lane the script gave nothing produces nothing"
    );
}

/// The skew an emitter fills its lanes by, which is what the shuffle cases rest on: the
/// named lane takes every row and the rest come back empty rather than absent.
#[test]
fn an_emitter_fills_the_lane_its_script_names() {
    let script = Script::default().with_emit(EmitRule::ToLane(2));
    let plan = unload(merge(emit(source("part", 1), 4)));
    let emitter_node = descend(plan.as_ref(), 2);
    let NodeExecutors::PartitionEmitter(mut emitter) =
        Mock::executors_for(&script, emitter_node, 0, 0).expect("an emitter builds")
    else {
        panic!("a scatter is an emitter");
    };
    let (out, _) = emitter.emit(mock_batch(12, 96)).expect("the scatter runs");
    let rows: Vec<usize> = out.iter().map(Batch::num_rows).collect();
    assert_eq!(
        rows,
        vec![0, 0, 12, 0],
        "every row into lane 2, four lanes out"
    );
}

/// Where an accumulator emits, which decides every queue-depth assertion made against it.
#[test]
fn an_accumulator_emits_where_its_script_says() {
    let arrivals = || vec![mock_batch(4, 32), mock_batch(6, 48)];
    let plan = unload(coalesce_all(source("part", 1)));
    let node = descend(plan.as_ref(), 1);

    let holding = Script::default().with_accumulator(AccRule::CoalesceAll);
    let (per_batch, at_done) = drive_accumulator(&holding, node, arrivals());
    assert_eq!(per_batch, 0, "a coalesce emits nothing until done");
    assert_eq!(at_done, 1, "and one batch when it comes");

    let streaming = Script::default().with_accumulator(AccRule::Streaming);
    let (per_batch, at_done) = drive_accumulator(&streaming, node, arrivals());
    assert_eq!(
        (per_batch, at_done),
        (2, 0),
        "a streaming one holds nothing"
    );

    let three = Script::default().with_accumulator(AccRule::EmitAtDone(3));
    let (per_batch, at_done) = drive_accumulator(&three, node, arrivals());
    assert_eq!(
        (per_batch, at_done),
        (0, 3),
        "and a scripted count is that count"
    );
}

/// Batches through one accumulator: how many it emitted per arrival, and how many at done.
fn drive_accumulator(
    script: &Script,
    node: &dyn GpuNode,
    arrivals: Vec<MockBatch>,
) -> (usize, usize) {
    let NodeExecutors::BatchAccumulator(mut accumulator) =
        Mock::executors_for(script, node, 0, 0).expect("an accumulator builds")
    else {
        panic!("a coalesce is a batch accumulator");
    };
    let mut per_batch = 0;
    for batch in arrivals {
        let (out, _) = accumulator
            .accumulate_and_fetch(batch)
            .expect("the arrival is accepted");
        per_batch += out.len();
    }
    let (out, _) = accumulator.mark_done_and_fetch().expect("done is accepted");
    (per_batch, out.len())
}


