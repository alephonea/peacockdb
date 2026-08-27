//! `GpuEmitPartitions`: what a scatter owes its driver, and what a join needs of it.
//!
//! Which lane a key lands in is not asserted here — that is the murmur3 rule, and the live
//! conformance gate is what ties it to the device. What is asserted is everything a plan
//! depends on: the count, that no row is lost or doubled, and that a key's lane does not
//! depend on the batch it arrived in.

use super::*;
use crate::batch_partitioned::cpu_backend::emit::CpuEmitter;
use crate::batch_partitioned::nodes::GpuEmitPartitions;

const LANES: usize = 4;

fn emitter() -> CpuEmitter {
    let node = GpuEmitPartitions::new(Given::of(&GROUPED), vec![0], LANES);
    CpuEmitter::new(&node, LANES, &schema_of(&GROUPED).fields).expect("the emitter builds")
}

fn keyed(keys: &[&str]) -> CpuBatch {
    grouped(
        keys.iter().map(|key| Some(*key)).collect(),
        keys.iter()
            .enumerate()
            .map(|(i, _)| Some(i as i64))
            .collect(),
    )
}

/// Which lane each key landed in, by reading the lanes back.
fn lane_of(emitted: &[CpuBatch]) -> Vec<(String, usize)> {
    let mut placed = Vec::new();
    for (lane, batch) in emitted.iter().enumerate() {
        let record = batch.record_batch();
        let keys = record
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("the key column");
        for row in 0..record.num_rows() {
            placed.push((keys.value(row).to_string(), lane));
        }
    }
    placed.sort();
    placed
}

/// The count is the contract: a driver reads output `p` as lane `p`'s, so an empty lane
/// still owes its batch — skipping it would shift every lane above it.
#[test]
fn a_scatter_answers_with_one_batch_per_lane_however_the_hash_fell() {
    let (emitted, _) = emitter().emit(keyed(&["a"])).expect("the scatter runs");
    assert_eq!(emitted.len(), LANES);
    let occupied = emitted
        .iter()
        .filter(|batch| batch.record_batch().num_rows() > 0)
        .count();
    assert_eq!(occupied, 1, "one row lands in one lane");
    for batch in &emitted {
        assert_eq!(
            batch.record_batch().schema().fields().len(),
            2,
            "an empty lane still carries the node's columns"
        );
    }
}

#[test]
fn every_row_lands_in_exactly_one_lane() {
    let keys = ["a", "b", "c", "d", "e", "f", "g", "h"];
    let (emitted, _) = emitter().emit(keyed(&keys)).expect("the scatter runs");
    let placed = lane_of(&emitted);
    assert_eq!(placed.len(), keys.len(), "no row lost and none doubled");
    let mut names: Vec<&str> = keys.to_vec();
    names.sort();
    assert_eq!(
        placed
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<&str>>(),
        names
    );
}

/// A join's two sides are scattered by two different emitters, and lane `p` of one has to
/// hold the rows that can match lane `p` of the other. That is only true if the same key
/// goes to the same lane every time it is seen.
#[test]
fn the_same_key_lands_in_the_same_lane_whatever_it_arrived_with() {
    let alone = emitter().emit(keyed(&["b"])).expect("the scatter runs").0;
    let crowded = emitter()
        .emit(keyed(&["a", "b", "c", "d", "e"]))
        .expect("the scatter runs")
        .0;
    let lane_for = |placed: Vec<(String, usize)>, key: &str| {
        placed
            .into_iter()
            .find(|(name, _)| name == key)
            .expect("the key was emitted")
            .1
    };
    assert_eq!(
        lane_for(lane_of(&alone), "b"),
        lane_for(lane_of(&crowded), "b"),
        "the lane is a property of the key, not of the batch it came in"
    );
}

#[test]
fn a_hash_key_past_the_inputs_columns_is_refused() {
    let node = GpuEmitPartitions::new(Given::of(&GROUPED), vec![7], LANES);
    let refused = match CpuEmitter::new(&node, LANES, &schema_of(&GROUPED).fields) {
        Err(refused) => refused,
        Ok(_) => panic!("there is no column 7"),
    };
    let message = format!("{refused}");
    assert!(
        message.contains("at 7") && message.contains("2 columns"),
        "the refusal has to name the key and the width: {message}"
    );
}
