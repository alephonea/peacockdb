//! The injector's own cases: what a rebuild keeps, what a selection covers, and what an
//! edge or a join type refuses.
//!
//! None of these reads a dataset. They are here rather than in the end-to-end tier for
//! that reason — a case that needs no parquet runs in a second on any host, and one buried
//! in a tier that takes minutes is one nobody runs while iterating.

mod common;

use common::injection::{
    CAP, Candidate, Dimensions, Drain, Edge, Empties, Hash, Injection, PlannedMode, Rebatch, SEED,
    emitter_over_four_lanes, interior_edges, merge_over_sorted, rebatch_at, select,
};
use common::rebuild::{every_kind, fields_with_one_value, rebuild_tree};
use peacockdb_core::batch_partitioned::GpuNode;
use peacockdb_core::batch_partitioned::batch::Batch;
use peacockdb_core::batch_partitioned::plan_text::render_plan;
use peacockdb_core::batch_partitioned::validate::validate;
use std::collections::BTreeSet;

// ── the rewrite, before anything is rewritten ───────────────────────────────

/// A node rebuilt over its own children is the node it was, per arm and per field.
///
/// A rebuild that drops a field is a plan that differs from the one under test for a
/// reason nobody chose, and every answer after it is about a different query.
///
/// Compared as debug output rather than as plan text: `survivors`, `can_be_null` and an
/// aggregate's intermediate schema reach no plan line, so a rebuild could drop any of the
/// three and render identically. The rendering is asserted beside it, as what a golden
/// reads.
#[test]
fn a_node_rebuilt_over_its_own_children_is_the_node_it_was() {
    let fixtures = every_kind();
    let mut reached: BTreeSet<&'static str> = BTreeSet::new();
    for plan in &fixtures {
        names_of(plan.as_ref(), &mut reached);
        let rebuilt = rebuild_tree(plan.as_ref());
        assert_eq!(
            format!("{plan:#?}"),
            format!("{rebuilt:#?}"),
            "{} lost a field in the rebuild",
            plan.name()
        );
        assert_eq!(
            render_plan(plan.as_ref()),
            render_plan(rebuilt.as_ref()),
            "{} renders differently after a rebuild",
            plan.name()
        );
    }
    // The count is what makes a nineteenth kind red here as well as a compile error in the
    // match: an arm nothing builds is an arm nothing proves.
    assert_eq!(
        reached.len(),
        18,
        "the fixtures reach {} of the eighteen kinds: {reached:?}",
        reached.len()
    );
    // And the level below that: a field every fixture gives the same value is a field
    // whose loss no fixture above could detect, so it is named the day it is added rather
    // than the day something depends on it.
    let unvaried = fields_with_one_value(&fixtures);
    assert!(
        unvaried.is_empty(),
        "no fixture varies these, so the rebuild could drop them unseen: {unvaried:#?}"
    );

    fn names_of(node: &dyn GpuNode, into: &mut BTreeSet<&'static str>) {
        into.insert(node.name());
        for child in node.children() {
            names_of(child, into);
        }
    }
}

// ── the selection, which is a claim about coverage ──────────────────────────

/// What the selector chose is asserted rather than trusted: a cap that took the first
/// thirty candidates would run thirty shapes and cover whichever dimensions happened to
/// sort early.
///
/// The requirements are written down here rather than read off `Dimensions`, which is what
/// makes the case able to go red — a requirement derived from the settings cannot catch a
/// settings list that lost a value. The thinned settings at the end are that red case,
/// kept rather than described.
#[test]
fn the_selection_covers_every_mode_and_every_boundary() {
    // Two lanes and a scatter at the second mode and neither at the first, which is the
    // shape that makes half the settings unreachable at one of them.
    let modes = [
        PlannedMode {
            name: "one-lane",
            lanes: 1,
            shuffles: false,
            owes_probe_when_empty: false,
        },
        PlannedMode {
            name: "four-lane",
            lanes: 4,
            shuffles: true,
            owes_probe_when_empty: false,
        },
    ];
    let chosen = select(&modes, &Dimensions::default(), CAP, SEED);
    assert!(
        chosen.len() <= CAP,
        "{} runs against a cap of {CAP}",
        chosen.len()
    );
    let carries = |what: &str, holds: &dyn Fn(&Candidate) -> bool| {
        assert!(
            chosen.iter().any(holds),
            "the selection carries no {what}: {:?}",
            chosen
                .iter()
                .map(|candidate| candidate.label(&modes))
                .collect::<Vec<String>>()
        );
    };
    for (index, mode) in modes.iter().enumerate() {
        carries(mode.name, &|candidate| candidate.mode == index);
    }
    carries("plan as planned", &|candidate| {
        candidate.injection == Injection::NONE
    });
    // Named, because the red case at the end is this same predicate over a thinned
    // settings list: what makes the assertion above load-bearing is that it fails there.
    let above_sources = |candidate: &Candidate| candidate.injection.rebatch == Rebatch::AboveSources;
    carries("rebatcher above the sources", &above_sources);
    carries("rebatcher at an interior edge", &|candidate| {
        candidate.injection.rebatch == Rebatch::AboveInterior
    });
    carries("drained lane", &|candidate| {
        candidate.injection.drain == Drain::FirstLane
    });
    carries("empty batches", &|candidate| {
        candidate.injection.empties == Empties::Sometimes(50)
    });
    carries("degenerate hash", &|candidate| {
        candidate.injection.hash == Hash::Degenerate
    });

    // A drain needs a lane to move the rows to and a degenerate hash needs a scatter, so
    // both belong to the four-lane mode alone — a candidate carrying one at the other
    // would be a run whose label says more than it did.
    for candidate in &chosen {
        if candidate.injection.drain != Drain::None || candidate.injection.hash != Hash::AsPlanned {
            assert_eq!(
                candidate.mode,
                1,
                "{} is a setting the mode cannot carry",
                candidate.label(&modes)
            );
        }
    }

    // The cover is at most one candidate per requirement — five modes and nine dimension
    // values — so a cap that low still carries everything asserted above. That is what
    // lets a run count be cut without cutting what is proved.
    let smallest = select(&modes, &Dimensions::default(), 14, SEED);
    assert!(smallest.len() <= 14, "{} runs at a cap of 14", smallest.len());
    for candidate in &chosen {
        let carried = |holds: &dyn Fn(&Candidate) -> bool| {
            !holds(candidate) || smallest.iter().any(holds)
        };
        assert!(
            carried(&|other| other.mode == candidate.mode)
                && carried(&|other| other.injection.rebatch == candidate.injection.rebatch)
                && carried(&|other| other.injection.drain == candidate.injection.drain)
                && carried(&|other| other.injection.empties == candidate.injection.empties)
                && carried(&|other| other.injection.hash == candidate.injection.hash),
            "a cap of 14 dropped what {} carries",
            candidate.label(&modes)
        );
    }

    // Seeded rather than sampled: a failure at one shape is reproducible only if the same
    // shapes are chosen next time.
    assert_eq!(
        chosen,
        select(&modes, &Dimensions::default(), CAP, SEED),
        "two selections at one seed differ"
    );

    // The red case for every assertion above: settings that lost a value cannot cover it,
    // and the cover check is what says so.
    let thinner = Dimensions {
        rebatch: vec![Rebatch::None],
        ..Dimensions::default()
    };
    assert!(
        !select(&modes, &thinner, CAP, SEED).iter().any(above_sources),
        "a settings list without a rebatcher still produced one, so the cover assertion \
         above would hold over a set that had lost the dimension"
    );
}

/// The edge a rebatcher may not go on, refused by the engine rather than by this file.
///
/// A coalesce clears its input's sort order, so an edge under a k-way merge is one the
/// plan cannot carry — and an injector that skipped such edges quietly would read as
/// having covered them. `interior_edges` names them, and this is the demonstration that
/// what it names is what the driver refuses.
#[test]
fn a_rebatcher_under_a_merge_is_refused_and_names_the_order() {
    let plan = merge_over_sorted();
    let edges = interior_edges(plan.as_ref());
    let refused: Vec<&Edge> = edges.iter().filter(|edge| edge.refused.is_some()).collect();
    // Every edge of this plan declares an order — the sort made one, the accumulator
    // merged its batches and the merge kept it — and only the edge under the merge is
    // refused by the engine. The others are refused by the rule: a node that carries rows
    // past an order would go green while changing which rows a limit above it takes.
    assert_eq!(
        refused.iter().map(|edge| edge.node).collect::<Vec<&str>>(),
        vec![
            "GpuMergeSortedPartitions",
            "GpuAccumulateBatchesAndSort",
            "GpuSort"
        ],
        "an edge declaring an order is not a rebatcher's"
    );
    let under_the_merge = refused
        .iter()
        .find(|edge| edge.node == "GpuAccumulateBatchesAndSort")
        .expect("the accumulator's edge");

    let injected = rebatch_at(plan.as_ref(), under_the_merge.child);
    match validate(injected.as_ref()) {
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("GpuMergeSortedPartitions") && message.contains("sorted"),
                "the refusal names neither the node nor the order: {message}"
            );
        }
        Ok(()) => panic!("a coalesce under a merge is not a plan, and validation took it"),
    }
    // And the plan it was made from is one, so what was refused is the insertion rather
    // than the fixture.
    validate(plan.as_ref()).expect("the plan the edge was taken from");
}

/// The fourth dimension, which the trace cannot show: a lane count is in the calls and a
/// row's lane is not.
///
/// Every key into lane 0 is a legal hash — a shuffle's contract is co-location, and
/// nothing above a scatter may depend on how evenly the lanes were loaded — so what has to
/// be true is that the batch count is still the lane count and that the rows are all in
/// one of them.
#[test]
fn the_degenerate_hash_puts_every_row_in_one_lane_and_still_emits_them_all() {
    let rows: Vec<i64> = (0..40).collect();
    let spread = emitter_over_four_lanes(&rows, Hash::AsPlanned);
    let degenerate = emitter_over_four_lanes(&rows, Hash::Degenerate);

    assert_eq!(spread.len(), 4, "a scatter emits one batch per lane");
    assert_eq!(degenerate.len(), 4, "and so does a degenerate one");
    assert!(
        spread.iter().filter(|batch| batch.num_rows() > 0).count() > 1,
        "the planned hash left every row in one lane, so the degenerate one proves nothing"
    );
    assert_eq!(
        degenerate[0].num_rows(),
        rows.len(),
        "lane 0 did not take every row"
    );
    assert!(
        degenerate[1..].iter().all(|batch| batch.num_rows() == 0),
        "a lane other than 0 was given rows"
    );
}
