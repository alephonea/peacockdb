//! The five batch-partitioned modes, and the knobs a run at one of them takes.
//!
//! One table, because three tiers plan the same five shapes: the plan goldens, the
//! end-to-end tier and the corpus tiers. Three copies of it would let one of them acquire
//! a sixth mode, or a different budget, without anything going red.

use peacockdb_core::batch_partitioned::plan::{BatchSizing, PlanKnobs};
use peacockdb_core::config::MemoryLimit;

/// The tier every mode is planned at. The plan goldens are written here, so a failure
/// anywhere reads against a committed plan rather than a shape nothing records.
pub const BUDGET: u64 = MemoryLimit::Mini.bytes() as u64;

/// Below this a table is small enough to broadcast rather than shuffle.
pub const SMALL_TABLE_BYTES: u64 = 5 * 1024 * 1024;

/// One planning mode: what the goldens call it, and the two knobs that make it distinct.
pub struct BpMode {
    /// The golden's spelling, `bp-tp4-sized`.
    pub name: &'static str,
    pub target_partitions: usize,
    pub sizing: BatchSizing,
}

impl BpMode {
    /// The macro's spelling of the same mode, `bp_tp4_sized` — one derivation rather than
    /// a second field, so the two cannot disagree.
    pub fn ident(&self) -> String {
        self.name.replace('-', "_")
    }

    pub fn knobs(&self) -> PlanKnobs {
        PlanKnobs {
            target_partitions: self.target_partitions,
            sizing: self.sizing,
            budget: BUDGET,
            small_table_bytes: SMALL_TABLE_BYTES,
        }
    }
}

/// The five, in the fixed sequence the widget and the `.result.txt` authority both read:
/// the last enabled one wins in each. One lane and one batch is the degenerate end,
/// row-group granularity is the finest the mapping expresses, and the sized mode is the
/// only one a budget moves.
pub const BP_MODES: [BpMode; 5] = [
    BpMode {
        name: "bp-tp1-single",
        target_partitions: 1,
        sizing: BatchSizing::OneBatchPerLane,
    },
    BpMode {
        name: "bp-tp1-rowgroup",
        target_partitions: 1,
        sizing: BatchSizing::OneBatchPerRowGroup,
    },
    BpMode {
        name: "bp-tp4-single",
        target_partitions: 4,
        sizing: BatchSizing::OneBatchPerLane,
    },
    BpMode {
        name: "bp-tp4-rowgroup",
        target_partitions: 4,
        sizing: BatchSizing::OneBatchPerRowGroup,
    },
    BpMode {
        name: "bp-tp4-sized",
        target_partitions: 4,
        sizing: BatchSizing::Budgeted,
    },
];

/// The mode a macro's ident names. Exhaustive over the table: an unlisted ident panics
/// naming the set, rather than being routed to whichever mode a prefix reached first.
pub fn mode_named(ident: &str) -> &'static BpMode {
    BP_MODES
        .iter()
        .find(|mode| mode.ident() == ident)
        .unwrap_or_else(|| {
            let known: Vec<String> = BP_MODES.iter().map(BpMode::ident).collect();
            panic!("unknown batch-partitioned mode '{ident}' (expected one of {known:?})")
        })
}
