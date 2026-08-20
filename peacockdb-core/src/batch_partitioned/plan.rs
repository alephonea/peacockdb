//! Planning a query in this mode: DataFusion's physical plan in, a node tree and its
//! memory model out.
//!
//! Two passes, because the two halves define each other — a batch size needs the tree it
//! flows through, and the tree's mapping needs a batch size. The first pass assumes one
//! number for every source and is used only to derive the real ones; the second is the
//! plan. The tree's shape does not move between them: lane counts come from row counts
//! and pushed-down limits, never from the batch size.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

use super::error::PlanError;
use super::estimator::{MemoryModel, estimate};
use super::node::GpuNode;
use super::nulls::refuse_null_unsafe_joins;
use super::partitioner::Batching;
use super::translate::Translator;

/// The planner inputs a mode fixes: how many lanes to aim for, whether a lane holds more
/// than one batch, the budget the estimator divides, and the byte count below which a
/// source stops being worth splitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanKnobs {
    pub target_partitions: usize,
    pub sizing: BatchSizing,
    /// Read only by [`BatchSizing::Budgeted`]; the other two forms reproduce a plan from
    /// the data alone.
    pub budget: u64,
    pub small_table_bytes: u64,
}

/// What a mode asks of the partitioner. The planner's half of [`Batching`]: `Budgeted` has
/// no number until the estimator solves for one, which is why it is a separate word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSizing {
    OneBatchPerLane,
    OneBatchPerRowGroup,
    Budgeted,
}

pub fn plan_batch_partitioned(
    root: &Arc<dyn ExecutionPlan>,
    knobs: PlanKnobs,
) -> Result<(Box<dyn GpuNode>, MemoryModel), PlanError> {
    let translator = |targets: Vec<u64>| {
        Translator::new(knobs.target_partitions, seed_batching(knobs))
            .with_small_table_bytes(knobs.small_table_bytes)
            .with_source_targets(targets)
    };

    let first = translator(Vec::new()).translate(root)?;
    if knobs.sizing != BatchSizing::Budgeted {
        let model = estimate(first.as_ref(), knobs.budget)?;
        validate(first.as_ref())?;
        refuse_null_unsafe_joins(first.as_ref())?;
        return Ok((first, model));
    }

    // Post-order sequence rises left to right across the sources, which is the order
    // translation reaches them, so sorting by it is the mapping between the two passes.
    let derived = estimate(first.as_ref(), knobs.budget)?;
    let targets = derived
        .sources
        .iter()
        .map(|source| source.target_batch_bytes)
        .collect();
    let tree = translator(targets).translate(root)?;
    let model = estimate(tree.as_ref(), knobs.budget)?;
    validate(tree.as_ref())?;
    refuse_null_unsafe_joins(tree.as_ref())?;
    Ok((tree, model))
}

/// Every node against what it requires of its children, post-order so a child's complaint
/// comes before its parent's. Nothing else calls this: a guard that only ever sees inputs
/// written by hand cannot fail on a plan, and a plan is what it exists to judge.
fn validate(root: &dyn GpuNode) -> Result<(), PlanError> {
    for child in root.children() {
        validate(child)?;
    }
    root.validate_schemas_and_partitions()
}

/// What the first pass assumes. Two of the three forms are the plan already; only the
/// budgeted one needs a number to start from, and the smallest the mapping can express is
/// the one that assumes least.
fn seed_batching(knobs: PlanKnobs) -> Batching {
    match knobs.sizing {
        BatchSizing::OneBatchPerLane => Batching::Off,
        BatchSizing::OneBatchPerRowGroup => Batching::PerRowGroup,
        BatchSizing::Budgeted => Batching::Sized {
            target_batch_bytes: super::estimator::MIN_TARGET_BATCH_BYTES as usize,
        },
    }
}
