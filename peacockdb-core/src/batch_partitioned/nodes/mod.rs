//! The plan nodes, grouped by family, and the one downcast registry over them.

pub mod accumulators;
pub mod aggregate;
pub mod exec_ops;
pub mod join;
pub mod partition_ops;
pub mod source;
pub mod union;
pub mod unload;

use super::error::PlanError;
use super::expr::Expr;
use super::layout::{ColumnOrder, KeyDistribution, PartitionLayout, SortOrder};
use super::node::GpuNode;
use super::schema::Schema;

pub use accumulators::{GpuAccumulateBatchesAndSort, GpuCoalesceAllBatches, GpuLimit};
pub use aggregate::{AggregateBody, GpuAggregate, GpuAggregateBatches};
pub use exec_ops::{GpuFilter, GpuProject, GpuSort};
pub use join::{GpuCrossJoin, GpuJoin, GpuNestedLoopJoin};
pub use partition_ops::{GpuEmitPartitions, GpuMergePartitions, GpuMergeSortedPartitions};
pub use source::GpuLoadParquet;
pub use union::{GpuInterleave, GpuUnion};
pub use unload::GpuUnload;

/// Every node kind, as a borrow of the concrete node. Adding a node is one line here and
/// an exhaustive match everywhere it is consumed — the renderer, a backend's executor
/// match, the serializer — rather than a downcast chain per consumer.
pub enum NodeRef<'a> {
    LoadParquet(&'a GpuLoadParquet),
    Filter(&'a GpuFilter),
    Project(&'a GpuProject),
    Sort(&'a GpuSort),
    CoalesceAllBatches(&'a GpuCoalesceAllBatches),
    AccumulateBatchesAndSort(&'a GpuAccumulateBatchesAndSort),
    Limit(&'a GpuLimit),
    Aggregate(&'a GpuAggregate),
    AggregateBatches(&'a GpuAggregateBatches),
    Join(&'a GpuJoin),
    CrossJoin(&'a GpuCrossJoin),
    NestedLoopJoin(&'a GpuNestedLoopJoin),
    MergePartitions(&'a GpuMergePartitions),
    EmitPartitions(&'a GpuEmitPartitions),
    MergeSortedPartitions(&'a GpuMergeSortedPartitions),
    Union(&'a GpuUnion),
    Interleave(&'a GpuInterleave),
    Unload(&'a GpuUnload),
}

pub fn as_node_ref(node: &dyn GpuNode) -> NodeRef<'_> {
    node_ref_of(node.as_any())
}

/// The registry without its panic, for the generic validation pass: a node it does not
/// know is a hand-built one under test, which declares none of the parameters that pass
/// reads. Every consumer of those parameters goes through `as_node_ref` instead.
pub(crate) fn try_as_node_ref(node: &dyn GpuNode) -> Option<NodeRef<'_>> {
    try_node_ref_of(node.as_any())
}

/// Off the erased value rather than the node, so a node can reach the registry from a
/// default trait method — where `Self` is not yet known to be sized.
fn node_ref_of(any: &dyn std::any::Any) -> NodeRef<'_> {
    try_node_ref_of(any).expect("a plan node outside the registry reached a consumer of it")
}

fn try_node_ref_of(any: &dyn std::any::Any) -> Option<NodeRef<'_>> {
    if let Some(n) = any.downcast_ref::<GpuLoadParquet>() {
        Some(NodeRef::LoadParquet(n))
    } else if let Some(n) = any.downcast_ref::<GpuFilter>() {
        Some(NodeRef::Filter(n))
    } else if let Some(n) = any.downcast_ref::<GpuProject>() {
        Some(NodeRef::Project(n))
    } else if let Some(n) = any.downcast_ref::<GpuSort>() {
        Some(NodeRef::Sort(n))
    } else if let Some(n) = any.downcast_ref::<GpuCoalesceAllBatches>() {
        Some(NodeRef::CoalesceAllBatches(n))
    } else if let Some(n) = any.downcast_ref::<GpuAccumulateBatchesAndSort>() {
        Some(NodeRef::AccumulateBatchesAndSort(n))
    } else if let Some(n) = any.downcast_ref::<GpuLimit>() {
        Some(NodeRef::Limit(n))
    } else if let Some(n) = any.downcast_ref::<GpuAggregate>() {
        Some(NodeRef::Aggregate(n))
    } else if let Some(n) = any.downcast_ref::<GpuAggregateBatches>() {
        Some(NodeRef::AggregateBatches(n))
    } else if let Some(n) = any.downcast_ref::<GpuJoin>() {
        Some(NodeRef::Join(n))
    } else if let Some(n) = any.downcast_ref::<GpuCrossJoin>() {
        Some(NodeRef::CrossJoin(n))
    } else if let Some(n) = any.downcast_ref::<GpuNestedLoopJoin>() {
        Some(NodeRef::NestedLoopJoin(n))
    } else if let Some(n) = any.downcast_ref::<GpuMergePartitions>() {
        Some(NodeRef::MergePartitions(n))
    } else if let Some(n) = any.downcast_ref::<GpuEmitPartitions>() {
        Some(NodeRef::EmitPartitions(n))
    } else if let Some(n) = any.downcast_ref::<GpuMergeSortedPartitions>() {
        Some(NodeRef::MergeSortedPartitions(n))
    } else if let Some(n) = any.downcast_ref::<GpuUnion>() {
        Some(NodeRef::Union(n))
    } else if let Some(n) = any.downcast_ref::<GpuInterleave>() {
        Some(NodeRef::Interleave(n))
    } else {
        any.downcast_ref::<GpuUnload>().map(NodeRef::Unload)
    }
}

/// Which executor trait drives a node. Read before an executor exists, since runnability
/// asks for it, so it is derived from the node rather than from what a backend returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorCategory {
    Source,
    Exec,
    BatchAccumulator,
    PartitionAccumulator,
    PartitionEmitter,
    Join,
    BatchForwarder,
    Unload,
}

impl ExecutorCategory {
    /// One instance per (node, lane); the rest are one per node, being the cross-lane
    /// points. A forwarder has no executor at all — the driver owns its rotation.
    pub fn is_lane_scoped(&self) -> bool {
        matches!(
            self,
            Self::Source | Self::Exec | Self::BatchAccumulator | Self::Join | Self::Unload
        )
    }
}

/// Off the registry rather than beside it: the node set and the category set are one
/// mapping, and a second source of truth for it is a second thing to keep true.
pub fn category_of(node: &dyn GpuNode) -> ExecutorCategory {
    match as_node_ref(node) {
        NodeRef::LoadParquet(_) => ExecutorCategory::Source,
        NodeRef::Filter(_) | NodeRef::Project(_) | NodeRef::Sort(_) | NodeRef::Aggregate(_) => {
            ExecutorCategory::Exec
        }
        NodeRef::CoalesceAllBatches(_)
        | NodeRef::AccumulateBatchesAndSort(_)
        | NodeRef::AggregateBatches(_)
        | NodeRef::Limit(_) => ExecutorCategory::BatchAccumulator,
        NodeRef::MergeSortedPartitions(_) => ExecutorCategory::PartitionAccumulator,
        NodeRef::EmitPartitions(_) => ExecutorCategory::PartitionEmitter,
        NodeRef::Join(_) | NodeRef::CrossJoin(_) | NodeRef::NestedLoopJoin(_) => {
            ExecutorCategory::Join
        }
        NodeRef::MergePartitions(_) | NodeRef::Union(_) | NodeRef::Interleave(_) => {
            ExecutorCategory::BatchForwarder
        }
        NodeRef::Unload(_) => ExecutorCategory::Unload,
    }
}

/// What a node is called, in a plan line and in the validation message that names it.
/// No `Exec` suffix: these are not DataFusion nodes, and after the wire-format rename a
/// line from either family says which mode produced it without a caption.
pub(crate) fn node_name(any: &dyn std::any::Any) -> &'static str {
    match node_ref_of(any) {
        NodeRef::LoadParquet(_) => "GpuLoadParquet",
        NodeRef::Filter(_) => "GpuFilter",
        NodeRef::Project(_) => "GpuProject",
        NodeRef::Sort(_) => "GpuSort",
        NodeRef::CoalesceAllBatches(_) => "GpuCoalesceAllBatches",
        NodeRef::AccumulateBatchesAndSort(_) => "GpuAccumulateBatchesAndSort",
        NodeRef::Limit(_) => "GpuLimit",
        NodeRef::Aggregate(_) => "GpuAggregate",
        NodeRef::AggregateBatches(_) => "GpuAggregateBatches",
        NodeRef::Join(_) => "GpuJoin",
        NodeRef::CrossJoin(_) => "GpuCrossJoin",
        NodeRef::NestedLoopJoin(_) => "GpuNestedLoopJoin",
        NodeRef::MergePartitions(_) => "GpuMergePartitions",
        NodeRef::EmitPartitions(_) => "GpuEmitPartitions",
        NodeRef::MergeSortedPartitions(_) => "GpuMergeSortedPartitions",
        NodeRef::Union(_) => "GpuUnion",
        NodeRef::Interleave(_) => "GpuInterleave",
        NodeRef::Unload(_) => "GpuUnload",
    }
}

/// The layout a node inherits from its input. A sink is the root, so it is never one.
pub(crate) fn input_layout(input: &dyn GpuNode) -> PartitionLayout {
    input
        .kind()
        .layout()
        .expect("a sink cannot be an input")
        .clone()
}

pub(crate) fn input_schema(input: &dyn GpuNode) -> Schema {
    input
        .kind()
        .schema()
        .expect("a sink cannot be an input")
        .clone()
}

/// Every column reference must be in range of the schema it reads AND carry the name of
/// the field at that position. The name is redundant on purpose: an ordinal read in the
/// wrong order is otherwise invisible until the final result (#135), and the layer
/// rebases ordinals at every node it inserts, so a stale reference is the likely slip.
pub(crate) fn check_column_refs(
    expr: &Expr,
    against: &Schema,
    site: &str,
) -> Result<(), PlanError> {
    match expr {
        Expr::Column(reference) => {
            let field = against
                .fields
                .fields()
                .get(reference.index as usize)
                .ok_or_else(|| {
                    PlanError::Invalid(format!(
                        "{site}: column {}@{} is past the {} columns its input has",
                        reference.name,
                        reference.index,
                        against.fields.fields().len()
                    ))
                })?;
            if field.name() != &reference.name {
                return Err(PlanError::Invalid(format!(
                    "{site}: column {}@{} reads {} at that position",
                    reference.name,
                    reference.index,
                    field.name()
                )));
            }
            Ok(())
        }
        Expr::Literal(_) => Ok(()),
        Expr::Binary { left, right, .. } => {
            check_column_refs(left, against, site)?;
            check_column_refs(right, against, site)
        }
        Expr::Unary { arg, .. } => check_column_refs(arg, against, site),
        Expr::Cast { expr, .. } => check_column_refs(expr, against, site),
        Expr::Like { expr, pattern, .. } => {
            check_column_refs(expr, against, site)?;
            check_column_refs(pattern, against, site)
        }
        Expr::Case {
            comparand,
            when_then,
            else_expr,
        } => {
            for part in comparand.iter().chain(else_expr.iter()) {
                check_column_refs(part, against, site)?;
            }
            for (when, then) in when_then {
                check_column_refs(when, against, site)?;
                check_column_refs(then, against, site)?;
            }
            Ok(())
        }
        Expr::ScalarFunction { args, .. } => {
            for arg in args {
                check_column_refs(arg, against, site)?;
            }
            Ok(())
        }
    }
}

/// A k-way merge reads one row per input at a time and takes the smallest, so the order it
/// merges on has to be the order its input's batches already carry: merging on a key the
/// batches are not sorted by emits rows in no order at all — and then declares them sorted,
/// which is the claim a top-N above it acts on.
///
/// A prefix is enough: batches sorted on `[a, b]` merged on `[a]` come out ordered by `a`.
pub(crate) fn check_merge_keys(
    node: &str,
    keys: &[ColumnOrder],
    input: &PartitionLayout,
) -> Result<(), PlanError> {
    let SortOrder::BatchSorted { columns } = &input.sort_order else {
        return Err(PlanError::Invalid(format!(
            "{node}: a merge needs sorted batches — the planner puts a GpuSort below it"
        )));
    };
    for (position, key) in keys.iter().enumerate() {
        match columns.get(position) {
            Some(sorted) if sorted == key => {}
            Some(sorted) => {
                return Err(PlanError::Invalid(format!(
                    "{node}: it merges on @{} {} at position {position} where its input's \
                     batches are sorted on @{} {}",
                    key.column,
                    direction(key),
                    sorted.column,
                    direction(sorted)
                )));
            }
            None => {
                return Err(PlanError::Invalid(format!(
                    "{node}: it merges on {} keys and its input's batches are sorted on {}",
                    keys.len(),
                    columns.len()
                )));
            }
        }
    }
    Ok(())
}

fn direction(order: &ColumnOrder) -> String {
    format!(
        "{} {}",
        if order.ascending { "asc" } else { "desc" },
        if order.nulls_first {
            "nulls first"
        } else {
            "nulls last"
        }
    )
}

/// Carry a layout's key distribution and sort order through a projection, keeping only
/// what a bare column reference re-exposes: a projected-away or computed column takes
/// its property with it, and a declaration that outlived its column would be a lie the
/// nodes above it act on.
pub(crate) fn rebase_through_projection(
    layout: &PartitionLayout,
    projected: &[Expr],
) -> PartitionLayout {
    let new_index = |old: u32| -> Option<u32> {
        projected
            .iter()
            .position(|expr| match expr {
                Expr::Column(reference) => reference.index == old,
                _ => false,
            })
            .map(|position| position as u32)
    };

    let key_distribution = match &layout.key_distribution {
        KeyDistribution::NotSpecified => KeyDistribution::NotSpecified,
        KeyDistribution::ByHash { hash_keys } => {
            match hash_keys
                .iter()
                .map(|k| new_index(*k))
                .collect::<Option<Vec<_>>>()
            {
                Some(hash_keys) => KeyDistribution::ByHash { hash_keys },
                None => KeyDistribution::NotSpecified,
            }
        }
    };

    let sort_order = match &layout.sort_order {
        SortOrder::NotSpecified => SortOrder::NotSpecified,
        SortOrder::BatchSorted { columns } => {
            let mapped: Option<Vec<_>> = columns
                .iter()
                .map(|order| {
                    new_index(order.column)
                        .map(|column| super::layout::ColumnOrder { column, ..*order })
                })
                .collect();
            // A prefix of the keys would still hold, but a sort key that vanished mid-list
            // leaves an order nothing downstream can name.
            mapped
                .map(SortOrder::batch_sorted)
                .unwrap_or(SortOrder::NotSpecified)
        }
    };

    PartitionLayout {
        key_distribution,
        sort_order,
        ..layout.clone()
    }
}

#[cfg(test)]
mod tests;
