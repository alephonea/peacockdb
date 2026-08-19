//! The aggregate nodes. Neither carries a phase: each declares aggregators over its own
//! input and, where it finishes the aggregate, one finalize expression per output column.
//! `GpuAggregate` runs per batch; `GpuAggregateBatches` merges pre-aggregated batches and
//! emits at done.

use std::any::Any;

use super::super::aggregates::AggCall;
use super::super::error::PlanError;
use super::super::expr::{Expr, NamedExpr};
use super::super::layout::{BatchLayout, KeyDistribution, NodeKind, SortOrder};
use super::super::node::GpuNode;
use super::super::schema::Schema;
use super::{check_column_refs, input_layout, input_schema};

/// The aggregators and the optional `final` list every aggregate node carries. A node
/// with no `final` emits its state; one with a `final` emits the finalized columns, and
/// that is the only thing distinguishing the positions — the single-node shortcut is
/// init aggregators and finalize expressions on the same node.
#[derive(Debug)]
pub struct AggregateBody {
    pub group_by: Vec<Expr>,
    /// One mask per grouping set, in key order — true where that key is NULL in that set.
    /// Empty unless this node expands grouping sets, which only an init node does: it
    /// emits `__grouping_id` as an ordinary column and every node above groups on the
    /// keys plus that column.
    pub grouping_sets: Vec<Vec<bool>>,
    /// The NULL substituted for each key a set excludes, in key order.
    pub null_exprs: Vec<Expr>,
    pub aggs: Vec<AggCall>,
    pub finalize: Option<Vec<NamedExpr>>,
}

impl AggregateBody {
    /// References inside `aggs` index the node's input; references inside `finalize`
    /// index the node's own intermediate table, `[group keys…, state columns…]`.
    fn validate(&self, node: &str, input: &Schema, intermediate: &Schema) -> Result<(), PlanError> {
        for key in &self.group_by {
            check_column_refs(key, input, node)?;
        }
        for call in &self.aggs {
            for arg in &call.args {
                check_column_refs(arg, input, node)?;
            }
        }
        for column in self.finalize.iter().flatten() {
            check_column_refs(&column.expr, intermediate, node)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct GpuAggregate {
    kind: NodeKind,
    pub body: AggregateBody,
    intermediate: Schema,
    input: Box<dyn GpuNode>,
}

impl GpuAggregate {
    /// `intermediate` is `[group keys…, state columns…]` — what the aggregators produce,
    /// and what a finalize expression reads. It is also the output schema where there is
    /// no finalize.
    pub fn new(
        input: Box<dyn GpuNode>,
        body: AggregateBody,
        intermediate: Schema,
        schema: Schema,
    ) -> Self {
        let mut layout = input_layout(input.as_ref());
        // Grouping re-keys the rows: no input order survives, and a hash the input
        // carried is about columns this node does not re-emit unchanged.
        layout.sort_order = SortOrder::NotSpecified;
        layout.key_distribution = KeyDistribution::NotSpecified;
        Self {
            kind: NodeKind::Intermediate { layout, schema },
            body,
            intermediate,
            input,
        }
    }
}

impl GpuNode for GpuAggregate {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        vec![self.input.as_ref()]
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        self.body.validate(
            "GpuAggregate",
            &input_schema(self.input.as_ref()),
            &self.intermediate,
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
pub struct GpuAggregateBatches {
    kind: NodeKind,
    pub body: AggregateBody,
    intermediate: Schema,
    input: Box<dyn GpuNode>,
}

impl GpuAggregateBatches {
    pub fn new(
        input: Box<dyn GpuNode>,
        body: AggregateBody,
        intermediate: Schema,
        schema: Schema,
    ) -> Self {
        let mut layout = input_layout(input.as_ref());
        layout.sort_order = SortOrder::NotSpecified;
        layout.key_distribution = KeyDistribution::NotSpecified;
        // It emits everything it merged, once, at done.
        layout.batch_layout = BatchLayout::SingleBatch;
        Self {
            kind: NodeKind::Intermediate { layout, schema },
            body,
            intermediate,
            input,
        }
    }
}

impl GpuNode for GpuAggregateBatches {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        vec![self.input.as_ref()]
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        let input = input_layout(self.input.as_ref());
        self.body.validate(
            "GpuAggregateBatches",
            &input_schema(self.input.as_ref()),
            &self.intermediate,
        )?;
        // A finalizing merge answers for whole groups, so every row of a group has to be
        // in this lane: either there is one lane, or the shuffle keyed on a subset of the
        // columns being grouped. Subset, not equality — a grouping-set rollup hashes on
        // the keys while grouping on keys plus __grouping_id.
        if self.body.finalize.is_some() && input.n > 1 {
            let grouped: Vec<u32> = self
                .body
                .group_by
                .iter()
                .filter_map(|key| match key {
                    Expr::Column(reference) => Some(reference.index),
                    _ => None,
                })
                .collect();
            if !input.key_distribution.is_subset_of(&grouped) {
                return Err(PlanError::Invalid(
                    "GpuAggregateBatches: a finalizing merge over several lanes needs its \
                     input hashed on a subset of its group columns — the planner inserts \
                     GpuMergePartitions + GpuEmitPartitions below it"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
