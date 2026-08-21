//! The aggregate nodes. Neither carries a phase: each declares aggregators over its own
//! input and, where it finishes the aggregate, how each aggregate finishes.
//! `GpuAggregate` runs per batch; `GpuAggregateBatches` merges pre-aggregated batches and
//! emits at done.

use std::any::Any;

use super::super::aggregates::{AggCall, Merge, decomposition};
use super::super::error::PlanError;
use super::super::expr::{Expr, NamedExpr};
use super::super::layout::{BatchLayout, KeyDistribution, NodeKind, PartitionLayout, SortOrder};
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
    /// One expression per aggregate output column, and not per output column: a group key is
    /// not finalized and is not here, so this list is shorter than the node's output
    /// schema by the number of keys. The project that carries it emits the keys first, and
    /// `recipe::aggregate_writer::finalize_project` is the one place that rule lives.
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
        // Grouping re-keys the rows, so no input order survives — but a hash on columns the
        // group list re-emits does: those rows are still where the hash put them, at the
        // ordinals the keys now occupy.
        layout.sort_order = SortOrder::NotSpecified;
        layout.key_distribution = regrouped_key_distribution(&layout, &body);
        Self {
            kind: NodeKind::Intermediate { layout, schema },
            body,
            intermediate,
            input,
        }
    }
}

impl GpuAggregate {
    /// `[group keys…, state columns…]` — what the aggregators produce, and where the
    /// state annotations live. The output schema is the finalized one where this node
    /// finalizes, so a consumer of the state reads this instead.
    pub fn intermediate(&self) -> &Schema {
        &self.intermediate
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
        layout.key_distribution = regrouped_key_distribution(&layout, &body);
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

impl GpuAggregateBatches {
    /// The state this node merges into, before any finalize of its own — see
    /// [`GpuAggregate::intermediate`].
    pub fn intermediate(&self) -> &Schema {
        &self.intermediate
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
        let input_columns = input_schema(self.input.as_ref());
        self.body
            .validate("GpuAggregateBatches", &input_columns, &self.intermediate)?;
        check_merges_the_state_it_was_given(&self.body, &input_columns)?;
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

/// A merge reads the state annotation rather than re-deriving where the state is: for each
/// aggregate its input declares, the columns at the declared positions must be merged, and
/// merged by the aggregators that aggregate's decomposition names — a count merges by sum,
/// and a Welford triple merges as one call. An annotation nothing checks is a comment.
///
/// `ddof` is not checked here because nothing at a merge names it: it separates the sample
/// forms from the population ones and appears only in the finalize expression, which is
/// built from the same spec that wrote the annotation. Nor are the state column TYPES,
/// which no rule anywhere derives from what produces them — [#163](../../../llm-wiki/tickets.md).
fn check_merges_the_state_it_was_given(
    body: &AggregateBody,
    input: &Schema,
) -> Result<(), PlanError> {
    let reads = |call: &AggCall| -> Vec<u32> {
        call.args
            .iter()
            .filter_map(|arg| match arg {
                Expr::Column(reference) => Some(reference.index),
                _ => None,
            })
            .collect()
    };
    for state in &input.agg_state {
        let rule = decomposition(state.func);
        match rule.merge {
            Merge::PerColumn(funcs) => {
                if funcs.len() != state.positions.len() {
                    return Err(PlanError::Invalid(format!(
                        "GpuAggregateBatches: {} declares {} state columns and its \
                         decomposition merges {}",
                        state.output,
                        state.positions.len(),
                        funcs.len()
                    )));
                }
                for (func, position) in funcs.iter().zip(state.positions.iter()) {
                    let merged = body
                        .aggs
                        .iter()
                        .find(|call| reads(call) == vec![*position])
                        .ok_or_else(|| {
                            PlanError::Invalid(format!(
                                "GpuAggregateBatches: nothing merges @{position}, which its \
                                 input declares as {} state",
                                state.output
                            ))
                        })?;
                    if merged.func != *func {
                        return Err(PlanError::Invalid(format!(
                            "GpuAggregateBatches: @{position} is {} state and is merged by \
                             {} rather than {}",
                            state.output,
                            merged.func.tag(),
                            func.tag()
                        )));
                    }
                }
            }
            Merge::Combined(func) => {
                let merged = body
                    .aggs
                    .iter()
                    .find(|call| reads(call) == state.positions)
                    .ok_or_else(|| {
                        PlanError::Invalid(format!(
                            "GpuAggregateBatches: {} merges its {} state columns in one call, \
                             and no aggregator reads them together",
                            state.output,
                            state.positions.len()
                        ))
                    })?;
                if merged.func != func {
                    return Err(PlanError::Invalid(format!(
                        "GpuAggregateBatches: {} state is merged by {} rather than {}",
                        state.output,
                        merged.func.tag(),
                        func.tag()
                    )));
                }
            }
        }
    }
    Ok(())
}

/// The input's hash as the group list re-numbers it. A hashed column that is not grouped on
/// takes the claim with it: rows of one group would then be spread across lanes, which is
/// exactly what the co-location rule above exists to refuse. A grouping set drops it the
/// same way — it substitutes NULL for the keys it excludes, so the rolled-up rows are no
/// longer where the hash put them even though the column is still in the group list.
///
/// A mask is one per set in key order and so is as long as the group list, but a short one
/// reads as excluding: over-claiming co-location is what this function exists to avoid.
fn regrouped_key_distribution(layout: &PartitionLayout, body: &AggregateBody) -> KeyDistribution {
    let KeyDistribution::ByHash { hash_keys } = &layout.key_distribution else {
        return KeyDistribution::NotSpecified;
    };
    let regrouped: Option<Vec<u32>> = hash_keys
        .iter()
        .map(|key| {
            body.group_by
                .iter()
                .position(|expr| matches!(expr, Expr::Column(reference) if reference.index == *key))
                .filter(|position| {
                    !body
                        .grouping_sets
                        .iter()
                        .any(|mask| mask.get(*position).copied().unwrap_or(true))
                })
                .map(|position| position as u32)
        })
        .collect();
    match regrouped {
        Some(hash_keys) => KeyDistribution::ByHash { hash_keys },
        None => KeyDistribution::NotSpecified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(group_by: Vec<u32>, grouping_sets: Vec<Vec<bool>>) -> AggregateBody {
        AggregateBody {
            group_by: group_by
                .into_iter()
                .map(|index| Expr::column(index, "k"))
                .collect(),
            grouping_sets,
            null_exprs: Vec::new(),
            aggs: Vec::new(),
            finalize: None,
        }
    }

    fn hashed_on(keys: Vec<u32>) -> PartitionLayout {
        PartitionLayout {
            n: 4,
            key_distribution: KeyDistribution::ByHash { hash_keys: keys },
            sort_order: SortOrder::NotSpecified,
            batch_layout: BatchLayout::MultipleBatches,
        }
    }

    #[test]
    fn a_hash_on_a_regrouped_key_survives_at_its_new_ordinal() {
        let claim = regrouped_key_distribution(&hashed_on(vec![3]), &body(vec![7, 3], Vec::new()));
        assert_eq!(claim, KeyDistribution::ByHash { hash_keys: vec![1] });
    }

    #[test]
    fn a_hash_on_a_key_a_grouping_set_drops_does_not_survive() {
        // The rollup's second set substitutes NULL for key 0, so the rows it produces are
        // no longer in the lane the hash on that column put them in — and a finalizing
        // merge that believed the claim would answer per lane for a group spread over all
        // of them.
        let sets = vec![vec![false, false], vec![true, false]];
        let claim = regrouped_key_distribution(&hashed_on(vec![7]), &body(vec![7, 3], sets));
        assert_eq!(claim, KeyDistribution::NotSpecified);
    }

    #[test]
    fn a_mask_shorter_than_the_group_list_drops_the_hash_rather_than_keeping_it() {
        // Masks are as long as the group list by construction, so this is the default and
        // not a shape: a missing entry has to read as excluding, or a malformed body would
        // buy a co-location claim the rows do not have.
        let claim =
            regrouped_key_distribution(&hashed_on(vec![3]), &body(vec![7, 3], vec![vec![false]]));
        assert_eq!(claim, KeyDistribution::NotSpecified);
    }

    #[test]
    fn a_grouping_set_that_drops_some_other_key_leaves_the_hash_alone() {
        let sets = vec![vec![false, false], vec![false, true]];
        let claim = regrouped_key_distribution(&hashed_on(vec![7]), &body(vec![7, 3], sets));
        assert_eq!(claim, KeyDistribution::ByHash { hash_keys: vec![0] });
    }
}
