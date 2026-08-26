//! The aggregate nodes. Neither carries a phase: each declares aggregators over its own
//! input and, where it finishes the aggregate, how each aggregate finishes.
//! `GpuAggregate` runs per batch; `GpuAggregateBatches` merges pre-aggregated batches and
//! emits at done.

use std::any::Any;

use super::super::aggregates::{AggCall, AggFunc, Merge, PlanAgg, decomposition};
use super::super::error::PlanError;
use super::super::expr::{Expr, NamedExpr};
use super::super::layout::{BatchLayout, KeyDistribution, NodeKind, PartitionLayout, SortOrder};
use super::super::node::GpuNode;
use super::super::schema::{AggStateColumns, Schema};
use super::{check_column_refs, input_layout, input_schema};

/// One aggregator as both engines name it: the SQL name the executor knows, the call whose
/// arguments it reads, and the output column it fills.
///
/// Order is load-bearing and not cosmetic. A state-shaped input is read positionally, by a
/// cursor walking each aggregate's state width, so an aggregate listed out of order reads
/// another one's columns. Everything but the Welford triple is one of ours to one SQL name;
/// the triple folds into the `stddev`/`var` it decomposes, at the position of the first of
/// its three, because neither engine has an `m2` of its own.
pub struct StateFunc<'a> {
    pub name: &'static str,
    pub call: &'a AggCall,
    pub alias: String,
    /// Whether this one is a folded triple, which is what makes its state three columns
    /// wide rather than one.
    pub welford: bool,
}

/// The aggregators a body declares over a state schema, in the order their state columns
/// appear. One rule, read by the recipe writer for the wire and by the CPU backend for
/// DataFusion — a second copy of it would be a second answer to which column is whose.
pub fn state_funcs<'a>(
    body: &'a AggregateBody,
    state: &'a Schema,
) -> Result<Vec<StateFunc<'a>>, PlanError> {
    let welford = welford_owners(body, state);
    let mut funcs: Vec<StateFunc<'a>> = Vec::new();
    for (position, call) in body.aggs.iter().enumerate() {
        match welford.get(position).copied().flatten() {
            // The second and third aggregators of a triple add nothing: the SQL name
            // covers all three columns.
            Some(owner) => {
                if funcs.iter().any(|f| f.alias == owner.output) {
                    continue;
                }
                funcs.push(StateFunc {
                    name: sql_name(owner.func, owner.ddof),
                    call,
                    alias: owner.output.clone(),
                    welford: true,
                });
            }
            None => {
                let alias = call
                    .outputs
                    .first()
                    .map(|field| field.name().clone())
                    .unwrap_or_default();
                funcs.push(StateFunc {
                    name: agg_name(call.func)?,
                    call,
                    alias,
                    welford: false,
                });
            }
        }
    }
    Ok(funcs)
}

/// The columns a state leads with before the first aggregate's: the group list, plus the
/// `__grouping_id` an init expanding grouping sets emits beside the keys and every node
/// above it groups on. The group list alone is one short of the state exactly there, and
/// a state position read one column early names the aggregator before the right one.
pub fn key_width(body: &AggregateBody) -> usize {
    body.group_by.len() + usize::from(!body.grouping_sets.is_empty())
}

/// Per aggregator, the Welford state it belongs to, if any.
///
/// An aggregator owns as many state columns as it emits — one each at the init, three for
/// a merge's `merge_m2` — so which aggregator a state position names is a walk over those
/// widths, never `position - group_width`.
fn welford_owners<'a>(body: &AggregateBody, state: &'a Schema) -> Vec<Option<&'a AggStateColumns>> {
    let mut owners = Vec::new();
    for (position, call) in body.aggs.iter().enumerate() {
        for _ in 0..call.outputs.len().max(1) {
            owners.push(position);
        }
    }
    let group_width = key_width(body);
    let mut per_agg = vec![None; body.aggs.len()];
    for columns in &state.agg_state {
        if !matches!(columns.func, AggFunc::Stddev | AggFunc::Var) {
            continue;
        }
        for position in &columns.positions {
            let Some(state_position) = (*position as usize).checked_sub(group_width) else {
                continue;
            };
            if let Some(agg) = owners.get(state_position) {
                per_agg[*agg] = Some(columns);
            }
        }
    }
    per_agg
}

/// What the executor calls this aggregate, which is DataFusion's own name plus the `ddof`
/// spelled into it: `stddev` is the sample form and `stddev_pop` the population one.
pub fn sql_name(func: AggFunc, ddof: u32) -> &'static str {
    match (func, ddof) {
        (AggFunc::Stddev, 0) => "stddev_pop",
        (AggFunc::Stddev, _) => "stddev",
        (AggFunc::Var, 0) => "var_pop",
        (AggFunc::Var, _) => "var",
        (AggFunc::Sum, _) => "sum",
        (AggFunc::Min, _) => "min",
        (AggFunc::Max, _) => "max",
        (AggFunc::Count, _) => "count",
        (AggFunc::Avg, _) => "avg",
    }
}

/// One of ours by the name the executor knows it by. `m2` and `merge_m2` have none of
/// their own: both are folded into the SQL aggregate whose state they are, above.
fn agg_name(agg: PlanAgg) -> Result<&'static str, PlanError> {
    Ok(match agg {
        PlanAgg::Sum => "sum",
        PlanAgg::Min => "min",
        PlanAgg::Max => "max",
        PlanAgg::Count => "count",
        PlanAgg::Mean => "mean",
        PlanAgg::M2 => {
            return Err(PlanError::Unsupported(
                "m2 alone has no name on the wire — it is written as part of the stddev or \
                 var it decomposes"
                    .to_string(),
            ));
        }
        PlanAgg::MergeM2 => {
            return Err(PlanError::Unsupported(
                "merge_m2 alone has no name on the wire either — it is written as the \
                 stddev or var whose three state columns it merges"
                    .to_string(),
            ));
        }
    })
}

/// The finalize as the project it becomes: the group keys straight through, then one
/// expression per aggregate output column, named as the node declares its output.
///
/// A project replaces the row, so the finalize list alone would answer with the finalized
/// columns and no keys to read them by. Both the width and the key positions are checked
/// rather than assumed: the keys are taken by position, so a state whose first columns are
/// not the keys would pass the width check and project state columns as keys.
pub fn finalize_columns(
    body: &AggregateBody,
    state: &Schema,
    output: &Schema,
) -> Result<Vec<NamedExpr>, PlanError> {
    let finalize = body
        .finalize
        .as_ref()
        .expect("a finalize project is built only where the node finalizes");
    let keys = key_width(body);
    let declared = output.fields.fields().len();
    if keys + finalize.len() != declared {
        return Err(PlanError::Invalid(format!(
            "an aggregate finalizing {keys} keys and {} columns declares {declared} output \
             columns — the project that finalizes is the whole row, so the two have to be \
             the same list",
            finalize.len()
        )));
    }
    let field_at = |schema: &Schema, ordinal: usize| -> String {
        schema
            .fields
            .fields()
            .get(ordinal)
            .expect("a plan that validated declares the columns it names")
            .name()
            .clone()
    };
    let mut columns = Vec::with_capacity(declared);
    for ordinal in 0..keys {
        let name = field_at(state, ordinal);
        let declared_name = field_at(output, ordinal);
        if name != declared_name {
            return Err(PlanError::Invalid(format!(
                "the finalized output names column {ordinal} `{declared_name}` and the \
                 state it is projected from names it `{name}` — the keys are taken by \
                 position, so the two orders have to be the same"
            )));
        }
        columns.push(NamedExpr::new(
            Expr::column(ordinal as u32, &name),
            &declared_name,
        ));
    }
    for (offset, column) in finalize.iter().enumerate() {
        columns.push(NamedExpr::new(
            column.expr.clone(),
            &field_at(output, keys + offset),
        ));
    }
    Ok(columns)
}

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
