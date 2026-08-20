//! The aggregate sequence: what a partial declares, decomposed into the parts each
//! position needs. Which parts are emitted is decided here rather than by the node kinds
//! alone, because the same DataFusion pair becomes a different tree at one lane and at
//! four.

use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Fields, Schema as ArrowSchema};
use datafusion::physical_expr::aggregate::AggregateFunctionExpr;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};

use super::super::aggregates::{AggCall, Merge, PlanAgg, decomposition, finalize, resolve};
use super::super::error::PlanError;
use super::super::expr::{Expr, NamedExpr};
use super::super::expr_translate::translate_expr;
use super::super::layout::BatchLayout;
use super::super::node::GpuNode;
use super::super::nodes::{AggregateBody, GpuAggregate, GpuAggregateBatches};
use super::super::schema::{AggStateColumns, Schema};
use super::{Translator, batches, hash_key_ordinals, lanes};

impl Translator {
    pub(super) fn aggregate(
        &self,
        aggregate: &AggregateExec,
    ) -> Result<Box<dyn GpuNode>, PlanError> {
        match aggregate.mode() {
            AggregateMode::Partial => self.aggregate_sequence(aggregate, None, Shuffle::None),
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                let (below, shuffle) = shuffle_below(aggregate.input());
                let partial = below
                    .as_any()
                    .downcast_ref::<AggregateExec>()
                    .filter(|partial| matches!(partial.mode(), AggregateMode::Partial))
                    .ok_or_else(|| {
                        PlanError::Unsupported(format!(
                            "a final aggregate over {} rather than a partial one",
                            below.name()
                        ))
                    })?;
                self.aggregate_sequence(partial, Some(aggregate), shuffle)
            }
            AggregateMode::Single | AggregateMode::SinglePartitioned => {
                self.aggregate_sequence(aggregate, Some(aggregate), Shuffle::None)
            }
        }
    }

    /// The whole sequence, from the aggregators the partial declares: init per batch, a
    /// per-lane merge where a lane holds several batches, the shuffle where the lanes must
    /// be re-landed by group key, and the merge that finishes it. Each part is emitted only
    /// where this lane count and batch layout need it — a one-lane region never splits, so
    /// there is nothing to merge back.
    fn aggregate_sequence(
        &self,
        partial: &AggregateExec,
        finisher: Option<&AggregateExec>,
        shuffle: Shuffle,
    ) -> Result<Box<dyn GpuNode>, PlanError> {
        let input = self.node(partial.input())?;
        let input_schema = partial.input().schema();
        let group = partial.group_expr();
        if partial.filter_expr().iter().any(Option::is_some) {
            return Err(PlanError::Unsupported(
                "a filtered aggregate (#161)".to_string(),
            ));
        }

        let mut group_by = Vec::with_capacity(group.expr().len());
        for (expr, _) in group.expr().iter() {
            group_by.push(translate_expr(expr, &input_schema)?);
        }
        // Grouping sets add one output column, `__grouping_id`, which the init emits like
        // any other and everything above groups on beside the keys. Its name and type are
        // DataFusion's, off the partial's own schema.
        let key_columns = group.expr().len() + usize::from(!group.is_single());
        let key_fields: Vec<Field> = (0..key_columns)
            .map(|index| partial.schema().field(index).clone())
            .collect();
        let mut null_exprs = Vec::new();
        for (expr, _) in group.null_expr().iter() {
            null_exprs.push(translate_expr(expr, &input_schema)?);
        }
        let grouping_sets: Vec<Vec<bool>> = if group.is_single() {
            Vec::new()
        } else {
            group.groups().to_vec()
        };

        let decomposed = decompose(partial.aggr_expr(), &input_schema, key_fields.len())?;
        let intermediate = Schema {
            fields: Arc::new(ArrowSchema::new(Fields::from(
                [key_fields.clone(), decomposed.state.clone()].concat(),
            ))),
            group_keys: (0..key_fields.len() as u32).collect(),
            agg_state: decomposed.annotations.clone(),
        };
        let keys_through: Vec<Expr> = key_fields
            .iter()
            .enumerate()
            .map(|(index, field)| Expr::column(index as u32, field.name()))
            .collect();

        // The output names are DataFusion's, so a finalized column lands where the plan
        // above it expects to read it.
        let finished = finisher.map(|finisher| {
            let names = finisher.schema();
            let finalize: Vec<NamedExpr> = decomposed
                .finalize
                .iter()
                .enumerate()
                .map(|(index, expr)| {
                    NamedExpr::new(expr.clone(), names.field(key_fields.len() + index).name())
                })
                .collect();
            // The finalized output holds the keys where the intermediate did and the
            // finalized columns where the state was, so the keys are still annotated and
            // the state is gone.
            let output = Schema {
                fields: finisher.schema(),
                group_keys: (0..key_fields.len() as u32).collect(),
                agg_state: Vec::new(),
            };
            (finalize, output)
        });

        // One batch in one lane is already the whole of every group, so the init node
        // finishes the aggregate itself.
        if let Some((finalize, output)) = &finished
            && batches(input.as_ref()) == BatchLayout::SingleBatch
            && lanes(input.as_ref()) == 1
        {
            return Ok(Box::new(GpuAggregate::new(
                input,
                AggregateBody {
                    group_by,
                    grouping_sets,
                    null_exprs,
                    aggs: decomposed.init,
                    finalize: Some(finalize.clone()),
                },
                intermediate,
                output.clone(),
            )));
        }

        let mut tree: Box<dyn GpuNode> = Box::new(GpuAggregate::new(
            input,
            AggregateBody {
                group_by,
                grouping_sets,
                null_exprs,
                aggs: decomposed.init,
                finalize: None,
            },
            intermediate.clone(),
            intermediate.clone(),
        ));

        // A merge groups on what the init emitted — keys and, where there was one, the
        // grouping id — and expands nothing: the sets were expanded once, below.
        let merge_body = |aggs: &[AggCall], finalize: Option<Vec<NamedExpr>>| AggregateBody {
            group_by: keys_through.clone(),
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: aggs.to_vec(),
            finalize,
        };

        // The per-lane half exists to shrink what crosses the shuffle; where the lanes
        // stay put there is nothing for it to do that the finishing merge does not.
        let regrouped = !matches!(shuffle, Shuffle::None) && lanes(tree.as_ref()) > 1;
        if regrouped && batches(tree.as_ref()) != BatchLayout::SingleBatch {
            tree = Box::new(GpuAggregateBatches::new(
                tree,
                merge_body(&decomposed.merge, None),
                intermediate.clone(),
                intermediate.clone(),
            ));
        }

        tree = match shuffle {
            Shuffle::ByHash { keys, n } if lanes(tree.as_ref()) > 1 => self.shuffled(tree, keys, n),
            // One lane holds every group already: v1 skips the shuffle for a one-lane
            // input exactly as it does for a keyless aggregate.
            Shuffle::ByHash { .. } => tree,
            Shuffle::Collapse => self.merged(tree),
            Shuffle::None => tree,
        };

        let (finalize, output) = match finished {
            Some((finalize, output)) => (Some(finalize), output),
            None => (None, intermediate.clone()),
        };
        Ok(Box::new(GpuAggregateBatches::new(
            tree,
            merge_body(&decomposed.merge, finalize),
            intermediate,
            output,
        )))
    }
}

/// What sits between a partial aggregate and the final one: DataFusion spells a shuffle as
/// a hash repartition and a lane collapse as a coalesce, and which one it chose is what
/// decides whether this aggregate re-lands its rows by key or merges them into one lane.
enum Shuffle {
    None,
    ByHash { keys: Vec<u32>, n: usize },
    Collapse,
}

/// Look through the nodes a shuffle is spelled with. A coalesce carrying a fetch is not
/// one of them — that is a limit, and it stops the walk.
fn shuffle_below(plan: &Arc<dyn ExecutionPlan>) -> (Arc<dyn ExecutionPlan>, Shuffle) {
    let mut node = plan.clone();
    let mut shuffle = Shuffle::None;
    loop {
        let any = node.as_any();
        if let Some(coalesce) = any.downcast_ref::<CoalesceBatchesExec>() {
            if coalesce.fetch().is_some() {
                return (node, shuffle);
            }
            node = coalesce.input().clone();
            continue;
        }
        if let Some(collapse) = any.downcast_ref::<CoalescePartitionsExec>() {
            shuffle = Shuffle::Collapse;
            node = collapse.input().clone();
            continue;
        }
        if let Some(repartition) = any.downcast_ref::<RepartitionExec>() {
            if let Partitioning::Hash(exprs, n) = repartition.partitioning() {
                match hash_key_ordinals(exprs, &repartition.input().schema()) {
                    Ok(keys) => shuffle = Shuffle::ByHash { keys, n: *n },
                    // Not a shape this walk can describe; the node arm will refuse it.
                    Err(_) => return (node, shuffle),
                }
            }
            node = repartition.input().clone();
            continue;
        }
        return (node, shuffle);
    }
}

/// The state field DataFusion declared for one of our aggregators. With a single state
/// column there is nothing to mismatch; beyond that the aggregator's tag is what names it
/// (`avg(x)[count]`), and a tag with no field is a drift this must not paper over.
fn declared_state<'a>(
    declared: &'a [Field],
    func: PlanAgg,
    aggregate: &str,
) -> Result<&'a Field, PlanError> {
    if declared.len() == 1 {
        return Ok(&declared[0]);
    }
    let tag = format!("[{}]", func.tag());
    declared
        .iter()
        .find(|field| field.name().ends_with(&tag))
        .ok_or_else(|| {
            PlanError::Invalid(format!(
                "{aggregate}: DataFusion declares no {tag} state column, so this mode's \
                 decomposition of it has drifted"
            ))
        })
}

/// What one aggregate node's aggregates become in each position.
struct Decomposed {
    init: Vec<AggCall>,
    merge: Vec<AggCall>,
    state: Vec<Field>,
    finalize: Vec<Expr>,
    /// One per aggregate sql asked for, naming the state columns it decomposed into —
    /// what a merge checks before trusting the positions it is about to merge.
    annotations: Vec<AggStateColumns>,
}

fn decompose(
    aggregates: &[Arc<AggregateFunctionExpr>],
    input_schema: &ArrowSchema,
    n_keys: usize,
) -> Result<Decomposed, PlanError> {
    let mut decomposed = Decomposed {
        init: Vec::new(),
        merge: Vec::new(),
        state: Vec::new(),
        finalize: Vec::new(),
        annotations: Vec::new(),
    };

    for aggregate in aggregates {
        if aggregate.is_distinct() {
            return Err(PlanError::Unsupported(format!(
                "DISTINCT inside {} (#62)",
                aggregate.name()
            )));
        }
        let spec = resolve(aggregate.fun().name())?;
        let rule = decomposition(spec.func);
        let declared = aggregate
            .state_fields()
            .map_err(|e| PlanError::Invalid(format!("{}: {e}", aggregate.name())))?;
        if declared.len() != rule.state.len() {
            return Err(PlanError::Invalid(format!(
                "{}: DataFusion declares {} state columns and this mode decomposes into {}",
                aggregate.name(),
                declared.len(),
                rule.state.len()
            )));
        }

        let mut args = Vec::with_capacity(aggregate.expressions().len());
        for arg in aggregate.expressions() {
            args.push(translate_expr(&arg, input_schema)?);
        }

        // The state names are ours — the golden and every later reference read them —
        // and the types are DataFusion's. Paired by the aggregator's tag rather than by
        // position: DataFusion declares avg as [count, sum] and this table reads
        // [sum, count], so a positional pairing types both of them wrongly.
        let state_at = n_keys + decomposed.state.len();
        let mut state = Vec::with_capacity(rule.state.len());
        for (suffix, func) in rule.state {
            let field = declared_state(&declared, *func, aggregate.name())?;
            state.push(Field::new(
                format!("{}{suffix}", aggregate.name()),
                field.data_type().clone(),
                field.is_nullable(),
            ));
        }

        for ((_, func), field) in rule.state.iter().zip(state.iter()) {
            decomposed.init.push(AggCall {
                func: *func,
                args: args.clone(),
                outputs: vec![field.clone()],
            });
        }

        let state_columns: Vec<Expr> = state
            .iter()
            .enumerate()
            .map(|(offset, field)| Expr::column((state_at + offset) as u32, field.name()))
            .collect();
        match rule.merge {
            Merge::PerColumn(funcs) => {
                for ((func, column), field) in
                    funcs.iter().zip(state_columns.iter()).zip(state.iter())
                {
                    decomposed.merge.push(AggCall {
                        func: *func,
                        args: vec![column.clone()],
                        outputs: vec![field.clone()],
                    });
                }
            }
            Merge::Combined(func) => decomposed.merge.push(AggCall {
                func,
                args: state_columns,
                outputs: state.clone(),
            }),
        }

        decomposed.finalize.push(finalize(
            spec,
            &state,
            state_at as u32,
            aggregate.field().data_type(),
        ));
        decomposed.annotations.push(AggStateColumns {
            output: aggregate.name().to_string(),
            func: spec.func,
            ddof: spec.ddof,
            positions: (state_at..state_at + state.len())
                .map(|position| position as u32)
                .collect(),
        });
        decomposed.state.extend(state);
    }

    Ok(decomposed)
}
