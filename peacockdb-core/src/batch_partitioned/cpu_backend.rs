//! The CPU backend's executors: one DataFusion operator per node, run one batch at a time.
//!
//! Reuse with the legacy modes is the point rather than something to avoid — both ask
//! DataFusion for the same operator — so this relays through [`execute_single_node`],
//! which `node_by_node` already uses. The operator is built at construction, and the
//! batch is what changes per call.
//!
//! The traits are synchronous and DataFusion's operator API is not, so each call blocks on
//! one node's stream. Sound for these operators, which spawn nothing, and the reason the
//! relay is per node: a whole plan would contain the operators that do.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef};
use datafusion::execution::{FunctionRegistry, TaskContext};
use datafusion::physical_expr::aggregate::AggregateExprBuilder;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;

use crate::executors::single_node::execute_single_node;

use super::cpu_batch::CpuBatch;
use super::error::PlanError;
use super::executor::{BackendError, CallResult, CallStats, RowRange};
use super::expr_physical::{physical_expr, physical_projection};
use super::layout::ColumnOrder;
use super::node::GpuNode;
use super::nodes::aggregate::{AggregateBody, finalize_columns, state_funcs};
use super::nodes::{GpuAggregate, GpuFilter, GpuProject, GpuSort};
use super::schema::Schema;

/// One DataFusion node with its child left as a placeholder, and the columns this mode
/// says it produces. [`execute_single_node`] replaces the children with the batches it is
/// handed, so what is stored is the operator and its expressions, never a source.
struct Stage {
    node: Arc<dyn ExecutionPlan>,
    /// The node's own schema, which is not always the one DataFusion answers with: a
    /// partial aggregate names its state columns after the accumulators it ran, and this
    /// mode names them in the schema every reference above resolves against.
    declared: SchemaRef,
}

/// A node's operators in call order, each one's output the next one's input — one for a
/// filter, two for an aggregate that finalizes, which is the recipe's own call list on the
/// other backend.
pub struct CpuExec {
    stages: Vec<Stage>,
    ctx: Arc<TaskContext>,
}

impl CpuExec {
    pub fn filter(
        node: &GpuFilter,
        input: &ArrowSchema,
        ctx: Arc<TaskContext>,
    ) -> Result<Self, PlanError> {
        let child = placeholder(input);
        let predicate = physical_expr(&node.predicate, input, ctx.as_ref())?;
        let filter = FilterExec::try_new(predicate, child)
            .map_err(|error| PlanError::Invalid(format!("GpuFilter: {error}")))?;
        let filter = match &node.projection {
            Some(columns) => filter
                .with_projection(Some(columns.iter().map(|c| *c as usize).collect()))
                .map_err(|error| PlanError::Invalid(format!("GpuFilter projection: {error}")))?,
            None => filter,
        };
        Ok(Self::of(vec![Arc::new(filter)], ctx))
    }

    pub fn project(
        node: &GpuProject,
        input: &ArrowSchema,
        ctx: Arc<TaskContext>,
    ) -> Result<Self, PlanError> {
        let exprs = physical_projection(&node.exprs, input, ctx.as_ref())?;
        let project = ProjectionExec::try_new(exprs, placeholder(input))
            .map_err(|error| PlanError::Invalid(format!("GpuProject: {error}")))?;
        Ok(Self::of(vec![Arc::new(project)], ctx))
    }

    /// The per-batch sort: `fetch` is the top-N within this batch, which is what makes a
    /// sort 1:1 per batch rather than an accumulator. Ordering the whole stream is
    /// `GpuAccumulateBatchesAndSort`, a different node.
    pub fn sort(
        node: &GpuSort,
        input: &ArrowSchema,
        ctx: Arc<TaskContext>,
    ) -> Result<Self, PlanError> {
        let ordering = lex_ordering(&node.keys, input)?;
        let sort = SortExec::new(ordering, placeholder(input)).with_fetch(node.fetch);
        Ok(Self::of(vec![Arc::new(sort)], ctx))
    }

    /// State from raw values, and the finalize where the node carries one — two operators,
    /// as the recipe is two calls. The split is not DataFusion's `Single` mode: this mode
    /// finalizes in a project so that both engines evaluate the one finalize expression,
    /// and a `Single` here would be a second implementation of it.
    pub fn aggregate(
        node: &GpuAggregate,
        input: &ArrowSchema,
        ctx: Arc<TaskContext>,
    ) -> Result<Self, PlanError> {
        let body = &node.body;
        let state = node.intermediate();
        let mut stages = vec![Stage {
            node: aggregate_exec(body, AggregateMode::Partial, input, state, ctx.as_ref())?,
            declared: state.fields.clone(),
        }];
        if body.finalize.is_some() {
            let output = node.kind().schema().expect("an aggregate is not a sink");
            let columns = finalize_columns(body, state, output)?;
            let exprs = physical_projection(&columns, &state.fields, ctx.as_ref())?;
            let project = ProjectionExec::try_new(exprs, placeholder(&state.fields))
                .map_err(|error| PlanError::Invalid(format!("the finalize project: {error}")))?;
            stages.push(Stage {
                node: Arc::new(project),
                declared: output.fields.clone(),
            });
        }
        Ok(Self { stages, ctx })
    }

    fn of(nodes: Vec<Arc<dyn ExecutionPlan>>, ctx: Arc<TaskContext>) -> Self {
        Self {
            stages: nodes
                .into_iter()
                .map(|node| Stage {
                    declared: node.schema(),
                    node,
                })
                .collect(),
            ctx,
        }
    }

    /// One batch in, one batch out — the contract every `Exec` node keeps. DataFusion may
    /// answer a batch with several or with none (a filter that kept nothing emits
    /// nothing), so the pieces are concatenated and an empty answer becomes an empty batch
    /// of the node's schema rather than a missing one.
    pub fn exec(&mut self, batch: CpuBatch) -> CallResult<CpuBatch> {
        let mut batches = vec![batch.into_record_batch()];
        for stage in &self.stages {
            let produced = self.run(&stage.node, vec![batches])?;
            batches = produced
                .into_iter()
                .map(|batch| declared_as(batch, &stage.declared))
                .collect::<Result<Vec<RecordBatch>, BackendError>>()?;
        }
        let schema = &self.stages.last().expect("a node has an operator").declared;
        let batch = concat_batches(schema, batches.iter())
            .map_err(|error| BackendError::new(format!("joining the node's output: {error}")))?;
        Ok((CpuBatch::new(batch), CallStats::default()))
    }

    fn run(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        inputs: Vec<Vec<RecordBatch>>,
    ) -> Result<Vec<RecordBatch>, BackendError> {
        futures::executor::block_on(execute_single_node(node, inputs, self.ctx.clone()))
            .map(|(batches, _)| batches)
            .map_err(|error| BackendError::new(error.to_string()))
    }
}

/// Where data leaves the device on the GPU path, and a slice on this one. The row range
/// arrives per call because a root-adjacent limit counts across lanes and only the driver
/// holds that count.
pub struct CpuUnload;

impl CpuUnload {
    pub fn unload(&mut self, batch: CpuBatch, rows: RowRange) -> CallResult<CpuBatch> {
        let batch = batch.into_record_batch();
        let n_rows = batch.num_rows() as u64;
        if rows.covers(n_rows) {
            return Ok((CpuBatch::new(batch), CallStats::default()));
        }
        let offset = rows.offset.min(n_rows);
        let length = rows.length.min(n_rows - offset);
        Ok((
            CpuBatch::new(batch.slice(offset as usize, length as usize)),
            CallStats::default(),
        ))
    }
}

/// The same columns under the names the node declares. Positional, and checked by arrow:
/// a column whose type is not the declared one is a state layout this mode and DataFusion
/// disagree about, which is a wrong answer everywhere above rather than an error.
fn declared_as(batch: RecordBatch, declared: &SchemaRef) -> Result<RecordBatch, BackendError> {
    if batch.schema() == *declared {
        return Ok(batch);
    }
    RecordBatch::try_new(declared.clone(), batch.columns().to_vec()).map_err(|error| {
        BackendError::new(format!(
            "the node declares {declared:?} and DataFusion answered with {:?}: {error}",
            batch.schema()
        ))
    })
}

/// A child of the right schema and nothing else: `execute_single_node` swaps it for a
/// stream over the batches the call was handed, so what it holds is never read.
fn placeholder(schema: &ArrowSchema) -> Arc<dyn ExecutionPlan> {
    Arc::new(EmptyExec::new(Arc::new(schema.clone())))
}

/// The aggregate as DataFusion runs it: the group list under the names the state gives
/// them, and one SQL aggregate per [`state_funcs`] entry — which is what makes the state
/// this produces the state the node declared, three columns at a time where a Welford
/// triple is one aggregate.
fn aggregate_exec(
    body: &AggregateBody,
    mode: AggregateMode,
    input: &ArrowSchema,
    state: &Schema,
    registry: &dyn FunctionRegistry,
) -> Result<Arc<dyn ExecutionPlan>, PlanError> {
    let input_schema = Arc::new(input.clone());
    let named = |exprs: &[super::expr::Expr]| -> Result<Vec<_>, PlanError> {
        exprs
            .iter()
            .enumerate()
            .map(|(position, expr)| {
                Ok((
                    physical_expr(expr, input, registry)?,
                    key_name(state, position),
                ))
            })
            .collect()
    };
    let keys = named(&body.group_by)?;
    let group_by = if body.grouping_sets.is_empty() {
        PhysicalGroupBy::new_single(keys)
    } else {
        PhysicalGroupBy::new(keys, named(&body.null_exprs)?, body.grouping_sets.clone())
    };

    let mut aggregates = Vec::new();
    for func in state_funcs(body, state)? {
        let udaf = registry.udaf(func.name).map_err(|error| {
            PlanError::Unsupported(format!(
                "`{}` is not one of this session's aggregates: {error}",
                func.name
            ))
        })?;
        let mut args = Vec::with_capacity(func.call.args.len());
        for arg in &func.call.args {
            args.push(physical_expr(arg, input, registry)?);
        }
        aggregates.push(Arc::new(
            AggregateExprBuilder::new(udaf, args)
                .schema(input_schema.clone())
                .alias(&func.alias)
                .build()
                .map_err(|error| PlanError::Invalid(format!("{}: {error}", func.name)))?,
        ));
    }

    let filters = vec![None; aggregates.len()];
    let aggregate = AggregateExec::try_new(
        mode,
        group_by,
        aggregates,
        filters,
        placeholder(input),
        input_schema,
    )
    .map_err(|error| PlanError::Invalid(format!("the aggregate: {error}")))?;
    let produced = aggregate.schema();
    check_state_layout(&produced, state)?;
    Ok(Arc::new(aggregate))
}

/// The state's key columns lead it, so a group position is a position in it — the same
/// rule the recipe writer's group names follow, since the two sides have to agree on which
/// column a key landed in.
fn key_name(state: &Schema, position: usize) -> String {
    state
        .fields
        .fields()
        .get(position)
        .expect("a plan that validated declares the keys it groups by")
        .name()
        .clone()
}

/// The state columns are relabelled positionally, so what DataFusion produces has to be
/// the shape the node declared. Types only: the names are what differ by design, and
/// nullability is DataFusion's own, copied into the declaration when the plan was built.
fn check_state_layout(produced: &ArrowSchema, declared: &Schema) -> Result<(), PlanError> {
    let ours = declared.fields.fields();
    if produced.fields().len() != ours.len() {
        return Err(PlanError::Invalid(format!(
            "the aggregate declares {} columns and DataFusion's accumulators produce {}",
            ours.len(),
            produced.fields().len()
        )));
    }
    for (position, (theirs, ours)) in produced.fields().iter().zip(ours.iter()).enumerate() {
        if theirs.data_type() != ours.data_type() {
            return Err(PlanError::Invalid(format!(
                "column {position} is {} in the declared state and {} in the one \
                 DataFusion's accumulators produce — a state read positionally has to be \
                 the state that was declared",
                ours.data_type(),
                theirs.data_type()
            )));
        }
    }
    Ok(())
}

fn lex_ordering(keys: &[ColumnOrder], input: &ArrowSchema) -> Result<LexOrdering, PlanError> {
    let mut exprs = Vec::with_capacity(keys.len());
    for key in keys {
        let field = input.fields().get(key.column as usize).ok_or_else(|| {
            PlanError::Invalid(format!(
                "sort key at {} and the input has {} columns",
                key.column,
                input.fields().len()
            ))
        })?;
        exprs.push(PhysicalSortExpr::new(
            Arc::new(datafusion::physical_expr::expressions::Column::new(
                field.name(),
                key.column as usize,
            )),
            datafusion::arrow::compute::SortOptions {
                descending: !key.ascending,
                nulls_first: key.nulls_first,
            },
        ));
    }
    Ok(LexOrdering::new(exprs))
}

#[cfg(test)]
mod tests;
