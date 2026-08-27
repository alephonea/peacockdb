//! The joins on the CPU: the decomposition the recipe names, one DataFusion operator per
//! call it names.
//!
//! A streamed probe cannot know which build rows matched, so the types that owe their
//! build side a row keep the probe keys per batch and answer once at done (#136). This
//! backend runs that decomposition rather than one whole join at done — the plan says the
//! probe streams, and an executor holding it all would be a different plan wearing the
//! same shape, and a poor oracle for the one the device runs.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{Field, Schema as ArrowSchema, SchemaRef};
use datafusion::common::{JoinSide as DfJoinSide, JoinType, ScalarValue};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::physical_plan::joins::{
    CrossJoinExec, HashJoinExec, NestedLoopJoinExec, PartitionMode,
};
use datafusion::physical_plan::projection::ProjectionExec;

use super::super::cpu_batch::CpuBatch;
use super::super::error::PlanError;
use super::super::executor::{BackendError, CallResult, CallStats};
use super::super::expr_physical::physical_expr;
use super::super::node::GpuNode;
use super::super::nodes::join::{
    JoinSide, NestedLoopJoinType, finish_join_type, per_call_join_type,
};
use super::super::nodes::{GpuCrossJoin, GpuJoin, GpuNestedLoopJoin};
use super::{declared_as, placeholder, run_node};

/// What a join does per call, built once. The `Option`s are the capability matrix in the
/// only form an executor needs it: a call it does not make is a call it does not have.
struct Calls {
    /// The join this probe batch runs, if any — absent for the build-side semi family,
    /// whose probe call is the key project alone.
    per_call: Option<Arc<dyn ExecutionPlan>>,
    /// The probe keys this batch contributes to the accumulation (#136).
    keys: Option<Arc<dyn ExecutionPlan>>,
    finish: Option<Arc<dyn ExecutionPlan>>,
    /// The build rows nothing matched, padded out to the joined schema.
    pad: Option<Arc<dyn ExecutionPlan>>,
    key_schema: SchemaRef,
    output: SchemaRef,
    ctx: Arc<TaskContext>,
}

/// A join before its build side arrives.
pub struct CpuJoin {
    calls: Calls,
}

impl CpuJoin {
    pub fn hash(
        node: &GpuJoin,
        build: &ArrowSchema,
        probe: &ArrowSchema,
        ctx: Arc<TaskContext>,
    ) -> Result<Self, PlanError> {
        let capability = node.capability()?;
        let output = node
            .kind()
            .schema()
            .expect("a join is not a sink")
            .fields
            .clone();
        if capability.answers_in_one_call() {
            return Ok(Self {
                calls: Calls {
                    per_call: Some(hash_join(node, node.join_type, build, probe, ctx.as_ref())?),
                    keys: None,
                    finish: None,
                    pad: None,
                    key_schema: Arc::new(ArrowSchema::empty()),
                    output,
                    ctx,
                },
            });
        }
        let per_call = match per_call_join_type(node.join_type) {
            Some(join_type) => Some(hash_join(node, join_type, build, probe, ctx.as_ref())?),
            None => None,
        };
        let (keys, key_schema) = key_project(node, probe)?;
        let finish = finish_join(node, build, &key_schema)?;
        let pad = match per_call.is_some() {
            true => Some(pad_project(node, build, probe)?),
            false => None,
        };
        Ok(Self {
            calls: Calls {
                per_call,
                keys: Some(keys),
                finish: Some(finish),
                pad,
                key_schema,
                output,
                ctx,
            },
        })
    }

    pub fn cross(
        node: &GpuCrossJoin,
        build: &ArrowSchema,
        probe: &ArrowSchema,
        ctx: Arc<TaskContext>,
    ) -> Result<Self, PlanError> {
        let cross = CrossJoinExec::new(placeholder(build), placeholder(probe));
        Ok(Self {
            calls: Self::one_call(Arc::new(cross), node.kind(), ctx),
        })
    }

    pub fn nested_loop(
        node: &GpuNestedLoopJoin,
        build: &ArrowSchema,
        probe: &ArrowSchema,
        ctx: Arc<TaskContext>,
    ) -> Result<Self, PlanError> {
        let filter = join_filter(
            &node.filter,
            &node.filter_columns,
            build,
            probe,
            ctx.as_ref(),
        )?;
        let join_type = match node.join_type {
            NestedLoopJoinType::Inner => JoinType::Inner,
            NestedLoopJoinType::Left => JoinType::Left,
        };
        let join = NestedLoopJoinExec::try_new(
            placeholder(build),
            placeholder(probe),
            Some(filter),
            &join_type,
            None,
        )
        .map_err(|error| PlanError::Invalid(format!("GpuNestedLoopJoin: {error}")))?;
        Ok(Self {
            calls: Self::one_call(Arc::new(join), node.kind(), ctx),
        })
    }

    /// The shape every join with no finish takes: one call per probe batch and nothing at
    /// done.
    fn one_call(
        join: Arc<dyn ExecutionPlan>,
        kind: &super::super::layout::NodeKind,
        ctx: Arc<TaskContext>,
    ) -> Calls {
        Calls {
            per_call: Some(join),
            keys: None,
            finish: None,
            pad: None,
            key_schema: Arc::new(ArrowSchema::empty()),
            output: kind.schema().expect("a join is not a sink").fields.clone(),
            ctx,
        }
    }

    /// Whether this join keeps probe keys and answers at done, rather than being one call
    /// and nothing else. Read by the test that holds the two readers of that rule to one
    /// answer; the rule itself is `JoinCapability::answers_in_one_call`.
    pub fn makes_a_finish_pass(&self) -> bool {
        self.calls.finish.is_some()
    }

    /// The build side, which is one batch per lane: the planner puts a
    /// `GpuCoalesceAllBatches` under it, so this is every row it will ever hold.
    pub fn set_build(self, batch: CpuBatch) -> CallResult<CpuProbingJoin> {
        Ok((
            CpuProbingJoin {
                build: batch.into_record_batch(),
                calls: self.calls,
                accumulated: Vec::new(),
            },
            CallStats::default(),
        ))
    }
}

/// A join with its build side set, taking probe batches.
pub struct CpuProbingJoin {
    build: RecordBatch,
    calls: Calls,
    accumulated: Vec<RecordBatch>,
}

impl CpuProbingJoin {
    pub fn probe_and_fetch(&mut self, batch: CpuBatch) -> CallResult<Vec<CpuBatch>> {
        let batch = batch.into_record_batch();
        if let Some(keys) = &self.calls.keys {
            let kept = run_node(keys, vec![vec![batch.clone()]], &self.calls.ctx)?;
            self.accumulated.extend(kept);
        }
        let Some(join) = &self.calls.per_call else {
            return Ok((Vec::new(), CallStats::default()));
        };
        let joined = run_node(
            join,
            vec![vec![self.build.clone()], vec![batch]],
            &self.calls.ctx,
        )?;
        Ok((declared(joined, &self.calls.output)?, CallStats::default()))
    }

    /// The question a streamed probe could not answer: which build rows nothing matched.
    pub fn finish_and_fetch(self) -> CallResult<Vec<CpuBatch>> {
        let Some(finish) = &self.calls.finish else {
            return Ok((Vec::new(), CallStats::default()));
        };
        let keys = concat_batches(&self.calls.key_schema, self.accumulated.iter())
            .map_err(|error| BackendError::new(format!("joining the probe keys: {error}")))?;
        let unmatched = run_node(finish, vec![vec![self.build], vec![keys]], &self.calls.ctx)?;
        let out = match &self.calls.pad {
            Some(pad) => run_node(pad, vec![unmatched], &self.calls.ctx)?,
            None => unmatched,
        };
        Ok((declared(out, &self.calls.output)?, CallStats::default()))
    }
}

fn declared(batches: Vec<RecordBatch>, schema: &SchemaRef) -> Result<Vec<CpuBatch>, BackendError> {
    batches
        .into_iter()
        .map(|batch| declared_as(batch, schema).map(CpuBatch::new))
        .collect()
}

/// The join a probe batch runs: the node's keys and residual, and the type the call emits
/// — the node's own where nothing finishes, the per-call one where something does.
fn hash_join(
    node: &GpuJoin,
    join_type: JoinType,
    build: &ArrowSchema,
    probe: &ArrowSchema,
    registry: &TaskContext,
) -> Result<Arc<dyn ExecutionPlan>, PlanError> {
    let mut on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> =
        Vec::with_capacity(node.keys.len());
    for (build_ordinal, probe_ordinal) in &node.keys {
        on.push((
            key_column(build, *build_ordinal)?,
            key_column(probe, *probe_ordinal)?,
        ));
    }
    let filter = match &node.filter {
        Some(expr) => Some(join_filter(
            expr,
            &node.filter_columns,
            build,
            probe,
            registry,
        )?),
        None => None,
    };
    let projection = node
        .projection
        .as_ref()
        .map(|columns| columns.iter().map(|column| *column as usize).collect());
    let join = HashJoinExec::try_new(
        placeholder(build),
        placeholder(probe),
        on,
        filter,
        &join_type,
        projection,
        // The build side is one resident table, which is what CollectLeft means here as
        // well as on the device.
        PartitionMode::CollectLeft,
        node.null_equals_null,
    )
    .map_err(|error| PlanError::Invalid(format!("GpuJoin: {error}")))?;
    Ok(Arc::new(join))
}

/// The probe keys this batch contributes, under the names they carry in the accumulation.
fn key_project(
    node: &GpuJoin,
    probe: &ArrowSchema,
) -> Result<(Arc<dyn ExecutionPlan>, SchemaRef), PlanError> {
    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = Vec::with_capacity(node.keys.len());
    for (_, probe_ordinal) in &node.keys {
        let field = field_at(probe, *probe_ordinal)?;
        exprs.push((key_column(probe, *probe_ordinal)?, field.name().clone()));
    }
    let project = ProjectionExec::try_new(exprs, placeholder(probe))
        .map_err(|error| PlanError::Invalid(format!("the probe key project: {error}")))?;
    let schema = project.schema();
    Ok((Arc::new(project), schema))
}

/// The finish join, against the accumulated probe keys. Their ordinals are `0..k` and
/// their names are the probe's, since the key project is what built that table — no
/// residual and no projection, because the question is only which build rows matched.
fn finish_join(
    node: &GpuJoin,
    build: &ArrowSchema,
    keys: &SchemaRef,
) -> Result<Arc<dyn ExecutionPlan>, PlanError> {
    let mut on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> =
        Vec::with_capacity(node.keys.len());
    for (position, (build_ordinal, _)) in node.keys.iter().enumerate() {
        on.push((
            key_column(build, *build_ordinal)?,
            key_column(keys, position as u32)?,
        ));
    }
    let join = HashJoinExec::try_new(
        placeholder(build),
        placeholder(keys),
        on,
        None,
        &finish_join_type(node.join_type),
        None,
        PartitionMode::CollectLeft,
        node.null_equals_null,
    )
    .map_err(|error| PlanError::Invalid(format!("the finish join: {error}")))?;
    Ok(Arc::new(join))
}

/// What the node declares, out of an anti join that emitted build columns only: each kept
/// column in the projection's order, a build one read from the anti join's output and a
/// probe one as a typed NULL.
fn pad_project(
    node: &GpuJoin,
    build: &ArrowSchema,
    probe: &ArrowSchema,
) -> Result<Arc<dyn ExecutionPlan>, PlanError> {
    let build_width = build.fields().len() as u32;
    let kept: Vec<u32> = match &node.projection {
        Some(columns) => columns.clone(),
        None => (0..build_width + probe.fields().len() as u32).collect(),
    };
    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = Vec::with_capacity(kept.len());
    for ordinal in kept {
        if ordinal < build_width {
            let field = field_at(build, ordinal)?;
            exprs.push((key_column(build, ordinal)?, field.name().clone()));
            continue;
        }
        let field = field_at(probe, ordinal - build_width)?;
        let null = ScalarValue::try_from(field.data_type()).map_err(|error| {
            PlanError::Invalid(format!("a typed NULL for {}: {error}", field.name()))
        })?;
        exprs.push((Arc::new(Literal::new(null)), field.name().clone()));
    }
    let project = ProjectionExec::try_new(exprs, placeholder(build))
        .map_err(|error| PlanError::Invalid(format!("the pad project: {error}")))?;
    Ok(Arc::new(project))
}

/// The residual, rebuilt against the intermediate schema its column map names — the same
/// reconstruction the legacy deserializer makes, since the map is the same map.
fn join_filter(
    filter: &super::super::expr::Expr,
    columns: &[super::super::nodes::join::JoinFilterColumn],
    build: &ArrowSchema,
    probe: &ArrowSchema,
    registry: &TaskContext,
) -> Result<JoinFilter, PlanError> {
    let mut fields: Vec<Field> = Vec::with_capacity(columns.len());
    let mut indices = Vec::with_capacity(columns.len());
    for column in columns {
        let (side, schema) = match column.side {
            JoinSide::Build => (DfJoinSide::Left, build),
            JoinSide::Probe => (DfJoinSide::Right, probe),
        };
        fields.push(field_at(schema, column.index)?.as_ref().clone());
        indices.push(ColumnIndex {
            index: column.index as usize,
            side,
        });
    }
    let intermediate = Arc::new(ArrowSchema::new(fields));
    let expression = physical_expr(filter, &intermediate, registry)?;
    Ok(JoinFilter::new(expression, indices, intermediate))
}

fn key_column(schema: &ArrowSchema, ordinal: u32) -> Result<Arc<dyn PhysicalExpr>, PlanError> {
    let field = field_at(schema, ordinal)?;
    Ok(Arc::new(Column::new(field.name(), ordinal as usize)))
}

fn field_at(schema: &ArrowSchema, ordinal: u32) -> Result<&Arc<Field>, PlanError> {
    schema.fields().get(ordinal as usize).ok_or_else(|| {
        PlanError::Invalid(format!(
            "a join reads column {ordinal} of an input with {} columns",
            schema.fields().len()
        ))
    })
}
