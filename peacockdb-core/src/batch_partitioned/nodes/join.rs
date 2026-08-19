//! The joins. All three take their build side as one batch per lane and stream the probe
//! where the capability matrix allows it; the equi-join is co-partitioned, while cross and
//! nested-loop need both inputs in one lane, having no key to co-locate on (#140).

use std::any::Any;

use super::super::error::PlanError;
use super::super::expr::ColumnRef;
use super::super::expr::Expr;
use datafusion::common::JoinType;

use super::super::layout::{BatchLayout, KeyDistribution, NodeKind, PartitionLayout, SortOrder};
use super::super::node::GpuNode;
use super::super::schema::Schema;
use super::{input_layout, input_schema};

/// DataFusion's join type, restricted to what a nested-loop join can run: the C++ rejects
/// anything else outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NestedLoopJoinType {
    Inner,
    Left,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinSide {
    Build,
    Probe,
}

/// Where one column of a join filter's own table comes from. The filter is written
/// against a schema of its own — neither side's, and not the joined one — so its
/// ordinals mean nothing without this map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinFilterColumn {
    pub side: JoinSide,
    pub index: u32,
}

#[derive(Debug)]
pub struct GpuCrossJoin {
    kind: NodeKind,
    build: Box<dyn GpuNode>,
    probe: Box<dyn GpuNode>,
}

impl GpuCrossJoin {
    pub fn new(build: Box<dyn GpuNode>, probe: Box<dyn GpuNode>, schema: Schema) -> Self {
        Self {
            kind: NodeKind::Intermediate {
                layout: joined_layout(),
                schema,
            },
            build,
            probe,
        }
    }
}

impl GpuNode for GpuCrossJoin {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        vec![self.build.as_ref(), self.probe.as_ref()]
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        check_join_inputs("GpuCrossJoin", self.build.as_ref(), self.probe.as_ref())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The predicate is the join: `conditional_inner_join` evaluates it per pair, or a cross
/// join and a mask where it is not AST-able.
#[derive(Debug)]
pub struct GpuNestedLoopJoin {
    kind: NodeKind,
    pub join_type: NestedLoopJoinType,
    pub filter: Expr,
    /// One entry per column the filter's own schema has, in its order.
    pub filter_columns: Vec<JoinFilterColumn>,
    build: Box<dyn GpuNode>,
    probe: Box<dyn GpuNode>,
}

impl GpuNestedLoopJoin {
    pub fn new(
        build: Box<dyn GpuNode>,
        probe: Box<dyn GpuNode>,
        join_type: NestedLoopJoinType,
        filter: Expr,
        filter_columns: Vec<JoinFilterColumn>,
        schema: Schema,
    ) -> Self {
        Self {
            kind: NodeKind::Intermediate {
                layout: joined_layout(),
                schema,
            },
            join_type,
            filter,
            filter_columns,
            build,
            probe,
        }
    }
}

impl GpuNode for GpuNestedLoopJoin {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        vec![self.build.as_ref(), self.probe.as_ref()]
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        check_join_inputs(
            "GpuNestedLoopJoin",
            self.build.as_ref(),
            self.probe.as_ref(),
        )?;
        check_filter_columns(
            &self.filter,
            &self.filter_columns,
            &input_schema(self.build.as_ref()),
            &input_schema(self.probe.as_ref()),
        )?;
        if self.join_type == NestedLoopJoinType::Left
            && input_layout(self.probe.as_ref()).batch_layout != BatchLayout::SingleBatch
        {
            return Err(PlanError::Invalid(
                "GpuNestedLoopJoin{Left}: its probe side must be one batch — the finish pass \
                 accumulates keys and a predicate join has none, so the planner puts a \
                 GpuCoalesceAllBatches below it"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// One lane, many batches, no order and no key: a join emits rows in the order its probe
/// batches arrive, and nothing about the inputs' layout survives it.
fn joined_layout() -> PartitionLayout {
    PartitionLayout {
        n: 1,
        key_distribution: KeyDistribution::NotSpecified,
        sort_order: SortOrder::NotSpecified,
        batch_layout: BatchLayout::MultipleBatches,
    }
}

fn check_join_inputs(
    node: &str,
    build: &dyn GpuNode,
    probe: &dyn GpuNode,
) -> Result<(), PlanError> {
    let (build_layout, probe_layout) = (input_layout(build), input_layout(probe));
    if build_layout.n != 1 || probe_layout.n != 1 {
        return Err(PlanError::Invalid(format!(
            "{node}: with no key to co-locate on both inputs must be one lane, not {} and {} \
             — the planner inserts GpuMergePartitions",
            build_layout.n, probe_layout.n
        )));
    }
    if build_layout.batch_layout != BatchLayout::SingleBatch {
        return Err(PlanError::Invalid(format!(
            "{node}: its build side must be one batch — the planner inserts \
             GpuCoalesceAllBatches below it"
        )));
    }
    Ok(())
}

/// Every reference in a join filter is an ordinal into the filter's own table, so each
/// one has to name a mapped column AND find its own name on the side that column comes
/// from — a mapping that points at the wrong side otherwise reads a valid column of the
/// wrong table, which nothing downstream can detect.
fn check_filter_columns(
    filter: &Expr,
    columns: &[JoinFilterColumn],
    build: &Schema,
    probe: &Schema,
) -> Result<(), PlanError> {
    let mut refs = Vec::new();
    collect_column_refs(filter, &mut refs);
    for reference in refs {
        let mapped = columns.get(reference.index as usize).ok_or_else(|| {
            PlanError::Invalid(format!(
                "GpuNestedLoopJoin: filter column {}@{} is past the {} its map has",
                reference.name,
                reference.index,
                columns.len()
            ))
        })?;
        let source = match mapped.side {
            JoinSide::Build => build,
            JoinSide::Probe => probe,
        };
        let field = source
            .fields
            .fields()
            .get(mapped.index as usize)
            .ok_or_else(|| {
                PlanError::Invalid(format!(
                    "GpuNestedLoopJoin: filter column {}@{} maps past its side's columns",
                    reference.name, reference.index
                ))
            })?;
        if field.name() != &reference.name {
            return Err(PlanError::Invalid(format!(
                "GpuNestedLoopJoin: filter column {}@{} maps to {} on the {:?} side",
                reference.name,
                reference.index,
                field.name(),
                mapped.side
            )));
        }
    }
    Ok(())
}

fn collect_column_refs<'a>(expr: &'a Expr, into: &mut Vec<&'a ColumnRef>) {
    match expr {
        Expr::Column(reference) => into.push(reference),
        Expr::Literal(_) => {}
        Expr::Binary { left, right, .. } => {
            collect_column_refs(left, into);
            collect_column_refs(right, into);
        }
        Expr::Unary { arg, .. } => collect_column_refs(arg, into),
        Expr::Cast { expr, .. } => collect_column_refs(expr, into),
        Expr::Like { expr, pattern, .. } => {
            collect_column_refs(expr, into);
            collect_column_refs(pattern, into);
        }
        Expr::Case {
            comparand,
            when_then,
            else_expr,
        } => {
            for part in comparand.iter().chain(else_expr.iter()) {
                collect_column_refs(part, into);
            }
            for (when, then) in when_then {
                collect_column_refs(when, into);
                collect_column_refs(then, into);
            }
        }
        Expr::ScalarFunction { args, .. } => {
            for arg in args {
                collect_column_refs(arg, into);
            }
        }
    }
}

/// What the capability matrix says about one join mode: whether the probe side can stream
/// batch by batch, and whether the lane owes a pass at done for what a streamed probe
/// cannot know — which build rows matched at least once (#136).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinCapability {
    pub probe_streams: bool,
    pub needs_finish: bool,
}

/// The three refused shapes are refusals of a defect or a missing cuDF variant, not of
/// this mode: an outer join's residual filter is applied after the outer gather and drops
/// the padded rows (#153), and no swapped `mixed_*` variant exists for the right-handed
/// semi family.
pub fn capability(join_type: JoinType, has_filter: bool) -> Result<JoinCapability, PlanError> {
    let streaming = |needs_finish| {
        Ok(JoinCapability {
            probe_streams: true,
            needs_finish,
        })
    };
    match join_type {
        JoinType::Inner => streaming(false),
        JoinType::Left | JoinType::Right | JoinType::Full if has_filter => {
            Err(PlanError::Unsupported(format!(
                "{join_type:?} join with a residual filter: the executor applies it after the \
                 outer gather, so a padded row's NULLs drop it (#153)"
            )))
        }
        // The build side is complete before the first probe call, so a probe row unmatched
        // in this batch is unmatched everywhere; only build-side rows need the finish.
        JoinType::Right => streaming(false),
        JoinType::Left | JoinType::Full => streaming(true),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
            // The per-call join disappears: a probe call is only the key project, and the
            // build side is not touched until the finish consumes it. The filtered forms
            // are one legacy call over a probe side the planner made single-batch.
            Ok(JoinCapability {
                probe_streams: !has_filter,
                needs_finish: true,
            })
        }
        JoinType::RightSemi | JoinType::RightAnti if has_filter => {
            Err(PlanError::Unsupported(format!(
                "{join_type:?} join with a residual filter: no swapped mixed_* variant exists — \
                 keeping the emitted side as the build turns it into a Left form"
            )))
        }
        JoinType::RightSemi | JoinType::RightAnti => streaming(false),
    }
}

/// An equi-join: the build side is one batch per lane, the probe streams unless the
/// capability matrix says otherwise, and lane p of each side holds exactly the rows that
/// can match lane p of the other.
#[derive(Debug)]
pub struct GpuJoin {
    kind: NodeKind,
    pub join_type: JoinType,
    /// (build ordinal, probe ordinal) per key, in the order the join hashes them.
    pub keys: Vec<(u32, u32)>,
    pub filter: Option<Expr>,
    pub filter_columns: Vec<JoinFilterColumn>,
    /// From DataFusion, per join: `false` — the SQL default — means a NULL key matches
    /// nothing, `true` is what a set operation lowered to a join needs.
    pub null_equals_null: bool,
    pub projection: Option<Vec<u32>>,
    build: Box<dyn GpuNode>,
    probe: Box<dyn GpuNode>,
}

impl GpuJoin {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        build: Box<dyn GpuNode>,
        probe: Box<dyn GpuNode>,
        join_type: JoinType,
        keys: Vec<(u32, u32)>,
        filter: Option<Expr>,
        filter_columns: Vec<JoinFilterColumn>,
        null_equals_null: bool,
        projection: Option<Vec<u32>>,
        schema: Schema,
    ) -> Self {
        let mut layout = input_layout(probe.as_ref());
        // Co-partitioned, so the lane count survives; nothing else about either input
        // does — the output is the join's own rows in the order its probe batches arrive.
        layout.key_distribution = KeyDistribution::NotSpecified;
        layout.sort_order = SortOrder::NotSpecified;
        layout.batch_layout = BatchLayout::MultipleBatches;
        Self {
            kind: NodeKind::Intermediate { layout, schema },
            join_type,
            keys,
            filter,
            filter_columns,
            null_equals_null,
            projection,
            build,
            probe,
        }
    }

    pub fn capability(&self) -> Result<JoinCapability, PlanError> {
        capability(self.join_type, self.filter.is_some())
    }
}

impl GpuNode for GpuJoin {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        vec![self.build.as_ref(), self.probe.as_ref()]
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        let (build, probe) = (
            input_layout(self.build.as_ref()),
            input_layout(self.probe.as_ref()),
        );
        if build.n != probe.n {
            return Err(PlanError::Invalid(format!(
                "GpuJoin: lane p of one side must hold what can match lane p of the other, \
                 but the sides carry {} and {} lanes",
                build.n, probe.n
            )));
        }
        if build.n > 1 {
            let hashed_on = |layout: &PartitionLayout, keys: Vec<u32>| {
                layout.key_distribution == KeyDistribution::ByHash { hash_keys: keys }
            };
            if !hashed_on(&build, self.keys.iter().map(|(b, _)| *b).collect())
                || !hashed_on(&probe, self.keys.iter().map(|(_, p)| *p).collect())
            {
                return Err(PlanError::Invalid(
                    "GpuJoin: several lanes are co-partitioned only if both sides were \
                     hashed on the join keys, in key order — the planner inserts \
                     GpuEmitPartitions on each side"
                        .to_string(),
                ));
            }
        }
        if build.batch_layout != BatchLayout::SingleBatch {
            return Err(PlanError::Invalid(
                "GpuJoin: its build side must be one batch per lane — the planner inserts \
                 GpuCoalesceAllBatches below it"
                    .to_string(),
            ));
        }
        let capability = self.capability()?;
        if !capability.probe_streams && probe.batch_layout != BatchLayout::SingleBatch {
            return Err(PlanError::Invalid(format!(
                "GpuJoin{{{:?}}} with a residual filter cannot stream its probe — the planner \
                 inserts GpuCoalesceAllBatches below it",
                self.join_type
            )));
        }
        if let Some(filter) = &self.filter {
            check_filter_columns(
                filter,
                &self.filter_columns,
                &input_schema(self.build.as_ref()),
                &input_schema(self.probe.as_ref()),
            )?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
