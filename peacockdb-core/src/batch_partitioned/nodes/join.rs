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
    /// Ordinals into the crossed table, `[build columns…, probe columns…]`. `None` is
    /// every column of it — a `CrossJoinExec` has no projection, and a predicate-free
    /// nested-loop join that lands here may.
    pub projection: Option<Vec<u32>>,
    build: Box<dyn GpuNode>,
    probe: Box<dyn GpuNode>,
}

impl GpuCrossJoin {
    pub fn new(
        build: Box<dyn GpuNode>,
        probe: Box<dyn GpuNode>,
        projection: Option<Vec<u32>>,
        schema: Schema,
    ) -> Self {
        Self {
            kind: NodeKind::Intermediate {
                layout: joined_layout(),
                schema,
            },
            projection,
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
        check_join_inputs("GpuCrossJoin", self.build.as_ref(), self.probe.as_ref())?;
        check_projection(
            "GpuCrossJoin",
            JoinType::Inner,
            self.projection.as_ref(),
            &input_schema(self.build.as_ref()),
            &input_schema(self.probe.as_ref()),
        )
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
    /// Ordinals into the crossed table, as DataFusion computed them. Dropping it leaves
    /// the node declaring the projected columns and emitting all of them, so every
    /// ordinal above it reads one column of some other one (#135).
    pub projection: Option<Vec<u32>>,
    build: Box<dyn GpuNode>,
    probe: Box<dyn GpuNode>,
}

impl GpuNestedLoopJoin {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        build: Box<dyn GpuNode>,
        probe: Box<dyn GpuNode>,
        join_type: NestedLoopJoinType,
        filter: Expr,
        filter_columns: Vec<JoinFilterColumn>,
        projection: Option<Vec<u32>>,
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
            projection,
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
            "GpuNestedLoopJoin",
            &self.filter,
            &self.filter_columns,
            &input_schema(self.build.as_ref()),
            &input_schema(self.probe.as_ref()),
        )?;
        check_projection(
            "GpuNestedLoopJoin",
            match self.join_type {
                NestedLoopJoinType::Inner => JoinType::Inner,
                NestedLoopJoinType::Left => JoinType::Left,
            },
            self.projection.as_ref(),
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

/// The distribution the output carries. A join does not move a row between lanes, so a
/// claim survives — but only the claim of a side whose rows are never padded, and only
/// where that side earned it by being hashed on this join's keys. Minting one instead,
/// from the keys and the output names alone, declares a hash over lanes nothing ever
/// scattered.
///
/// Padding is what decides the side, not hashing: a Left join emits its unmatched build
/// rows with the probe columns NULL, so only the build column holds the value that placed
/// every output row; Right is the mirror; Full pads both ways and so has no such column.
fn joined_key_distribution(
    join_type: JoinType,
    keys: &[(u32, u32)],
    build: &PartitionLayout,
    probe: &PartitionLayout,
    build_width: u32,
    projection: Option<&Vec<u32>>,
) -> KeyDistribution {
    let sides = unpadded_sides(join_type);
    if sides.is_empty() {
        return KeyDistribution::NotSpecified;
    }
    let ordinals = |side: JoinSide| -> Vec<u32> {
        match side {
            JoinSide::Build => keys.iter().map(|(b, _)| *b).collect(),
            JoinSide::Probe => keys.iter().map(|(_, p)| *p).collect(),
        }
    };
    // Every unpadded side must carry the same fact, or the two disagree about where the
    // rows are and neither claim describes the output.
    for side in sides.iter().copied() {
        let layout = match side {
            JoinSide::Build => build,
            JoinSide::Probe => probe,
        };
        if layout.key_distribution
            != (KeyDistribution::ByHash {
                hash_keys: ordinals(side),
            })
        {
            return KeyDistribution::NotSpecified;
        }
    }
    // Re-numbered structurally rather than by name: the join knows its side widths and
    // its projection, so an output ordinal is derivable even where the key name appears
    // twice in the output or not at all.
    for side in sides.iter().copied() {
        let renumbered: Option<Vec<u32>> = ordinals(side)
            .iter()
            .map(|ordinal| output_ordinal(join_type, side, *ordinal, build_width, projection))
            .collect();
        if let Some(hash_keys) = renumbered {
            return KeyDistribution::ByHash { hash_keys };
        }
    }
    KeyDistribution::NotSpecified
}

/// The sides whose rows this join type emits unpadded. Empty for Full, which pads both.
fn unpadded_sides(join_type: JoinType) -> &'static [JoinSide] {
    match join_type {
        JoinType::Inner => &[JoinSide::Build, JoinSide::Probe],
        JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
            &[JoinSide::Build]
        }
        JoinType::Right | JoinType::RightSemi | JoinType::RightAnti => &[JoinSide::Probe],
        JoinType::Full => &[],
    }
}

/// Where one side's column lands in the output: into the emitted table first — which is
/// the crossed pair, or the one side a semi form emits — and then through the projection,
/// where a dropped column has no output ordinal at all.
fn output_ordinal(
    join_type: JoinType,
    side: JoinSide,
    ordinal: u32,
    build_width: u32,
    projection: Option<&Vec<u32>>,
) -> Option<u32> {
    // Exhaustive rather than defaulted: a side the output does not hold has no ordinal,
    // and a join type reaching a catch-all would silently lose a claim it earned.
    let emitted = match (emits(join_type), side) {
        (Emits::BothSides, JoinSide::Build) => ordinal,
        (Emits::BothSides, JoinSide::Probe) => build_width + ordinal,
        (Emits::BuildSide | Emits::BuildSideAndMark, JoinSide::Build) => ordinal,
        (Emits::ProbeSide, JoinSide::Probe) => ordinal,
        (Emits::BuildSide | Emits::BuildSideAndMark, JoinSide::Probe) => return None,
        (Emits::ProbeSide, JoinSide::Build) => return None,
    };
    match projection {
        Some(columns) => columns
            .iter()
            .position(|column| *column == emitted)
            .map(|at| at as u32),
        None => Some(emitted),
    }
}

/// Which columns a join type emits, before any projection of its own. A semi or anti join
/// is a filter written as a join — it emits the side it decides about — and a mark join
/// emits that side plus the boolean it computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Emits {
    BothSides,
    BuildSide,
    /// The build side and one more column, the mark.
    BuildSideAndMark,
    ProbeSide,
}

fn emits(join_type: JoinType) -> Emits {
    match join_type {
        JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => Emits::BothSides,
        JoinType::LeftSemi | JoinType::LeftAnti => Emits::BuildSide,
        JoinType::LeftMark => Emits::BuildSideAndMark,
        JoinType::RightSemi | JoinType::RightAnti => Emits::ProbeSide,
    }
}

/// One lane, many batches, no order and no key: cross and nested-loop joins have no key to
/// co-locate on, so nothing about the inputs' layout survives them.
fn joined_layout() -> PartitionLayout {
    PartitionLayout {
        n: 1,
        key_distribution: KeyDistribution::NotSpecified,
        sort_order: SortOrder::NotSpecified,
        batch_layout: BatchLayout::MultipleBatches,
    }
}

/// A key pair is one ordinal into each side. An out-of-range one does not merely read the
/// wrong column: `joined_key_distribution` cannot name it, so the node quietly declares no
/// distribution at all and a co-partitioned join above it loses the claim it earned.
fn check_keys(keys: &[(u32, u32)], build: &Schema, probe: &Schema) -> Result<(), PlanError> {
    if keys.is_empty() {
        return Err(PlanError::Invalid(
            "GpuJoin: an equi-join with no keys is a cross join — the planner emits \
             GpuCrossJoin for that shape"
                .to_string(),
        ));
    }
    for (side, schema, ordinals) in [
        (
            "build",
            build,
            keys.iter().map(|(b, _)| *b).collect::<Vec<_>>(),
        ),
        (
            "probe",
            probe,
            keys.iter().map(|(_, p)| *p).collect::<Vec<_>>(),
        ),
    ] {
        let width = schema.fields.fields().len();
        for ordinal in ordinals {
            if ordinal as usize >= width {
                return Err(PlanError::Invalid(format!(
                    "GpuJoin: {side} key @{ordinal} is past the {width} columns that side has"
                )));
            }
        }
    }
    Ok(())
}

/// A projected ordinal indexes the table the join emits before its own projection — both
/// sides for the four that pair rows, one side for a semi or anti join, and that side plus
/// the mark for a mark join. Bounding it by both sides instead would leave the five
/// one-sided types checked against a table wider than the one they have.
fn check_projection(
    node: &str,
    join_type: JoinType,
    projection: Option<&Vec<u32>>,
    build: &Schema,
    probe: &Schema,
) -> Result<(), PlanError> {
    let width = emitted_columns(
        join_type,
        build.fields.fields().len(),
        probe.fields.fields().len(),
    );
    for ordinal in projection.into_iter().flatten() {
        if *ordinal as usize >= width {
            return Err(PlanError::Invalid(format!(
                "{node}: projected column @{ordinal} is past the {width} a {join_type:?} join \
                 emits"
            )));
        }
    }
    Ok(())
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
    node: &str,
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
                "{node}: filter column {}@{} is past the {} its map has",
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
                    "{node}: filter column {}@{} maps past its side's columns",
                    reference.name, reference.index
                ))
            })?;
        if field.name() != &reference.name {
            return Err(PlanError::Invalid(format!(
                "{node}: filter column {}@{} maps to {} on the {:?} side",
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

impl JoinCapability {
    /// Whether the whole join is one call: no probe keys kept and no finish pass.
    ///
    /// True where nothing needs a finish, and also where the probe cannot stream — the
    /// planner makes that probe a single batch, and one call over the whole of it is the
    /// same legacy node the other modes emit. Asked by the recipe writer for what to
    /// publish and by an executor for what to build, because answering it twice is how a
    /// filtered semi join ended up on the finish path with its residual dropped.
    pub fn answers_in_one_call(&self) -> bool {
        !self.probe_streams || !self.needs_finish
    }
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
                 keeping the emitted side as the build turns it into a Left form (#159)"
            )))
        }
        JoinType::RightSemi | JoinType::RightAnti => streaming(false),
    }
}

/// What the per-probe-batch join emits, which is not what the node is: a Left emits this
/// batch's matches and waits for the finish, and a Full also emits the probe rows this
/// batch had no match for — batch-local, because the build side was complete before the
/// first call.
///
/// `None` is the build-side semi family, whose probe call is only the key project.
pub fn per_call_join_type(join_type: JoinType) -> Option<JoinType> {
    match join_type {
        JoinType::Left => Some(JoinType::Inner),
        JoinType::Full => Some(JoinType::Right),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => None,
        other => unreachable!("{other:?} needs no finish pass"),
    }
}

/// The join the finish pass runs against the accumulated keys. Left and Full ask which
/// build rows nothing ever matched; the semi family asks its own question, and asks it
/// with the node's own NULL semantics, so the pass substitutes for a legacy single call
/// rather than improving on it (#59, #80).
pub fn finish_join_type(join_type: JoinType) -> JoinType {
    match join_type {
        JoinType::Left | JoinType::Full => JoinType::LeftAnti,
        semi @ (JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark) => semi,
        other => unreachable!("{other:?} needs no finish pass"),
    }
}

/// How many columns a join emits before any projection of its own — the count half of
/// [`emits`], which is the same fact its carry-over rule reads.
pub(crate) fn emitted_columns(join_type: JoinType, build: usize, probe: usize) -> usize {
    match emits(join_type) {
        Emits::BothSides => build + probe,
        Emits::BuildSide => build,
        Emits::BuildSideAndMark => build + 1,
        Emits::ProbeSide => probe,
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
        let build_layout = input_layout(build.as_ref());
        let mut layout = input_layout(probe.as_ref());
        // The lane count survives, and so does a hash the input earned: a join does not
        // move a row between lanes. Dropping that is what makes a correct plan fail the
        // co-location guard above an aggregate.
        layout.key_distribution = joined_key_distribution(
            join_type,
            &keys,
            &build_layout,
            &layout,
            input_schema(build.as_ref()).fields.fields().len() as u32,
            projection.as_ref(),
        );
        // The output is the join's own rows in the order its probe batches arrive.
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
                "GpuJoin",
                filter,
                &self.filter_columns,
                &input_schema(self.build.as_ref()),
                &input_schema(self.probe.as_ref()),
            )?;
        }
        let (build_schema, probe_schema) = (
            input_schema(self.build.as_ref()),
            input_schema(self.probe.as_ref()),
        );
        check_keys(&self.keys, &build_schema, &probe_schema)?;
        check_projection(
            "GpuJoin",
            self.join_type,
            self.projection.as_ref(),
            &build_schema,
            &probe_schema,
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
