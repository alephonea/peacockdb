//! Validation of a whole tree: each node against its children, then against itself.
//!
//! The two halves answer different questions. A node's own
//! [`validate_schemas_and_partitions`](GpuNode::validate_schemas_and_partitions) judges
//! what its children hand it and can name the node that fixes it. The structural rules
//! here judge what a node says about its *own* output — the ordinals in its layout and
//! its annotations — which no child can be blamed for and which no node-local check
//! looks at.

use datafusion::arrow::datatypes::Schema as ArrowSchema;

use super::error::PlanError;
use super::layout::{KeyDistribution, NodeKind, SortOrder};
use super::node::GpuNode;
use super::nodes::join::emitted_columns;
use super::nodes::{AggregateBody, GpuCrossJoin, GpuNestedLoopJoin, NodeRef, try_as_node_ref};
use super::schema::Schema;

/// Post-order, so a child's complaint comes before its parent's.
pub(super) fn validate(root: &dyn GpuNode) -> Result<(), PlanError> {
    if !matches!(root.kind(), NodeKind::Sink) {
        return Err(PlanError::Invalid(format!(
            "{}: a plan ends at the crossing back to the host — the planner roots it in \
             GpuUnload",
            root.name()
        )));
    }
    check_canonical_form(root)?;
    walk(root)
}

/// Which lowering a limit got is a question about position, and a node cannot see what is
/// above it: root-adjacent means the interval belongs to `GpuUnload`, so a limit node whose
/// only consumer is the sink is a tree the planner does not emit.
fn limit_positions(node: &dyn GpuNode, parent_is_sink: bool) -> Result<(), PlanError> {
    if node.row_interval().is_some() && parent_is_sink {
        return Err(PlanError::Invalid(format!(
            "{}: a limit feeding only the sink is not a node — the planner puts its \
             skip/fetch on GpuUnload, so the driver can release the batches it does not want \
             instead of unloading them and throwing the rows away",
            node.name()
        )));
    }
    let is_sink = matches!(node.kind(), NodeKind::Sink);
    for child in node.children() {
        limit_positions(child, is_sink)?;
    }
    Ok(())
}

/// The canonical-form rules a driver needs to have been applied before it runs, so a mock
/// plan meets the same refusal a planned one would.
pub(crate) fn check_canonical_form(root: &dyn GpuNode) -> Result<(), PlanError> {
    limit_positions(root, false)
}

fn walk(node: &dyn GpuNode) -> Result<(), PlanError> {
    for child in node.children() {
        walk(child)?;
    }
    node.validate_schemas_and_partitions()?;
    structural(node)?;
    declared_width(node)?;
    types_across_the_edge(node)?;
    earned_claims(node)
}

/// Where a declaration must have come from. A hash is made by exactly one node and an
/// order by three, so a claim with no such node beneath it was minted rather than carried
/// — the shape I4 was: a join deriving `ByHash` from its keys and its column names, over
/// lanes nothing had scattered.
///
/// Subtree presence, not a path: which nodes may carry a claim past themselves is a rule
/// each node states about its own output, and this is the cross-check that the chain
/// starts somewhere real.
fn earned_claims(node: &dyn GpuNode) -> Result<(), PlanError> {
    let Some(layout) = node.kind().layout() else {
        return Ok(());
    };
    let name = node.name();
    if matches!(layout.key_distribution, KeyDistribution::ByHash { .. })
        && !below(node, &|found| {
            matches!(try_as_node_ref(found), Some(NodeRef::EmitPartitions(_)))
        })
    {
        return Err(PlanError::Invalid(format!(
            "{name}: it declares rows placed by a hash, and nothing below it scattered them \
             — only GpuEmitPartitions does"
        )));
    }
    if layout.sort_order.is_batch_sorted()
        && !below(node, &|found| {
            matches!(
                try_as_node_ref(found),
                Some(
                    NodeRef::Sort(_)
                        | NodeRef::AccumulateBatchesAndSort(_)
                        | NodeRef::MergeSortedPartitions(_)
                )
            )
        })
    {
        return Err(PlanError::Invalid(format!(
            "{name}: it declares sorted batches, and nothing below it sorted any"
        )));
    }
    Ok(())
}

/// The node itself or anything under it — a sort declares its own order, and a scatter its
/// own hash.
fn below(node: &dyn GpuNode, accept: &dyn Fn(&dyn GpuNode) -> bool) -> bool {
    accept(node) || node.children().iter().any(|child| below(*child, accept))
}

/// What the plan promised its caller. The layer's contract is that the rows it hands back
/// are the ones DataFusion planned, and nothing else states it: every node below is
/// checked against its own children, so a whole tree can be internally consistent and
/// answer a different query.
pub(super) fn check_output_schema(
    root: &dyn GpuNode,
    planned: &ArrowSchema,
) -> Result<(), PlanError> {
    let emitted = root
        .children()
        .first()
        .and_then(|input| input.kind().schema())
        .expect("a sink has an input");
    if emitted.fields.fields().len() != planned.fields().len() {
        return Err(PlanError::Invalid(format!(
            "the plan emits {} columns and the query asked for {}",
            emitted.fields.fields().len(),
            planned.fields().len()
        )));
    }
    let mismatch = emitted
        .fields
        .fields()
        .iter()
        .zip(planned.fields().iter())
        .find(|(ours, theirs)| {
            ours.name() != theirs.name() || ours.data_type() != theirs.data_type()
        });
    if let Some((ours, theirs)) = mismatch {
        return Err(PlanError::Invalid(format!(
            "the plan emits {} as {:?} where the query asked for {} as {:?}",
            ours.name(),
            ours.data_type(),
            theirs.name(),
            theirs.data_type()
        )));
    }
    Ok(())
}

/// A node's declared column count against the parameters that produce those columns. The
/// two are written in different places — the schema comes from DataFusion, the parameters
/// from this layer's own rebasing — and a node emitting a different number of columns from
/// the one it declares shifts every ordinal above it.
fn declared_width(node: &dyn GpuNode) -> Result<(), PlanError> {
    let Some(schema) = node.kind().schema() else {
        return Ok(());
    };
    let declared = schema.fields.fields().len();
    let width_of = |input: &dyn GpuNode| {
        input
            .kind()
            .schema()
            .expect("a sink cannot be an input")
            .fields
            .fields()
            .len()
    };
    // A node the registry does not know is a hand-built one under test, and declares
    // none of the parameters this rule reads.
    let Some(kind) = try_as_node_ref(node) else {
        return Ok(());
    };
    let (emitted, from) = match kind {
        NodeRef::LoadParquet(load) => (load.projection.len(), "the columns it reads"),
        NodeRef::Project(project) => (project.exprs.len(), "its expression list"),
        NodeRef::Filter(filter) => match &filter.projection {
            Some(columns) => (columns.len(), "its projection"),
            None => (width_of(node.children()[0]), "its input"),
        },
        NodeRef::Join(join) => match &join.projection {
            Some(columns) => (columns.len(), "its projection"),
            None => (
                emitted_columns(
                    join.join_type,
                    width_of(node.children()[0]),
                    width_of(node.children()[1]),
                ),
                "the sides its join type emits",
            ),
        },
        NodeRef::CrossJoin(GpuCrossJoin {
            projection: Some(columns),
            ..
        })
        | NodeRef::NestedLoopJoin(GpuNestedLoopJoin {
            projection: Some(columns),
            ..
        }) => (columns.len(), "its projection"),
        NodeRef::CrossJoin(_) | NodeRef::NestedLoopJoin(_) => (
            width_of(node.children()[0]) + width_of(node.children()[1]),
            "its two sides",
        ),
        NodeRef::Aggregate(aggregate) => aggregate_width(&aggregate.body),
        NodeRef::AggregateBatches(aggregate) => aggregate_width(&aggregate.body),
        // The rest emit their input's columns, which `types_across_the_edge` compares
        // field for field — a stronger statement than a count.
        _ => return Ok(()),
    };
    if declared != emitted {
        return Err(PlanError::Invalid(format!(
            "{}: it declares {declared} columns and {from} produces {emitted}",
            node.name()
        )));
    }
    Ok(())
}

/// Group keys, then what the node emits per aggregate: its state columns where it hands
/// state on, and one finalized column each where it finishes. A grouping-set expansion
/// emits `__grouping_id` beside the keys, which is a key everywhere above it.
fn aggregate_width(body: &AggregateBody) -> (usize, &'static str) {
    let keys = body.group_by.len() + usize::from(!body.grouping_sets.is_empty());
    match &body.finalize {
        Some(finalize) => (keys + finalize.len(), "its keys and its finalize list"),
        None => (
            keys + body
                .aggs
                .iter()
                .map(|call| call.outputs.len())
                .sum::<usize>(),
            "its keys and its state columns",
        ),
    }
}

/// A node that moves rows rather than changing them emits its input's columns, so the two
/// schemas must be the same fields and not merely the same count. Types are the half no
/// name check and no per-node byte count can see: both engines derive their bytes from
/// this same declaration, so a column that changed type across an edge costs the same on
/// either and surfaces only in the answer.
///
/// It covers the nodes that CARRY a column. Deriving the type a computed column would
/// actually have — a project's expression, an aggregate's state — and checking the
/// declaration against it is [#163](../../../llm-wiki/tickets.md).
fn types_across_the_edge(node: &dyn GpuNode) -> Result<(), PlanError> {
    let carried: &dyn GpuNode = match try_as_node_ref(node) {
        Some(
            NodeRef::Sort(_)
            | NodeRef::CoalesceAllBatches(_)
            | NodeRef::AccumulateBatchesAndSort(_)
            | NodeRef::Limit(_)
            | NodeRef::MergePartitions(_)
            | NodeRef::EmitPartitions(_)
            | NodeRef::MergeSortedPartitions(_),
        ) => node.children()[0],
        // A filter that projects re-selects columns rather than carrying them; its
        // projection is checked by width above and by ordinal at the node.
        Some(NodeRef::Filter(filter)) if filter.projection.is_none() => node.children()[0],
        _ => return Ok(()),
    };
    let (ours, theirs) = (
        node.kind().schema().expect("not a sink"),
        carried.kind().schema().expect("a sink cannot be an input"),
    );
    if let Some((ours, theirs)) = ours
        .fields
        .fields()
        .iter()
        .zip(theirs.fields.fields().iter())
        .find(|(ours, theirs)| {
            ours.name() != theirs.name() || ours.data_type() != theirs.data_type()
        })
    {
        return Err(PlanError::Invalid(format!(
            "{}: it moves rows rather than changing them, and declares {} as {:?} where its \
             input holds {} as {:?}",
            node.name(),
            ours.name(),
            ours.data_type(),
            theirs.name(),
            theirs.data_type()
        )));
    }
    Ok(())
}

fn structural(node: &dyn GpuNode) -> Result<(), PlanError> {
    let name = node.name();
    let children = node.children();
    for child in &children {
        if matches!(child.kind(), NodeKind::Sink) {
            return Err(PlanError::Invalid(format!(
                "{name}: its input is a {}, whose output has already crossed to the host",
                child.name()
            )));
        }
    }
    match (node.kind(), children.is_empty()) {
        (NodeKind::Source { .. }, false) => {
            return Err(PlanError::Invalid(format!(
                "{name}: a source reads a table, so it takes no input"
            )));
        }
        (NodeKind::Intermediate { .. }, true) => {
            return Err(PlanError::Invalid(format!(
                "{name}: nothing produces the rows it declares — only a source has no input"
            )));
        }
        _ => {}
    }

    let (Some(layout), Some(schema)) = (node.kind().layout(), node.kind().schema()) else {
        return Ok(());
    };
    if layout.n == 0 {
        return Err(PlanError::Invalid(format!(
            "{name}: no lanes, so it declares rows nothing can read"
        )));
    }
    let columns = schema.fields.fields().len();
    let in_range = |ordinal: u32, what: &str| -> Result<(), PlanError> {
        if ordinal as usize >= columns {
            return Err(PlanError::Invalid(format!(
                "{name}: {what} @{ordinal} is past the {columns} columns it emits"
            )));
        }
        Ok(())
    };

    if let KeyDistribution::ByHash { hash_keys } = &layout.key_distribution {
        for key in hash_keys {
            in_range(*key, "hash key")?;
        }
    }
    if let SortOrder::BatchSorted { columns } = &layout.sort_order {
        for order in columns {
            in_range(order.column, "sort key")?;
        }
    }
    for key in &schema.group_keys {
        in_range(*key, "group key")?;
    }
    annotated_state(name, schema, &in_range)
}

/// An aggregate state annotation names the columns it decomposed into, so those columns
/// have to be there: a merge reads the positions rather than re-deriving them, and a
/// position past the output would read whatever the next node emits at that ordinal.
fn annotated_state(
    name: &str,
    schema: &Schema,
    in_range: &dyn Fn(u32, &str) -> Result<(), PlanError>,
) -> Result<(), PlanError> {
    for state in &schema.agg_state {
        if state.positions.is_empty() {
            return Err(PlanError::Invalid(format!(
                "{name}: {} declares aggregate state in no column",
                state.output
            )));
        }
        for position in &state.positions {
            in_range(*position, &format!("{} state", state.output))?;
        }
        if state
            .positions
            .iter()
            .any(|position| schema.group_keys.contains(position))
        {
            return Err(PlanError::Invalid(format!(
                "{name}: {} declares its state in a column that is also a group key",
                state.output
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_partitioned::aggregates::AggFunc;
    use crate::batch_partitioned::expr::{Expr, NamedExpr};
    use crate::batch_partitioned::layout::{BatchLayout, ColumnOrder, PartitionLayout, SortOrder};
    use crate::batch_partitioned::node::RowInterval;
    use crate::batch_partitioned::nodes::{
        GpuFilter, GpuLimit, GpuMergePartitions, GpuProject, GpuUnload,
    };
    use crate::batch_partitioned::schema::AggStateColumns;
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use std::any::Any;
    use std::sync::Arc;

    /// A node whose kind the test writes: the structural rules are about what a node
    /// declares of itself, and the translation layer is what makes those declarations
    /// agree with the schema — so a plan that breaks one is unreachable from sql.
    #[derive(Debug)]
    struct Declaring {
        kind: NodeKind,
        children: Vec<Box<dyn GpuNode>>,
    }

    impl GpuNode for Declaring {
        fn kind(&self) -> &NodeKind {
            &self.kind
        }

        fn name(&self) -> &'static str {
            "GpuDeclaring"
        }

        fn children(&self) -> Vec<&dyn GpuNode> {
            self.children.iter().map(|c| c.as_ref()).collect()
        }

        fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
            Ok(())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn schema_of(columns: &[&str]) -> Schema {
        Schema::new(Arc::new(ArrowSchema::new(
            columns
                .iter()
                .map(|name| Field::new(*name, DataType::Int64, true))
                .collect::<Vec<Field>>(),
        )))
    }

    /// A source, since a leaf that is not one is itself a finding below.
    fn source(schema: Schema, layout: PartitionLayout) -> Box<dyn GpuNode> {
        Box::new(Declaring {
            kind: NodeKind::Source { layout, schema },
            children: Vec::new(),
        })
    }

    fn plain_source() -> Box<dyn GpuNode> {
        source(schema_of(&["a"]), PartitionLayout::new(1))
    }

    /// The tree as the planner shapes it: whatever is given, under an unload.
    fn rooted(input: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
        Box::new(GpuUnload::new(input, None))
    }

    fn invalid(result: Result<(), PlanError>, mentions: &str) {
        match result {
            Err(PlanError::Invalid(what)) => assert!(
                what.contains(mentions),
                "the error names the wrong thing: {what}"
            ),
            other => panic!("expected an invalid plan naming {mentions}, got {other:?}"),
        }
    }

    #[test]
    fn a_tree_the_planner_shapes_passes_both_halves() {
        let tree = rooted(Box::new(GpuMergePartitions::new(plain_source())));
        assert_eq!(validate(tree.as_ref()), Ok(()));
    }

    #[test]
    fn a_limit_whose_only_consumer_is_the_sink_is_refused() {
        // Root-adjacent is the other lowering: the interval belongs to the unload, which is
        // what lets the driver release a batch it wants none of rather than move it and
        // throw the rows away.
        let tree = rooted(Box::new(GpuLimit::new(
            plain_source(),
            RowInterval {
                skip: 0,
                fetch: Some(5),
            },
        )));
        invalid(
            validate(tree.as_ref()),
            "a limit feeding only the sink is not a node",
        );
        invalid(
            check_canonical_form(tree.as_ref()),
            "a limit feeding only the sink is not a node",
        );
    }

    #[test]
    fn a_limit_with_a_real_consumer_above_it_is_the_shape_that_passes() {
        let tree = rooted(Box::new(GpuMergePartitions::new(Box::new(GpuLimit::new(
            plain_source(),
            RowInterval {
                skip: 2,
                fetch: Some(5),
            },
        )))));
        assert_eq!(check_canonical_form(tree.as_ref()), Ok(()));
        assert_eq!(validate(tree.as_ref()), Ok(()));
    }

    #[test]
    fn the_unloads_own_interval_is_the_root_adjacent_lowering_and_not_a_finding() {
        let tree: Box<dyn GpuNode> = Box::new(GpuUnload::new(
            plain_source(),
            Some(RowInterval {
                skip: 3,
                fetch: Some(20),
            }),
        ));
        assert_eq!(check_canonical_form(tree.as_ref()), Ok(()));
    }

    #[test]
    fn a_plan_that_does_not_end_in_a_crossing_is_refused() {
        invalid(validate(plain_source().as_ref()), "GpuUnload");
    }

    #[test]
    fn a_sink_below_the_root_is_refused() {
        // Two unloads: the inner one already moved its rows to the host, so the outer
        // one is reading something that is no longer on the device.
        let tree = rooted(rooted(plain_source()));
        invalid(validate(tree.as_ref()), "already crossed to the host");
    }

    #[test]
    fn an_intermediate_with_no_input_is_refused() {
        let leaf = Box::new(Declaring {
            kind: NodeKind::Intermediate {
                layout: PartitionLayout::new(1),
                schema: schema_of(&["a"]),
            },
            children: Vec::new(),
        });
        invalid(
            validate(rooted(leaf).as_ref()),
            "only a source has no input",
        );
    }

    #[test]
    fn a_source_with_an_input_is_refused() {
        let loaded = Box::new(Declaring {
            kind: NodeKind::Source {
                layout: PartitionLayout::new(1),
                schema: schema_of(&["a"]),
            },
            children: vec![plain_source()],
        });
        invalid(validate(rooted(loaded).as_ref()), "takes no input");
    }

    #[test]
    fn a_node_declaring_no_lanes_is_refused() {
        let empty = source(schema_of(&["a"]), PartitionLayout::new(0));
        invalid(validate(rooted(empty).as_ref()), "no lanes");
    }

    #[test]
    fn a_hash_key_past_the_columns_it_emits_is_refused() {
        let layout = PartitionLayout {
            key_distribution: KeyDistribution::ByHash { hash_keys: vec![3] },
            ..PartitionLayout::new(4)
        };
        // The node-local checks read a node's keys against its input; this one reads the
        // claim it makes about its own output, which nothing else looks at.
        let node = source(schema_of(&["a", "b"]), layout);
        invalid(
            validate(rooted(node).as_ref()),
            "hash key @3 is past the 2 columns",
        );
    }

    #[test]
    fn a_sort_key_past_the_columns_it_emits_is_refused() {
        let layout = PartitionLayout {
            sort_order: SortOrder::batch_sorted(vec![ColumnOrder {
                column: 2,
                ascending: true,
                nulls_first: false,
            }]),
            batch_layout: BatchLayout::SingleBatch,
            ..PartitionLayout::new(1)
        };
        let node = source(schema_of(&["a", "b"]), layout);
        invalid(
            validate(rooted(node).as_ref()),
            "sort key @2 is past the 2 columns",
        );
    }

    #[test]
    fn a_group_key_past_the_columns_it_emits_is_refused() {
        let mut schema = schema_of(&["k", "n"]);
        schema.group_keys = vec![0, 5];
        let node = source(schema, PartitionLayout::new(1));
        invalid(
            validate(rooted(node).as_ref()),
            "group key @5 is past the 2 columns",
        );
    }

    fn state_at(positions: Vec<u32>) -> AggStateColumns {
        AggStateColumns {
            output: "avg(l_quantity)".to_string(),
            func: AggFunc::Avg,
            ddof: 0,
            positions,
        }
    }

    #[test]
    fn a_state_column_past_the_columns_it_emits_is_refused() {
        let mut schema = schema_of(&["k", "avg$sum", "avg$count"]);
        schema.group_keys = vec![0];
        schema.agg_state = vec![state_at(vec![1, 4])];
        let node = source(schema, PartitionLayout::new(1));
        invalid(
            validate(rooted(node).as_ref()),
            "avg(l_quantity) state @4 is past the 3 columns",
        );
    }

    #[test]
    fn an_aggregate_declaring_state_in_no_column_is_refused() {
        let mut schema = schema_of(&["k"]);
        schema.group_keys = vec![0];
        schema.agg_state = vec![state_at(Vec::new())];
        let node = source(schema, PartitionLayout::new(1));
        invalid(validate(rooted(node).as_ref()), "state in no column");
    }

    #[test]
    fn state_declared_in_a_group_key_is_refused() {
        // The two overlap only if the positions were derived against a different column
        // order, and a merge reading a key as state would merge the thing it groups by.
        let mut schema = schema_of(&["k", "avg$sum"]);
        schema.group_keys = vec![0];
        schema.agg_state = vec![state_at(vec![0, 1])];
        let node = source(schema, PartitionLayout::new(1));
        invalid(
            validate(rooted(node).as_ref()),
            "state in a column that is also a group key",
        );
    }

    #[test]
    fn a_hash_nothing_below_scattered_is_refused() {
        // The shape a join minted before it read its children: lanes hashed on a column
        // nothing ever scattered by, which a co-partitioned join above would then trust.
        let claiming = source(
            schema_of(&["k"]),
            PartitionLayout {
                key_distribution: KeyDistribution::ByHash { hash_keys: vec![0] },
                ..PartitionLayout::new(4)
            },
        );
        invalid(
            validate(rooted(claiming).as_ref()),
            "nothing below it scattered them",
        );
    }

    #[test]
    fn an_order_nothing_below_sorted_is_refused() {
        let claiming = source(
            schema_of(&["a"]),
            PartitionLayout {
                sort_order: SortOrder::batch_sorted(vec![ColumnOrder {
                    column: 0,
                    ascending: true,
                    nulls_first: false,
                }]),
                ..PartitionLayout::new(1)
            },
        );
        invalid(
            validate(rooted(claiming).as_ref()),
            "nothing below it sorted any",
        );
    }

    /// A schema of one column, of the type given: the pair below is Int64 against
    /// Decimal128(15,2), because a decimal read as its neighbour is what these rules
    /// exist to catch and what no per-node byte count can show.
    fn one_column(name: &str, data_type: DataType) -> Schema {
        Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
            name, data_type, true,
        )])))
    }

    #[test]
    fn a_column_that_changes_type_across_an_edge_is_refused() {
        // A filter drops rows and keeps columns, so declaring a decimal where its input
        // holds an integer is a claim about a column nobody produced. Both engines derive
        // their per-node bytes from this same declaration, so it costs the same on either
        // and shows up only in the answer.
        let input = source(one_column("a", DataType::Int64), PartitionLayout::new(1));
        let filter = Box::new(GpuFilter::new(
            input,
            Expr::column(0, "a"),
            None,
            one_column("a", DataType::Decimal128(15, 2)),
        ));
        invalid(
            validate(rooted(filter).as_ref()),
            "declares a as Decimal128(15, 2) where its input holds a as Int64",
        );
    }

    #[test]
    fn a_node_declaring_more_columns_than_it_produces_is_refused() {
        let input = source(one_column("a", DataType::Int64), PartitionLayout::new(1));
        let project = Box::new(GpuProject::new(
            input,
            vec![NamedExpr::new(Expr::column(0, "a"), "a")],
            Schema::new(Arc::new(ArrowSchema::new(vec![
                Field::new("a", DataType::Int64, true),
                Field::new("b", DataType::Int64, true),
            ]))),
        ));
        invalid(
            validate(rooted(project).as_ref()),
            "it declares 2 columns and its expression list produces 1",
        );
    }

    #[test]
    fn an_aggregate_declaring_a_width_its_body_does_not_produce_is_refused() {
        // Keys plus state where it hands state on. The aggregate is the node whose schema
        // carries the annotations a merge reads, so a width slip there mis-numbers the
        // state columns rather than only the output.
        use crate::batch_partitioned::aggregates::{AggCall, PlanAgg};
        use crate::batch_partitioned::nodes::GpuAggregate;

        let input = source(one_column("n", DataType::Int64), PartitionLayout::new(1));
        let body = AggregateBody {
            group_by: Vec::new(),
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![AggCall {
                func: PlanAgg::Sum,
                args: vec![Expr::column(0, "n")],
                outputs: vec![Field::new("n", DataType::Int64, true)],
            }],
            finalize: None,
        };
        let declared = Schema::new(Arc::new(ArrowSchema::new(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("spare", DataType::Int64, true),
        ])));
        let aggregate = Box::new(GpuAggregate::new(input, body, declared.clone(), declared));
        invalid(
            validate(rooted(aggregate).as_ref()),
            "it declares 2 columns and its keys and its state columns produces 1",
        );
    }

    #[test]
    fn a_plan_emitting_a_different_column_count_from_the_query_is_refused() {
        let tree = rooted(source(
            one_column("a", DataType::Int64),
            PartitionLayout::new(1),
        ));
        let asked = ArrowSchema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
        ]);
        invalid(
            check_output_schema(tree.as_ref(), &asked),
            "the plan emits 1 columns and the query asked for 2",
        );
    }

    #[test]
    fn a_plan_emitting_a_column_of_another_type_than_the_query_is_refused() {
        // The layer's contract with its caller: every node below is checked against its
        // own children, so a tree can be internally consistent and answer in a type the
        // query did not ask for.
        let tree = rooted(source(
            one_column("total", DataType::Int64),
            PartitionLayout::new(1),
        ));
        let asked = ArrowSchema::new(vec![Field::new("total", DataType::Decimal128(15, 2), true)]);
        invalid(
            check_output_schema(tree.as_ref(), &asked),
            "the plan emits total as Int64 where the query asked for total as Decimal128(15, 2)",
        );
    }

    #[test]
    fn a_childs_complaint_comes_before_its_parents() {
        // Post-order, so the deepest defect is the one reported: a parent's message names
        // a fix that would not help if its input is already wrong.
        let broken = source(schema_of(&["a"]), PartitionLayout::new(0));
        let tree = rooted(Box::new(GpuMergePartitions::new(broken)));
        invalid(validate(tree.as_ref()), "no lanes");
    }
}
