//! Rebuilding a plan node, and the fixtures that prove the rebuild keeps every field.
//!
//! Rebuild rather than edit, for the reason the prototype's `LayoutInjector` records: a
//! node's partitioning is not a field but a value its constructor computed from the child
//! it was handed, so a node with a new child has to be built again.
//!
//! Three levels, each catching what the one below cannot: the exhaustive match makes a new
//! kind a compile error, the fixtures are asserted to reach every variant, and
//! [`fields_with_one_value`] names a field no fixture varies — the one a later arm could
//! drop unseen.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::{JoinType, ScalarValue};

use peacockdb_core::batch_partitioned::GpuNode;
use peacockdb_core::batch_partitioned::aggregates::{AggCall, AggFunc, PlanAgg, decomposition};
use peacockdb_core::batch_partitioned::expr::{Expr, NamedExpr};
use peacockdb_core::batch_partitioned::layout::ColumnOrder;
use peacockdb_core::batch_partitioned::node::RowInterval;
use peacockdb_core::batch_partitioned::nodes::join::{
    JoinFilterColumn, JoinSide, NestedLoopJoinType,
};
use peacockdb_core::batch_partitioned::nodes::{
    AggregateBody, GpuAccumulateBatchesAndSort, GpuAggregate, GpuAggregateBatches,
    GpuCoalesceAllBatches, GpuCrossJoin, GpuEmitPartitions, GpuFilter, GpuInterleave, GpuJoin,
    GpuLimit, GpuLoadParquet, GpuMergePartitions, GpuMergeSortedPartitions, GpuNestedLoopJoin,
    GpuProject, GpuSort, GpuUnion, GpuUnload, NodeRef, as_node_ref,
};
use peacockdb_core::batch_partitioned::parquet_meta::ScanMetadata;
use peacockdb_core::batch_partitioned::partitioner::RowGroupMeta;
use peacockdb_core::batch_partitioned::schema::Schema;

/// `node` rebuilt over `children`, which are the rewritten children in the order
/// [`GpuNode::children`] reports them. Handed a node's own children back it is the
/// identity, which is the property everything else here rests on.
pub fn rebuild(node: &dyn GpuNode, children: Vec<Box<dyn GpuNode>>) -> Box<dyn GpuNode> {
    let name = node.name();
    let mut children = children.into_iter();
    let mut one = || {
        children
            .next()
            .unwrap_or_else(|| panic!("{name} was rebuilt with no child to take"))
    };
    match as_node_ref(node) {
        NodeRef::LoadParquet(load) => {
            let scan = scan_of(load);
            Box::new(GpuLoadParquet::new(
                load.table.clone(),
                load.projection.clone(),
                load.partition_groups.clone(),
                &scan,
                load.limit,
                schema_of(node),
            ))
        }
        NodeRef::Filter(filter) => Box::new(GpuFilter::new(
            one(),
            filter.predicate.clone(),
            filter.projection.clone(),
            schema_of(node),
        )),
        NodeRef::Project(project) => Box::new(GpuProject::new(
            one(),
            project.exprs.clone(),
            schema_of(node),
        )),
        NodeRef::Sort(sort) => Box::new(GpuSort::new(one(), sort.keys.clone(), sort.fetch)),
        NodeRef::CoalesceAllBatches(_) => Box::new(GpuCoalesceAllBatches::new(one())),
        NodeRef::AccumulateBatchesAndSort(sort) => Box::new(GpuAccumulateBatchesAndSort::new(
            one(),
            sort.keys.clone(),
            sort.fetch,
        )),
        NodeRef::Limit(limit) => Box::new(GpuLimit::new(one(), limit.interval)),
        NodeRef::Aggregate(aggregate) => Box::new(GpuAggregate::new(
            one(),
            body_of(&aggregate.body),
            aggregate.intermediate().clone(),
            schema_of(node),
        )),
        NodeRef::AggregateBatches(merge) => Box::new(GpuAggregateBatches::new(
            one(),
            body_of(&merge.body),
            merge.intermediate().clone(),
            schema_of(node),
        )),
        NodeRef::Join(join) => {
            // Named rather than two calls in argument position: both sides are the same
            // type, so a swap is a wrong plan the compiler cannot see.
            let (build, probe) = (one(), one());
            Box::new(GpuJoin::new(
                build,
                probe,
                join.join_type,
                join.keys.clone(),
                join.filter.clone(),
                join.filter_columns.clone(),
                join.null_equals_null,
                join.projection.clone(),
                schema_of(node),
            ))
        }
        NodeRef::CrossJoin(join) => {
            let (build, probe) = (one(), one());
            Box::new(GpuCrossJoin::new(
                build,
                probe,
                join.projection.clone(),
                schema_of(node),
            ))
        }
        NodeRef::NestedLoopJoin(join) => {
            let (build, probe) = (one(), one());
            Box::new(GpuNestedLoopJoin::new(
                build,
                probe,
                join.join_type,
                join.filter.clone(),
                join.filter_columns.clone(),
                join.projection.clone(),
                schema_of(node),
            ))
        }
        NodeRef::MergePartitions(_) => Box::new(GpuMergePartitions::new(one())),
        NodeRef::EmitPartitions(emit) => Box::new(GpuEmitPartitions::new(
            one(),
            emit.hash_keys.clone(),
            lanes_of(node),
        )),
        NodeRef::MergeSortedPartitions(merge) => Box::new(GpuMergeSortedPartitions::new(
            one(),
            merge.keys.clone(),
            merge.fetch,
        )),
        NodeRef::Union(_) => Box::new(GpuUnion::new(children.collect(), schema_of(node))),
        NodeRef::Interleave(_) => Box::new(GpuInterleave::new(children.collect(), schema_of(node))),
        NodeRef::Unload(unload) => Box::new(GpuUnload::new(one(), unload.interval)),
    }
}

/// The whole tree rebuilt bottom-up, each node over its rebuilt children — the identity
/// rewrite, and the walk every injected one is a variation of.
pub fn rebuild_tree(node: &dyn GpuNode) -> Box<dyn GpuNode> {
    let children = node
        .children()
        .into_iter()
        .map(rebuild_tree)
        .collect::<Vec<Box<dyn GpuNode>>>();
    rebuild(node, children)
}

/// The metadata read a loader was built from, back out of the fields it kept — the file
/// the row-group indices are numbered in, the survivors and their statistics. One
/// statement of it, since the injector rebuilds loaders too.
pub fn scan_of(load: &GpuLoadParquet) -> ScanMetadata {
    ScanMetadata {
        file: load.file.clone(),
        groups: load.survivors.clone(),
        can_be_null: load.can_be_null.clone(),
    }
}

/// Field by field rather than a clone, so a field added to the body is a compile error
/// here — the same guard the exhaustive match gives the kinds.
fn body_of(body: &AggregateBody) -> AggregateBody {
    AggregateBody {
        group_by: body.group_by.clone(),
        grouping_sets: body.grouping_sets.clone(),
        null_exprs: body.null_exprs.clone(),
        aggs: body.aggs.clone(),
        finalize: body.finalize.clone(),
    }
}

pub fn schema_of(node: &dyn GpuNode) -> Schema {
    node.kind()
        .schema()
        .unwrap_or_else(|| panic!("{} declares a schema", node.name()))
        .clone()
}

/// An emitter's lane count is what it emits into, which is its own declared layout — its
/// input's is where the rows come from and is a different number.
pub fn lanes_of(node: &dyn GpuNode) -> usize {
    node.kind()
        .layout()
        .unwrap_or_else(|| panic!("{} declares a layout", node.name()))
        .n
}

// ── fixtures for the identity case ──────────────────────────────────────────

/// One plan per kind, twice: every field a kind has takes two values across the pair —
/// the optional ones present and absent, the rest simply different, the child from a
/// different table with a different schema.
///
/// Hand-built rather than planned from sql, because what is proved is per arm and per
/// field: a corpus plan covers only the combinations its queries happen to produce, and
/// several of these fields appear in no corpus query at all.
pub fn every_kind() -> Vec<Box<dyn GpuNode>> {
    vec![
        source(Some(7)),
        other_source(),
        Box::new(GpuFilter::new(
            source(None),
            column(0, "k"),
            Some(vec![1]),
            one_column(),
        )),
        Box::new(GpuFilter::new(
            other_source(),
            column(1, "n"),
            None,
            other_columns(),
        )),
        Box::new(GpuProject::new(
            source(None),
            vec![NamedExpr::new(column(1, "v"), "v")],
            one_column(),
        )),
        Box::new(GpuProject::new(
            other_source(),
            vec![
                NamedExpr::new(column(0, "r"), "r"),
                NamedExpr::new(column(1, "n"), "n"),
            ],
            other_columns(),
        )),
        Box::new(GpuSort::new(source(None), vec![key(0)], Some(9))),
        Box::new(GpuSort::new(other_source(), vec![key(1)], None)),
        Box::new(GpuCoalesceAllBatches::new(source(None))),
        Box::new(GpuCoalesceAllBatches::new(other_source())),
        Box::new(GpuAccumulateBatchesAndSort::new(
            source(None),
            vec![key(0)],
            Some(4),
        )),
        Box::new(GpuAccumulateBatchesAndSort::new(
            other_source(),
            vec![key(1)],
            None,
        )),
        // A limit with both ends, and a pure offset — the form that never satisfies.
        Box::new(GpuLimit::new(
            source(None),
            RowInterval {
                skip: 3,
                fetch: Some(11),
            },
        )),
        Box::new(GpuLimit::new(
            other_source(),
            RowInterval {
                skip: 5,
                fetch: None,
            },
        )),
        // An aggregate with a finalize, grouping sets and their null expressions, over the
        // Welford triple — the widest body there is — against a keyless sum with none.
        Box::new(GpuAggregate::new(
            source(None),
            welford_body(true),
            state(),
            one_column(),
        )),
        Box::new(GpuAggregate::new(
            other_source(),
            sum_body(false),
            other_columns(),
            one_column(),
        )),
        Box::new(GpuAggregateBatches::new(
            source(None),
            welford_body(true),
            state(),
            one_column(),
        )),
        Box::new(GpuAggregateBatches::new(
            other_source(),
            sum_body(false),
            other_columns(),
            one_column(),
        )),
        // A join carrying its residual, the filter's own column map, NULL = NULL and a
        // projection; and one carrying none of them.
        Box::new(GpuJoin::new(
            source(None),
            source(Some(7)),
            JoinType::LeftSemi,
            vec![(0, 1)],
            Some(column(0, "f")),
            vec![
                JoinFilterColumn {
                    side: JoinSide::Build,
                    index: 1,
                },
                JoinFilterColumn {
                    side: JoinSide::Probe,
                    index: 0,
                },
            ],
            true,
            Some(vec![0, 3]),
            columns(),
        )),
        Box::new(GpuJoin::new(
            other_source(),
            source(None),
            JoinType::Inner,
            vec![(0, 0)],
            None,
            Vec::new(),
            false,
            None,
            other_columns(),
        )),
        Box::new(GpuCrossJoin::new(
            source(None),
            source(Some(7)),
            Some(vec![1, 2]),
            columns(),
        )),
        Box::new(GpuCrossJoin::new(
            other_source(),
            source(None),
            None,
            other_columns(),
        )),
        Box::new(GpuNestedLoopJoin::new(
            source(None),
            source(Some(7)),
            NestedLoopJoinType::Left,
            column(0, "f"),
            vec![JoinFilterColumn {
                side: JoinSide::Probe,
                index: 1,
            }],
            Some(vec![0, 2]),
            columns(),
        )),
        Box::new(GpuNestedLoopJoin::new(
            other_source(),
            source(None),
            NestedLoopJoinType::Inner,
            column(1, "f"),
            Vec::new(),
            None,
            other_columns(),
        )),
        Box::new(GpuMergePartitions::new(source(None))),
        Box::new(GpuMergePartitions::new(other_source())),
        Box::new(GpuEmitPartitions::new(source(None), vec![1], 4)),
        Box::new(GpuEmitPartitions::new(other_source(), vec![0], 2)),
        Box::new(GpuMergeSortedPartitions::new(
            sorted(source(None), key(0)),
            vec![key(0)],
            Some(6),
        )),
        Box::new(GpuMergeSortedPartitions::new(
            sorted(other_source(), key(1)),
            vec![key(1)],
            None,
        )),
        Box::new(GpuUnion::new(
            vec![source(None), source(Some(7))],
            columns(),
        )),
        Box::new(GpuUnion::new(
            vec![other_source(), source(None), source(Some(7))],
            other_columns(),
        )),
        Box::new(GpuInterleave::new(
            vec![scattered(source(None), 2), scattered(source(Some(7)), 2)],
            columns(),
        )),
        Box::new(GpuInterleave::new(
            vec![scattered(other_source(), 3), scattered(source(None), 3)],
            other_columns(),
        )),
        Box::new(GpuUnload::new(
            source(None),
            Some(RowInterval {
                skip: 2,
                fetch: Some(8),
            }),
        )),
        Box::new(GpuUnload::new(other_source(), None)),
    ]
}

/// Two lanes, three row groups between them, and a column that holds a NULL beside one
/// that does not — the loader's fields that no plan line prints.
pub fn source(limit: Option<usize>) -> Box<dyn GpuNode> {
    let groups: Vec<RowGroupMeta> = (0..3)
        .map(|index| RowGroupMeta {
            index,
            rows: 100 + u64::from(index),
            bytes: 800,
        })
        .collect();
    let scan = ScanMetadata {
        file: "/part.parquet".to_string(),
        groups,
        can_be_null: vec![false, true],
    };
    Box::new(GpuLoadParquet::new(
        "part".to_string(),
        vec![0, 1],
        vec![vec![vec![0], vec![1]], vec![vec![2]]],
        &scan,
        limit,
        columns(),
    ))
}

/// The second table, differing from the first in every field a loader carries: the file,
/// the projection, the mapping's shape, the survivors, which column can be NULL, and the
/// schema above it.
fn other_source() -> Box<dyn GpuNode> {
    let scan = ScanMetadata {
        file: "/region.parquet".to_string(),
        groups: vec![RowGroupMeta {
            index: 4,
            rows: 7,
            bytes: 64,
        }],
        can_be_null: vec![true, false],
    };
    Box::new(GpuLoadParquet::new(
        "region".to_string(),
        vec![0, 2],
        vec![vec![vec![4]]],
        &scan,
        Some(3),
        other_columns(),
    ))
}

/// A sorted stream, as the planner makes one: a per-batch sort, and the accumulator that
/// merges those batches into one. The accumulator requires the order below it, so the sort
/// is not decoration.
pub fn sorted(input: Box<dyn GpuNode>, key: ColumnOrder) -> Box<dyn GpuNode> {
    let sorted = Box::new(GpuSort::new(input, vec![key], None));
    Box::new(GpuAccumulateBatchesAndSort::new(sorted, vec![key], None))
}

/// Hash-partitioned lanes, which is what an interleave requires of every branch.
fn scattered(input: Box<dyn GpuNode>, lanes: usize) -> Box<dyn GpuNode> {
    Box::new(GpuEmitPartitions::new(input, vec![0], lanes))
}

fn columns() -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
    ])))
}

fn other_columns() -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(vec![
        Field::new("r", DataType::Int32, false),
        Field::new("n", DataType::Utf8, true),
    ])))
}

fn one_column() -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        true,
    )])))
}

/// An aggregate's intermediate schema, which is a second schema on the node and reaches
/// no plan line — so a rebuild dropping it shows only in the fields themselves.
fn state() -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
        "sum(v)",
        DataType::Int64,
        true,
    )])))
}

fn column(index: u32, name: &str) -> Expr {
    Expr::column(index, name)
}

pub fn key(column: u32) -> ColumnOrder {
    ColumnOrder {
        column,
        ascending: column == 0,
        nulls_first: column != 0,
    }
}

/// Grouping sets, their null expressions and a finalize project, over the three-column
/// Welford state — every optional field of a body at once.
fn welford_body(finalize: bool) -> AggregateBody {
    AggregateBody {
        group_by: vec![column(0, "k")],
        grouping_sets: vec![vec![false], vec![true]],
        null_exprs: vec![Expr::Literal(ScalarValue::Int64(None))],
        aggs: decomposition(AggFunc::Stddev)
            .state
            .iter()
            .map(|(suffix, func)| AggCall {
                func: *func,
                args: vec![column(1, "v")],
                outputs: vec![Field::new(
                    format!("stddev(v){suffix}"),
                    DataType::Float64,
                    true,
                )],
            })
            .collect(),
        finalize: finalize.then(|| vec![NamedExpr::new(column(1, "stddev(v)"), "stddev(v)")]),
    }
}

fn sum_body(finalize: bool) -> AggregateBody {
    AggregateBody {
        group_by: Vec::new(),
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![AggCall {
            func: PlanAgg::Sum,
            args: vec![column(1, "v")],
            outputs: vec![Field::new("sum(v)", DataType::Int64, true)],
        }],
        finalize: finalize.then(|| vec![NamedExpr::new(column(0, "sum(v)"), "sum(v)")]),
    }
}

// ── what the fixtures actually vary ────────────────────────────────────────

/// Every field of a node that the fixture set gives one value, named `Node.field`.
///
/// A fixture list is a hand-maintained claim, and the miss it allows is the one that
/// happens: a field added to a node, no fixture varying it, the arm free to drop it, the
/// identity case green. Derived from the debug output rather than listed, so it goes red
/// the day the field is added rather than the day something depends on it.
///
/// A node's own fields only. The tree is walked through `children()` and each node's debug
/// split at its outermost braces, so a nested `Expr` or `Schema` is one value rather than
/// something to parse — what is asserted is the level `rebuild` writes.
pub fn fields_with_one_value(fixtures: &[Box<dyn GpuNode>]) -> Vec<String> {
    let mut seen: BTreeMap<(&'static str, String), BTreeSet<String>> = BTreeMap::new();
    for fixture in fixtures {
        for node in every_node(fixture.as_ref()) {
            let debug = format!("{node:?}");
            for (field, value) in own_fields(&debug) {
                seen.entry((node.name(), field.to_string()))
                    .or_default()
                    .insert(value.to_string());
            }
        }
    }
    seen.into_iter()
        .filter_map(|((node, field), values)| {
            let name = format!("{node}.{field}");
            match (values.len() < 2, BY_CONSTRUCTION.contains(&name.as_str())) {
                (true, false) => Some(name),
                // The other direction, so the list cannot rot: a field that starts varying
                // is one whose exemption has stopped being true.
                (false, true) => Some(format!("{name} varies and is listed as constant")),
                _ => None,
            }
        })
        .collect()
}

/// What no fixture can vary. A sink's kind is `NodeKind::Sink`, which carries neither a
/// layout nor a schema — there is nothing in it for a rebuild to drop.
const BY_CONSTRUCTION: [&str; 1] = ["GpuUnload.kind"];

fn every_node<'a>(node: &'a dyn GpuNode) -> Vec<&'a dyn GpuNode> {
    let mut all = vec![node];
    for child in node.children() {
        all.extend(every_node(child));
    }
    all
}

/// The `field: value` pairs at the outermost brace level of `Name { … }`.
fn own_fields(debug: &str) -> Vec<(&str, &str)> {
    let Some(open) = debug.find(" { ") else {
        return Vec::new();
    };
    let body = debug[open + 3..].trim_end().trim_end_matches('}');
    split_top_level(body)
        .into_iter()
        .filter_map(|part| {
            let colon = part.find(": ")?;
            Some((part[..colon].trim(), part[colon + 2..].trim()))
        })
        .collect()
}

/// Split at the commas that are not inside a nesting or a string.
fn split_top_level(text: &str) -> Vec<&str> {
    let (mut depth, mut quoted, mut start) = (0i32, false, 0);
    let mut parts = Vec::new();
    for (index, character) in text.char_indices() {
        match character {
            '"' => quoted = !quoted,
            '{' | '[' | '(' if !quoted => depth += 1,
            '}' | ']' | ')' if !quoted => depth -= 1,
            ',' if depth == 0 && !quoted => {
                parts.push(text[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    let last = text[start..].trim();
    if !last.is_empty() {
        parts.push(last);
    }
    parts
}
