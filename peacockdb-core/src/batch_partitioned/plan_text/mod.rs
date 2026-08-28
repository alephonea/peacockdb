//! Rendering a plan as text: one line per node, indentation as the tree.
//!
//! One rule runs through it — the ordinal is authoritative and the name comes from the
//! declared schema at that position, so every column reference reads `name@ordinal` and a
//! name disagreeing with its ordinal is visible rather than invisible. The rest is what a
//! reader of a golden needs and cannot derive: the layout a node declares, every `fetch`
//! that trims rows, the loader's mapping verbatim, and the declared schema, without which
//! an explicit cast's target means nothing.

mod expr_text;
mod fb_text;
mod memory;
mod recipes;
mod run_text;

pub use expr_text::expr_text;
use expr_text::join_filter_text;
pub use memory::render_plan_memory;
pub use recipes::{Payloads, render_plan_recipes};
pub use run_text::render_run;

use std::fmt::Write as _;

use datafusion::arrow::datatypes::DataType;

use super::aggregates::{AggCall, PlanAgg};
use super::expr::{Expr, NamedExpr};
use super::layout::{BatchLayout, ColumnOrder, KeyDistribution, PartitionLayout, SortOrder};
use super::node::{GpuNode, RowInterval};
use super::nodes::{NodeRef, as_node_ref};
use super::schema::Schema;

/// The plan under `root`, one line per node. The plan golden carries the declared schema
/// per node; an execution golden does not, since what it records is what ran.
pub fn render_plan(root: &dyn GpuNode) -> String {
    let mut text = String::new();
    render_node(root, 0, &mut text);
    text
}

fn render_node(node: &dyn GpuNode, depth: usize, text: &mut String) {
    let _ = writeln!(text, "{}{}", "  ".repeat(depth), node_line(node));
    for child in node.children() {
        render_node(child, depth + 1, text);
    }
}

fn node_line(node: &dyn GpuNode) -> String {
    let mut parts = node_line_parts(node);
    if let Some(schema) = node.kind().schema() {
        parts.push(schema_text(schema));
    }
    join_parts(parts)
}

/// The name, the node's own fields and its layout — everything both goldens carry. The
/// declared schema is the plan golden's alone and what the run produced is the execution
/// golden's, so each caller appends its own tail.
pub(super) fn node_line_parts(node: &dyn GpuNode) -> Vec<String> {
    let mut parts = vec![node.name().to_string()];
    parts.extend(node_fields(node));
    if let Some(layout) = node.kind().layout() {
        parts.push(layout_text(layout, schema_of(node)));
    }
    parts
}

/// `Name: field, field` — the name alone where a node has no fields at all.
pub(super) fn join_parts(mut parts: Vec<String>) -> String {
    let mut line = parts.remove(0);
    if !parts.is_empty() {
        let _ = write!(line, ": {}", parts.join(", "));
    }
    line
}

fn schema_of(node: &dyn GpuNode) -> Option<&Schema> {
    node.kind().schema()
}

/// What this node was asked to do — the parameters that decide its answer, and nothing
/// that is derived from its children or repeated by the layout.
fn node_fields(node: &dyn GpuNode) -> Vec<String> {
    let input_schema = node.children().first().and_then(|child| schema_of(*child));
    let mut fields = Vec::new();
    match as_node_ref(node) {
        NodeRef::LoadParquet(load) => {
            fields.push(format!("table={}", load.table));
            let schema = schema_of(node);
            let projected: Vec<String> = load
                .projection
                .iter()
                .enumerate()
                .map(|(position, file_ordinal)| {
                    format!("{}@{file_ordinal}", name_at(schema, position as u32))
                })
                .collect();
            fields.push(format!("projections=[{}]", projected.join(", ")));
            // The mapping verbatim: partitions outermost, batches within them, row groups
            // innermost. Which batch sits in which partition is the whole content.
            fields.push(format!(
                "partition_groups={}",
                nested(&load.partition_groups)
            ));
            if let Some(limit) = load.limit {
                fields.push(format!("limit={limit}"));
            }
        }
        NodeRef::Filter(filter) => {
            fields.push(format!("predicate={}", expr_text(&filter.predicate)));
            if let Some(projection) = &filter.projection {
                let kept: Vec<String> = projection
                    .iter()
                    .map(|ordinal| format!("{}@{ordinal}", name_at(input_schema, *ordinal)))
                    .collect();
                fields.push(format!("projection=[{}]", kept.join(", ")));
            }
        }
        NodeRef::Project(project) => {
            fields.push(format!("exprs=[{}]", named_exprs(&project.exprs)));
        }
        NodeRef::Sort(sort) => {
            fields.push(format!("by=[{}]", sort_keys(&sort.keys, input_schema)));
            push_fetch(&mut fields, sort.fetch);
        }
        NodeRef::AccumulateBatchesAndSort(accumulator) => {
            fields.push(format!(
                "by=[{}]",
                sort_keys(&accumulator.keys, input_schema)
            ));
            push_fetch(&mut fields, accumulator.fetch);
        }
        NodeRef::MergeSortedPartitions(merge) => {
            fields.push(format!("by=[{}]", sort_keys(&merge.keys, input_schema)));
            push_fetch(&mut fields, merge.fetch);
        }
        NodeRef::Limit(limit) => fields.extend(interval_fields(limit.interval)),
        NodeRef::Unload(unload) => {
            if let Some(interval) = unload.interval {
                fields.extend(interval_fields(interval));
            }
        }
        NodeRef::Aggregate(aggregate) => aggregate_fields(&mut fields, &aggregate.body),
        NodeRef::AggregateBatches(aggregate) => aggregate_fields(&mut fields, &aggregate.body),
        NodeRef::Join(join) => {
            fields.push(format!("join_type={:?}", join.join_type));
            let build = schema_of(node.children()[0]);
            let probe = schema_of(node.children()[1]);
            let on: Vec<String> = join
                .keys
                .iter()
                .map(|(left, right)| {
                    format!(
                        "({}@{left}, {}@{right})",
                        name_at(build, *left),
                        name_at(probe, *right)
                    )
                })
                .collect();
            fields.push(format!("on=[{}]", on.join(", ")));
            if let Some(filter) = &join.filter {
                fields.push(format!(
                    "filter={}",
                    join_filter_text(filter, &join.filter_columns, build, probe)
                ));
            }
            // Only the non-default prints: the SQL default is false, which is what nearly
            // every join carries, and a line saying so on each of them would say nothing.
            if join.null_equals_null {
                fields.push("null_equals_null=true".to_string());
            }
            projection_field(&mut fields, join.projection.as_ref(), build, probe);
        }
        NodeRef::NestedLoopJoin(join) => {
            let (build, probe) = (schema_of(node.children()[0]), schema_of(node.children()[1]));
            fields.push(format!("join_type={:?}", join.join_type));
            fields.push(format!(
                "filter={}",
                join_filter_text(&join.filter, &join.filter_columns, build, probe)
            ));
            projection_field(&mut fields, join.projection.as_ref(), build, probe);
        }
        NodeRef::CrossJoin(join) => projection_field(
            &mut fields,
            join.projection.as_ref(),
            schema_of(node.children()[0]),
            schema_of(node.children()[1]),
        ),
        NodeRef::EmitPartitions(emit) => {
            let keys: Vec<String> = emit
                .hash_keys
                .iter()
                .map(|key| format!("{}@{key}", name_at(input_schema, *key)))
                .collect();
            fields.push(format!("hash=[{}]", keys.join(", ")));
        }
        NodeRef::CoalesceAllBatches(_)
        | NodeRef::MergePartitions(_)
        | NodeRef::Union(_)
        | NodeRef::Interleave(_) => {}
    }
    fields
}

/// Ordinals into the crossed table, so they are named from both sides in order — the same
/// rule as every other reference.
fn projection_field(
    fields: &mut Vec<String>,
    projection: Option<&Vec<u32>>,
    build: Option<&Schema>,
    probe: Option<&Schema>,
) {
    let Some(projection) = projection else {
        return;
    };
    let build_width = build
        .map(|schema| schema.fields.fields().len() as u32)
        .unwrap_or(0);
    let joined: Vec<String> = projection
        .iter()
        .map(|ordinal| {
            if *ordinal < build_width {
                format!("{}@{ordinal}", name_at(build, *ordinal))
            } else {
                format!("{}@{ordinal}", name_at(probe, *ordinal - build_width))
            }
        })
        .collect();
    fields.push(format!("projection=[{}]", joined.join(", ")));
}

fn aggregate_fields(fields: &mut Vec<String>, body: &super::nodes::AggregateBody) {
    let keys: Vec<String> = body.group_by.iter().map(expr_text).collect();
    fields.push(format!("group_by=[{}]", keys.join(", ")));
    if !body.grouping_sets.is_empty() {
        // The keys each set groups on — the complement of the mask, which is the half a
        // reader checks against the rollup the query asked for.
        let sets: Vec<String> = body
            .grouping_sets
            .iter()
            .map(|mask| {
                let held: Vec<&str> = mask
                    .iter()
                    .enumerate()
                    .filter(|(_, is_null)| !**is_null)
                    .filter_map(|(index, _)| keys.get(index).map(String::as_str))
                    .collect();
                format!("[{}]", held.join(", "))
            })
            .collect();
        fields.push(format!("grouping_sets=[{}]", sets.join(", ")));
    }
    let aggs: Vec<String> = body.aggs.iter().map(agg_call_text).collect();
    fields.push(format!("aggs=[{}]", aggs.join(", ")));
    if let Some(finalize) = &body.finalize {
        fields.push(format!("final=[{}]", named_exprs(finalize)));
    }
}

fn agg_call_text(call: &AggCall) -> String {
    let args: Vec<String> = call.args.iter().map(expr_text).collect();
    let outputs: Vec<&str> = call
        .outputs
        .iter()
        .map(|field| field.name().as_str())
        .collect();
    let produced = if outputs.len() == 1 {
        quoted(outputs[0])
    } else {
        format!(
            "[{}]",
            outputs
                .iter()
                .map(|name| quoted(name))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    format!(
        "{}({}) as {produced}",
        plan_agg_name(call.func),
        args.join(", ")
    )
}

fn plan_agg_name(func: PlanAgg) -> &'static str {
    match func {
        PlanAgg::Sum => "sum",
        PlanAgg::Min => "min",
        PlanAgg::Max => "max",
        PlanAgg::Count => "count",
        PlanAgg::Mean => "mean",
        PlanAgg::M2 => "m2",
        PlanAgg::MergeM2 => "merge_m2",
    }
}

fn interval_fields(interval: RowInterval) -> Vec<String> {
    let mut fields = vec![format!("skip={}", interval.skip)];
    if let Some(fetch) = interval.fetch {
        fields.push(format!("fetch={fetch}"));
    }
    fields
}

fn push_fetch(fields: &mut Vec<String>, fetch: Option<usize>) {
    if let Some(fetch) = fetch {
        fields.push(format!("fetch={fetch}"));
    }
}

/// Lane count and batch layout always; a hash or an order only where one is declared,
/// since a line saying a node declares nothing says nothing.
fn layout_text(layout: &PartitionLayout, schema: Option<&Schema>) -> String {
    let mut text = format!("lanes={}", layout.n);
    let _ = write!(
        text,
        ", batches={}",
        match layout.batch_layout {
            BatchLayout::SingleBatch => "single",
            BatchLayout::MultipleBatches => "multiple",
        }
    );
    if let KeyDistribution::ByHash { hash_keys } = &layout.key_distribution {
        let keys: Vec<String> = hash_keys
            .iter()
            .map(|key| format!("{}@{key}", name_at(schema, *key)))
            .collect();
        let _ = write!(text, ", hashed_on=[{}]", keys.join(", "));
    }
    if let SortOrder::BatchSorted { columns } = &layout.sort_order {
        let _ = write!(text, ", sorted_on=[{}]", sort_keys(columns, schema));
    }
    text
}

fn schema_text(schema: &Schema) -> String {
    let columns: Vec<String> = schema
        .fields
        .fields()
        .iter()
        .map(|field| format!("{}:{}", quoted(field.name()), type_text(field.data_type())))
        .collect();
    format!("schema=[{}]", columns.join(", "))
}

/// Arrow's own rendering, minus the noise a plan reader does not need. Decimal precision
/// and scale stay: an explicit cast's target is unreadable without them.
pub(super) fn type_text(data_type: &DataType) -> String {
    match data_type {
        DataType::Decimal128(precision, scale) => format!("Decimal128({precision},{scale})"),
        other => format!("{other:?}"),
    }
}

fn sort_keys(keys: &[ColumnOrder], schema: Option<&Schema>) -> String {
    keys.iter()
        .map(|key| {
            format!(
                "{}@{} {} {}",
                name_at(schema, key.column),
                key.column,
                if key.ascending { "asc" } else { "desc" },
                if key.nulls_first {
                    "nulls_first"
                } else {
                    "nulls_last"
                }
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn named_exprs(exprs: &[NamedExpr]) -> String {
    exprs
        .iter()
        .map(|named| {
            let rendered = expr_text(&named.expr);
            match &named.expr {
                Expr::Column(reference) if reference.name == named.name => rendered,
                _ => format!("{rendered} as {}", quoted(&named.name)),
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// A name as a single token: bare where it already is one, in backticks where it holds a
/// character this rendering punctuates with — whitespace, a comma, a bracket, an `@`, or a
/// backtick, which doubles.
///
/// A name is not always an identifier: DataFusion names an aggregate output by its own
/// expression text, and tpcds aliases like `order count`. Unquoted, `order count@0 as
/// order count` leaves a reader no way to see where the name ends. Backticks rather than
/// quotes because a name can hold a rendered literal — `Utf8("PROMO%")` — and doubling
/// those is the unreadable half of the problem.
fn quoted(name: &str) -> String {
    let plain = !name.is_empty()
        && !name
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, ',' | '[' | ']' | '@' | '`'));
    if plain {
        name.to_string()
    } else {
        format!("`{}`", name.replace('`', "``"))
    }
}

/// The name a schema declares at that position. An empty name is what a reference past
/// the end renders as, and validation is what refuses the plan.
pub(super) fn name_at(schema: Option<&Schema>, index: u32) -> String {
    schema
        .and_then(|schema| schema.fields.fields().get(index as usize))
        .map(|field| quoted(field.name()))
        .unwrap_or_else(|| "?".to_string())
}

fn nested(groups: &[Vec<Vec<u32>>]) -> String {
    let partitions: Vec<String> = groups
        .iter()
        .map(|batches| {
            let rendered: Vec<String> = batches
                .iter()
                .map(|row_groups| {
                    format!(
                        "[{}]",
                        row_groups
                            .iter()
                            .map(u32::to_string)
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                })
                .collect();
            format!("[{}]", rendered.join(","))
        })
        .collect();
    format!("[{}]", partitions.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_partitioned::partitioner::Batching;
    use crate::batch_partitioned::translate::Translator;
    use std::path::PathBuf;

    async fn rendered(sql: &str, target_partitions: usize) -> String {
        let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal");
        let ctx = crate::register_tables_for(crate::build_session_state(target_partitions), &data)
            .await
            .expect("register the minimal tables");
        let plan = ctx
            .sql(sql)
            .await
            .expect("plan the query")
            .create_physical_plan()
            .await
            .expect("physical plan");
        let tree = Translator::new(
            target_partitions,
            Batching::Sized {
                target_batch_bytes: 1 << 20,
            },
        )
        .translate(&plan)
        .expect("translate the plan");
        render_plan(tree.as_ref())
    }

    fn line_with<'a>(text: &'a str, node: &str) -> &'a str {
        text.lines()
            .find(|line| line.trim_start().starts_with(node))
            .unwrap_or_else(|| panic!("no {node} line in:\n{text}"))
    }

    #[tokio::test]
    async fn every_column_reference_renders_name_at_ordinal() {
        let text = rendered("SELECT c_name FROM customer WHERE c_nationkey > 1", 1).await;
        // The ordinal is authoritative and the name is the declared schema's at that
        // position, so a name disagreeing with its ordinal is visible on the line.
        assert!(
            line_with(&text, "GpuFilter").contains("predicate=c_nationkey@1 > 1"),
            "{text}"
        );
        // The scan's projections carry the file ordinal each column came from.
        assert!(
            line_with(&text, "GpuLoadParquet").contains("projections=[c_name@1, c_nationkey@3]"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn the_layout_replaces_the_lane_count() {
        let text = rendered(
            "SELECT c_nationkey, count(*) FROM customer GROUP BY c_nationkey",
            4,
        )
        .await;
        // Lane count and batch layout on every node; a hash or an order only where one is
        // declared. Three of these four are invisible in a legacy line.
        assert!(
            line_with(&text, "GpuLoadParquet").contains("lanes=4, batches=multiple"),
            "{text}"
        );
        assert!(
            line_with(&text, "GpuEmitPartitions")
                .contains("lanes=4, batches=multiple, hashed_on=[c_nationkey@0]"),
            "{text}"
        );
        assert!(
            line_with(&text, "GpuCoalesceAllBatches").contains("lanes=1, batches=single"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn every_node_carrying_a_fetch_prints_it() {
        let text = rendered(
            "SELECT c_name FROM customer ORDER BY c_name LIMIT 5 OFFSET 2",
            4,
        )
        .await;
        // A merge that turns 40 rows into 7 says so on its own line; in legacy only the
        // sort beneath it does, and the merge is silent.
        assert!(line_with(&text, "GpuSort").contains("fetch=7"), "{text}");
        assert!(
            line_with(&text, "GpuMergeSortedPartitions").contains("fetch=7"),
            "{text}"
        );
        // The interval rides the boundary crossing, and prints there.
        assert!(
            line_with(&text, "GpuUnload").contains("skip=2, fetch=5"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn a_mid_plan_limit_prints_its_interval_on_its_own_node() {
        let text = rendered(
            "SELECT count(*) FROM (SELECT * FROM customer WHERE c_nationkey > 1 LIMIT 3) t",
            1,
        )
        .await;
        assert!(
            line_with(&text, "GpuLimit").contains("skip=0, fetch=3"),
            "{text}"
        );
    }

    #[test]
    fn a_name_that_is_not_a_token_is_backquoted_and_an_ordinary_one_is_not() {
        assert_eq!(quoted("c_name"), "c_name");
        assert_eq!(
            quoted("sum(lineitem.l_quantity)"),
            "sum(lineitem.l_quantity)"
        );
        assert_eq!(quoted("order count"), "`order count`");
        assert_eq!(quoted("a,b"), "`a,b`");
        assert_eq!(quoted("x@1"), "`x@1`");
        assert_eq!(quoted("a`b"), "`a``b`");
        assert_eq!(quoted(""), "``");
    }

    #[tokio::test]
    async fn a_spaced_column_name_prints_as_one_token_everywhere_it_appears() {
        // Both halves of `… as name` and the schema entry, which is where a reader
        // resolves an ordinal: unquoted, the name's own space reads as the separator.
        let text = rendered(
            "SELECT \"nat key\", count(*) FROM \
             (SELECT c_nationkey AS \"nat key\" FROM customer) t GROUP BY \"nat key\"",
            1,
        )
        .await;
        assert!(text.contains("as `nat key`"), "{text}");
        assert!(text.contains("schema=[`nat key`:"), "{text}");
        assert!(text.contains("group_by=[`nat key`@0]"), "{text}");
    }

    #[tokio::test]
    async fn a_filter_that_projects_declares_and_prints_what_it_keeps() {
        // DataFusion's filter drops columns as well as rows. A node that declared its
        // child's schema here would emit an extra column and shift every ordinal above it.
        let text = rendered(
            "SELECT c_name FROM customer WHERE c_nationkey > 1 ORDER BY c_name",
            4,
        )
        .await;
        let filter = line_with(&text, "GpuFilter");
        assert!(filter.contains("projection=[c_name@0]"), "{text}");
        assert!(filter.contains("schema=[c_name:Utf8View]"), "{text}");
    }

    #[tokio::test]
    async fn the_declared_schema_prints_name_and_type_per_column() {
        let text = rendered("SELECT c_acctbal FROM customer", 1).await;
        // Precision and scale stay: an explicit cast's target is unreadable without the
        // state column's declared scale beside it.
        assert!(
            line_with(&text, "GpuLoadParquet").contains("schema=[c_acctbal:Decimal128(15,2)]"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn a_source_prints_the_partitioners_mapping_verbatim() {
        let text = rendered("SELECT c_name FROM customer", 4).await;
        // Partitions outermost, batches within them, row groups innermost — and a lane
        // the mapping left empty renders as one, because it is one.
        assert!(
            line_with(&text, "GpuLoadParquet").contains("partition_groups=[[[0]],[[1]],[],[]]"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn node_names_carry_no_exec_suffix() {
        let text = rendered("SELECT c_name FROM customer ORDER BY c_name", 4).await;
        assert!(!text.contains("Exec"), "{text}");
        assert!(text.contains("GpuMergeSortedPartitions"), "{text}");
    }

    #[tokio::test]
    async fn an_aggregate_prints_its_aggregators_and_its_final_list() {
        let text = rendered("SELECT stddev(c_acctbal) FROM customer", 1).await;
        let merge = line_with(&text, "GpuAggregateBatches");
        // merge_m2 returns its three state columns together, and the line spells them out.
        assert!(merge.contains("merge_m2("), "{text}");
        assert!(
            merge.contains(
                "$count, stddev(customer.c_acctbal)$mean, stddev(customer.c_acctbal)$m2]"
            ),
            "{text}"
        );
        assert!(merge.contains("final=[CASE WHEN"), "{text}");
        // The init node runs the three Welford aggregators over raw rows.
        assert!(
            line_with(&text, "GpuAggregate:").contains("m2(c_acctbal@0)"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn a_join_prints_its_keys_and_its_projection_by_name() {
        let text = rendered(
            "SELECT c.c_name, s.s_name FROM customer c JOIN supplier s ON c.c_nationkey = s.s_nationkey",
            4,
        )
        .await;
        let join = line_with(&text, "GpuJoin");
        assert!(
            join.contains("on=[(s_nationkey@1, c_nationkey@1)]"),
            "{text}"
        );
        // A projection is ordinals into the joined table, so it is named from both sides
        // rather than printed as bare positions.
        assert!(join.contains("projection=[s_name@0, c_name@2]"), "{text}");
    }

    #[tokio::test]
    async fn a_join_filter_resolves_each_reference_onto_the_side_it_came_from() {
        // The filter's ordinals index a table of its own, which appears on no line. Both
        // sides carry their key at ordinal 0 here, so a side mix-up changes the text — the
        // case that caught the same slip when the validator went red at T4.
        let text = rendered(
            "SELECT * FROM nation n, region r WHERE n.n_nationkey < r.r_regionkey",
            1,
        )
        .await;
        let join = line_with(&text, "GpuNestedLoopJoin");
        // region is the build side — DataFusion put the smaller table there and flipped
        // the predicate — and both sides' ordinal 0 is a different column, which is what
        // makes a mix-up visible.
        assert!(
            join.contains("filter=r_regionkey@build:0 > n_nationkey@probe:0"),
            "{text}"
        );
    }
}
