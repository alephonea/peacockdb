//! Rendering a plan as text: one line per node, indentation as the tree.
//!
//! One rule runs through it — the ordinal is authoritative and the name comes from the
//! declared schema at that position, so every column reference reads `name@ordinal` and a
//! name disagreeing with its ordinal is visible rather than invisible. The rest is what a
//! reader of a golden needs and cannot derive: the layout a node declares, every `fetch`
//! that trims rows, the loader's mapping verbatim, and the declared schema, without which
//! an explicit cast's target means nothing.

use std::fmt::Write as _;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::ScalarValue;

use super::aggregates::{AggCall, PlanAgg};
use super::estimator::MemoryModel;
use super::expr::{BinaryOp, ColumnRef, Expr, NamedExpr, UnaryOp};
use super::layout::{BatchLayout, ColumnOrder, KeyDistribution, PartitionLayout, SortOrder};
use super::node::{GpuNode, RowInterval};
use super::nodes::join::{JoinFilterColumn, JoinSide};
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
    let mut parts = vec![node_name(node).to_string()];
    parts.extend(node_fields(node));
    if let Some(layout) = node.kind().layout() {
        parts.push(layout_text(layout, schema_of(node)));
    }
    if let Some(schema) = node.kind().schema() {
        parts.push(schema_text(schema));
    }
    let mut line = parts.remove(0);
    if !parts.is_empty() {
        let _ = write!(line, ": {}", parts.join(", "));
    }
    line
}

fn schema_of(node: &dyn GpuNode) -> Option<&Schema> {
    node.kind().schema()
}

/// No `Exec` suffix: these are not DataFusion nodes, and after the wire-format rename a
/// line from either family says which mode produced it without a caption.
fn node_name(node: &dyn GpuNode) -> &'static str {
    match as_node_ref(node) {
        NodeRef::LoadParquet(_) => "GpuLoadParquet",
        NodeRef::Filter(_) => "GpuFilter",
        NodeRef::Project(_) => "GpuProject",
        NodeRef::Sort(_) => "GpuSort",
        NodeRef::CoalesceAllBatches(_) => "GpuCoalesceAllBatches",
        NodeRef::AccumulateBatchesAndSort(_) => "GpuAccumulateBatchesAndSort",
        NodeRef::Limit(_) => "GpuLimit",
        NodeRef::Aggregate(_) => "GpuAggregate",
        NodeRef::AggregateBatches(_) => "GpuAggregateBatches",
        NodeRef::Join(_) => "GpuJoin",
        NodeRef::CrossJoin(_) => "GpuCrossJoin",
        NodeRef::NestedLoopJoin(_) => "GpuNestedLoopJoin",
        NodeRef::MergePartitions(_) => "GpuMergePartitions",
        NodeRef::EmitPartitions(_) => "GpuEmitPartitions",
        NodeRef::MergeSortedPartitions(_) => "GpuMergeSortedPartitions",
        NodeRef::Union(_) => "GpuUnion",
        NodeRef::Interleave(_) => "GpuInterleave",
        NodeRef::Unload(_) => "GpuUnload",
    }
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
            if let Some(projection) = &join.projection {
                // Ordinals into the joined table, so they are named from both sides in
                // order — the same rule as every other reference.
                let joined: Vec<String> = projection
                    .iter()
                    .map(|ordinal| {
                        let build_width = build
                            .map(|schema| schema.fields.fields().len() as u32)
                            .unwrap_or(0);
                        if *ordinal < build_width {
                            format!("{}@{ordinal}", name_at(build, *ordinal))
                        } else {
                            format!("{}@{ordinal}", name_at(probe, *ordinal - build_width))
                        }
                    })
                    .collect();
                fields.push(format!("projection=[{}]", joined.join(", ")));
            }
        }
        NodeRef::NestedLoopJoin(join) => {
            fields.push(format!("join_type={:?}", join.join_type));
            fields.push(format!(
                "filter={}",
                join_filter_text(
                    &join.filter,
                    &join.filter_columns,
                    schema_of(node.children()[0]),
                    schema_of(node.children()[1]),
                )
            ));
        }
        NodeRef::EmitPartitions(emit) => {
            let keys: Vec<String> = emit
                .hash_keys
                .iter()
                .map(|key| format!("{}@{key}", name_at(input_schema, *key)))
                .collect();
            fields.push(format!("hash=[{}]", keys.join(", ")));
        }
        NodeRef::CoalesceAllBatches(_)
        | NodeRef::CrossJoin(_)
        | NodeRef::MergePartitions(_)
        | NodeRef::Union(_)
        | NodeRef::Interleave(_) => {}
    }
    fields
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
        outputs[0].to_string()
    } else {
        format!("[{}]", outputs.join(", "))
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
        .map(|field| format!("{}:{}", field.name(), type_text(field.data_type())))
        .collect();
    format!("schema=[{}]", columns.join(", "))
}

/// Arrow's own rendering, minus the noise a plan reader does not need. Decimal precision
/// and scale stay: an explicit cast's target is unreadable without them.
fn type_text(data_type: &DataType) -> String {
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
                _ => format!("{rendered} as {}", named.name),
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The name a schema declares at that position. An empty name is what a reference past
/// the end renders as, and validation is what refuses the plan.
fn name_at(schema: Option<&Schema>, index: u32) -> String {
    schema
        .and_then(|schema| schema.fields.fields().get(index as usize))
        .map(|field| field.name().clone())
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

pub fn expr_text(expr: &Expr) -> String {
    // An ordinary reference indexes the node's input, which is the line below it.
    expr_text_resolved(expr, &|reference| {
        format!("{}@{}", reference.name, reference.index)
    })
}

/// The same rendering with the column form supplied, because a join filter's ordinals
/// index a table of the filter's own that appears on no line.
fn expr_text_resolved(expr: &Expr, column: &dyn Fn(&ColumnRef) -> String) -> String {
    let nested = |expr: &Expr| nested_expr_resolved(expr, column);
    let plain = |expr: &Expr| expr_text_resolved(expr, column);
    match expr {
        Expr::Column(reference) => column(reference),
        Expr::Literal(value) => literal_text(value),
        Expr::Binary {
            left, op, right, ..
        } => format!("{} {} {}", nested(left), binary_op_text(*op), nested(right)),
        Expr::Unary { op, arg } => match op {
            UnaryOp::Not => format!("NOT {}", nested(arg)),
            UnaryOp::IsNull => format!("{} IS NULL", nested(arg)),
            UnaryOp::IsNotNull => format!("{} IS NOT NULL", nested(arg)),
            UnaryOp::Negative => format!("-{}", nested(arg)),
            UnaryOp::Sqrt => format!("sqrt({})", plain(arg)),
        },
        Expr::Cast { expr, target } => {
            format!("CAST({} AS {})", plain(expr), type_text(target))
        }
        Expr::Like {
            expr,
            pattern,
            negated,
            case_insensitive,
        } => format!(
            "{} {}{} {}",
            nested(expr),
            if *negated { "NOT " } else { "" },
            if *case_insensitive { "ILIKE" } else { "LIKE" },
            nested(pattern)
        ),
        Expr::Case {
            comparand,
            when_then,
            else_expr,
        } => {
            let mut text = "CASE".to_string();
            if let Some(comparand) = comparand {
                let _ = write!(text, " {}", plain(comparand));
            }
            for (when, then) in when_then {
                let _ = write!(text, " WHEN {} THEN {}", plain(when), plain(then));
            }
            if let Some(otherwise) = else_expr {
                let _ = write!(text, " ELSE {}", plain(otherwise));
            }
            text + " END"
        }
        Expr::ScalarFunction { name, args, .. } => format!(
            "{name}({})",
            args.iter().map(plain).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// A sub-expression that is itself an operator is parenthesized, so precedence is read off
/// the line rather than assumed.
fn nested_expr_resolved(expr: &Expr, column: &dyn Fn(&ColumnRef) -> String) -> String {
    match expr {
        Expr::Binary { .. } | Expr::Case { .. } => {
            format!("({})", expr_text_resolved(expr, column))
        }
        other => expr_text_resolved(other, column),
    }
}

/// A join filter's reference, resolved onto the side it came from and that side's own
/// ordinal — `k@build:0` is column 0 of the build child, whose schema is on its own line.
/// Left as its filter-schema ordinal only where the map is short, which validation refuses.
fn join_filter_text(
    filter: &Expr,
    columns: &[JoinFilterColumn],
    build: Option<&Schema>,
    probe: Option<&Schema>,
) -> String {
    expr_text_resolved(
        filter,
        &|reference| match columns.get(reference.index as usize) {
            Some(mapped) => {
                let (side, schema) = match mapped.side {
                    JoinSide::Build => ("build", build),
                    JoinSide::Probe => ("probe", probe),
                };
                format!("{}@{side}:{}", name_at(schema, mapped.index), mapped.index)
            }
            None => format!("{}@{}", reference.name, reference.index),
        },
    )
}

/// A decimal scalar prints as its value, not as the triple its `Display` gives: a plan
/// reader compares a literal against the column beside it.
fn literal_text(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Decimal128(Some(unscaled), _, scale) => decimal_text(*unscaled, *scale),
        ScalarValue::Decimal256(Some(unscaled), _, scale) => decimal_text(
            unscaled.to_string().parse::<i128>().unwrap_or_default(),
            *scale,
        ),
        ScalarValue::IntervalYearMonth(Some(months)) => interval_text(*months, 0, 0),
        ScalarValue::IntervalDayTime(Some(interval)) => {
            interval_text(0, interval.days, interval.milliseconds as i64 * 1_000_000)
        }
        ScalarValue::IntervalMonthDayNano(Some(interval)) => {
            interval_text(interval.months, interval.days, interval.nanoseconds)
        }
        other => other.to_string(),
    }
}

/// The parts that are not zero, in the units they are declared in — `90 days` where the
/// struct form spends sixty characters saying the same thing. All-zero prints as `0 days`,
/// since an interval of nothing is still an interval.
fn interval_text(months: i32, days: i32, nanoseconds: i64) -> String {
    let parts: Vec<String> = [
        (months as i64, "mons"),
        (days as i64, "days"),
        (nanoseconds, "nanos"),
    ]
    .into_iter()
    .filter(|(value, _)| *value != 0)
    .map(|(value, unit)| format!("{value} {unit}"))
    .collect();
    if parts.is_empty() {
        "0 days".to_string()
    } else {
        parts.join(" ")
    }
}

fn decimal_text(unscaled: i128, scale: i8) -> String {
    if scale <= 0 {
        return unscaled.to_string();
    }
    let divisor = 10i128.pow(scale as u32);
    let (sign, magnitude) = if unscaled < 0 {
        ("-", -unscaled)
    } else {
        ("", unscaled)
    };
    format!(
        "{sign}{}.{:0width$}",
        magnitude / divisor,
        magnitude % divisor,
        width = scale as usize
    )
}

fn binary_op_text(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Eq => "=",
        BinaryOp::NotEq => "!=",
        BinaryOp::Lt => "<",
        BinaryOp::LtEq => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::GtEq => ">=",
        BinaryOp::Plus => "+",
        BinaryOp::Minus => "-",
        BinaryOp::Multiply => "*",
        BinaryOp::Divide => "/",
        BinaryOp::Modulo => "%",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::BitwiseAnd => "&",
        BinaryOp::BitwiseOr => "|",
        BinaryOp::BitwiseXor => "^",
        BinaryOp::BitwiseShiftLeft => "<<",
        BinaryOp::BitwiseShiftRight => ">>",
        BinaryOp::StringConcat => "||",
        BinaryOp::IsDistinctFrom => "IS DISTINCT FROM",
        BinaryOp::IsNotDistinctFrom => "IS NOT DISTINCT FROM",
    }
}

/// The sibling memory golden: the same tree, carrying what the estimator derived rather
/// than what the planner decided. Sources also print the batch size they were given, which
/// is the one number the partitioner reads back.
pub fn render_plan_memory(root: &dyn GpuNode, model: &MemoryModel) -> String {
    let mut text = format!(
        "budget={}, accumulators={}, certain={}\n",
        model.budget, model.accumulator_bytes, model.certain_accumulator_bytes
    );
    render_memory_node(root, 0, &mut Sequence::default(), model, &mut text);
    text
}

/// Canonical post-order, the order the estimator indexes by: children left to right, then
/// the node.
#[derive(Default)]
struct Sequence {
    next: usize,
}

fn render_memory_node(
    node: &dyn GpuNode,
    depth: usize,
    sequence: &mut Sequence,
    model: &MemoryModel,
    text: &mut String,
) {
    let children: Vec<&dyn GpuNode> = node.children();
    let mut lines = Vec::new();
    for child in children {
        let mut child_text = String::new();
        render_memory_node(child, depth + 1, sequence, model, &mut child_text);
        lines.push(child_text);
    }
    let seq = sequence.next;
    sequence.next += 1;

    let mut fields = vec![format!(
        "estimated_max_resident_size={}",
        model.resident.get(seq).copied().unwrap_or(0)
    )];
    if let Some(source) = model.sources.iter().find(|source| source.seq == seq) {
        // What the source holds is what makes the rest of the line legible: it is the cap
        // on the target, so a reader sees whether the target is the source or the budget
        // talking, and it is what the small-table threshold compared to decide the lanes.
        if let NodeRef::LoadParquet(load) = as_node_ref(node) {
            fields.push(format!("source_bytes={}", load.bytes()));
        }
        fields.push(format!(
            "target_batch_bytes={}, amplification={:.1}",
            source.target_batch_bytes, source.amplification
        ));
    }
    let _ = writeln!(
        text,
        "{}{}: {}",
        "  ".repeat(depth),
        node_name(node),
        fields.join(", ")
    );
    for line in lines {
        text.push_str(&line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_partitioned::partitioner::Batching;
    use crate::batch_partitioned::translate::Translator;
    use datafusion::common::ScalarValue;
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

    #[test]
    fn an_interval_literal_prints_the_parts_that_are_not_zero() {
        use datafusion::arrow::datatypes::IntervalMonthDayNano;
        let ninety_days = Expr::Literal(ScalarValue::IntervalMonthDayNano(Some(
            IntervalMonthDayNano::new(0, 90, 0),
        )));
        assert_eq!(expr_text(&ninety_days), "90 days");
        let mixed = Expr::Literal(ScalarValue::IntervalMonthDayNano(Some(
            IntervalMonthDayNano::new(2, 1, 500),
        )));
        assert_eq!(expr_text(&mixed), "2 mons 1 days 500 nanos");
        let nothing = Expr::Literal(ScalarValue::IntervalMonthDayNano(Some(
            IntervalMonthDayNano::new(0, 0, 0),
        )));
        assert_eq!(expr_text(&nothing), "0 days");
    }

    #[test]
    fn a_decimal_literal_prints_as_a_value_rather_than_its_parts() {
        let money = Expr::Literal(ScalarValue::Decimal128(Some(-123_456), 15, 2));
        assert_eq!(expr_text(&money), "-1234.56");
        let whole = Expr::Literal(ScalarValue::Decimal128(Some(7), 15, 0));
        assert_eq!(expr_text(&whole), "7");
    }

    #[test]
    fn every_expression_form_renders_readably() {
        use crate::batch_partitioned::expr::{BinaryOp, UnaryOp};
        let column = Expr::column(2, "s");
        let cast = Expr::Cast {
            expr: Box::new(Expr::column(0, "a")),
            target: DataType::Decimal128(38, 6),
        };
        assert_eq!(expr_text(&cast), "CAST(a@0 AS Decimal128(38,6))");

        let like = Expr::Like {
            expr: Box::new(column.clone()),
            pattern: Box::new(Expr::Literal(ScalarValue::Utf8(Some("%x%".to_string())))),
            negated: true,
            case_insensitive: false,
        };
        assert_eq!(expr_text(&like), "s@2 NOT LIKE %x%");

        // A nested operator is parenthesized, so precedence is read off the line.
        let nested = Expr::binary(
            Expr::binary(
                Expr::column(0, "a"),
                BinaryOp::Plus,
                Expr::column(1, "b"),
                DataType::Int64,
            ),
            BinaryOp::Gt,
            Expr::Literal(ScalarValue::Int64(Some(3))),
            DataType::Boolean,
        );
        assert_eq!(expr_text(&nested), "(a@0 + b@1) > 3");

        let guard = Expr::Case {
            comparand: None,
            when_then: vec![(
                Expr::unary(UnaryOp::IsNull, Expr::column(0, "a")),
                Expr::Literal(ScalarValue::Int64(None)),
            )],
            else_expr: Some(Box::new(Expr::unary(UnaryOp::Sqrt, Expr::column(0, "a")))),
        };
        assert_eq!(
            expr_text(&guard),
            "CASE WHEN a@0 IS NULL THEN NULL ELSE sqrt(a@0) END"
        );

        let call = Expr::ScalarFunction {
            name: "date_part".to_string(),
            args: vec![
                Expr::Literal(ScalarValue::Utf8(Some("year".to_string()))),
                column,
            ],
            return_type: DataType::Int32,
            nullable: true,
        };
        assert_eq!(expr_text(&call), "date_part(year, s@2)");
    }
}
