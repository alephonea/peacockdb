//! What a recipe-plan node carries, read back out of the bytes and written as text.
//!
//! Read back rather than described from the plan node it came from: a description written
//! beside the writer agrees with the writer by construction, and would keep agreeing on
//! the day the writer became wrong. This one can only say what is in the buffer.

use std::fmt::Write as _;

use crate::generated::gpu_plan_generated::peacock::plan as fb;

pub(super) use super::super::recipe::node_at;

/// One field per line, indented under the call that addresses it. Only what the node
/// carries: a field the writer left unset is absent here rather than printed as a default,
/// since "not set" and "set to zero" are different instructions to the executor.
pub(super) fn payload_text(node: &fb::PlanNode<'_>, indent: &str) -> String {
    let mut text = String::new();
    let mut field = |name: &str, value: String| {
        let _ = writeln!(text, "{indent}{name}: {value}");
    };
    match node.node_type() {
        fb::PlanNodeKind::CudfScan => {
            let scan = node.node_as_cudf_scan().expect("a scan");
            if let Some(paths) = scan.file_paths() {
                field("files", paths.iter().collect::<Vec<_>>().join(", "));
            }
            if let Some(schema) = scan.file_schema() {
                field("schema", schema_text(&schema));
            }
            if scan.limit() > 0 {
                field("limit", scan.limit().to_string());
            }
        }
        fb::PlanNodeKind::CudfFilter => {
            let filter = node.node_as_cudf_filter().expect("a filter");
            if let Some(predicate) = filter.predicate() {
                field("predicate", expr_text(&predicate));
            }
            if let Some(projection) = filter.projection() {
                field("projection", ordinals(projection.iter()));
            }
        }
        fb::PlanNodeKind::CudfProject => {
            let project = node.node_as_cudf_project().expect("a project");
            let exprs = project.exprs().map(|v| v.iter().collect::<Vec<_>>());
            let aliases = project.aliases().map(|v| v.iter().collect::<Vec<_>>());
            if let (Some(exprs), Some(aliases)) = (exprs, aliases) {
                for (expr, alias) in exprs.iter().zip(aliases) {
                    field(alias, expr_text(expr));
                }
            }
        }
        fb::PlanNodeKind::CudfAggregate => {
            let aggregate = node.node_as_cudf_aggregate().expect("an aggregate");
            field("mode", format!("{:?}", aggregate.mode()));
            if let Some(groups) = aggregate.group_exprs() {
                let written: Vec<String> = groups.iter().map(|e| expr_text(&e)).collect();
                field("group_by", written.join(", "));
            }
            if let Some(funcs) = aggregate.aggr_funcs() {
                for func in funcs.iter() {
                    let args = func
                        .args()
                        .map(|args| {
                            let written: Vec<String> =
                                args.iter().map(|a| expr_text(&a)).collect();
                            written.join(", ")
                        })
                        .unwrap_or_default();
                    field(
                        func.alias().unwrap_or_default(),
                        format!("{}({args})", func.name().unwrap_or("?")),
                    );
                }
            }
            // Only where there are any: a plain group-by writes both empty, and a line
            // saying so on every aggregate in the file would bury the ones that carry them.
            if let Some(nulls) = aggregate.null_exprs().filter(|nulls| !nulls.is_empty()) {
                let names = aggregate.null_names();
                let written: Vec<String> = nulls
                    .iter()
                    .enumerate()
                    .map(|(position, expr)| {
                        let name = names
                            .filter(|names| position < names.len())
                            .map(|names| names.get(position))
                            .unwrap_or("?");
                        format!("{name}={}", expr_text(&expr))
                    })
                    .collect();
                field("null_exprs", written.join(", "));
            }
            if let Some(sets) = aggregate.grouping_sets().filter(|sets| !sets.is_empty()) {
                let written: Vec<String> = sets
                    .iter()
                    .map(|set| {
                        let mask: Vec<&str> = set
                            .values()
                            .map(|values| {
                                values.iter().map(|on| if on { "-" } else { "k" }).collect()
                            })
                            .unwrap_or_default();
                        mask.join("")
                    })
                    .collect();
                field("grouping_sets", written.join(", "));
            }
            if aggregate.mergeable_agg_state() {
                field("mergeable_agg_state", "true".to_string());
            }
        }
        fb::PlanNodeKind::CudfHashJoin => {
            let join = node.node_as_cudf_hash_join().expect("a hash join");
            field("join_type", format!("{:?}", join.join_type()));
            if let Some(keys) = join.keys() {
                let pairs: Vec<String> = keys
                    .iter()
                    .map(|key| {
                        format!(
                            "({}, {})",
                            key.left().map(|e| expr_text(&e)).unwrap_or_default(),
                            key.right().map(|e| expr_text(&e)).unwrap_or_default()
                        )
                    })
                    .collect();
                field("keys", pairs.join(", "));
            }
            if let Some(filter) = join.filter() {
                field("filter", expr_text(&filter));
            }
            if let Some(columns) = join.filter_columns() {
                let mapped: Vec<String> = columns
                    .iter()
                    .map(|column| format!("{:?}@{}", column.side(), column.index()))
                    .collect();
                field("filter_columns", format!("[{}]", mapped.join(", ")));
            }
            if let Some(projection) = join.projection() {
                field("projection", ordinals(projection.iter()));
            }
            if join.null_equals_null() {
                field("null_equals_null", "true".to_string());
            }
        }
        fb::PlanNodeKind::CudfNestedLoopJoin => {
            let join = node.node_as_cudf_nested_loop_join().expect("a nlj");
            field("join_type", format!("{:?}", join.join_type()));
            if let Some(filter) = join.filter() {
                field("filter", expr_text(&filter));
            }
            if let Some(projection) = join.projection() {
                field("projection", ordinals(projection.iter()));
            }
        }
        fb::PlanNodeKind::CudfSort => {
            let sort = node.node_as_cudf_sort().expect("a sort");
            if let Some(exprs) = sort.exprs() {
                field("by", sort_keys(exprs));
            }
            if sort.fetch() >= 0 {
                field("fetch", sort.fetch().to_string());
            }
        }
        fb::PlanNodeKind::CudfSortPreservingMerge => {
            let merge = node
                .node_as_cudf_sort_preserving_merge()
                .expect("a sort-preserving merge");
            if let Some(exprs) = merge.exprs() {
                field("by", sort_keys(exprs));
            }
            if merge.fetch() >= 0 {
                field("fetch", merge.fetch().to_string());
            }
        }
        fb::PlanNodeKind::CudfRepartition => {
            let repartition = node.node_as_cudf_repartition().expect("a repartition");
            field(
                "partitioning",
                format!("{:?}, {}", repartition.kind(), repartition.num_partitions()),
            );
            if let Some(keys) = repartition.hash_exprs() {
                field(
                    "hash",
                    keys.iter().map(|e| expr_text(&e)).collect::<Vec<_>>().join(", "),
                );
            }
        }
        // A coalesce and a cross join carry their inputs and nothing else, which is the
        // whole content of the collapse arm: it concatenates whatever it is handed.
        _ => {}
    }
    text
}

fn ordinals(columns: impl Iterator<Item = u32>) -> String {
    format!(
        "[{}]",
        columns.map(|c| c.to_string()).collect::<Vec<_>>().join(", ")
    )
}

fn sort_keys(
    exprs: flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<fb::SortExprNode<'_>>>,
) -> String {
    exprs
        .iter()
        .map(|key| {
            format!(
                "{} {} {}",
                key.expr().map(|e| expr_text(&e)).unwrap_or_default(),
                if key.asc() { "asc" } else { "desc" },
                if key.nulls_first() {
                    "nulls first"
                } else {
                    "nulls last"
                }
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn schema_text(schema: &fb::Schema<'_>) -> String {
    let fields = schema
        .fields()
        .map(|fields| {
            fields
                .iter()
                .map(|f| format!("{}:{:?}", f.name().unwrap_or("?"), f.data_type()))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!("[{fields}]")
}

/// The expression as the buffer holds it. Ordinals rather than names where the wire
/// carries both, since the ordinal is what the executor reads.
fn expr_text(expr: &fb::Expr<'_>) -> String {
    match expr.node_type() {
        fb::ExprNode::ColumnRef => {
            let column = expr.node_as_column_ref().expect("a column");
            format!("{}@{}", column.name().unwrap_or("?"), column.index())
        }
        fb::ExprNode::LiteralExpr => expr
            .node_as_literal_expr()
            .and_then(|literal| literal.value())
            .map(|value| scalar_text(&value))
            .unwrap_or_else(|| "?".to_string()),
        fb::ExprNode::BinaryExprNode => {
            let binary = expr.node_as_binary_expr_node().expect("a binary");
            let out = if binary.out_decimal_precision() == 0 {
                String::new()
            } else {
                format!(
                    "::Decimal128({}, {})",
                    binary.out_decimal_precision(),
                    binary.out_decimal_scale()
                )
            };
            format!(
                "({} {:?} {}){out}",
                binary.left().map(|e| expr_text(&e)).unwrap_or_default(),
                binary.op(),
                binary.right().map(|e| expr_text(&e)).unwrap_or_default()
            )
        }
        fb::ExprNode::UnaryExprNode => {
            let unary = expr.node_as_unary_expr_node().expect("a unary");
            format!(
                "{:?}({})",
                unary.op(),
                unary.arg().map(|e| expr_text(&e)).unwrap_or_default()
            )
        }
        fb::ExprNode::CastExprNode => {
            let cast = expr.node_as_cast_expr_node().expect("a cast");
            let target = if cast.decimal_precision() == 0 {
                format!("{:?}", cast.target_type())
            } else {
                format!(
                    "Decimal128({}, {})",
                    cast.decimal_precision(),
                    cast.decimal_scale()
                )
            };
            format!(
                "cast({} as {target})",
                cast.expr().map(|e| expr_text(&e)).unwrap_or_default()
            )
        }
        fb::ExprNode::LikeExprNode => {
            let like = expr.node_as_like_expr_node().expect("a like");
            format!(
                "{} {}{} {}",
                like.expr().map(|e| expr_text(&e)).unwrap_or_default(),
                if like.negated() { "not " } else { "" },
                if like.case_insensitive() {
                    "ilike"
                } else {
                    "like"
                },
                like.pattern().map(|e| expr_text(&e)).unwrap_or_default()
            )
        }
        fb::ExprNode::CaseExprNode => {
            let case = expr.node_as_case_expr_node().expect("a case");
            let mut text = String::from("case");
            if let Some(comparand) = case.expr() {
                let _ = write!(text, " {}", expr_text(&comparand));
            }
            if let Some(arms) = case.when_thens() {
                for arm in arms.iter() {
                    let _ = write!(
                        text,
                        " when {} then {}",
                        arm.when().map(|e| expr_text(&e)).unwrap_or_default(),
                        arm.then().map(|e| expr_text(&e)).unwrap_or_default()
                    );
                }
            }
            if let Some(otherwise) = case.else_expr() {
                let _ = write!(text, " else {}", expr_text(&otherwise));
            }
            text.push_str(" end");
            text
        }
        fb::ExprNode::ScalarFunctionExprNode => {
            let function = expr
                .node_as_scalar_function_expr_node()
                .expect("a scalar function");
            let args = function
                .args()
                .map(|args| args.iter().map(|a| expr_text(&a)).collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            format!("{}({args})", function.name().unwrap_or("?"))
        }
        other => format!("{other:?}"),
    }
}

/// A literal as the wire holds it: the value, and for a decimal the precision and scale
/// that decide what the value means.
fn scalar_text(value: &fb::ScalarValue<'_>) -> String {
    if value.is_null() {
        return format!("null::{:?}", value.type_());
    }
    match value.type_() {
        fb::DataType::Boolean => value.bool_val().to_string(),
        fb::DataType::Utf8 | fb::DataType::LargeUtf8 | fb::DataType::Utf8View => {
            format!("'{}'", value.string_val().unwrap_or(""))
        }
        fb::DataType::Float32 | fb::DataType::Float64 => value.float_val().to_string(),
        fb::DataType::Decimal128 => {
            let raw = ((value.decimal_hi() as i128) << 64) | value.decimal_lo() as i128;
            format!(
                "{raw}::Decimal128({}, {})",
                value.decimal_precision(),
                value.decimal_scale()
            )
        }
        fb::DataType::UInt8
        | fb::DataType::UInt16
        | fb::DataType::UInt32
        | fb::DataType::UInt64 => value.uint_val().to_string(),
        _ => value.int_val().to_string(),
    }
}
