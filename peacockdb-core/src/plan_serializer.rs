// Serialize a DataFusion GPU physical plan tree into a FlatBuffer.
//
// Walks the `ExecutionPlan` tree produced by GpuExecutionRule, extracts the
// inner DataFusion nodes (FilterExec, ProjectionExec, etc.), and writes the
// corresponding FlatBuffer plan via the generated `peacock::plan` types.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType as ArrowDataType, Schema, SchemaRef};
use datafusion::common::ScalarValue as DfScalarValue;
use datafusion::physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr,
    Literal, NegativeExpr, NotExpr,
};
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::PhysicalExpr;
use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::generated::gpu_plan_generated::peacock::plan as fb;
use crate::gpu_rule::{
    GpuAggregateExec, GpuCoalesceBatchesExec, GpuCoalescePartitionsExec, GpuCrossJoinExec,
    GpuFilterExec, GpuGlobalLimitExec, GpuHashJoinExec, GpuInterleaveExec, GpuNestedLoopJoinExec,
    GpuProjectExec, GpuRepartitionExec, GpuScanExec, GpuSortExec, GpuSortPreservingMergeExec,
    GpuUnionExec, GpuWindowExec, PartitionMode,
};

/// Serialize an entire GPU execution plan tree into a FlatBuffer byte vector.
///
/// Returns `Err` if the plan contains nodes that cannot be serialized (e.g.
/// unsupported expression types or plan nodes)
pub fn serialize_plan(plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<u8>, String> {
    serialize_plan_mode(plan, PartitionMode::SinglePartition)
}

/// Like [`serialize_plan`] but at an explicit [`PartitionMode`]. The mode is
/// threaded to every `CudfAggregate` node so its `mergeable_agg_state` flag is set
/// iff [`PartitionMode::RealMultiPartition`] — driving the STDDEV/VAR partial-state
/// shape (see the flatbuffer field doc / #25). `serialize_plan` defaults to
/// `SinglePartition`, so existing callers and the flatbuffer roundtrips stay
/// byte-identical (the flag serializes as its `false` default and is omitted).
pub fn serialize_plan_mode(
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
) -> Result<Vec<u8>, String> {
    let mut builder = FlatBufferBuilder::with_capacity(4096);
    let root = serialize_plan_node(&mut builder, plan, pm)?;
    let gpu_plan = fb::GpuPlan::create(&mut builder, &fb::GpuPlanArgs { root: Some(root) });
    builder.finish(gpu_plan, None);
    Ok(builder.finished_data().to_vec())
}

// ---------------------------------------------------------------------------
// Plan nodes
// ---------------------------------------------------------------------------

pub(crate) fn serialize_plan_node<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
) -> Result<WIPOffset<fb::PlanNode<'a>>, String> {
    let output_schema = serialize_schema(b, &plan.schema());

    let (node_type, node_offset) = if let Some(scan) = plan.as_any().downcast_ref::<GpuScanExec>()
    {
        crate::operators::scan::serialize_cudf_scan(b, scan)?
    } else if plan.as_any().is::<GpuFilterExec>() {
        crate::operators::filter::serialize_cudf_filter(b, plan, pm)?
    } else if plan.as_any().is::<GpuProjectExec>() {
        crate::operators::project::serialize_cudf_project(b, plan, pm)?
    } else if plan.as_any().is::<GpuAggregateExec>() {
        crate::operators::aggregate::serialize_cudf_aggregate(b, plan, pm)?
    } else if plan.as_any().is::<GpuHashJoinExec>() {
        crate::operators::join::serialize_cudf_hash_join(b, plan, pm)?
    } else if plan.as_any().is::<GpuCrossJoinExec>() {
        crate::operators::join::serialize_cudf_cross_join(b, plan, pm)?
    } else if plan.as_any().is::<GpuNestedLoopJoinExec>() {
        crate::operators::join::serialize_cudf_nested_loop_join(b, plan, pm)?
    } else if plan.as_any().is::<GpuSortExec>() {
        crate::operators::sort::serialize_cudf_sort(b, plan, pm)?
    } else if plan.as_any().is::<GpuCoalesceBatchesExec>() {
        crate::operators::coalesce::serialize_cudf_coalesce_batches(b, plan, pm)?
    } else if plan.as_any().is::<GpuCoalescePartitionsExec>() {
        crate::operators::coalesce::serialize_cudf_coalesce_partitions(b, plan, pm)?
    } else if plan.as_any().is::<GpuRepartitionExec>() {
        crate::operators::repartition::serialize_cudf_repartition(b, plan, pm)?
    } else if plan.as_any().is::<GpuSortPreservingMergeExec>() {
        crate::operators::sort::serialize_cudf_sort_preserving_merge(b, plan, pm)?
    } else if plan.as_any().is::<GpuUnionExec>() {
        crate::operators::union::serialize_cudf_union(b, plan, false, pm)?
    } else if plan.as_any().is::<GpuInterleaveExec>() {
        crate::operators::union::serialize_cudf_union(b, plan, true, pm)?
    } else if plan.as_any().is::<GpuGlobalLimitExec>() {
        crate::operators::limit::serialize_cudf_limit(b, plan, pm)?
    } else if plan.as_any().is::<GpuWindowExec>() {
        crate::operators::window::serialize_cudf_window(b, plan, pm)?
    } else {
        return Err(format!("unsupported plan node: {}", plan.name()));
    };

    Ok(fb::PlanNode::create(
        b,
        &fb::PlanNodeArgs {
            node_type,
            node: Some(node_offset),
            output_schema: Some(output_schema),
        },
    ))
}


// ---------------------------------------------------------------------------
// The per-operator serialize arms live in `crate::operators::<family>` —
// statement order is the wire format, so they were relocated verbatim, not
// rewritten. What remains here is the framework: the dispatcher above (which still
// writes each node's output_schema FIRST and builds the enclosing PlanNode itself,
// because moving either into the operators would shift every node's bytes) plus the
// shared expression/schema/type helpers below, which more than one arm needs.
// ---------------------------------------------------------------------------


pub(crate) fn serialize_expr<'a>(
    b: &mut FlatBufferBuilder<'a>,
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> Result<WIPOffset<fb::Expr<'a>>, String> {
    let any = expr.as_any();

    let (node_type, node_offset) = if let Some(col) = any.downcast_ref::<Column>() {
        let name = b.create_string(col.name());
        let cr = fb::ColumnRef::create(
            b,
            &fb::ColumnRefArgs {
                index: col.index() as u32,
                name: Some(name),
            },
        );
        (fb::ExprNode::ColumnRef, cr.as_union_value())
    } else if let Some(lit) = any.downcast_ref::<Literal>() {
        let sv = serialize_scalar_value(b, lit.value())?;
        let le = fb::LiteralExpr::create(b, &fb::LiteralExprArgs { value: Some(sv) });
        (fb::ExprNode::LiteralExpr, le.as_union_value())
    } else if let Some(bin) = any.downcast_ref::<BinaryExpr>() {
        let left = serialize_expr(b, bin.left(), schema)?;
        let right = serialize_expr(b, bin.right(), schema)?;
        let op = convert_operator(bin.op())?;
        // DataFusion's declared decimal output scale, so the executor can match
        // its fixed_point result scale (esp. division, where cuDF differs).
        let (out_decimal_precision, out_decimal_scale) = match bin.data_type(schema) {
            Ok(ArrowDataType::Decimal128(p, s)) => (p, s),
            _ => (0, 0),
        };
        let be = fb::BinaryExprNode::create(
            b,
            &fb::BinaryExprNodeArgs {
                left: Some(left),
                op,
                right: Some(right),
                out_decimal_precision,
                out_decimal_scale,
            },
        );
        (fb::ExprNode::BinaryExprNode, be.as_union_value())
    } else if let Some(not) = any.downcast_ref::<NotExpr>() {
        let arg = serialize_expr(b, not.arg(), schema)?;
        let ue = fb::UnaryExprNode::create(
            b,
            &fb::UnaryExprNodeArgs {
                op: fb::UnaryOp::Not,
                arg: Some(arg),
            },
        );
        (fb::ExprNode::UnaryExprNode, ue.as_union_value())
    } else if let Some(is_null) = any.downcast_ref::<IsNullExpr>() {
        let arg = serialize_expr(b, is_null.arg(), schema)?;
        let ue = fb::UnaryExprNode::create(
            b,
            &fb::UnaryExprNodeArgs {
                op: fb::UnaryOp::IsNull,
                arg: Some(arg),
            },
        );
        (fb::ExprNode::UnaryExprNode, ue.as_union_value())
    } else if let Some(is_not_null) = any.downcast_ref::<IsNotNullExpr>() {
        let arg = serialize_expr(b, is_not_null.arg(), schema)?;
        let ue = fb::UnaryExprNode::create(
            b,
            &fb::UnaryExprNodeArgs {
                op: fb::UnaryOp::IsNotNull,
                arg: Some(arg),
            },
        );
        (fb::ExprNode::UnaryExprNode, ue.as_union_value())
    } else if let Some(neg) = any.downcast_ref::<NegativeExpr>() {
        let arg = serialize_expr(b, neg.arg(), schema)?;
        let ue = fb::UnaryExprNode::create(
            b,
            &fb::UnaryExprNodeArgs {
                op: fb::UnaryOp::Negative,
                arg: Some(arg),
            },
        );
        (fb::ExprNode::UnaryExprNode, ue.as_union_value())
    } else if let Some(cast) = any.downcast_ref::<CastExpr>() {
        let inner = serialize_expr(b, cast.expr(), schema)?;
        let target = convert_data_type(cast.cast_type())?;
        // The DataType enum can't carry decimal precision/scale, but the
        // executor needs the scale to reconstruct the cuDF fixed_point type.
        let (decimal_precision, decimal_scale) = match cast.cast_type() {
            ArrowDataType::Decimal128(p, s) => (*p, *s),
            _ => (0, 0),
        };
        let ce = fb::CastExprNode::create(
            b,
            &fb::CastExprNodeArgs {
                expr: Some(inner),
                target_type: target,
                decimal_precision,
                decimal_scale,
            },
        );
        (fb::ExprNode::CastExprNode, ce.as_union_value())
    } else if let Some(like) = any.downcast_ref::<LikeExpr>() {
        let inner = serialize_expr(b, like.expr(), schema)?;
        let pattern = serialize_expr(b, like.pattern(), schema)?;
        let le = fb::LikeExprNode::create(
            b,
            &fb::LikeExprNodeArgs {
                expr: Some(inner),
                pattern: Some(pattern),
                negated: like.negated(),
                case_insensitive: like.case_insensitive(),
            },
        );
        (fb::ExprNode::LikeExprNode, le.as_union_value())
    } else if let Some(case) = any.downcast_ref::<CaseExpr>() {
        let comparand = match case.expr() {
            Some(e) => Some(serialize_expr(b, e, schema)?),
            None => None,
        };
        let mut whens = Vec::new();
        for (when, then) in case.when_then_expr() {
            let w = serialize_expr(b, when, schema)?;
            let t = serialize_expr(b, then, schema)?;
            whens.push(fb::CaseWhenThen::create(
                b,
                &fb::CaseWhenThenArgs {
                    when: Some(w),
                    then: Some(t),
                },
            ));
        }
        let whens_vec = b.create_vector(&whens);
        let else_ = match case.else_expr() {
            Some(e) => Some(serialize_expr(b, e, schema)?),
            None => None,
        };
        let ce = fb::CaseExprNode::create(
            b,
            &fb::CaseExprNodeArgs {
                expr: comparand,
                when_thens: Some(whens_vec),
                else_expr: else_,
            },
        );
        (fb::ExprNode::CaseExprNode, ce.as_union_value())
    } else if any.downcast_ref::<InListExpr>().is_some() {
        // IN-lists are lowered to OR-chains by GpuExecutionRule before the plan
        // reaches the serializer (cuDF AST has no IN opcode). Hitting one here
        // means a plan node carrying an IN-list wasn't covered by that pass.
        return Err(format!(
            "InListExpr reached the serializer un-lowered (GpuExecutionRule should \
             have expanded it to an OR-chain): {expr}"
        ));
    } else if let Some(sf) = any.downcast_ref::<ScalarFunctionExpr>() {
        let name = b.create_string(sf.name());
        let mut args = Vec::new();
        for arg in sf.args() {
            args.push(serialize_expr(b, arg, schema)?);
        }
        let args_vec = b.create_vector(&args);
        let ret = convert_data_type(sf.return_type())?;
        let (return_decimal_precision, return_decimal_scale) = match sf.return_type() {
            ArrowDataType::Decimal128(p, s) => (*p, *s),
            _ => (0, 0),
        };
        let sfn = fb::ScalarFunctionExprNode::create(
            b,
            &fb::ScalarFunctionExprNodeArgs {
                name: Some(name),
                args: Some(args_vec),
                return_type: ret,
                return_decimal_precision,
                return_decimal_scale,
                nullable: sf.nullable(),
            },
        );
        (fb::ExprNode::ScalarFunctionExprNode, sfn.as_union_value())
    } else {
        return Err(format!(
            "unsupported physical expression: {}",
            expr
        ));
    };

    Ok(fb::Expr::create(
        b,
        &fb::ExprArgs {
            node_type,
            node: Some(node_offset),
        },
    ))
}

// ---------------------------------------------------------------------------
// Scalar values
// ---------------------------------------------------------------------------

pub(crate) fn serialize_scalar_value<'a>(
    b: &mut FlatBufferBuilder<'a>,
    sv: &DfScalarValue,
) -> Result<WIPOffset<fb::ScalarValue<'a>>, String> {
    let mut args = fb::ScalarValueArgs::default();

    match sv {
        DfScalarValue::Null => {
            args.type_ = fb::DataType::Null;
        }
        DfScalarValue::Boolean(Some(v)) => {
            args.type_ = fb::DataType::Boolean;
            args.bool_val = *v;
        }
        DfScalarValue::Int8(Some(v)) => {
            args.type_ = fb::DataType::Int8;
            args.int_val = *v as i64;
        }
        DfScalarValue::Int16(Some(v)) => {
            args.type_ = fb::DataType::Int16;
            args.int_val = *v as i64;
        }
        DfScalarValue::Int32(Some(v)) => {
            args.type_ = fb::DataType::Int32;
            args.int_val = *v as i64;
        }
        DfScalarValue::Int64(Some(v)) => {
            args.type_ = fb::DataType::Int64;
            args.int_val = *v;
        }
        DfScalarValue::UInt8(Some(v)) => {
            args.type_ = fb::DataType::UInt8;
            args.uint_val = *v as u64;
        }
        DfScalarValue::UInt16(Some(v)) => {
            args.type_ = fb::DataType::UInt16;
            args.uint_val = *v as u64;
        }
        DfScalarValue::UInt32(Some(v)) => {
            args.type_ = fb::DataType::UInt32;
            args.uint_val = *v as u64;
        }
        DfScalarValue::UInt64(Some(v)) => {
            args.type_ = fb::DataType::UInt64;
            args.uint_val = *v;
        }
        DfScalarValue::Float32(Some(v)) => {
            args.type_ = fb::DataType::Float32;
            args.float_val = *v as f64;
        }
        DfScalarValue::Float64(Some(v)) => {
            args.type_ = fb::DataType::Float64;
            args.float_val = *v;
        }
        DfScalarValue::Utf8(Some(s)) | DfScalarValue::LargeUtf8(Some(s)) => {
            args.type_ = fb::DataType::Utf8;
            args.string_val = Some(b.create_string(s));
        }
        DfScalarValue::Utf8View(Some(s)) => {
            // Utf8View is a DataFusion 45+ optimizer rewrite of string literals;
            // cuDF doesn't distinguish view vs. owned strings. Preserve the type
            // tag for faithful roundtrip, but the wire payload is identical.
            args.type_ = fb::DataType::Utf8View;
            args.string_val = Some(b.create_string(s));
        }
        DfScalarValue::Date32(Some(d)) => {
            args.type_ = fb::DataType::Date32;
            args.int_val = *d as i64;
        }
        DfScalarValue::Decimal128(Some(v), prec, scale) => {
            args.type_ = fb::DataType::Decimal128;
            args.decimal_hi = (*v >> 64) as i64;
            args.decimal_lo = *v as u64;
            args.decimal_precision = *prec;
            args.decimal_scale = *scale as i8;
        }
        // Treat any None variant as typed null. The `is_null` flag is what
        // distinguishes it from a zero value on the wire.
        other if other.is_null() => {
            args.type_ = convert_data_type(&other.data_type())?;
            args.is_null = true;
            if let DfScalarValue::Decimal128(_, prec, scale) = other {
                args.decimal_precision = *prec;
                args.decimal_scale = *scale as i8;
            }
        }
        other => {
            return Err(format!("unsupported scalar value: {other:?}"));
        }
    }

    Ok(fb::ScalarValue::create(b, &args))
}

// ---------------------------------------------------------------------------
// Arrow type / operator conversions
// ---------------------------------------------------------------------------

pub(crate) fn convert_data_type(dt: &ArrowDataType) -> Result<fb::DataType, String> {
    Ok(match dt {
        ArrowDataType::Null => fb::DataType::Null,
        ArrowDataType::Boolean => fb::DataType::Boolean,
        ArrowDataType::Int8 => fb::DataType::Int8,
        ArrowDataType::Int16 => fb::DataType::Int16,
        ArrowDataType::Int32 => fb::DataType::Int32,
        ArrowDataType::Int64 => fb::DataType::Int64,
        ArrowDataType::UInt8 => fb::DataType::UInt8,
        ArrowDataType::UInt16 => fb::DataType::UInt16,
        ArrowDataType::UInt32 => fb::DataType::UInt32,
        ArrowDataType::UInt64 => fb::DataType::UInt64,
        ArrowDataType::Float16 => fb::DataType::Float16,
        ArrowDataType::Float32 => fb::DataType::Float32,
        ArrowDataType::Float64 => fb::DataType::Float64,
        ArrowDataType::Utf8 => fb::DataType::Utf8,
        ArrowDataType::LargeUtf8 => fb::DataType::LargeUtf8,
        ArrowDataType::Binary => fb::DataType::Binary,
        ArrowDataType::LargeBinary => fb::DataType::LargeBinary,
        ArrowDataType::Date32 => fb::DataType::Date32,
        ArrowDataType::Date64 => fb::DataType::Date64,
        ArrowDataType::Decimal128(_, _) => fb::DataType::Decimal128,
        ArrowDataType::Utf8View => fb::DataType::Utf8View,
        ArrowDataType::BinaryView => fb::DataType::BinaryView,
        other => return Err(format!("unsupported Arrow data type: {other:?}")),
    })
}

fn convert_operator(
    op: &datafusion::logical_expr::Operator,
) -> Result<fb::BinaryOp, String> {
    use datafusion::logical_expr::Operator as Op;
    Ok(match op {
        Op::Eq => fb::BinaryOp::Eq,
        Op::NotEq => fb::BinaryOp::NotEq,
        Op::Lt => fb::BinaryOp::Lt,
        Op::LtEq => fb::BinaryOp::LtEq,
        Op::Gt => fb::BinaryOp::Gt,
        Op::GtEq => fb::BinaryOp::GtEq,
        Op::Plus => fb::BinaryOp::Plus,
        Op::Minus => fb::BinaryOp::Minus,
        Op::Multiply => fb::BinaryOp::Multiply,
        Op::Divide => fb::BinaryOp::Divide,
        Op::Modulo => fb::BinaryOp::Modulo,
        Op::And => fb::BinaryOp::And,
        Op::Or => fb::BinaryOp::Or,
        Op::BitwiseAnd => fb::BinaryOp::BitwiseAnd,
        Op::BitwiseOr => fb::BinaryOp::BitwiseOr,
        Op::BitwiseXor => fb::BinaryOp::BitwiseXor,
        Op::BitwiseShiftLeft => fb::BinaryOp::BitwiseShiftLeft,
        Op::BitwiseShiftRight => fb::BinaryOp::BitwiseShiftRight,
        Op::StringConcat => fb::BinaryOp::StringConcat,
        Op::IsDistinctFrom => fb::BinaryOp::IsDistinctFrom,
        Op::IsNotDistinctFrom => fb::BinaryOp::IsNotDistinctFrom,
        other => return Err(format!("unsupported binary operator: {other:?}")),
    })
}

// ---------------------------------------------------------------------------
// Schema serialization
// ---------------------------------------------------------------------------

pub(crate) fn serialize_schema<'a>(
    b: &mut FlatBufferBuilder<'a>,
    schema: &SchemaRef,
) -> WIPOffset<fb::Schema<'a>> {
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| {
            let name = b.create_string(f.name());
            let dt = convert_data_type(f.data_type()).unwrap_or(fb::DataType::Null);
            let (decimal_precision, decimal_scale) = match f.data_type() {
                ArrowDataType::Decimal128(p, s) => (*p, *s),
                _ => (0, 0),
            };
            fb::Field::create(
                b,
                &fb::FieldArgs {
                    name: Some(name),
                    data_type: dt,
                    nullable: f.is_nullable(),
                    decimal_precision,
                    decimal_scale,
                },
            )
        })
        .collect();
    let fields_vec = b.create_vector(&fields);
    fb::Schema::create(b, &fb::SchemaArgs { fields: Some(fields_vec) })
}

// ---------------------------------------------------------------------------
// Deserialization: FlatBuffer → ExecutionPlan
// ---------------------------------------------------------------------------

/// Deserialize a FlatBuffer byte buffer into an `ExecutionPlan` tree.
///
/// The reconstructed plan uses the same GPU exec node types
/// (`GpuScanExec`, `GpuFilterExec`, etc.) wrapping real DataFusion nodes
/// built from the serialized expressions and schemas. Pass-through CPU nodes
/// (CoalesceBatches, Repartition, etc.) are not present in the flatbuffer
/// and are therefore not reconstructed.
pub fn deserialize_plan(bytes: &[u8]) -> Result<Arc<dyn ExecutionPlan>, String> {
    // Plans nest arbitrarily deep (TPC-DS q8 exceeds the verifier's default
    // depth limit, and the verifier + recursive descent below overflow the
    // default 2 MiB thread stack on the deepest plans). Run both on a thread
    // with a generous stack and a raised `max_depth`, which keeps the verifier's
    // malformed-buffer guard intact.
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn_scoped(s, || {
                let opts = flatbuffers::VerifierOptions {
                    max_depth: 1024,
                    ..Default::default()
                };
                let gpu_plan = flatbuffers::root_with_opts::<fb::GpuPlan>(&opts, bytes)
                    .map_err(|e| format!("invalid FlatBuffer: {e}"))?;
                let root = gpu_plan.root().ok_or("GpuPlan has no root node")?;
                deserialize_plan_node(&root)
            })
            .expect("spawn deserialization thread")
            .join()
            .map_err(|_| "deserialization thread panicked".to_string())?
    })
}

pub(crate) fn deserialize_plan_node(node: &fb::PlanNode) -> Result<Arc<dyn ExecutionPlan>, String> {
    match node.node_type() {
        fb::PlanNodeKind::CudfScan => {
            let scan = node.node_as_cudf_scan().ok_or("expected CudfScan")?;
            crate::operators::scan::deserialize_cudf_scan(&scan, node)
        }
        fb::PlanNodeKind::CudfFilter => {
            let filter = node.node_as_cudf_filter().ok_or("expected CudfFilter")?;
            crate::operators::filter::deserialize_cudf_filter(&filter, node)
        }
        fb::PlanNodeKind::CudfProject => {
            let proj = node.node_as_cudf_project().ok_or("expected CudfProject")?;
            crate::operators::project::deserialize_cudf_project(&proj, node)
        }
        fb::PlanNodeKind::CudfAggregate => {
            let agg = node.node_as_cudf_aggregate().ok_or("expected CudfAggregate")?;
            crate::operators::aggregate::deserialize_cudf_aggregate(&agg, node)
        }
        fb::PlanNodeKind::CudfHashJoin => {
            let join = node.node_as_cudf_hash_join().ok_or("expected CudfHashJoin")?;
            crate::operators::join::deserialize_cudf_hash_join(&join, node)
        }
        fb::PlanNodeKind::CudfCrossJoin => {
            let join = node.node_as_cudf_cross_join().ok_or("expected CudfCrossJoin")?;
            crate::operators::join::deserialize_cudf_cross_join(&join)
        }
        fb::PlanNodeKind::CudfNestedLoopJoin => {
            let join = node
                .node_as_cudf_nested_loop_join()
                .ok_or("expected CudfNestedLoopJoin")?;
            crate::operators::join::deserialize_cudf_nested_loop_join(&join, node)
        }
        fb::PlanNodeKind::CudfSort => {
            let sort = node.node_as_cudf_sort().ok_or("expected CudfSort")?;
            crate::operators::sort::deserialize_cudf_sort(&sort, node)
        }
        fb::PlanNodeKind::CudfCoalesceBatches => {
            let cb = node.node_as_cudf_coalesce_batches().ok_or("expected CudfCoalesceBatches")?;
            crate::operators::coalesce::deserialize_cudf_coalesce_batches(&cb)
        }
        fb::PlanNodeKind::CudfCoalescePartitions => {
            let cp = node.node_as_cudf_coalesce_partitions().ok_or("expected CudfCoalescePartitions")?;
            crate::operators::coalesce::deserialize_cudf_coalesce_partitions(&cp)
        }
        fb::PlanNodeKind::CudfRepartition => {
            let rp = node.node_as_cudf_repartition().ok_or("expected CudfRepartition")?;
            crate::operators::repartition::deserialize_cudf_repartition(&rp)
        }
        fb::PlanNodeKind::CudfSortPreservingMerge => {
            let spm = node.node_as_cudf_sort_preserving_merge().ok_or("expected CudfSortPreservingMerge")?;
            crate::operators::sort::deserialize_cudf_sort_preserving_merge(&spm)
        }
        fb::PlanNodeKind::CudfUnion => {
            let u = node.node_as_cudf_union().ok_or("expected CudfUnion")?;
            crate::operators::union::deserialize_cudf_union(&u)
        }
        fb::PlanNodeKind::CudfLimit => {
            let l = node.node_as_cudf_limit().ok_or("expected CudfLimit")?;
            crate::operators::limit::deserialize_cudf_limit(&l)
        }
        fb::PlanNodeKind::CudfWindow => {
            let w = node.node_as_cudf_window().ok_or("expected CudfWindow")?;
            crate::operators::window::deserialize_cudf_window(&w)
        }
        other => Err(format!("unknown PlanNodeKind: {:?}", other)),
    }
}

pub(crate) fn deserialize_schema(schema: &fb::Schema) -> SchemaRef {
    let fields: Vec<datafusion::arrow::datatypes::Field> = schema
        .fields()
        .map(|v| {
            (0..v.len())
                .map(|i| {
                    let f = v.get(i);
                    // Decimal128 carries its precision/scale in dedicated fields
                    // (the DataType enum can't); reconstruct the exact type so the
                    // schema — and every downstream expression result scale derived
                    // from it — round-trips faithfully.
                    let dt = match f.data_type() {
                        fb::DataType::Decimal128 => {
                            ArrowDataType::Decimal128(f.decimal_precision(), f.decimal_scale())
                        }
                        other => fb_to_arrow_type(other),
                    };
                    datafusion::arrow::datatypes::Field::new(
                        f.name().unwrap_or(""),
                        dt,
                        f.nullable(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    Arc::new(datafusion::arrow::datatypes::Schema::new(fields))
}

pub(crate) fn fb_to_arrow_type(dt: fb::DataType) -> ArrowDataType {
    match dt {
        fb::DataType::Null => ArrowDataType::Null,
        fb::DataType::Boolean => ArrowDataType::Boolean,
        fb::DataType::Int8 => ArrowDataType::Int8,
        fb::DataType::Int16 => ArrowDataType::Int16,
        fb::DataType::Int32 => ArrowDataType::Int32,
        fb::DataType::Int64 => ArrowDataType::Int64,
        fb::DataType::UInt8 => ArrowDataType::UInt8,
        fb::DataType::UInt16 => ArrowDataType::UInt16,
        fb::DataType::UInt32 => ArrowDataType::UInt32,
        fb::DataType::UInt64 => ArrowDataType::UInt64,
        fb::DataType::Float16 => ArrowDataType::Float16,
        fb::DataType::Float32 => ArrowDataType::Float32,
        fb::DataType::Float64 => ArrowDataType::Float64,
        fb::DataType::Utf8 => ArrowDataType::Utf8,
        fb::DataType::LargeUtf8 => ArrowDataType::LargeUtf8,
        fb::DataType::Binary => ArrowDataType::Binary,
        fb::DataType::LargeBinary => ArrowDataType::LargeBinary,
        fb::DataType::Date32 => ArrowDataType::Date32,
        fb::DataType::Date64 => ArrowDataType::Date64,
        fb::DataType::Decimal128 => ArrowDataType::Decimal128(38, 10),
        fb::DataType::Utf8View => ArrowDataType::Utf8View,
        fb::DataType::BinaryView => ArrowDataType::BinaryView,
        _ => ArrowDataType::Null,
    }
}

pub(crate) fn deserialize_expr(expr: &fb::Expr) -> Result<Arc<dyn PhysicalExpr>, String> {
    match expr.node_type() {
        fb::ExprNode::ColumnRef => {
            let col = expr.node_as_column_ref().ok_or("expected ColumnRef")?;
            Ok(Arc::new(Column::new(
                col.name().unwrap_or(""),
                col.index() as usize,
            )))
        }
        fb::ExprNode::LiteralExpr => {
            let lit = expr.node_as_literal_expr().ok_or("expected LiteralExpr")?;
            let sv = lit.value().ok_or("LiteralExpr has no value")?;
            Ok(Arc::new(Literal::new(deserialize_scalar(&sv)?)))
        }
        fb::ExprNode::BinaryExprNode => {
            let bin = expr
                .node_as_binary_expr_node()
                .ok_or("expected BinaryExprNode")?;
            let left = deserialize_expr(&bin.left().ok_or("BinaryExpr missing left")?)?;
            let right = deserialize_expr(&bin.right().ok_or("BinaryExpr missing right")?)?;
            let op = fb_to_operator(bin.op())?;
            Ok(Arc::new(BinaryExpr::new(left, op, right)))
        }
        fb::ExprNode::UnaryExprNode => {
            let un = expr
                .node_as_unary_expr_node()
                .ok_or("expected UnaryExprNode")?;
            let arg = deserialize_expr(&un.arg().ok_or("UnaryExpr missing arg")?)?;
            match un.op() {
                fb::UnaryOp::Not => Ok(Arc::new(NotExpr::new(arg))),
                fb::UnaryOp::IsNull => Ok(Arc::new(IsNullExpr::new(arg))),
                fb::UnaryOp::IsNotNull => Ok(Arc::new(IsNotNullExpr::new(arg))),
                fb::UnaryOp::Negative => Ok(Arc::new(NegativeExpr::new(arg))),
                other => Err(format!("unsupported UnaryOp: {:?}", other)),
            }
        }
        fb::ExprNode::CastExprNode => {
            let cast = expr
                .node_as_cast_expr_node()
                .ok_or("expected CastExprNode")?;
            let inner = deserialize_expr(&cast.expr().ok_or("CastExpr missing expr")?)?;
            // Decimal128 carries its precision/scale in dedicated fields (the
            // DataType enum can't), so reconstruct the exact type rather than
            // the placeholder fb_to_arrow_type returns.
            let target = match cast.target_type() {
                fb::DataType::Decimal128 => {
                    ArrowDataType::Decimal128(cast.decimal_precision(), cast.decimal_scale())
                }
                other => fb_to_arrow_type(other),
            };
            Ok(Arc::new(CastExpr::new(inner, target, None)))
        }
        fb::ExprNode::LikeExprNode => {
            let l = expr.node_as_like_expr_node().ok_or("expected LikeExprNode")?;
            let inner = deserialize_expr(&l.expr().ok_or("LikeExpr missing expr")?)?;
            let pat = deserialize_expr(&l.pattern().ok_or("LikeExpr missing pattern")?)?;
            Ok(Arc::new(LikeExpr::new(
                l.negated(),
                l.case_insensitive(),
                inner,
                pat,
            )))
        }
        fb::ExprNode::CaseExprNode => {
            let c = expr.node_as_case_expr_node().ok_or("expected CaseExprNode")?;
            let comparand = match c.expr() {
                Some(e) => Some(deserialize_expr(&e)?),
                None => None,
            };
            let mut whens = Vec::new();
            if let Some(wts) = c.when_thens() {
                for i in 0..wts.len() {
                    let wt = wts.get(i);
                    let when = deserialize_expr(&wt.when().ok_or("CaseWhenThen missing when")?)?;
                    let then = deserialize_expr(&wt.then().ok_or("CaseWhenThen missing then")?)?;
                    whens.push((when, then));
                }
            }
            let else_ = match c.else_expr() {
                Some(e) => Some(deserialize_expr(&e)?),
                None => None,
            };
            Ok(Arc::new(
                CaseExpr::try_new(comparand, whens, else_)
                    .map_err(|e| format!("CaseExpr::try_new: {e}"))?,
            ))
        }
        fb::ExprNode::ScalarFunctionExprNode => {
            let s = expr
                .node_as_scalar_function_expr_node()
                .ok_or("expected ScalarFunctionExprNode")?;
            let name = s.name().ok_or("ScalarFunctionExpr missing name")?;
            let mut args = Vec::new();
            if let Some(a) = s.args() {
                for i in 0..a.len() {
                    args.push(deserialize_expr(&a.get(i))?);
                }
            }
            let udf = datafusion::functions::all_default_functions()
                .into_iter()
                .find(|u| u.name() == name)
                .ok_or_else(|| format!("unknown scalar function: {name}"))?;
            let return_type = match s.return_type() {
                fb::DataType::Decimal128 => ArrowDataType::Decimal128(
                    s.return_decimal_precision(),
                    s.return_decimal_scale(),
                ),
                other => fb_to_arrow_type(other),
            };
            // ScalarFunctionExpr::new defaults nullable=true; restore the
            // serialized nullability so the result field round-trips.
            Ok(Arc::new(
                ScalarFunctionExpr::new(name, udf, args, return_type)
                    .with_nullable(s.nullable()),
            ))
        }
        other => Err(format!("unsupported ExprNode type: {:?}", other)),
    }
}

pub(crate) fn deserialize_scalar(sv: &fb::ScalarValue) -> Result<DfScalarValue, String> {
    // A typed NULL literal: reconstruct the `None` variant of the right type.
    if sv.is_null() {
        return Ok(match sv.type_() {
            fb::DataType::Decimal128 => {
                DfScalarValue::Decimal128(None, sv.decimal_precision(), sv.decimal_scale() as i8)
            }
            other => {
                let dt = fb_to_arrow_type(other);
                DfScalarValue::try_from(&dt)
                    .map_err(|e| format!("null scalar of type {dt:?}: {e}"))?
            }
        });
    }
    Ok(match sv.type_() {
        fb::DataType::Null => DfScalarValue::Null,
        fb::DataType::Boolean => DfScalarValue::Boolean(Some(sv.bool_val())),
        fb::DataType::Int8 => DfScalarValue::Int8(Some(sv.int_val() as i8)),
        fb::DataType::Int16 => DfScalarValue::Int16(Some(sv.int_val() as i16)),
        fb::DataType::Int32 => DfScalarValue::Int32(Some(sv.int_val() as i32)),
        fb::DataType::Int64 => DfScalarValue::Int64(Some(sv.int_val())),
        fb::DataType::UInt8 => DfScalarValue::UInt8(Some(sv.uint_val() as u8)),
        fb::DataType::UInt16 => DfScalarValue::UInt16(Some(sv.uint_val() as u16)),
        fb::DataType::UInt32 => DfScalarValue::UInt32(Some(sv.uint_val() as u32)),
        fb::DataType::UInt64 => DfScalarValue::UInt64(Some(sv.uint_val())),
        fb::DataType::Float32 => DfScalarValue::Float32(Some(sv.float_val() as f32)),
        fb::DataType::Float64 => DfScalarValue::Float64(Some(sv.float_val())),
        fb::DataType::Utf8 => {
            DfScalarValue::Utf8(Some(sv.string_val().unwrap_or("").to_string()))
        }
        fb::DataType::LargeUtf8 => {
            DfScalarValue::LargeUtf8(Some(sv.string_val().unwrap_or("").to_string()))
        }
        fb::DataType::Utf8View => {
            DfScalarValue::Utf8View(Some(sv.string_val().unwrap_or("").to_string()))
        }
        fb::DataType::Date32 => DfScalarValue::Date32(Some(sv.int_val() as i32)),
        fb::DataType::Decimal128 => {
            let hi = sv.decimal_hi() as i128;
            let lo = sv.decimal_lo() as i128;
            let val = (hi << 64) | (lo & 0xFFFF_FFFF_FFFF_FFFF);
            DfScalarValue::Decimal128(Some(val), sv.decimal_precision(), sv.decimal_scale() as i8)
        }
        other => return Err(format!("unsupported scalar DataType: {:?}", other)),
    })
}

pub(crate) fn fb_to_operator(op: fb::BinaryOp) -> Result<datafusion::logical_expr::Operator, String> {
    use datafusion::logical_expr::Operator as Op;
    Ok(match op {
        fb::BinaryOp::Eq => Op::Eq,
        fb::BinaryOp::NotEq => Op::NotEq,
        fb::BinaryOp::Lt => Op::Lt,
        fb::BinaryOp::LtEq => Op::LtEq,
        fb::BinaryOp::Gt => Op::Gt,
        fb::BinaryOp::GtEq => Op::GtEq,
        fb::BinaryOp::Plus => Op::Plus,
        fb::BinaryOp::Minus => Op::Minus,
        fb::BinaryOp::Multiply => Op::Multiply,
        fb::BinaryOp::Divide => Op::Divide,
        fb::BinaryOp::Modulo => Op::Modulo,
        fb::BinaryOp::And => Op::And,
        fb::BinaryOp::Or => Op::Or,
        fb::BinaryOp::BitwiseAnd => Op::BitwiseAnd,
        fb::BinaryOp::BitwiseOr => Op::BitwiseOr,
        fb::BinaryOp::BitwiseXor => Op::BitwiseXor,
        fb::BinaryOp::BitwiseShiftLeft => Op::BitwiseShiftLeft,
        fb::BinaryOp::BitwiseShiftRight => Op::BitwiseShiftRight,
        fb::BinaryOp::StringConcat => Op::StringConcat,
        fb::BinaryOp::IsDistinctFrom => Op::IsDistinctFrom,
        fb::BinaryOp::IsNotDistinctFrom => Op::IsNotDistinctFrom,
        other => return Err(format!("unsupported BinaryOp: {:?}", other)),
    })
}

// ---------------------------------------------------------------------------
// The per-operator deserialize arms live in `crate::operators::<family>`,
// each one sitting next to the serializer it mirrors. They are one contract: the
// round-trip identity depends on pairings that are invisible if the halves live in
// different files (GpuScanExec's row_groups override; the batches_map path-dedup
// that keeps GpuRepartitionExec.input_partitions from flipping 8<->1; Decimal128
// precision/scale; window partition_keys, written-and-deliberately-not-read).
// What remains here is the framework + the shared expr/schema/scalar helpers.
// ---------------------------------------------------------------------------
