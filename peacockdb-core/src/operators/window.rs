//! Window family.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};

#[allow(unused_imports)]
use super::{gpu_exec_node, GpuExtraDisplay};
#[allow(unused_imports)]
use super::operator::{Operator, PartitionTopology};
use datafusion::physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};


gpu_exec_node!(GpuWindowExec);
impl GpuExtraDisplay for GpuWindowExec {
    fn extra_display_info(&self) -> String {
        // Window exprs live on either WindowAggExec or BoundedWindowAggExec.
        let names: Vec<String> = if let Some(w) =
            self.inner.as_any().downcast_ref::<WindowAggExec>()
        {
            w.window_expr().iter().map(|e| e.name().to_string()).collect()
        } else if let Some(w) = self.inner.as_any().downcast_ref::<BoundedWindowAggExec>() {
            w.window_expr().iter().map(|e| e.name().to_string()).collect()
        } else {
            vec![]
        };
        format!("wdw=[{}]", names.join(", "))
    }
}

// ---------------------------------------------------------------------------
// FlatBuffer wire format
//
// STATEMENT ORDER IS THE WIRE FORMAT. FlatBufferBuilder is a no-interning bump
// arena, so every builder call appends and returns an offset — reordering the
// statements below, or hoisting a create_string, changes the bytes even though the
// values are identical. Do not "tidy" these bodies. testdata/goldens/plan_bytes.sha256
// pins them; the C++ side reads what they emit.
// ---------------------------------------------------------------------------

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::generated::gpu_plan_generated::peacock::plan as fb;
use datafusion::common::ScalarValue as DfScalarValue;
use crate::plan_serializer::{deserialize_expr, deserialize_plan_node};
use datafusion::arrow::compute::SortOptions;
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::arrow::datatypes::DataType as ArrowDataType;
use datafusion::physical_plan::PhysicalExpr;
use crate::plan_serializer::convert_data_type;
use crate::plan_serializer::{serialize_expr, serialize_plan_node};
use crate::PartitionMode;

pub(crate) fn serialize_cudf_window<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    use datafusion::logical_expr::WindowFrameBound as DfBound;
    use datafusion::physical_expr::window::{
        PlainAggregateWindowExpr, SlidingAggregateWindowExpr, WindowExpr,
    };
    use datafusion::physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};

    let gpu_win = plan.as_any().downcast_ref::<GpuWindowExec>().unwrap();
    let inner = gpu_win.inner();

    // Window exprs + input live on either WindowAggExec (whole-partition frames)
    // or BoundedWindowAggExec (running / ranking frames).
    let (window_exprs, input_plan): (&[Arc<dyn WindowExpr>], &Arc<dyn ExecutionPlan>) =
        if let Some(w) = inner.as_any().downcast_ref::<WindowAggExec>() {
            (w.window_expr(), w.input())
        } else if let Some(w) = inner.as_any().downcast_ref::<BoundedWindowAggExec>() {
            (w.window_expr(), w.input())
        } else {
            return Err("GpuWindowExec inner is not a window exec".to_string());
        };
    let input_schema = input_plan.schema();

    let mut expr_offsets = Vec::new();
    for we in window_exprs {
        // Only aggregate windows (sum/avg/max/min/count) are supported today;
        // ranking functions (rank/row_number → StandardWindowExpr) are not yet.
        let (func_name_str, arg_exprs): (String, Vec<Arc<dyn PhysicalExpr>>) =
            if let Some(p) = we.as_any().downcast_ref::<PlainAggregateWindowExpr>() {
                let a = p.get_aggregate_expr();
                (a.fun().name().to_string(), a.expressions())
            } else if let Some(s) = we.as_any().downcast_ref::<SlidingAggregateWindowExpr>() {
                let a = s.get_aggregate_expr();
                (a.fun().name().to_string(), a.expressions())
            } else {
                return Err(format!(
                    "unsupported window function: {} (only aggregate windows supported)",
                    we.name()
                ));
            };

        let func_name = b.create_string(&func_name_str);
        let alias = b.create_string(we.name());

        let mut args = Vec::new();
        for arg in &arg_exprs {
            args.push(serialize_expr(b, arg, &input_schema)?);
        }
        let args_vec = b.create_vector(&args);

        let mut pby = Vec::new();
        for e in we.partition_by() {
            pby.push(serialize_expr(b, e, &input_schema)?);
        }
        let pby_vec = b.create_vector(&pby);

        let mut oby = Vec::new();
        for se in we.order_by().iter() {
            let e = serialize_expr(b, &se.expr, &input_schema)?;
            oby.push(fb::SortExprNode::create(
                b,
                &fb::SortExprNodeArgs {
                    expr: Some(e),
                    asc: !se.options.descending,
                    nulls_first: se.options.nulls_first,
                },
            ));
        }
        let oby_vec = b.create_vector(&oby);

        // Supported frames: start = UnboundedPreceding; end = CurrentRow
        // (running) or UnboundedFollowing (whole partition).
        let frame = we.get_window_frame();
        if !frame.start_bound.is_unbounded() {
            return Err(format!(
                "unsupported window frame start: {:?} (expected UNBOUNDED PRECEDING)",
                frame.start_bound
            ));
        }
        let frame_start = fb::WindowFrameBound::UnboundedPreceding;
        let frame_end = match &frame.end_bound {
            DfBound::CurrentRow => fb::WindowFrameBound::CurrentRow,
            bound if bound.is_unbounded() => fb::WindowFrameBound::UnboundedFollowing,
            other => {
                return Err(format!(
                    "unsupported window frame end: {other:?} (expected CURRENT ROW or UNBOUNDED FOLLOWING)"
                ))
            }
        };

        let out_field = we.field().map_err(|e| format!("window field: {e}"))?;
        let return_type = convert_data_type(out_field.data_type()).unwrap_or(fb::DataType::Null);
        let (out_decimal_precision, out_decimal_scale) = match out_field.data_type() {
            ArrowDataType::Decimal128(p, s) => (*p, *s),
            _ => (0, 0),
        };

        expr_offsets.push(fb::WindowExprNode::create(
            b,
            &fb::WindowExprNodeArgs {
                func_name: Some(func_name),
                args: Some(args_vec),
                partition_by: Some(pby_vec),
                order_by: Some(oby_vec),
                frame_start,
                frame_end,
                alias: Some(alias),
                return_type,
                out_decimal_precision,
                out_decimal_scale,
            },
        ));
    }
    let exprs_vec = b.create_vector(&expr_offsets);
    let input = serialize_plan_node(b, input_plan, pm)?;

    let node = fb::CudfWindow::create(
        b,
        &fb::CudfWindowArgs {
            window_exprs: Some(exprs_vec),
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::CudfWindow, node.as_union_value()))
}

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_cudf_window(win: &fb::CudfWindow) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::logical_expr::{
        WindowFrame, WindowFrameBound as DfBound, WindowFrameUnits, WindowFunctionDefinition,
    };
    use datafusion::physical_expr::LexOrdering;
    use datafusion::physical_plan::windows::{
        create_window_expr, BoundedWindowAggExec, WindowAggExec,
    };
    use datafusion::physical_plan::InputOrderMode;

    let input = deserialize_plan_node(&win.input().ok_or("CudfWindow missing input")?)?;
    let input_schema = input.schema();

    // DataFusion plans a running frame (… AND CURRENT ROW) as a streaming
    // BoundedWindowAggExec, which preserves the input's partitioning, but a
    // whole-partition frame (… AND UNBOUNDED FOLLOWING) as a WindowAggExec, which
    // collapses to a single partition. The exec type isn't on the wire (it doesn't
    // affect the GPU executor), but it changes the output partitioning that parent
    // nodes display, so pick it from the frame to keep the round-trip faithful.
    let mut running_frame = false;
    let mut window_exprs = Vec::new();
    if let Some(exprs) = win.window_exprs() {
        for i in 0..exprs.len() {
            let we = exprs.get(i);
            let func_name = we.func_name().ok_or("WindowExpr missing func_name")?;

            // Only aggregate windows are serialized (serialize_cudf_window rejects
            // ranking functions), so look the function up among aggregate UDFs.
            let udf = datafusion::functions_aggregate::all_default_aggregate_functions()
                .into_iter()
                .find(|u| u.name() == func_name)
                .ok_or_else(|| format!("unknown window aggregate function: {func_name}"))?;

            let args: Vec<Arc<dyn PhysicalExpr>> = we
                .args()
                .map(|a| {
                    (0..a.len())
                        .map(|j| deserialize_expr(&a.get(j)))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();

            let partition_by: Vec<Arc<dyn PhysicalExpr>> = we
                .partition_by()
                .map(|p| {
                    (0..p.len())
                        .map(|j| deserialize_expr(&p.get(j)))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();

            let order_by_exprs: Vec<PhysicalSortExpr> = we
                .order_by()
                .map(|ob| {
                    (0..ob.len())
                        .map(|j| {
                            let se = ob.get(j);
                            let expr =
                                deserialize_expr(&se.expr().ok_or("SortExpr missing expr")?)?;
                            Ok::<_, String>(PhysicalSortExpr::new(
                                expr,
                                SortOptions {
                                    descending: !se.asc(),
                                    nulls_first: se.nulls_first(),
                                },
                            ))
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
                .transpose()?
                .unwrap_or_default();
            let order_by: LexOrdering = order_by_exprs.into();

            // Supported frames: start = UNBOUNDED PRECEDING; end = CURRENT ROW or
            // UNBOUNDED FOLLOWING. The wire omits the frame units (irrelevant to the
            // GPU executor, which keys off the bounds), so reconstruct as RANGE —
            // the units affect neither re-serialization nor the round-trip oracle.
            let start_bound = DfBound::Preceding(DfScalarValue::Null);
            let end_bound = match we.frame_end() {
                fb::WindowFrameBound::CurrentRow => {
                    running_frame = true;
                    DfBound::CurrentRow
                }
                fb::WindowFrameBound::UnboundedFollowing => {
                    DfBound::Following(DfScalarValue::Null)
                }
                other => return Err(format!("unsupported window frame end: {other:?}")),
            };
            let frame = Arc::new(WindowFrame::new_bounds(
                WindowFrameUnits::Range,
                start_bound,
                end_bound,
            ));

            let alias = we.alias().unwrap_or(func_name).to_string();
            let fun = WindowFunctionDefinition::AggregateUDF(udf);
            let wexpr = create_window_expr(
                &fun,
                alias,
                &args,
                &partition_by,
                &order_by,
                frame,
                input_schema.as_ref(),
                false,
            )
            .map_err(|e| format!("create_window_expr: {e}"))?;
            window_exprs.push(wexpr);
        }
    }

    // partition_keys (repartition keys) aren't serialized and aren't read back on
    // re-serialization, so an empty set is faithful for the round-trip.
    let exec: Arc<dyn ExecutionPlan> = if running_frame {
        Arc::new(
            BoundedWindowAggExec::try_new(window_exprs, input, vec![], InputOrderMode::Sorted)
                .map_err(|e| format!("BoundedWindowAggExec: {e}"))?,
        )
    } else {
        Arc::new(
            WindowAggExec::try_new(window_exprs, input, vec![])
                .map_err(|e| format!("WindowAggExec: {e}"))?,
        )
    };
    Ok(Arc::new(GpuWindowExec::new(exec)))
}
// --- Operator: partition topology + strip behavior ------------------------

/// NOT stripped, same reason as `GpuCrossJoinExec` in `join.rs`.
impl Operator for GpuWindowExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Map
    }
    fn strips_to_inner(&self) -> bool {
        false
    }
}
