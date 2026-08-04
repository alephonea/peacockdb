//! Aggregate family.

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
use datafusion::physical_plan::aggregates::AggregateExec;


gpu_exec_node!(GpuAggregateExec);
impl GpuExtraDisplay for GpuAggregateExec {
    fn extra_display_info(&self) -> String {
        let agg = self.inner.as_any().downcast_ref::<AggregateExec>().unwrap();
        let groups: Vec<&str> = agg.group_expr().expr().iter()
            .map(|(_, name): &(_, String)| name.as_str())
            .collect();
        let aggrs: Vec<&str> = agg.aggr_expr().iter()
            .map(|e| e.name())
            .collect();
        format!("group_by=[{}], aggr=[{}]", groups.join(", "), aggrs.join(", "))
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
use datafusion::physical_expr::aggregate::AggregateExprBuilder;
use datafusion::physical_plan::aggregates::PhysicalGroupBy;
use datafusion::physical_plan::PhysicalExpr;
use crate::plan_serializer::{deserialize_expr, deserialize_plan_node, deserialize_schema};
use datafusion::arrow::datatypes::DataType as ArrowDataType;
use datafusion::physical_plan::aggregates::AggregateMode as DfAggMode;
use crate::plan_serializer::{serialize_expr, serialize_plan_node, serialize_schema};
use crate::PartitionMode;

pub(crate) fn serialize_gpu_aggregate<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu_agg = plan
        .as_any()
        .downcast_ref::<GpuAggregateExec>()
        .unwrap();
    let agg = gpu_agg
        .inner()
        .as_any()
        .downcast_ref::<AggregateExec>()
        .ok_or("GpuAggregateExec inner is not AggregateExec")?;

    let mode = match agg.mode() {
        DfAggMode::Partial => fb::AggregateMode::Partial,
        DfAggMode::Final => fb::AggregateMode::Final,
        DfAggMode::FinalPartitioned => fb::AggregateMode::FinalPartitioned,
        DfAggMode::Single => fb::AggregateMode::Single,
        DfAggMode::SinglePartitioned => fb::AggregateMode::SinglePartitioned,
    };

    let group_by = agg.group_expr();
    let mut group_exprs = Vec::new();
    let mut group_names = Vec::new();
    for (expr, name) in group_by.expr() {
        group_exprs.push(serialize_expr(b, expr, &agg.input().schema())?);
        group_names.push(b.create_string(name));
    }
    let group_exprs_vec = b.create_vector(&group_exprs);
    let group_names_vec = b.create_vector(&group_names);

    // ROLLUP/CUBE/GROUPING SETS state. Empty for regular GROUP BY.
    let mut null_exprs = Vec::new();
    let mut null_names = Vec::new();
    for (expr, name) in group_by.null_expr() {
        null_exprs.push(serialize_expr(b, expr, &agg.input().schema())?);
        null_names.push(b.create_string(name));
    }
    let null_exprs_vec = b.create_vector(&null_exprs);
    let null_names_vec = b.create_vector(&null_names);

    let mut grouping_set_offsets = Vec::new();
    for set in group_by.groups() {
        let values = b.create_vector(set.as_slice());
        grouping_set_offsets.push(fb::GroupingSetMask::create(
            b,
            &fb::GroupingSetMaskArgs {
                values: Some(values),
            },
        ));
    }
    let grouping_sets_vec = b.create_vector(&grouping_set_offsets);

    let mut aggr_funcs = Vec::new();
    for aggr in agg.aggr_expr() {
        let func_name = b.create_string(aggr.fun().name());
        let alias = b.create_string(aggr.name());
        let mut arg_offsets = Vec::new();
        for arg in aggr.expressions() {
            arg_offsets.push(serialize_expr(b, &arg, &agg.input_schema())?);
        }
        let args = b.create_vector(&arg_offsets);
        // DataFusion's declared final output type (e.g. avg(Decimal(p,s)) →
        // Decimal(p+4, s+4)); cuDF's mean keeps the input scale, so the executor
        // casts the input to this scale before averaging.
        let (out_decimal_precision, out_decimal_scale) = match aggr.field().data_type() {
            ArrowDataType::Decimal128(p, s) => (*p, *s),
            _ => (0, 0),
        };
        let func = fb::AggregateFuncNode::create(
            b,
            &fb::AggregateFuncNodeArgs {
                name: Some(func_name),
                args: Some(args),
                distinct: aggr.is_distinct(),
                alias: Some(alias),
                out_decimal_precision,
                out_decimal_scale,
            },
        );
        aggr_funcs.push(func);
    }
    let aggr_funcs_vec = b.create_vector(&aggr_funcs);

    let aggr_input_schema = serialize_schema(b, &agg.input_schema());

    let input = serialize_plan_node(b, agg.input(), pm)?;

    let node = fb::GpuAggregate::create(
        b,
        &fb::GpuAggregateArgs {
            mode,
            group_exprs: Some(group_exprs_vec),
            group_names: Some(group_names_vec),
            aggr_funcs: Some(aggr_funcs_vec),
            input: Some(input),
            null_exprs: Some(null_exprs_vec),
            null_names: Some(null_names_vec),
            grouping_sets: Some(grouping_sets_vec),
            aggr_input_schema: Some(aggr_input_schema),
            // Set iff this run merges partial state across real hash partitions;
            // the executor uses it to pick the 3-col Welford stddev/var state (see
            // the flatbuffer field doc / #25). AVG is unaffected in either mode.
            mergeable_agg_state: pm == PartitionMode::RealMultiPartition,
        },
    );
    Ok((fb::PlanNodeKind::GpuAggregate, node.as_union_value()))
}

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_gpu_aggregate(
    agg: &fb::GpuAggregate,
    _node: &fb::PlanNode,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let input = deserialize_plan_node(&agg.input().ok_or("GpuAggregate missing input")?)?;

    let mode = match agg.mode() {
        fb::AggregateMode::Partial => DfAggMode::Partial,
        fb::AggregateMode::Final => DfAggMode::Final,
        fb::AggregateMode::FinalPartitioned => DfAggMode::FinalPartitioned,
        fb::AggregateMode::Single => DfAggMode::Single,
        fb::AggregateMode::SinglePartitioned => DfAggMode::SinglePartitioned,
        other => return Err(format!("unsupported AggregateMode: {:?}", other)),
    };

    // Reconstruct group-by expressions.
    let group_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = agg
        .group_exprs()
        .zip(agg.group_names())
        .map(|(exprs, names)| {
            (0..exprs.len())
                .map(|i| {
                    let expr = deserialize_expr(&exprs.get(i))?;
                    let name = names.get(i).to_string();
                    Ok((expr, name))
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?
        .unwrap_or_default();

    // ROLLUP/CUBE/GROUPING SETS: reconstruct null exprs and per-set masks.
    let null_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = match (agg.null_exprs(), agg.null_names()) {
        (Some(exprs), Some(names)) => (0..exprs.len())
            .map(|i| {
                let expr = deserialize_expr(&exprs.get(i))?;
                let name = names.get(i).to_string();
                Ok::<_, String>((expr, name))
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };
    let groups: Vec<Vec<bool>> = agg
        .grouping_sets()
        .map(|sets| {
            (0..sets.len())
                .map(|i| {
                    sets.get(i)
                        .values()
                        .map(|v| (0..v.len()).map(|j| v.get(j)).collect::<Vec<bool>>())
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // `is_single` is equivalent to "null_expr is empty" in DataFusion. Keep the
    // same convention here: anything with a non-empty null_expr came from
    // ROLLUP/CUBE/GROUPING SETS and must be reconstructed via `new`.
    let group_by = if null_exprs.is_empty() {
        PhysicalGroupBy::new_single(group_exprs)
    } else {
        PhysicalGroupBy::new(group_exprs, null_exprs, groups)
    };

    // Reconstruct aggregate function expressions. Aggregate args resolve against
    // the pre-aggregation input schema, which differs from `input.schema()` for
    // Final/FinalPartitioned stages (whose input is the Partial output and lacks
    // the original columns the args reference).
    let input_schema = deserialize_schema(
        &agg.aggr_input_schema().ok_or("GpuAggregate missing aggr_input_schema")?,
    );
    let aggr_exprs: Vec<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>> = agg
        .aggr_funcs()
        .map(|funcs| {
            (0..funcs.len())
                .map(|i| {
                    let f = funcs.get(i);
                    let name = f.name().unwrap_or("count");

                    // Reconstruct args.
                    let args: Vec<Arc<dyn PhysicalExpr>> = f
                        .args()
                        .map(|a| {
                            (0..a.len())
                                .map(|j| deserialize_expr(&a.get(j)))
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()?
                        .unwrap_or_default();

                    // Look up the aggregate UDF by name.
                    let udf = datafusion::functions_aggregate::all_default_aggregate_functions()
                        .into_iter()
                        .find(|u| u.name() == name)
                        .ok_or_else(|| format!("unknown aggregate function: {name}"))?;

                    let alias = f.alias().unwrap_or(f.name().unwrap_or("?"));
                    let mut builder = AggregateExprBuilder::new(udf, args)
                        .schema(input_schema.clone())
                        .alias(alias);

                    if f.distinct() {
                        builder = builder.distinct();
                    }

                    builder.build()
                        .map(|e| Arc::new(e))
                        .map_err(|e| format!("AggregateExprBuilder error: {e}"))
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?
        .unwrap_or_default();

    let agg_exec = AggregateExec::try_new(
        mode,
        group_by,
        aggr_exprs,
        vec![None; agg.aggr_funcs().map(|f| f.len()).unwrap_or(0)], // no per-aggregate filters
        input,
        input_schema,
    )
    .map_err(|e| format!("AggregateExec: {e}"))?;

    Ok(Arc::new(GpuAggregateExec::new(Arc::new(agg_exec))))
}


// --- Operator: partition topology + strip behavior ------------------------

impl Operator for GpuAggregateExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Map
    }
}
