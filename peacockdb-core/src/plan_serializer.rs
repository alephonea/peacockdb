// Serialize a DataFusion GPU physical plan tree into a FlatBuffer.
//
// Walks the `ExecutionPlan` tree produced by GpuExecutionRule, extracts the
// inner DataFusion nodes (FilterExec, ProjectionExec, etc.), and writes the
// corresponding FlatBuffer plan via the generated `peacock::plan` types.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType as ArrowDataType, Field, Schema, SchemaRef};
use datafusion::common::ScalarValue as DfScalarValue;
use datafusion::datasource::physical_plan::ParquetExec;
use datafusion::physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr,
    Literal, NegativeExpr, NotExpr,
};
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode as DfAggMode};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::common::JoinSide;
use datafusion::common::JoinType as DfJoinType;
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::physical_plan::joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::PhysicalExpr;
use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::generated::gpu_plan_generated::peacock::plan as fb;
use crate::gpu_rule::{
    GpuAggregateExec, GpuCoalesceBatchesExec, GpuCoalescePartitionsExec, GpuCrossJoinExec,
    GpuFilterExec, GpuGlobalLimitExec, GpuHashJoinExec, GpuInterleaveExec, GpuNestedLoopJoinExec,
    GpuProjectExec, GpuRepartitionExec, GpuScanExec, GpuSortExec, GpuSortPreservingMergeExec,
    GpuUnionExec, GpuWindowExec,
};

/// Serialize an entire GPU execution plan tree into a FlatBuffer byte vector.
///
/// Returns `Err` if the plan contains nodes that cannot be serialized (e.g.
/// unsupported expression types or plan nodes)
pub fn serialize_plan(plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<u8>, String> {
    let mut builder = FlatBufferBuilder::with_capacity(4096);
    let root = serialize_plan_node(&mut builder, plan)?;
    let gpu_plan = fb::GpuPlan::create(&mut builder, &fb::GpuPlanArgs { root: Some(root) });
    builder.finish(gpu_plan, None);
    Ok(builder.finished_data().to_vec())
}

// ---------------------------------------------------------------------------
// Plan nodes
// ---------------------------------------------------------------------------

fn serialize_plan_node<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<WIPOffset<fb::PlanNode<'a>>, String> {
    let output_schema = serialize_schema(b, &plan.schema());

    let (node_type, node_offset) = if let Some(scan) = plan.as_any().downcast_ref::<GpuScanExec>()
    {
        serialize_gpu_scan(b, scan)?
    } else if plan.as_any().is::<GpuFilterExec>() {
        serialize_gpu_filter(b, plan)?
    } else if plan.as_any().is::<GpuProjectExec>() {
        serialize_gpu_project(b, plan)?
    } else if plan.as_any().is::<GpuAggregateExec>() {
        serialize_gpu_aggregate(b, plan)?
    } else if plan.as_any().is::<GpuHashJoinExec>() {
        serialize_gpu_hash_join(b, plan)?
    } else if plan.as_any().is::<GpuCrossJoinExec>() {
        serialize_gpu_cross_join(b, plan)?
    } else if plan.as_any().is::<GpuNestedLoopJoinExec>() {
        serialize_gpu_nested_loop_join(b, plan)?
    } else if plan.as_any().is::<GpuSortExec>() {
        serialize_gpu_sort(b, plan)?
    } else if plan.as_any().is::<GpuCoalesceBatchesExec>() {
        serialize_gpu_coalesce_batches(b, plan)?
    } else if plan.as_any().is::<GpuCoalescePartitionsExec>() {
        serialize_gpu_coalesce_partitions(b, plan)?
    } else if plan.as_any().is::<GpuRepartitionExec>() {
        serialize_gpu_repartition(b, plan)?
    } else if plan.as_any().is::<GpuSortPreservingMergeExec>() {
        serialize_gpu_sort_preserving_merge(b, plan)?
    } else if plan.as_any().is::<GpuUnionExec>() {
        serialize_gpu_union(b, plan, false)?
    } else if plan.as_any().is::<GpuInterleaveExec>() {
        serialize_gpu_union(b, plan, true)?
    } else if plan.as_any().is::<GpuGlobalLimitExec>() {
        serialize_gpu_limit(b, plan)?
    } else if plan.as_any().is::<GpuWindowExec>() {
        serialize_gpu_window(b, plan)?
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

// --- GpuScanExec ---

fn serialize_gpu_scan<'a>(
    b: &mut FlatBufferBuilder<'a>,
    scan: &GpuScanExec,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    // The inner plan is a ParquetExec.
    let inner = scan.inner();
    let parquet = inner
        .as_any()
        .downcast_ref::<ParquetExec>()
        .ok_or_else(|| "GpuScanExec inner is not ParquetExec".to_string())?;

    let config = parquet.base_config();

    // The wire format requires absolute filesystem paths. `object_meta.location`
    // is an `object_store::path::Path`, which strips the leading `/` during
    // normalization, so we re-add it here. ListingTableUrl canonicalizes to
    // absolute at registration time, so the original input was always absolute
    // by the time we get here.
    let path_strings: Vec<String> = config
        .file_groups
        .iter()
        .flat_map(|group| {
            group
                .iter()
                .map(|pf| format!("/{}", pf.object_meta.location))
        })
        .collect();
    let paths: Vec<_> = path_strings.iter().map(|s| b.create_string(s)).collect();
    let file_paths = b.create_vector(&paths);

    let file_schema = serialize_schema(b, &config.file_schema);

    let projection = config.projection.as_ref().map(|proj| {
        let indices: Vec<u32> = proj.iter().map(|&i| i as u32).collect();
        b.create_vector(&indices)
    });

    let limit = config.limit.unwrap_or(0) as u64;

    // Surviving row groups under the static pushdown predicate (same PruningPredicate
    // DataFusion uses for the CPU path -> identical set -> parity with the .cpu.txt
    // oracle). A deserialized scan carries the override verbatim (its reconstructed
    // ParquetExec has no predicate to recompute from) so the roundtrip stays
    // byte-stable; a fresh plan computes from the predicate. None/empty for the
    // no-predicate / multi-file / #16-dynamic cases -> read all.
    let survivors = scan
        .row_groups_override()
        .cloned()
        .or_else(|| crate::gpu_rowgroup_prune::surviving_row_groups(parquet));
    let row_groups = survivors
        .filter(|v| !v.is_empty())
        .map(|v| b.create_vector(&v));

    let gpu_scan = fb::GpuScan::create(
        b,
        &fb::GpuScanArgs {
            file_paths: Some(file_paths),
            file_schema: Some(file_schema),
            projection,
            batch_size: scan.gpu_batch_size as u32,
            limit,
            row_groups,
        },
    );

    Ok((fb::PlanNodeKind::GpuScan, gpu_scan.as_union_value()))
}

// --- GpuFilterExec ---

fn serialize_gpu_filter<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu_filter = plan
        .as_any()
        .downcast_ref::<GpuFilterExec>()
        .unwrap();
    let filter = gpu_filter
        .inner()
        .as_any()
        .downcast_ref::<FilterExec>()
        .ok_or("GpuFilterExec inner is not FilterExec")?;

    let predicate = serialize_expr(b, filter.predicate(), &filter.input().schema())?;
    let input_plan = filter.input();
    let input = serialize_plan_node(b, input_plan)?;

    let projection = filter.projection().map(|p| {
        let indices: Vec<u32> = p.iter().map(|&i| i as u32).collect();
        b.create_vector(&indices)
    });

    let node = fb::GpuFilter::create(
        b,
        &fb::GpuFilterArgs {
            predicate: Some(predicate),
            input: Some(input),
            projection,
        },
    );
    Ok((fb::PlanNodeKind::GpuFilter, node.as_union_value()))
}

// --- GpuProjectExec ---

fn serialize_gpu_project<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu_proj = plan
        .as_any()
        .downcast_ref::<GpuProjectExec>()
        .unwrap();
    let proj = gpu_proj
        .inner()
        .as_any()
        .downcast_ref::<ProjectionExec>()
        .ok_or("GpuProjectExec inner is not ProjectionExec")?;

    let mut exprs = Vec::new();
    let mut alias_offsets = Vec::new();
    for (expr, alias) in proj.expr() {
        exprs.push(serialize_expr(b, expr, &proj.input().schema())?);
        alias_offsets.push(b.create_string(alias));
    }
    let exprs_vec = b.create_vector(&exprs);
    let aliases_vec = b.create_vector(&alias_offsets);

    let input = serialize_plan_node(b, proj.input())?;

    let node = fb::GpuProject::create(
        b,
        &fb::GpuProjectArgs {
            exprs: Some(exprs_vec),
            aliases: Some(aliases_vec),
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::GpuProject, node.as_union_value()))
}

// --- GpuAggregateExec ---

fn serialize_gpu_aggregate<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
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

    let input = serialize_plan_node(b, agg.input())?;

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
        },
    );
    Ok((fb::PlanNodeKind::GpuAggregate, node.as_union_value()))
}

// --- GpuHashJoinExec ---

fn serialize_gpu_hash_join<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu_join = plan
        .as_any()
        .downcast_ref::<GpuHashJoinExec>()
        .unwrap();
    let join = gpu_join
        .inner()
        .as_any()
        .downcast_ref::<HashJoinExec>()
        .ok_or("GpuHashJoinExec inner is not HashJoinExec")?;

    let join_type = match join.join_type() {
        DfJoinType::Inner => fb::JoinType::Inner,
        DfJoinType::Left => fb::JoinType::Left,
        DfJoinType::Right => fb::JoinType::Right,
        DfJoinType::Full => fb::JoinType::Full,
        DfJoinType::LeftSemi => fb::JoinType::LeftSemi,
        DfJoinType::RightSemi => fb::JoinType::RightSemi,
        DfJoinType::LeftAnti => fb::JoinType::LeftAnti,
        DfJoinType::RightAnti => fb::JoinType::RightAnti,
        DfJoinType::LeftMark => fb::JoinType::LeftMark,
    };

    let mut keys = Vec::new();
    for (left_key, right_key) in join.on() {
        let left = serialize_expr(b, left_key, &join.left().schema())?;
        let right = serialize_expr(b, right_key, &join.right().schema())?;
        keys.push(fb::JoinKey::create(
            b,
            &fb::JoinKeyArgs {
                left: Some(left),
                right: Some(right),
            },
        ));
    }
    let keys_vec = b.create_vector(&keys);

    // Serialize the residual filter verbatim, along with its column-origin map.
    // The expression's ColumnRefs index the filter's intermediate schema; the
    // C++ executor remaps them to its post-join table via `filter_columns`.
    let (filter, filter_columns) = if let Some(jf) = join.filter() {
        let expr = serialize_expr(b, jf.expression(), jf.schema())?;
        let cols: Vec<fb::JoinFilterColumn> = jf
            .column_indices()
            .iter()
            .map(|ci| {
                let side = match ci.side {
                    JoinSide::Left => fb::JoinSide::Left,
                    JoinSide::Right => fb::JoinSide::Right,
                    JoinSide::None => {
                        return Err("join filter references a mark-join column".to_string())
                    }
                };
                Ok(fb::JoinFilterColumn::new(ci.index as u32, side))
            })
            .collect::<Result<_, String>>()?;
        (Some(expr), Some(b.create_vector(&cols)))
    } else {
        (None, None)
    };

    let left = serialize_plan_node(b, join.left())?;
    let right = serialize_plan_node(b, join.right())?;

    let projection = join.projection.as_ref().map(|proj| {
        let indices: Vec<u32> = proj.iter().map(|&i| i as u32).collect();
        b.create_vector(&indices)
    });

    let node = fb::GpuHashJoin::create(
        b,
        &fb::GpuHashJoinArgs {
            join_type,
            keys: Some(keys_vec),
            filter,
            filter_columns,
            left: Some(left),
            right: Some(right),
            projection,
        },
    );
    Ok((fb::PlanNodeKind::GpuHashJoin, node.as_union_value()))
}

// --- GpuCrossJoinExec ---

fn serialize_gpu_cross_join<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu = plan.as_any().downcast_ref::<GpuCrossJoinExec>().unwrap();
    let cross = gpu
        .inner()
        .as_any()
        .downcast_ref::<CrossJoinExec>()
        .ok_or("GpuCrossJoinExec inner is not CrossJoinExec")?;

    let left = serialize_plan_node(b, cross.left())?;
    let right = serialize_plan_node(b, cross.right())?;

    let node = fb::GpuCrossJoin::create(
        b,
        &fb::GpuCrossJoinArgs {
            left: Some(left),
            right: Some(right),
        },
    );
    Ok((fb::PlanNodeKind::GpuCrossJoin, node.as_union_value()))
}

// --- GpuNestedLoopJoinExec ---

fn serialize_gpu_nested_loop_join<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu = plan.as_any().downcast_ref::<GpuNestedLoopJoinExec>().unwrap();
    let nlj = gpu
        .inner()
        .as_any()
        .downcast_ref::<NestedLoopJoinExec>()
        .ok_or("GpuNestedLoopJoinExec inner is not NestedLoopJoinExec")?;

    // The C++ executor only implements Inner and Left nested-loop joins, so keep
    // the serializable surface equal to the executable one: reject the rest here
    // rather than failing at GPU runtime.
    let join_type = match nlj.join_type() {
        DfJoinType::Inner => fb::JoinType::Inner,
        DfJoinType::Left => fb::JoinType::Left,
        other => {
            return Err(format!(
                "NestedLoopJoin join type {other:?} is not supported on GPU (only Inner/Left)"
            ))
        }
    };

    // Same convention as GpuHashJoin: serialize the predicate verbatim with its
    // column-origin map; the C++ executor remaps the ColumnRefs.
    let (filter, filter_columns) = if let Some(jf) = nlj.filter() {
        let expr = serialize_expr(b, jf.expression(), jf.schema())?;
        let cols: Vec<fb::JoinFilterColumn> = jf
            .column_indices()
            .iter()
            .map(|ci| {
                let side = match ci.side {
                    JoinSide::Left => fb::JoinSide::Left,
                    JoinSide::Right => fb::JoinSide::Right,
                    JoinSide::None => {
                        return Err(
                            "nested-loop join filter references a mark-join column".to_string()
                        )
                    }
                };
                Ok(fb::JoinFilterColumn::new(ci.index as u32, side))
            })
            .collect::<Result<_, String>>()?;
        (Some(expr), Some(b.create_vector(&cols)))
    } else {
        (None, None)
    };

    let left = serialize_plan_node(b, nlj.left())?;
    let right = serialize_plan_node(b, nlj.right())?;

    let projection = nlj.projection().map(|proj| {
        let indices: Vec<u32> = proj.iter().map(|&i| i as u32).collect();
        b.create_vector(&indices)
    });

    let node = fb::GpuNestedLoopJoin::create(
        b,
        &fb::GpuNestedLoopJoinArgs {
            join_type,
            filter,
            filter_columns,
            left: Some(left),
            right: Some(right),
            projection,
        },
    );
    Ok((fb::PlanNodeKind::GpuNestedLoopJoin, node.as_union_value()))
}

// --- GpuSortExec ---

fn serialize_gpu_sort<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu_sort = plan
        .as_any()
        .downcast_ref::<GpuSortExec>()
        .unwrap();
    let sort = gpu_sort
        .inner()
        .as_any()
        .downcast_ref::<SortExec>()
        .ok_or("GpuSortExec inner is not SortExec")?;

    let mut sort_exprs = Vec::new();
    for se in sort.expr().iter() {
        let expr = serialize_expr(b, &se.expr, &sort.input().schema())?;
        sort_exprs.push(fb::SortExprNode::create(
            b,
            &fb::SortExprNodeArgs {
                expr: Some(expr),
                asc: !se.options.descending,
                nulls_first: se.options.nulls_first,
            },
        ));
    }
    let exprs_vec = b.create_vector(&sort_exprs);

    let fetch = sort.fetch().map(|f| f as i64).unwrap_or(-1);

    let input = serialize_plan_node(b, sort.input())?;

    let node = fb::GpuSort::create(
        b,
        &fb::GpuSortArgs {
            exprs: Some(exprs_vec),
            fetch,
            preserve_partitioning: sort.preserve_partitioning(),
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::GpuSort, node.as_union_value()))
}

// --- GpuCoalesceBatchesExec ---

fn serialize_gpu_coalesce_batches<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu_cb = plan.as_any().downcast_ref::<GpuCoalesceBatchesExec>().unwrap();
    let cb = gpu_cb
        .inner()
        .as_any()
        .downcast_ref::<datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec>()
        .ok_or("GpuCoalesceBatchesExec inner is not CoalesceBatchesExec")?;

    let input = serialize_plan_node(b, cb.input())?;
    let node = fb::GpuCoalesceBatches::create(
        b,
        &fb::GpuCoalesceBatchesArgs {
            target_batch_size: cb.target_batch_size() as u32,
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::GpuCoalesceBatches, node.as_union_value()))
}

// --- GpuCoalescePartitionsExec ---

fn serialize_gpu_coalesce_partitions<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu_cp = plan.as_any().downcast_ref::<GpuCoalescePartitionsExec>().unwrap();
    let cp = gpu_cp
        .inner()
        .as_any()
        .downcast_ref::<datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec>()
        .ok_or("GpuCoalescePartitionsExec inner is not CoalescePartitionsExec")?;

    let input = serialize_plan_node(b, cp.input())?;
    let node = fb::GpuCoalescePartitions::create(
        b,
        &fb::GpuCoalescePartitionsArgs {
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::GpuCoalescePartitions, node.as_union_value()))
}

// --- GpuRepartitionExec ---

fn serialize_gpu_repartition<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    use datafusion::physical_plan::repartition::RepartitionExec;
    use datafusion::physical_plan::Partitioning;

    let gpu_rp = plan.as_any().downcast_ref::<GpuRepartitionExec>().unwrap();
    let rp = gpu_rp
        .inner()
        .as_any()
        .downcast_ref::<RepartitionExec>()
        .ok_or("GpuRepartitionExec inner is not RepartitionExec")?;

    let input = serialize_plan_node(b, rp.input())?;

    let (kind, num_partitions, hash_exprs) = match rp.partitioning() {
        Partitioning::RoundRobinBatch(n) => (fb::PartitioningKind::RoundRobinBatch, *n, None),
        Partitioning::Hash(exprs, n) => {
            let mut expr_offsets = Vec::new();
            for expr in exprs {
                expr_offsets.push(serialize_expr(b, expr, &rp.input().schema())?);
            }
            let exprs_vec = b.create_vector(&expr_offsets);
            (fb::PartitioningKind::Hash, *n, Some(exprs_vec))
        }
        Partitioning::UnknownPartitioning(n) => (fb::PartitioningKind::Unknown, *n, None),
    };

    let node = fb::GpuRepartition::create(
        b,
        &fb::GpuRepartitionArgs {
            kind,
            num_partitions: num_partitions as u32,
            hash_exprs,
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::GpuRepartition, node.as_union_value()))
}

// --- GpuSortPreservingMergeExec ---

fn serialize_gpu_sort_preserving_merge<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

    let gpu_spm = plan.as_any().downcast_ref::<GpuSortPreservingMergeExec>().unwrap();
    let spm = gpu_spm
        .inner()
        .as_any()
        .downcast_ref::<SortPreservingMergeExec>()
        .ok_or("GpuSortPreservingMergeExec inner is not SortPreservingMergeExec")?;

    let input = serialize_plan_node(b, spm.input())?;

    let mut sort_exprs = Vec::new();
    for se in spm.expr().iter() {
        let expr = serialize_expr(b, &se.expr, &spm.input().schema())?;
        sort_exprs.push(fb::SortExprNode::create(
            b,
            &fb::SortExprNodeArgs {
                expr: Some(expr),
                asc: !se.options.descending,
                nulls_first: se.options.nulls_first,
            },
        ));
    }
    let exprs_vec = b.create_vector(&sort_exprs);

    let fetch = spm.fetch().map(|f| f as i64).unwrap_or(-1);

    let node = fb::GpuSortPreservingMerge::create(
        b,
        &fb::GpuSortPreservingMergeArgs {
            exprs: Some(exprs_vec),
            fetch,
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::GpuSortPreservingMerge, node.as_union_value()))
}

// --- GpuUnionExec / GpuInterleaveExec ---

fn serialize_gpu_union<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    interleave: bool,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    // UnionExec and InterleaveExec carry no extra state beyond their children,
    // so serialize the inputs directly off the wrapper (no inner downcast).
    let mut inputs = Vec::with_capacity(plan.children().len());
    for child in plan.children() {
        inputs.push(serialize_plan_node(b, child)?);
    }
    let inputs_vec = b.create_vector(&inputs);

    // Carry the declared output schema so the executor can normalize each
    // branch's decimal scale before concatenate (see GpuUnion.output_schema).
    let output_schema = serialize_schema(b, &plan.schema());

    let node = fb::GpuUnion::create(
        b,
        &fb::GpuUnionArgs {
            inputs: Some(inputs_vec),
            interleave,
            output_schema: Some(output_schema),
        },
    );
    Ok((fb::PlanNodeKind::GpuUnion, node.as_union_value()))
}

// --- GpuGlobalLimitExec ---

fn serialize_gpu_limit<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    use datafusion::physical_plan::limit::GlobalLimitExec;

    let gpu_limit = plan.as_any().downcast_ref::<GpuGlobalLimitExec>().unwrap();
    let limit = gpu_limit
        .inner()
        .as_any()
        .downcast_ref::<GlobalLimitExec>()
        .ok_or("GpuGlobalLimitExec inner is not GlobalLimitExec")?;

    let input = serialize_plan_node(b, limit.input())?;
    let fetch = limit.fetch().map(|f| f as i64).unwrap_or(-1);

    let node = fb::GpuLimit::create(
        b,
        &fb::GpuLimitArgs {
            skip: limit.skip() as u64,
            fetch,
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::GpuLimit, node.as_union_value()))
}

// --- GpuWindowExec ---

fn serialize_gpu_window<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
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
    let input = serialize_plan_node(b, input_plan)?;

    let node = fb::GpuWindow::create(
        b,
        &fb::GpuWindowArgs {
            window_exprs: Some(exprs_vec),
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::GpuWindow, node.as_union_value()))
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

fn serialize_expr<'a>(
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

fn serialize_scalar_value<'a>(
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

fn convert_data_type(dt: &ArrowDataType) -> Result<fb::DataType, String> {
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

fn serialize_schema<'a>(
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

fn deserialize_plan_node(node: &fb::PlanNode) -> Result<Arc<dyn ExecutionPlan>, String> {
    match node.node_type() {
        fb::PlanNodeKind::GpuScan => {
            let scan = node.node_as_gpu_scan().ok_or("expected GpuScan")?;
            deserialize_gpu_scan(&scan, node)
        }
        fb::PlanNodeKind::GpuFilter => {
            let filter = node.node_as_gpu_filter().ok_or("expected GpuFilter")?;
            deserialize_gpu_filter(&filter, node)
        }
        fb::PlanNodeKind::GpuProject => {
            let proj = node.node_as_gpu_project().ok_or("expected GpuProject")?;
            deserialize_gpu_project(&proj, node)
        }
        fb::PlanNodeKind::GpuAggregate => {
            let agg = node.node_as_gpu_aggregate().ok_or("expected GpuAggregate")?;
            deserialize_gpu_aggregate(&agg, node)
        }
        fb::PlanNodeKind::GpuHashJoin => {
            let join = node.node_as_gpu_hash_join().ok_or("expected GpuHashJoin")?;
            deserialize_gpu_hash_join(&join, node)
        }
        fb::PlanNodeKind::GpuCrossJoin => {
            let join = node.node_as_gpu_cross_join().ok_or("expected GpuCrossJoin")?;
            deserialize_gpu_cross_join(&join)
        }
        fb::PlanNodeKind::GpuNestedLoopJoin => {
            let join = node
                .node_as_gpu_nested_loop_join()
                .ok_or("expected GpuNestedLoopJoin")?;
            deserialize_gpu_nested_loop_join(&join, node)
        }
        fb::PlanNodeKind::GpuSort => {
            let sort = node.node_as_gpu_sort().ok_or("expected GpuSort")?;
            deserialize_gpu_sort(&sort, node)
        }
        fb::PlanNodeKind::GpuCoalesceBatches => {
            let cb = node.node_as_gpu_coalesce_batches().ok_or("expected GpuCoalesceBatches")?;
            deserialize_gpu_coalesce_batches(&cb)
        }
        fb::PlanNodeKind::GpuCoalescePartitions => {
            let cp = node.node_as_gpu_coalesce_partitions().ok_or("expected GpuCoalescePartitions")?;
            deserialize_gpu_coalesce_partitions(&cp)
        }
        fb::PlanNodeKind::GpuRepartition => {
            let rp = node.node_as_gpu_repartition().ok_or("expected GpuRepartition")?;
            deserialize_gpu_repartition(&rp)
        }
        fb::PlanNodeKind::GpuSortPreservingMerge => {
            let spm = node.node_as_gpu_sort_preserving_merge().ok_or("expected GpuSortPreservingMerge")?;
            deserialize_gpu_sort_preserving_merge(&spm)
        }
        fb::PlanNodeKind::GpuUnion => {
            let u = node.node_as_gpu_union().ok_or("expected GpuUnion")?;
            deserialize_gpu_union(&u)
        }
        fb::PlanNodeKind::GpuLimit => {
            let l = node.node_as_gpu_limit().ok_or("expected GpuLimit")?;
            deserialize_gpu_limit(&l)
        }
        fb::PlanNodeKind::GpuWindow => {
            let w = node.node_as_gpu_window().ok_or("expected GpuWindow")?;
            deserialize_gpu_window(&w)
        }
        other => Err(format!("unknown PlanNodeKind: {:?}", other)),
    }
}

fn deserialize_schema(schema: &fb::Schema) -> SchemaRef {
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

fn fb_to_arrow_type(dt: fb::DataType) -> ArrowDataType {
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

fn deserialize_expr(expr: &fb::Expr) -> Result<Arc<dyn PhysicalExpr>, String> {
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

fn deserialize_scalar(sv: &fb::ScalarValue) -> Result<DfScalarValue, String> {
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

fn fb_to_operator(op: fb::BinaryOp) -> Result<datafusion::logical_expr::Operator, String> {
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

// --- Plan node deserialization ---

use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::physical_expr::aggregate::AggregateExprBuilder;
use datafusion::physical_plan::aggregates::PhysicalGroupBy;
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::arrow::compute::SortOptions;

fn deserialize_gpu_scan(
    scan: &fb::GpuScan,
    node: &fb::PlanNode,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let file_schema = node
        .output_schema()
        .map(|s| deserialize_schema(&s))
        .unwrap_or_else(|| {
            scan.file_schema()
                .map(|s| deserialize_schema(&s))
                .unwrap_or_else(|| Arc::new(datafusion::arrow::datatypes::Schema::empty()))
        });

    let full_schema = scan
        .file_schema()
        .map(|s| deserialize_schema(&s))
        .unwrap_or_else(|| file_schema.clone());

    let file_groups: Vec<Vec<PartitionedFile>> = scan
        .file_paths()
        .map(|v| {
            (0..v.len())
                .map(|i| {
                    let path = v.get(i);
                    vec![PartitionedFile::new(path.to_string(), 0)]
                })
                .collect()
        })
        .unwrap_or_default();

    let projection = scan.projection().map(|v| {
        (0..v.len()).map(|i| v.get(i) as usize).collect::<Vec<_>>()
    });

    let limit = if scan.limit() > 0 {
        Some(scan.limit() as usize)
    } else {
        None
    };

    let config = FileScanConfig::new(
        datafusion::execution::object_store::ObjectStoreUrl::local_filesystem(),
        full_schema,
    )
    .with_file_groups(file_groups)
    .with_projection(projection)
    .with_limit(limit);

    let parquet = ParquetExec::builder(config).build_arc();

    // Carry the surviving row groups verbatim so re-serialization reproduces them
    // (the reconstructed ParquetExec has no predicate to recompute from).
    let row_groups: Option<Vec<u32>> = scan
        .row_groups()
        .map(|v| (0..v.len()).map(|i| v.get(i)).collect::<Vec<u32>>())
        .filter(|v| !v.is_empty());

    Ok(Arc::new(GpuScanExec::with_row_groups(
        parquet,
        scan.batch_size() as usize,
        row_groups,
    )))
}

fn deserialize_gpu_filter(
    filter: &fb::GpuFilter,
    _node: &fb::PlanNode,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let input = deserialize_plan_node(&filter.input().ok_or("GpuFilter missing input")?)?;
    let predicate = deserialize_expr(&filter.predicate().ok_or("GpuFilter missing predicate")?)?;
    let mut filter_exec =
        FilterExec::try_new(predicate, input).map_err(|e| format!("FilterExec: {e}"))?;
    if let Some(proj) = filter.projection() {
        let indices: Vec<usize> = (0..proj.len()).map(|i| proj.get(i) as usize).collect();
        filter_exec = filter_exec
            .with_projection(Some(indices))
            .map_err(|e| format!("FilterExec::with_projection: {e}"))?;
    }
    Ok(Arc::new(GpuFilterExec::new(Arc::new(filter_exec))))
}

fn deserialize_gpu_project(
    proj: &fb::GpuProject,
    _node: &fb::PlanNode,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let input = deserialize_plan_node(&proj.input().ok_or("GpuProject missing input")?)?;
    let exprs_fb = proj.exprs().ok_or("GpuProject missing exprs")?;
    let aliases_fb = proj.aliases().ok_or("GpuProject missing aliases")?;

    let expr_pairs: Vec<(Arc<dyn PhysicalExpr>, String)> = (0..exprs_fb.len())
        .map(|i| {
            let expr = deserialize_expr(&exprs_fb.get(i))?;
            let alias = aliases_fb.get(i).to_string();
            Ok((expr, alias))
        })
        .collect::<Result<_, String>>()?;

    let proj_exec =
        ProjectionExec::try_new(expr_pairs, input).map_err(|e| format!("ProjectionExec: {e}"))?;
    Ok(Arc::new(GpuProjectExec::new(Arc::new(proj_exec))))
}

fn deserialize_gpu_aggregate(
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

fn deserialize_gpu_hash_join(
    join: &fb::GpuHashJoin,
    _node: &fb::PlanNode,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let left = deserialize_plan_node(&join.left().ok_or("GpuHashJoin missing left")?)?;
    let right = deserialize_plan_node(&join.right().ok_or("GpuHashJoin missing right")?)?;

    let join_type = match join.join_type() {
        fb::JoinType::Inner => DfJoinType::Inner,
        fb::JoinType::Left => DfJoinType::Left,
        fb::JoinType::Right => DfJoinType::Right,
        fb::JoinType::Full => DfJoinType::Full,
        fb::JoinType::LeftSemi => DfJoinType::LeftSemi,
        fb::JoinType::RightSemi => DfJoinType::RightSemi,
        fb::JoinType::LeftAnti => DfJoinType::LeftAnti,
        fb::JoinType::RightAnti => DfJoinType::RightAnti,
        fb::JoinType::LeftMark => DfJoinType::LeftMark,
        other => return Err(format!("unsupported JoinType: {:?}", other)),
    };

    let on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> = join
        .keys()
        .map(|keys| {
            (0..keys.len())
                .map(|i| {
                    let k = keys.get(i);
                    let l = deserialize_expr(&k.left().ok_or("JoinKey missing left")?)?;
                    let r = deserialize_expr(&k.right().ok_or("JoinKey missing right")?)?;
                    Ok((l, r))
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?
        .unwrap_or_default();

    let projection: Option<Vec<usize>> = join.projection().map(|v| {
        (0..v.len()).map(|i| v.get(i) as usize).collect()
    });

    // Rebuild the residual JoinFilter from the verbatim expression + its
    // column-origin map. The intermediate schema is reconstructed by pulling
    // each referenced field from the left/right input schemas.
    let filter = match (join.filter(), join.filter_columns()) {
        (Some(expr), Some(cols)) => {
            let expression = deserialize_expr(&expr)?;
            let left_schema = left.schema();
            let right_schema = right.schema();
            let mut column_indices = Vec::with_capacity(cols.len());
            let mut fields: Vec<Field> = Vec::with_capacity(cols.len());
            for i in 0..cols.len() {
                let c = cols.get(i);
                let idx = c.index() as usize;
                let (side, schema) = match c.side() {
                    fb::JoinSide::Left => (JoinSide::Left, &left_schema),
                    fb::JoinSide::Right => (JoinSide::Right, &right_schema),
                    other => return Err(format!("invalid JoinSide: {other:?}")),
                };
                fields.push(schema.field(idx).clone());
                column_indices.push(ColumnIndex { index: idx, side });
            }
            Some(JoinFilter::new(expression, column_indices, Schema::new(fields).into()))
        }
        _ => None,
    };

    let join_exec = HashJoinExec::try_new(
        left,
        right,
        on,
        filter,
        &join_type,
        projection,
        datafusion::physical_plan::joins::PartitionMode::CollectLeft,
        false, // null_equals_null
    )
    .map_err(|e| format!("HashJoinExec: {e}"))?;

    Ok(Arc::new(GpuHashJoinExec::new(Arc::new(join_exec))))
}

fn deserialize_gpu_cross_join(join: &fb::GpuCrossJoin) -> Result<Arc<dyn ExecutionPlan>, String> {
    let left = deserialize_plan_node(&join.left().ok_or("GpuCrossJoin missing left")?)?;
    let right = deserialize_plan_node(&join.right().ok_or("GpuCrossJoin missing right")?)?;
    let join_exec = CrossJoinExec::new(left, right);
    Ok(Arc::new(GpuCrossJoinExec::new(Arc::new(join_exec))))
}

fn deserialize_gpu_nested_loop_join(
    join: &fb::GpuNestedLoopJoin,
    _node: &fb::PlanNode,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let left = deserialize_plan_node(&join.left().ok_or("GpuNestedLoopJoin missing left")?)?;
    let right = deserialize_plan_node(&join.right().ok_or("GpuNestedLoopJoin missing right")?)?;

    let join_type = match join.join_type() {
        fb::JoinType::Inner => DfJoinType::Inner,
        fb::JoinType::Left => DfJoinType::Left,
        fb::JoinType::Right => DfJoinType::Right,
        fb::JoinType::Full => DfJoinType::Full,
        fb::JoinType::LeftSemi => DfJoinType::LeftSemi,
        fb::JoinType::RightSemi => DfJoinType::RightSemi,
        fb::JoinType::LeftAnti => DfJoinType::LeftAnti,
        fb::JoinType::RightAnti => DfJoinType::RightAnti,
        fb::JoinType::LeftMark => DfJoinType::LeftMark,
        other => return Err(format!("unsupported JoinType: {:?}", other)),
    };

    let projection: Option<Vec<usize>> = join
        .projection()
        .map(|v| (0..v.len()).map(|i| v.get(i) as usize).collect());

    // Rebuild the join predicate from the verbatim expression + column-origin
    // map (same convention as the hash-join residual filter).
    let filter = match (join.filter(), join.filter_columns()) {
        (Some(expr), Some(cols)) => {
            let expression = deserialize_expr(&expr)?;
            let left_schema = left.schema();
            let right_schema = right.schema();
            let mut column_indices = Vec::with_capacity(cols.len());
            let mut fields: Vec<Field> = Vec::with_capacity(cols.len());
            for i in 0..cols.len() {
                let c = cols.get(i);
                let idx = c.index() as usize;
                let (side, schema) = match c.side() {
                    fb::JoinSide::Left => (JoinSide::Left, &left_schema),
                    fb::JoinSide::Right => (JoinSide::Right, &right_schema),
                    other => return Err(format!("invalid JoinSide: {other:?}")),
                };
                fields.push(schema.field(idx).clone());
                column_indices.push(ColumnIndex { index: idx, side });
            }
            Some(JoinFilter::new(expression, column_indices, Schema::new(fields).into()))
        }
        _ => None,
    };

    let join_exec = NestedLoopJoinExec::try_new(left, right, filter, &join_type, projection)
        .map_err(|e| format!("NestedLoopJoinExec: {e}"))?;

    Ok(Arc::new(GpuNestedLoopJoinExec::new(Arc::new(join_exec))))
}

fn deserialize_gpu_sort(
    sort: &fb::GpuSort,
    _node: &fb::PlanNode,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let input = deserialize_plan_node(&sort.input().ok_or("GpuSort missing input")?)?;

    let sort_exprs: Vec<PhysicalSortExpr> = sort
        .exprs()
        .map(|exprs| {
            (0..exprs.len())
                .map(|i| {
                    let se = exprs.get(i);
                    let expr = deserialize_expr(&se.expr().ok_or("SortExpr missing expr")?)?;
                    Ok(PhysicalSortExpr::new(
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

    let mut sort_exec = SortExec::new(sort_exprs.into(), input)
        .with_preserve_partitioning(sort.preserve_partitioning());
    if sort.fetch() >= 0 {
        sort_exec = sort_exec.with_fetch(Some(sort.fetch() as usize));
    }

    Ok(Arc::new(GpuSortExec::new(Arc::new(sort_exec))))
}

fn deserialize_gpu_coalesce_batches(
    cb: &fb::GpuCoalesceBatches,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;

    let input = deserialize_plan_node(&cb.input().ok_or("GpuCoalesceBatches missing input")?)?;
    let inner: Arc<dyn ExecutionPlan> =
        Arc::new(CoalesceBatchesExec::new(input, cb.target_batch_size() as usize));
    Ok(Arc::new(GpuCoalesceBatchesExec::new(inner)))
}

fn deserialize_gpu_coalesce_partitions(
    cp: &fb::GpuCoalescePartitions,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;

    let input = deserialize_plan_node(&cp.input().ok_or("GpuCoalescePartitions missing input")?)?;
    let inner: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(input));
    Ok(Arc::new(GpuCoalescePartitionsExec::new(inner)))
}

fn deserialize_gpu_repartition(
    rp: &fb::GpuRepartition,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::physical_plan::repartition::RepartitionExec;
    use datafusion::physical_plan::Partitioning;

    let input = deserialize_plan_node(&rp.input().ok_or("GpuRepartition missing input")?)?;

    let partitioning = match rp.kind() {
        fb::PartitioningKind::RoundRobinBatch => {
            Partitioning::RoundRobinBatch(rp.num_partitions() as usize)
        }
        fb::PartitioningKind::Hash => {
            let exprs: Vec<Arc<dyn PhysicalExpr>> = rp
                .hash_exprs()
                .map(|v| {
                    (0..v.len())
                        .map(|i| deserialize_expr(&v.get(i)))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();
            Partitioning::Hash(exprs, rp.num_partitions() as usize)
        }
        fb::PartitioningKind::Unknown => {
            Partitioning::UnknownPartitioning(rp.num_partitions() as usize)
        }
        other => return Err(format!("unsupported PartitioningKind: {:?}", other)),
    };

    let inner: Arc<dyn ExecutionPlan> = Arc::new(
        RepartitionExec::try_new(input, partitioning)
            .map_err(|e| format!("RepartitionExec: {e}"))?,
    );
    Ok(Arc::new(GpuRepartitionExec::new(inner)))
}

fn deserialize_gpu_sort_preserving_merge(
    spm: &fb::GpuSortPreservingMerge,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

    let input = deserialize_plan_node(
        &spm.input().ok_or("GpuSortPreservingMerge missing input")?,
    )?;

    let sort_exprs: Vec<PhysicalSortExpr> = spm
        .exprs()
        .map(|exprs| {
            (0..exprs.len())
                .map(|i| {
                    let se = exprs.get(i);
                    let expr = deserialize_expr(&se.expr().ok_or("SortExpr missing expr")?)?;
                    Ok(PhysicalSortExpr::new(
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

    let mut merge_exec = SortPreservingMergeExec::new(sort_exprs.into(), input);
    if spm.fetch() >= 0 {
        merge_exec = merge_exec.with_fetch(Some(spm.fetch() as usize));
    }

    Ok(Arc::new(GpuSortPreservingMergeExec::new(Arc::new(merge_exec))))
}

fn deserialize_gpu_union(u: &fb::GpuUnion) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::physical_plan::union::{InterleaveExec, UnionExec};

    let inputs: Vec<Arc<dyn ExecutionPlan>> = u
        .inputs()
        .map(|v| {
            (0..v.len())
                .map(|i| deserialize_plan_node(&v.get(i)))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    if inputs.is_empty() {
        return Err("GpuUnion has no inputs".into());
    }

    if u.interleave() {
        let inner = InterleaveExec::try_new(inputs).map_err(|e| format!("InterleaveExec: {e}"))?;
        Ok(Arc::new(GpuInterleaveExec::new(Arc::new(inner))))
    } else {
        let inner = UnionExec::new(inputs);
        Ok(Arc::new(GpuUnionExec::new(Arc::new(inner))))
    }
}

fn deserialize_gpu_limit(l: &fb::GpuLimit) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::physical_plan::limit::GlobalLimitExec;

    let input = deserialize_plan_node(&l.input().ok_or("GpuLimit missing input")?)?;
    let fetch = if l.fetch() >= 0 {
        Some(l.fetch() as usize)
    } else {
        None
    };
    let inner = GlobalLimitExec::new(input, l.skip() as usize, fetch);
    Ok(Arc::new(GpuGlobalLimitExec::new(Arc::new(inner))))
}

fn deserialize_gpu_window(win: &fb::GpuWindow) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::logical_expr::{
        WindowFrame, WindowFrameBound as DfBound, WindowFrameUnits, WindowFunctionDefinition,
    };
    use datafusion::physical_expr::LexOrdering;
    use datafusion::physical_plan::windows::{
        create_window_expr, BoundedWindowAggExec, WindowAggExec,
    };
    use datafusion::physical_plan::InputOrderMode;

    let input = deserialize_plan_node(&win.input().ok_or("GpuWindow missing input")?)?;
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

            // Only aggregate windows are serialized (serialize_gpu_window rejects
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