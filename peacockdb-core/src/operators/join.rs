//! Join family: hash, cross, and nested-loop.

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
use datafusion::physical_plan::joins::{HashJoinExec, NestedLoopJoinExec};


gpu_exec_node!(GpuHashJoinExec);
impl GpuExtraDisplay for GpuHashJoinExec {
    fn extra_display_info(&self) -> String {
        let hj = self.inner.as_any().downcast_ref::<HashJoinExec>().unwrap();
        let on: Vec<String> = hj
            .on()
            .iter()
            .map(|(l, r)| format!("({l}, {r})"))
            .collect();
        let mut s = format!("join_type={:?}, on=[{}]", hj.join_type(), on.join(", "));
        if let Some(jf) = hj.filter() {
            s.push_str(&format!(", filter={}", jf.expression()));
        }
        if let Some(proj) = hj.projection.as_ref() {
            let cols: Vec<String> = proj.iter().map(|i| i.to_string()).collect();
            s.push_str(&format!(", projection=[{}]", cols.join(", ")));
        }
        s
    }
}

gpu_exec_node!(GpuCrossJoinExec);
impl GpuExtraDisplay for GpuCrossJoinExec {}

gpu_exec_node!(GpuNestedLoopJoinExec);
impl GpuExtraDisplay for GpuNestedLoopJoinExec {
    fn extra_display_info(&self) -> String {
        let nlj = self
            .inner
            .as_any()
            .downcast_ref::<NestedLoopJoinExec>()
            .unwrap();
        let mut s = format!("join_type={:?}", nlj.join_type());
        if let Some(jf) = nlj.filter() {
            s.push_str(&format!(", filter={}", jf.expression()));
        }
        if let Some(proj) = nlj.projection() {
            let cols: Vec<String> = proj.iter().map(|i| i.to_string()).collect();
            s.push_str(&format!(", projection=[{}]", cols.join(", ")));
        }
        s
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
use datafusion::arrow::datatypes::{Field, Schema};
use datafusion::physical_plan::PhysicalExpr;
use crate::plan_serializer::{deserialize_expr, deserialize_plan_node};
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::common::JoinSide;
use datafusion::common::JoinType as DfJoinType;
use datafusion::physical_plan::joins::CrossJoinExec;
use crate::plan_serializer::{serialize_expr, serialize_plan_node};
use crate::PartitionMode;

pub(crate) fn serialize_cudf_hash_join<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
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

    let left = serialize_plan_node(b, join.left(), pm)?;
    let right = serialize_plan_node(b, join.right(), pm)?;

    let projection = join.projection.as_ref().map(|proj| {
        let indices: Vec<u32> = proj.iter().map(|&i| i as u32).collect();
        b.create_vector(&indices)
    });

    let node = fb::CudfHashJoin::create(
        b,
        &fb::CudfHashJoinArgs {
            join_type,
            keys: Some(keys_vec),
            filter,
            filter_columns,
            left: Some(left),
            right: Some(right),
            projection,
            null_equals_null: join.null_equals_null(),
        },
    );
    Ok((fb::PlanNodeKind::CudfHashJoin, node.as_union_value()))
}
pub(crate) fn serialize_cudf_cross_join<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu = plan.as_any().downcast_ref::<GpuCrossJoinExec>().unwrap();
    let cross = gpu
        .inner()
        .as_any()
        .downcast_ref::<CrossJoinExec>()
        .ok_or("GpuCrossJoinExec inner is not CrossJoinExec")?;

    let left = serialize_plan_node(b, cross.left(), pm)?;
    let right = serialize_plan_node(b, cross.right(), pm)?;

    let node = fb::CudfCrossJoin::create(
        b,
        &fb::CudfCrossJoinArgs {
            left: Some(left),
            right: Some(right),
        },
    );
    Ok((fb::PlanNodeKind::CudfCrossJoin, node.as_union_value()))
}
pub(crate) fn serialize_cudf_nested_loop_join<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
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

    // Same convention as CudfHashJoin: serialize the predicate verbatim with its
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

    let left = serialize_plan_node(b, nlj.left(), pm)?;
    let right = serialize_plan_node(b, nlj.right(), pm)?;

    let projection = nlj.projection().map(|proj| {
        let indices: Vec<u32> = proj.iter().map(|&i| i as u32).collect();
        b.create_vector(&indices)
    });

    let node = fb::CudfNestedLoopJoin::create(
        b,
        &fb::CudfNestedLoopJoinArgs {
            join_type,
            filter,
            filter_columns,
            left: Some(left),
            right: Some(right),
            projection,
        },
    );
    Ok((fb::PlanNodeKind::CudfNestedLoopJoin, node.as_union_value()))
}

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_cudf_hash_join(
    join: &fb::CudfHashJoin,
    _node: &fb::PlanNode,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let left = deserialize_plan_node(&join.left().ok_or("CudfHashJoin missing left")?)?;
    let right = deserialize_plan_node(&join.right().ok_or("CudfHashJoin missing right")?)?;

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
        join.null_equals_null(), // mirrors DataFusion's per-join NULL key-equality
    )
    .map_err(|e| format!("HashJoinExec: {e}"))?;

    Ok(Arc::new(GpuHashJoinExec::new(Arc::new(join_exec))))
}

pub(crate) fn deserialize_cudf_cross_join(join: &fb::CudfCrossJoin) -> Result<Arc<dyn ExecutionPlan>, String> {
    let left = deserialize_plan_node(&join.left().ok_or("CudfCrossJoin missing left")?)?;
    let right = deserialize_plan_node(&join.right().ok_or("CudfCrossJoin missing right")?)?;
    let join_exec = CrossJoinExec::new(left, right);
    Ok(Arc::new(GpuCrossJoinExec::new(Arc::new(join_exec))))
}

pub(crate) fn deserialize_cudf_nested_loop_join(
    join: &fb::CudfNestedLoopJoin,
    _node: &fb::PlanNode,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let left = deserialize_plan_node(&join.left().ok_or("CudfNestedLoopJoin missing left")?)?;
    let right = deserialize_plan_node(&join.right().ok_or("CudfNestedLoopJoin missing right")?)?;

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


// --- Operator: partition topology + strip behavior ------------------------

impl Operator for GpuHashJoinExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Join
    }
}

/// NOT stripped — load-bearing: flipping it changes execution substitution and
/// the reported `NodeMemoryStats.node_name`.
impl Operator for GpuCrossJoinExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Join
    }
    fn strips_to_inner(&self) -> bool {
        false
    }
}

/// NOT stripped, same reason as GpuCrossJoinExec.
impl Operator for GpuNestedLoopJoinExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Join
    }
    fn strips_to_inner(&self) -> bool {
        false
    }
}
