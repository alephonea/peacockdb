//! Sort family: sort and the order-preserving k-way merge.

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
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;


gpu_exec_node!(GpuSortExec);
impl GpuExtraDisplay for GpuSortExec {
    fn extra_display_info(&self) -> String {
        let se = self.inner.as_any().downcast_ref::<SortExec>().unwrap();
        let mut s = format!("expr=[{}]", se.expr());
        if let Some(f) = se.fetch() {
            s.push_str(&format!(", fetch={f}"));
        }
        s
    }
}

gpu_exec_node!(GpuSortPreservingMergeExec);
impl GpuExtraDisplay for GpuSortPreservingMergeExec {
    fn extra_display_info(&self) -> String {
        let spm = self.inner.as_any().downcast_ref::<SortPreservingMergeExec>().unwrap();
        format!("[{}]", spm.expr())
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
use crate::plan_serializer::{deserialize_expr, deserialize_plan_node};
use datafusion::arrow::compute::SortOptions;
use datafusion::physical_expr::PhysicalSortExpr;
use crate::plan_serializer::{serialize_expr, serialize_plan_node};
use crate::PartitionMode;

pub(crate) fn serialize_gpu_sort<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
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

    let input = serialize_plan_node(b, sort.input(), pm)?;

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
pub(crate) fn serialize_gpu_sort_preserving_merge<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

    let gpu_spm = plan.as_any().downcast_ref::<GpuSortPreservingMergeExec>().unwrap();
    let spm = gpu_spm
        .inner()
        .as_any()
        .downcast_ref::<SortPreservingMergeExec>()
        .ok_or("GpuSortPreservingMergeExec inner is not SortPreservingMergeExec")?;

    let input = serialize_plan_node(b, spm.input(), pm)?;

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

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_gpu_sort(
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

pub(crate) fn deserialize_gpu_sort_preserving_merge(
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


// --- Operator: partition topology + strip behavior ------------------------

impl Operator for GpuSortExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Map
    }
}

impl Operator for GpuSortPreservingMergeExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::KWayMerge
    }
}
