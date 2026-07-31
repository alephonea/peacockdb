//! Projection family.

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
use datafusion::physical_plan::projection::ProjectionExec;


gpu_exec_node!(GpuProjectExec);
impl GpuExtraDisplay for GpuProjectExec {
    fn extra_display_info(&self) -> String {
        let pe = self.inner.as_any().downcast_ref::<ProjectionExec>().unwrap();
        let exprs: Vec<String> = pe
            .expr()
            .iter()
            .map(|(e, alias)| format!("{e} as {alias}"))
            .collect();
        format!("expr=[{}]", exprs.join(", "))
    }
}


// ---------------------------------------------------------------------------
// FlatBuffer wire format (Inc3: moved verbatim from plan_serializer.rs)
//
// STATEMENT ORDER IS THE WIRE FORMAT. FlatBufferBuilder is a no-interning bump
// arena, so every builder call appends and returns an offset — reordering the
// statements below, or hoisting a create_string, changes the bytes even though the
// values are identical. Do not "tidy" these bodies. testdata/goldens/plan_bytes.sha256
// pins them; the C++ side reads what they emit.
// ---------------------------------------------------------------------------

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::generated::gpu_plan_generated::peacock::plan as fb;
use datafusion::physical_plan::PhysicalExpr;
use crate::plan_serializer::{deserialize_expr, deserialize_plan_node};
use crate::plan_serializer::{serialize_expr, serialize_plan_node};
use crate::PartitionMode;

pub(crate) fn serialize_gpu_project<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
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

    let input = serialize_plan_node(b, proj.input(), pm)?;

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

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_gpu_project(
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


// --- Operator: partition topology + strip behavior ------------------------

impl Operator for GpuProjectExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Map
    }
}
