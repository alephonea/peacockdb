//! Filter family.

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
use datafusion::physical_plan::filter::FilterExec;


gpu_exec_node!(GpuFilterExec);
impl GpuExtraDisplay for GpuFilterExec {
    fn extra_display_info(&self) -> String {
        let fe = self.inner.as_any().downcast_ref::<FilterExec>().unwrap();
        let mut s = format!("predicate={}", fe.predicate());
        if let Some(proj) = fe.projection() {
            let cols: Vec<String> = proj.iter().map(|i| i.to_string()).collect();
            s.push_str(&format!(", projection=[{}]", cols.join(", ")));
        }
        s
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
use crate::plan_serializer::{deserialize_expr, deserialize_plan_node};
use crate::plan_serializer::{serialize_expr, serialize_plan_node};
use crate::PartitionMode;

pub(crate) fn serialize_gpu_filter<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
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
    let input = serialize_plan_node(b, input_plan, pm)?;

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

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_gpu_filter(
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


// --- Operator: partition topology + strip behavior ------------------------

impl Operator for GpuFilterExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Map
    }
}
