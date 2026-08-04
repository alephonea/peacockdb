//! Union family. Interleave lives HERE, not in repartition: it shares union's
//! serialized form (both emit PlanNodeKind::GpuUnion) and its serializer body.

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


gpu_exec_node!(GpuUnionExec);
impl GpuExtraDisplay for GpuUnionExec {}

gpu_exec_node!(GpuInterleaveExec);
impl GpuExtraDisplay for GpuInterleaveExec {}


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
use crate::plan_serializer::deserialize_plan_node;
use crate::plan_serializer::{serialize_plan_node, serialize_schema};
use crate::PartitionMode;

pub(crate) fn serialize_gpu_union<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    interleave: bool,
    pm: PartitionMode,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    // UnionExec and InterleaveExec carry no extra state beyond their children,
    // so serialize the inputs directly off the wrapper (no inner downcast).
    let mut inputs = Vec::with_capacity(plan.children().len());
    for child in plan.children() {
        inputs.push(serialize_plan_node(b, child, pm)?);
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

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_gpu_union(u: &fb::GpuUnion) -> Result<Arc<dyn ExecutionPlan>, String> {
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


// --- Operator: partition topology + strip behavior ------------------------

/// NOT stripped, same reason as `GpuCrossJoinExec` in `join.rs`. Contrast
/// GpuInterleaveExec, which IS.
impl Operator for GpuUnionExec {
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

/// IS stripped, and it MATTERS: build_stream feeds single-partition stream stubs, under which InterleaveExec::try_new cannot interleave, so it is rebuilt as an equivalent UnionExec. Unstripped, the wrapper's with_new_children surfaces that as an error instead.
impl Operator for GpuInterleaveExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Map
    }
}
