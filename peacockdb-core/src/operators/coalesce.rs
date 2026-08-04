//! Coalesce family: batch coalescing and partition collapse.

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


gpu_exec_node!(GpuCoalesceBatchesExec);
// target_batch_size is deliberately NOT displayed: it is budget-derived
// (budget / subtree_max_row_bytes) and vestigial for observable output — see the
// note on `GpuScanExec`'s display in `scan.rs`.
impl GpuExtraDisplay for GpuCoalesceBatchesExec {}

gpu_exec_node!(GpuCoalescePartitionsExec);
impl GpuExtraDisplay for GpuCoalescePartitionsExec {}


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
use crate::plan_serializer::serialize_plan_node;
use crate::PartitionMode;

pub(crate) fn serialize_gpu_coalesce_batches<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu_cb = plan.as_any().downcast_ref::<GpuCoalesceBatchesExec>().unwrap();
    let cb = gpu_cb
        .inner()
        .as_any()
        .downcast_ref::<datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec>()
        .ok_or("GpuCoalesceBatchesExec inner is not CoalesceBatchesExec")?;

    let input = serialize_plan_node(b, cb.input(), pm)?;
    let node = fb::GpuCoalesceBatches::create(
        b,
        &fb::GpuCoalesceBatchesArgs {
            target_batch_size: cb.target_batch_size() as u32,
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::GpuCoalesceBatches, node.as_union_value()))
}
pub(crate) fn serialize_gpu_coalesce_partitions<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    let gpu_cp = plan.as_any().downcast_ref::<GpuCoalescePartitionsExec>().unwrap();
    let cp = gpu_cp
        .inner()
        .as_any()
        .downcast_ref::<datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec>()
        .ok_or("GpuCoalescePartitionsExec inner is not CoalescePartitionsExec")?;

    let input = serialize_plan_node(b, cp.input(), pm)?;
    let node = fb::GpuCoalescePartitions::create(
        b,
        &fb::GpuCoalescePartitionsArgs {
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::GpuCoalescePartitions, node.as_union_value()))
}

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_gpu_coalesce_batches(
    cb: &fb::GpuCoalesceBatches,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;

    let input = deserialize_plan_node(&cb.input().ok_or("GpuCoalesceBatches missing input")?)?;
    let inner: Arc<dyn ExecutionPlan> =
        Arc::new(CoalesceBatchesExec::new(input, cb.target_batch_size() as usize));
    Ok(Arc::new(GpuCoalesceBatchesExec::new(inner)))
}

pub(crate) fn deserialize_gpu_coalesce_partitions(
    cp: &fb::GpuCoalescePartitions,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;

    let input = deserialize_plan_node(&cp.input().ok_or("GpuCoalescePartitions missing input")?)?;
    let inner: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(input));
    Ok(Arc::new(GpuCoalescePartitionsExec::new(inner)))
}


// --- Operator: partition topology + strip behavior ------------------------

impl Operator for GpuCoalesceBatchesExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Map
    }
}

impl Operator for GpuCoalescePartitionsExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::Collapse
    }
}
