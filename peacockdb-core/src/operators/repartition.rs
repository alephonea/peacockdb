//! Repartition family: the Spark-murmur3 hash shuffle.

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
use datafusion::physical_plan::repartition::RepartitionExec;


gpu_exec_node!(GpuRepartitionExec);
impl GpuExtraDisplay for GpuRepartitionExec {
    fn extra_display_info(&self) -> String {
        let rp = self.inner.as_any().downcast_ref::<RepartitionExec>().unwrap();
        let partitioning = rp.partitioning();
        let input_partitions = rp.input().properties().output_partitioning().partition_count();
        format!("partitioning={partitioning}, input_partitions={input_partitions}")
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
use datafusion::physical_plan::PhysicalExpr;
use crate::plan_serializer::{deserialize_expr, deserialize_plan_node};
use crate::plan_serializer::{serialize_expr, serialize_plan_node};
use crate::PartitionMode;

pub(crate) fn serialize_cudf_repartition<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    use datafusion::physical_plan::repartition::RepartitionExec;
    use datafusion::physical_plan::Partitioning;

    let gpu_rp = plan.as_any().downcast_ref::<GpuRepartitionExec>().unwrap();
    let rp = gpu_rp
        .inner()
        .as_any()
        .downcast_ref::<RepartitionExec>()
        .ok_or("GpuRepartitionExec inner is not RepartitionExec")?;

    let input = serialize_plan_node(b, rp.input(), pm)?;

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

    let node = fb::CudfRepartition::create(
        b,
        &fb::CudfRepartitionArgs {
            kind,
            num_partitions: num_partitions as u32,
            hash_exprs,
            input: Some(input),
        },
    );
    Ok((fb::PlanNodeKind::CudfRepartition, node.as_union_value()))
}

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_cudf_repartition(
    rp: &fb::CudfRepartition,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    use datafusion::physical_plan::repartition::RepartitionExec;
    use datafusion::physical_plan::Partitioning;

    let input = deserialize_plan_node(&rp.input().ok_or("CudfRepartition missing input")?)?;

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


// --- Operator: partition topology + strip behavior ------------------------

impl Operator for GpuRepartitionExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::RepartitionHash
    }
}
