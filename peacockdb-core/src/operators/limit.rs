//! Limit family.

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
use datafusion::physical_plan::limit::GlobalLimitExec;


gpu_exec_node!(GpuGlobalLimitExec);
impl GpuExtraDisplay for GpuGlobalLimitExec {
    fn extra_display_info(&self) -> String {
        let gl = self.inner.as_any().downcast_ref::<GlobalLimitExec>().unwrap();
        match gl.fetch() {
            Some(f) => format!("skip={}, fetch={}", gl.skip(), f),
            None => format!("skip={}, fetch=None", gl.skip()),
        }
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
use crate::plan_serializer::deserialize_plan_node;
use crate::plan_serializer::serialize_plan_node;
use crate::PartitionMode;

pub(crate) fn serialize_gpu_limit<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Arc<dyn ExecutionPlan>,
    pm: PartitionMode,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    use datafusion::physical_plan::limit::GlobalLimitExec;

    let gpu_limit = plan.as_any().downcast_ref::<GpuGlobalLimitExec>().unwrap();
    let limit = gpu_limit
        .inner()
        .as_any()
        .downcast_ref::<GlobalLimitExec>()
        .ok_or("GpuGlobalLimitExec inner is not GlobalLimitExec")?;

    let input = serialize_plan_node(b, limit.input(), pm)?;
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

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_gpu_limit(l: &fb::GpuLimit) -> Result<Arc<dyn ExecutionPlan>, String> {
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


// --- Operator: partition topology + strip behavior ------------------------

/// NOT stripped, same reason as `GpuCrossJoinExec` in `join.rs`.
impl Operator for GpuGlobalLimitExec {
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
