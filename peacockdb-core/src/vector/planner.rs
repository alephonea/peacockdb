//! Physical planning for [`VectorTopK`]: an [`ExtensionPlanner`] lowers it to a
//! [`GpuVectorSearchExec`] (strategy pinned ExactBrute, ann_index_id 0), wrapped
//! in a [`QueryPlanner`] so `create_physical_plan` picks it up. Registered on both
//! session builders (see `lib.rs`).

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::{internal_err, Result};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::session_state::SessionState;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_expr::create_physical_expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{
    DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner,
};

use super::exec::GpuVectorSearchExec;
use super::logical::VectorTopK;

#[derive(Debug, Default)]
pub struct VectorTopKPlanner;

#[async_trait]
impl ExtensionPlanner for VectorTopKPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(vtopk) = node.as_any().downcast_ref::<VectorTopK>() else {
            return Ok(None);
        };
        if logical_inputs.len() != 1 || physical_inputs.len() != 1 {
            return internal_err!("VectorTopK expects exactly one input");
        }
        // Lower the logical distance expr against the input schema.
        let distance = create_physical_expr(
            vtopk.distance(),
            logical_inputs[0].schema(),
            session_state.execution_props(),
        )?;
        let exec = GpuVectorSearchExec::new(
            physical_inputs[0].clone(),
            Some(distance),
            vtopk.k(),
            vtopk.query().to_vec(),
            vtopk.dim(),
        );
        Ok(Some(Arc::new(exec)))
    }
}

/// Query planner delegating to the default physical planner plus our vector
/// extension planner.
#[derive(Debug, Default)]
pub struct VectorQueryPlanner;

#[async_trait]
impl QueryPlanner for VectorQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(VectorTopKPlanner)]);
        planner.create_physical_plan(logical_plan, session_state).await
    }
}
