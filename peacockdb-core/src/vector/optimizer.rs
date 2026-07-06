//! `PushFilterIntoVectorTopK` — a `Filter` directly above a [`VectorTopK`] moves
//! below it, so the selection pre-filters the candidate set the top-k scores
//! (the vector-search "filtered top-k" shape). The filter references input
//! columns, never the (non-materialized) distance, so this is always valid here.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::Result;
use datafusion::logical_expr::{Extension, Filter, LogicalPlan};
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use super::logical::VectorTopK;

#[derive(Debug, Default)]
pub struct PushFilterIntoVectorTopK;

impl OptimizerRule for PushFilterIntoVectorTopK {
    fn name(&self) -> &str {
        "push_filter_into_vector_top_k"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        plan.transform_down(push_one)
    }
}

fn push_one(node: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Filter(filter) = &node else {
        return Ok(Transformed::no(node));
    };
    let LogicalPlan::Extension(ext) = filter.input.as_ref() else {
        return Ok(Transformed::no(node));
    };
    let Some(vtopk) = ext.node.as_any().downcast_ref::<VectorTopK>() else {
        return Ok(Transformed::no(node));
    };

    // Filter(VectorTopK(input))  ->  VectorTopK(Filter(input))
    let pushed = LogicalPlan::Filter(Filter::try_new(
        filter.predicate.clone(),
        Arc::new(vtopk.input().clone()),
    )?);
    let new_vtopk = VectorTopK::new(
        pushed,
        vtopk.distance().clone(),
        vtopk.k(),
        vtopk.query().to_vec(),
        vtopk.dim(),
    );
    Ok(Transformed::yes(LogicalPlan::Extension(Extension {
        node: Arc::new(new_vtopk),
    })))
}
