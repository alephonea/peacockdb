//! Union and interleave: lane relabeling, and nothing else. Neither touches a row — the
//! branch type normalization legacy does inside the executor is a per-branch `GpuProject`
//! the planner inserts, which is what leaves these two as pure routing.

use std::any::Any;

use super::super::error::PlanError;
use super::super::layout::{BatchLayout, KeyDistribution, NodeKind, PartitionLayout, SortOrder};
use super::super::node::GpuNode;
use super::super::schema::Schema;
use super::input_layout;

/// Output lanes are the sum of its branches', and output lane k is served by exactly one
/// of them, so no row changes lane and no lane waits on another. The hash a branch
/// carried says nothing about the union's numbering, so it goes.
#[derive(Debug)]
pub struct GpuUnion {
    kind: NodeKind,
    branches: Vec<Box<dyn GpuNode>>,
}

impl GpuUnion {
    pub fn new(branches: Vec<Box<dyn GpuNode>>, schema: Schema) -> Self {
        let n = branches.iter().map(|b| input_layout(b.as_ref()).n).sum();
        // Output lane k is one branch's lane, forwarded batch for batch, so an order and a
        // one-batch lane survive wherever every branch has them. The hash does not: lane k
        // means something different in each branch's numbering.
        let layout = PartitionLayout {
            n,
            key_distribution: KeyDistribution::NotSpecified,
            sort_order: agreed_sort_order(&branches),
            batch_layout: agreed_batch_layout(&branches),
        };
        Self {
            kind: NodeKind::Intermediate { layout, schema },
            branches,
        }
    }
}

impl GpuNode for GpuUnion {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        self.branches.iter().map(|b| b.as_ref()).collect()
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        check_branch_schemas("GpuUnion", self.kind.schema(), &self.branches)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Output lane p is lane p of each branch, which is why every branch must carry the same
/// hash: the distribution is what makes lane p of one branch belong beside lane p of the
/// next, and it survives because no row changes lane.
#[derive(Debug)]
pub struct GpuInterleave {
    kind: NodeKind,
    branches: Vec<Box<dyn GpuNode>>,
}

impl GpuInterleave {
    pub fn new(branches: Vec<Box<dyn GpuNode>>, schema: Schema) -> Self {
        let first = input_layout(branches.first().expect("interleave has branches").as_ref());
        // Lane p holds every branch's lane p, so each batch is still whatever it was — an
        // order within a batch survives — but k branches make k batches out of one lane.
        let batch_layout = if branches.len() == 1 {
            agreed_batch_layout(&branches)
        } else {
            BatchLayout::MultipleBatches
        };
        let layout = PartitionLayout {
            n: first.n,
            key_distribution: first.key_distribution.clone(),
            sort_order: agreed_sort_order(&branches),
            batch_layout,
        };
        Self {
            kind: NodeKind::Intermediate { layout, schema },
            branches,
        }
    }
}

impl GpuNode for GpuInterleave {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        self.branches.iter().map(|b| b.as_ref()).collect()
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        check_branch_schemas("GpuInterleave", self.kind.schema(), &self.branches)?;
        let first = input_layout(self.branches[0].as_ref());
        for branch in &self.branches[1..] {
            let layout = input_layout(branch.as_ref());
            if layout.n != first.n || layout.key_distribution != first.key_distribution {
                return Err(PlanError::Invalid(
                    "GpuInterleave: lane p is lane p of every branch, so all of them must \
                     carry the same hash distribution — otherwise this is a GpuUnion"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The order every branch agrees on, or none. Routing does not touch a batch, so an order
/// that holds within each branch's batches holds within the output's.
fn agreed_sort_order(branches: &[Box<dyn GpuNode>]) -> SortOrder {
    let first = input_layout(branches.first().expect("union has branches").as_ref()).sort_order;
    let agreed = branches
        .iter()
        .all(|branch| input_layout(branch.as_ref()).sort_order == first);
    if agreed {
        first
    } else {
        SortOrder::NotSpecified
    }
}

fn agreed_batch_layout(branches: &[Box<dyn GpuNode>]) -> BatchLayout {
    if branches
        .iter()
        .all(|branch| input_layout(branch.as_ref()).batch_layout == BatchLayout::SingleBatch)
    {
        BatchLayout::SingleBatch
    } else {
        BatchLayout::MultipleBatches
    }
}

/// Routing cannot retype anything, so a branch whose columns differ from the declared
/// output is a missing cast rather than something the executor should fix up (#41).
fn check_branch_schemas(
    node: &str,
    declared: Option<&Schema>,
    branches: &[Box<dyn GpuNode>],
) -> Result<(), PlanError> {
    let declared = declared.expect("a union is not a sink");
    for (index, branch) in branches.iter().enumerate() {
        let schema = branch.kind().schema().expect("a sink cannot be a branch");
        let mismatched = schema
            .fields
            .fields()
            .iter()
            .zip(declared.fields.fields().iter())
            .find(|(branch_field, out)| branch_field.data_type() != out.data_type());
        if schema.fields.fields().len() != declared.fields.fields().len() {
            return Err(PlanError::Invalid(format!(
                "{node}: branch {index} has {} columns and the output declares {}",
                schema.fields.fields().len(),
                declared.fields.fields().len()
            )));
        }
        if let Some((branch_field, out)) = mismatched {
            return Err(PlanError::Invalid(format!(
                "{node}: branch {index} emits {} as {:?} where the output declares {:?} — the \
                 planner inserts a casting GpuProject on that branch",
                branch_field.name(),
                branch_field.data_type(),
                out.data_type()
            )));
        }
    }
    Ok(())
}
