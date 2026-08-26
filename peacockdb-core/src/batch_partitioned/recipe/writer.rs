//! The walk that builds the recipe plan's bytes, and with them the seq numbers.
//!
//! A seq is a post-order index into the tree the C++ indexes, so it cannot be counted
//! from the recipes alone: whatever fills a structural slot is indexed too. Hence one
//! walk that builds the buffer and numbers as it goes. Three rules keep the order it
//! creates nodes in equal to the order `NodeSession::index_post_order` visits them, and
//! each is stated where it is applied — [`Writer::take`], [`Writer::stub`] and
//! [`Writer::reduce`]. Nothing addresses a stub or a structural union: they hold the tree
//! together, and the seqs a recipe publishes point at the nodes that carry a call.

use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, WIPOffset};

use crate::generated::gpu_plan_generated::peacock::plan as fb;

use super::super::error::PlanError;
use super::types::Seq;

/// The seq a failing payload would have taken, appended to the reason rather than wrapped
/// around it: a second `PlanError` around the first prints its prefix twice.
fn at_seq(why: PlanError, seq: Seq) -> PlanError {
    match why {
        PlanError::Unsupported(what) => PlanError::Unsupported(format!("{what} at #{seq}")),
        PlanError::Invalid(what) => PlanError::Invalid(format!("{what} at #{seq}")),
    }
}

/// A recipe-plan node under construction: which kind it is and its union payload.
pub(super) struct Payload {
    pub kind: fb::PlanNodeKind,
    pub value: WIPOffset<UnionWIPOffset>,
}

pub(super) struct Writer<'a> {
    builder: FlatBufferBuilder<'a>,
    /// Created and not yet taken as somebody's child, oldest first.
    pool: Vec<WIPOffset<fb::PlanNode<'a>>>,
    next_seq: Seq,
}

impl<'a> Writer<'a> {
    pub(super) fn new() -> Self {
        Self {
            builder: FlatBufferBuilder::new(),
            pool: Vec::new(),
            next_seq: 0,
        }
    }

    /// Build one addressed node: take `arity` inputs, stub whatever is missing, number it.
    ///
    /// A payload that cannot be written fails the whole plan, and the seq it would have
    /// taken is in the message. Nothing is substituted: a placeholder would be a node the
    /// recipe names as one kind and the buffer holds as another, and running it would
    /// return an empty or concatenated table rather than an error.
    pub(super) fn node<F>(&mut self, arity: usize, build: F) -> Result<Seq, PlanError>
    where
        F: FnOnce(
            &mut FlatBufferBuilder<'a>,
            &[WIPOffset<fb::PlanNode<'a>>],
        ) -> Result<Payload, PlanError>,
    {
        let taken = self.take(arity);
        let seq = self.next_seq;
        let payload = build(&mut self.builder, &taken).map_err(|why| at_seq(why, seq))?;
        Ok(self.push(payload).0)
    }

    /// The offsets for one node's slots: the unconsumed ones first, then stubs.
    ///
    /// The first rule — a node takes the most recently created unconsumed offsets, so its
    /// children are the contiguous block immediately below it, which is what post-order
    /// means. A stub is numbered here, before the node that takes it, for the same reason.
    fn take(&mut self, arity: usize) -> Vec<WIPOffset<fb::PlanNode<'a>>> {
        let from_pool = arity.min(self.pool.len());
        let mut taken = self.pool.split_off(self.pool.len() - from_pool);
        for _ in from_pool..arity {
            taken.push(self.stub());
        }
        taken
    }

    /// A scan of nothing, holding a slot open: the second rule, since an empty slot is a
    /// crash at plan load rather than an absent child. `CudfScan` is the only leaf the
    /// schema has — every other table carries `input`, `left`/`right` or `inputs`.
    fn stub(&mut self) -> WIPOffset<fb::PlanNode<'a>> {
        let scan = fb::CudfScan::create(&mut self.builder, &fb::CudfScanArgs::default());
        let (_, offset) = self.push(Payload {
            kind: fb::PlanNodeKind::CudfScan,
            value: scan.as_union_value(),
        });
        // Taken by the node being built, so it does not stay in the pool.
        self.pool.pop();
        offset
    }

    fn push(&mut self, payload: Payload) -> (Seq, WIPOffset<fb::PlanNode<'a>>) {
        let node = fb::PlanNode::create(
            &mut self.builder,
            &fb::PlanNodeArgs {
                node_type: payload.kind,
                node: Some(payload.value),
                output_schema: None,
            },
        );
        self.pool.push(node);
        let seq = self.next_seq;
        self.next_seq += 1;
        (seq, node)
    }

    /// How many nodes are unconsumed — a caller's mark for [`Writer::reduce`].
    pub(super) fn mark(&self) -> usize {
        self.pool.len()
    }

    /// Leave one offset for everything created since `mark`: what the node produced if
    /// that is all there is, or — the third rule — a structural union over the branches it
    /// did not consume, because an unreachable node is never indexed and every seq above
    /// it would shift with nothing saying why.
    pub(super) fn reduce(&mut self, mark: usize) -> Result<(), PlanError> {
        if self.pool.len() - mark <= 1 {
            return Ok(());
        }
        let inputs = self.pool.split_off(mark);
        let inputs = self.builder.create_vector(&inputs);
        let union = fb::CudfUnion::create(
            &mut self.builder,
            &fb::CudfUnionArgs {
                inputs: Some(inputs),
                interleave: false,
                output_schema: None,
            },
        );
        self.push(Payload {
            kind: fb::PlanNodeKind::CudfUnion,
            value: union.as_union_value(),
        });
        Ok(())
    }

    /// Finish on the one offset left, which is the root by construction: the last node
    /// created is the last visited in post-order. Returns the bytes and how many fb nodes
    /// went into them — stubs and structural unions included, since the C++ indexes those.
    pub(super) fn finish(mut self) -> Result<(Vec<u8>, Seq), PlanError> {
        let root = match self.pool.as_slice() {
            [root] => *root,
            other => {
                return Err(PlanError::Invalid(format!(
                    "the recipe plan ended with {} unconsumed nodes rather than one root",
                    other.len()
                )));
            }
        };
        let plan = fb::GpuPlan::create(&mut self.builder, &fb::GpuPlanArgs { root: Some(root) });
        self.builder.finish(plan, None);
        Ok((self.builder.finished_data().to_vec(), self.next_seq))
    }
}
