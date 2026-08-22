//! [`Backend`] — one impl per backend, naming a concrete type for the batch and for
//! every executor category.
//!
//! The drivers are generic over it, so each backend monomorphizes: no vtable and no
//! allocation for a batch, and `Drop` on a GPU handle is a direct call. Backend choice
//! is a turbofish at the entry point, not a selector consulted per node. It is also what
//! keeps the tier boundary structural — a rust-only target instantiates the drivers only
//! at the CPU backend, so the GPU types are never named and never monomorphized.

use super::executor::{
    BatchAccumulatorExecutor, ExecExecutor, JoinExecutor, PartitionAccumulatorExecutor,
    PartitionEmitterExecutor, SourceExecutor, UnloadExecutor,
};
use super::forwarder::Forwarder;
use super::node::GpuNode;

pub trait Backend: Sized {
    /// GPU: the open `NodeSession`; CPU: ().
    type Context;
    type Batch: super::batch::Batch;
    type Source: SourceExecutor<Self>;
    type Exec: ExecExecutor<Self>;
    type BatchAcc: BatchAccumulatorExecutor<Self>;
    type PartAcc: PartitionAccumulatorExecutor<Self>;
    type Emitter: PartitionEmitterExecutor<Self>;
    type Join: JoinExecutor<Self>;
    type Unload: UnloadExecutor<Self>;

    /// A fresh instance set per call, so the driver instantiates per lane; `lane` is
    /// needed because a loader's lane picks its own row groups out of the partitioner's
    /// mapping. Construction lives here rather than on the node so that a node describes
    /// what it computes and stops knowing that backends exist.
    fn executors_for(
        ctx: &Self::Context,
        node: &dyn GpuNode,
        lane: usize,
    ) -> Result<NodeExecutors<Self>, super::error::PlanError>;
}

/// Which trait drives a node, with the executor stored inline — the match compiles to a
/// jump. `ProbingJoin` is absent because it comes from `set_build`, not from the backend.
pub enum NodeExecutors<B: Backend> {
    Source(B::Source),
    Exec(B::Exec),
    BatchAccumulator(B::BatchAcc),
    PartitionAccumulator(B::PartAcc),
    PartitionEmitter(B::Emitter),
    Join(B::Join),
    Unload(B::Unload),
    /// GpuMergePartitions, GpuUnion, GpuInterleave — routing only, no backend.
    BatchForwarder(Forwarder),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_partitioned::batch::Batch;
    use crate::batch_partitioned::cpu_batch::CpuBatch;
    use crate::batch_partitioned::error::PlanError;
    use crate::batch_partitioned::executor::{
        CallStats, Executor, LaneEvent, ProbingJoin, RowRange, SourceStep,
    };
    use crate::batch_partitioned::layout::NodeKind;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::arrow::datatypes::Schema as ArrowSchema;
    use std::sync::Arc;

    /// A batch that is a handle and a row count, as `GpuBatch` is — the second backend
    /// exists to differ from the first here, which is what the generic code must absorb.
    struct HandleBatch {
        rows: usize,
    }
    impl Batch for HandleBatch {
        fn num_rows(&self) -> usize {
            self.rows
        }
        fn byte_size(&self) -> usize {
            self.rows * 8
        }
    }

    fn empty_cpu_batch() -> CpuBatch {
        CpuBatch::new(RecordBatch::new_empty(Arc::new(ArrowSchema::empty())))
    }

    // One backend per invocation: every category is the same stub type, since what is
    // under test is that the seven associated types resolve and the generic driver
    // monomorphizes — not what any operator computes.
    macro_rules! stub_backend {
        ($backend:ident, $ops:ident, $probing:ident, $batch:ty, $make:expr) => {
            struct $backend;
            struct $ops;
            struct $probing;

            impl Executor for $ops {
                fn resident_bytes(&self) -> usize {
                    0
                }
                fn scratch_bytes(&self, _n_rows: u64, n_bytes: usize) -> usize {
                    n_bytes
                }
            }
            impl Executor for $probing {
                fn resident_bytes(&self) -> usize {
                    1
                }
                fn scratch_bytes(&self, _n_rows: u64, _n_bytes: usize) -> usize {
                    0
                }
            }

            impl SourceExecutor<$backend> for $ops {
                fn next_batch(self) -> SourceStep<$backend> {
                    SourceStep::Batch {
                        batch: $make,
                        stats: CallStats::default(),
                        source: self,
                    }
                }
            }
            impl ExecExecutor<$backend> for $ops {
                fn exec(&mut self, batch: $batch) -> ($batch, CallStats) {
                    (batch, CallStats::default())
                }
            }
            impl BatchAccumulatorExecutor<$backend> for $ops {
                fn accumulate_and_fetch(&mut self, batch: $batch) -> (Vec<$batch>, CallStats) {
                    (vec![batch], CallStats::default())
                }
                fn mark_done_and_fetch(self) -> (Vec<$batch>, CallStats) {
                    (Vec::new(), CallStats::default())
                }
            }
            impl PartitionAccumulatorExecutor<$backend> for $ops {
                fn accumulate_and_fetch(
                    &mut self,
                    _partition: usize,
                    event: LaneEvent<$backend>,
                ) -> (Vec<$batch>, CallStats) {
                    let out = match event {
                        LaneEvent::Batch(batch) => vec![batch],
                        LaneEvent::Done => Vec::new(),
                    };
                    (out, CallStats::default())
                }
            }
            impl PartitionEmitterExecutor<$backend> for $ops {
                fn emit(&mut self, batch: $batch) -> (Vec<$batch>, CallStats) {
                    (vec![batch], CallStats::default())
                }
            }
            impl JoinExecutor<$backend> for $ops {
                type Probing = $probing;
                fn set_build(self, _batch: $batch) -> ($probing, CallStats) {
                    ($probing, CallStats::default())
                }
            }
            impl ProbingJoin<$backend> for $probing {
                fn probe_and_fetch(&mut self, batch: $batch) -> (Vec<$batch>, CallStats) {
                    (vec![batch], CallStats::default())
                }
                fn finish_and_fetch(self) -> (Vec<$batch>, CallStats) {
                    (Vec::new(), CallStats::default())
                }
            }
            impl UnloadExecutor<$backend> for $ops {
                fn unload(&mut self, _batch: $batch, _rows: RowRange) -> (CpuBatch, CallStats) {
                    (empty_cpu_batch(), CallStats::default())
                }
            }

            impl Backend for $backend {
                type Context = ();
                type Batch = $batch;
                type Source = $ops;
                type Exec = $ops;
                type BatchAcc = $ops;
                type PartAcc = $ops;
                type Emitter = $ops;
                type Join = $ops;
                type Unload = $ops;

                fn executors_for(
                    _ctx: &(),
                    node: &dyn GpuNode,
                    _lane: usize,
                ) -> Result<NodeExecutors<Self>, PlanError> {
                    match node.kind() {
                        NodeKind::Source { .. } => Ok(NodeExecutors::Source($ops)),
                        NodeKind::Intermediate { .. } => Ok(NodeExecutors::Exec($ops)),
                        NodeKind::Sink => Ok(NodeExecutors::Unload($ops)),
                    }
                }
            }
        };
    }

    stub_backend!(
        FirstBackend,
        FirstOps,
        FirstProbing,
        CpuBatch,
        empty_cpu_batch()
    );
    stub_backend!(
        SecondBackend,
        SecondOps,
        SecondProbing,
        HandleBatch,
        HandleBatch { rows: 7 }
    );

    /// One build -> probe -> finish transition, written once for every backend.
    fn drive_join<B: Backend>(join: B::Join, build: B::Batch, probe: B::Batch) -> usize {
        let (mut probing, _) = join.set_build(build);
        let (probed, _) = probing.probe_and_fetch(probe);
        let held = probing.resident_bytes();
        let (finished, _) = probing.finish_and_fetch();
        held + probed
            .iter()
            .chain(finished.iter())
            .map(Batch::num_rows)
            .sum::<usize>()
    }

    /// One source step, written once for every backend.
    fn drive_source<B: Backend>(source: B::Source) -> usize {
        match source.next_batch() {
            SourceStep::Batch { batch, .. } => batch.num_rows(),
            SourceStep::Exhausted => 0,
        }
    }

    #[test]
    fn one_generic_driver_serves_two_backends_with_different_batch_types() {
        assert_eq!(
            drive_join::<FirstBackend>(FirstOps, empty_cpu_batch(), empty_cpu_batch()),
            1
        );
        assert_eq!(
            drive_join::<SecondBackend>(
                SecondOps,
                HandleBatch { rows: 3 },
                HandleBatch { rows: 4 }
            ),
            5
        );
        assert_eq!(drive_source::<FirstBackend>(FirstOps), 0);
        assert_eq!(drive_source::<SecondBackend>(SecondOps), 7);
    }
}
