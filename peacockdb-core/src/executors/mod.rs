//! Execution modes, the contract they share, and the machinery beneath them.
//!
//! Layering:
//!   executor.rs      the [`Executor`] / [`InstrumentedExecutor`] contract + stats types
//!   node_by_node.rs  the backend-agnostic driver + the [`NodeExecutor`] trait
//!   backend/         the two [`NodeExecutor`] impls (DataFusion, C++/cuDF FFI)
//!   stream.rs        streaming plumbing + the resident-OOM enforcer
//!   single_node.rs   the per-node CPU primitive and shared helpers
//!
//! Five mode classes sit on top. Four ride the shared driver; full_table_cpu does
//! NOT — it is a recursive streaming path that coalesces to one partition and owns
//! the resident-OOM hook. Partition mode is implied by the class, never a param:
//!
//!   full_table_cpu     recursive streaming, tp1 or tp8 hint, SinglePartition
//!   partitioned_cpu    driver + CPU backend, tp8, RealMultiPartition
//!   full_table_gpu     driver + GPU backend, tp1, SinglePartition
//!   partitioned_gpu    driver + GPU backend, tp8, RealMultiPartition
//!   all_at_once_gpu    one `peacock_execute` FFI call; [`Executor`] only (no per-node stats)
//!
//! full_table_gpu and partitioned_gpu are deliberately THIN: same backend, same
//! driver, differing only by constructed [`crate::PartitionMode`].

pub mod backend;
pub mod executor;
pub mod node_by_node;
pub mod single_node;
pub mod stream;

pub mod full_table_cpu_executor;
pub mod partitioned_cpu_executor;

#[cfg(not(feature = "rust-only"))]
pub mod all_at_once_gpu_executor;
#[cfg(not(feature = "rust-only"))]
pub mod full_table_gpu_executor;
#[cfg(not(feature = "rust-only"))]
pub mod partitioned_gpu_executor;

pub use executor::{Executor, InstrumentedExecutor, NodeMemoryStats, PartitionStat};
pub use full_table_cpu_executor::FullTableCpuExecutor;
pub use node_by_node::{execute_node_by_node, NodeExecutor};
pub use partitioned_cpu_executor::PartitionedCpuExecutor;

#[cfg(not(feature = "rust-only"))]
pub use all_at_once_gpu_executor::{AllAtOnceGpuExecutor, GpuExecutor};
#[cfg(not(feature = "rust-only"))]
pub use backend::gpu_node_executor::{
    install_rmm_pool, set_node_timing, RmmPool,
};
#[cfg(not(feature = "rust-only"))]
pub use full_table_gpu_executor::FullTableGpuExecutor;
#[cfg(not(feature = "rust-only"))]
pub use partitioned_gpu_executor::PartitionedGpuExecutor;
