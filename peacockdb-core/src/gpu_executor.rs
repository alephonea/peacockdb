//! Compatibility facade. [`GpuExecutor`] moved to
//! [`crate::executors::all_at_once_gpu_executor`], alongside the all-at-once
//! mode class it backs. This keeps `peacockdb_core::gpu_executor::GpuExecutor`
//! resolving for existing consumers.

pub use crate::executors::all_at_once_gpu_executor::{AllAtOnceGpuExecutor, GpuExecutor};

/// Benchmark-mode instrumentation: the per-node timing switch, the resolution floor of
/// what it reports, and the pooled device allocator those numbers should be taken under
/// — see
/// [`crate::executors::backend::gpu_node_executor::set_node_timing`],
/// [`crate::executors::backend::gpu_node_executor::measure_timing_floor_us`] and
/// [`crate::executors::backend::gpu_node_executor::install_rmm_pool`].
pub use crate::executors::backend::gpu_node_executor::{
    install_rmm_pool, measure_timing_floor_us, set_node_timing, RmmPool,
};
