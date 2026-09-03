//! Compatibility facade. [`GpuExecutor`] moved to
//! [`crate::executors::all_at_once_gpu_executor`], alongside the all-at-once
//! mode class it backs. This keeps `peacockdb_core::gpu_executor::GpuExecutor`
//! resolving for existing consumers.

pub use crate::executors::all_at_once_gpu_executor::{AllAtOnceGpuExecutor, GpuExecutor};

/// Benchmark-mode instrumentation: the per-node timing switch, the resolution floor of
/// what it reports, and the pooled device allocator those numbers should be taken under.
/// See [`crate::executors::backend::gpu_node_executor`].
pub use crate::executors::backend::gpu_node_executor::{
    NodeTiming, NvtxRange, RmmPool, install_rmm_pool, measure_timing_floor_us,
    node_timing_on, nvtx_range, set_node_timing, set_nvtx_ranges,
};
