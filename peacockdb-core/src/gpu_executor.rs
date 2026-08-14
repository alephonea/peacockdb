//! Compatibility facade. [`GpuExecutor`] moved to
//! [`crate::executors::all_at_once_gpu_executor`], alongside the all-at-once
//! mode class it backs. This keeps `peacockdb_core::gpu_executor::GpuExecutor`
//! resolving for existing consumers.

pub use crate::executors::all_at_once_gpu_executor::{AllAtOnceGpuExecutor, GpuExecutor};

/// Per-node GPU timing switch (benchmark mode) and the resolution floor of what it
/// reports — see
/// [`crate::executors::backend::gpu_node_executor::set_node_timing`] and
/// [`crate::executors::backend::gpu_node_executor::measure_timing_floor_us`].
pub use crate::executors::backend::gpu_node_executor::{measure_timing_floor_us, set_node_timing};
