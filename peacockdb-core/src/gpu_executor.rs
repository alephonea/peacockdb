//! Compatibility facade. [`GpuExecutor`] moved to
//! [`crate::executors::all_at_once_gpu_executor`], alongside the all-at-once
//! mode class it backs. This keeps `peacockdb_core::gpu_executor::GpuExecutor`
//! resolving for existing consumers.

pub use crate::executors::all_at_once_gpu_executor::{AllAtOnceGpuExecutor, GpuExecutor};
