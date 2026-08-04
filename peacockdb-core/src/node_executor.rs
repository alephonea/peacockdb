//! Compatibility facade. The node-by-node driver, its `NodeExecutor` trait, and the
//! CPU/GPU backends moved to [`crate::executors`]. This module keeps the
//! `peacockdb_core::node_executor::…` paths resolving for existing consumers.

pub use crate::executors::node_by_node::{execute_node_by_node, NodeExecutor};

pub use crate::executors::backend::cpu_node_executor::CpuNodeExecutor;

#[cfg(not(feature = "rust-only"))]
pub use crate::executors::backend::gpu_node_executor::GpuNodeExecutor;
