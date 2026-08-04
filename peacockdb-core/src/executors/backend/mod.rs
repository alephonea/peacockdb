//! Backends for the node-by-node driver. Both implement
//! [`super::node_by_node::NodeExecutor`]; they differ only in how a single node is
//! executed — DataFusion vs the C++/cuDF FFI.

pub mod cpu_node_executor;

// The GPU backend links libpeacock_gpu; `rust-only` builds must not see it, nor any
// re-export of it (diag_flip_audit / test_node_executor depend on that build working).
#[cfg(not(feature = "rust-only"))]
pub mod gpu_node_executor;
