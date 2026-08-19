//! The batch-partitioned mode: a lane holds a stream of batches rather than one
//! resident table.
//!
//! This module is the vocabulary — plan nodes, the layout and schema they declare, and
//! the executor contracts the drivers call. The design and the reasons behind each
//! shape are in `llm-wiki/tasks/batch_partitioned_executor.md`; the legacy modes under
//! `executors/` are untouched by any of it.

pub mod aggregates;
pub mod backend;
pub mod batch;
pub mod cpu_batch;
pub mod error;
pub mod executor;
pub mod forwarder;
pub mod layout;
pub mod node;
pub mod partitioner;
pub mod schema;

#[cfg(not(feature = "rust-only"))]
pub mod gpu_batch;

pub use backend::{Backend, NodeExecutors};
pub use batch::Batch;
pub use cpu_batch::CpuBatch;
pub use error::PlanError;
pub use executor::{CallStats, Executor};
pub use layout::{BatchLayout, KeyDistribution, NodeKind, PartitionLayout, SortOrder};
pub use node::{GpuNode, RowInterval};
pub use partitioner::{Batching, RowGroupMeta};
pub use schema::Schema;

#[cfg(not(feature = "rust-only"))]
pub use gpu_batch::GpuBatch;
