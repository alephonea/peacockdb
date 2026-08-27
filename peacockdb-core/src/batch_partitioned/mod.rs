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
pub mod cpu_backend;
pub mod cpu_batch;
pub mod driver;
pub mod error;
pub mod estimator;
pub mod executor;
pub mod expr;
pub mod expr_physical;
pub mod expr_translate;
pub mod forwarder;
pub mod layout;
pub mod node;
pub mod nodes;
pub mod nulls;
pub mod parquet_meta;
pub mod partitioner;
pub mod plan;
pub mod plan_text;
pub mod recipe;
pub mod schema;
pub mod translate;
pub(crate) mod validate;

#[cfg(not(feature = "rust-only"))]
pub mod gpu_backend;
#[cfg(not(feature = "rust-only"))]
pub mod gpu_batch;

pub use backend::{Backend, NodeExecutors};
pub use batch::Batch;
pub use cpu_batch::CpuBatch;
pub use error::{PlanError, RunError, When};
pub use executor::{CallStats, Executor};
pub use expr::{BinaryOp, ColumnRef, Expr, UnaryOp};
pub use layout::{BatchLayout, KeyDistribution, NodeKind, PartitionLayout, SortOrder};
pub use node::{GpuNode, RowInterval};
pub use nodes::{ExecutorCategory, category_of};
pub use partitioner::{Batching, RowGroupMeta};
pub use schema::Schema;

#[cfg(not(feature = "rust-only"))]
pub use gpu_batch::GpuBatch;
