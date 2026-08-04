//! Compatibility facade. The memory-accounting helpers moved to [`crate::memory`]
//! and the executors to [`crate::executors`]; this keeps every
//! `peacockdb_core::cpu_executor::…` path resolving for existing consumers.

pub use crate::executors::executor::{NodeMemoryStats, PartitionStat};
pub use crate::memory::{
    batch_allocated_size, batch_logical_size, batch_varlen_content_bytes, logical_size_from_schema,
};

pub(crate) use crate::memory::{assert_type_accountable, ColAccum};
