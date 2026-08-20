//! The batch value type: one table's worth of rows, and nothing else about it.
//!
//! Ownership is by move — every executor method takes a batch by value, so reuse after
//! consumption is a compile error rather than an unknown-handle throw from C++. Neither
//! implementation is `Clone`: a `GpuBatch` cannot be, and the free `RecordBatch` clone
//! is not worth the asymmetry (a future dual consumer writes an explicit copy, #140).

pub trait Batch {
    fn num_rows(&self) -> usize;
    fn byte_size(&self) -> usize;
}
