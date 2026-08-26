//! [`GpuBatch`]: a handle to a resident `cudf::table`, plus the session it belongs to.
//!
//! The handle is the whole value — no box, no vtable — and `Drop` releases it, which is
//! what keeps a batch the driver abandons from leaking VRAM. A handle an FFI call
//! consumed must skip that drop: C++ erased it, and releasing it again is a use of a
//! dead handle. [`GpuBatch::consume`] is that boundary, and the only place the release
//! is skipped.

use std::fmt;
use std::mem::ManuallyDrop;

use peacockdb_ffi::raw::{PeacockExecutor, peacock_handle_release};

use super::batch::Batch;

/// The executor pointer is BORROWED, as everywhere else on the GPU path: the session
/// outlives every batch drawn from it.
pub struct GpuBatch {
    executor: *mut PeacockExecutor,
    handle: u64,
    num_rows: usize,
    byte_size: usize,
}

impl GpuBatch {
    pub fn new(
        executor: *mut PeacockExecutor,
        handle: u64,
        num_rows: usize,
        byte_size: usize,
    ) -> Self {
        Self {
            executor,
            handle,
            num_rows,
            byte_size,
        }
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }

    pub fn executor(&self) -> *mut PeacockExecutor {
        self.executor
    }

    /// Hand the handle to an FFI call that consumes it — a slice, or an executor call
    /// taking it as an input. The batch is gone by move, and its release is skipped
    /// because C++ has erased the registry entry: releasing again would be a use of a
    /// dead handle. Every other way out of a `GpuBatch` runs `Drop`.
    pub fn consume(self) -> (*mut PeacockExecutor, u64) {
        let batch = ManuallyDrop::new(self);
        (batch.executor, batch.handle)
    }
}

impl Batch for GpuBatch {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn byte_size(&self) -> usize {
        self.byte_size
    }
}

impl Drop for GpuBatch {
    fn drop(&mut self) {
        unsafe { peacock_handle_release(self.executor, self.handle) };
    }
}

impl fmt::Debug for GpuBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuBatch")
            .field("handle", &self.handle)
            .field("num_rows", &self.num_rows)
            .finish()
    }
}
