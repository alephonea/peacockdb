//! [`GpuBatch`]: a handle to a resident `cudf::table`, plus the session it belongs to.
//!
//! The handle is the whole value — no box, no vtable — and `Drop` releases it, which is
//! what keeps a batch the driver abandons from leaking VRAM. A handle an FFI call
//! consumed must skip that drop: C++ erased it, and releasing it again is a use of a
//! dead handle. That boundary is owned by the helper the GPU backend will add (T9), not
//! by callers.

use std::fmt;

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
