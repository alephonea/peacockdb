// Raw FFI bindings to libpeacock_gpu.
//
// In rust-only mode none of these symbols are linked; callers must gate
// usage behind `#[cfg(not(feature = "rust-only"))]`.

#[cfg(not(feature = "rust-only"))]
pub mod raw {
    use std::ffi::c_char;

    #[repr(C)]
    pub struct PeacockExecutor {
        _opaque: [u8; 0],
    }

    /// Actual per-node costs (node-by-node interface). Rust applies the shared
    /// ColAccum overhead from rows+schema and adds `varlen_content_bytes`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct PeacockNodeStats {
        pub rows: u64,
        pub varlen_content_bytes: u64,
        /// Microseconds this output partition took; 0 unless
        /// [`peacock_set_node_timing`] is on. A node's time is Σ over partitions.
        pub time_us: u64,
    }

    #[link(name = "peacock_gpu")]
    unsafe extern "C" {
        pub fn peacock_gpu_version() -> *const c_char;

        pub fn peacock_executor_create(
            gpu_memory_limit: u64,
            out_executor: *mut *mut PeacockExecutor,
        ) -> i32;

        pub fn peacock_executor_destroy(executor: *mut PeacockExecutor);

        pub fn peacock_execute(
            executor: *mut PeacockExecutor,
            plan_bytes: *const u8,
            plan_len: u64,
            out_result_bytes: *mut *mut u8,
            out_result_len: *mut u64,
        ) -> i32;

        pub fn peacock_result_free(result_bytes: *mut u8);
        pub fn peacock_last_error(executor: *mut PeacockExecutor) -> *const c_char;

        /// Turn per-node timing on/off (process-global; OFF by default). When on,
        /// `peacock_executor_execute_node` synchronizes the default stream at every
        /// measurement boundary and fills `PeacockNodeStats::time_us`. The sync is
        /// what makes the number real (cuDF work is async and this path has no sync
        /// of its own) and also what makes it costly — hence opt-in. Used by the
        /// `peacock_gpu_benchmarks` target.
        pub fn peacock_set_node_timing(enable: i32);

        /// Cost of the measurement itself, in microseconds: the same timed region a
        /// node pays, around no work. A node's `time_us` is real work PLUS one of
        /// these, so a node at or below this is below the method's resolution rather
        /// than cheap. Report alongside; never subtract. Returns the second-smallest
        /// of `samples` (clamped to >= 2), or 0 if CUDA errored.
        ///
        /// Needs a live CUDA context and no concurrent work on the default stream.
        pub fn peacock_measure_timing_floor_us(samples: u32) -> u64;

        // --- node-by-node execution (unified node-executor interface) ---
        pub fn peacock_executor_begin_plan(
            executor: *mut PeacockExecutor,
            plan_bytes: *const u8,
            plan_len: u64,
            out_node_count: *mut u64,
        ) -> i32;

        pub fn peacock_executor_execute_node(
            executor: *mut PeacockExecutor,
            seq: u64,
            input_handles: *const u64,
            input_child_counts: *const u64,
            n_children: u64,
            out_handles: *mut u64,
            out_cap: u64,
            out_count: *mut u64,
            out_stats: *mut PeacockNodeStats,
        ) -> i32;

        /// Execute the `CudfScan` at post-order `seq` reading exactly
        /// `row_groups[0..n)` rather than the list the node carries, storing its one
        /// output as a new resident handle — the batch-partitioned loader's one call
        /// per batch. `n == 0` is refused rather than read as "every group", and a
        /// `seq` naming any other kind of node fails saying which.
        pub fn peacock_executor_execute_scan_rowgroups(
            executor: *mut PeacockExecutor,
            seq: u64,
            row_groups: *const u32,
            n: u64,
            out_handle: *mut u64,
            out_stats: *mut PeacockNodeStats,
        ) -> i32;

        /// Copy rows `[offset, offset+length)` of `handle` into a new resident handle,
        /// CONSUMING `handle`: C++ erases it, so the caller must not release it
        /// afterwards. Ranges clamp as on [`peacock_result_from_handle`]. Serves a
        /// mid-plan limit, whose kept rows feed further GPU work and so must stay
        /// resident rather than become a result.
        pub fn peacock_executor_slice_handle(
            executor: *mut PeacockExecutor,
            handle: u64,
            offset: u64,
            length: u64,
            out_handle: *mut u64,
        ) -> i32;

        /// Materialize rows `[offset, offset+length)` of a resident handle as an Arrow
        /// IPC stream; `length == u64::MAX` means to the end, which is what a caller
        /// wanting the whole table passes. A range naming no rows of a non-empty table
        /// exports nothing (`*out_ipc_len == 0`, nothing to free) and one running past
        /// the end clamps, because a limit's fetch legitimately overruns the batch it
        /// straddles. Does NOT release the handle.
        pub fn peacock_result_from_handle(
            executor: *mut PeacockExecutor,
            handle: u64,
            offset: u64,
            length: u64,
            out_ipc: *mut *mut u8,
            out_ipc_len: *mut u64,
        ) -> i32;

        pub fn peacock_handle_release(executor: *mut PeacockExecutor, handle: u64);
        pub fn peacock_executor_end_plan(executor: *mut PeacockExecutor);

        // Conformance hook (stateless): GPU Spark-murmur3 partition ids for the
        // key columns of an Arrow C-Data struct array. `schema`/`array` are pointers
        // to arrow's FFI_ArrowSchema/FFI_ArrowArray (ABI-compatible with cuDF's
        // ArrowSchema/ArrowArray). Used by the live GPU↔CPU conformance test.
        pub fn peacock_spark_partition_ids(
            schema: *const std::ffi::c_void,
            array: *const std::ffi::c_void,
            key_cols: *const u32,
            num_keys: u64,
            num_partitions: u32,
            seed: u32,
            out_pids: *mut i32,
            out_cap: u64,
            out_n: *mut u64,
        ) -> i32;
    }
}

#[cfg(not(feature = "rust-only"))]
pub fn version() -> &'static str {
    let ptr = unsafe { raw::peacock_gpu_version() };
    let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
    cstr.to_str().expect("version string is valid UTF-8")
}

#[cfg(feature = "rust-only")]
pub fn version() -> &'static str {
    "0.1.0-cpu"
}
