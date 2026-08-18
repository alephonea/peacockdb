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
        /// Host microseconds this output partition spent BEFORE its first device
        /// touch: flatbuffer decode, handle lookups, AST construction. The
        /// peacockdb-only prologue bare cuDF has no analogue for, which is why it is
        /// its own field. 0 unless [`peacock_set_node_timing`] selected a mode.
        pub host_setup_us: u64,
        /// Host microseconds from the first device touch to the end of the region; the
        /// two modes are NOT comparable. Under [`PEACOCK_NODE_TIMING_EVENTS`] there is
        /// no explicit drain and the device work arrives via
        /// [`peacock_executor_collect_node_times`] — still not launch cost, since cuDF
        /// and rmm synchronize internally (tpch q3: within 0.01% of device). Under
        /// [`PEACOCK_NODE_TIMING_SYNC`] it contains the device execution outright.
        pub host_submit_us: u64,
        /// C++'s own reconstruction of the byte total, from cuDF types. Present to
        /// be COMPARED against the Rust value, never consumed in its place — see
        /// the debug assertion in `gpu_node_executor`.
        pub logical_bytes: u64,
        /// 1 when the partition's columns are, one for one, the types the node's
        /// `output_schema` declares. Scopes the comparison above — see the debug
        /// assertion in `gpu_node_executor` for which nodes legitimately report 0.
        pub schema_faithful: u64,
    }

    /// What [`peacock_install_rmm_pool`] did — sizes are 0 unless `state` is
    /// [`PEACOCK_RMM_POOL_INSTALLED`]. Mirrors `PeacockRmmPoolInfo` in
    /// `cpp/include/peacock_gpu.h`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct PeacockRmmPoolInfo {
        pub state: i32,
        /// 1 on an integrated part, which is sized by a different rule.
        pub integrated: i32,
        pub free_bytes: u64,
        pub initial_bytes: u64,
        pub maximum_bytes: u64,
    }

    /// A pool is the current device resource.
    pub const PEACOCK_RMM_POOL_INSTALLED: i32 = 0;
    /// The pool could not be built: the default resource, NOT on purpose.
    pub const PEACOCK_RMM_POOL_UNAVAILABLE: i32 = 1;

    /// One collected device interval — which node output partition, and what the
    /// device spent on it. Mirrors `PeacockNodeDeviceTime` in `cpp/include/peacock_gpu.h`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct PeacockNodeDeviceTime {
        pub seq: u64,
        pub partition: u64,
        pub device_us: u64,
    }

    /// The default, and the only mode a shipping query runs in.
    pub const PEACOCK_NODE_TIMING_OFF: i32 = 0;
    /// Host clock per region, closed by a default-stream sync. The sync serializes what
    /// cuDF would pipeline, so measuring changes what is measured; the baseline events
    /// get checked against.
    pub const PEACOCK_NODE_TIMING_SYNC: i32 = 1;
    /// CUDA events around the device work, host clock around the host work, no sync
    /// inside the region. Device times are read afterwards with
    /// [`peacock_executor_collect_node_times`].
    pub const PEACOCK_NODE_TIMING_EVENTS: i32 = 2;

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

        /// Install rmm's pooled device resource (process-global, idempotent, not
        /// installed by default). Must be called before any GPU work; without it every
        /// cuDF intermediate is a `cudaMalloc`/`cudaFree` round trip.
        ///
        /// Without it every cuDF intermediate the engine allocates is a
        /// `cudaMalloc`/`cudaFree` round trip. The C++ gtest binaries install the same
        /// pool from their `main()`; this exists so `peacock_gpu_benchmarks` — which is
        /// Rust and cannot include the C++ header that owns the sizing rule — measures
        /// the engine under the same allocator rather than producing node times that get
        /// compared with theirs anyway.
        ///
        /// Returns 0 unless `out_info` is null; NOT non-zero on
        /// [`PEACOCK_RMM_POOL_UNAVAILABLE`], which still leaves a runnable default
        /// resource. The caller decides whether that is fatal — for the benchmark
        /// harness it is.
        pub fn peacock_install_rmm_pool(out_info: *mut PeacockRmmPoolInfo) -> i32;

        /// Select the per-node timing mode (process-global; `PEACOCK_NODE_TIMING_OFF`
        /// by default). Opt-in because neither mode is free: SYNC drains the default
        /// stream at every boundary, EVENTS allocates a CUDA event pair per region and
        /// holds it until collection. Unknown values are treated as OFF. Used by the
        /// `peacock_gpu_benchmarks` target.
        pub fn peacock_set_node_timing(mode: i32);

        /// Emit NVTX ranges around plan nodes and their output partitions
        /// (process-global; off by default). A separate switch from
        /// [`peacock_set_node_timing`]: a profiling run wants the node boundaries
        /// without the event pairs, whose recording is device work a capture would
        /// show inside the node. Nonzero to emit.
        pub fn peacock_set_nvtx_ranges(on: i32);

        /// Drain the device intervals recorded since the last call, in execution
        /// order. Only [`PEACOCK_NODE_TIMING_EVENTS`] produces any. Call AFTER the
        /// root [`peacock_result_from_handle`] and BEFORE
        /// [`peacock_executor_end_plan`], which destroys the events.
        ///
        /// `out_count` is set to the number RECORDED, not the number that fit: when it
        /// exceeds `cap` the call returns non-zero and the surplus is gone (the drain
        /// already happened). Regions with an incomplete pair — a node that threw, one
        /// that never touched the device — are absent rather than zero.
        pub fn peacock_executor_collect_node_times(
            executor: *mut PeacockExecutor,
            out: *mut PeacockNodeDeviceTime,
            cap: u64,
            out_count: *mut u64,
        ) -> i32;

        /// Cost of the measurement itself under [`PEACOCK_NODE_TIMING_SYNC`]: that
        /// mode's region around no work. A SYNC node time is real work plus one of
        /// these, so a node at or below it is unresolvable rather than cheap — report
        /// alongside, never subtract. Measures SYNC even under events, since it is what
        /// says what events bought. Second-smallest of `samples` (clamped to >= 2), or
        /// 0 on CUDA error.
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
