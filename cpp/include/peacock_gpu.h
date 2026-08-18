#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Versioning
// ---------------------------------------------------------------------------

/// Returns a null-terminated version string, e.g. "0.1.0".
/// The returned pointer is valid for the lifetime of the process.
const char* peacock_gpu_version(void);

// ---------------------------------------------------------------------------
// Executor lifecycle
// ---------------------------------------------------------------------------

/// Opaque handle to a GPU executor instance.
typedef struct peacock_executor peacock_executor_t;

/// Create a GPU executor.
///
/// @param gpu_memory_limit  Maximum GPU memory (bytes) the executor may use.
///                          Pass 0 to use all available memory.
/// @param out_executor      Set to the newly created executor on success.
/// @return                  0 on success, non-zero on failure.
int peacock_executor_create(uint64_t gpu_memory_limit,
                            peacock_executor_t** out_executor);

/// Destroy a GPU executor and free associated resources.
void peacock_executor_destroy(peacock_executor_t* executor);

// ---------------------------------------------------------------------------
// Query execution
// ---------------------------------------------------------------------------

/// Execute a serialised physical plan.
///
/// @param executor          Executor handle.
/// @param plan_bytes        Flatbuffer-encoded physical plan.
/// @param plan_len          Length of plan_bytes in bytes.
/// @param out_result_bytes  On success, set to a newly allocated buffer
///                          containing the result (Arrow IPC stream).
///                          Caller must free with peacock_result_free().
///                          An empty result (no rows) is signalled by
///                          *out_result_len == 0; *out_result_bytes is
///                          unspecified in that case and must not be freed.
/// @param out_result_len    Set to the length of out_result_bytes.
/// @return                  0 on success, non-zero on failure. On failure
///                          *out_result_bytes and *out_result_len are
///                          unspecified — the caller must not read or free
///                          them, and should retrieve the error via
///                          peacock_last_error().
int peacock_execute(peacock_executor_t* executor,
                    const uint8_t* plan_bytes,
                    uint64_t plan_len,
                    uint8_t** out_result_bytes,
                    uint64_t* out_result_len);

/// Free a result buffer returned by peacock_execute().
void peacock_result_free(uint8_t* result_bytes);

/// Return the last error message set by peacock_execute(), or an empty string.
/// The returned pointer is valid until the next call on this executor.
const char* peacock_last_error(peacock_executor_t* executor);

// ---------------------------------------------------------------------------
// Node-by-node execution (unified CPU/GPU node-executor interface)
//
// The Rust orchestrator drives ONE plan node at a time: load the plan once, then
// call peacock_executor_execute_node per node (canonical post-order) with the child
// output handles. Intermediates stay GPU-resident behind handles; Arrow IPC crosses
// the boundary only at peacock_result_from_handle — once, at the root, for that walk,
// and once per unloaded batch for a driver using the three per-batch entry points
// below (a scan read per row-group subset, a sliced handle, a ranged export).
// ---------------------------------------------------------------------------

/// Actual per-node costs. Rust applies the shared ColAccum overhead (validity +
/// fixed-width + var-length offset buffers, from rows+schema) and adds
/// varlen_content_bytes — keeping the byte-accounting formula single-sourced in Rust.
typedef struct PeacockNodeStats {
  uint64_t rows;
  uint64_t varlen_content_bytes;
  /// Host microseconds this output partition spent before its first device touch:
  /// flatbuffer decode, handle lookups, AST construction — the peacockdb-only prologue
  /// bare cuDF has no analogue for, which is why it is separate. 0 unless
  /// peacock_set_node_timing() enabled a mode; a node's total is the Σ over partitions.
  uint64_t host_setup_us;
  /// Host microseconds from the first device touch to the end of the region. The two
  /// modes are NOT comparable here.
  ///
  /// Under EVENTS the region ends without an explicit drain and the device work is
  /// reported by peacock_executor_collect_node_times. Not launch cost, though: cuDF
  /// returns owned columns and rmm frees them, both synchronizing internally, so this
  /// tracks device time closely (tpch q3: within 0.01% — see test_node_timing). Under
  /// SYNC the region ends in a stream sync and contains the device execution outright.
  uint64_t host_submit_us;
  /// The same total Rust derives from rows+schema, recomputed here from cuDF types.
  /// To be COMPARED, not consumed: a difference means the two ends of the calibration
  /// are not counting bytes the same way.
  uint64_t logical_bytes;
  /// 1 when the partition's columns are one for one the types `output_schema` declares.
  /// Scopes the comparison above, which is only askable where both ends look at the
  /// same columns — a few nodes legitimately carry a different shape (grouping-set AVG
  /// state, `__grouping_id`, a union branch's decimal literal). Compare bytes only at 1.
  uint64_t schema_faithful;
} PeacockNodeStats;

/// What peacock_install_rmm_pool() did. Values, not a bitfield.
enum {
  /// A pool is the current device resource.
  PEACOCK_RMM_POOL_INSTALLED = 0,
  /// The pool could not be built: the default resource, NOT on purpose.
  PEACOCK_RMM_POOL_UNAVAILABLE = 1
};

/// Outcome of installing the pooled device allocator — sizes in bytes, 0 unless
/// `state` is PEACOCK_RMM_POOL_INSTALLED.
typedef struct PeacockRmmPoolInfo {
  int32_t state;       ///< one of PEACOCK_RMM_POOL_*
  int32_t integrated;  ///< 1 on an integrated part, which is sized differently
  uint64_t free_bytes;
  uint64_t initial_bytes;
  uint64_t maximum_bytes;
} PeacockRmmPoolInfo;

/// Install rmm's pooled device resource for the current device (process-global,
/// idempotent, NOT installed by default). Call before any GPU work.
///
/// Without it every cuDF intermediate is a cudaMalloc/cudaFree round trip. The gtest
/// binaries install the same pool from their main(); this entry point exists so
/// `peacock_gpu_benchmarks` — a Rust target, which cannot include
/// cpp/include/peacock/rmm_pool.hpp — measures under the same allocator rather than
/// producing numbers quietly compared with theirs.
///
/// The engine does not call this for itself, so a shipping query is unaffected; making
/// it self-installing is #148.
///
/// @param out_info  Filled with what actually happened. Required.
/// @return 0 unless out_info is NULL — NOT non-zero on UNAVAILABLE, which still
///         leaves a runnable default resource. Whether that is fatal is the
///         caller's call, and for anything being timed it is: a time taken with a
///         pool and one taken without differ by more than noise.
int peacock_install_rmm_pool(PeacockRmmPoolInfo* out_info);

/// Per-node timing modes for peacock_set_node_timing.
enum {
  /// The default, and the only mode a shipping query runs in.
  PEACOCK_NODE_TIMING_OFF = 0,
  /// Host clock around each region, closed by a default-stream sync. The sync
  /// serializes what cuDF would pipeline, so measuring changes what is measured; kept
  /// as the baseline events are checked against.
  PEACOCK_NODE_TIMING_SYNC = 1,
  /// CUDA events around the device work, host clock around the host work, no sync
  /// inside the region. Device times do not exist when a node returns — read them with
  /// peacock_executor_collect_node_times after the root materialize.
  PEACOCK_NODE_TIMING_EVENTS = 2
};

/// Select the per-node timing mode (process-global; OFF by default).
///
/// Opt-in because neither mode is free: SYNC drains the default stream at every
/// boundary, EVENTS allocates an event pair per region and holds it until collection.
///
/// @param mode one of PEACOCK_NODE_TIMING_*. Unknown values are treated as OFF.
void peacock_set_node_timing(int mode);

/// Emit NVTX ranges around plan nodes and their output partitions (process-global;
/// off by default). Independent of peacock_set_node_timing: a profiling run wants
/// the node boundaries without the event pairs, whose recording is device work a
/// capture would attribute to the node.
///
/// @param on nonzero to emit. Cheap but not free when on, and nothing reads the
///        ranges unless a profiler is attached.
void peacock_set_nvtx_ranges(int on);

/// One collected device interval: which node output partition it belongs to, and
/// what the device spent on it.
typedef struct PeacockNodeDeviceTime {
  uint64_t seq;
  uint64_t partition;
  uint64_t device_us;
} PeacockNodeDeviceTime;

/// Drain the device intervals recorded since the last call, in execution order.
///
/// Only PEACOCK_NODE_TIMING_EVENTS produces any. Separate from
/// peacock_executor_execute_node because the answer does not exist when a node returns
/// — the point of events. Call it after the root peacock_result_from_handle and before
/// peacock_executor_end_plan, which destroys the events.
///
/// What is returned is released, so two calls do not double-report. Regions with an
/// incomplete pair (a node that threw, or one that never touched the device) are absent
/// rather than zero; their host halves are still in PeacockNodeStats.
///
/// @param out      Caller array of `cap` entries.
/// @param out_count Number of regions RECORDED, not the number that fit — that is what
///                  the caller must size against. Exceeding `cap` FAILS the call (the
///                  first `cap` are still written) and the surplus is gone, the drain
///                  having happened: a truncated collection is indistinguishable from a
///                  device that did less work. Size `cap` as node count ×
///                  target_partitions and it cannot arise.
/// @return 0 on success, non-zero on failure (see peacock_last_error).
int peacock_executor_collect_node_times(peacock_executor_t* executor,
                                        PeacockNodeDeviceTime* out, uint64_t cap,
                                        uint64_t* out_count);

/// Cost of the measurement itself under PEACOCK_NODE_TIMING_SYNC, in microseconds:
/// that mode's region around no work (clock reads + a sync of an idle stream).
///
/// A SYNC node time is real work plus one of these, so a node at or below it is not
/// cheap but unresolvable. Report it beside the node times; do not subtract it.
/// Measures SYNC even when the caller runs with events: it says what events bought.
///
/// Returns the second-smallest of `samples` (clamped to >= 2). Requires a live CUDA
/// context and no concurrent work on the default stream. Returns 0 on CUDA error.
uint64_t peacock_measure_timing_floor_us(unsigned samples);

/// Load a plan for node-by-node execution. Parses + verifies once and indexes
/// nodes in post-order. Replaces any previously loaded plan on this executor.
/// @param out_node_count  Set to the number of plan nodes (post-order 0..n-1).
/// @return 0 on success, non-zero on failure (see peacock_last_error).
int peacock_executor_begin_plan(peacock_executor_t* executor,
                                const uint8_t* plan_bytes, uint64_t plan_len,
                                uint64_t* out_node_count);

// FAILURE POLICY at the four doors below. The three that execute — execute_node,
// execute_scan_rowgroups, slice_handle — end the query once work has begun: the loaded plan
// goes, and every resident handle with it, which is what makes a release on the failure path
// a no-op and keeps a driver's holds equal to its releases. Their validation arms are the
// exception — a null out-param, no plan loaded, an empty row-group list — since those refuse
// before any work and leave the session as it was. peacock_result_from_handle never ends a
// query: it reads a handle and touches nothing.

/// Execute the node at post-order `seq` with already-resident child output handles,
/// storing each output partition as a new resident handle. A failure ends the query.
///
/// Each child contributes a VECTOR of partition handles: `input_handles` is the
/// flattened concatenation grouped by child, `input_child_counts[c]` is child c's
/// partition count, `n_children` the number of children. Output handles go to
/// `out_handles[0..*out_count]` (caller buffer of `out_cap`; the partition count is
/// bounded by target_partitions). `out_stats[0..*out_count]` is a caller array
/// filled PER PARTITION (parallel to out_handles) so Rust sums the ColAccum
/// overhead per partition: cost = Σ_p ColAccum(rows_p), NOT ColAccum(Σ rows).
/// @return 0 on success, non-zero on failure.
int peacock_executor_execute_node(peacock_executor_t* executor,
                                  uint64_t seq,
                                  const uint64_t* input_handles,
                                  const uint64_t* input_child_counts,
                                  uint64_t n_children,
                                  uint64_t* out_handles, uint64_t out_cap,
                                  uint64_t* out_count,
                                  PeacockNodeStats* out_stats);

/// Execute the CudfScan at post-order `seq` reading exactly `row_groups[0..n)` rather
/// than the list the plan node carries, and store its one output as a new resident
/// handle — one call per batch for the batch-partitioned loader. `n == 0` is refused
/// rather than read as "every group"; `out_stats` may be NULL; a failure ends the query.
///
/// Every OTHER field of the node still applies per call, and `limit` is the one that does
/// not compose: each call sets it as the reader's row cap, so B calls answer B x limit
/// rows, which is why a limit-carrying source is planned as one lane and one batch.
/// @return 0 on success, non-zero on failure (a `seq` that is not a scan, an empty list,
///         a row group the file does not have).
int peacock_executor_execute_scan_rowgroups(peacock_executor_t* executor, uint64_t seq,
                                            const uint32_t* row_groups, uint64_t n,
                                            uint64_t* out_handle, PeacockNodeStats* out_stats);

/// Copy rows [offset, offset+length) of `handle` into a new resident handle, CONSUMING
/// `handle` as every operation on a resident table does. Range semantics as
/// peacock_result_from_handle. Serves a mid-plan limit, whose kept rows feed further
/// GPU work and so must stay a handle rather than become a result. A failure ends the
/// query.
/// @return 0 on success, non-zero on failure.
int peacock_executor_slice_handle(peacock_executor_t* executor, uint64_t handle, uint64_t offset,
                                  uint64_t length, uint64_t* out_handle);

/// Materialize rows [offset, offset+length) of a resident handle to an Arrow IPC stream
/// (called once per handle, at root). `length == UINT64_MAX` means to the end, which is
/// what a caller wanting the whole table passes. An offset at or past the end, and any
/// other range naming no rows of a non-empty table, is an empty result → *out_ipc_len==0
/// and nothing to free; a range running past the end clamps to it rather than failing,
/// because a limit's fetch legitimately overruns the batch it straddles. An EMPTY table
/// still exports its schema, as whole-table callers have always received.
/// Caller frees *out_ipc with peacock_result_free(). Does NOT release the handle, and a
/// failure leaves the session standing.
/// @return 0 on success, non-zero on failure.
int peacock_result_from_handle(peacock_executor_t* executor, uint64_t handle, uint64_t offset,
                               uint64_t length, uint8_t** out_ipc, uint64_t* out_ipc_len);

/// Release a resident intermediate handle (idempotent).
void peacock_handle_release(peacock_executor_t* executor, uint64_t handle);

/// Drop the loaded plan and all remaining resident handles.
void peacock_executor_end_plan(peacock_executor_t* executor);

// ---------------------------------------------------------------------------
// Conformance hook (stateless; no executor needed). Computes the GPU
// Spark-murmur3 partition ids for `key_cols` of the Arrow table described by the
// C-Data Interface (`schema`/`array` are `ArrowSchema*`/`ArrowArray*`; a struct
// array = the table), so the live conformance test can assert the REAL GPU path
// == the REAL comet CPU helper over the SAME bytes. Writes up to `out_cap` ids
// into `out_pids` and sets `*out_n`.
/// @return 0 on success; non-zero on failure (message via peacock_last_error(NULL)).
int peacock_spark_partition_ids(const void* schema, const void* array,
                                const uint32_t* key_cols, uint64_t num_keys,
                                uint32_t num_partitions, uint32_t seed,
                                int32_t* out_pids, uint64_t out_cap, uint64_t* out_n);

#ifdef __cplusplus
} // extern "C"
#endif
