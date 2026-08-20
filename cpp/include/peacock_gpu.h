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
  /// Microseconds this OUTPUT PARTITION took; 0 unless peacock_set_node_timing(1)
  /// is in effect. A node's time is the Σ over its partitions.
  uint64_t time_us;
} PeacockNodeStats;

/// Turn per-node timing on/off (process-global; off by default).
///
/// Enabling it makes peacock_executor_execute_node synchronize the default stream at
/// every measurement boundary and fill PeacockNodeStats::time_us. That sync is what
/// makes the number real and also what makes it costly, so this is opt-in: correct
/// for a benchmark, wrong for production. Intended for peacock_gpu_benchmarks.
void peacock_set_node_timing(int enable);

/// Cost of the measurement itself, in microseconds: the timed region every node
/// pays, wrapped around no work (clock reads + a sync of an already-idle stream).
///
/// A node's time_us is real work PLUS one of these, so a node at or below this
/// number is not "cheap" — it is below what the method can resolve. Report it next
/// to the node times; do NOT subtract it from them.
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
