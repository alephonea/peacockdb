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
// the boundary only once, at peacock_result_from_handle (root).
// ---------------------------------------------------------------------------

/// Actual per-node costs. Rust applies the shared ColAccum overhead (validity +
/// fixed-width + var-length offset buffers, from rows+schema) and adds
/// varlen_content_bytes — keeping the byte-accounting formula single-sourced in Rust.
typedef struct PeacockNodeStats {
  uint64_t rows;
  uint64_t varlen_content_bytes;
} PeacockNodeStats;

/// Load a plan for node-by-node execution. Parses + verifies once and indexes
/// nodes in post-order. Replaces any previously loaded plan on this executor.
/// @param out_node_count  Set to the number of plan nodes (post-order 0..n-1).
/// @return 0 on success, non-zero on failure (see peacock_last_error).
int peacock_executor_begin_plan(peacock_executor_t* executor,
                                const uint8_t* plan_bytes, uint64_t plan_len,
                                uint64_t* out_node_count);

/// Execute the node at post-order `seq` with already-resident child output handles,
/// storing each output partition as a new resident handle.
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

/// Materialize a resident handle to an Arrow IPC stream (called once, at root).
/// Caller frees *out_ipc with peacock_result_free(). Empty result → *out_ipc_len==0.
/// Does NOT release the handle.
/// @return 0 on success, non-zero on failure.
int peacock_result_from_handle(peacock_executor_t* executor, uint64_t handle,
                               uint8_t** out_ipc, uint64_t* out_ipc_len);

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
