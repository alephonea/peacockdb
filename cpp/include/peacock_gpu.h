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
// Query execution  (placeholder — full interface defined in Phase 2)
// ---------------------------------------------------------------------------

/// Execute a serialised physical plan.
///
/// @param executor          Executor handle.
/// @param plan_bytes        Flatbuffer-encoded physical plan.
/// @param plan_len          Length of plan_bytes in bytes.
/// @param out_result_bytes  On success, set to a newly allocated buffer
///                          containing the result (Arrow IPC stream).
///                          Caller must free with peacock_result_free().
/// @param out_result_len    Set to the length of out_result_bytes.
/// @return                  0 on success, non-zero on failure.
int peacock_execute(peacock_executor_t* executor,
                    const uint8_t* plan_bytes,
                    uint64_t plan_len,
                    uint8_t** out_result_bytes,
                    uint64_t* out_result_len);

/// Free a result buffer returned by peacock_execute().
void peacock_result_free(uint8_t* result_bytes);

#ifdef __cplusplus
} // extern "C"
#endif
