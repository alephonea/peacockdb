#include "peacock_gpu.h"

#include <cudf/null_mask.hpp>

#include <cstdlib>
#include <cstring>
#include <new>

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct peacock_executor {
  uint64_t memory_limit;
};

// ---------------------------------------------------------------------------
// Versioning
// ---------------------------------------------------------------------------

const char* peacock_gpu_version() {
  // Trivial cudf call to ensure libcudf.so appears as a dynamic dependency.
  (void)cudf::num_bitmask_words(0);
  return "0.1.0";
}

// ---------------------------------------------------------------------------
// Executor lifecycle
// ---------------------------------------------------------------------------

int peacock_executor_create(uint64_t gpu_memory_limit,
                            peacock_executor_t** out_executor) {
  if (!out_executor) return 1;

  auto* ex = new (std::nothrow) peacock_executor{gpu_memory_limit};
  if (!ex) return 1;

  *out_executor = ex;
  return 0;
}

void peacock_executor_destroy(peacock_executor_t* executor) {
  delete executor;
}

// ---------------------------------------------------------------------------
// Query execution  (stub — implemented in Phase 2)
// ---------------------------------------------------------------------------

int peacock_execute(peacock_executor_t* /*executor*/,
                    const uint8_t* /*plan_bytes*/,
                    uint64_t /*plan_len*/,
                    uint8_t** out_result_bytes,
                    uint64_t* out_result_len) {
  if (!out_result_bytes || !out_result_len) return 1;

  // Placeholder: return an empty byte buffer.
  *out_result_bytes = static_cast<uint8_t*>(std::malloc(0));
  *out_result_len = 0;
  return 0;
}

void peacock_result_free(uint8_t* result_bytes) {
  std::free(result_bytes);
}
