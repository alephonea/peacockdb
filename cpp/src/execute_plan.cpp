// The public entry points: execute_plan (the recursive production fast path) and
// varlen_content_bytes.

#include "peacock/operators.h"
#include "plan_executor.h"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <stdexcept>
#include <string>

namespace peacock {


uint64_t varlen_content_bytes(const cudf::table_view& table) {
  uint64_t total = 0;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    auto col = table.column(i);
    // Flat string columns only — no nested List types reach here. Matches the Rust
    // ColAccum content term (Σ value byte lengths = offsets[n]-offsets[0]).
    if (col.type().id() == cudf::type_id::STRING) {
      total += static_cast<uint64_t>(
          cudf::strings_column_view(col).chars_size(cudf::get_default_stream()));
    }
  }
  return total;
}

TableResult execute_plan(const uint8_t* plan_bytes, uint64_t plan_len) {
  auto* gpu_plan = fb::GetGpuPlan(plan_bytes);
  if (!gpu_plan)
    throw std::runtime_error("failed to parse FlatBuffer GpuPlan");

  // Deeply nested plans (e.g. TPC-DS q8/q64) exceed the verifier's default
  // max_depth of 64; raise it to match the Rust serializer's VerifierOptions.
  flatbuffers::Verifier verifier(plan_bytes, plan_len, /*max_depth=*/1024);
  if (!gpu_plan->Verify(verifier))
    throw std::runtime_error("FlatBuffer verification failed");

  auto* root = gpu_plan->root();
  if (!root)
    throw std::runtime_error("GpuPlan has no root node");

  return execute_node(root, nullptr);
}


}  // namespace peacock
