#include "peacock_gpu.h"
#include "plan_executor.h"

#include <cudf/interop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/unary.hpp>

#include <arrow/buffer.h>
#include <arrow/c/bridge.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct peacock_executor {
  uint64_t memory_limit;
  std::string last_error;
  // Node-by-node session (unified node-executor interface). Holds the parsed
  // plan + resident intermediate handles for the current plan; null otherwise.
  std::unique_ptr<peacock::NodeSession> session;
};

// Export a cuDF table to an Arrow IPC stream buffer (malloc'd; free with
// peacock_result_free). Shared by peacock_execute (fast path) and
// peacock_result_from_handle (node-by-node root). Widens DECIMAL32/64→128 since
// the Rust arrow-ipc reader rejects narrow decimals.
static void export_table_to_ipc(const cudf::table_view& tview,
                                const std::vector<std::string>& column_names,
                                uint8_t** out_bytes, uint64_t* out_len) {
  std::vector<cudf::column_metadata> col_meta;
  col_meta.reserve(column_names.size());
  for (const auto& name : column_names) col_meta.push_back({name});

  std::vector<std::unique_ptr<cudf::column>> widened;
  std::vector<cudf::column_view> widened_views;
  widened_views.reserve(tview.num_columns());
  for (cudf::size_type i = 0; i < tview.num_columns(); ++i) {
    auto col = tview.column(i);
    auto t = col.type();
    if (t.id() == cudf::type_id::DECIMAL32 || t.id() == cudf::type_id::DECIMAL64) {
      auto w = cudf::cast(col, cudf::data_type{cudf::type_id::DECIMAL128, t.scale()});
      widened_views.push_back(w->view());
      widened.push_back(std::move(w));
    } else {
      widened_views.push_back(col);
    }
  }
  cudf::table_view export_view{widened_views};

  auto c_schema = cudf::to_arrow_schema(export_view, col_meta);
  auto schema = arrow::ImportSchema(c_schema.get()).ValueOrDie();
  auto c_array = cudf::to_arrow_host(export_view);
  auto batch = arrow::ImportRecordBatch(&c_array->array, schema).ValueOrDie();

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  auto writer = arrow::ipc::MakeStreamWriter(sink.get(), schema).ValueOrDie();
  auto st = writer->WriteRecordBatch(*batch);
  if (!st.ok()) throw std::runtime_error("IPC write: " + st.ToString());
  st = writer->Close();
  if (!st.ok()) throw std::runtime_error("IPC close: " + st.ToString());
  auto buffer = sink->Finish().ValueOrDie();

  *out_len = static_cast<uint64_t>(buffer->size());
  *out_bytes = static_cast<uint8_t*>(std::malloc(*out_len));
  if (!*out_bytes) throw std::runtime_error("malloc failed for result buffer");
  std::memcpy(*out_bytes, buffer->data(), *out_len);
}

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

  auto* ex = new (std::nothrow) peacock_executor{gpu_memory_limit, {}};
  if (!ex) return 1;

  *out_executor = ex;
  return 0;
}

void peacock_executor_destroy(peacock_executor_t* executor) {
  delete executor;
}

// ---------------------------------------------------------------------------
// Query execution
// ---------------------------------------------------------------------------

int peacock_execute(peacock_executor_t* executor,
                    const uint8_t* plan_bytes,
                    uint64_t plan_len,
                    uint8_t** out_result_bytes,
                    uint64_t* out_result_len) {
  if (!executor || !plan_bytes || !out_result_bytes || !out_result_len)
    return 1;

  try {
    auto result = peacock::execute_plan(plan_bytes, plan_len);
    export_table_to_ipc(result.table->view(), result.column_names, out_result_bytes,
                        out_result_len);
    return 0;
  } catch (const std::exception& e) {
    executor->last_error = e.what();
    std::fprintf(stderr, "[peacock_execute] error: %s\n", e.what());
    return 1;
  } catch (...) {
    executor->last_error = "unknown exception";
    std::fprintf(stderr, "[peacock_execute] unknown exception\n");
    return 1;
  }
}

void peacock_result_free(uint8_t* result_bytes) {
  std::free(result_bytes);
}

const char* peacock_last_error(peacock_executor_t* executor) {
  if (!executor) return "";
  return executor->last_error.c_str();
}

// ---------------------------------------------------------------------------
// Node-by-node execution (unified CPU/GPU node-executor interface)
// ---------------------------------------------------------------------------

int peacock_executor_begin_plan(peacock_executor_t* executor,
                                const uint8_t* plan_bytes, uint64_t plan_len,
                                uint64_t* out_node_count) {
  if (!executor || !plan_bytes || !out_node_count) return 1;
  try {
    executor->session = std::make_unique<peacock::NodeSession>(plan_bytes, plan_len);
    *out_node_count = static_cast<uint64_t>(executor->session->node_count());
    return 0;
  } catch (const std::exception& e) {
    executor->last_error = e.what();
    return 1;
  } catch (...) {
    executor->last_error = "unknown exception";
    return 1;
  }
}

int peacock_executor_execute_node(peacock_executor_t* executor, uint64_t seq,
                                  const uint64_t* input_handles, uint64_t n_inputs,
                                  uint64_t* out_handle, PeacockNodeStats* out_stats) {
  if (!executor || !out_handle || (n_inputs > 0 && !input_handles)) return 1;
  if (!executor->session) {
    executor->last_error = "no plan loaded (call peacock_executor_begin_plan first)";
    return 1;
  }
  try {
    peacock::NodeStats stats;
    *out_handle = executor->session->execute_node(seq, input_handles,
                                                  static_cast<size_t>(n_inputs), &stats);
    if (out_stats) {
      out_stats->rows = stats.rows;
      out_stats->varlen_content_bytes = stats.varlen_content_bytes;
    }
    return 0;
  } catch (const std::exception& e) {
    executor->last_error = e.what();
    // On failure mid-walk, drop the whole plan + all resident intermediates.
    executor->session.reset();
    return 1;
  } catch (...) {
    executor->last_error = "unknown exception";
    executor->session.reset();
    return 1;
  }
}

int peacock_result_from_handle(peacock_executor_t* executor, uint64_t handle,
                               uint8_t** out_ipc, uint64_t* out_ipc_len) {
  if (!executor || !out_ipc || !out_ipc_len) return 1;
  if (!executor->session) {
    executor->last_error = "no plan loaded";
    return 1;
  }
  try {
    const auto& result = executor->session->table_for(handle);
    export_table_to_ipc(result.table->view(), result.column_names, out_ipc, out_ipc_len);
    return 0;
  } catch (const std::exception& e) {
    executor->last_error = e.what();
    return 1;
  } catch (...) {
    executor->last_error = "unknown exception";
    return 1;
  }
}

void peacock_handle_release(peacock_executor_t* executor, uint64_t handle) {
  if (executor && executor->session) executor->session->release(handle);
}

void peacock_executor_end_plan(peacock_executor_t* executor) {
  if (executor) executor->session.reset();
}