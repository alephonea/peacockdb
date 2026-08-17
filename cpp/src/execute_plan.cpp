// The public entry points: execute_plan (the recursive production fast path) and
// varlen_content_bytes.

#include "peacock/expr.h"
#include "peacock/operators.h"
#include "plan_executor.h"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

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

uint64_t logical_size_from_table(const cudf::table_view& table, uint64_t varlen_content) {
  const uint64_t rows = static_cast<uint64_t>(table.num_rows());
  // Charged per column, nullable or not — mirroring the Rust formula, which adds it
  // from the row count alone and never looks at the null mask.
  const uint64_t bitmap = (rows + 7) / 8;
  uint64_t total = varlen_content;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    const auto id = table.column(i).type().id();
    uint64_t data = 0;
    switch (id) {
      // A bit per row, NOT cuDF's byte per row: this models the Rust number.
      case cudf::type_id::BOOL8: data = (rows + 7) / 8; break;
      case cudf::type_id::INT8:
      case cudf::type_id::UINT8: data = rows; break;
      case cudf::type_id::INT16:
      case cudf::type_id::UINT16: data = rows * 2; break;
      // TIMESTAMP_DAYS is Arrow Date32. Arrow Timestamp, which Rust widths at 8,
      // cannot reach the GPU at all — `convert_data_type` has no arm for it — so
      // the 4-byte reading is the only one available here.
      case cudf::type_id::INT32:
      case cudf::type_id::UINT32:
      case cudf::type_id::FLOAT32:
      case cudf::type_id::TIMESTAMP_DAYS: data = rows * 4; break;
      case cudf::type_id::INT64:
      case cudf::type_id::UINT64:
      case cudf::type_id::FLOAT64:
      case cudf::type_id::TIMESTAMP_MILLISECONDS: data = rows * 8; break;
      // Only the offset buffer is structural; the content arrived as varlen_content.
      // See the header on why 4-byte offsets are assumed.
      case cudf::type_id::STRING: data = (rows + 1) * 4; break;
      case cudf::type_id::DECIMAL128: data = rows * 16; break;
      default:
        throw std::runtime_error(
            "logical_size_from_table: unhandled cudf type id " +
            std::to_string(static_cast<int>(id)) +
            " — add a deterministic arm here AND in type_structural_size (Rust)");
    }
    // Diagnostic, off unless asked for. When the Rust cross-check in
    // `gpu_node_executor` disagrees, it can name the Arrow schema but not the types
    // cuDF actually materialized — and that gap is the whole finding (#154, #41).
    // Printing the device side turns "the totals differ by N" into "column i came back
    // as type T". Correlate with the assertion by (rows, varlen, total).
    static const bool log_cols = std::getenv("PEACOCK_LOG_LOGICAL_BYTES") != nullptr;
    if (log_cols) {
      std::fprintf(stderr,
                   "[logical_bytes] col=%d type_id=%d scale=%d rows=%llu bytes=%llu\n", i,
                   static_cast<int>(id), table.column(i).type().scale(),
                   static_cast<unsigned long long>(rows),
                   static_cast<unsigned long long>(bitmap + data));
    }
    total += bitmap + data;
  }
  return total;
}

// Does `table` carry, one for one, the types `declared` says it does? Scopes the
// byte cross-check — see `NodeStats::schema_faithful`. Deliberately a TYPE
// comparison and not a byte one: it must be able to disagree with
// `logical_size_from_table` rather than restate it.
//
// `fb_to_type_id` collapses Utf8/LargeUtf8/Utf8View onto one STRING, so a declared
// string matches any string column. That ambiguity is unresolvable on the device
// and is exactly what the byte comparison downstream is left to catch.
static bool types_match_declared(const cudf::table_view& table,
                                 const fb::Schema* declared) {
  if (!declared || !declared->fields()) return true;  // nothing to contradict
  const auto* fields = declared->fields();
  if (static_cast<size_t>(table.num_columns()) != fields->size()) return false;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    const auto* f = fields->Get(static_cast<flatbuffers::uoffset_t>(i));
    // An unmapped declared type is not a mismatch to report here — it means the
    // serializer emitted something this path never handles, and execution would
    // have thrown long before.
    cudf::type_id want;
    try {
      want = fb_to_type_id(f->data_type());
    } catch (const std::exception&) {
      return false;
    }
    if (table.column(i).type().id() != want) return false;
  }
  return true;
}

NodeStats node_stats_for(const TableResult& result, uint64_t host_setup_us,
                         uint64_t host_submit_us, const fb::Schema* declared) {
  const auto full = result.table->view();

  // Drop `__rowcount__` before costing. `execute_project` synthesizes that column
  // when the projection is empty (see project.cpp): cuDF cannot represent a
  // 0-column table with rows, but DataFusion's `output_schema` for such a node
  // genuinely has no fields. It is an artifact of the device representation, not a
  // logical column, and charging it puts this end of the byte axis above Rust's by
  // a whole column (tpcds q88/q90/q96).
  std::vector<cudf::column_view> logical_cols;
  logical_cols.reserve(full.num_columns());
  for (cudf::size_type i = 0; i < full.num_columns(); ++i) {
    if (i < static_cast<cudf::size_type>(result.column_names.size()) &&
        result.column_names[i] == "__rowcount__")
      continue;
    logical_cols.push_back(full.column(i));
  }
  const cudf::table_view logical{logical_cols};

  // Rows come from `full`: dropping the placeholder must not change the row count,
  // and an all-placeholder table's `logical` view reports 0 rows.
  const uint64_t varlen = varlen_content_bytes(logical);
  // Faithfulness is judged on `logical` too: for an empty projection the declared
  // schema has no fields and the placeholder is exactly what was just dropped, so
  // the two agree at zero columns rather than off by one.
  return NodeStats{static_cast<uint64_t>(full.num_rows()), varlen, host_setup_us,
                   host_submit_us, logical_size_from_table(logical, varlen),
                   types_match_declared(logical, declared) ? 1u : 0u};
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
