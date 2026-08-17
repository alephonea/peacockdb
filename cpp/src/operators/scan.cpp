// CudfScan -- Parquet reads, row-group selection, projection pushdown.

#include "peacock/operators.h"
#include "peacock/expr.h"

#include <cudf/io/parquet.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace peacock {

TableResult execute_scan(const fb::CudfScan* scan,
                         cudf::host_span<const uint32_t> row_groups_override) {
  if (!scan->file_paths() || scan->file_paths()->size() == 0)
    throw std::runtime_error("CudfScan: no file paths");

  // Wire-format contract (see gpu_plan.fbs::CudfScan): every path must be
  // absolute. We reject anything else with a clear error rather than
  // resolving against an implicit root.
  std::vector<std::string> paths;
  paths.reserve(scan->file_paths()->size());
  for (auto* p : *scan->file_paths()) {
    auto s = p->str();
    if (s.empty() || s.front() != '/') {
      throw std::runtime_error(
          "CudfScan: file path must be absolute (got \"" + s + "\")");
    }
    paths.push_back(std::move(s));
  }

  // Build column name list from file_schema + projection.
  std::vector<std::string> all_names;
  if (scan->file_schema() && scan->file_schema()->fields()) {
    for (auto* f : *scan->file_schema()->fields()) {
      all_names.push_back(f->name()->str());
    }
  }

  std::vector<std::string> projected_names;
  if (scan->projection() && scan->projection()->size() > 0) {
    for (auto idx : *scan->projection()) {
      if (idx < all_names.size()) {
        projected_names.push_back(all_names[idx]);
      }
    }
  } else {
    projected_names = all_names;
  }

  auto opts = cudf::io::parquet_reader_options::builder(
                  cudf::io::source_info{paths})
                  .columns(projected_names)
                  .build();

  if (scan->limit() > 0) {
    opts.set_num_rows(static_cast<cudf::size_type>(scan->limit()));
  }

  // Row-group pruning: decode ONLY the groups the serializer computed (the same
  // DataFusion PruningPredicate the CPU path prunes with) unless this call overrides
  // them — with the RG→partition map, or one batch of the batch-partitioned loader.
  // An empty override defers to the node's list, and an empty list reads all (#16).
  std::vector<cudf::size_type> rgs;
  if (!row_groups_override.empty()) {
    rgs.reserve(row_groups_override.size());
    for (auto rg : row_groups_override) rgs.push_back(static_cast<cudf::size_type>(rg));
  } else if (auto* own = scan->row_groups()) {
    rgs.reserve(own->size());
    for (auto rg : *own) rgs.push_back(static_cast<cudf::size_type>(rg));
  }
  if (!rgs.empty()) {
    opts.set_row_groups({rgs});
  }

  // Everything above is flatbuffer decode and reader-option assembly. Note that the
  // "device" interval for a scan also contains host file I/O and parquet decode —
  // cuDF's reader does both behind one call, and nothing here can separate them.
  mark_device_start();
  auto result = cudf::io::read_parquet(opts);

  // Optional diagnostic (PEACOCK_LOG_SCAN_ROWS=1): rows actually decoded and
  // whether row-group pruning applied — evidence that a clustered-predicate scan
  // reads only surviving groups (q6 lineitem -> 983040, not 6001215).
  if (std::getenv("PEACOCK_LOG_SCAN_ROWS")) {
    std::fprintf(stderr, "[PEACOCK_SCAN] %s rows=%ld row_groups=%s(%zu)\n",
                 paths.empty() ? "?" : paths[0].c_str(), static_cast<long>(result.tbl->num_rows()),
                 rgs.empty() ? "all" : "pruned", rgs.size());
  }

  // Use column names from the reader metadata.
  std::vector<std::string> col_names;
  for (auto& ci : result.metadata.schema_info) {
    col_names.push_back(ci.name);
  }

  // Widen narrow decimals to DECIMAL128 (scale preserved). The cuDF parquet reader
  // picks the smallest fixed_point width that fits, but DataFusion — and so our
  // serialized literals and the CPU oracle — use Decimal128 throughout, and cuDF's
  // binary_operation rejects mixed fixed_point widths. Subsumes the decimal64→128
  // widening the Arrow IPC export path also does.
  auto cols = result.tbl->release();
  for (auto& c : cols) {
    auto id = c->type().id();
    if (id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64) {
      c = cudf::cast(c->view(),
                     cudf::data_type{cudf::type_id::DECIMAL128, c->type().scale()});
    }
  }

  return {std::make_unique<cudf::table>(std::move(cols)), std::move(col_names)};
}

}  // namespace peacock
