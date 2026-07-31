// Split out of the former src/plan_executor.cpp monolith.
//
// GpuScan -- Parquet reads, row-group selection, projection pushdown.

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

// ============================================================================
// GpuScan — read Parquet files
// ============================================================================

TableResult execute_scan(
    const fb::GpuScan* scan,
    const flatbuffers::Vector<uint32_t>* row_groups_override) {
  if (!scan->file_paths() || scan->file_paths()->size() == 0)
    throw std::runtime_error("GpuScan: no file paths");

  // Wire-format contract (see gpu_plan.fbs::GpuScan): every path must be
  // absolute. We reject anything else with a clear error rather than
  // resolving against an implicit root.
  std::vector<std::string> paths;
  paths.reserve(scan->file_paths()->size());
  for (auto* p : *scan->file_paths()) {
    auto s = p->str();
    if (s.empty() || s.front() != '/') {
      throw std::runtime_error(
          "GpuScan: file path must be absolute (got \"" + s + "\")");
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

  // Row-group pruning: decode ONLY the surviving groups the serializer computed
  // (same DataFusion PruningPredicate as the CPU path). One inner vector = single
  // source. Empty/absent => read all groups (no predicate / multi-file / #16).
  // A per-partition override (Inc1 RG→partition map) takes precedence over the
  // scan's single-partition `row_groups`; both name the SAME global RG indices, so
  // the GPU decodes exactly the groups the CPU oracle / golden generator read.
  const flatbuffers::Vector<uint32_t>* rg_src =
      row_groups_override ? row_groups_override : scan->row_groups();
  if (rg_src && rg_src->size() > 0) {
    std::vector<cudf::size_type> rgs;
    rgs.reserve(rg_src->size());
    for (auto rg : *rg_src) {
      rgs.push_back(static_cast<cudf::size_type>(rg));
    }
    opts.set_row_groups({std::move(rgs)});
  }

  auto result = cudf::io::read_parquet(opts);

  // Optional diagnostic (PEACOCK_LOG_SCAN_ROWS=1): how many rows the GPU scan
  // actually decoded + whether row-group pruning was applied. Evidence that a
  // clustered-predicate scan reads only the surviving groups (e.g. q6 lineitem ->
  // 983040, not 6001215). Off by default; no effect on results.
  if (std::getenv("PEACOCK_LOG_SCAN_ROWS")) {
    bool pruned = rg_src && rg_src->size() > 0;
    std::fprintf(stderr, "[PEACOCK_SCAN] %s rows=%ld row_groups=%s(%u)\n",
                 paths.empty() ? "?" : paths[0].c_str(),
                 static_cast<long>(result.tbl->num_rows()),
                 pruned ? "pruned" : "all",
                 pruned ? rg_src->size() : 0u);
  }

  // Use column names from the reader metadata.
  std::vector<std::string> col_names;
  for (auto& ci : result.metadata.schema_info) {
    col_names.push_back(ci.name);
  }

  // Widen narrow decimals to DECIMAL128. The cuDF parquet reader picks the
  // smallest fixed_point width that fits (decimal32/64 for small precision),
  // but DataFusion — and therefore our serialized literals and the CPU
  // ground-truth executor — represent every decimal as Decimal128. cuDF's
  // binary_operation rejects mixed fixed_point widths ("Unsupported operator
  // for these types"), so normalize the scan output to a uniform DECIMAL128
  // representation (scale preserved). This also subsumes the decimal64→128
  // widening the result-export path does for Arrow IPC.
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
