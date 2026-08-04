// Limit (LIMIT / OFFSET) -- slice rows [skip, skip + fetch).

#include "peacock/operators.h"
#include "peacock/expr.h"

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace peacock {

TableResult execute_limit(const fb::GpuLimit* limit, NodeInputs* in) {
  auto input = execute_node(limit->input(), in);
  auto tv = input.table->view();
  auto num_rows = tv.num_rows();

  auto skip = std::min(static_cast<cudf::size_type>(limit->skip()), num_rows);
  auto end = num_rows;
  if (limit->fetch() >= 0) {
    // skip + fetch, clamped to num_rows (skip is already <= num_rows).
    auto want = static_cast<int64_t>(skip) + limit->fetch();
    end = static_cast<cudf::size_type>(std::min<int64_t>(want, num_rows));
  }

  if (skip == 0 && end == num_rows) return std::move(input);

  std::vector<cudf::size_type> slice_indices{skip, end};
  auto sliced = cudf::slice(tv, slice_indices);
  auto result = std::make_unique<cudf::table>(sliced[0]);
  return {std::move(result), std::move(input.column_names)};
}


}  // namespace peacock
