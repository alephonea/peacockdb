// GpuSort -- sort by expressions.

#include "peacock/operators.h"
#include "peacock/expr.h"

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>
#include <cudf/merge.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace peacock {

TableResult execute_sort(const fb::GpuSort* sort, NodeInputs* in) {
  auto input = execute_node(sort->input(), in);
  auto tv = input.table->view();

  if (!sort->exprs() || sort->exprs()->size() == 0)
    return std::move(input);

  // Build the key table and sort orders.
  std::vector<cudf::column_view> key_cols;
  std::vector<cudf::order> orders;
  std::vector<cudf::null_order> null_orders;
  // Owns columns materialised from expression sort keys (e.g. q89 sorts by
  // sum_sales - avg_monthly_sales); kept alive until after the gather.
  std::vector<std::unique_ptr<cudf::column>> owned_keys;

  for (flatbuffers::uoffset_t i = 0; i < sort->exprs()->size(); ++i) {
    auto* se = sort->exprs()->Get(i);
    auto* expr = se->expr();
    if (!expr)
      throw std::runtime_error("GpuSort: missing sort key expression");
    if (expr->node_type() == fb::ExprNode_ColumnRef) {
      auto idx = static_cast<cudf::size_type>(expr->node_as_ColumnRef()->index());
      key_cols.push_back(tv.column(idx));
    } else {
      owned_keys.push_back(build_column(expr, tv));
      key_cols.push_back(owned_keys.back()->view());
    }
    orders.push_back(se->asc() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    null_orders.push_back(se->nulls_first() ? cudf::null_order::BEFORE
                                            : cudf::null_order::AFTER);
  }

  cudf::table_view keys{key_cols};
  auto sorted_indices = cudf::sorted_order(keys, orders, null_orders);
  auto result = cudf::gather(tv, sorted_indices->view());

  // Apply fetch (LIMIT).
  if (sort->fetch() > 0) {
    auto n = std::min(static_cast<cudf::size_type>(sort->fetch()),
                      result->view().num_rows());
    std::vector<cudf::size_type> slice_indices{0, n};
    auto sliced = cudf::slice(result->view(), slice_indices);
    result = std::make_unique<cudf::table>(sliced[0]);
  }

  return {std::move(result), std::move(input.column_names)};
}


}  // namespace peacock
