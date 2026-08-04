// Split out of the former src/plan_executor.cpp monolith.
//
// GpuFilter -- apply a boolean predicate.

#include "peacock/operators.h"
#include "peacock/expr.h"

#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/table/table.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace peacock {

// ============================================================================
// GpuFilter — apply boolean predicate
// ============================================================================

TableResult execute_filter(const fb::GpuFilter* filter, NodeInputs* in) {
  auto input = execute_node(filter->input(), in);

  // AST fast path when the predicate has no LIKE / CASE / ScalarFunction nodes;
  // otherwise produce the bool mask via the column-producing evaluator.
  std::unique_ptr<cudf::column> mask;
  if (is_ast_able(filter->predicate(), input.table->view())) {
    ExprContext ctx;
    auto& predicate = build_expr(filter->predicate(), ctx);
    mask = cudf::compute_column(input.table->view(), predicate);
  } else {
    mask = build_column(filter->predicate(), input.table->view());
  }
  auto filtered = cudf::apply_boolean_mask(input.table->view(), mask->view());

  // Optional projection (set when the planner fused a downstream
  // ProjectionExec into the filter). Without this, all input columns survive
  // and downstream column indices are wrong by exactly the number of dropped
  // columns.
  if (filter->projection() && filter->projection()->size() > 0) {
    auto fv = filtered->view();
    std::vector<std::unique_ptr<cudf::column>> proj_cols;
    std::vector<std::string> proj_names;
    proj_cols.reserve(filter->projection()->size());
    proj_names.reserve(filter->projection()->size());
    for (auto idx : *filter->projection()) {
      proj_cols.push_back(std::make_unique<cudf::column>(fv.column(idx)));
      proj_names.push_back(input.column_names[idx]);
    }
    return {std::make_unique<cudf::table>(std::move(proj_cols)),
            std::move(proj_names)};
  }

  return {std::move(filtered), std::move(input.column_names)};
}


}  // namespace peacock
