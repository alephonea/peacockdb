// GpuProject -- column selection / renaming / computed columns.

#include "peacock/operators.h"
#include "peacock/expr.h"

#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace peacock {

TableResult execute_project(const fb::GpuProject* proj, NodeInputs* in) {
  auto input = execute_node(proj->input(), in);

  if (!proj->exprs() || proj->exprs()->size() == 0) {
    // Empty projection (DataFusion emits one feeding count(*) — it needs no
    // input columns, only the row count). A 0-column table would lose that
    // count, so emit a single non-null placeholder column of the input length;
    // count(*) reads column 0 as size − null_count and gets the right answer.
    auto n_rows = input.table->num_rows();
    cudf::numeric_scalar<int8_t> zero(0, true);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_column_from_scalar(zero, n_rows));
    std::vector<std::string> names{"__rowcount__"};
    return {std::make_unique<cudf::table>(std::move(columns)), std::move(names)};
  }

  auto tv = input.table->view();
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<std::string> names;

  for (flatbuffers::uoffset_t i = 0; i < proj->exprs()->size(); ++i) {
    auto* expr = proj->exprs()->Get(i);

    // Fast path: simple column reference → just copy the column view.
    if (expr->node_type() == fb::ExprNode_ColumnRef) {
      auto* col = expr->node_as_ColumnRef();
      auto idx = static_cast<cudf::size_type>(col->index());
      columns.push_back(std::make_unique<cudf::column>(tv.column(idx)));
    } else if (is_ast_able(expr, tv)) {
      // Pure AST expression: fuse via cudf::compute_column.
      ExprContext ctx;
      auto& ast = build_expr(expr, ctx);
      columns.push_back(cudf::compute_column(tv, ast));
    } else {
      // Contains LIKE / CASE / ScalarFunction — column-producing path.
      columns.push_back(build_column(expr, tv));
    }

    if (proj->aliases() && i < proj->aliases()->size()) {
      names.push_back(proj->aliases()->Get(i)->str());
    } else {
      names.push_back("col" + std::to_string(i));
    }
  }

  auto result = std::make_unique<cudf::table>(std::move(columns));
  return {std::move(result), std::move(names)};
}


}  // namespace peacock
