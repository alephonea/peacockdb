#pragma once
// PRIVATE header -- deliberately under src/, NOT include/. CMakeLists ships
// include/ wholesale via install(DIRECTORY include/), so anything placed there
// becomes public API sitting next to the stable C FFI surface (peacock_gpu.h).
// These are internal executor guts and are not part of that contract.
//
// Expression building: the AST fast path and the column-producing path.
//
// plan_executor_internal.h is deliberately NOT folded in here. It is the narrow
// contract the Tier-1b host-only CPU tests compile against (binop_output_type,
// is_ast_able) and it carries the rationale for why those two are exposed at all.
// Including it keeps that contract -- and cpp/tests/cpu/test_executor.cpp --
// untouched by this split.

#include "peacock/plan_types.h"
#include "plan_executor_internal.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace peacock {

/// Owns all AST sub-expressions so references remain valid for cuDF.
struct ExprContext {
  std::vector<std::unique_ptr<cudf::ast::expression>> owned;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

  cudf::ast::expression& keep(std::unique_ptr<cudf::ast::expression> e) {
    owned.push_back(std::move(e));
    return *owned.back();
  }
};

// When non-null (join-filter context), a ColumnRef(i) in the expression is
// remapped to column_reference(col_map[i].index, LEFT|RIGHT) so a mixed
// semi/anti join's AST predicate can address its two conditional tables.
using JoinFilterColMap = flatbuffers::Vector<const fb::JoinFilterColumn*>;

// Default argument lives on the DECLARATION only -- repeating it on the definition
// is a hard error. (Already the pattern here; preserved through the move.)
cudf::ast::expression& build_expr(const fb::Expr* expr, ExprContext& ctx,
                                  const JoinFilterColMap* col_map = nullptr);

// Materialize an expression into a column (the non-AST path).
std::unique_ptr<cudf::column> build_column(const fb::Expr* expr,
                                           cudf::table_view const& table);

cudf::type_id fb_to_type_id(fb::DataType dt);

// binop_output_type and is_ast_able are declared by plan_executor_internal.h
// (included above) -- they are the host-only test contract, not repeated here.

}  // namespace peacock
