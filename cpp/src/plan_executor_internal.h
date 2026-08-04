#pragma once
//
// Internal declarations of pure, host-only helpers defined in src/expr.cpp.
// Exposed solely so the Tier-1b CPU unit tests can exercise the decimal-scale and
// AST-routing rules without a GPU (they run on every push, unlike the GPU result
// tests). NOT part of the stable FFI surface — do not depend on it outside tests.

#include "generated/gpu_plan_generated.h"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

namespace peacock {
namespace fb = peacock::plan;

// Output cuDF type for a binary op: BOOL8 for predicates; for decimal
// arithmetic the DataFusion-matching fixed_point result scale (ADD/SUB take
// min(scale), MUL adds scales, DIV subtracts); otherwise the lhs type.
cudf::data_type binop_output_type(fb::BinaryOp op, cudf::data_type lhs,
                                  cudf::data_type rhs);

// Whether an expression can be evaluated through the cuDF AST fast path (true)
// or must take the column-producing path (false). Routes decimal operands and
// un-inferrable / mismatched-type binary ops to the column path.
bool is_ast_able(const fb::Expr* expr, cudf::table_view const& table);

}  // namespace peacock
