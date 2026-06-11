#include "plan_executor.h"
#include "plan_executor_internal.h"
#include "generated/gpu_plan_generated.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/datetime.hpp>
#include <cudf/filling.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#if __has_include(<cudf/join/join.hpp>)
#include <cudf/join/join.hpp>
#else
#include <cudf/join.hpp>
#endif
#if __has_include(<cudf/join/filtered_join.hpp>)
#include <cudf/join/filtered_join.hpp>
#define PEACOCK_HAVE_FILTERED_JOIN 1
#endif
// cuDF 26.02 moved the mixed (equality + AST-conditional) join functions out of
// the monolithic <cudf/join.hpp> into their own header; older versions declare
// them in <cudf/join.hpp> (included above). Pull in the split header when present.
#if __has_include(<cudf/join/mixed_join.hpp>)
#include <cudf/join/mixed_join.hpp>
#endif
// Likewise, 26.02 split the conditional (pure-AST) join functions
// (conditional_inner_join / conditional_left_join, used by GpuNestedLoopJoin)
// into their own header; older versions declare them in <cudf/join.hpp>.
#if __has_include(<cudf/join/conditional_join.hpp>)
#include <cudf/join/conditional_join.hpp>
#endif
#include <cudf/reduction.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/unary.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <cuda_runtime.h>
#include <rmm/cuda_stream_view.hpp>

namespace peacock {
namespace fb = peacock::plan;

// ---------------------------------------------------------------------------
// Debug instrumentation (PEACOCK_GPU_DEBUG=1 to enable)
// ---------------------------------------------------------------------------
// Prints each plan node + expression as it executes and synchronizes the
// default cuDF stream after each step so async CUDA errors surface at the
// call site instead of cascading several ops later.

static bool debug_enabled() {
  static const bool e = []() {
    const char* v = std::getenv("PEACOCK_GPU_DEBUG");
    return v && v[0] && v[0] != '0';
  }();
  return e;
}

#define PCK_TRACE(...) do {                                  \
    if (debug_enabled()) {                                   \
      std::fprintf(stderr, "[peacock] " __VA_ARGS__);        \
      std::fprintf(stderr, "\n");                            \
    }                                                        \
  } while (0)

// Synchronize the default stream and check for errors. When debug is on,
// we always sync (to localize errors); when off, this is a no-op.
static void debug_sync(const char* tag) {
  if (!debug_enabled()) return;
  auto err = cudaStreamSynchronize(cudf::get_default_stream().value());
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[peacock] CUDA sync after %s: %s\n",
                 tag, cudaGetErrorString(err));
    throw std::runtime_error(std::string("CUDA error after ") + tag +
                             ": " + cudaGetErrorString(err));
  }
}

// ============================================================================
// Expression evaluation context
// ============================================================================

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

// Forward declarations
static TableResult execute_node(const fb::PlanNode* node);
static cudf::ast::expression& build_expr(const fb::Expr* expr, ExprContext& ctx,
                                         const JoinFilterColMap* col_map = nullptr);
static bool is_predicate_op(fb::BinaryOp op);
static cudf::type_id infer_expr_type(const fb::Expr* expr,
                                     cudf::table_view const& table);

// ============================================================================
// FlatBuffer DataType → cuDF type_id
// ============================================================================

static cudf::type_id fb_to_type_id(fb::DataType dt) {
  switch (dt) {
    case fb::DataType_Boolean:    return cudf::type_id::BOOL8;
    case fb::DataType_Int8:       return cudf::type_id::INT8;
    case fb::DataType_Int16:      return cudf::type_id::INT16;
    case fb::DataType_Int32:      return cudf::type_id::INT32;
    case fb::DataType_Int64:      return cudf::type_id::INT64;
    case fb::DataType_UInt8:      return cudf::type_id::UINT8;
    case fb::DataType_UInt16:     return cudf::type_id::UINT16;
    case fb::DataType_UInt32:     return cudf::type_id::UINT32;
    case fb::DataType_UInt64:     return cudf::type_id::UINT64;
    case fb::DataType_Float32:    return cudf::type_id::FLOAT32;
    case fb::DataType_Float64:    return cudf::type_id::FLOAT64;
    case fb::DataType_Utf8:
    case fb::DataType_LargeUtf8:
    case fb::DataType_Utf8View:   return cudf::type_id::STRING;
    case fb::DataType_Date32:     return cudf::type_id::TIMESTAMP_DAYS;
    case fb::DataType_Date64:     return cudf::type_id::TIMESTAMP_MILLISECONDS;
    case fb::DataType_Decimal128: return cudf::type_id::DECIMAL128;
    default:                      return cudf::type_id::EMPTY;
  }
}

// ============================================================================
// FlatBuffer BinaryOp → cuDF AST operator
// ============================================================================

static cudf::ast::ast_operator fb_to_ast_op(fb::BinaryOp op) {
  switch (op) {
    case fb::BinaryOp_Eq:     return cudf::ast::ast_operator::EQUAL;
    case fb::BinaryOp_NotEq:  return cudf::ast::ast_operator::NOT_EQUAL;
    case fb::BinaryOp_Lt:     return cudf::ast::ast_operator::LESS;
    case fb::BinaryOp_LtEq:   return cudf::ast::ast_operator::LESS_EQUAL;
    case fb::BinaryOp_Gt:     return cudf::ast::ast_operator::GREATER;
    case fb::BinaryOp_GtEq:   return cudf::ast::ast_operator::GREATER_EQUAL;
    case fb::BinaryOp_Plus:   return cudf::ast::ast_operator::ADD;
    case fb::BinaryOp_Minus:  return cudf::ast::ast_operator::SUB;
    case fb::BinaryOp_Multiply: return cudf::ast::ast_operator::MUL;
    case fb::BinaryOp_Divide: return cudf::ast::ast_operator::DIV;
    case fb::BinaryOp_Modulo: return cudf::ast::ast_operator::MOD;
    // SQL three-valued logic: `TRUE OR NULL` is TRUE and `FALSE AND NULL` is
    // FALSE, whereas cuDF's plain LOGICAL_AND/OR propagate the null (so
    // `TRUE OR NULL` → NULL). For a top-level filter, AND is unaffected (a NULL
    // mask and a FALSE mask both drop the row), but plain OR silently DROPS rows
    // a disjunctive predicate should keep — e.g. `p_channel_email = 'N' OR
    // p_channel_event = 'N'` when one channel is NULL (TPC-DS q7/q15/q26/q79).
    // NULL_LOGICAL_* match the peacock CPU oracle's SQL semantics and reduce to
    // LOGICAL_* when both operands are non-null, so non-null predicates (the
    // whole passing suite) are unchanged.
    case fb::BinaryOp_And:    return cudf::ast::ast_operator::NULL_LOGICAL_AND;
    case fb::BinaryOp_Or:     return cudf::ast::ast_operator::NULL_LOGICAL_OR;
    case fb::BinaryOp_BitwiseAnd: return cudf::ast::ast_operator::BITWISE_AND;
    case fb::BinaryOp_BitwiseOr:  return cudf::ast::ast_operator::BITWISE_OR;
    case fb::BinaryOp_BitwiseXor: return cudf::ast::ast_operator::BITWISE_XOR;
    default:
      throw std::runtime_error("unsupported BinaryOp: " + std::to_string(op));
  }
}

// ============================================================================
// AST expression builder
// ============================================================================

static cudf::ast::expression& build_expr(const fb::Expr* expr, ExprContext& ctx,
                                         const JoinFilterColMap* col_map) {
  if (!expr || !expr->node())
    throw std::runtime_error("null expression");

  switch (expr->node_type()) {
    case fb::ExprNode_ColumnRef: {
      auto* col = expr->node_as_ColumnRef();
      if (col_map) {
        // Join-filter predicate: ColumnRef(i) indexes the filter's intermediate
        // schema; remap to (side, index) so the AST addresses the mixed join's
        // left/right conditional tables directly.
        if (col->index() >= col_map->size())
          throw std::runtime_error("join filter ColumnRef out of range of filter_columns");
        auto* fc = col_map->Get(col->index());
        auto side = fc->side() == fb::JoinSide_Right
                        ? cudf::ast::table_reference::RIGHT
                        : cudf::ast::table_reference::LEFT;
        return ctx.keep(std::make_unique<cudf::ast::column_reference>(
            static_cast<cudf::size_type>(fc->index()), side));
      }
      return ctx.keep(std::make_unique<cudf::ast::column_reference>(
          static_cast<cudf::size_type>(col->index())));
    }

    case fb::ExprNode_LiteralExpr: {
      auto* lit = expr->node_as_LiteralExpr();
      auto* sv = lit->value();
      if (!sv) throw std::runtime_error("LiteralExpr has no value");

      switch (sv->type()) {
        case fb::DataType_Boolean: {
          auto s = std::make_unique<cudf::numeric_scalar<bool>>(
              sv->bool_val(), true);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Int8: {
          auto s = std::make_unique<cudf::numeric_scalar<int8_t>>(
              static_cast<int8_t>(sv->int_val()), true);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Int16: {
          auto s = std::make_unique<cudf::numeric_scalar<int16_t>>(
              static_cast<int16_t>(sv->int_val()), true);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Int32: {
          auto s = std::make_unique<cudf::numeric_scalar<int32_t>>(
              static_cast<int32_t>(sv->int_val()), true);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Int64: {
          auto s = std::make_unique<cudf::numeric_scalar<int64_t>>(
              sv->int_val(), true);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Float32: {
          auto s = std::make_unique<cudf::numeric_scalar<float>>(
              static_cast<float>(sv->float_val()), true);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Float64: {
          auto s = std::make_unique<cudf::numeric_scalar<double>>(
              sv->float_val(), true);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Decimal128: {
          // cuDF AST does not directly support Decimal128 literals.
          // Promote to float64 for comparison.
          __int128 val = (static_cast<__int128>(sv->decimal_hi()) << 64) |
                         static_cast<unsigned __int128>(sv->decimal_lo());
          int8_t scale = sv->decimal_scale();
          double dval = static_cast<double>(val);
          for (int8_t i = 0; i < scale; ++i) dval /= 10.0;
          for (int8_t i = 0; i > scale; --i) dval *= 10.0;
          auto s = std::make_unique<cudf::numeric_scalar<double>>(dval, true);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Utf8:
        case fb::DataType_LargeUtf8:
        case fb::DataType_Utf8View: {
          // cuDF AST literals accept string_scalar; cuDF doesn't distinguish
          // owned vs. view strings on the device side, so all three flavors
          // map to the same scalar type.
          auto s = std::make_unique<cudf::string_scalar>(
              std::string(sv->string_val() ? sv->string_val()->str() : ""), true);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Date32: {
          // Date32 = days since UNIX epoch (int32).
          auto s = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
              cudf::duration_D{static_cast<int32_t>(sv->int_val())}, true);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        default:
          throw std::runtime_error(
              "unsupported literal type: " + std::to_string(sv->type()));
      }
    }

    case fb::ExprNode_BinaryExprNode: {
      auto* bin = expr->node_as_BinaryExprNode();
      auto& left = build_expr(bin->left(), ctx, col_map);
      auto& right = build_expr(bin->right(), ctx, col_map);
      auto op = fb_to_ast_op(bin->op());
      return ctx.keep(std::make_unique<cudf::ast::operation>(op, left, right));
    }

    case fb::ExprNode_UnaryExprNode: {
      auto* un = expr->node_as_UnaryExprNode();
      auto& arg = build_expr(un->arg(), ctx, col_map);
      switch (un->op()) {
        case fb::UnaryOp_Not:
          return ctx.keep(std::make_unique<cudf::ast::operation>(
              cudf::ast::ast_operator::NOT, arg));
        case fb::UnaryOp_IsNull:
          return ctx.keep(std::make_unique<cudf::ast::operation>(
              cudf::ast::ast_operator::IS_NULL, arg));
        case fb::UnaryOp_IsNotNull: {
          // cuDF AST has no IS_NOT_NULL; compose as NOT(IS_NULL(arg)).
          auto& is_null = ctx.keep(std::make_unique<cudf::ast::operation>(
              cudf::ast::ast_operator::IS_NULL, arg));
          return ctx.keep(std::make_unique<cudf::ast::operation>(
              cudf::ast::ast_operator::NOT, is_null));
        }
        case fb::UnaryOp_Negative: {
          // -x = 0 - x
          auto zero = std::make_unique<cudf::numeric_scalar<int64_t>>(0, true);
          auto& zref = *zero;
          ctx.scalars.push_back(std::move(zero));
          auto& lit = ctx.keep(std::make_unique<cudf::ast::literal>(zref));
          return ctx.keep(std::make_unique<cudf::ast::operation>(
              cudf::ast::ast_operator::SUB, lit, arg));
        }
        default:
          throw std::runtime_error(
              "unsupported UnaryOp: " + std::to_string(un->op()));
      }
    }

    case fb::ExprNode_CastExprNode: {
      auto* cast = expr->node_as_CastExprNode();
      auto& inner = build_expr(cast->expr(), ctx, col_map);
      auto target = fb_to_type_id(cast->target_type());
      cudf::ast::ast_operator cast_op;
      switch (target) {
        case cudf::type_id::INT64:  cast_op = cudf::ast::ast_operator::CAST_TO_INT64; break;
        case cudf::type_id::FLOAT64: cast_op = cudf::ast::ast_operator::CAST_TO_FLOAT64; break;
        default:
          throw std::runtime_error("unsupported CAST target type");
      }
      return ctx.keep(std::make_unique<cudf::ast::operation>(cast_op, inner));
    }

    default:
      throw std::runtime_error(
          "unsupported expression node type: " + std::to_string(expr->node_type()));
  }
}

// ============================================================================
// Column-producing expression evaluator
// ============================================================================
//
// cuDF AST has no operators for LIKE, substr, date_part (extract), or CASE
// WHEN. Expressions that contain any of these nodes are evaluated outside the
// AST: each subexpression produces a `cudf::column`, which we combine via
// cudf row-wise APIs (binary_operation, copy_if_else, strings::like, ...).
//
// AST-able subtrees still go through `compute_column` for fusion; the
// column-path is a recursive fallback that calls into the AST evaluator
// whenever it encounters a fully AST-able subexpression.

// String/binary literal types whose AST evaluation isn't supported by cuDF
// (compute_column allocates fixed-width output, so a string compare aborts).
static bool is_string_like_literal(const fb::Expr* expr) {
  if (expr->node_type() != fb::ExprNode_LiteralExpr) return false;
  auto* sv = expr->node_as_LiteralExpr()->value();
  if (!sv) return false;
  switch (sv->type()) {
    case fb::DataType_Utf8:
    case fb::DataType_LargeUtf8:
    case fb::DataType_Utf8View:
    case fb::DataType_Binary:
    case fb::DataType_LargeBinary:
    case fb::DataType_BinaryView:
      return true;
    default:
      return false;
  }
}

// Best-effort static type of an expression, resolved against the input table
// (column refs need the schema). Used only to decide AST-ability; returns
// EMPTY for shapes we can't infer, which callers treat conservatively.
static cudf::type_id infer_expr_type(const fb::Expr* expr,
                                     cudf::table_view const& table) {
  switch (expr->node_type()) {
    case fb::ExprNode_ColumnRef: {
      auto idx = static_cast<cudf::size_type>(
          expr->node_as_ColumnRef()->index());
      if (idx < 0 || idx >= table.num_columns()) return cudf::type_id::EMPTY;
      return table.column(idx).type().id();
    }
    case fb::ExprNode_LiteralExpr: {
      auto* sv = expr->node_as_LiteralExpr()->value();
      return sv ? fb_to_type_id(sv->type()) : cudf::type_id::EMPTY;
    }
    case fb::ExprNode_BinaryExprNode: {
      auto* b = expr->node_as_BinaryExprNode();
      if (is_predicate_op(b->op())) return cudf::type_id::BOOL8;
      auto lt = infer_expr_type(b->left(), table);
      auto rt = infer_expr_type(b->right(), table);
      auto is_float = [](cudf::type_id t) {
        return t == cudf::type_id::FLOAT32 || t == cudf::type_id::FLOAT64;
      };
      if (is_float(lt) || is_float(rt)) return cudf::type_id::FLOAT64;
      if (lt == cudf::type_id::DECIMAL128 || rt == cudf::type_id::DECIMAL128)
        return cudf::type_id::DECIMAL128;
      return lt;
    }
    case fb::ExprNode_UnaryExprNode: {
      auto* u = expr->node_as_UnaryExprNode();
      switch (u->op()) {
        case fb::UnaryOp_Not:
        case fb::UnaryOp_IsNull:
        case fb::UnaryOp_IsNotNull:
          return cudf::type_id::BOOL8;
        default:  // Negative
          return infer_expr_type(u->arg(), table);
      }
    }
    case fb::ExprNode_CastExprNode:
      return fb_to_type_id(expr->node_as_CastExprNode()->target_type());
    case fb::ExprNode_LikeExprNode:
      return cudf::type_id::BOOL8;
    case fb::ExprNode_CaseExprNode: {
      auto* c = expr->node_as_CaseExprNode();
      if (c->when_thens() && c->when_thens()->size() > 0)
        return infer_expr_type(c->when_thens()->Get(0)->then(), table);
      return cudf::type_id::EMPTY;
    }
    case fb::ExprNode_ScalarFunctionExprNode:
      return fb_to_type_id(
          expr->node_as_ScalarFunctionExprNode()->return_type());
    default:
      return cudf::type_id::EMPTY;
  }
}

bool is_ast_able(const fb::Expr* expr, cudf::table_view const& table) {
  switch (expr->node_type()) {
    case fb::ExprNode_LikeExprNode:
    case fb::ExprNode_CaseExprNode:
    case fb::ExprNode_ScalarFunctionExprNode:
      return false;
    case fb::ExprNode_LiteralExpr:
      // A standalone string/binary literal can't go through compute_column
      // (it allocates fixed-width output); route it to build_column, which
      // broadcasts the scalar. Numeric literals stay AST-able. Literals inside
      // binary ops are classified by the BinaryExprNode arm below, not here.
      return !is_string_like_literal(expr);
    case fb::ExprNode_BinaryExprNode: {
      auto* b = expr->node_as_BinaryExprNode();
      // cuDF AST has no string ops; string literal on either side forces the
      // column path (cudf::binary_operation, which does support strings).
      if (is_string_like_literal(b->left()) || is_string_like_literal(b->right()))
        return false;
      // cuDF AST requires both operands to be the identical type and has no
      // decimal support at all (it only auto-promotes nothing — unlike the
      // binaryop API). DataFusion emits matched decimal operands and same-type
      // comparisons, but our AST literal path promotes decimal literals to
      // float64, and int widths can differ. Any decimal operand, or a type
      // mismatch between the two sides, routes to the column path, where
      // cudf::binary_operation coerces (and handles fixed_point) natively.
      auto lt = infer_expr_type(b->left(), table);
      auto rt = infer_expr_type(b->right(), table);
      // Un-inferrable operand → route to the column path conservatively. (Two
      // EMPTYs would otherwise compare equal below and be treated as AST-able,
      // the opposite of the intended fallback.)
      if (lt == cudf::type_id::EMPTY || rt == cudf::type_id::EMPTY)
        return false;
      if (lt == cudf::type_id::DECIMAL128 || rt == cudf::type_id::DECIMAL128)
        return false;
      if (lt != rt)
        return false;
      return is_ast_able(b->left(), table) && is_ast_able(b->right(), table);
    }
    case fb::ExprNode_UnaryExprNode:
      return is_ast_able(expr->node_as_UnaryExprNode()->arg(), table);
    case fb::ExprNode_CastExprNode: {
      // cuDF AST only has CAST_TO_INT64 / CAST_TO_FLOAT64. Any other target
      // (notably Decimal128) must go through the column path, which uses
      // cudf::cast.
      auto target = fb_to_type_id(expr->node_as_CastExprNode()->target_type());
      if (target != cudf::type_id::INT64 && target != cudf::type_id::FLOAT64)
        return false;
      return is_ast_able(expr->node_as_CastExprNode()->expr(), table);
    }
    default:
      return true;
  }
}

static std::unique_ptr<cudf::scalar> build_scalar(const fb::ScalarValue* sv) {
  // A typed NULL literal is encoded with is_null set; the value fields are
  // unused. Each scalar is built invalid so cuDF treats it as null of `type`.
  bool valid = !sv->is_null();
  switch (sv->type()) {
    case fb::DataType_Boolean:
      return std::make_unique<cudf::numeric_scalar<bool>>(sv->bool_val(), valid);
    case fb::DataType_Int8:
      return std::make_unique<cudf::numeric_scalar<int8_t>>(
          static_cast<int8_t>(sv->int_val()), valid);
    case fb::DataType_Int16:
      return std::make_unique<cudf::numeric_scalar<int16_t>>(
          static_cast<int16_t>(sv->int_val()), valid);
    case fb::DataType_Int32:
      return std::make_unique<cudf::numeric_scalar<int32_t>>(
          static_cast<int32_t>(sv->int_val()), valid);
    case fb::DataType_Int64:
      return std::make_unique<cudf::numeric_scalar<int64_t>>(sv->int_val(), valid);
    case fb::DataType_Float32:
      return std::make_unique<cudf::numeric_scalar<float>>(
          static_cast<float>(sv->float_val()), valid);
    case fb::DataType_Float64:
      return std::make_unique<cudf::numeric_scalar<double>>(sv->float_val(), valid);
    case fb::DataType_Utf8:
    case fb::DataType_LargeUtf8:
    case fb::DataType_Utf8View:
      return std::make_unique<cudf::string_scalar>(
          std::string(sv->string_val() ? sv->string_val()->str() : ""), valid);
    case fb::DataType_Date32:
      return std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
          cudf::duration_D{static_cast<int32_t>(sv->int_val())}, valid);
    case fb::DataType_Decimal128: {
      // Reassemble the 128-bit value; Arrow scale (fractional digits, positive)
      // negates to cuDF's base-10 exponent.
      __int128 val = (static_cast<__int128>(sv->decimal_hi()) << 64) |
                     static_cast<unsigned __int128>(sv->decimal_lo());
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
          val, numeric::scale_type{-static_cast<int32_t>(sv->decimal_scale())}, valid);
    }
    default:
      throw std::runtime_error(
          "unsupported scalar type in column path: " + std::to_string(sv->type()));
  }
}

static cudf::binary_operator fb_to_binop(fb::BinaryOp op) {
  switch (op) {
    case fb::BinaryOp_Eq:    return cudf::binary_operator::EQUAL;
    case fb::BinaryOp_NotEq: return cudf::binary_operator::NOT_EQUAL;
    case fb::BinaryOp_Lt:    return cudf::binary_operator::LESS;
    case fb::BinaryOp_LtEq:  return cudf::binary_operator::LESS_EQUAL;
    case fb::BinaryOp_Gt:    return cudf::binary_operator::GREATER;
    case fb::BinaryOp_GtEq:  return cudf::binary_operator::GREATER_EQUAL;
    case fb::BinaryOp_Plus:  return cudf::binary_operator::ADD;
    case fb::BinaryOp_Minus: return cudf::binary_operator::SUB;
    case fb::BinaryOp_Multiply: return cudf::binary_operator::MUL;
    case fb::BinaryOp_Divide:   return cudf::binary_operator::DIV;
    case fb::BinaryOp_Modulo:   return cudf::binary_operator::MOD;
    // SQL three-valued logic (see fb_to_ast_op): use the NULL_LOGICAL_* variants
    // so a disjunction with a NULL operand keeps the row when the other side is
    // TRUE. This column path handles OR over string comparisons / IN-lists
    // (e.g. q15's `ca_state IN (...)`, which the AST path can't express).
    case fb::BinaryOp_And:      return cudf::binary_operator::NULL_LOGICAL_AND;
    case fb::BinaryOp_Or:       return cudf::binary_operator::NULL_LOGICAL_OR;
    case fb::BinaryOp_BitwiseAnd: return cudf::binary_operator::BITWISE_AND;
    case fb::BinaryOp_BitwiseOr:  return cudf::binary_operator::BITWISE_OR;
    case fb::BinaryOp_BitwiseXor: return cudf::binary_operator::BITWISE_XOR;
    default:
      throw std::runtime_error(
          "unsupported BinaryOp in column path: " + std::to_string(op));
  }
}

// Forward declaration.
static std::unique_ptr<cudf::column> build_column(
    const fb::Expr* expr, cudf::table_view const& table);

// Evaluate an AST-able subtree by routing it through cudf::compute_column.
static std::unique_ptr<cudf::column> eval_ast_subtree(
    const fb::Expr* expr, cudf::table_view const& table) {
  ExprContext ctx;
  auto& ast = build_expr(expr, ctx);
  return cudf::compute_column(table, ast);
}

// Returns true if the binary op produces a bool column (comparison/logical).
static bool is_predicate_op(fb::BinaryOp op) {
  switch (op) {
    case fb::BinaryOp_Eq:
    case fb::BinaryOp_NotEq:
    case fb::BinaryOp_Lt:
    case fb::BinaryOp_LtEq:
    case fb::BinaryOp_Gt:
    case fb::BinaryOp_GtEq:
    case fb::BinaryOp_And:
    case fb::BinaryOp_Or:
      return true;
    default:
      return false;
  }
}

// Pick an output type for binary_operation. Boolean for predicates; otherwise
// promote to the wider of the two input types (cuDF's binary_operation does
// the actual coercion under the hood, but it needs us to declare an output).
cudf::data_type binop_output_type(
    fb::BinaryOp op, cudf::data_type lhs, cudf::data_type rhs) {
  if (is_predicate_op(op)) return cudf::data_type{cudf::type_id::BOOL8};
  // Fixed-point arithmetic: cuDF requires the output type's scale to equal the
  // scale it computes for the operation, so we can't just echo lhs. The rules
  // (scales are base-10 exponents, negative for fractional digits): ADD/SUB/MOD
  // take min(s_l, s_r); MUL adds; DIV subtracts. Matches SQL decimal semantics.
  if (lhs.id() == cudf::type_id::DECIMAL128 ||
      rhs.id() == cudf::type_id::DECIMAL128) {
    int32_t ls = lhs.scale();
    int32_t rs = rhs.scale();
    int32_t out_scale;
    switch (op) {
      case fb::BinaryOp_Multiply: out_scale = ls + rs; break;
      case fb::BinaryOp_Divide:   out_scale = ls - rs; break;
      default:                    out_scale = std::min(ls, rs); break;
    }
    return cudf::data_type{cudf::type_id::DECIMAL128, out_scale};
  }
  // Fall back to lhs type — adequate for the queries we care about
  // (arithmetic in projections is rare in this code path; the heavy
  // arithmetic still goes through AST).
  (void)rhs;
  return lhs;
}

static std::unique_ptr<cudf::column> build_column_binary(
    const fb::BinaryExprNode* bin, cudf::table_view const& table) {
  auto* lhs = bin->left();
  auto* rhs = bin->right();
  auto op = fb_to_binop(bin->op());

  // Decimal division: cuDF's fixed_point DIV yields scale s_l-s_r (e.g. 0 for
  // two scale-4 sums → truncates to 0), but DataFusion boosts the scale (its
  // declared value rides on out_decimal_precision/scale). Reproduce it by
  // pre-scaling the numerator so DIV lands on DataFusion's scale: with output
  // exponent e_o = −out_scale and denominator exponent e_r, set numerator
  // exponent e_l = e_o + e_r (since DIV gives e_l − e_r = e_o).
  if (bin->op() == fb::BinaryOp_Divide && bin->out_decimal_precision() != 0) {
    auto lcol = build_column(lhs, table);
    auto rcol = build_column(rhs, table);
    if (lcol->type().id() == cudf::type_id::DECIMAL128 &&
        rcol->type().id() == cudf::type_id::DECIMAL128) {
      int32_t e_o = -static_cast<int32_t>(bin->out_decimal_scale());
      int32_t e_r = rcol->type().scale();
      auto num = cudf::cast(
          lcol->view(), cudf::data_type{cudf::type_id::DECIMAL128, e_o + e_r});
      return cudf::binary_operation(
          num->view(), rcol->view(), op,
          cudf::data_type{cudf::type_id::DECIMAL128, e_o});
    }
    // out_decimal_precision != 0 means DataFusion declared a Decimal128 result,
    // which after scan-widening implies both operands materialise as DECIMAL128.
    throw std::runtime_error(
        "decimal division declared Decimal128 output but operand columns are "
        "not both DECIMAL128");
  }

  // Column-scalar fast path when one side is a literal.
  if (rhs->node_type() == fb::ExprNode_LiteralExpr &&
      lhs->node_type() != fb::ExprNode_LiteralExpr) {
    auto lcol = build_column(lhs, table);
    auto rsv = rhs->node_as_LiteralExpr()->value();
    auto rscalar = build_scalar(rsv);
    auto out = binop_output_type(bin->op(), lcol->type(), rscalar->type());
    return cudf::binary_operation(lcol->view(), *rscalar, op, out);
  }
  if (lhs->node_type() == fb::ExprNode_LiteralExpr &&
      rhs->node_type() != fb::ExprNode_LiteralExpr) {
    auto rcol = build_column(rhs, table);
    auto lsv = lhs->node_as_LiteralExpr()->value();
    auto lscalar = build_scalar(lsv);
    auto out = binop_output_type(bin->op(), lscalar->type(), rcol->type());
    return cudf::binary_operation(*lscalar, rcol->view(), op, out);
  }

  // Both sides materialise to columns.
  auto lcol = build_column(lhs, table);
  auto rcol = build_column(rhs, table);
  auto out = binop_output_type(bin->op(), lcol->type(), rcol->type());
  return cudf::binary_operation(lcol->view(), rcol->view(), op, out);
}

static std::unique_ptr<cudf::column> build_column_scalar_fn(
    const fb::ScalarFunctionExprNode* sf, cudf::table_view const& table) {
  auto name = sf->name() ? sf->name()->str() : std::string{};
  auto* args = sf->args();
  if (!args || args->size() == 0)
    throw std::runtime_error("ScalarFunction " + name + ": no args");

  // date_part(field, ts) — DataFusion encodes the field as a string literal.
  if (name == "date_part") {
    if (args->size() != 2)
      throw std::runtime_error("date_part expects 2 args");
    auto* field_expr = args->Get(0);
    if (field_expr->node_type() != fb::ExprNode_LiteralExpr)
      throw std::runtime_error("date_part: field must be a literal");
    auto* fsv = field_expr->node_as_LiteralExpr()->value();
    auto field = fsv && fsv->string_val() ? fsv->string_val()->str() : std::string{};
    for (auto& c : field) c = static_cast<char>(std::toupper(c));
    cudf::datetime::datetime_component comp;
    if      (field == "YEAR")    comp = cudf::datetime::datetime_component::YEAR;
    else if (field == "MONTH")   comp = cudf::datetime::datetime_component::MONTH;
    else if (field == "DAY")     comp = cudf::datetime::datetime_component::DAY;
    else if (field == "HOUR")    comp = cudf::datetime::datetime_component::HOUR;
    else if (field == "MINUTE")  comp = cudf::datetime::datetime_component::MINUTE;
    else if (field == "SECOND")  comp = cudf::datetime::datetime_component::SECOND;
    else throw std::runtime_error("date_part: unsupported field " + field);
    auto ts = build_column(args->Get(1), table);
    return cudf::datetime::extract_datetime_component(ts->view(), comp);
  }

  // substr(s, start, length) — SQL semantics: 1-based start.
  if (name == "substr" || name == "substring") {
    if (args->size() < 2 || args->size() > 3)
      throw std::runtime_error("substr expects 2 or 3 args");
    auto strcol = build_column(args->Get(0), table);

    auto lit_int = [&](const fb::Expr* e) -> int32_t {
      if (e->node_type() != fb::ExprNode_LiteralExpr)
        throw std::runtime_error("substr: position/length must be literals");
      auto* v = e->node_as_LiteralExpr()->value();
      return static_cast<int32_t>(v->int_val());
    };

    int32_t start_1 = lit_int(args->Get(1));            // 1-based
    int32_t start = start_1 > 0 ? start_1 - 1 : start_1;  // → 0-based
    cudf::numeric_scalar<cudf::size_type> start_s(start, true);

    if (args->size() == 3) {
      int32_t len = lit_int(args->Get(2));
      int32_t stop = start + len;
      cudf::numeric_scalar<cudf::size_type> stop_s(stop, true);
      cudf::numeric_scalar<cudf::size_type> step_s(1, true);
      return cudf::strings::slice_strings(
          cudf::strings_column_view{strcol->view()}, start_s, stop_s, step_s);
    }
    // No length → slice through end.
    cudf::numeric_scalar<cudf::size_type> stop_s(0, false);  // null = "to end"
    cudf::numeric_scalar<cudf::size_type> step_s(1, true);
    return cudf::strings::slice_strings(
        cudf::strings_column_view{strcol->view()}, start_s, stop_s, step_s);
  }

  // abs(x) — numeric/decimal absolute value.
  if (name == "abs") {
    if (args->size() != 1)
      throw std::runtime_error("abs expects 1 arg");
    auto col = build_column(args->Get(0), table);
    return cudf::unary_operation(col->view(), cudf::unary_operator::ABS);
  }

  // lower(s) — lowercase a string column.
  if (name == "lower") {
    if (args->size() != 1)
      throw std::runtime_error("lower expects 1 arg");
    auto col = build_column(args->Get(0), table);
    return cudf::strings::to_lower(cudf::strings_column_view{col->view()});
  }

  // upper(s) — uppercase a string column.
  if (name == "upper") {
    if (args->size() != 1)
      throw std::runtime_error("upper expects 1 arg");
    auto col = build_column(args->Get(0), table);
    return cudf::strings::to_upper(cudf::strings_column_view{col->view()});
  }

  // concat(a, b, …) — string concatenation. DataFusion's `concat` treats NULL
  // as the empty string, so map nulls to "" (narep) rather than nulling the row.
  if (name == "concat") {
    std::vector<std::unique_ptr<cudf::column>> owned;
    std::vector<cudf::column_view> views;
    owned.reserve(args->size());
    views.reserve(args->size());
    for (flatbuffers::uoffset_t k = 0; k < args->size(); ++k) {
      owned.push_back(build_column(args->Get(k), table));
      views.push_back(owned.back()->view());
    }
    cudf::string_scalar separator("", true);
    cudf::string_scalar narep("", true);
    return cudf::strings::concatenate(cudf::table_view{views}, separator, narep);
  }

  // coalesce(a, b, …) — first non-null per row. Fold from the last arg back,
  // selecting arg_k where it is valid, otherwise the accumulated result.
  if (name == "coalesce") {
    auto n = args->size();
    auto result = build_column(args->Get(n - 1), table);
    for (int k = static_cast<int>(n) - 2; k >= 0; --k) {
      auto col = build_column(args->Get(k), table);
      auto mask = cudf::is_valid(col->view());
      result = cudf::copy_if_else(col->view(), result->view(), mask->view());
    }
    return result;
  }

  throw std::runtime_error("unsupported scalar function in column path: " + name);
}

static std::unique_ptr<cudf::column> build_column_case(
    const fb::CaseExprNode* c, cudf::table_view const& table) {
  // Search-form CASE only. Value-form (`CASE x WHEN v THEN t ... END`) is NOT
  // rewritten away by DataFusion — it survives to the physical plan and
  // plan_serializer.rs forwards `case.expr()` as the comparand — so it does
  // reach here for e.g. TPC-DS q39 (`CASE mean WHEN 0 THEN NULL/0 ELSE
  // stdev/mean END`). A column-path implementation (comparand + per-branch
  // `comparand == when` + copy_if_else fold) was attempted but produced
  // incorrect results on the GPU (every output row came back 0/null — #57), so
  // it is withheld rather than shipped unverified. q39 is the only value-form
  // user and is already disabled (stddev ULP, #54); this throw is therefore
  // unreachable from the enabled suite. Re-implement, with a direct GPU test,
  // when q39 is enabled (#57).
  if (c->expr())
    throw std::runtime_error("value-form CASE not supported in column path");
  auto* whens = c->when_thens();
  if (!whens || whens->size() == 0)
    throw std::runtime_error("CASE has no WHEN/THEN pairs");

  // Build the ELSE column first (or null if none); fold from the last WHEN
  // backward so each step produces `if cond_i then then_i else accumulated`.
  std::unique_ptr<cudf::column> result;
  if (c->else_expr()) {
    result = build_column(c->else_expr(), table);
  } else {
    // Use the THEN type of the last branch as a reference for null fill.
    auto last_then = build_column(whens->Get(whens->size() - 1)->then(), table);
    auto null_scalar = cudf::make_default_constructed_scalar(last_then->type());
    result = cudf::make_column_from_scalar(*null_scalar, last_then->size());
  }

  for (cudf::size_type i = static_cast<cudf::size_type>(whens->size()) - 1; i >= 0; --i) {
    auto* wt = whens->Get(static_cast<flatbuffers::uoffset_t>(i));
    auto cond = build_column(wt->when(), table);
    auto then = build_column(wt->then(), table);
    result = cudf::copy_if_else(then->view(), result->view(), cond->view());
  }
  return result;
}

static const char* expr_kind_name(fb::ExprNode k) {
  switch (k) {
    case fb::ExprNode_ColumnRef:               return "ColumnRef";
    case fb::ExprNode_LiteralExpr:             return "Literal";
    case fb::ExprNode_BinaryExprNode:          return "Binary";
    case fb::ExprNode_UnaryExprNode:           return "Unary";
    case fb::ExprNode_CastExprNode:            return "Cast";
    case fb::ExprNode_LikeExprNode:            return "Like";
    case fb::ExprNode_CaseExprNode:            return "Case";
    case fb::ExprNode_ScalarFunctionExprNode:  return "ScalarFn";
    default:                                    return "?";
  }
}

static std::unique_ptr<cudf::column> build_column(
    const fb::Expr* expr, cudf::table_view const& table) {
  if (debug_enabled()) {
    PCK_TRACE("  build_column kind=%s rows=%d cols=%d",
              expr_kind_name(expr->node_type()),
              table.num_rows(), table.num_columns());
  }
  // Plain literal: broadcast scalar to the table's row count. cudf::ast
  // doesn't have a defined behaviour for literal-only expressions in
  // compute_column, so handle this case before the AST fast path.
  if (expr->node_type() == fb::ExprNode_LiteralExpr) {
    auto sc = build_scalar(expr->node_as_LiteralExpr()->value());
    auto out = cudf::make_column_from_scalar(*sc, table.num_rows());
    debug_sync("Literal->make_column_from_scalar");
    return out;
  }

  // Bare column reference: copy the column view directly. compute_column
  // would allocate fixed-width output and reject strings/lists/structs.
  if (expr->node_type() == fb::ExprNode_ColumnRef) {
    auto* c = expr->node_as_ColumnRef();
    auto idx = static_cast<cudf::size_type>(c->index());
    if (idx < 0 || idx >= table.num_columns()) {
      throw std::runtime_error(
          "ColumnRef index " + std::to_string(idx) +
          " out of range (cols=" + std::to_string(table.num_columns()) + ")");
    }
    auto cv = table.column(idx);
    if (debug_enabled()) {
      PCK_TRACE("  ColumnRef idx=%d type_id=%d size=%d null_count=%d",
                static_cast<int>(idx),
                static_cast<int>(cv.type().id()),
                static_cast<int>(cv.size()),
                static_cast<int>(cv.null_count()));
    }
    auto out = std::make_unique<cudf::column>(cv);
    debug_sync("ColumnRef->copy");
    return out;
  }

  // AST-able expressions go through cudf::compute_column for fusion.
  if (is_ast_able(expr, table)) {
    auto out = eval_ast_subtree(expr, table);
    debug_sync("AST->compute_column");
    return out;
  }

  switch (expr->node_type()) {
    case fb::ExprNode_BinaryExprNode:
      return build_column_binary(expr->node_as_BinaryExprNode(), table);

    case fb::ExprNode_UnaryExprNode: {
      auto* un = expr->node_as_UnaryExprNode();
      auto arg = build_column(un->arg(), table);
      switch (un->op()) {
        case fb::UnaryOp_Not:
          return cudf::unary_operation(arg->view(), cudf::unary_operator::NOT);
        case fb::UnaryOp_IsNull:
          return cudf::is_null(arg->view());
        case fb::UnaryOp_IsNotNull:
          return cudf::is_valid(arg->view());
        default:
          throw std::runtime_error(
              "UnaryOp not supported in column path: " + std::to_string(un->op()));
      }
    }

    case fb::ExprNode_LikeExprNode: {
      auto* l = expr->node_as_LikeExprNode();
      auto strcol = build_column(l->expr(), table);
      auto* psv = l->pattern() && l->pattern()->node_type() == fb::ExprNode_LiteralExpr
                      ? l->pattern()->node_as_LiteralExpr()->value()
                      : nullptr;
      if (!psv || !psv->string_val())
        throw std::runtime_error("LIKE pattern must be a string literal");
      cudf::string_scalar pattern(psv->string_val()->str(), true);
      auto mask = cudf::strings::like(
          cudf::strings_column_view{strcol->view()}, pattern);
      if (l->negated()) {
        return cudf::unary_operation(mask->view(), cudf::unary_operator::NOT);
      }
      return mask;
    }

    case fb::ExprNode_CaseExprNode:
      return build_column_case(expr->node_as_CaseExprNode(), table);

    case fb::ExprNode_ScalarFunctionExprNode:
      return build_column_scalar_fn(expr->node_as_ScalarFunctionExprNode(), table);

    case fb::ExprNode_CastExprNode: {
      auto* cast = expr->node_as_CastExprNode();
      auto inner = build_column(cast->expr(), table);
      auto target_id = fb_to_type_id(cast->target_type());
      // Decimal types need a scale. Arrow/DataFusion scale counts fractional
      // digits (positive); cuDF's fixed_point scale is the base-10 exponent
      // (negated). Other types use the default (scale 0).
      cudf::data_type target =
          target_id == cudf::type_id::DECIMAL128
              ? cudf::data_type{target_id, -static_cast<int32_t>(cast->decimal_scale())}
              : cudf::data_type{target_id};
      return cudf::cast(inner->view(), target);
    }

    default:
      // Other nodes (Column, Literal) are AST-able and were handled above.
      throw std::runtime_error(
          "unexpected non-AST expression: " + std::to_string(expr->node_type()));
  }
}

// ============================================================================
// GpuScan — read Parquet files
// ============================================================================

static TableResult execute_scan(const fb::GpuScan* scan) {
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

  auto result = cudf::io::read_parquet(opts);

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

// ============================================================================
// GpuFilter — apply boolean predicate
// ============================================================================

static TableResult execute_filter(const fb::GpuFilter* filter) {
  auto input = execute_node(filter->input());

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

// ============================================================================
// GpuProject — column selection / renaming
// ============================================================================

static TableResult execute_project(const fb::GpuProject* proj) {
  auto input = execute_node(proj->input());

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

// ============================================================================
// GpuAggregate — group-by aggregation
// ============================================================================

// DataFusion lowers stddev_samp/stddev → "stddev" and stddev_pop → "stddev_pop"
// (variance variants likewise). cuDF's STD aggregation takes a ddof: sample std
// uses ddof=1, population std ddof=0.
static bool is_stddev_name(const std::string& f) {
  return f == "stddev" || f == "STDDEV" || f == "stddev_samp" ||
         f == "STDDEV_SAMP" || f == "stddev_pop" || f == "STDDEV_POP";
}
static bool is_avg_name(const std::string& f) {
  return f == "avg" || f == "AVG" || f == "mean" || f == "MEAN";
}
static cudf::size_type stddev_ddof(const std::string& f) {
  return (f == "stddev_pop" || f == "STDDEV_POP") ? 0 : 1;
}

static std::unique_ptr<cudf::groupby_aggregation> make_agg(
    const std::string& func_name, bool is_final) {
  // In Final mode, count→sum (sum partial counts), others stay the same.
  if (func_name == "count" || func_name == "COUNT") {
    if (is_final)
      return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    else
      return cudf::make_count_aggregation<cudf::groupby_aggregation>();
  }
  if (func_name == "sum" || func_name == "SUM")
    return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  if (func_name == "min" || func_name == "MIN")
    return cudf::make_min_aggregation<cudf::groupby_aggregation>();
  if (func_name == "max" || func_name == "MAX")
    return cudf::make_max_aggregation<cudf::groupby_aggregation>();
  if (func_name == "avg" || func_name == "AVG" ||
      func_name == "mean" || func_name == "MEAN") {
    // GpuRepartition executes as passthrough, so the Partial stage already
    // groups the full input (one row per key); the Final stage then regroups
    // those unique keys, making MEAN-of-singleton an identity. So a plain
    // groupby MEAN is correct in both Partial and Final modes here.
    // NOTE: correct only while Partial output is one-row-per-key. Multi-
    // partition repartition breaks this (mean-of-means); execute_aggregate
    // guards against it at runtime — see the has_avg_final check there.
    // Decompose AVG into SUM+COUNT to lift this restriction:
    // https://github.com/asymptote-tech/peacockdb/issues/25
    return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  }
  if (is_stddev_name(func_name)) {
    // Two-phase stddev: the Partial stage computes the real per-group std over
    // the raw rows; with passthrough repartition that output is one row per key,
    // so the Final regroup is an identity. cuDF cannot merge partial std states,
    // so model the Final stage as a singleton-identity (MEAN over the lone
    // partial row), exactly as AVG does above. The has_singleton_final guard in
    // execute_aggregate fails loudly if a real multi-row merge ever happens.
    if (is_final)
      return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    return cudf::make_std_aggregation<cudf::groupby_aggregation>(
        stddev_ddof(func_name));
  }
  throw std::runtime_error("unsupported aggregate function: " + func_name);
}

static std::unique_ptr<cudf::reduce_aggregation> make_reduce_agg(
    const std::string& func_name) {
  if (func_name == "count" || func_name == "COUNT")
    throw std::runtime_error("count handled inline — make_reduce_agg should not be called for count");
  if (func_name == "sum" || func_name == "SUM")
    return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  if (func_name == "min" || func_name == "MIN")
    return cudf::make_min_aggregation<cudf::reduce_aggregation>();
  if (func_name == "max" || func_name == "MAX")
    return cudf::make_max_aggregation<cudf::reduce_aggregation>();
  if (func_name == "avg" || func_name == "AVG" ||
      func_name == "mean" || func_name == "MEAN")
    return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
  throw std::runtime_error("unsupported aggregate function: " + func_name);
}

static TableResult execute_aggregate(const fb::GpuAggregate* agg) {
  auto input = execute_node(agg->input());
  auto tv = input.table->view();

  bool is_final = (agg->mode() == fb::AggregateMode_Final ||
                   agg->mode() == fb::AggregateMode_FinalPartitioned);

  // Build group-by keys.
  std::vector<cudf::size_type> key_indices;
  std::vector<std::string> key_names;
  if (agg->group_exprs()) {
    for (flatbuffers::uoffset_t i = 0; i < agg->group_exprs()->size(); ++i) {
      auto* expr = agg->group_exprs()->Get(i);
      if (expr->node_type() != fb::ExprNode_ColumnRef)
        throw std::runtime_error("GpuAggregate: only ColumnRef group exprs supported");
      auto* col = expr->node_as_ColumnRef();
      key_indices.push_back(static_cast<cudf::size_type>(col->index()));
      if (agg->group_names() && i < agg->group_names()->size())
        key_names.push_back(agg->group_names()->Get(i)->str());
      else
        key_names.push_back(input.column_names[col->index()]);
    }
  }

  // Build key table.
  std::vector<cudf::column_view> key_cols;
  for (auto idx : key_indices) key_cols.push_back(tv.column(idx));

  // Owns columns materialised for aggregate arguments that aren't plain column
  // references (e.g. sum(a*b), sum(CASE ...)). Must outlive the aggregate call
  // below, since the column_views handed to cuDF point into these.
  std::vector<std::unique_ptr<cudf::column>> computed_args;

  // Helper to determine the values column for a function node.
  auto get_values_col = [&](const fb::AggregateFuncNode* func) -> cudf::column_view {
    cudf::column_view base;
    if (func->args() && func->args()->size() > 0) {
      auto* arg = func->args()->Get(0);
      if (arg->node_type() == fb::ExprNode_ColumnRef) {
        base = tv.column(static_cast<cudf::size_type>(arg->node_as_ColumnRef()->index()));
      } else {
        // Aggregate over a computed expression: DataFusion inlines the argument
        // (no preceding ProjectionExec). Materialise it against the input table
        // rather than silently aggregating the wrong column.
        computed_args.push_back(build_column(arg, tv));
        base = computed_args.back()->view();
      }
    } else {
      base = tv.column(0);  // count(*) or no args: dummy first column
    }
    // avg over a decimal: DataFusion's result scale is s+4, but cuDF's mean
    // keeps the input scale s (truncating precision). Cast the input up to the
    // declared output scale first so the mean carries the right value, not just
    // a zero-padded display (out_decimal_scale rides on the func node).
    std::string fname = func->name() ? func->name()->str() : "";
    bool is_avg = (fname == "avg" || fname == "AVG" ||
                   fname == "mean" || fname == "MEAN");
    if (is_avg && func->out_decimal_precision() != 0 &&
        base.type().id() == cudf::type_id::DECIMAL128) {
      int32_t want_exp = -static_cast<int32_t>(func->out_decimal_scale());
      if (base.type().scale() != want_exp) {
        computed_args.push_back(cudf::cast(
            base, cudf::data_type{cudf::type_id::DECIMAL128, want_exp}));
        base = computed_args.back()->view();
      }
    }
    return base;
  };

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  std::vector<std::string> out_names;

  if (key_cols.empty()) {
    // Global aggregation (no group-by): use cudf::reduce to produce one row.
    if (agg->aggr_funcs()) {
      for (flatbuffers::uoffset_t i = 0; i < agg->aggr_funcs()->size(); ++i) {
        auto* func = agg->aggr_funcs()->Get(i);
        std::string name = func->name() ? func->name()->str() : "count";

        cudf::column_view values_col = get_values_col(func);
        bool is_count = (name == "count" || name == "COUNT");

        bool is_avg = is_avg_name(name);
        bool is_std = is_stddev_name(name);
        // Same shortcut/guard as the grouped path: a Final-stage global AVG (or
        // STDDEV) reduces over the Partial outputs. With passthrough repartition
        // there is exactly one partial row, so MEAN/STDDEV-of-one is an identity.
        // More than one row means a multi-partition merge → silently-wrong
        // mean-of-means / std-of-stds (decompose AVG into SUM+COUNT to lift this:
        // https://github.com/asymptote-tech/peacockdb/issues/25).
        if (is_final && (is_avg || is_std) && values_col.size() > 1) {
          throw std::runtime_error(
              "Final-stage AVG/STDDEV merged multiple partial rows "
              "(mean-of-means is wrong); must be decomposed into "
              "additive state before multi-partition GPU repartition is enabled");
        }

        bool is_sum = (name == "sum" || name == "SUM");
        std::unique_ptr<cudf::column> result_col;
        if (is_count) {
          // Avoid make_count_aggregation<reduce_aggregation> which is not
          // exported in all cudf versions. Count = size - null_count.
          int64_t cnt = static_cast<int64_t>(values_col.size()) -
                        static_cast<int64_t>(values_col.null_count());
          cudf::numeric_scalar<int64_t> s(cnt, true);
          result_col = cudf::make_column_from_scalar(s, 1);
        } else if (values_col.type().id() == cudf::type_id::DECIMAL128 &&
                   (is_sum || is_avg)) {
          // cudf::reduce supports only min/max on fixed_point types; sum/mean
          // throw "non-arithmetic types" (TPC-H q22's global avg of a decimal).
          // The groupby path *does* support decimal sum/mean, so tag every row
          // with one constant key and run a single-group aggregation.
          if (values_col.size() == 0) {
            // SQL sum/avg over zero rows is NULL; groupby would emit no groups,
            // so build the one-row null result of the input type directly.
            result_col = cudf::make_column_from_scalar(
                *cudf::make_default_constructed_scalar(values_col.type()), 1);
          } else {
            cudf::numeric_scalar<int32_t> zero(0, true);
            auto key = cudf::make_column_from_scalar(zero, values_col.size());
            cudf::table_view kview{{key->view()}};
            cudf::groupby::groupby gb1{kview};
            std::vector<cudf::groupby::aggregation_request> reqs(1);
            reqs[0].values = values_col;
            reqs[0].aggregations.push_back(make_agg(name, is_final));
            auto [gk, res] = gb1.aggregate(reqs);
            result_col = std::move(res[0].results[0]);
          }
        } else if (is_std) {
          // Global stddev. cuDF's compound STD reduction outputs FLOAT64. In the
          // Final stage the single partial row already holds the real std, so
          // take it as-is (MEAN-of-one identity); otherwise reduce STD directly.
          std::unique_ptr<cudf::reduce_aggregation> ragg =
              is_final ? cudf::make_mean_aggregation<cudf::reduce_aggregation>()
                       : cudf::make_std_aggregation<cudf::reduce_aggregation>(
                             stddev_ddof(name));
          auto scalar_result = cudf::reduce(
              values_col, *ragg, cudf::data_type{cudf::type_id::FLOAT64});
          result_col = cudf::make_column_from_scalar(*scalar_result, 1);
        } else {
          // avg/mean's compound reduction must output FLOAT64 (cuDF rejects an
          // integer output type); sum/min/max keep the input type.
          cudf::data_type out_t =
              is_avg ? cudf::data_type{cudf::type_id::FLOAT64} : values_col.type();
          auto scalar_result =
              cudf::reduce(values_col, *make_reduce_agg(name), out_t);
          result_col = cudf::make_column_from_scalar(*scalar_result, 1);
        }
        out_cols.push_back(std::move(result_col));

        if (func->alias())
          out_names.push_back(func->alias()->str());
        else
          out_names.push_back(name);
      }
    }
    return {std::make_unique<cudf::table>(std::move(out_cols)), std::move(out_names)};
  }

  cudf::table_view keys_view{key_cols};
  // SQL GROUP BY puts NULL keys in their own group; cuDF's groupby defaults to
  // null_policy::EXCLUDE, which DROPS every row that has a NULL in any grouping
  // key — silently losing the NULL group (e.g. TPC-DS q15's NULL ca_zip row).
  // INCLUDE matches the peacock CPU oracle. Non-null keys are unaffected.
  cudf::groupby::groupby gb{keys_view, cudf::null_policy::INCLUDE};

  // Build aggregation requests — one per aggregate function.
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::string> agg_names;
  std::vector<bool> agg_is_count;
  bool has_avg_final = false;
  if (agg->aggr_funcs()) {
    for (flatbuffers::uoffset_t i = 0; i < agg->aggr_funcs()->size(); ++i) {
      auto* func = agg->aggr_funcs()->Get(i);
      std::string name = func->name() ? func->name()->str() : "count";

      cudf::groupby::aggregation_request req;
      req.values = get_values_col(func);
      req.aggregations.push_back(make_agg(name, is_final));
      requests.push_back(std::move(req));

      if (func->alias())
        agg_names.push_back(func->alias()->str());
      else
        agg_names.push_back(name);
      agg_is_count.push_back(name == "count" || name == "COUNT");
      // STDDEV uses the same singleton-identity shortcut as AVG in Final mode.
      if (is_final && (is_avg_name(name) || is_stddev_name(name)))
        has_avg_final = true;
    }
  }

  auto [group_keys, agg_results] = gb.aggregate(requests);

  // Guard the AVG-as-plain-MEAN shortcut (see make_agg). It is correct only
  // when the Partial stage already produced one row per key, so the Final
  // regroup is an identity (MEAN-of-singleton). Today that holds because
  // GpuRepartition runs as passthrough (single partition). If real multi-
  // partition repartition is ever enabled, the same key arrives from several
  // Partial outputs and this Final groupby actually merges rows — turning AVG
  // into a silently-wrong mean-of-means. Detect that here (fewer output groups
  // than input rows) and fail loudly instead. The fix at that point is to
  // decompose AVG into SUM+COUNT across the Partial/Final boundary:
  // https://github.com/asymptote-tech/peacockdb/issues/25
  if (has_avg_final &&
      group_keys->num_rows() < static_cast<cudf::size_type>(tv.num_rows())) {
    throw std::runtime_error(
        "Final-stage AVG merged multiple partial rows per key "
        "(mean-of-means is wrong); AVG must be decomposed into SUM+COUNT "
        "before multi-partition GPU repartition is enabled");
  }

  // Assemble output: key columns + aggregation result columns.
  for (cudf::size_type i = 0; i < group_keys->num_columns(); ++i) {
    out_cols.push_back(std::make_unique<cudf::column>(group_keys->view().column(i)));
    out_names.push_back(key_names[i]);
  }
  for (size_t i = 0; i < agg_results.size(); ++i) {
    // Each aggregation_result has one column per aggregation; we have one each.
    auto col = std::move(agg_results[i].results[0]);
    // cuDF count returns INT32; cast to INT64 for SQL BIGINT compatibility.
    if (agg_is_count[i] && col->type().id() == cudf::type_id::INT32)
      col = cudf::cast(*col, cudf::data_type{cudf::type_id::INT64});
    out_cols.push_back(std::move(col));
    out_names.push_back(agg_names[i]);
  }

  return {std::make_unique<cudf::table>(std::move(out_cols)), std::move(out_names)};
}

// ============================================================================
// GpuHashJoin — equi-join
// ============================================================================

static TableResult execute_hash_join(const fb::GpuHashJoin* join) {
  auto left = execute_node(join->left());
  auto right = execute_node(join->right());

  auto ltv = left.table->view();
  auto rtv = right.table->view();

  // Build key tables.
  std::vector<cudf::column_view> left_key_cols, right_key_cols;
  if (join->keys()) {
    for (flatbuffers::uoffset_t i = 0; i < join->keys()->size(); ++i) {
      auto* key = join->keys()->Get(i);
      auto* lk = key->left();
      auto* rk = key->right();
      if (!lk || !rk || lk->node_type() != fb::ExprNode_ColumnRef ||
          rk->node_type() != fb::ExprNode_ColumnRef)
        throw std::runtime_error("GpuHashJoin: only ColumnRef keys supported");
      left_key_cols.push_back(
          ltv.column(static_cast<cudf::size_type>(lk->node_as_ColumnRef()->index())));
      right_key_cols.push_back(
          rtv.column(static_cast<cudf::size_type>(rk->node_as_ColumnRef()->index())));
    }
  }

  cudf::table_view left_keys{left_key_cols};
  cudf::table_view right_keys{right_key_cols};

  // Semi/anti joins emit only one side's columns and use a different cuDF API
  // (single index vector instead of a pair). Right{Semi,Anti} = Left{Semi,Anti}
  // with sides swapped, so we normalise to Left{Semi,Anti} and remember which
  // side to gather from.
  bool is_semi_or_anti = false;
  bool emit_left = true;  // false → emit right side instead
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> single_indices;

  // A residual filter on a semi/anti join (e.g. EXISTS / NOT EXISTS with a `<>`
  // correlation, as in TPC-H q21) is NOT optional: the key-only cuDF semi/anti
  // joins ignore it, which silently changes the result — a LeftAnti on the key
  // alone excludes every left row whose key trivially exists in the right side,
  // collapsing the output to zero rows. Route those to the mixed_* variants,
  // whose AST predicate is evaluated during the join. Build the predicate only
  // for semi/anti types so inner-join (non-AST) filters aren't forced through
  // the AST path.
  auto jt = join->join_type();
  bool semi_anti_type =
      jt == fb::JoinType_LeftSemi || jt == fb::JoinType_LeftAnti ||
      jt == fb::JoinType_RightSemi || jt == fb::JoinType_RightAnti;
  ExprContext semi_ctx;
  const cudf::ast::expression* semi_pred = nullptr;
  if (semi_anti_type && join->filter()) {
    if (!join->filter_columns())
      throw std::runtime_error(
          "semi/anti join has a filter but no filter_columns map");
    semi_pred = &build_expr(join->filter(), semi_ctx, join->filter_columns());
  }

  // cuDF replaced free `left_{semi,anti}_join` with `filtered_join` (build the
  // hash table from one side, then probe). For Left{Semi,Anti} the right side
  // is the filter; for Right{Semi,Anti} we swap.
  switch (jt) {
    case fb::JoinType_LeftSemi: {
      if (semi_pred) {
        single_indices = cudf::mixed_left_semi_join(
            left_keys, right_keys, ltv, rtv, *semi_pred,
            cudf::null_equality::EQUAL);
      } else {
#ifdef PEACOCK_HAVE_FILTERED_JOIN
        cudf::filtered_join fj(right_keys, cudf::null_equality::EQUAL,
                               cudf::set_as_build_table::RIGHT, 0.5);
        single_indices = fj.semi_join(left_keys);
#else
        single_indices = cudf::left_semi_join(left_keys, right_keys);
#endif
      }
      is_semi_or_anti = true;
      emit_left = true;
      break;
    }
    case fb::JoinType_LeftAnti: {
      if (semi_pred) {
        single_indices = cudf::mixed_left_anti_join(
            left_keys, right_keys, ltv, rtv, *semi_pred,
            cudf::null_equality::EQUAL);
      } else {
#ifdef PEACOCK_HAVE_FILTERED_JOIN
        cudf::filtered_join fj(right_keys, cudf::null_equality::EQUAL,
                               cudf::set_as_build_table::RIGHT, 0.5);
        single_indices = fj.anti_join(left_keys);
#else
        single_indices = cudf::left_anti_join(left_keys, right_keys);
#endif
      }
      is_semi_or_anti = true;
      emit_left = true;
      break;
    }
    case fb::JoinType_RightSemi: {
      if (semi_pred)
        throw std::runtime_error(
            "residual filter on RightSemi join not supported (no swapped "
            "mixed-join path); should not arise from DataFusion decorrelation");
#ifdef PEACOCK_HAVE_FILTERED_JOIN
      cudf::filtered_join fj(left_keys, cudf::null_equality::EQUAL,
                             cudf::set_as_build_table::RIGHT, 0.5);
      single_indices = fj.semi_join(right_keys);
#else
      single_indices = cudf::left_semi_join(right_keys, left_keys);
#endif
      is_semi_or_anti = true;
      emit_left = false;
      break;
    }
    case fb::JoinType_RightAnti: {
      if (semi_pred)
        throw std::runtime_error(
            "residual filter on RightAnti join not supported (no swapped "
            "mixed-join path); should not arise from DataFusion decorrelation");
#ifdef PEACOCK_HAVE_FILTERED_JOIN
      cudf::filtered_join fj(left_keys, cudf::null_equality::EQUAL,
                             cudf::set_as_build_table::RIGHT, 0.5);
      single_indices = fj.anti_join(right_keys);
#else
      single_indices = cudf::left_anti_join(right_keys, left_keys);
#endif
      is_semi_or_anti = true;
      emit_left = false;
      break;
    }
    default:
      break;
  }

  if (is_semi_or_anti) {
    auto& side_tv = emit_left ? ltv : rtv;
    auto& side_names = emit_left ? left.column_names : right.column_names;
    auto m = static_cast<cudf::size_type>(single_indices->size());
    cudf::column_view idx_col{cudf::data_type{cudf::type_id::INT32}, m,
                              single_indices->data(), nullptr, 0, 0, {}};
    auto gathered = cudf::gather(side_tv, idx_col);
    auto gtv = gathered->view();
    std::vector<std::unique_ptr<cudf::column>> cols;
    std::vector<std::string> names;
    for (cudf::size_type i = 0; i < gtv.num_columns(); ++i) {
      cols.push_back(std::make_unique<cudf::column>(gtv.column(i)));
      names.push_back(side_names[i]);
    }
    auto t = std::make_unique<cudf::table>(std::move(cols));
    if (join->projection() && join->projection()->size() > 0) {
      auto tv = t->view();
      std::vector<std::unique_ptr<cudf::column>> p_cols;
      std::vector<std::string> p_names;
      for (auto idx : *join->projection()) {
        p_cols.push_back(std::make_unique<cudf::column>(tv.column(idx)));
        p_names.push_back(names[idx]);
      }
      return {std::make_unique<cudf::table>(std::move(p_cols)),
              std::move(p_names)};
    }
    return {std::move(t), std::move(names)};
  }

  // LeftMark: one row per left row, plus a trailing boolean "mark" column that
  // is true iff the left row has >=1 match in the right input (DataFusion's
  // EXISTS-in-disjunction decorrelation). cuDF has no mark join, so compute the
  // matched left-row indices with a (mixed) left semi-join and scatter `true`
  // into an all-false boolean column.
  if (jt == fb::JoinType_LeftMark) {
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> matched;
    if (join->filter()) {
      if (!join->filter_columns())
        throw std::runtime_error(
            "LeftMark join has a filter but no filter_columns map");
      ExprContext mctx;
      const auto& pred = build_expr(join->filter(), mctx, join->filter_columns());
      matched = cudf::mixed_left_semi_join(left_keys, right_keys, ltv, rtv, pred,
                                           cudf::null_equality::EQUAL);
    } else {
#ifdef PEACOCK_HAVE_FILTERED_JOIN
      cudf::filtered_join fj(right_keys, cudf::null_equality::EQUAL,
                             cudf::set_as_build_table::RIGHT, 0.5);
      matched = fj.semi_join(left_keys);
#else
      matched = cudf::left_semi_join(left_keys, right_keys);
#endif
    }
    auto nrows = ltv.num_rows();
    auto m = static_cast<cudf::size_type>(matched->size());
    cudf::numeric_scalar<bool> true_s(true), false_s(false);
    auto target = cudf::make_column_from_scalar(false_s, nrows);
    auto src = cudf::make_column_from_scalar(true_s, m);
    cudf::column_view map_col{cudf::data_type{cudf::type_id::INT32}, m,
                              matched->data(), nullptr, 0, 0, {}};
    auto scattered = cudf::scatter(cudf::table_view{{src->view()}}, map_col,
                                   cudf::table_view{{target->view()}});
    auto scattered_cols = scattered->release();

    std::vector<std::unique_ptr<cudf::column>> cols;
    std::vector<std::string> names;
    for (cudf::size_type i = 0; i < ltv.num_columns(); ++i) {
      cols.push_back(std::make_unique<cudf::column>(ltv.column(i)));
      names.push_back(left.column_names[i]);
    }
    cols.push_back(std::move(scattered_cols.front()));
    names.push_back("mark");
    auto t = std::make_unique<cudf::table>(std::move(cols));
    if (join->projection() && join->projection()->size() > 0) {
      auto tv = t->view();
      std::vector<std::unique_ptr<cudf::column>> p_cols;
      std::vector<std::string> p_names;
      for (auto idx : *join->projection()) {
        p_cols.push_back(std::make_unique<cudf::column>(tv.column(idx)));
        p_names.push_back(names[idx]);
      }
      return {std::make_unique<cudf::table>(std::move(p_cols)),
              std::move(p_names)};
    }
    return {std::move(t), std::move(names)};
  }

  // SQL equi-joins never match on NULL keys (NULL = NULL is unknown, not true),
  // but cuDF's join APIs default to null_equality::EQUAL, which pairs NULL keys
  // together and invents rows the SQL oracle excludes — e.g. TPC-DS q50/q6/q81,
  // where a spurious NULL=NULL match inflates a downstream count/sum by one.
  // UNEQUAL restores SQL semantics for inner/left/full/right. (Non-null keys —
  // the whole passing suite — are unaffected.)
  constexpr auto kJoinNulls = cudf::null_equality::UNEQUAL;

  // Execute join — returns index pairs.
  auto [left_indices, right_indices] = [&]() {
    switch (join->join_type()) {
      case fb::JoinType_Inner:
        return cudf::inner_join(left_keys, right_keys, kJoinNulls);
      case fb::JoinType_Left:
        return cudf::left_join(left_keys, right_keys, kJoinNulls);
      case fb::JoinType_Full:
        return cudf::full_join(left_keys, right_keys, kJoinNulls);
      case fb::JoinType_Right: {
        // cuDF has no right_join; right_join(L,R) == left_join(R,L) with the
        // returned (right_idx, left_idx) pair swapped back to (left_idx,
        // right_idx). Unmatched left rows then carry JoinNoneValue and are
        // NULLIFY-gathered below (see left_policy).
        auto p = cudf::left_join(right_keys, left_keys, kJoinNulls);
        return std::make_pair(std::move(p.second), std::move(p.first));
      }
      default:
        throw std::runtime_error(
            "unsupported join type: " + std::to_string(join->join_type()));
    }
  }();

  // Gather rows from both sides.
  //
  // For LEFT/FULL outer joins, cuDF signals unmatched rows with
  // JoinNoneValue (INT32_MIN) in the corresponding index vector — gathering
  // those with the default DONT_CHECK policy reads out of bounds and faults
  // with cudaErrorIllegalAddress. NULLIFY converts sentinel indices to nulls.
  using cudf::out_of_bounds_policy;
  auto kind = join->join_type();
  auto right_policy = (kind == fb::JoinType_Left || kind == fb::JoinType_Full)
                          ? out_of_bounds_policy::NULLIFY
                          : out_of_bounds_policy::DONT_CHECK;
  auto left_policy = (kind == fb::JoinType_Full || kind == fb::JoinType_Right)
                         ? out_of_bounds_policy::NULLIFY
                         : out_of_bounds_policy::DONT_CHECK;

  auto n = static_cast<cudf::size_type>(left_indices->size());
  cudf::column_view left_idx_col{cudf::data_type{cudf::type_id::INT32},
                                  n, left_indices->data(),
                                  nullptr, 0, 0, {}};
  cudf::column_view right_idx_col{cudf::data_type{cudf::type_id::INT32},
                                   n, right_indices->data(),
                                   nullptr, 0, 0, {}};
  auto left_gathered = cudf::gather(ltv, left_idx_col, left_policy);
  auto right_gathered = cudf::gather(rtv, right_idx_col, right_policy);

  // Concatenate columns: [left_cols..., right_cols...].
  std::vector<std::unique_ptr<cudf::column>> all_cols;
  std::vector<std::string> all_names;

  auto lgv = left_gathered->view();
  for (cudf::size_type i = 0; i < lgv.num_columns(); ++i) {
    all_cols.push_back(std::make_unique<cudf::column>(lgv.column(i)));
    all_names.push_back(left.column_names[i]);
  }
  auto rgv = right_gathered->view();
  for (cudf::size_type i = 0; i < rgv.num_columns(); ++i) {
    all_cols.push_back(std::make_unique<cudf::column>(rgv.column(i)));
    all_names.push_back(right.column_names[i]);
  }

  auto full_table = std::make_unique<cudf::table>(std::move(all_cols));

  // Residual (non-equi) join filter: DataFusion attaches a predicate the
  // equijoin can't express (e.g. q17's `l_quantity < 0.2 * avg`). It's
  // serialized verbatim, with its ColumnRefs indexing the filter's intermediate
  // schema; `filter_columns` maps intermediate column i to the (side, index) in
  // the join inputs. Build that intermediate view over the gathered
  // [left_cols..., right_cols...] table, evaluate the filter, and drop failing
  // rows before applying the output projection.
  if (join->filter()) {
    auto left_width = static_cast<cudf::size_type>(left.table->num_columns());
    std::vector<cudf::column_view> inter_cols;
    if (join->filter_columns()) {
      for (const auto* fc : *join->filter_columns()) {
        cudf::size_type combined =
            fc->side() == fb::JoinSide_Right
                ? left_width + static_cast<cudf::size_type>(fc->index())
                : static_cast<cudf::size_type>(fc->index());
        inter_cols.push_back(full_table->view().column(combined));
      }
    }
    cudf::table_view inter{inter_cols};
    auto mask = build_column(join->filter(), inter);
    full_table = cudf::apply_boolean_mask(full_table->view(), mask->view());
  }

  // Apply output projection if present.
  if (join->projection() && join->projection()->size() > 0) {
    auto ftv = full_table->view();
    std::vector<std::unique_ptr<cudf::column>> proj_cols;
    std::vector<std::string> proj_names;
    for (auto idx : *join->projection()) {
      proj_cols.push_back(std::make_unique<cudf::column>(ftv.column(idx)));
      proj_names.push_back(all_names[idx]);
    }
    return {std::make_unique<cudf::table>(std::move(proj_cols)),
            std::move(proj_names)};
  }

  return {std::move(full_table), std::move(all_names)};
}

// ============================================================================
// GpuCrossJoin — cartesian product
// ============================================================================

static TableResult execute_cross_join(const fb::GpuCrossJoin* join) {
  auto left = execute_node(join->left());
  auto right = execute_node(join->right());

  auto out = cudf::cross_join(left.table->view(), right.table->view());
  std::vector<std::string> names = std::move(left.column_names);
  names.insert(names.end(), right.column_names.begin(), right.column_names.end());
  return {std::move(out), std::move(names)};
}

// ============================================================================
// GpuNestedLoopJoin — cross product filtered by a non-equi predicate
// ============================================================================

static TableResult execute_nested_loop_join(const fb::GpuNestedLoopJoin* join) {
  auto jt = join->join_type();
  if (jt != fb::JoinType_Inner && jt != fb::JoinType_Left)
    throw std::runtime_error(
        "GpuNestedLoopJoin: only Inner/Left join types supported (got " +
        std::to_string(jt) + ")");

  auto left = execute_node(join->left());
  auto right = execute_node(join->right());
  auto ltv = left.table->view();
  auto rtv = right.table->view();

  std::vector<std::string> all_names = left.column_names;
  all_names.insert(all_names.end(), right.column_names.begin(),
                   right.column_names.end());

  std::unique_ptr<cudf::table> full_table;
  if (!join->filter()) {
    // Unconditional NestedLoopJoin = cartesian product: every left row pairs
    // with every right row. For a LEFT join this equals the cross product only
    // when the right side is non-empty — an empty right would have to emit each
    // left row once with null right columns, but cross_join yields zero rows.
    // The only source of an unconditional LEFT NLJ is a decorrelated scalar
    // subquery whose (group-by-less) aggregate always returns exactly one row,
    // so we assert that invariant rather than special-casing empty-right.
    if (jt == fb::JoinType_Left && rtv.num_rows() == 0)
      throw std::runtime_error(
          "unconditional LEFT NestedLoopJoin with an empty right side is "
          "unsupported (cross_join would drop all left rows); expected a "
          "single-row scalar aggregate on the right");
    full_table = cudf::cross_join(ltv, rtv);
  } else if (!is_ast_able(
                 join->filter(),
                 // Type-only view of the referenced columns in filter_columns
                 // order: ColumnRef(i) -> filter_columns[i] -> a left/right
                 // column. is_ast_able only inspects types, but a table_view
                 // requires equal column sizes — and left/right differ — so use
                 // zero-row slices (which preserve type) to make them uniform.
                 [&] {
                   std::vector<cudf::column_view> cols;
                   for (flatbuffers::uoffset_t i = 0;
                        i < join->filter_columns()->size(); ++i) {
                     auto* fc = join->filter_columns()->Get(i);
                     auto src = fc->side() == fb::JoinSide_Right
                                    ? rtv.column(fc->index())
                                    : ltv.column(fc->index());
                     cols.push_back(cudf::slice(src, {0, 0}).front());
                   }
                   return cudf::table_view{cols};
                 }())) {
    // Filter isn't expressible in the cuDF AST (e.g. a CAST to Decimal128, as in
    // TPC-H q11/q22) so conditional_*_join can't evaluate it. Fall back to the
    // column path: materialise the full cross product, evaluate the predicate as
    // a boolean column, and apply it as a mask. Only Inner is handled — a LEFT
    // join would additionally have to re-emit unmatched left rows with null
    // right columns, which the mask can't express.
    if (jt != fb::JoinType_Inner)
      throw std::runtime_error(
          "non-AST-able NestedLoopJoin filter is only supported for Inner joins");
    if (!join->filter_columns())
      throw std::runtime_error(
          "GpuNestedLoopJoin has a filter but no filter_columns map");
    auto crossed = cudf::cross_join(ltv, rtv);
    auto cv = crossed->view();
    // build_column resolves ColumnRef(i) directly against column i of the table
    // it is given, so arrange the cross-product columns in filter_columns order:
    // left columns occupy [0, L), right columns [L, L+R).
    auto left_ncols = ltv.num_columns();
    std::vector<cudf::column_view> mask_cols;
    for (flatbuffers::uoffset_t i = 0; i < join->filter_columns()->size(); ++i) {
      auto* fc = join->filter_columns()->Get(i);
      mask_cols.push_back(fc->side() == fb::JoinSide_Right
                              ? cv.column(left_ncols + fc->index())
                              : cv.column(fc->index()));
    }
    cudf::table_view mask_src{mask_cols};
    auto mask = build_column(join->filter(), mask_src);
    full_table = cudf::apply_boolean_mask(cv, mask->view());
  } else {
    // Build the predicate as a cuDF AST over the two tables — build_expr maps
    // each ColumnRef to table_reference::LEFT/RIGHT via filter_columns — and run
    // a conditional join, which evaluates the predicate per (left,right) pair.
    if (!join->filter_columns())
      throw std::runtime_error(
          "GpuNestedLoopJoin has a filter but no filter_columns map");
    ExprContext ctx;
    const auto& pred = build_expr(join->filter(), ctx, join->filter_columns());

    auto [left_indices, right_indices] =
        jt == fb::JoinType_Left ? cudf::conditional_left_join(ltv, rtv, pred)
                                : cudf::conditional_inner_join(ltv, rtv, pred);

    // For a LEFT join, unmatched left rows carry an out-of-bounds right index;
    // NULLIFY turns those into nulls. All left indices are in-bounds.
    using cudf::out_of_bounds_policy;
    auto right_policy = (jt == fb::JoinType_Left)
                            ? out_of_bounds_policy::NULLIFY
                            : out_of_bounds_policy::DONT_CHECK;
    auto n = static_cast<cudf::size_type>(left_indices->size());
    cudf::column_view left_idx_col{cudf::data_type{cudf::type_id::INT32}, n,
                                   left_indices->data(), nullptr, 0, 0, {}};
    cudf::column_view right_idx_col{cudf::data_type{cudf::type_id::INT32}, n,
                                    right_indices->data(), nullptr, 0, 0, {}};
    auto left_gathered =
        cudf::gather(ltv, left_idx_col, out_of_bounds_policy::DONT_CHECK);
    auto right_gathered = cudf::gather(rtv, right_idx_col, right_policy);

    // Concatenate columns: [left_cols..., right_cols...].
    std::vector<std::unique_ptr<cudf::column>> all_cols;
    auto lgv = left_gathered->view();
    for (cudf::size_type i = 0; i < lgv.num_columns(); ++i)
      all_cols.push_back(std::make_unique<cudf::column>(lgv.column(i)));
    auto rgv = right_gathered->view();
    for (cudf::size_type i = 0; i < rgv.num_columns(); ++i)
      all_cols.push_back(std::make_unique<cudf::column>(rgv.column(i)));
    full_table = std::make_unique<cudf::table>(std::move(all_cols));
  }

  // Apply output projection if present.
  if (join->projection() && join->projection()->size() > 0) {
    auto ftv = full_table->view();
    std::vector<std::unique_ptr<cudf::column>> proj_cols;
    std::vector<std::string> proj_names;
    for (auto idx : *join->projection()) {
      proj_cols.push_back(std::make_unique<cudf::column>(ftv.column(idx)));
      proj_names.push_back(all_names[idx]);
    }
    return {std::make_unique<cudf::table>(std::move(proj_cols)),
            std::move(proj_names)};
  }

  return {std::move(full_table), std::move(all_names)};
}

// ============================================================================
// GpuSort — sort by expressions
// ============================================================================

static TableResult execute_sort(const fb::GpuSort* sort) {
  auto input = execute_node(sort->input());
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

// ============================================================================
// Union (UNION ALL / interleave): concatenate the rows of all inputs
// ============================================================================

static TableResult execute_union(const fb::GpuUnion* u) {
  if (!u->inputs() || u->inputs()->size() == 0)
    throw std::runtime_error("GpuUnion has no inputs");

  // Execute each input fully, then concatenate the materialized tables.
  std::vector<TableResult> inputs;
  inputs.reserve(u->inputs()->size());
  for (flatbuffers::uoffset_t i = 0; i < u->inputs()->size(); ++i) {
    inputs.push_back(execute_node(u->inputs()->Get(i)));
  }

  // A single input needs no copy.
  if (inputs.size() == 1) return std::move(inputs[0]);

  // Each branch is planned independently, so a column can land a different cuDF
  // type per branch even though DataFusion declares one union output type: q5's
  // inner UNION pairs a real decimal measure in one branch against a `0` literal
  // (materialized as FLOAT64) at the same position in the other, and cuDF's SUM
  // also drifts the fixed_point scale per branch. cudf::concatenate requires
  // identical column types, so cast every branch column to the union's declared
  // output type (id + decimal scale) before stacking.
  if (u->output_schema() && u->output_schema()->fields()) {
    auto* fields = u->output_schema()->fields();
    for (auto& in : inputs) {
      auto cols = in.table->release();
      auto n = std::min<std::size_t>(cols.size(), fields->size());
      for (std::size_t c = 0; c < n; ++c) {
        auto* f = fields->Get(static_cast<flatbuffers::uoffset_t>(c));
        auto want_id = fb_to_type_id(f->data_type());
        cudf::data_type want =
            (f->data_type() == fb::DataType_Decimal128)
                ? cudf::data_type{want_id, -static_cast<int32_t>(f->decimal_scale())}
                : cudf::data_type{want_id};
        // STRING/EMPTY aren't producible by cudf::cast and already agree across
        // branches; only retype numeric/decimal columns that actually differ.
        if (want_id != cudf::type_id::STRING && want_id != cudf::type_id::EMPTY &&
            cols[c]->type() != want) {
          cols[c] = cudf::cast(cols[c]->view(), want);
        }
      }
      in.table = std::make_unique<cudf::table>(std::move(cols));
    }
  }

  std::vector<cudf::table_view> views;
  views.reserve(inputs.size());
  for (auto& in : inputs) views.push_back(in.table->view());

  auto out = cudf::concatenate(views);
  return {std::move(out), std::move(inputs[0].column_names)};
}

// ============================================================================
// Limit (LIMIT / OFFSET): slice rows [skip, skip + fetch)
// ============================================================================

static TableResult execute_limit(const fb::GpuLimit* limit) {
  auto input = execute_node(limit->input());
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

// ============================================================================
// GpuWindow — window functions (append one column per window expr)
// ============================================================================

static std::unique_ptr<cudf::rolling_aggregation> make_rolling_agg(
    const std::string& func_name, cudf::null_policy count_nulls) {
  if (func_name == "sum" || func_name == "SUM")
    return cudf::make_sum_aggregation<cudf::rolling_aggregation>();
  if (func_name == "min" || func_name == "MIN")
    return cudf::make_min_aggregation<cudf::rolling_aggregation>();
  if (func_name == "max" || func_name == "MAX")
    return cudf::make_max_aggregation<cudf::rolling_aggregation>();
  if (func_name == "avg" || func_name == "AVG" || func_name == "mean" ||
      func_name == "MEAN")
    return cudf::make_mean_aggregation<cudf::rolling_aggregation>();
  if (func_name == "count" || func_name == "COUNT")
    return cudf::make_count_aggregation<cudf::rolling_aggregation>(count_nulls);
  throw std::runtime_error("unsupported window function: " + func_name);
}

static TableResult execute_window(const fb::GpuWindow* win) {
  auto input = execute_node(win->input());
  auto tv = input.table->view();

  // Output = all input columns (in order) followed by one column per window expr.
  // The input arrives pre-sorted by [partition_by, order_by] (DataFusion's
  // SortExec), so consecutive equal partition keys form one group, and
  // grouped_rolling_window preserves input row order.
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  std::vector<std::string> out_names;
  for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
    out_cols.push_back(std::make_unique<cudf::column>(tv.column(i)));
    out_names.push_back(input.column_names[i]);
  }

  if (win->window_exprs()) {
    for (flatbuffers::uoffset_t i = 0; i < win->window_exprs()->size(); ++i) {
      auto* we = win->window_exprs()->Get(i);
      std::string fname = we->func_name() ? we->func_name()->str() : "";

      // Partition key table (the group keys). build_column handles ColumnRef and
      // computed partition exprs (e.g. CASE in a ROLLUP grouping key).
      std::vector<std::unique_ptr<cudf::column>> key_owned;
      std::vector<cudf::column_view> key_views;
      if (we->partition_by()) {
        for (flatbuffers::uoffset_t k = 0; k < we->partition_by()->size(); ++k) {
          key_owned.push_back(build_column(we->partition_by()->Get(k), tv));
          key_views.push_back(key_owned.back()->view());
        }
      }
      cudf::table_view keys{key_views};

      // Argument column. count() may have no args (COUNT(*)); use the first
      // column as a placeholder since COUNT(*) counts every row.
      std::unique_ptr<cudf::column> arg_owned;
      cudf::column_view arg_view;
      bool has_arg = we->args() && we->args()->size() > 0;
      if (has_arg) {
        arg_owned = build_column(we->args()->Get(0), tv);
        arg_view = arg_owned->view();
      } else {
        arg_view = tv.column(0);
      }

      // avg over a decimal: cuDF's mean keeps the input scale while DataFusion
      // boosts it to s+4. Cast the input up to the declared output scale first so
      // the averaged value matches (mirrors execute_aggregate).
      bool is_avg = (fname == "avg" || fname == "AVG" || fname == "mean" ||
                     fname == "MEAN");
      if (is_avg && we->out_decimal_precision() != 0 &&
          arg_view.type().id() == cudf::type_id::DECIMAL128) {
        int32_t want_exp = -static_cast<int32_t>(we->out_decimal_scale());
        if (arg_view.type().scale() != want_exp) {
          arg_owned = cudf::cast(
              arg_view, cudf::data_type{cudf::type_id::DECIMAL128, want_exp});
          arg_view = arg_owned->view();
        }
      }

      // Frame: start is always UNBOUNDED PRECEDING; end is CURRENT ROW (running)
      // or UNBOUNDED FOLLOWING (whole partition).
      auto preceding = cudf::window_bounds::unbounded();
      auto following = (we->frame_end() == fb::WindowFrameBound_UnboundedFollowing)
                           ? cudf::window_bounds::unbounded()
                           : cudf::window_bounds::get(0);

      // COUNT(*) counts every row, so the placeholder column's nulls must be
      // included; COUNT(col) counts non-null values (cuDF's EXCLUDE default).
      auto count_nulls = has_arg ? cudf::null_policy::EXCLUDE
                                 : cudf::null_policy::INCLUDE;
      auto agg = make_rolling_agg(fname, count_nulls);
      auto col = cudf::grouped_rolling_window(keys, arg_view, preceding, following,
                                              /*min_periods=*/1, *agg);

      out_cols.push_back(std::move(col));
      out_names.push_back(we->alias() ? we->alias()->str()
                                      : ("window" + std::to_string(i)));
    }
  }

  auto result = std::make_unique<cudf::table>(std::move(out_cols));
  return {std::move(result), std::move(out_names)};
}

// ============================================================================
// Pass-through nodes (single-GPU: just execute input)
// ============================================================================

static TableResult execute_passthrough(const fb::PlanNode* input_node) {
  return execute_node(input_node);
}

// ============================================================================
// Plan node dispatcher
// ============================================================================

static const char* plan_node_kind_name(fb::PlanNodeKind k) {
  switch (k) {
    case fb::PlanNodeKind_GpuScan:                return "GpuScan";
    case fb::PlanNodeKind_GpuFilter:              return "GpuFilter";
    case fb::PlanNodeKind_GpuProject:             return "GpuProject";
    case fb::PlanNodeKind_GpuAggregate:           return "GpuAggregate";
    case fb::PlanNodeKind_GpuHashJoin:            return "GpuHashJoin";
    case fb::PlanNodeKind_GpuCrossJoin:           return "GpuCrossJoin";
    case fb::PlanNodeKind_GpuNestedLoopJoin:      return "GpuNestedLoopJoin";
    case fb::PlanNodeKind_GpuSort:                return "GpuSort";
    case fb::PlanNodeKind_GpuCoalesceBatches:     return "GpuCoalesceBatches";
    case fb::PlanNodeKind_GpuCoalescePartitions:  return "GpuCoalescePartitions";
    case fb::PlanNodeKind_GpuRepartition:         return "GpuRepartition";
    case fb::PlanNodeKind_GpuSortPreservingMerge: return "GpuSortPreservingMerge";
    case fb::PlanNodeKind_GpuUnion:               return "GpuUnion";
    case fb::PlanNodeKind_GpuLimit:               return "GpuLimit";
    case fb::PlanNodeKind_GpuWindow:              return "GpuWindow";
    default:                                       return "Unknown";
  }
}

static TableResult execute_node(const fb::PlanNode* node) {
  if (!node) throw std::runtime_error("null PlanNode");

  const char* kind = plan_node_kind_name(node->node_type());
  PCK_TRACE("enter %s", kind);

  TableResult result;
  try {
    switch (node->node_type()) {
      case fb::PlanNodeKind_GpuScan:
        result = execute_scan(node->node_as_GpuScan()); break;
      case fb::PlanNodeKind_GpuFilter:
        result = execute_filter(node->node_as_GpuFilter()); break;
      case fb::PlanNodeKind_GpuProject:
        result = execute_project(node->node_as_GpuProject()); break;
      case fb::PlanNodeKind_GpuAggregate:
        result = execute_aggregate(node->node_as_GpuAggregate()); break;
      case fb::PlanNodeKind_GpuHashJoin:
        result = execute_hash_join(node->node_as_GpuHashJoin()); break;
      case fb::PlanNodeKind_GpuCrossJoin:
        result = execute_cross_join(node->node_as_GpuCrossJoin()); break;
      case fb::PlanNodeKind_GpuNestedLoopJoin:
        result = execute_nested_loop_join(node->node_as_GpuNestedLoopJoin()); break;
      case fb::PlanNodeKind_GpuSort:
        result = execute_sort(node->node_as_GpuSort()); break;
      case fb::PlanNodeKind_GpuCoalesceBatches:
        result = execute_passthrough(node->node_as_GpuCoalesceBatches()->input()); break;
      case fb::PlanNodeKind_GpuCoalescePartitions:
        result = execute_passthrough(node->node_as_GpuCoalescePartitions()->input()); break;
      case fb::PlanNodeKind_GpuRepartition:
        result = execute_passthrough(node->node_as_GpuRepartition()->input()); break;
      case fb::PlanNodeKind_GpuSortPreservingMerge:
        result = execute_passthrough(node->node_as_GpuSortPreservingMerge()->input()); break;
      case fb::PlanNodeKind_GpuUnion:
        result = execute_union(node->node_as_GpuUnion()); break;
      case fb::PlanNodeKind_GpuLimit:
        result = execute_limit(node->node_as_GpuLimit()); break;
      case fb::PlanNodeKind_GpuWindow:
        result = execute_window(node->node_as_GpuWindow()); break;
      default:
        throw std::runtime_error(
            "unsupported PlanNodeKind: " + std::to_string(node->node_type()));
    }
  } catch (const std::exception& e) {
    std::string msg = e.what();
    if (msg.find("[in ") == std::string::npos) {
      throw std::runtime_error(std::string("[in ") + kind + "] " + msg);
    }
    throw;
  }

  debug_sync(kind);
  if (debug_enabled()) {
    auto tv = result.table->view();
    PCK_TRACE("leave %s rows=%d cols=%d", kind, tv.num_rows(), tv.num_columns());
  }
  return result;
}

// ============================================================================
// Public API
// ============================================================================

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

  return execute_node(root);
}

}  // namespace peacock
