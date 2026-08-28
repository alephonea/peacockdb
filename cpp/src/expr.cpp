// Expression evaluation: the AST fast path (build_expr) and the column-producing
// path (build_column), plus the shared debug/trace definitions.

#include "peacock/expr.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/filling.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>

namespace peacock {

// Debug instrumentation (PEACOCK_GPU_DEBUG=1): traces each plan node/expression
// and syncs the default cuDF stream after each step, so async CUDA errors surface
// at the call site instead of cascading several ops later.

bool debug_enabled() {
  static const bool e = []() {
    const char* v = std::getenv("PEACOCK_GPU_DEBUG");
    return v && v[0] && v[0] != '0';
  }();
  return e;
}


// No-op unless debug is on; otherwise sync + check, to localize async errors.
void debug_sync(const char* tag) {
  if (!debug_enabled()) return;
  auto err = cudaStreamSynchronize(cudf::get_default_stream().value());
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[peacock] CUDA sync after %s: %s\n",
                 tag, cudaGetErrorString(err));
    throw std::runtime_error(std::string("CUDA error after ") + tag +
                             ": " + cudaGetErrorString(err));
  }
}

// TU-private forward declarations: the expression builders below are mutually
// recursive. (Cross-TU declarations live in peacock/expr.h.)
static bool is_predicate_op(fb::BinaryOp op);
static cudf::type_id infer_expr_type(const fb::Expr* expr,
                                     cudf::table_view const& table);
static std::unique_ptr<cudf::column> build_column_binary(
    const fb::Expr* expr, cudf::table_view const& table);


cudf::type_id fb_to_type_id(fb::DataType dt) {
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
    // MUST be NULL_LOGICAL_*, not LOGICAL_*: SQL three-valued logic says
    // `TRUE OR NULL` = TRUE, but plain LOGICAL_OR propagates the null and thus
    // silently DROPS rows a disjunctive predicate should keep (TPC-DS
    // q7/q15/q26/q79). Reduces to LOGICAL_* when both operands are non-null.
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

cudf::ast::expression& build_expr(const fb::Expr* expr, ExprContext& ctx,
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
      // The same flag `build_scalar` reads below, and it has to be read in both places: a
      // cuDF AST literal carries its scalar's validity, so a typed NULL built valid here is
      // a literal ZERO of that type rather than a null — which is what a Left outer's pad
      // put in its probe columns until the two engines were asked the same question. The
      // string arms have no such bug because cuDF's AST has no string literal and they fall
      // through to `build_scalar`.
      const bool valid = !sv->is_null();

      switch (sv->type()) {
        case fb::DataType_Boolean: {
          auto s = std::make_unique<cudf::numeric_scalar<bool>>(
              sv->bool_val(), valid);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Int8: {
          auto s = std::make_unique<cudf::numeric_scalar<int8_t>>(
              static_cast<int8_t>(sv->int_val()), valid);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Int16: {
          auto s = std::make_unique<cudf::numeric_scalar<int16_t>>(
              static_cast<int16_t>(sv->int_val()), valid);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Int32: {
          auto s = std::make_unique<cudf::numeric_scalar<int32_t>>(
              static_cast<int32_t>(sv->int_val()), valid);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Int64: {
          auto s = std::make_unique<cudf::numeric_scalar<int64_t>>(
              sv->int_val(), valid);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Float32: {
          auto s = std::make_unique<cudf::numeric_scalar<float>>(
              static_cast<float>(sv->float_val()), valid);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Float64: {
          auto s = std::make_unique<cudf::numeric_scalar<double>>(
              sv->float_val(), valid);
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
          auto s = std::make_unique<cudf::numeric_scalar<double>>(dval, valid);
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
              std::string(sv->string_val() ? sv->string_val()->str() : ""), valid);
          auto& ref = *s;
          ctx.scalars.push_back(std::move(s));
          return ctx.keep(std::make_unique<cudf::ast::literal>(ref));
        }
        case fb::DataType_Date32: {
          // Date32 = days since UNIX epoch (int32).
          auto s = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
              cudf::duration_D{static_cast<int32_t>(sv->int_val())}, valid);
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
        case fb::UnaryOp_Sqrt:
          return ctx.keep(std::make_unique<cudf::ast::operation>(
              cudf::ast::ast_operator::SQRT, arg));
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
// cuDF AST has no LIKE / substr / date_part / CASE WHEN. Expressions containing
// those are evaluated outside the AST: each subexpression becomes a
// `cudf::column`, combined via row-wise cuDF APIs. AST-able subtrees still route
// back through `compute_column` for fusion.

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
        // Negative and Sqrt both answer in their operand's type; a stddev's
        // finalize takes the root of a FLOAT64 quotient.
        default:
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
      // cuDF AST never coerces: operands must be the identical type, and it has
      // no decimal support. So any decimal operand or type mismatch goes to the
      // column path, where cudf::binary_operation coerces (and handles
      // fixed_point) natively.
      auto lt = infer_expr_type(b->left(), table);
      auto rt = infer_expr_type(b->right(), table);
      // Un-inferrable operand → column path. Checked BEFORE the lt != rt test,
      // which two EMPTYs would otherwise pass as "AST-able".
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
    // NULL_LOGICAL_* for SQL three-valued logic — see fb_to_ast_op. This path
    // carries OR over string comparisons / IN-lists (q15's `ca_state IN (...)`).
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

std::unique_ptr<cudf::column> build_column(
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
  // Otherwise echo lhs; the heavy arithmetic goes through the AST path anyway.
  (void)rhs;
  return lhs;
}

static std::unique_ptr<cudf::column> build_column_binary(
    const fb::BinaryExprNode* bin, cudf::table_view const& table) {
  auto* lhs = bin->left();
  auto* rhs = bin->right();
  auto op = fb_to_binop(bin->op());

  // Decimal division: cuDF's fixed_point DIV yields scale s_l-s_r (0 for two
  // scale-4 sums → truncation), but DataFusion declares a boosted scale on
  // out_decimal_precision/scale. Pre-scale the numerator to hit it: with output
  // exponent e_o = −out_scale and denominator exponent e_r, set e_l = e_o + e_r.
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

  // round(x [, places]) — evaluate in FLOAT64 with HALF_UP: that is DataFusion's
  // semantics (f64::round, half away from zero), and cudf::round has no
  // DECIMAL128 overload though our scan widens every decimal to DECIMAL128.
  // `places`, when given, must be an integer literal.
  if (name == "round") {
    if (args->size() < 1 || args->size() > 2)
      throw std::runtime_error("round expects 1 or 2 args");
    auto col = build_column(args->Get(0), table);
    int32_t places = 0;
    if (args->size() == 2) {
      auto* e = args->Get(1);
      if (e->node_type() != fb::ExprNode_LiteralExpr)
        throw std::runtime_error("round: decimal places must be a literal");
      places = static_cast<int32_t>(e->node_as_LiteralExpr()->value()->int_val());
    }
    auto fcol = col->type().id() == cudf::type_id::FLOAT64
                    ? std::move(col)
                    : cudf::cast(col->view(),
                                 cudf::data_type{cudf::type_id::FLOAT64});
    return cudf::round(fcol->view(), places, cudf::rounding_method::HALF_UP);
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
  // Search-form CASE only. Value-form (`CASE x WHEN v THEN t END`) survives
  // DataFusion's rewrites and does reach here (only TPC-DS q39, itself disabled
  // for stddev ULP, #54), but a copy_if_else fold on the comparand came back
  // wrong on the GPU, so it is withheld: implement with a GPU test under #57.
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

std::unique_ptr<cudf::column> build_column(
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
        case fb::UnaryOp_Sqrt:
          return cudf::unary_operation(arg->view(), cudf::unary_operator::SQRT);
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
      // Valid by construction rather than by assumption: the guard above refuses a pattern
      // with no `string_val`, and the serializer writes a typed null as `is_null` with none —
      // so anything reaching here is a present string. The ten literal arms read the flag
      // instead; that this one does not is [#200](../../llm-wiki/tickets.md#t200).
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
      // String->string cast is a no-op: cuDF maps every Arrow string variant to
      // the single STRING type, so DataFusion's coercion of two char keys has
      // nothing to convert. cudf::cast has no STRING overload and would throw
      // "Unary cast type must be fixed-width" (#45); a genuine non-string ->
      // STRING conversion isn't producible by cudf::cast at all.
      if (target_id == cudf::type_id::STRING) {
        if (inner->type().id() == cudf::type_id::STRING)
          return inner;
        throw std::runtime_error(
            "cast to STRING from a non-string type not supported in column path");
      }
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


}  // namespace peacock
