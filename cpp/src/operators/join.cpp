// Split out of the former src/plan_executor.cpp monolith.
//
// GpuHashJoin / GpuCrossJoin / GpuNestedLoopJoin.

#include "peacock/operators.h"
#include "peacock/expr.h"

#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>

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

#include <algorithm>
#include <stdexcept>
#include <string>

namespace peacock {

// ============================================================================
// GpuHashJoin — equi-join
// ============================================================================

TableResult execute_hash_join(const fb::GpuHashJoin* join, NodeInputs* in) {
  auto left = execute_node(join->left(), in);
  auto right = execute_node(join->right(), in);

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
  // Per-join NULL key-equality, mirrored from DataFusion's null_equals_null
  // (serialized into the plan). true → NULL keys match (set/INTERSECT, q14);
  // false → NULL keys never match (SQL IN/EXISTS three-valued, q33). Drives the
  // equi and SEMI cuDF null_equality. ANTI/mark stay EQUAL (see below).
  auto join_nulls = join->null_equals_null() ? cudf::null_equality::EQUAL
                                             : cudf::null_equality::UNEQUAL;
  ExprContext semi_ctx;
  const cudf::ast::expression* semi_pred = nullptr;
  if (semi_anti_type && join->filter()) {
    if (!join->filter_columns())
      throw std::runtime_error(
          "semi/anti join has a filter but no filter_columns map");
    semi_pred = &build_expr(join->filter(), semi_ctx, join->filter_columns());
  }

  // NULL semantics by join kind (see #59 semi / #80 anti):
  //  - SEMI (Left/Right) and the EQUI joins use `join_nulls`, mirrored from
  //    DataFusion's per-join `null_equals_null` (serialized in the plan). The two
  //    cases genuinely need OPPOSITE NULL behavior and only the source plan can
  //    tell them apart, so a blanket choice is wrong:
  //      * IN/EXISTS-derived semi (null_equals_null=false → UNEQUAL): SQL
  //        `x IN (...)` excludes NULLs (NULL IN (...,NULL) = UNKNOWN). q33 hit
  //        this — item has one Electronics row with NULL i_manufact_id; under
  //        cuDF's default EQUAL the semi paired NULL↔NULL → a spurious extra
  //        group (708 vs DuckDB/DataFusion's correct 707).
  //      * INTERSECT-derived semi (null_equals_null=true → EQUAL): set semantics
  //        treat NULLs as equal. q14 (cross-channel INTERSECT on
  //        brand/class/category) needs NULL=NULL → 3837; UNEQUAL wrongly drops
  //        the NULL-composite-key rows (3792).
  //    cuDF honors compare_nulls on both the free `left_{semi}_join` and the
  //    newer `filtered_join` (selected at compile time by __has_include; this
  //    build uses the free functions — same compare_nulls contract).
  //  - ANTI (Left/Right) and the mark join below intentionally STAY at EQUAL and
  //    are NOT driven by the flag. Anti is the NOT IN/NOT EXISTS
  //    three-valued-logic trap (`x NOT IN (..., NULL)` is NULL/false for every
  //    x), which neither EQUAL nor UNEQUAL implements on its own; driving it from
  //    the flag could regress. Dedicated NOT IN vs NOT EXISTS semantics + a
  //    demonstrating test + a DuckDB oracle are tracked in issue #80.
  //
  // For Left{Semi,Anti} the right side is the membership/filter; for
  // Right{Semi,Anti} we swap.
  switch (jt) {
    case fb::JoinType_LeftSemi: {
      if (semi_pred) {
        single_indices = cudf::mixed_left_semi_join(
            left_keys, right_keys, ltv, rtv, *semi_pred, join_nulls);
      } else {
#ifdef PEACOCK_HAVE_FILTERED_JOIN
        cudf::filtered_join fj(right_keys, join_nulls,
                               cudf::set_as_build_table::RIGHT, 0.5);
        single_indices = fj.semi_join(left_keys);
#else
        single_indices = cudf::left_semi_join(left_keys, right_keys,
                                              join_nulls);
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
        single_indices = cudf::left_anti_join(left_keys, right_keys,
                                              cudf::null_equality::EQUAL);
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
      cudf::filtered_join fj(left_keys, join_nulls,
                             cudf::set_as_build_table::RIGHT, 0.5);
      single_indices = fj.semi_join(right_keys);
#else
      single_indices = cudf::left_semi_join(right_keys, left_keys,
                                            join_nulls);
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
      single_indices = cudf::left_anti_join(right_keys, left_keys,
                                            cudf::null_equality::EQUAL);
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
    // null_equality::EQUAL kept on purpose here too — see the semi/anti note
    // above (nullable mark-key semantics tracked in issue #59).
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
  // We drive this from DataFusion's per-join null_equals_null (join_nulls):
  // its default false → UNEQUAL restores SQL semantics for inner/left/full/right
  // (the whole passing suite, non-null keys, is unaffected), while a set-semantics
  // join that asks for NULL=NULL gets EQUAL.
  auto kJoinNulls = join_nulls;

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

TableResult execute_cross_join(const fb::GpuCrossJoin* join, NodeInputs* in) {
  auto left = execute_node(join->left(), in);
  auto right = execute_node(join->right(), in);

  auto out = cudf::cross_join(left.table->view(), right.table->view());
  std::vector<std::string> names = std::move(left.column_names);
  names.insert(names.end(), right.column_names.begin(), right.column_names.end());
  return {std::move(out), std::move(names)};
}

// ============================================================================
// GpuNestedLoopJoin — cross product filtered by a non-equi predicate
// ============================================================================

TableResult execute_nested_loop_join(const fb::GpuNestedLoopJoin* join, NodeInputs* in) {
  auto jt = join->join_type();
  if (jt != fb::JoinType_Inner && jt != fb::JoinType_Left)
    throw std::runtime_error(
        "GpuNestedLoopJoin: only Inner/Left join types supported (got " +
        std::to_string(jt) + ")");

  auto left = execute_node(join->left(), in);
  auto right = execute_node(join->right(), in);
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


}  // namespace peacock
