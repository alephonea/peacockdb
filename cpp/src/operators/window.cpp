// CudfWindow -- window functions (append one column per window expr).

#include "peacock/operators.h"
#include "peacock/expr.h"

#include <cudf/rolling.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace peacock {

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

TableResult execute_window(const fb::CudfWindow* win, NodeInputs* in) {
  auto input = execute_node(win->input(), in);
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


}  // namespace peacock
