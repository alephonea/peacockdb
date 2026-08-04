// Union (UNION ALL / interleave) -- concatenate the rows of all inputs.

#include "peacock/operators.h"
#include "peacock/expr.h"

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace peacock {

TableResult execute_union(const fb::GpuUnion* u, NodeInputs* in) {
  if (!u->inputs() || u->inputs()->size() == 0)
    throw std::runtime_error("GpuUnion has no inputs");

  // Execute each input fully, then concatenate the materialized tables.
  std::vector<TableResult> inputs;
  inputs.reserve(u->inputs()->size());
  for (flatbuffers::uoffset_t i = 0; i < u->inputs()->size(); ++i) {
    inputs.push_back(execute_node(u->inputs()->Get(i), in));
  }

  // A single input needs no copy.
  if (inputs.size() == 1) return std::move(inputs[0]);

  // Branches are planned independently, so one column can land a different cuDF
  // type per branch despite the single declared union output type (q5 pairs a
  // decimal measure against a `0` literal materialized as FLOAT64; cuDF's SUM also
  // drifts fixed_point scale per branch). cudf::concatenate requires identical
  // types, so retype every branch column to the declared output first.
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


}  // namespace peacock
