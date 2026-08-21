// CudfAggregate -- group-by and scalar (whole-table) aggregation.

#include "peacock/operators.h"
#include "peacock/expr.h"

#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/unary.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace peacock {

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
static bool is_var_name(const std::string& f) {
  return f == "var" || f == "VAR" || f == "var_samp" || f == "VAR_SAMP" ||
         f == "var_pop" || f == "VAR_POP" || f == "variance" || f == "VARIANCE";
}
// ddof (delta degrees of freedom) for the STDDEV/VAR divisor n-ddof: population
// variants (STDDEV_POP/VAR_POP) use ddof=0 (divisor n), the sample default
// (STDDEV/STDDEV_SAMP/VAR/VAR_SAMP/VARIANCE) uses ddof=1 (divisor n-1). Matches
// DataFusion (StddevPop/VarPop = population; the rest = sample).
static cudf::size_type stddev_ddof(const std::string& f) {
  return (f == "stddev_pop" || f == "STDDEV_POP" || f == "var_pop" || f == "VAR_POP")
             ? 0
             : 1;
}

// Aggregate execution phase. MUST stay three-way: the `is_final` bool below
// collapses Single and Partial, which two-phase AVG state has to tell apart.
//   Single  = one pass over raw rows                  -> AVG = plain MEAN (1 col)
//   Partial = first pass, emits per-partition STATE   -> AVG = [sum, count] (2 cols)
//   Final   = merges partial STATE across the shuffle -> AVG = Σsum / Σcount
// SUM/COUNT/MIN/MAX are phase-insensitive except count-Final -> sum.
//   Merge   = merges partial STATE and emits STATE, not a finished value. The
//             batch-partitioned mode merges once per lane and again across lanes,
//             and finalizes in a project of its own, so the merge and the finalize
//             Final couples are separate here.
enum class AggPhase { Single, Partial, Final, Merge };
static AggPhase agg_phase(fb::AggregateMode m) {
  switch (m) {
    case fb::AggregateMode_Partial:
      return AggPhase::Partial;
    case fb::AggregateMode_Final:
    case fb::AggregateMode_FinalPartitioned:
      return AggPhase::Final;
    case fb::AggregateMode_Single:
    case fb::AggregateMode_SinglePartitioned:
      return AggPhase::Single;
    case fb::AggregateMode_Merge:
      return AggPhase::Merge;
    default:
      throw std::runtime_error("agg_phase: unsupported AggregateMode");
  }
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
    // Valid ONLY while Partial output is one row per key (CudfRepartition as
    // passthrough): the Final regroup is then a MEAN-of-singleton identity.
    // Multi-partition repartition breaks it (mean-of-means) — execute_aggregate
    // guards at runtime; decomposing AVG into SUM+COUNT lifts the restriction,
    // ticket #25 (llm-wiki/tickets.md).
    return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  }
  if (is_stddev_name(func_name) || is_var_name(func_name)) {
    // SINGLE-PARTITION stddev/var only. Partial computes the real per-group
    // std/var; with passthrough repartition that is one row per key, so Final is
    // a singleton identity (MEAN of the lone partial row), exactly as AVG. The
    // grouped guard in execute_aggregate fails loudly on a real multi-row merge;
    // real cross-partition merging takes the 3-col Welford M2 path instead
    // (mergeable_agg_state), NOT this call. #25.
    if (is_final)
      return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    if (is_var_name(func_name))
      return cudf::make_variance_aggregation<cudf::groupby_aggregation>(
          stddev_ddof(func_name));
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

TableResult execute_aggregate(const fb::CudfAggregate* agg, NodeInputs* in) {
  auto input = execute_node(agg->input(), in);
  auto tv = input.table->view();

  bool is_final = (agg->mode() == fb::AggregateMode_Final ||
                   agg->mode() == fb::AggregateMode_FinalPartitioned);

  // make_agg would silently compute the NON-distinct value for a DISTINCT
  // aggregate (needs cuDF nunique/distinct, unimplemented), while the CPU oracle
  // honours the flag — a silent divergence, so fail loudly. Unreachable today:
  // DataFusion rewrites a standalone count(DISTINCT x) into GROUP BY + count; the
  // flag only survives when DISTINCT is mixed with other aggregates (#62).
  if (agg->aggr_funcs()) {
    for (flatbuffers::uoffset_t i = 0; i < agg->aggr_funcs()->size(); ++i) {
      if (agg->aggr_funcs()->Get(i)->distinct())
        throw std::runtime_error(
            "DISTINCT aggregate (e.g. count(DISTINCT)) not yet supported on the "
            "GPU when the flag survives to the executor (mixed with other "
            "aggregates); needs cuDF nunique/distinct aggregations — see #62");
    }
  }

  // Build group-by keys.
  std::vector<cudf::size_type> key_indices;
  std::vector<std::string> key_names;
  if (agg->group_exprs()) {
    for (flatbuffers::uoffset_t i = 0; i < agg->group_exprs()->size(); ++i) {
      auto* expr = agg->group_exprs()->Get(i);
      if (expr->node_type() != fb::ExprNode_ColumnRef)
        throw std::runtime_error("CudfAggregate: only ColumnRef group exprs supported");
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

  // Helper to determine the values column for a function node. `agg_idx` is the
  // aggregate's positional index, used only for the Final-stage case.
  auto get_values_col = [&](const fb::AggregateFuncNode* func,
                            size_t agg_idx) -> cudf::column_view {
    cudf::column_view base;
    if (is_final) {
      // Final's input is the Partial stage's OUTPUT (group keys + one accumulator
      // per aggregate), so func->args() — which index the original input — are
      // meaningless here. Resolve positionally: aggregate columns sit right after
      // the group keys (whose count already includes __grouping_id for a
      // grouping-set Final).
      base = tv.column(
          static_cast<cudf::size_type>(key_indices.size() + agg_idx));
    } else if (func->args() && func->args()->size() > 0) {
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

        cudf::column_view values_col = get_values_col(func, i);
        bool is_count = (name == "count" || name == "COUNT");

        bool is_avg = is_avg_name(name);
        bool is_std = is_stddev_name(name);
        // Same guard as the grouped path: a Final-stage global AVG/STDDEV reduces
        // over the Partial outputs, and with passthrough repartition there is
        // exactly one partial row (identity). More than one row means a
        // multi-partition merge → silently-wrong mean-of-means / std-of-stds.
        // Decomposing AVG into SUM+COUNT lifts this: ticket #25 (llm-wiki/tickets.md).
        if (is_final && (is_avg || is_std) && values_col.size() > 1) {
          throw std::runtime_error(
              "Final-stage AVG/STDDEV merged multiple partial rows "
              "(mean-of-means is wrong); must be decomposed into "
              "additive state before multi-partition GPU repartition is enabled");
        }

        bool is_sum = (name == "sum" || name == "SUM");
        std::unique_ptr<cudf::column> result_col;
        if (is_count) {
          if (is_final) {
            // #100: a GLOBAL count at Final must SUM the per-partition partial
            // counts (one per row of values_col), NOT re-count the partial rows —
            // else 8-way gives 8. Mirrors the grouped count→sum-at-Final. The
            // partial count column is INT64; reduce-sum to INT64.
            auto ragg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
            auto s = cudf::reduce(values_col, *ragg,
                                  cudf::data_type{cudf::type_id::INT64});
            result_col = cudf::make_column_from_scalar(*s, 1);
          } else {
            // Partial/Single: count the actual (non-null) rows.
            // Avoid make_count_aggregation<reduce_aggregation> which is not
            // exported in all cudf versions. Count = size - null_count.
            int64_t cnt = static_cast<int64_t>(values_col.size()) -
                          static_cast<int64_t>(values_col.null_count());
            cudf::numeric_scalar<int64_t> s(cnt, true);
            result_col = cudf::make_column_from_scalar(s, 1);
          }
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

  // ---- ROLLUP / CUBE / GROUPING SETS ----
  // DataFusion's Partial expands these into N grouping sets and appends a
  // `__grouping_id` column after the group columns; the Final then re-groups by
  // [group cols..., __grouping_id] as a plain GROUP BY and the outer projection
  // drops the id. cuDF has no native ROLLUP/CUBE, so per set we substitute the
  // per-position NULL placeholder (null_exprs[i]) for masked group columns, run
  // the same aggregations, tag rows with a distinct id, and concatenate.
  //
  // NON-EMPTY null_exprs is the discriminator, not grouping_sets: a plain GROUP BY
  // — and the Final of a grouping-set agg — still serializes one all-false mask
  // but leaves null_exprs EMPTY, and must take the single-groupby path below.
  if (agg->null_exprs() && agg->null_exprs()->size() > 0 &&
      agg->grouping_sets() && agg->grouping_sets()->size() > 0) {
    auto* sets = agg->grouping_sets();
    auto nkeys = static_cast<cudf::size_type>(key_cols.size());
    if (agg->null_exprs()->size() != static_cast<flatbuffers::uoffset_t>(nkeys))
      throw std::runtime_error(
          "grouping sets: null_exprs length != group_exprs");

    // NULL placeholder column per group position (full input length): a masked
    // position contributes an all-NULL key of the matching type, which is
    // concatenate-compatible with the real column for that position.
    std::vector<std::unique_ptr<cudf::column>> null_placeholders;
    null_placeholders.reserve(nkeys);
    for (cudf::size_type i = 0; i < nkeys; ++i)
      null_placeholders.push_back(build_column(agg->null_exprs()->Get(i), tv));

    // Aggregate value columns + metadata — identical for every set. Reserve
    // computed_args up front so the views get_values_col returns stay valid as
    // it fills the vector across funcs (no reallocation).
    std::vector<cudf::column_view> gs_values;
    std::vector<std::string> gs_func_names;
    std::vector<std::string> gs_agg_names;
    std::vector<bool> gs_is_count;
    if (agg->aggr_funcs()) {
      computed_args.reserve(agg->aggr_funcs()->size() * 2 + 4);
      for (flatbuffers::uoffset_t i = 0; i < agg->aggr_funcs()->size(); ++i) {
        auto* func = agg->aggr_funcs()->Get(i);
        std::string name = func->name() ? func->name()->str() : "count";
        gs_values.push_back(get_values_col(func, i));
        gs_func_names.push_back(name);
        gs_agg_names.push_back(func->alias() ? func->alias()->str() : name);
        gs_is_count.push_back(name == "count" || name == "COUNT");
      }
    }

    std::vector<std::unique_ptr<cudf::table>> set_tables;
    set_tables.reserve(sets->size());
    for (flatbuffers::uoffset_t s = 0; s < sets->size(); ++s) {
      auto* mask = sets->Get(s)->values();
      if (!mask || mask->size() != static_cast<flatbuffers::uoffset_t>(nkeys))
        throw std::runtime_error("grouping set mask length != group_exprs length");

      std::vector<cudf::column_view> set_keys;
      set_keys.reserve(nkeys);
      // gid only has to be DISTINCT per set (so Final keeps sets apart when a
      // placeholder NULL collides with a natural NULL). It does NOT match
      // DataFusion's __grouping_id bit convention; unobservable while no enabled
      // query projects/orders by GROUPING(col). Make it match before one does (#65).
      int32_t gid = 0;
      for (cudf::size_type i = 0; i < nkeys; ++i) {
        if (mask->Get(i)) {  // masked -> NULL placeholder; record the bit
          set_keys.push_back(null_placeholders[i]->view());
          gid |= (1 << i);
        } else {
          set_keys.push_back(key_cols[i]);
        }
      }

      cudf::groupby::groupby gbs{cudf::table_view{set_keys},
                                 cudf::null_policy::INCLUDE};
      std::vector<cudf::groupby::aggregation_request> reqs;
      reqs.reserve(gs_values.size());
      for (size_t a = 0; a < gs_values.size(); ++a) {
        cudf::groupby::aggregation_request r;
        r.values = gs_values[a];
        r.aggregations.push_back(make_agg(gs_func_names[a], is_final));
        reqs.push_back(std::move(r));
      }
      auto [gk, res] = gbs.aggregate(reqs);

      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.reserve(static_cast<size_t>(nkeys) + 1 + res.size());
      for (cudf::size_type i = 0; i < gk->num_columns(); ++i)
        cols.push_back(std::make_unique<cudf::column>(gk->view().column(i)));
      cudf::numeric_scalar<int32_t> gid_s(gid, true);
      cols.push_back(cudf::make_column_from_scalar(gid_s, gk->num_rows()));
      for (size_t a = 0; a < res.size(); ++a) {
        auto col = std::move(res[a].results[0]);
        if (gs_is_count[a] && col->type().id() == cudf::type_id::INT32)
          col = cudf::cast(*col, cudf::data_type{cudf::type_id::INT64});
        cols.push_back(std::move(col));
      }
      set_tables.push_back(std::make_unique<cudf::table>(std::move(cols)));
    }

    std::vector<cudf::table_view> views;
    views.reserve(set_tables.size());
    for (auto& t : set_tables) views.push_back(t->view());
    auto out = cudf::concatenate(views);

    std::vector<std::string> names = key_names;
    names.push_back("__grouping_id");
    for (auto& n : gs_agg_names) names.push_back(n);
    return {std::move(out), std::move(names)};
  }

  cudf::table_view keys_view{key_cols};
  // SQL GROUP BY puts NULL keys in their own group; cuDF's groupby defaults to
  // null_policy::EXCLUDE, which DROPS every row that has a NULL in any grouping
  // key — silently losing the NULL group (e.g. TPC-DS q15's NULL ca_zip row).
  // INCLUDE matches the peacock CPU oracle. Non-null keys are unaffected.
  cudf::groupby::groupby gb{keys_view, cudf::null_policy::INCLUDE};

  AggPhase phase = agg_phase(agg->mode());

  // Raw value column for an aggregate's argument (Partial/Single stages), from
  // func->args() over the ORIGINAL input. NO avg out-scale cast here — see below.
  auto arg_col = [&](const fb::AggregateFuncNode* func) -> cudf::column_view {
    if (func->args() && func->args()->size() > 0) {
      auto* arg = func->args()->Get(0);
      if (arg->node_type() == fb::ExprNode_ColumnRef)
        return tv.column(static_cast<cudf::size_type>(arg->node_as_ColumnRef()->index()));
      computed_args.push_back(build_column(arg, tv));
      return computed_args.back()->view();
    }
    return tv.column(0);  // count(*) / no args: dummy column
  };

  // Build aggregation requests. AVG's two-phase STATE is (sum, count) — unlike a
  // pre-divided mean it IS additive, so Final merges Σsum/Σcount. Branch on the
  // THREE-way `phase`, never the 2-way is_final:
  //   Single-avg  -> 1 req [mean] over the out-scale-cast value      (1 out col)
  //   Partial-avg -> 1 req [sum, count] over the raw value           (2 out cols: STATE)
  //   Final-avg   -> 2 reqs [sum(partial_sum)], [sum(partial_count)] (1 out col)
  // The Final-input cursor `in_off` must advance by each agg's STATE WIDTH (avg
  // = 2, not 1), symmetrically with the Partial output assembly.
  // Every field carries a default: the arms below fill an OutBuild by name and set only
  // what they use, so an uninitialized one is read as whichever bit pattern the stack
  // held — `res` indexes a result vector and `avg_div` selects a whole assembly path.
  struct OutBuild {
    std::string name;
    int req = 0;              // request index producing the (primary) result
    int res = 0;              // result index within that request
    bool count_cast = false;  // INT32 count -> INT64
    int req_div = -1;         // Final-avg: request producing Σcount (-1 otherwise)
    bool avg_div = false;     // Final-avg: out = Σsum / Σcount
    int32_t out_scale = 0;      // decimal out scale for the avg divide
    uint8_t out_precision = 0;  // 0 => float avg
    // Final-stddev/var: the request result is a MERGE_M2 struct {count, mean, m2};
    // finalize to var = m2/(count-ddof) (NULL when count-ddof<=0), stddev = √var.
    bool std_finalize = false;
    bool is_variance = false;  // finalize as variance (skip the sqrt)
    int ddof = 1;              // divisor n-ddof (0 = population, 1 = sample)
    // Merge-stddev/var: the same MERGE_M2 struct, handed back as state rather than
    // finalized. One OutBuild per child, so the node emits [count, mean, m2] in the
    // order the next merge expects to read them.
    int struct_child = -1;
  };
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<OutBuild> builds;
  bool has_stddev_or_var_final = false;
  size_t in_off = key_indices.size();  // Final positional input cursor

  // Set by the serializer iff this run merges partial state across REAL hash
  // partitions. Drives the STDDEV/VAR state shape: 3-col Welford [count,mean,m2]
  // + MERGE_M2 when set, else the 1-col make_std singleton (byte-stable for the
  // single-partition goldens, e.g. tpcds q17). AVG is unaffected.
  const bool mergeable = agg->mergeable_agg_state();
  // Per-agg Final STRIDE (input columns consumed): count/sum/min/max = 1;
  // avg = 2 [sum,count] or 1 (grouping-set/ROLLUP mean); stddev/var = 3 Welford
  // cols when `mergeable`, else 1 (singleton). q17 interleaves all three.
  const size_t stddev_stride = mergeable ? 3 : 1;

  // Recover avg's 2-vs-1 stride from the RESIDUAL after removing keys + the
  // flag-known stddev strides + the 1-col others — NOT from the total column
  // count, which stddev's variable stride would confound. A grouping-set/ROLLUP
  // Partial emits 1 MEAN col per avg and its Final lands here; reading 2 cols for
  // such a 1-col avg over-runs the input (q18/q22 OOB).
  bool avg_state_2col = true;
  const bool reads_state = phase == AggPhase::Final || phase == AggPhase::Merge;
  if (reads_state && agg->aggr_funcs()) {
    size_t n_funcs = agg->aggr_funcs()->size();
    size_t n_avg = 0, n_std = 0;
    for (flatbuffers::uoffset_t i = 0; i < n_funcs; ++i) {
      std::string nm = agg->aggr_funcs()->Get(i)->name()
                           ? agg->aggr_funcs()->Get(i)->name()->str()
                           : "";
      if (is_avg_name(nm))
        ++n_avg;
      else if (is_stddev_name(nm) || is_var_name(nm))
        ++n_std;
    }
    size_t n_other = n_funcs - n_avg - n_std;  // count/sum/min/max = 1 col each
    size_t fixed = key_indices.size() + n_std * stddev_stride + n_other;
    size_t got = static_cast<size_t>(tv.num_columns());
    size_t residual = (got >= fixed) ? got - fixed : ~static_cast<size_t>(0);
    if (n_avg > 0 && residual == n_avg)  // 1 col per avg -> grouping-set/ROLLUP mean
      avg_state_2col = false;
    else if (residual != n_avg * 2)
      throw std::runtime_error(
          "state-stage aggregate input width does not match the expected "
          "count(1)/avg(1|2)/stddev(1|3) state layout");
  }

  if (agg->aggr_funcs()) {
    for (flatbuffers::uoffset_t i = 0; i < agg->aggr_funcs()->size(); ++i) {
      auto* func = agg->aggr_funcs()->Get(i);
      std::string name = func->name() ? func->name()->str() : "count";
      std::string alias = func->alias() ? func->alias()->str() : name;
      bool is_avg = is_avg_name(name);
      // Guard only the NON-mergeable (singleton) stddev/var Final: it must see one
      // partial row per key. The mergeable path below (MERGE_M2) is DESIGNED to
      // merge many partial rows, so it must NOT trip the guard.
      if (phase == AggPhase::Final && (is_stddev_name(name) || is_var_name(name)) &&
          !mergeable)
        has_stddev_or_var_final = true;

      if (is_avg && phase == AggPhase::Partial) {
        // Emit the (sum, count) STATE. Sum is over the RAW value (input scale) —
        // the out-scale cast belongs at the Final divide, not the partial sum.
        cudf::groupby::aggregation_request req;
        req.values = arg_col(func);
        req.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        req.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
        int r = static_cast<int>(requests.size());
        builds.push_back({alias, r, 0, false, -1, false, 0, 0});      // partial_sum
        builds.push_back({alias, r, 1, true, -1, false, 0, 0});       // partial_count -> INT64
        requests.push_back(std::move(req));
      } else if (is_avg && phase == AggPhase::Final && !avg_state_2col) {
        // Grouping-set (ROLLUP/CUBE) Partial emitted a 1-col MEAN, not [sum,count]
        // state; with one partial row per key, mean-of-singleton is exact. Consume
        // ONE input column (the shared in_off += 1 below), NOT two. Real
        // ROLLUP-avg state merge is #18.
        cudf::groupby::aggregation_request req;
        req.values = tv.column(static_cast<cudf::size_type>(in_off));
        req.aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        builds.push_back({alias, static_cast<int>(requests.size()), 0, false,
                          -1, false, 0, 0});
        requests.push_back(std::move(req));
        // fall through to the shared `in_off += 1` (single Final input column)
      } else if (is_avg && phase == AggPhase::Merge && !avg_state_2col) {
        // A grouping-set/ROLLUP Partial emits ONE mean column rather than [sum, count]
        // state, and the same width recovery that guards the Final arm (#18, the q18/q22
        // over-run) guards this one: consume ONE column through the shared `in_off += 1`
        // below. Mean-of-singleton is exact while there is one partial row per key, which
        // is the same condition the Final arm relies on.
        cudf::groupby::aggregation_request req;
        req.values = tv.column(static_cast<cudf::size_type>(in_off));
        req.aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        builds.push_back({alias, static_cast<int>(requests.size()), 0});
        requests.push_back(std::move(req));
        // falls through to the shared `in_off += 1`
      } else if (is_avg && phase == AggPhase::Merge) {
        // Σ(partial_sum) and Σ(partial_count), emitted as state: the divide is the
        // finalize and belongs to whoever finalizes. Consumes the same TWO columns
        // Final does.
        cudf::groupby::aggregation_request req_s, req_c;
        req_s.values = tv.column(static_cast<cudf::size_type>(in_off));
        req_s.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        req_c.values = tv.column(static_cast<cudf::size_type>(in_off + 1));
        req_c.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        int r = static_cast<int>(requests.size());
        builds.push_back({alias, r, 0, false, -1, false, 0, 0});      // Σ partial_sum
        builds.push_back({alias, r + 1, 0, false, -1, false, 0, 0});  // Σ partial_count
        requests.push_back(std::move(req_s));
        requests.push_back(std::move(req_c));
        in_off += 2;
        continue;  // avg consumed TWO state columns
      } else if (is_avg && phase == AggPhase::Final) {
        // Merge: Σ(partial_sum) / Σ(partial_count). The two state cols sit at
        // [in_off, in_off+1] (running offset accounts for prior avgs' 2 cols).
        cudf::groupby::aggregation_request req_s, req_c;
        req_s.values = tv.column(static_cast<cudf::size_type>(in_off));
        req_s.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        req_c.values = tv.column(static_cast<cudf::size_type>(in_off + 1));
        req_c.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        int r_s = static_cast<int>(requests.size());
        requests.push_back(std::move(req_s));
        int r_c = static_cast<int>(requests.size());
        requests.push_back(std::move(req_c));
        builds.push_back({alias, r_s, 0, false, r_c, true,
                          static_cast<int32_t>(func->out_decimal_scale()),
                          func->out_decimal_precision()});
        in_off += 2;
        continue;  // avg consumed TWO Final input columns
      } else if ((is_stddev_name(name) || is_var_name(name)) &&
                 phase == AggPhase::Partial && mergeable) {
        // Welford STATE [count, mean, m2] over the value cast to FLOAT64 (stddev/
        // var are float-valued in DataFusion). Three aggregations in one request,
        // matching DataFusion's 3-col Partial state schema so the Final can
        // MERGE_M2 across real hash partitions (#25). count -> INT64.
        computed_args.push_back(
            cudf::cast(arg_col(func), cudf::data_type{cudf::type_id::FLOAT64}));
        cudf::groupby::aggregation_request req;
        req.values = computed_args.back()->view();
        req.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
        req.aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        req.aggregations.push_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());
        int r = static_cast<int>(requests.size());
        builds.push_back({alias, r, 0, true, -1, false, 0, 0});   // count -> INT64
        builds.push_back({alias, r, 1, false, -1, false, 0, 0});  // mean
        builds.push_back({alias, r, 2, false, -1, false, 0, 0});  // m2
        requests.push_back(std::move(req));
      } else if ((is_stddev_name(name) || is_var_name(name)) &&
                 phase == AggPhase::Merge && mergeable) {
        // The Final arm's merge without its finalize: pack [count, mean, m2] at
        // [in_off .. in_off+2] and MERGE_M2 per group, then hand the struct's three
        // children back as columns. Count MUST be INT32 going in — cuDF 25.02's
        // group_merge_m2 rejects INT64 at runtime — and comes back out as INT64, so
        // a second merge reads the same widths this one did.
        auto cnt = cudf::cast(tv.column(static_cast<cudf::size_type>(in_off)),
                              cudf::data_type{cudf::type_id::INT32});
        auto mean = std::make_unique<cudf::column>(
            tv.column(static_cast<cudf::size_type>(in_off + 1)));
        auto m2 = std::make_unique<cudf::column>(
            tv.column(static_cast<cudf::size_type>(in_off + 2)));
        std::vector<std::unique_ptr<cudf::column>> members;
        members.push_back(std::move(cnt));
        members.push_back(std::move(mean));
        members.push_back(std::move(m2));
        computed_args.push_back(cudf::make_structs_column(
            tv.num_rows(), std::move(members), 0, rmm::device_buffer{}));
        cudf::groupby::aggregation_request req;
        req.values = computed_args.back()->view();
        req.aggregations.push_back(
            cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>());
        int r = static_cast<int>(requests.size());
        for (int child = 0; child < 3; ++child) {
          OutBuild ob;
          ob.name = alias;
          ob.req = r;
          ob.struct_child = child;
          builds.push_back(ob);
        }
        requests.push_back(std::move(req));
        in_off += 3;
        continue;  // stddev/var consumed THREE state columns
      } else if ((is_stddev_name(name) || is_var_name(name)) &&
                 phase == AggPhase::Final && mergeable) {
        // Merge Welford state across partitions: pack the 3 partial cols
        // [count, mean, m2] at [in_off .. in_off+2] into a struct and MERGE_M2 per
        // group (Welford-Chan); the finalize happens in the assembly below.
        // Consumes THREE Final cols. Child ORDER + TYPES are fixed by cuDF's
        // group_merge_m2: child(0)=valid_count INT32, child(1)=mean f64,
        // child(2)=M2 f64. Count MUST be INT32 — cuDF 25.02 rejects INT64 at
        // runtime (25.10 relaxed it), so an INT64 here compiles and then fails.
        auto cnt = cudf::cast(tv.column(static_cast<cudf::size_type>(in_off)),
                              cudf::data_type{cudf::type_id::INT32});
        auto mean = std::make_unique<cudf::column>(
            tv.column(static_cast<cudf::size_type>(in_off + 1)));
        auto m2 = std::make_unique<cudf::column>(
            tv.column(static_cast<cudf::size_type>(in_off + 2)));
        std::vector<std::unique_ptr<cudf::column>> members;
        members.push_back(std::move(cnt));
        members.push_back(std::move(mean));
        members.push_back(std::move(m2));
        computed_args.push_back(cudf::make_structs_column(
            tv.num_rows(), std::move(members), 0, rmm::device_buffer{}));
        cudf::groupby::aggregation_request req;
        req.values = computed_args.back()->view();
        req.aggregations.push_back(
            cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>());
        OutBuild ob;
        ob.name = alias;
        ob.req = static_cast<int>(requests.size());
        ob.std_finalize = true;
        ob.is_variance = is_var_name(name);
        ob.ddof = static_cast<int>(stddev_ddof(name));
        builds.push_back(ob);
        requests.push_back(std::move(req));
        in_off += 3;
        continue;  // stddev/var consumed THREE Final input columns
      } else {
        // Single-avg (plain mean) OR sum/count/min/max/stddev — one request.
        cudf::groupby::aggregation_request req;
        if (reads_state) {
          req.values = tv.column(static_cast<cudf::size_type>(in_off));
        } else if (is_avg) {
          // Single-avg: mean over the value cast up to the declared out scale
          // (cuDF mean keeps the input scale otherwise).
          cudf::column_view base = arg_col(func);
          if (func->out_decimal_precision() != 0 &&
              base.type().id() == cudf::type_id::DECIMAL128) {
            int32_t want_exp = -static_cast<int32_t>(func->out_decimal_scale());
            if (base.type().scale() != want_exp) {
              computed_args.push_back(
                  cudf::cast(base, cudf::data_type{cudf::type_id::DECIMAL128, want_exp}));
              base = computed_args.back()->view();
            }
          }
          req.values = base;
        } else if ((is_stddev_name(name) || is_var_name(name)) &&
                   (phase == AggPhase::Partial || phase == AggPhase::Single)) {
          // SINGLE-PARTITION stddev/var: cuDF's make_std/make_variance keep the
          // input type (e.g. DECIMAL l_quantity), but DataFusion's stddev/var are
          // FLOAT64 — cast first, as the mergeable M2 path does. (Final on this leg
          // is a singleton MEAN over the already-f64 partial, so it needs no cast.)
          computed_args.push_back(
              cudf::cast(arg_col(func), cudf::data_type{cudf::type_id::FLOAT64}));
          req.values = computed_args.back()->view();
        } else {
          req.values = arg_col(func);
        }
        req.aggregations.push_back(make_agg(name, reads_state));
        builds.push_back({alias, static_cast<int>(requests.size()), 0,
                          (name == "count" || name == "COUNT"), -1, false, 0, 0});
        requests.push_back(std::move(req));
      }
      if (reads_state) in_off += 1;
    }
  }

  auto [group_keys, agg_results] = gb.aggregate(requests);

  // Guards the NON-mergeable stddev/var Final only (has_stddev_or_var_final is
  // set only there): merging >1 partial row per key on that path would silently
  // produce std-of-stds. AVG is deliberately unguarded — its (sum,count) state
  // merges correctly above (#25); the global/ungrouped avg has its own guard in
  // the reduce path.
  if (has_stddev_or_var_final &&
      group_keys->num_rows() < static_cast<cudf::size_type>(tv.num_rows())) {
    throw std::runtime_error(
        "Final-stage STDDEV/VAR merged multiple partial rows per key "
        "(std-of-stds is wrong); STDDEV/VAR partial-moment merge is Inc5");
  }

  // Assemble output: key columns then aggregate columns (per `builds`).
  for (cudf::size_type i = 0; i < group_keys->num_columns(); ++i) {
    out_cols.push_back(std::make_unique<cudf::column>(group_keys->view().column(i)));
    out_names.push_back(key_names[i]);
  }
  for (auto& b : builds) {
    std::unique_ptr<cudf::column> col;
    if (b.struct_child >= 0) {
      // Merge-stddev/var: state out, not a value. cuDF's merged count child is
      // INT32; widen it so both merges of a chain read the same layout.
      auto merged = agg_results[b.req].results[b.res]->view();
      auto child = merged.child(b.struct_child);
      col = child.type().id() == cudf::type_id::INT32
                ? cudf::cast(child, cudf::data_type{cudf::type_id::INT64})
                : std::make_unique<cudf::column>(child);
    } else if (b.std_finalize) {
      // MERGE_M2 result is a struct {count, mean, m2}. Finalize:
      //   var = m2 / (count - ddof);  stddev = sqrt(var)
      // with DataFusion's sample-NULL semantics: count-ddof <= 0 (a single-row
      // sample group, or an empty group) -> NULL, not NaN/inf.
      auto merged = agg_results[b.req].results[b.res]->view();  // struct
      auto count_v = merged.child(0);  // valid count
      auto m2_v = merged.child(2);     // FLOAT64
      auto count_f = cudf::cast(count_v, cudf::data_type{cudf::type_id::FLOAT64});
      cudf::numeric_scalar<double> ddof_s(static_cast<double>(b.ddof), true);
      auto denom = cudf::binary_operation(count_f->view(), ddof_s,
                                          cudf::binary_operator::SUB,
                                          cudf::data_type{cudf::type_id::FLOAT64});
      auto var = cudf::binary_operation(m2_v, denom->view(),
                                        cudf::binary_operator::DIV,
                                        cudf::data_type{cudf::type_id::FLOAT64});
      cudf::numeric_scalar<double> zero(0.0, true);
      auto valid = cudf::binary_operation(denom->view(), zero,
                                          cudf::binary_operator::GREATER,
                                          cudf::data_type{cudf::type_id::BOOL8});
      auto null_f64 =
          cudf::make_default_constructed_scalar(cudf::data_type{cudf::type_id::FLOAT64});
      auto var_masked = cudf::copy_if_else(var->view(), *null_f64, valid->view());
      col = b.is_variance
                ? std::move(var_masked)
                : cudf::unary_operation(var_masked->view(), cudf::unary_operator::SQRT);
    } else if (b.avg_div) {
      // Σsum / Σcount at DataFusion's declared out type.
      auto sum_v = agg_results[b.req].results[b.res]->view();
      auto cnt_v = agg_results[b.req_div].results[0]->view();
      if (b.out_precision != 0) {
        // Decimal avg: DIV result scale = lhs.scale - rhs.scale, so put Σsum at the
        // out scale and Σcount at scale 0 -> result carries the out scale.
        int32_t want_exp = -b.out_scale;
        auto sum_scaled =
            cudf::cast(sum_v, cudf::data_type{cudf::type_id::DECIMAL128, want_exp});
        auto cnt_dec = cudf::cast(cnt_v, cudf::data_type{cudf::type_id::DECIMAL128, 0});
        col = cudf::binary_operation(sum_scaled->view(), cnt_dec->view(),
                                     cudf::binary_operator::DIV,
                                     cudf::data_type{cudf::type_id::DECIMAL128, want_exp});
      } else {
        auto sum_f = cudf::cast(sum_v, cudf::data_type{cudf::type_id::FLOAT64});
        auto cnt_f = cudf::cast(cnt_v, cudf::data_type{cudf::type_id::FLOAT64});
        col = cudf::binary_operation(sum_f->view(), cnt_f->view(),
                                     cudf::binary_operator::DIV,
                                     cudf::data_type{cudf::type_id::FLOAT64});
      }
    } else {
      col = std::move(agg_results[b.req].results[b.res]);
      // cuDF count returns INT32; cast to INT64 for SQL BIGINT compatibility.
      if (b.count_cast && col->type().id() == cudf::type_id::INT32)
        col = cudf::cast(*col, cudf::data_type{cudf::type_id::INT64});
    }
    out_cols.push_back(std::move(col));
    out_names.push_back(b.name);
  }

  return {std::make_unique<cudf::table>(std::move(out_cols)), std::move(out_names)};
}


}  // namespace peacock
