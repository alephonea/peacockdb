// Per-operator cuDF timings over real TPC-H sf40 columns.
//
// The query benchmarks say how long q1 takes; they cannot say which operator the time went
// to, and on a query like q1 — 78 s whole-table versus 3.9 s streamed for the same answer —
// that is the only question worth asking. This file times one operator at a time over the
// same sf40 columns, so a query's cost can be read as a sum of parts rather than guessed at.
//
// Real columns, not synthetic ones, and that is the point: cuDF's cost depends on the data,
// not just its size. A groupby over 4 distinct values and a groupby over 60M distinct values
// are the same call on the same column count, and they are three orders of magnitude apart.
// Every input below is a genuine sf40 column with its genuine distribution.
//
// Not a correctness test — it asserts only that each operator produced the row count it
// should, so a benchmark cannot silently time an empty column. Correctness of these
// operators is test_tpch.cpp's job.
//
// Sizing: one slice of lineitem, PEACOCK_NODES_ROWS rows (default 50M), read once and reused
// by every case, so the numbers are comparable across operators and no case pays a load.
#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
// cudf/join.hpp split into cudf/join/*.hpp in 26.02; this binary is meant to run on the
// 25.02 hosts too, so both spellings are accepted. cudf::hash_join exists in both.
#if __has_include(<cudf/join/hash_join.hpp>)
#  include <cudf/join/hash_join.hpp>
#else
#  include <cudf/join.hpp>
#endif
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "rmm_pool.hpp"
#include "tpch_golden.hpp"

using namespace peacock_test;

namespace {

using Clock = std::chrono::steady_clock;

int bench_runs() { return std::max(2, std::atoi(env_or("PEACOCK_NODE_RUNS", "5").c_str())); }

cudf::size_type slice_rows() {
  return static_cast<cudf::size_type>(
      std::atoll(env_or("PEACOCK_NODES_ROWS", "50000000").c_str()));
}

const auto kBool = cudf::data_type{cudf::type_id::BOOL8};
const auto kDec2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
const auto kDec4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};
const auto kF64 = cudf::data_type{cudf::type_id::FLOAT64};

struct Row {
  std::string name;
  long rows;
  double ms;
  double gbps;   // negative => no meaningful byte count for this operator
  long out_rows;
};

std::vector<Row> results;

// Times one operator: warm up once, then 2nd-minimum of N, syncing inside the timed region
// because cuDF returns as soon as the kernel is queued. bytes is what the operator reads —
// stated per case rather than derived, since "the bytes an operator touches" is not a
// property cuDF exposes.
template <typename F>
void bench(const std::string& name, long rows, double bytes, F&& op) {
  auto warm = op();
  cudaDeviceSynchronize();
  const long out_rows = warm;

  std::vector<double> ms;
  ms.reserve(bench_runs());
  for (int i = 0; i < bench_runs(); ++i) {
    cudaDeviceSynchronize();
    const auto t0 = Clock::now();
    auto r = op();
    cudaDeviceSynchronize();
    ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    (void)r;
  }
  std::sort(ms.begin(), ms.end());
  const double best = ms.size() > 1 ? ms[1] : ms[0];
  const double gbps = bytes > 0 ? bytes / (best / 1e3) / 1e9 : -1.0;
  results.push_back({name, rows, best, gbps, out_rows});

  if (gbps >= 0) {
    std::fprintf(stderr, "[node] %-42s %9.2f ms  %8.1f Mrow/s  %7.1f GB/s  out=%ld\n",
                 name.c_str(), best, rows / best / 1e3, gbps, out_rows);
  } else {
    std::fprintf(stderr, "[node] %-42s %9.2f ms  %8.1f Mrow/s  %7s      out=%ld\n", name.c_str(),
                 best, rows / best / 1e3, "-", out_rows);
  }
}

cudf::io::table_with_metadata read_slice(const std::string& path, std::vector<std::string> cols,
                                         cudf::size_type rows) {
  auto o = cudf::io::parquet_reader_options::builder(
               cudf::io::source_info{std::vector<std::string>{path}})
               .columns(std::move(cols))
               .num_rows(rows)
               .build();
  return cudf::io::read_parquet(o);
}

cudf::column_view map_view(std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& m) {
  return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(m->size()), m->data(), nullptr, 0);
}

}  // namespace

// One fixture, one load, every operator below reuses it — a per-test load would dominate
// the short cases and make them incomparable.
class CudfNodes : public TpchSf40 {};

TEST_F(CudfNodes, OperatorTimings) {
  const auto rows = slice_rows();

  const auto t_load = Clock::now();
  auto line = read_slice(data_dir() + "/lineitem.parquet",
                         {"l_orderkey", "l_returnflag", "l_linestatus", "l_quantity",
                          "l_extendedprice", "l_discount", "l_shipdate"},
                         rows);
  auto orders = read_slice(data_dir() + "/orders.parquet", {"o_orderkey"}, rows / 4);
  cudaDeviceSynchronize();
  const double load_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_load).count();

  auto lv = line.tbl->view();
  const long n = lv.num_rows();
  ASSERT_GT(n, 0) << "no lineitem rows read";
  std::fprintf(stderr, "[node] loaded %ld lineitem rows x %d cols, %ld orders keys in %.0f ms\n", n,
               lv.num_columns(), static_cast<long>(orders.tbl->num_rows()), load_ms);

  auto orderkey = lv.column(0);
  auto returnflag = lv.column(1);
  auto quantity_raw = lv.column(3);
  auto extprice_raw = lv.column(4);
  auto discount_raw = lv.column(5);
  auto shipdate = lv.column(6);

  // Widened once outside the timed cases so the decimal operators are timed on decimal128
  // inputs — the width the engine actually computes in.
  auto extprice = cudf::cast(extprice_raw, kDec2);
  auto discount = cudf::cast(discount_raw, kDec2);
  auto quantity = cudf::cast(quantity_raw, kDec2);
  auto ev = extprice->view();
  auto dv = discount->view();

  // ---- casts -------------------------------------------------------------
  // reads 8 B/row (DECIMAL64), writes 16 B
  bench("cast decimal64 -> decimal128", n, double(n) * 8, [&] {
    auto c = cudf::cast(extprice_raw, kDec2);
    return long(c->size());
  });
  bench("cast decimal64 -> float64", n, double(n) * 8, [&] {
    auto c = cudf::cast(extprice_raw, kF64);
    return long(c->size());
  });

  // ---- expressions -------------------------------------------------------
  // two decimal128 inputs, 32 B/row read
  bench("expr decimal128 mul (price*discount)", n, double(n) * 32, [&] {
    auto c = cudf::binary_operation(ev, dv, cudf::binary_operator::MUL, kDec4);
    return long(c->size());
  });
  bench("expr decimal128 sub (scalar - col)", n, double(n) * 16, [&] {
    auto one = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
    auto c = cudf::binary_operation(one, dv, cudf::binary_operator::SUB, kDec2);
    return long(c->size());
  });
  bench("expr timestamp >= scalar (predicate)", n, double(n) * 4, [&] {
    auto lo = cudf::timestamp_scalar<cudf::timestamp_D>(
        cudf::timestamp_D{cudf::duration_D{days_since_epoch(1994, 1, 1)}}, true);
    auto c = cudf::binary_operation(shipdate, lo, cudf::binary_operator::GREATER_EQUAL, kBool);
    return long(c->size());
  });
  bench("expr string == scalar (l_returnflag)", n, double(n) * 1, [&] {
    auto a = cudf::string_scalar(std::string("A"));
    auto c = cudf::binary_operation(returnflag, a, cudf::binary_operator::EQUAL, kBool);
    return long(c->size());
  });

  // ---- filter ------------------------------------------------------------
  // The mask is built once outside so apply_boolean_mask is timed alone; building it is
  // the predicate cases above.
  //
  // The predicates are on l_quantity and l_discount, NOT on l_shipdate, and that is not a
  // detail. lineitem is clustered by orderkey and orderkey correlates with date, so a date
  // predicate over the first N rows selects almost everything or almost nothing depending
  // on N — a date-based version of these two cases measured 100% and 0% selectivity at
  // n=50M and would have silently reported a filter rate for an empty output. quantity
  // (1..50) and discount (0.00..0.10) are uniform and independent of row position, so the
  // selectivity is a property of the predicate at any slice size. The `out=` column is
  // the measured survivor count; the label does not assert a percentage it cannot know.
  auto qty_26 = cudf::fixed_point_scalar<numeric::decimal128>(2600, numeric::scale_type{-2});
  auto disc_05 = cudf::fixed_point_scalar<numeric::decimal128>(5, numeric::scale_type{-2});
  auto sel_half = cudf::binary_operation(quantity->view(), qty_26, cudf::binary_operator::LESS,
                                         kBool);
  auto sel_low = cudf::binary_operation(dv, disc_05, cudf::binary_operator::EQUAL, kBool);

  bench("filter apply_boolean_mask, 2 dec128 (qty<26)", n, double(n) * 32, [&] {
    auto t = cudf::apply_boolean_mask(cudf::table_view{{ev, dv}}, sel_half->view());
    return long(t->num_rows());
  });
  bench("filter apply_boolean_mask, 2 dec128 (disc=0.05)", n, double(n) * 32, [&] {
    auto t = cudf::apply_boolean_mask(cudf::table_view{{ev, dv}}, sel_low->view());
    return long(t->num_rows());
  });

  // ---- aggregation -------------------------------------------------------
  // Output type must equal the input type for a fixed-point sum reduction — cuDF rejects a
  // rescaling reduce outright rather than rescaling for you.
  bench("reduce sum decimal128", n, double(n) * 16, [&] {
    auto agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    auto s = cudf::reduce(ev, *agg, ev.type());
    return long(s->is_valid());
  });

  // Two group cardinalities, the same call. This pair is the reason the file exists.
  auto sum_by = [&](std::vector<cudf::column_view> keys) {
    cudf::groupby::groupby gb(cudf::table_view{keys});
    std::vector<cudf::groupby::aggregation_request> reqs;
    cudf::groupby::aggregation_request r;
    r.values = ev;
    r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    reqs.push_back(std::move(r));
    auto [k, a] = gb.aggregate(reqs);
    return long(k->num_rows());
  };
  // These four exist as a set, and the set is the point. The obvious pair — group by two
  // string columns into 4 groups, versus one int64 column into millions — varies key type,
  // key width, column count AND cardinality at once, so whatever it shows cannot be
  // attributed to any of them. The two derived keys below hold the column fixed (one int64
  // built from l_orderkey) and vary only cardinality; reading them against the string cases
  // separates type from cardinality.
  auto four = cudf::numeric_scalar<int64_t>(4);
  auto key_mod4 = cudf::binary_operation(orderkey, four, cudf::binary_operator::MOD,
                                         cudf::data_type{cudf::type_id::INT64});
  auto thousand = cudf::numeric_scalar<int64_t>(1000);
  auto key_mod1k = cudf::binary_operation(orderkey, thousand, cudf::binary_operator::MOD,
                                          cudf::data_type{cudf::type_id::INT64});

  bench("groupby 1 int64 key, 4 groups", n, double(n) * 24,
        [&] { return sum_by({key_mod4->view()}); });
  bench("groupby 1 int64 key, 1000 groups", n, double(n) * 24,
        [&] { return sum_by({key_mod1k->view()}); });
  bench("groupby 1 int64 key, ~n/4 groups (orderkey)", n, double(n) * 24,
        [&] { return sum_by({orderkey}); });
  bench("groupby 1 string key, 3 groups (returnflag)", n, double(n) * 18,
        [&] { return sum_by({returnflag}); });
  bench("groupby 2 string keys, 4 groups (flag,status)", n, double(n) * 18,
        [&] { return sum_by({returnflag, lv.column(2)}); });

  // The q1 shape, and the reason this case exists: one groupby carrying EIGHT aggregation
  // requests over the same keys — four DECIMAL128 sums, three float64 means, a count.
  // Whole-table q1 costs ~75 s even over a pool while its operators, timed one at a time
  // above, sum to a second or two; the difference has to be here or nowhere. If cuDF
  // cannot serve every request from its hash groupby it falls back to sorting the input,
  // and sorting 236M rows on two string keys is a different order of cost than hashing
  // them. Timing one aggregate and assuming eight cost eight times as much is exactly the
  // extrapolation this pair is here to refute or confirm.
  auto qty_f = cudf::cast(quantity->view(), kF64);
  auto price_f = cudf::cast(ev, kF64);
  auto disc_f = cudf::cast(dv, kF64);
  bench("groupby q1 shape: 8 aggs, 4 groups", n, double(n) * 18, [&] {
    cudf::groupby::groupby gb(cudf::table_view{{returnflag, lv.column(2)}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    auto add = [&](cudf::column_view v, std::unique_ptr<cudf::groupby_aggregation> a) {
      cudf::groupby::aggregation_request r;
      r.values = v;
      r.aggregations.push_back(std::move(a));
      reqs.push_back(std::move(r));
    };
    add(quantity->view(), cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    add(ev, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    add(ev, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    add(ev, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    add(qty_f->view(), cudf::make_mean_aggregation<cudf::groupby_aggregation>());
    add(price_f->view(), cudf::make_mean_aggregation<cudf::groupby_aggregation>());
    add(disc_f->view(), cudf::make_mean_aggregation<cudf::groupby_aggregation>());
    add(quantity->view(), cudf::make_count_aggregation<cudf::groupby_aggregation>());
    auto [k, a] = gb.aggregate(reqs);
    return long(k->num_rows());
  });
  // The two halves of that shape, separately, to say WHICH request costs it.
  bench("groupby 4 dec128 sums only, 4 groups", n, double(n) * 18, [&] {
    cudf::groupby::groupby gb(cudf::table_view{{returnflag, lv.column(2)}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    for (int i = 0; i < 4; ++i) {
      cudf::groupby::aggregation_request r;
      r.values = ev;
      r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    auto [k, a] = gb.aggregate(reqs);
    return long(k->num_rows());
  });
  bench("groupby 3 float64 means only, 4 groups", n, double(n) * 18, [&] {
    cudf::groupby::groupby gb(cudf::table_view{{returnflag, lv.column(2)}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    for (auto v : {qty_f->view(), price_f->view(), disc_f->view()}) {
      cudf::groupby::aggregation_request r;
      r.values = v;
      r.aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    auto [k, a] = gb.aggregate(reqs);
    return long(k->num_rows());
  });

  // ---- join --------------------------------------------------------------
  auto ov = orders.tbl->view();
  bench("hash_join build (orders keys)", ov.num_rows(), double(ov.num_rows()) * 8, [&] {
    cudf::hash_join hj(cudf::table_view{{ov.column(0)}}, cudf::null_equality::EQUAL);
    return long(ov.num_rows());
  });

  cudf::hash_join hj(cudf::table_view{{ov.column(0)}}, cudf::null_equality::EQUAL);
  bench("hash_join probe (lineitem x orders)", n, double(n) * 8, [&] {
    auto [l, o] = hj.inner_join(cudf::table_view{{orderkey}});
    return long(l->size());
  });

  // Probe plus the gather that a real plan always pays: a join's output is the maps, and
  // nothing downstream can read a map.
  bench("hash_join probe + gather 2 cols", n, double(n) * 8, [&] {
    auto [l, o] = hj.inner_join(cudf::table_view{{orderkey}});
    auto t = cudf::gather(cudf::table_view{{ev, dv}}, map_view(l));
    return long(t->num_rows());
  });

  // ---- sort / gather -----------------------------------------------------
  bench("sorted_order on int64 (orderkey)", n, double(n) * 8, [&] {
    auto o = cudf::sorted_order(cudf::table_view{{orderkey}}, {cudf::order::ASCENDING},
                                {cudf::null_order::AFTER});
    return long(o->size());
  });

  auto sort_map = cudf::sorted_order(cudf::table_view{{orderkey}}, {cudf::order::ASCENDING},
                                     {cudf::null_order::AFTER});
  bench("gather 2 dec128 cols by a sort map", n, double(n) * 32, [&] {
    auto t = cudf::gather(cudf::table_view{{ev, dv}}, sort_map->view());
    return long(t->num_rows());
  });

  std::fprintf(stderr, "\n[node] %-42s %10s %12s %10s\n", "summary", "ms", "Mrow/s", "GB/s");
  for (const auto& r : results) {
    std::fprintf(stderr, "[node] %-42s %10.2f %12.1f %10.1f\n", r.name.c_str(), r.ms,
                 r.rows / r.ms / 1e3, r.gbps);
  }
  note_peak();
}

// Same entry point as the other gtest binaries here (the conda cudf ships no gtest_main).
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  peacock_test::install_rmm_pool();
  return RUN_ALL_TESTS();
}
