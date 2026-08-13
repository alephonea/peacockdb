// TPC-H query plans hand-written in bare cuDF (no Rust/DataFusion/SQL), checked against
// DuckDB over the SAME sf40 parquet files.
//
// Two phases, mirroring peacockdb's load-then-execute model: phase 1 reads COLUMNS only
// (row filtering is deliberately NOT pushed into the reader); phase 2 runs the operator
// chain over the resident columns. If a predicate ever migrates into phase 1 this test
// stops testing what it exists for.
//
// Data is read-only, in place; absent data => SKIP loudly (see the fixture).
#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/groupby.hpp>
// cudf/join.hpp moved to cudf/join/join.hpp in 26.02; CI builds both legs. Path change
// only — the inner_join signature is identical in both versions.
#if __has_include(<cudf/join/join.hpp>)
#  include <cudf/join/join.hpp>   // cudf >= 26.02
#else
#  include <cudf/join.hpp>        // cudf 25.02
#endif
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

#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "rmm_pool.hpp"
#include "tpch_golden.hpp"

using namespace peacock_test;

// ===========================================================================
// Q6  — revenue from a discounted-order filter
//
//   SELECT sum(l_extendedprice * l_discount)
//   FROM lineitem
//   WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
//     AND l_discount BETWEEN 0.05 AND 0.07
//     AND l_quantity < 24
//
// operator chain: scan -> filter(3 predicates) -> project(mul) -> reduce(sum)
//
// Exact, no tolerance: money columns decode as DECIMAL64 scale -2, widened to DECIMAL128
// for headroom, fixed-point end to end. Predicate constants are decimal scalars, so the
// boundaries 0.05/0.07/24 have no float error. Sum accumulates at scale -4; worst case
// ~1e16 vs a DECIMAL128 ceiling of ~1.7e38 — cannot overflow.
// ===========================================================================
TEST_F(TpchSf40, Q6ExactDecimal) {
  const auto lineitem_path = data_dir() + "/lineitem.parquet";
  const auto golden_path = golden_dir() + "/duckdb_q6.csv";
  ASSERT_TRUE(file_exists(golden_path))
      << "golden missing: " << golden_path
      << " (regenerate with testdata/gen_duckdb_goldens.sh --sf 40)";

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD (columns only; predicates stay in phase 2) ----------------
  auto opts = cudf::io::parquet_reader_options::builder(
                  cudf::io::source_info{std::vector<std::string>{lineitem_path}})
                  .columns({"l_quantity", "l_extendedprice", "l_discount", "l_shipdate"})
                  .build();
  auto loaded = cudf::io::read_parquet(opts);
  auto tbl = loaded.tbl->view();
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();

  ASSERT_EQ(tbl.num_columns(), 4);
  std::fprintf(stderr, "[q6] loaded %ld rows x 4 cols\n", static_cast<long>(tbl.num_rows()));

  auto quantity_raw = tbl.column(0);
  auto extprice_raw = tbl.column(1);
  auto discount_raw = tbl.column(2);
  auto shipdate = tbl.column(3);

  // Assert decoded types — if a future cudf decodes differently, the arithmetic below
  // changes meaning.
  ASSERT_EQ(shipdate.type().id(), cudf::type_id::TIMESTAMP_DAYS);
  for (auto c : {quantity_raw, extprice_raw, discount_raw}) {
    ASSERT_TRUE(c.type().id() == cudf::type_id::DECIMAL64 ||
                c.type().id() == cudf::type_id::DECIMAL128)
        << "expected fixed-point money columns, got type id "
        << static_cast<int>(c.type().id());
    ASSERT_EQ(c.type().scale(), -2);
  }

  // widen to DECIMAL128 (same scale => exact, value-preserving) for accumulation headroom
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};

  // ---------------- PHASE 2: EXECUTE ----------------
  // Closure so the test verifies ONCE and benchmark_execute times the SAME code. The
  // widening casts are operators and live inside the timed region.
  auto execute = [&]() -> std::unique_ptr<cudf::scalar> {
    auto quantity = cudf::cast(quantity_raw, dec128_s2);
    auto extprice = cudf::cast(extprice_raw, dec128_s2);
    auto discount = cudf::cast(discount_raw, dec128_s2);

    // filter: three predicates, AND-ed
    auto lo_date = cudf::timestamp_scalar<cudf::timestamp_D>(
        cudf::timestamp_D{cudf::duration_D{days_since_epoch(1994, 1, 1)}}, true);
    auto hi_date = cudf::timestamp_scalar<cudf::timestamp_D>(
        cudf::timestamp_D{cudf::duration_D{days_since_epoch(1995, 1, 1)}}, true);
    // decimal scalars at scale -2: 0.05 -> 5, 0.07 -> 7, 24 -> 2400. Exact boundaries.
    auto disc_lo = cudf::fixed_point_scalar<numeric::decimal128>(5, numeric::scale_type{-2});
    auto disc_hi = cudf::fixed_point_scalar<numeric::decimal128>(7, numeric::scale_type{-2});
    auto qty_hi = cudf::fixed_point_scalar<numeric::decimal128>(2400, numeric::scale_type{-2});

    const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
    auto m1 = cudf::binary_operation(shipdate, lo_date, cudf::binary_operator::GREATER_EQUAL, boolean);
    auto m2 = cudf::binary_operation(shipdate, hi_date, cudf::binary_operator::LESS, boolean);
    auto m3 = cudf::binary_operation(discount->view(), disc_lo, cudf::binary_operator::GREATER_EQUAL, boolean);
    auto m4 = cudf::binary_operation(discount->view(), disc_hi, cudf::binary_operator::LESS_EQUAL, boolean);
    auto m5 = cudf::binary_operation(quantity->view(), qty_hi, cudf::binary_operator::LESS, boolean);

    auto mask = cudf::binary_operation(m1->view(), m2->view(), cudf::binary_operator::LOGICAL_AND, boolean);
    mask = cudf::binary_operation(mask->view(), m3->view(), cudf::binary_operator::LOGICAL_AND, boolean);
    mask = cudf::binary_operation(mask->view(), m4->view(), cudf::binary_operator::LOGICAL_AND, boolean);
    mask = cudf::binary_operation(mask->view(), m5->view(), cudf::binary_operator::LOGICAL_AND, boolean);

    note_peak();
    auto kept = cudf::apply_boolean_mask(
        cudf::table_view{{extprice->view(), discount->view()}}, mask->view());
    EXPECT_GT(kept->num_rows(), 0) << "filter kept no rows — the predicates or the data are wrong";
    note_peak();

    // project: extendedprice * discount  (scale -2 * scale -2 -> scale -4)
    auto revenue_col = cudf::binary_operation(kept->get_column(0).view(),
                                              kept->get_column(1).view(),
                                              cudf::binary_operator::MUL,
                                              cudf::data_type{cudf::type_id::DECIMAL128, -4});
    note_peak();
    // reduce: exact fixed-point sum
    auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    return cudf::reduce(revenue_col->view(), *sum_agg,
                        cudf::data_type{cudf::type_id::DECIMAL128, -4});
  };

  // verify once (also the benchmark's implicit warm-up)
  auto result = execute();
  note_peak();
  auto* fp = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal128>*>(result.get());
  ASSERT_NE(fp, nullptr) << "sum did not come back as a decimal128 scalar";
  ASSERT_TRUE(fp->is_valid());

  Decimal got;
  got.unscaled = static_cast<__int128>(fp->value());
  got.scale = -fp->type().scale();  // cudf scale is negative: -4 => 4 digits after point

  const auto t_done = std::chrono::steady_clock::now();

  // ---------------- COMPARE (exact, value-normalized) ----------------
  const auto golden_raw = read_single_value_golden(golden_path);
  ASSERT_FALSE(golden_raw.empty()) << "empty golden: " << golden_path;
  const Decimal want = parse_decimal(golden_raw);

  std::fprintf(stderr, "[q6] cudf  = %s (scale %d)\n", decimal_to_string(got).c_str(), got.scale);
  std::fprintf(stderr, "[q6] duckdb= %s (scale %d)\n", decimal_to_string(want).c_str(), want.scale);

  EXPECT_TRUE(decimal_values_equal(got, want))
      << "Q6 mismatch (EXACT decimal comparison, no tolerance)\n"
      << "  cudf   : " << decimal_to_string(got) << "\n"
      << "  duckdb : " << decimal_to_string(want) << "\n"
      << "  golden : " << golden_path;

  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q6] load %ld ms, execute %ld ms, total %ld ms\n",
               ms(t0, t_loaded), ms(t_loaded, t_done), ms(t0, t_done));

  // no-op unless PEACOCK_BENCHMARK is set; times the same closure verified above
  benchmark_execute("q6", execute,
                    std::chrono::duration<double, std::milli>(t_loaded - t0).count());
}

// ===========================================================================
// Q1 — pricing summary report
//
//   SELECT l_returnflag, l_linestatus,
//          sum(l_quantity), sum(l_extendedprice),
//          sum(l_extendedprice*(1-l_discount)),
//          sum(l_extendedprice*(1-l_discount)*(1+l_tax)),
//          avg(l_quantity), avg(l_extendedprice), avg(l_discount), count(*)
//   FROM lineitem WHERE l_shipdate <= DATE '1998-09-02'
//   GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus
//
// operator chain: scan -> filter -> project(2 derived cols) -> groupby(8 aggs) -> sort
//
// Per-column comparison semantics (a blanket tolerance would hide errors in columns that
// CAN be exact): the four SUMs are DECIMAL128 compared exactly (scale-normalized), count
// is exact; only the three AVGs get a 1e-9 relative tolerance — forced because both
// DuckDB and cuDF return DOUBLE for AVG and accumulate in different orders. 1e-9 sits ~4
// orders above the observed ~1e-13 drift and ~7 orders below any real bug's effect.
// ===========================================================================

TEST_F(TpchSf40, Q1GroupByAggregates) {
  const auto lineitem_path = data_dir() + "/lineitem.parquet";
  const auto golden_path = golden_dir() + "/duckdb_q1.csv";
  ASSERT_TRUE(file_exists(golden_path))
      << "golden missing: " << golden_path
      << " (regenerate with testdata/gen_duckdb_goldens.sh --sf 40)";

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD (columns only) ----------------
  auto opts = cudf::io::parquet_reader_options::builder(
                  cudf::io::source_info{std::vector<std::string>{lineitem_path}})
                  .columns({"l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice",
                            "l_discount", "l_tax", "l_shipdate"})
                  .build();
  auto loaded = cudf::io::read_parquet(opts);
  auto tbl = loaded.tbl->view();
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();
  ASSERT_EQ(tbl.num_columns(), 7);
  std::fprintf(stderr, "[q1] loaded %ld rows x 7 cols\n", static_cast<long>(tbl.num_rows()));

  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  auto returnflag = tbl.column(0);
  auto linestatus = tbl.column(1);
  auto shipdate = tbl.column(6);

  // ---------------- PHASE 2: EXECUTE ----------------
  // Closure so verify and benchmark run the SAME code; the casts are inside the timed region.
  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    auto quantity = cudf::cast(tbl.column(2), dec128_s2);
    auto extprice = cudf::cast(tbl.column(3), dec128_s2);
    auto discount = cudf::cast(tbl.column(4), dec128_s2);
    auto tax = cudf::cast(tbl.column(5), dec128_s2);

    // filter: l_shipdate <= 1998-09-02
    auto cutoff = cudf::timestamp_scalar<cudf::timestamp_D>(
        cudf::timestamp_D{cudf::duration_D{days_since_epoch(1998, 9, 2)}}, true);
    const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
    auto mask = cudf::binary_operation(shipdate, cutoff, cudf::binary_operator::LESS_EQUAL, boolean);
    auto kept = cudf::apply_boolean_mask(
        cudf::table_view{{returnflag, linestatus, quantity->view(), extprice->view(),
                          discount->view(), tax->view()}},
        mask->view());
    EXPECT_GT(kept->num_rows(), 0);
    note_peak();

    auto k_flag = kept->get_column(0).view();
    auto k_status = kept->get_column(1).view();
    auto k_qty = kept->get_column(2).view();
    auto k_price = kept->get_column(3).view();
    auto k_disc = kept->get_column(4).view();
    auto k_tax = kept->get_column(5).view();

    // project: disc_price = extendedprice * (1 - discount)          scale -4
    //          charge     = disc_price     * (1 + tax)              scale -6
    // The literals 1 are DECIMAL scalars at scale -2 (value 100), so these stay exact.
    auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
    auto one_minus_disc = cudf::binary_operation(one_s2, k_disc, cudf::binary_operator::SUB, dec128_s2);
    auto one_plus_tax = cudf::binary_operation(one_s2, k_tax, cudf::binary_operator::ADD, dec128_s2);
    auto disc_price = cudf::binary_operation(k_price, one_minus_disc->view(),
                                             cudf::binary_operator::MUL,
                                             cudf::data_type{cudf::type_id::DECIMAL128, -4});
    auto charge = cudf::binary_operation(disc_price->view(), one_plus_tax->view(),
                                         cudf::binary_operator::MUL,
                                         cudf::data_type{cudf::type_id::DECIMAL128, -6});

    // AVG must be double on both sides (DuckDB's AVG over DECIMAL returns DOUBLE) — the
    // one place a tolerance applies
    const auto f64 = cudf::data_type{cudf::type_id::FLOAT64};
    auto qty_f = cudf::cast(k_qty, f64);
    auto price_f = cudf::cast(k_price, f64);
    auto disc_f = cudf::cast(k_disc, f64);

    // groupby (l_returnflag, l_linestatus) -> 8 aggregates
    cudf::groupby::groupby gb(cudf::table_view{{k_flag, k_status}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    auto add = [&](cudf::column_view v, std::unique_ptr<cudf::groupby_aggregation> a) {
      cudf::groupby::aggregation_request r;
      r.values = v;
      r.aggregations.push_back(std::move(a));
      reqs.push_back(std::move(r));
    };
    add(k_qty, cudf::make_sum_aggregation<cudf::groupby_aggregation>());          // 0 sum_qty
    add(k_price, cudf::make_sum_aggregation<cudf::groupby_aggregation>());        // 1 sum_base_price
    add(disc_price->view(), cudf::make_sum_aggregation<cudf::groupby_aggregation>());  // 2
    add(charge->view(), cudf::make_sum_aggregation<cudf::groupby_aggregation>());      // 3
    add(qty_f->view(), cudf::make_mean_aggregation<cudf::groupby_aggregation>());      // 4 avg_qty
    add(price_f->view(), cudf::make_mean_aggregation<cudf::groupby_aggregation>());    // 5 avg_price
    add(disc_f->view(), cudf::make_mean_aggregation<cudf::groupby_aggregation>());     // 6 avg_disc
    add(k_qty, cudf::make_count_aggregation<cudf::groupby_aggregation>());             // 7 count

    auto [keys_tbl, agg_results] = gb.aggregate(reqs);
    note_peak();

    // sort by the two group keys — unique per group, so the order is total
    auto keys_view = keys_tbl->view();
    std::vector<std::unique_ptr<cudf::column>> agg_cols;
    for (auto& r : agg_results) agg_cols.push_back(std::move(r.results[0]));
    std::vector<cudf::column_view> all_views{keys_view.column(0), keys_view.column(1)};
    for (auto& c : agg_cols) all_views.push_back(c->view());

    auto order = cudf::sorted_order(cudf::table_view{{keys_view.column(0), keys_view.column(1)}},
                                    {cudf::order::ASCENDING, cudf::order::ASCENDING},
                                    {cudf::null_order::AFTER, cudf::null_order::AFTER});
    return cudf::gather(cudf::table_view{all_views}, order->view());
  };

  auto sorted = execute();
  note_peak();
  const auto t_done = std::chrono::steady_clock::now();

  // ---------------- COMPARE (SUMs and COUNT exact; only AVGs toleranced) ----------------
  constexpr double kAvgRelTol = 1e-9;  // rationale in the header comment
  const std::vector<ColSpec> kQ1Spec = {
      {"l_returnflag", Cmp::ExactString},
      {"l_linestatus", Cmp::ExactString},
      {"sum_qty", Cmp::ExactDecimal},
      {"sum_base_price", Cmp::ExactDecimal},
      {"sum_disc_price", Cmp::ExactDecimal},
      {"sum_charge", Cmp::ExactDecimal},
      {"avg_qty", Cmp::TolerantDouble, kAvgRelTol},
      {"avg_price", Cmp::TolerantDouble, kAvgRelTol},
      {"avg_disc", Cmp::TolerantDouble, kAvgRelTol},
      {"count_order", Cmp::ExactInt},
  };
  const auto golden = read_csv_golden(golden_path);
  const double worst_rel =
      compare_table_to_golden(sorted->view(), golden, kQ1Spec, "q1");

  std::fprintf(stderr, "[q1] worst AVG relative error %.3e (tolerance %.1e)\n",
               worst_rel, kAvgRelTol);
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q1] load %ld ms, execute %ld ms, total %ld ms\n",
               ms(t0, t_loaded), ms(t_loaded, t_done), ms(t0, t_done));

  benchmark_execute("q1", execute,
                    std::chrono::duration<double, std::milli>(t_loaded - t0).count());
}

// ===========================================================================
// Q3 — shipping priority (the first query needing JOINS)
//
//   SELECT l_orderkey, sum(l_extendedprice*(1-l_discount)) AS revenue,
//          o_orderdate, o_shippriority
//   FROM customer, orders, lineitem
//   WHERE c_mktsegment='BUILDING' AND c_custkey=o_custkey AND l_orderkey=o_orderkey
//     AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15'
//   GROUP BY l_orderkey, o_orderdate, o_shippriority
//   ORDER BY revenue DESC, o_orderdate, l_orderkey LIMIT 10
//
// operator chain: scan x3 -> filter x3 -> join -> join -> project -> groupby -> sort -> limit
//
// All comparisons EXACT (revenue DECIMAL128 scale -4; no float enters).
// TIE-BREAK: ORDER BY revenue DESC, o_orderdate is not a total order, so l_orderkey is
// appended here AND in the golden generator — otherwise ties at the LIMIT 10 boundary
// would be flaky.
// ===========================================================================
TEST_F(TpchSf40, Q3JoinsGroupByTopN) {
  const auto golden_path = golden_dir() + "/duckdb_q3.csv";
  ASSERT_TRUE(file_exists(golden_path))
      << "golden missing: " << golden_path
      << " (regenerate with testdata/gen_duckdb_goldens.sh --sf 40)";

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD (all three tables, columns only) ----------------
  auto read_cols = [](const std::string& path, std::vector<std::string> cols) {
    auto o = cudf::io::parquet_reader_options::builder(
                 cudf::io::source_info{std::vector<std::string>{path}})
                 .columns(std::move(cols))
                 .build();
    return cudf::io::read_parquet(o);
  };
  auto cust_in = read_cols(data_dir() + "/customer.parquet", {"c_custkey", "c_mktsegment"});
  auto ord_in = read_cols(data_dir() + "/orders.parquet",
                          {"o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"});
  auto line_in = read_cols(data_dir() + "/lineitem.parquet",
                           {"l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"});
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();
  std::fprintf(stderr, "[q3] loaded customer %ld, orders %ld, lineitem %ld rows\n",
               static_cast<long>(cust_in.tbl->num_rows()),
               static_cast<long>(ord_in.tbl->num_rows()),
               static_cast<long>(line_in.tbl->num_rows()));

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};

  // cudf::inner_join returns gather MAPS, not tables — this wraps one as a column_view so
  // only the needed columns get gathered.
  auto map_view = [](std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& m) {
    return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                             static_cast<cudf::size_type>(m->size()), m->data(), nullptr, 0);
  };

  // Closure so verify and benchmark run the SAME code; LIMIT 10 is a zero-copy slice
  // applied outside in the verify path.
  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    // filter customer: c_mktsegment = 'BUILDING'
    auto seg = cudf::string_scalar(std::string("BUILDING"));
    auto cust_mask = cudf::binary_operation(cust_in.tbl->view().column(1), seg,
                                            cudf::binary_operator::EQUAL, boolean);
    auto cust_f = cudf::apply_boolean_mask(
        cudf::table_view{{cust_in.tbl->view().column(0)}}, cust_mask->view());

    // filter orders: o_orderdate < 1995-03-15
    auto d1995 = cudf::timestamp_scalar<cudf::timestamp_D>(
        cudf::timestamp_D{cudf::duration_D{days_since_epoch(1995, 3, 15)}}, true);
    auto ord_mask = cudf::binary_operation(ord_in.tbl->view().column(2), d1995,
                                           cudf::binary_operator::LESS, boolean);
    auto ord_f = cudf::apply_boolean_mask(ord_in.tbl->view(), ord_mask->view());

    // filter lineitem: l_shipdate > 1995-03-15
    auto line_mask = cudf::binary_operation(line_in.tbl->view().column(3), d1995,
                                            cudf::binary_operator::GREATER, boolean);
    auto line_f = cudf::apply_boolean_mask(line_in.tbl->view(), line_mask->view());
    EXPECT_GT(cust_f->num_rows(), 0);
    EXPECT_GT(ord_f->num_rows(), 0);
    EXPECT_GT(line_f->num_rows(), 0);

    // join 1: customer.c_custkey = orders.o_custkey
    auto [c_map, o_map] = cudf::inner_join(cudf::table_view{{cust_f->get_column(0).view()}},
                                           cudf::table_view{{ord_f->get_column(1).view()}});
    // only orders columns are needed downstream (c_custkey is consumed by the join)
    auto co = cudf::gather(cudf::table_view{{ord_f->get_column(0).view(),    // o_orderkey
                                             ord_f->get_column(2).view(),    // o_orderdate
                                             ord_f->get_column(3).view()}},  // o_shippriority
                           map_view(o_map));

    // join 2: (customer|X|orders).o_orderkey = lineitem.l_orderkey
    auto [co_map, l_map] = cudf::inner_join(cudf::table_view{{co->get_column(0).view()}},
                                            cudf::table_view{{line_f->get_column(0).view()}});
    auto co_side = cudf::gather(cudf::table_view{{co->get_column(0).view(),    // o_orderkey
                                                  co->get_column(1).view(),    // o_orderdate
                                                  co->get_column(2).view()}},  // o_shippriority
                                map_view(co_map));
    auto l_side = cudf::gather(cudf::table_view{{line_f->get_column(1).view(),    // extendedprice
                                                 line_f->get_column(2).view()}},  // discount
                               map_view(l_map));
    note_peak();
    EXPECT_GT(co_side->num_rows(), 0) << "join produced no rows";

    // project: revenue = l_extendedprice * (1 - l_discount), exact decimal
    auto price = cudf::cast(l_side->get_column(0).view(), dec128_s2);
    auto disc = cudf::cast(l_side->get_column(1).view(), dec128_s2);
    auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
    auto one_minus_disc = cudf::binary_operation(one_s2, disc->view(), cudf::binary_operator::SUB, dec128_s2);
    auto revenue = cudf::binary_operation(price->view(), one_minus_disc->view(),
                                          cudf::binary_operator::MUL, dec128_s4);

    // groupby (l_orderkey, o_orderdate, o_shippriority) -> sum(revenue)
    cudf::groupby::groupby gb(cudf::table_view{{co_side->get_column(0).view(),
                                                co_side->get_column(1).view(),
                                                co_side->get_column(2).view()}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    {
      cudf::groupby::aggregation_request r;
      r.values = revenue->view();
      r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    auto [gkeys, gaggs] = gb.aggregate(reqs);

    // sort: revenue DESC, o_orderdate ASC, l_orderkey ASC (total order — see header)
    auto gk = gkeys->view();
    auto rev_col = std::move(gaggs[0].results[0]);
    auto order = cudf::sorted_order(
        cudf::table_view{{rev_col->view(), gk.column(1), gk.column(0)}},
        {cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING},
        {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER});
    return cudf::gather(
        cudf::table_view{{gk.column(0), rev_col->view(), gk.column(1), gk.column(2)}},
        order->view());
  };

  auto sorted = execute();
  note_peak();
  // limit 10 (zero-copy view into the sorted result)
  auto top = cudf::slice(sorted->view(), {0, 10})[0];
  const auto t_done = std::chrono::steady_clock::now();

  // ---------------- COMPARE (exact, no tolerance; o_orderdate compared as days) ----------------
  const std::vector<ColSpec> kQ3Spec = {
      {"l_orderkey", Cmp::ExactInt},
      {"revenue", Cmp::ExactDecimal},
      {"o_orderdate", Cmp::ExactDate},
      {"o_shippriority", Cmp::ExactInt},
  };
  const auto golden = read_csv_golden(golden_path);
  ASSERT_EQ(static_cast<int>(golden.size()), 10) << "golden should hold 10 rows";
  compare_table_to_golden(top, golden, kQ3Spec, "q3");

  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q3] load %ld ms, execute %ld ms, total %ld ms\n",
               ms(t0, t_loaded), ms(t_loaded, t_done), ms(t0, t_done));

  benchmark_execute("q3", execute,
                    std::chrono::duration<double, std::milli>(t_loaded - t0).count());
}

// ===========================================================================
// Q8 — national market share. SEVEN distinct tables, the most of any TPC-H query
// (customer, lineitem, nation, orders, part, region, supplier; nation twice as n1/n2).
//
//   SELECT o_year, sum(CASE WHEN nation='BRAZIL' THEN volume ELSE 0 END)/sum(volume)
//   FROM (part, supplier, lineitem, orders, customer, nation n1, nation n2, region
//         joined on the key chain, with r_name='AMERICA',
//         o_orderdate BETWEEN 1995-01-01 AND 1996-12-31,
//         p_type='ECONOMY ANODIZED STEEL')
//   GROUP BY o_year ORDER BY o_year
//
// Join order is hand-chosen (no optimizer), driven by measured sf40 selectivities.
// BUSHY plan, not left-deep: subtree A (region |X| n1 |X| customer |X| filtered orders)
// and subtree B (filtered part |X| lineitem) are built independently and meet on
// orderkey. B must happen early: lineitem carries no filter of its own, so joining it
// against the ~54k matching parts cuts 240M -> ~1.6M; the textual order (lineitem |X|
// orders first) would materialize ~68M rows — 40x larger for the same answer.
//
// o_year and both volume sums compared EXACTLY; only mkt_share is toleranced (DuckDB's
// DECIMAL/DECIMAL division returns DOUBLE). The golden carries the two sums as their own
// columns specifically so compensating errors cannot cancel inside the ratio.
// ===========================================================================
TEST_F(TpchSf40, Q8SevenTableJoin) {
  const auto golden_path = golden_dir() + "/duckdb_q8.csv";
  ASSERT_TRUE(file_exists(golden_path))
      << "golden missing: " << golden_path
      << " (regenerate with testdata/gen_duckdb_goldens.sh --sf 40)";

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD (seven files, columns only) ----------------
  auto read_cols = [](const std::string& path, std::vector<std::string> cols) {
    auto o = cudf::io::parquet_reader_options::builder(
                 cudf::io::source_info{std::vector<std::string>{path}})
                 .columns(std::move(cols))
                 .build();
    return cudf::io::read_parquet(o);
  };
  auto part_in = read_cols(data_dir() + "/part.parquet", {"p_partkey", "p_type"});
  auto supp_in = read_cols(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  auto line_in = read_cols(data_dir() + "/lineitem.parquet",
                           {"l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount"});
  auto ord_in = read_cols(data_dir() + "/orders.parquet",
                          {"o_orderkey", "o_custkey", "o_orderdate"});
  auto cust_in = read_cols(data_dir() + "/customer.parquet", {"c_custkey", "c_nationkey"});
  // nation is loaded once and joined twice (n1 via customer reaches region; n2 via
  // supplier supplies the output nation name) — same in-memory table, no copy.
  auto nation_in = read_cols(data_dir() + "/nation.parquet",
                             {"n_nationkey", "n_name", "n_regionkey"});
  auto region_in = read_cols(data_dir() + "/region.parquet", {"r_regionkey", "r_name"});
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();
  std::fprintf(stderr,
               "[q8] loaded part %ld, supplier %ld, lineitem %ld, orders %ld, customer %ld, "
               "nation %ld, region %ld\n",
               static_cast<long>(part_in.tbl->num_rows()), static_cast<long>(supp_in.tbl->num_rows()),
               static_cast<long>(line_in.tbl->num_rows()), static_cast<long>(ord_in.tbl->num_rows()),
               static_cast<long>(cust_in.tbl->num_rows()), static_cast<long>(nation_in.tbl->num_rows()),
               static_cast<long>(region_in.tbl->num_rows()));

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};
  const auto f64 = cudf::data_type{cudf::type_id::FLOAT64};
  auto map_view = [](std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& m) {
    return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                             static_cast<cudf::size_type>(m->size()), m->data(), nullptr, 0);
  };

  auto nation_v = nation_in.tbl->view();

  // Closure so verify and benchmark run the SAME code.
  auto execute = [&]() -> std::unique_ptr<cudf::table> {
  // --- subtree A ---
  // A1: region where r_name='AMERICA', then nation n1 on n_regionkey
  auto america = cudf::string_scalar(std::string("AMERICA"));
  auto rmask = cudf::binary_operation(region_in.tbl->view().column(1), america,
                                      cudf::binary_operator::EQUAL, boolean);
  auto region_f = cudf::apply_boolean_mask(
      cudf::table_view{{region_in.tbl->view().column(0)}}, rmask->view());
  auto [n1_map, r_map] = cudf::inner_join(cudf::table_view{{nation_v.column(2)}},   // n_regionkey
                                          cudf::table_view{{region_f->get_column(0).view()}});
  auto n1 = cudf::gather(cudf::table_view{{nation_v.column(0)}}, map_view(n1_map));  // n_nationkey
  std::fprintf(stderr, "[q8] A1 region|X|nation n1 -> %ld nations\n",
               static_cast<long>(n1->num_rows()));

  // A2: customer on c_nationkey
  auto [c_map, n1b_map] = cudf::inner_join(cudf::table_view{{cust_in.tbl->view().column(1)}},
                                           cudf::table_view{{n1->get_column(0).view()}});
  auto cust_am = cudf::gather(cudf::table_view{{cust_in.tbl->view().column(0)}}, map_view(c_map));
  std::fprintf(stderr, "[q8] A2 |X|customer -> %ld\n", static_cast<long>(cust_am->num_rows()));

  // A3: orders filtered to 1995-96, joined on o_custkey
  auto lo = cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(1995, 1, 1)}}, true);
  auto hi = cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(1996, 12, 31)}}, true);
  auto om1 = cudf::binary_operation(ord_in.tbl->view().column(2), lo,
                                    cudf::binary_operator::GREATER_EQUAL, boolean);
  auto om2 = cudf::binary_operation(ord_in.tbl->view().column(2), hi,
                                    cudf::binary_operator::LESS_EQUAL, boolean);
  auto omask = cudf::binary_operation(om1->view(), om2->view(),
                                      cudf::binary_operator::LOGICAL_AND, boolean);
  auto ord_f = cudf::apply_boolean_mask(ord_in.tbl->view(), omask->view());
  note_peak();
  auto [o_map, ca_map] = cudf::inner_join(cudf::table_view{{ord_f->get_column(1).view()}},  // o_custkey
                                          cudf::table_view{{cust_am->get_column(0).view()}});
  auto orders_am = cudf::gather(cudf::table_view{{ord_f->get_column(0).view(),    // o_orderkey
                                                  ord_f->get_column(2).view()}},  // o_orderdate
                                map_view(o_map));
  note_peak();
  std::fprintf(stderr, "[q8] A3 |X|orders(1995-96) -> %ld\n",
               static_cast<long>(orders_am->num_rows()));

  // --- subtree B: part |X| lineitem, the step that keeps this query small ---
  auto ptype = cudf::string_scalar(std::string("ECONOMY ANODIZED STEEL"));
  auto pmask = cudf::binary_operation(part_in.tbl->view().column(1), ptype,
                                      cudf::binary_operator::EQUAL, boolean);
  auto part_f = cudf::apply_boolean_mask(
      cudf::table_view{{part_in.tbl->view().column(0)}}, pmask->view());
  std::fprintf(stderr, "[q8] B0 part filtered -> %ld\n", static_cast<long>(part_f->num_rows()));
  auto [l_map, p_map] = cudf::inner_join(cudf::table_view{{line_in.tbl->view().column(1)}},  // l_partkey
                                         cudf::table_view{{part_f->get_column(0).view()}});
  auto line_p = cudf::gather(cudf::table_view{{line_in.tbl->view().column(0),    // l_orderkey
                                               line_in.tbl->view().column(2),    // l_suppkey
                                               line_in.tbl->view().column(3),    // l_extendedprice
                                               line_in.tbl->view().column(4)}},  // l_discount
                             map_view(l_map));
  note_peak();
  std::fprintf(stderr, "[q8] B1 part|X|lineitem -> %ld (from %ld)\n",
               static_cast<long>(line_p->num_rows()),
               static_cast<long>(line_in.tbl->num_rows()));

  // --- C: the two subtrees meet on orderkey ---
  auto [lp_map, oa_map] = cudf::inner_join(cudf::table_view{{line_p->get_column(0).view()}},
                                           cudf::table_view{{orders_am->get_column(0).view()}});
  auto lp_side = cudf::gather(cudf::table_view{{line_p->get_column(1).view(),    // l_suppkey
                                                line_p->get_column(2).view(),    // l_extendedprice
                                                line_p->get_column(3).view()}},  // l_discount
                              map_view(lp_map));
  auto oa_side = cudf::gather(cudf::table_view{{orders_am->get_column(1).view()}},  // o_orderdate
                              map_view(oa_map));
  note_peak();
  std::fprintf(stderr, "[q8] C A|X|B on orderkey -> %ld\n", static_cast<long>(lp_side->num_rows()));
  EXPECT_GT(lp_side->num_rows(), 0) << "seven-table join produced no rows";

  // --- D: supplier, then nation n2 (the SECOND use of the same nation table) ---
  auto [s_map2, sp_map] = cudf::inner_join(cudf::table_view{{lp_side->get_column(0).view()}},  // l_suppkey
                                           cudf::table_view{{supp_in.tbl->view().column(0)}});
  auto d_price = cudf::gather(cudf::table_view{{lp_side->get_column(1).view(),
                                                lp_side->get_column(2).view()}},
                              map_view(s_map2));
  auto d_date = cudf::gather(cudf::table_view{{oa_side->get_column(0).view()}}, map_view(s_map2));
  auto d_snation = cudf::gather(cudf::table_view{{supp_in.tbl->view().column(1)}},  // s_nationkey
                                map_view(sp_map));
  auto [sn_map, n2_map] = cudf::inner_join(cudf::table_view{{d_snation->get_column(0).view()}},
                                           cudf::table_view{{nation_v.column(0)}});  // n_nationkey
  auto e_price = cudf::gather(cudf::table_view{{d_price->get_column(0).view(),
                                                d_price->get_column(1).view()}},
                              map_view(sn_map));
  auto e_date = cudf::gather(cudf::table_view{{d_date->get_column(0).view()}}, map_view(sn_map));
  auto e_nname = cudf::gather(cudf::table_view{{nation_v.column(1)}}, map_view(n2_map));  // n_name
  note_peak();
  std::fprintf(stderr, "[q8] D |X|supplier|X|nation n2 -> %ld\n",
               static_cast<long>(e_price->num_rows()));

  // --- project: volume, o_year, and the BRAZIL-only volume ---
  auto price = cudf::cast(e_price->get_column(0).view(), dec128_s2);
  auto disc = cudf::cast(e_price->get_column(1).view(), dec128_s2);
  auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
  auto one_minus_disc =
      cudf::binary_operation(one_s2, disc->view(), cudf::binary_operator::SUB, dec128_s2);
  auto volume = cudf::binary_operation(price->view(), one_minus_disc->view(),
                                       cudf::binary_operator::MUL, dec128_s4);
  // extract_datetime_component, NOT extract_year — deprecated in 25.02, REMOVED in 26.02;
  // the generic form is identical in both, so no #if needed.
  auto o_year = cudf::datetime::extract_datetime_component(
      e_date->get_column(0).view(), cudf::datetime::datetime_component::YEAR);
  // CASE WHEN nation='BRAZIL' THEN volume ELSE 0 — copy_if_else against a zero decimal
  auto brazil = cudf::string_scalar(std::string("BRAZIL"));
  auto is_brazil = cudf::binary_operation(e_nname->get_column(0).view(), brazil,
                                          cudf::binary_operator::EQUAL, boolean);
  auto zero_s4 = cudf::fixed_point_scalar<numeric::decimal128>(0, numeric::scale_type{-4});
  auto brazil_volume = cudf::copy_if_else(volume->view(), zero_s4, is_brazil->view());
  note_peak();

  // --- groupby o_year -> sum(brazil_volume), sum(volume) ---
  cudf::groupby::groupby gb(cudf::table_view{{o_year->view()}});
  std::vector<cudf::groupby::aggregation_request> reqs;
  auto add = [&](cudf::column_view v) {
    cudf::groupby::aggregation_request r;
    r.values = v;
    r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    reqs.push_back(std::move(r));
  };
  add(brazil_volume->view());
  add(volume->view());
  auto [ykeys, yaggs] = gb.aggregate(reqs);
  auto brazil_sum = std::move(yaggs[0].results[0]);
  auto total_sum = std::move(yaggs[1].results[0]);

  // mkt_share division in float64 on BOTH sides — DuckDB returns DOUBLE for
  // DECIMAL/DECIMAL, so a decimal quotient would round differently.
  auto brazil_f = cudf::cast(brazil_sum->view(), f64);
  auto total_f = cudf::cast(total_sum->view(), f64);
  auto mkt_share =
      cudf::binary_operation(brazil_f->view(), total_f->view(), cudf::binary_operator::DIV, f64);

  // sort by o_year (one row per year; the key is unique so the order is total)
  auto order = cudf::sorted_order(cudf::table_view{{ykeys->view().column(0)}},
                                  {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  return cudf::gather(cudf::table_view{{ykeys->view().column(0), brazil_sum->view(),
                                        total_sum->view(), mkt_share->view()}},
                      order->view());
  };  // end execute

  auto sorted = execute();
  note_peak();
  const auto t_done = std::chrono::steady_clock::now();

  // ---------------- COMPARE (sums exact; only mkt_share toleranced) ----------------
  constexpr double kShareRelTol = 1e-9;
  const std::vector<ColSpec> kQ8Spec = {
      {"o_year", Cmp::ExactInt},
      {"brazil_volume", Cmp::ExactDecimal},
      {"total_volume", Cmp::ExactDecimal},
      {"mkt_share", Cmp::TolerantDouble, kShareRelTol},
  };
  const auto golden = read_csv_golden(golden_path);
  const double worst_rel = compare_table_to_golden(sorted->view(), golden, kQ8Spec, "q8");

  std::fprintf(stderr, "[q8] worst mkt_share relative error %.3e (tolerance %.1e)\n",
               worst_rel, kShareRelTol);
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q8] load %ld ms, execute %ld ms, total %ld ms\n",
               ms(t0, t_loaded), ms(t_loaded, t_done), ms(t0, t_done));

  benchmark_execute("q8", execute,
                    std::chrono::duration<double, std::milli>(t_loaded - t0).count());
}

// Same entry point as the other gtest binaries here (the conda cudf ships no gtest_main).
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  peacock_test::install_rmm_pool();
  return RUN_ALL_TESTS();
}
