// The SAME TPC-H queries as test_tpch.cpp, executed as MULTI-GPU plans over all visible GPUs
// and checked against the SAME committed DuckDB goldens BYTE-FOR-BYTE (shared helper in
// tpch_golden.hpp).
//
// THE PLANS (all size to cudaGetDeviceCount() — never hardcode a GPU count):
//   Partition lineitem across the G GPUs on WHOLE parquet row-group boundaries (data
//   distribution, NOT predicate pushdown — every query filter still runs as an operator). Each
//   worker owns one contiguous span of row groups for the whole run.
//
//   q6  — embarrassingly parallel. Each worker filters its partition and reduces its partial
//         sum(l_extendedprice*l_discount); the G partial sums are gathered to GPU0 and reduced
//         to the final scalar. Exact decimal throughout, so it matches the single-GPU golden
//         bit-for-bit.
//
//   q1  — each worker does a LOCAL groupby over its partition producing RE-AGGREGATABLE partial
//         aggregates. The subtlety: the three MEAN columns cannot be averaged across partitions
//         — a mean of partial means is wrong. So each worker emits partial SUMs and a partial
//         COUNT per group; the merge on GPU0 sums those and recomputes each mean as
//         sum(partial sums)/sum(partial counts). The exact sums re-aggregate exactly (integer
//         decimal accumulation) so they match the golden exactly; the means match within the
//         same 1e-9 tolerance the single-GPU test uses (they are doubles on both sides).
//
// The plans are correct at ANY partition count, so they run for any G >= 1 (skip only if 0 GPUs
// or the sf40 data is absent, same as test_tpch.cpp). At G=1 it degenerates to one span / one
// worker — the FAIR single-GPU baseline: identical code and identical (pooled) allocator, just
// one partition — so a G=1-vs-G=2 comparison isolates parallelism from allocator choice. NOT
// wired into CI (built for compile coverage; run by hand on a GPU host). PEACOCK_BENCHMARK
// times the execute (inputs resident, the cross-GPU merge included, an all-device sync at the
// boundary, 2nd-min of 6).
//
// The worker-per-GPU model (a device object's whole lifetime stays on its device's thread) and
// all the scaffolding come from multi_gpu.hpp / multi_gpu.cpp.
//
// SHARED PROCESS-WIDE WorkerPool: exactly ONE WorkerPool for the whole binary, owned by
// MultiGpuEnvironment (a gtest Environment) and reused by every query test. Per-test pool
// teardown churns a cudf-26.02 process-global and the next test's benchmark then intermittently
// throws "invalid device ordinal"; with one pool held for the run there is no teardown to
// pollute, so the whole suite (PEACOCK_BENCHMARK, 7 executes each) benchmarks cleanly in ONE
// process.

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/datetime.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#if __has_include(<cudf/join/join.hpp>)
#  include <cudf/join/join.hpp>  // cudf >= 26.02
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

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include "multi_gpu.hpp"
#include "tpch_golden.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace peacock_test;  // TpchSf40 fixture, ColSpec/compare_table_to_golden, Decimal, ...
using namespace peacock_mgpu;  // WorkerPool, partition_row_groups, gather_here, ...

namespace {

const cudf::data_type kBool     = cudf::data_type{cudf::type_id::BOOL8};
const cudf::data_type kDec2     = cudf::data_type{cudf::type_id::DECIMAL128, -2};
const cudf::data_type kDec4     = cudf::data_type{cudf::type_id::DECIMAL128, -4};
const cudf::data_type kDec6     = cudf::data_type{cudf::type_id::DECIMAL128, -6};
const cudf::data_type kFloat64  = cudf::data_type{cudf::type_id::FLOAT64};

// wrap a cudf::inner_join gather map (device_uvector<size_type>) as a column_view for gather().
cudf::column_view map_view(std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& m) {
  return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(m->size()), m->data(), nullptr, 0);
}

cudf::timestamp_scalar<cudf::timestamp_D> date_scalar(int y, unsigned mo, unsigned d,
                                                      rmm::cuda_stream_view s) {
  return cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(y, mo, d)}}, true, s);
}

// Decimal read out of a fixed_point_scalar (for q6's final reduced scalar).
Decimal decimal_from_scalar(cudf::scalar const& s) {
  auto const* fp = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal128> const*>(&s);
  Decimal d;
  if (!fp) {
    ADD_FAILURE() << "expected a decimal128 scalar";
    return d;
  }
  d.unscaled = static_cast<__int128>(fp->value());
  d.scale    = -fp->type().scale();
  return d;
}

// Q6 — multi-GPU embarrassingly-parallel sum. Same query as test_tpch.cpp Q6ExactDecimal.
TEST_F(TpchSf40, Q6MultiGpu) {
  const int G = gpu_count();
  if (G < 1) GTEST_SKIP() << "no visible GPU (found " << G << ")";

  const auto lineitem_path = data_dir() + "/lineitem.parquet";
  const auto golden_path   = golden_dir() + "/duckdb_q6.csv";
  ASSERT_TRUE(file_exists(golden_path))
      << "golden missing: " << golden_path << " (regenerate with gen_duckdb_goldens.sh --sf 40)";

  const int num_rg = parquet_num_row_groups(lineitem_path);
  const auto spans = partition_row_groups(num_rg, G);
  auto& pool = shared_pool();  // one pool for the whole binary (see MultiGpuEnvironment)

  // ---- LOAD: each worker reads ONLY its row-group span (columns only, no predicate). Kept
  //      resident across benchmark iterations. ----
  const auto t_load0 = std::chrono::steady_clock::now();
  std::vector<cudf::io::table_with_metadata> parts(G);
  {
    std::vector<std::future<void>> fs;
    for (int g = 0; g < G; ++g)
      fs.push_back(pool[g].submit([&, g] {
        auto s = pool.stream(g);  // this worker's persistent stream (outlives every object it makes)
        parts[g] = read_row_group_span(
            lineitem_path, {"l_quantity", "l_extendedprice", "l_discount", "l_shipdate"},
            spans[g], s);
        s.synchronize();
      }));
    for (auto& f : fs) f.get();
  }
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

  // ---- EXECUTE: each worker filters + reduces its span to a partial-sum SCALAR; the G
  //      scalars are summed on the HOST. This is the OPTIMAL merge for q6 — the whole result
  //      is one decimal per GPU, so the table-gather machinery (pack/peer-copy/unpack/concat/
  //      reduce, used by q1 for real partial TABLES) is pure overhead here. The partials all
  //      share scale -4, so the host sum is exact __int128 integer addition — bit-identical to
  //      the single-GPU golden. ----
  auto execute = [&]() -> Decimal {
    std::vector<std::future<Decimal>> pf;
    for (int g = 0; g < G; ++g) {
      pf.push_back(pool[g].submit([&, g]() -> Decimal {
        auto s = pool.stream(g);
        auto v = parts[g].tbl->view();
        auto quantity = cudf::cast(v.column(0), kDec2, s);
        auto extprice = cudf::cast(v.column(1), kDec2, s);
        auto discount = cudf::cast(v.column(2), kDec2, s);
        auto shipdate = v.column(3);

        auto lo = cudf::timestamp_scalar<cudf::timestamp_D>(
            cudf::timestamp_D{cudf::duration_D{days_since_epoch(1994, 1, 1)}}, true, s);
        auto hi = cudf::timestamp_scalar<cudf::timestamp_D>(
            cudf::timestamp_D{cudf::duration_D{days_since_epoch(1995, 1, 1)}}, true, s);
        auto dlo = cudf::fixed_point_scalar<numeric::decimal128>(5, numeric::scale_type{-2}, true, s);
        auto dhi = cudf::fixed_point_scalar<numeric::decimal128>(7, numeric::scale_type{-2}, true, s);
        auto qhi = cudf::fixed_point_scalar<numeric::decimal128>(2400, numeric::scale_type{-2}, true, s);

        auto m1 = cudf::binary_operation(shipdate, lo, cudf::binary_operator::GREATER_EQUAL, kBool, s);
        auto m2 = cudf::binary_operation(shipdate, hi, cudf::binary_operator::LESS, kBool, s);
        auto m3 = cudf::binary_operation(discount->view(), dlo, cudf::binary_operator::GREATER_EQUAL, kBool, s);
        auto m4 = cudf::binary_operation(discount->view(), dhi, cudf::binary_operator::LESS_EQUAL, kBool, s);
        auto m5 = cudf::binary_operation(quantity->view(), qhi, cudf::binary_operator::LESS, kBool, s);
        auto mask = cudf::binary_operation(m1->view(), m2->view(), cudf::binary_operator::LOGICAL_AND, kBool, s);
        mask = cudf::binary_operation(mask->view(), m3->view(), cudf::binary_operator::LOGICAL_AND, kBool, s);
        mask = cudf::binary_operation(mask->view(), m4->view(), cudf::binary_operator::LOGICAL_AND, kBool, s);
        mask = cudf::binary_operation(mask->view(), m5->view(), cudf::binary_operator::LOGICAL_AND, kBool, s);

        auto kept = cudf::apply_boolean_mask(
            cudf::table_view{{extprice->view(), discount->view()}}, mask->view(), s);
        auto revenue = cudf::binary_operation(kept->get_column(0).view(), kept->get_column(1).view(),
                                              cudf::binary_operator::MUL, kDec4, s);
        auto sum_agg     = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto partial_sum = cudf::reduce(revenue->view(), *sum_agg, kDec4, s);
        s.synchronize();
        return decimal_from_scalar(*partial_sum);  // read the partial scalar to host
      }));
    }
    // Host merge: exact __int128 addition of the G partials (all scale -4).
    __int128 total = 0;
    int scale      = 4;
    for (int g = 0; g < G; ++g) {
      Decimal d = pf[g].get();
      total += d.unscaled;
      scale = d.scale;  // uniform across partials (reduce output_type kDec4)
    }
    Decimal out;
    out.unscaled = total;
    out.scale    = scale;
    return out;
  };

  Decimal got = execute();
  note_peak();
  const Decimal want = parse_decimal(read_single_value_golden(golden_path));
  ASSERT_TRUE(want.ok) << "golden is not a plain decimal: " << golden_path;
  EXPECT_TRUE(decimal_values_equal(got, want))
      << "Q6 multi-GPU (" << G << " GPUs) EXACT decimal mismatch\n"
      << "  cudf   : " << decimal_to_string(got) << "\n"
      << "  duckdb : " << decimal_to_string(want);
  std::fprintf(stderr, "[q6-mgpu] %d GPUs, %d row groups -> spans; result=%s\n", G, num_rg,
               decimal_to_string(got).c_str());

  benchmark_mgpu("q6-mgpu", execute, G, load_ms);

  // Release each worker's loaded partition ON its owning worker — the columns were allocated
  // from that GPU's pool, and must be freed before WorkerPool tears the pools down (and never
  // on the main thread). See the lifetime note on WorkerPool.
  release_partitions(pool, parts);
}

// Q1 — multi-GPU groupby with partial-aggregate merge. Same query as test_tpch.cpp
// Q1GroupByAggregates. Each worker emits partial SUMs + partial COUNT per group; GPU0 merges.
TEST_F(TpchSf40, Q1MultiGpu) {
  const int G = gpu_count();
  if (G < 1) GTEST_SKIP() << "no visible GPU (found " << G << ")";

  const auto lineitem_path = data_dir() + "/lineitem.parquet";
  const auto golden_path   = golden_dir() + "/duckdb_q1.csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  const int num_rg = parquet_num_row_groups(lineitem_path);
  const auto spans = partition_row_groups(num_rg, G);
  auto& pool = shared_pool();  // one pool for the whole binary (see MultiGpuEnvironment)

  // ---- LOAD (per-worker row-group span). ----
  const auto t_load0 = std::chrono::steady_clock::now();
  std::vector<cudf::io::table_with_metadata> parts(G);
  {
    std::vector<std::future<void>> fs;
    for (int g = 0; g < G; ++g)
      fs.push_back(pool[g].submit([&, g] {
        auto s = pool.stream(g);  // this worker's persistent stream (outlives every object it makes)
        parts[g] = read_row_group_span(
            lineitem_path,
            {"l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice", "l_discount",
             "l_tax", "l_shipdate"},
            spans[g], s);
        s.synchronize();
      }));
    for (auto& f : fs) f.get();
  }
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

  // PARTIAL SCHEMA emitted by each worker's local groupby (keys first, then re-aggregatable
  // partials). Column order is fixed and shared by the merge below:
  //   0 l_returnflag (string, key)   1 l_linestatus (string, key)
  //   2 psum_qty (dec -2)   3 psum_base_price (dec -2)   4 psum_disc_price (dec -4)
  //   5 psum_charge (dec -6)   6 psum_disc (dec -2)   7 pcount (int)
  // psum_disc (col 6) is the extra partial the single-GPU plan does not need: avg_disc must be
  // recombined as sum(disc)/count, and unlike avg_qty/avg_price there is no sum(disc) already
  // among the output columns.
  auto local_partial = [&](int g, rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    auto v = parts[g].tbl->view();
    auto returnflag = v.column(0);
    auto linestatus = v.column(1);
    auto quantity = cudf::cast(v.column(2), kDec2, s);
    auto extprice = cudf::cast(v.column(3), kDec2, s);
    auto discount = cudf::cast(v.column(4), kDec2, s);
    auto tax      = cudf::cast(v.column(5), kDec2, s);
    auto shipdate = v.column(6);

    auto cutoff = cudf::timestamp_scalar<cudf::timestamp_D>(
        cudf::timestamp_D{cudf::duration_D{days_since_epoch(1998, 9, 2)}}, true, s);
    auto mask = cudf::binary_operation(shipdate, cutoff, cudf::binary_operator::LESS_EQUAL, kBool, s);
    auto kept = cudf::apply_boolean_mask(
        cudf::table_view{{returnflag, linestatus, quantity->view(), extprice->view(),
                          discount->view(), tax->view()}},
        mask->view(), s);

    auto k_flag  = kept->get_column(0).view();
    auto k_stat  = kept->get_column(1).view();
    auto k_qty   = kept->get_column(2).view();
    auto k_price = kept->get_column(3).view();
    auto k_disc  = kept->get_column(4).view();
    auto k_tax   = kept->get_column(5).view();

    auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2}, true, s);
    auto one_minus_disc = cudf::binary_operation(one_s2, k_disc, cudf::binary_operator::SUB, kDec2, s);
    auto one_plus_tax   = cudf::binary_operation(one_s2, k_tax, cudf::binary_operator::ADD, kDec2, s);
    auto disc_price = cudf::binary_operation(k_price, one_minus_disc->view(), cudf::binary_operator::MUL, kDec4, s);
    auto charge     = cudf::binary_operation(disc_price->view(), one_plus_tax->view(), cudf::binary_operator::MUL, kDec6, s);

    cudf::groupby::groupby gb(cudf::table_view{{k_flag, k_stat}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    auto add = [&](cudf::column_view val) {
      cudf::groupby::aggregation_request r;
      r.values = val;
      r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    };
    add(k_qty);                 // -> psum_qty
    add(k_price);               // -> psum_base_price
    add(disc_price->view());    // -> psum_disc_price
    add(charge->view());        // -> psum_charge
    add(k_disc);                // -> psum_disc (for avg_disc)
    // a COUNT of any non-null value column gives the per-group row count
    {
      cudf::groupby::aggregation_request r;
      r.values = k_qty;
      r.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    auto [keys, aggs] = gb.aggregate(reqs, s);

    std::vector<std::unique_ptr<cudf::column>> out;
    auto kcols = keys->release();
    out.push_back(std::move(kcols[0]));  // returnflag
    out.push_back(std::move(kcols[1]));  // linestatus
    for (auto& a : aggs) out.push_back(std::move(a.results[0]));  // psum_qty..psum_disc, pcount
    return std::make_unique<cudf::table>(std::move(out));
  };

  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    std::vector<std::unique_ptr<cudf::table>> partial(G);  // kept on each worker
    std::vector<cudf::packed_columns> packed(G);

    std::vector<std::future<PackedPartial>> pf;
    for (int g = 0; g < G; ++g)
      pf.push_back(pool[g].submit([&, g]() -> PackedPartial {
        auto s = pool.stream(g);
        partial[g] = local_partial(g, s);
        packed[g]  = cudf::pack(partial[g]->view(), s);
        s.synchronize();  // partial + packed gpu_data ready before GPU0 peer-copies
        return describe_packed(g, packed[g]);
      }));
    std::vector<PackedPartial> handles(G);
    for (int g = 0; g < G; ++g) handles[g] = pf[g].get();

    // GPU0: gather all G partial tables, concatenate, then MERGE-groupby (sum the partial sums
    // and counts per group), recompute the three means, and sort.
    auto result = pool[0]
        .submit([&]() -> std::unique_ptr<cudf::table> {
          auto s = pool.stream(0);
          std::vector<GatheredTable> gathered;
          gathered.reserve(G);
          std::vector<cudf::table_view> views;
          views.reserve(G);
          for (int g = 0; g < G; ++g) {
            gathered.push_back(gather_here(handles[g], s));
            views.push_back(gathered[g].view);
          }
          auto merged_in = cudf::concatenate(views, s);  // [keys | psums | pcount] * G

          auto mv = merged_in->view();
          // merge-groupby on (returnflag, linestatus): SUM every partial column.
          cudf::groupby::groupby gb(cudf::table_view{{mv.column(0), mv.column(1)}});
          std::vector<cudf::groupby::aggregation_request> reqs;
          for (int c = 2; c <= 7; ++c) {  // psum_qty..psum_disc, pcount
            cudf::groupby::aggregation_request r;
            r.values = mv.column(c);
            r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            reqs.push_back(std::move(r));
          }
          auto [keys, aggs] = gb.aggregate(reqs, s);
          auto sum_qty        = std::move(aggs[0].results[0]);  // dec -2
          auto sum_base_price = std::move(aggs[1].results[0]);  // dec -2
          auto sum_disc_price = std::move(aggs[2].results[0]);  // dec -4
          auto sum_charge     = std::move(aggs[3].results[0]);  // dec -6
          auto sum_disc       = std::move(aggs[4].results[0]);  // dec -2
          auto count_order    = std::move(aggs[5].results[0]);  // int (sum of partial counts)

          // means = sum / count, in float64 (same tolerant-double semantics as the golden).
          auto count_f = cudf::cast(count_order->view(), kFloat64, s);
          auto meanf   = [&](cudf::column_view sum_col) {
            auto sf = cudf::cast(sum_col, kFloat64, s);
            return cudf::binary_operation(sf->view(), count_f->view(), cudf::binary_operator::DIV,
                                          kFloat64, s);
          };
          auto avg_qty   = meanf(sum_qty->view());
          auto avg_price = meanf(sum_base_price->view());
          auto avg_disc  = meanf(sum_disc->view());

          // assemble in the golden's column order, then sort by (returnflag, linestatus).
          auto kv = keys->view();
          std::vector<cudf::column_view> final_cols = {
              kv.column(0),          kv.column(1),        sum_qty->view(),
              sum_base_price->view(), sum_disc_price->view(), sum_charge->view(),
              avg_qty->view(),        avg_price->view(),   avg_disc->view(),
              count_order->view()};
          auto order = cudf::sorted_order(cudf::table_view{{kv.column(0), kv.column(1)}},
                                          {cudf::order::ASCENDING, cudf::order::ASCENDING},
                                          {cudf::null_order::AFTER, cudf::null_order::AFTER},
                                          s);
          auto sorted = cudf::gather(cudf::table_view{final_cols}, order->view(),
                                     cudf::out_of_bounds_policy::DONT_CHECK, s);
          s.synchronize();
          return sorted;
        })
        .get();

    std::vector<std::future<void>> rf;
    for (int g = 0; g < G; ++g)
      rf.push_back(pool[g].submit([&, g] {
        packed[g]  = cudf::packed_columns{};
        partial[g] = nullptr;
      }));
    for (auto& f : rf) f.get();
    return result;
  };

  auto result = execute();
  note_peak();

  // Same per-column semantics as the single-GPU q1: 4 sums + count EXACT, 3 avgs tolerant.
  constexpr double kAvgRelTol = 1e-9;
  const std::vector<ColSpec> spec = {
      {"l_returnflag", Cmp::ExactString},   {"l_linestatus", Cmp::ExactString},
      {"sum_qty", Cmp::ExactDecimal},       {"sum_base_price", Cmp::ExactDecimal},
      {"sum_disc_price", Cmp::ExactDecimal},{"sum_charge", Cmp::ExactDecimal},
      {"avg_qty", Cmp::TolerantDouble, kAvgRelTol}, {"avg_price", Cmp::TolerantDouble, kAvgRelTol},
      {"avg_disc", Cmp::TolerantDouble, kAvgRelTol}, {"count_order", Cmp::ExactInt},
  };
  const auto golden = read_csv_golden(golden_path);
  const double worst = compare_table_to_golden(result->view(), golden, spec, "q1-mgpu");
  std::fprintf(stderr, "[q1-mgpu] %d GPUs, %d row groups; worst AVG rel err %.3e (tol %.1e)\n", G,
               num_rg, worst, kAvgRelTol);

  benchmark_mgpu("q1-mgpu", execute, G, load_ms);
  release_partitions(pool, parts);  // free pool-allocated partitions on their workers
}

// Q3 — 3-way join + high-cardinality group-by, via BROADCAST joins and a HASH-SHUFFLE.
// Same query as test_tpch.cpp Q3JoinsGroupByTopN, same golden.
//
// PLAN & CARDINALITIES (sf40):
//   - lineitem (240M, the fact) is ROW-GROUP-PARTITIONED across the GPUs (columns only).
//   - customer (6M) and orders (60M) are the dimension side; both are BROADCAST (read whole on
//     every GPU) and their join is built LOCALLY on each GPU — customer filters to BUILDING
//     (~1.2M) and orders to o_orderdate<1995-03-15 (~30M), and customer|X|orders on custkey is
//     ~6M rows keyed by orderkey. That ~6M build is redundant per GPU but tiny next to lineitem;
//     broadcasting it avoids shuffling a large join. (Reading full orders per GPU is the
//     broadcast cost; at higher G, build customer|X|orders once and broadcast the 6M result.)
//   - Each GPU joins its lineitem partition against the local customer|X|orders on orderkey and
//     computes revenue -> per-GPU pre-aggregation rows (orderkey, orderdate, shippriority, revenue).
//   - The GROUP-BY key is l_orderkey: HIGH cardinality (~millions of distinct keys), where a
//     gather-all-partials-to-GPU0 merge does NOT scale. Instead HASH-SHUFFLE (murmur3) the
//     pre-agg rows by orderkey so every orderkey lives entirely on one GPU; each GPU then does a
//     COMPLETE local group-by (no cross-GPU partial merge), takes its local top-10, and the final
//     merge is a trivial gather of G×10 rows + a global top-10. Revenue is exact decimal (−4)
//     throughout, so it matches the golden bit-for-bit.
TEST_F(TpchSf40, Q3MultiGpu) {
  const int G = gpu_count();
  if (G < 1) GTEST_SKIP() << "no visible GPU (found " << G << ")";
  const auto golden_path = golden_dir() + "/duckdb_q3.csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  const auto line_path = data_dir() + "/lineitem.parquet";
  const auto cust_path = data_dir() + "/customer.parquet";
  const auto ord_path  = data_dir() + "/orders.parquet";
  const int  num_rg    = parquet_num_row_groups(line_path);
  const auto spans     = partition_row_groups(num_rg, G);
  auto&      pool      = shared_pool();

  // ---- LOAD: lineitem partitioned; customer + orders BROADCAST (whole) on every worker. ----
  const auto t_load0 = std::chrono::steady_clock::now();
  std::vector<cudf::io::table_with_metadata> line(G), cust(G), ord(G);
  {
    std::vector<std::future<void>> fs;
    for (int g = 0; g < G; ++g)
      fs.push_back(pool[g].submit([&, g] {
        auto s   = pool.stream(g);
        line[g]  = read_row_group_span(line_path,
            {"l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"}, spans[g], s);
        cust[g]  = read_full_table(cust_path, {"c_custkey", "c_mktsegment"}, s);
        ord[g]   = read_full_table(ord_path, {"o_orderkey", "o_custkey", "o_orderdate",
                                              "o_shippriority"}, s);
        s.synchronize();
      }));
    for (auto& f : fs) f.get();
  }
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    // Phase 1: per-GPU build customer|X|orders, join lineitem, revenue -> preagg on each worker.
    std::vector<std::unique_ptr<cudf::table>> preagg(G);
    {
      std::vector<std::future<void>> fs;
      for (int g = 0; g < G; ++g)
        fs.push_back(pool[g].submit([&, g] {
          auto s  = pool.stream(g);
          auto lv = line[g].tbl->view();  // orderkey, extprice, discount, shipdate
          auto cv = cust[g].tbl->view();  // c_custkey, c_mktsegment
          auto ov = ord[g].tbl->view();   // o_orderkey, o_custkey, o_orderdate, o_shippriority
          auto d1995 = date_scalar(1995, 3, 15, s);

          // customer where c_mktsegment='BUILDING' -> c_custkey
          auto seg    = cudf::string_scalar(std::string("BUILDING"), true, s);
          auto cmask  = cudf::binary_operation(cv.column(1), seg, cudf::binary_operator::EQUAL, kBool, s);
          auto cust_f = cudf::apply_boolean_mask(cudf::table_view{{cv.column(0)}}, cmask->view(), s);
          // orders where o_orderdate < 1995-03-15
          auto omask  = cudf::binary_operation(ov.column(2), d1995, cudf::binary_operator::LESS, kBool, s);
          auto ord_f  = cudf::apply_boolean_mask(ov, omask->view(), s);
          // customer |X| orders on custkey -> dim(orderkey, orderdate, shippriority)
          auto [c_map, o_map] = cudf::inner_join(cudf::table_view{{cust_f->get_column(0).view()}},
                                                 cudf::table_view{{ord_f->get_column(1).view()}},
                                                 cudf::null_equality::EQUAL, s);
          auto dim = cudf::gather(cudf::table_view{{ord_f->get_column(0).view(),
                                                    ord_f->get_column(2).view(),
                                                    ord_f->get_column(3).view()}},
                                  map_view(o_map), cudf::out_of_bounds_policy::DONT_CHECK, s);
          // lineitem where l_shipdate > 1995-03-15 -> orderkey, extprice, discount
          auto lmask  = cudf::binary_operation(lv.column(3), d1995, cudf::binary_operator::GREATER, kBool, s);
          auto line_f = cudf::apply_boolean_mask(
              cudf::table_view{{lv.column(0), lv.column(1), lv.column(2)}}, lmask->view(), s);
          // lineitem |X| dim on orderkey
          auto [l_map, d_map] = cudf::inner_join(cudf::table_view{{line_f->get_column(0).view()}},
                                                 cudf::table_view{{dim->get_column(0).view()}},
                                                 cudf::null_equality::EQUAL, s);
          auto lj = cudf::gather(cudf::table_view{{line_f->get_column(1).view(),
                                                   line_f->get_column(2).view()}},
                                 map_view(l_map), cudf::out_of_bounds_policy::DONT_CHECK, s);
          auto dj = cudf::gather(dim->view(), map_view(d_map),
                                 cudf::out_of_bounds_policy::DONT_CHECK, s);  // orderkey,date,prio
          // revenue = extprice*(1-discount), exact decimal (-2 * -2 -> -4)
          auto price = cudf::cast(lj->get_column(0).view(), kDec2, s);
          auto disc  = cudf::cast(lj->get_column(1).view(), kDec2, s);
          auto one   = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2}, true, s);
          auto omd   = cudf::binary_operation(one, disc->view(), cudf::binary_operator::SUB, kDec2, s);
          auto rev   = cudf::binary_operation(price->view(), omd->view(), cudf::binary_operator::MUL, kDec4, s);
          // preagg = (orderkey, orderdate, shippriority, revenue)
          std::vector<std::unique_ptr<cudf::column>> cols;
          cols.push_back(std::make_unique<cudf::column>(dj->get_column(0).view(), s));
          cols.push_back(std::make_unique<cudf::column>(dj->get_column(1).view(), s));
          cols.push_back(std::make_unique<cudf::column>(dj->get_column(2).view(), s));
          cols.push_back(std::move(rev));
          preagg[g] = std::make_unique<cudf::table>(std::move(cols));
          s.synchronize();
        }));
      for (auto& f : fs) f.get();
    }

    // Phase 2: HASH-SHUFFLE the pre-agg rows by orderkey (col 0) so each orderkey lands on one GPU.
    std::vector<cudf::table_view> preagg_views;
    preagg_views.reserve(G);
    for (int g = 0; g < G; ++g) preagg_views.push_back(preagg[g]->view());
    auto shuffled = hash_shuffle(pool, preagg_views, {0});
    {  // preagg is fully consumed by the shuffle; release it on its workers
      std::vector<std::future<void>> fs;
      for (int g = 0; g < G; ++g) fs.push_back(pool[g].submit([&, g] { preagg[g].reset(); }));
      for (auto& f : fs) f.get();
    }

    // Phase 3: each GPU does a COMPLETE local group-by (orderkey,orderdate,shippriority)->sum(rev),
    // takes its local top-10 (revenue DESC, orderdate, orderkey), packs it for the final gather.
    std::vector<cudf::packed_columns> local_top(G);
    std::vector<PackedPartial>        tops(G);
    {
      std::vector<std::future<void>> fs;
      for (int p = 0; p < G; ++p)
        fs.push_back(pool[p].submit([&, p] {
          auto s  = pool.stream(p);
          auto sv = shuffled[p]->view();  // orderkey, orderdate, shippriority, revenue
          cudf::groupby::groupby gb(cudf::table_view{{sv.column(0), sv.column(1), sv.column(2)}});
          std::vector<cudf::groupby::aggregation_request> reqs;
          {
            cudf::groupby::aggregation_request r;
            r.values = sv.column(3);
            r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            reqs.push_back(std::move(r));
          }
          auto [keys, aggs] = gb.aggregate(reqs, s);
          auto kv  = keys->view();
          auto rev = std::move(aggs[0].results[0]);
          // order by revenue DESC, orderdate ASC, orderkey ASC (total order)
          auto order = cudf::sorted_order(
              cudf::table_view{{rev->view(), kv.column(1), kv.column(0)}},
              {cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING},
              {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER}, s);
          // output cols in golden order: orderkey, revenue, orderdate, shippriority
          auto sorted = cudf::gather(
              cudf::table_view{{kv.column(0), rev->view(), kv.column(1), kv.column(2)}},
              order->view(), cudf::out_of_bounds_policy::DONT_CHECK, s);
          const cudf::size_type n = std::min<cudf::size_type>(10, sorted->num_rows());
          auto top      = cudf::slice(sorted->view(), {0, n})[0];
          local_top[p]  = cudf::pack(top, s);
          tops[p]       = describe_packed(p, local_top[p]);
          s.synchronize();
        }));
      for (auto& f : fs) f.get();
    }

    // Phase 4: gather the G local top-10s to GPU0, concat, global top-10.
    auto result = pool[0]
        .submit([&]() -> std::unique_ptr<cudf::table> {
          auto s = pool.stream(0);
          std::vector<GatheredTable> gts;
          gts.reserve(G);
          std::vector<cudf::table_view> views;
          views.reserve(G);
          for (int p = 0; p < G; ++p) {
            gts.push_back(gather_here(tops[p], s));
            views.push_back(gts.back().view);
          }
          auto merged = cudf::concatenate(views, s);
          auto mv     = merged->view();  // orderkey, revenue, orderdate, shippriority
          auto order  = cudf::sorted_order(
              cudf::table_view{{mv.column(1), mv.column(2), mv.column(0)}},
              {cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING},
              {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER}, s);
          auto sorted = cudf::gather(mv, order->view(), cudf::out_of_bounds_policy::DONT_CHECK, s);
          const cudf::size_type n = std::min<cudf::size_type>(10, sorted->num_rows());
          auto top = std::make_unique<cudf::table>(cudf::slice(sorted->view(), {0, n})[0], s);
          s.synchronize();
          return top;
        })
        .get();

    {  // release the per-GPU packed tops on their workers
      std::vector<std::future<void>> fs;
      for (int p = 0; p < G; ++p)
        fs.push_back(pool[p].submit([&, p] { local_top[p] = cudf::packed_columns{}; }));
      for (auto& f : fs) f.get();
    }
    // BUG (ticket #121): `shuffled` is destroyed at closure end on the CALLING thread,
    // so shuffled[p] for p!=0 frees GPU-p pool memory with device 0 current —
    // the worker-per-GPU destruction rule this file states everywhere else.
    (void)shuffled;
    return result;
  };

  auto result = execute();
  const std::vector<ColSpec> spec = {
      {"l_orderkey", Cmp::ExactInt},   {"revenue", Cmp::ExactDecimal},
      {"o_orderdate", Cmp::ExactDate}, {"o_shippriority", Cmp::ExactInt},
  };
  const auto golden = read_csv_golden(golden_path);
  ASSERT_EQ(static_cast<int>(golden.size()), 10) << "golden should hold 10 rows";
  compare_table_to_golden(result->view(), golden, spec, "q3-mgpu");
  std::fprintf(stderr, "[q3-mgpu] %d GPUs, %d row groups, hash-shuffle by orderkey\n", G, num_rg);

  benchmark_mgpu("q3-mgpu", execute, G, load_ms);
  release_partitions(pool, line);
  release_partitions(pool, cust);
  release_partitions(pool, ord);
}

// Q8 — 7 tables, bushy join order, LOW-cardinality group-by. Same query as test_tpch.cpp
// Q8SevenTableJoin, same golden.
//
// PLAN & CARDINALITIES (sf40):
//   - lineitem (240M) is the fact — ROW-GROUP-PARTITIONED across the GPUs.
//   - part/supplier/orders/customer/nation/region are all SMALL and BROADCAST (read whole on
//     every GPU). The whole dimension subtree (region=AMERICA -> nation n1 -> customer ->
//     orders(1995-96), and part=ECONOMY-ANODIZED-STEEL, and supplier -> nation n2) is built
//     LOCALLY on each GPU from those broadcast tables — identical on every GPU, cheap next to
//     lineitem, and it lets every lineitem partition join a fully-local dim side with no shuffle.
//   - The single step that keeps this query small is part |X| lineitem: p_type='ECONOMY ANODIZED
//     STEEL' keeps ~1/150 of part, so joining it onto lineitem FIRST collapses the 240M fact to a
//     few hundred K rows BEFORE any other join. That collapse happens independently inside each
//     partition (part_f is broadcast), so partitioning loses nothing.
//   - GROUP-BY key is o_year: only 1995 and 1996 — LOW cardinality. So NO shuffle: each GPU
//     emits a partial group-by (o_year -> partial sum(brazil_volume), partial sum(volume)), the G
//     partials are gathered to GPU0 (pack->peer-copy->unpack->concat) and merged with a final
//     sum-group-by. Both sums are EXACT decimal(-4) so they re-aggregate bit-for-bit; mkt_share is
//     recomputed as sum(brazil)/sum(total) in float64 on GPU0 (DuckDB returns DOUBLE for the
//     decimal/decimal division — matching semantics, tolerant 1e-9), NEVER by averaging partial
//     ratios.
TEST_F(TpchSf40, Q8MultiGpu) {
  const int G = gpu_count();
  if (G < 1) GTEST_SKIP() << "no visible GPU (found " << G << ")";
  const auto golden_path = golden_dir() + "/duckdb_q8.csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  const auto line_path = data_dir() + "/lineitem.parquet";
  const int  num_rg    = parquet_num_row_groups(line_path);
  const auto spans     = partition_row_groups(num_rg, G);
  auto&      pool      = shared_pool();

  // ---- LOAD: lineitem partitioned; all six dimension tables BROADCAST on every worker. ----
  const auto t_load0 = std::chrono::steady_clock::now();
  std::vector<cudf::io::table_with_metadata> line(G), part(G), supp(G), ord(G), cust(G),
      nation(G), region(G);
  {
    std::vector<std::future<void>> fs;
    for (int g = 0; g < G; ++g)
      fs.push_back(pool[g].submit([&, g] {
        auto s    = pool.stream(g);
        line[g]   = read_row_group_span(line_path,
            {"l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount"}, spans[g], s);
        part[g]   = read_full_table(data_dir() + "/part.parquet", {"p_partkey", "p_type"}, s);
        supp[g]   = read_full_table(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"}, s);
        ord[g]    = read_full_table(data_dir() + "/orders.parquet",
                                    {"o_orderkey", "o_custkey", "o_orderdate"}, s);
        cust[g]   = read_full_table(data_dir() + "/customer.parquet", {"c_custkey", "c_nationkey"}, s);
        nation[g] = read_full_table(data_dir() + "/nation.parquet",
                                    {"n_nationkey", "n_name", "n_regionkey"}, s);
        region[g] = read_full_table(data_dir() + "/region.parquet", {"r_regionkey", "r_name"}, s);
        s.synchronize();
      }));
    for (auto& f : fs) f.get();
  }
  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

  // Per-GPU partial: build the broadcast dim subtree, join the local lineitem partition through it,
  // group by o_year -> (o_year, partial sum(brazil_volume), partial sum(volume)). All exact decimal.
  auto local_partial = [&](int g, rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    auto lv = line[g].tbl->view();     // orderkey, partkey, suppkey, extprice, discount
    auto pv = part[g].tbl->view();     // p_partkey, p_type
    auto sv = supp[g].tbl->view();     // s_suppkey, s_nationkey
    auto ov = ord[g].tbl->view();      // o_orderkey, o_custkey, o_orderdate
    auto cv = cust[g].tbl->view();     // c_custkey, c_nationkey
    auto nv = nation[g].tbl->view();   // n_nationkey, n_name, n_regionkey
    auto rv = region[g].tbl->view();   // r_regionkey, r_name

    // A1: region='AMERICA' -> nation n1 on regionkey -> n_nationkey
    auto america = cudf::string_scalar(std::string("AMERICA"), true, s);
    auto rmask   = cudf::binary_operation(rv.column(1), america, cudf::binary_operator::EQUAL, kBool, s);
    auto region_f = cudf::apply_boolean_mask(cudf::table_view{{rv.column(0)}}, rmask->view(), s);
    auto [n1_map, r_map] = cudf::inner_join(cudf::table_view{{nv.column(2)}},
                                            cudf::table_view{{region_f->get_column(0).view()}},
                                            cudf::null_equality::EQUAL, s);
    auto n1 = cudf::gather(cudf::table_view{{nv.column(0)}}, map_view(n1_map),
                           cudf::out_of_bounds_policy::DONT_CHECK, s);  // n_nationkey (in AMERICA)
    // A2: customer on c_nationkey -> c_custkey
    auto [c_map, n1b_map] = cudf::inner_join(cudf::table_view{{cv.column(1)}},
                                             cudf::table_view{{n1->get_column(0).view()}},
                                             cudf::null_equality::EQUAL, s);
    auto cust_am = cudf::gather(cudf::table_view{{cv.column(0)}}, map_view(c_map),
                               cudf::out_of_bounds_policy::DONT_CHECK, s);  // c_custkey
    // A3: orders in 1995-96 on o_custkey -> (o_orderkey, o_orderdate)
    auto d95 = date_scalar(1995, 1, 1, s);
    auto d96 = date_scalar(1996, 12, 31, s);
    auto om1 = cudf::binary_operation(ov.column(2), d95, cudf::binary_operator::GREATER_EQUAL, kBool, s);
    auto om2 = cudf::binary_operation(ov.column(2), d96, cudf::binary_operator::LESS_EQUAL, kBool, s);
    auto omask = cudf::binary_operation(om1->view(), om2->view(), cudf::binary_operator::LOGICAL_AND, kBool, s);
    auto ord_f = cudf::apply_boolean_mask(ov, omask->view(), s);
    auto [o_map, ca_map] = cudf::inner_join(cudf::table_view{{ord_f->get_column(1).view()}},
                                            cudf::table_view{{cust_am->get_column(0).view()}},
                                            cudf::null_equality::EQUAL, s);
    auto orders_am = cudf::gather(cudf::table_view{{ord_f->get_column(0).view(),
                                                    ord_f->get_column(2).view()}},
                                  map_view(o_map), cudf::out_of_bounds_policy::DONT_CHECK, s);  // orderkey, orderdate

    // B: part='ECONOMY ANODIZED STEEL' -> part_f, then part_f |X| lineitem (collapses the fact)
    auto ptype = cudf::string_scalar(std::string("ECONOMY ANODIZED STEEL"), true, s);
    auto pmask = cudf::binary_operation(pv.column(1), ptype, cudf::binary_operator::EQUAL, kBool, s);
    auto part_f = cudf::apply_boolean_mask(cudf::table_view{{pv.column(0)}}, pmask->view(), s);
    auto [l_map, p_map] = cudf::inner_join(cudf::table_view{{lv.column(1)}},  // l_partkey
                                           cudf::table_view{{part_f->get_column(0).view()}},
                                           cudf::null_equality::EQUAL, s);
    auto line_p = cudf::gather(cudf::table_view{{lv.column(0), lv.column(2), lv.column(3), lv.column(4)}},
                               map_view(l_map), cudf::out_of_bounds_policy::DONT_CHECK, s);
    //             line_p: l_orderkey, l_suppkey, l_extendedprice, l_discount

    // C: line_p |X| orders_am on orderkey
    auto [lp_map, oa_map] = cudf::inner_join(cudf::table_view{{line_p->get_column(0).view()}},
                                             cudf::table_view{{orders_am->get_column(0).view()}},
                                             cudf::null_equality::EQUAL, s);
    auto lp_side = cudf::gather(cudf::table_view{{line_p->get_column(1).view(),   // l_suppkey
                                                  line_p->get_column(2).view(),   // extprice
                                                  line_p->get_column(3).view()}}, // discount
                                map_view(lp_map), cudf::out_of_bounds_policy::DONT_CHECK, s);
    auto oa_side = cudf::gather(cudf::table_view{{orders_am->get_column(1).view()}},  // o_orderdate
                                map_view(oa_map), cudf::out_of_bounds_policy::DONT_CHECK, s);

    // D: |X| supplier on suppkey -> reach s_nationkey; carry price/discount and date
    auto [s_map2, sp_map] = cudf::inner_join(cudf::table_view{{lp_side->get_column(0).view()}},  // l_suppkey
                                             cudf::table_view{{sv.column(0)}},                    // s_suppkey
                                             cudf::null_equality::EQUAL, s);
    auto d_price = cudf::gather(cudf::table_view{{lp_side->get_column(1).view(),
                                                  lp_side->get_column(2).view()}},
                                map_view(s_map2), cudf::out_of_bounds_policy::DONT_CHECK, s);
    auto d_date  = cudf::gather(cudf::table_view{{oa_side->get_column(0).view()}}, map_view(s_map2),
                                cudf::out_of_bounds_policy::DONT_CHECK, s);
    auto d_snation = cudf::gather(cudf::table_view{{sv.column(1)}}, map_view(sp_map),  // s_nationkey
                                  cudf::out_of_bounds_policy::DONT_CHECK, s);
    // E: |X| nation n2 on nationkey -> n_name (the SECOND use of the same nation table)
    auto [sn_map, n2_map] = cudf::inner_join(cudf::table_view{{d_snation->get_column(0).view()}},
                                             cudf::table_view{{nv.column(0)}},
                                             cudf::null_equality::EQUAL, s);
    auto e_price = cudf::gather(cudf::table_view{{d_price->get_column(0).view(),
                                                  d_price->get_column(1).view()}},
                                map_view(sn_map), cudf::out_of_bounds_policy::DONT_CHECK, s);
    auto e_date  = cudf::gather(cudf::table_view{{d_date->get_column(0).view()}}, map_view(sn_map),
                                cudf::out_of_bounds_policy::DONT_CHECK, s);
    auto e_nname = cudf::gather(cudf::table_view{{nv.column(1)}}, map_view(n2_map),  // n_name
                                cudf::out_of_bounds_policy::DONT_CHECK, s);

    // project: volume = extprice*(1-discount) (exact dec-4), o_year, brazil-only volume
    auto price = cudf::cast(e_price->get_column(0).view(), kDec2, s);
    auto disc  = cudf::cast(e_price->get_column(1).view(), kDec2, s);
    auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2}, true, s);
    auto omd    = cudf::binary_operation(one_s2, disc->view(), cudf::binary_operator::SUB, kDec2, s);
    auto volume = cudf::binary_operation(price->view(), omd->view(), cudf::binary_operator::MUL, kDec4, s);
    auto o_year = cudf::datetime::extract_datetime_component(
        e_date->get_column(0).view(), cudf::datetime::datetime_component::YEAR);
    auto brazil = cudf::string_scalar(std::string("BRAZIL"), true, s);
    auto is_brazil = cudf::binary_operation(e_nname->get_column(0).view(), brazil,
                                            cudf::binary_operator::EQUAL, kBool, s);
    auto zero_s4 = cudf::fixed_point_scalar<numeric::decimal128>(0, numeric::scale_type{-4}, true, s);
    auto brazil_volume = cudf::copy_if_else(volume->view(), zero_s4, is_brazil->view(), s);

    // local partial group-by o_year -> sum(brazil_volume), sum(volume)
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
    auto [ykeys, yaggs] = gb.aggregate(reqs, s);

    std::vector<std::unique_ptr<cudf::column>> out;
    auto kcols = ykeys->release();
    out.push_back(std::move(kcols[0]));           // o_year
    out.push_back(std::move(yaggs[0].results[0]));  // partial sum(brazil_volume)
    out.push_back(std::move(yaggs[1].results[0]));  // partial sum(volume)
    return std::make_unique<cudf::table>(std::move(out));
  };

  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    // Phase 1: each worker builds its partial (o_year, brazil_psum, total_psum) and packs it.
    std::vector<std::unique_ptr<cudf::table>> partial(G);
    std::vector<cudf::packed_columns>         packed(G);
    std::vector<std::future<PackedPartial>>   pf;
    for (int g = 0; g < G; ++g)
      pf.push_back(pool[g].submit([&, g]() -> PackedPartial {
        auto s     = pool.stream(g);
        partial[g] = local_partial(g, s);
        packed[g]  = cudf::pack(partial[g]->view(), s);
        s.synchronize();
        return describe_packed(g, packed[g]);
      }));
    std::vector<PackedPartial> handles(G);
    for (int g = 0; g < G; ++g) handles[g] = pf[g].get();

    // Phase 2: GPU0 gathers the G partials, concatenates, merge-group-by o_year (SUM the partial
    // sums — exact), recomputes mkt_share in float64, sorts by o_year.
    auto result = pool[0]
        .submit([&]() -> std::unique_ptr<cudf::table> {
          auto s = pool.stream(0);
          std::vector<GatheredTable> gathered;
          gathered.reserve(G);
          std::vector<cudf::table_view> views;
          views.reserve(G);
          for (int g = 0; g < G; ++g) {
            gathered.push_back(gather_here(handles[g], s));
            views.push_back(gathered[g].view);
          }
          auto merged = cudf::concatenate(views, s);  // [o_year | brazil_psum | total_psum] * G
          auto mv     = merged->view();
          cudf::groupby::groupby gb(cudf::table_view{{mv.column(0)}});
          std::vector<cudf::groupby::aggregation_request> reqs;
          for (int c = 1; c <= 2; ++c) {
            cudf::groupby::aggregation_request r;
            r.values = mv.column(c);
            r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            reqs.push_back(std::move(r));
          }
          auto [keys, aggs] = gb.aggregate(reqs, s);
          auto brazil_sum = std::move(aggs[0].results[0]);  // dec -4, exact
          auto total_sum  = std::move(aggs[1].results[0]);  // dec -4, exact
          auto brazil_f = cudf::cast(brazil_sum->view(), kFloat64, s);
          auto total_f  = cudf::cast(total_sum->view(), kFloat64, s);
          auto mkt_share = cudf::binary_operation(brazil_f->view(), total_f->view(),
                                                  cudf::binary_operator::DIV, kFloat64, s);
          auto kv    = keys->view();
          auto order = cudf::sorted_order(cudf::table_view{{kv.column(0)}}, {cudf::order::ASCENDING},
                                          {cudf::null_order::AFTER}, s);
          auto sorted = cudf::gather(
              cudf::table_view{{kv.column(0), brazil_sum->view(), total_sum->view(), mkt_share->view()}},
              order->view(), cudf::out_of_bounds_policy::DONT_CHECK, s);
          s.synchronize();
          return sorted;
        })
        .get();

    std::vector<std::future<void>> rf;
    for (int g = 0; g < G; ++g)
      rf.push_back(pool[g].submit([&, g] {
        packed[g]  = cudf::packed_columns{};
        partial[g] = nullptr;
      }));
    for (auto& f : rf) f.get();
    return result;
  };

  auto result = execute();
  constexpr double kShareRelTol = 1e-9;
  const std::vector<ColSpec> spec = {
      {"o_year", Cmp::ExactInt},          {"brazil_volume", Cmp::ExactDecimal},
      {"total_volume", Cmp::ExactDecimal},{"mkt_share", Cmp::TolerantDouble, kShareRelTol},
  };
  const auto golden = read_csv_golden(golden_path);
  const double worst = compare_table_to_golden(result->view(), golden, spec, "q8-mgpu");
  std::fprintf(stderr, "[q8-mgpu] %d GPUs, %d row groups; worst mkt_share rel err %.3e (tol %.1e)\n",
               G, num_rg, worst, kShareRelTol);

  benchmark_mgpu("q8-mgpu", execute, G, load_ms);
  release_partitions(pool, line);
  release_partitions(pool, part);
  release_partitions(pool, supp);
  release_partitions(pool, ord);
  release_partitions(pool, cust);
  release_partitions(pool, nation);
  release_partitions(pool, region);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MultiGpuEnvironment);  // one shared WorkerPool
  return RUN_ALL_TESTS();
}
