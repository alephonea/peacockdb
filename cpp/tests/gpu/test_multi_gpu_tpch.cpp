// test_multi_gpu_tpch.cpp — the SAME TPC-H q6 and q1 as test_tpch.cpp, but executed as
// MULTI-GPU plans over all visible GPUs, and checked against the SAME committed DuckDB goldens
// BYTE-FOR-BYTE (via the shared helper in tpch_golden.hpp). Milestone 1: no joins, no shuffle.
//
// THE PLANS (both size to cudaGetDeviceCount() — never hardcode a GPU count):
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
// The plan is correct at ANY partition count, so it runs for any G >= 1 (skip only if 0 GPUs
// or the sf40 data is absent, same as test_tpch.cpp). At G=1 it degenerates to one span / one
// worker — the FAIR single-GPU baseline: identical code and identical (pooled) allocator, just
// one partition — so a G=1-vs-G=2 comparison isolates parallelism from allocator choice. NOT
// wired into CI (built for compile coverage; run by hand on a GPU host). PEACOCK_BENCHMARK
// times the execute (inputs resident, the cross-GPU merge included, an all-device sync at the
// boundary, 2nd-min of 6).
//
// The worker-per-GPU model (a device object's whole lifetime stays on its device's thread) and
// all the scaffolding come from the shared test library multi_gpu.hpp / multi_gpu.cpp.
//
// BENCHMARK ONE QUERY PER PROCESS. Correctness (a single execute per test) is reliable running
// the whole binary in one process. But running MULTIPLE query BENCHMARKS (PEACOCK_BENCHMARK,
// which executes each plan 7x) in one process is currently FLAKY at G>=2: a cudf-26.02
// process-global piece of state (its internal stream/scratch pool, device-0-biased) is churned
// when one test's WorkerPool tears its per-device pools down, and the next test's multi-execute
// benchmark then intermittently throws "invalid device ordinal" in a downstream scan. So to
// benchmark, run each query in its OWN process, e.g.
//     PEACOCK_BENCHMARK=1 ./peacock_multi_gpu_tpch_tests --gtest_filter='*Q6*'
// The real fix is a PROCESS-WIDE shared WorkerPool (created once for the whole binary, reused
// across every test, so there is no per-test teardown to pollute) — M2's first task.

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream.hpp>

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
using namespace peacock_mgpu;  // WorkerPool, partition_row_groups, gather_to_gpu0, ...

namespace {

const cudf::data_type kBool     = cudf::data_type{cudf::type_id::BOOL8};
const cudf::data_type kDec2     = cudf::data_type{cudf::type_id::DECIMAL128, -2};
const cudf::data_type kDec4     = cudf::data_type{cudf::type_id::DECIMAL128, -4};
const cudf::data_type kDec6     = cudf::data_type{cudf::type_id::DECIMAL128, -6};
const cudf::data_type kFloat64  = cudf::data_type{cudf::type_id::FLOAT64};

int gpu_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    cudaGetLastError();
    n = 0;
  }
  return n;
}

// Execute-time benchmark for a multi-GPU plan: same 2nd-min-of-6 protocol as test_tpch.cpp,
// but the boundary sync drains EVERY device (the plan's work is spread across all of them),
// not just the current one. No-op unless PEACOCK_BENCHMARK is set.
template <typename F>
void benchmark_mgpu(const char* tag, F&& execute, int num_gpus, double load_ms) {
  if (!benchmark_enabled()) return;
  const int runs = std::max(2, std::atoi(env_or("PEACOCK_BENCHMARK_RUNS", "6").c_str()));
  auto sync_all = [num_gpus] {
    for (int g = 0; g < num_gpus; ++g) {
      cudaSetDevice(g);
      cudaDeviceSynchronize();
    }
    cudaSetDevice(0);
  };
  std::vector<double> ms;
  ms.reserve(runs);
  for (int i = 0; i < runs; ++i) {
    sync_all();
    const auto t0 = std::chrono::steady_clock::now();
    auto result = execute();
    sync_all();  // include the cross-GPU merge and every device's work in the timed region
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    (void)result;
  }
  std::sort(ms.begin(), ms.end());
  const double second_min = ms.size() > 1 ? ms[1] : ms[0];
  std::string all;
  for (size_t i = 0; i < ms.size(); ++i) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%s%.3f", i ? "," : "", ms[i]);
    all += buf;
  }
  std::fprintf(stderr, "[bench] %s execute_ms=%.3f load_ms=%.3f gpus=%d runs=%d all=[%s]\n", tag,
               second_min, load_ms, num_gpus, runs, all.c_str());
}

// Free each worker's loaded partition ON its owning worker (device current), so pool-allocated
// columns are destroyed on the right thread/device before the WorkerPool tears its pools down.
void release_partitions(WorkerPool& pool, std::vector<cudf::io::table_with_metadata>& parts) {
  std::vector<std::future<void>> fs;
  for (int g = 0; g < pool.size(); ++g)
    fs.push_back(pool[g].submit([&, g] { parts[g] = cudf::io::table_with_metadata{}; }));
  for (auto& f : fs) f.get();
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

// ===========================================================================
// Q6 — multi-GPU embarrassingly-parallel sum. Same query as test_tpch.cpp Q6ExactDecimal.
// ===========================================================================
TEST_F(TpchSf40, Q6MultiGpu) {
  const int G = gpu_count();
  if (G < 1) GTEST_SKIP() << "no visible GPU (found " << G << ")";

  const auto lineitem_path = data_dir() + "/lineitem.parquet";
  const auto golden_path   = golden_dir() + "/duckdb_q6.csv";
  ASSERT_TRUE(file_exists(golden_path))
      << "golden missing: " << golden_path << " (regenerate with gen_duckdb_goldens.sh --sf 40)";

  const int num_rg = parquet_num_row_groups(lineitem_path);
  const auto spans = partition_row_groups(num_rg, G);
  WorkerPool pool(G);

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
  //      the single-GPU golden. Reading a final scalar to host is what q6 does anyway to
  //      compare against the golden; this is idiomatic, not CPU-emulation. ----
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

// ===========================================================================
// Q1 — multi-GPU groupby with partial-aggregate merge. Same query as test_tpch.cpp
// Q1GroupByAggregates. Each worker emits partial SUMs + partial COUNT per group; GPU0 merges.
// ===========================================================================
TEST_F(TpchSf40, Q1MultiGpu) {
  const int G = gpu_count();
  if (G < 1) GTEST_SKIP() << "no visible GPU (found " << G << ")";

  const auto lineitem_path = data_dir() + "/lineitem.parquet";
  const auto golden_path   = golden_dir() + "/duckdb_q1.csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  const int num_rg = parquet_num_row_groups(lineitem_path);
  const auto spans = partition_row_groups(num_rg, G);
  WorkerPool pool(G);

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
            gathered.push_back(gather_to_gpu0(handles[g], s));
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

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
