// TPC-H sf40 in bare cuDF, streamed — the same four queries as test_tpch.cpp and the same
// goldens, but never holding a whole table resident.
//
// Why a second file rather than a flag on the first: test_tpch.cpp loads each table whole
// and then times the operator chain over resident columns, which is the measurement the
// minimal benchmark reports. That shape needs ~10 GiB for q8 and cannot run at all where
// the device has a gigabyte. Here every scan is a `chunked_parquet_reader` bounded by a
// byte budget, and every operator above it is rewritten to accept a batch at a time and
// keep bounded state. The two files therefore answer different questions and both are
// worth having: the first says how fast the operators are, this one says whether the
// query fits.
//
// Verification is unchanged and non-negotiable: the streamed answer is compared to the SAME
// committed DuckDB golden, exactly, through the same tpch_golden.hpp comparators. A
// streaming plan that is fast and wrong is the whole risk of this file, so nothing here
// relaxes a comparison — every decomposition below (partial sums, sum+count for an average,
// per-batch joins) is exact by construction, not by tolerance.
//
// Timing is reported as LOAD and EXECUTE separately even though they interleave: the phases
// alternate per batch, so each is accumulated into its own total with a device sync at the
// boundary. That sync is what makes the split meaningful — cuDF is async, so without it the
// reader's time would absorb the previous batch's kernels.
#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
// cuDF >= 26.02 only, unlike test_tpch.cpp which builds on both legs. 26.02 removed the
// free cudf::left_semi_join in favour of the filtered_join object, and this file wants the
// object form regardless: a streamed probe should build the hash table once and probe it
// per batch, not rebuild it for every batch. The target is EXCLUDE_FROM_ALL, so the 25.02
// CI leg never tries to compile it.
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
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
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rmm_pool.hpp"
#include "tpch_golden.hpp"

using namespace peacock_test;

namespace {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// The two budgets cuDF's chunked reader takes. chunk is the ceiling on one output batch;
// pass is the ceiling on the reader's own decode scratch, which is the larger of the two
// costs and the one that actually decides whether a scan fits. Both are settable because
// the right values depend on what else holds the device, which on a shared machine is not
// a property of this test.
size_t chunk_read_limit() {
  return size_t(std::atoll(env_or("PEACOCK_STREAM_CHUNK_MB", "128").c_str())) << 20;
}
size_t pass_read_limit() {
  return size_t(std::atoll(env_or("PEACOCK_STREAM_PASS_MB", "256").c_str())) << 20;
}

// Accumulated per query, printed as one machine-readable line.
struct Phases {
  double load_ms = 0;
  double exec_ms = 0;
  long batches = 0;
  long rows_in = 0;
};

// Streams one parquet file column-projected, calling per_batch on each chunk.
//
// The sync placement is the contract: read_chunk returns as soon as the copy is queued, so
// the load timer closes on a sync and the execute timer opens after it. per_batch must
// leave nothing running, which the second sync enforces.
template <typename F>
void stream_parquet(const std::string& path, std::vector<std::string> cols, Phases& ph,
                    F&& per_batch) {
  auto opts = cudf::io::parquet_reader_options::builder(
                  cudf::io::source_info{std::vector<std::string>{path}})
                  .columns(std::move(cols))
                  .build();
  cudf::io::chunked_parquet_reader reader(chunk_read_limit(), pass_read_limit(), opts);

  while (reader.has_next()) {
    const auto t0 = Clock::now();
    auto chunk = reader.read_chunk();
    cudaDeviceSynchronize();
    const auto t1 = Clock::now();
    ph.load_ms += ms_between(t0, t1);

    auto view = chunk.tbl->view();
    if (view.num_rows() > 0) {
      ++ph.batches;
      ph.rows_in += view.num_rows();
      per_batch(view);
    }
    cudaDeviceSynchronize();
    ph.exec_ms += ms_between(t1, Clock::now());
  }
}

// cudf::inner_join hands back gather maps rather than a table; this is test_tpch.cpp's
// wrapper, repeated because that file is a test binary and not a library.
cudf::column_view map_view(std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& m) {
  return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(m->size()), m->data(), nullptr, 0);
}

// filtered_join's three-argument form is ambiguous in 26.02 — one overload takes a stream
// fourth, the other a load factor — so the load factor is passed explicitly at cuDF's own
// default to name which one is meant.
constexpr double kLoadFactor = 0.5;

const auto kBool = cudf::data_type{cudf::type_id::BOOL8};
const auto kDec2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
const auto kDec4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};
const auto kF64 = cudf::data_type{cudf::type_id::FLOAT64};

cudf::timestamp_scalar<cudf::timestamp_D> date_scalar(int y, unsigned m, unsigned d) {
  return cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(y, m, d)}}, true);
}

// Holds partial results between batches and folds them together when they grow.
//
// Compaction is what keeps a streamed aggregate bounded: without it the partials are a
// second copy of the input for a group-per-row aggregate. The threshold doubles after each
// fold, so a query whose groups genuinely are numerous stops paying for folds that do not
// reduce — the same discover-the-regime rule the batch-partitioned design uses, because the
// planner has no cardinality estimate to be told the answer by (#19, #146).
class PartialSet {
 public:
  explicit PartialSet(long compact_above) : compact_above_(compact_above) {}

  // fold: (concatenated partials) -> merged partials, same schema in and out. It takes
  // ownership so the identity fold — a build side, which accumulates and never reduces —
  // costs nothing beyond the concatenate.
  template <typename Fold>
  void add(std::unique_ptr<cudf::table> t, Fold&& fold) {
    rows_ += t->num_rows();
    parts_.push_back(std::move(t));
    if (rows_ > compact_above_) {
      auto merged = fold(concat());
      parts_.clear();
      rows_ = merged->num_rows();
      parts_.push_back(std::move(merged));
      compact_above_ *= 2;
      ++compactions_;
    }
  }

  std::unique_ptr<cudf::table> concat() const {
    std::vector<cudf::table_view> views;
    views.reserve(parts_.size());
    for (auto const& p : parts_) views.push_back(p->view());
    return cudf::concatenate(views);
  }

  bool empty() const { return parts_.empty(); }
  int compactions() const { return compactions_; }

 private:
  std::vector<std::unique_ptr<cudf::table>> parts_;
  long rows_ = 0;
  long compact_above_;
  int compactions_ = 0;
};

// A build side accumulates and never reduces, so its fold is the identity — the concat
// PartialSet already did is the whole operation.
std::unique_ptr<cudf::table> keep_all(std::unique_ptr<cudf::table> t) { return t; }

// Streams a table, keeps whatever each batch's transform returns, and hands back the
// concatenation. Returns nullptr when nothing survived, which every caller treats as a
// failed precondition rather than an empty join.
//
// This is the shape of every build side below: a streamed scan cannot make a join's build
// side smaller than it is, it only avoids holding the *unfiltered* table. That is the real
// limit of streaming a join and the reason #136 and #140 exist — the probe side streams,
// the build side must fit.
template <typename F>
std::unique_ptr<cudf::table> stream_collect(const std::string& path,
                                            std::vector<std::string> cols, Phases& ph,
                                            F&& make_batch) {
  PartialSet acc(1 << 22);
  bool any = false;
  stream_parquet(path, std::move(cols), ph, [&](cudf::table_view batch) {
    auto out = make_batch(batch);
    if (out && out->num_rows() > 0) {
      acc.add(std::move(out), keep_all);
      any = true;
    }
  });
  return any ? acc.concat() : nullptr;
}

// Reads a small table whole. Used only for the dimensions that are trivially small at any
// scale factor (nation, region, supplier); everything that grows with the scale factor is
// streamed.
cudf::io::table_with_metadata read_whole(const std::string& path,
                                         std::vector<std::string> cols) {
  auto o = cudf::io::parquet_reader_options::builder(
               cudf::io::source_info{std::vector<std::string>{path}})
               .columns(std::move(cols))
               .build();
  return cudf::io::read_parquet(o);
}

void report(const char* qtag, const Phases& ph, double compare_ms, size_t peak_bytes) {
  std::fprintf(stderr,
               "[stream] %s load_ms=%.1f execute_ms=%.1f total_ms=%.1f batches=%ld "
               "rows_in=%ld peak_mib=%.0f chunk_mib=%zu pass_mib=%zu\n",
               qtag, ph.load_ms, ph.exec_ms, ph.load_ms + ph.exec_ms + compare_ms, ph.batches,
               ph.rows_in, peak_bytes / 1048576.0, chunk_read_limit() >> 20,
               pass_read_limit() >> 20);
}

class TpchSf40Streamed : public TpchSf40 {
 protected:
  size_t peak() const { return peak_bytes(); }
};

}  // namespace

// ===========================================================================
// Q6 — the streaming base case: every operator is per-batch and the state is one number.
//
// sum(l_extendedprice * l_discount) over a filtered lineitem. The partial sums are exact
// DECIMAL128 at scale -4 and are accumulated on the HOST as __int128, so batch count cannot
// change the answer by even a unit in the last place. A batch whose filter keeps nothing
// contributes nothing and is skipped — cudf::reduce over an empty column returns a null
// scalar, which is not zero.
// ===========================================================================
TEST_F(TpchSf40Streamed, Q6Streamed) {
  const auto golden_path = golden_dir() + "/duckdb_q6.csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  Phases ph;
  __int128 total = 0;  // unscaled, scale -4
  long kept_rows = 0;

  stream_parquet(data_dir() + "/lineitem.parquet",
                 {"l_quantity", "l_extendedprice", "l_discount", "l_shipdate"}, ph,
                 [&](cudf::table_view batch) {
    auto quantity = cudf::cast(batch.column(0), kDec2);
    auto extprice = cudf::cast(batch.column(1), kDec2);
    auto discount = cudf::cast(batch.column(2), kDec2);
    auto shipdate = batch.column(3);

    auto lo_date = date_scalar(1994, 1, 1);
    auto hi_date = date_scalar(1995, 1, 1);
    auto disc_lo = cudf::fixed_point_scalar<numeric::decimal128>(5, numeric::scale_type{-2});
    auto disc_hi = cudf::fixed_point_scalar<numeric::decimal128>(7, numeric::scale_type{-2});
    auto qty_hi = cudf::fixed_point_scalar<numeric::decimal128>(2400, numeric::scale_type{-2});

    auto m = cudf::binary_operation(shipdate, lo_date, cudf::binary_operator::GREATER_EQUAL, kBool);
    auto and_with = [&](std::unique_ptr<cudf::column> const& lhs, auto&& rhs) {
      return cudf::binary_operation(lhs->view(), rhs->view(), cudf::binary_operator::LOGICAL_AND,
                                    kBool);
    };
    auto m2 = cudf::binary_operation(shipdate, hi_date, cudf::binary_operator::LESS, kBool);
    auto m3 = cudf::binary_operation(discount->view(), disc_lo,
                                     cudf::binary_operator::GREATER_EQUAL, kBool);
    auto m4 = cudf::binary_operation(discount->view(), disc_hi, cudf::binary_operator::LESS_EQUAL,
                                     kBool);
    auto m5 = cudf::binary_operation(quantity->view(), qty_hi, cudf::binary_operator::LESS, kBool);
    m = and_with(m, m2);
    m = and_with(m, m3);
    m = and_with(m, m4);
    m = and_with(m, m5);

    auto kept = cudf::apply_boolean_mask(
        cudf::table_view{{extprice->view(), discount->view()}}, m->view());
    if (kept->num_rows() == 0) return;
    kept_rows += kept->num_rows();

    auto revenue = cudf::binary_operation(kept->get_column(0).view(), kept->get_column(1).view(),
                                          cudf::binary_operator::MUL, kDec4);
    auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    auto part = cudf::reduce(revenue->view(), *sum_agg, kDec4);
    auto* fp = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal128>*>(part.get());
    ASSERT_NE(fp, nullptr) << "partial sum did not come back as a decimal128 scalar";
    total += static_cast<__int128>(fp->value());
    note_peak();
  });

  EXPECT_GT(kept_rows, 0) << "filter kept no rows across every batch";

  const auto t_cmp = Clock::now();
  Decimal got{total, 4};
  const Decimal want = parse_decimal(read_single_value_golden(golden_path));
  EXPECT_TRUE(decimal_values_equal(got, want))
      << "Q6 mismatch (EXACT decimal comparison)\n"
      << "  cudf   : " << decimal_to_string(got) << "\n"
      << "  duckdb : " << decimal_to_string(want);

  report("q6", ph, ms_between(t_cmp, Clock::now()), peak());
}

// ===========================================================================
// Q1 — a streamed grouped aggregate. Four groups, so the partials are trivially small and
// the interesting part is only that the decomposition is exact.
//
// AVG IS NOT AGGREGATED. Each batch emits SUM and COUNT, and the averages are divided once
// at the end — averaging partial means would weight every batch equally regardless of how
// many rows it held, which is wrong whenever the reader's batches differ in size, i.e.
// always. This is the same rule the multi-GPU path states in build-test.md, and it is the
// single most likely way for a streamed rewrite of this query to be quietly wrong.
// ===========================================================================
TEST_F(TpchSf40Streamed, Q1Streamed) {
  const auto golden_path = golden_dir() + "/duckdb_q1.csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  Phases ph;
  // partial schema: flag, status, sum_qty, sum_price, sum_disc_price, sum_charge, sum_disc,
  // count — the same schema the fold below produces, which is what lets it be re-applied.
  auto fold = [](std::unique_ptr<cudf::table> p) {
    auto partials = p->view();
    cudf::groupby::groupby gb(cudf::table_view{{partials.column(0), partials.column(1)}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    for (int c = 2; c < partials.num_columns(); ++c) {
      cudf::groupby::aggregation_request r;
      r.values = partials.column(c);
      r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    auto [keys, aggs] = gb.aggregate(reqs);
    std::vector<std::unique_ptr<cudf::column>> cols;
    auto kv = keys->release();
    cols.push_back(std::move(kv[0]));
    cols.push_back(std::move(kv[1]));
    for (auto& a : aggs) cols.push_back(std::move(a.results[0]));
    return std::make_unique<cudf::table>(std::move(cols));
  };

  PartialSet partials(1 << 16);
  stream_parquet(data_dir() + "/lineitem.parquet",
                 {"l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice", "l_discount",
                  "l_tax", "l_shipdate"},
                 ph, [&](cudf::table_view batch) {
    auto quantity = cudf::cast(batch.column(2), kDec2);
    auto extprice = cudf::cast(batch.column(3), kDec2);
    auto discount = cudf::cast(batch.column(4), kDec2);
    auto tax = cudf::cast(batch.column(5), kDec2);

    auto cutoff = date_scalar(1998, 9, 2);
    auto mask = cudf::binary_operation(batch.column(6), cutoff, cudf::binary_operator::LESS_EQUAL,
                                       kBool);
    auto kept = cudf::apply_boolean_mask(
        cudf::table_view{{batch.column(0), batch.column(1), quantity->view(), extprice->view(),
                          discount->view(), tax->view()}},
        mask->view());
    if (kept->num_rows() == 0) return;

    auto k_qty = kept->get_column(2).view();
    auto k_price = kept->get_column(3).view();
    auto k_disc = kept->get_column(4).view();
    auto k_tax = kept->get_column(5).view();

    auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
    auto one_minus_disc = cudf::binary_operation(one_s2, k_disc, cudf::binary_operator::SUB, kDec2);
    auto one_plus_tax = cudf::binary_operation(one_s2, k_tax, cudf::binary_operator::ADD, kDec2);
    auto disc_price = cudf::binary_operation(k_price, one_minus_disc->view(),
                                             cudf::binary_operator::MUL, kDec4);
    auto charge = cudf::binary_operation(disc_price->view(), one_plus_tax->view(),
                                         cudf::binary_operator::MUL,
                                         cudf::data_type{cudf::type_id::DECIMAL128, -6});

    cudf::groupby::groupby gb(
        cudf::table_view{{kept->get_column(0).view(), kept->get_column(1).view()}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    auto add = [&](cudf::column_view v, std::unique_ptr<cudf::groupby_aggregation> a) {
      cudf::groupby::aggregation_request r;
      r.values = v;
      r.aggregations.push_back(std::move(a));
      reqs.push_back(std::move(r));
    };
    add(k_qty, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    add(k_price, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    add(disc_price->view(), cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    add(charge->view(), cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    add(k_disc, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    add(k_qty, cudf::make_count_aggregation<cudf::groupby_aggregation>());

    auto [keys, aggs] = gb.aggregate(reqs);
    std::vector<std::unique_ptr<cudf::column>> cols;
    auto kv = keys->release();
    cols.push_back(std::move(kv[0]));
    cols.push_back(std::move(kv[1]));
    for (auto& a : aggs) cols.push_back(std::move(a.results[0]));
    partials.add(std::make_unique<cudf::table>(std::move(cols)), fold);
    note_peak();
  });

  ASSERT_FALSE(partials.empty()) << "no batch produced a partial aggregate";

  const auto t_final = Clock::now();
  auto merged = fold(partials.concat());
  auto mv = merged->view();

  // avg = sum / count, once, in float64 — the only float on this path, and the only place
  // the golden allows a tolerance.
  auto count_f = cudf::cast(mv.column(7), kF64);
  auto avg_of = [&](cudf::column_view sum_col) {
    auto sum_f = cudf::cast(sum_col, kF64);
    return cudf::binary_operation(sum_f->view(), count_f->view(), cudf::binary_operator::DIV, kF64);
  };
  auto avg_qty = avg_of(mv.column(2));
  auto avg_price = avg_of(mv.column(3));
  auto avg_disc = avg_of(mv.column(6));

  auto order = cudf::sorted_order(cudf::table_view{{mv.column(0), mv.column(1)}},
                                  {cudf::order::ASCENDING, cudf::order::ASCENDING},
                                  {cudf::null_order::AFTER, cudf::null_order::AFTER});
  auto sorted = cudf::gather(
      cudf::table_view{{mv.column(0), mv.column(1), mv.column(2), mv.column(3), mv.column(4),
                        mv.column(5), avg_qty->view(), avg_price->view(), avg_disc->view(),
                        mv.column(7)}},
      order->view());
  ph.exec_ms += ms_between(t_final, Clock::now());
  note_peak();

  constexpr double kAvgRelTol = 1e-9;
  const std::vector<ColSpec> kQ1Spec = {
      {"l_returnflag", Cmp::ExactString},   {"l_linestatus", Cmp::ExactString},
      {"sum_qty", Cmp::ExactDecimal},       {"sum_base_price", Cmp::ExactDecimal},
      {"sum_disc_price", Cmp::ExactDecimal}, {"sum_charge", Cmp::ExactDecimal},
      {"avg_qty", Cmp::TolerantDouble, kAvgRelTol},
      {"avg_price", Cmp::TolerantDouble, kAvgRelTol},
      {"avg_disc", Cmp::TolerantDouble, kAvgRelTol},
      {"count_order", Cmp::ExactInt},
  };
  const auto t_cmp = Clock::now();
  const double worst_rel =
      compare_table_to_golden(sorted->view(), read_csv_golden(golden_path), kQ1Spec, "q1");
  std::fprintf(stderr, "[q1] worst AVG relative error %.3e (tolerance %.1e), %d compactions\n",
               worst_rel, kAvgRelTol, partials.compactions());
  report("q1", ph, ms_between(t_cmp, Clock::now()), peak());
}

// ===========================================================================
// Q3 — a streamed join, and the first query where streaming changes the plan rather than
// just the scan.
//
// The two small sides are built first, each by a streamed filtered scan: BUILDING customers
// (a key list), then orders before 1995-03-15 that belong to one of them. Only then is
// lineitem streamed, each batch joined against the resident order table and aggregated into
// partials keyed by orderkey.
//
// The join direction is forced, not chosen: the streamed side must be the PROBE side,
// because a build side has to be complete before the first probe row can be answered. That
// is why orders is fully materialized (~5.8M rows after both filters) while lineitem — 6x
// larger — never is.
//
// The group key is orderkey, so the partials are nearly as numerous as the output and
// compaction earns its place here in a way it does not in q1.
// ===========================================================================
TEST_F(TpchSf40Streamed, Q3Streamed) {
  const auto golden_path = golden_dir() + "/duckdb_q3.csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  Phases ph;
  auto d1995 = date_scalar(1995, 3, 15);

  auto cust = stream_collect(data_dir() + "/customer.parquet",
                             {"c_custkey", "c_mktsegment"}, ph,
                             [&](cudf::table_view b) {
    auto seg = cudf::string_scalar(std::string("BUILDING"));
    auto mask = cudf::binary_operation(b.column(1), seg, cudf::binary_operator::EQUAL, kBool);
    return cudf::apply_boolean_mask(cudf::table_view{{b.column(0)}}, mask->view());
  });
  ASSERT_NE(cust, nullptr) << "no BUILDING customers";

  // c_custkey is unique, so a semi-join is exactly right and cheaper than a full inner
  // join: no row can be duplicated and no customer column is needed downstream. The object
  // is built once here and probed by every orders batch.
  cudf::filtered_join cust_fj(cudf::table_view{{cust->get_column(0).view()}},
                              cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT,
                              kLoadFactor);
  auto orders = stream_collect(
      data_dir() + "/orders.parquet",
      {"o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"}, ph,
      [&](cudf::table_view b) -> std::unique_ptr<cudf::table> {
        auto mask = cudf::binary_operation(b.column(2), d1995, cudf::binary_operator::LESS, kBool);
        auto f = cudf::apply_boolean_mask(b, mask->view());
        if (f->num_rows() == 0) return nullptr;
        auto sm = cust_fj.semi_join(cudf::table_view{{f->get_column(1).view()}});
        if (sm->size() == 0) return nullptr;
        return cudf::gather(cudf::table_view{{f->get_column(0).view(), f->get_column(2).view(),
                                              f->get_column(3).view()}},
                            map_view(sm));
      });
  ASSERT_NE(orders, nullptr) << "no orders survived the filter and the customer join";
  std::fprintf(stderr, "[q3] build side: %ld customers, %ld orders\n",
               static_cast<long>(cust->num_rows()), static_cast<long>(orders->num_rows()));

  // partial schema: o_orderkey, o_orderdate, o_shippriority, sum(revenue)
  auto fold = [](std::unique_ptr<cudf::table> p) {
    auto v = p->view();
    cudf::groupby::groupby gb(cudf::table_view{{v.column(0), v.column(1), v.column(2)}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    cudf::groupby::aggregation_request r;
    r.values = v.column(3);
    r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    reqs.push_back(std::move(r));
    auto [keys, aggs] = gb.aggregate(reqs);
    auto cols = keys->release();
    cols.push_back(std::move(aggs[0].results[0]));
    return std::make_unique<cudf::table>(std::move(cols));
  };

  PartialSet partials(1 << 21);
  auto ord_v = orders->view();
  // hash_join takes the BUILD table and returns [probe_indices, build_indices] per probe.
  cudf::hash_join ord_hj(cudf::table_view{{ord_v.column(0)}}, cudf::null_equality::EQUAL);
  stream_parquet(data_dir() + "/lineitem.parquet",
                 {"l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"}, ph,
                 [&](cudf::table_view b) {
    auto mask = cudf::binary_operation(b.column(3), d1995, cudf::binary_operator::GREATER, kBool);
    auto f = cudf::apply_boolean_mask(
        cudf::table_view{{b.column(0), b.column(1), b.column(2)}}, mask->view());
    if (f->num_rows() == 0) return;

    auto [l_map, o_map] = ord_hj.inner_join(cudf::table_view{{f->get_column(0).view()}});
    if (l_map->size() == 0) return;
    auto l_side = cudf::gather(
        cudf::table_view{{f->get_column(1).view(), f->get_column(2).view()}}, map_view(l_map));
    auto o_side = cudf::gather(
        cudf::table_view{{ord_v.column(0), ord_v.column(1), ord_v.column(2)}}, map_view(o_map));

    auto price = cudf::cast(l_side->get_column(0).view(), kDec2);
    auto disc = cudf::cast(l_side->get_column(1).view(), kDec2);
    auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
    auto one_minus_disc = cudf::binary_operation(one_s2, disc->view(), cudf::binary_operator::SUB,
                                                 kDec2);
    auto revenue = cudf::binary_operation(price->view(), one_minus_disc->view(),
                                          cudf::binary_operator::MUL, kDec4);

    std::vector<std::unique_ptr<cudf::column>> cols = o_side->release();
    cols.push_back(std::move(revenue));
    partials.add(fold(std::make_unique<cudf::table>(std::move(cols))), fold);
    note_peak();
  });
  ASSERT_FALSE(partials.empty()) << "the streamed join produced no rows";

  const auto t_final = Clock::now();
  auto merged = fold(partials.concat());
  auto mv = merged->view();
  auto order = cudf::sorted_order(
      cudf::table_view{{mv.column(3), mv.column(1), mv.column(0)}},
      {cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING},
      {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER});
  auto sorted = cudf::gather(
      cudf::table_view{{mv.column(0), mv.column(3), mv.column(1), mv.column(2)}}, order->view());
  auto top = cudf::slice(sorted->view(), {0, 10})[0];
  ph.exec_ms += ms_between(t_final, Clock::now());
  note_peak();

  const std::vector<ColSpec> kQ3Spec = {
      {"l_orderkey", Cmp::ExactInt},
      {"revenue", Cmp::ExactDecimal},
      {"o_orderdate", Cmp::ExactDate},
      {"o_shippriority", Cmp::ExactInt},
  };
  const auto t_cmp = Clock::now();
  const auto golden = read_csv_golden(golden_path);
  ASSERT_EQ(static_cast<int>(golden.size()), 10) << "golden should hold 10 rows";
  compare_table_to_golden(top, golden, kQ3Spec, "q3");
  std::fprintf(stderr, "[q3] %d compactions\n", partials.compactions());
  report("q3", ph, ms_between(t_cmp, Clock::now()), peak());
}

// ===========================================================================
// Q8 — seven tables, streamed. Six build sides and one streamed probe.
//
// The build order is the whole trick, and it is the same bushy plan test_tpch.cpp uses,
// just with each side built by a streamed scan: region -> nation n1 -> customer -> orders
// gives the American orders in 1995-96, and part is filtered to the one type. Only then is
// lineitem streamed, and each batch is cut by the partkey join FIRST — that is what takes
// 240M rows down to ~1.6M before any wider intermediate exists. Joining orders first would
// materialize ~68M rows per pass and defeat the point.
//
// nation and region are read whole: 25 and 5 rows, constant at every scale factor.
// ===========================================================================
TEST_F(TpchSf40Streamed, Q8Streamed) {
  const auto golden_path = golden_dir() + "/duckdb_q8.csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  Phases ph;
  const auto t_small = Clock::now();
  auto nation_in = read_whole(data_dir() + "/nation.parquet",
                              {"n_nationkey", "n_name", "n_regionkey"});
  auto region_in = read_whole(data_dir() + "/region.parquet", {"r_regionkey", "r_name"});
  auto supp_in = read_whole(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  cudaDeviceSynchronize();
  ph.load_ms += ms_between(t_small, Clock::now());
  auto nation_v = nation_in.tbl->view();

  const auto t_n1 = Clock::now();
  auto america = cudf::string_scalar(std::string("AMERICA"));
  auto rmask = cudf::binary_operation(region_in.tbl->view().column(1), america,
                                      cudf::binary_operator::EQUAL, kBool);
  auto region_f = cudf::apply_boolean_mask(
      cudf::table_view{{region_in.tbl->view().column(0)}}, rmask->view());
  cudf::filtered_join region_fj(cudf::table_view{{region_f->get_column(0).view()}},
                                cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT,
                              kLoadFactor);
  auto n1_sm = region_fj.semi_join(cudf::table_view{{nation_v.column(2)}});
  auto n1 = cudf::gather(cudf::table_view{{nation_v.column(0)}}, map_view(n1_sm));
  ph.exec_ms += ms_between(t_n1, Clock::now());

  cudf::filtered_join n1_fj(cudf::table_view{{n1->get_column(0).view()}},
                            cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT,
                              kLoadFactor);
  auto cust = stream_collect(data_dir() + "/customer.parquet", {"c_custkey", "c_nationkey"}, ph,
                             [&](cudf::table_view b) -> std::unique_ptr<cudf::table> {
    auto sm = n1_fj.semi_join(cudf::table_view{{b.column(1)}});
    if (sm->size() == 0) return nullptr;
    return cudf::gather(cudf::table_view{{b.column(0)}}, map_view(sm));
  });
  ASSERT_NE(cust, nullptr) << "no customers in American nations";

  auto lo = date_scalar(1995, 1, 1);
  auto hi = date_scalar(1996, 12, 31);
  cudf::filtered_join cust_fj(cudf::table_view{{cust->get_column(0).view()}},
                              cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT,
                              kLoadFactor);
  auto orders = stream_collect(data_dir() + "/orders.parquet",
                               {"o_orderkey", "o_custkey", "o_orderdate"}, ph,
                               [&](cudf::table_view b) -> std::unique_ptr<cudf::table> {
    auto m1 = cudf::binary_operation(b.column(2), lo, cudf::binary_operator::GREATER_EQUAL, kBool);
    auto m2 = cudf::binary_operation(b.column(2), hi, cudf::binary_operator::LESS_EQUAL, kBool);
    auto m = cudf::binary_operation(m1->view(), m2->view(), cudf::binary_operator::LOGICAL_AND,
                                    kBool);
    auto f = cudf::apply_boolean_mask(b, m->view());
    if (f->num_rows() == 0) return nullptr;
    auto sm = cust_fj.semi_join(cudf::table_view{{f->get_column(1).view()}});
    if (sm->size() == 0) return nullptr;
    return cudf::gather(
        cudf::table_view{{f->get_column(0).view(), f->get_column(2).view()}}, map_view(sm));
  });
  ASSERT_NE(orders, nullptr) << "no American orders in 1995-96";

  auto part = stream_collect(data_dir() + "/part.parquet", {"p_partkey", "p_type"}, ph,
                             [&](cudf::table_view b) {
    auto ptype = cudf::string_scalar(std::string("ECONOMY ANODIZED STEEL"));
    auto pmask = cudf::binary_operation(b.column(1), ptype, cudf::binary_operator::EQUAL, kBool);
    return cudf::apply_boolean_mask(cudf::table_view{{b.column(0)}}, pmask->view());
  });
  ASSERT_NE(part, nullptr) << "no parts of the requested type";
  std::fprintf(stderr, "[q8] build sides: %ld nations, %ld customers, %ld orders, %ld parts\n",
               static_cast<long>(n1->num_rows()), static_cast<long>(cust->num_rows()),
               static_cast<long>(orders->num_rows()), static_cast<long>(part->num_rows()));

  // partial schema: o_year, sum(brazil_volume), sum(volume)
  auto fold = [](std::unique_ptr<cudf::table> p) {
    auto v = p->view();
    cudf::groupby::groupby gb(cudf::table_view{{v.column(0)}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    for (int c = 1; c < v.num_columns(); ++c) {
      cudf::groupby::aggregation_request r;
      r.values = v.column(c);
      r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    auto [keys, aggs] = gb.aggregate(reqs);
    auto cols = keys->release();
    for (auto& a : aggs) cols.push_back(std::move(a.results[0]));
    return std::make_unique<cudf::table>(std::move(cols));
  };

  PartialSet partials(1 << 12);
  auto ord_v = orders->view();
  auto supp_v = supp_in.tbl->view();
  // Four build sides, four objects, all built once before the probe stream starts.
  cudf::filtered_join part_fj(cudf::table_view{{part->get_column(0).view()}},
                              cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT,
                              kLoadFactor);
  cudf::hash_join ord_hj(cudf::table_view{{ord_v.column(0)}}, cudf::null_equality::EQUAL);
  cudf::hash_join supp_hj(cudf::table_view{{supp_v.column(0)}}, cudf::null_equality::EQUAL);
  cudf::hash_join nation_hj(cudf::table_view{{nation_v.column(0)}}, cudf::null_equality::EQUAL);
  stream_parquet(data_dir() + "/lineitem.parquet",
                 {"l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount"}, ph,
                 [&](cudf::table_view b) {
    // partkey first — the only filter that makes this query small
    auto p_sm = part_fj.semi_join(cudf::table_view{{b.column(1)}});
    if (p_sm->size() == 0) return;
    auto lp = cudf::gather(
        cudf::table_view{{b.column(0), b.column(2), b.column(3), b.column(4)}}, map_view(p_sm));

    auto [lp_map, oa_map] = ord_hj.inner_join(cudf::table_view{{lp->get_column(0).view()}});
    if (lp_map->size() == 0) return;
    auto lp_side = cudf::gather(cudf::table_view{{lp->get_column(1).view(),
                                                  lp->get_column(2).view(),
                                                  lp->get_column(3).view()}},
                                map_view(lp_map));
    auto date_side = cudf::gather(cudf::table_view{{ord_v.column(1)}}, map_view(oa_map));

    auto [s_map, sp_map] = supp_hj.inner_join(cudf::table_view{{lp_side->get_column(0).view()}});
    if (s_map->size() == 0) return;
    auto d_price = cudf::gather(
        cudf::table_view{{lp_side->get_column(1).view(), lp_side->get_column(2).view()}},
        map_view(s_map));
    auto d_date = cudf::gather(cudf::table_view{{date_side->get_column(0).view()}},
                               map_view(s_map));
    auto d_snation = cudf::gather(cudf::table_view{{supp_v.column(1)}}, map_view(sp_map));

    auto [sn_map, n2_map] = nation_hj.inner_join(
        cudf::table_view{{d_snation->get_column(0).view()}});
    auto e_price = cudf::gather(
        cudf::table_view{{d_price->get_column(0).view(), d_price->get_column(1).view()}},
        map_view(sn_map));
    auto e_date = cudf::gather(cudf::table_view{{d_date->get_column(0).view()}}, map_view(sn_map));
    auto e_nname = cudf::gather(cudf::table_view{{nation_v.column(1)}}, map_view(n2_map));

    auto price = cudf::cast(e_price->get_column(0).view(), kDec2);
    auto disc = cudf::cast(e_price->get_column(1).view(), kDec2);
    auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
    auto one_minus_disc = cudf::binary_operation(one_s2, disc->view(), cudf::binary_operator::SUB,
                                                 kDec2);
    auto volume = cudf::binary_operation(price->view(), one_minus_disc->view(),
                                         cudf::binary_operator::MUL, kDec4);
    auto o_year = cudf::datetime::extract_datetime_component(
        e_date->get_column(0).view(), cudf::datetime::datetime_component::YEAR);
    auto brazil = cudf::string_scalar(std::string("BRAZIL"));
    auto is_brazil = cudf::binary_operation(e_nname->get_column(0).view(), brazil,
                                            cudf::binary_operator::EQUAL, kBool);
    auto zero_s4 = cudf::fixed_point_scalar<numeric::decimal128>(0, numeric::scale_type{-4});
    auto brazil_volume = cudf::copy_if_else(volume->view(), zero_s4, is_brazil->view());

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(o_year));
    cols.push_back(std::move(brazil_volume));
    cols.push_back(std::move(volume));
    partials.add(fold(std::make_unique<cudf::table>(std::move(cols))), fold);
    note_peak();
  });
  ASSERT_FALSE(partials.empty()) << "the seven-table streamed join produced no rows";

  const auto t_final = Clock::now();
  auto merged = fold(partials.concat());
  auto mv = merged->view();
  auto brazil_f = cudf::cast(mv.column(1), kF64);
  auto total_f = cudf::cast(mv.column(2), kF64);
  auto mkt_share = cudf::binary_operation(brazil_f->view(), total_f->view(),
                                          cudf::binary_operator::DIV, kF64);
  auto order = cudf::sorted_order(cudf::table_view{{mv.column(0)}}, {cudf::order::ASCENDING},
                                  {cudf::null_order::AFTER});
  auto sorted = cudf::gather(
      cudf::table_view{{mv.column(0), mv.column(1), mv.column(2), mkt_share->view()}},
      order->view());
  ph.exec_ms += ms_between(t_final, Clock::now());
  note_peak();

  constexpr double kShareRelTol = 1e-9;
  const std::vector<ColSpec> kQ8Spec = {
      {"o_year", Cmp::ExactInt},
      {"brazil_volume", Cmp::ExactDecimal},
      {"total_volume", Cmp::ExactDecimal},
      {"mkt_share", Cmp::TolerantDouble, kShareRelTol},
  };
  const auto t_cmp = Clock::now();
  const double worst_rel =
      compare_table_to_golden(sorted->view(), read_csv_golden(golden_path), kQ8Spec, "q8");
  std::fprintf(stderr, "[q8] worst mkt_share relative error %.3e (tolerance %.1e)\n", worst_rel,
               kShareRelTol);
  report("q8", ph, ms_between(t_cmp, Clock::now()), peak());
}

// Same entry point as the other gtest binaries here (the conda cudf ships no gtest_main).
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  peacock_test::install_rmm_pool();
  return RUN_ALL_TESTS();
}
