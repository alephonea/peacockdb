// TPC-H+V (vector-augmented) queries in bare cuDF + cuVS brute force, checked against
// DuckDB. Same rules as test_tpch.cpp: two separate phases; every predicate — including
// the distance predicate — is a phase-2 operator, never pushed into the reader.
//
// cuVS linking gotcha: rmm is header-only here, so libcudf.so DEFINES rmm::logger::~logger
// and libcuvs.so only references it. Loading cuVS alone fails with "undefined symbol:
// _ZN3rmm6loggerD1Ev"; libcudf must be loaded/linked first. `ldd libcuvs.so` reports no
// missing deps, so only an actual dlopen/link exposes this.
#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/datetime.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#if __has_include(<cudf/join/join.hpp>)
#  include <cudf/join/join.hpp>   // cudf >= 26.02
#else
#  include <cudf/join.hpp>        // cudf 25.02
#endif

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

#include "peacock/rmm_pool.hpp"
#include "tpch_golden.hpp"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

using namespace peacock_test;

namespace {

// One (q, D) probe from the committed query_params.jsonl — never hardcoded, so the golden
// and the GPU side cannot drift apart.
struct VecProbe {
  std::string id;
  int k = 0;
  double D = 0.0;
  std::vector<float> q;
};

// Minimal field extraction; the file is one flat JSON object per line.
std::vector<VecProbe> load_probes(const std::string& path, std::vector<std::string> const& want) {
  std::vector<VecProbe> out;
  std::ifstream f(path);
  std::string line;
  // json.dump puts a space after each colon — skip whitespace rather than assuming a
  // fixed offset (that assumption silently found zero probes once).
  auto field = [](const std::string& s, const std::string& key) -> std::string {
    auto p = s.find("\"" + key + "\":");
    if (p == std::string::npos) return {};
    p += key.size() + 3;
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
    auto e = s.find_first_of(",}", p);
    return s.substr(p, e - p);
  };
  while (std::getline(f, line)) {
    std::string id = field(line, "id");
    if (!id.empty() && id.front() == '"') id = id.substr(1, id.size() - 2);
    if (std::find(want.begin(), want.end(), id) == want.end()) continue;
    VecProbe p;
    p.id = id;
    p.k = std::atoi(field(line, "k").c_str());
    p.D = std::strtod(field(line, "D").c_str(), nullptr);
    auto qs = line.find("\"q\":");
    auto lb = line.find('[', qs), rb = line.find(']', lb);
    std::string nums = line.substr(lb + 1, rb - lb - 1), tok;
    for (char c : nums) {
      if (c == ',') { p.q.push_back(std::strtof(tok.c_str(), nullptr)); tok.clear(); }
      else tok += c;
    }
    if (!tok.empty()) p.q.push_back(std::strtof(tok.c_str(), nullptr));
    out.push_back(std::move(p));
  }
  return out;
}

std::string vec_params_path() {
  return env_or("PEACOCK_TPCH_VEC_PARAMS",
                "/home/info/peacock-datasets/testdata/tpch-vec-queries/query_params.jsonl");
}

// EMBEDDING LOAD — do NOT "simplify" the loop back into a single read_parquet.
// A cuDF LIST column's child length is int32, capping it at 2^31-1 elements. At sf40:
//   partsupp.ps_image_embedding  32M x 96  = 3.07e9 -> OVER the ceiling, must chunk
//   part.p_text_embedding         8M x 100 = 0.80e9 -> one read
// A single read of the over-ceiling column fails with "std::bad_alloc: out_of_memory:
// cudaErrorMemoryAllocation" — misleading: it is an overflowed size computation, not a
// real allocation shortfall (device had 11x the needed memory; bisect confirms the
// boundary). The batching is NOT a memory optimisation and NOT predicate pushdown: every
// row group is read, every row is resident before any operator runs. cuVS/raft use int64
// extents, so the reassembled flat buffer is representable even though a cuDF list column
// of the same data is not.
// One function for both regimes: batch size derives from ceiling/width, and each caller
// asserts which regime it expects, so a column silently crossing the boundary fails
// loudly. Always copies into an owning buffer — one lifetime rule instead of two.
struct EmbeddingMatrix {
  rmm::device_uvector<float> buf;
  int64_t rows;
  int dim;
  int batches;
};

EmbeddingMatrix load_embedding_column(const std::string& path, const std::string& col,
                                      int dim, rmm::cuda_stream_view stream) {
  auto meta = cudf::io::read_parquet_metadata(
      cudf::io::source_info{std::vector<std::string>{path}});
  const int64_t n_rows = meta.num_rows();
  const int n_rg = meta.num_rowgroups();

  EmbeddingMatrix m{rmm::device_uvector<float>(static_cast<size_t>(n_rows) * dim, stream),
                    n_rows, dim, 0};

  // Half the ceiling: the loop can only bound a batch by the file-average row-group size
  // (exact per-group counts are not exposed), so leave headroom for uneven groups.
  constexpr int64_t kListChildCeiling = 2147483647;
  const int64_t max_rows_per_batch = (kListChildCeiling / dim) / 2;
  const int64_t avg_rg_rows = (n_rows + n_rg - 1) / n_rg;

  int64_t copied = 0;
  for (int rg = 0; rg < n_rg;) {
    std::vector<cudf::size_type> group;
    int64_t batch_rows = 0;
    while (rg < n_rg && batch_rows < max_rows_per_batch) {
      group.push_back(rg);
      batch_rows += avg_rg_rows;
      ++rg;
    }
    auto o = cudf::io::parquet_reader_options::builder(
                 cudf::io::source_info{std::vector<std::string>{path}})
                 .columns({col})
                 .build();
    o.set_row_groups({group});
    auto chunk = cudf::io::read_parquet(o);
    auto child = cudf::lists_column_view(chunk.tbl->view().column(0)).child();
    EXPECT_EQ(child.type().id(), cudf::type_id::FLOAT32)
        << col << ": embedding child must be float32";
    EXPECT_EQ(static_cast<int64_t>(child.size()),
              static_cast<int64_t>(chunk.tbl->num_rows()) * dim)
        << col << ": batch is not a fixed " << dim << "-wide list";
    CUDF_CUDA_TRY(cudaMemcpyAsync(m.buf.data() + copied * dim, child.data<float>(),
                                  static_cast<size_t>(child.size()) * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream.value()));
    cudaStreamSynchronize(stream.value());
    copied += chunk.tbl->num_rows();
    ++m.batches;
  }
  // A dropped or double-counted batch would silently change every distance result.
  EXPECT_EQ(copied, n_rows) << col << ": chunked load reassembled " << copied
                            << " rows but the file has " << n_rows
                            << " — a batch was dropped or double-counted";
  std::fprintf(stderr, "[vec] %s: %ld rows x %d loaded in %d batch(es)\n", col.c_str(),
               static_cast<long>(copied), dim, m.batches);
  return m;
}

// RANGE-VS-TOP-K: cuVS brute_force has only a top-K search, no range/radius API, but the
// queries need every row with distance < D. So: search top-K with K well above the rows
// that can fall under D, then filter by D. Exact ONLY if the top-K did not saturate — the
// saturation guard (K'th neighbour still inside D => truncated answer) is load-bearing.
//
// Distance space: cuVS defaults to L2Expanded (SQUARED L2); DuckDB's array_distance is the
// root. The index is built with L2SqrtExpanded so D is used as-is on both sides.
//
// The count corroboration against DuckDB (expect_hit_count) pins the row SET the distance
// predicate selects before any join can mask a discrepancy; it runs only in the verify
// path, so vector_range_hits itself reads no golden and both verify and benchmark time the
// same code.
template <typename Index>
std::vector<int32_t> vector_range_hits(raft::resources& handle, Index const& index,
                                       VecProbe const& probe, int dim, int64_t K,
                                       const char* tag) {
  auto d_q = raft::make_device_matrix<float, int64_t>(handle, 1, dim);
  raft::copy(d_q.data_handle(), probe.q.data(), dim, raft::resource::get_cuda_stream(handle));
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(handle, 1, K);
  auto distances = raft::make_device_matrix<float, int64_t>(handle, 1, K);
  cuvs::neighbors::brute_force::search(handle, cuvs::neighbors::brute_force::search_params{},
                                       index, raft::make_const_mdspan(d_q.view()),
                                       neighbors.view(), distances.view());
  raft::resource::sync_stream(handle);

  // range-filter on the host: K is O(1e5), trivial next to the search
  std::vector<int64_t> h_nb(K);
  std::vector<float> h_di(K);
  raft::copy(h_nb.data(), neighbors.data_handle(), K, raft::resource::get_cuda_stream(handle));
  raft::copy(h_di.data(), distances.data_handle(), K, raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);

  std::vector<int32_t> hits;
  // saturation guard — see file header note
  EXPECT_GE(static_cast<double>(h_di[K - 1]), probe.D)
      << tag << "/" << probe.id << ": top-K SATURATED (K=" << K << ", furthest distance "
      << h_di[K - 1] << " < D=" << probe.D
      << ") — the range answer would be silently truncated. Raise K.";

  for (int64_t i = 0; i < K; ++i) {
    if (static_cast<double>(h_di[i]) < probe.D) hits.push_back(static_cast<int32_t>(h_nb[i]));
  }
  std::fprintf(stderr, "[%s] %s: D=%.17g -> %zu rows under D (furthest returned %.9g)\n", tag,
               probe.id.c_str(), probe.D, hits.size(), h_di[K - 1]);
  return hits;
}

// Count corroboration — verify path only (reads a golden, so kept out of the timed region).
inline void expect_hit_count(size_t got, const std::string& count_golden, const char* tag,
                             const std::string& id) {
  EXPECT_TRUE(file_exists(count_golden)) << "missing " << count_golden;
  if (file_exists(count_golden)) {
    const int64_t want =
        std::strtoll(read_single_value_golden(count_golden).c_str(), nullptr, 10);
    EXPECT_EQ(static_cast<int64_t>(got), want)
        << tag << "/" << id << ": cuVS found " << got << " rows under D but DuckDB found "
        << want << ". A one-row difference here means cuVS and DuckDB landed on opposite "
        << "sides of the distance boundary — a real numerical divergence, NOT something to "
        << "absorb with a tolerance.";
  }
}

// host hit list -> an owning device int32 column, ready to be joined against a row index
struct DeviceHits {
  rmm::device_uvector<int32_t> buf;
  cudf::column_view view() const {
    return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                             static_cast<cudf::size_type>(buf.size()), buf.data(), nullptr, 0);
  }
};

DeviceHits to_device_hits(std::vector<int32_t> const& hits) {
  DeviceHits d{rmm::device_uvector<int32_t>(hits.size(), cudf::get_default_stream())};
  CUDF_CUDA_TRY(cudaMemcpyAsync(d.buf.data(), hits.data(), hits.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice, cudf::get_default_stream().value()));
  cudf::get_default_stream().synchronize();
  return d;
}

// read a set of columns from one parquet file — no predicate, no row selection
cudf::io::table_with_metadata read_cols(const std::string& path, std::vector<std::string> cols) {
  auto o = cudf::io::parquet_reader_options::builder(
               cudf::io::source_info{std::vector<std::string>{path}})
               .columns(std::move(cols))
               .build();
  return cudf::io::read_parquet(o);
}

// cudf::inner_join returns gather MAPS, not tables; this wraps one as a column_view so it
// can be handed to cudf::gather.
cudf::column_view map_view(std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& m) {
  return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(m->size()), m->data(), nullptr, 0);
}

cudf::timestamp_scalar<cudf::timestamp_D> date_scalar(int y, unsigned mo, unsigned d) {
  return cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(y, mo, d)}}, true);
}

// K for top-K-then-filter. Overridable (PEACOCK_TPCHV_K=64) only so the saturation guard
// can be exercised. Default sized from measured sf40 hit counts with generous headroom.
int64_t search_k() {
  return std::strtoll(env_or("PEACOCK_TPCHV_K", "131072").c_str(), nullptr, 10);
}

// cuVS brute-force index build — the SETUP phase, timed separately from load and execute:
// a one-time O(n) preprocess amortised across probes. L2SqrtExpanded == DuckDB's
// array_distance (see distance-space note above).
inline auto build_bf_index(raft::resources& handle, const float* data, int64_t rows, int dim) {
  auto dataset = raft::make_device_matrix_view<const float, int64_t>(data, rows, dim);
  cuvs::neighbors::brute_force::index_params ip;
  ip.metric = cuvs::distance::DistanceType::L2SqrtExpanded;
  return cuvs::neighbors::brute_force::build(handle, ip, dataset);
}

// The vector operator for the execute closures: search part under D, gather surviving
// p_partkey values. Verify path and benchmark both go through here.
template <typename Index>
std::unique_ptr<cudf::table> parts_under_d(raft::resources& handle, Index const& index,
                                           cudf::column_view partkey, VecProbe const& probe,
                                           int dim, int64_t K, const char* tag) {
  auto hits = vector_range_hits(handle, index, probe, dim, K, tag);
  auto d_hits = to_device_hits(hits);
  return cudf::gather(cudf::table_view{{partkey}}, d_hits.view());
}

}  // namespace

// TPC-H+V q11 — national market value, restricted to a vector neighbourhood.
//
//   SELECT ps_partkey, sum(ps_supplycost*ps_availqty) AS value
//   FROM partsupp, supplier, nation
//   WHERE ps_suppkey=s_suppkey AND s_nationkey=n_nationkey AND n_name='GERMANY'
//     AND ps_image_embedding <-> ${q} < ${D}
//   GROUP BY ps_partkey
//   HAVING sum(ps_supplycost*ps_availqty) > (same sum over ALL German rows) * 0.000002
//   ORDER BY value DESC
//
// The HAVING subquery deliberately does NOT carry the vector predicate: its threshold is
// over every German partsupp row, so one join serves both the unfiltered threshold and the
// vector-filtered groups.
//
// All comparisons EXACT (value is DECIMAL128 scale -4; no float enters the aggregate).
// Distances are float32 but only ever COMPARED to D; the selected row set is exact as long
// as the boundary margin (measured >= 4.5e-6) exceeds float32 disagreement (~5e-7).
TEST_F(TpchSf40, Q11VectorBruteForce) {
  const auto params_path = vec_params_path();
  if (!file_exists(params_path)) {
    GTEST_SKIP() << "query_params.jsonl not found at " << params_path
                 << " — set PEACOCK_TPCH_VEC_PARAMS. NOTHING WAS VERIFIED.";
  }
  // one probe per k tier: 10 / 100 / 1000
  const std::vector<std::string> want{"img_000", "img_017", "img_034"};
  auto probes = load_probes(params_path, want);
  ASSERT_EQ(probes.size(), want.size()) << "missing probes in " << params_path;
  for (auto const& p : probes) ASSERT_EQ(p.q.size(), 96u) << p.id << ": expected 96 dims";

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD ----------------
  // Columns only; the embedding column is loaded IN FULL (12.3 GB) — the distance
  // predicate is not pushed into the reader, which is the constraint under test.
  auto ps_in = read_cols(data_dir() + "/partsupp.parquet",
                         {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"});
  auto sup_in = read_cols(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  auto nat_in = read_cols(data_dir() + "/nation.parquet", {"n_nationkey", "n_name"});
  raft::resources handle;
  auto emb = load_embedding_column(data_dir() + "/partsupp.parquet", "ps_image_embedding", 96,
                                   raft::resource::get_cuda_stream(handle));
  // Over the int32 list-child ceiling at sf40 — must chunk. One batch would mean the
  // ceiling arithmetic in load_embedding_column is no longer exercised.
  ASSERT_GT(emb.batches, 1) << "ps_image_embedding at sf40 must need chunking";
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();
  const auto n_ps = ps_in.tbl->num_rows();
  ASSERT_EQ(emb.rows, static_cast<int64_t>(n_ps))
      << "embedding row count disagrees with the partsupp scalar columns";
  std::fprintf(stderr, "[q11v] loaded partsupp %ld, supplier %ld, nation %ld\n",
               static_cast<long>(n_ps), static_cast<long>(sup_in.tbl->num_rows()),
               static_cast<long>(nat_in.tbl->num_rows()));

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};

  // SETUP: cuVS index build, timed separately (reported as index-build in the benchmark).
  const auto t_index0 = std::chrono::steady_clock::now();
  auto bf_index = build_bf_index(handle, emb.buf.data(), emb.rows, 96);
  cudaDeviceSynchronize();
  const auto t_index1 = std::chrono::steady_clock::now();
  const double index_ms =
      std::chrono::duration<double, std::milli>(t_index1 - t_index0).count();
  note_peak();
  std::fprintf(stderr, "[q11v] cuVS brute-force index built over %ld x 96 vectors (%.1f ms)\n",
               static_cast<long>(n_ps), index_ms);

  // --- the GERMANY join, built ONCE and used twice (threshold + filtered groups) ---
  auto germany = cudf::string_scalar(std::string("GERMANY"));
  auto nmask = cudf::binary_operation(nat_in.tbl->view().column(1), germany,
                                      cudf::binary_operator::EQUAL, boolean);
  auto nat_de = cudf::apply_boolean_mask(
      cudf::table_view{{nat_in.tbl->view().column(0)}}, nmask->view());
  auto [s_map, n_map] = cudf::inner_join(cudf::table_view{{sup_in.tbl->view().column(1)}},
                                         cudf::table_view{{nat_de->get_column(0).view()}});
  auto sup_de = cudf::gather(cudf::table_view{{sup_in.tbl->view().column(0)}}, map_view(s_map));
  std::fprintf(stderr, "[q11v] german suppliers: %ld\n",
               static_cast<long>(sup_de->num_rows()));

  auto [ps_map, sd_map] = cudf::inner_join(cudf::table_view{{ps_in.tbl->view().column(1)}},
                                           cudf::table_view{{sup_de->get_column(0).view()}});
  // partsupp rows belonging to German suppliers: partkey, availqty, supplycost, and the
  // ORIGINAL row index, which is what lets the vector hit-list be intersected below
  auto row_idx = cudf::sequence(static_cast<cudf::size_type>(n_ps),
                                *cudf::make_fixed_width_scalar<int32_t>(0),
                                *cudf::make_fixed_width_scalar<int32_t>(1));
  auto de = cudf::gather(cudf::table_view{{ps_in.tbl->view().column(0),   // ps_partkey
                                           ps_in.tbl->view().column(2),   // ps_availqty
                                           ps_in.tbl->view().column(3),   // ps_supplycost
                                           row_idx->view()}},             // original index
                         map_view(ps_map));
  note_peak();
  std::fprintf(stderr, "[q11v] german partsupp rows: %ld\n", static_cast<long>(de->num_rows()));
  ASSERT_GT(de->num_rows(), 0);

  // value = ps_supplycost * ps_availqty, exact decimal (scale -2 * scale -2 -> -4)
  auto de_avail = cudf::cast(de->get_column(1).view(), dec128_s2);
  auto de_cost = cudf::cast(de->get_column(2).view(), dec128_s2);
  auto de_value = cudf::binary_operation(de_cost->view(), de_avail->view(),
                                         cudf::binary_operator::MUL, dec128_s4);

  // HAVING threshold: sum over ALL German rows (no vector predicate), times 0.000002.
  // Kept in decimal; the 0.000002 factor is applied by DIVIDING by 500000, which is exact,
  // rather than multiplying by a float literal that has no exact decimal representation.
  auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto total = cudf::reduce(de_value->view(), *sum_agg, dec128_s4);
  auto* total_fp = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal128>*>(total.get());
  ASSERT_NE(total_fp, nullptr);
  const __int128 total_unscaled = static_cast<__int128>(total_fp->value());
  std::fprintf(stderr, "[q11v] german total value (scale -4 units): %s\n",
               to_string_i128(total_unscaled).c_str());

  const auto t_setup = std::chrono::steady_clock::now();

  // Per-probe execute: search -> range filter -> intersect with the (shared) German rows
  // -> groupby -> HAVING -> sort. The German join and index build are setup, reused across
  // all 3 probes.
  const int64_t K = search_k();
  ASSERT_LE(K, n_ps);
  for (auto const& probe : probes) {
    // Execute closure — verified once, then the benchmark times the SAME code. The cuVS
    // search is inside; golden reads are not.
    auto execute = [&]() -> std::unique_ptr<cudf::table> {
      auto hits = vector_range_hits(handle, bf_index, probe, 96, K, "q11v");

      // de column 3 is each German row's ORIGINAL partsupp index, so intersecting with the
      // vector hits is an inner join against the hit list.
      auto d_hits = to_device_hits(hits);
      auto [de_map, hit_map] = cudf::inner_join(
          cudf::table_view{{de->get_column(3).view()}}, cudf::table_view{{d_hits.view()}});
      auto sel = cudf::gather(cudf::table_view{{de->get_column(0).view(), de_value->view()}},
                              map_view(de_map));

      cudf::groupby::groupby gb(cudf::table_view{{sel->get_column(0).view()}});
      std::vector<cudf::groupby::aggregation_request> reqs;
      {
        cudf::groupby::aggregation_request r;
        r.values = sel->get_column(1).view();
        r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        reqs.push_back(std::move(r));
      }
      auto [gk, ga] = gb.aggregate(reqs);
      auto values = std::move(ga[0].results[0]);

      // HAVING value > total/500000  (0.000002 exactly, in decimal — no float literal)
      auto thresh = cudf::fixed_point_scalar<numeric::decimal128>(
          total_unscaled / 500000, numeric::scale_type{-4});
      auto keep = cudf::binary_operation(values->view(), thresh,
                                         cudf::binary_operator::GREATER, boolean);
      auto kept = cudf::apply_boolean_mask(
          cudf::table_view{{gk->view().column(0), values->view()}}, keep->view());

      // ORDER BY value DESC, ps_partkey  (partkey breaks value ties -> total order)
      auto order = cudf::sorted_order(
          cudf::table_view{{kept->get_column(1).view(), kept->get_column(0).view()}},
          {cudf::order::DESCENDING, cudf::order::ASCENDING},
          {cudf::null_order::AFTER, cudf::null_order::AFTER});
      return cudf::gather(
          cudf::table_view{{kept->get_column(0).view(), kept->get_column(1).view()}},
          order->view());
    };

    // corroboration #1 (verify path only): row count under D vs DuckDB
    auto count_hits = vector_range_hits(handle, bf_index, probe, 96, K, "q11v");
    expect_hit_count(count_hits.size(),
                     golden_dir() + "/duckdb_psimage_" + probe.id + ".count.csv", "q11v",
                     probe.id);

    auto sorted = execute();
    note_peak();
    // corroboration #2: the final grouped result, exact on both columns
    const std::vector<ColSpec> spec = {
        {"ps_partkey", Cmp::ExactInt},
        {"value", Cmp::ExactDecimal},
    };
    const auto golden = read_csv_golden(golden_dir() + "/duckdb_q11v_" + probe.id + ".csv");
    compare_table_to_golden(sorted->view(), golden, spec, ("q11v/" + probe.id).c_str());
    std::fprintf(stderr, "[q11v] %s: %ld result rows matched\n", probe.id.c_str(),
                 static_cast<long>(sorted->num_rows()));

    // index-build is amortised across the 3 probes: per-probe cuDF cost for the DuckDB
    // comparison is execute + index_ms/3.
    benchmark_execute(("q11v/" + probe.id).c_str(), execute,
                      std::chrono::duration<double, std::milli>(t_loaded - t0).count(),
                      index_ms);
  }

  const auto t_done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q11v] load %ld ms, setup %ld ms, search+verify %ld ms, total %ld ms\n",
               ms(t0, t_loaded), ms(t_loaded, t_setup), ms(t_setup, t_done), ms(t0, t_done));
}

// q12v, q10v and q9v all restrict `part` by the same p_text_embedding predicate (via
// parts_under_d) and differ only in what they join and aggregate afterwards. part at sf40
// is under the list-child ceiling, so its embedding loads in ONE read (asserted per test,
// so a scale factor crossing the boundary fails loudly).

// ===========================================================================
// TPC-H+V q12v — shipping-mode SLA counts inside a vector neighbourhood.
//
//   SELECT l_shipmode,
//          sum(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END),
//          sum(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END)
//   FROM orders, lineitem, part
//   WHERE o_orderkey=l_orderkey AND l_partkey=p_partkey
//     AND l_shipmode IN ('MAIL','SHIP')
//     AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
//     AND l_receiptdate >= 1994-01-01 AND l_receiptdate < 1995-01-01
//     AND p_text_embedding <-> ${q} < ${D}
//   GROUP BY l_shipmode ORDER BY l_shipmode
//
// Unique coverage: column-to-column TIMESTAMP_DAYS comparisons (elsewhere dates are only
// compared to literals — the same type-id trap that once produced a silent-zero bug when a
// reader assumed an integer); all-integer outputs; a string group key.
//
// Join order (hand-chosen): filter lineitem locally first (largest table), join the
// vector-restricted part before orders (far more selective). All comparisons EXACT
// (cudf SUM over INT32 -> INT64 vs DuckDB HUGEINT, both far from any limit).
// ===========================================================================
TEST_F(TpchSf40, Q12VectorShipModeCounts) {
  const auto params_path = vec_params_path();
  if (!file_exists(params_path)) {
    GTEST_SKIP() << "query_params.jsonl not found at " << params_path
                 << " — set PEACOCK_TPCH_VEC_PARAMS. NOTHING WAS VERIFIED.";
  }
  auto probes = load_probes(params_path, {"txt_000"});
  ASSERT_EQ(probes.size(), 1u) << "missing probe txt_000 in " << params_path;
  const auto& probe = probes.front();
  ASSERT_EQ(probe.q.size(), 100u) << "txt_000 should be a 100-dim text probe";
  const auto golden_path = golden_dir() + "/duckdb_q12v_" + probe.id + ".csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD ----------------
  auto ord_in = read_cols(data_dir() + "/orders.parquet", {"o_orderkey", "o_orderpriority"});
  auto line_in = read_cols(data_dir() + "/lineitem.parquet",
                           {"l_orderkey", "l_partkey", "l_shipmode", "l_commitdate",
                            "l_receiptdate", "l_shipdate"});
  const auto part_path = data_dir() + "/part.parquet";
  auto part_in = read_cols(part_path, {"p_partkey"});
  raft::resources handle;
  auto emb = load_embedding_column(part_path, "p_text_embedding", 100,
                                   raft::resource::get_cuda_stream(handle));
  EXPECT_EQ(emb.batches, 1) << "p_text_embedding at sf40 must load in a single read";
  EXPECT_EQ(emb.rows, static_cast<int64_t>(part_in.tbl->num_rows()));
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();

  // SETUP: cuVS index build, timed separately from load and execute.
  const int64_t K = search_k();
  ASSERT_LE(K, emb.rows);
  const auto t_index0 = std::chrono::steady_clock::now();
  auto index = build_bf_index(handle, emb.buf.data(), emb.rows, 100);
  cudaDeviceSynchronize();
  const double index_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_index0).count();
  std::fprintf(stderr, "[q12v] loaded orders %ld, lineitem %ld, part %ld; index built %.1f ms\n",
               static_cast<long>(ord_in.tbl->num_rows()),
               static_cast<long>(line_in.tbl->num_rows()),
               static_cast<long>(part_in.tbl->num_rows()), index_ms);

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto int32 = cudf::data_type{cudf::type_id::INT32};
  auto lv = line_in.tbl->view();
  auto pk = part_in.tbl->view().column(0);

  // Closure so the test verifies once and the benchmark times the SAME code. The vector
  // search is an operator, timed inside execute; only the index build is setup.
  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    auto sel_partkeys = parts_under_d(handle, index, pk, probe, 100, K, "q12v");

    // l_shipmode IN ('MAIL','SHIP')
    auto mail = cudf::string_scalar(std::string("MAIL"));
    auto ship = cudf::string_scalar(std::string("SHIP"));
    auto is_mail = cudf::binary_operation(lv.column(2), mail, cudf::binary_operator::EQUAL, boolean);
    auto is_ship = cudf::binary_operation(lv.column(2), ship, cudf::binary_operator::EQUAL, boolean);
    auto mode_ok = cudf::binary_operation(is_mail->view(), is_ship->view(),
                                          cudf::binary_operator::LOGICAL_OR, boolean);

    // column-to-column TIMESTAMP_DAYS comparisons — no casts, types stay as read
    auto commit_lt_receipt = cudf::binary_operation(lv.column(3), lv.column(4),
                                                    cudf::binary_operator::LESS, boolean);
    auto ship_lt_commit = cudf::binary_operation(lv.column(5), lv.column(3),
                                                 cudf::binary_operator::LESS, boolean);

    auto d1994 = date_scalar(1994, 1, 1);
    auto d1995 = date_scalar(1995, 1, 1);
    auto recv_ge = cudf::binary_operation(lv.column(4), d1994,
                                          cudf::binary_operator::GREATER_EQUAL, boolean);
    auto recv_lt = cudf::binary_operation(lv.column(4), d1995, cudf::binary_operator::LESS, boolean);

    auto m1 = cudf::binary_operation(mode_ok->view(), commit_lt_receipt->view(),
                                     cudf::binary_operator::LOGICAL_AND, boolean);
    auto m2 = cudf::binary_operation(m1->view(), ship_lt_commit->view(),
                                     cudf::binary_operator::LOGICAL_AND, boolean);
    auto m3 = cudf::binary_operation(m2->view(), recv_ge->view(),
                                     cudf::binary_operator::LOGICAL_AND, boolean);
    auto line_mask = cudf::binary_operation(m3->view(), recv_lt->view(),
                                            cudf::binary_operator::LOGICAL_AND, boolean);
    auto line_f = cudf::apply_boolean_mask(
        cudf::table_view{{lv.column(0), lv.column(1), lv.column(2)}}, line_mask->view());
    EXPECT_GT(line_f->num_rows(), 0);

    // join 1: lineitem.l_partkey = (parts under D).p_partkey
    auto [l_map, p_map] = cudf::inner_join(
        cudf::table_view{{line_f->get_column(1).view()}},
        cudf::table_view{{sel_partkeys->get_column(0).view()}});
    auto lp = cudf::gather(cudf::table_view{{line_f->get_column(0).view(),    // l_orderkey
                                             line_f->get_column(2).view()}},  // l_shipmode
                           map_view(l_map));
    EXPECT_GT(lp->num_rows(), 0) << "vector predicate and filters left no rows to join";

    // join 2: |X| orders on orderkey
    auto [lp_map, o_map] = cudf::inner_join(cudf::table_view{{lp->get_column(0).view()}},
                                            cudf::table_view{{ord_in.tbl->view().column(0)}});
    auto mode_col = cudf::gather(cudf::table_view{{lp->get_column(1).view()}}, map_view(lp_map));
    auto prio_col = cudf::gather(cudf::table_view{{ord_in.tbl->view().column(1)}}, map_view(o_map));

    // the two CASE expressions, as boolean masks cast to counters
    auto urgent = cudf::string_scalar(std::string("1-URGENT"));
    auto high = cudf::string_scalar(std::string("2-HIGH"));
    auto is_urgent = cudf::binary_operation(prio_col->get_column(0).view(), urgent,
                                            cudf::binary_operator::EQUAL, boolean);
    auto is_high = cudf::binary_operation(prio_col->get_column(0).view(), high,
                                          cudf::binary_operator::EQUAL, boolean);
    auto is_hi = cudf::binary_operation(is_urgent->view(), is_high->view(),
                                        cudf::binary_operator::LOGICAL_OR, boolean);
    // low = NOT is_hi: DuckDB writes <> AND <>, equivalent here because o_orderpriority is
    // NOT NULL in TPC-H; negation keeps the two counters from drifting apart.
    auto is_lo = cudf::unary_operation(is_hi->view(), cudf::unary_operator::NOT);
    auto hi_i = cudf::cast(is_hi->view(), int32);
    auto lo_i = cudf::cast(is_lo->view(), int32);

    // groupby l_shipmode -> sum(hi), sum(lo)
    cudf::groupby::groupby gb(cudf::table_view{{mode_col->get_column(0).view()}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    {
      cudf::groupby::aggregation_request r;
      r.values = hi_i->view();
      r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    {
      cudf::groupby::aggregation_request r;
      r.values = lo_i->view();
      r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    auto [gk, ga] = gb.aggregate(reqs);
    auto hi_sum = std::move(ga[0].results[0]);
    auto lo_sum = std::move(ga[1].results[0]);

    // ORDER BY l_shipmode — the group key, unique per row, so this is a TOTAL order
    auto order = cudf::sorted_order(cudf::table_view{{gk->view().column(0)}},
                                    {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
    return cudf::gather(
        cudf::table_view{{gk->view().column(0), hi_sum->view(), lo_sum->view()}}, order->view());
  };

  // count corroboration (verify path only), and vacuity guard
  auto count_hits = vector_range_hits(handle, index, probe, 100, K, "q12v");
  expect_hit_count(count_hits.size(),
                   golden_dir() + "/duckdb_ptext_" + probe.id + ".count.csv", "q12v", probe.id);
  ASSERT_GT(count_hits.size(), 0u) << "no part rows under D — test would be vacuous";

  auto sorted = execute();
  note_peak();
  const auto t_done = std::chrono::steady_clock::now();

  // ---------------- COMPARE (exact throughout) ----------------
  const std::vector<ColSpec> spec = {
      {"l_shipmode", Cmp::ExactString},
      {"high_line_count", Cmp::ExactInt},
      {"low_line_count", Cmp::ExactInt},
  };
  const auto golden = read_csv_golden(golden_path);
  ASSERT_GT(golden.size(), 0u) << "golden is empty: " << golden_path;
  compare_table_to_golden(sorted->view(), golden, spec, "q12v");
  std::fprintf(stderr, "[q12v] %ld result rows matched\n", static_cast<long>(sorted->num_rows()));

  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q12v] load %ld ms, execute %ld ms, total %ld ms\n", ms(t0, t_loaded),
               ms(t_loaded, t_done), ms(t0, t_done));

  benchmark_execute("q12v", execute,
                    std::chrono::duration<double, std::milli>(t_loaded - t0).count(), index_ms);
}

// ===========================================================================
// TPC-H+V q10v — returned-item revenue by customer, inside a vector neighbourhood.
//
//   SELECT c_custkey, c_name, sum(l_extendedprice*(1-l_discount)) AS revenue,
//          c_acctbal, n_name, c_address, c_phone, c_comment
//   FROM customer, orders, lineitem, nation, part
//   WHERE c_custkey=o_custkey AND l_orderkey=o_orderkey AND l_partkey=p_partkey
//     AND o_orderdate >= 1993-10-01 AND o_orderdate < 1994-01-01
//     AND l_returnflag='R' AND c_nationkey=n_nationkey
//     AND p_text_embedding <-> ${q} < ${D}
//   GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
//   ORDER BY revenue DESC, c_custkey LIMIT 20
//
// Unique coverage: GROUP BY seven columns (five strings, c_comment ~72 chars) — the
// variable-width hash path is otherwise untested — and a top-N gathering six string
// columns.
//
// TIE-BREAK: Q10's ORDER BY revenue alone is not a total order, so LIMIT 20 would be
// flaky; c_custkey (unique per group) is appended on BOTH sides.
//
// Join order (hand-chosen): part join first — it is the only thing that cuts lineitem
// down, keeping the orders join small. All comparisons EXACT (revenue DECIMAL128 scale
// -4; c_acctbal carried through the groupby as a KEY, never re-derived). c_address and
// c_comment contain commas, so the golden uses RFC4180 quoting, which read_csv_golden
// parses.
// ===========================================================================
TEST_F(TpchSf40, Q10VectorCustomerTopN) {
  const auto params_path = vec_params_path();
  if (!file_exists(params_path)) {
    GTEST_SKIP() << "query_params.jsonl not found at " << params_path
                 << " — set PEACOCK_TPCH_VEC_PARAMS. NOTHING WAS VERIFIED.";
  }
  auto probes = load_probes(params_path, {"txt_017"});
  ASSERT_EQ(probes.size(), 1u) << "missing probe txt_017 in " << params_path;
  const auto& probe = probes.front();
  ASSERT_EQ(probe.q.size(), 100u) << "txt_017 should be a 100-dim text probe";
  const auto golden_path = golden_dir() + "/duckdb_q10v_" + probe.id + ".csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD ----------------
  auto cust_in = read_cols(data_dir() + "/customer.parquet",
                           {"c_custkey", "c_name", "c_address", "c_nationkey", "c_phone",
                            "c_acctbal", "c_comment"});
  auto ord_in = read_cols(data_dir() + "/orders.parquet",
                          {"o_orderkey", "o_custkey", "o_orderdate"});
  auto line_in = read_cols(data_dir() + "/lineitem.parquet",
                           {"l_orderkey", "l_partkey", "l_extendedprice", "l_discount",
                            "l_returnflag"});
  auto nat_in = read_cols(data_dir() + "/nation.parquet", {"n_nationkey", "n_name"});
  const auto part_path = data_dir() + "/part.parquet";
  auto part_in = read_cols(part_path, {"p_partkey"});
  raft::resources handle;
  auto emb = load_embedding_column(part_path, "p_text_embedding", 100,
                                   raft::resource::get_cuda_stream(handle));
  EXPECT_EQ(emb.batches, 1) << "p_text_embedding at sf40 must load in a single read";
  EXPECT_EQ(emb.rows, static_cast<int64_t>(part_in.tbl->num_rows()));
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();

  // SETUP: cuVS index build, timed separately.
  const int64_t K = search_k();
  ASSERT_LE(K, emb.rows);
  const auto t_index0 = std::chrono::steady_clock::now();
  auto index = build_bf_index(handle, emb.buf.data(), emb.rows, 100);
  cudaDeviceSynchronize();
  const double index_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_index0).count();
  std::fprintf(stderr, "[q10v] loaded customer %ld, orders %ld, lineitem %ld, part %ld; index %.1f ms\n",
               static_cast<long>(cust_in.tbl->num_rows()),
               static_cast<long>(ord_in.tbl->num_rows()),
               static_cast<long>(line_in.tbl->num_rows()),
               static_cast<long>(part_in.tbl->num_rows()), index_ms);

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};
  auto cv = cust_in.tbl->view();
  auto ov = ord_in.tbl->view();
  auto lv = line_in.tbl->view();
  auto pk = part_in.tbl->view().column(0);

  // Closure so verify and benchmark run the SAME code; LIMIT 20 is a zero-copy slice
  // applied outside, in the verify path.
  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    auto sel_partkeys = parts_under_d(handle, index, pk, probe, 100, K, "q10v");

    // filter lineitem: l_returnflag = 'R'
    auto flag_r = cudf::string_scalar(std::string("R"));
    auto line_mask = cudf::binary_operation(lv.column(4), flag_r,
                                            cudf::binary_operator::EQUAL, boolean);
    auto line_f = cudf::apply_boolean_mask(
        cudf::table_view{{lv.column(0), lv.column(1), lv.column(2), lv.column(3)}},
        line_mask->view());
    note_peak();

    // filter orders: the 1993-10-01 .. 1994-01-01 quarter
    auto d_lo = date_scalar(1993, 10, 1);
    auto d_hi = date_scalar(1994, 1, 1);
    auto o_ge = cudf::binary_operation(ov.column(2), d_lo,
                                       cudf::binary_operator::GREATER_EQUAL, boolean);
    auto o_lt = cudf::binary_operation(ov.column(2), d_hi, cudf::binary_operator::LESS, boolean);
    auto ord_mask = cudf::binary_operation(o_ge->view(), o_lt->view(),
                                           cudf::binary_operator::LOGICAL_AND, boolean);
    auto ord_f = cudf::apply_boolean_mask(cudf::table_view{{ov.column(0), ov.column(1)}},
                                          ord_mask->view());
    EXPECT_GT(line_f->num_rows(), 0);
    EXPECT_GT(ord_f->num_rows(), 0);

    // join 1: lineitem.l_partkey = (parts under D).p_partkey
    auto [l_map, p_map] = cudf::inner_join(
        cudf::table_view{{line_f->get_column(1).view()}},
        cudf::table_view{{sel_partkeys->get_column(0).view()}});
    auto lp = cudf::gather(cudf::table_view{{line_f->get_column(0).view(),    // l_orderkey
                                             line_f->get_column(2).view(),    // l_extendedprice
                                             line_f->get_column(3).view()}},  // l_discount
                           map_view(l_map));
    note_peak();

    // join 2: |X| orders on orderkey
    auto [lp_map, o_map] = cudf::inner_join(cudf::table_view{{lp->get_column(0).view()}},
                                            cudf::table_view{{ord_f->get_column(0).view()}});
    auto lo_l = cudf::gather(cudf::table_view{{lp->get_column(1).view(), lp->get_column(2).view()}},
                             map_view(lp_map));
    auto lo_o = cudf::gather(cudf::table_view{{ord_f->get_column(1).view()}},  // o_custkey
                             map_view(o_map));
    EXPECT_GT(lo_o->num_rows(), 0) << "vector predicate and filters left no rows to join";

    // join 3: |X| customer on custkey
    auto [c_map, lo_map] = cudf::inner_join(cudf::table_view{{cv.column(0)}},
                                            cudf::table_view{{lo_o->get_column(0).view()}});
    auto cust_side = cudf::gather(cudf::table_view{{cv.column(0),    // c_custkey
                                                    cv.column(1),    // c_name
                                                    cv.column(5),    // c_acctbal
                                                    cv.column(4),    // c_phone
                                                    cv.column(2),    // c_address
                                                    cv.column(6),    // c_comment
                                                    cv.column(3)}},  // c_nationkey
                                  map_view(c_map));
    auto val_side = cudf::gather(
        cudf::table_view{{lo_l->get_column(0).view(), lo_l->get_column(1).view()}},
        map_view(lo_map));

    // join 4: |X| nation on nationkey
    auto [cn_map, n_map] = cudf::inner_join(cudf::table_view{{cust_side->get_column(6).view()}},
                                            cudf::table_view{{nat_in.tbl->view().column(0)}});
    auto cust_j = cudf::gather(cudf::table_view{{cust_side->get_column(0).view(),
                                                 cust_side->get_column(1).view(),
                                                 cust_side->get_column(2).view(),
                                                 cust_side->get_column(3).view(),
                                                 cust_side->get_column(4).view(),
                                                 cust_side->get_column(5).view()}},
                               map_view(cn_map));
    auto val_j = cudf::gather(
        cudf::table_view{{val_side->get_column(0).view(), val_side->get_column(1).view()}},
        map_view(cn_map));
    auto nat_j = cudf::gather(cudf::table_view{{nat_in.tbl->view().column(1)}}, map_view(n_map));
    note_peak();
    EXPECT_GT(cust_j->num_rows(), 0);

    // revenue = l_extendedprice * (1 - l_discount), exact decimal (scale -2 * -2 -> -4)
    auto price = cudf::cast(val_j->get_column(0).view(), dec128_s2);
    auto disc = cudf::cast(val_j->get_column(1).view(), dec128_s2);
    auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
    auto one_minus_disc =
        cudf::binary_operation(one_s2, disc->view(), cudf::binary_operator::SUB, dec128_s2);
    auto revenue = cudf::binary_operation(price->view(), one_minus_disc->view(),
                                          cudf::binary_operator::MUL, dec128_s4);

    // groupby on all SEVEN key columns, in DuckDB's GROUP BY order
    cudf::groupby::groupby gb(cudf::table_view{{cust_j->get_column(0).view(),   // c_custkey
                                                cust_j->get_column(1).view(),   // c_name
                                                cust_j->get_column(2).view(),   // c_acctbal
                                                cust_j->get_column(3).view(),   // c_phone
                                                nat_j->get_column(0).view(),    // n_name
                                                cust_j->get_column(4).view(),   // c_address
                                                cust_j->get_column(5).view()}});// c_comment
    std::vector<cudf::groupby::aggregation_request> reqs;
    {
      cudf::groupby::aggregation_request r;
      r.values = revenue->view();
      r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    auto [gk, ga] = gb.aggregate(reqs);
    auto rev_col = std::move(ga[0].results[0]);
    EXPECT_GE(gk->num_rows(), 20) << "fewer than 20 groups — LIMIT 20 would not be exercised";

    // ORDER BY revenue DESC, c_custkey ASC  (total order — see the header)
    auto gkv = gk->view();
    auto order = cudf::sorted_order(cudf::table_view{{rev_col->view(), gkv.column(0)}},
                                    {cudf::order::DESCENDING, cudf::order::ASCENDING},
                                    {cudf::null_order::AFTER, cudf::null_order::AFTER});
    // output column order matches the golden: custkey, name, revenue, acctbal, nation,
    // address, phone, comment
    return cudf::gather(cudf::table_view{{gkv.column(0), gkv.column(1), rev_col->view(),
                                          gkv.column(2), gkv.column(4), gkv.column(5),
                                          gkv.column(3), gkv.column(6)}},
                        order->view());
  };

  // count corroboration (verify path only), and vacuity guard
  auto count_hits = vector_range_hits(handle, index, probe, 100, K, "q10v");
  expect_hit_count(count_hits.size(),
                   golden_dir() + "/duckdb_ptext_" + probe.id + ".count.csv", "q10v", probe.id);
  ASSERT_GT(count_hits.size(), 0u) << "no part rows under D — test would be vacuous";

  auto sorted = execute();
  note_peak();
  auto top = cudf::slice(sorted->view(), {0, 20})[0];  // LIMIT 20 (zero-copy view)
  const auto t_done = std::chrono::steady_clock::now();

  // ---------------- COMPARE (exact throughout) ----------------
  const std::vector<ColSpec> spec = {
      {"c_custkey", Cmp::ExactInt},   {"c_name", Cmp::ExactString},
      {"revenue", Cmp::ExactDecimal}, {"c_acctbal", Cmp::ExactDecimal},
      {"n_name", Cmp::ExactString},   {"c_address", Cmp::ExactString},
      {"c_phone", Cmp::ExactString},  {"c_comment", Cmp::ExactString},
  };
  const auto golden = read_csv_golden(golden_path);
  ASSERT_EQ(static_cast<int>(golden.size()), 20) << "golden should hold 20 rows";
  compare_table_to_golden(top, golden, spec, "q10v");
  std::fprintf(stderr, "[q10v] 20 result rows matched\n");

  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q10v] load %ld ms, execute %ld ms, total %ld ms\n", ms(t0, t_loaded),
               ms(t_loaded, t_done), ms(t0, t_done));

  benchmark_execute("q10v", execute,
                    std::chrono::duration<double, std::milli>(t_loaded - t0).count(), index_ms);
}

// ===========================================================================
// TPC-H+V q9v — product-line profit by nation and year, inside a vector neighbourhood.
//
//   SELECT nation, o_year, sum(amount)
//   FROM (SELECT n_name, extract(year FROM o_orderdate),
//                l_extendedprice*(1-l_discount) - ps_supplycost*l_quantity
//         FROM part, supplier, lineitem, partsupp, orders, nation
//         WHERE s_suppkey=l_suppkey AND ps_suppkey=l_suppkey AND ps_partkey=l_partkey
//           AND p_partkey=l_partkey AND o_orderkey=l_orderkey AND s_nationkey=n_nationkey
//           AND p_name LIKE '%green%'
//           AND p_text_embedding <-> ${q} < ${D})
//   GROUP BY nation, o_year ORDER BY nation, o_year DESC
//
// Unique coverage: the suite's only COMPOSITE two-column join (partsupp |X| lineitem on
// partkey+suppkey — the shape most likely to be quietly wrong: match on one column and
// row counts still look plausible while every supplycost comes from the wrong supplier);
// a substring predicate (strings::contains); a decimal subtraction of two products; a
// two-key GROUP BY with descending secondary sort.
//
// Join order (hand-chosen): part filtered first — it carries both selective predicates;
// everything after operates on ~1e4 rows instead of 1e8. All comparisons EXACT
// (o_year INT16 vs DuckDB BIGINT, compared as integers; sum_profit DECIMAL128 scale -4).
// ===========================================================================
TEST_F(TpchSf40, Q9VectorCompositeJoin) {
  const auto params_path = vec_params_path();
  if (!file_exists(params_path)) {
    GTEST_SKIP() << "query_params.jsonl not found at " << params_path
                 << " — set PEACOCK_TPCH_VEC_PARAMS. NOTHING WAS VERIFIED.";
  }
  auto probes = load_probes(params_path, {"txt_034"});
  ASSERT_EQ(probes.size(), 1u) << "missing probe txt_034 in " << params_path;
  const auto& probe = probes.front();
  ASSERT_EQ(probe.q.size(), 100u) << "txt_034 should be a 100-dim text probe";
  const auto golden_path = golden_dir() + "/duckdb_q9v_" + probe.id + ".csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  const auto t0 = std::chrono::steady_clock::now();

  // ---------------- PHASE 1: LOAD ----------------
  const auto part_path = data_dir() + "/part.parquet";
  auto part_name_in = read_cols(part_path, {"p_partkey", "p_name"});
  auto sup_in = read_cols(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  auto line_in = read_cols(data_dir() + "/lineitem.parquet",
                           {"l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice",
                            "l_discount", "l_quantity"});
  auto ps_in = read_cols(data_dir() + "/partsupp.parquet",
                         {"ps_partkey", "ps_suppkey", "ps_supplycost"});
  auto ord_in = read_cols(data_dir() + "/orders.parquet", {"o_orderkey", "o_orderdate"});
  auto nat_in = read_cols(data_dir() + "/nation.parquet", {"n_nationkey", "n_name"});
  raft::resources handle;
  auto emb = load_embedding_column(part_path, "p_text_embedding", 100,
                                   raft::resource::get_cuda_stream(handle));
  EXPECT_EQ(emb.batches, 1) << "p_text_embedding at sf40 must load in a single read";
  EXPECT_EQ(emb.rows, static_cast<int64_t>(part_name_in.tbl->num_rows()));
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();

  // SETUP: cuVS index build, timed separately.
  const int64_t K = search_k();
  ASSERT_LE(K, emb.rows);
  const auto t_index0 = std::chrono::steady_clock::now();
  auto index = build_bf_index(handle, emb.buf.data(), emb.rows, 100);
  cudaDeviceSynchronize();
  const double index_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_index0).count();
  std::fprintf(stderr,
               "[q9v] loaded part %ld, supplier %ld, lineitem %ld, partsupp %ld, orders %ld;"
               " index %.1f ms\n",
               static_cast<long>(part_name_in.tbl->num_rows()),
               static_cast<long>(sup_in.tbl->num_rows()),
               static_cast<long>(line_in.tbl->num_rows()),
               static_cast<long>(ps_in.tbl->num_rows()),
               static_cast<long>(ord_in.tbl->num_rows()), index_ms);

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};
  auto lv = line_in.tbl->view();
  auto pk = part_name_in.tbl->view().column(0);
  auto pname = part_name_in.tbl->view().column(1);

  // Closure so verify and benchmark run the SAME code.
  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    auto sel_partkeys = parts_under_d(handle, index, pk, probe, 100, K, "q9v");

    // p_name LIKE '%green%' — substring match over all of part, then intersected with the
    // vector hits; contains() over 8M short strings is cheap next to the distance scan.
    auto green = cudf::string_scalar(std::string("green"));
    auto green_mask = cudf::strings::contains(cudf::strings_column_view(pname), green);
    auto green_parts = cudf::apply_boolean_mask(cudf::table_view{{pk}}, green_mask->view());
    EXPECT_GT(green_parts->num_rows(), 0);

    auto [g_map, v_map] = cudf::inner_join(
        cudf::table_view{{green_parts->get_column(0).view()}},
        cudf::table_view{{sel_partkeys->get_column(0).view()}});
    auto part_sel = cudf::gather(cudf::table_view{{green_parts->get_column(0).view()}},
                                 map_view(g_map));
    EXPECT_GT(part_sel->num_rows(), 0) << "the two part predicates have no overlap";

    // join 1: part |X| lineitem on partkey — the decisive reduction, 240M -> ~63k
    auto [ps_map, l_map] = cudf::inner_join(cudf::table_view{{part_sel->get_column(0).view()}},
                                            cudf::table_view{{lv.column(1)}});
    auto li = cudf::gather(cudf::table_view{{lv.column(0),    // l_orderkey
                                             lv.column(1),    // l_partkey
                                             lv.column(2),    // l_suppkey
                                             lv.column(3),    // l_extendedprice
                                             lv.column(4),    // l_discount
                                             lv.column(5)}},  // l_quantity
                           map_view(l_map));
    note_peak();
    EXPECT_GT(li->num_rows(), 0);

    // join 2: THE COMPOSITE JOIN — (l_partkey, l_suppkey) vs (ps_partkey, ps_suppkey),
    // matched positionally. Joining on partkey alone would grow the row count ~4x with
    // wrong supplycosts, hence the exact row-count assertion below.
    auto [li_map, psx_map] = cudf::inner_join(
        cudf::table_view{{li->get_column(1).view(), li->get_column(2).view()}},
        cudf::table_view{{ps_in.tbl->view().column(0), ps_in.tbl->view().column(1)}});
    auto li_j = cudf::gather(cudf::table_view{{li->get_column(0).view(),    // l_orderkey
                                               li->get_column(2).view(),    // l_suppkey
                                               li->get_column(3).view(),    // l_extendedprice
                                               li->get_column(4).view(),    // l_discount
                                               li->get_column(5).view()}},  // l_quantity
                             map_view(li_map));
    auto cost_j = cudf::gather(cudf::table_view{{ps_in.tbl->view().column(2)}},  // ps_supplycost
                               map_view(psx_map));
    note_peak();
    // (partkey, suppkey) is UNIQUE in partsupp -> the join must preserve the lineitem row
    // count exactly. That is the point of the test.
    EXPECT_EQ(li_j->num_rows(), li->num_rows())
        << "composite join changed the lineitem row count — (ps_partkey, ps_suppkey) is "
           "unique in partsupp, so an inner join on both keys must preserve it exactly. "
           "A larger count means the join matched on fewer keys than it was given.";

    // join 3: |X| orders on orderkey (for o_orderdate)
    auto [lo_map, o_map] = cudf::inner_join(cudf::table_view{{li_j->get_column(0).view()}},
                                            cudf::table_view{{ord_in.tbl->view().column(0)}});
    auto li_o = cudf::gather(cudf::table_view{{li_j->get_column(1).view(),   // l_suppkey
                                               li_j->get_column(2).view(),   // l_extendedprice
                                               li_j->get_column(3).view(),   // l_discount
                                               li_j->get_column(4).view()}}, // l_quantity
                             map_view(lo_map));
    auto cost_o = cudf::gather(cudf::table_view{{cost_j->get_column(0).view()}}, map_view(lo_map));
    auto date_o = cudf::gather(cudf::table_view{{ord_in.tbl->view().column(1)}}, map_view(o_map));
    note_peak();
    EXPECT_GT(li_o->num_rows(), 0);

    // join 4: |X| supplier on suppkey, then |X| nation on nationkey
    auto [ls_map, s_map] = cudf::inner_join(cudf::table_view{{li_o->get_column(0).view()}},
                                            cudf::table_view{{sup_in.tbl->view().column(0)}});
    auto vals_s = cudf::gather(cudf::table_view{{li_o->get_column(1).view(),
                                                 li_o->get_column(2).view(),
                                                 li_o->get_column(3).view()}},
                               map_view(ls_map));
    auto cost_s = cudf::gather(cudf::table_view{{cost_o->get_column(0).view()}}, map_view(ls_map));
    auto date_s = cudf::gather(cudf::table_view{{date_o->get_column(0).view()}}, map_view(ls_map));
    auto natkey_s = cudf::gather(cudf::table_view{{sup_in.tbl->view().column(1)}}, map_view(s_map));

    auto [sn_map, n_map] = cudf::inner_join(cudf::table_view{{natkey_s->get_column(0).view()}},
                                            cudf::table_view{{nat_in.tbl->view().column(0)}});
    auto vals_n = cudf::gather(cudf::table_view{{vals_s->get_column(0).view(),
                                                 vals_s->get_column(1).view(),
                                                 vals_s->get_column(2).view()}},
                               map_view(sn_map));
    auto cost_n = cudf::gather(cudf::table_view{{cost_s->get_column(0).view()}}, map_view(sn_map));
    auto date_n = cudf::gather(cudf::table_view{{date_s->get_column(0).view()}}, map_view(sn_map));
    auto name_n = cudf::gather(cudf::table_view{{nat_in.tbl->view().column(1)}}, map_view(n_map));
    note_peak();
    EXPECT_GT(vals_n->num_rows(), 0);

    // extract_datetime_component (returns INT16), not extract_year — deleted in cudf 26.02
    auto o_year = cudf::datetime::extract_datetime_component(
        date_n->get_column(0).view(), cudf::datetime::datetime_component::YEAR);

    // amount = l_extendedprice*(1-l_discount) - ps_supplycost*l_quantity, exact decimal:
    // both products are scale -4 and the difference stays there.
    auto price = cudf::cast(vals_n->get_column(0).view(), dec128_s2);
    auto disc = cudf::cast(vals_n->get_column(1).view(), dec128_s2);
    auto qty = cudf::cast(vals_n->get_column(2).view(), dec128_s2);
    auto cost = cudf::cast(cost_n->get_column(0).view(), dec128_s2);
    auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
    auto one_minus_disc =
        cudf::binary_operation(one_s2, disc->view(), cudf::binary_operator::SUB, dec128_s2);
    auto gross = cudf::binary_operation(price->view(), one_minus_disc->view(),
                                        cudf::binary_operator::MUL, dec128_s4);
    auto outlay =
        cudf::binary_operation(cost->view(), qty->view(), cudf::binary_operator::MUL, dec128_s4);
    auto amount = cudf::binary_operation(gross->view(), outlay->view(),
                                         cudf::binary_operator::SUB, dec128_s4);

    // groupby (n_name, o_year) -> sum(amount)
    cudf::groupby::groupby gb(
        cudf::table_view{{name_n->get_column(0).view(), o_year->view()}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    {
      cudf::groupby::aggregation_request r;
      r.values = amount->view();
      r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    auto [gk, ga] = gb.aggregate(reqs);
    auto profit = std::move(ga[0].results[0]);

    // ORDER BY nation ASC, o_year DESC — together the group key, so this is a TOTAL order
    auto gkv = gk->view();
    auto order = cudf::sorted_order(cudf::table_view{{gkv.column(0), gkv.column(1)}},
                                    {cudf::order::ASCENDING, cudf::order::DESCENDING},
                                    {cudf::null_order::AFTER, cudf::null_order::AFTER});
    return cudf::gather(
        cudf::table_view{{gkv.column(0), gkv.column(1), profit->view()}}, order->view());
  };

  // count corroboration (verify path only), and vacuity guard
  auto count_hits = vector_range_hits(handle, index, probe, 100, K, "q9v");
  expect_hit_count(count_hits.size(),
                   golden_dir() + "/duckdb_ptext_" + probe.id + ".count.csv", "q9v", probe.id);
  ASSERT_GT(count_hits.size(), 0u) << "no part rows under D — test would be vacuous";

  auto sorted = execute();
  note_peak();
  const auto t_done = std::chrono::steady_clock::now();

  // ---------------- COMPARE (exact throughout) ----------------
  const std::vector<ColSpec> spec = {
      {"nation", Cmp::ExactString},
      {"o_year", Cmp::ExactInt},
      {"sum_profit", Cmp::ExactDecimal},
  };
  const auto golden = read_csv_golden(golden_path);
  ASSERT_GT(golden.size(), 0u) << "golden is empty: " << golden_path;
  compare_table_to_golden(sorted->view(), golden, spec, "q9v");
  std::fprintf(stderr, "[q9v] %ld result rows matched\n", static_cast<long>(sorted->num_rows()));

  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q9v] load %ld ms, execute %ld ms, total %ld ms\n", ms(t0, t_loaded),
               ms(t_loaded, t_done), ms(t0, t_done));

  benchmark_execute("q9v", execute,
                    std::chrono::duration<double, std::milli>(t_loaded - t0).count(), index_ms);
}

// Same entry point as the other gtest binaries here (the conda cudf ships no gtest_main).
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  peacock::install_rmm_pool();
  return RUN_ALL_TESTS();
}
