// test_tpchv.cpp — TPC-H+V (vector-augmented) query in BARE cuDF + cuVS brute force,
// checked against DuckDB.
//
// Same rules as test_tpch.cpp: no Rust, no DataFusion, no SQL; two visibly separate
// phases; every predicate is an operator in phase 2, never pushed into the reader —
// INCLUDING the distance predicate, which is the whole point of the exercise.
//
// LINKING cuVS — READ THIS BEFORE DEBUGGING A LOAD FAILURE:
// libcuvs.so cannot be dlopen'd on its own in this environment. rmm is header-only here
// (there is no librmm.so), so rmm symbols are compiled INTO consumers: libcudf.so DEFINES
// rmm::logger::~logger (_ZN3rmm6loggerD1Ev) and libcuvs.so REFERENCES it. Loading cuVS
// alone therefore fails with "undefined symbol: _ZN3rmm6loggerD1Ev", while loading libcudf
// first and cuVS second works. This binary links both, so the link order resolves it — but
// note that `ldd libcuvs.so` reports NO missing dependencies and is thus misleading; only
// an actual dlopen/link exposes it.
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

#include "tpch_golden.hpp"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

using namespace peacock_test;

namespace {

// One (q, D) probe resolved from the COMMITTED query_params.jsonl — never hardcoded here,
// so the golden and the GPU side cannot drift apart.
struct VecProbe {
  std::string id;
  int k = 0;
  double D = 0.0;
  std::vector<float> q;
};

// Minimal extraction of the three fields we need. The file is one JSON object per line
// with no nested objects, so a full parser would be more machinery than the format needs.
std::vector<VecProbe> load_probes(const std::string& path, std::vector<std::string> const& want) {
  std::vector<VecProbe> out;
  std::ifstream f(path);
  std::string line;
  // The file is written by json.dump, so there IS a space after each colon
  // ("id": "img_000"). Skip any whitespace rather than assuming a fixed offset — that
  // assumption silently found zero probes on the first run.
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

// ===========================================================================
// EMBEDDING LOAD — AN EMBEDDING COLUMN MAY NOT BE READABLE IN ONE CALL. READ THIS
// BEFORE "SIMPLIFYING" THE LOOP BELOW BACK INTO A SINGLE read_parquet.
//
// A cuDF LIST column stores its values in ONE contiguous child column whose length is a
// cudf::size_type, i.e. INT32. That caps the child at 2,147,483,647 elements:
//     96-wide image embeddings  -> max 22,369,621 rows  (~tpch.sf27)
//     100-wide text embeddings  -> max 21,474,836 rows  (~tpch.sf26)
// This is a real architectural limit for anything that stores embeddings as cuDF lists,
// and it is worth more than these tests. At sf40 the two columns fall on OPPOSITE SIDES
// of it:
//     partsupp.ps_image_embedding  32,000,000 x 96  = 3.072e9  -> 1.43x OVER, must chunk
//     part.p_text_embedding         8,000,000 x 100 = 0.800e9  -> 0.37x, one read
// A single read of the over-ceiling column fails in about one second with
//     std::bad_alloc: out_of_memory: cudaErrorMemoryAllocation
// WHICH IS A LIE: the device had 137 GB free and the column needs 12.3 GB. The message is
// the symptom of an overflowed size computation, not a real allocation shortfall.
// Verified three ways: the arithmetic above; an isolated repro in plain cuDF Python
// (cudf.read_parquet(..., columns=['ps_image_embedding']) alone fails); and a bisect that
// reads OK at 22,241,280 rows (0.99x the limit) and fails at 23,592,960 (1.05x).
//
// WHAT THE BATCHING IS NOT:
//   NOT a memory optimisation — there is 11x more device memory than needed.
//   NOT predicate pushdown — every row group is read, no rows are skipped or filtered,
//     and the distance predicate still runs in phase 2. The reader batches I/O; it does
//     not select data.
// The load-then-execute boundary is unchanged: every row is resident before any operator
// runs. cuVS/raft use INT64 extents (probed directly: build + search over a 32M x 96
// matrix succeed), which is why the reassembled buffer is representable even though a
// cuDF list column of the same data is not.
//
// ONE FUNCTION FOR BOTH REGIMES, deliberately: the batch size is derived from the ceiling
// and the width, so a 100-wide column at sf40 comes back in ONE batch through exactly the
// code that assembles the 96-wide one in several. Each caller asserts which regime it
// expects, so a change that silently moved a column across the boundary fails loudly.
// It always copies into an owning buffer, even in the one-batch case where the list
// child could be pointed at directly — 3.2 GB to keep one lifetime rule instead of two.
// ===========================================================================
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

  // Half the ceiling, so no single batch can approach it even if the row groups are very
  // uneven — the loop can only bound a batch by the file-average row-group size, since
  // exact per-group counts are not exposed.
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
  // COMPLETENESS: a dropped or double-counted batch would silently shorten the dataset and
  // change every distance result. Assert the reassembly, do not assume it.
  EXPECT_EQ(copied, n_rows) << col << ": chunked load reassembled " << copied
                            << " rows but the file has " << n_rows
                            << " — a batch was dropped or double-counted";
  std::fprintf(stderr, "[vec] %s: %ld rows x %d loaded in %d batch(es)\n", col.c_str(),
               static_cast<long>(copied), dim, m.batches);
  return m;
}

// ===========================================================================
// RANGE-VS-TOP-K, the semantic gap every query here has to bridge.
//
// cuVS 25.02 brute_force exposes only a TOP-K search; there is no range/radius API (I read
// the header — no range, radius or epsilon entry point exists). The queries need a RANGE:
// every row with distance < D. Resolution: search top-K with K chosen well above the
// number of rows that can fall under D, then filter by D. That is exact ONLY IF the top-K
// did not saturate, so the saturation guard is load-bearing, not decorative: if the K'th
// returned neighbour is still inside D, the answer was truncated and the test FAILS rather
// than silently returning a subset. That is the silent-wrong-answer shape this suite keeps
// finding, so it gets an explicit check.
//
// DISTANCE SPACE: cuVS defaults to L2Expanded, which returns SQUARED L2, whereas DuckDB's
// array_distance returns the root. Comparing those directly would be a silent factor-level
// error. Rather than square D or sqrt the results by hand, the index is built with
// L2SqrtExpanded, which returns true L2 — the same quantity DuckDB computes, so D is used
// as-is on both sides with no conversion to get wrong.
//
// CORROBORATION: the returned row count is checked against a DuckDB count over the same
// parquet BEFORE any join runs. That pins the row SET the distance predicate selects,
// independent of everything the joins and aggregations do afterwards. A final result can
// coincidentally match while the search returned the wrong neighbours; the count cannot.
// ===========================================================================
template <typename Index>
std::vector<int32_t> vector_range_hits(raft::resources& handle, Index const& index,
                                       VecProbe const& probe, int dim, int64_t K,
                                       const std::string& count_golden,
                                       const char* tag) {
  auto d_q = raft::make_device_matrix<float, int64_t>(handle, 1, dim);
  raft::copy(d_q.data_handle(), probe.q.data(), dim, raft::resource::get_cuda_stream(handle));
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(handle, 1, K);
  auto distances = raft::make_device_matrix<float, int64_t>(handle, 1, K);
  cuvs::neighbors::brute_force::search(handle, cuvs::neighbors::brute_force::search_params{},
                                       index, raft::make_const_mdspan(d_q.view()),
                                       neighbors.view(), distances.view());
  raft::resource::sync_stream(handle);

  // pull the hit list back and range-filter it on the host: K is O(1e5), trivial next to
  // the multi-million-row search that produced it
  std::vector<int64_t> h_nb(K);
  std::vector<float> h_di(K);
  raft::copy(h_nb.data(), neighbors.data_handle(), K, raft::resource::get_cuda_stream(handle));
  raft::copy(h_di.data(), distances.data_handle(), K, raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);

  std::vector<int32_t> hits;
  // SATURATION GUARD — see above. If the furthest neighbour cuVS returned is still inside
  // D, then rows beyond K also qualify and the range answer is TRUNCATED.
  EXPECT_GE(static_cast<double>(h_di[K - 1]), probe.D)
      << tag << "/" << probe.id << ": top-K SATURATED (K=" << K << ", furthest distance "
      << h_di[K - 1] << " < D=" << probe.D
      << ") — the range answer would be silently truncated. Raise K.";

  for (int64_t i = 0; i < K; ++i) {
    if (static_cast<double>(h_di[i]) < probe.D) hits.push_back(static_cast<int32_t>(h_nb[i]));
  }
  std::fprintf(stderr, "[%s] %s: D=%.17g -> %zu rows under D (furthest returned %.9g)\n", tag,
               probe.id.c_str(), probe.D, hits.size(), h_di[K - 1]);

  EXPECT_TRUE(file_exists(count_golden)) << "missing " << count_golden;
  if (file_exists(count_golden)) {
    const int64_t want =
        std::strtoll(read_single_value_golden(count_golden).c_str(), nullptr, 10);
    EXPECT_EQ(static_cast<int64_t>(hits.size()), want)
        << tag << "/" << probe.id << ": cuVS found " << hits.size()
        << " rows under D but DuckDB found " << want
        << ". A one-row difference here means cuVS and DuckDB landed on opposite sides of "
        << "the distance boundary — a real numerical divergence, NOT something to absorb "
        << "with a tolerance.";
  }
  return hits;
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

// K for the top-K-then-filter search. Overridable ONLY so the saturation guard can be
// exercised: a guard that has never fired is not a guard. Setting PEACOCK_TPCHV_K=64 makes
// the top-K too small to cover the rows under D, which must produce a loud failure rather
// than a truncated answer. Default is the real value, sized from the measured sf40 counts
// (307/3237/40413 image, ~400/4000/40000 text) with generous headroom.
int64_t search_k() {
  return std::strtoll(env_or("PEACOCK_TPCHV_K", "131072").c_str(), nullptr, 10);
}

}  // namespace

// ===========================================================================
// TPC-H+V q11 — national market value, restricted to a vector neighbourhood
//
//   SELECT ps_partkey, sum(ps_supplycost*ps_availqty) AS value
//   FROM partsupp, supplier, nation
//   WHERE ps_suppkey=s_suppkey AND s_nationkey=n_nationkey AND n_name='GERMANY'
//     AND ps_image_embedding <-> ${q} < ${D}
//   GROUP BY ps_partkey
//   HAVING sum(ps_supplycost*ps_availqty) > (same sum over ALL German rows) * 0.000002
//   ORDER BY value DESC
//
// NOTE the HAVING subquery deliberately does NOT carry the vector predicate: its threshold
// is computed over every German partsupp row. So the join is needed twice over — once
// unfiltered for the threshold, once vector-filtered for the groups — from ONE join.
//
// The range-vs-top-K gap, the distance space, the saturation guard and the count
// corroboration are all handled by vector_range_hits() above — see its comment block.
//
// REPRESENTATION, per column:
//   ps_partkey : integer, EXACT
//   value      : sum(ps_supplycost*ps_availqty), DECIMAL128 scale -4, EXACT (both inputs
//                are decimal(15,2); no float enters the aggregate)
//   row count under D : integer, EXACT — corroboration against DuckDB's own count
// No tolerance anywhere. The distances themselves are float32 on both sides, but they are
// only ever COMPARED to D, never asserted; what is asserted is which rows that comparison
// selects, which is exact as long as the boundary margin holds (measured: 1.14e-4 / 3.0e-5
// / 4.47e-6 for the three probes, against ~5e-7 expected float32 disagreement).
// ===========================================================================
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
  // Columns only. In particular the embedding column is loaded IN FULL for all 32M rows —
  // the distance predicate is NOT pushed into the reader, which is the constraint under
  // test. 32M x 96 x 4B = 12.3 GB of vectors before a single operator runs.
  // Scalar columns in one read — 32M fixed-width rows are nowhere near any limit.
  auto ps_in = read_cols(data_dir() + "/partsupp.parquet",
                         {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"});
  auto sup_in = read_cols(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  auto nat_in = read_cols(data_dir() + "/nation.parquet", {"n_nationkey", "n_name"});
  raft::resources handle;
  auto emb = load_embedding_column(data_dir() + "/partsupp.parquet", "ps_image_embedding", 96,
                                   raft::resource::get_cuda_stream(handle));
  // OVER the int32 list-child ceiling at sf40, so this column MUST come back in >1 batch.
  // If it ever comes back in one, either the scale factor shrank or cuDF changed its
  // representation — and the arithmetic in load_embedding_column needs revisiting rather
  // than silently no longer being exercised.
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

  auto dataset = raft::make_device_matrix_view<const float, int64_t>(emb.buf.data(), emb.rows, 96);
  cuvs::neighbors::brute_force::index_params ip;
  // L2SqrtExpanded == DuckDB's array_distance. See the header comment on distance space.
  ip.metric = cuvs::distance::DistanceType::L2SqrtExpanded;
  auto bf_index = cuvs::neighbors::brute_force::build(handle, ip, dataset);
  note_peak();
  std::fprintf(stderr, "[q11v] cuVS brute-force index built over %ld x 96 vectors\n",
               static_cast<long>(n_ps));

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

  // --- per-probe: cuVS search -> range filter -> intersect with the German rows ---
  for (auto const& probe : probes) {
    const int64_t K = search_k();
    ASSERT_LE(K, n_ps);
    auto hits = vector_range_hits(handle, bf_index, probe, 96, K,
                                  golden_dir() + "/duckdb_psimage_" + probe.id + ".count.csv",
                                  "q11v");
    note_peak();

    // --- intersect the vector hits with the German rows, then group ---
    // de column 3 holds each German row's ORIGINAL partsupp index, so the intersection is
    // an inner join of that index against the hit list.
    auto d_hits = to_device_hits(hits);
    auto [de_map, hit_map] = cudf::inner_join(
        cudf::table_view{{de->get_column(3).view()}}, cudf::table_view{{d_hits.view()}});
    auto sel = cudf::gather(cudf::table_view{{de->get_column(0).view(), de_value->view()}},
                            map_view(de_map));
    std::fprintf(stderr, "[q11v] %s: german AND under-D rows: %ld\n", probe.id.c_str(),
                 static_cast<long>(sel->num_rows()));

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
    auto sorted = cudf::gather(
        cudf::table_view{{kept->get_column(0).view(), kept->get_column(1).view()}},
        order->view());

    // CORROBORATION #2: the final grouped result, exact on both columns
    const std::vector<ColSpec> spec = {
        {"ps_partkey", Cmp::ExactInt},
        {"value", Cmp::ExactDecimal},
    };
    const auto golden = read_csv_golden(golden_dir() + "/duckdb_q11v_" + probe.id + ".csv");
    compare_table_to_golden(sorted->view(), golden, spec,
                            ("q11v/" + probe.id).c_str());
    std::fprintf(stderr, "[q11v] %s: %ld result rows matched\n", probe.id.c_str(),
                 static_cast<long>(sorted->num_rows()));
  }

  const auto t_done = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::fprintf(stderr, "[q11v] load %ld ms, setup %ld ms, search+verify %ld ms, total %ld ms\n",
               ms(t0, t_loaded), ms(t_loaded, t_setup), ms(t_setup, t_done), ms(t0, t_done));
}

// ===========================================================================
// SHARED SETUP FOR THE THREE p_text_embedding QUERIES
//
// q12v, q10v and q9v all restrict `part` by the SAME predicate — p_text_embedding within
// D of q — and differ only in what they join and aggregate afterwards. This helper does
// the common half: load part's key + embedding, build the cuVS index, run the range
// search, corroborate the count against DuckDB, and hand back the surviving p_partkey
// values as a column.
//
// THE ONE-READ SIDE OF THE CEILING, and the reason these three exist as a set:
// part at sf40 is 8,000,000 x 100 = 800,000,000 child elements, 0.37x the int32 list-child
// ceiling, so this column comes back in ONE read where partsupp's 96-wide column needs
// several. Same function, both regimes; the assertion below pins which one applies here so
// that a future scale factor crossing the boundary fails loudly instead of quietly
// changing the code path under test.
//
// NOTE the ORDER: the search runs over ALL 8M part rows. The distance predicate is not
// pushed into the reader and no part row is skipped before it — which is the constraint
// the whole exercise is about.
// ===========================================================================
namespace {

struct PartProbeResult {
  std::unique_ptr<cudf::table> partkeys;  // one column: the p_partkey values under D
  int64_t n_part;
  size_t n_hits;
};

PartProbeResult select_parts_by_text_embedding(raft::resources& handle, VecProbe const& probe,
                                               const char* tag) {
  const auto part_path = data_dir() + "/part.parquet";
  auto part_in = read_cols(part_path, {"p_partkey"});
  auto emb = load_embedding_column(part_path, "p_text_embedding", 100,
                                   raft::resource::get_cuda_stream(handle));
  // UNDER the ceiling at sf40 -> exactly one read. See load_embedding_column's comment.
  EXPECT_EQ(emb.batches, 1) << "p_text_embedding at sf40 must load in a single read";
  EXPECT_EQ(emb.rows, static_cast<int64_t>(part_in.tbl->num_rows()));

  auto dataset = raft::make_device_matrix_view<const float, int64_t>(emb.buf.data(), emb.rows, 100);
  cuvs::neighbors::brute_force::index_params ip;
  ip.metric = cuvs::distance::DistanceType::L2SqrtExpanded;  // == DuckDB array_distance
  auto index = cuvs::neighbors::brute_force::build(handle, ip, dataset);

  auto hits = vector_range_hits(handle, index, probe, 100, search_k(),
                                golden_dir() + "/duckdb_ptext_" + probe.id + ".count.csv", tag);
  // The hit list is a list of PART ROW INDICES; gather turns it into the p_partkey values
  // the joins downstream actually need.
  auto d_hits = to_device_hits(hits);
  auto keys = cudf::gather(cudf::table_view{{part_in.tbl->view().column(0)}}, d_hits.view());
  return PartProbeResult{std::move(keys), emb.rows, hits.size()};
}

}  // namespace

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
// WHAT THIS COVERS THAT NOTHING ELSE HERE DOES:
//  * COLUMN-TO-COLUMN DATE COMPARISON. Q1/Q3/Q6/Q8/q11v all compare a date to a LITERAL.
//    Here two of the three date predicates compare one TIMESTAMP_DAYS column to another.
//    That matters because o_orderdate's type id (12, TIMESTAMP_DAYS) has already produced
//    one silent-zero bug in this suite when a reader assumed an integer; a column-column
//    compare is the same trap in operator form.
//  * NO DECIMAL ANYWHERE. Both outputs are integer counters, so the whole result is
//    exactly comparable and no tolerance is even representable.
//  * A STRING GROUP KEY.
//
// JOIN ORDER — hand-chosen, no optimizer:
//   1. part restricted by the vector predicate            8,000,000 -> ~400 (txt_000)
//   2. lineitem restricted by shipmode + the three dates  240,000,000 -> ~2.4M
//   3. (2) |X| (1) on partkey                             -> ~120 rows
//   4. (3) |X| orders on orderkey                         -> ~120 rows
// The lineitem filter runs BEFORE the join because it is a local predicate on the largest
// table; the part join runs before orders because it is by far the more selective of the
// two.
//
// REPRESENTATION, per column:
//   l_shipmode       : string, EXACT
//   high_line_count  : cudf SUM over INT32 -> INT64; DuckDB sum(INTEGER) -> HUGEINT. Both
//                      print as plain integers and the values are ~1e2-1e4 here, nowhere
//                      near either width's limit. EXACT.
//   low_line_count   : same
// No tolerance anywhere.
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
  // Columns only, every row. No predicate reaches the reader.
  auto ord_in = read_cols(data_dir() + "/orders.parquet", {"o_orderkey", "o_orderpriority"});
  auto line_in = read_cols(data_dir() + "/lineitem.parquet",
                           {"l_orderkey", "l_partkey", "l_shipmode", "l_commitdate",
                            "l_receiptdate", "l_shipdate"});
  raft::resources handle;
  auto sel_parts = select_parts_by_text_embedding(handle, probe, "q12v");
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();
  std::fprintf(stderr, "[q12v] loaded orders %ld, lineitem %ld, part %ld (%zu under D)\n",
               static_cast<long>(ord_in.tbl->num_rows()),
               static_cast<long>(line_in.tbl->num_rows()),
               static_cast<long>(sel_parts.n_part), sel_parts.n_hits);
  ASSERT_GT(sel_parts.partkeys->num_rows(), 0) << "no part rows under D — test would be vacuous";

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto int32_t_ = cudf::data_type{cudf::type_id::INT32};
  auto lv = line_in.tbl->view();

  // l_shipmode IN ('MAIL','SHIP')
  auto mail = cudf::string_scalar(std::string("MAIL"));
  auto ship = cudf::string_scalar(std::string("SHIP"));
  auto is_mail = cudf::binary_operation(lv.column(2), mail, cudf::binary_operator::EQUAL, boolean);
  auto is_ship = cudf::binary_operation(lv.column(2), ship, cudf::binary_operator::EQUAL, boolean);
  auto mode_ok = cudf::binary_operation(is_mail->view(), is_ship->view(),
                                        cudf::binary_operator::LOGICAL_OR, boolean);

  // THE COLUMN-TO-COLUMN COMPARISONS: both operands are TIMESTAMP_DAYS columns, not
  // scalars. Nothing is cast; the types stay what the reader produced.
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
  note_peak();
  std::fprintf(stderr, "[q12v] lineitem after local filters: %ld\n",
               static_cast<long>(line_f->num_rows()));
  ASSERT_GT(line_f->num_rows(), 0);

  // join 1: lineitem.l_partkey = (parts under D).p_partkey
  auto [l_map, p_map] = cudf::inner_join(
      cudf::table_view{{line_f->get_column(1).view()}},
      cudf::table_view{{sel_parts.partkeys->get_column(0).view()}});
  auto lp = cudf::gather(cudf::table_view{{line_f->get_column(0).view(),    // l_orderkey
                                           line_f->get_column(2).view()}},  // l_shipmode
                         map_view(l_map));
  note_peak();
  std::fprintf(stderr, "[q12v] lineitem |X| part -> %ld rows\n", static_cast<long>(lp->num_rows()));
  ASSERT_GT(lp->num_rows(), 0) << "vector predicate and filters left no rows to join";

  // join 2: |X| orders on orderkey
  auto [lp_map, o_map] = cudf::inner_join(cudf::table_view{{lp->get_column(0).view()}},
                                          cudf::table_view{{ord_in.tbl->view().column(0)}});
  auto mode_col = cudf::gather(cudf::table_view{{lp->get_column(1).view()}}, map_view(lp_map));
  auto prio_col = cudf::gather(cudf::table_view{{ord_in.tbl->view().column(1)}}, map_view(o_map));
  note_peak();
  std::fprintf(stderr, "[q12v] |X| orders -> %ld rows\n", static_cast<long>(mode_col->num_rows()));
  ASSERT_GT(mode_col->num_rows(), 0);

  // the two CASE expressions, as boolean masks cast to counters
  auto urgent = cudf::string_scalar(std::string("1-URGENT"));
  auto high = cudf::string_scalar(std::string("2-HIGH"));
  auto is_urgent = cudf::binary_operation(prio_col->get_column(0).view(), urgent,
                                          cudf::binary_operator::EQUAL, boolean);
  auto is_high = cudf::binary_operation(prio_col->get_column(0).view(), high,
                                        cudf::binary_operator::EQUAL, boolean);
  auto is_hi = cudf::binary_operation(is_urgent->view(), is_high->view(),
                                      cudf::binary_operator::LOGICAL_OR, boolean);
  // low is the exact complement: DuckDB writes it as <> AND <>, which is NOT is_hi by De
  // Morgan. o_orderpriority is NOT NULL in TPC-H so the two forms agree; computing it as
  // a negation rather than restating the comparisons keeps them from drifting apart.
  auto is_lo = cudf::unary_operation(is_hi->view(), cudf::unary_operator::NOT);
  auto hi_i = cudf::cast(is_hi->view(), int32_t_);
  auto lo_i = cudf::cast(is_lo->view(), int32_t_);

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
  note_peak();

  // ORDER BY l_shipmode — the group key, unique per row, so this is a TOTAL order
  auto order = cudf::sorted_order(cudf::table_view{{gk->view().column(0)}},
                                  {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  auto sorted = cudf::gather(
      cudf::table_view{{gk->view().column(0), hi_sum->view(), lo_sum->view()}}, order->view());
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
// WHAT THIS COVERS THAT NOTHING ELSE HERE DOES:
//  * GROUP BY SEVEN COLUMNS, FIVE OF THEM STRINGS, one of which (c_comment) runs to ~72
//    characters. Every other groupby in this suite keys on integers, decimals or a single
//    short string, so the variable-width hash path is otherwise untested.
//  * A TOP-N WITH A WIDE STRING PAYLOAD: the sort gathers six string columns along with
//    the decimals.
//
// TIE-BREAK: TPC-H Q10 orders by revenue alone, which is not a total order, so LIMIT 20
// could return different rows run to run. c_custkey is appended — unique per group — on
// BOTH sides. Without it this test would be flaky rather than wrong, which is worse.
//
// JOIN ORDER — hand-chosen:
//   1. part restricted by the vector predicate       8,000,000 -> ~4,000 (txt_017)
//   2. lineitem restricted by l_returnflag='R'       240,000,000 -> ~59M
//   3. (2) |X| (1) on partkey                        -> ~30k
//   4. orders restricted to the 3-month window       60,000,000 -> ~1.8M
//   5. (3) |X| (4) on orderkey                       -> ~900
//   6. |X| customer |X| nation                       -> ~900
// Step 3 before step 5 for the same reason as Q8: the part join is the only thing that can
// cut lineitem down, and doing it first keeps the orders join small.
//
// REPRESENTATION, per column:
//   c_custkey  : integer, EXACT
//   c_name     : string, EXACT
//   revenue    : sum(l_extendedprice*(1-l_discount)), DECIMAL128 scale -4, EXACT
//   c_acctbal  : DECIMAL, EXACT — carried through the groupby as a KEY, never re-derived
//   n_name, c_address, c_phone, c_comment : strings, EXACT
// No tolerance anywhere. c_address and c_comment contain commas, so DuckDB quotes those
// fields in the golden — read_csv_golden parses RFC4180 quoting for exactly this reason.
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
  raft::resources handle;
  auto sel_parts = select_parts_by_text_embedding(handle, probe, "q10v");
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();
  std::fprintf(stderr, "[q10v] loaded customer %ld, orders %ld, lineitem %ld, part %ld (%zu under D)\n",
               static_cast<long>(cust_in.tbl->num_rows()),
               static_cast<long>(ord_in.tbl->num_rows()),
               static_cast<long>(line_in.tbl->num_rows()),
               static_cast<long>(sel_parts.n_part), sel_parts.n_hits);
  ASSERT_GT(sel_parts.partkeys->num_rows(), 0) << "no part rows under D — test would be vacuous";

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};
  auto cv = cust_in.tbl->view();
  auto ov = ord_in.tbl->view();
  auto lv = line_in.tbl->view();

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
  note_peak();
  std::fprintf(stderr, "[q10v] after local filters: lineitem %ld, orders %ld\n",
               static_cast<long>(line_f->num_rows()), static_cast<long>(ord_f->num_rows()));
  ASSERT_GT(line_f->num_rows(), 0);
  ASSERT_GT(ord_f->num_rows(), 0);

  // join 1: lineitem.l_partkey = (parts under D).p_partkey
  auto [l_map, p_map] = cudf::inner_join(
      cudf::table_view{{line_f->get_column(1).view()}},
      cudf::table_view{{sel_parts.partkeys->get_column(0).view()}});
  auto lp = cudf::gather(cudf::table_view{{line_f->get_column(0).view(),    // l_orderkey
                                           line_f->get_column(2).view(),    // l_extendedprice
                                           line_f->get_column(3).view()}},  // l_discount
                         map_view(l_map));
  note_peak();
  std::fprintf(stderr, "[q10v] lineitem |X| part -> %ld rows\n", static_cast<long>(lp->num_rows()));

  // join 2: |X| orders on orderkey
  auto [lp_map, o_map] = cudf::inner_join(cudf::table_view{{lp->get_column(0).view()}},
                                          cudf::table_view{{ord_f->get_column(0).view()}});
  auto lo_l = cudf::gather(cudf::table_view{{lp->get_column(1).view(), lp->get_column(2).view()}},
                           map_view(lp_map));
  auto lo_o = cudf::gather(cudf::table_view{{ord_f->get_column(1).view()}},  // o_custkey
                           map_view(o_map));
  note_peak();
  std::fprintf(stderr, "[q10v] |X| orders -> %ld rows\n", static_cast<long>(lo_o->num_rows()));
  ASSERT_GT(lo_o->num_rows(), 0) << "vector predicate and filters left no rows to join";

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
  note_peak();

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
  std::fprintf(stderr, "[q10v] |X| customer |X| nation -> %ld rows\n",
               static_cast<long>(cust_j->num_rows()));
  ASSERT_GT(cust_j->num_rows(), 0);

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
  note_peak();
  std::fprintf(stderr, "[q10v] groups: %ld\n", static_cast<long>(gk->num_rows()));
  ASSERT_GE(gk->num_rows(), 20) << "fewer than 20 groups — LIMIT 20 would not be exercised";

  // ORDER BY revenue DESC, c_custkey ASC  (total order — see the header)
  auto gkv = gk->view();
  auto order = cudf::sorted_order(cudf::table_view{{rev_col->view(), gkv.column(0)}},
                                  {cudf::order::DESCENDING, cudf::order::ASCENDING},
                                  {cudf::null_order::AFTER, cudf::null_order::AFTER});
  // output column order matches the golden: custkey, name, revenue, acctbal, nation,
  // address, phone, comment
  auto sorted = cudf::gather(cudf::table_view{{gkv.column(0), gkv.column(1), rev_col->view(),
                                               gkv.column(2), gkv.column(4), gkv.column(5),
                                               gkv.column(3), gkv.column(6)}},
                             order->view());
  auto top = cudf::slice(sorted->view(), {0, 20})[0];
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
// WHAT THIS COVERS THAT NOTHING ELSE HERE DOES:
//  * A COMPOSITE TWO-COLUMN JOIN. partsupp is joined to lineitem on BOTH ps_partkey and
//    ps_suppkey. Every other join in this suite is single-key. A multi-key inner_join is
//    the shape most likely to be quietly wrong — pair up on one column and the row counts
//    still look plausible while every supplycost is drawn from the wrong supplier.
//  * A SUBSTRING PREDICATE. p_name LIKE '%green%' is strings::contains, not a prefix or an
//    equality; nothing else here does substring matching.
//  * A DECIMAL SUBTRACTION of two products, rather than a sum of one. Both products land
//    at scale -4 and the difference stays there, so the arithmetic is still exact — but it
//    is the first place the scales of two independently derived decimals have to agree.
//  * A TWO-KEY GROUP BY with a DESCENDING secondary sort.
//
// JOIN ORDER — hand-chosen:
//   1. part restricted by the vector predicate AND '%green%'   8,000,000 -> ~2,100
//   2. (1) |X| lineitem on partkey                             240,000,000 -> ~63k
//   3. (2) |X| partsupp on (partkey, suppkey)   <- THE COMPOSITE JOIN     -> ~63k
//   4. |X| orders on orderkey, |X| supplier |X| nation on suppkey/nationkey
// part is filtered first because it carries both selective predicates; everything after it
// operates on tens of thousands of rows rather than hundreds of millions.
//
// REPRESENTATION, per column:
//   nation     : string, EXACT
//   o_year     : cudf extract_datetime_component(YEAR) -> INT16; DuckDB extract() -> BIGINT.
//                Compared as integers, EXACT (int64_at reads either width).
//   sum_profit : DECIMAL128 scale -4, EXACT — no float enters the expression at any point
// No tolerance anywhere.
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
  auto part_name_in = read_cols(data_dir() + "/part.parquet", {"p_partkey", "p_name"});
  auto sup_in = read_cols(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  auto line_in = read_cols(data_dir() + "/lineitem.parquet",
                           {"l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice",
                            "l_discount", "l_quantity"});
  auto ps_in = read_cols(data_dir() + "/partsupp.parquet",
                         {"ps_partkey", "ps_suppkey", "ps_supplycost"});
  auto ord_in = read_cols(data_dir() + "/orders.parquet", {"o_orderkey", "o_orderdate"});
  auto nat_in = read_cols(data_dir() + "/nation.parquet", {"n_nationkey", "n_name"});
  raft::resources handle;
  auto sel_parts = select_parts_by_text_embedding(handle, probe, "q9v");
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();
  std::fprintf(stderr,
               "[q9v] loaded part %ld, supplier %ld, lineitem %ld, partsupp %ld, orders %ld"
               " (%zu parts under D)\n",
               static_cast<long>(part_name_in.tbl->num_rows()),
               static_cast<long>(sup_in.tbl->num_rows()),
               static_cast<long>(line_in.tbl->num_rows()),
               static_cast<long>(ps_in.tbl->num_rows()),
               static_cast<long>(ord_in.tbl->num_rows()), sel_parts.n_hits);
  ASSERT_GT(sel_parts.partkeys->num_rows(), 0) << "no part rows under D — test would be vacuous";

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};
  auto lv = line_in.tbl->view();

  // p_name LIKE '%green%' — a SUBSTRING match over the whole part table, then intersected
  // with the vector hits. Both predicates are on part, so which runs first is arbitrary;
  // running contains() over 8M short strings is cheap next to the 8M x 100 distance scan
  // that already happened.
  auto green = cudf::string_scalar(std::string("green"));
  auto green_mask = cudf::strings::contains(
      cudf::strings_column_view(part_name_in.tbl->view().column(1)), green);
  auto green_parts = cudf::apply_boolean_mask(
      cudf::table_view{{part_name_in.tbl->view().column(0)}}, green_mask->view());
  note_peak();
  std::fprintf(stderr, "[q9v] parts matching '%%green%%': %ld\n",
               static_cast<long>(green_parts->num_rows()));
  ASSERT_GT(green_parts->num_rows(), 0);

  auto [g_map, v_map] = cudf::inner_join(
      cudf::table_view{{green_parts->get_column(0).view()}},
      cudf::table_view{{sel_parts.partkeys->get_column(0).view()}});
  auto part_sel = cudf::gather(cudf::table_view{{green_parts->get_column(0).view()}},
                               map_view(g_map));
  std::fprintf(stderr, "[q9v] parts under D AND '%%green%%': %ld\n",
               static_cast<long>(part_sel->num_rows()));
  ASSERT_GT(part_sel->num_rows(), 0) << "the two part predicates have no overlap";

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
  std::fprintf(stderr, "[q9v] part |X| lineitem -> %ld rows\n", static_cast<long>(li->num_rows()));
  ASSERT_GT(li->num_rows(), 0);

  // join 2: THE COMPOSITE JOIN. Two key columns on each side, matched positionally:
  // (l_partkey, l_suppkey) against (ps_partkey, ps_suppkey). Joining on partkey alone would
  // multiply every lineitem row by the four suppliers that stock the part and silently
  // charge it the wrong supplycost — the row count would grow 4x, which is why the
  // assertion below pins it against the left input rather than merely checking >0.
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
  std::fprintf(stderr, "[q9v] |X| partsupp on (partkey,suppkey) -> %ld rows\n",
               static_cast<long>(li_j->num_rows()));
  // (partkey, suppkey) is UNIQUE in partsupp, so the composite join must be row-preserving
  // on the lineitem side: every lineitem row has exactly one matching partsupp row.
  // A single-key join here would return ~4x this. That is the whole point of the test.
  ASSERT_EQ(li_j->num_rows(), li->num_rows())
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
  std::fprintf(stderr, "[q9v] |X| orders -> %ld rows\n", static_cast<long>(li_o->num_rows()));
  ASSERT_GT(li_o->num_rows(), 0);

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
  std::fprintf(stderr, "[q9v] |X| supplier |X| nation -> %ld rows\n",
               static_cast<long>(vals_n->num_rows()));
  ASSERT_GT(vals_n->num_rows(), 0);

  // o_year — extract_datetime_component, not the 25.02-only extract_year (deleted in
  // 26.02). Returns INT16.
  auto o_year = cudf::datetime::extract_datetime_component(
      date_n->get_column(0).view(), cudf::datetime::datetime_component::YEAR);

  // amount = l_extendedprice*(1-l_discount) - ps_supplycost*l_quantity, all exact decimal:
  // both products are (scale -2) x (scale -2) -> scale -4, and the difference stays at -4.
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
  note_peak();

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
  note_peak();
  std::fprintf(stderr, "[q9v] groups: %ld\n", static_cast<long>(gk->num_rows()));

  // ORDER BY nation ASC, o_year DESC — together the group key, so this is a TOTAL order
  auto gkv = gk->view();
  auto order = cudf::sorted_order(cudf::table_view{{gkv.column(0), gkv.column(1)}},
                                  {cudf::order::ASCENDING, cudf::order::DESCENDING},
                                  {cudf::null_order::AFTER, cudf::null_order::AFTER});
  auto sorted = cudf::gather(
      cudf::table_view{{gkv.column(0), gkv.column(1), profit->view()}}, order->view());
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
}

// Same entry point as the other gtest binaries here (the conda cudf ships no gtest_main).
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
