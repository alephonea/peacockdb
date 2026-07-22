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
#include <cudf/stream_compaction.hpp>
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
// RANGE-VS-TOP-K, the semantic gap I had to bridge explicitly:
// cuVS 25.02 brute_force exposes only a TOP-K search; there is no range/radius API (I read
// the header — no range, radius or epsilon entry point exists). The query needs a RANGE:
// every row with distance < D. Resolution: search top-K with K chosen well above the
// number of rows that can fall under D, then filter by D. That is exact ONLY IF the top-K
// did not saturate, so the saturation guard below is load-bearing, not decorative: if the
// K'th returned neighbour is still inside D, the answer was truncated and the test FAILS
// rather than silently returning a subset. That is the silent-wrong-answer shape this
// suite keeps finding, so it gets an explicit check.
//
// DISTANCE SPACE: cuVS defaults to L2Expanded, which returns SQUARED L2, whereas DuckDB's
// array_distance returns the root. Comparing those directly would be a silent factor-level
// error. Rather than square D or sqrt the results by hand, the index is built with
// L2SqrtExpanded, which returns true L2 — the same quantity DuckDB computes, so D is used
// as-is on both sides with no conversion to get wrong.
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
  auto read_cols = [](const std::string& path, std::vector<std::string> cols) {
    auto o = cudf::io::parquet_reader_options::builder(
                 cudf::io::source_info{std::vector<std::string>{path}})
                 .columns(std::move(cols))
                 .build();
    return cudf::io::read_parquet(o);
  };
  // Scalar columns in one read — 32M fixed-width rows are nowhere near any limit.
  auto ps_in = read_cols(data_dir() + "/partsupp.parquet",
                         {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"});
  auto sup_in = read_cols(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  auto nat_in = read_cols(data_dir() + "/nation.parquet", {"n_nationkey", "n_name"});
  const auto t_loaded = std::chrono::steady_clock::now();
  note_peak();
  const auto n_ps = ps_in.tbl->num_rows();
  std::fprintf(stderr, "[q11v] loaded partsupp %ld, supplier %ld, nation %ld\n",
               static_cast<long>(n_ps), static_cast<long>(sup_in.tbl->num_rows()),
               static_cast<long>(nat_in.tbl->num_rows()));

  // ---------------- PHASE 2: EXECUTE ----------------
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};
  auto map_view = [](std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& m) {
    return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                             static_cast<cudf::size_type>(m->size()), m->data(), nullptr, 0);
  };

  // ===========================================================================
  // THE EMBEDDING COLUMN CANNOT BE READ IN ONE CALL AT sf40 — READ THIS BEFORE
  // "SIMPLIFYING" THE LOOP BELOW BACK INTO A SINGLE read_parquet.
  //
  // A cuDF LIST column stores its values in ONE contiguous child column whose length is a
  // cudf::size_type, i.e. INT32. That caps the child at 2,147,483,647 elements:
  //     96-wide image embeddings  -> max 22,369,621 rows  (~tpch.sf27)
  //     100-wide text embeddings  -> max 21,474,836 rows  (~tpch.sf26)
  // partsupp at sf40 is 32,000,000 rows x 96 = 3,072,000,000 elements — 1.43x over the
  // ceiling. A single read fails in about one second with
  //     std::bad_alloc: out_of_memory: cudaErrorMemoryAllocation
  // WHICH IS A LIE: the device had 137 GB free and the column needs 12.3 GB. The message is
  // the symptom of an overflowed size computation, not a real allocation shortfall.
  // Verified three ways: the arithmetic above; an isolated repro in plain cuDF Python
  // (cudf.read_parquet(..., columns=['ps_image_embedding']) alone fails); and a bisect that
  // reads OK at 22,241,280 rows (0.99x the limit) and fails at 23,592,960 (1.05x).
  //
  // So the column is read in ROW-GROUP BATCHES and reassembled into one device buffer.
  // WHAT THIS IS NOT:
  //   NOT a memory optimisation — there is 11x more device memory than needed.
  //   NOT predicate pushdown — every row group is read, no rows are skipped or filtered,
  //     and the distance predicate still runs in phase 2. The reader batches I/O; it does
  //     not select data.
  // The load-then-execute boundary is unchanged: every row is resident before any operator
  // runs. cuVS/raft use INT64 extents (probed directly: build + search over a 32M x 96
  // matrix succeed), which is why the reassembled buffer is representable even though a
  // cuDF list column of the same data is not.
  // ===========================================================================
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  const auto ps_path = data_dir() + "/partsupp.parquet";
  auto pq_meta = cudf::io::read_parquet_metadata(
      cudf::io::source_info{std::vector<std::string>{ps_path}});
  const int64_t n_ps_meta = pq_meta.num_rows();
  const int n_rg = pq_meta.num_rowgroups();
  ASSERT_EQ(n_ps_meta, static_cast<int64_t>(n_ps)) << "row count disagrees with the footer";

  rmm::device_uvector<float> emb_buf(static_cast<size_t>(n_ps_meta) * 96, stream);
  // batch cap well under the int32 ceiling so no single batch can trip it
  constexpr int64_t kMaxRowsPerBatch = 8'000'000;
  int64_t copied_rows = 0;
  int batches = 0;
  for (int rg = 0; rg < n_rg;) {
    std::vector<cudf::size_type> group;
    int64_t batch_rows = 0;
    // grow the batch until adding another row group would exceed the cap
    while (rg < n_rg && batch_rows < kMaxRowsPerBatch) {
      group.push_back(rg);
      // exact per-group row counts are not exposed, so bound by the cap using the file
      // average; the assertion after the loop is what actually guarantees completeness
      batch_rows += (n_ps_meta + n_rg - 1) / n_rg;
      ++rg;
    }
    auto o = cudf::io::parquet_reader_options::builder(
                 cudf::io::source_info{std::vector<std::string>{ps_path}})
                 .columns({"ps_image_embedding"})
                 .build();
    o.set_row_groups({group});
    auto part = cudf::io::read_parquet(o);
    auto lv = cudf::lists_column_view(part.tbl->view().column(0));
    auto child = lv.child();
    ASSERT_EQ(child.type().id(), cudf::type_id::FLOAT32) << "embedding child must be float32";
    ASSERT_EQ(static_cast<int64_t>(child.size()),
              static_cast<int64_t>(part.tbl->num_rows()) * 96)
        << "batch is not a fixed 96-wide list";
    CUDF_CUDA_TRY(cudaMemcpyAsync(emb_buf.data() + copied_rows * 96, child.data<float>(),
                                  static_cast<size_t>(child.size()) * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream.value()));
    cudaStreamSynchronize(stream.value());
    copied_rows += part.tbl->num_rows();
    ++batches;
    note_peak();
  }
  // COMPLETENESS: a dropped or double-counted batch would silently shorten the dataset and
  // change every distance result. Assert the reassembly, do not assume it.
  ASSERT_EQ(copied_rows, n_ps_meta)
      << "chunked embedding load reassembled " << copied_rows << " rows but the file has "
      << n_ps_meta << " — a batch was dropped or double-counted";
  std::fprintf(stderr, "[q11v] embeddings loaded in %d batches, %ld rows reassembled\n",
               batches, static_cast<long>(copied_rows));

  auto dataset = raft::make_device_matrix_view<const float, int64_t>(
      emb_buf.data(), static_cast<int64_t>(n_ps_meta), 96);
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
    // K must exceed the number of rows that can fall under D. Sized from the measured
    // counts at sf40 (307 / 3237 / 40413 for k=10/100/1000) with generous headroom; the
    // guard below turns a bad guess into a loud failure rather than a truncated answer.
    // K is overridable ONLY so the saturation guard below can be exercised: a guard that
    // has never fired is not a guard. Setting PEACOCK_TPCHV_K=64 makes the top-K too small
    // to cover the rows under D, which must produce a loud failure rather than a truncated
    // answer. Default is the real value.
    const int64_t K = std::strtoll(env_or("PEACOCK_TPCHV_K", "131072").c_str(), nullptr, 10);
    ASSERT_LE(K, n_ps);
    auto d_q = raft::make_device_matrix<float, int64_t>(handle, 1, 96);
    raft::copy(d_q.data_handle(), probe.q.data(), 96, raft::resource::get_cuda_stream(handle));
    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(handle, 1, K);
    auto distances = raft::make_device_matrix<float, int64_t>(handle, 1, K);
    cuvs::neighbors::brute_force::search(
        handle, cuvs::neighbors::brute_force::search_params{}, bf_index,
        raft::make_const_mdspan(d_q.view()), neighbors.view(), distances.view());
    raft::resource::sync_stream(handle);
    note_peak();

    // pull the hit list back and range-filter it on the host: K is 131072, so this is
    // trivial next to the 32M-row search that produced it
    std::vector<int64_t> h_nb(K);
    std::vector<float> h_di(K);
    raft::copy(h_nb.data(), neighbors.data_handle(), K, raft::resource::get_cuda_stream(handle));
    raft::copy(h_di.data(), distances.data_handle(), K, raft::resource::get_cuda_stream(handle));
    raft::resource::sync_stream(handle);

    // SATURATION GUARD — see the header comment. If the furthest neighbour cuVS returned is
    // still inside D, then rows beyond K also qualify and the range answer is TRUNCATED.
    ASSERT_GE(static_cast<double>(h_di[K - 1]), probe.D)
        << probe.id << ": top-K SATURATED (K=" << K << ", furthest distance " << h_di[K - 1]
        << " < D=" << probe.D << ") — the range answer would be silently truncated. "
        << "Raise K.";

    std::vector<int32_t> hits;
    for (int64_t i = 0; i < K; ++i) {
      if (static_cast<double>(h_di[i]) < probe.D) hits.push_back(static_cast<int32_t>(h_nb[i]));
    }
    std::fprintf(stderr, "[q11v] %s: D=%.17g -> %zu rows under D (furthest returned %.9g)\n",
                 probe.id.c_str(), probe.D, hits.size(), h_di[K - 1]);

    // CORROBORATION #1: the row count under D must equal DuckDB's own count, computed
    // independently over the same parquet. This pins the row SET the distance predicate
    // selects, before any join or aggregation can mask a discrepancy.
    const auto count_golden = golden_dir() + "/duckdb_q11v_" + probe.id + ".count.csv";
    ASSERT_TRUE(file_exists(count_golden)) << "missing " << count_golden;
    const int64_t want_count =
        std::strtoll(read_single_value_golden(count_golden).c_str(), nullptr, 10);
    ASSERT_EQ(static_cast<int64_t>(hits.size()), want_count)
        << probe.id << ": cuVS found " << hits.size() << " rows under D but DuckDB found "
        << want_count << ". A one-row difference here means cuVS and DuckDB landed on "
        << "opposite sides of the distance boundary — a real numerical divergence, NOT "
        << "something to absorb with a tolerance.";

    // --- intersect the vector hits with the German rows, then group ---
    // de column 3 holds each German row's ORIGINAL partsupp index, so the intersection is
    // an inner join of that index against the hit list.
    // host hit list -> device int32 column (no cudf test-util helpers in the conda build)
    rmm::device_uvector<int32_t> d_hits(hits.size(), cudf::get_default_stream());
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_hits.data(), hits.data(), hits.size() * sizeof(int32_t),
                                  cudaMemcpyHostToDevice, cudf::get_default_stream().value()));
    cudf::get_default_stream().synchronize();
    auto hits_col = cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                                      static_cast<cudf::size_type>(d_hits.size()),
                                      d_hits.data(), nullptr, 0);
    auto [de_map, hit_map] = cudf::inner_join(
        cudf::table_view{{de->get_column(3).view()}}, cudf::table_view{{hits_col}});
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

// Same entry point as the other gtest binaries here (the conda cudf ships no gtest_main).
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
