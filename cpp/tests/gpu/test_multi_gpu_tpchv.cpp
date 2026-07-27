// test_multi_gpu_tpchv.cpp — the SAME TPC-H+V vector queries as test_tpchv.cpp (q9v/q10v/q11v/
// q12v), but with the cuVS brute-force SEARCH SHARDED across all visible GPUs, checked against
// the SAME committed DuckDB goldens BYTE-FOR-BYTE (via tpch_golden.hpp). Milestone 3.
//
// THE PARALLEL WIN IS THE SEARCH. Each vector query restricts a table (partsupp.ps_image_embedding
// for q11v, part.p_text_embedding for q12v/q10v/q9v) to the rows within distance D of a probe.
// That distance scan over millions of embeddings is the expensive, embarrassingly parallel part;
// the relational joins/group-by wrapped around the O(1e5) hit list are small. So we SHARD the
// embedding table across the GPUs on WHOLE parquet row-group boundaries (exactly as the fact
// tables are partitioned in M1/M2), build a cuVS brute_force index over each shard, search every
// shard for the probe's top-K, and MERGE the shards to the EXACT global result. The relational
// tail then runs on GPU0 and is verbatim the single-GPU plan from test_tpchv.cpp.
//
// WHY SHARDING IS EXACT (the correctness crux):
//   A globally-top-K point is necessarily top-K within its OWN shard, so the union of the
//   per-shard top-K lists CONTAINS the true global top-K. Request K from every shard, gather the
//   G*K (global_key, distance) candidates, sort by distance, take the global top-K — exact. The
//   SATURATION GUARD generalizes to the GLOBAL K-th distance: if it is < D, more than K rows
//   globally fall under D and the range answer is truncated -> FAIL loudly (EXPECT_GE), same
//   spirit as the single-GPU guard. Then range-filter distance < D -> the global hit keys, exactly
//   the set the single-GPU vector_range_hits returns. Because the shards are contiguous row-group
//   spans in natural parquet order, a shard-local row index i maps to the GLOBAL row index
//   row_offset[g] + i — the very row index the single-GPU search returns — so the relational tail
//   (which gathers a key column at those indices on GPU0) is unchanged.
//
// WHY THE int32 LIST-CHILD CEILING MOTIVATES THIS: a cuDF LIST column's child is capped at 2^31
// elements, so partsupp.ps_image_embedding (32M x 96 = 3.07e9) EXCEEDS it single-GPU and must be
// chunk-loaded (see test_tpchv.cpp). Sharding by row groups keeps each shard's child under the
// cap (at G=2, 16M x 96 = 1.5e9 fits in one read); the shard loader still batches WITHIN a span
// so the G=1 baseline (whole table on one GPU) stays correct too.
//
// MANUAL sharding — NOT cuvs::neighbors::mg (SNMG): the cross-GPU movement stays explicit and
// consistent with the rest of the suite (WorkerPool, gather_here, hash_shuffle); mg would hide it.
// cuVS/raft allocate from the per-device RMM POOL (the current-device resource on each worker IS
// the pool); the q11v test asserts this empirically (free VRAM unchanged across a search).
//
// LINKING cuVS — see the note in test_tpchv.cpp: libcuvs references rmm symbols defined in
// libcudf, so cudf must link before cuvs (the CMake target orders them).
//
// SHARED PROCESS-WIDE WorkerPool: one WorkerPool for the whole binary, owned by MultiGpuEnvironment
// (now in multi_gpu.hpp, shared with test_multi_gpu_tpch.cpp). PEACOCK_BENCHMARK times the execute
// (the sharded search + host merge + relational tail), all-device-synced, 2nd-min of 6.

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
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
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#if __has_include(<cudf/join/join.hpp>)
#  include <cudf/join/join.hpp>  // cudf >= 26.02
#else
#  include <cudf/join.hpp>
#endif

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

#include "multi_gpu.hpp"
#include "tpch_golden.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace peacock_test;   // TpchSf40 fixture, ColSpec/compare_table_to_golden, Decimal, ...
using namespace peacock_mgpu;   // WorkerPool, shared_pool, partition_row_groups, benchmark_mgpu, ...

namespace {

// ===========================================================================
// Probe loading — identical to test_tpchv.cpp: one (q, D, k) probe resolved from the COMMITTED
// query_params.jsonl, never hardcoded, so the golden and the GPU side cannot drift apart.
// ===========================================================================
struct VecProbe {
  std::string id;
  int k = 0;
  double D = 0.0;
  std::vector<float> q;
};

std::vector<VecProbe> load_probes(const std::string& path, std::vector<std::string> const& want) {
  std::vector<VecProbe> out;
  std::ifstream f(path);
  std::string line;
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

// K for the top-K-then-filter search — requested from EACH shard. Overridable ONLY so the
// saturation guard can be exercised (a guard that never fires is not a guard): PEACOCK_TPCHV_K=64
// makes the per-shard top-K too small to cover the rows under D and must fail loudly. Default is
// the real value used single-GPU, sized from the measured sf40 counts with generous headroom.
int64_t search_k() {
  return std::strtoll(env_or("PEACOCK_TPCHV_K", "131072").c_str(), nullptr, 10);
}

// ===========================================================================
// STAGE PROFILER (measurement only; env-gated by PEACOCK_PROFILE) — a per-stage ms breakdown +
// per-stage row counts for the EXECUTE path, to settle where the serial GPU0 remainder goes.
// When `on` is false EVERY method is a no-op, so the committed correctness/benchmark path is
// byte-for-byte unchanged (the ticks are threaded through execute but cost nothing off-profile).
// Isolating a stage requires an all-device sync at its boundary; those syncs SERIALIZE work that
// may otherwise overlap, so the summed breakdown is an UPPER bound and differs from the clean
// benchmark total — the report prints the sum and the caller compares it to the 2nd-min benchmark.
// tick() accumulates one ms sample per stage per iteration; report() takes the 2nd-min per stage.
// ===========================================================================
inline bool profile_enabled() {
  const char* v = std::getenv("PEACOCK_PROFILE");
  return v && *v && std::string(v) != "0";
}

struct StageProfiler {
  bool on = false;
  int num_gpus = 1;
  std::vector<std::string> order;                        // stage names, first-seen order
  std::map<std::string, std::vector<double>> times;      // stage -> per-iteration ms samples
  std::vector<std::pair<std::string, long>> rows;        // stage row counts, insertion order
  std::chrono::steady_clock::time_point last;

  void sync_all() {
    for (int g = 0; g < num_gpus; ++g) {
      cudaSetDevice(g);
      cudaDeviceSynchronize();
    }
    cudaSetDevice(0);
  }
  void begin() {
    if (!on) return;
    sync_all();
    last = std::chrono::steady_clock::now();
  }
  void tick(const char* name) {
    if (!on) return;
    sync_all();
    auto now = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - last).count();
    if (times.find(name) == times.end())
      order.push_back(name);
    times[name].push_back(ms);
    last = now;
  }
  void rowcount(const char* name, long n) {
    if (!on) return;
    for (auto& pr : rows)
      if (pr.first == name) { pr.second = n; return; }
    rows.emplace_back(name, n);
  }
  void report(const char* tag, int G) {
    if (!on) return;
    const size_t iters = times.empty() ? 0 : times.begin()->second.size();
    std::fprintf(stderr, "[profile] %s G=%d — per-stage 2nd-min ms (%zu iters):\n", tag, G, iters);
    double total = 0;
    for (auto const& name : order) {
      auto v = times[name];
      std::sort(v.begin(), v.end());
      const double m = v.size() > 1 ? v[1] : (v.empty() ? 0.0 : v[0]);
      total += m;
      std::fprintf(stderr, "[profile]   %-18s %9.3f ms\n", name.c_str(), m);
    }
    std::fprintf(stderr, "[profile]   %-18s %9.3f ms  (sum; boundary syncs remove overlap)\n",
                 "= stage-sum", total);
    for (auto const& pr : rows)
      std::fprintf(stderr, "[profile]   rows %-13s %ld\n", pr.first.c_str(), pr.second);
  }
};

// ===========================================================================
// SHARD EMBEDDING LOADER — read ONLY a row-group span's embedding column into a resident
// row-major float matrix on the CURRENT device. Batches WITHIN the span so no single read's list
// child exceeds the int32 ceiling (the G=1 baseline reads the whole over-ceiling column and must
// still chunk; at G>=2 each span fits in one read). Exact shard row count comes from summing the
// batches — parquet does not expose per-row-group counts, so the buffer is assembled, not
// presized from metadata.
// ===========================================================================
struct EmbShard {
  rmm::device_uvector<float> buf;  // rows*dim, row-major, on the current device
  int64_t rows;
  int batches;
};

EmbShard load_embedding_shard(const std::string& path, const std::string& col, int dim,
                              std::vector<cudf::size_type> const& span,
                              rmm::cuda_stream_view stream) {
  auto meta = cudf::io::read_parquet_metadata(
      cudf::io::source_info{std::vector<std::string>{path}});
  const int64_t n_rows_file = meta.num_rows();
  const int n_rg = meta.num_rowgroups();

  // Half the ceiling per batch, so no single read approaches it even with uneven row groups.
  constexpr int64_t kListChildCeiling = 2147483647;
  const int64_t max_rows_per_batch = (kListChildCeiling / dim) / 2;
  const int64_t avg_rg_rows = (n_rows_file + n_rg - 1) / std::max(1, n_rg);

  // Read the span's row groups in cap-bounded batches; keep each batch's contiguous child data in
  // its own device buffer, then assemble once the exact total is known.
  std::vector<rmm::device_uvector<float>> pieces;
  int64_t total = 0;
  int batches = 0;
  for (size_t i = 0; i < span.size();) {
    std::vector<cudf::size_type> group;
    int64_t batch_rows = 0;
    while (i < span.size() && batch_rows < max_rows_per_batch) {
      group.push_back(span[i]);
      batch_rows += avg_rg_rows;
      ++i;
    }
    auto o = cudf::io::parquet_reader_options::builder(
                 cudf::io::source_info{std::vector<std::string>{path}})
                 .columns({col})
                 .build();
    o.set_row_groups({group});
    auto chunk = cudf::io::read_parquet(o, stream);
    auto child = cudf::lists_column_view(chunk.tbl->view().column(0)).child();
    EXPECT_EQ(child.type().id(), cudf::type_id::FLOAT32)
        << col << ": embedding child must be float32";
    EXPECT_EQ(static_cast<int64_t>(child.size()),
              static_cast<int64_t>(chunk.tbl->num_rows()) * dim)
        << col << ": batch is not a fixed " << dim << "-wide list";
    rmm::device_uvector<float> piece(child.size(), stream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(piece.data(), child.data<float>(),
                                  static_cast<size_t>(child.size()) * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream.value()));
    stream.synchronize();  // chunk (and its child) freed at loop end; piece now owns the data
    total += chunk.tbl->num_rows();
    pieces.push_back(std::move(piece));
    ++batches;
  }

  EmbShard m{rmm::device_uvector<float>(static_cast<size_t>(total) * dim, stream), total, batches};
  int64_t off = 0;
  for (auto& piece : pieces) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(m.buf.data() + off, piece.data(),
                                  piece.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                                  stream.value()));
    off += static_cast<int64_t>(piece.size());
  }
  stream.synchronize();  // assembled before `pieces` free here
  return m;
}

// build a cuVS L2Sqrt brute-force index over a resident row-major float matrix — L2SqrtExpanded
// returns TRUE L2 (DuckDB's array_distance), so D is used as-is with no conversion. Identical to
// test_tpchv.cpp's build_bf_index.
inline auto build_bf_index(raft::resources& handle, const float* data, int64_t rows, int dim) {
  auto dataset = raft::make_device_matrix_view<const float, int64_t>(data, rows, dim);
  cuvs::neighbors::brute_force::index_params ip;
  ip.metric = cuvs::distance::DistanceType::L2SqrtExpanded;
  return cuvs::neighbors::brute_force::build(handle, ip, dataset);
}

// ===========================================================================
// SHARDED SEARCH — one cuVS brute_force index per GPU over that GPU's embedding shard.
//
// A ShardSearcher type-erases the per-GPU raft handle, the shard matrix, and the cuVS index (all
// device objects that MUST live and die on their owning worker thread) behind two closures:
//   search(probe, K, host_idx, host_dist) — runs the top-K search on the shard (called ON the
//     worker), returns the SHARD-LOCAL neighbour indices and their distances to the host.
//   The captured shared_ptrs keep the device objects alive; clearing `search` on the worker
//     thread at teardown drops the last reference there (never on the main thread / device 0).
// ===========================================================================
struct ShardSearcher {
  int64_t rows = 0;           // rows in this shard (0 => empty span, skipped)
  int64_t global_offset = 0;  // number of rows in all earlier shards (natural parquet order)
  std::function<void(VecProbe const&, int64_t, std::vector<int64_t>&, std::vector<float>&)> search;
};

// Bytes of free VRAM on the current device — used to prove cuVS drew from the pool (a from-pool
// allocation does not change free VRAM, since the pool reserved it up front; a stray cudaMalloc
// outside the pool would drop it).
inline size_t free_vram() {
  size_t f = 0, t = 0;
  cudaMemGetInfo(&f, &t);
  return f;
}

// Build one shard index per worker; return the searchers (with global offsets filled in) and the
// max per-worker index-build time. Also asserts the shards reassemble to `expect_rows` and that
// each non-empty shard has >= K rows (cuVS requires K <= shard size; true by miles at sf40/G<=8).
std::vector<ShardSearcher> build_sharded_index(WorkerPool& pool, const std::string& path,
                                               const std::string& col, int dim, int64_t K,
                                               int64_t expect_rows, double& index_ms_out,
                                               const char* tag) {
  const int G = pool.size();
  const int n_rg = parquet_num_row_groups(path);
  const auto spans = partition_row_groups(n_rg, G);

  struct BuildOut {
    int64_t rows;
    double build_ms;
    std::function<void(VecProbe const&, int64_t, std::vector<int64_t>&, std::vector<float>&)> search;
  };

  std::vector<std::future<BuildOut>> fs;
  for (int g = 0; g < G; ++g)
    fs.push_back(pool[g].submit([&, g]() -> BuildOut {
      auto s = pool.stream(g);
      if (spans[g].empty()) return BuildOut{0, 0.0, {}};  // empty span -> contributes nothing

      auto handle = std::make_shared<raft::resources>();  // ctor on device g -> stream on device g
      auto shard  = load_embedding_shard(path, col, dim, spans[g], s);
      auto emb    = std::make_shared<rmm::device_uvector<float>>(std::move(shard.buf));

      const auto t0 = std::chrono::steady_clock::now();
      auto idx = build_bf_index(*handle, emb->data(), shard.rows, dim);
      raft::resource::sync_stream(*handle);
      const double build_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

      using IdxT = decltype(idx);
      auto idxp  = std::make_shared<IdxT>(std::move(idx));

      // capture order [handle, emb, idxp] => destroyed idxp, emb, handle (index frees on the
      // handle's stream while the handle is still alive).
      auto search = [handle, emb, idxp, dim](VecProbe const& p, int64_t Kq,
                                             std::vector<int64_t>& hidx, std::vector<float>& hdist) {
        auto stream = raft::resource::get_cuda_stream(*handle);
        auto d_q = raft::make_device_matrix<float, int64_t>(*handle, 1, dim);
        raft::copy(d_q.data_handle(), p.q.data(), dim, stream);
        auto nb = raft::make_device_matrix<int64_t, int64_t>(*handle, 1, Kq);
        auto ds = raft::make_device_matrix<float, int64_t>(*handle, 1, Kq);
        cuvs::neighbors::brute_force::search(*handle, cuvs::neighbors::brute_force::search_params{},
                                             *idxp, raft::make_const_mdspan(d_q.view()), nb.view(),
                                             ds.view());
        raft::resource::sync_stream(*handle);
        hidx.resize(Kq);
        hdist.resize(Kq);
        raft::copy(hidx.data(), nb.data_handle(), Kq, stream);
        raft::copy(hdist.data(), ds.data_handle(), Kq, stream);
        raft::resource::sync_stream(*handle);
      };
      return BuildOut{shard.rows, build_ms, std::move(search)};
    }));

  std::vector<ShardSearcher> out(G);
  int64_t offset = 0, total = 0;
  double max_build = 0.0;
  for (int g = 0; g < G; ++g) {
    auto b = fs[g].get();
    out[g].rows          = b.rows;
    out[g].global_offset = offset;
    out[g].search        = std::move(b.search);
    offset += b.rows;
    total += b.rows;
    max_build = std::max(max_build, b.build_ms);
    if (b.rows > 0)
      EXPECT_GE(b.rows, K) << tag << ": shard " << g << " has " << b.rows
                           << " rows < K=" << K << " — cuVS cannot return K neighbours from it.";
  }
  EXPECT_EQ(total, expect_rows) << tag << ": shards reassembled " << total << " embedding rows but "
                                << "the table has " << expect_rows << " — a span was dropped.";
  index_ms_out = max_build;
  std::fprintf(stderr, "[%s] sharded %ld embeddings over %d GPU(s), max index-build %.1f ms\n", tag,
               static_cast<long>(total), G, max_build);
  return out;
}

// Free the per-worker device objects (raft handle, shard matrix, cuVS index) ON their owning
// worker by clearing the search closure there — never on the main thread.
void release_searchers(WorkerPool& pool, std::vector<ShardSearcher>& sh) {
  std::vector<std::future<void>> fs;
  for (int g = 0; g < pool.size(); ++g)
    fs.push_back(pool[g].submit([&, g] { sh[g].search = nullptr; }));
  for (auto& f : fs) f.get();
}

// THE SHARDED SEARCH OPERATOR (timed, inside execute): search every shard for K, merge the G*K
// candidates on the host to the EXACT global result, and range-filter by D. Returns the global
// row indices (into the table's natural parquet order) of the rows under D — the same set the
// single-GPU vector_range_hits returns. The GLOBAL saturation guard is load-bearing.
std::vector<int32_t> sharded_range_hits(WorkerPool& pool, std::vector<ShardSearcher>& sh,
                                        VecProbe const& probe, int64_t K, const char* tag,
                                        StageProfiler* prof = nullptr) {
  const int G = pool.size();
  std::vector<std::vector<int64_t>> idx(G);
  std::vector<std::vector<float>> dist(G);

  std::vector<std::future<void>> fs;
  for (int g = 0; g < G; ++g) {
    if (sh[g].rows == 0) continue;
    fs.push_back(pool[g].submit([&, g] { sh[g].search(probe, K, idx[g], dist[g]); }));
  }
  for (auto& f : fs) f.get();
  if (prof) prof->tick("1_search");  // per-shard cuVS distance scan (parallel; wall = max worker)

  // merge: (distance, GLOBAL row index) candidates from every shard.
  struct Cand {
    float d;
    int32_t gi;
  };
  std::vector<Cand> cands;
  cands.reserve(static_cast<size_t>(G) * K);
  for (int g = 0; g < G; ++g)
    for (size_t i = 0; i < idx[g].size(); ++i)
      cands.push_back(Cand{dist[g][i], static_cast<int32_t>(sh[g].global_offset + idx[g][i])});

  if (static_cast<int64_t>(cands.size()) < K) {
    ADD_FAILURE() << tag << "/" << probe.id << ": only " << cands.size()
                  << " candidates merged (< K=" << K << ") — data too small for K.";
    return {};
  }
  // Only two facts are needed from the G*K candidates — the global K-th distance (for the guard)
  // and the set with distance < D (for the tail, order-irrelevant). A full sort is wasteful and
  // its O(G*K*log) SERIAL cost grows with G; nth_element gives the K-th in O(G*K) and the hits are
  // a single linear scan. (Each shard's list is already distance-sorted, but nth_element over the
  // union is simplest and the merge is no longer the bottleneck.)
  std::nth_element(cands.begin(), cands.begin() + (K - 1), cands.end(),
                   [](Cand const& a, Cand const& b) { return a.d < b.d; });
  const double kth = static_cast<double>(cands[K - 1].d);

  // GLOBAL SATURATION GUARD — the K-th smallest distance across ALL shards must be >= D, else more
  // than K rows globally fall under D and the union of per-shard top-K missed some: TRUNCATED.
  EXPECT_GE(kth, probe.D) << tag << "/" << probe.id << ": GLOBAL top-K SATURATED (K=" << K
                          << " per shard, global K-th distance " << kth << " < D=" << probe.D
                          << ") — the range answer would be silently truncated. Raise K.";

  std::vector<int32_t> hits;
  for (auto const& c : cands)
    if (static_cast<double>(c.d) < probe.D) hits.push_back(c.gi);
  if (prof) {
    prof->tick("2_search_merge");  // gather G*K cands + nth_element K-th guard + range-filter <D
    prof->rowcount("hits_under_D", static_cast<long>(hits.size()));
  }
  std::fprintf(stderr, "[%s] %s: D=%.17g -> %zu rows under D (global K-th %.9g)\n", tag,
               probe.id.c_str(), probe.D, hits.size(), kth);
  return hits;
}

// COUNT CORROBORATION — verify path only (reads a golden, kept out of the timed region). Same as
// single-GPU: pins the row SET the distance predicate selects against DuckDB's own count.
inline void expect_hit_count(size_t got, const std::string& count_golden, const char* tag,
                             const std::string& id) {
  EXPECT_TRUE(file_exists(count_golden)) << "missing " << count_golden;
  if (file_exists(count_golden)) {
    const int64_t want =
        std::strtoll(read_single_value_golden(count_golden).c_str(), nullptr, 10);
    EXPECT_EQ(static_cast<int64_t>(got), want)
        << tag << "/" << id << ": cuVS found " << got << " rows under D but DuckDB found " << want
        << ". A one-row difference means cuVS and DuckDB landed on opposite sides of the distance "
        << "boundary — a real numerical divergence, NOT something to absorb with a tolerance.";
  }
}

// host hit list -> owning device int32 column on GPU0, ready to join against a row index.
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

// read a set of columns from one parquet file on the CURRENT device (GPU0, for the relational
// tail) — no predicate, no row selection. Same as test_tpchv.cpp's read_cols.
cudf::io::table_with_metadata read_cols(const std::string& path, std::vector<std::string> cols) {
  auto o = cudf::io::parquet_reader_options::builder(
               cudf::io::source_info{std::vector<std::string>{path}})
               .columns(std::move(cols))
               .build();
  return cudf::io::read_parquet(o);
}

cudf::column_view map_view(std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& m) {
  return cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                           static_cast<cudf::size_type>(m->size()), m->data(), nullptr, 0);
}

cudf::timestamp_scalar<cudf::timestamp_D> date_scalar(int y, unsigned mo, unsigned d) {
  return cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(y, mo, d)}}, true);
}

// stream-taking variant, for the worker-side `project` closures (off GPU0 every cudf op needs a
// device-local stream).
cudf::timestamp_scalar<cudf::timestamp_D> date_scalar(int y, unsigned mo, unsigned d,
                                                      rmm::cuda_stream_view s) {
  return cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(y, mo, d)}}, true, s);
}

// ===========================================================================
// PARALLEL LINEITEM TAIL — the M4 step that scales the RELATIONAL tail.
//
// The sharded search yields a small hit-key set (~1e5 part/partsupp keys under D, resident on
// GPU0). This BROADCASTS that key column to every GPU (pack once on GPU0, peer-copy via
// gather_here — keeps the column's native type, so the semi-join key types match; a host int32
// round-trip silently mistyped int64 keys), PARTITIONS the 240M-row lineitem across the GPUs
// (whole row groups, as M1/M2 do; the shards are loaded resident once in the LOAD phase), and on
// each GPU runs `project` — the query's lineitem-LOCAL predicates + a SEMI-JOIN against the
// broadcast keys + the projection the downstream tail needs. `project` gets (shard_view,
// hit_keys_column_on_this_device, stream) and must thread the stream through every cudf op
// (device-local stream required off GPU0). Because the hit-key semi-join is highly selective
// (~1e5 keys vs 240M rows), each shard's survivors are small, so they are gathered to GPU0
// (gather_here + concatenate) and the query's remaining dim joins + group-by finish there on the
// full survivor set — EXACT, no partial-aggregate merge and no dim-table broadcast redundancy
// needed. That makes the parallelized part the expensive 240M lineitem scan/filter/semi-join; the
// small tail after it stays on GPU0 (a residual, reported).
// ===========================================================================
std::unique_ptr<cudf::table> semijoin_gather(
    WorkerPool& pool, std::vector<cudf::io::table_with_metadata>& shards,
    cudf::table_view hit_keys,  // single-column key table resident on GPU0
    std::function<std::unique_ptr<cudf::table>(cudf::table_view, cudf::column_view,
                                               rmm::cuda_stream_view)> const& project,
    StageProfiler* prof = nullptr) {
  const int G = pool.size();
  // Broadcast the hit-key column: pack once on GPU0, each worker peer-copies it in (gather_here).
  auto packed_keys = cudf::pack(hit_keys, cudf::get_default_stream());
  cudf::get_default_stream().synchronize();
  const PackedPartial keys_handle = describe_packed(0, packed_keys);

  std::vector<std::unique_ptr<cudf::table>> surv(G);
  std::vector<cudf::packed_columns> packed(G);

  std::vector<std::future<PackedPartial>> pf;
  for (int g = 0; g < G; ++g)
    pf.push_back(pool[g].submit([&, g]() -> PackedPartial {
      auto s   = pool.stream(g);
      auto kt  = gather_here(keys_handle, s);  // the broadcast key column, now on device g
      surv[g]  = project(shards[g].tbl->view(), kt.view.column(0), s);
      packed[g] = cudf::pack(surv[g]->view(), s);
      s.synchronize();
      return describe_packed(g, packed[g]);
    }));
  std::vector<PackedPartial> handles(G);
  for (int g = 0; g < G; ++g) handles[g] = pf[g].get();
  if (prof) prof->tick("3_tail_scan");  // broadcast keys + per-GPU lineitem filter+semijoin

  auto out = pool[0]
                 .submit([&]() -> std::unique_ptr<cudf::table> {
                   auto s = pool.stream(0);
                   std::vector<GatheredTable> gts;
                   gts.reserve(G);
                   std::vector<cudf::table_view> views;
                   views.reserve(G);
                   for (int g = 0; g < G; ++g) {
                     gts.push_back(gather_here(handles[g], s));
                     views.push_back(gts.back().view);
                   }
                   auto merged = cudf::concatenate(views, s);
                   s.synchronize();
                   return merged;
                 })
                 .get();
  if (prof) {
    prof->tick("4_gather_survivors");  // gather_here + concatenate the survivors to GPU0
    prof->rowcount("survivors_total", static_cast<long>(out->num_rows()));
  }

  std::vector<std::future<void>> rf;
  for (int g = 0; g < G; ++g)
    rf.push_back(pool[g].submit([&, g] {
      packed[g] = cudf::packed_columns{};
      surv[g].reset();
    }));
  for (auto& f : rf) f.get();
  return out;
}

// Load lineitem's row-group shards resident on each worker (once, in the LOAD phase), so the
// per-probe execute only does compute over them — same discipline as M1/M2.
std::vector<cudf::io::table_with_metadata> load_lineitem_shards(
    WorkerPool& pool, std::string const& path, std::vector<std::string> const& cols) {
  const int G = pool.size();
  const int nrg = parquet_num_row_groups(path);
  const auto spans = partition_row_groups(nrg, G);
  std::vector<cudf::io::table_with_metadata> shards(G);
  std::vector<std::future<void>> fs;
  for (int g = 0; g < G; ++g)
    fs.push_back(pool[g].submit([&, g] {
      auto s   = pool.stream(g);
      shards[g] = read_row_group_span(path, cols, spans[g], s);
      s.synchronize();
    }));
  for (auto& f : fs) f.get();
  return shards;
}

}  // namespace

// ===========================================================================
// TPC-H+V q11v — national market value in a vector neighbourhood (partsupp.ps_image_embedding).
// Same query, joins and golden as test_tpchv.cpp Q11VectorBruteForce; only the search is sharded.
// ===========================================================================
TEST_F(TpchSf40, Q11VectorBruteForceMultiGpu) {
  const int G = gpu_count();
  if (G < 1) GTEST_SKIP() << "no visible GPU";
  const auto params_path = vec_params_path();
  if (!file_exists(params_path))
    GTEST_SKIP() << "query_params.jsonl not found at " << params_path << " — NOTHING VERIFIED.";
  const std::vector<std::string> want{"img_000", "img_017", "img_034"};
  auto probes = load_probes(params_path, want);
  ASSERT_EQ(probes.size(), want.size()) << "missing probes in " << params_path;
  for (auto const& p : probes) ASSERT_EQ(p.q.size(), 96u) << p.id << ": expected 96 dims";

  auto& pool = shared_pool();
  const auto boolean   = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};

  // ---------------- LOAD ----------------
  // Relational (scalar) columns on GPU0; the embedding is SHARDED across all GPUs + per-shard
  // cuVS index built (the setup phase, timed separately).
  const auto ps_path = data_dir() + "/partsupp.parquet";
  auto ps_in  = read_cols(ps_path, {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"});
  auto sup_in = read_cols(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  auto nat_in = read_cols(data_dir() + "/nation.parquet", {"n_nationkey", "n_name"});
  const int64_t n_ps = ps_in.tbl->num_rows();
  const int64_t K = search_k();
  ASSERT_LE(K, n_ps);

  double index_ms = 0.0;
  auto searchers = build_sharded_index(pool, ps_path, "ps_image_embedding", 96, K, n_ps, index_ms,
                                       "q11v-mgpu");

  // --- the GERMANY join, built ONCE and used twice (threshold + filtered groups) ---
  auto germany = cudf::string_scalar(std::string("GERMANY"));
  auto nmask = cudf::binary_operation(nat_in.tbl->view().column(1), germany,
                                      cudf::binary_operator::EQUAL, boolean);
  auto nat_de = cudf::apply_boolean_mask(cudf::table_view{{nat_in.tbl->view().column(0)}},
                                         nmask->view());
  auto [s_map, n_map] = cudf::inner_join(cudf::table_view{{sup_in.tbl->view().column(1)}},
                                         cudf::table_view{{nat_de->get_column(0).view()}});
  auto sup_de = cudf::gather(cudf::table_view{{sup_in.tbl->view().column(0)}}, map_view(s_map));

  auto [ps_map, sd_map] = cudf::inner_join(cudf::table_view{{ps_in.tbl->view().column(1)}},
                                           cudf::table_view{{sup_de->get_column(0).view()}});
  auto row_idx = cudf::sequence(static_cast<cudf::size_type>(n_ps),
                                *cudf::make_fixed_width_scalar<int32_t>(0),
                                *cudf::make_fixed_width_scalar<int32_t>(1));
  auto de = cudf::gather(cudf::table_view{{ps_in.tbl->view().column(0),   // ps_partkey
                                           ps_in.tbl->view().column(2),   // ps_availqty
                                           ps_in.tbl->view().column(3),   // ps_supplycost
                                           row_idx->view()}},             // ORIGINAL partsupp index
                         map_view(ps_map));
  ASSERT_GT(de->num_rows(), 0);

  auto de_avail = cudf::cast(de->get_column(1).view(), dec128_s2);
  auto de_cost  = cudf::cast(de->get_column(2).view(), dec128_s2);
  auto de_value = cudf::binary_operation(de_cost->view(), de_avail->view(),
                                         cudf::binary_operator::MUL, dec128_s4);
  auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto total = cudf::reduce(de_value->view(), *sum_agg, dec128_s4);
  auto* total_fp = dynamic_cast<cudf::fixed_point_scalar<numeric::decimal128>*>(total.get());
  ASSERT_NE(total_fp, nullptr);
  const __int128 total_unscaled = static_cast<__int128>(total_fp->value());

  // --- per-probe execute: SHARDED search -> range filter -> intersect German rows -> groupby ---
  for (auto const& probe : probes) {
    StageProfiler prof;
    auto execute = [&]() -> std::unique_ptr<cudf::table> {
      prof.begin();
      auto hits = sharded_range_hits(pool, searchers, probe, K, "q11v-mgpu", &prof);
      auto d_hits = to_device_hits(hits);
      auto [de_map, hit_map] = cudf::inner_join(cudf::table_view{{de->get_column(3).view()}},
                                                cudf::table_view{{d_hits.view()}});
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

      auto thresh = cudf::fixed_point_scalar<numeric::decimal128>(total_unscaled / 500000,
                                                                  numeric::scale_type{-4});
      auto keep = cudf::binary_operation(values->view(), thresh, cudf::binary_operator::GREATER,
                                         boolean);
      auto kept = cudf::apply_boolean_mask(
          cudf::table_view{{gk->view().column(0), values->view()}}, keep->view());
      auto order = cudf::sorted_order(
          cudf::table_view{{kept->get_column(1).view(), kept->get_column(0).view()}},
          {cudf::order::DESCENDING, cudf::order::ASCENDING},
          {cudf::null_order::AFTER, cudf::null_order::AFTER});
      auto res = cudf::gather(
          cudf::table_view{{kept->get_column(0).view(), kept->get_column(1).view()}},
          order->view());
      prof.tick("5_intersect_groupby");  // q11's trivial tail: intersect precomputed German rows + groupby
      prof.rowcount("groups", static_cast<long>(gk->num_rows()));
      return res;
    };

    auto count_hits = sharded_range_hits(pool, searchers, probe, K, "q11v-mgpu");
    expect_hit_count(count_hits.size(),
                     golden_dir() + "/duckdb_psimage_" + probe.id + ".count.csv", "q11v-mgpu",
                     probe.id);

    auto sorted = execute();
    const std::vector<ColSpec> spec = {{"ps_partkey", Cmp::ExactInt}, {"value", Cmp::ExactDecimal}};
    const auto golden = read_csv_golden(golden_dir() + "/duckdb_q11v_" + probe.id + ".csv");
    compare_table_to_golden(sorted->view(), golden, spec, ("q11v-mgpu/" + probe.id).c_str());
    std::fprintf(stderr, "[q11v-mgpu] %s: %ld result rows matched\n", probe.id.c_str(),
                 static_cast<long>(sorted->num_rows()));

    benchmark_mgpu(("q11v-mgpu/" + probe.id).c_str(), execute, G, index_ms);
    if (profile_enabled()) {
      prof.on = true;
      prof.num_gpus = G;
      for (int i = 0; i < 6; ++i) { auto r = execute(); (void)r; }
      prof.report(("q11v-mgpu/" + probe.id).c_str(), G);
    }
  }

  // POOL-DRAW PROOF (q11v only): a search must not reduce free VRAM on any worker — that would
  // mean cuVS allocated OUTSIDE the per-device pool (which reserved its memory at ctor).
  for (int g = 0; g < G; ++g) {
    if (searchers[g].rows == 0) continue;
    auto delta = pool[g]
                     .submit([&, g]() -> long {
                       const size_t before = free_vram();
                       std::vector<int64_t> hi;
                       std::vector<float> hd;
                       searchers[g].search(probes.front(), K, hi, hd);
                       cudaDeviceSynchronize();
                       const size_t after = free_vram();
                       return static_cast<long>(before) - static_cast<long>(after);
                     })
                     .get();
    EXPECT_LE(std::llabs(delta), 64L << 20)
        << "q11v-mgpu: GPU " << g << " free VRAM changed by " << delta
        << " B across a cuVS search — cuVS did NOT allocate from the RMM pool.";
    std::fprintf(stderr, "[q11v-mgpu] GPU %d free-VRAM delta across search: %ld B (pool-drawn)\n", g,
                 delta);
  }

  release_searchers(pool, searchers);
}

// ===========================================================================
// TPC-H+V q12v — shipping-mode SLA counts in a vector neighbourhood (part.p_text_embedding).
// Same query/joins/golden as test_tpchv.cpp Q12VectorShipModeCounts; only the search is sharded.
// ===========================================================================
TEST_F(TpchSf40, Q12VectorShipModeCountsMultiGpu) {
  const int G = gpu_count();
  if (G < 1) GTEST_SKIP() << "no visible GPU";
  const auto params_path = vec_params_path();
  if (!file_exists(params_path))
    GTEST_SKIP() << "query_params.jsonl not found at " << params_path << " — NOTHING VERIFIED.";
  auto probes = load_probes(params_path, {"txt_000"});
  ASSERT_EQ(probes.size(), 1u) << "missing probe txt_000 in " << params_path;
  const auto& probe = probes.front();
  ASSERT_EQ(probe.q.size(), 100u) << "txt_000 should be a 100-dim text probe";
  const auto golden_path = golden_dir() + "/duckdb_q12v_" + probe.id + ".csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  auto& pool = shared_pool();
  const auto boolean = cudf::data_type{cudf::type_id::BOOL8};
  const auto int32   = cudf::data_type{cudf::type_id::INT32};

  // ---------------- LOAD ----------------
  // orders + part-key on GPU0 (the post-search dim tables); the embedding SHARDED for the search;
  // and lineitem PARTITIONED across the GPUs (M4: the 240M-row scan is now parallel, not GPU0-only).
  auto ord_in = read_cols(data_dir() + "/orders.parquet", {"o_orderkey", "o_orderpriority"});
  const auto part_path = data_dir() + "/part.parquet";
  auto part_in = read_cols(part_path, {"p_partkey"});
  const int64_t n_part = part_in.tbl->num_rows();
  const int64_t K = search_k();
  ASSERT_LE(K, n_part);
  double index_ms = 0.0;
  auto searchers = build_sharded_index(pool, part_path, "p_text_embedding", 100, K, n_part,
                                       index_ms, "q12v-mgpu");
  auto line_shards = load_lineitem_shards(
      pool, data_dir() + "/lineitem.parquet",
      {"l_orderkey", "l_partkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"});

  auto pk = part_in.tbl->view().column(0);

  // Worker-side project: filter the lineitem shard (shipmode + 3 date predicates) and SEMI-JOIN it
  // against the broadcast part-hit keys, projecting (l_orderkey, l_shipmode). Every cudf op takes
  // the device-local stream. Shard cols: 0 orderkey,1 partkey,2 shipmode,3 commit,4 receipt,5 ship.
  auto project = [](cudf::table_view lv, cudf::column_view hk,
                    rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    const auto b = cudf::data_type{cudf::type_id::BOOL8};
    auto mail = cudf::string_scalar(std::string("MAIL"), true, s);
    auto ship = cudf::string_scalar(std::string("SHIP"), true, s);
    auto is_mail = cudf::binary_operation(lv.column(2), mail, cudf::binary_operator::EQUAL, b, s);
    auto is_ship = cudf::binary_operation(lv.column(2), ship, cudf::binary_operator::EQUAL, b, s);
    auto mode_ok = cudf::binary_operation(is_mail->view(), is_ship->view(),
                                          cudf::binary_operator::LOGICAL_OR, b, s);
    auto commit_lt_receipt =
        cudf::binary_operation(lv.column(3), lv.column(4), cudf::binary_operator::LESS, b, s);
    auto ship_lt_commit =
        cudf::binary_operation(lv.column(5), lv.column(3), cudf::binary_operator::LESS, b, s);
    auto d1994 = date_scalar(1994, 1, 1, s);
    auto d1995 = date_scalar(1995, 1, 1, s);
    auto recv_ge =
        cudf::binary_operation(lv.column(4), d1994, cudf::binary_operator::GREATER_EQUAL, b, s);
    auto recv_lt = cudf::binary_operation(lv.column(4), d1995, cudf::binary_operator::LESS, b, s);
    auto m1 = cudf::binary_operation(mode_ok->view(), commit_lt_receipt->view(),
                                     cudf::binary_operator::LOGICAL_AND, b, s);
    auto m2 = cudf::binary_operation(m1->view(), ship_lt_commit->view(),
                                     cudf::binary_operator::LOGICAL_AND, b, s);
    auto m3 = cudf::binary_operation(m2->view(), recv_ge->view(),
                                     cudf::binary_operator::LOGICAL_AND, b, s);
    auto mask = cudf::binary_operation(m3->view(), recv_lt->view(),
                                       cudf::binary_operator::LOGICAL_AND, b, s);
    auto line_f = cudf::apply_boolean_mask(
        cudf::table_view{{lv.column(0), lv.column(1), lv.column(2)}}, mask->view(), s);
    // SEMI-JOIN l_partkey ∈ hit keys (hit keys are distinct part keys -> inner join is 1:1)
    auto [l_map, h_map] = cudf::inner_join(cudf::table_view{{line_f->get_column(1).view()}},
                                           cudf::table_view{{hk}}, cudf::null_equality::EQUAL, s);
    return cudf::gather(
        cudf::table_view{{line_f->get_column(0).view(), line_f->get_column(2).view()}},
        map_view(l_map), cudf::out_of_bounds_policy::DONT_CHECK, s);  // (l_orderkey, l_shipmode)
  };

  StageProfiler prof;  // no-op unless PEACOCK_PROFILE; ticks below cost nothing off-profile
  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    prof.begin();
    // 1) sharded search -> global hit row indices; 2) materialize the hit part KEYS on GPU0 and
    // copy to host (to broadcast); 3) parallel lineitem filter+semijoin -> gather survivors to GPU0.
    auto hits = sharded_range_hits(pool, searchers, probe, K, "q12v-mgpu", &prof);
    auto d_hits = to_device_hits(hits);
    auto sel = cudf::gather(cudf::table_view{{pk}}, d_hits.view());  // the hit part keys, on GPU0
    auto lp = semijoin_gather(pool, line_shards, sel->view(), project, &prof);  // (orderkey, shipmode)
    EXPECT_GT(lp->num_rows(), 0) << "vector predicate and filters left no rows to join";

    // 4) FINISH on GPU0 (survivor set is tiny): join broadcast orders, count by priority, group.
    auto [lp_map, o_map] = cudf::inner_join(cudf::table_view{{lp->get_column(0).view()}},
                                            cudf::table_view{{ord_in.tbl->view().column(0)}});
    auto mode_col = cudf::gather(cudf::table_view{{lp->get_column(1).view()}}, map_view(lp_map));
    auto prio_col = cudf::gather(cudf::table_view{{ord_in.tbl->view().column(1)}}, map_view(o_map));
    prof.tick("5_dim_orders");
    prof.rowcount("after_orders", static_cast<long>(mode_col->num_rows()));

    auto urgent = cudf::string_scalar(std::string("1-URGENT"));
    auto high = cudf::string_scalar(std::string("2-HIGH"));
    auto is_urgent = cudf::binary_operation(prio_col->get_column(0).view(), urgent,
                                            cudf::binary_operator::EQUAL, boolean);
    auto is_high = cudf::binary_operation(prio_col->get_column(0).view(), high,
                                          cudf::binary_operator::EQUAL, boolean);
    auto is_hi = cudf::binary_operation(is_urgent->view(), is_high->view(),
                                        cudf::binary_operator::LOGICAL_OR, boolean);
    auto is_lo = cudf::unary_operation(is_hi->view(), cudf::unary_operator::NOT);
    auto hi_i = cudf::cast(is_hi->view(), int32);
    auto lo_i = cudf::cast(is_lo->view(), int32);

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
    auto order = cudf::sorted_order(cudf::table_view{{gk->view().column(0)}},
                                    {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
    auto res = cudf::gather(
        cudf::table_view{{gk->view().column(0), hi_sum->view(), lo_sum->view()}}, order->view());
    prof.tick("6_groupby_sort");
    prof.rowcount("groups", static_cast<long>(gk->num_rows()));
    return res;
  };

  auto count_hits = sharded_range_hits(pool, searchers, probe, K, "q12v-mgpu");
  expect_hit_count(count_hits.size(),
                   golden_dir() + "/duckdb_ptext_" + probe.id + ".count.csv", "q12v-mgpu",
                   probe.id);
  ASSERT_GT(count_hits.size(), 0u) << "no part rows under D — test would be vacuous";

  auto sorted = execute();
  const std::vector<ColSpec> spec = {
      {"l_shipmode", Cmp::ExactString},
      {"high_line_count", Cmp::ExactInt},
      {"low_line_count", Cmp::ExactInt},
  };
  const auto golden = read_csv_golden(golden_path);
  ASSERT_GT(golden.size(), 0u) << "golden is empty: " << golden_path;
  compare_table_to_golden(sorted->view(), golden, spec, "q12v-mgpu");
  std::fprintf(stderr, "[q12v-mgpu] %ld result rows matched\n",
               static_cast<long>(sorted->num_rows()));

  benchmark_mgpu("q12v-mgpu", execute, G, index_ms);
  if (profile_enabled()) {
    prof.on = true;
    prof.num_gpus = G;
    for (int i = 0; i < 6; ++i) { auto r = execute(); (void)r; }
    prof.report("q12v-mgpu", G);
  }
  release_searchers(pool, searchers);
  release_partitions(pool, line_shards);
}

// ===========================================================================
// TPC-H+V q10v — returned-item revenue by customer in a vector neighbourhood (part.p_text_embedding).
// Same query/joins/golden as test_tpchv.cpp Q10VectorCustomerTopN; only the search is sharded.
// ===========================================================================
TEST_F(TpchSf40, Q10VectorCustomerTopNMultiGpu) {
  const int G = gpu_count();
  if (G < 1) GTEST_SKIP() << "no visible GPU";
  const auto params_path = vec_params_path();
  if (!file_exists(params_path))
    GTEST_SKIP() << "query_params.jsonl not found at " << params_path << " — NOTHING VERIFIED.";
  auto probes = load_probes(params_path, {"txt_017"});
  ASSERT_EQ(probes.size(), 1u) << "missing probe txt_017 in " << params_path;
  const auto& probe = probes.front();
  ASSERT_EQ(probe.q.size(), 100u) << "txt_017 should be a 100-dim text probe";
  const auto golden_path = golden_dir() + "/duckdb_q10v_" + probe.id + ".csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  auto& pool = shared_pool();
  const auto boolean   = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};

  // ---------------- LOAD ----------------
  // dim tables (customer/orders/nation/part-key) on GPU0; embedding SHARDED for the search;
  // lineitem PARTITIONED across the GPUs (M4).
  auto cust_in = read_cols(data_dir() + "/customer.parquet",
                           {"c_custkey", "c_name", "c_address", "c_nationkey", "c_phone",
                            "c_acctbal", "c_comment"});
  auto ord_in = read_cols(data_dir() + "/orders.parquet", {"o_orderkey", "o_custkey", "o_orderdate"});
  auto nat_in = read_cols(data_dir() + "/nation.parquet", {"n_nationkey", "n_name"});
  const auto part_path = data_dir() + "/part.parquet";
  auto part_in = read_cols(part_path, {"p_partkey"});
  const int64_t n_part = part_in.tbl->num_rows();
  const int64_t K = search_k();
  ASSERT_LE(K, n_part);
  double index_ms = 0.0;
  auto searchers = build_sharded_index(pool, part_path, "p_text_embedding", 100, K, n_part,
                                       index_ms, "q10v-mgpu");
  auto line_shards = load_lineitem_shards(
      pool, data_dir() + "/lineitem.parquet",
      {"l_orderkey", "l_partkey", "l_extendedprice", "l_discount", "l_returnflag"});

  auto cv = cust_in.tbl->view();
  auto ov = ord_in.tbl->view();
  auto pk = part_in.tbl->view().column(0);

  // Worker-side project: filter the lineitem shard (l_returnflag='R') + SEMI-JOIN part-hit keys,
  // projecting (l_orderkey, l_extendedprice, l_discount). Shard cols: 0 orderkey,1 partkey,2
  // extprice,3 discount,4 returnflag.
  auto project = [](cudf::table_view lv, cudf::column_view hk,
                    rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    const auto b = cudf::data_type{cudf::type_id::BOOL8};
    auto flag_r = cudf::string_scalar(std::string("R"), true, s);
    auto mask = cudf::binary_operation(lv.column(4), flag_r, cudf::binary_operator::EQUAL, b, s);
    auto line_f = cudf::apply_boolean_mask(
        cudf::table_view{{lv.column(0), lv.column(1), lv.column(2), lv.column(3)}}, mask->view(), s);
    auto [l_map, h_map] = cudf::inner_join(cudf::table_view{{line_f->get_column(1).view()}},
                                           cudf::table_view{{hk}}, cudf::null_equality::EQUAL, s);
    return cudf::gather(cudf::table_view{{line_f->get_column(0).view(), line_f->get_column(2).view(),
                                          line_f->get_column(3).view()}},
                        map_view(l_map), cudf::out_of_bounds_policy::DONT_CHECK, s);
  };

  StageProfiler prof;
  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    prof.begin();
    auto hits = sharded_range_hits(pool, searchers, probe, K, "q10v-mgpu", &prof);
    auto d_hits = to_device_hits(hits);
    auto sel = cudf::gather(cudf::table_view{{pk}}, d_hits.view());  // the hit part keys, on GPU0
    auto lp = semijoin_gather(pool, line_shards, sel->view(), project, &prof);  // (orderkey,price,disc)

    // orders filtered to the 1993-10..1994-01 quarter (GPU0) — a residual GPU0 scan (reported).
    auto d_lo = date_scalar(1993, 10, 1);
    auto d_hi = date_scalar(1994, 1, 1);
    auto o_ge = cudf::binary_operation(ov.column(2), d_lo, cudf::binary_operator::GREATER_EQUAL,
                                       boolean);
    auto o_lt = cudf::binary_operation(ov.column(2), d_hi, cudf::binary_operator::LESS, boolean);
    auto ord_mask = cudf::binary_operation(o_ge->view(), o_lt->view(),
                                           cudf::binary_operator::LOGICAL_AND, boolean);
    auto ord_f = cudf::apply_boolean_mask(cudf::table_view{{ov.column(0), ov.column(1)}},
                                          ord_mask->view());
    EXPECT_GT(lp->num_rows(), 0);
    EXPECT_GT(ord_f->num_rows(), 0);

    auto [lp_map, o_map] = cudf::inner_join(cudf::table_view{{lp->get_column(0).view()}},
                                            cudf::table_view{{ord_f->get_column(0).view()}});
    auto lo_l = cudf::gather(cudf::table_view{{lp->get_column(1).view(), lp->get_column(2).view()}},
                             map_view(lp_map));
    auto lo_o = cudf::gather(cudf::table_view{{ord_f->get_column(1).view()}}, map_view(o_map));
    EXPECT_GT(lo_o->num_rows(), 0) << "vector predicate and filters left no rows to join";
    prof.tick("5_dim_orders");
    prof.rowcount("after_orders", static_cast<long>(lo_o->num_rows()));

    auto [c_map, lo_map] = cudf::inner_join(cudf::table_view{{cv.column(0)}},
                                            cudf::table_view{{lo_o->get_column(0).view()}});
    auto cust_side = cudf::gather(cudf::table_view{{cv.column(0), cv.column(1), cv.column(5),
                                                    cv.column(4), cv.column(2), cv.column(6),
                                                    cv.column(3)}},
                                  map_view(c_map));
    auto val_side = cudf::gather(
        cudf::table_view{{lo_l->get_column(0).view(), lo_l->get_column(1).view()}},
        map_view(lo_map));

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
    EXPECT_GT(cust_j->num_rows(), 0);
    prof.tick("5b_dim_customer_nation");
    prof.rowcount("after_customer", static_cast<long>(cust_j->num_rows()));

    auto price = cudf::cast(val_j->get_column(0).view(), dec128_s2);
    auto disc = cudf::cast(val_j->get_column(1).view(), dec128_s2);
    auto one_s2 = cudf::fixed_point_scalar<numeric::decimal128>(100, numeric::scale_type{-2});
    auto one_minus_disc =
        cudf::binary_operation(one_s2, disc->view(), cudf::binary_operator::SUB, dec128_s2);
    auto revenue = cudf::binary_operation(price->view(), one_minus_disc->view(),
                                          cudf::binary_operator::MUL, dec128_s4);

    cudf::groupby::groupby gb(cudf::table_view{{cust_j->get_column(0).view(),
                                                cust_j->get_column(1).view(),
                                                cust_j->get_column(2).view(),
                                                cust_j->get_column(3).view(),
                                                nat_j->get_column(0).view(),
                                                cust_j->get_column(4).view(),
                                                cust_j->get_column(5).view()}});
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

    auto gkv = gk->view();
    auto order = cudf::sorted_order(cudf::table_view{{rev_col->view(), gkv.column(0)}},
                                    {cudf::order::DESCENDING, cudf::order::ASCENDING},
                                    {cudf::null_order::AFTER, cudf::null_order::AFTER});
    auto res = cudf::gather(cudf::table_view{{gkv.column(0), gkv.column(1), rev_col->view(),
                                              gkv.column(2), gkv.column(4), gkv.column(5),
                                              gkv.column(3), gkv.column(6)}},
                            order->view());
    prof.tick("6_groupby_sort");
    prof.rowcount("groups", static_cast<long>(gk->num_rows()));
    return res;
  };

  auto count_hits = sharded_range_hits(pool, searchers, probe, K, "q10v-mgpu");
  expect_hit_count(count_hits.size(),
                   golden_dir() + "/duckdb_ptext_" + probe.id + ".count.csv", "q10v-mgpu",
                   probe.id);
  ASSERT_GT(count_hits.size(), 0u) << "no part rows under D — test would be vacuous";

  auto sorted = execute();
  auto top = cudf::slice(sorted->view(), {0, 20})[0];  // LIMIT 20 (zero-copy view)
  const std::vector<ColSpec> spec = {
      {"c_custkey", Cmp::ExactInt},   {"c_name", Cmp::ExactString},
      {"revenue", Cmp::ExactDecimal}, {"c_acctbal", Cmp::ExactDecimal},
      {"n_name", Cmp::ExactString},   {"c_address", Cmp::ExactString},
      {"c_phone", Cmp::ExactString},  {"c_comment", Cmp::ExactString},
  };
  const auto golden = read_csv_golden(golden_path);
  ASSERT_EQ(static_cast<int>(golden.size()), 20) << "golden should hold 20 rows";
  compare_table_to_golden(top, golden, spec, "q10v-mgpu");
  std::fprintf(stderr, "[q10v-mgpu] 20 result rows matched\n");

  benchmark_mgpu("q10v-mgpu", execute, G, index_ms);
  if (profile_enabled()) {
    prof.on = true;
    prof.num_gpus = G;
    for (int i = 0; i < 6; ++i) { auto r = execute(); (void)r; }
    prof.report("q10v-mgpu", G);
  }
  release_searchers(pool, searchers);
  release_partitions(pool, line_shards);
}

// ===========================================================================
// TPC-H+V q9v — product-line profit by nation and year in a vector neighbourhood
// (part.p_text_embedding + p_name LIKE '%green%'). Same query/joins/golden as test_tpchv.cpp
// Q9VectorCompositeJoin; only the search is sharded.
// ===========================================================================
TEST_F(TpchSf40, Q9VectorCompositeJoinMultiGpu) {
  const int G = gpu_count();
  if (G < 1) GTEST_SKIP() << "no visible GPU";
  const auto params_path = vec_params_path();
  if (!file_exists(params_path))
    GTEST_SKIP() << "query_params.jsonl not found at " << params_path << " — NOTHING VERIFIED.";
  auto probes = load_probes(params_path, {"txt_034"});
  ASSERT_EQ(probes.size(), 1u) << "missing probe txt_034 in " << params_path;
  const auto& probe = probes.front();
  ASSERT_EQ(probe.q.size(), 100u) << "txt_034 should be a 100-dim text probe";
  const auto golden_path = golden_dir() + "/duckdb_q9v_" + probe.id + ".csv";
  ASSERT_TRUE(file_exists(golden_path)) << "golden missing: " << golden_path;

  auto& pool = shared_pool();
  const auto boolean   = cudf::data_type{cudf::type_id::BOOL8};
  const auto dec128_s2 = cudf::data_type{cudf::type_id::DECIMAL128, -2};
  const auto dec128_s4 = cudf::data_type{cudf::type_id::DECIMAL128, -4};

  // ---------------- LOAD ----------------
  // part(key+name), supplier/partsupp/orders/nation on GPU0; embedding SHARDED for the search;
  // lineitem PARTITIONED across the GPUs (M4). partsupp (32M) stays on GPU0 — the composite join
  // against it is a residual GPU0 scan (reported); lineitem is the dominant scan we parallelize.
  const auto part_path = data_dir() + "/part.parquet";
  auto part_name_in = read_cols(part_path, {"p_partkey", "p_name"});
  auto sup_in = read_cols(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  auto ps_in = read_cols(data_dir() + "/partsupp.parquet",
                         {"ps_partkey", "ps_suppkey", "ps_supplycost"});
  auto ord_in = read_cols(data_dir() + "/orders.parquet", {"o_orderkey", "o_orderdate"});
  auto nat_in = read_cols(data_dir() + "/nation.parquet", {"n_nationkey", "n_name"});
  const int64_t n_part = part_name_in.tbl->num_rows();
  const int64_t K = search_k();
  ASSERT_LE(K, n_part);
  double index_ms = 0.0;
  auto searchers = build_sharded_index(pool, part_path, "p_text_embedding", 100, K, n_part,
                                       index_ms, "q9v-mgpu");
  auto line_shards = load_lineitem_shards(
      pool, data_dir() + "/lineitem.parquet",
      {"l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount", "l_quantity"});

  auto pk = part_name_in.tbl->view().column(0);
  auto pname = part_name_in.tbl->view().column(1);

  // Worker-side project: SEMI-JOIN the lineitem shard against the broadcast hit keys (q9's hit set
  // is green ∩ under-D, intersected on GPU0 below), projecting all six lineitem columns the
  // composite join needs. Shard cols: 0 orderkey,1 partkey,2 suppkey,3 extprice,4 discount,5 qty.
  auto project = [](cudf::table_view lv, cudf::column_view hk,
                    rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    auto [l_map, h_map] = cudf::inner_join(cudf::table_view{{lv.column(1)}}, cudf::table_view{{hk}},
                                           cudf::null_equality::EQUAL, s);
    return cudf::gather(cudf::table_view{{lv.column(0), lv.column(1), lv.column(2), lv.column(3),
                                          lv.column(4), lv.column(5)}},
                        map_view(l_map), cudf::out_of_bounds_policy::DONT_CHECK, s);
  };

  StageProfiler prof;
  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    prof.begin();
    auto hits = sharded_range_hits(pool, searchers, probe, K, "q9v-mgpu", &prof);
    auto d_hits = to_device_hits(hits);
    auto sel_partkeys = cudf::gather(cudf::table_view{{pk}}, d_hits.view());

    // q9's part predicate is TWO-fold: under D AND p_name LIKE '%green%'. Intersect on GPU0 to get
    // the effective hit key set, then broadcast it for the partitioned lineitem semi-join.
    auto green = cudf::string_scalar(std::string("green"));
    auto green_mask = cudf::strings::contains(cudf::strings_column_view(pname), green);
    auto green_parts = cudf::apply_boolean_mask(cudf::table_view{{pk}}, green_mask->view());
    EXPECT_GT(green_parts->num_rows(), 0);
    auto [g_map, v_map] = cudf::inner_join(cudf::table_view{{green_parts->get_column(0).view()}},
                                           cudf::table_view{{sel_partkeys->get_column(0).view()}});
    auto part_sel = cudf::gather(cudf::table_view{{green_parts->get_column(0).view()}},
                                 map_view(g_map));
    EXPECT_GT(part_sel->num_rows(), 0) << "the two part predicates have no overlap";
    prof.tick("2b_green_intersect");  // GPU0: p_name LIKE %green% over 8M ∩ under-D keys
    prof.rowcount("green_and_underD_keys", static_cast<long>(part_sel->num_rows()));

    // parallel lineitem semi-join -> survivors gathered to GPU0 (== single-GPU part⋈lineitem `li`)
    auto li = semijoin_gather(pool, line_shards, part_sel->view(), project, &prof);
    EXPECT_GT(li->num_rows(), 0);

    auto [li_map, psx_map] = cudf::inner_join(
        cudf::table_view{{li->get_column(1).view(), li->get_column(2).view()}},
        cudf::table_view{{ps_in.tbl->view().column(0), ps_in.tbl->view().column(1)}});
    auto li_j = cudf::gather(cudf::table_view{{li->get_column(0).view(), li->get_column(2).view(),
                                               li->get_column(3).view(), li->get_column(4).view(),
                                               li->get_column(5).view()}},
                             map_view(li_map));
    auto cost_j = cudf::gather(cudf::table_view{{ps_in.tbl->view().column(2)}}, map_view(psx_map));
    EXPECT_EQ(li_j->num_rows(), li->num_rows())
        << "composite join changed the lineitem row count — (ps_partkey, ps_suppkey) is unique in "
           "partsupp, so an inner join on both keys must preserve it exactly.";
    prof.tick("5_dim_partsupp");  // GPU0: composite (partkey,suppkey) join vs 32M-row partsupp
    prof.rowcount("after_partsupp", static_cast<long>(li_j->num_rows()));

    auto [lo_map, o_map] = cudf::inner_join(cudf::table_view{{li_j->get_column(0).view()}},
                                            cudf::table_view{{ord_in.tbl->view().column(0)}});
    auto li_o = cudf::gather(cudf::table_view{{li_j->get_column(1).view(), li_j->get_column(2).view(),
                                               li_j->get_column(3).view(), li_j->get_column(4).view()}},
                             map_view(lo_map));
    auto cost_o = cudf::gather(cudf::table_view{{cost_j->get_column(0).view()}}, map_view(lo_map));
    auto date_o = cudf::gather(cudf::table_view{{ord_in.tbl->view().column(1)}}, map_view(o_map));
    EXPECT_GT(li_o->num_rows(), 0);
    prof.tick("5b_dim_orders");  // GPU0: join vs 60M-row orders
    prof.rowcount("after_orders", static_cast<long>(li_o->num_rows()));

    auto [ls_map, s_map] = cudf::inner_join(cudf::table_view{{li_o->get_column(0).view()}},
                                            cudf::table_view{{sup_in.tbl->view().column(0)}});
    auto vals_s = cudf::gather(cudf::table_view{{li_o->get_column(1).view(), li_o->get_column(2).view(),
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
    EXPECT_GT(vals_n->num_rows(), 0);
    prof.tick("5c_dim_supp_nation");  // GPU0: supplier + nation joins (small dims)
    prof.rowcount("group_in", static_cast<long>(vals_n->num_rows()));

    auto o_year = cudf::datetime::extract_datetime_component(
        date_n->get_column(0).view(), cudf::datetime::datetime_component::YEAR);
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

    cudf::groupby::groupby gb(cudf::table_view{{name_n->get_column(0).view(), o_year->view()}});
    std::vector<cudf::groupby::aggregation_request> reqs;
    {
      cudf::groupby::aggregation_request r;
      r.values = amount->view();
      r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      reqs.push_back(std::move(r));
    }
    auto [gk, ga] = gb.aggregate(reqs);
    auto profit = std::move(ga[0].results[0]);
    auto gkv = gk->view();
    auto order = cudf::sorted_order(cudf::table_view{{gkv.column(0), gkv.column(1)}},
                                    {cudf::order::ASCENDING, cudf::order::DESCENDING},
                                    {cudf::null_order::AFTER, cudf::null_order::AFTER});
    auto res = cudf::gather(cudf::table_view{{gkv.column(0), gkv.column(1), profit->view()}},
                            order->view());
    prof.tick("6_groupby_sort");
    prof.rowcount("groups", static_cast<long>(gk->num_rows()));
    return res;
  };

  auto count_hits = sharded_range_hits(pool, searchers, probe, K, "q9v-mgpu");
  expect_hit_count(count_hits.size(),
                   golden_dir() + "/duckdb_ptext_" + probe.id + ".count.csv", "q9v-mgpu", probe.id);
  ASSERT_GT(count_hits.size(), 0u) << "no part rows under D — test would be vacuous";

  auto sorted = execute();
  const std::vector<ColSpec> spec = {
      {"nation", Cmp::ExactString},
      {"o_year", Cmp::ExactInt},
      {"sum_profit", Cmp::ExactDecimal},
  };
  const auto golden = read_csv_golden(golden_path);
  ASSERT_GT(golden.size(), 0u) << "golden is empty: " << golden_path;
  compare_table_to_golden(sorted->view(), golden, spec, "q9v-mgpu");
  std::fprintf(stderr, "[q9v-mgpu] %ld result rows matched\n", static_cast<long>(sorted->num_rows()));

  benchmark_mgpu("q9v-mgpu", execute, G, index_ms);
  if (profile_enabled()) {
    prof.on = true;
    prof.num_gpus = G;
    for (int i = 0; i < 6; ++i) { auto r = execute(); (void)r; }
    prof.report("q9v-mgpu", G);
  }
  release_searchers(pool, searchers);
  release_partitions(pool, line_shards);
}

// Same entry point as the other multi-GPU binary; registers the ONE shared WorkerPool.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new peacock_mgpu::MultiGpuEnvironment);
  return RUN_ALL_TESTS();
}
