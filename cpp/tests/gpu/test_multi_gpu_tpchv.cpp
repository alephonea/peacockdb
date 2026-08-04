// The same TPC-H+V vector queries as test_tpchv.cpp, with the cuVS brute-force SEARCH
// sharded across all visible GPUs, checked against the same committed DuckDB goldens.
//
// The search is the parallel win: the distance scan over millions of embeddings is
// embarrassingly parallel; the relational tail around the O(1e5) hit list is small.
// The embedding table is sharded on whole parquet row-group boundaries, one cuVS index
// per shard.
//
// Why sharding is EXACT: a globally-top-K point is top-K within its own shard, so the
// union of per-shard top-K lists contains the true global top-K; merge and range-filter
// by D. The saturation guard generalizes globally (see sharded_range_hits). Shards are
// contiguous row-group spans in natural parquet order, so shard-local index i maps to
// global index row_offset[g] + i — the same row index the single-GPU search returns,
// leaving the relational tail unchanged.
//
// Sharding also sidesteps the int32 list-child ceiling: each shard's child stays under
// the cap (the loader still batches within a span so the G=1 baseline works too).
//
// MANUAL sharding, not cuvs::neighbors::mg (SNMG): cross-GPU movement stays explicit and
// consistent with the rest of the suite. cuVS/raft allocate from the per-device RMM pool;
// the q11v test asserts this empirically. cuVS linking: see test_tpchv.cpp (cudf must
// link before cuvs). One process-wide WorkerPool, owned by MultiGpuEnvironment.

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

// Probe loading — identical to test_tpchv.cpp: probes come from the committed
// query_params.jsonl, never hardcoded, so golden and GPU side cannot drift.
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

// K for top-K-then-filter, requested from EACH shard. Overridable (PEACOCK_TPCHV_K=64)
// only so the saturation guard can be exercised; default matches single-GPU.
int64_t search_k() {
  return std::strtoll(env_or("PEACOCK_TPCHV_K", "131072").c_str(), nullptr, 10);
}

// Stage profiler, env-gated by PEACOCK_PROFILE; every method is a no-op when off, so the
// correctness/benchmark path is unchanged. Stage boundaries need all-device syncs, which
// serialize otherwise-overlapping work — the summed breakdown is an UPPER bound on the
// clean benchmark total. report() takes the 2nd-min per stage.
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

// Shard embedding loader — reads one row-group span's embedding column into a resident
// row-major float matrix on the CURRENT device. Batches within the span so no read's list
// child exceeds the int32 ceiling (matters for the G=1 baseline). The buffer is assembled
// from the batches, not presized — parquet does not expose per-row-group counts.
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

  // read in cap-bounded batches; assemble once the exact total is known
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

// L2SqrtExpanded returns true L2 (DuckDB's array_distance), so D is used as-is.
// Identical to test_tpchv.cpp's build_bf_index.
inline auto build_bf_index(raft::resources& handle, const float* data, int64_t rows, int dim) {
  auto dataset = raft::make_device_matrix_view<const float, int64_t>(data, rows, dim);
  cuvs::neighbors::brute_force::index_params ip;
  ip.metric = cuvs::distance::DistanceType::L2SqrtExpanded;
  return cuvs::neighbors::brute_force::build(handle, ip, dataset);
}

// ShardSearcher type-erases the per-GPU raft handle, shard matrix and cuVS index — device
// objects that MUST live and die on their owning worker thread — behind the `search`
// closure. The captured shared_ptrs keep them alive; clearing `search` on the worker at
// teardown drops the last reference there, never on the main thread.
struct ShardSearcher {
  int64_t rows = 0;           // rows in this shard (0 => empty span, skipped)
  int64_t global_offset = 0;  // number of rows in all earlier shards (natural parquet order)
  std::function<void(VecProbe const&, int64_t, std::vector<int64_t>&, std::vector<float>&)> search;
};

// Free VRAM on the current device — a from-pool allocation does not change it (the pool
// reserved up front); a stray cudaMalloc outside the pool would drop it.
inline size_t free_vram() {
  size_t f = 0, t = 0;
  cudaMemGetInfo(&f, &t);
  return f;
}

// Build one shard index per worker; return searchers (global offsets filled) and the max
// per-worker build time. Asserts shards reassemble to `expect_rows` and each non-empty
// shard has >= K rows (cuVS requires K <= shard size).
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

// The sharded search operator (timed, inside execute): search every shard for K, merge on
// the host, range-filter by D. Returns global row indices (natural parquet order) — the
// same set single-GPU vector_range_hits returns.
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

  // Merge is O(#hits), not O(G*K): collect only the under-D entries from each shard's
  // top-K list.
  // Saturation guard by COUNT: hits.size() < K  <=>  global K-th distance >= D  <=>  every
  // shard's top-K captured ALL its under-D points, so the union is the exact global set.
  // If any shard saturated it alone contributes K hits and the guard fails — correct,
  // since that shard may hide under-D points beyond its K.
  std::vector<int32_t> hits;
  for (int g = 0; g < G; ++g)
    for (size_t i = 0; i < idx[g].size(); ++i)
      if (static_cast<double>(dist[g][i]) < probe.D)
        hits.push_back(static_cast<int32_t>(sh[g].global_offset + idx[g][i]));

  EXPECT_LT(static_cast<int64_t>(hits.size()), K)
      << tag << "/" << probe.id << ": top-K SATURATED — " << hits.size()
      << " points fall under D across the per-shard top-K lists (>= K=" << K
      << "), so a shard's top-K may have truncated the range. Raise K.";

  if (prof) {
    prof->tick("2_search_merge");  // per-shard <D scan + count guard (O(hits), not O(G*K))
    prof->rowcount("hits_under_D", static_cast<long>(hits.size()));
  }
  std::fprintf(stderr, "[%s] %s: D=%.17g -> %zu rows under D\n", tag, probe.id.c_str(), probe.D,
               hits.size());
  return hits;
}

// Count corroboration — verify path only (reads a golden, kept out of the timed region).
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

// read columns from one parquet file on the CURRENT device — no predicate, no row selection
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

// stream-taking variant — off GPU0 every cudf op needs a device-local stream
cudf::timestamp_scalar<cudf::timestamp_D> date_scalar(int y, unsigned mo, unsigned d,
                                                      rmm::cuda_stream_view s) {
  return cudf::timestamp_scalar<cudf::timestamp_D>(
      cudf::timestamp_D{cudf::duration_D{days_since_epoch(y, mo, d)}}, true, s);
}

// broadcast_join_gather: broadcast a small intermediate (resident on GPU0) to every GPU,
// run `join_fn` against each GPU's row-group shard of a big table, gather the small
// results back to GPU0 and concatenate.
// - The broadcast packs once on GPU0 and peer-copies via gather_here, keeping the
//   column's NATIVE type so semi-join key types match (a host int32 round-trip once
//   silently mistyped int64 keys).
// - `join_fn` must thread the device-local stream through every cudf op (required off
//   GPU0).
// - Correctness for unique-key joins: the key is unique and the spans disjoint, so each
//   probe key matches at most one row on exactly one shard — gathering matches across
//   shards is the full join, no double-count.
// `scan_label`/`gather_label` name the profiler stages.
std::unique_ptr<cudf::table> broadcast_join_gather(
    WorkerPool& pool, std::vector<cudf::io::table_with_metadata>& shards, cudf::table_view bcast,
    std::function<std::unique_ptr<cudf::table>(cudf::table_view, cudf::table_view,
                                               rmm::cuda_stream_view)> const& join_fn,
    StageProfiler* prof = nullptr, const char* scan_label = nullptr,
    const char* gather_label = nullptr) {
  const int G = pool.size();
  auto packed_bcast = cudf::pack(bcast, cudf::get_default_stream());
  cudf::get_default_stream().synchronize();
  const PackedPartial bcast_handle = describe_packed(0, packed_bcast);

  std::vector<std::unique_ptr<cudf::table>> res(G);
  std::vector<cudf::packed_columns> packed(G);

  std::vector<std::future<PackedPartial>> pf;
  for (int g = 0; g < G; ++g)
    pf.push_back(pool[g].submit([&, g]() -> PackedPartial {
      auto s    = pool.stream(g);
      auto bt   = gather_here(bcast_handle, s);  // broadcast intermediate, now on device g
      res[g]    = join_fn(shards[g].tbl->view(), bt.view, s);
      packed[g] = cudf::pack(res[g]->view(), s);
      s.synchronize();
      return describe_packed(g, packed[g]);
    }));
  std::vector<PackedPartial> handles(G);
  for (int g = 0; g < G; ++g) handles[g] = pf[g].get();
  if (prof && scan_label) prof->tick(scan_label);

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
  if (prof && gather_label) prof->tick(gather_label);

  std::vector<std::future<void>> rf;
  for (int g = 0; g < G; ++g)
    rf.push_back(pool[g].submit([&, g] {
      packed[g] = cudf::packed_columns{};
      res[g].reset();
    }));
  for (auto& f : rf) f.get();
  return out;
}

// Load a table's row-group shards resident on each worker once, in the LOAD phase, so the
// per-probe execute only does compute over them.
std::vector<cudf::io::table_with_metadata> load_table_shards(
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
  // Scalar columns on GPU0; embedding sharded across all GPUs + per-shard index (setup,
  // timed separately).
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

  // Pool-draw proof (q11v only): a search must not change free VRAM on any worker — a
  // change would mean cuVS allocated outside the per-device pool.
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
  // part-key on GPU0; embedding sharded for the search; lineitem AND orders partitioned
  // across the GPUs (the 60M orders join dominated the serial remainder).
  auto ord_shards = load_table_shards(pool, data_dir() + "/orders.parquet",
                                      {"o_orderkey", "o_orderpriority"});
  const auto part_path = data_dir() + "/part.parquet";
  auto part_in = read_cols(part_path, {"p_partkey"});
  const int64_t n_part = part_in.tbl->num_rows();
  const int64_t K = search_k();
  ASSERT_LE(K, n_part);
  double index_ms = 0.0;
  auto searchers = build_sharded_index(pool, part_path, "p_text_embedding", 100, K, n_part,
                                       index_ms, "q12v-mgpu");
  auto line_shards = load_table_shards(
      pool, data_dir() + "/lineitem.parquet",
      {"l_orderkey", "l_partkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"});

  auto pk = part_in.tbl->view().column(0);

  // Worker-side project: filter the lineitem shard (shipmode + 3 date predicates), then
  // semi-join the broadcast part-hit keys, projecting (l_orderkey, l_shipmode).
  // Shard cols: 0 orderkey,1 partkey,2 shipmode,3 commit,4 receipt,5 ship.
  auto project = [](cudf::table_view lv, cudf::table_view hkt,
                    rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    auto hk = hkt.column(0);  // broadcast single-column hit-key table
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

  // Each GPU joins its orders shard against the broadcast lineitem survivors on
  // o_orderkey (unique -> 1 match on one shard), producing (l_shipmode, o_orderpriority).
  auto orders_join = [](cudf::table_view ordv /*o_orderkey,o_orderpriority*/,
                        cudf::table_view lpv /*l_orderkey,l_shipmode*/,
                        rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    auto [o_map, l_map] = cudf::inner_join(cudf::table_view{{ordv.column(0)}},
                                           cudf::table_view{{lpv.column(0)}},
                                           cudf::null_equality::EQUAL, s);
    auto sm = cudf::gather(cudf::table_view{{lpv.column(1)}}, map_view(l_map),
                           cudf::out_of_bounds_policy::DONT_CHECK, s);  // l_shipmode
    auto pr = cudf::gather(cudf::table_view{{ordv.column(1)}}, map_view(o_map),
                           cudf::out_of_bounds_policy::DONT_CHECK, s);  // o_orderpriority
    std::vector<std::unique_ptr<cudf::column>> out;
    auto smc = sm->release();
    auto prc = pr->release();
    out.push_back(std::move(smc[0]));
    out.push_back(std::move(prc[0]));
    return std::make_unique<cudf::table>(std::move(out));
  };

  StageProfiler prof;  // no-op unless PEACOCK_PROFILE; ticks below cost nothing off-profile
  auto execute = [&]() -> std::unique_ptr<cudf::table> {
    prof.begin();
    // sharded search -> hit part keys on GPU0 -> parallel lineitem filter+semijoin
    auto hits = sharded_range_hits(pool, searchers, probe, K, "q12v-mgpu", &prof);
    auto d_hits = to_device_hits(hits);
    auto sel = cudf::gather(cudf::table_view{{pk}}, d_hits.view());  // the hit part keys, on GPU0
    auto lp = broadcast_join_gather(pool, line_shards, sel->view(), project, &prof, "3_tail_scan",
                                    "4_gather_survivors");  // (l_orderkey, l_shipmode)
    prof.rowcount("survivors_total", static_cast<long>(lp->num_rows()));
    EXPECT_GT(lp->num_rows(), 0) << "vector predicate and filters left no rows to join";

    // partitioned orders join -> (l_shipmode, o_orderpriority)
    auto mp = broadcast_join_gather(pool, ord_shards, lp->view(), orders_join, &prof, "5_dim_orders",
                                    "5_dim_orders_gather");
    prof.rowcount("after_orders", static_cast<long>(mp->num_rows()));
    auto mode = mp->view().column(0);  // l_shipmode
    auto prio = mp->view().column(1);  // o_orderpriority

    auto urgent = cudf::string_scalar(std::string("1-URGENT"));
    auto high = cudf::string_scalar(std::string("2-HIGH"));
    auto is_urgent = cudf::binary_operation(prio, urgent, cudf::binary_operator::EQUAL, boolean);
    auto is_high = cudf::binary_operation(prio, high, cudf::binary_operator::EQUAL, boolean);
    auto is_hi = cudf::binary_operation(is_urgent->view(), is_high->view(),
                                        cudf::binary_operator::LOGICAL_OR, boolean);
    auto is_lo = cudf::unary_operation(is_hi->view(), cudf::unary_operator::NOT);
    auto hi_i = cudf::cast(is_hi->view(), int32);
    auto lo_i = cudf::cast(is_lo->view(), int32);

    cudf::groupby::groupby gb(cudf::table_view{{mode}});
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
  release_partitions(pool, ord_shards);
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
  auto line_shards = load_table_shards(
      pool, data_dir() + "/lineitem.parquet",
      {"l_orderkey", "l_partkey", "l_extendedprice", "l_discount", "l_returnflag"});

  auto cv = cust_in.tbl->view();
  auto ov = ord_in.tbl->view();
  auto pk = part_in.tbl->view().column(0);

  // Worker-side project: filter the lineitem shard (l_returnflag='R') + SEMI-JOIN part-hit keys,
  // projecting (l_orderkey, l_extendedprice, l_discount). Shard cols: 0 orderkey,1 partkey,2
  // extprice,3 discount,4 returnflag.
  // Worker-side project: filter l_returnflag='R' then SEMI-JOIN the hit part keys, projecting
  // (l_orderkey, l_extendedprice, l_discount). Shard cols: 0 orderkey,1 partkey,2 extprice,3
  // discount,4 returnflag.
  // NOTE (Lever C evaluated + REVERTED): the profiler showed this returnflag-first order's tail
  // scales only 1.09x (it materializes a ~59M-row intermediate that barely halves). Reordering to
  // semi-join FIRST does make the tail scale 1.95x — BUT the semi-join then probes the full 240M
  // lineitem instead of the 59M post-filter, which costs MORE per-GPU: measured, semi-join-first is
  // ~4ms SLOWER at G=2 (39.7 vs 35.5 ms) even though its G=1-vs-G=2 RATIO looks better (that ratio
  // is inflated by a slower G=1). So we keep returnflag-first for the better G=2 wall-clock; q10v's
  // ~1.08x is the honest residual — it is bound by touching a large lineitem subset either way.
  auto project = [](cudf::table_view lv, cudf::table_view hkt,
                    rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    auto hk = hkt.column(0);  // broadcast single-column hit-key table
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
    auto lp = broadcast_join_gather(pool, line_shards, sel->view(), project, &prof, "3_tail_scan",
                                    "4_gather_survivors");  // (orderkey, extprice, discount)
    prof.rowcount("survivors_total", static_cast<long>(lp->num_rows()));

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
  // part(key+name), supplier + nation (small) on GPU0; embedding SHARDED for the search; lineitem
  // (M4) AND partsupp+orders (M5 Lever B — they were 70% of q9v's serial remainder, 23ms) all
  // PARTITIONED across the GPUs on row groups.
  const auto part_path = data_dir() + "/part.parquet";
  auto part_name_in = read_cols(part_path, {"p_partkey", "p_name"});
  auto sup_in = read_cols(data_dir() + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  auto nat_in = read_cols(data_dir() + "/nation.parquet", {"n_nationkey", "n_name"});
  const int64_t n_part = part_name_in.tbl->num_rows();
  const int64_t K = search_k();
  ASSERT_LE(K, n_part);
  double index_ms = 0.0;
  auto searchers = build_sharded_index(pool, part_path, "p_text_embedding", 100, K, n_part,
                                       index_ms, "q9v-mgpu");
  auto line_shards = load_table_shards(
      pool, data_dir() + "/lineitem.parquet",
      {"l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount", "l_quantity"});
  auto ps_shards = load_table_shards(pool, data_dir() + "/partsupp.parquet",
                                     {"ps_partkey", "ps_suppkey", "ps_supplycost"});
  auto ord_shards = load_table_shards(pool, data_dir() + "/orders.parquet",
                                      {"o_orderkey", "o_orderdate"});

  auto pk = part_name_in.tbl->view().column(0);
  auto pname = part_name_in.tbl->view().column(1);

  // Worker-side project: SEMI-JOIN the lineitem shard against the broadcast hit keys (q9's hit set
  // is green ∩ under-D, intersected on GPU0 below), projecting all six lineitem columns the
  // composite join needs. Shard cols: 0 orderkey,1 partkey,2 suppkey,3 extprice,4 discount,5 qty.
  auto project = [](cudf::table_view lv, cudf::table_view hkt,
                    rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    auto hk = hkt.column(0);  // broadcast single-column hit-key table
    auto [l_map, h_map] = cudf::inner_join(cudf::table_view{{lv.column(1)}}, cudf::table_view{{hk}},
                                           cudf::null_equality::EQUAL, s);
    return cudf::gather(cudf::table_view{{lv.column(0), lv.column(1), lv.column(2), lv.column(3),
                                          lv.column(4), lv.column(5)}},
                        map_view(l_map), cudf::out_of_bounds_policy::DONT_CHECK, s);
  };

  // Lever B join_fn #1 — each GPU COMPOSITE-joins its partsupp shard against the broadcast lineitem
  // survivors on (l_partkey,l_suppkey)==(ps_partkey,ps_suppkey) (unique -> 1 match on one shard),
  // producing (l_orderkey, l_suppkey, extprice, discount, quantity, ps_supplycost).
  auto ps_join = [](cudf::table_view psv /*ps_partkey,ps_suppkey,ps_supplycost*/,
                    cudf::table_view liv /*orderkey,partkey,suppkey,extprice,discount,quantity*/,
                    rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    auto [li_map, ps_map] = cudf::inner_join(
        cudf::table_view{{liv.column(1), liv.column(2)}},
        cudf::table_view{{psv.column(0), psv.column(1)}}, cudf::null_equality::EQUAL, s);
    auto lic = cudf::gather(cudf::table_view{{liv.column(0), liv.column(2), liv.column(3),
                                              liv.column(4), liv.column(5)}},
                            map_view(li_map), cudf::out_of_bounds_policy::DONT_CHECK, s);
    auto cost = cudf::gather(cudf::table_view{{psv.column(2)}}, map_view(ps_map),
                             cudf::out_of_bounds_policy::DONT_CHECK, s);
    std::vector<std::unique_ptr<cudf::column>> out;
    for (auto& c : lic->release()) out.push_back(std::move(c));
    out.push_back(std::move(cost->release()[0]));
    return std::make_unique<cudf::table>(std::move(out));  // orderkey,suppkey,price,disc,qty,cost
  };

  // Lever B join_fn #2 — each GPU joins its orders shard against the broadcast (post-partsupp)
  // survivors on o_orderkey (unique), producing (l_suppkey, price, disc, qty, cost, o_orderdate).
  auto ord_join = [](cudf::table_view ordv /*o_orderkey,o_orderdate*/,
                     cudf::table_view lpv /*orderkey,suppkey,price,disc,qty,cost*/,
                     rmm::cuda_stream_view s) -> std::unique_ptr<cudf::table> {
    auto [o_map, l_map] = cudf::inner_join(cudf::table_view{{ordv.column(0)}},
                                           cudf::table_view{{lpv.column(0)}},
                                           cudf::null_equality::EQUAL, s);
    auto lpc = cudf::gather(cudf::table_view{{lpv.column(1), lpv.column(2), lpv.column(3),
                                              lpv.column(4), lpv.column(5)}},
                            map_view(l_map), cudf::out_of_bounds_policy::DONT_CHECK, s);
    auto dt = cudf::gather(cudf::table_view{{ordv.column(1)}}, map_view(o_map),
                           cudf::out_of_bounds_policy::DONT_CHECK, s);
    std::vector<std::unique_ptr<cudf::column>> out;
    for (auto& c : lpc->release()) out.push_back(std::move(c));
    out.push_back(std::move(dt->release()[0]));
    return std::make_unique<cudf::table>(std::move(out));  // suppkey,price,disc,qty,cost,orderdate
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
    auto li = broadcast_join_gather(pool, line_shards, part_sel->view(), project, &prof,
                                    "3_tail_scan", "4_gather_survivors");
    prof.rowcount("survivors_total", static_cast<long>(li->num_rows()));
    EXPECT_GT(li->num_rows(), 0);

    // Lever B round 1: PARTITIONED composite join vs partsupp (replaces the 10.6ms GPU0 scan).
    auto li_ps = broadcast_join_gather(pool, ps_shards, li->view(), ps_join, &prof, "5_dim_partsupp",
                                       "5_dim_partsupp_gather");  // orderkey,suppkey,price,disc,qty,cost
    EXPECT_EQ(li_ps->num_rows(), li->num_rows())
        << "composite join changed the row count — (ps_partkey, ps_suppkey) is unique in partsupp, "
           "so the partitioned inner join on both keys must preserve the survivor count exactly.";
    prof.rowcount("after_partsupp", static_cast<long>(li_ps->num_rows()));

    // Lever B round 2: PARTITIONED orders join (replaces the 12.5ms GPU0 scan).
    auto li_ord = broadcast_join_gather(pool, ord_shards, li_ps->view(), ord_join, &prof,
                                        "5b_dim_orders", "5b_dim_orders_gather");  // suppkey,price,disc,qty,cost,orderdate
    EXPECT_GT(li_ord->num_rows(), 0);
    prof.rowcount("after_orders", static_cast<long>(li_ord->num_rows()));

    // supplier + nation joins stay on GPU0 (small: 400k supplier, 25 nation).
    auto lov = li_ord->view();  // 0 suppkey,1 price,2 disc,3 qty,4 cost,5 orderdate
    auto [ls_map, s_map] = cudf::inner_join(cudf::table_view{{lov.column(0)}},
                                            cudf::table_view{{sup_in.tbl->view().column(0)}});
    auto vals_s = cudf::gather(cudf::table_view{{lov.column(1), lov.column(2), lov.column(3)}},
                               map_view(ls_map));
    auto cost_s = cudf::gather(cudf::table_view{{lov.column(4)}}, map_view(ls_map));
    auto date_s = cudf::gather(cudf::table_view{{lov.column(5)}}, map_view(ls_map));
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
  release_partitions(pool, ps_shards);
  release_partitions(pool, ord_shards);
}

// Same entry point as the other multi-GPU binary; registers the ONE shared WorkerPool.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new peacock_mgpu::MultiGpuEnvironment);
  return RUN_ALL_TESTS();
}
