// Non-template definitions for the multi-GPU test scaffolding declared in multi_gpu.hpp.
// Body of the TEST-ONLY static library multi_gpu_testlib: not installed, not in CI.

#include "multi_gpu.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>

#include <rmm/aligned.hpp>

namespace peacock_mgpu {

namespace {
// round DOWN to the RMM allocation alignment (pool sizes must be aligned).
std::size_t align_down(std::size_t n) {
  return n - (n % rmm::CUDA_ALLOCATION_ALIGNMENT);
}
}  // namespace

using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;
struct WorkerSetup {
  rmm::mr::cuda_memory_resource* upstream;
  pool_mr*                       pool;
  rmm::cuda_stream*              stream;
};

WorkerPool::WorkerPool(int num_gpus) {
  workers_.reserve(num_gpus);
  upstream_.resize(num_gpus);
  pools_.resize(num_gpus);
  streams_.resize(num_gpus);
  for (int g = 0; g < num_gpus; ++g) workers_.push_back(std::make_unique<GpuWorker>(g));

  for (int g = 0; g < num_gpus; ++g) {
    // Peer access + pool install + persistent stream, all on device g's own worker.
    auto s = workers_[g]
                 ->submit([g, num_gpus]() -> WorkerSetup {
                   for (int p = 0; p < num_gpus; ++p) {
                     if (p == g) continue;
                     int can = 0;
                     cudaDeviceCanAccessPeer(&can, g, p);
                     if (!can) continue;
                     cudaError_t e = cudaDeviceEnablePeerAccess(p, 0);
                     if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                   }
                   // Size the pool off THIS device's free memory (never hardcode). 85% initial
                   // so a query's whole working set fits without a mid-query growth event
                   // (another cudaMalloc — the very sync we are avoiding); 95% ceiling.
                   std::size_t free = 0, total = 0;
                   MG_CUDA_TRY(cudaMemGetInfo(&free, &total));
                   const std::size_t initial = align_down(free * 85 / 100);
                   const std::size_t maximum = align_down(free * 95 / 100);
                   auto* up   = new rmm::mr::cuda_memory_resource();
                   auto* pool = new pool_mr(up, initial, maximum);  // reserves on device g
                   rmm::mr::set_per_device_resource(rmm::cuda_device_id{g}, pool);  // syncs the ref too
                   auto* stream = new rmm::cuda_stream();  // persistent, on device g
                   return WorkerSetup{up, pool, stream};
                 })
                 .get();
    upstream_[g].reset(s.upstream);
    pools_[g].reset(s.pool);
    streams_[g].reset(s.stream);
  }
}

WorkerPool::~WorkerPool() {
  // On each worker (device current): drain the device, reset its RMM resource OFF the pool
  // (back to RMM's initial cuda resource) BEFORE destroying the pool + upstream + stream — so
  // the RMM per-device map never dangles and the pool's buffer is freed under the owning device.
  for (int g = 0; g < static_cast<int>(workers_.size()); ++g) {
    workers_[g]
        ->submit([this, g] {
          cudaDeviceSynchronize();
          // FOOTGUN: set_per_device_resource(id, nullptr) resets ONLY the pointer map — its
          // internal ref-sync is guarded by `if (new_mr != nullptr)`, so it LEAVES the ref map
          // pointing at the about-to-be-destroyed pool, and a later WorkerPool in the same
          // process inherits a dangling ref until it reinstalls (surfaces as "invalid device
          // ordinal"). Both maps must therefore be reset to RMM's static initial resource.
          rmm::mr::set_per_device_resource(rmm::cuda_device_id{g}, nullptr);   // pointer -> initial
          rmm::mr::reset_per_device_resource_ref(rmm::cuda_device_id{g});      // ref     -> initial
          pools_[g].reset();     // pool dtor frees the reserved buffer to upstream, on device g
          upstream_[g].reset();
          streams_[g].reset();   // destroy the persistent stream last, after all frees on it
        })
        .get();
  }
  // Members now hold nullptrs; workers_ join as the object is destroyed.
}

int parquet_num_row_groups(std::string const& path) {
  auto meta = cudf::io::read_parquet_metadata(
      cudf::io::source_info{std::vector<std::string>{path}});
  return meta.num_rowgroups();
}

std::vector<std::vector<cudf::size_type>> partition_row_groups(int num_row_groups, int num_gpus) {
  std::vector<std::vector<cudf::size_type>> out(num_gpus);
  const int base = num_row_groups / num_gpus;
  const int rem  = num_row_groups % num_gpus;
  int idx        = 0;
  for (int g = 0; g < num_gpus; ++g) {
    const int cnt = base + (g < rem ? 1 : 0);
    for (int k = 0; k < cnt; ++k) out[g].push_back(idx++);
  }
  return out;
}

cudf::io::table_with_metadata read_row_group_span(std::string const& path,
                                                  std::vector<std::string> columns,
                                                  std::vector<cudf::size_type> const& row_groups,
                                                  rmm::cuda_stream_view stream) {
  auto opts = cudf::io::parquet_reader_options::builder(
                  cudf::io::source_info{std::vector<std::string>{path}})
                  .columns(std::move(columns))
                  .build();
  opts.set_row_groups({row_groups});  // one inner vector = row groups from source 0
  return cudf::io::read_parquet(opts, stream);
}

cudf::io::table_with_metadata read_full_table(std::string const& path,
                                              std::vector<std::string> columns,
                                              rmm::cuda_stream_view stream) {
  auto opts = cudf::io::parquet_reader_options::builder(
                  cudf::io::source_info{std::vector<std::string>{path}})
                  .columns(std::move(columns))
                  .build();
  return cudf::io::read_parquet(opts, stream);
}

PackedPartial describe_packed(int device, cudf::packed_columns const& p) {
  return PackedPartial{device, p.metadata->data(),
                       static_cast<const uint8_t*>(p.gpu_data->data()), p.gpu_data->size()};
}

GatheredTable gather_here(PackedPartial const& p, rmm::cuda_stream_view stream) {
  rmm::device_buffer buf(p.gpu_data_size, stream);
  int cur = 0;
  MG_CUDA_TRY(cudaGetDevice(&cur));
  MG_CUDA_TRY(cudaMemcpyPeerAsync(buf.data(), cur, p.gpu_data, p.device, p.gpu_data_size,
                                  stream.value()));
  MG_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  cudf::table_view view = cudf::unpack(p.metadata, static_cast<const uint8_t*>(buf.data()));
  return GatheredTable{std::move(buf), view};
}

std::vector<std::unique_ptr<cudf::table>> hash_shuffle(
    WorkerPool& pool, std::vector<cudf::table_view> const& locals,
    std::vector<cudf::size_type> const& key_cols) {
  const int G = pool.size();

  // Phase 1: each worker g hash_partitions its local table into G contiguous buckets and packs
  // each one. parted[g] and packed[g][p] stay alive on worker g until the exchange copies them.
  std::vector<std::unique_ptr<cudf::table>>       parted(G);
  std::vector<std::vector<cudf::packed_columns>>  packed(G);
  std::vector<std::vector<PackedPartial>>         handles(G, std::vector<PackedPartial>(G));
  {
    std::vector<std::future<void>> fs;
    for (int g = 0; g < G; ++g)
      fs.push_back(pool[g].submit([&, g] {
        auto s   = pool.stream(g);
        auto res = cudf::hash_partition(locals[g], key_cols, G, cudf::hash_id::HASH_MURMUR3,
                                        cudf::DEFAULT_HASH_SEED, s);
        parted[g]          = std::move(res.first);
        auto const& starts = res.second;  // size G: start row of each bucket
        auto full          = parted[g]->view();
        const cudf::size_type nrows = full.num_rows();
        packed[g].resize(G);
        for (int p = 0; p < G; ++p) {
          const cudf::size_type start = starts[p];
          const cudf::size_type end   = (p + 1 < G) ? starts[p + 1] : nrows;
          auto bucket   = cudf::slice(full, {start, end})[0];  // zero-copy view of bucket p
          packed[g][p]  = cudf::pack(bucket, s);
          handles[g][p] = describe_packed(g, packed[g][p]);
        }
        s.synchronize();  // buckets fully materialized before any peer reads them
      }));
    for (auto& f : fs) f.get();
  }

  // Phase 2: each worker p pulls bucket p from every worker g and concatenates -> result[p].
  std::vector<std::unique_ptr<cudf::table>> result(G);
  {
    std::vector<std::future<void>> fs;
    for (int p = 0; p < G; ++p)
      fs.push_back(pool[p].submit([&, p] {
        auto s = pool.stream(p);
        std::vector<GatheredTable> gathered;
        gathered.reserve(G);
        std::vector<cudf::table_view> views;
        views.reserve(G);
        for (int g = 0; g < G; ++g) {
          gathered.push_back(gather_here(handles[g][p], s));
          views.push_back(gathered[g].view);
        }
        result[p] = cudf::concatenate(views, s);
        s.synchronize();  // result[p] owns its data before `gathered` buffers free here
      }));
    for (auto& f : fs) f.get();
  }

  // Phase 3: release the packed source tables on their producing workers (after the exchange).
  {
    std::vector<std::future<void>> fs;
    for (int g = 0; g < G; ++g)
      fs.push_back(pool[g].submit([&, g] {
        packed[g].clear();
        parted[g].reset();
      }));
    for (auto& f : fs) f.get();
  }
  return result;
}

}  // namespace peacock_mgpu
