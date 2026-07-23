// multi_gpu.hpp — shared scaffolding for the multi-GPU GPU tests (interface).
//
// This is the header of a TEST-ONLY static library (multi_gpu_testlib = multi_gpu.cpp +
// multi_gpu.hpp), linked into every multi-GPU test executable so they share ONE copy of:
//   * GpuWorker      — a persistent thread pinned to a GPU (the correct multi-GPU cuDF model:
//                      a device object's WHOLE lifetime, alloc->compute->free, stays on the
//                      thread whose current device it was created under; see the long note in
//                      test_basic_multi_gpu.cpp for why). TEMPLATE (submit) -> stays in header.
//   * WorkerPool     — one GpuWorker per visible GPU (sized to cudaGetDeviceCount(), never
//                      hardcoded), with all-pairs peer access enabled up front.
//   * row-group partitioning — split a parquet file's row groups across the GPUs on WHOLE
//                      row-group boundaries; each worker reads only its span. DATA
//                      DISTRIBUTION, not predicate pushdown — predicates still run as operators.
//   * broadcast read — load a whole (small) table on the calling GPU, for broadcast joins.
//   * cross-GPU gather — move a (small) cuDF table from a worker's GPU to the CURRENT GPU via
//                      pack -> peer-copy the contiguous buffer -> unpack, for a final merge.
//   * hash-shuffle — all-to-all MURMUR3 repartition of large tables by a key, so every row with
//                      a given key ends up on one GPU (for a high-cardinality group-by/join).
//
// Non-template definitions live in multi_gpu.cpp; only the GpuWorker template and small
// structs/inline accessors are here.
#pragma once

#include <cudf/contiguous_split.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace peacock_mgpu {

// A CUDA check usable INSIDE worker tasks: it THROWS rather than using gtest's ASSERT_ (whose
// `return;` cannot leave a value-returning lambda, and whose fatal-failure semantics do not
// propagate cleanly off the main thread). The packaged_task captures the exception; the main
// thread rethrows it at future.get() and gtest reports it as a normal test failure.
#define MG_CUDA_TRY(call)                                                                \
  do {                                                                                   \
    cudaError_t _s = (call);                                                             \
    if (_s != cudaSuccess)                                                               \
      throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") +             \
                               std::to_string(__LINE__) + " -> " #call " -> " +          \
                               cudaGetErrorString(_s));                                  \
  } while (0)

// ---------------------------------------------------------------------------
// GpuWorker — one persistent thread pinned to a GPU. It sets its device ONCE and then owns
// every allocation/compute/free submitted to it. submit() returns a std::future carrying the
// task's result (or any exception it threw). Members are ordered so the thread starts LAST.
// TEMPLATE submit() keeps the whole class in the header.
// ---------------------------------------------------------------------------
class GpuWorker {
 public:
  explicit GpuWorker(int device) : device_(device), worker_([this] { run(); }) {}

  ~GpuWorker() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    worker_.join();
  }

  GpuWorker(const GpuWorker&)            = delete;
  GpuWorker& operator=(const GpuWorker&) = delete;

  template <typename F>
  auto submit(F f) -> std::future<decltype(f())> {
    using R  = decltype(f());
    auto job = std::make_shared<std::packaged_task<R()>>(std::move(f));
    auto fut = job->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.emplace([job] { (*job)(); });
    }
    cv_.notify_one();
    return fut;
  }

  int device() const { return device_; }

 private:
  void run() {
    if (cudaSetDevice(device_) != cudaSuccess) cudaGetLastError();
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) return;
        job = std::move(queue_.front());
        queue_.pop();
      }
      job();
    }
  }

  int                               device_;
  std::mutex                        mu_;
  std::condition_variable           cv_;
  std::queue<std::function<void()>> queue_;
  bool                              stop_ = false;
  std::thread                       worker_;  // MUST be last (starts after the rest exist)
};

// ---------------------------------------------------------------------------
// WorkerPool — one GpuWorker per visible GPU, sized to cudaGetDeviceCount(). The constructor
// (in multi_gpu.cpp): enables all-pairs peer access, and installs a PER-DEVICE RMM POOL so
// that every cudf op on that GPU allocates from a pre-reserved pool instead of a per-op
// cudaMalloc/cudaFree (which are synchronous and driver-serialized across threads — the fixed
// overhead that otherwise stops cheap queries scaling). Each pool is created ON its worker
// thread (device current) so its reservation lives on the right GPU.
//
// PERSISTENT PER-WORKER STREAM (required by the pool): the pool's deallocate is STREAM-ORDERED,
// so a device object frees itself on the stream it was created with. A local rmm::cuda_stream
// inside a task is destroyed when the task returns, so any object that OUTLIVES the task (q1's
// partial groupbys, its packed partials, the merged result) would later deallocate on a dead
// stream -> crash. So the pool owns one persistent stream per GPU, exposed via stream(g); every
// multi-GPU-query op runs on it, and it outlives every object allocated from that GPU's pool.
// (With the default un-pooled resource, dealloc is a stream-independent cudaFree, so the PoC's
// local streams are fine — this only bites the pooled path.)
//
// LIFETIME (the footgun): set_per_device_resource stores a NON-owning reference, so on teardown
// the destructor, ON EACH WORKER THREAD: (1) syncs the device, (2) resets that device's RMM
// resource OFF the pool (back to RMM's initial cuda resource) BEFORE (3) destroying the pool,
// its upstream, and the stream — so nothing dangles and the pool's big buffer is freed under the
// right device. Everything allocated from a pool (a test's loaded partitions, its result) must
// be freed before the WorkerPool is destroyed; the tests release their per-worker state first.
// ---------------------------------------------------------------------------
class WorkerPool {
 public:
  explicit WorkerPool(int num_gpus);
  ~WorkerPool();

  WorkerPool(const WorkerPool&)            = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  int                   size() const { return static_cast<int>(workers_.size()); }
  GpuWorker&            operator[](int g) { return *workers_[g]; }
  rmm::cuda_stream_view stream(int g) const { return streams_[g]->view(); }

 private:
  using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;

  // Declared before workers_ so member destruction (reverse order) tears the threads down first;
  // the pools/streams are actually destroyed on their worker threads inside ~WorkerPool first.
  std::vector<std::unique_ptr<rmm::mr::cuda_memory_resource>> upstream_;
  std::vector<std::unique_ptr<pool_mr>>                       pools_;
  std::vector<std::unique_ptr<rmm::cuda_stream>>              streams_;
  std::vector<std::unique_ptr<GpuWorker>>                     workers_;
};

// ---------------------------------------------------------------------------
// Row-group partitioning — DATA DISTRIBUTION on WHOLE parquet row-group boundaries.
// ---------------------------------------------------------------------------

// Number of row groups in a parquet file (its natural, indivisible read unit).
int parquet_num_row_groups(std::string const& path);

// Split [0, num_row_groups) into `num_gpus` CONTIGUOUS spans, as even as possible. Contiguous
// (not round-robin) so each worker reads one sequential run of the file. If num_gpus exceeds
// num_row_groups the tail workers get EMPTY spans; callers must treat an empty span as an empty
// partition (produce a 0-row partial), never as "read all row groups".
std::vector<std::vector<cudf::size_type>> partition_row_groups(int num_row_groups, int num_gpus);

// Read ONLY the given row groups of a parquet file, columns projected, NO predicate pushed to
// the reader (the query filters run as operators). The stream is REQUIRED (no default): on a
// worker pinned to a non-zero GPU, read_parquet with cudf::get_default_stream() uses a stream
// bound to device 0 and fails with cudaErrorInvalidDevice. Pass a stream created on the
// caller's device.
cudf::io::table_with_metadata read_row_group_span(std::string const& path,
                                                  std::vector<std::string> columns,
                                                  std::vector<cudf::size_type> const& row_groups,
                                                  rmm::cuda_stream_view stream);

// BROADCAST read: load a whole (small) table, all columns projected, on the caller's device —
// the replicate-the-small-dimension-side half of a broadcast join. NO predicate pushdown.
// The stream is REQUIRED for the same device-binding reason as read_row_group_span.
cudf::io::table_with_metadata read_full_table(std::string const& path,
                                              std::vector<std::string> columns,
                                              rmm::cuda_stream_view stream);

// ---------------------------------------------------------------------------
// Cross-GPU gather — move a (small) partial-result table from a worker's GPU to GPU0.
//
// The producing worker packs its table into ONE contiguous device buffer + host metadata
// (cudf::pack). The buffer is then peer-copied to GPU0 and reconstructed with cudf::unpack —
// the same serialize/transfer/deserialize a real shuffle does, here between GPUs. The packed
// source must stay alive on the producing worker until GPU0 has copied it (the future ordering
// in the test guarantees that), and the returned GPU0 buffer must outlive the returned view.
// ---------------------------------------------------------------------------

// A non-owning description of a packed partial living on some worker's GPU. `metadata` and
// `gpu_data` point into a cudf::packed_columns kept alive by that worker.
struct PackedPartial {
  int            device;         // GPU the gpu_data lives on
  const uint8_t* metadata;       // HOST pointer (readable from any thread)
  const uint8_t* gpu_data;       // DEVICE pointer on `device`
  std::size_t    gpu_data_size;  // bytes
};

// Build a PackedPartial view over a cudf::packed_columns (call cudf::pack yourself so the
// packed_columns is owned where you want it kept alive).
PackedPartial describe_packed(int device, cudf::packed_columns const& p);

// A table reconstructed on the CURRENT GPU: the owning device buffer plus the view into it. The
// view is valid only while `buffer` is alive.
struct GatheredTable {
  rmm::device_buffer buffer;  // on the current device
  cudf::table_view   view;    // points into `buffer`
};

// On the CURRENT device (the calling worker's GPU): peer-copy a packed partial's buffer in and
// unpack it. A same-device partial is copied too — a device-to-device copy on one GPU is cheap
// and keeps the code uniform.
GatheredTable gather_here(PackedPartial const& p, rmm::cuda_stream_view stream);

// ---------------------------------------------------------------------------
// hash_shuffle — all-to-all cross-GPU repartition by a key (MURMUR3).
//
// `locals[g]` is a table resident on GPU g (same schema on every GPU). This repartitions ALL
// rows so that every row (from any GPU) whose key hashes to bucket p lands on GPU p. Returns G
// owning tables; result[p] is resident on GPU p and MUST be consumed and freed on worker p.
//
// Every worker hashes with the SAME function and seed, so a given key maps to the same bucket on
// every GPU — that is what makes the repartition consistent (all rows for a key converge on one
// GPU). Mechanism: each worker cudf::hash_partition's its local table into G contiguous buckets,
// packs each bucket, and the all-to-all peer exchange delivers bucket p from every worker to
// worker p, which concatenates them. This is the genuine repartition a high-cardinality group-by
// needs — after it, each GPU owns a DISJOINT key range and can group locally with no cross-GPU
// partial merge. (For a low-cardinality group-by, prefer the gather_here partial-merge instead.)
std::vector<std::unique_ptr<cudf::table>> hash_shuffle(
    WorkerPool& pool, std::vector<cudf::table_view> const& locals,
    std::vector<cudf::size_type> const& key_cols);

}  // namespace peacock_mgpu
