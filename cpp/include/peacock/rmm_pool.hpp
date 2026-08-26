// Pooled device allocator: the sizing rule, and the one function that installs it.
//
// cuDF allocates every intermediate through rmm's current device resource, and rmm's
// default is a bare cudaMalloc/cudaFree per allocation. A query that materializes dozens of
// multi-GiB intermediates therefore pays a driver round trip — and on a unified-memory part
// a page-table walk and a zeroing pass — per intermediate. That is not background noise:
// on GB10, TPC-H q1 over the whole table measured 76.5 s without a pool.
//
// This header lives under `cpp/include/` rather than `cpp/tests/` because three different
// callers install the same pool and must not drift:
//   - the single-GPU gtest binaries, from their main();
//   - `peacock_install_rmm_pool` in the FFI, which is how `peacock_gpu_benchmarks` — a Rust
//     target that cannot include a C++ header — gets the same allocator the gtest binaries
//     have. Without it the two families of numbers in the tree are taken under different
//     allocators and quietly compared (llm-wiki/archive/archived-tickets.md #151);
//   - `multi_gpu.cpp` keeps its own per-device installation, because a pool per worker
//     thread on the device that worker owns is a different lifecycle, but reads the
//     percentages from here.
//
// The engine still does not install this on its own behalf: no path under `cpp/src/` calls
// it except the FFI entry point above, which nothing but the benchmark harness invokes. So
// a shipping query still allocates the expensive way, and `gpu_memory_limit` is still
// accepted and ignored. That is #148, deliberately left open here: making the engine
// self-install changes where every shipping query's memory comes from and what
// `gpu_memory_limit` means, which is a decision about the product, not about measurement.
#pragma once

// RMM flattened rmm/mr/device/*.hpp into rmm/mr/*.hpp after 25.02, and peacock_tpch_tests
// builds on both CI legs, so both spellings are accepted. Same treatment as the cudf join
// header in test_tpch.cpp.
#if __has_include(<rmm/mr/cuda_memory_resource.hpp>)
#  include <rmm/mr/cuda_memory_resource.hpp>
#  include <rmm/mr/per_device_resource.hpp>
#  include <rmm/mr/pool_memory_resource.hpp>
#  include <rmm/mr/statistics_resource_adaptor.hpp>
#else
#  include <rmm/mr/device/cuda_memory_resource.hpp>
#  include <rmm/mr/device/per_device_resource.hpp>
#  include <rmm/mr/device/pool_memory_resource.hpp>
#  include <rmm/mr/device/statistics_resource_adaptor.hpp>
#endif
#include <rmm/aligned.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>

namespace peacock {

// Pool sizes must be aligned to rmm's allocation granularity.
inline std::size_t pool_align_down(std::size_t n) {
  return n - (n % rmm::CUDA_ALLOCATION_ALIGNMENT);
}

// Two sizing regimes, because "free device memory" means different things on the two kinds
// of machine.
//
// On a DISCRETE part, VRAM is the GPU's alone: reserving 85% of it up front costs the host
// nothing, and it buys a query whose whole working set fits without a mid-query growth
// event — itself another cudaMalloc, the very sync the pool exists to avoid.
//
// On an INTEGRATED part there is one pool of memory and the "device" reservation comes
// straight out of what the OS has for page cache and for the parquet reader's own host
// buffers. Reserving 85% there would starve the read path to speed up the compute path. So
// the initial reservation is small and the ceiling is what stays generous: growth events
// are a handful of cudaMallocs across a query, not one per intermediate, which is the cost
// that actually mattered.
inline constexpr int kDiscreteInitialPercent = 85;
inline constexpr int kDiscreteMaximumPercent = 95;
inline constexpr int kIntegratedInitialPercent = 25;
inline constexpr int kIntegratedMaximumPercent = 90;

// What install_rmm_pool() actually did — not what it was asked to do.
//
// Unavailable is a host problem, not a configuration: nothing asks for it and the sizes
// below are meaningless when it happens. The gtest binaries carry on regardless, since a
// correctness result does not depend on the allocator; the benchmark harness asserts,
// since a time does.
struct RmmPoolStatus {
  enum class State {
    Installed,    // a pool is the current device resource
    Unavailable,  // the pool could not be built; the default resource is still in place
  };
  State state = State::Unavailable;
  bool integrated = false;
  std::size_t free_bytes = 0;
  std::size_t initial_bytes = 0;
  std::size_t maximum_bytes = 0;
};

using StatsMr =
    rmm::mr::statistics_resource_adaptor<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>;

// Function-local static rather than a namespace-scope one: this is a header, and rmm keeps
// a non-owning pointer to whatever is installed, so the object must outlive every test.
inline std::unique_ptr<StatsMr>& stats_mr() {
  static std::unique_ptr<StatsMr> mr;
  return mr;
}

// Installs a pooled resource for the current device and returns what happened. Call before
// any cuDF work; the resources are function-local statics because rmm stores a non-owning
// pointer to the current resource and the callers outlive any narrower scope.
//
// IDEMPOTENT, and that is load-bearing now that this is reachable from the FFI: a second
// call returns the first call's outcome without building a second pool. Overwriting the
// statics instead would drop a resource that live allocations still point into, and the
// benchmark harness — 127 separate #[test] functions in one process — is exactly the shape
// that would find it.
inline const RmmPoolStatus& install_rmm_pool() {
  static RmmPoolStatus status;
  static bool done = false;
  if (done) return status;
  done = true;

  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  status.integrated = prop.integrated != 0;

  std::size_t free_bytes = 0, total = 0;
  if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess || free_bytes == 0) {
    std::fprintf(stderr, "[rmm] no device memory info; leaving the default resource in place\n");
    return status;  // Unavailable
  }
  status.free_bytes = free_bytes;

  // The initial percentage is overridable so "is this cost pool growth?" can be answered by
  // measurement rather than argument: reserve enough up front that no growth is possible and
  // see whether the cost moves.
  int init_pct = prop.integrated ? kIntegratedInitialPercent : kDiscreteInitialPercent;
  const char* pct = std::getenv("PEACOCK_RMM_POOL_INIT_PCT");
  if (pct && *pct) init_pct = std::atoi(pct);
  const int max_pct = prop.integrated ? kIntegratedMaximumPercent : kDiscreteMaximumPercent;
  const std::size_t initial = pool_align_down(free_bytes / 100 * init_pct);
  const std::size_t maximum = pool_align_down(free_bytes / 100 * max_pct);

  static auto upstream = std::make_unique<rmm::mr::cuda_memory_resource>();
  static std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool;
  // Both percentages come off FREE memory, so on a shared host the reservation can fail
  // outright — a neighbour holding most of the device leaves an initial size that was
  // computed a moment ago and is no longer there. Report it instead of aborting here: the
  // correctness binaries are still right without a pool, and the one caller for which that
  // is not true — the benchmark harness — refuses the run on Unavailable itself.
  try {
    pool = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
        upstream.get(), initial, maximum);
    // The statistics adaptor sits ABOVE the pool, so it counts what the query asked for
    // rather than what the pool reserved. That distinction is the whole reason it is here:
    // once a pool owns the memory, cudaMemGetInfo stops moving during a query, so the
    // fixture's free-memory delta reads a peak of nearly zero — a number that looks like a
    // triumph and measures nothing. See peak_allocated_bytes().
    stats_mr() = std::make_unique<StatsMr>(pool.get());
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "[rmm] pool of %.1f GiB could not be built (%s); leaving the default "
                 "resource in place\n",
                 initial / 1073741824.0, e.what());
    pool.reset();
    return status;  // Unavailable
  }
  rmm::mr::set_current_device_resource(stats_mr().get());

  status.state = RmmPoolStatus::State::Installed;
  status.initial_bytes = initial;
  status.maximum_bytes = maximum;
  std::fprintf(stderr, "[rmm] pool on %s: initial %.1f GiB, max %.1f GiB of %.1f GiB free\n",
               prop.integrated ? "an integrated device" : "a discrete device",
               initial / 1073741824.0, maximum / 1073741824.0, free_bytes / 1073741824.0);
  return status;
}

// Peak bytes handed out since the pool was installed, or 0 when there is no pool (in which
// case the caller's cudaMemGetInfo delta is still the right measurement).
inline std::size_t peak_allocated_bytes() {
  if (!stats_mr()) return 0;
  return static_cast<std::size_t>(stats_mr()->get_bytes_counter().peak);
}

// Per-test scoping. The adaptor cascades nested counters, so pushing at the start of a test
// and popping at its end makes the peak that test's high-water mark rather than the whole
// run's — without it the second test in a binary inherits the first one's peak and every
// number after the first is wrong in the flattering direction.
inline void begin_peak_scope() {
  if (stats_mr()) stats_mr()->push_counters();
}
inline void end_peak_scope() {
  if (stats_mr()) stats_mr()->pop_counters();
}

}  // namespace peacock
