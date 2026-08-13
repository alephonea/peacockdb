// RAM <-> VRAM throughput, measured directly.
//
// Build — CUDA runtime only, no cuDF and no cmake, so this runs on a host long before a
// RAPIDS environment exists:
//
//   nvcc -O3 -std=c++17 -arch=native -o bench_ram_vram cpp/benchmarks/bench_ram_vram.cu
//
// Usage: bench_ram_vram [buffer_MiB]   (default: an eighth of free memory, capped at 1 GiB)
//
// Why it is not derived from the interconnect width: on a single-pool part (GB10) "host to
// device" is not a transfer across anything — both allocations land in the same physical
// LPDDR5X — so the numbers a discrete GPU would give do not transfer, in either direction.
// Every figure below is timed, and the reference cases (host memcpy, device-to-device, a
// copy kernel) are here so the transfer numbers can be read against the memory system they
// share rather than against a bus that may not exist.
//
// Three cases worth keeping apart, because only the first is what this file is running on:
// GB10 has ONE pool (device total == system RAM, no framebuffer); Grace-Hopper has TWO real
// pools, LPDDR5X and HBM3, joined by NVLink-C2C, so a copy there does cross a link; a
// discrete GPU has two pools joined by PCIe. The printed `integrated` flag and the
// device-total-vs-system-RAM comparison are how a reader tells which one they are on.
//
// Each phase allocates what it needs and frees it before the next one starts, so the peak
// is one or two buffers rather than all of them. That is what lets the buffer be sized in
// the tens of GiB — the size at which a "this is just cache" explanation stops being
// available. A phase that does not fit says so and is skipped; it is never silently
// dropped, because a missing row that looks like an absent measurement rather than an
// absent capability is how a benchmark misleads.
//
// Protocol matches llm-wiki/reports/benchmark-minimal.md: warm up, then report the
// 2nd-minimum of N runs. GB/s is 1e9 bytes per second.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                                 \
  do {                                                                                   \
    cudaError_t _err = (expr);                                                           \
    if (_err != cudaSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,               \
                   cudaGetErrorString(_err));                                            \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

namespace {

constexpr int kWarmup = 1;
constexpr int kRuns = 5;

// Headroom left unallocated so a phase cannot push the machine into reclaim and time that
// instead of the copy.
constexpr size_t kMargin = 2ull << 30;

int g_skipped = 0;

// 2nd-minimum, the repo's benchmark statistic: it discards the single luckiest run without
// letting a stray slow run in, which matters here because the first touch of a fresh
// mapping faults pages in and is not the steady state we are reporting.
double second_min(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples.size() > 1 ? samples[1] : samples[0];
}

double gbps(size_t bytes, double seconds) { return double(bytes) / seconds / 1e9; }

struct Row {
  std::string name;
  size_t bytes;
  double seconds;
};

std::vector<Row> rows;

template <typename F>
void measure(const std::string& name, size_t bytes, F&& f) {
  for (int i = 0; i < kWarmup; ++i) f();
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> samples;
  samples.reserve(kRuns);
  for (int i = 0; i < kRuns; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  double s = second_min(std::move(samples));
  rows.push_back({name, bytes, s});
  std::printf("  %-46s %8.2f GB/s   %9.3f ms\n", name.c_str(), gbps(bytes, s), s * 1e3);
  std::fflush(stdout);
}

// A phase runs only if its allocations fit alongside the margin. Device and host memory are
// the same pool on an integrated part, so one check covers both.
bool fits(const char* phase, size_t need) {
  size_t free_bytes = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
  if (need + kMargin <= free_bytes) return true;
  std::printf("  [skipped: %s needs %.1f GiB + %.0f GiB margin, %.1f GiB free]\n", phase,
              need / 1073741824.0, kMargin / 1073741824.0, free_bytes / 1073741824.0);
  ++g_skipped;
  return false;
}

// float4 so a thread issues a 16-byte access; the grid-stride loop keeps the launch shape
// independent of size.
__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst,
                            size_t n) {
  size_t stride = size_t(blockDim.x) * gridDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    dst[i] = src[i];
  }
}

// Reads only, accumulating into a sink the compiler cannot drop. Every element is read
// exactly once per launch, so no cache can serve a second access — which is the property
// that makes the zero-copy number below a memory-system measurement rather than a cache hit.
__global__ void read_kernel(const float4* __restrict__ src, size_t n, float* sink) {
  size_t stride = size_t(blockDim.x) * gridDim.x;
  float acc = 0.f;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 12345.678f) *sink = acc;  // never true; keeps the loads live
}

void launch_copy(const void* src, void* dst, size_t bytes) {
  size_t n = bytes / sizeof(float4);
  int block = 256;
  int grid = int(std::min<size_t>((n + block - 1) / block, 8192));
  copy_kernel<<<grid, block>>>(static_cast<const float4*>(src), static_cast<float4*>(dst), n);
}

void launch_read(const void* src, size_t bytes, float* sink) {
  size_t n = bytes / sizeof(float4);
  int block = 256;
  int grid = int(std::min<size_t>((n + block - 1) / block, 8192));
  read_kernel<<<grid, block>>>(static_cast<const float4*>(src), n, sink);
}

}  // namespace

int main(int argc, char** argv) {
  int device = 0;
  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  size_t free_bytes = 0, total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

  size_t bytes = std::min<size_t>(1ull << 30, free_bytes / 8);
  if (argc > 1) bytes = size_t(std::atof(argv[1])) * (1ull << 20);
  bytes &= ~size_t(sizeof(float4) - 1);

  std::printf("device            : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("integrated        : %s\n",
              prop.integrated ? "yes (unified memory system)" : "no (discrete)");
  std::printf("can map host mem  : %s\n", prop.canMapHostMemory ? "yes" : "no");
  std::printf("pageableAccess    : %s\n", prop.pageableMemoryAccess ? "yes" : "no");
  // No clock here: CUDA 13 removed cudaDeviceProp::memoryClockRate, and a nominal bandwidth
  // computed from width x clock is the derived number this benchmark exists to avoid.
  std::printf("memory bus        : %d-bit, L2 %.1f MiB\n", prop.memoryBusWidth,
              prop.l2CacheSize / 1048576.0);
  std::printf("device memory     : %.1f GiB free / %.1f GiB total\n", free_bytes / 1073741824.0,
              total_bytes / 1073741824.0);
  std::printf("buffer            : %.0f MiB (%.2f GiB)\n\n", bytes / 1048576.0,
              bytes / 1073741824.0);

  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

  // ---- host reference ----------------------------------------------------
  if (fits("host memcpy", 2 * bytes)) {
    void* a = std::malloc(bytes);
    void* b = std::malloc(bytes);
    if (a && b) {
      std::memset(a, 1, bytes);
      std::memset(b, 2, bytes);
      std::printf("host reference (single-threaded memcpy — one core, not the DRAM ceiling)\n");
      measure("host memcpy (RAM -> RAM)", bytes, [&] { std::memcpy(b, a, bytes); });
    }
    std::free(a);
    std::free(b);
  }

  // ---- cudaMemcpy, pageable ----------------------------------------------
  if (fits("pageable copies", 2 * bytes)) {
    void* h = std::malloc(bytes);
    void* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    if (h) {
      std::memset(h, 3, bytes);
      std::printf("\ncudaMemcpy, pageable host memory\n");
      measure("H2D pageable", bytes,
              [&] { CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); });
      measure("D2H pageable", bytes,
              [&] { CUDA_CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost)); });
    }
    std::free(h);
    CUDA_CHECK(cudaFree(d));
  }

  // ---- cudaMemcpy, pinned, sync and async --------------------------------
  if (fits("pinned copies", 2 * bytes)) {
    void* h = nullptr;
    void* d = nullptr;
    cudaError_t pin = cudaHostAlloc(&h, bytes, cudaHostAllocDefault);
    if (pin != cudaSuccess) {
      std::printf("  [skipped: pinning %.1f GiB failed -> %s]\n", bytes / 1073741824.0,
                  cudaGetErrorString(pin));
      ++g_skipped;
      cudaGetLastError();
    } else {
      CUDA_CHECK(cudaMalloc(&d, bytes));
      std::memset(h, 4, bytes);
      std::printf("\ncudaMemcpy, pinned host memory\n");
      measure("H2D pinned", bytes,
              [&] { CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); });
      measure("D2H pinned", bytes,
              [&] { CUDA_CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost)); });

      // Chunked across streams: what a loader overlapping transfer with compute would see,
      // and on a discrete part the only way to reach the bus ceiling.
      constexpr int kStreams = 4;
      cudaStream_t streams[kStreams];
      for (auto& s : streams) CUDA_CHECK(cudaStreamCreate(&s));
      size_t chunk = ((bytes / kStreams) + 4095) & ~size_t(4095);
      std::printf("\ncudaMemcpyAsync, pinned, %d streams, %.0f MiB chunks\n", kStreams,
                  chunk / 1048576.0);
      auto async_copy = [&](cudaMemcpyKind kind) {
        for (int i = 0; i < kStreams; ++i) {
          size_t off = size_t(i) * chunk;
          if (off >= bytes) break;
          size_t len = std::min(chunk, bytes - off);
          void* dst = kind == cudaMemcpyHostToDevice ? static_cast<void*>(static_cast<char*>(d) + off)
                                                     : static_cast<void*>(static_cast<char*>(h) + off);
          const void* src = kind == cudaMemcpyHostToDevice
                                ? static_cast<const void*>(static_cast<char*>(h) + off)
                                : static_cast<const void*>(static_cast<char*>(d) + off);
          CUDA_CHECK(cudaMemcpyAsync(dst, src, len, kind, streams[i]));
        }
      };
      measure("H2D pinned async", bytes, [&] { async_copy(cudaMemcpyHostToDevice); });
      measure("D2H pinned async", bytes, [&] { async_copy(cudaMemcpyDeviceToHost); });
      for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
      CUDA_CHECK(cudaFreeHost(h));
      CUDA_CHECK(cudaFree(d));
    }
  }

  // ---- device-side reference ---------------------------------------------
  if (fits("device-to-device", 2 * bytes)) {
    void *a = nullptr, *b = nullptr;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMemset(a, 5, bytes));
    std::printf("\ndevice-side reference\n");
    measure("D2D cudaMemcpy", bytes,
            [&] { CUDA_CHECK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice)); });
    // A copy kernel moves the bytes twice — once read, once written — so the traffic it puts
    // on the memory system is 2x the buffer, and that is what the rate is reported against.
    measure("copy kernel (VRAM r+w, 2x bytes)", 2 * bytes, [&] { launch_copy(a, b, bytes); });
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
  }

  // ---- managed prefetch ---------------------------------------------------
  if (prop.concurrentManagedAccess && fits("managed prefetch", bytes)) {
    void* m = nullptr;
    CUDA_CHECK(cudaMallocManaged(&m, bytes));
    std::memset(m, 6, bytes);
    std::printf("\nmanaged memory, cudaMemPrefetchAsync\n");
#if CUDART_VERSION >= 13000
    // CUDA 13 takes a cudaMemLocation rather than a device ordinal, so the host destination
    // is a location kind and not the cudaCpuDeviceId sentinel. Both spellings are kept
    // because this benchmark is meant to be run on the discrete CUDA 12 hosts too — a
    // measurement you cannot take on both machines cannot compare them.
    cudaMemLocation dev_loc{cudaMemLocationTypeDevice, device};
    cudaMemLocation host_loc{cudaMemLocationTypeHost, 0};
    measure("prefetch host -> device", bytes,
            [&] { CUDA_CHECK(cudaMemPrefetchAsync(m, bytes, dev_loc, 0, 0)); });
    measure("prefetch device -> host", bytes,
            [&] { CUDA_CHECK(cudaMemPrefetchAsync(m, bytes, host_loc, 0, 0)); });
#else
    measure("prefetch host -> device", bytes,
            [&] { CUDA_CHECK(cudaMemPrefetchAsync(m, bytes, device, 0)); });
    measure("prefetch device -> host", bytes,
            [&] { CUDA_CHECK(cudaMemPrefetchAsync(m, bytes, cudaCpuDeviceId, 0)); });
#endif
    CUDA_CHECK(cudaFree(m));
  }

  // ---- the comparison that decides whether to stage a copy at all ---------
  // One buffer each, measured in separate phases, so these two survive at sizes where
  // nothing else fits — which is exactly where a cache explanation would have to be
  // abandoned.
  if (prop.canMapHostMemory && fits("zero-copy read", bytes)) {
    void *h = nullptr, *dp = nullptr;
    cudaError_t map = cudaHostAlloc(&h, bytes, cudaHostAllocMapped);
    if (map != cudaSuccess) {
      std::printf("  [skipped: mapping %.1f GiB failed -> %s]\n", bytes / 1073741824.0,
                  cudaGetErrorString(map));
      ++g_skipped;
      cudaGetLastError();
    } else {
      std::memset(h, 7, bytes);
      CUDA_CHECK(cudaHostGetDevicePointer(&dp, h, 0));
      std::printf("\nzero-copy (kernel reads mapped host memory in place)\n");
      measure("kernel read of mapped host memory", bytes, [&] { launch_read(dp, bytes, d_sink); });
      CUDA_CHECK(cudaFreeHost(h));
    }
  }

  if (fits("device read", bytes)) {
    void* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemset(d, 8, bytes));
    std::printf("\ndevice-memory read (same kernel, for the comparison above)\n");
    measure("kernel read of device memory", bytes, [&] { launch_read(d, bytes, d_sink); });
    CUDA_CHECK(cudaFree(d));
  }

  std::printf("\n%-46s %10s %12s\n", "summary", "GB/s", "ms");
  for (const auto& r : rows) {
    std::printf("%-46s %10.2f %12.3f\n", r.name.c_str(), gbps(r.bytes, r.seconds),
                r.seconds * 1e3);
  }
  if (g_skipped) {
    std::printf("\n%d phase(s) skipped at this buffer size — see the [skipped] lines above.\n",
                g_skipped);
  }

  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
