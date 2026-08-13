// Properties of a single-pool (integrated) GPU that decide engine design, and that the
// query benchmarks cannot see. CUDA runtime only.
//
//   nvcc -O3 -std=c++17 -arch=native -o dgx_spark_probe cpp/benchmarks/dgx_spark_probe.cu
//
// Each probe answers a question peacockdb has to answer for GB10-class hardware:
//   oversubscribe  - can a query allocate more than physical memory and page, or does it die?
//                    (#142: the batch-partitioned mode has no recourse for an oversized batch)
//   launch         - what does an empty kernel launch cost? the batch-partitioned mode runs
//                    one call per node per batch, so this is the floor on batching finer.
//   first-touch    - on one pool, does it matter whether the CPU or the GPU wrote a page
//                    first? if it does, where a buffer is filled changes what reading costs.
//   alloc          - what do cudaMalloc/cudaFree cost by size? #148 rests on this.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                            \
  do {                                                                                   \
    cudaError_t e = (x);                                                                 \
    if (e != cudaSuccess) {                                                              \
      std::printf("    %s -> %s\n", #x, cudaGetErrorString(e));                          \
      cudaGetLastError();                                                                \
    }                                                                                    \
  } while (0)

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
double second_min(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.size() > 1 ? v[1] : v[0];
}

__global__ void empty_kernel() {}
__global__ void touch_kernel(float4* p, size_t n) {
  size_t s = size_t(blockDim.x) * gridDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += s)
    p[i] = make_float4(1.f, 1.f, 1.f, 1.f);
}
__global__ void read_kernel(const float4* p, size_t n, float* sink) {
  size_t s = size_t(blockDim.x) * gridDim.x;
  float acc = 0.f;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += s) {
    float4 v = p[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 12345.678f) *sink = acc;
}

void probe_launch() {
  std::printf("\n== launch overhead ==\n");
  cudaFree(nullptr);
  for (int batch : {1, 10, 100, 1000}) {
    std::vector<double> s;
    for (int r = 0; r < 6; ++r) {
      auto t0 = Clock::now();
      for (int i = 0; i < batch; ++i) empty_kernel<<<1, 32>>>();
      cudaDeviceSynchronize();
      s.push_back(ms(t0, Clock::now()));
    }
    double t = second_min(s);
    std::printf("  %4d empty launches: %8.3f ms total, %7.2f us each\n", batch, t,
                t * 1000.0 / batch);
  }
}

void probe_alloc() {
  std::printf("\n== cudaMalloc / cudaFree by size ==\n");
  for (size_t mb : {1ull, 16ull, 256ull, 1024ull, 4096ull}) {
    size_t bytes = mb << 20;
    std::vector<double> ma, fr;
    for (int r = 0; r < 4; ++r) {
      void* p = nullptr;
      auto t0 = Clock::now();
      if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); break; }
      auto t1 = Clock::now();
      cudaFree(p);
      auto t2 = Clock::now();
      ma.push_back(ms(t0, t1));
      fr.push_back(ms(t1, t2));
    }
    if (ma.empty()) { std::printf("  %5zu MiB: allocation failed\n", mb); continue; }
    std::printf("  %5zu MiB: malloc %8.2f ms  free %8.2f ms  (%.2f GB/s alloc)\n", mb,
                second_min(ma), second_min(fr), bytes / (second_min(ma) / 1e3) / 1e9);
  }
}

void probe_first_touch(size_t bytes) {
  std::printf("\n== first touch: who writes a managed page first ==\n");
  size_t n = bytes / sizeof(float4);
  int block = 256, grid = int(std::min<size_t>((n + block - 1) / block, 8192));
  float* sink = nullptr;
  CK(cudaMalloc(&sink, sizeof(float)));

  for (int cpu_first : {1, 0}) {
    void* m = nullptr;
    if (cudaMallocManaged(&m, bytes) != cudaSuccess) { cudaGetLastError(); continue; }
    if (cpu_first) {
      std::memset(m, 1, bytes);  // CPU faults the pages in
    } else {
      touch_kernel<<<grid, block>>>(static_cast<float4*>(m), n);
      cudaDeviceSynchronize();
    }
    std::vector<double> s;
    for (int r = 0; r < 5; ++r) {
      auto t0 = Clock::now();
      read_kernel<<<grid, block>>>(static_cast<const float4*>(m), n, sink);
      cudaDeviceSynchronize();
      s.push_back(ms(t0, Clock::now()));
    }
    double t = second_min(s);
    std::printf("  %-22s GPU read %7.2f ms  %7.1f GB/s\n",
                cpu_first ? "CPU wrote first:" : "GPU wrote first:", t, bytes / (t / 1e3) / 1e9);
    cudaFree(m);
  }
  cudaFree(sink);
}

// Can a managed allocation exceed physical memory, and what does using it cost? On a discrete
// GPU this is the classic oversubscription case; on one pool there is nothing to migrate to,
// so the interesting answer is whether it fails, swaps, or silently works.
void probe_oversubscribe() {
  std::printf("\n== managed oversubscription ==\n");
  size_t freeb = 0, total = 0;
  CK(cudaMemGetInfo(&freeb, &total));
  std::printf("  total %.1f GiB, free %.1f GiB\n", total / 1073741824.0, freeb / 1073741824.0);

  for (double frac : {0.5, 0.9, 1.2}) {
    size_t bytes = size_t(total * frac);
    void* m = nullptr;
    auto t0 = Clock::now();
    cudaError_t e = cudaMallocManaged(&m, bytes);
    auto t1 = Clock::now();
    if (e != cudaSuccess) {
      std::printf("  %4.0f%% of total (%.1f GiB): cudaMallocManaged -> %s\n", frac * 100,
                  bytes / 1073741824.0, cudaGetErrorString(e));
      cudaGetLastError();
      continue;
    }
    // Touch a 1 GiB window from the GPU so the mapping is exercised without writing the whole
    // thing, which at 1.2x total would be the point of no return on a machine we share.
    size_t win = std::min<size_t>(bytes, 1ull << 30);
    size_t n = win / sizeof(float4);
    int block = 256, grid = int(std::min<size_t>((n + block - 1) / block, 8192));
    auto t2 = Clock::now();
    touch_kernel<<<grid, block>>>(static_cast<float4*>(m), n);
    cudaError_t k = cudaDeviceSynchronize();
    auto t3 = Clock::now();
    std::printf("  %4.0f%% of total (%6.1f GiB): alloc %7.1f ms, touch 1 GiB %7.1f ms -> %s\n",
                frac * 100, bytes / 1073741824.0, ms(t0, t1), ms(t2, t3),
                k == cudaSuccess ? "ok" : cudaGetErrorString(k));
    cudaGetLastError();
    cudaFree(m);
  }
}
}  // namespace

int main(int argc, char** argv) {
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp p{};
  CK(cudaGetDeviceProperties(&p, dev));
  std::printf("device %s sm_%d%d, integrated=%d, SMs=%d\n", p.name, p.major, p.minor,
              p.integrated, p.multiProcessorCount);

  size_t buf = (argc > 1) ? size_t(std::atoll(argv[1])) << 20 : (2ull << 30);
  probe_launch();
  probe_alloc();
  probe_first_touch(buf);
  probe_oversubscribe();
  std::printf("\n=== PROBE COMPLETE ===\n");
  return 0;
}
