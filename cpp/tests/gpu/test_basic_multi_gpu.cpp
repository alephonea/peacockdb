// A minimal TWO-GPU pipeline exercising BOTH cuDF and cuVS on BOTH devices, with a
// bidirectional EXCHANGE across the NVLink in the middle. Input is generated in-process from a
// fixed seed and every check is a self-consistency assertion — no goldens, no external files.
//
// WHY A WORKER-PER-GPU THREAD POOL — a hard constraint, not cosmetic:
//
//   cudaSetDevice is THREAD-LOCAL, and a cuDF column/table (and cuVS/raft device objects)
//   must be DESTROYED while its owning device is current — not merely created there. A
//   unique_ptr<column> allocated on GPU0 that destructs on a thread where GPU1 is current
//   makes RMM free device-0 memory under device 1 -> corruption / CUDA error. So each
//   GpuWorker is a PERSISTENT thread that calls cudaSetDevice(g) once at startup and then
//   services a task queue; ALL of device g's cuDF and cuVS work — and the destruction of its
//   objects — happens on that one thread. That is what Dask-cuDF and every real multi-GPU
//   cuDF deployment do; a single-thread version flipping cudaSetDevice between phases only
//   dodges the problem by never letting an object outlive its phase.
//
//   cuVS is softer (every call takes a raft::resources bound to the current device) but the
//   same discipline applies, so it rides the same workers. cuVS's native
//   single-node-multi-GPU path (cuvs::neighbors::mg, NCCL under the hood) is deliberately NOT
//   used here: it would hide the very cross-GPU exchange this test exists to make visible.
//
// THE PIPELINE — a SYMMETRIC EXCHANGE (each worker g runs the identical steps on its device):
//   1. [cuDF] generate a LOCAL partition of nHalf D-dim points from the fixed seed — device 0
//             gets partition A, device 1 gets partition B, and A != B. Filter by a box
//             predicate, interleave survivors into a row-major [n_g x D] matrix M_local on
//             device g. Sync.                                 (barrier: both M_local ready)
//   2. EXCHANGE (both directions, each copy issued from the DESTINATION worker):
//        worker0: cudaMemcpyPeerAsync  B (device 1) --NVLink--> M_remote on device 0
//        worker1: cudaMemcpyPeerAsync  A (device 0) --NVLink--> M_remote on device 1
//      Each M_local stays alive on its owner (step 3 also consumes it locally), so it is only
//      ever read across the link, never destroyed off-thread.
//   3. [cuDF] concatenate into the SAME canonical order [A ; B] on BOTH devices —
//             device 0 does concat(local=A, remote=B); device 1 does concat(remote=A, local=B)
//             -> both yield the identical [A;B]. checksum = cuDF sum-reduce of M_full.
//   4. [cuVS] brute-force self-kNN over M_full on device g -> every point's NN is itself at
//             distance ~0 (A and B are distinct random points, so no cross-partition ties).
//   5. Join. Assert checksum(device 0) == checksum(device 1).
//
//   WHY THIS CATCHES REAL BUGS: each device's final set depends on data that ORIGINATED ON THE
//   OTHER device (device 0's = A_local + B_received; device 1's = A_received + B_local), so
//   neither GPU is a passive sink and BOTH copy directions are exercised. The cross-device
//   checksum equality requires BOTH receptions to be faithful; the per-device cuVS identity
//   check catches gross garbling of the combined set.
//
// LINKING cuVS — READ THIS BEFORE DEBUGGING A LOAD FAILURE (same note as test_tpchv.cpp):
// libcuvs.so cannot be loaded on its own here. rmm is header-only (no librmm.so), so rmm
// symbols are compiled INTO consumers: libcudf.so DEFINES rmm::logger::~logger and libcuvs.so
// REFERENCES it. Loading cuVS alone fails with "undefined symbol: _ZN3rmm6loggerD1Ev"; loading
// libcudf first and cuVS second works. This binary links both, so link order resolves it.
//
// NOT WIRED INTO CI: needs two visible GPUs. Built for compile coverage but left out of the CI
// test set (no add_test, not installed) until CI grows multi-GPU runner support. Run it by hand
// on a >=2-GPU host; it GTEST_SKIPs itself otherwise.

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include "multi_gpu.hpp"  // GpuWorker + MG_CUDA_TRY, shared by all multi-GPU test files

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using peacock_mgpu::GpuWorker;  // GpuWorker + MG_CUDA_TRY come from multi_gpu.hpp

constexpr int   kHalf   = 2500;   // points per partition (before filtering) -> 5000 total
constexpr int   kDim    = 16;     // dimensionality of each point
constexpr float kBoxCut = 0.5f;   // keep points whose coord 0 >= this (~half survive)
constexpr int   kSeed   = 42;     // fixed -> deterministic input, no goldens needed

// cuDF sum-reduce of a FLOAT32 device column, accumulated in FLOAT64. Runs on the current
// device (the calling worker's). Used to checksum each device's combined matrix.
double checksum(cudf::column_view const& col, rmm::cuda_stream_view stream) {
  auto agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto s   = cudf::reduce(col, *agg, cudf::data_type{cudf::type_id::FLOAT64}, stream);
  return static_cast<cudf::numeric_scalar<double>*>(s.get())->value(stream);
}

// Brute-force self-kNN (queries == dataset, k = 1) over a resident row-major [rows x dim]
// float matrix on the current device. A correct run over distinct points yields {0, ~0} — a
// point's nearest neighbour in its own dataset is itself at distance 0. All cuVS/raft objects
// here are created AND destroyed on the calling worker thread.
struct KnnCheck {
  int64_t violations;     // points whose NN(i) != i
  float   max_self_dist;  // largest returned nearest distance
};
KnnCheck self_knn_identity(const float* data, int64_t rows, int dim) {
  raft::resources handle;  // binds to the current device, owns a stream

  auto dataset = raft::make_device_matrix_view<const float, int64_t>(data, rows, dim);
  cuvs::neighbors::brute_force::index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2SqrtExpanded;  // true (non-squared) L2
  auto index = cuvs::neighbors::brute_force::build(handle, index_params, dataset);

  auto queries   = raft::make_device_matrix_view<const float, int64_t>(data, rows, dim);
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(handle, rows, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(handle, rows, 1);
  cuvs::neighbors::brute_force::search(handle, cuvs::neighbors::brute_force::search_params{},
                                       index, queries, neighbors.view(), distances.view());
  raft::resource::sync_stream(handle);

  std::vector<int64_t> nn(rows);
  std::vector<float>   dist(rows);
  MG_CUDA_TRY(cudaMemcpy(nn.data(), neighbors.data_handle(), rows * sizeof(int64_t),
                        cudaMemcpyDeviceToHost));
  MG_CUDA_TRY(cudaMemcpy(dist.data(), distances.data_handle(), rows * sizeof(float),
                        cudaMemcpyDeviceToHost));

  KnnCheck out{0, 0.0f};
  for (int64_t i = 0; i < rows; ++i) {
    if (nn[i] != i) ++out.violations;
    out.max_self_dist = std::max(out.max_self_dist, dist[i]);
  }
  return out;
}

// Build a cuDF table of kDim FLOAT32 columns (column d = coordinate d) from host data on the
// current device.
cudf::table make_points_table(std::vector<std::vector<float>> const& host_cols,
                              rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(host_cols.size());
  for (auto const& hc : host_cols) {
    auto c = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT32},
                                       static_cast<cudf::size_type>(hc.size()),
                                       cudf::mask_state::UNALLOCATED, stream);
    MG_CUDA_TRY(cudaMemcpyAsync(c->mutable_view().data<float>(), hc.data(),
                               hc.size() * sizeof(float), cudaMemcpyHostToDevice,
                               stream.value()));
    cols.push_back(std::move(c));
  }
  return cudf::table(std::move(cols));
}

// A device's LOCAL partition after filter+interleave: its device pointer and row count, with
// the owning column kept alive elsewhere (in an m_local holder touched only by its worker).
struct Local {
  const float* ptr;  // device pointer to the row-major [n x kDim] matrix on this worker's GPU
  int64_t      n;    // surviving points in this partition
};

// STEP 1 on a worker: generate the partition, filter, interleave -> M_local; keep M_local in
// `holder` (owned by this worker) and hand back its pointer + row count.
Local build_local(std::vector<std::vector<float>> const& host_part,
                  std::unique_ptr<cudf::column>& holder) {
  rmm::cuda_stream stream;
  auto table = make_points_table(host_part, stream.view());

  cudf::numeric_scalar<float> cut(kBoxCut, true, stream.view());
  auto mask     = cudf::binary_operation(table.view().column(0), cut,
                                         cudf::binary_operator::GREATER_EQUAL,
                                         cudf::data_type{cudf::type_id::BOOL8}, stream.view());
  auto filtered = cudf::apply_boolean_mask(table.view(), mask->view(), stream.view());
  auto row_major = cudf::interleave_columns(filtered->view(), stream.view());
  stream.synchronize();  // M_local fully materialized before it is read locally or across the link

  const float* ptr = row_major->view().data<float>();
  int64_t n        = filtered->num_rows();
  holder           = std::move(row_major);  // KEEP M_local alive on this worker
  return Local{ptr, n};
}

// STEPS 2-4 on a worker: peer-copy the PEER's partition into M_remote, concatenate into the
// canonical [A;B] order, checksum, and self-kNN over the combined matrix. `local` is this
// worker's own M_local (already resident); `peer` is the other device's partition (read across
// the link). `local_is_A` selects the concat order so BOTH devices produce the identical [A;B]:
// device 0 (local=A) -> concat(local, remote); device 1 (local=B) -> concat(remote, local).
struct Full {
  double   checksum;
  KnnCheck knn;
};
Full exchange_and_combine(cudf::column_view const& local, int this_dev, Local const& peer,
                          int peer_dev, bool local_is_A) {
  rmm::cuda_stream stream;
  const std::size_t rcount = static_cast<std::size_t>(peer.n) * kDim;
  rmm::device_uvector<float> remote(rcount, stream.view());
  // Explicit src/dst device ids: this worker (this_dev current) reads the peer's device memory
  // by pointer. Rides P2P when peer access is enabled, host-stages otherwise.
  MG_CUDA_TRY(cudaMemcpyPeerAsync(remote.data(), this_dev, peer.ptr, peer_dev,
                                 rcount * sizeof(float), stream.value()));
  MG_CUDA_TRY(cudaStreamSynchronize(stream.value()));  // this worker's copy done before concat

  cudf::column_view remote_view(cudf::data_type{cudf::type_id::FLOAT32},
                                static_cast<cudf::size_type>(rcount), remote.data(), nullptr, 0);
  // canonical order [A ; B] regardless of which device we are on
  std::vector<cudf::column_view> in =
      local_is_A ? std::vector<cudf::column_view>{local, remote_view}
                 : std::vector<cudf::column_view>{remote_view, local};
  auto full = cudf::concatenate(in, stream.view());

  double cs = checksum(full->view(), stream.view());
  stream.synchronize();

  const int64_t total = full->size() / kDim;
  KnnCheck knn        = self_knn_identity(full->view().data<float>(), total, kDim);
  return Full{cs, knn};  // remote + full (device buffers) destroyed here, on this worker
}

TEST(BasicMultiGpu, CudfAndCuvsAcrossTwoGpus) {
  int device_count = 0;
  // Do NOT hard-assert this: on a host with no driver / no GPU, cudaGetDeviceCount returns an
  // error rather than 0, and a fatal check there would FAIL where we want to SKIP.
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    cudaGetLastError();
    device_count = 0;
  }
  if (device_count < 2) {
    GTEST_SKIP() << "needs >= 2 visible GPUs, found " << device_count;
  }

  // P2P capability, queried on the main thread (cudaDeviceCanAccessPeer does not need either
  // device current — it only reports capability). The enable happens on the workers.
  int can01 = 0, can10 = 0;
  cudaDeviceCanAccessPeer(&can01, 0, 1);
  cudaDeviceCanAccessPeer(&can10, 1, 0);
  std::fprintf(stderr, "[multi-gpu] P2P 0<->1: %s\n",
               (can01 && can10) ? "yes" : "no (copies will stage through host)");

  GpuWorker w0(0);
  GpuWorker w1(1);

  // Best-effort: each worker enables peer access to the other so the exchange rides P2P.
  // cudaDeviceEnablePeerAccess is relative to the CURRENT device, so it runs on the workers.
  auto enable_peer = [](int peer) {
    cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
  };
  if (can01) w0.submit([&] { enable_peer(1); }).get();
  if (can10) w1.submit([&] { enable_peer(0); }).get();

  // Synthetic input: 2*kHalf points, fixed seed -> deterministic. Partition A = first kHalf,
  // B = second kHalf. Distinct data, so the two partitions differ (A != B).
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  std::vector<std::vector<float>> host_A(kDim, std::vector<float>(kHalf));
  std::vector<std::vector<float>> host_B(kDim, std::vector<float>(kHalf));
  for (int d = 0; d < kDim; ++d) {
    for (int i = 0; i < kHalf; ++i) host_A[d][i] = unit(rng);
    for (int i = 0; i < kHalf; ++i) host_B[d][i] = unit(rng);
  }

  // Each partition's M_local lives in a holder touched ONLY by its owning worker: created in
  // build_local, destroyed in the release task at the very end. The unique_ptr wrappers sit on
  // this stack, but the cuDF columns they own are allocated and freed exclusively on-worker.
  std::unique_ptr<cudf::column> m_local0;  // partition A, device 0
  std::unique_ptr<cudf::column> m_local1;  // partition B, device 1

  // ---- STEP 1: build both local partitions in parallel; barrier on both. ----
  auto fa = w0.submit([&] { return build_local(host_A, m_local0); });
  auto fb = w1.submit([&] { return build_local(host_B, m_local1); });
  Local A = fa.get();  // resident on device 0
  Local B = fb.get();  // resident on device 1
  ASSERT_GT(A.n, 0) << "partition A filter kept no rows";
  ASSERT_GT(B.n, 0) << "partition B filter kept no rows";

  // ---- STEPS 2-4: exchange (both directions) + concat + checksum + self-kNN, in parallel. ----
  // worker0 receives B (1->0); its local is A, so concat order is (local, remote) = [A;B].
  // worker1 receives A (0->1); its local is B, so concat order is (remote, local) = [A;B].
  auto ff0 = w0.submit([&] {
    return exchange_and_combine(m_local0->view(), /*this_dev=*/0, B, /*peer_dev=*/1,
                                /*local_is_A=*/true);
  });
  auto ff1 = w1.submit([&] {
    return exchange_and_combine(m_local1->view(), /*this_dev=*/1, A, /*peer_dev=*/0,
                                /*local_is_A=*/false);
  });
  Full F0 = ff0.get();  // rethrows here if worker0 threw; its copy is complete once this returns
  Full F1 = ff1.get();

  EXPECT_EQ(F0.knn.violations, 0) << "GPU0 cuVS: some combined points' NN is not itself";
  EXPECT_LT(F0.knn.max_self_dist, 1e-3f) << "GPU0 cuVS: self-distance too large";
  EXPECT_EQ(F1.knn.violations, 0) << "GPU1 cuVS: some combined points' NN is not itself";
  EXPECT_LT(F1.knn.max_self_dist, 1e-3f) << "GPU1 cuVS: self-distance too large";

  // The cross-device equality is the exchange's proof: device 0's full set is A_local +
  // B_received and device 1's is A_received + B_local; equal checksums require BOTH transfer
  // directions to have been faithful.
  EXPECT_DOUBLE_EQ(F0.checksum, F1.checksum)
      << "cuDF checksum of the combined [A;B] differs across devices (" << F0.checksum << " vs "
      << F1.checksum << ") — a transfer in one of the two directions corrupted the data";

  // ---- STEP 5: release each M_local on its OWNING worker (never on this thread), then join. ----
  // Safe now: both exchange tasks have completed above, so neither partition is still being
  // read across the link.
  w0.submit([&] { m_local0.reset(); }).get();
  w1.submit([&] { m_local1.reset(); }).get();

  std::fprintf(stderr,
               "[multi-gpu] OK: A=%ld B=%ld combined=%ld dim=%d checksum=%.6f "
               "(cuDF+cuVS on both GPUs, %s bidirectional exchange)\n",
               static_cast<long>(A.n), static_cast<long>(B.n),
               static_cast<long>(A.n + B.n), kDim, F0.checksum,
               (can01 && can10) ? "P2P" : "host-staged");
  // Workers join cleanly in their destructors as this scope exits.
}

}  // namespace

// Own entry point: the conda cuDF ships no gtest_main, and the CMake target links
// GTest::gtest (not ::gtest_main) — same convention as the other test binaries here.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
