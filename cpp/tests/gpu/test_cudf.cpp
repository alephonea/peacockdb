#include <cudf/aggregation.hpp>
#include <cudf/filling.hpp>
#include <cudf/hashing.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/column_wrapper.hpp>

#include "peacock/partitioning.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

// strings_column_wrapper references cudf::test::get_default_stream(), which lives in
// libcudftestutil — NOT shipped in our conda cudf. Provide that single symbol by
// delegating to libcudf's real default stream (this is the only testutil symbol the
// column wrappers need for construction).
namespace cudf {
namespace test {
rmm::cuda_stream_view const get_default_stream() { return cudf::get_default_stream(); }
}  // namespace test
}  // namespace cudf

#include <cstdint>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "plan_executor.h"
#include "rmm_pool.hpp"

// measure_timing_floor_us turns the global timing switch on for its own samples, so
// it has to hand back what it found. It is called mid-benchmark, between a warm-up
// and the measured runs, and a leak in either direction changes what every later node
// in that process measures — without failing anything.
TEST(NodeTiming, FloorRestoresTheSwitch) {
  for (bool start : {false, true}) {
    peacock::set_node_timing(start);
    (void)peacock::measure_timing_floor_us(4);
    EXPECT_EQ(peacock::node_timing_enabled(), start);
  }
  peacock::set_node_timing(false);
}

// The header promises a clamp rather than undefined behaviour: a second-smallest
// needs two samples, and the caller passes a constant that someone will one day tune.
TEST(NodeTiming, FloorClampsSampleCount) {
  EXPECT_NO_THROW({ (void)peacock::measure_timing_floor_us(0); });
  EXPECT_NO_THROW({ (void)peacock::measure_timing_floor_us(1); });
}

TEST(CudfGpu, SequenceSum) {
  // Generate [1, 2, 3, ..., 100] on the GPU.
  constexpr cudf::size_type N = 100;
  auto init = cudf::make_fixed_width_scalar<int64_t>(1);
  auto step = cudf::make_fixed_width_scalar<int64_t>(1);
  auto col  = cudf::sequence(N, *init, *step);

  ASSERT_EQ(col->size(), N);
  ASSERT_EQ(col->type().id(), cudf::type_id::INT64);

  // Sum on the GPU; expected = N*(N+1)/2
  auto agg    = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto result = cudf::reduce(col->view(), *agg, cudf::data_type{cudf::type_id::INT64});

  auto* scalar = dynamic_cast<cudf::numeric_scalar<int64_t>*>(result.get());
  ASSERT_NE(scalar, nullptr);
  ASSERT_TRUE(scalar->is_valid());
  EXPECT_EQ(scalar->value(), static_cast<int64_t>(N) * (N + 1) / 2);
}

// Conformance: the GPU Spark-murmur3 partition-id path
// (peacock::partitioning::spark_partition_ids) must match comet's CPU twin
// bit-exact. The reference values below come from the real comet helper
// (create_murmur3_hashes seed=42 -> pmod) in
// peacockdb-core/tests/test_inc2_conformance.rs.
namespace {
std::vector<int32_t> gpu_partition_ids(cudf::table_view const& keys,
                                       std::vector<cudf::size_type> const& cols,
                                       cudf::size_type n_parts) {
  auto pid = peacock::partitioning::spark_partition_ids(keys, cols, n_parts, /*seed=*/42);
  auto v   = pid->view();
  std::vector<int32_t> ids(v.size());
  cudaMemcpy(ids.data(), v.data<int32_t>(), v.size() * sizeof(int32_t),
             cudaMemcpyDeviceToHost);
  return ids;
}
}  // namespace

TEST(CudfGpu, SparkPartitionIdsMatchCometSingleCol) {
  using cudf::test::strings_column_wrapper;
  strings_column_wrapper rf({"A", "N", "R", "F", "O"});  // q1 chars, no nulls
  cudf::table_view keys{{rf}};
  auto ids = gpu_partition_ids(keys, {0}, 8);
  std::fprintf(stderr, "GPU spark_partition_ids 1-col(['A','N','R','F','O'],8) =");
  for (auto p : ids) std::fprintf(stderr, " %d", p);
  std::fprintf(stderr, "\n");
  // comet reference: A->2 N->0 R->1 F->4 O->6
  EXPECT_EQ(ids, (std::vector<int32_t>{2, 0, 1, 4, 6}));
}

TEST(CudfGpu, SparkPartitionIdsMatchComet2ColWithNulls) {
  using cudf::test::strings_column_wrapper;
  // Full proof-query key shape (l_returnflag, l_linestatus) + a NULL in each col.
  strings_column_wrapper rf({"A", "N", "N", "R", "x", "A"},
                            {true, true, true, true, false, true});
  strings_column_wrapper ls({"F", "F", "O", "F", "F", "x"},
                            {true, true, true, true, true, false});
  cudf::table_view keys{{rf, ls}};
  auto ids = gpu_partition_ids(keys, {0, 1}, 8);
  std::fprintf(stderr, "GPU spark_partition_ids 2-col(rf,ls)+nulls,8 =");
  for (auto p : ids) std::fprintf(stderr, " %d", p);
  std::fprintf(stderr, "\n  rows: (A,F)(N,F)(N,O)(R,F)(NULL,F)(A,NULL)\n");
  // comet reference (multi-key left-to-right seed + Spark null-skip):
  // (A,F)->3 (N,F)->7 (N,O)->1 (R,F)->0 (NULL,F)->4 (A,NULL)->2
  EXPECT_EQ(ids, (std::vector<int32_t>{3, 7, 1, 0, 4, 2}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  peacock_test::install_rmm_pool();
  return RUN_ALL_TESTS();
}
