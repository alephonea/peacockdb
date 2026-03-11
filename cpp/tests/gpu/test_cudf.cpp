#include <cudf/aggregation.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>

#include <gtest/gtest.h>

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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
