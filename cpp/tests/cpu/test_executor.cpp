#include <peacock_gpu.h>

#include <gtest/gtest.h>

TEST(PeacockGpu, Version) {
  EXPECT_STREQ(peacock_gpu_version(), "0.1.0");
}

TEST(PeacockGpu, ExecutorCreateDestroy) {
  peacock_executor_t* ex = nullptr;
  ASSERT_EQ(peacock_executor_create(/*gpu_memory_limit=*/0, &ex), 0);
  ASSERT_NE(ex, nullptr);
  peacock_executor_destroy(ex);
}

TEST(PeacockGpu, ExecutorNullOut) {
  EXPECT_NE(peacock_executor_create(0, nullptr), 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
