cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

set(CUDF_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cudf/cpp")

# ---------------------------------------------------------------------------
# Bootstrap rapids-cmake via cuDF's own RAPIDS.cmake.
# This downloads rapids-cmake (a small cmake-only repo) at configure time
# and makes include(rapids-*) available.  Subsequent includes of the same
# file are no-ops because FetchContent tracks the populated state.
# ---------------------------------------------------------------------------
include("${CUDF_SOURCE_DIR}/../cmake/rapids_config.cmake")
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(peacock_gpu)

project(peacock_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# cuDF (built from submodule — no install, tests, or benchmarks)
# ---------------------------------------------------------------------------
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CUDF_BUILD_TESTUTIL OFF CACHE BOOL "" FORCE)

add_subdirectory("${CUDF_SOURCE_DIR}" cudf-build EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# libpeacock_gpu
# ---------------------------------------------------------------------------
add_library(peacock_gpu SHARED
  src/gpu_executor.cpp
)

target_include_directories(peacock_gpu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(peacock_gpu
  PRIVATE
    cudf::cudf
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

include("${CUDF_SOURCE_DIR}/cmake/thirdparty/get_gtest.cmake")
enable_testing()

# CPU tests — no GPU required
add_executable(peacock_cpu_tests tests/cpu/test_executor.cpp)
target_include_directories(peacock_cpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(peacock_cpu_tests PRIVATE peacock_gpu GTest::gtest)
set_target_properties(peacock_cpu_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME peacock_cpu_tests COMMAND peacock_cpu_tests)
set_tests_properties(peacock_cpu_tests PROPERTIES LABELS "cpu")

# GPU tests — requires a CUDA-capable device
add_executable(peacock_gpu_tests tests/gpu/test_cudf.cpp)
target_include_directories(peacock_gpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(peacock_gpu_tests PRIVATE peacock_gpu cudf::cudf GTest::gtest)
set_target_properties(peacock_gpu_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME peacock_gpu_tests COMMAND peacock_gpu_tests)
set_tests_properties(peacock_gpu_tests PROPERTIES LABELS "gpu")

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS peacock_gpu
  LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
