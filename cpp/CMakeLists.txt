cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

option(CUDF_BUILD_FROM_SOURCE "Build cuDF from the vendored submodule instead of using a host installation" OFF)

set(CUDF_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cudf/cpp")

# ---------------------------------------------------------------------------
# Bootstrap rapids-cmake and declare CUDA language
# ---------------------------------------------------------------------------
include("${CUDF_SOURCE_DIR}/../cmake/rapids_config.cmake")
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()
rapids_cuda_init_architectures(peacock_gpu)

# nvcc finds its host compiler via PATH (`gcc`/`g++`) by default, ignoring
# CMAKE_CXX_COMPILER. When the system default gcc is newer than what
# nvcc supports (e.g. gcc-14 with CUDA 12.2), nvcc fails host_config.h's
# version check even though CMAKE_CXX_COMPILER points at a supported gcc.
# Force nvcc's -ccbin to match the configured CXX.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  if(DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "")
  elseif(DEFINED ENV{CXX})
    set(CMAKE_CUDA_HOST_COMPILER "$ENV{CXX}" CACHE FILEPATH "")
  endif()
endif()

project(peacock_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
if(CUDF_BUILD_FROM_SOURCE)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

# ---------------------------------------------------------------------------
# cuDF
# ---------------------------------------------------------------------------
if(CUDF_BUILD_FROM_SOURCE)
  message(STATUS "Building cudf from submodule")
  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(CUDF_BUILD_TESTUTIL OFF CACHE BOOL "" FORCE)
  add_subdirectory("${CUDF_SOURCE_DIR}" cudf-build EXCLUDE_FROM_ALL)
else()
  if(NOT DEFINED cudf_ROOT AND NOT DEFINED ENV{cudf_ROOT}
     AND NOT DEFINED CMAKE_PREFIX_PATH AND NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    message(FATAL_ERROR
      "cudf_ROOT is not set. Either:\n"
      "  - Set -Dcudf_ROOT=<path> or the cudf_ROOT environment variable to a cudf installation, or\n"
      "  - Set -DCUDF_BUILD_FROM_SOURCE=ON to build cudf from the vendored submodule.")
  endif()
  find_package(cudf REQUIRED CONFIG)
  message(STATUS "Using host cudf: ${cudf_VERSION}")
endif()

# ---------------------------------------------------------------------------
# Arrow (IPC serialization of query results)
# ---------------------------------------------------------------------------
# Arrow must come from the same prefix as cuDF. Otherwise CMake can pick up
# the system arrow (e.g. Ubuntu's libarrow-dev at /lib/x86_64-linux-gnu),
# which has a different version and links against an incompatible
# curl/openssl — surfaces as a link-time undefined reference to
# curl_easy_getinfo@CURL_OPENSSL_4 or as ABI mismatches with cuDF.
if(NOT DEFINED Arrow_ROOT AND NOT DEFINED ENV{Arrow_ROOT})
  if(DEFINED cudf_ROOT)
    set(Arrow_ROOT "${cudf_ROOT}" CACHE PATH "Match Arrow with cuDF prefix")
  elseif(DEFINED ENV{cudf_ROOT})
    set(Arrow_ROOT "$ENV{cudf_ROOT}" CACHE PATH "Match Arrow with cuDF prefix")
  endif()
endif()
find_package(Arrow REQUIRED CONFIG)
message(STATUS "Using Arrow: ${Arrow_VERSION} from ${Arrow_DIR}")

# ---------------------------------------------------------------------------
# FlatBuffers (header-only runtime + flatc compiler for codegen)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG        v25.12.19
  GIT_SHALLOW    TRUE
)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
# FlatBuffers is a BUILD-time dependency — a header-only runtime we link in and a
# flatc we run for codegen. Nothing downstream consumes it as an installed package,
# so its install rules are pure liability: with them on, `--target install` copies
# flatbuffers' headers and cmake config into CMAKE_INSTALL_PREFIX, which for the
# main build is cpp/install — the tree the deploy phase rsyncs to the GPU host. It
# is bytes over a slow link for something nothing there reads.
#
# Not to be confused with the "Permission denied installing to /usr/local" failure
# that shows up on a cold peacockdb-ffi reconfigure. That one is NOT about which
# install rules exist: peacockdb-ffi's build.rs runs its own cmake tree in cargo's
# OUT_DIR and the cmake crate does pass -DCMAKE_INSTALL_PREFIX=<OUT_DIR>, but if
# that tree's cache was written with different compilers, cmake prints "you have
# changed variables that require your cache to be deleted", wipes it, reconfigures,
# and the prefix falls back to the default /usr/local. Turning install rules off
# only moves which target trips over it first. The fix is to let the reconfigure
# settle (re-run the build) or delete that OUT_DIR — not to prune install rules.
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(flatbuffers)

# ---------------------------------------------------------------------------
# Generate C++ header from the FlatBuffers schema
# ---------------------------------------------------------------------------
set(FBS_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/../flatbuffers/gpu_plan.fbs")
set(FBS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(FBS_GENERATED_HEADER "${FBS_GENERATED_DIR}/gpu_plan_generated.h")

file(MAKE_DIRECTORY ${FBS_GENERATED_DIR})

add_custom_command(
  OUTPUT  ${FBS_GENERATED_HEADER}
  COMMAND flatc --cpp -o ${FBS_GENERATED_DIR} ${FBS_SCHEMA}
  DEPENDS ${FBS_SCHEMA} flatc
  COMMENT "Generating gpu_plan_generated.h from FlatBuffers schema"
)
add_custom_target(gpu_plan_fbs DEPENDS ${FBS_GENERATED_HEADER})

# ---------------------------------------------------------------------------
# libpeacock_gpu
# ---------------------------------------------------------------------------
# src/plan_executor.cpp was a 3271-line monolith; Inc4b split it into the units
# below. The private headers it grew (src/peacock/*.h) stay under src/ on purpose —
# install(DIRECTORY include/) would otherwise ship executor internals as public API
# alongside the stable C FFI surface in peacock_gpu.h.
add_library(peacock_gpu SHARED
  src/gpu_executor.cpp
  src/expr.cpp
  src/execute_plan.cpp
  src/node_session.cpp
  src/operators/dispatch.cpp
  src/operators/scan.cpp
  src/operators/filter.cpp
  src/operators/project.cpp
  src/operators/aggregate.cpp
  src/operators/join.cpp
  src/operators/sort.cpp
  src/operators/union.cpp
  src/operators/limit.cpp
  src/operators/window.cpp
  src/operators/coalesce.cpp
  src/operators/repartition.cpp
  src/spark_hash_partition.cu)
add_dependencies(peacock_gpu gpu_plan_fbs)
target_include_directories(peacock_gpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${FBS_GENERATED_DIR}/..>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(peacock_gpu PRIVATE cudf::cudf flatbuffers Arrow::arrow_shared)

# Make TESTDATA_DIR available to test sources.
set(PEACOCK_TESTDATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../testdata")
get_filename_component(PEACOCK_TESTDATA_DIR "${PEACOCK_TESTDATA_DIR}" ABSOLUTE)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
include("${CUDF_SOURCE_DIR}/cmake/thirdparty/get_gtest.cmake")
enable_testing()

# CPU tests — no GPU required at runtime, but needs libpeacock_gpu
add_executable(peacock_cpu_tests tests/cpu/test_executor.cpp)
target_include_directories(peacock_cpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# cudf::cudf + flatbuffers for the host-only unit tests of plan_executor_internal.h
# (decimal-scale / AST-routing helpers); no GPU is touched at runtime.
# Arrow because `with_declared_precision` is an Arrow-schema rewrite: peacock_gpu links it
# PRIVATE, so the test needs its own to construct the schemas it hands that function.
target_link_libraries(peacock_cpu_tests PRIVATE peacock_gpu cudf::cudf flatbuffers GTest::gtest
                      Arrow::arrow_shared)
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

# Plan executor GPU tests — executes FlatBuffer plans against testdata Parquet files
add_executable(peacock_plan_tests tests/gpu/test_plan_executor.cpp)
target_include_directories(peacock_plan_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${FBS_GENERATED_DIR}/..
)
target_compile_definitions(peacock_plan_tests PRIVATE
  PEACOCK_TESTDATA_DIR="${PEACOCK_TESTDATA_DIR}"
)
target_link_libraries(peacock_plan_tests PRIVATE peacock_gpu cudf::cudf flatbuffers GTest::gtest)
set_target_properties(peacock_plan_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_dependencies(peacock_plan_tests gpu_plan_fbs)
add_test(NAME peacock_plan_tests COMMAND peacock_plan_tests)
set_tests_properties(peacock_plan_tests PROPERTIES LABELS "gpu")

# TPC-H bare-cudf query tests — hand-written operator chains over the LARGE (sf40)
# dataset that lives on the GPU host, checked against committed DuckDB goldens. Skips
# itself when that dataset is absent, so it is safe to run anywhere.
add_executable(peacock_tpch_tests tests/gpu/test_tpch.cpp)
target_include_directories(peacock_tpch_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(peacock_tpch_tests PRIVATE cudf::cudf GTest::gtest)
set_target_properties(peacock_tpch_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME peacock_tpch_tests COMMAND peacock_tpch_tests)
set_tests_properties(peacock_tpch_tests PROPERTIES LABELS "gpu")

# The same four TPC-H queries streamed through a byte-bounded chunked reader, for a device
# that cannot hold a table. Not registered with add_test and not installed: it is a
# measurement binary for small-VRAM hosts, it needs the sf40 dataset, and the answer it
# verifies is already verified by peacock_tpch_tests wherever that one fits. Build it
# explicitly: cmake --build . --target peacock_tpch_streamed_tests
add_executable(peacock_tpch_streamed_tests EXCLUDE_FROM_ALL tests/gpu/test_tpch_streamed.cpp)
target_include_directories(peacock_tpch_streamed_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(peacock_tpch_streamed_tests PRIVATE cudf::cudf GTest::gtest)
set_target_properties(peacock_tpch_streamed_tests PROPERTIES CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED ON)

# Per-operator cuDF timings over sf40 columns — what a query benchmark cannot tell you.
# EXCLUDE_FROM_ALL and no add_test for the same reasons as the streamed binary: it needs the
# sf40 dataset and it measures rather than verifies.
add_executable(peacock_cudf_node_tests EXCLUDE_FROM_ALL tests/gpu/test_cudf_nodes.cpp)
target_include_directories(peacock_cudf_node_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(peacock_cudf_node_tests PRIVATE cudf::cudf GTest::gtest)
set_target_properties(peacock_cudf_node_tests PROPERTIES CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED ON)

# TPC-H+V vector query tests — bare cudf + cuVS BRUTE FORCE (exact; an ANN index cannot
# match an exact DuckDB golden) over the sf40 dataset on the GPU host. Skips itself when
# that dataset or query_params.jsonl is absent.
#
# cuVS must be found the same way cudf is (same conda prefix), and libcuvs.so is unusable
# standalone: rmm is header-only, so libcudf DEFINES the rmm symbols libcuvs REFERENCES.
# Linking both here is what resolves it — see the comment at the top of test_tpchv.cpp.
find_package(cuvs REQUIRED CONFIG)
message(STATUS "Using cuVS: ${cuvs_VERSION}")
add_executable(peacock_tpchv_tests tests/gpu/test_tpchv.cpp)
target_include_directories(peacock_tpchv_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(peacock_tpchv_tests PRIVATE cudf::cudf cuvs::cuvs GTest::gtest)
set_target_properties(peacock_tpchv_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME peacock_tpchv_tests COMMAND peacock_tpchv_tests)
set_tests_properties(peacock_tpchv_tests PROPERTIES LABELS "gpu")

# ---------------------------------------------------------------------------
# Multi-GPU test scaffolding + tests. ALL of this is TEST-ONLY and DELIBERATELY NOT hooked into
# CI: no add_test(), and nothing here is added to install(TARGETS ...), so neither ctest nor the
# "run every shipped binary" CI glob (which enumerates install/bin) picks any of it up. The
# binaries are built for compile coverage and run by hand on a >=2-GPU host; each GTEST_SKIPs
# itself when fewer than two GPUs are visible. Multi-GPU CI runner support is a later task.
# ---------------------------------------------------------------------------

# Shared scaffolding as a TEST-ONLY static library: the worker-per-GPU pool, the row-group
# partitioner, and the cross-GPU peer/gather helpers. Linked into every multi-GPU test binary
# so there is exactly one copy. Not installed, not in CI.
# EXCLUDE_FROM_ALL: these multi-GPU targets are TEST-ONLY and GPU-ONLY — never built or run in
# CI (the CI `all` build runs in a CPU container that can't resolve the rmm headers this test lib
# pulls in during C++20 dep-scan). They are built by hand on a GPU host via explicit
# `cmake --build . --target <name>`, which EXCLUDE_FROM_ALL still permits. Not installed either.
add_library(multi_gpu_testlib STATIC EXCLUDE_FROM_ALL tests/gpu/multi_gpu.cpp)
# `include` is PUBLIC, not PRIVATE: multi_gpu.hpp reaches peacock/rmm_pool.hpp through
# tpch_golden.hpp, so every consumer of this lib compiles that header too. Each multi-GPU
# binary also names the directory itself, which is why the omission stayed invisible.
target_include_directories(multi_gpu_testlib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
# GTest is PUBLIC: multi_gpu.hpp now hosts the shared MultiGpuEnvironment (a ::testing::Environment)
# and benchmark_mgpu, used by both multi-GPU test binaries.
target_link_libraries(multi_gpu_testlib PUBLIC cudf::cudf GTest::gtest)
set_target_properties(multi_gpu_testlib PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
                                                   POSITION_INDEPENDENT_CODE ON)

# Basic two-GPU PoC — cuDF + cuVS on BOTH devices, with a bidirectional cudaMemcpyPeer exchange
# in the middle (see the header of test_basic_multi_gpu.cpp). Needs two visible GPUs to run.
add_executable(peacock_multi_gpu_tests EXCLUDE_FROM_ALL tests/gpu/test_basic_multi_gpu.cpp)
target_include_directories(peacock_multi_gpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(peacock_multi_gpu_tests PRIVATE multi_gpu_testlib cudf::cudf cuvs::cuvs GTest::gtest)
set_target_properties(peacock_multi_gpu_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# Multi-GPU TPC-H query plans (M1: q6 + q1). SAME queries as test_tpch.cpp, checked against the
# SAME committed DuckDB goldens via the shared helper in tpch_golden.hpp. No cuVS needed.
add_executable(peacock_multi_gpu_tpch_tests EXCLUDE_FROM_ALL tests/gpu/test_multi_gpu_tpch.cpp)
target_include_directories(peacock_multi_gpu_tpch_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(peacock_multi_gpu_tpch_tests PRIVATE multi_gpu_testlib cudf::cudf GTest::gtest)
set_target_properties(peacock_multi_gpu_tpch_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# Multi-GPU TPC-H+V vector query plans (M3: q9v/q10v/q11v/q12v). SAME queries as test_tpchv.cpp,
# checked against the SAME committed DuckDB goldens, with the cuVS brute-force SEARCH sharded
# across all visible GPUs. cuVS is REQUIRED; cudf must be linked BEFORE cuvs so the rmm symbols
# libcuvs references (e.g. rmm::logger::~logger) resolve — same order note as peacock_tpchv_tests.
# TEST-ONLY: no add_test, not installed, not in CI.
add_executable(peacock_multi_gpu_tpchv_tests EXCLUDE_FROM_ALL tests/gpu/test_multi_gpu_tpchv.cpp)
target_include_directories(peacock_multi_gpu_tpchv_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(peacock_multi_gpu_tpchv_tests PRIVATE multi_gpu_testlib cudf::cudf cuvs::cuvs GTest::gtest)
set_target_properties(peacock_multi_gpu_tpchv_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS peacock_gpu LIBRARY DESTINATION lib)
install(TARGETS peacock_gpu_tests peacock_cpu_tests peacock_plan_tests peacock_tpch_tests
        peacock_tpchv_tests RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Ensure installed test binaries can find libpeacock_gpu.so in ../lib.
set_target_properties(peacock_cpu_tests peacock_gpu_tests peacock_plan_tests PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"
)

# Helper macro used by both install(CODE) blocks below.
# Copies each resolved dep to install/lib, patches its rpath to $ORIGIN,
# and creates a soname symlink if needed.
set(_install_dep_code "
  foreach(dep IN LISTS resolved)
    file(INSTALL \"\${dep}\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/lib\" FOLLOW_SYMLINK_CHAIN)
    get_filename_component(dep_name \"\${dep}\" NAME)
    set(dep_installed \"\${CMAKE_INSTALL_PREFIX}/lib/\${dep_name}\")
    execute_process(COMMAND patchelf --set-rpath \"\$ORIGIN\" \"\${dep_installed}\")
    execute_process(
      COMMAND patchelf --print-soname \"\${dep_installed}\"
      OUTPUT_VARIABLE dep_soname OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(dep_soname AND NOT dep_soname STREQUAL dep_name)
      file(CREATE_LINK \"\${dep_name}\" \"\${CMAKE_INSTALL_PREFIX}/lib/\${dep_soname}\" SYMBOLIC)
    endif()
  endforeach()
  if(unresolved)
    message(WARNING \"Unresolved runtime dependencies: \${unresolved}\")
  endif()
")

# Always bundle Arrow and its own dependencies so the installed test binaries
# can find them via $ORIGIN/../lib regardless of how cudf was provided.
# Collecting from Arrow::arrow_shared directly avoids accidentally dragging in
# libcudf.so (which is large and host-provided) via peacock_gpu's dep tree.
install(CODE "
  file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES      \"\$<TARGET_FILE:Arrow::arrow_shared>\"
    RESOLVED_DEPENDENCIES_VAR   resolved
    UNRESOLVED_DEPENDENCIES_VAR unresolved
    DIRECTORIES    \"\$<TARGET_FILE_DIR:Arrow::arrow_shared>\"
    PRE_EXCLUDE_REGEXES  \"libcuda\\\\.so\" \"libcudart\\\\.so\" \"libnvidia\"
    POST_EXCLUDE_REGEXES \"^/lib/\" \"^/usr/lib/\" \"^/lib64/\"
  )
  ${_install_dep_code}
")

# When cudf is built from source also bundle cudf and nvcomp themselves.
if(CUDF_BUILD_FROM_SOURCE)
install(CODE "
  file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES
      \"\$<TARGET_FILE:cudf::cudf>\"
      \"\$<TARGET_FILE:nvcomp::nvcomp>\"
    RESOLVED_DEPENDENCIES_VAR   resolved
    UNRESOLVED_DEPENDENCIES_VAR unresolved
    DIRECTORIES
      \"\$<TARGET_FILE_DIR:cudf::cudf>\"
      \"\$<TARGET_FILE_DIR:nvcomp::nvcomp>\"
    PRE_EXCLUDE_REGEXES  \"libcuda\\\\.so\" \"libcudart\\\\.so\" \"libnvidia\"
    POST_EXCLUDE_REGEXES \"^/lib/\" \"^/usr/lib/\" \"^/lib64/\"
  )
  ${_install_dep_code}
")
endif()
