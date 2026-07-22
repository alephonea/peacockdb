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
add_library(peacock_gpu SHARED src/gpu_executor.cpp src/plan_executor.cpp src/spark_hash_partition.cu)
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
target_link_libraries(peacock_cpu_tests PRIVATE peacock_gpu cudf::cudf flatbuffers GTest::gtest)
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

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS peacock_gpu LIBRARY DESTINATION lib)
install(TARGETS peacock_gpu_tests peacock_cpu_tests peacock_plan_tests peacock_tpch_tests
        RUNTIME DESTINATION bin)
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
