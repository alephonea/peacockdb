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
# libpeacock_gpu
# ---------------------------------------------------------------------------
add_library(peacock_gpu SHARED src/gpu_executor.cpp)
target_include_directories(peacock_gpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(peacock_gpu PRIVATE cudf::cudf)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
include("${CUDF_SOURCE_DIR}/cmake/thirdparty/get_gtest.cmake")
enable_testing()

# CPU tests — no GPU required at runtime, but needs libpeacock_gpu
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
install(TARGETS peacock_gpu LIBRARY DESTINATION lib)
install(TARGETS peacock_gpu_tests peacock_cpu_tests RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Ensure installed test binaries can find libpeacock_gpu.so in ../lib.
set_target_properties(peacock_cpu_tests peacock_gpu_tests PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"
)

# Collect and install all transitive shared library dependencies of
# libpeacock_gpu.so, excluding system libs and the host GPU driver.
# Skipped when using host cudf — the run host is expected to provide it.
if(CUDF_BUILD_FROM_SOURCE)
install(CODE "
  file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES
      \"\$<TARGET_FILE:peacock_gpu>\"
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
endif()
