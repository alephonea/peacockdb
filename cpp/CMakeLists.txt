cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

option(USE_HOST_LIBCUDF "Use cuDF installed on the host instead of building from submodule" OFF)

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
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# ---------------------------------------------------------------------------
# cuDF
# ---------------------------------------------------------------------------
if(USE_HOST_LIBCUDF)
  find_package(cudf REQUIRED)
  message(STATUS "Using host cudf: ${cudf_VERSION}")
else()
  message(STATUS "Building cudf from submodule")
  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(CUDF_BUILD_TESTUTIL OFF CACHE BOOL "" FORCE)
  add_subdirectory("${CUDF_SOURCE_DIR}" cudf-build EXCLUDE_FROM_ALL)
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

# Collect and install all transitive shared library dependencies of
# libpeacock_gpu.so, excluding system libs and the host GPU driver.
install(CODE "
  file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES
      \"\$<TARGET_FILE:peacock_gpu>\"
      \"\$<TARGET_FILE:cudf::cudf>\"
    RESOLVED_DEPENDENCIES_VAR   resolved
    UNRESOLVED_DEPENDENCIES_VAR unresolved
    DIRECTORIES    \"\$<TARGET_FILE_DIR:cudf::cudf>\"
    PRE_EXCLUDE_REGEXES  \"libcuda\\\\.so\" \"libnvidia\"
    POST_EXCLUDE_REGEXES \"^/lib/\" \"^/usr/lib/\" \"^/lib64/\"
  )
  foreach(dep IN LISTS resolved)
    file(INSTALL \"\${dep}\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/lib\")
    get_filename_component(dep_name \"\${dep}\" NAME)
    execute_process(COMMAND patchelf --set-rpath \"\$ORIGIN\"
      \"\${CMAKE_INSTALL_PREFIX}/lib/\${dep_name}\")
  endforeach()
  if(unresolved)
    message(WARNING \"Unresolved runtime dependencies: \${unresolved}\")
  endif()
")
