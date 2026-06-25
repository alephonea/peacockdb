#!/bin/bash

set -e

# sm_80 = Ampere (A100), sm_90 = Hopper (H100)
CUDA_ARCHITECTURES="80;90"

DO_CONFIGURE=0
DO_BUILD=0
DO_INSTALL=0
CUDF_ROOT=""
CUDF_BUILD_FROM_SOURCE=0
TARGET="cpp"
GCC_VERSION=""  # empty = use default below
COMPILER_LAUNCHER=""  # empty = call the compiler directly (e.g. set to "ccache")

while [ $# -gt 0 ]; do
  case "$1" in
    --configure)              DO_CONFIGURE=1 ;;
    --build)                  DO_BUILD=1 ;;
    --install)                DO_INSTALL=1 ;;
    --all)                    DO_CONFIGURE=1; DO_BUILD=1; DO_INSTALL=1 ;;
    --cudf_ROOT)              CUDF_ROOT="$2"; shift ;;
    --cudf-build-from-source) CUDF_BUILD_FROM_SOURCE=1 ;;
    --target)                 TARGET="$2"; shift ;;
    --gcc-version)            GCC_VERSION="$2"; shift ;;
    --compiler-launcher)      COMPILER_LAUNCHER="$2"; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

BUILD_DIR="${TARGET}/build"
INSTALL_DIR="${TARGET}/install"

# Default to gcc-14 (matches Ubuntu's current default). For nvcc toolkits
# that don't yet support gcc-14 (e.g. CUDA <= 12.2, which caps at gcc-12),
# pass --gcc-version 12 to pick gcc-12/g++-12.
if [ -z "$GCC_VERSION" ]; then
  GCC_VERSION=14
fi
export CC=$(which "gcc-${GCC_VERSION}")
export CXX=$(which "g++-${GCC_VERSION}")
export CUDACXX=$(which nvcc)
if [ -z "$CC" ] || [ -z "$CXX" ]; then
  echo "error: gcc-${GCC_VERSION} / g++-${GCC_VERSION} not found on PATH" >&2
  exit 1
fi

if [ $DO_CONFIGURE -eq 1 ]; then
  mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

  CUDF_CMAKE_FLAGS=""
  if [ $CUDF_BUILD_FROM_SOURCE -eq 1 ]; then
    CUDF_CMAKE_FLAGS="-DCUDF_BUILD_FROM_SOURCE=ON"
  elif [ -n "$CUDF_ROOT" ]; then
    CUDF_CMAKE_FLAGS="-Dcudf_ROOT=${CUDF_ROOT}"
  else
    echo "error: cudf not configured. Either:" >&2
    echo "  --cudf_ROOT <path>        use a host cudf installation" >&2
    echo "  --cudf-build-from-source  build cudf from the vendored submodule" >&2
    exit 1
  fi

  LAUNCHER_CMAKE_FLAGS=""
  if [ -n "$COMPILER_LAUNCHER" ]; then
    # Wrap the host C/C++ compilers (e.g. with ccache). Left off CUDA/nvcc on
    # purpose: ccache + nvcc is unreliable, and the device compile isn't the bottleneck.
    LAUNCHER_CMAKE_FLAGS="-DCMAKE_C_COMPILER_LAUNCHER=${COMPILER_LAUNCHER} -DCMAKE_CXX_COMPILER_LAUNCHER=${COMPILER_LAUNCHER}"
  fi

  cmake -S cpp -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    ${LAUNCHER_CMAKE_FLAGS} \
    ${CUDF_CMAKE_FLAGS}
fi

if [ $DO_BUILD -eq 1 ]; then
  cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
fi

if [ $DO_INSTALL -eq 1 ]; then
  cmake --install "${BUILD_DIR}" --prefix "${INSTALL_DIR}"
fi
