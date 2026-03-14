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

for arg in "$@"; do
  case "$arg" in
    --configure)              DO_CONFIGURE=1 ;;
    --build)                  DO_BUILD=1 ;;
    --install)                DO_INSTALL=1 ;;
    --all)                    DO_CONFIGURE=1; DO_BUILD=1; DO_INSTALL=1 ;;
    --cudf_ROOT=*)            CUDF_ROOT="${arg#--cudf_ROOT=}" ;;
    --cudf-build-from-source) CUDF_BUILD_FROM_SOURCE=1 ;;
    --target=*)               TARGET="${arg#--target=}" ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

BUILD_DIR="${TARGET}/build"
INSTALL_DIR="${TARGET}/install"

if [ $DO_CONFIGURE -eq 1 ]; then
  mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

  CUDF_CMAKE_FLAGS=""
  if [ $CUDF_BUILD_FROM_SOURCE -eq 1 ]; then
    CUDF_CMAKE_FLAGS="-DCUDF_BUILD_FROM_SOURCE=ON"
  elif [ -n "$CUDF_ROOT" ]; then
    CUDF_CMAKE_FLAGS="-Dcudf_ROOT=${CUDF_ROOT}"
  else
    echo "error: cudf not configured. Either:" >&2
    echo "  --cudf_ROOT=<path>        use a host cudf installation" >&2
    echo "  --cudf-build-from-source  build cudf from the vendored submodule" >&2
    exit 1
  fi

  cmake -S cpp -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    ${CUDF_CMAKE_FLAGS}
fi

if [ $DO_BUILD -eq 1 ]; then
  cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
fi

if [ $DO_INSTALL -eq 1 ]; then
  cmake --install "${BUILD_DIR}" --prefix "${INSTALL_DIR}"
fi
