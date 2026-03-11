#!/bin/bash

set -e

# Hopper (H100) is 90
CUDA_ARCHITECTURES="70;80;90"

DO_CONFIGURE=0
DO_BUILD=0
DO_INSTALL=0
CUDF_ROOT=""

for arg in "$@"; do
  case "$arg" in
    --configure)    DO_CONFIGURE=1 ;;
    --build)        DO_BUILD=1 ;;
    --install)      DO_INSTALL=1 ;;
    --all)          DO_CONFIGURE=1; DO_BUILD=1; DO_INSTALL=1 ;;
    --cudf_ROOT=*)  CUDF_ROOT="${arg#--cudf_ROOT=}" ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

if [ $DO_CONFIGURE -eq 1 ]; then
  mkdir -p cpp/build cpp/install

  CUDF_CMAKE_FLAGS=""
  if [ -n "$CUDF_ROOT" ]; then
    CUDF_CMAKE_FLAGS="-DUSE_HOST_LIBCUDF=ON -Dcudf_ROOT=${CUDF_ROOT}"
  fi

  cmake -S cpp -B cpp/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX=cpp/install \
    ${CUDF_CMAKE_FLAGS}
fi

if [ $DO_BUILD -eq 1 ]; then
  cmake --build cpp/build --parallel "$(nproc)"
fi

if [ $DO_INSTALL -eq 1 ]; then
  cmake --install cpp/build --prefix cpp/install
fi
