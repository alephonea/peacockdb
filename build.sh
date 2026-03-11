#!/bin/bash

# Hopper (H100) is 90
CUDA_ARCHITECTURES="70;80;90"

mkdir -p cpp/build cpp/install

cmake -S cpp -B cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX=cpp/install

cmake --build cpp/build --parallel 24

cmake --install cpp/build --prefix cpp/install
