# Reproducible build environment for peacockdb against a pinned cuDF release.
#
# WHY THIS EXISTS
# ---------------
# The committed build scripts hardcode one developer's machine:
#
#   scripts/build-test-shadgpu.sh:5   CUDF_ROOT=/home/dmitry/data/miniforge3/envs/rapids-cuda-12.2
#   scripts/build-test-shadgpu.sh:14  CC=/usr/bin/gcc-12
#
# Instead of editing them, this image MATERIALIZES those exact paths as symlinks
# onto the RAPIDS image's own /opt/conda and its conda-forge gcc. The committed
# scripts then run here unmodified, and the developer's own machine keeps working
# exactly as before. Both hardcoded paths are ARGs, so a different fork of the
# script (a different conda prefix, a different gcc) is a --build-arg away.
#
# The recipe otherwise mirrors .github/actions/cpp-build/action.yml, which is the
# authoritative, green build for this project: build inside the official
# rapidsai/base image with CUDF_ROOT=/opt/conda and conda-forge gcc/gxx.
#
# Build it via scripts/docker-build.sh (which passes every ARG below); this file
# is not meant to be `docker build`-ed by hand.

ARG RAPIDS_IMAGE=rapidsai/base:25.02-cuda12.0-py3.12
FROM ${RAPIDS_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# nvcc from CUDA 12.x hard-rejects gcc newer than 12 in host_config.h, and the
# RAPIDS image ships no host compiler of its own — so pin the conda-forge one.
ARG GCC_VERSION=12
# CMake >= 3.30.4 is required by cpp/CMakeLists.txt (rapids-cmake).
ARG CMAKE_VERSION=3.31.6
ARG NINJA_VERSION=1.12.1

# `gcc`/`gxx` (not gcc_linux-64) is what CI installs: it puts an unversioned
# gcc/g++ directly on $CONDA_PREFIX/bin, which is what the symlinks below need.
# cuda-nvcc is pulled explicitly rather than assumed: rapidsai/base is a RUNTIME
# image, so nvcc is not guaranteed to be in it. Requesting the version already
# present is a no-op; requesting it when absent installs it.
RUN conda install -y -n base -c conda-forge -c nvidia \
      "gcc=${GCC_VERSION}" "gxx=${GCC_VERSION}" \
      "cmake=${CMAKE_VERSION}" "ninja=${NINJA_VERSION}" \
      binutils ccache git curl make patchelf \
      cuda-nvcc \
 && conda clean -afy

# --- Arrow: match the GPU host, not the stock image -----------------------------
# libpeacock_gpu.so is the ONE library in this project that links Arrow, and the
# soname it records has to be one the GPU host can load. Those two disagreed:
#
#   rapidsai/base:25.02          libarrow 18.1.0  -> libarrow.so.1801
#   host rapids-cuda-12.2 env    libarrow 19.0.1  -> libarrow.so.1900
#
# and NOT because the cuDF releases differ — the libcudf build string is identical
# on both sides (25.02.02-cuda12_250303_g8139f3c84f_0). libcudf does not link Arrow
# at all (`readelf -d libcudf.so | grep arrow` is empty), so conda was free to solve
# a different Arrow into each, and did. Arrow is purely OUR dependency, which is why
# moving it here is safe: nothing in the RAPIDS stack has an opinion about it.
#
# The `rapids` metapackage is what pinned Arrow down. It drags custreamz -> cudf_kafka
# -> librdkafka -> lz4-c <1.10, and Arrow 19 needs lz4-c 1.10; the solve is
# unsatisfiable until it is gone. None of that subtree is used to BUILD peacockdb —
# this image needs libcudf and its headers, not the kafka/streamz surface — and the
# host env does not have it either (164 packages, no `rapids`, no kafka). So removing
# it moves this image TOWARD the host rather than away from it.
#
# Without this the artifacts have to be shipped with their own Arrow tree and put
# ahead of the host's on LD_LIBRARY_PATH, which is what CI still does
# (.github/workflows/pipeline.yml, "Bundle Arrow/Parquet runtime libraries"). CI is
# untouched by this file and keeps working; this path no longer needs the bundle.
#
# `=*_cpu` is not decoration: conda-forge also publishes `*_cuda` Arrow builds that
# depend on the __cuda virtual package, which is absent during `docker build` (no
# GPU), and the solve fails on it. CONDA_OVERRIDE_CUDA is belt-and-braces for the
# rest of the solve, which sees cuda-version from the already-installed libcudf.
#
# The closing `test -e` asserts the soname the version implies (Arrow encodes it as
# major*100+minor), so a wrong ARROW_VERSION fails the image build here instead of
# surfacing three phases later as a link error that mentions nothing about conda.
ARG ARROW_VERSION=19.0.1
RUN conda remove -y -n base --force \
      rapids custreamz cudf_kafka libcudf_kafka librdkafka streamz \
 && CONDA_OVERRIDE_CUDA=12.0 conda install -y -n base -c conda-forge \
      "libarrow=${ARROW_VERSION}=*_cpu" \
      "libarrow-acero=${ARROW_VERSION}=*_cpu" \
      "libarrow-dataset=${ARROW_VERSION}=*_cpu" \
      "libarrow-substrait=${ARROW_VERSION}=*_cpu" \
 && conda clean -afy \
 && soname=$(echo "${ARROW_VERSION}" | awk -F. '{printf "%d%02d", $1, $2}') \
 && test -e "/opt/conda/lib/libarrow.so.${soname}"

# --- Shim 1: the gcc-12 / g++-12 paths scripts/build.sh resolves via `which` ---
RUN ln -sf /opt/conda/bin/gcc /usr/bin/gcc-${GCC_VERSION} \
 && ln -sf /opt/conda/bin/g++ /usr/bin/g++-${GCC_VERSION}

# --- Shim 2: the conda prefix build-test-shadgpu.sh hardcodes as CUDF_ROOT -----
ARG SHIM_CUDF_ROOT=/home/dmitry/data/miniforge3/envs/rapids-cuda-12.2
RUN mkdir -p "$(dirname "${SHIM_CUDF_ROOT}")" \
 && ln -sfn /opt/conda "${SHIM_CUDF_ROOT}"

# --- Rust: edition 2024 needs rustc >= 1.85, older than any current stable -----
# Installed to a world-readable prefix so the container can run as the HOST user
# (uid/gid passed at `docker run` time) and still use the toolchain. CARGO_HOME is
# re-pointed at a writable mounted cache dir at run time; the toolchain itself is
# read-only, which is all cargo needs of it.
ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path \
 && chmod -R a+rX /opt/rust

# /opt/conda/bin must precede the system PATH so conda's gcc/cmake/ninja/python
# win; the RAPIDS entrypoint normally does this via conda activate, which a
# non-interactive `docker run <cmd>` would skip.
ENV PATH=/opt/rust/cargo/bin:/opt/conda/bin:${PATH} \
    CUDF_ROOT=/opt/conda \
    LD_LIBRARY_PATH=/opt/conda/lib \
    LDFLAGS=-Wl,-rpath-link,/opt/conda/lib \
    CMAKE_GENERATOR=Ninja \
    CONDA_PREFIX=/opt/conda

WORKDIR /work
