#!/usr/bin/env bash
#
# Run a peacockdb build inside a pinned-cuDF container.
#
# Building peacockdb needs a full RAPIDS/cuDF prefix, nvcc, a gcc the CUDA release
# accepts, CMake >= 3.30.4 and rustc >= 1.85 — a version-coupled stack that has to
# match the cuDF release the GPU host runs. CI builds inside the official
# rapidsai/base image (pipeline.yml, job cpp-build-2502); this gives a developer the
# same thing locally, for any cuDF release.
#
# It does not reimplement the build. docker/cudf-build.Dockerfile materializes the
# paths scripts/lib/shadgpu-env.sh hardcodes — its conda prefix and /usr/bin/gcc-12 —
# as symlinks onto the image's /opt/conda and conda gcc, so the committed scripts run
# inside unmodified: they stay the source of truth for how to build, this one supplies
# where.
#
# Everything large (the CMake tree, the cargo target dir, the registry, ccache) lands
# in --cache-dir on the host rather than in the repo, which is usually not on the
# biggest disk. cpp/install/ is the exception: build-test-shadgpu.sh --push-binaries
# reads it from there. The container runs as the invoking uid:gid.
#
# This entry point and a native build-test-shadgpu.sh --build do not share a cargo
# cache: here CARGO_TARGET_DIR is /cache/cargo-target with RUSTFLAGS=-C debuginfo=0,
# natively it is $PWD/target-cudf-<cudf>. Alternating between them costs a cold
# DataFusion rebuild each way — a consequence of the isolation, not cache thrash to
# diagnose. Point CARGO_TARGET_DIR at one of them to change that, and accept that the
# debuginfo flag then differs per build.
#
# USAGE
#   scripts/docker-build.sh                       # build cuDF 25.02, the default
#   scripts/docker-build.sh --cudf 25.04
#   scripts/docker-build.sh --cache-dir ~/big-disk/peacock
#   scripts/docker-build.sh --image-only          # (re)build the builder image, stop
#   scripts/docker-build.sh --shell               # interactive shell in the container
#   scripts/docker-build.sh -- cargo test -p peacockdb-core --no-run   # any command
#   scripts/docker-build.sh --no-image -- ./scripts/build-test-shadgpu.sh --build-benchmarks
#
# After it succeeds, deploy and run with the committed script, unchanged:
#   ./scripts/build-test-shadgpu.sh --push-binaries --patch --run

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# --- knobs (env or flag; flag wins) -------------------------------------------
CUDF_VERSION=${PEACOCK_CUDF_VERSION:-25.02}
BASE_IMAGE=${PEACOCK_RAPIDS_IMAGE:-}
BUILDER_TAG=${PEACOCK_BUILDER_IMAGE:-}
CACHE_DIR=${PEACOCK_CACHE_DIR:-}
GCC_VERSION=${PEACOCK_GCC_VERSION:-12}
# The conda prefix the committed build scripts hardcode, read out of
# scripts/lib/shadgpu-env.sh so the two cannot drift: repoint that line and this
# shim follows.
SHIM_CUDF_ROOT=${PEACOCK_SHIM_CUDF_ROOT:-}
IMAGE_ONLY=0
SKIP_IMAGE=0
INNER_CMD=()

# Prints the header block above verbatim, so -h cannot drift from it. The range ends
# at the last comment line before `set -euo pipefail`; keep it in step when the
# header grows, or --help truncates mid-USAGE.
usage() { sed -n '2,39p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --cudf)        CUDF_VERSION=$2; shift 2 ;;
    --image)       BASE_IMAGE=$2; shift 2 ;;
    --tag)         BUILDER_TAG=$2; shift 2 ;;
    --cache-dir)   CACHE_DIR=$2; shift 2 ;;
    --gcc)         GCC_VERSION=$2; shift 2 ;;
    --image-only)  IMAGE_ONLY=1; shift ;;
    --no-image)    SKIP_IMAGE=1; shift ;;
    --shell)       INNER_CMD=(bash); shift ;;
    -h|--help)     usage 0 ;;
    --)            shift; INNER_CMD=("$@"); break ;;
    *)             echo "Unknown flag: $1" >&2; usage 1 ;;
  esac
done

: "${BASE_IMAGE:=rapidsai/base:${CUDF_VERSION}-cuda12.0-py3.12}"
: "${BUILDER_TAG:=peacockdb-build:cudf-${CUDF_VERSION}}"

if [ -z "$SHIM_CUDF_ROOT" ]; then
  SHIM_CUDF_ROOT=$(sed -n 's/^CUDF_ROOT=\(.*\)$/\1/p' "$REPO_DIR/scripts/lib/shadgpu-env.sh" | head -1)
  [ -n "$SHIM_CUDF_ROOT" ] || {
    echo "ERROR: could not read CUDF_ROOT out of scripts/lib/shadgpu-env.sh." >&2
    echo "       Pass it explicitly: --cache-dir ... PEACOCK_SHIM_CUDF_ROOT=<prefix>" >&2
    exit 1
  }
fi

# The biggest mounted filesystem we can name cheaply: a dedicated /build volume if
# the machine has one, else under the home dir.
if [ -z "$CACHE_DIR" ]; then
  if [ -d /build ] && [ -w /build ]; then CACHE_DIR=/build/peacock; else CACHE_DIR="$HOME/.cache/peacock-build"; fi
fi
CACHE_DIR=$(mkdir -p "$CACHE_DIR" && cd "$CACHE_DIR" && pwd)

# Default job: the committed build phase, nothing else. No runtime libs are copied
# out of the conda prefix afterwards — this image pins Arrow to the version the GPU
# host runs (ARROW_VERSION in docker/cudf-build.Dockerfile), so what gets built here
# already asks for a soname the host can resolve.
if [ ${#INNER_CMD[@]} -eq 0 ]; then
  INNER_CMD=(bash -c './scripts/build-test-shadgpu.sh --build')
fi

command -v docker >/dev/null || { echo "ERROR: docker not found on PATH" >&2; exit 1; }

# --- builder image ------------------------------------------------------------
if [ "$SKIP_IMAGE" -eq 0 ]; then
  echo "==> ensuring builder image $BUILDER_TAG (from $BASE_IMAGE)"
  docker build \
    -f "$REPO_DIR/docker/cudf-build.Dockerfile" \
    -t "$BUILDER_TAG" \
    --build-arg "RAPIDS_IMAGE=$BASE_IMAGE" \
    --build-arg "GCC_VERSION=$GCC_VERSION" \
    --build-arg "SHIM_CUDF_ROOT=$SHIM_CUDF_ROOT" \
    "$REPO_DIR/docker"
fi
[ "$IMAGE_ONLY" -eq 1 ] && { echo "==> image only; done"; exit 0; }

# --- host-side cache tree -----------------------------------------------------
# Created here as the invoking user rather than by docker as root: a root-owned
# cache dir makes every later build fail on permissions in a way that reads as a
# compiler error.
for d in cpp-build cargo-target cargo-home ccache home cpm git-mirrors; do
  mkdir -p "$CACHE_DIR/$d"
done
mkdir -p "$REPO_DIR/cpp/install"

echo "==> repo      : $REPO_DIR"
echo "==> cache     : $CACHE_DIR"
echo "==> cudf root : $SHIM_CUDF_ROOT -> /opt/conda (inside container)"
echo "==> command   : ${INNER_CMD[*]}"

docker_args=(
  --rm
  --user "$(id -u):$(id -g)"
  --workdir /work
  -v "$REPO_DIR:/work"
  # Keep the multi-GiB CMake tree and cargo artifacts off the repo's filesystem.
  -v "$CACHE_DIR/cpp-build:/work/cpp/build"
  -v "$CACHE_DIR/cargo-target:/cache/cargo-target"
  -v "$CACHE_DIR/cargo-home:/cache/cargo-home"
  -v "$CACHE_DIR/ccache:/cache/ccache"
  -v "$CACHE_DIR/home:/cache/home"
  -v "$CACHE_DIR/cpm:/cache/cpm"
  -e HOME=/cache/home
  # rapids-cmake fetches dependencies through CPM, and the project is configured
  # twice with separate build trees — once by scripts/build.sh, again by
  # peacockdb-ffi's build.rs inside cargo's OUT_DIR. A shared source cache stops the
  # second tree downloading the same sources again.
  -e CPM_SOURCE_CACHE=/cache/cpm
  -e CARGO_HOME=/cache/cargo-home
  -e CARGO_TARGET_DIR=/cache/cargo-target
  -e CCACHE_DIR=/cache/ccache
  -e CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-5G}"
  # content, not mtime: the mounted repo's timestamps are not stable across
  # checkouts, so a size/mtime check would miss every real cache hit.
  -e CCACHE_COMPILERCHECK=content
  # Matches CI. Debug info is ~400MB per rust GPU test binary, all of which would
  # then cross a slow link to the GPU host for nothing.
  -e RUSTFLAGS="${RUSTFLAGS:--C debuginfo=0}"
  # The repo is bind-mounted from another uid's checkout; without this every git
  # invocation (cmake FetchContent, build scripts) dies on "dubious ownership".
  -e GIT_CONFIG_COUNT=2
  -e GIT_CONFIG_KEY_0=safe.directory
  -e GIT_CONFIG_VALUE_0='*'
  # cpp/CMakeLists.txt pulls flatbuffers via FetchContent rather than CPM, so unlike
  # the CPM packages it is re-cloned per build tree — and there are two. That clone
  # is the one thing that would still reach for the network after everything else is
  # cached. Populate the mirror with:
  #   git clone --mirror https://github.com/google/flatbuffers.git \
  #             "$CACHE_DIR/git-mirrors/flatbuffers.git"
  # Harmless when absent: git only rewrites URLs it has a mapping for.
  -e GIT_CONFIG_KEY_1=url./cache/git-mirrors/flatbuffers.git.insteadOf
  -e GIT_CONFIG_VALUE_1=https://github.com/google/flatbuffers.git
  -v "$CACHE_DIR/git-mirrors:/cache/git-mirrors:ro"
)
# Pass through the committed scripts' own knobs when the caller set them.
for v in CARGO_BUILD_JOBS PEACOCK_FFI_CLEAN CMAKE_BUILD_PARALLEL_LEVEL; do
  [ -n "${!v:-}" ] && docker_args+=(-e "$v=${!v}")
done
[ -t 0 ] && docker_args+=(-it)

exec docker run "${docker_args[@]}" "$BUILDER_TAG" "${INNER_CMD[@]}"
