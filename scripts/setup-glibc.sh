#!/bin/bash
#
# Download and build glibc 2.35 into a local prefix (no sudo required).
# Then patchelf the test binaries to use it.
#
# Usage:
#   ./scripts/setup-glibc.sh --repo-dir /path/to/peacockdb --install --patch
#   ./scripts/setup-glibc.sh --repo-dir /path/to/peacockdb --install
#   ./scripts/setup-glibc.sh --repo-dir /path/to/peacockdb --patch --cuda-dir /usr/local/cuda-12.2
#
# After patching, run tests with:
#   ./cpp/build/peacock_plan_tests

set -euo pipefail

GLIBC_VERSION="2.35"
PREFIX="$HOME/glibc-${GLIBC_VERSION}"
BUILD_DIR="/tmp/glibc-build-${GLIBC_VERSION}"
SRC_DIR="/tmp/glibc-${GLIBC_VERSION}"
TARBALL="/tmp/glibc-${GLIBC_VERSION}.tar.xz"
DO_INSTALL=0
DO_PATCH=0
REPO_DIR=""
CUDA_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --install)       DO_INSTALL=1 ;;
    --patch)         DO_PATCH=1 ;;
    --repo-dir)      REPO_DIR="$2"; shift ;;
    --cuda-dir)      CUDA_DIR="$2"; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

if [ "$DO_INSTALL" -eq 0 ] && [ "$DO_PATCH" -eq 0 ]; then
  echo "ERROR: specify at least one of --install or --patch"
  exit 1
fi

if [ -z "$REPO_DIR" ]; then
  echo "ERROR: --repo-dir <path> is required"
  exit 1
fi

# Resolve CUDA lib directory.
CUDA_LIB_DIR=""
if [ -n "$CUDA_DIR" ]; then
  for candidate in \
    "${CUDA_DIR}/targets/x86_64-linux/lib" \
    "${CUDA_DIR}/lib64" \
    "${CUDA_DIR}/lib"; do
    if [ -d "$candidate" ]; then
      CUDA_LIB_DIR="$candidate"
      break
    fi
  done
  if [ -z "$CUDA_LIB_DIR" ]; then
    echo "WARNING: --cuda-dir=${CUDA_DIR} specified but no lib directory found"
  else
    echo "--- Using CUDA libs from ${CUDA_LIB_DIR}"
  fi
fi

REPO_DIR="$(cd "$REPO_DIR" && pwd)"
CPP_BUILD_DIR="${REPO_DIR}/cpp/build"

# -----------------------------------------------------------------------
# Step 1: Build glibc
# -----------------------------------------------------------------------

if [ "$DO_INSTALL" -eq 1 ]; then

  # Ensure bison is available (glibc configure requires it).
  if ! command -v bison &>/dev/null || ! bison --version | head -1 | grep -qE '3\.[0-9]'; then
    BISON_VERSION="3.8.2"
    BISON_PREFIX="$HOME/.local"
    BISON_TAR="/tmp/bison-${BISON_VERSION}.tar.xz"
    BISON_SRC="/tmp/bison-${BISON_VERSION}"

    if [ ! -x "${BISON_PREFIX}/bin/bison" ]; then
      echo "==> Building bison ${BISON_VERSION} (required by glibc)..."
      if [ ! -f "$BISON_TAR" ]; then
        curl -fSL "https://ftp.gnu.org/gnu/bison/bison-${BISON_VERSION}.tar.xz" -o "$BISON_TAR"
      fi
      if [ ! -d "$BISON_SRC" ]; then
        tar -xf "$BISON_TAR" -C /tmp
      fi
      cd "$BISON_SRC"
      ./configure --prefix="$BISON_PREFIX" 2>&1 | tail -3
      make -j"$(nproc)" 2>&1 | tail -3
      make install 2>&1 | tail -3
      rm -rf "$BISON_SRC" "$BISON_TAR"
    fi

    export PATH="${BISON_PREFIX}/bin:$PATH"
    echo "--- Using bison: $(bison --version | head -1)"
  fi

  echo "==> Building glibc ${GLIBC_VERSION} into ${PREFIX}"

  # Download source.
  if [ ! -f "$TARBALL" ]; then
    echo "--- Downloading glibc ${GLIBC_VERSION} source..."
    curl -fSL "https://ftp.gnu.org/gnu/glibc/glibc-${GLIBC_VERSION}.tar.xz" -o "$TARBALL"
  fi

  # Extract.
  if [ ! -d "$SRC_DIR" ]; then
    echo "--- Extracting..."
    tar -xf "$TARBALL" -C /tmp
  fi

  # Build out-of-tree.
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"

  echo "--- Configuring..."
  "${SRC_DIR}/configure" \
    --prefix="$PREFIX" \
    --disable-werror \
    --disable-profile \
    --enable-shared \
    --without-selinux \
    CFLAGS="-O2 -g0" 2>&1 | tail -5

  echo "--- Building (this takes a few minutes)..."
  make -j"$(nproc)" 2>&1 | tail -3

  echo "--- Installing to ${PREFIX}..."
  make install 2>&1 | tail -3

  # Clean up build artifacts (keep the prefix).
  rm -rf "$BUILD_DIR" "$SRC_DIR" "$TARBALL"

  echo "==> glibc ${GLIBC_VERSION} installed to ${PREFIX}"
fi

if [ "$DO_PATCH" -eq 0 ]; then
  echo "==> Done (install only, skipping patch)."
  exit 0
fi

# -----------------------------------------------------------------------
# Step 2: Ensure patchelf is available
# -----------------------------------------------------------------------

if ! command -v patchelf &>/dev/null; then
  echo "--- patchelf not found, installing locally..."
  PATCHELF_VERSION="0.18.0"
  PATCHELF_DIR="/tmp/patchelf-${PATCHELF_VERSION}"
  PATCHELF_TAR="/tmp/patchelf-${PATCHELF_VERSION}.tar.gz"

  if [ ! -f "$HOME/.local/bin/patchelf" ]; then
    curl -fSL "https://github.com/NixOS/patchelf/releases/download/${PATCHELF_VERSION}/patchelf-${PATCHELF_VERSION}-x86_64.tar.gz" \
      -o "$PATCHELF_TAR"
    mkdir -p "$PATCHELF_DIR"
    tar -xzf "$PATCHELF_TAR" -C "$PATCHELF_DIR"
    mkdir -p "$HOME/.local/bin"
    cp "$PATCHELF_DIR/bin/patchelf" "$HOME/.local/bin/"
    rm -rf "$PATCHELF_DIR" "$PATCHELF_TAR"
  fi

  export PATH="$HOME/.local/bin:$PATH"
fi

# -----------------------------------------------------------------------
# Step 3: Patch binaries to use the local glibc
# -----------------------------------------------------------------------

INTERP="${PREFIX}/lib/ld-linux-x86-64.so.2"

if [ ! -f "$INTERP" ]; then
  echo "ERROR: interpreter not found at ${INTERP}"
  echo "Run with --install first to build glibc."
  exit 1
fi

CPP_INSTALL_DIR="${REPO_DIR}/cpp/install"

# The set of binaries to patch is DERIVED from what is actually shipped, not hardcoded.
# It used to be a fixed list, and a new test binary (peacock_tpch_tests) was silently
# missed: it shipped with the stock ELF interpreter and died with an instant SIGSEGV and
# no output — a symptom that looks nothing like a packaging fault and costs an hour to
# diagnose. Anything ELF+executable in the install bin dir (or matching peacock_* in a
# build dir) gets patched, so a new target cannot be forgotten. verify_patched() below
# then fails the script if any shipped executable slipped through anyway.
LIBS=(
  libpeacock_gpu.so
)

# Which shipped executables actually need the newer glibc: the ones that link OUR stack
# (libcudf / libpeacock_gpu / libarrow). Deriving the list from the directory alone is too
# broad — cpp/install/bin also holds vendored HOST tools (flatc), and repointing those at
# glibc-2.35 breaks them outright: flatc came back
#   "error while loading shared libraries: libstdc++.so.6: cannot open shared object file"
# because the new loader no longer finds the system C++ runtime. So the rule is
# capability-based, not name-based: patch what links our libraries, leave everything else
# on the system loader. A future test binary is covered whatever it is called.
#
# ABOUT flatc SPECIFICALLY, so nobody "fixes" the exclusion: the shipped flatc is
# INDEPENDENTLY BROKEN on the GPU host and always has been. Built on the build host
# against a newer toolchain, it needs GLIBC_2.33/2.34 and GLIBCXX_3.4.29; that host is
# Ubuntu GLIBC 2.31. Patching it to our glibc-2.35 loader does not rescue it either (it
# then fails to find libstdc++) — it only changes which error prints. This is latent, not
# a live failure: codegen runs on the build host and nothing on the GPU host invokes
# flatc. Excluding it here is correct; making it run would mean shipping a matching
# libstdc++, which is a separate decision, not a patcher bug.
needs_patch() {
  local needed
  needed="$(patchelf --print-needed "$1" 2>/dev/null || true)"
  echo "$needed" | grep -qE '^(libcudf|libpeacock_gpu|libarrow)' && return 0
  # rust test binaries link libpeacock_gpu at runtime via rpath, not always DT_NEEDED
  case "$(basename "$1")" in peacock_*) return 0 ;; esac
  return 1
}

patch_rpath() {
  local target="$1"
  local current
  current="$(patchelf --print-rpath "$target" 2>/dev/null || true)"

  # Strip previous glibc and cuda entries to make this idempotent.
  # `|| true`: under `set -euo pipefail`, if `grep -v` filters out EVERY line (an
  # already-patched lib whose only rpath entry is the one being stripped) grep exits
  # 1 and pipefail would abort the whole script — making re-patching non-idempotent.
  local cleaned
  cleaned="$( { echo "$current" | tr ':' '\n' \
    | grep -v "^${PREFIX}/lib\$" \
    | { if [ -n "$CUDA_LIB_DIR" ]; then grep -v "^${CUDA_LIB_DIR}\$"; else cat; fi; } \
    | paste -sd ':'; } || true)"

  # Build new rpath: glibc first, then cuda, then original entries.
  local new_rpath="${PREFIX}/lib"
  if [ -n "$CUDA_LIB_DIR" ]; then
    new_rpath="${new_rpath}:${CUDA_LIB_DIR}"
  fi
  if [ -n "$cleaned" ]; then
    new_rpath="${new_rpath}:${cleaned}"
  fi

  patchelf --set-rpath "$new_rpath" "$target"
}

patch_dir() {
  local dir="$1"
  local label="$2"

  if [ ! -d "$dir" ]; then
    echo "--- Skipping ${label} (directory not found)"
    return
  fi

  echo "==> Patching binaries in ${dir} (${label})"

  local target found=0
  for target in "${dir}"/bin/* "${dir}"/peacock_*; do
    [ -f "$target" ] && [ -x "$target" ] || continue
    # ELF check: patchelf refuses non-ELF, so this doubles as the filter
    patchelf --print-interpreter "$target" >/dev/null 2>&1 || continue
    needs_patch "$target" || { echo "--- Leaving $(basename "$target") on the system loader (does not link our stack)"; continue; }
    echo "--- Patching $(basename "$target")"
    patchelf --set-interpreter "$INTERP" "$target"
    patch_rpath "$target"
    found=$((found + 1))
  done
  [ "$found" -gt 0 ] || echo "--- No ELF executables found in ${dir}"

  for lib in "${LIBS[@]}"; do
    target="${dir}/${lib}"
    [ -d "${dir}/lib" ] && target="${dir}/lib/${lib}"
    if [ -f "$target" ]; then
      echo "--- Patching ${lib} rpath"
      patch_rpath "$target"
    fi
  done
}

# Rust integration test binaries (cargo test --no-run output, staged by
# build-test-shadgpu.sh / CI under cpp/install/rust-tests/). They live one
# directory deep next to cpp/install/lib, so $ORIGIN/../lib resolves
# libpeacock_gpu.so. Filenames vary, so we ELF-detect by trying patchelf.
patch_rust_dir() {
  local dir="$1"

  if [ ! -d "$dir" ]; then
    echo "--- Skipping rust tests (directory not found: ${dir})"
    return
  fi

  echo "==> Patching rust test binaries in ${dir}"

  local f current cleaned new_rpath
  for f in "$dir"/*; do
    [ -f "$f" ] && [ -x "$f" ] || continue
    if ! patchelf --print-interpreter "$f" >/dev/null 2>&1; then
      continue
    fi

    echo "--- Patching $(basename "$f")"
    patchelf --set-interpreter "$INTERP" "$f"

    # Same approach as patch_rpath, plus a $ORIGIN/../lib entry so libpeacock_gpu.so
    # resolves from the sibling lib/ dir without depending on LD_LIBRARY_PATH.
    current="$(patchelf --print-rpath "$f" 2>/dev/null || true)"
    # `|| true`: tolerate grep -v filtering out every entry (idempotent re-patch) —
    # otherwise pipefail aborts before the next binary / the rust-tests step.
    cleaned="$( { echo "$current" | tr ':' '\n' \
      | grep -v "^${PREFIX}/lib\$" \
      | { if [ -n "$CUDA_LIB_DIR" ]; then grep -v "^${CUDA_LIB_DIR}\$"; else cat; fi; } \
      | grep -vF '$ORIGIN/../lib' \
      | paste -sd ':'; } || true)"

    new_rpath="${PREFIX}/lib"
    if [ -n "$CUDA_LIB_DIR" ]; then
      new_rpath="${new_rpath}:${CUDA_LIB_DIR}"
    fi
    new_rpath="${new_rpath}:\$ORIGIN/../lib"
    if [ -n "$cleaned" ]; then
      new_rpath="${new_rpath}:${cleaned}"
    fi

    patchelf --set-rpath "$new_rpath" "$f"
  done
}

# Belt and braces: patching is derived from the filesystem now, but an unpatched binary
# fails at RUNTIME with a bare SIGSEGV that reads like a code bug, so prove the outcome
# instead of trusting the loop. Every shipped executable must carry the patched
# interpreter; name the ones that don't and fail here, where the cause is obvious.
verify_patched() {
  local unpatched=() f interp
  for f in "${CPP_INSTALL_DIR}"/bin/* "${CPP_INSTALL_DIR}"/rust-tests/* \
           "${CPP_INSTALL_DIR}"/rust-benchmarks/*; do
    [ -f "$f" ] && [ -x "$f" ] || continue
    interp="$(patchelf --print-interpreter "$f" 2>/dev/null || true)"
    [ -n "$interp" ] || continue          # not an ELF executable
    # Same predicate as patch_dir: a vendored host tool left on the system loader is
    # correct, not a miss. Only binaries linking our stack must be patched.
    needs_patch "$f" || continue
    [ "$interp" = "$INTERP" ] || unpatched+=("$f (interp: $interp)")
  done
  if [ ${#unpatched[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: shipped executables are NOT patched for glibc ${GLIBC_VERSION:-2.35}:" >&2
    printf '  %s\n' "${unpatched[@]}" >&2
    echo "They would die at startup with a bare SIGSEGV and no output." >&2
    return 1
  fi
  echo "==> Verified: every shipped executable uses ${INTERP}"
}

patch_dir "$CPP_BUILD_DIR" "build"
patch_dir "$CPP_INSTALL_DIR" "install"
patch_rust_dir "${CPP_INSTALL_DIR}/rust-tests"
# Benchmarks are staged in their OWN directory so build-test-shadgpu.sh's --run
# glob over rust-tests/ does not sweep a tens-of-minutes measurement run into the
# correctness gate (that script's --run enforces it). They are the same kind of
# binary and need the same patching; patch_rust_dir no-ops when the directory is
# absent, so this is free for anyone who never builds them.
patch_rust_dir "${CPP_INSTALL_DIR}/rust-benchmarks"
verify_patched

echo ""
echo "==> Done. Run tests with:"
echo "    LD_LIBRARY_PATH=${PREFIX}/lib:\$LD_LIBRARY_PATH ${CPP_BUILD_DIR}/peacock_plan_tests"
echo ""
echo "    TRAP: apply that LD_LIBRARY_PATH PER-COMMAND (env LD_LIBRARY_PATH=... ./binary)."
echo "    EXPORTING it into your shell makes the HOST's own coreutils load the patched"
echo "    glibc and segfault — readelf/mkdir/grep/tail all die with the SAME signature as"
echo "    an unpatched test binary, which sends you diagnosing the wrong thing entirely."
