
# Build instructions for Ubuntu

## Build and run CPU-only tests (no GPU needed)

Setup the environment by installing necessary packages:

```
scripts/init.sh
```

Packages included g++-14, cuda-tookkit-12, recent cmake version, and cudf. cudf is installed with conda.
Make sure `auto_activate_base` is set to false in conda config, such that builds don't attempt to take cuda toolkit and driver from conda environment.

To build just the C++ part of peacockdb, run:

```
scripts/build.sh --cudf_ROOT=$HOME/miniforge3/envs/rapids --configure
scripts/build.sh --cudf_ROOT=$HOME/miniforge3/envs/rapids --build
scripts/build.sh --cudf_ROOT=$HOME/miniforge3/envs/rapids --install
```

`scripts/build.sh` defaults to `gcc-14`. If your conda env's nvcc is older
than CUDA 12.4 it will reject `gcc-14` (nvcc 12.2 caps at gcc-12, nvcc 12.3
caps at gcc-13). In that case install the matching gcc and pass
`--gcc-version`:

```
sudo apt install gcc-12 g++-12
scripts/build.sh --cudf_ROOT=... --gcc-version 12 --configure
```

`build.sh` also forces nvcc's `-ccbin` to match the configured CXX
(`CMAKE_CUDA_HOST_COMPILER`), so the host compiler nvcc invokes for `.cu`
files won't drift back to the system `gcc`.

When the build is done, resulting binaries will link with cudf dynamically.

To invoke C++ tests, run

```
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/rapids/lib
cpp/install/build/peacock_gpu_tests
```

To build and run rust components, use cargo

```
cargo test --features rust-only
```

cargo can also be used to build the system end-to-end (
```
export CUDF_ROOT=$HOME/miniforge3/envs/rapids
cargo build 
```

`peacockdb-ffi/build.rs` runs its own cmake invocation (separate from
`scripts/build.sh`), so the `--gcc-version` flag does not reach it. If you
need a non-default gcc, export `CC`/`CXX` before invoking cargo:

```
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
cargo build
```

After changing the host compiler, wipe the stale cmake cache so the
compiler-ID test reruns: `cargo clean -p peacockdb-ffi` and (for the
C++ tree) `rm -rf cpp/build`.

GPU TO CPU TESTS

```
 # all 5 tests in cpu_executor                                                                                              
  cargo test -p peacockdb-core --lib cpu_executor                                                                            
   
  # one specific test                                                                                                        
  cargo test -p peacockdb-core --test test_cpu_executor_misc test_execution_strips_gpu_nodes
                                                                                                                             
  # with output printed (useful for seeing node names etc.)                                                                  
  cargo test -p peacockdb-core --lib cpu_executor -- --nocapture                                                             
                                                                                                                             
  # everything in the crate                                 
  cargo test -p peacockdb-core                                                                                               
```

## Run GPU tests

Set up environment on the GPU machine in the same way, by installing all packages using init.sh.
System images for GPU machines sometimes come with Cuda toolkit preinstalled. Cuda toolkit version must match the constraint for cudf (conda-install-cudf.sh), i.e. fall between 12.1 and 12.9

Copy the build artifacts from the build machine to the GPU machine:

```
scp -r cpp/install/ <gpu-machine>:~/peacockdb/cpp/install/
```

On the GPU machine, set the library path to the cudf conda environment and run the GPU test binary:

```
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/rapids-26.02/lib
~/peacockdb/cpp/install/build/peacock_gpu_tests
```

If the Rust build was used (`cargo build`), copy the peacockdb binary:

```
scp target/debug/peacockdb <gpu-machine>:~/peacockdb/
```

Run it on the GPU machine with the cudf library path set:

```
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/rapids-26.02/lib
~/peacockdb/peacockdb
```

## RUN Tests

cargo test -p peacockdb-core --test test_query_plan
cargo test -p peacockdb-core --test test_cpu_full_table

LD_LIBRARY_PATH=/home/babanov1403/miniforge3/envs/rapids/lib CUDF_ROOT=/home/babanov1403/miniforge3/envs/rapids cargo test -p peacockdb-core --test test_gpu_full_table -- --nocapture

## RUN All rust non-gpu tests

cargo test --features rust-only
