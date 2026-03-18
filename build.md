
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
scripts/build.sh --cudf_ROOT=$HOME/miniforge3/envs/rapids-26.02 --configure
scripts/build.sh --cudf_ROOT=$HOME/miniforge3/envs/rapids-26.02 --build
scripts/build.sh --cudf_ROOT=$HOME/miniforge3/envs/rapids-26.02 --install
```

When the build is done, resulting binaries will link with cudf dynamically.

To invoke C++ tests, run

```
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/rapids-26.02/lib
cpp/install/build/peacock_gpu_tests
```

To build and run rust components, use cargo

```
cargo test --features rust-only
```

cargo can also be used to build the system end-to-end (
```
export CUDF_ROOT=$HOME/miniforge3/envs/rapids-26.02
cargo build 
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
