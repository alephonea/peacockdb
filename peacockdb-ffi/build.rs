fn main() {
    // In cpu-only mode the C++ library is not built or linked.
    if cfg!(feature = "cpu-only") {
        return;
    }

    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root exists");

    let cpp_dir = workspace_root.join("cpp");

    let num_jobs = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let install_dir = cmake::Config::new(&cpp_dir)
        .build_arg("--parallel")
        .build_arg(num_jobs.to_string())
        .build();

    // Tell rustc where to find libpeacock_gpu.so.
    println!(
        "cargo:rustc-link-search=native={}/lib",
        install_dir.display()
    );
    println!("cargo:rustc-link-lib=dylib=peacock_gpu");

    // Re-run if C++ sources or the cuDF submodule HEAD changes.
    println!("cargo:rerun-if-changed={}", cpp_dir.display());
    println!(
        "cargo:rerun-if-changed={}",
        workspace_root.join("third_party/cudf").display()
    );
}
