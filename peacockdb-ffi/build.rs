fn main() {
    // In cpu-only mode the C++ library is not built or linked.
    if cfg!(feature = "rust-only") {
        return;
    }

    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root exists");

    let cpp_dir = workspace_root.join("cpp");

    let mut cfg = cmake::Config::new(&cpp_dir);
    if let Ok(cudf_root) = std::env::var("CUDF_ROOT") {
        cfg.define("USE_HOST_LIBCUDF", "ON");
        cfg.define("cudf_ROOT", &cudf_root);
        cfg.define("CMAKE_PREFIX_PATH", &cudf_root);
    }
    println!("cargo:rerun-if-env-changed=CUDF_ROOT");
    let install_dir = cfg.build();

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
