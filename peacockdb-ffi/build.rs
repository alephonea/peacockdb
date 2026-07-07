fn main() {
    // In rust-only mode the C++ library is not built or linked.
    if cfg!(feature = "rust-only") {
        return;
    }

    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root exists");

    let cpp_dir = workspace_root.join("cpp");

    let mut cfg = cmake::Config::new(&cpp_dir);

    // Build the C++/CUDA executor optimized. The cmake crate otherwise mirrors
    // cargo's profile (Debug for `cargo build`/`cargo test`), which builds the
    // cuDF-backed executor unoptimized and far too slow. RelWithDebInfo is
    // release-grade optimization (same runtime as Release) while keeping -g debug
    // symbols, matching the debug=true we keep on the Rust side for backtraces.
    cfg.profile("RelWithDebInfo");

    let cudf_root = std::env::var("CUDF_ROOT").ok();
    let build_from_source = std::env::var("CUDF_BUILD_FROM_SOURCE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if build_from_source {
        cfg.define("CUDF_BUILD_FROM_SOURCE", "ON");
    } else if let Some(root) = &cudf_root {
        cfg.define("cudf_ROOT", root);
        cfg.define("CMAKE_PREFIX_PATH", root);
    } else {
        panic!(
            "cudf not configured. Either:\n\
             - Set CUDF_ROOT=<path> to a cudf installation, or\n\
             - Set CUDF_BUILD_FROM_SOURCE=1 to build from the vendored submodule."
        );
    }

    println!("cargo:rerun-if-env-changed=CUDF_ROOT");
    println!("cargo:rerun-if-env-changed=CUDF_BUILD_FROM_SOURCE");
    let install_dir = cfg.build();

    // Tell rustc where to find libpeacock_gpu.so.
    println!(
        "cargo:rustc-link-search=native={}/lib",
        install_dir.display()
    );
    println!("cargo:rustc-link-lib=dylib=peacock_gpu");

    // Re-run only when actual C++ SOURCES (or the fb schema / cuDF submodule) change —
    // NOT on every `cpp/` mtime bump. `cpp/` also holds build OUTPUTS (cpp/build*,
    // cpp/install, compile_commands.json) that the out-of-crate C++ builds
    // (scripts/build-test*.sh) write into; watching all of `cpp/` made every such
    // build spuriously mark this FFI build script stale → re-run its cmake build and
    // cascade a rebuild through peacockdb-core into the test binaries, even though no
    // source changed. Watch the inputs the cmake build actually compiles instead.
    for rel in ["cpp/CMakeLists.txt", "cpp/src", "cpp/include", "cpp/tests", "flatbuffers"] {
        println!("cargo:rerun-if-changed={}", workspace_root.join(rel).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        workspace_root.join("third_party/cudf").display()
    );
}
