use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let schema = workspace_root.join("flatbuffers/gpu_plan.fbs");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed={}", schema.display());

    // Use the vendored flatc binary built by the flatc-fork crate.
    let flatc = flatc_fork::flatc();

    let status = Command::new(flatc)
        .args(["--rust", "-o"])
        .arg(&out_dir)
        .arg(&schema)
        .status()
        .unwrap_or_else(|e| panic!("failed to run flatc: {e}"));

    assert!(status.success(), "flatc failed with {status}");

    emit_build_profile(&out_dir);
}

/// Bake how this crate was compiled into it, for `peacock_gpu_benchmarks` to write
/// into every measurement record (`build_profile=`).
///
/// Two records built under different profiles look directly comparable and are not
/// (`[profile.benchmarks]` in the workspace Cargo.toml says why). Same trap
/// `sync_floor_us` and `allocator` exist to close: a number whose meaning depends on
/// how it was produced has to carry that with it. Of the three this is the only one
/// baked at compile time, because it is the only one the running process cannot ask
/// about itself.
///
/// Cargo hands a build script `OPT_LEVEL` directly, but not the profile NAME:
/// `PROFILE` collapses every release-inheriting profile to "release". The profile
/// directory does carry it, and `OUT_DIR` is
/// `<target>/<profile-dir>/build/<pkg>-<hash>/out` — hence the fourth ancestor.
/// Unknown rather than a panic if that shape ever changes: a build script is the
/// wrong place to fail the build over a diagnostic string.
fn emit_build_profile(out_dir: &std::path::Path) {
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".into());
    println!("cargo:rustc-env=PEACOCK_BUILD_PROFILE={profile_dir}");
    println!("cargo:rustc-env=PEACOCK_BUILD_OPT_LEVEL={opt_level}");
}
