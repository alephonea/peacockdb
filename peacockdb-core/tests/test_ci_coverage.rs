//! Asserts every integration-test target is actually NAMED by a CI workflow step.
//!
//! Why this exists. CI does not sweep Rust test targets as a set — pipeline.yml lists
//! each `cargo test ... --test <name>` by hand. So a new `tests/test_*.rs` is invisible
//! to CI until someone remembers to add it, and `cargo test` locally still runs it, which
//! makes the gap look like coverage. This repo has now been bitten twice:
//!
//!   - `test_plan_bytes` (the FlatBuffer wire-format guard) shipped able to go red
//!     locally but NOT at the merge gate — a guard that cannot fail where it matters.
//!   - `peacock_tpchv_tests` on the C++ side was built, shipped, glibc-patched and
//!     verified present, and still never executed; CI was green having never run it.
//!     That was fixed with a glob + a ran-any assertion (see the gpu-tests job).
//!
//! The C++ side got a glob. Rust cannot glob (targets are named in Cargo/CI), so this
//! test is the equivalent guard: it fails when a target exists that no workflow runs.
//!
//! If this fails you have two honest options — wire the target into pipeline.yml, or
//! add it to `INTENTIONALLY_NOT_IN_CI` below WITH a reason. Deleting the test to make
//! the failure go away recreates the exact hole it exists to close.

use std::collections::BTreeSet;

/// Targets deliberately absent from CI, each with the reason it is exempt.
const INTENTIONALLY_NOT_IN_CI: &[(&str, &str)] = &[
    ("test_gpu", "GPU host only — run by the gpu-tests job from a prebuilt binary, not via cargo"),
    ("test_inc2_conformance", "GPU host only — same as test_gpu"),
    ("test_gpu_executor_misc", "needs the linked C++/CUDA executor; not built in the CPU tiers"),
    ("test_cpu_h200", "H200-device goldens; exercised on the GPU host, not in the CPU tiers"),
    ("test_ci_coverage", "this test"),
];

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn every_rust_test_target_is_named_by_ci() {
    let tests_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut targets: BTreeSet<String> = BTreeSet::new();
    for entry in std::fs::read_dir(&tests_dir).expect("read tests/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        // `common/` is a shared module, not a target; only test_*.rs are targets.
        if stem.starts_with("test_") {
            targets.insert(stem);
        }
    }
    assert!(!targets.is_empty(), "found no tests/test_*.rs — the glob is wrong, not the repo");

    // Read every workflow, not just pipeline.yml: a target named by any of them counts.
    let wf_dir = repo_root().join(".github/workflows");
    let mut workflows = String::new();
    for entry in std::fs::read_dir(&wf_dir).expect("read .github/workflows/") {
        let path = entry.expect("dir entry").path();
        if matches!(path.extension().and_then(|e| e.to_str()), Some("yml") | Some("yaml")) {
            workflows.push_str(&std::fs::read_to_string(&path).expect("read workflow"));
        }
    }
    assert!(!workflows.is_empty(), "no workflow files found under .github/workflows");

    let exempt: BTreeSet<&str> = INTENTIONALLY_NOT_IN_CI.iter().map(|(n, _)| *n).collect();

    let missing: Vec<&String> = targets
        .iter()
        .filter(|t| !exempt.contains(t.as_str()))
        // `--test <name>`; require the flag so a passing mention in prose doesn't count.
        .filter(|t| !workflows.contains(&format!("--test {t}")))
        .collect();

    assert!(
        missing.is_empty(),
        "these test targets exist but NO CI workflow runs them, so they cannot go red at the \
         merge gate:\n{}\n\nWire each into .github/workflows/pipeline.yml, or add it to \
         INTENTIONALLY_NOT_IN_CI with a reason.",
        missing.iter().map(|t| format!("  - {t}")).collect::<Vec<_>>().join("\n")
    );

    // Keep the exemption list honest: an entry for a target that no longer exists is
    // stale and would silently excuse a future target that reuses the name.
    let stale: Vec<&str> =
        exempt.iter().filter(|e| !targets.contains(**e)).copied().collect();
    assert!(
        stale.is_empty(),
        "INTENTIONALLY_NOT_IN_CI names targets that no longer exist: {stale:?} — remove them"
    );
}
