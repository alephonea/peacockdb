//! Asserts every integration-test target is actually NAMED by a CI workflow step.
//!
//! Why this exists. CI does not sweep Rust test targets as a set — pipeline.yml lists
//! each `cargo test ... --test <name>` by hand. So a new `tests/test_*.rs` is invisible
//! to CI until someone remembers to add it, and `cargo test` locally still runs it,
//! which makes the gap look like coverage (it has happened: a guard shipped able to go
//! red locally but not at the merge gate, and a C++ test binary was built and shipped
//! but never executed). The C++ side got a glob + ran-any assertion (gpu-tests job);
//! Rust cannot glob (targets are named in Cargo/CI), so this test is the equivalent
//! guard: it fails when a target exists that no workflow runs.
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
    // test_cpu_h200 is NOT exempt: it needs no GPU (runs in ~25s) and owns the
    // REVERSE half of the cost-registry check for the ftc_tp1 column — leaving it
    // out of CI would let a CSV row claim coverage no test provides.
    ("test_ci_coverage", "this test"),
];

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Does this workflow line actually RUN `--test <name>`?
///
/// Matching is line-wise and deliberately strict, because a coverage guard that
/// reports FALSE coverage is worse than no guard at all. Two ways a naive
/// `workflows.contains("--test {name}")` lies:
///
///   - PREFIX COLLISION. `--test test_cpu_executor` is a substring of
///     `--test test_cpu_executor_misc`. This repo HAS that prefix pair, so deleting
///     the standalone `test_cpu_executor` step would still report it covered — one
///     edit away from a live hole. Fixed by requiring a word boundary (whitespace
///     or end-of-line) after the name.
///   - `--no-run` BLINDNESS. `cargo test --no-run ... --test X` BUILDS X without
///     running it. A target named only in such a step is "wired" while never
///     executing — precisely the built-but-never-run hole (peacock_tpchv_tests) that
///     this guard exists to close. Fixed by skipping those lines entirely.
///
/// Both are safe to check line-wise: every `--no-run` invocation in the workflows
/// carries its `--test` flags on the SAME line.
fn line_runs_target(line: &str, name: &str) -> bool {
    if line.contains("--no-run") {
        return false;
    }
    let needle = format!("--test {name}");
    let mut from = 0;
    while let Some(i) = line[from..].find(&needle) {
        let end = from + i + needle.len();
        match line[end..].chars().next() {
            // end-of-line, or a separator -> a real, whole-name mention
            None => return true,
            Some(c) if c.is_whitespace() => return true,
            // otherwise this was a longer target name that merely starts with `name`
            _ => {}
        }
        from = end;
    }
    false
}

/// The matcher's own guard. Both cases below PASS under the naive
/// `workflows.contains("--test {name}")` this replaced, which is the point:
/// without these, a regression back to substring matching is invisible.
#[test]
fn line_matcher_rejects_both_false_coverage_modes() {
    // (1) prefix collision — a longer target name must not cover a shorter one.
    let only_misc = "          cargo test -p peacockdb-core --test test_cpu_executor_misc";
    assert!(
        !line_runs_target(only_misc, "test_cpu_executor"),
        "prefix collision: `--test test_cpu_executor_misc` must NOT count as running \
         test_cpu_executor"
    );
    assert!(line_runs_target(only_misc, "test_cpu_executor_misc"));

    // (2) --no-run blindness — building a target is not running it.
    let build_only =
        "          cargo test --no-run -p peacockdb-core --test test_query_plan --test test_ffi";
    assert!(
        !line_runs_target(build_only, "test_query_plan"),
        "--no-run builds without running; it must not count as CI coverage"
    );

    // A genuine run step still counts, including at end-of-line and mid-line.
    assert!(line_runs_target("cargo test -p x --test test_query_plan", "test_query_plan"));
    assert!(line_runs_target(
        "cargo test -p x --test test_query_plan --test test_ffi",
        "test_query_plan"
    ));
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
    let mut workflow_lines: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&wf_dir).expect("read .github/workflows/") {
        let path = entry.expect("dir entry").path();
        if matches!(path.extension().and_then(|e| e.to_str()), Some("yml") | Some("yaml")) {
            let text = std::fs::read_to_string(&path).expect("read workflow");
            workflow_lines.extend(text.lines().map(str::to_string));
        }
    }
    assert!(!workflow_lines.is_empty(), "no workflow files found under .github/workflows");

    let exempt: BTreeSet<&str> = INTENTIONALLY_NOT_IN_CI.iter().map(|(n, _)| *n).collect();

    let missing: Vec<&String> = targets
        .iter()
        .filter(|t| !exempt.contains(t.as_str()))
        // Must be RUN by some line: `--test <name>` at a word boundary, in a step
        // that is not `--no-run`. See line_runs_target for why both matter.
        .filter(|t| !workflow_lines.iter().any(|l| line_runs_target(l, t)))
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
