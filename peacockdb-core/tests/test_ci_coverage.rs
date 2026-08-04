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
    ("test_gpu_full_table", "GPU host only — run by the gpu-tests job from a prebuilt binary, not via cargo"),
    ("test_gpu_partitioned", "GPU host only — same as test_gpu_full_table"),
    ("test_inc2_conformance", "GPU host only — same as test_gpu_full_table"),
    ("test_gpu_executor_misc", "needs the linked C++/CUDA executor; not built in the CPU tiers"),
    // test_cpu_partitioned is NOT exempt: it needs no GPU (its tp8-standard goldens
    // are CPU-emulated) and it owns the cost-registry check for the partitioned_cpu
    // column — leaving it out of CI would let a CSV row claim coverage no test
    // provides.
    ("test_ci_coverage", "this test"),
];

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Join backslash-continued shell lines into one logical line.
///
/// A shell command in a workflow may be split across `\` continuations, and YAML makes
/// that idiomatic for long `cargo test` invocations. [`line_runs_target`] decides
/// `--no-run` per line, so an unfolded build invocation hands it continuation lines
/// that carry `--test` flags but not the `--no-run` that disqualifies them — they read
/// as run steps and the guard reports coverage that does not exist.
///
/// Folding here rather than requiring one physical line per invocation: continuations
/// are legitimate YAML and the next person will reintroduce them.
fn fold_continuations(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut acc = String::new();
    for line in text.lines() {
        let trimmed = line.trim_end();
        match trimmed.strip_suffix('\\') {
            // Keep a separator, or `--test a \` + `--test b` would fuse into one token.
            Some(head) => {
                acc.push_str(head);
                acc.push(' ');
            }
            None => {
                acc.push_str(trimmed);
                out.push(std::mem::take(&mut acc));
            }
        }
    }
    // A trailing continuation with no terminating line still has to be emitted.
    if !acc.is_empty() {
        out.push(acc);
    }
    out
}

/// Does this workflow line actually RUN `--test <name>`?
///
/// Matching is line-wise and deliberately strict, because a coverage guard that
/// reports FALSE coverage is worse than no guard at all. Two ways a naive
/// `workflows.contains("--test {name}")` lies:
///
///   - PREFIX COLLISION. `--test test_query_plan` is a substring of
///     `--test test_query_plan_misc`. This repo HAS that prefix pair, so deleting
///     the standalone `test_query_plan` step would still report it covered — one
///     edit away from a live hole. Fixed by requiring a word boundary (whitespace
///     or end-of-line) after the name.
///   - `--no-run` BLINDNESS. `cargo test --no-run ... --test X` BUILDS X without
///     running it. A target named only in such a step is "wired" while never
///     executing — precisely the built-but-never-run hole (peacock_tpchv_tests) that
///     this guard exists to close. Fixed by skipping those lines entirely.
///
/// `--no-run` is detected per LINE, so callers MUST pass lines that have already been
/// through [`fold_continuations`]. A build invocation split across `\` continuations
/// carries `--no-run` only on its first physical line, and the continuation lines then
/// read as genuine run steps — which is exactly how this guard silently weakened once
/// (five targets were counted as run by a build continuation).
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
    let only_misc = "          cargo test -p peacockdb-core --test test_query_plan_misc";
    assert!(
        !line_runs_target(only_misc, "test_query_plan"),
        "prefix collision: `--test test_query_plan_misc` must NOT count as running \
         test_query_plan"
    );
    assert!(line_runs_target(only_misc, "test_query_plan_misc"));

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

    // (3) LINE CONTINUATION — the mode that actually shipped. A --no-run build split
    // across `\` carries the flag only on its first physical line, so every target
    // named on a continuation looked like a run step. Five real targets were counted
    // that way; coverage survived only because each ALSO had a genuine run line, i.e.
    // the guard had stopped guarding while still reporting green.
    let build_continued = "          cargo test --no-run -p peacockdb-core --test test_a \\\n\
                           --test test_b --test test_c";
    let folded = fold_continuations(build_continued);
    assert_eq!(folded.len(), 1, "the continuation must fold into ONE logical line: {folded:?}");
    for t in ["test_a", "test_b", "test_c"] {
        assert!(
            !folded.iter().any(|l| line_runs_target(l, t)),
            "{t} is BUILT, not run — a continuation must not count as coverage"
        );
    }
    // ...while a continued RUN step still counts, on any of its physical lines.
    let run_continued = "          cargo test -p peacockdb-core --test test_a \\\n\
                         --test test_b";
    let folded = fold_continuations(run_continued);
    for t in ["test_a", "test_b"] {
        assert!(folded.iter().any(|l| line_runs_target(l, t)), "{t} IS run");
    }
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
            // Folded, NOT raw: see fold_continuations — a build invocation split
            // across `\` would otherwise contribute continuation lines that look
            // like run steps.
            workflow_lines.extend(fold_continuations(&text));
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
