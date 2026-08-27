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

/// Why a target is absent from the normal CI tiers.
///
/// An enum rather than a free-text reason plus a bool, because the two kinds differ in
/// what can be CHECKED: [`Exemption::GpuJob`] makes a claim about a committed workflow
/// array, so the claim is verified below; [`Exemption::NotRun`] asserts only that
/// nothing runs the target, which nothing can confirm.
#[derive(Debug)]
enum Exemption {
    /// Run on the GPU host by pipeline.yml's gpu-tests job, from a prebuilt binary
    /// rather than via cargo. VERIFIED: the target must appear in that job's
    /// `for t in …` staging array. Without that check, dropping a target from the
    /// array retires it silently while this exemption still excuses it — an execution
    /// mode disappearing with nothing red, which is the failure the pipeline comment
    /// warns about in prose but nothing enforced.
    GpuJob,
    /// Not run by any workflow, for the stated reason.
    NotRun(&'static str),
}

/// Targets deliberately absent from the CI tiers this guard sweeps.
const INTENTIONALLY_NOT_IN_CI: &[(&str, Exemption)] = &[
    ("test_gpu_full_table", Exemption::GpuJob),
    ("test_gpu_partitioned", Exemption::GpuJob),
    ("test_inc2_conformance", Exemption::GpuJob),
    ("test_gpu_abi", Exemption::GpuJob),
    ("test_gpu_recipe_walk", Exemption::GpuJob),
    ("test_gpu_executors", Exemption::GpuJob),
    ("test_gpu_executor_misc", Exemption::NotRun(
        "needs the linked C++/CUDA executor; not built in the CPU tiers and not staged \
         for the GPU job",
    )),
    ("peacock_gpu_benchmarks", Exemption::NotRun(
        "GPU host only, tens of minutes, and it MEASURES rather than asserts — there is \
         nothing for a merge gate to go red on. Correctness for the very same case list \
         (all three targets include! common/gpu_cases.inc) is owned by \
         test_gpu_full_table / test_gpu_partitioned, which is where a regression shows up. \
         NOT Exemption::GpuJob: that variant claims membership in the gpu-tests staging \
         array and is verified against it — this target is deliberately not in it",
    )),
    ("diag_flip_audit", Exemption::NotRun(
        "diagnostic printer, no assertions — run by hand while #97/#95 gate the tp8 \
         rollout; wiring it to CI would add a step that cannot fail",
    )),
    // test_cpu_partitioned is NOT exempt: it needs no GPU (its tp8-standard goldens
    // are CPU-emulated) and it owns the cost-registry check for the partitioned_cpu
    // column — leaving it out of CI would let a CSV row claim coverage no test
    // provides.
    ("test_gpu_bp_corpus", Exemption::GpuJob),
    ("test_ci_coverage", Exemption::NotRun("this test")),
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

/// Does this workflow line RUN the lib unit tests (`--lib`, not `--no-run`)?
///
/// The inline `#[cfg(test)]` modules are a target class this guard was blind to: every
/// other invocation in the workflows passes `--test`, which selects integration
/// targets ONLY, so `config`/`gpu_rule`/`resident`'s unit tests ran locally and never
/// at the merge gate. Being invisible to the guard AND to CI is the same hole one
/// level down — a target class nothing enumerates.
fn line_runs_lib_tests(line: &str) -> bool {
    if line.contains("--no-run") {
        return false;
    }
    // Word-boundary check, same reasoning as line_runs_target: `--library` or a longer
    // flag starting with `--lib` must not count.
    let Some(i) = line.find("--lib") else { return false };
    let after_ok = line[i + "--lib".len()..].chars().next().is_none_or(char::is_whitespace);
    after_ok && line.contains("cargo test") && line.contains("-p peacockdb-core")
}

/// The test targets pipeline.yml's gpu-tests job stages and runs, read out of the
/// committed workflow (`for t in <names>; do`).
///
/// Parsed rather than duplicated: this is the array a [`Exemption::GpuJob`] entry
/// points at, so a copy here would defeat the check it exists to make.
fn gpu_job_staged_targets() -> BTreeSet<String> {
    let text = std::fs::read_to_string(repo_root().join(".github/workflows/pipeline.yml"))
        .expect("read pipeline.yml");
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("for t in test_"))
        .expect(
            "pipeline.yml has no `for t in test_…; do` staging loop — the gpu-tests \
             staging step was renamed or removed, and every GpuJob exemption now rests \
             on an array that does not exist",
        );
    line.trim()
        .trim_start_matches("for t in ")
        .split(';')
        .next()
        .unwrap_or("")
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

/// The GPU test binaries `scripts/build-test-shadgpu.sh` stages, from its `RUST_TESTS`
/// array — what a developer's own run ships to the host.
fn shadgpu_staged_targets() -> BTreeSet<String> {
    let text = std::fs::read_to_string(repo_root().join("scripts/build-test-shadgpu.sh"))
        .expect("read build-test-shadgpu.sh");
    let start = text.find("RUST_TESTS=(").expect(
        "build-test-shadgpu.sh has no RUST_TESTS=( array — the staging list was renamed, \
         and the check that it matches CI now rests on a list that does not exist",
    ) + "RUST_TESTS=(".len();
    let end = start + text[start..].find(')').expect("unterminated RUST_TESTS array");
    text[start..end].split_whitespace().map(str::to_string).collect()
}

/// The targets `scripts/build-test.sh` treats as needing a device at run time, from its
/// `gpu_runtime_targets()` heredoc, as `(crate, target)`. That set is SUBTRACTED from the
/// CPU modes and matched WHOLE against `<crate>:<target>`, so the crate is kept rather than
/// discarded: a line with the wrong crate subtracts nothing, and the target it names then
/// runs on a machine with no GPU.
fn gpu_runtime_targets() -> BTreeSet<(String, String)> {
    let text = std::fs::read_to_string(repo_root().join("scripts/build-test.sh"))
        .expect("read build-test.sh");
    let start = text.find("<<'GPUSET'").expect(
        "build-test.sh has no GPUSET heredoc — gpu_runtime_targets() was renamed or \
         reshaped, and both the mode ladder and this check depend on its contents",
    ) + "<<'GPUSET'".len();
    let end = start + text[start..].find("\nGPUSET").expect("unterminated GPUSET heredoc");
    text[start..end]
        .lines()
        .filter_map(|l| l.trim().split_once(':').map(|(c, t)| (c.to_string(), t.to_string())))
        .collect()
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
    // --lib detection: a build is not a run, and the flag needs a word boundary.
    assert!(line_runs_lib_tests("          cargo test --features rust-only -p peacockdb-core --lib"));
    assert!(!line_runs_lib_tests(
        "          cargo test --no-run --features rust-only -p peacockdb-core --lib --test test_plan_bytes"
    ), "--no-run builds the lib target without running it");
    assert!(!line_runs_lib_tests("          cargo test -p peacockdb-core --test test_query_plan"),
            "an integration-only invocation does not run the lib tests");

    // ...while a continued RUN step still counts, on any of its physical lines.
    let run_continued = "          cargo test -p peacockdb-core --test test_a \\\n\
                         --test test_b";
    let folded = fold_continuations(run_continued);
    for t in ["test_a", "test_b"] {
        assert!(folded.iter().any(|l| line_runs_target(l, t)), "{t} IS run");
    }
}

/// Every integration-test target in the WORKSPACE, as `(crate, target)`.
///
/// Two scoping bugs this closes, both of which let a target escape the gate by being
/// somewhere or something the enumeration did not think to look for:
///   - ONE CRATE. This used to read `CARGO_MANIFEST_DIR/tests` only, so
///     peacockdb-ffi's targets were invisible. They happen to be wired, but a new one
///     would never have been noticed. Crates come from the workspace manifest, so
///     adding a member cannot silently shrink the guard's scope.
///   - THE `test_` PREFIX. This used to collect only stems starting with `test_`,
///     which made a naming convention load-bearing and unenforced: `tests/audit_foo.rs`
///     is a real cargo target and was invisible purely because of its name. Every
///     `tests/*.rs` counts now.
fn workspace_test_targets() -> BTreeSet<(String, String)> {
    let root = repo_root();
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("read workspace Cargo.toml");
    // Members are the authority; a hardcoded crate list here would be a second source
    // of truth and would drift exactly as the single-crate glob did.
    let members: Vec<String> = manifest
        .lines()
        .skip_while(|l| !l.starts_with("[workspace]"))
        .skip(1)
        .take_while(|l| !l.starts_with('['))
        .filter_map(|l| l.trim().trim_end_matches(',').strip_prefix('"')?.strip_suffix('"').map(str::to_string))
        .collect();
    assert!(!members.is_empty(), "no [workspace] members parsed from Cargo.toml");

    let mut targets = BTreeSet::new();
    for m in &members {
        let dir = root.join(m).join("tests");
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            // `common/` is a shared module dir, not a target; only *.rs files are.
            targets.insert((m.clone(), path.file_stem().unwrap().to_string_lossy().to_string()));
        }
    }
    targets
}

#[test]
fn every_rust_test_target_is_named_by_ci() {
    let all = workspace_test_targets();
    let targets: BTreeSet<String> = all.iter().map(|(_, t)| t.clone()).collect();
    assert!(!targets.is_empty(), "found no tests/*.rs in any workspace crate — the enumeration is wrong, not the repo");

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

    // The lib unit tests are not an integration target, so the sweep above cannot see
    // them. Assert the run line exists directly, or deleting it silently un-gates the
    // inline #[cfg(test)] modules exactly as it did before this check.
    assert!(
        workflow_lines.iter().any(|l| line_runs_lib_tests(l)),
        "no workflow line runs the peacockdb-core LIB unit tests. Every other cargo \
         invocation passes --test, which selects integration targets only, so the \
         inline #[cfg(test)] modules (batch_partitioned, config, gpu_rule, resident) would run locally \
         and never at the merge gate. Add `cargo test --features rust-only \
         -p peacockdb-core --lib` to the CPU tier."
    );

    // F5: a GpuJob exemption CLAIMS the gpu-tests job runs the target. Verify that
    // against the committed workflow text rather than trusting the claim. The array is
    // read from pipeline.yml, never copied here — a hardcoded list would be a second
    // source of truth and would drift exactly as this exemption did.
    let gpu_staging = gpu_job_staged_targets();
    let unstaged: Vec<&str> = INTENTIONALLY_NOT_IN_CI
        .iter()
        .filter(|(_, e)| matches!(e, Exemption::GpuJob))
        .map(|(n, _)| *n)
        .filter(|n| !gpu_staging.contains(*n))
        .collect();
    assert!(
        unstaged.is_empty(),
        "these targets are exempt on the grounds that the gpu-tests job runs them, but \
         they do NOT appear in its staging array in pipeline.yml: {unstaged:?}\n\
         Either add them back to that array or change their exemption — as it stands \
         they run nowhere and nothing would go red.\nArray currently names: {:?}",
        gpu_staging
    );

    // Keep the exemption list honest: an entry for a target that no longer exists is
    // stale and would silently excuse a future target that reuses the name. Report the
    // stated reason alongside, because that is what the reader has to judge — "is this
    // claim still true?" is answerable, "is `test_foo` still exempt?" is not.
    let stale: Vec<String> = INTENTIONALLY_NOT_IN_CI
        .iter()
        .filter(|(n, _)| !targets.contains(*n))
        .map(|(n, e)| match e {
            Exemption::GpuJob => format!("{n} (exempt as: run by the gpu-tests job)"),
            Exemption::NotRun(why) => format!("{n} (exempt as: {why})"),
        })
        .collect();
    assert!(
        stale.is_empty(),
        "INTENTIONALLY_NOT_IN_CI names targets that no longer exist — remove them:\n  {}",
        stale.join("\n  ")
    );
}

/// Three committed lists name the GPU test binaries — pipeline.yml's staging array,
/// build-test-shadgpu.sh's `RUST_TESTS`, and build-test.sh's `gpu_runtime_targets()` —
/// each read by a different runner, so a target in one and not the others is invisible in
/// exactly the direction that matters. The third is the dangerous one: the CPU modes are
/// built by SUBTRACTING it, whole-line, so an entry with the wrong crate or a stale target
/// name subtracts nothing and a file-gated GPU test RUNS on a host with no device.
///
/// Not checked: a target exempted as `NotRun` may be absent from `gpu_runtime_targets()`
/// with nothing red. No file says "needs a device" — that is what the hand-declared set is
/// for, and `test_gpu_executor_misc` is that shape today.
#[test]
fn the_three_gpu_target_lists_agree() {
    let ci = gpu_job_staged_targets();
    let dev = shadgpu_staged_targets();
    let runtime = gpu_runtime_targets();

    assert_eq!(
        ci, dev,
        "pipeline.yml's gpu-tests staging array and build-test-shadgpu.sh's RUST_TESTS \
         name different sets. A developer's run then proves a different set from the \
         merge gate's, in whichever direction the lists disagree."
    );

    // The heredoc is the only one of the three that names a crate, so it is the only one
    // that can be wrong about it — and it lists the same tests/ tree workspace_test_targets
    // reads, which is what makes a bad prefix or a renamed file checkable here at all.
    let on_disk = workspace_test_targets();
    let unmatched: Vec<String> = runtime.difference(&on_disk).map(qualified).collect();
    assert!(
        unmatched.is_empty(),
        "gpu_runtime_targets() names entries that no workspace test target matches: \
         {unmatched:?}\nbuild-test.sh matches those lines whole against `<crate>:<target>`, so \
         each subtracts nothing — either the crate is wrong or the test file was renamed, and \
         the target it means to hold back now runs on a CPU-only host."
    );

    // The other two lists are bare names, and every target they carry is peacockdb-core's,
    // so that is what a bare name means when set against the crate-qualified heredoc.
    let ci_qualified: BTreeSet<(String, String)> =
        ci.iter().map(|t| ("peacockdb-core".to_string(), t.clone())).collect();

    let unsubtracted: Vec<String> = ci_qualified.difference(&runtime).map(qualified).collect();
    assert!(
        unsubtracted.is_empty(),
        "CI stages these for the GPU host, but build-test.sh's gpu_runtime_targets() does \
         not name them: {unsubtracted:?}\nThat set is subtracted to build the CPU modes, \
         so each of these is currently both skipped by --gpu and RUN on a CPU-only host."
    );

    let exempt: BTreeSet<&str> = INTENTIONALLY_NOT_IN_CI.iter().map(|(n, _)| *n).collect();
    let orphaned: Vec<String> = runtime
        .difference(&ci_qualified)
        .filter(|(_, t)| !exempt.contains(t.as_str()))
        .map(qualified)
        .collect();
    assert!(
        orphaned.is_empty(),
        "gpu_runtime_targets() names these, no CI staging step runs them, and no \
         INTENTIONALLY_NOT_IN_CI entry says that is deliberate: {orphaned:?}"
    );
}

/// The `<crate>:<target>` form build-test.sh matches on, for failure messages.
fn qualified((krate, target): &(String, String)) -> String {
    format!("{krate}:{target}")
}

/// The line that actually runs a staged rust GPU binary, out of a committed runner.
///
/// Scoped to the rust loop and read line-wise, because both shortcuts are false-green.
/// The C++ loop above it in each file uses the same `$t` and rightly passes no flag, so the
/// file's first invocation is the wrong one; and every rust invocation carries a comment
/// saying why the flag is mandatory, so a file-wide `contains` survives exactly the edit
/// that matters — the flag dropped from the command, the comment left behind.
fn rust_gpu_runner_invocation(rel: &str) -> String {
    let text = std::fs::read_to_string(repo_root().join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"));
    let mut after_header = text.lines().skip_while(|l| !is_rust_gpu_runner_loop_header(l));
    after_header.next().unwrap_or_else(|| {
        panic!(
            "{rel} has no `for t in …rust-tests/*` loop — the runner was reshaped, and the \
             single-tenant GPU invariant now rests on a loop this guard cannot find"
        )
    });
    for line in after_header {
        if line.trim_start().starts_with("done") {
            break;
        }
        if is_rust_gpu_runner_invocation(line) {
            return line.trim().to_string();
        }
    }
    panic!(
        "{rel}'s rust-tests loop runs no binary as `\"\\$t\" …` — the invocation was renamed or \
         moved out of the loop, and this guard now reads nothing"
    )
}

fn is_rust_gpu_runner_loop_header(line: &str) -> bool {
    line.contains("for t in") && line.contains("rust-tests/")
}

/// Is this the line that executes the binary, rather than a comment about it?
fn is_rust_gpu_runner_invocation(line: &str) -> bool {
    line.trim_start().starts_with("\"\\$t\"")
}

/// Single-tenant GPU is one flag on two committed runner lines and nothing else. cuDF and
/// RMM share a process-wide pool, so `--test-threads=1` is what keeps the device to one test
/// at a time — and `test_gpu_bp_corpus.rs` spends it further, setting environment variables
/// in an `unsafe` block whose safety argument is that flag. Dropping it from either runner
/// makes that argument false, and until this test existed nothing said so.
#[test]
fn both_gpu_runners_pass_test_threads_one() {
    for rel in [".github/workflows/pipeline.yml", "scripts/build-test-shadgpu.sh"] {
        let line = rust_gpu_runner_invocation(rel);
        assert!(
            line.contains("--test-threads=1"),
            "{rel} runs the staged GPU binaries without --test-threads=1:\n  {line}\n\
             cuDF/RMM share one process-wide pool, so concurrent cases OOM the device, and the \
             env-var writes in test_gpu_bp_corpus.rs are sound only while this flag holds."
        );
    }
}

/// The reader's own guard, over both false-green modes: each block below passes a file-wide
/// search for the flag while the rust binary runs without it.
#[test]
fn the_runner_reader_takes_the_rust_command_and_not_a_comment_or_the_cpp_loop() {
    // (1) the flag dropped from the command, its comment left above it.
    let dropped = "for t in $REMOTE_DIR/cpp/install/rust-tests/*; do\n\
                   \x20 # --test-threads=1: cuDF/RMM share one process-wide pool.\n\
                   \x20 \"\\$t\" --nocapture > \"\\$tlog\" 2>&1\n\
                   done";
    assert!(dropped.contains("--test-threads=1"), "a file-wide search is green on this");
    let read = dropped
        .lines()
        .skip_while(|l| !is_rust_gpu_runner_loop_header(l))
        .find(|l| is_rust_gpu_runner_invocation(l))
        .expect("the invocation");
    assert!(!read.contains("--test-threads=1"), "the command itself has lost the flag");

    // (2) the C++ loop first, sharing the same `$t`: taking the file's first invocation
    // reads that one and never reaches the rust loop at all.
    let both_loops = "for t in $REMOTE_DIR/cpp/install/bin/peacock_*_tests; do\n\
                      \x20 \"\\$t\" > \"\\$tlog\" 2>&1\n\
                      done\n\
                      for t in $REMOTE_DIR/cpp/install/rust-tests/*; do\n\
                      \x20 \"\\$t\" --nocapture --test-threads=1 > \"\\$tlog\" 2>&1\n\
                      done";
    let first = both_loops.lines().find(|l| is_rust_gpu_runner_invocation(l)).expect("a line");
    assert!(!first.contains("--test-threads=1"), "the C++ invocation is the file's first");
    let scoped = both_loops
        .lines()
        .skip_while(|l| !is_rust_gpu_runner_loop_header(l))
        .find(|l| is_rust_gpu_runner_invocation(l))
        .expect("the rust invocation");
    assert!(scoped.contains("--test-threads=1"), "scoping to the rust loop reads the right one");
}
