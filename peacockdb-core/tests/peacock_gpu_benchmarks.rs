//! Benchmark run over the batch-partitioned corpus: every query
//! [`corpus_benchmark_cases.inc`](common/corpus_benchmark_cases.inc) names, timed at the
//! modes it names.
//!
//! Asserts nothing about an answer — correctness for these queries is `test_gpu_bp_corpus`.
//! Each case warms up once, runs `BENCH_MEASURED_RUNS` times and keeps the second-smallest
//! by end-to-end time; see `common::corpus_benchmark`.
//!
//! NOT run by CI (see `INTENTIONALLY_NOT_IN_CI` in test_ci_coverage.rs): it needs a GPU,
//! takes tens of minutes, and its output is a measurement rather than a gate. Build and
//! run it with `scripts/build-test-shadgpu.sh --build-benchmarks` and `--run-benchmarks`,
//! which is also what compiles it under `[profile.benchmarks]` rather than the
//! opt-level-1 default, or on the host directly:
//!
//!   ./peacock_gpu_benchmarks --nocapture --test-threads=1
//!
//! `--test-threads=1` is not optional — cuDF/RMM share one process-wide pool, and
//! concurrent queries would time each other's contention.
#![cfg(not(feature = "rust-only"))]
mod common;

/// A benchmark declaration's reading: one test per enabled mode, and no test at all for
/// `none` — a query written down and deliberately not timed.
///
/// Both arms register, and that asymmetry with the tests is the point: `none` is a record
/// rather than an omission, so the file of every mode carries a marker for it instead of
/// leaving it absent for an unstated reason.
///
/// The mode reaches the case as its own spelling rather than as a resolved `BpMode`, for
/// the reason `corpus_gpu` does the same: a `&'static BpMode` in a generated function body
/// would need the table indexed at expansion time, and `mode_named`'s panic — which names
/// the five — is a better failure than a subscript.
macro_rules! corpus_query_benchmark {
    ($dataset:ident, $sf:expr, $query:ident, none) => {
        inventory::submit! {
            common::corpus_benchmark::BenchmarkCase {
                dataset: stringify!($dataset),
                sf: stringify!($sf),
                query: stringify!($query),
                mode: common::corpus_benchmark::NOT_TIMED,
            }
        }
    };
    ($dataset:ident, $sf:expr, $query:ident, $($mode:ident)|+) => {
        $(
            paste::paste! {
                #[tokio::test]
                async fn [<bench_bp_ $dataset _sf $sf _ $query _ $mode>]() {
                    common::corpus_benchmark::benchmark_case(
                        stringify!($dataset),
                        stringify!($sf),
                        &stringify!($query).replace('_', "-"),
                        stringify!($mode),
                    )
                    .await;
                }
            }
            inventory::submit! {
                common::corpus_benchmark::BenchmarkCase {
                    dataset: stringify!($dataset),
                    sf: stringify!($sf),
                    query: stringify!($query),
                    mode: stringify!($mode),
                }
            }
        )+
    };
}

include!("common/corpus_benchmark_cases.inc");

/// The declarations reach the inventory, and the file each mode is to hold is derived from
/// them rather than from the case list read as text.
///
/// Runs on this binary because `inventory` collects per linked binary: the same assertion
/// in another target would sweep an empty list and pass for the wrong reason. It needs no
/// device — what it checks is the expansion, not a run.
#[test]
fn every_declared_mode_names_the_queries_its_file_will_hold() {
    use common::corpus_benchmark::{BenchmarkCase, NOT_TIMED, declared_for};

    let modes: std::collections::BTreeSet<(&str, &str, &str)> = inventory::iter::<BenchmarkCase>
        .into_iter()
        .filter(|case| case.mode != NOT_TIMED)
        .map(|case| (case.dataset, case.sf, case.mode))
        .collect();
    assert!(!modes.is_empty(), "the case list declares at least one mode");

    for (dataset, sf, mode) in modes {
        let declared = declared_for(dataset, sf, mode);
        assert!(
            !declared.is_empty(),
            "{dataset}/sf{sf} at {mode} names no query, so its file would have no sections"
        );
        let mut names: Vec<&str> = declared.iter().map(|(name, _)| name.as_str()).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            before,
            "{dataset}/sf{sf} at {mode} names a query twice, and a section cannot be two"
        );
    }
}

/// Numeric order within the prefix, so a file reads `q2` before `q10` rather than in the
/// order a linker happened to lay the declarations down.
#[test]
fn the_sections_of_a_file_are_ordered_numerically() {
    let declared = common::corpus_benchmark::declared_for("tpch", "40", "bp_tp1_single");
    let names: Vec<&str> = declared.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["q6", "q19"], "6 before 19, not '19' before '6'");
}

/// Where a mode's results land: one file per (dataset, mode), stemmed by the mode.
#[test]
fn a_modes_results_go_to_one_file_per_dataset_and_mode() {
    use common::corpus_benchmark::results_file;

    let path = results_file("tpch", "40", "bp_tp1_single");
    assert!(
        path.ends_with("benchmark-results/tpch.sf40/tp1_single.benchmark.txt"),
        "{}",
        path.display()
    );
}

/// A run that produced one query's section leaves every other query's alone — the property
/// the whole merge exists for, asserted on the declarations this binary actually carries.
///
/// On `merged_text` rather than on the file: what is in question is the merge's semantics
/// under `Sections`, and writing into `testdata/` to ask about them would leave the tree
/// dirty for the answer.
#[test]
fn a_filtered_run_keeps_the_sections_it_did_not_produce() {
    use common::corpus_benchmark::declared_for;
    use common::corpus_golden::{Regeneration, merged_text};

    let declared = declared_for("tpch", "40", "bp_tp1_single");
    let after_six = merged_text("", &declared, "q6", "six\n", Regeneration::Sections);
    let after_both = merged_text(
        &after_six,
        &declared,
        "q19",
        "nineteen\n",
        Regeneration::Sections,
    );
    assert!(after_both.contains("six\n"), "q6 survived q19's write");
    assert!(after_both.contains("nineteen\n"), "q19 was written");
    assert!(
        after_both.find("six").unwrap() < after_both.find("nineteen").unwrap(),
        "sections keep declaration order, not write order"
    );
}

/// The record's switch: the variable names the file, and its absence is what makes an
/// ordinary run a measurement rather than a collection.
///
/// Both branches, because only the pair has teeth: an assertion that no file appeared
/// passes just as well when the writer is broken and writes nothing ever.
///
/// Asserted on the writer rather than by running a case — what is in question is the
/// switch, and a device run to ask about it would cost minutes for one branch. The
/// variable is process-wide and this binary runs `--test-threads=1`, which is what makes
/// setting it here safe.
#[test]
fn the_record_is_written_only_when_a_path_is_named() {
    use common::record::{RECORD_PATH_ENV, RunMeta, append_records};

    let meta = RunMeta {
        dataset: "tpch",
        sf: "40",
        query: "q6",
        mode: "bp-tp1-single",
        timing_mode: "events",
        build_profile: "test",
        allocator: "none",
    };
    let row = "one\ttwo\tthree".to_string();
    let dir = std::env::temp_dir().join(format!("peacock-record-{}", std::process::id()));
    let path = dir.join("records.tsv");
    let restore = std::env::var_os(RECORD_PATH_ENV);
    let _ = std::fs::remove_dir_all(&dir);

    unsafe { std::env::remove_var(RECORD_PATH_ENV) };
    append_records(std::slice::from_ref(&row), &meta);
    assert!(!path.exists(), "no path was named, so nothing should have been written");

    unsafe { std::env::set_var(RECORD_PATH_ENV, &path) };
    append_records(std::slice::from_ref(&row), &meta);
    let written = std::fs::read_to_string(&path).expect("a named path is written to");
    assert!(written.contains(&row), "the row reached the file");
    assert!(
        written.lines().any(|line| line.starts_with("# run: ")),
        "a fresh file carries the run's conditions: {written}"
    );

    let _ = std::fs::remove_dir_all(&dir);
    match restore {
        Some(path) => unsafe { std::env::set_var(RECORD_PATH_ENV, path) },
        None => unsafe { std::env::remove_var(RECORD_PATH_ENV) },
    }
}

/// The record is checked against the plan on every case; here is what that check refuses.
///
/// Both branches, for the reason the switch test above has both: an assertion that valid
/// rows pass is satisfied just as well by a checker that accepts everything, and the two
/// shapes below are exactly the ones this record can take while still looking well formed.
///
/// On literal rows rather than on a run: what is in question is the checker, and the
/// pairing it rejects is one no correct run produces.
#[test]
fn the_record_is_checked_against_what_the_plan_declares() {
    use std::collections::{BTreeMap, BTreeSet};

    use common::record::{COLUMNS, rows_match_the_recipes};

    let at = |column: &str| COLUMNS.iter().position(|name| *name == column).unwrap();
    // One execution's rows, so `run_index` is 0 throughout — a batch spanning two is its
    // own failure and is asserted last.
    let row = |node: usize, seq: u32, call: u64| {
        let mut fields = vec![String::new(); COLUMNS.len()];
        fields[at("node_seq")] = node.to_string();
        fields[at("recipe_seq")] = seq.to_string();
        fields[at("call_index")] = call.to_string();
        fields[at("run_index")] = "0".to_string();
        fields.join("\t")
    };
    let of_run = |run: u64, node: usize, seq: u32, call: u64| {
        let mut fields: Vec<String> = row(node, seq, call).split('\t').map(str::to_string).collect();
        fields[at("run_index")] = run.to_string();
        fields.join("\t")
    };
    // Node 0 publishes one step, node 1 two — one plan node with two calls, which is why
    // `node_seq` and `recipe_seq` are separate columns in the first place.
    let declared: BTreeMap<usize, BTreeSet<u32>> =
        BTreeMap::from([(0, BTreeSet::from([0])), (1, BTreeSet::from([1, 2]))]);

    let good = [row(0, 0, 0), row(0, 0, 1), row(1, 1, 0), row(1, 2, 0)];
    assert_eq!(rows_match_the_recipes(&good, &declared), Ok(()));

    // The pre-order confusion: node 0 exists, step #1 exists, and only the pair is wrong.
    let crossed = [row(0, 1, 0)];
    assert!(
        rows_match_the_recipes(&crossed, &declared).is_err(),
        "a step belonging to another node's recipe is not this node's"
    );

    // A dropped row leaves the rest well formed and the totals merely smaller.
    let gap = [row(0, 0, 0), row(0, 0, 2)];
    assert!(
        rows_match_the_recipes(&gap, &declared).is_err(),
        "call 1 was made and never written down"
    );

    // Ten executions appended together, which is what the file holds and what this check
    // must therefore be given one at a time. Caught by `run_index` and named as that,
    // rather than surfacing as the step-called-twice it would otherwise look like.
    let two_runs = [row(0, 0, 0), of_run(1, 0, 0, 0)];
    let mixed = rows_match_the_recipes(&two_runs, &declared).unwrap_err();
    assert!(
        mixed.contains("executions 0 and 1"),
        "two executions at once should be named as that, not as a repeated call: {mixed}"
    );
}
