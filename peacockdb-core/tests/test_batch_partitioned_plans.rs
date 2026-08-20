//! Plan goldens for the batch-partitioned mode: one file per (bench, mode), holding every
//! query the bench has.
//!
//! A section per query — the tree, then `--- recipes ---` and what each node asks of the
//! device, then `--- memory ---` and the estimator's figures. The legacy `.plan.txt`
//! goldens carry the first and the last of those. Refusals are content: a query
//! this mode declines renders its reason where its tree would be, so the file says what
//! the mode does and does not run.

mod common;

use std::path::{Path, PathBuf};

use peacockdb_core::batch_partitioned::plan::{BatchSizing, PlanKnobs, plan_batch_partitioned};
use peacockdb_core::batch_partitioned::plan_text::{
    render_plan, render_plan_memory, render_plan_recipes,
};
use peacockdb_core::batch_partitioned::recipe::attach_recipes;
use peacockdb_core::config::MemoryLimit;

use common::{data_dir_for, golden_dir_for, queries_dir_for};

/// The tier `bp-tp4-sized` is canonized at — the one mode that reads a budget, so changing
/// this regenerates that file alone. Mini is the tier the legacy plan and `.cpu.txt` goldens
/// use, so a figure here reads against one from there; the memory summary line records it
/// per query.
const BUDGET: u64 = MemoryLimit::Mini.bytes() as u64;

/// A scan reading less than this stops being worth splitting: it has nothing to gain from
/// lanes and would pay a shuffle for them.
///
/// From the sf1 measurement at full projection: the largest table that must stay on one lane
/// is tpcds date_dim at 4,006,445 bytes, the smallest that must not is tpcds web_returns at
/// 8,041,397, and tpch supplier at 1,532,237 sets the floor. 5 MiB sits in that gap nearer
/// the lower end, so date_dim would have to grow 31% to cross it and web_returns shrink 35%.
/// It reads the projected bytes of the surviving row groups, so a narrow scan of a big table
/// falls below it — the rule working, not a value to retune.
const SMALL_TABLE_BYTES: u64 = 5 * 1024 * 1024;

struct Mode {
    name: &'static str,
    knobs: PlanKnobs,
}

fn mode(name: &'static str, target_partitions: usize, sizing: BatchSizing) -> Mode {
    Mode {
        name,
        knobs: PlanKnobs {
            target_partitions,
            sizing,
            budget: BUDGET,
            small_table_bytes: SMALL_TABLE_BYTES,
        },
    }
}

fn queries(dataset: &str) -> Vec<(String, PathBuf)> {
    let mut found: Vec<(String, PathBuf)> = std::fs::read_dir(queries_dir_for(dataset))
        .expect("the query directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()?.to_str()? != "sql" {
                return None;
            }
            Some((path.file_stem()?.to_str()?.to_string(), path))
        })
        .collect();
    found.sort();
    found
}

async fn render_bench(dataset: &str, sf: &str, mode: &Mode) -> String {
    let ctx = peacockdb_core::register_tables_for(
        peacockdb_core::build_session_state(mode.knobs.target_partitions),
        &data_dir_for(dataset, sf),
    )
    .await
    .expect("register the tables");

    let mut text = String::new();
    for (name, path) in queries(dataset) {
        let sql = std::fs::read_to_string(&path).expect("the query text");
        text.push_str(&format!("== {name}\n"));
        text.push_str(&render_query(&ctx, &sql, mode.knobs).await);
    }
    text
}

async fn render_query(
    ctx: &datafusion::execution::context::SessionContext,
    sql: &str,
    knobs: PlanKnobs,
) -> String {
    let planned = match ctx.sql(sql).await {
        Ok(frame) => frame.create_physical_plan().await,
        Err(e) => Err(e),
    };
    let plan = match planned {
        Ok(plan) => plan,
        // A query DataFusion itself declines never reaches this mode's planner.
        Err(e) => {
            return format!(
                "refused by datafusion: {}\n",
                relative_to_testdata(&e.to_string())
            );
        }
    };
    match plan_batch_partitioned(&plan, knobs) {
        Ok((tree, model)) => format!(
            "{}--- recipes ---\n{}--- memory ---\n{}",
            render_plan(tree.as_ref()),
            render_plan_recipes(tree.as_ref(), &attach_recipes(tree.as_ref())),
            render_plan_memory(tree.as_ref(), &model)
        ),
        Err(e) => format!("refused: {}\n", relative_to_testdata(&e.to_string())),
    }
}

/// A refusal renders the error it was given, and DataFusion's own error text can carry a
/// plan dump — which names every file by absolute path. Canonized as-is, the golden matches
/// only a checkout at the same path: ours passed here and had never passed in CI, on any
/// machine. So the testdata root renders as `testdata`, in both the leading-slash form and
/// the object-store form that drops it. Nothing else in the message changes: what
/// DataFusion said is the content, and only where it lives on this disk is not.
fn relative_to_testdata(text: &str) -> String {
    let root = std::fs::canonicalize(common::testdata_root())
        .unwrap_or_else(|_| common::testdata_root())
        .to_string_lossy()
        .into_owned();
    text.replace(&root, "testdata")
        .replace(root.trim_start_matches('/'), "testdata")
}

fn golden(dataset: &str, sf: &str, mode: &Mode) -> PathBuf {
    golden_dir_for(dataset, sf).join(format!("{}.plans.txt", mode.name))
}

fn assert_or_update(path: &Path, actual: &str) {
    if std::env::var("UPDATE_CANONICAL").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, actual).unwrap();
        eprintln!("Updated canonical plans: {}", path.display());
        return;
    }
    let canonical = std::fs::read_to_string(path).unwrap_or_else(|_| {
        panic!(
            "canonical file not found: {}\nRun with UPDATE_CANONICAL=1 to generate it.",
            path.display()
        )
    });
    if actual == canonical {
        return;
    }
    let said = section_differences(&canonical, actual);
    panic!(
        "{}: {} of {} queries differ\n{}",
        path.display(),
        said.len(),
        ordered_sections(&canonical).len(),
        said.join("\n")
    );
}

/// Every query whose section moved, one short line each. One file holds every query, so a
/// whole-file `assert_eq!` says only that a two-megabyte golden differs — and a plan line
/// runs past a thousand characters, which the CI log drops, so even the dump it prints
/// arrives unreadable. One line per query, each naming the column that moved, is what
/// survives the log and what a person can scan.
fn section_differences(canonical: &str, actual: &str) -> Vec<String> {
    let expected = ordered_sections(canonical);
    let produced = ordered_sections(actual);
    let mut said = Vec::new();
    for (position, (name, body)) in expected.iter().enumerate() {
        let Some((produced_name, produced_body)) = produced.get(position) else {
            said.push(format!(
                "{name}: in the golden, and the run produced no section there"
            ));
            continue;
        };
        if produced_name != name {
            said.push(format!(
                "section {position}: `{name}` in the golden and `{produced_name}` in the run"
            ));
            continue;
        }
        if produced_body != body {
            said.push(format!("{name}: {}", line_difference(body, produced_body)));
        }
    }
    for (name, _) in produced.iter().skip(expected.len()) {
        said.push(format!(
            "{name}: produced by the run, and the golden has no such section"
        ));
    }
    if said.is_empty() && canonical != actual {
        said.push("every section matches — a header or a trailing byte moved".to_string());
    }
    said
}

/// The first line of a section that differs, as the column it differs at and a window
/// either side. A plan line is long enough that printing two of them whole says less than
/// pointing at the character, and a long line is the one the log drops.
fn line_difference(expected: &str, actual: &str) -> String {
    for (number, (want, got)) in expected.lines().zip(actual.lines()).enumerate() {
        if want == got {
            continue;
        }
        let at = want
            .char_indices()
            .zip(got.char_indices())
            .find(|((_, a), (_, b))| a != b)
            .map(|((index, _), _)| index)
            .unwrap_or_else(|| want.len().min(got.len()));
        let rest = expected
            .lines()
            .zip(actual.lines())
            .skip(number + 1)
            .filter(|(want, got)| want != got)
            .count();
        return format!(
            "line {}, column {at} — expected `{}` — actual `{}`{}",
            number + 1,
            window(want, at),
            window(got, at),
            if rest > 0 {
                format!(" (+{rest} more lines)")
            } else {
                String::new()
            }
        );
    }
    format!(
        "{} lines in the golden and {} in the run",
        expected.lines().count(),
        actual.lines().count()
    )
}

/// 40 characters either side of `at`, elided at both ends — narrow enough that a run of
/// them stays inside what a CI log carries per line.
fn window(line: &str, at: usize) -> String {
    let start = at.saturating_sub(40);
    let end = (at + 40).min(line.len());
    let start = (start..=at)
        .find(|i| line.is_char_boundary(*i))
        .unwrap_or(at);
    let end = (at..=end)
        .rev()
        .find(|i| line.is_char_boundary(*i))
        .unwrap_or(at);
    format!(
        "{}{}{}",
        if start > 0 { "…" } else { "" },
        &line[start..end],
        if end < line.len() { "…" } else { "" }
    )
}

/// Sections in file order: `(query, body)` at each `== ` header, names as the file writes
/// them. `sections_of` below reads the same shape into a map for the registry check, which
/// wants lookup rather than order.
fn ordered_sections(text: &str) -> Vec<(String, String)> {
    let mut sections: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        match line.strip_prefix("== ") {
            Some(header) => sections.push((header.to_string(), String::new())),
            None => {
                if let Some((_, body)) = sections.last_mut() {
                    body.push_str(line);
                    body.push('\n');
                }
            }
        }
    }
    sections
}

#[test]
fn a_section_that_moved_is_named_with_the_column_that_moved() {
    let golden = "== q1\nGpuUnload\n  GpuSort: lanes=1\n== q2\nGpuUnload\n";
    let run = "== q1\nGpuUnload\n  GpuSort: lanes=4\n== q2\nGpuUnload\n";
    let said = section_differences(golden, run);
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(said[0].starts_with("q1: line 2, column "), "{said:?}");
    assert!(
        said[0].contains("lanes=1") && said[0].contains("lanes=4"),
        "{said:?}"
    );
}

#[test]
fn a_section_missing_from_one_side_says_which_side() {
    let both = "== q1\nGpuUnload\n== q2\nGpuUnload\n";
    let short = "== q1\nGpuUnload\n";
    assert!(
        section_differences(both, short)[0].starts_with("q2: in the golden"),
        "{:?}",
        section_differences(both, short)
    );
    assert!(
        section_differences(short, both)[0].starts_with("q2: produced by the run"),
        "{:?}",
        section_differences(short, both)
    );
}

#[test]
fn sections_out_of_order_are_named_by_position() {
    let golden = "== q1\nGpuUnload\n== q2\nGpuUnload\n";
    let run = "== q2\nGpuUnload\n== q1\nGpuUnload\n";
    let said = section_differences(golden, run);
    assert!(
        said[0].starts_with("section 0: `q1` in the golden"),
        "{said:?}"
    );
}

async fn check(dataset: &str, sf: &str, mode: Mode) {
    let actual = render_bench(dataset, sf, &mode).await;
    assert_or_update(&golden(dataset, sf, &mode), &actual);
}

/// The five modes, as the goldens and the registry columns name them: the three batching
/// forms crossed with the lane counts that make them distinct. Each label names its own
/// form, so no label means one thing at one lane count and another at the other. There is
/// no tp1 sized mode because at one lane a source takes essentially the whole budget, which
/// collapses Sized to Off. Only the last reads a budget.
const MODES: [(&str, usize, BatchSizing); 5] = [
    ("bp-tp1-single", 1, BatchSizing::OneBatchPerLane),
    ("bp-tp1-rowgroup", 1, BatchSizing::OneBatchPerRowGroup),
    ("bp-tp4-single", 4, BatchSizing::OneBatchPerLane),
    ("bp-tp4-rowgroup", 4, BatchSizing::OneBatchPerRowGroup),
    ("bp-tp4-sized", 4, BatchSizing::Budgeted),
];

#[tokio::test]
async fn tpch_bp_tp1_single() {
    check(
        "tpch",
        "1",
        mode("bp-tp1-single", 1, BatchSizing::OneBatchPerLane),
    )
    .await;
}

#[tokio::test]
async fn tpch_bp_tp1_rowgroup() {
    check(
        "tpch",
        "1",
        mode("bp-tp1-rowgroup", 1, BatchSizing::OneBatchPerRowGroup),
    )
    .await;
}

#[tokio::test]
async fn tpch_bp_tp4_single() {
    check(
        "tpch",
        "1",
        mode("bp-tp4-single", 4, BatchSizing::OneBatchPerLane),
    )
    .await;
}

#[tokio::test]
async fn tpch_bp_tp4_rowgroup() {
    check(
        "tpch",
        "1",
        mode("bp-tp4-rowgroup", 4, BatchSizing::OneBatchPerRowGroup),
    )
    .await;
}

#[tokio::test]
async fn tpch_bp_tp4_sized() {
    check("tpch", "1", mode("bp-tp4-sized", 4, BatchSizing::Budgeted)).await;
}

#[tokio::test]
async fn tpcds_bp_tp1_single() {
    check(
        "tpcds",
        "1",
        mode("bp-tp1-single", 1, BatchSizing::OneBatchPerLane),
    )
    .await;
}

#[tokio::test]
async fn tpcds_bp_tp1_rowgroup() {
    check(
        "tpcds",
        "1",
        mode("bp-tp1-rowgroup", 1, BatchSizing::OneBatchPerRowGroup),
    )
    .await;
}

#[tokio::test]
async fn tpcds_bp_tp4_single() {
    check(
        "tpcds",
        "1",
        mode("bp-tp4-single", 4, BatchSizing::OneBatchPerLane),
    )
    .await;
}

#[tokio::test]
async fn tpcds_bp_tp4_rowgroup() {
    check(
        "tpcds",
        "1",
        mode("bp-tp4-rowgroup", 4, BatchSizing::OneBatchPerRowGroup),
    )
    .await;
}

#[tokio::test]
async fn tpcds_bp_tp4_sized() {
    check("tpcds", "1", mode("bp-tp4-sized", 4, BatchSizing::Budgeted)).await;
}

/// The registry's five `bp_` columns against the goldens, in both directions: every cell
/// says what its query's section says, and every section has a cell. Nothing registers
/// these at link time — one golden holds every query — so the golden is what declares
/// them and this is where the two are held to each other.
#[test]
fn the_registry_matches_the_goldens_in_both_directions() {
    let rows = common::registry::load_csv();
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        for (name, target_partitions, sizing) in MODES {
            let mode = mode(name, target_partitions, sizing);
            let sections = sections_of(&golden(dataset, sf, &mode));
            let column = name.replace('-', "_");

            for row in rows
                .iter()
                .filter(|row| row.dataset == dataset && row.sf == sf)
            {
                let body = sections.get(&row.query).unwrap_or_else(|| {
                    panic!("{column}: {} has a registry row and no section", row.query)
                });
                let declared = if body.starts_with("refused by datafusion") {
                    // Not this mode declining: the query never reaches its planner.
                    "na"
                } else if body.starts_with("refused") {
                    "disabled"
                } else {
                    "enabled"
                };
                assert_eq!(
                    row.states[&column], declared,
                    "{column}: {} is {} in the registry and {declared} in the golden",
                    row.query, row.states[&column]
                );
            }

            let known: std::collections::BTreeSet<&str> = rows
                .iter()
                .filter(|row| row.dataset == dataset && row.sf == sf)
                .map(|row| row.query.as_str())
                .collect();
            for query in sections.keys() {
                assert!(
                    known.contains(query.as_str()),
                    "{column}: {query} has a section and no registry row"
                );
            }
        }
    }
}

/// Every mode has a golden and every golden has a mode. The per-mode test functions spell
/// their knobs out one by one, so a mode added to `MODES` and to no test function would
/// otherwise be checked by nobody — and its file would silently not exist.
#[test]
fn every_mode_has_a_golden_and_every_golden_has_a_mode() {
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        let mut expected: Vec<String> = MODES
            .iter()
            .map(|(name, ..)| format!("{name}.plans.txt"))
            .collect();
        expected.sort();
        let mut found: Vec<String> = std::fs::read_dir(golden_dir_for(dataset, sf))
            .expect("the golden directory")
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().to_str()?.to_string();
                (name.starts_with("bp-") && name.ends_with(".plans.txt")).then_some(name)
            })
            .collect();
        found.sort();
        assert_eq!(found, expected, "{dataset}: goldens and modes disagree");
    }
}

/// No refusal carries a path from the machine that wrote it. This class did not come from
/// our renderer — it arrived inside a third-party error we embed verbatim, where the
/// name@ordinal discipline the rest of the file keeps could not see it — so the check is on
/// the rendered text rather than on any one producer of it.
#[test]
fn no_refusal_in_a_golden_carries_a_host_path() {
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        for (name, ..) in MODES {
            let path = golden_dir_for(dataset, sf).join(format!("{name}.plans.txt"));
            let text = std::fs::read_to_string(&path).expect("a golden");
            // Every line, not only those opening with `refused`: the error text carries
            // its own newlines, so the path sits on a continuation line.
            for line in text.lines() {
                if let Some(at) = absolute_path_in(line) {
                    panic!(
                        "{dataset}/{name}: a refusal carries a host path, so this golden \
                         matches only a checkout at that path:\n  {}",
                        &line[at..(at + 90).min(line.len())]
                    );
                }
            }
        }
    }
}

/// Where a line first names a file by machine rather than by repository: a `/testdata/`
/// with something before it, or a token opening with a slash. Division has spaces or a
/// closing paren before it, so it is not one.
fn absolute_path_in(line: &str) -> Option<usize> {
    if let Some(at) = line.find("/testdata/") {
        return Some(
            line[..at]
                .rfind(['[', ' ', '"'])
                .map_or(0, |start| start + 1),
        );
    }
    let bytes = line.as_bytes();
    (1..bytes.len().saturating_sub(1)).find(|&at| {
        bytes[at] == b'/'
            && matches!(bytes[at - 1], b'[' | b' ' | b'"' | b'(' | b',')
            && bytes[at + 1].is_ascii_alphabetic()
    })
}

/// A refusal names a blocker, and that blocker exists. The reason a query does not plan is
/// the content of these files, and a ticket number that has been renumbered or never
/// existed reads as an explanation while pointing at nothing.
///
/// Both lists: a ticket keeps its number when it closes and moves to the archive, so
/// reading only the open one would turn this red for a refusal whose blocker was fixed
/// somewhere else — which is a stale refusal to delete, not a broken reference.
#[test]
fn every_refusal_names_a_ticket_that_exists() {
    let wiki = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../llm-wiki");
    let tickets = ["tickets.md", "archive/archived-tickets.md"]
        .iter()
        .map(|name| std::fs::read_to_string(wiki.join(name)).expect("the ticket list"))
        .collect::<String>();
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        for (name, ..) in MODES {
            let text = std::fs::read_to_string(
                golden_dir_for(dataset, sf).join(format!("{name}.plans.txt")),
            )
            .expect("a golden");
            for line in text.lines().filter(|line| line.starts_with("refused")) {
                let cited: Vec<&str> = line
                    .match_indices('#')
                    .filter_map(|(at, _)| {
                        line[at + 1..].split(|c: char| !c.is_ascii_digit()).next()
                    })
                    .filter(|number| !number.is_empty())
                    .collect();
                for number in cited {
                    assert!(
                        tickets.contains(&format!("### #{number} ")),
                        "{dataset}/{name}: a refusal names #{number}, which tickets.md does not have:\n{line}"
                    );
                }
            }
        }
    }
}

/// Query name to section body. Names carry underscores here, as the registry writes them.
fn sections_of(path: &Path) -> std::collections::BTreeMap<String, String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut sections = std::collections::BTreeMap::new();
    let mut name: Option<String> = None;
    let mut body = String::new();
    for line in text.lines() {
        if let Some(header) = line.strip_prefix("== ") {
            if let Some(previous) = name.take() {
                sections.insert(previous, std::mem::take(&mut body));
            }
            name = Some(header.replace('-', "_"));
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    if let Some(previous) = name {
        sections.insert(previous, body);
    }
    sections
}
