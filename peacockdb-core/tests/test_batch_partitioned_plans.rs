//! Plan goldens for the batch-partitioned mode: one file per (bench, mode), holding every
//! query the bench has.
//!
//! A section per query — the tree, then `--- recipes ---` and what each node asks of the
//! device, then `--- memory ---` and the estimator's figures. The legacy `.plan.txt`
//! goldens carry the first and the last of those. Refusals are content: a query
//! this mode declines renders its reason where its tree would be, so the file says what
//! the mode does and does not run.

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use peacockdb_core::batch_partitioned::node::GpuNode;
use peacockdb_core::batch_partitioned::plan::{BatchSizing, PlanKnobs, plan_batch_partitioned};
use peacockdb_core::batch_partitioned::plan_text::{
    Payloads, render_plan, render_plan_memory, render_plan_recipes,
};
use peacockdb_core::batch_partitioned::recipe::{attach_recipes, check_seq_kinds, depth, node_at};
use peacockdb_core::batch_partitioned::{ExecutorCategory, category_of};
use peacockdb_core::config::MemoryLimit;
use peacockdb_core::generated::gpu_plan_generated::peacock::plan as fb;

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
            recipes_of(tree.as_ref()),
            render_plan_memory(tree.as_ref(), &model)
        ),
        Err(e) => format!("refused: {}\n", relative_to_testdata(&e.to_string())),
    }
}

/// The recipes, or the one line saying there are none.
///
/// `not runnable`, never `refused:` — the plan plans, validates and runs on the CPU, and
/// what failed is the crossing to the device. Naming its ticket is not decoration: the
/// meta test below asserts that every line of this shape names a ticket that exists.
fn recipes_of(tree: &dyn peacockdb_core::batch_partitioned::GpuNode) -> String {
    match attach_recipes(tree) {
        Ok(plan) => render_plan_recipes(tree, &plan, Payloads::Omitted),
        Err(e) => format!("not runnable: {}\n", relative_to_testdata(&e.to_string())),
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

// --- the payload golden ------------------------------------------------------
//
// One file, one mode, twelve queries: what each call actually hands the executor, which
// the ten mode goldens deliberately do not carry. They answer different questions — those
// say which kernel a node addresses and how often, this says what is in the buffer — and
// one renderer serves both, so a plan cannot move in one and stand still in the other.

/// The mode the payloads are read at. tp4 because tp1 emits no repartition at all, and
/// rowgroup because it is the only form with many batches per lane, which is what makes an
/// accumulating sort and a compaction real rather than degenerate. Not the sized mode:
/// that one moves whenever the estimator does, and this file should move when a payload
/// does.
const PAYLOAD_MODE: (&str, usize, BatchSizing) = ("bp-tp4-rowgroup", 4, BatchSizing::OneBatchPerRowGroup);

/// The queries, and the rule for adding one.
///
/// Chosen by cover rather than by taste: every fb node kind the mapping emits (join types
/// spelled out, both project roles, the repartition, the two symbols that address no seq,
/// the rows that emit nothing), every call shape longer than one call, and every
/// expression kind a payload can carry — LIKE, CASE, casts, scalar functions, decimal
/// scales, grouping sets, null substitutions, Welford state, an avg's finalize, fetch and
/// skip.
///
/// Which half of that a test enforces, so a reader knows what rests on a person: the fb
/// kinds and the call shapes are asserted against the ten mode goldens by
/// `the_payload_golden_covers_every_kind_and_call_shape_the_modes_produce` — a mapping arm
/// or a call pattern that appears there and not here goes red. The expression features are
/// prose, checked by reading.
///
/// So a query earns a place here by covering something no other query does. Adding one
/// that covers nothing new adds lines no reader can check against anything.
const PAYLOAD_QUERIES: [(&str, &str); 16] = [
    // Nested-loop join lives nowhere else in the corpus; the stddev query is the Welford
    // init, both merges and the finalize project; q13 is the outer join's finish pass.
    ("tpch", "q13"),
    ("tpch", "shuffle-stddev"),
    ("tpch", "nested-loop-join"),
    // q22 is the build-side semi family's finish, whose join type is the node's OWN
    // LeftAnti: a key project per batch, then coalesce and that join at done. The other
    // LeftAnti here is a Left outer's DERIVED finish — different keys, no projection, and
    // a pad project after it — so neither covers the other.
    ("tpch", "q22"),
    // q21 is the single-batch probe, where a filtered semi or anti join is one legacy call
    // that hands the build side over rather than copying it — the same join types as q22
    // and q13, and a different shape.
    ("tpch", "q21"),
    // q15 is the aggregate that finalizes on its own: one batch in one lane is already the
    // whole of every group, so the init node carries the finalize and its recipe is the
    // partial plus a finalize project. Every other finalize here rides a merge.
    ("tpch", "q15"),
    ("tpcds", "q14"),
    ("tpcds", "q97"),
    ("tpcds", "q61"),
    ("tpcds", "q87"),
    ("tpcds", "q45"),
    ("tpcds", "q41"),
    ("tpcds", "q5"),
    ("tpcds", "q91"),
    ("tpcds", "q39"),
    // q40 is a plain Right outer: probe-local, one call, no finish. The Right in q97 is a
    // Full outer's per-batch call, which is the same kind doing a different thing.
    ("tpcds", "q40"),
];

/// A digest of the bytes beside the text, because the two can disagree: a field the
/// renderer does not print, an ordering that moves. `plan_bytes.sha256` pins the legacy
/// wire form the same way and for the same reason.
fn digest_of(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[tokio::test]
async fn the_payload_golden_carries_what_each_call_hands_the_executor() {
    let mode = mode(PAYLOAD_MODE.0, PAYLOAD_MODE.1, PAYLOAD_MODE.2);
    let mut text = String::new();
    for dataset in ["tpch", "tpcds"] {
        let wanted: Vec<&str> = PAYLOAD_QUERIES
            .iter()
            .filter(|(bench, _)| *bench == dataset)
            .map(|(_, query)| *query)
            .collect();
        if wanted.is_empty() {
            continue;
        }
        // Through the canonical root, so the paths the buffer embeds — and the digest
        // over them — are the same on every machine. `test_plan_bytes` holds the legacy
        // wire form still the same way, and for the same reason.
        let ctx = peacockdb_core::register_tables_for(
            peacockdb_core::build_session_state(mode.knobs.target_partitions),
            &common::canonical_data_dir(dataset, "1"),
        )
        .await
        .expect("register the tables");
        for (name, path) in queries(dataset) {
            if !wanted.contains(&name.as_str()) {
                continue;
            }
            let sql = std::fs::read_to_string(&path).expect("the query text");
            let plan = ctx
                .sql(&sql)
                .await
                .expect("the query parses")
                .create_physical_plan()
                .await
                .expect("the query plans");
            let (tree, _) = plan_batch_partitioned(&plan, mode.knobs).expect("this mode runs it");
            let recipes = attach_recipes(tree.as_ref()).expect("a plan's recipes are structural");
            text.push_str(&format!("== {dataset} {name}\n"));
            text.push_str(&format!("sha256={}\n", digest_of(recipes.bytes())));
            text.push_str(&render_plan_recipes(
                tree.as_ref(),
                &recipes,
                Payloads::Shown,
            ));
        }
    }
    let path = common::testdata_root()
        .join("goldens")
        .join("bp-recipe-payloads.txt");
    // Three states, as `plan_bytes.sha256` has for the legacy wire form and for the same
    // reason: the digests here are this branch's only byte-level pin on the newer
    // serializer, and the documented way to refresh goldens is a bulk --update-canonical on
    // verda. Without the second variable that run would rewrite the evidence and the diff
    // would come home among hundreds. With it, a moved payload goes red DURING the regen.
    let update = std::env::var("UPDATE_CANONICAL").is_ok();
    let rewrite = std::env::var("PEACOCK_REWRITE_RECIPE_BYTES").is_ok();
    if update && !rewrite {
        let canonical = std::fs::read_to_string(&path).expect("the payload golden");
        eprintln!(
            "NOT regenerating {} — verifying instead. The `sha256=` lines are a fixed \
             expectation from before a change, and the C++ reads these bytes. If the move is \
             intended, set PEACOCK_REWRITE_RECIPE_BYTES=1 alongside UPDATE_CANONICAL.",
            path.display()
        );
        assert_eq!(
            digests_of(&canonical),
            digests_of(&text),
            "{}: the recipe bytes moved. Regenerating cannot make this green — see the \
             message above.",
            path.display()
        );
        // The text may still be regenerated: it describes the same bytes.
        assert_or_update(&path, &text);
        return;
    }
    assert_or_update(&path, &text);
}

/// The `sha256=` line per query, which is the half a bulk regen may not rewrite.
fn digests_of(text: &str) -> std::collections::BTreeMap<String, String> {
    let mut digests = std::collections::BTreeMap::new();
    let mut query = String::new();
    for line in text.lines() {
        if let Some(header) = line.strip_prefix("== ") {
            query = header.to_string();
        } else if let Some(digest) = line.strip_prefix("sha256=") {
            digests.insert(query.clone(), digest.to_string());
        }
    }
    digests
}

/// Every seq every recipe publishes addresses a node of the kind it claims — over every
/// query both benches have, not just the twelve the payload golden carries.
///
/// It is the assertion behind the three structural rules in `recipe/writer.rs`: a stub for
/// an unfilled slot, a node hanging off its own previous call, and a union over what a
/// node did not consume. Each keeps the order the walk creates nodes in equal to the
/// post-order the C++ indexes by, and a break in any of them shows up here as a seq that
/// resolves to nothing or to the wrong kind. The mode goldens cannot catch it: they render
/// with `Payloads::Omitted` and never open the buffer at all.
#[tokio::test]
async fn every_published_seq_addresses_the_kind_its_recipe_claims() {
    let mode = mode(PAYLOAD_MODE.0, PAYLOAD_MODE.1, PAYLOAD_MODE.2);
    let mut checked = 0;
    let mut faults: Vec<String> = Vec::new();
    let mut uncrossable: Vec<String> = Vec::new();
    let mut deepest = (0usize, String::new());
    for dataset in ["tpch", "tpcds"] {
        let ctx = peacockdb_core::register_tables_for(
            peacockdb_core::build_session_state(mode.knobs.target_partitions),
            &data_dir_for(dataset, "1"),
        )
        .await
        .expect("register the tables");
        for (name, path) in queries(dataset) {
            let sql = std::fs::read_to_string(&path).expect("the query text");
            let Ok(frame) = ctx.sql(&sql).await else { continue };
            let Ok(plan) = frame.create_physical_plan().await else {
                continue;
            };
            // A query this mode refuses has no recipes to check; a query it plans has to
            // publish seqs that resolve.
            let Ok((tree, _)) = plan_batch_partitioned(&plan, mode.knobs) else {
                continue;
            };
            match attach_recipes(tree.as_ref()) {
                Ok(recipes) => {
                    checked += 1;
                    if let Err(e) = check_seq_kinds(&recipes) {
                        faults.push(format!("{dataset} {name}: {e}"));
                    }
                    let reached = depth(&recipes).expect("the plan we just wrote");
                    deepest = deepest.max((reached, format!("{dataset} {name}")));
                }
                // A plan the wire cannot carry has no recipe plan to check. Named rather
                // than counted: the day a second query joins mixed-join here, that is a
                // fact about the mode worth a red test rather than a smaller number.
                Err(_) => uncrossable.push(format!("{dataset} {name}")),
            }
        }
    }
    assert!(checked > 100, "only {checked} plans reached the check");
    // #169: a recipe plan is a chain, so its depth is its length, and the C++ verifier
    // refuses one past 1024 at begin_plan — the whole query, before any call. A tripwire
    // rather than a note: the number was found by hand once and would drift silently.
    let (reached, where_) = &deepest;
    assert!(
        *reached < 900,
        "{where_} builds a recipe plan {reached} deep against the verifier's 1024 (#169) — \
         the shape has to split before it reaches the limit, not the limit be raised"
    );
    assert_eq!(
        uncrossable,
        ["tpch mixed-join"],
        "the queries this mode plans but cannot cross to the device are #168's, and only its"
    );
    assert!(
        faults.is_empty(),
        "{} of {checked} plans publish a seq that does not hold what it claims:\n{}",
        faults.len(),
        faults.join("\n")
    );
}

/// The queries this mode PLANS but cannot cross to the device, and the ticket for each.
///
/// A `not runnable` line means the plan validates and runs on the CPU while one node's
/// payload has no shape on the wire — so the registry stays out of it: its cell answers
/// "does this query plan", which for these is yes, and a fourth state would make one cell
/// answer two questions at once.
///
/// Asserted both ways below. A new unwritable expression goes red until someone adds it
/// here with a ticket, and an entry that stops being true goes red until it is removed —
/// which is the direction that matters when #168 closes, since a stale entry is how a
/// declared set rots into a list nobody trusts.
const NOT_RUNNABLE: &[(&str, &str, &str)] = &[("tpch", "mixed-join", "168")];

#[test]
fn every_query_that_cannot_cross_the_wire_is_declared_and_every_declaration_is_true() {
    let mut found: Vec<(String, String)> = Vec::new();
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        for (name, ..) in MODES {
            let path = golden_dir_for(dataset, sf).join(format!("{name}.plans.txt"));
            for (query, body) in ordered_sections(&std::fs::read_to_string(&path).expect("a golden"))
            {
                let Some(line) = body.lines().find(|line| line.starts_with("not runnable")) else {
                    continue;
                };
                let declared = NOT_RUNNABLE
                    .iter()
                    .find(|(bench, declared, _)| *bench == dataset && *declared == query);
                match declared {
                    Some((_, _, ticket)) => assert!(
                        line.contains(&format!("(#{ticket})")),
                        "{dataset}/{name} {query}: declared under #{ticket} and the line cites \
                         something else:\n{line}"
                    ),
                    None => panic!(
                        "{dataset}/{name} {query}: cannot cross the wire and is not in \
                         NOT_RUNNABLE — add it with the ticket that explains it:\n{line}"
                    ),
                }
                found.push((dataset.to_string(), query.clone()));
            }
        }
    }
    for (dataset, query, ticket) in NOT_RUNNABLE {
        let carried = found
            .iter()
            .filter(|(bench, name)| bench == dataset && name == query)
            .count();
        assert_eq!(
            carried,
            MODES.len(),
            "{dataset}/{query} is declared under #{ticket} but carries the line in {carried} of \
             the {} modes — a plan that crosses in one mode and not another is a finding, and a \
             declaration that has become false is stale",
            MODES.len()
        );
    }
}

/// What a recipes line says once its seqs are stripped: the node, its call pattern, the
/// kinds it addresses and the handles they take — the shape, not the numbering.
fn call_shapes(text: &str) -> std::collections::BTreeSet<String> {
    let mut shapes = std::collections::BTreeSet::new();
    let mut in_recipes = false;
    for line in text.lines() {
        match line {
            // Both files hold the same section, headed differently: the mode goldens put
            // it between two markers, the payload golden gives a whole section to it after
            // the digest line.
            "--- recipes ---" => in_recipes = true,
            _ if line.starts_with("sha256=") => in_recipes = true,
            "--- memory ---" => in_recipes = false,
            _ if line.starts_with("== ") => in_recipes = false,
            _ if in_recipes => {
                let line = line.trim();
                // Payload lines are indented under their call and carry no `: ` shape;
                // a `none` row and a `not runnable` line are shapes of their own.
                if !line.contains(": ") && !line.ends_with(": none") {
                    continue;
                }
                // The lane count is a per-plan number rather than a shape: the same call
                // at four lanes and at nine is one thing to cover.
                let line = match (line.find("calling_lanes="), line.find(", ")) {
                    (Some(at), Some(comma)) if comma > at => {
                        format!("{}{}", &line[..at], &line[comma + 2..])
                    }
                    _ => line.to_string(),
                };
                let stripped = line
                    .split('#')
                    .enumerate()
                    .map(|(at, part)| {
                        if at == 0 {
                            part.to_string()
                        } else {
                            // Drop the digits of the seq and keep what follows it.
                            format!("#{}", part.trim_start_matches(|c: char| c.is_ascii_digit()))
                        }
                    })
                    .collect::<String>();
                shapes.insert(stripped);
            }
            _ => {}
        }
    }
    shapes
}

/// Every fb kind and every call shape the ten mode goldens hold appears in the payload
/// golden too.
///
/// This is the checked half of [`PAYLOAD_QUERIES`]'s claim. A mapping arm added later
/// without a payload query for it goes red here rather than waiting for a careful reader —
/// and the shapes matter as much as the kinds, since the build-side semi family's finish
/// and a Left outer's derived finish both address a `CudfHashJoin{LeftAnti}` while doing
/// different things with different arguments.
#[test]
fn the_payload_golden_covers_every_kind_and_call_shape_the_modes_produce() {
    let payloads = std::fs::read_to_string(
        common::testdata_root()
            .join("goldens")
            .join("bp-recipe-payloads.txt"),
    )
    .expect("the payload golden");
    let covered = call_shapes(&payloads);

    let mut wanted = std::collections::BTreeSet::new();
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        for (name, ..) in MODES {
            let text = std::fs::read_to_string(
                golden_dir_for(dataset, sf).join(format!("{name}.plans.txt")),
            )
            .expect("a golden");
            wanted.extend(call_shapes(&text));
        }
    }

    let kind = |shapes: &std::collections::BTreeSet<String>| -> std::collections::BTreeSet<String> {
        shapes
            .iter()
            .flat_map(|shape| {
                shape
                    .match_indices("Cudf")
                    .map(|(at, _)| {
                        let rest = &shape[at..];
                        let end = rest
                            .find(|c: char| c == ',' || c == ')')
                            .unwrap_or(rest.len());
                        rest[..end].to_string()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let missing_kinds: Vec<String> = kind(&wanted).difference(&kind(&covered)).cloned().collect();
    assert!(
        missing_kinds.is_empty(),
        "the modes emit fb kinds the payload golden never shows, so nothing pins what they \
         carry: {missing_kinds:?}"
    );

    let missing_shapes: Vec<&String> = wanted
        .difference(&covered)
        // A shape the modes hold and the payload file cannot: `not runnable` is a
        // property of mixed-join, which by definition has no payload to show.
        .filter(|shape| !shape.starts_with("not runnable"))
        .collect();
    assert!(
        missing_shapes.is_empty(),
        "{} call shapes appear in the mode goldens and in no payload query — add a query \
         that carries one, per PAYLOAD_QUERIES' rule:\n{}",
        missing_shapes.len(),
        missing_shapes
            .iter()
            .map(|shape| format!("  {shape}"))
            .collect::<Vec<_>>()
            .join("\n")
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
            // Both shapes of line that decline something: `refused` is the planner's,
            // and `not runnable` is the crossing's — the plan runs on the CPU and only the
            // buffer cannot be built. A ticket is cited in PARENTHESES, which is what
            // separates it from the `#5` in "at #5", a seq, and from a DataFusion refusal
            // that cites nothing of ours at all.
            let declines =
                |line: &&str| line.starts_with("refused") || line.starts_with("not runnable");
            for line in text.lines().filter(declines) {
                let cited: Vec<&str> = line
                    .match_indices("(#")
                    .filter_map(|(at, _)| {
                        line[at + 2..].split(|c: char| !c.is_ascii_digit()).next()
                    })
                    .filter(|number| !number.is_empty())
                    .collect();
                assert!(
                    !cited.is_empty() || !line.starts_with("not runnable"),
                    "{dataset}/{name}: a plan the mode cannot cross names no ticket, so a \
                     reader has nowhere to go:\n{line}"
                );
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

/// Which fields each writer sets, per fb table, unioned over the whole corpus.
///
/// `expr_writer`'s tests compare the two writers byte for byte, which is right for
/// expressions and wrong for node payloads: the two legitimately differ there. What a
/// difference must not be is unexamined — a field the legacy writer sets and this one
/// leaves at its default is either a decision with a reason beside it or the grouping-set
/// omission again, and from outside the two look identical.
///
/// Names come from the fbs rather than from a list here, so a field added to a table is
/// covered the moment it exists.
fn fbs_offset_fields() -> std::collections::BTreeMap<String, Vec<String>> {
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../flatbuffers/gpu_plan.fbs"),
    )
    .expect("the schema");
    // A scalar equal to its default is written by nobody, so presence says nothing about
    // it: `interleave: false` and an unset `interleave` are the same bytes. Enums are
    // scalars too, which is why the enum names are collected rather than assumed.
    let mut scalars: BTreeSet<String> = [
        "bool", "byte", "ubyte", "short", "ushort", "int", "uint", "long", "ulong", "float",
        "double", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
        "float32", "float64",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let mut unions: BTreeSet<String> = BTreeSet::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("enum ") {
            scalars.insert(rest.split([':', ' ']).next().unwrap().to_string());
        } else if let Some(rest) = line.strip_prefix("union ") {
            unions.insert(rest.split([' ', '{']).next().unwrap().to_string());
        }
    }

    let mut tables = std::collections::BTreeMap::new();
    let mut open: Option<(String, Vec<String>)> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("table ") {
            open = Some((rest.trim_end_matches(" {").trim().to_string(), Vec::new()));
        } else if line == "}" {
            if let Some((name, fields)) = open.take() {
                tables.insert(name, fields);
            }
        } else if let Some((_, fields)) = open.as_mut() {
            if line.starts_with("//") || line.is_empty() {
                continue;
            }
            let Some((name, declared)) = line.split_once(':') else {
                continue;
            };
            // Every field takes a vtable slot, so the position is the declaration order
            // whatever the type; only the ones worth comparing carry a name.
            let declared = declared
                .split(|c: char| c == '=' || c == ';' || c == '(')
                .next()
                .unwrap_or("")
                .trim();
            // A union takes two slots — the type byte, then the offset — so a table
            // holding one numbers every field after it two higher than it is declared.
            if unions.contains(declared) {
                fields.push(String::new());
            }
            fields.push(if scalars.contains(declared) {
                String::new()
            } else {
                name.trim().to_string()
            });
        }
    }
    tables
}

/// The field indices a table actually set, read off its vtable so a field nobody wrote an
/// accessor call for is still counted.
fn fields_set(table: &flatbuffers::Table<'_>) -> BTreeSet<usize> {
    let vtable = table.vtable();
    (0..vtable.num_fields())
        .filter(|index| {
            vtable.get(flatbuffers::field_index_to_field_offset(
                *index as flatbuffers::VOffsetT,
            )) != 0
        })
        .collect()
}

/// Per node kind, the union of the field indices set across every node of that kind.
///
/// `PlanNode` and each node's top-level payload, and nothing below them: a field of a
/// nested table is outside this comparison, as `AggregateFuncNode`'s decimal pair is.
/// Those two are answered at `recipe::aggregate_writer::named_func` instead.
fn written_fields(bytes: &[u8]) -> std::collections::BTreeMap<String, BTreeSet<usize>> {
    let options = flatbuffers::VerifierOptions {
        max_depth: 1024,
        ..Default::default()
    };
    let plan = flatbuffers::root_with_opts::<fb::GpuPlan>(&options, bytes).expect("it verifies");
    let mut per_kind: std::collections::BTreeMap<String, BTreeSet<usize>> =
        std::collections::BTreeMap::new();
    let mut seq = 0;
    while let Some(node) = node_at(&plan, seq) {
        per_kind
            .entry("PlanNode".to_string())
            .or_default()
            .extend(fields_set(&node._tab));
        if let Some(payload) = node.node() {
            per_kind
                .entry(format!("{:?}", node.node_type()))
                .or_default()
                .extend(fields_set(&payload));
        }
        seq += 1;
    }
    per_kind
}
/// The differences between the two writers, each with the reason it is one. Everything not
/// listed has to match: a field the legacy writer sets and this one does not is how the
/// grouping-set omission shipped, and the payload golden pinned it faithfully.
const WRITER_DIFFERENCES: [(&str, &str, &str); 5] = [
    (
        "CudfScan",
        "projection",
        "the file schema written here is already the projected fields, so an empty \
         projection reads as every column of it",
    ),
    (
        "CudfScan",
        "row_groups",
        "the node's own list is overridden per call by execute_scan_rowgroups, so a list \
         on the node would be a second answer to the same question",
    ),
    (
        "CudfScan",
        "batches",
        "the RG map is this mode's own and is executed one batch per call, not read off \
         the node",
    ),
    (
        "CudfUnion",
        "output_schema",
        "the only unions here are structural, gathering a node's unconsumed branches so \
         nothing is left unreachable; no recipe publishes a seq for one, so execute_union \
         never runs on it and never reads this",
    ),
    (
        "PlanNode",
        "output_schema",
        "nothing on the C++ side reads it — the executors take their types from each \
         node's own payload. The one arm that would have is the collapse of no input \
         handles, which needed a schema to answer with and is a refusal instead (#173), \
         so this stays absent rather than becoming a field written for one caller",
    ),
];

/// The two writers run over different query sets — this one only where planning and
/// `attach_recipes` both succeed, the legacy one over everything — so a field the legacy
/// writer sets only on a query this mode refuses arrives here as a difference, and the
/// entry explaining it would be describing the refusal rather than the writer.
#[tokio::test]
async fn every_field_the_legacy_writer_sets_is_set_here_or_declared_a_difference() {
    let names = fbs_offset_fields();
    let mut legacy: std::collections::BTreeMap<String, BTreeSet<usize>> = Default::default();
    let mut recipe: std::collections::BTreeMap<String, BTreeSet<usize>> = Default::default();
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        let mode = mode(PAYLOAD_MODE.0, PAYLOAD_MODE.1, PAYLOAD_MODE.2);
        let ctx = peacockdb_core::register_tables_for(
            peacockdb_core::build_session_state(mode.knobs.target_partitions),
            &data_dir_for(dataset, sf),
        )
        .await
        .expect("register the tables");
        for (query, path) in queries(dataset) {
            let sql = std::fs::read_to_string(&path).expect("the query text");
            let Ok(frame) = ctx.sql(&sql).await else {
                continue;
            };
            let Ok(plan) = frame.create_physical_plan().await else {
                continue;
            };
            if let Ok((tree, _)) = plan_batch_partitioned(&plan, mode.knobs)
                && let Ok(recipes) = attach_recipes(tree.as_ref())
            {
                merge(&mut recipe, written_fields(recipes.bytes()));
            }
            let legacy_plan = common::plan_for(dataset, sf, &query, "tp8-standard").await;
            if let Ok(bytes) = peacockdb_core::plan_serializer::serialize_plan_mode(
                &legacy_plan,
                common::plan_partition_mode("tp8-standard"),
            ) {
                merge(&mut legacy, written_fields(&bytes));
            }
        }
    }

    let mut unexplained: Vec<String> = Vec::new();
    for (kind, set) in &legacy {
        let Some(here) = recipe.get(kind) else {
            continue;
        };
        for index in set.difference(here) {
            let field = names
                .get(kind)
                .and_then(|fields| fields.get(*index))
                .map(String::as_str)
                .unwrap_or("?");
            if field.is_empty() {
                continue;
            }
            if !WRITER_DIFFERENCES
                .iter()
                .any(|(table, name, _)| *table == kind && *name == field)
            {
                unexplained.push(format!("{kind}.{field}"));
            }
        }
    }
    assert!(
        unexplained.is_empty(),
        "the legacy writer sets these and the recipe writer leaves them at their default, \
         with no entry in WRITER_DIFFERENCES saying why: {unexplained:?}"
    );

    let stale: Vec<String> = WRITER_DIFFERENCES
        .iter()
        .filter(|(table, name, _)| {
            let index = names
                .get(*table)
                .and_then(|fields| fields.iter().position(|field| field == name));
            match (index, legacy.get(*table), recipe.get(*table)) {
                (Some(index), Some(set), Some(here)) => {
                    !set.contains(&index) || here.contains(&index)
                }
                _ => true,
            }
        })
        .map(|(table, name, _)| format!("{table}.{name}"))
        .collect();
    assert!(
        stale.is_empty(),
        "WRITER_DIFFERENCES explains differences that are not there — either the field \
         moved or this writer now sets it: {stale:?}"
    );
}

fn merge(
    into: &mut std::collections::BTreeMap<String, BTreeSet<usize>>,
    from: std::collections::BTreeMap<String, BTreeSet<usize>>,
) {
    for (kind, set) in from {
        into.entry(kind).or_default().extend(set);
    }
}

/// The driver's numbering and the recipe list's are the same numbering.
///
/// `PlanIndex` numbers pre-order for the schedule and records each node's children-first
/// position beside it; `attach_recipes` numbers children-first and stores a recipe at that
/// position. Two walks, two files, one at plan time and one at run time — and a backend
/// looks a recipe up by the number the index handed it. Nothing else compares them, which
/// is [#134](../../llm-wiki/tickets.md#t134)'s shape one boundary in.
///
/// Two claims, and the first is what the second rests on: the position the index recorded
/// is the node's own, against a children-first walk written here rather than the one under
/// test; and the recipe standing at that position is present exactly where the node makes
/// calls. Checked over the corpus rather than a fixture: every numbering agrees on a tree
/// with one child per node, and they part company at the first node with two.
#[tokio::test]
async fn the_index_and_the_recipes_number_the_same_nodes_the_same_way() {
    // Four lanes at row-group granularity: lanes and many batches at once, which is the
    // mode whose trees branch most.
    let mode = mode("bp-tp4-rowgroup", 4, BatchSizing::OneBatchPerRowGroup);
    let mut checked = 0;
    for dataset in ["tpch", "tpcds"] {
        let ctx = peacockdb_core::register_tables_for(
            peacockdb_core::build_session_state(mode.knobs.target_partitions),
            &data_dir_for(dataset, "1"),
        )
        .await
        .expect("register the tables");
        for (_, path) in queries(dataset) {
            let sql = std::fs::read_to_string(&path).expect("the query text");
            let Ok(frame) = ctx.sql(&sql).await else {
                continue;
            };
            let Ok(plan) = frame.create_physical_plan().await else {
                continue;
            };
            let Ok((tree, _)) = plan_batch_partitioned(&plan, mode.knobs) else {
                continue;
            };
            let Ok(recipes) = attach_recipes(tree.as_ref()) else {
                continue;
            };
            let positions =
                peacockdb_core::batch_partitioned::driver::post_order_of_every_node(tree.as_ref())
                    .expect("the plan indexes");
            let mut nodes = Vec::new();
            collect(tree.as_ref(), &mut nodes);
            let mut children_first = Vec::new();
            addresses_children_first(tree.as_ref(), &mut children_first);
            assert_eq!(positions.len(), nodes.len());
            assert_eq!(
                recipes.nodes(),
                nodes.len(),
                "the recipe list holds one entry per node of the tree"
            );
            for (node, position) in nodes.iter().zip(&positions) {
                assert_eq!(
                    children_first.get(*position).copied(),
                    Some(address_of(*node)),
                    "{} was numbered {position}, which is where a children-first walk puts \
                     another node",
                    node.name()
                );
                let makes_calls = category_of(*node) != ExecutorCategory::BatchForwarder;
                assert_eq!(
                    recipes.get(*position).is_some(),
                    makes_calls,
                    "{} sits at {position} in the index's numbering, and the recipe there \
                     is {}",
                    node.name(),
                    match recipes.get(*position) {
                        Some(_) => "some other node's",
                        None => "absent where this node makes calls",
                    }
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} plans reached the check");
}

/// The tree in the index's own order, so the pairing above is against an independent walk
/// rather than against the walk under test.
fn collect<'a>(node: &'a dyn GpuNode, into: &mut Vec<&'a dyn GpuNode>) {
    into.push(node);
    for child in node.children() {
        collect(child, into);
    }
}

/// The same tree children-first, which is the numbering the recipes are addressed by.
fn addresses_children_first(node: &dyn GpuNode, into: &mut Vec<usize>) {
    for child in node.children() {
        addresses_children_first(child, into);
    }
    into.push(address_of(node));
}

/// A node's identity as a number. Only the data half of the trait-object pointer is taken:
/// the vtable half is not stable across casts, and identity is what is being compared.
fn address_of(node: &dyn GpuNode) -> usize {
    node as *const dyn GpuNode as *const () as usize
}
