//! What proves the corpus infra, since every way it fails is quiet.
//!
//! A dropped section, a filtered regeneration that deletes coverage, a marker a reader
//! cannot tell from an absence — none of these turn a run red on their own. So each has a
//! case here, built from strings and temporary files rather than by editing a committed
//! golden and undoing it, and every one is deterministic: this tier's whole claim is that a
//! golden means something, and a case that passes on the third run is a golden that means
//! nothing.
//!
//! The second half reads the committed corpus goldens back and checks them against their
//! own redundancy. The cost tree has no external oracle — the golden is written by the run
//! that will later check it — but a file that contradicts itself is a file the renderer got
//! wrong, and the renderer is most of what could be wrong.
#[macro_use]
mod common;

use std::path::{Path, PathBuf};

use common::bp_mode::{BP_MODES, mode_named};
use common::corpus_golden::{
    Regeneration, SKIPPED, cost_golden, cpu_golden, merge_section, merged_text, result_golden,
};
use common::golden_text::{ordered_sections, parse_node_line, parse_run_section};

// --- the write path ------------------------------------------------------------

/// A skeleton as `declared_sections` builds one: `None` where a run writes the section,
/// the marker where the declaration says it is not enabled.
fn skeleton(entries: &[(&str, bool)]) -> Vec<(String, Option<String>)> {
    entries
        .iter()
        .map(|(query, enabled)| {
            (
                query.to_string(),
                (!enabled).then(|| format!("{SKIPPED}not enabled at this mode\n")),
            )
        })
        .collect()
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("peacock-corpus-{}-{name}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

/// Two writers, forced rather than raced: the second merges with the first's section
/// already on disk, which is what the lock guarantees can be the only ordering.
#[test]
fn two_writers_at_the_merge_keep_both_sections() {
    let path = scratch("both");
    let declared = skeleton(&[("q1", true), ("q2", true)]);
    merge_section(&path, &declared, "q1", "one\n", Regeneration::Whole);
    merge_section(&path, &declared, "q2", "two\n", Regeneration::Whole);
    assert_eq!(
        std::fs::read_to_string(&path).expect("the file"),
        "== q1\none\n== q2\ntwo\n"
    );
}

/// The same pair with the lock's guarantee removed — a writer that read the file before the
/// other wrote, and published afterwards. Its section is the only one left. This is what the
/// lock buys, stated as the outcome it prevents rather than as a mechanism nobody checks.
#[test]
fn a_writer_that_read_before_the_other_wrote_loses_that_section() {
    let path = scratch("lost");
    let declared = skeleton(&[("q1", true), ("q2", true)]);
    // What a writer holding no lock would have read: the file before q2 arrived.
    let read_first = String::new();
    merge_section(&path, &declared, "q2", "two\n", Regeneration::Whole);
    let published = merged_text(&read_first, &declared, "q1", "one\n", Regeneration::Whole);
    std::fs::write(&path, &published).expect("the write that loses it");
    assert_eq!(
        std::fs::read_to_string(&path).expect("the file"),
        "== q1\none\n",
        "q2's section survived a write that could not have seen it"
    );
}

/// And the lock is what makes that ordering impossible: a second handle cannot take it
/// while the first holds it. Asserted with `try_lock` rather than by racing two threads —
/// two threads that merely start together prove nothing on a fast machine.
#[test]
fn the_lock_excludes_a_second_writer_of_the_same_file() {
    let path = scratch("exclusive");
    std::fs::write(&path, "").expect("the file");
    let held = std::fs::File::options()
        .read(true)
        .write(true)
        .open(&path)
        .expect("open");
    held.lock().expect("the lock");
    let other = std::fs::File::options()
        .read(true)
        .write(true)
        .open(&path)
        .expect("open again");
    assert!(
        other.try_lock().is_err(),
        "a second writer took the lock while the first held it"
    );
    held.unlock().expect("unlock");
    assert!(
        other.try_lock().is_ok(),
        "and could not take it once the first let go"
    );
}

/// A filtered regeneration rewrites its own section and leaves every other byte alone,
/// including sections for queries no declaration accounts for — a filtered run has no
/// standing to call one stale.
#[test]
fn a_filtered_regeneration_leaves_every_other_byte_identical() {
    let before = "== q1\none\n== q2\ntwo\n== q9\nnine\n";
    let declared = skeleton(&[("q1", true), ("q2", true)]);
    let after = merged_text(before, &declared, "q1", "ONE\n", Regeneration::Sections);
    assert_eq!(after, "== q1\nONE\n== q2\ntwo\n== q9\nnine\n");
}

/// A whole-file regeneration is the one that may drop a section, and only one no
/// declaration accounts for.
#[test]
fn a_whole_regeneration_drops_only_what_no_declaration_accounts_for() {
    let before = "== q1\none\n== q2\ntwo\n== q9\nnine\n";
    let declared = skeleton(&[("q1", true), ("q2", true)]);
    let after = merged_text(before, &declared, "q1", "ONE\n", Regeneration::Whole);
    assert_eq!(after, "== q1\nONE\n== q2\ntwo\n");
}

/// A query enabled at this mode and absent from both the file and the run is left OUT, not
/// filled with a marker. The two absences are different facts: this run did not produce it,
/// which is not the same as the declaration saying it never runs here.
#[test]
fn a_section_the_run_did_not_produce_is_absent_rather_than_skipped() {
    let declared = skeleton(&[("q1", true), ("q2", true)]);
    let after = merged_text("", &declared, "q1", "one\n", Regeneration::Sections);
    assert_eq!(after, "== q1\none\n", "q2 was invented as skipped");
}

/// And the marker round-trips: a query whose bit is clear carries one, and clearing a bit
/// that was set turns a real section into a marker rather than deleting it.
#[test]
fn a_cleared_bit_turns_a_real_section_into_a_marker() {
    let both = skeleton(&[("q1", true), ("q2", true)]);
    let written = merged_text("", &both, "q2", "two\n", Regeneration::Whole);
    assert_eq!(written, "== q2\ntwo\n");

    let cleared = skeleton(&[("q1", true), ("q2", false)]);
    let after = merged_text(&written, &cleared, "q1", "one\n", Regeneration::Whole);
    assert_eq!(
        after,
        format!("== q1\none\n== q2\n{SKIPPED}not enabled at this mode\n"),
        "q2 was deleted rather than marked"
    );
}

/// The over-cap result keeps its section, says why, and names who decided — in that order.
///
/// Built by calling `over_cap` rather than by formatting a twin: nine sites read a leading
/// SKIPPED as "this section holds no rows", `corpus_gpu` among them, so a mode line ahead of
/// it would let a `golden_exact` device case compare against nothing. A hand-built body pins
/// the test's own string and leaves that ordering asserted nowhere.
#[test]
fn an_over_cap_result_is_a_marker_and_not_a_deletion() {
    let declared = skeleton(&[("q1", true)]);
    let mode = mode_named("bp_tp4_sized");
    let body = common::corpus::over_cap(Some(300_000), mode);
    let after = merged_text("", &declared, "q1", &body, Regeneration::Whole);
    assert_eq!(after, format!("== q1\n{body}"));
    let (_, held) = ordered_sections(&after).remove(0);
    assert!(
        held.starts_with(SKIPPED),
        "a reader cannot tell why it is absent, and every SKIPPED reader now sees rows: {held}"
    );
    assert_eq!(
        held.lines().nth(1),
        Some(format!("mode={}", mode.name).as_str()),
        "the marker must name its author on the SECOND line: {held}"
    );
}

// --- the committed goldens, against their own redundancy ------------------------

fn corpus_files() -> Vec<(String, String, &'static str, PathBuf)> {
    let mut files = Vec::new();
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        for mode in &BP_MODES {
            files.push((
                dataset.to_string(),
                sf.to_string(),
                mode.name,
                cpu_golden(dataset, sf, mode.name),
            ));
        }
    }
    files
}

fn sections_with_content(path: &Path) -> Vec<(String, String)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    ordered_sections(&text)
        .into_iter()
        .filter(|(_, body)| !body.starts_with(SKIPPED))
        .collect()
}

/// Every node's totals are the sums of its own per-batch lists, and the lane count of those
/// lists is the `lanes=N` the node declares.
#[test]
fn a_nodes_totals_are_the_sums_of_its_batches() {
    for (_, _, mode, path) in corpus_files() {
        for (query, body) in sections_with_content(&path) {
            let (_, nodes) = parse_run_section(&body);
            for node in &nodes {
                let at = format!("{mode} {query} {}", node.line.name);
                assert_eq!(
                    node.line.count("output_rows"),
                    Some(node.batch_rows.iter().flatten().sum()),
                    "{at}: output_rows is not the sum of batch_rows"
                );
                assert_eq!(
                    node.line.count("output_bytes"),
                    Some(node.batch_bytes.iter().flatten().sum()),
                    "{at}: output_bytes is not the sum of batch_bytes"
                );
                assert_eq!(
                    node.batch_rows.len(),
                    node.batch_bytes.len(),
                    "{at}: the two lists span different lanes"
                );
                // A node that declares no layout — the unload — declares no lane count
                // either, and its one lane is the list's own.
                if let Some(lanes) = node.line.count("lanes") {
                    assert_eq!(
                        node.batch_rows.len() as u64,
                        lanes,
                        "{at}: {} lanes of batches against lanes={lanes}",
                        node.batch_rows.len()
                    );
                }
            }
        }
    }
}

/// The conservation law over the file: what a node consumed from a child, plus what that
/// child abandoned, is what the child emitted — per the child's own lane, which is why
/// `in_rows` is indexed by it.
#[test]
fn every_row_a_node_emitted_was_consumed_or_abandoned() {
    let mut checked = 0;
    for (_, _, mode, path) in corpus_files() {
        for (query, body) in sections_with_content(&path) {
            let (_, nodes) = parse_run_section(&body);
            for node in &nodes {
                assert_eq!(
                    node.in_rows.len(),
                    node.children.len(),
                    "{mode} {query} {}: in_rows spans {} children and the node has {}",
                    node.line.name,
                    node.in_rows.len(),
                    node.children.len()
                );
                for (slot, child) in node.children.iter().enumerate() {
                    let child = &nodes[*child];
                    for (lane, emitted) in child.batch_rows.iter().enumerate() {
                        let consumed = node.in_rows[slot][lane];
                        let abandoned = child.abandoned.get(lane).copied().unwrap_or(0);
                        assert_eq!(
                            consumed + abandoned,
                            emitted.iter().sum::<u64>(),
                            "{mode} {query} {} lane {lane}: against {}",
                            node.line.name,
                            child.line.name
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0, "the law was asserted over nothing");
}

/// A loader emits one batch per entry of its own lane's mapping, and a prefix of it where
/// the run stopped early: the scheduler stops where it is, so one run can have lane 0
/// complete and lane 1 short. Compared per lane rather than over the flattened shape, which
/// is what the nesting was chosen for.
#[test]
fn a_loaders_batches_line_up_with_the_row_groups_that_made_them() {
    for (_, _, mode, path) in corpus_files() {
        for (query, body) in sections_with_content(&path) {
            let (marker, nodes) = parse_run_section(&body);
            for node in &nodes {
                let Some(groups) = node.line.field("partition_groups") else {
                    continue;
                };
                let planned = lane_batch_counts(groups);
                let at = format!("{mode} {query} {}", node.line.name);
                assert_eq!(
                    node.batch_rows.len(),
                    planned.len(),
                    "{at}: the mapping and the batches span different lanes"
                );
                for (lane, count) in planned.iter().enumerate() {
                    let emitted = node.batch_rows[lane].len();
                    match marker == "none" {
                        true => assert_eq!(
                            emitted, *count,
                            "{at} lane {lane}: {emitted} batches against {count} in the mapping"
                        ),
                        false => assert!(
                            emitted <= *count,
                            "{at} lane {lane}: {emitted} batches, more than the {count} planned"
                        ),
                    }
                }
            }
        }
    }
}

/// How many batches each lane of `partition_groups=[[[0,1],[2]],[[3]]]` plans: the count of
/// second-level lists, per first-level one.
fn lane_batch_counts(groups: &str) -> Vec<usize> {
    let inner = groups
        .strip_prefix('[')
        .and_then(|text| text.strip_suffix(']'))
        .unwrap_or_else(|| panic!("not a mapping: {groups}"));
    let mut lanes = Vec::new();
    let mut depth = 0usize;
    let mut batches = 0usize;
    for c in inner.chars() {
        match c {
            '[' => {
                depth += 1;
                if depth == 2 {
                    batches += 1;
                }
            }
            ']' => {
                depth -= 1;
                if depth == 0 {
                    lanes.push(batches);
                    batches = 0;
                }
            }
            _ => {}
        }
    }
    lanes
}

/// The root's `out_rows` against the row count in `.result.txt` — the one check in the file
/// with an external oracle behind it, since that result was compared to DataFusion. Only
/// the mode that authored the section can be held to it: where the SQL does not fix the row
/// set, another mode's answer is not the same answer.
#[test]
fn the_root_emitted_the_rows_the_result_golden_holds() {
    let mut checked = 0;
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        let results =
            std::fs::read_to_string(result_golden(dataset, sf)).expect("the result golden");
        for (query, result) in ordered_sections(&results) {
            if result.starts_with(SKIPPED) {
                continue;
            }
            let mode = result
                .lines()
                .next()
                .and_then(|line| line.strip_prefix("mode="))
                .expect("a result section names the mode that wrote it")
                .to_string();
            let body = sections_with_content(&cpu_golden(dataset, sf, &mode))
                .into_iter()
                .find(|(name, _)| *name == query)
                .map(|(_, body)| body)
                .unwrap_or_else(|| panic!("{mode}: no run for {query}, which authored its result"));
            let (_, nodes) = parse_run_section(&body);
            assert_eq!(
                nodes[0].line.count("output_rows"),
                Some(rendered_rows(&result) as u64),
                "{dataset}/{query} at {mode}: the root's rows and the result's disagree"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no result section was checked");
}

/// Data rows in a rendered result: the bordered table's `|` lines, less its header.
fn rendered_rows(result: &str) -> usize {
    result
        .lines()
        .filter(|line| line.starts_with('|'))
        .count()
        .saturating_sub(1)
}

/// Every cost section is derived from a `.cpu.txt` section that exists, and a skipped query
/// is skipped in both. The two files are written by different calls, so nothing else holds
/// them to the same set of queries.
#[test]
fn the_cost_golden_holds_the_same_queries_as_its_cpu_golden() {
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        for mode in &BP_MODES {
            let cpu = std::fs::read_to_string(cpu_golden(dataset, sf, mode.name)).expect("cpu");
            let cost = std::fs::read_to_string(cost_golden(dataset, sf, mode.name)).expect("cost");
            let names = |text: &str| -> Vec<(String, bool)> {
                ordered_sections(text)
                    .into_iter()
                    .map(|(name, body)| (name, body.starts_with(SKIPPED)))
                    .collect()
            };
            assert_eq!(
                names(&cpu),
                names(&cost),
                "{dataset} {}: the two goldens disagree about which queries ran",
                mode.name
            );
        }
    }
}

/// Every node line in every corpus golden parses, and carries the fields the arithmetic
/// above reads. A section that stopped parsing would make each of those checks pass over
/// nothing, which is the failure a count assertion cannot see from inside one test.
#[test]
fn every_node_line_in_every_corpus_golden_parses() {
    let mut nodes = 0;
    for (_, _, mode, path) in corpus_files() {
        for (query, body) in sections_with_content(&path) {
            let (_, parsed) = parse_run_section(&body);
            assert!(!parsed.is_empty(), "{mode} {query}: no nodes");
            for node in &parsed {
                let at = format!("{mode} {query} {}", node.line.name);
                assert!(
                    node.line.count("output_rows").is_some(),
                    "{at}: no output_rows"
                );
                assert!(
                    node.line.count("output_bytes").is_some(),
                    "{at}: no output_bytes"
                );
                nodes += 1;
            }
            // Every line is either a node or a per-batch line under one, so the count of
            // node lines is half the section's non-blank lines less the marker.
            let lines = body.lines().filter(|line| !line.trim().is_empty()).count();
            assert_eq!(
                parsed.len() * 2 + 1,
                lines,
                "{mode} {query}: {lines} lines against {} nodes",
                parsed.len()
            );
        }
    }
    assert!(nodes > 100, "only {nodes} nodes were read back");
}

/// The tree the parser rebuilds is the tree the indentation draws: every node but the root
/// has exactly one parent, and the root is the only node at depth zero.
#[test]
fn the_parsed_tree_is_the_one_the_indentation_draws() {
    for (_, _, mode, path) in corpus_files() {
        for (query, body) in sections_with_content(&path) {
            let (_, nodes) = parse_run_section(&body);
            let mut parents = vec![0usize; nodes.len()];
            for node in &nodes {
                for child in &node.children {
                    parents[*child] += 1;
                }
            }
            assert_eq!(parents[0], 0, "{mode} {query}: the root has a parent");
            assert!(
                parents[1..].iter().all(|count| *count == 1),
                "{mode} {query}: a node has {:?} parents",
                parents
            );
            assert_eq!(
                nodes[0].line.depth, 0,
                "{mode} {query}: the root is indented"
            );
            assert_eq!(
                nodes.iter().filter(|node| node.line.depth == 0).count(),
                1,
                "{mode} {query}: more than one node at depth zero"
            );
        }
    }
}

/// A section a reader takes for a run is one, and a marker is not: the two are told apart by
/// the prefix rather than by whether the parse happens to fail.
#[test]
fn a_marker_is_never_read_as_a_run() {
    let mut markers = 0;
    for (_, _, mode, path) in corpus_files() {
        let text = std::fs::read_to_string(&path).expect("a golden");
        for (query, body) in ordered_sections(&text) {
            if !body.starts_with(SKIPPED) {
                continue;
            }
            assert_eq!(
                body.lines().count(),
                1,
                "{mode} {query}: a marker is one line"
            );
            assert!(
                parse_node_line(body.lines().next().expect("the line")).is_none(),
                "{mode} {query}: a marker reads as a node line"
            );
            markers += 1;
        }
    }
    assert!(markers > 0, "no marker was checked, so nothing here ran");
}

/// The third leg of the three that have to agree. The registry is held to the macro by each
/// binary's inventory check, and the goldens are written FROM the registry — but nothing
/// read them back against it, so a hand-edited section or a stale file agreed with nobody
/// and went green.
///
/// Every enabled cell means a section carrying content; every disabled one means a marker.
/// The converse matters as much: a disabled cell whose section is full is coverage nobody
/// is reading.
#[test]
fn every_enabled_cell_has_a_section_with_content_and_every_disabled_one_a_marker() {
    let rows = common::registry::load_csv();
    let mut wrong: Vec<String> = Vec::new();
    let mut checked = 0;
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        for mode in &BP_MODES {
            let column = format!("bp_cpu_{}", mode.ident().trim_start_matches("bp_"));
            let text = std::fs::read_to_string(cpu_golden(dataset, sf, mode.name))
                .expect("the mode's golden");
            let sections: std::collections::BTreeMap<String, String> =
                ordered_sections(&text).into_iter().collect();
            for row in rows.iter().filter(|r| r.dataset == dataset && r.sf == sf) {
                let state = row.states.get(&column).map(String::as_str).unwrap_or("na");
                let held = sections.get(&common::registry::stem(&row.query));
                let at = format!("{dataset} {} {}", mode.name, common::registry::stem(&row.query));
                match (state, held) {
                    ("enabled", None) => wrong.push(format!("{at}: enabled and has no section")),
                    ("enabled", Some(body)) if body.starts_with(SKIPPED) => {
                        wrong.push(format!("{at}: enabled and its section is a marker"))
                    }
                    ("disabled", None) => wrong.push(format!("{at}: disabled and has no section")),
                    ("disabled", Some(body)) if !body.starts_with(SKIPPED) => wrong
                        .push(format!("{at}: disabled and its section holds a run — coverage \
                                       nobody is reading")),
                    ("na", Some(_)) => wrong.push(format!("{at}: not declared and has a section")),
                    _ => {}
                }
                checked += 1;
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert!(checked > 500, "only {checked} cells were checked");
}

/// Every section's `mode=` is the mode that would author it today.
///
/// The author is the LAST mode a query declares, so a cut removing trailing modes moves the
/// authorship — and a section written by the old author is correct-looking, current and stale. The
/// absent case is caught by the two tests that read the section; this is the present one, which
/// nothing else looks at.
///
/// Read off `ordered_sections` rather than `sections_with_content`, and discriminating on a `mode=`
/// line being PRESENT rather than on SKIPPED being absent: an over-cap section carries both, and it
/// is the section this guard most needs to see.
#[test]
fn every_result_section_names_the_mode_that_would_author_it_now() {
    let rows = common::registry::load_csv();
    let mut compared = 0;
    let mut markers = 0;
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        let path = result_golden(dataset, sf);
        let text = std::fs::read_to_string(&path).expect("the result golden");
        for (query, body) in ordered_sections(&text) {
            let Some(wrote) = body.lines().find_map(|l| l.strip_prefix("mode=")) else {
                markers += 1;
                continue;
            };
            let row = rows
                .iter()
                .find(|r| r.dataset == dataset && r.sf == sf && common::registry::stem(&r.query) == query)
                .unwrap_or_else(|| panic!("{dataset}/{query}: a section with no registry row"));
            // `enabled | skip` is the pair `declared_sections` uses to decide a query has a
            // section at all, so counting only `enabled` would disagree with the writer about
            // who the author is — and disagree on the rarest cells.
            let author = BP_MODES.iter().rev().find(|mode| {
                let column = format!("bp_cpu_{}", mode.ident().trim_start_matches("bp_"));
                row.states.get(&column).is_some_and(|s| s == "enabled" || s == "skip")
            });
            assert_eq!(
                Some(wrote),
                author.map(|mode| mode.name),
                "{dataset}/{query}: the section says {wrote} and the last mode it declares is {}",
                author.map(|mode| mode.name).unwrap_or("none")
            );
            compared += 1;
        }
    }
    // The exact count, derived rather than floored: every row with an authority has a
    // section carrying that mode, so the two numbers are the same set counted twice. A
    // floor of one passes on the day all but one stop being compared, which is the failure
    // a guard over a whole corpus is most likely to have.
    let expected = rows
        .iter()
        .filter(|r| ["tpch", "tpcds"].contains(&r.dataset.as_str()) && r.sf == "1")
        .filter(|r| {
            BP_MODES.iter().any(|mode| {
                let column = format!("bp_cpu_{}", mode.ident().trim_start_matches("bp_"));
                r.states.get(&column).is_some_and(|s| s == "enabled" || s == "skip")
            })
        })
        .count();
    println!("compared {compared} sections, {markers} markers");
    assert_eq!(compared, expected, "every row with an authority has one section naming it");
}

/// The one golden whose key carries no mode names the mode that wrote it, and that mode has
/// to be the authority: the LAST the query declares, in the fixed sequence. A section
/// written by any other mode is one a filtered regeneration re-authored from a run that was
/// not entitled to, which is the failure the authority rule exists to prevent — and the
/// body's own line would say so while nobody read it.
#[test]
fn each_result_section_was_written_by_the_mode_entitled_to_write_it() {
    let rows = common::registry::load_csv();
    for (dataset, sf) in [("tpch", "1"), ("tpcds", "1")] {
        let text = std::fs::read_to_string(result_golden(dataset, sf)).expect("the result golden");
        for (query, body) in ordered_sections(&text) {
            let row = rows
                .iter()
                .find(|r| r.dataset == dataset && r.sf == sf && common::registry::stem(&r.query) == query)
                .unwrap_or_else(|| panic!("{dataset}/{query}: a section with no registry row"));
            let entitled = BP_MODES.iter().rev().find(|mode| {
                let column = format!("bp_cpu_{}", mode.ident().trim_start_matches("bp_"));
                row.states.get(&column).map(String::as_str) == Some("enabled")
            });
            match (entitled, body.starts_with(SKIPPED)) {
                (None, true) => {}
                (None, false) => panic!("{dataset}/{query}: no mode is enabled and its section holds a run"),
                (Some(mode), true) => {
                    // Over the cap is the one reason an enabled query carries a marker, and
                    // it says so in words rather than by being absent.
                    assert!(
                        body.contains("cap"),
                        "{dataset}/{query}: enabled at {} and its section is a marker that does \
                         not say why:\n{body}",
                        mode.name
                    );
                }
                (Some(mode), false) => {
                    let author = body
                        .lines()
                        .next()
                        .and_then(|line| line.strip_prefix("mode="))
                        .unwrap_or_else(|| panic!("{dataset}/{query}: no `mode=` line"));
                    assert_eq!(
                        author, mode.name,
                        "{dataset}/{query}: written at {author}, and {} is the last mode it \
                         declares",
                        mode.name
                    );
                }
            }
        }
    }
}

/// The read-only path is read-only under a regeneration, which is what the device side
/// rests on: it calls `assert_section` and never `assert_or_merge`, and a device that can
/// author its own golden proves nothing against it.
///
/// Both variables set, a body that does NOT match, and the file must come back byte for
/// byte with the verification still failing. Asserting the property rather than trusting
/// that nothing on that path happens to call the write — the gpu binary links the write
/// path through `mod common` exactly like every other binary.
#[test]
fn a_regeneration_does_not_make_the_read_only_path_write() {
    let path = scratch("read-only");
    let before = "== q1\nthe committed body\n";
    std::fs::write(&path, before).expect("the file");
    // SAFETY: this binary's other cases drive the merge with an explicit `Regeneration`
    // and never read the environment, so nothing else here observes these.
    unsafe {
        std::env::set_var("UPDATE_CANONICAL", "1");
        std::env::set_var("PCK_UPDATE_SECTIONS", "1");
    }
    let wrote_anyway = std::panic::catch_unwind(|| {
        common::corpus_golden::assert_section(&path, "q1", "a body from a run\n");
    });
    unsafe {
        std::env::remove_var("UPDATE_CANONICAL");
        std::env::remove_var("PCK_UPDATE_SECTIONS");
    }
    assert!(wrote_anyway.is_err(), "it did not even verify");
    assert_eq!(
        std::fs::read_to_string(&path).expect("the file"),
        before,
        "the read-only path wrote under a regeneration"
    );
}

/// Every `total_us` in a benchmark file is the sum of the `time_us` beside it.
///
/// The one thing about that file checkable without a device and without a second oracle:
/// the renderer writes both numbers from the same measurement, so a file where they
/// disagree is a file the renderer got wrong. That is the same claim the cost-tree cases
/// above make — a golden with no external oracle is still checkable against its own
/// redundancy.
///
/// `-` is NOT "measured and unknown" — it is "no region opened", which is what a call
/// that made no ABI call of its own leaves behind: an accumulator below its compaction
/// threshold, an unload exporting through a door that opens none. No device time is
/// attributable to it, so the total sums the numeric entries and skips those. A node
/// whose entries are ALL `-` has nothing to sum and carries `-` itself.
///
/// Which is why this is worth pinning rather than obvious: the other reading — any `-`
/// poisons the total — is equally plausible from the file alone, and the two differ on
/// every node that has both.
///
/// Absent files are skipped rather than failed: this tree is written by a run on a GPU
/// host and a fresh checkout has none. What is NOT skipped is finding no file at all with
/// timing lines in it — a check that silently examines nothing is the failure mode this
/// suite exists to close.
#[test]
fn every_total_us_is_the_sum_of_the_time_us_beside_it() {
    let root = common::testdata_root().join("benchmark-results");
    let mut files = Vec::new();
    collect_benchmark_files(&root, &mut files);

    let mut checked = 0;
    for path in &files {
        let text = std::fs::read_to_string(path).expect("a benchmark file");
        for (n, line) in text.lines().enumerate() {
            let Some(rest) = line.trim().strip_prefix("time_us=") else {
                continue;
            };
            let (array, total) = rest
                .split_once(" total_us=")
                .unwrap_or_else(|| panic!("{}:{}: no total_us on {line:?}", path.display(), n + 1));
            // Every number between the brackets, in one pass: the nesting says which lane
            // a call was on and the sum does not care.
            let entries: Vec<&str> = array
                .trim_matches(|c| c == '[' || c == ']')
                .split(|c| c == ',' || c == '[' || c == ']')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            let numbers: Vec<u64> = entries.iter().filter_map(|e| e.parse().ok()).collect();
            let unknown = entries.len() - numbers.len();
            let want = match numbers.is_empty() {
                false => numbers.iter().sum::<u64>().to_string(),
                true => "-".to_string(),
            };
            assert_eq!(
                total,
                want,
                "{}:{}: total_us disagrees with the {} measured entries beside it \
                 ({unknown} opened no region) — {line:?}",
                path.display(),
                n + 1,
                numbers.len()
            );
            checked += 1;
        }
    }
    assert!(
        files.is_empty() || checked > 0,
        "{} benchmark file(s) and not one timing line: the format moved and this check \
         is reading past it",
        files.len()
    );
}

fn collect_benchmark_files(dir: &Path, into: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_benchmark_files(&path, into);
        } else if path.to_string_lossy().ends_with(".benchmark.txt") {
            into.push(path);
        }
    }
}
