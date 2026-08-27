//! The corpus goldens: where they live, how a case reads one section, and how a
//! regenerating case writes one without losing the others.
//!
//! One file per mode holds every query, so several test cases write one path. A whole-file
//! write would be last-writer-wins — every other query's section dropped, and the run
//! green. So a write takes an advisory lock on the file, merges its own section into what
//! is there, and publishes by renaming a sibling onto the name.

use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use super::bp_mode::TIER;
use super::golden_text::{line_difference, ordered_sections};
use super::{golden_dir_for, registry};

/// What a section says when it holds no content. One prefix for every such reason, so a
/// reader scanning a file sees the same word wherever a section is not a run.
pub const SKIPPED: &str = "skipped: ";

/// `<mode>-<tier>.cpu.txt` — the per-node tree of every query that ran at this mode.
pub fn cpu_golden(dataset: &str, sf: &str, mode: &str) -> PathBuf {
    golden_dir_for(dataset, sf).join(format!("{mode}-{}.cpu.txt", TIER.label()))
}

/// `<mode>-<tier>.cost.txt`, derived per section from the `.cpu.txt` beside it.
pub fn cost_golden(dataset: &str, sf: &str, mode: &str) -> PathBuf {
    golden_dir_for(dataset, sf).join(format!("{mode}-{}.cost.txt", TIER.label()))
}

/// `bp-<tier>.result.txt` — one entry per query, keyed by the query alone, because the
/// modes are supposed to agree on results. The tier is in the name for the same reason it
/// is in the others': a second tier would be a second file rather than a silent overwrite.
pub fn result_golden(dataset: &str, sf: &str) -> PathBuf {
    golden_dir_for(dataset, sf).join(format!("bp-{}.result.txt", TIER.label()))
}

/// Whether this run writes goldens, and how much of the file it owns when it does.
///
/// `UPDATE_CANONICAL`'s contract is a whole file, which a corpus file cannot honour from
/// one case: the sections belong to different cases and a filtered run has only some of
/// them. So the whole-file form is what a full run means, and `PCK_UPDATE_SECTIONS` is the
/// filtered one — the same merge, without the pruning that a whole-file rewrite implies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Regeneration {
    /// Verify. The default, and what CI always does.
    No,
    /// Merge this section and prune sections no declaration accounts for.
    Whole,
    /// Merge this section and leave every other byte of the file alone.
    Sections,
}

pub fn regeneration() -> Regeneration {
    match (
        std::env::var("UPDATE_CANONICAL").is_ok(),
        std::env::var("PCK_UPDATE_SECTIONS").is_ok(),
    ) {
        (_, true) => Regeneration::Sections,
        (true, false) => Regeneration::Whole,
        (false, false) => Regeneration::No,
    }
}

/// Read this query's section, or panic naming what a reader has to do next. A missing file
/// and a missing section are different failures and say so: the first is a golden nobody
/// has generated, the second is a query whose coverage is claimed and not recorded.
pub fn section_of(path: &Path, query: &str) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_else(|_| {
        panic!(
            "golden not found: {}\nRun with UPDATE_CANONICAL=1 to generate it.",
            path.display()
        )
    });
    ordered_sections(&text)
        .into_iter()
        .find(|(name, _)| name == query)
        .map(|(_, body)| body)
        .unwrap_or_else(|| {
            panic!(
                "{}: no `== {query}` section. A query enabled at this mode and absent from \
                 its golden is coverage claimed and not recorded; regenerate with \
                 PCK_UPDATE_SECTIONS=1 and a filter naming it.",
                path.display()
            )
        })
}

/// Verify one section against `body`, or write it, depending on the run.
pub fn assert_or_merge(
    path: &Path,
    dataset: &str,
    sf: &str,
    columns: &[&str],
    query: &str,
    body: &str,
) {
    // Every body ends with a newline, because a section read back always does: the
    // comparison would otherwise turn a caller's missing trailing byte into a moved section.
    let body = match body.ends_with('\n') {
        true => body.to_string(),
        false => format!("{body}\n"),
    };
    match regeneration() {
        Regeneration::No => assert_section(path, query, &body),
        mode => merge(path, dataset, sf, columns, query, &body, mode),
    }
}

/// Verify one section and never write, whatever the run was asked to regenerate. What the
/// device side uses: a device that can author its own golden proves nothing against it.
pub fn assert_section(path: &Path, query: &str, body: &str) {
    let canonical = section_of(path, query);
    assert!(
        canonical == body,
        "{}: `{query}` moved — {}",
        path.display(),
        line_difference(&canonical, body)
    );
}

/// Merge one section into the file under an advisory lock, and publish by rename.
///
/// The lock is on the file rather than in the process, and the distinction is what makes it
/// sufficient: libtest runs a binary's cases as threads, so a `Mutex` would serialize
/// those and nothing else — not `cargo nextest`, which gives each case its own process, and
/// not two shells regenerating at once. The rename is for the other half: a crash mid-write
/// must not leave a truncated golden.
///
/// The read is inside the critical section, which is the whole point: a writer that read
/// before another wrote would publish a file missing the other's section.
pub fn merge_section(
    path: &Path,
    declared: &[(String, Option<String>)],
    query: &str,
    body: &str,
    mode: Regeneration,
) {
    std::fs::create_dir_all(path.parent().expect("a golden directory")).expect("the directory");
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap_or_else(|e| panic!("cannot open {} to merge into: {e}", path.display()));
    file.lock()
        .unwrap_or_else(|e| panic!("cannot lock {}: {e}", path.display()));
    let mut text = String::new();
    file.read_to_string(&mut text)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let merged = merged_text(&text, declared, query, body, mode);
    publish(path, &merged);
    // Explicit rather than left to the drop, so the unlock is ordered after the rename.
    let _ = file.unlock();
}

fn merge(
    path: &Path,
    dataset: &str,
    sf: &str,
    columns: &[&str],
    query: &str,
    body: &str,
    mode: Regeneration,
) {
    let declared = declared_sections(dataset, sf, columns);
    merge_section(path, &declared, query, body, mode);
}

/// The file as it will be written: every declared query in declaration order, this one's
/// section replaced, and each of the others kept as it stands.
///
/// A query enabled at this mode and absent from both the file and this run is left out
/// rather than filled with a marker: a filtered run cannot produce it and has no standing
/// to say it is skipped, and the read path names it as missing, which is what a golden is
/// for. A disabled one always renders its marker, since that fact needs no run.
pub fn merged_text(
    text: &str,
    declared: &[(String, Option<String>)],
    query: &str,
    body: &str,
    mode: Regeneration,
) -> String {
    let mut held: Vec<(String, String)> = ordered_sections(text);
    match held.iter_mut().find(|(name, _)| name == query) {
        Some(section) => section.1 = body.to_string(),
        None => held.push((query.to_string(), body.to_string())),
    }
    let mut out = String::new();
    for (name, marker) in declared {
        match (marker, held.iter().find(|(held, _)| held == name)) {
            (Some(marker), _) => push_section(&mut out, name, marker),
            (None, Some((_, body))) => push_section(&mut out, name, body),
            (None, None) => continue,
        }
    }
    if mode == Regeneration::Sections {
        // Nothing the declarations do not account for is dropped: a filtered run has no
        // standing to say a section is stale.
        for (name, body) in &held {
            if !declared.iter().any(|(declared, _)| declared == name) {
                push_section(&mut out, name, body);
            }
        }
    }
    out
}

fn push_section(out: &mut String, name: &str, body: &str) {
    out.push_str(&format!("== {name}\n"));
    out.push_str(body);
    if !body.ends_with('\n') {
        out.push('\n');
    }
}

/// Every query that should have a section in this file, in the registry's row order, with
/// the marker for those whose cells are not enabled. The registry is the skeleton because
/// it is the one place that knows a query is declared and disabled — a disabled mode
/// submits no registration, so the inventory cannot say it.
///
/// `columns` is one column for a per-mode file and all five for the result golden, which
/// holds one entry per query across the modes: enabled anywhere is enabled there.
pub fn declared_sections(
    dataset: &str,
    sf: &str,
    columns: &[&str],
) -> Vec<(String, Option<String>)> {
    registry::load_csv()
        .into_iter()
        .filter(|row| row.dataset == dataset && row.sf == sf)
        .filter_map(|row| {
            let states: Vec<&str> = columns
                .iter()
                .map(|column| row.states.get(*column).map(String::as_str).unwrap_or("na"))
                .collect();
            if states
                .iter()
                .any(|state| matches!(*state, "enabled" | "skip"))
            {
                return Some((row.query, None));
            }
            // The result golden spans the modes, so its marker cannot say "this mode" —
            // absent for one reason reading as absent for another is the whole thing these
            // markers exist to stop.
            let where_ = match columns.len() {
                1 => "at this mode",
                _ => "at any mode",
            };
            states
                .contains(&"disabled")
                .then(|| (row.query, Some(format!("{SKIPPED}not enabled {where_}\n"))))
        })
        .collect()
}

/// Write a sibling and rename onto the name, so a reader of the path sees the whole file or
/// the previous one and never a partial write. The sibling carries the process id: two
/// binaries reaching one path at the same time is what `canonical_root` records the same
/// surprise about.
fn publish(path: &Path, text: &str) {
    let staged = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = std::fs::File::create(&staged)
        .unwrap_or_else(|e| panic!("cannot stage {}: {e}", staged.display()));
    file.write_all(text.as_bytes())
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", staged.display()));
    file.sync_all()
        .unwrap_or_else(|e| panic!("cannot flush {}: {e}", staged.display()));
    std::fs::rename(&staged, path)
        .unwrap_or_else(|e| panic!("cannot publish {}: {e}", path.display()));
}

/// Re-read a file after a merge. Only the tests of this module need it; a case reads its
/// own section through [`section_of`].
pub fn read_back(path: &Path) -> String {
    let mut file = std::fs::File::open(path).expect("the golden");
    let mut text = String::new();
    file.rewind().expect("rewind");
    file.read_to_string(&mut text).expect("read");
    text
}
