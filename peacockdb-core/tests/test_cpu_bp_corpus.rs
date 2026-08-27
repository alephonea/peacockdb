//! The corpus on the CPU backend: every query in
//! [`corpus_cases.inc`](common/corpus_cases.inc) at every mode it declares.
//!
//! This file holds the macro and nothing else — the declarations are the include, shared
//! with the device binary so one line carries a query's coverage on both engines. What a
//! case does lives in `common::corpus`.
#[macro_use]
mod common;

use common::registry::RegistryEntry;

/// `corpus_query!(dataset, sf, query, cpu_modes, gpu_modes, cpu_oracle, gpu_oracle)` — one
/// test and one registration per enabled cpu mode. The mode arguments read as a bitwise or
/// and are matched as idents, which is what lets the expansion produce a case per mode
/// rather than a case that decides at run time whether it is one: a disabled mode has no
/// test to name and no registration to explain.
///
/// The device's four arguments are consumed and dropped here. That is the point of one
/// list: this binary cannot silently disagree with the other about which query exists.
macro_rules! corpus_query {
    ($dataset:ident, $sf:expr, $query:ident, none, $($gpu:ident)|+, $cpu_oracle:ident, $gpu_oracle:ident) => {
        declare_corpus_query!($dataset, $sf, $query, $cpu_oracle, $gpu_oracle);
    };
    ($dataset:ident, $sf:expr, $query:ident, $($cpu:ident)|+, $($gpu:ident)|+, $cpu_oracle:ident, $gpu_oracle:ident) => {
        declare_corpus_query!($dataset, $sf, $query, $cpu_oracle, $gpu_oracle);
        $(
            paste::paste! {
                #[tokio::test]
                async fn [<bp_cpu_ $dataset _ $query _ $cpu>]() {
                    common::corpus::cpu_case(
                        stringify!($dataset),
                        stringify!($sf),
                        &stringify!($query).replace('_', "-"),
                        stringify!($cpu),
                        stringify!($cpu_oracle),
                    )
                    .await;
                }
            }
            inventory::submit! {
                RegistryEntry {
                    kind: "bp_cpu",
                    dataset: stringify!($dataset),
                    sf: stringify!($sf),
                    query: stringify!($query),
                    device: stringify!($cpu),
                    state: "enabled",
                }
            }
        )+
    };
}

/// The line itself, submitted by both arms — a query with no enabled mode still declared
/// two oracles, and the pairing between them is a property of the line rather than of a run.
macro_rules! declare_corpus_query {
    ($dataset:ident, $sf:expr, $query:ident, $cpu_oracle:ident, $gpu_oracle:ident) => {
        inventory::submit! {
            common::registry::CorpusDeclaration {
                dataset: stringify!($dataset),
                sf: stringify!($sf),
                query: stringify!($query),
                cpu_oracle: stringify!($cpu_oracle),
                gpu_oracle: stringify!($gpu_oracle),
            }
        }
    };
}

include!("common/corpus_cases.inc");

/// The two oracles of one line have to suit each other, and both directions are asserted
/// rather than trusted.
///
/// A `golden_exact` where no committed section can serve is a test that fails on correct
/// behaviour: the result is over the cap and has a marker instead, or the query's rows are
/// not determined across modes and one mode's answer cannot be the authority for five. A
/// `live_cpu` where a section does serve spends a device-side cpu run on a comparison a
/// committed file makes faster and harder.
///
/// Derivable is why a CHECK can exist here, never why either value would be absent from the
/// line. Read off the declaration and the committed golden, so it needs no run — which is
/// A device cell exists only where the cpu has one at the same mode.
///
/// The device tier asserts read-only against the section the cpu authored AT THAT MODE, so a
/// gpu cell whose cpu twin is off compares against a skipped marker and passes having checked
/// nothing. It holds across all 600 cells today by six hand-chosen cells rather than by a
/// rule, and the moment [#152] clears somebody enables device modes in bulk.
///
/// Read off the registry rather than the declarations: `CorpusDeclaration` carries the two
/// oracles and not the modes, and a disabled mode submits no registration to compare.
#[test]
fn every_device_cell_has_a_cpu_cell_at_the_same_mode() {
    let mut wrong: Vec<String> = Vec::new();
    let mut checked = 0;
    for row in common::registry::load_csv() {
        for mode in &common::bp_mode::BP_MODES {
            let suffix = mode.ident().trim_start_matches("bp_").to_string();
            let live = |prefix: &str| {
                row.states
                    .get(&format!("{prefix}{suffix}"))
                    .is_some_and(|s| s == "enabled" || s == "skip")
            };
            checked += 1;
            if live("bp_gpu_") && !live("bp_cpu_") {
                wrong.push(format!("{}/{} at {}", row.dataset, row.query, mode.name));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "these device cells have no cpu cell at the same mode, so each compares against a \
         marker and passes having checked nothing: {wrong:?}"
    );
    assert_eq!(checked, common::registry::load_csv().len() * common::bp_mode::BP_MODES.len());
}

/// what makes it catch the first `live_cpu` query BEFORE the rollout that needs it, rather
/// than during.
#[test]
fn each_declarations_two_oracles_suit_each_other() {
    let mut wrong: Vec<String> = Vec::new();
    for declared in inventory::iter::<common::registry::CorpusDeclaration> {
        let query = common::registry::stem(declared.query);
        let authority = common::corpus::authoritative_mode(declared.dataset, declared.sf, &query);
        // A query with no enabled mode has no result section to reason about, and its
        // oracles are inert until one is enabled.
        if authority.is_none() {
            continue;
        }
        let section = common::corpus_golden::section_of(
            &common::corpus_golden::result_golden(declared.dataset, declared.sf),
            &query,
        );
        // The two conditions the entry names, read off what is committed and off the line.
        let over_cap = section.starts_with(common::corpus_golden::SKIPPED);
        let undetermined = declared.cpu_oracle == "data_fusion_subset";
        let needs_live = over_cap || undetermined;
        let says_live = declared.gpu_oracle == "live_cpu";
        if needs_live && !says_live {
            wrong.push(format!(
                "{}/{query}: gpu_oracle is {} where no committed section can serve it ({}), \
                 so it fails on correct behaviour",
                declared.dataset,
                declared.gpu_oracle,
                match (over_cap, undetermined) {
                    (true, true) => "the result is over the cap AND its rows are undetermined",
                    (true, false) => "the result is over the cap",
                    _ => "cpu_oracle is data_fusion_subset, so the modes need not agree",
                }
            ));
        }
        if says_live && !needs_live {
            wrong.push(format!(
                "{}/{query}: gpu_oracle is live_cpu and `.result.txt` holds this query's rows \
                 — a device-side cpu run per mode for what the committed section already says",
                declared.dataset
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A hyphenated query resolves its authority, and that authority is what silence would cost.
///
/// The CSV spells a query as an identifier and every golden section spells it with hyphens, so
/// a reader comparing the two directly finds no row and answers None — and None is a legal
/// answer here, meaning "no mode is enabled". So nothing writes the query's `.result.txt`
/// section, nothing reads it, and the test above excuses the query at its `authority.is_none()`
/// guard. T18's twenty are all `qNN` and cannot reach it; T19's first batch is five queries that
/// can, `scan-limit` among them — which is the query that test was written for.
#[test]
fn a_hyphenated_query_resolves_its_authority_and_has_its_result_section() {
    let rows = common::registry::load_csv();
    let mut checked = 0;
    for declared in inventory::iter::<common::registry::CorpusDeclaration> {
        let query = common::registry::stem(declared.query);
        if query == declared.query {
            continue;
        }
        let row = rows
            .iter()
            .find(|r| r.dataset == declared.dataset && r.sf == declared.sf && r.query == declared.query)
            .unwrap_or_else(|| panic!("{}/{query}: declared and not in the registry", declared.dataset));
        // A query enabled at no mode has no authority to resolve, which is the same None for
        // an entirely different reason — the one this test exists to tell apart.
        if !row.states.iter().any(|(col, state)| col.starts_with("bp_cpu_") && state == "enabled") {
            continue;
        }
        let authority = common::corpus::authoritative_mode(declared.dataset, declared.sf, &query);
        assert!(
            authority.is_some(),
            "{}/{query} is enabled and resolves no authoritative mode. The registry spells it \
             {} and this asked for {query}, so the row was never found — no result section is \
             written, nothing checks it, and the oracle pairing above excuses the query.",
            declared.dataset,
            declared.query
        );
        common::corpus_golden::section_of(
            &common::corpus_golden::result_golden(declared.dataset, declared.sf),
            &query,
        );
        checked += 1;
    }
    // The exact count rather than a floor: a floor of one passes on the day all but one
    // hyphenated query stops being checked, and the set is derivable from the same two
    // sources the loop reads.
    let expected = inventory::iter::<common::registry::CorpusDeclaration>
        .into_iter()
        .filter(|d| common::registry::stem(d.query) != d.query)
        .filter(|d| {
            rows.iter()
                .find(|r| r.dataset == d.dataset && r.sf == d.sf && r.query == d.query)
                .is_some_and(|r| {
                    r.states.iter().any(|(col, state)| col.starts_with("bp_cpu_") && state == "enabled")
                })
        })
        .count();
    assert_eq!(checked, expected, "every enabled hyphenated query is checked, and only those");
}

/// The five `bp_cpu_` columns against what this binary declares, in both directions: a
/// registration whose cell says otherwise, and a cell no case backs, both fail. The device
/// half is checked in the device binary, because `inventory` collects per linked binary.
#[test]
fn the_registry_matches_the_cpu_corpus_in_both_directions() {
    common::registry::assert_registry_matches_csv(
        &[
            "bp_cpu_tp1_single",
            "bp_cpu_tp1_rowgroup",
            "bp_cpu_tp4_single",
            "bp_cpu_tp4_rowgroup",
            "bp_cpu_tp4_sized",
        ],
        &[],
    );
}
