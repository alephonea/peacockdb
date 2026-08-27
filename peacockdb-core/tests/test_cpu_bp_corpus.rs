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
/// what makes it catch the first `live_cpu` query BEFORE the rollout that needs it, rather
/// than during.
#[test]
fn each_declarations_two_oracles_suit_each_other() {
    let mut wrong: Vec<String> = Vec::new();
    for declared in inventory::iter::<common::registry::CorpusDeclaration> {
        let query = declared.query.replace('_', "-");
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
