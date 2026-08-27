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
    ($dataset:ident, $sf:expr, $query:ident, none, $($gpu:ident)|+, $cpu_oracle:ident, $gpu_oracle:ident) => {};
    ($dataset:ident, $sf:expr, $query:ident, $($cpu:ident)|+, $($gpu:ident)|+, $cpu_oracle:ident, $gpu_oracle:ident) => {
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

include!("common/corpus_cases.inc");

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
