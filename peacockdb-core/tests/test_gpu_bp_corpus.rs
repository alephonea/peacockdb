//! The corpus on a device: every query in [`corpus_cases.inc`](common/corpus_cases.inc) at
//! every mode its `gpu_modes` declares.
//!
//! The same declaration list the cpu binary reads, so the two engines' coverage is one line
//! per query. This binary never writes a golden — see `common::corpus_gpu`.
#![cfg(not(feature = "rust-only"))]
#[macro_use]
mod common;

use common::registry::RegistryEntry;

/// The device's reading of a declaration: one test and one registration per enabled gpu
/// mode, and nothing at all for `none`. The cpu arguments are consumed and dropped, which
/// is what makes one list serve both binaries.
macro_rules! corpus_query {
    ($dataset:ident, $sf:expr, $query:ident, $($cpu:ident)|+, none, $cpu_oracle:ident, $gpu_oracle:ident) => {};
    ($dataset:ident, $sf:expr, $query:ident, $($cpu:ident)|+, $($gpu:ident)|+, $cpu_oracle:ident, $gpu_oracle:ident) => {
        $(
            paste::paste! {
                #[tokio::test]
                async fn [<bp_gpu_ $dataset _ $query _ $gpu>]() {
                    common::corpus_gpu::gpu_case(
                        stringify!($dataset),
                        stringify!($sf),
                        &stringify!($query).replace('_', "-"),
                        stringify!($gpu),
                        stringify!($gpu_oracle),
                    )
                    .await;
                }
            }
            inventory::submit! {
                RegistryEntry {
                    kind: "bp_gpu",
                    dataset: stringify!($dataset),
                    sf: stringify!($sf),
                    query: stringify!($query),
                    device: stringify!($gpu),
                    state: "enabled",
                }
            }
        )+
    };
}

include!("common/corpus_cases.inc");

/// The five `bp_gpu_` columns against what this binary declares, in both directions. It
/// runs only on the gpu host, because that is where these registrations are linked — so a
/// gpu-column drift does not go red on the cpu leg, which is a consequence to know rather
/// than a gap to close: `inventory` collects per linked binary.
#[test]
fn the_registry_matches_the_gpu_corpus_in_both_directions() {
    common::registry::assert_registry_matches_csv(
        &[
            "bp_gpu_tp1_single",
            "bp_gpu_tp1_rowgroup",
            "bp_gpu_tp4_single",
            "bp_gpu_tp4_rowgroup",
            "bp_gpu_tp4_sized",
        ],
        &[],
    );
}
