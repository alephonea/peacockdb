//! Cost-model goldens: derive each `<query>.<mode>-<tp>-<tier>.cost.txt` from its
//! sibling `<query>.<mode>-<tp>-<tier>.cpu.txt` text and assert it matches (or
//! regenerate it under `UPDATE_CANONICAL=1`). Pure text — no executor, no dataset —
//! so it runs in the
//! plain CPU CI tier. The taxonomy + multipliers live in `common/cost_model.rs`.
//!
//! Byte-identity invariant: at today's all-1.0 multipliers the `.cost.txt` total
//! equals `Σ output_bytes` over the `.cpu.txt` tree.
//!
//! `COST_FILTER=<substr>` restricts the run to `.cpu.txt` files whose name
//! contains `<substr>` (regenerate a single golden).
#[macro_use]
mod common;

use std::path::PathBuf;

use common::cost_model::CostModel;

/// Σ of every `output_bytes=` value in a `.cpu.txt` body.
fn sum_output_bytes(cpu_text: &str) -> u64 {
    const KEY: &str = "output_bytes=";
    cpu_text
        .lines()
        .filter_map(|l| l.find(KEY).map(|p| &l[p + KEY.len()..]))
        .map(|tail| {
            tail.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<u64>()
                .unwrap()
        })
        .sum()
}

/// Σ of every `peacockdb_cost=` in the text — one in a legacy cost golden, one per section
/// in a corpus one, and the sum is what the `.cpu.txt` beside it has to account for either
/// way.
fn cost_total(cost_text: &str) -> u64 {
    const KEY: &str = "peacockdb_cost=";
    let totals: Vec<u64> = cost_text
        .lines()
        .filter_map(|line| line.strip_prefix(KEY))
        .map(|total| total.parse().expect("a cost total"))
        .collect();
    assert!(!totals.is_empty(), "cost text missing total footer");
    totals.iter().sum()
}

#[test]
fn cost_goldens_match_and_total_is_byte_identical() {
    let update = std::env::var("UPDATE_CANONICAL").is_ok();
    let filter = std::env::var("COST_FILTER").ok();
    let model = CostModel::load();

    let dirs = [
        common::golden_dir_for("tpch", "1"),
        common::golden_dir_for("tpcds", "1"),
    ];
    let mut cpu_files: Vec<PathBuf> = dirs
        .iter()
        .flat_map(|d| {
            std::fs::read_dir(d).unwrap_or_else(|e| panic!("read_dir {}: {e}", d.display()))
        })
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_str().map(|s| s.ends_with(".cpu.txt")).unwrap_or(false))
        .filter(|p| match &filter {
            Some(f) => p
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.contains(f.as_str()))
                .unwrap_or(false),
            None => true,
        })
        .collect();
    cpu_files.sort();
    assert!(
        !cpu_files.is_empty(),
        "no .cpu.txt goldens matched (filter {filter:?})"
    );

    let mut mismatches: Vec<String> = Vec::new();
    for cpu_path in &cpu_files {
        let name = cpu_path.file_name().and_then(|s| s.to_str()).unwrap();
        let cpu_text = std::fs::read_to_string(cpu_path).unwrap();
        // A corpus golden holds every query in sections and its cost golden mirrors them,
        // so the derivation is per section there and per file for the legacy one-query
        // form. Decided by what the file holds rather than by its name: the two forms are
        // distinguishable, and a name convention is a second thing to keep true.
        let sectioned = cpu_text.starts_with("== ");
        let actual = match sectioned {
            true => model.cost_text_from_sections(&cpu_text, name),
            false => model.cost_text_from_cpu(&cpu_text, name),
        };

        // Invariant: total == Σ output_bytes in the .cpu.txt (multipliers all 1.0).
        let total = cost_total(&actual);
        let expected_total = sum_output_bytes(&cpu_text);
        assert_eq!(
            total, expected_total,
            "{name}: .cost.txt total {total} != Σ output_bytes {expected_total} in .cpu.txt"
        );

        let cost_path = cpu_path.with_file_name(name.replace(".cpu.txt", ".cost.txt"));
        if update {
            std::fs::write(&cost_path, &actual).unwrap();
            eprintln!("Updated cost golden: {}", cost_path.display());
            continue;
        }
        match std::fs::read_to_string(&cost_path) {
            Ok(golden) if golden.trim_end() == actual.trim_end() => {}
            Ok(_) => mismatches.push(format!(
                "{name}: .cost.txt does not match (run with UPDATE_CANONICAL=1)"
            )),
            Err(_) => mismatches.push(format!(
                "{}: missing (run with UPDATE_CANONICAL=1)",
                cost_path.display()
            )),
        }
    }
    assert!(
        mismatches.is_empty(),
        "cost golden mismatches:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn generator_bins_and_totals_synthetic_tree() {
    // Covers: a node with no args renders bare (`GpuCoalescePartitionsExec,
    // output_bytes=…`, no colon) and must still bin; nodes sharing a category sum;
    // an unmapped category lands at 0; total == Σ output_bytes (multipliers 1.0).
    let cpu = "\
GpuSortExec: expr=[x], output_bytes=10, output_rows=1
  GpuCoalescePartitionsExec, output_bytes=20, output_rows=2
    GpuScanExec: table=t, output_bytes=30, output_rows=3";
    let cost = CostModel::load().cost_text_from_cpu(cpu, "synthetic");
    assert!(cost.contains("storage_read_bytes=30 #"));
    assert!(cost.contains("cuda_sort_bytes=10 #"));
    assert!(cost.contains("cuda_shuffle_bytes=20 #")); // bare CoalescePartitions binned
    assert!(cost.contains("cuda_window_bytes=0 #")); // unused category present at 0
    assert!(cost.contains("ram_to_vram_bytes=0 # (placeholder, no node mapping)"));
    assert_eq!(cost_total(&cost), 60); // 10 + 20 + 30 == Σ output_bytes
}

/// Every batch-partitioned node kind is in the taxonomy, and every batch-partitioned name
/// in the taxonomy is a kind — read off the exhaustive match in `nodes::node_name`, which
/// is the one place a kind is named. A nineteenth kind adds an arm there and reddens this
/// until it has an entry, which is what makes "all eighteen at once" a property rather
/// than a moment: T19 enables queries and must never touch this file.
#[test]
fn every_node_kind_is_in_exactly_one_cost_category() {
    let model = CostModel::load();
    let mut binned: Vec<(&str, usize)> = Vec::new();
    for name in node_kind_names() {
        let categories = model
            .categories
            .iter()
            .filter(|c| c.nodes.iter().any(|n| *n == name))
            .count();
        binned.push((name, categories));
    }
    let missing: Vec<&str> = binned
        .iter()
        .filter(|(_, n)| *n == 0)
        .map(|(name, _)| *name)
        .collect();
    assert!(
        missing.is_empty(),
        "node kinds with no cost category: {missing:?}\n\
         Add each to a category in testdata/cost_model.conf — the taxonomy must be total, \
         or the first .cpu.txt holding one of them cannot derive its .cost.txt."
    );
    let twice: Vec<&str> = binned
        .iter()
        .filter(|(_, n)| *n > 1)
        .map(|(name, _)| *name)
        .collect();
    assert!(
        twice.is_empty(),
        "node kinds in more than one category: {twice:?}"
    );

    // And back: a name here that no kind carries is a category counting bytes nothing
    // can produce, which reads as coverage and is not.
    let kinds: std::collections::BTreeSet<&str> = node_kind_names().into_iter().collect();
    for category in &model.categories {
        for node in &category.nodes {
            assert!(
                node.ends_with("Exec") || kinds.contains(node.as_str()),
                "cost_model.conf names '{node}', which is neither a legacy wrapper nor a \
                 batch-partitioned node kind"
            );
        }
    }
}

/// The node names, parsed out of `node_name`'s match arms. Reading the source rather than
/// constructing one node of each kind, for the reason the writer's field cover reads the
/// fbs: a list of instances can miss a kind, where the exhaustive match cannot.
fn node_kind_names() -> Vec<&'static str> {
    const SOURCE: &str = include_str!("../src/batch_partitioned/nodes/mod.rs");
    let body = SOURCE
        .split_once("pub(crate) fn node_name(")
        .expect("nodes/mod.rs declares node_name")
        .1;
    let body = body.split_once("\n}").expect("node_name has a body").0;
    let names: Vec<&str> = body
        .lines()
        .filter_map(|line| line.split_once("=> \"")?.1.split_once('"'))
        .map(|(name, _)| name)
        .collect();
    // Against the arms, not against a floor: an arm this scan misses — a long one rustfmt
    // wrapped onto two lines — would leave its kind quietly outside the taxonomy check,
    // which is the shape of failure the check exists to prevent one level up.
    assert_eq!(
        names.len(),
        body.matches("=>").count(),
        "node_name has {} arms and {} were parsed — the scan is missing one",
        body.matches("=>").count(),
        names.len()
    );
    names
}
