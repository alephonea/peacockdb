//! Cost-model goldens: derive each `<query>.<device>.cost.txt` from its sibling
//! `<query>.<device>.cpu.txt` text and assert it matches (or regenerate it under
//! `UPDATE_CANONICAL=1`). Pure text — no executor, no dataset — so it runs in the
//! plain CPU CI tier. The taxonomy + multipliers live in `common/cost_model.rs`.
//!
//! Also guards the byte-identity invariant the refactor must preserve: at today's
//! all-1.0 multipliers the `.cost.txt` total equals `Σ output_bytes` over the
//! `.cpu.txt` tree (the value the old `peacockdb_cost=` footer carried).
//!
//! `COST_FILTER=<substr>` restricts the run to `.cpu.txt` files whose name
//! contains `<substr>` (used to regenerate a single golden at the review gate).
#[macro_use]
mod common;

use std::path::PathBuf;

use common::cost_model::CostModel;

/// Σ of every `output_bytes=` value in a `.cpu.txt` body (the old footer total).
fn sum_output_bytes(cpu_text: &str) -> u64 {
    const KEY: &str = "output_bytes=";
    cpu_text
        .lines()
        .filter_map(|l| l.find(KEY).map(|p| &l[p + KEY.len()..]))
        .map(|tail| tail.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse::<u64>().unwrap())
        .sum()
}

fn cost_total(cost_text: &str) -> u64 {
    const KEY: &str = "peacockdb_cost=";
    let line = cost_text.lines().find(|l| l.starts_with(KEY)).expect("cost text missing total footer");
    line[KEY.len()..].parse().unwrap()
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
        .flat_map(|d| std::fs::read_dir(d).unwrap_or_else(|e| panic!("read_dir {}: {e}", d.display())))
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_str().map(|s| s.ends_with(".cpu.txt")).unwrap_or(false))
        .filter(|p| match &filter {
            Some(f) => p.file_name().and_then(|n| n.to_str()).map(|n| n.contains(f.as_str())).unwrap_or(false),
            None => true,
        })
        .collect();
    cpu_files.sort();
    assert!(!cpu_files.is_empty(), "no .cpu.txt goldens matched (filter {filter:?})");

    let mut mismatches: Vec<String> = Vec::new();
    for cpu_path in &cpu_files {
        let name = cpu_path.file_name().and_then(|s| s.to_str()).unwrap();
        let cpu_text = std::fs::read_to_string(cpu_path).unwrap();
        let actual = model.cost_text_from_cpu(&cpu_text, name);

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
            Ok(golden) if golden.trim_end() == actual => {}
            Ok(_) => mismatches.push(format!("{name}: .cost.txt does not match (run with UPDATE_CANONICAL=1)")),
            Err(_) => mismatches.push(format!("{}: missing (run with UPDATE_CANONICAL=1)", cost_path.display())),
        }
    }
    assert!(mismatches.is_empty(), "cost golden mismatches:\n{}", mismatches.join("\n"));
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
