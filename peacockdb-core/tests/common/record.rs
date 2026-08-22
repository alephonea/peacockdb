//! The cost-model calibration record: one line per timed region, in the one format
//! both measurement sources write.
//!
//! The fit needs peacockdb sf1 rows and bare-cuDF sf40 rows on the same axes, so the
//! format is defined once here and implemented twice — this side from the plan tree
//! and [`NodeMemoryStats`], the sf40 side by hand in the C++ tests, which have
//! no plan to walk. Sharing the format rather than the code is forced: the two
//! sources agree on what a row MEANS and on nothing else.
//!
//! TSV, because `build_profile` and `allocator` contain spaces and commas and are
//! the fields a reader most needs verbatim.
//!
//! `hbm_bytes` is deliberately absent: it comes from Nsight and is joined in
//! later on `(query, node_seq)`.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use peacockdb_core::cpu_executor::NodeMemoryStats;

use super::cost_model::CostModel;

/// Env var naming the file rows are APPENDED to. Unset ⇒ no record is written, which
/// is why every caller can emit unconditionally.
pub const RECORD_PATH_ENV: &str = "PEACOCK_RECORD_PATH";

pub const COLUMNS: &[&str] = &[
    "source",
    "dataset",
    "sf",
    "query",
    "label",
    "node_seq",
    "node_type",
    "category",
    "partitions",
    "partition",
    "in_rows",
    "in_bytes",
    "out_rows",
    "out_bytes",
    "cuda_bytes",
    "peacock_host_us",
    "cudf_host_us",
    "device_us",
    "wall_us",
    "timing_mode",
    "build_profile",
    "allocator",
];

/// What a row cannot be recovered from: which engine produced it, over what data, and
/// under the three settings that change what its microseconds mean.
pub struct RunMeta<'a> {
    /// `peacockdb` here; `cudf` from the sf40 side. The fit's `const_peacock` is the
    /// difference between the two, so this is the column it keys on.
    pub source: &'a str,
    pub dataset: &'a str,
    pub sf: &'a str,
    pub query: &'a str,
    /// `<mode>-<tp>-<tier>`, as on the goldens: the same query at the same device is a
    /// different plan full-table vs partitioned.
    pub label: &'a str,
    pub timing_mode: &'a str,
    pub build_profile: &'a str,
    pub allocator: &'a str,
}

/// One row per timed region — per (node, output partition), NOT per node. Three of the
/// four `execute_node` branches loop over partitions, so a node line would average away
/// the structure the per-partition times have; the repartition prologue charged to p0
/// is the clearest case.
///
/// Post-order, so `node_seq` indexes `stats` directly and matches the order the
/// `.cpu.txt` and `.benchmark.txt` trees are built in.
pub fn record_rows(
    plan: &Arc<dyn ExecutionPlan>,
    stats: &[NodeMemoryStats],
    meta: &RunMeta<'_>,
    model: &CostModel,
) -> Vec<String> {
    struct Node<'a> {
        stat: &'a NodeMemoryStats,
        plan: &'a Arc<dyn ExecutionPlan>,
        children: Vec<Node<'a>>,
        seq: usize,
    }
    fn collect<'a>(
        plan: &'a Arc<dyn ExecutionPlan>,
        stats: &'a [NodeMemoryStats],
        idx: &mut usize,
    ) -> Node<'a> {
        let children: Vec<Node<'a>> =
            plan.children().iter().map(|c| collect(c, stats, idx)).collect();
        let seq = *idx;
        *idx += 1;
        Node { stat: &stats[seq], plan, children, seq }
    }

    /// Input bytes/rows for output partition `k` of a node with `n` of them, summed over
    /// children — unlike the `.cpu.txt` golden, which renders the first child's only. A
    /// join's cost depends on both sides, and this column exists to be regressed on.
    ///
    /// Partition k of a child maps to partition k of the node only when the two counts
    /// agree. Where they do not, the child is charged wholly to k=0: a coalesce (8→1)
    /// would otherwise report one eighth of what it concatenates, and a repartition
    /// (1→8) already bills its shared prologue to p0. Either way Σ over k is the child's
    /// total, which is the invariant the fit needs.
    fn input_for(node: &Node, k: usize, n: usize) -> (usize, usize) {
        let mut rows = 0;
        let mut bytes = 0;
        for c in &node.children {
            if c.stat.part_stats.len() == n {
                rows += c.stat.part_stats[k].out_rows;
                bytes += c.stat.part_stats[k].out_bytes;
            } else if k == 0 {
                rows += c.stat.row_count;
                bytes += c.stat.output_bytes;
            }
        }
        (rows, bytes)
    }

    fn walk(node: &Node, meta: &RunMeta<'_>, model: &CostModel, out: &mut Vec<String>) {
        let node_type = node.plan.name();
        // Panics rather than emitting an untagged row: an unbinned node type is a hole
        // in the taxonomy, and a record that quietly drops it is one the fit cannot
        // notice is incomplete.
        let category = model.category_name_of(node_type).unwrap_or_else(|| {
            panic!("{} node type '{node_type}' is not in the cost taxonomy", meta.query)
        });
        let n = node.stat.part_stats.len().max(1);
        for k in 0..n {
            let (out_rows, out_bytes, setup, submit, device) = match node.stat.part_stats.get(k) {
                Some(ps) => {
                    (ps.out_rows, ps.out_bytes, ps.host_setup_us, ps.host_submit_us, ps.device_us)
                }
                // N=1: part_stats is empty by the golden's convention, and the node
                // totals ARE the single partition's.
                None => (
                    node.stat.row_count,
                    node.stat.output_bytes,
                    node.stat.host_setup_us,
                    node.stat.host_submit_us,
                    node.stat.device_us,
                ),
            };
            let (in_rows, in_bytes) = input_for(node, k, n);
            out.push(
                [
                    meta.source.to_string(),
                    meta.dataset.to_string(),
                    meta.sf.to_string(),
                    meta.query.to_string(),
                    meta.label.to_string(),
                    node.seq.to_string(),
                    node_type.to_string(),
                    category.to_string(),
                    n.to_string(),
                    k.to_string(),
                    in_rows.to_string(),
                    in_bytes.to_string(),
                    out_rows.to_string(),
                    out_bytes.to_string(),
                    // The regressor. Equal to out_bytes on this source because that is
                    // what `cost_model.conf` charges a node's category; the sf40 side
                    // fills it from its hand-written call→category mapping, where the
                    // two need not coincide.
                    out_bytes.to_string(),
                    setup.to_string(),
                    submit.to_string(),
                    device.to_string(),
                    (setup + submit).to_string(),
                    meta.timing_mode.to_string(),
                    meta.build_profile.to_string(),
                    meta.allocator.to_string(),
                ]
                .join("\t"),
            );
        }
        for child in &node.children {
            walk(child, meta, model, out);
        }
    }

    let root = collect(plan, stats, &mut 0);
    let mut rows = Vec::new();
    walk(&root, meta, model, &mut rows);
    rows
}

/// The `#` preamble, written once per file. A record has to be readable without this
/// source, and every line below is one a reader would otherwise guess wrong.
pub fn record_header() -> String {
    format!("{HEADER_NOTES}\n{}", COLUMNS.join("\t"))
}

const HEADER_NOTES: &str = "\
# peacockdb cost-model calibration record. One row per timed region — per
# (node, output partition), not per node: the partitioned branches time each partition
# separately, and a hash repartition bills its shared concat+scatter prologue to p0.
# source = which engine produced the row. peacockdb pays a host prologue that bare cuDF
#   has no analogue for; that difference is what const_peacock is fitted to, so rows
#   from the two sources are not interchangeable.
# peacock_host_us = host time before the region's first device touch.
# cudf_host_us = host time from that touch to the end of the region. Under
#   timing_mode=sync it CONTAINS the device execution and device_us is 0; under events
#   it does not. Not launch cost in either mode — cuDF and rmm synchronize internally.
# device_us = between the region's CUDA events. events mode only.
# wall_us = peacock_host_us + cudf_host_us, the host side of the region.
# cuda_bytes = the bytes this region's category is charged for; the regressor. Equal to
#   out_bytes on the peacockdb source, the mapped call's bytes on the bare-cuDF one.
# in_rows/in_bytes = summed over ALL children, unlike the .cpu.txt golden, which
#   renders the first child's.
# hbm_bytes is NOT here: it comes from Nsight and joins on (query, node_seq).
# timing_mode, build_profile and allocator each change what the microseconds MEAN. Rows
#   disagreeing on any of them are not comparable — hence columns, not a run heading.";

/// Append this run's rows to `$PEACOCK_RECORD_PATH`, or do nothing if it is unset.
///
/// Appends rather than writes: a collection run is many queries in one process, and
/// one file per run is what the fit reads. The header goes in only when the file is
/// created, so concatenating runs stays a valid file.
pub fn append_records(
    plan: &Arc<dyn ExecutionPlan>,
    stats: &[NodeMemoryStats],
    meta: &RunMeta<'_>,
) {
    let Ok(path) = std::env::var(RECORD_PATH_ENV) else { return };
    let path = PathBuf::from(path);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let fresh = std::fs::metadata(&path).map(|m| m.len() == 0).unwrap_or(true);
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("cannot open {} for records: {e}", path.display()));
    let mut text = String::new();
    if fresh {
        text.push_str(&record_header());
        text.push('\n');
    }
    for row in record_rows(plan, stats, meta, &CostModel::load()) {
        text.push_str(&row);
        text.push('\n');
    }
    f.write_all(text.as_bytes())
        .unwrap_or_else(|e| panic!("cannot append records to {}: {e}", path.display()));
}
