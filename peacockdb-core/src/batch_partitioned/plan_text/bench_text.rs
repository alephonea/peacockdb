//! The benchmark record's text: the plan tree, and what each node's calls cost.
//!
//! A sibling of [`run_text`](super::run_text) rather than a mode of it. The two answer
//! different questions of the same tree — what it produced, what it took — and they are
//! written by different runs into different files. What they share is the node line, and
//! that is shared by calling one function rather than by one renderer growing a flag.
//!
//! Rows and bytes are deliberately absent. They are a correctness fact, asserted in the
//! `.cpu.txt` golden and carried per call in `records.tsv`; repeating them here would give
//! one fact two homes and make this file diff on a data change that has nothing to do with
//! time.
//!
//! Its tests are the driver's, in `driver/tests/render.rs`, for the reason `run_text`'s
//! are: rendering takes a run, and the mock backend that makes a deterministic one is
//! there.

use std::fmt::Write as _;

use super::{join_parts, node_line_parts};
use crate::batch_partitioned::driver::index::{PlanIndex, ROOT};
use crate::batch_partitioned::driver::{Measurements, node_measured};
use crate::batch_partitioned::node::GpuNode;

/// What a call reports when the device recorded no region for it — a call that opened
/// none, or a run nobody measured.
///
/// Not `0`: the call took time, and a zero would say it did not. The distinction is the
/// same one `AbiCalls` carries from the backend up, and this is where it reaches a reader.
const UNMEASURED: &str = "-";

/// `root` as it was timed: one node per line with the microseconds its calls spent on the
/// device beneath it.
pub fn render_timings(root: &dyn GpuNode, times: &Measurements) -> String {
    let index = PlanIndex::build(root).expect("the tree the run was made over indexes");
    let mut text = String::new();
    render_node(&index, times, ROOT, 0, &mut text);
    text
}

fn render_node(
    index: &PlanIndex<'_>,
    times: &Measurements,
    node: usize,
    depth: usize,
    text: &mut String,
) {
    let indent = "  ".repeat(depth);
    let _ = writeln!(
        text,
        "{indent}{}",
        join_parts(node_line_parts(index.nodes[node].node))
    );
    // Lanes outermost and calls within, the nesting `batch_rows` and `partition_groups`
    // use, so a reader comparing this file against the execution golden compares index for
    // index rather than by counting brackets.
    let _ = writeln!(
        text,
        "{indent}  time_us={} total_us={}",
        lanes_of(times, node),
        total_of(times, node),
    );
    for child in &index.nodes[node].children {
        render_node(index, times, *child, depth + 1, text);
    }
}

/// `[[22,37,40],[55,-,30]]` — one list per driving lane, one entry per call that lane made.
///
/// A lane the node was never driven on renders `[]` rather than being left out: a routing
/// node the driver answers itself has as many empty lanes as it has, and dropping them
/// would make its four lanes read as none.
fn lanes_of(times: &Measurements, node: usize) -> String {
    let rendered: Vec<String> = times
        .lanes(node)
        .iter()
        .map(|lane| {
            let calls: Vec<String> = lane
                .iter()
                .map(|call| match call {
                    Some(time) if time.regions > 0 => time.device_us.to_string(),
                    _ => UNMEASURED.to_string(),
                })
                .collect();
            format!("[{}]", calls.join(","))
        })
        .collect();
    format!("[{}]", rendered.join(","))
}

/// The node's whole device time, over every lane and call. `-` where nothing was measured,
/// which is not the same statement as a node that cost nothing.
fn total_of(times: &Measurements, node: usize) -> String {
    match node_measured(times, node) {
        Some(time) if time.regions > 0 => time.device_us.to_string(),
        _ => UNMEASURED.to_string(),
    }
}
