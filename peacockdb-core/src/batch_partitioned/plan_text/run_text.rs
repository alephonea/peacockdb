//! The execution golden's text: the plan tree, and what the run produced at each node.
//!
//! The node line is the plan golden's minus the declared schema — a plan fact, recorded
//! once beside the plan — plus the two totals both mode families carry. Under it goes one
//! line holding what the node consumed and the size of every batch it emitted.
//!
//! Its tests are the driver's, in `driver/tests/render.rs`: rendering a report takes a run,
//! and the mock backend that makes one deterministic lives there.
//!
//! The tree is walked through the driver's own [`PlanIndex`], not through a second walk of
//! the same shape: the report is addressed by that index, and two walks agreeing is a
//! coincidence to be avoided rather than relied on.

use std::fmt::Write as _;

use super::{join_parts, node_line_parts};
use crate::batch_partitioned::driver::RunReport;
use crate::batch_partitioned::driver::index::{PlanIndex, ROOT};
use crate::batch_partitioned::node::GpuNode;

/// `root` as it ran: the early-exit marker, then one node per line with its per-batch
/// record beneath it.
pub fn render_run(root: &dyn GpuNode, report: &RunReport) -> String {
    let index = PlanIndex::build(root).expect("the tree the run was made over indexes");
    let mut text = format!("early_exit={}\n", early_exit(&index, report));
    render_node(&index, report, ROOT, 0, &mut text);
    text
}

/// Which limits were satisfied, by name and post-order address — the address because that
/// is what both engines number a node by, and the name because nobody reads an address.
/// `none` where the run drained, stated rather than left to an absence, since a smaller
/// number with no cause reads exactly like a plan that produced less.
fn early_exit(index: &PlanIndex<'_>, report: &RunReport) -> String {
    if report.satisfied.is_empty() {
        return "none".to_string();
    }
    let named: Vec<String> = report
        .satisfied
        .iter()
        .map(|node| {
            format!(
                "{}@{}",
                index.nodes[*node].node.name(),
                index.nodes[*node].post_order
            )
        })
        .collect();
    named.join(",")
}

fn render_node(
    index: &PlanIndex<'_>,
    report: &RunReport,
    node: usize,
    depth: usize,
    text: &mut String,
) {
    let batches = &report.emitted[node];
    let mut parts = node_line_parts(index.nodes[node].node);
    parts.push(format!(
        "output_rows={}",
        batches
            .iter()
            .flatten()
            .map(|batch| batch.rows)
            .sum::<u64>()
    ));
    parts.push(format!(
        "output_bytes={}",
        batches
            .iter()
            .flatten()
            .map(|batch| batch.bytes)
            .sum::<usize>()
    ));
    let indent = "  ".repeat(depth);
    let _ = writeln!(text, "{indent}{}", join_parts(parts));
    // Lanes outermost and batches within, which is `partition_groups`' own nesting: on a
    // source, element i of lane j is one batch in all three lists and in the mapping.
    let _ = write!(
        text,
        "{indent}  in_rows={} batch_rows={} batch_bytes={}",
        lanes_of(&report.consumed[node], |rows| *rows),
        lanes_of(batches, |batch| batch.rows),
        lanes_of(batches, |batch| batch.bytes as u64),
    );
    // Both are early-exit quantities, so both are absent from every node of every run that
    // drained rather than a zero on each of them.
    let abandoned = &report.abandoned[node];
    if abandoned.iter().any(|rows| *rows > 0) {
        let _ = write!(text, " abandoned={}", numbers(abandoned));
    }
    if report.rows_skipped[node] > 0 {
        let _ = write!(text, " rows_skipped={}", report.rows_skipped[node]);
    }
    text.push('\n');
    for child in &index.nodes[node].children {
        render_node(index, report, *child, depth + 1, text);
    }
}

/// `[[a,b],[c]]` — the bracket style `partition_groups` renders in, so the two nestings a
/// reader compares index for index also read alike.
fn lanes_of<T>(lanes: &[Vec<T>], of: impl Fn(&T) -> u64) -> String {
    let rendered: Vec<String> = lanes
        .iter()
        .map(|lane| numbers(&lane.iter().map(&of).collect::<Vec<u64>>()))
        .collect();
    format!("[{}]", rendered.join(","))
}

fn numbers(values: &[u64]) -> String {
    let rendered: Vec<String> = values.iter().map(u64::to_string).collect();
    format!("[{}]", rendered.join(","))
}
