//! The `--- memory ---` section: the estimator's per-node figures under the same tree the
//! plan renders, so a line reads against the node above it.

use std::fmt::Write as _;

use super::super::estimator::MemoryModel;
use super::super::node::GpuNode;
use super::super::nodes::{NodeRef, as_node_ref};

/// The sibling memory golden: the same tree, carrying what the estimator derived rather
/// than what the planner decided. Sources also print the batch size they were given, which
/// is the one number the partitioner reads back.
pub fn render_plan_memory(root: &dyn GpuNode, model: &MemoryModel) -> String {
    let mut text = format!(
        "budget={}, accumulators={}, certain={}\n",
        model.budget, model.accumulator_bytes, model.certain_accumulator_bytes
    );
    render_memory_node(root, 0, &mut Sequence::default(), model, &mut text);
    text
}

/// Canonical post-order, the order the estimator indexes by: children left to right, then
/// the node.
#[derive(Default)]
struct Sequence {
    next: usize,
}

fn render_memory_node(
    node: &dyn GpuNode,
    depth: usize,
    sequence: &mut Sequence,
    model: &MemoryModel,
    text: &mut String,
) {
    let children: Vec<&dyn GpuNode> = node.children();
    let mut lines = Vec::new();
    for child in children {
        let mut child_text = String::new();
        render_memory_node(child, depth + 1, sequence, model, &mut child_text);
        lines.push(child_text);
    }
    let seq = sequence.next;
    sequence.next += 1;

    let mut fields = vec![format!(
        "estimated_max_resident_size={}",
        model.resident.get(seq).copied().unwrap_or(0)
    )];
    if let Some(source) = model.sources.iter().find(|source| source.seq == seq) {
        // What the source holds is what makes the rest of the line legible: it is the cap
        // on the target, so a reader sees whether the target is the source or the budget
        // talking, and it is what the small-table threshold compared to decide the lanes.
        if let NodeRef::LoadParquet(load) = as_node_ref(node) {
            fields.push(format!("source_bytes={}", load.bytes()));
        }
        fields.push(format!(
            "target_batch_bytes={}, amplification={:.1}",
            source.target_batch_bytes, source.amplification
        ));
    }
    let _ = writeln!(
        text,
        "{}{}: {}",
        "  ".repeat(depth),
        node.name(),
        fields.join(", ")
    );
    for line in lines {
        text.push_str(&line);
    }
}
