//! The `--- recipes ---` section: what each node asks of the device, under the same tree
//! the plan renders, so a line reads against the node above it.

use std::fmt::Write as _;

use crate::generated::gpu_plan_generated::peacock::plan as fb;

use super::super::node::GpuNode;
use super::super::nodes::category_of;
use super::super::recipe::RecipePlan;
use super::fb_text;

/// Whether the section prints what each call passes the executor, or only which kernel it
/// addresses. One renderer either way: two would drift, and the ten mode goldens and the
/// payload golden would then disagree about a plan neither of them changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Payloads {
    Omitted,
    Shown,
}

/// One line per node, in plan order. A node that makes no ABI call says `none` rather
/// than nothing: "this node touches no device" and "the pass missed it" have to look
/// different, and a blank line makes them look the same.
///
/// Which is why the coverage check comes first: a plan shorter than the tree renders its
/// tail as `none` too, and that reads as a forwarder rather than as the gap it is.
pub fn render_plan_recipes(root: &dyn GpuNode, plan: &RecipePlan, payloads: Payloads) -> String {
    let nodes = count_nodes(root);
    assert_eq!(
        plan.nodes(),
        nodes,
        "the recipe plan covers {} nodes and this tree has {nodes} — the pass and the \
         renderer disagree about the tree, and every node past the end would render as \
         `none`, which is what a node that makes no call says",
        plan.nodes()
    );
    let mut text = String::new();
    let buffer = (payloads == Payloads::Shown).then(|| {
        // The same depth the C++ verifier allows (`NodeSession`): a recipe plan is a
        // chain, one node per seq, so it is as deep as it is long — where a legacy plan is
        // a tree and never near the default 64.
        let options = flatbuffers::VerifierOptions {
            max_depth: 1024,
            ..Default::default()
        };
        flatbuffers::root_with_opts::<fb::GpuPlan>(&options, plan.bytes())
            .expect("the plan we just wrote")
    });
    render_recipe_node(root, 0, &mut Sequence::default(), plan, buffer.as_ref(), &mut text);
    text
}

fn count_nodes(node: &dyn GpuNode) -> usize {
    1 + node.children().into_iter().map(count_nodes).sum::<usize>()
}

/// How many lanes call, which is the multiplier on every `per batch` below it: one per
/// lane where the executor is lane-scoped, and exactly one at the cross-lane points. An
/// emitter scatters one lane into N and a partition accumulator merges N into one, so
/// neither calls per lane whatever its output declares — which is why the section spells
/// this `calling_lanes` rather than reusing the tree's `lanes`.
fn instances(node: &dyn GpuNode) -> usize {
    if !category_of(node).is_lane_scoped() {
        return 1;
    }
    match node.kind().layout() {
        Some(layout) => layout.n,
        // A sink declares no layout of its own: it calls once per lane of its input.
        None => {
            let input = node
                .children()
                .into_iter()
                .next()
                .expect("a sink has an input");
            input.kind().layout().expect("a sink cannot be an input").n
        }
    }
}

/// The node's post-order position, which is what the recipe plan is indexed by — the same
/// order the memory section numbers in, and not a `Seq`.
#[derive(Default)]
struct Sequence {
    next: usize,
}

fn render_recipe_node(
    node: &dyn GpuNode,
    depth: usize,
    sequence: &mut Sequence,
    plan: &RecipePlan,
    buffer: Option<&fb::GpuPlan<'_>>,
    text: &mut String,
) {
    let mut lines = Vec::new();
    for child in node.children() {
        let mut child_text = String::new();
        render_recipe_node(child, depth + 1, sequence, plan, buffer, &mut child_text);
        lines.push(child_text);
    }
    let position = sequence.next;
    sequence.next += 1;

    // `calling_lanes` is how many lanes make the call, not how many the node's output
    // declares: an emitter reads one lane and scatters into four, so it calls from one
    // while its tree line says four. On the line with the calls and nowhere else — a node
    // that makes none has no number to give.
    let line = match plan.get(position) {
        Some(recipe) => format!("calling_lanes={}, {recipe}", instances(node)),
        None => "none".to_string(),
    };
    let _ = writeln!(text, "{}{}: {line}", "  ".repeat(depth), node.name());
    if let (Some(buffer), Some(recipe)) = (buffer, plan.get(position)) {
        // Under the call it belongs to, indented past the tree, so a long payload reads
        // as this node's rather than as the next line of the plan.
        let indent = format!("{}    ", "  ".repeat(depth));
        for call in &recipe.calls {
            let Some((seq, kind)) = call.target else {
                continue;
            };
            let _ = writeln!(text, "{indent}#{seq} {kind}");
            // Expect rather than default: a seq with no node is the numbering outrunning
            // the tree, and printing nothing there would render a plan whose every later
            // seq addresses the wrong node as if it were fine.
            let node = fb_text::node_at(buffer, seq)
                .unwrap_or_else(|| panic!("seq #{seq} is published as {kind} and the plan has no node there"));
            assert_eq!(
                node.node_type(),
                kind.wire_kind(),
                "seq #{seq} is published as {kind} and holds a different kind"
            );
            text.push_str(&fb_text::payload_text(&node, &format!("{indent}  ")));
        }
    }
    for line in lines {
        text.push_str(&line);
    }
}
