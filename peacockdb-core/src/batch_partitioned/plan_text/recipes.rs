//! The `--- recipes ---` section: what each node asks of the device, under the same tree
//! the plan renders, so a line reads against the node above it.

use std::fmt::Write as _;

use super::super::node::GpuNode;
use super::super::nodes::category_of;
use super::super::recipe::RecipePlan;

/// One line per node, in plan order. A node that makes no ABI call says `none` rather
/// than nothing: "this node touches no device" and "the pass missed it" have to look
/// different, and a blank line makes them look the same.
///
/// Which is why the coverage check comes first: a plan shorter than the tree renders its
/// tail as `none` too, and that reads as a forwarder rather than as the gap it is.
pub fn render_plan_recipes(root: &dyn GpuNode, plan: &RecipePlan) -> String {
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
    render_recipe_node(root, 0, &mut Sequence::default(), plan, &mut text);
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
    text: &mut String,
) {
    let mut lines = Vec::new();
    for child in node.children() {
        let mut child_text = String::new();
        render_recipe_node(child, depth + 1, sequence, plan, &mut child_text);
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
    for line in lines {
        text.push_str(&line);
    }
}
