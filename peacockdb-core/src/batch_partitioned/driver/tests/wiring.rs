//! What the driver hands a backend when it builds an executor set.
//!
//! The GPU backend looks a recipe up by the post-order it is given here, and a wrong
//! number reaches it as another node's recipe rather than as an error — so the claim is
//! checked against a walk written in this file, not against the one under test.

use super::super::mock::{Script, address_of, spec};
use super::super::plans::*;
use super::*;

/// The tree children-first, which is what `attach_recipes` numbers by.
fn children_first<'a>(node: &'a dyn GpuNode, into: &mut Vec<usize>) {
    for child in node.children() {
        children_first(child, into);
    }
    into.push(address_of(node));
}

/// Three call sites hand a backend a post-order, and they read it from different places:
/// a lane site carries one, and the forwarder wiring and the cross-lane build each take it
/// off the index themselves. So the plan reaches all three — a scatter and a sorted merge
/// under the probe side, with a routing merge between them.
#[test]
fn an_executor_is_built_with_its_own_nodes_post_order() {
    // A join, so the walk has a node with two children: every order agrees on a chain.
    let inner = join(coalesce_all(source("a", 1)), filter(source("b", 1)));
    let probe = merge_sorted(merge(emit(source("c", 1), 4)));
    let plan = unload(join(coalesce_all(inner), probe));
    let script = Script::default()
        .source("a", vec![vec![spec(4, 32)]])
        .source("b", vec![vec![spec(6, 48)]])
        .source("c", vec![vec![spec(8, 64)]]);
    run(plan.as_ref(), &script);

    let mut expected = Vec::new();
    children_first(plan.as_ref(), &mut expected);
    let built = script.built_at.lock().expect("the recording").clone();
    assert_eq!(
        built.len(),
        expected.len(),
        "every node of the plan built an executor set, and no node built two"
    );
    for (node, position) in &built {
        assert_eq!(
            expected.get(*position).copied(),
            Some(*node),
            "the executor set built for one node was handed post-order {position}, which \
             is where a children-first walk puts another"
        );
    }
    let mut positions: Vec<usize> = built.iter().map(|(_, position)| *position).collect();
    positions.sort_unstable();
    assert_eq!(
        positions,
        (0..expected.len()).collect::<Vec<usize>>(),
        "each node was handed a position of its own"
    );
}
