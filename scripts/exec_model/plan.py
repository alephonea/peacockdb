"""Plan indexing: heights, left-to-right order, and the structural rules the driver relies on.

Two facts the scheduler is built on, both computed here once:

- **height** — distance to the root, root = 0. The driver always runs the runnable node
  with the *smallest* height, which is what pushes a batch as far up the tree as it can
  go before any new batch is produced below it.
- **order** — pre-order (left-to-right) index, the tie-break among equal heights. In a
  tree, pre-order visits same-depth nodes left to right, so "smallest order" is exactly
  "leftmost".

Both are pure functions of the tree, so scheduling is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PlanError
from .layout import BatchLayout, NodeKind
from .node import ExecutorCategory, GpuNode, NodeExecutors


@dataclass(frozen=True)
class PlanNodeInfo:
    id: int
    node: GpuNode
    executors: NodeExecutors
    height: int
    order: int
    parent: int | None
    child_slot: int | None
    children: tuple[int, ...]
    n_lanes: int

    @property
    def category(self) -> ExecutorCategory:
        return self.executors.category

    def __str__(self) -> str:
        return f"{self.node.name()}#{self.id}"


class Plan:
    """An indexed, validated plan tree."""

    def __init__(self, nodes: list[PlanNodeInfo], root: int):
        self.nodes = nodes
        self.root = root

    def __getitem__(self, node_id: int) -> PlanNodeInfo:
        return self.nodes[node_id]

    def __len__(self) -> int:
        return len(self.nodes)

    def child(self, node_id: int, slot: int) -> PlanNodeInfo:
        return self.nodes[self.nodes[node_id].children[slot]]

    def validate(self) -> None:
        for info in self.nodes:
            _validate_kind(self, info)
            _validate_structure(self, info)
            info.node.validate_schemas_and_partitions()

    @classmethod
    def build(cls, root: GpuNode) -> "Plan":
        nodes: list[PlanNodeInfo] = []
        counter = [0]

        def walk(node: GpuNode, height: int, parent: int | None, slot: int | None) -> int:
            node_id = len(nodes)
            order = counter[0]
            counter[0] += 1
            nodes.append(None)  # placeholder; children need the id first
            child_ids = tuple(
                walk(child, height + 1, node_id, i) for i, child in enumerate(node.children())
            )
            executors = node.make_executors()
            nodes[node_id] = PlanNodeInfo(
                id=node_id,
                node=node,
                executors=executors,
                height=height,
                order=order,
                parent=parent,
                child_slot=slot,
                children=child_ids,
                n_lanes=_lane_count(node, [nodes[c] for c in child_ids]),
            )
            return node_id

        root_id = walk(root, 0, None, None)
        plan = cls(nodes, root_id)
        plan.validate()
        return plan


def _lane_count(node: GpuNode, children: list[PlanNodeInfo]) -> int:
    """A sink declares no layout, so it inherits its child's lane count."""
    layout = node.output_partitions()
    if layout is not None:
        return layout.n
    if node.kind() is not NodeKind.SINK:
        raise PlanError(f"{node.name()}: only a sink may omit output_partitions()")
    if len(children) != 1:
        raise PlanError(f"{node.name()}: a sink takes exactly one child")
    return children[0].n_lanes


#: Categories that legitimately take two children. Everything else is a pipe.
_BINARY = frozenset({ExecutorCategory.JOIN, ExecutorCategory.BATCH_FORWARDER})


def _validate_structure(plan: Plan, info: PlanNodeInfo) -> None:
    category = info.category
    n_children = len(info.children)

    if n_children > 2:
        raise PlanError(f"{info}: {n_children} children — the plan tree is binary")
    if n_children == 2 and category not in _BINARY:
        raise PlanError(f"{info}: {category.value} takes at most one child")

    if category is ExecutorCategory.SOURCE:
        if n_children:
            raise PlanError(f"{info}: a source takes no children")
        return

    if n_children == 0:
        raise PlanError(f"{info}: {category.value} needs a child")

    if category in (ExecutorCategory.EXEC, ExecutorCategory.BATCH_ACCUMULATOR):
        child = plan.child(info.id, 0)
        if child.n_lanes != info.n_lanes:
            raise PlanError(
                f"{info}: {category.value} is 1:1 per lane but child has "
                f"{child.n_lanes} lanes against {info.n_lanes}"
            )

    elif category is ExecutorCategory.PARTITION_EMITTER:
        child = plan.child(info.id, 0)
        if child.n_lanes != 1:
            raise PlanError(f"{info}: an emitter scatters 1 → N, child has {child.n_lanes} lanes")

    elif category is ExecutorCategory.PARTITION_ACCUMULATOR:
        if info.n_lanes != 1:
            raise PlanError(f"{info}: a partition accumulator outputs 1 lane, not {info.n_lanes}")

    elif category is ExecutorCategory.BATCH_FORWARDER:
        forwarder = info.executors.forwarder
        if forwarder.out_lanes() != info.n_lanes:
            raise PlanError(
                f"{info}: forwarder declares {forwarder.out_lanes()} lanes, "
                f"layout declares {info.n_lanes}"
            )
        for out_lane in range(info.n_lanes):
            for child_idx, child_lane in forwarder.sources_of(out_lane):
                if child_idx >= n_children:
                    raise PlanError(f"{info}: sources_of({out_lane}) names child {child_idx}")
                if child_lane >= plan.child(info.id, child_idx).n_lanes:
                    raise PlanError(
                        f"{info}: sources_of({out_lane}) names lane {child_lane} of a "
                        f"{plan.child(info.id, child_idx).n_lanes}-lane child"
                    )

    elif category is ExecutorCategory.JOIN:
        if n_children != 2:
            raise PlanError(f"{info}: a join takes exactly two children")
        build, probe = plan.child(info.id, 0), plan.child(info.id, 1)
        # The build side is always left. That orientation is what removes the wait:
        # among equal heights the driver picks the leftmost node, so the build subtree
        # drains before the probe subtree can starve the join.
        build_layout = build.node.output_partitions()
        if build_layout.batch_layout is not BatchLayout.SINGLE_BATCH:
            raise PlanError(
                f"{info}: the build side (left child {build}) must declare SingleBatch — "
                "the planner inserts GpuCoalesceAllBatches"
            )
        if not (build.n_lanes == probe.n_lanes == info.n_lanes):
            raise PlanError(
                f"{info}: join lanes must agree — build {build.n_lanes}, "
                f"probe {probe.n_lanes}, output {info.n_lanes}"
            )


def _validate_kind(plan: Plan, info: PlanNodeInfo) -> None:
    kind = info.node.kind()
    if kind is NodeKind.SINK and info.id != plan.root:
        raise PlanError(f"{info}: a sink may only be the root")
    if kind is NodeKind.SOURCE and info.category is not ExecutorCategory.SOURCE:
        raise PlanError(f"{info}: NodeKind.SOURCE requires the Source executor category")


