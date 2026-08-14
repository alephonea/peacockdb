"""Mock trait implementations and plan builders.

The driver tests drive *these*, never real operators: the strategy under test is which
node runs when, so the executors only need to record what they were asked to do and hand
back batches whose tags say where they came from.
"""

from __future__ import annotations

from typing import Callable, Iterable

from ..batch import Batch, CallStats
from ..executors import (
    BatchAccumulatorExecutor,
    ExecExecutor,
    Executor,
    JoinExecutor,
    LaneEvent,
    PartitionAccumulatorExecutor,
    PartitionEmitterExecutor,
    SourceExecutor,
)
from ..forwarder import (
    BatchForwarder,
    InterleaveForwarder,
    MergePartitionsForwarder,
    UnionForwarder,
)
from ..layout import BatchLayout, KeyDistribution, NodeKind, PartitionLayout, SortOrder
from ..node import (
    BackendSelector,
    ExecutorBackends,
    ExecutorCategory,
    GpuNode,
    NodeExecutors,
)

BYTES_PER_ROW = 10


class MockBatch(Batch):
    """Carries a provenance tag. Consumption is one-shot, as it is in Rust."""

    def __init__(self, tag: str, rows: int = 1, nbytes: int | None = None):
        self.tag = tag
        self.rows = rows
        self.nbytes = rows * BYTES_PER_ROW if nbytes is None else nbytes
        self.consumed = False

    def num_rows(self) -> int:
        return self.rows

    def byte_size(self) -> int:
        return self.nbytes

    def consume(self) -> "MockBatch":
        if self.consumed:
            raise AssertionError(f"batch {self.tag} consumed twice")
        self.consumed = True
        return self

    def __repr__(self) -> str:
        return f"MockBatch({self.tag!r}, rows={self.rows})"


class MockExecutor(Executor):
    """Shared accounting behaviour: no state, scratch equal to the input size."""

    def resident_bytes(self) -> int:
        return 0

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return n_bytes


class ScriptedSource(MockExecutor, SourceExecutor):
    def __init__(self, batches: Iterable[MockBatch]):
        self._queue = list(batches)
        self.calls = 0

    def next_batch(self):
        self.calls += 1
        if not self._queue:
            return None
        return self._queue.pop(0), CallStats(scratch_bytes=0)


class MapExec(MockExecutor, ExecExecutor):
    """1:1. `selectivity` scales the row count, so 0.0 models an empty-batch operator."""

    def __init__(self, name: str, selectivity: float = 1.0):
        self.name = name
        self.selectivity = selectivity
        self.calls = 0

    def exec(self, batch: MockBatch):
        self.calls += 1
        batch.consume()
        rows = int(batch.num_rows() * self.selectivity)
        return MockBatch(f"{batch.tag}>{self.name}", rows), CallStats(scratch_bytes=batch.nbytes)


class CollectAccumulator(MockExecutor, BatchAccumulatorExecutor):
    """`GpuCoalesceAllBatches` / `GpuAggregateBatches`: emits exactly one batch at done."""

    def __init__(self, name: str, rows_at_done: Callable[[int], int] | None = None):
        self.name = name
        self.rows_at_done = rows_at_done or (lambda rows: rows)
        self.held_rows = 0
        self.held_bytes = 0
        self.tags: list[str] = []

    def resident_bytes(self) -> int:
        return self.held_bytes

    def accumulate_and_fetch(self, batch: MockBatch):
        batch.consume()
        self.held_rows += batch.num_rows()
        self.held_bytes += batch.byte_size()
        self.tags.append(batch.tag)
        return [], CallStats(scratch_bytes=0)

    def mark_done_and_fetch(self):
        rows = self.rows_at_done(self.held_rows)
        out = MockBatch(f"[{'+'.join(self.tags)}]>{self.name}", rows)
        self.held_rows = self.held_bytes = 0
        return [out], CallStats(scratch_bytes=out.byte_size())


class TargetCoalescer(MockExecutor, BatchAccumulatorExecutor):
    """`GpuCoalesceBatches[target]`: emits mid-stream whenever the threshold is crossed.

    Deferred to #139 in Rust v1, but the drivers must tolerate it at any tree position,
    so the prototype keeps it.
    """

    def __init__(self, name: str, target_rows: int):
        self.name = name
        self.target_rows = target_rows
        self.held_rows = 0
        self.held_bytes = 0
        self.tags: list[str] = []

    def resident_bytes(self) -> int:
        return self.held_bytes

    def _flush(self) -> list[MockBatch]:
        if not self.tags:
            return []
        out = MockBatch(f"[{'+'.join(self.tags)}]>{self.name}", self.held_rows)
        self.held_rows = self.held_bytes = 0
        self.tags = []
        return [out]

    def accumulate_and_fetch(self, batch: MockBatch):
        batch.consume()
        self.held_rows += batch.num_rows()
        self.held_bytes += batch.byte_size()
        self.tags.append(batch.tag)
        outputs = self._flush() if self.held_rows >= self.target_rows else []
        return outputs, CallStats(scratch_bytes=batch.byte_size())

    def mark_done_and_fetch(self):
        return self._flush(), CallStats(scratch_bytes=self.held_bytes)


class MergeSortedPartitions(MockExecutor, PartitionAccumulatorExecutor):
    """Buffers every lane's batches and emits one batch when the last lane says Done."""

    def __init__(self, name: str, n_lanes: int):
        self.name = name
        self.n_lanes = n_lanes
        self.done_lanes: set[int] = set()
        self.held_rows = 0
        self.held_bytes = 0
        self.tags: list[str] = []
        self.events: list[tuple[int, str]] = []

    def resident_bytes(self) -> int:
        return self.held_bytes

    def accumulate_and_fetch(self, partition: int, event: LaneEvent):
        if event.is_done:
            self.events.append((partition, "done"))
            self.done_lanes.add(partition)
            if len(self.done_lanes) < self.n_lanes:
                return [], CallStats(scratch_bytes=0)
            out = MockBatch(f"[{'+'.join(self.tags)}]>{self.name}", self.held_rows)
            self.held_rows = self.held_bytes = 0
            return [out], CallStats(scratch_bytes=out.byte_size())
        event.batch.consume()
        self.events.append((partition, event.batch.tag))
        self.tags.append(event.batch.tag)
        self.held_rows += event.batch.num_rows()
        self.held_bytes += event.batch.byte_size()
        return [], CallStats(scratch_bytes=0)


class EagerMergePartitions(MockExecutor, PartitionAccumulatorExecutor):
    """Emits on every lane event instead of only at the last Done.

    Not a shape any planned node has — it exists so the queue-bound assertion has an
    input that turns it red. A partition accumulator declares one output lane and the
    driver feeds it every input lane in a single step, so k lanes emitting at once puts
    k batches into a 1-lane queue.
    """

    def __init__(self, name: str, n_lanes: int):
        self.name = name
        self.n_lanes = n_lanes
        self.done_lanes: set[int] = set()

    def accumulate_and_fetch(self, partition: int, event: LaneEvent):
        if event.is_done:
            self.done_lanes.add(partition)
            return [], CallStats(scratch_bytes=0)
        event.batch.consume()
        out = MockBatch(f"{event.batch.tag}>{self.name}", event.batch.num_rows())
        return [out], CallStats(scratch_bytes=0)


class ScriptedEmitter(MockExecutor, PartitionEmitterExecutor):
    """`router(batch) -> rows per output lane`; empty lanes come back as 0-row batches."""

    def __init__(self, name: str, n_lanes: int, router: Callable[[MockBatch], list[int]]):
        self.name = name
        self.n_lanes = n_lanes
        self.router = router
        self.calls = 0

    def emit(self, batch: MockBatch):
        self.calls += 1
        batch.consume()
        per_lane = self.router(batch)
        assert len(per_lane) == self.n_lanes
        outputs = [
            MockBatch(f"{batch.tag}>{self.name}.p{lane}", rows)
            for lane, rows in enumerate(per_lane)
        ]
        return outputs, CallStats(scratch_bytes=batch.byte_size())


class RecordingJoin(MockExecutor, JoinExecutor):
    """Inner-join shape by default: probes emit, finish emits nothing."""

    def __init__(self, name: str, emit_on_finish: int = 0):
        self.name = name
        self.emit_on_finish = emit_on_finish
        self.build_tag: str | None = None
        self.build_bytes = 0
        self.probe_tags: list[str] = []

    def resident_bytes(self) -> int:
        return self.build_bytes

    def set_build(self, batch: MockBatch) -> CallStats:
        batch.consume()
        self.build_tag = batch.tag
        self.build_bytes = batch.byte_size()
        return CallStats(scratch_bytes=batch.byte_size())

    def probe_and_fetch(self, batch: MockBatch):
        if self.build_tag is None:
            raise AssertionError(f"{self.name}: probed before set_build")
        batch.consume()
        self.probe_tags.append(batch.tag)
        out = MockBatch(f"({self.build_tag}⋈{batch.tag})>{self.name}", batch.num_rows())
        return [out], CallStats(scratch_bytes=batch.byte_size())

    def finish_and_fetch(self):
        if not self.emit_on_finish:
            return [], CallStats(scratch_bytes=0)
        out = MockBatch(f"({self.build_tag}⋈unmatched)>{self.name}", self.emit_on_finish)
        self.build_bytes = 0
        return [out], CallStats(scratch_bytes=out.byte_size())


class MockSelector(BackendSelector):
    """Always the CPU constructor, and it records every instantiation."""

    def __init__(self):
        self.instantiated: list[tuple[str, int | None]] = []

    def select(self, category: ExecutorCategory, backends: ExecutorBackends, lane: int | None):
        executor = backends.cpu(lane)
        self.instantiated.append((type(executor).__name__, lane))
        return executor


# -- nodes -----------------------------------------------------------------------


class MockNode(GpuNode):
    """One node class for every category; the category is a constructor argument.

    `validate_schemas_and_partitions` is deliberately empty: the prototype's validation
    scope is partitioning and the SingleBatch constraint, and those are checked centrally
    in `plan.py` where the whole tree is visible. Schema checks are out of T0's scope.
    """

    def __init__(
        self,
        name: str,
        kind: NodeKind,
        layout: PartitionLayout | None,
        category: ExecutorCategory,
        children: Iterable[GpuNode] = (),
        factory: Callable[[int | None], Executor] | None = None,
        forwarder: BatchForwarder | None = None,
    ):
        self._name = name
        self._kind = kind
        self._layout = layout
        self._category = category
        self._children = list(children)
        self._factory = factory
        self._forwarder = forwarder

    def name(self) -> str:
        return self._name

    def kind(self) -> NodeKind:
        return self._kind

    def output_partitions(self):
        return self._layout

    def output_schema(self):
        return None

    def children(self):
        return self._children

    def make_executors(self) -> NodeExecutors:
        if self._forwarder is not None:
            return NodeExecutors(self._category, forwarder=self._forwarder)
        return NodeExecutors(self._category, backends=ExecutorBackends(cpu=self._factory))

    def validate_schemas_and_partitions(self) -> None:
        return None


def _layout(n, batch_layout=BatchLayout.MULTIPLE_BATCHES, hash_keys=None, sort=None):
    return PartitionLayout(
        n=n,
        key_distribution=(
            KeyDistribution.by_hash(hash_keys)
            if hash_keys is not None
            else KeyDistribution.not_specified()
        ),
        sort_order=SortOrder.batch_sorted(sort) if sort else SortOrder.not_specified(),
        batch_layout=batch_layout,
    )


def source(name: str, batches_by_lane: list[list[int]], **layout_kwargs) -> MockNode:
    """`batches_by_lane[p]` is the row count of each batch lane p produces."""

    def factory(lane):
        return ScriptedSource(
            MockBatch(f"{name}.p{lane}.b{i}", rows)
            for i, rows in enumerate(batches_by_lane[lane])
        )

    return MockNode(
        name,
        NodeKind.SOURCE,
        _layout(len(batches_by_lane), **layout_kwargs),
        ExecutorCategory.SOURCE,
        factory=factory,
    )


def exec_node(name: str, child: MockNode, selectivity: float = 1.0, **layout_kwargs) -> MockNode:
    n = child.output_partitions().n
    return MockNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(n, **layout_kwargs),
        ExecutorCategory.EXEC,
        children=[child],
        factory=lambda lane: MapExec(name, selectivity),
    )


def sink(name: str, child: MockNode) -> MockNode:
    return MockNode(
        name,
        NodeKind.SINK,
        None,
        ExecutorCategory.EXEC,
        children=[child],
        factory=lambda lane: MapExec(name),
    )


def coalesce_all(name: str, child: MockNode, **layout_kwargs) -> MockNode:
    n = child.output_partitions().n
    layout_kwargs.setdefault("batch_layout", BatchLayout.SINGLE_BATCH)
    return MockNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(n, **layout_kwargs),
        ExecutorCategory.BATCH_ACCUMULATOR,
        children=[child],
        factory=lambda lane: CollectAccumulator(name),
    )


def coalesce_target(name: str, child: MockNode, target_rows: int) -> MockNode:
    """A batch accumulator that keeps `MultipleBatches` — legal at any tree position."""
    n = child.output_partitions().n
    return MockNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(n),
        ExecutorCategory.BATCH_ACCUMULATOR,
        children=[child],
        factory=lambda lane: TargetCoalescer(name, target_rows),
    )


def merge_partitions(name: str, child: MockNode) -> MockNode:
    return MockNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(1),
        ExecutorCategory.BATCH_FORWARDER,
        children=[child],
        forwarder=MergePartitionsForwarder(child.output_partitions().n),
    )


def emit_partitions(
    name: str, child: MockNode, n: int, router: Callable[[MockBatch], list[int]], hash_keys=(0,)
) -> MockNode:
    return MockNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(n, hash_keys=hash_keys),
        ExecutorCategory.PARTITION_EMITTER,
        children=[child],
        factory=lambda lane: ScriptedEmitter(name, n, router),
    )


def merge_sorted_partitions(name: str, child: MockNode) -> MockNode:
    n_in = child.output_partitions().n
    return MockNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(1, batch_layout=BatchLayout.SINGLE_BATCH),
        ExecutorCategory.PARTITION_ACCUMULATOR,
        children=[child],
        factory=lambda lane: MergeSortedPartitions(name, n_in),
    )


def eager_merge_partitions(name: str, child: MockNode) -> MockNode:
    """A partition accumulator that emits per lane event — see `EagerMergePartitions`."""
    n_in = child.output_partitions().n
    return MockNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(1),
        ExecutorCategory.PARTITION_ACCUMULATOR,
        children=[child],
        factory=lambda lane: EagerMergePartitions(name, n_in),
    )


def join(name: str, build: MockNode, probe: MockNode, emit_on_finish: int = 0) -> MockNode:
    return MockNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(build.output_partitions().n),
        ExecutorCategory.JOIN,
        children=[build, probe],
        factory=lambda lane: RecordingJoin(f"{name}.p{lane}", emit_on_finish),
    )


def union(name: str, children: list[MockNode]) -> MockNode:
    counts = [c.output_partitions().n for c in children]
    return MockNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(sum(counts)),
        ExecutorCategory.BATCH_FORWARDER,
        children=children,
        forwarder=UnionForwarder(counts),
    )


def interleave(name: str, children: list[MockNode], hash_keys=(0,)) -> MockNode:
    n = children[0].output_partitions().n
    return MockNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(n, hash_keys=hash_keys),
        ExecutorCategory.BATCH_FORWARDER,
        children=children,
        forwarder=InterleaveForwarder(len(children), n),
    )


# -- routers ---------------------------------------------------------------------


def even_router(n: int) -> Callable[[MockBatch], list[int]]:
    """Splits a batch's rows as evenly as the row count allows."""

    def route(batch: MockBatch) -> list[int]:
        base, extra = divmod(batch.num_rows(), n)
        return [base + (1 if lane < extra else 0) for lane in range(n)]

    return route


def skew_router(n: int, hot_lane: int) -> Callable[[MockBatch], list[int]]:
    """Every row to one lane — the starved-sibling case."""

    def route(batch: MockBatch) -> list[int]:
        return [batch.num_rows() if lane == hot_lane else 0 for lane in range(n)]

    return route
