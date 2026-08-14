"""`GpuNode` implementations backed by the pandas operators, plus plan builders.

Same trait surface the mocks implement, so the same two drivers run both. Partition
layouts are declared here because the plan validator checks them; the executors never see
them.
"""

from __future__ import annotations

from typing import Callable, Iterable

from ..executors import Executor
from ..forwarder import (
    BatchForwarder,
    InterleaveForwarder,
    MergePartitionsForwarder,
    UnionForwarder,
)
from ..layout import (
    BatchLayout,
    ColumnOrder,
    KeyDistribution,
    NodeKind,
    PartitionLayout,
    SortOrder,
)
from ..node import ExecutorBackends, ExecutorCategory, GpuNode, NodeExecutors
from . import accumulators, exec_ops, joins, partition_ops, source


class PandasNode(GpuNode):
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
        return None  # schema registry is T7; out of the prototype's validation scope

    def children(self):
        return self._children

    def make_executors(self) -> NodeExecutors:
        if self._forwarder is not None:
            return NodeExecutors(self._category, forwarder=self._forwarder)
        return NodeExecutors(self._category, backends=ExecutorBackends(cpu=self._factory))

    def validate_schemas_and_partitions(self) -> None:
        return None


def _layout(n, batch_layout=BatchLayout.MULTIPLE_BATCHES, hash_keys=None, sort_by=None):
    return PartitionLayout(
        n=n,
        key_distribution=(
            KeyDistribution.by_hash(range(len(hash_keys)))
            if hash_keys is not None
            else KeyDistribution.not_specified()
        ),
        sort_order=(
            SortOrder.batch_sorted([ColumnOrder(i) for i in range(len(sort_by))])
            if sort_by
            else SortOrder.not_specified()
        ),
        batch_layout=batch_layout,
    )


def _lanes(child: PandasNode) -> int:
    return child.output_partitions().n


# -- source -----------------------------------------------------------------------


def scan(name, frame, n_partitions=1, rows_per_group=4, target_batch_rows=None) -> PandasNode:
    """A table split into row groups, mapped to (partition, batch) by the T2 policy."""
    row_groups = source.split_row_groups(len(frame), rows_per_group)
    mapping = source.partition_row_groups(row_groups, n_partitions, target_batch_rows)
    node = PandasNode(
        name,
        NodeKind.SOURCE,
        _layout(n_partitions),
        ExecutorCategory.SOURCE,
        factory=lambda lane: source.TableSource(frame, row_groups, mapping[lane], name, lane),
    )
    node.partition_groups = mapping  # what the plan golden would render verbatim
    return node


# -- exec -------------------------------------------------------------------------


def filter_(name, child, predicate) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child)),
        ExecutorCategory.EXEC,
        [child],
        factory=lambda lane: exec_ops.FilterExec(predicate, name),
    )


def project(name, child, exprs) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child)),
        ExecutorCategory.EXEC,
        [child],
        factory=lambda lane: exec_ops.ProjectExec(exprs, name),
    )


def sort(name, child, by, ascending=None, nulls_first=False, fetch=None) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child), sort_by=by),
        ExecutorCategory.EXEC,
        [child],
        factory=lambda lane: exec_ops.SortExec(by, ascending, nulls_first, fetch, name),
    )


def partial_aggregate(name, child, keys, aggs) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child)),
        ExecutorCategory.EXEC,
        [child],
        factory=lambda lane: exec_ops.PartialAggregateExec(keys, aggs, name),
    )


def limit(name, child, skip=0, fetch=None) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child), batch_layout=BatchLayout.SINGLE_BATCH),
        ExecutorCategory.EXEC,
        [child],
        factory=lambda lane: exec_ops.LimitExec(skip, fetch, name),
    )


def unload(name, child) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.SINK,
        None,
        ExecutorCategory.EXEC,
        [child],
        factory=lambda lane: exec_ops.UnloadExec(name),
    )


# -- accumulators -----------------------------------------------------------------


def coalesce_all(name, child, schema=None) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child), batch_layout=BatchLayout.SINGLE_BATCH),
        ExecutorCategory.BATCH_ACCUMULATOR,
        [child],
        factory=lambda lane: accumulators.CoalesceAllBatches(name, schema),
    )


def aggregate_batches(name, child, keys, aggs, final) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child), batch_layout=BatchLayout.SINGLE_BATCH),
        ExecutorCategory.BATCH_ACCUMULATOR,
        [child],
        factory=lambda lane: accumulators.AggregateBatches(keys, aggs, final, name),
    )


def accumulate_and_sort(name, child, by, ascending=None, nulls_first=False, fetch=None):
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child), batch_layout=BatchLayout.SINGLE_BATCH, sort_by=by),
        ExecutorCategory.BATCH_ACCUMULATOR,
        [child],
        factory=lambda lane: accumulators.AccumulateBatchesAndSort(
            by, ascending, nulls_first, fetch, name
        ),
    )


def merge_sorted_partitions(name, child, by, ascending=None, nulls_first=False, fetch=None):
    n_in = _lanes(child)
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(1, batch_layout=BatchLayout.SINGLE_BATCH, sort_by=by),
        ExecutorCategory.PARTITION_ACCUMULATOR,
        [child],
        factory=lambda lane: accumulators.MergeSortedPartitions(
            n_in, by, ascending, nulls_first, fetch, name
        ),
    )


# -- partition ops ----------------------------------------------------------------


def merge_partitions(name, child) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(1),
        ExecutorCategory.BATCH_FORWARDER,
        [child],
        forwarder=MergePartitionsForwarder(_lanes(child)),
    )


def emit_partitions(name, child, keys, n_partitions) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(n_partitions, hash_keys=keys),
        ExecutorCategory.PARTITION_EMITTER,
        [child],
        factory=lambda lane: partition_ops.EmitPartitions(keys, n_partitions, name),
    )


def union(name, children) -> PandasNode:
    counts = [_lanes(c) for c in children]
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(sum(counts)),
        ExecutorCategory.BATCH_FORWARDER,
        children,
        forwarder=UnionForwarder(counts),
    )


def interleave(name, children) -> PandasNode:
    n = _lanes(children[0])
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(n),
        ExecutorCategory.BATCH_FORWARDER,
        children,
        forwarder=InterleaveForwarder(len(children), n),
    )


# -- join -------------------------------------------------------------------------


def hash_join(
    name, build, probe, join_type, build_keys, probe_keys, null_equals_null=False
) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(build)),
        ExecutorCategory.JOIN,
        [build, probe],
        factory=lambda lane: joins.HashJoin(
            join_type, build_keys, probe_keys, null_equals_null, f"{name}.p{lane}"
        ),
    )


def cross_join(name, build, probe) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(build)),
        ExecutorCategory.JOIN,
        [build, probe],
        factory=lambda lane: joins.HashJoin(joins.JoinType.CROSS, [], [], False, f"{name}.p{lane}"),
    )
