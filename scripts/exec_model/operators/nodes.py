"""`GpuNode` implementations backed by the pandas operators, plus plan builders.

Same trait surface the mocks implement, so the same two drivers run both. Partition
layouts are declared here because the plan validator checks them; the executors never see
them.

Every builder records how it was called (`Recipe`), so a plan can be rebuilt with different
arguments — see `injection.py`. Layouts here are computed from constructor arguments and
captured in closures, not stored where they could be edited afterwards, so re-running the
builder is the only way to change one and keep the plan consistent.
"""

from __future__ import annotations

import functools
import inspect
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable

from ..errors import PlanError
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
from ..limit import RowInterval
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
        row_interval: RowInterval | None = None,
        validator: Callable[["PandasNode"], None] | None = None,
    ):
        self._row_interval = row_interval
        self._validator = validator
        self._name = name
        self._kind = kind
        self._layout = layout
        self._category = category
        self._children = list(children)
        self._factory = factory
        self._forwarder = forwarder
        #: set by the builder that made this node; None for a hand-constructed one
        self.recipe: Recipe | None = None

    def name(self) -> str:
        return self._name

    def kind(self) -> NodeKind:
        return self._kind

    def category(self) -> ExecutorCategory:
        return self._category

    def row_interval(self) -> RowInterval | None:
        return self._row_interval

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
        if self._validator is not None:
            self._validator(self)


@dataclass(frozen=True)
class Recipe:
    """The builder call that produced a node, so a rewriter can make the call again.

    `args` holds every parameter by name, children included, which is what lets a rewrite
    be expressed as "same call, these arguments replaced" without the rewriter knowing
    what any particular builder does with them.
    """

    builder: Callable[..., "PandasNode"]
    args: dict

    def child_args(self) -> dict:
        """The arguments that are child nodes, by parameter name."""
        found = {}
        for key, value in self.args.items():
            if isinstance(value, PandasNode):
                found[key] = value
            elif isinstance(value, (list, tuple)) and value and all(
                isinstance(item, PandasNode) for item in value
            ):
                found[key] = list(value)
        return found

    def rebuild(self, **overrides) -> "PandasNode":
        return self.builder(**{**self.args, **overrides})


def _records_its_recipe(builder):
    """Record the call on the node it returns."""
    signature = inspect.signature(builder)

    @functools.wraps(builder)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        node = builder(*args, **kwargs)
        node.recipe = Recipe(wrapper, dict(bound.arguments))
        return node

    return wrapper


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


@_records_its_recipe
def scan(
    name,
    frame,
    n_partitions=1,
    rows_per_group=4,
    target_batch_rows=None,
    empty_lanes=(),
    empty_batch_probability=0.0,
    seed=0,
) -> PandasNode:
    """A table split into row groups, mapped to (partition, batch) by the T2 policy.

    The last three arguments are shapes the policy would not choose but the runtime must
    survive: lanes with nothing to read, and zero-row batches arriving mid-stream.
    """
    row_groups = source.split_row_groups(len(frame), rows_per_group)
    mapping = source.partition_row_groups(row_groups, n_partitions, target_batch_rows)
    if empty_lanes:
        mapping = source.drain_lanes(mapping, empty_lanes)
    node = PandasNode(
        name,
        NodeKind.SOURCE,
        _layout(n_partitions),
        ExecutorCategory.SOURCE,
        factory=lambda lane: source.TableSource(
            frame,
            row_groups,
            mapping[lane],
            name,
            lane,
            empty_batch_probability,
            # Per lane, so lanes do not inject in lockstep, and derived from the name so
            # two scans in one plan differ. crc32 rather than hash(): str hashing is
            # salted per process and this has to reproduce.
            zlib.crc32(f"{name}.{lane}".encode(), seed),
        ),
    )
    node.partition_groups = mapping  # what the plan golden would render verbatim
    return node


# -- exec -------------------------------------------------------------------------


@_records_its_recipe
def filter_(name, child, predicate) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child)),
        ExecutorCategory.EXEC,
        [child],
        factory=lambda lane: exec_ops.FilterExec(predicate, name),
    )


@_records_its_recipe
def project(name, child, exprs) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child)),
        ExecutorCategory.EXEC,
        [child],
        factory=lambda lane: exec_ops.ProjectExec(exprs, name),
    )


@_records_its_recipe
def sort(name, child, by, ascending=None, nulls_first=False, fetch=None) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child), sort_by=by),
        ExecutorCategory.EXEC,
        [child],
        factory=lambda lane: exec_ops.SortExec(by, ascending, nulls_first, fetch, name),
    )


@_records_its_recipe
def partial_aggregate(name, child, keys, aggs) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child)),
        ExecutorCategory.EXEC,
        [child],
        factory=lambda lane: exec_ops.PartialAggregateExec(keys, aggs, name),
    )


@_records_its_recipe
def limit(name, child, skip=0, fetch=None) -> PandasNode:
    """The mid-plan `GpuLimit`: one partition in, any number of batches, streaming out.

    Its input needs `GpuMergePartitions` beneath it — an interval over N lanes names no
    rows — but *not* a `GpuCoalesceAllBatches`. Requiring one batch would make
    `... JOIN (SELECT * FROM orders LIMIT 100)` read the whole of orders to answer a
    hundred rows. See `accumulators.LimitStream`.

    The output layout follows the input: a limit is a prefix of its stream, so it neither
    increases the batch count nor disturbs an order. Feeding only the sink it is not this
    node at all — `skip`/`fetch` go on `unload`.
    """
    child_layout = child.output_partitions()
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        PartitionLayout(
            n=1,
            key_distribution=KeyDistribution.not_specified(),
            sort_order=child_layout.sort_order,
            batch_layout=child_layout.batch_layout,
        ),
        ExecutorCategory.BATCH_ACCUMULATOR,
        [child],
        factory=lambda lane: accumulators.LimitStream(skip, fetch, name),
        row_interval=RowInterval(skip, fetch),
        validator=_one_partition_in,
    )


def _one_partition_in(node: PandasNode) -> None:
    lanes = _lanes(node.children()[0])
    if lanes != 1:
        raise PlanError(
            f"{node.name()}: a limit is an interval over one stream, and its input has "
            f"{lanes} lanes — the planner inserts GpuMergePartitions below it"
        )


@_records_its_recipe
def unload(name, child, skip=0, fetch=None) -> PandasNode:
    """The boundary crossing, and where a root-adjacent limit's interval lives.

    `skip`/`fetch` are the limit: what the translation layer emits instead of a `GpuLimit`
    node when the limit feeds only the unload. Written here rather than as a rewrite of a
    limit node above, because that is what T4 will do — the node never exists.
    """
    return PandasNode(
        name,
        NodeKind.SINK,
        None,
        ExecutorCategory.UNLOAD,
        [child],
        factory=lambda lane: exec_ops.UnloadExec(name),
        row_interval=RowInterval(skip, fetch) if (skip or fetch is not None) else None,
    )


# -- accumulators -----------------------------------------------------------------


@_records_its_recipe
def coalesce_all(name, child, schema=None) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child), batch_layout=BatchLayout.SINGLE_BATCH),
        ExecutorCategory.BATCH_ACCUMULATOR,
        [child],
        factory=lambda lane: accumulators.CoalesceAllBatches(name, schema),
    )


@_records_its_recipe
def aggregate_batches(name, child, keys, aggs, final, schema=None) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child), batch_layout=BatchLayout.SINGLE_BATCH),
        ExecutorCategory.BATCH_ACCUMULATOR,
        [child],
        factory=lambda lane: accumulators.AggregateBatches(keys, aggs, final, name, schema),
    )


@_records_its_recipe
def rebatch(name, child, target_rows) -> PandasNode:
    """Re-cut a lane's stream to `target_rows` — merging, splitting, or both (#139)."""
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child)),
        ExecutorCategory.BATCH_ACCUMULATOR,
        [child],
        factory=lambda lane: accumulators.ReBatchToTarget(target_rows, name),
    )


@_records_its_recipe
def accumulate_and_sort(name, child, by, ascending=None, nulls_first=False, fetch=None, schema=None):
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(child), batch_layout=BatchLayout.SINGLE_BATCH, sort_by=by),
        ExecutorCategory.BATCH_ACCUMULATOR,
        [child],
        factory=lambda lane: accumulators.AccumulateBatchesAndSort(
            by, ascending, nulls_first, fetch, name, schema
        ),
    )


@_records_its_recipe
def merge_sorted_partitions(name, child, by, ascending=None, nulls_first=False, fetch=None, schema=None):
    n_in = _lanes(child)
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(1, batch_layout=BatchLayout.SINGLE_BATCH, sort_by=by),
        ExecutorCategory.PARTITION_ACCUMULATOR,
        [child],
        factory=lambda lane: accumulators.MergeSortedPartitions(
            n_in, by, ascending, nulls_first, fetch, name, schema
        ),
    )


# -- partition ops ----------------------------------------------------------------


@_records_its_recipe
def merge_partitions(name, child) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(1),
        ExecutorCategory.BATCH_FORWARDER,
        [child],
        forwarder=MergePartitionsForwarder(_lanes(child)),
    )


@_records_its_recipe
def emit_partitions(name, child, keys, n_partitions, hash_fn=None) -> PandasNode:
    """`hash_fn` defaults to the real placement; anything key-deterministic is legal."""
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(n_partitions, hash_keys=keys),
        ExecutorCategory.PARTITION_EMITTER,
        [child],
        factory=lambda lane: partition_ops.EmitPartitions(keys, n_partitions, name, hash_fn),
    )


@_records_its_recipe
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


@_records_its_recipe
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


@_records_its_recipe
def hash_join(
    name, build, probe, join_type, build_keys, probe_keys, null_equals_null=False,
    fanout=joins.TRIVIAL_FANOUT, probe_schema=None,
) -> PandasNode:
    """`fanout` is the optimizer's cardinality estimate for this join, carried as a node
    property so the executor built from it can model its own scratch. `probe_schema` is
    the probe side's column list, consulted only when an outer finish saw no probe batch."""
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(build)),
        ExecutorCategory.JOIN,
        [build, probe],
        factory=lambda lane: joins.HashJoin(
            join_type, build_keys, probe_keys, null_equals_null, f"{name}.p{lane}", fanout,
            probe_schema,
        ),
    )


@_records_its_recipe
def cross_join(name, build, probe) -> PandasNode:
    return PandasNode(
        name,
        NodeKind.INTERMEDIATE,
        _layout(_lanes(build)),
        ExecutorCategory.JOIN,
        [build, probe],
        factory=lambda lane: joins.HashJoin(joins.JoinType.CROSS, [], [], False, f"{name}.p{lane}"),
    )
