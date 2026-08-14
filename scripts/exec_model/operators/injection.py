"""`LayoutInjector` — re-shape a plan's partitioning and batching, keep its answer.

A query's result is a function of its rows, not of how those rows were divided. Everything
the batch-partitioned model adds — lanes, batches, shuffles, accumulators — is machinery
underneath that invariant, and the way to test machinery like that is to vary it as
violently as the engine permits and demand the same answer every time.

That is what this does. Give it a plan built from `nodes.py` and it hands back an
equivalent plan whose lanes, row groups, batch sizes and hash placement are whatever the
chosen preset says. The plans in `test_tpch.py` are written once, at whatever partitioning
reads clearly, and then run at every shape.

**Why rebuilding rather than editing.** A node's partitioning is not a field: the source
captured a row-group mapping computed at build time, the emitter captured its lane count,
every parent computed its own layout from its child's. Editing one would leave the rest
inconsistent — so a rewrite re-runs the builder with different arguments, which is what
`nodes.Recipe` records for.

**What may not be re-partitioned.** A join is correct at N lanes only if both sides are
hash-partitioned on the join keys into the same N. Where a plan joins two streams that were
never shuffled — a small build side against a scan, the shape v1 leans on — its one lane is
load-bearing, and the injector leaves that subtree's lane count alone while still varying
its batching. Getting this wrong would not fail the plan validator; it would silently
compute a join of matching slices and return too few rows.

**Why a degenerate hash is a legal hash.** The submodes include placements that send three
quarters of the keys to one lane, or all of them. A shuffle's contract is co-location —
equal keys reach the same lane — and nothing above it may depend on how evenly the lanes
were loaded. Every placement here is a pure function of the key columns, so every one of
them satisfies the contract, and a plan that only works under a well-spread hash is broken
rather than unlucky.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from ..node import ExecutorCategory
from . import nodes as N
from . import partition_ops
from .nodes import PandasNode


class HashMode(Enum):
    """How `GpuEmitPartitions` places rows. All key-deterministic; see the module note."""

    SPREAD = "spread"
    SKEWED = "skewed"
    ALL_TO_FIRST = "all_to_first"
    ALL_TO_LAST = "all_to_last"

    @property
    def hash_fn(self):
        return {
            HashMode.SPREAD: partition_ops.partition_ids,
            HashMode.SKEWED: partition_ops.skewed_ids,
            HashMode.ALL_TO_FIRST: partition_ops.first_lane_ids,
            HashMode.ALL_TO_LAST: partition_ops.last_lane_ids,
        }[self]


@dataclass(frozen=True)
class Shape:
    """One preset's arithmetic, in units a table's row count can be divided into."""

    lanes: int
    #: row groups per lane — the source's read granularity
    groups_per_lane: int
    #: row groups packed into one batch; None means one batch per lane
    groups_per_batch: int | None
    #: lanes drained of their batches, so they are live but produce nothing
    empty_lanes: tuple[int, ...] = ()
    #: insert a re-batching node above every source
    rebatch: bool = False


class LayoutPreset(Enum):
    ONE_PARTITION_ONE_BATCH = "one_partition_one_batch"
    FEW_PARTITIONS_FEW_BATCHES = "few_partitions_few_batches"
    MANY_SMALL_PARTITIONS = "many_small_partitions"
    EMPTY_PARTITIONS = "empty_partitions"
    REBATCHED = "rebatched"

    @property
    def shape(self) -> Shape:
        return _SHAPES[self]


_SHAPES = {
    # The degenerate end: one lane, one batch, the shape the oracle itself has. The layout
    # still declares MultipleBatches and no key distribution — describing the plan as
    # single-batch would let a downstream node take a shortcut, and the point is to run
    # the general path over degenerate data, not to test the shortcut.
    LayoutPreset.ONE_PARTITION_ONE_BATCH: Shape(
        lanes=1, groups_per_lane=1, groups_per_batch=None
    ),
    LayoutPreset.FEW_PARTITIONS_FEW_BATCHES: Shape(
        lanes=3, groups_per_lane=6, groups_per_batch=2
    ),
    # Small enough that per-batch overheads dominate and short batches are the norm.
    LayoutPreset.MANY_SMALL_PARTITIONS: Shape(
        lanes=8, groups_per_lane=4, groups_per_batch=1
    ),
    LayoutPreset.EMPTY_PARTITIONS: Shape(
        lanes=5, groups_per_lane=6, groups_per_batch=2, empty_lanes=(1, 3)
    ),
    LayoutPreset.REBATCHED: Shape(
        lanes=5, groups_per_lane=6, groups_per_batch=2, empty_lanes=(1, 3), rebatch=True
    ),
}

#: How far a re-batching node moves off the source's batch size, in each direction.
_REBATCH_FACTOR = 3


@dataclass
class LayoutInjector:
    """Rewrite a plan into one preset and one hash mode. `apply` does not mutate its input."""

    preset: LayoutPreset
    hash_mode: HashMode = HashMode.SPREAD
    #: chance that a source emits a zero-row batch instead of advancing, per call
    empty_batch_probability: float = 0.0
    seed: int = 0
    #: sources rewritten so far, in pre-order — decides which way a re-batch node cuts
    _sources: int = field(default=0, init=False, repr=False)

    @property
    def label(self) -> str:
        return f"{self.preset.value}/{self.hash_mode.value}"

    def apply(self, root: PandasNode) -> PandasNode:
        self._sources = 0
        return self._rewrite(root, free=True)

    # -- the rewrite -------------------------------------------------------------

    def _rewrite(self, node: PandasNode, free: bool) -> PandasNode:
        if node.recipe is None:
            raise ValueError(
                f"{node.name()}: built without a recipe, so it cannot be rebuilt — "
                "use the builders in nodes.py"
            )
        if node.category() is ExecutorCategory.JOIN and not _is_shuffled_join(node):
            free = False  # this join's lane count is part of its correctness

        overrides = {
            key: (
                self._rewrite(value, free)
                if isinstance(value, PandasNode)
                else [self._rewrite(child, free) for child in value]
            )
            for key, value in node.recipe.child_args().items()
        }

        if node.category() is ExecutorCategory.SOURCE:
            return self._rewrite_source(node, free)
        if node.category() is ExecutorCategory.PARTITION_EMITTER:
            overrides["hash_fn"] = self.hash_mode.hash_fn
            if free:
                overrides["n_partitions"] = self.preset.shape.lanes
        return node.recipe.rebuild(**overrides)

    def _rewrite_source(self, node: PandasNode, free: bool) -> PandasNode:
        shape = self.preset.shape
        args = node.recipe.args
        rows = len(args["frame"])
        lanes = shape.lanes if free else args["n_partitions"]

        rows_per_group = max(1, math.ceil(rows / max(1, lanes * shape.groups_per_lane)))
        target = None if shape.groups_per_batch is None else rows_per_group * shape.groups_per_batch
        # A lane can only be drained if another lane is left to take its rows.
        empty = tuple(lane for lane in shape.empty_lanes if lane < lanes)

        rebuilt = node.recipe.rebuild(
            n_partitions=lanes,
            rows_per_group=rows_per_group,
            target_batch_rows=target,
            empty_lanes=empty,
            empty_batch_probability=self.empty_batch_probability,
            seed=self.seed,
        )
        index = self._sources
        self._sources += 1
        if not shape.rebatch:
            return rebuilt
        # Alternate the direction so one plan gets both: a node that merges the stream's
        # batches and a node that splits them.
        base = target if target is not None else rows_per_group
        target_rows = base * _REBATCH_FACTOR if index % 2 == 0 else max(1, base // _REBATCH_FACTOR)
        return N.rebatch(f"{args['name']}$rebatch", rebuilt, target_rows)


# -- what a join's lanes mean ------------------------------------------------------


class _LaneOrigin(Enum):
    """What decides a subtree's lane count, which is what says whether it may be changed."""

    SCAN = "scan"          # the source's own partitioning, with no shuffle above it
    SHUFFLE = "shuffle"    # a hash emitter, so lanes carry co-located keys
    COLLAPSED = "collapsed"  # an N→1 node, so the lane count is 1 whatever happened below


def _is_shuffled_join(join: PandasNode) -> bool:
    """True when both sides reach the join through the same kind of lane-defining node.

    Two shuffled sides may be re-partitioned together: change both emitters to N and the
    keys still meet. Two collapsed sides are already pinned at one lane and nothing below
    can change that. Anything else — the small-build-side-against-a-scan shape — must keep
    the lane count the plan was written with.
    """
    build, probe = (_lane_origin(child) for child in join.children())
    return build is probe and build in (_LaneOrigin.SHUFFLE, _LaneOrigin.COLLAPSED)


def _lane_origin(node: PandasNode) -> _LaneOrigin:
    category = node.category()
    if category is ExecutorCategory.SOURCE:
        return _LaneOrigin.SCAN
    if category is ExecutorCategory.PARTITION_EMITTER:
        return _LaneOrigin.SHUFFLE
    if category is ExecutorCategory.PARTITION_ACCUMULATOR:
        return _LaneOrigin.COLLAPSED
    children = node.children()
    if not children:
        return _LaneOrigin.SCAN
    if category is ExecutorCategory.BATCH_FORWARDER:
        if node.output_partitions().n == 1:
            return _LaneOrigin.COLLAPSED   # GpuMergePartitions
        origins = {_lane_origin(child) for child in children}
        # Union relabels lanes; it preserves a distribution only when every branch has
        # the same one, which is the rule that makes GpuInterleave selectable at all.
        return _LaneOrigin.SHUFFLE if origins == {_LaneOrigin.SHUFFLE} else _LaneOrigin.SCAN
    # A pipe, or a join — whose lane count is its build side's.
    return _lane_origin(children[0])
