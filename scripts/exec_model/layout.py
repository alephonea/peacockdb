"""Partition layout vocabulary — the properties a node declares about its output.

Declarations only; no execution logic. Mirrors the Rust types in
`llm-wiki/tasks/batch_partitioned_executor.md` (Traits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeKind(Enum):
    SOURCE = "Source"
    INTERMEDIATE = "Intermediate"
    SINK = "Sink"


class BatchLayout(Enum):
    SINGLE_BATCH = "SingleBatch"
    MULTIPLE_BATCHES = "MultipleBatches"


@dataclass(frozen=True)
class ColumnOrder:
    column: int
    ascending: bool = True
    nulls_first: bool = False


class KeyDistributionKind(Enum):
    NOT_SPECIFIED = "NotSpecified"
    BY_HASH = "ByHash"


@dataclass(frozen=True)
class KeyDistribution:
    """Spark murmur3, seed 42 — the only routing `GpuEmitPartitions` has."""

    kind: KeyDistributionKind = KeyDistributionKind.NOT_SPECIFIED
    hash_keys: tuple[int, ...] = ()

    @classmethod
    def not_specified(cls) -> "KeyDistribution":
        return cls()

    @classmethod
    def by_hash(cls, hash_keys) -> "KeyDistribution":
        return cls(KeyDistributionKind.BY_HASH, tuple(hash_keys))

    def is_subset_of(self, group_columns) -> bool:
        """The `hashKeys ⊆ group columns` rule a final aggregate's input must satisfy."""
        if self.kind is KeyDistributionKind.NOT_SPECIFIED:
            return False
        return set(self.hash_keys).issubset(set(group_columns))


class SortKind(Enum):
    """Two-valued on purpose.

    A third `PartitionSorted` — whole stream ordered, not merely each batch — would be
    a synonym: the only nodes that order a whole stream (`GpuAccumulateBatchesAndSort`,
    `GpuMergeSortedPartitions`) emit exactly one batch, so the property holds precisely
    when `BatchSorted` meets `SingleBatch` and `PartitionLayout.is_stream_sorted` derives
    it. It becomes a real third state only under #138's ranged merge emission, which
    orders a stream across several batches.
    """

    NOT_SPECIFIED = "NotSpecified"
    BATCH_SORTED = "BatchSorted"


@dataclass(frozen=True)
class SortOrder:
    kind: SortKind = SortKind.NOT_SPECIFIED
    columns: tuple[ColumnOrder, ...] = ()

    @classmethod
    def not_specified(cls) -> "SortOrder":
        return cls()

    @classmethod
    def batch_sorted(cls, columns) -> "SortOrder":
        return cls(SortKind.BATCH_SORTED, tuple(columns))

    @property
    def is_batch_sorted(self) -> bool:
        return self.kind is SortKind.BATCH_SORTED


class UniqueScope(Enum):
    """How far a uniqueness guarantee reaches."""

    PER_BATCH = "PerBatch"          # no two rows of one batch share the key
    PER_PARTITION = "PerPartition"  # …of one lane, across its batches
    GLOBAL = "Global"               # …of the whole output


@dataclass(frozen=True)
class UniqueKeys:
    """A key set this output is unique on, and how widely.

    Declared and *not* checked. An aggregate knows the scope for free — its init is unique
    per batch, its per-lane merge per partition, and a final under a shuffle on those keys
    globally — and later work can read it rather than re-deriving it: a distinct over an
    already-globally-unique key is a no-op, a join on a globally unique build key cannot
    fan out, and a merge whose input is unique per partition needs no re-aggregation.
    Recording it now costs nothing and means the fact is not lost by the node that knew it.
    """

    columns: tuple[int, ...]
    scope: UniqueScope


@dataclass(frozen=True)
class PartitionLayout:
    n: int
    key_distribution: KeyDistribution = field(default_factory=KeyDistribution.not_specified)
    sort_order: SortOrder = field(default_factory=SortOrder.not_specified)
    batch_layout: BatchLayout = BatchLayout.MULTIPLE_BATCHES
    #: key sets this output is unique on; declared for later use, never checked here
    unique_keys: tuple[UniqueKeys, ...] = ()

    @property
    def is_stream_sorted(self) -> bool:
        """Whole stream ordered, not merely each batch — what a top-N after a sort needs.

        Derived rather than declared, so there is no second way to say it and no pair of
        fields that can disagree.
        """
        return self.sort_order.is_batch_sorted and self.batch_layout is BatchLayout.SINGLE_BATCH
