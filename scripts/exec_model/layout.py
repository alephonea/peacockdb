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
    NOT_SPECIFIED = "NotSpecified"
    BATCH_SORTED = "BatchSorted"
    PARTITION_SORTED = "PartitionSorted"


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

    @classmethod
    def partition_sorted(cls, columns) -> "SortOrder":
        return cls(SortKind.PARTITION_SORTED, tuple(columns))

    @property
    def is_batch_sorted(self) -> bool:
        """PartitionSorted implies BatchSorted; validation accepts the weaker property."""
        return self.kind in (SortKind.BATCH_SORTED, SortKind.PARTITION_SORTED)

    @property
    def is_partition_sorted(self) -> bool:
        return self.kind is SortKind.PARTITION_SORTED


@dataclass(frozen=True)
class PartitionLayout:
    n: int
    key_distribution: KeyDistribution = field(default_factory=KeyDistribution.not_specified)
    sort_order: SortOrder = field(default_factory=SortOrder.not_specified)
    batch_layout: BatchLayout = BatchLayout.MULTIPLE_BATCHES

    def __post_init__(self):
        # Under SingleBatch the two sort properties coincide — canonicalize to the
        # stronger one so equality comparisons do not distinguish equivalent layouts.
        if (
            self.batch_layout is BatchLayout.SINGLE_BATCH
            and self.sort_order.kind is SortKind.BATCH_SORTED
        ):
            object.__setattr__(
                self, "sort_order", SortOrder.partition_sorted(self.sort_order.columns)
            )
