"""`GpuNode` — what a plan node offers the driver — plus backend selection.

A node declares its output layout and which executor category drives it; the category
carries the backend pair inside, so the driver never downcasts a node to find out how to
run it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .executors import Executor
from .forwarder import BatchForwarder
from .layout import NodeKind, PartitionLayout


class ExecutorCategory(Enum):
    SOURCE = "Source"
    EXEC = "Exec"
    BATCH_ACCUMULATOR = "BatchAccumulator"
    PARTITION_ACCUMULATOR = "PartitionAccumulator"
    PARTITION_EMITTER = "PartitionEmitter"
    JOIN = "Join"
    BATCH_FORWARDER = "BatchForwarder"
    UNLOAD = "Unload"


#: Categories with one executor instance per (node, lane); the rest are per node.
LANE_SCOPED = frozenset(
    {
        ExecutorCategory.SOURCE,
        ExecutorCategory.EXEC,
        ExecutorCategory.BATCH_ACCUMULATOR,
        ExecutorCategory.JOIN,
        ExecutorCategory.UNLOAD,
    }
)

#: Categories that consume one input lane and produce one, 1:1 per batch.
ONE_TO_ONE = frozenset(
    {
        ExecutorCategory.EXEC,
        ExecutorCategory.BATCH_ACCUMULATOR,
        ExecutorCategory.UNLOAD,
    }
)


@dataclass(frozen=True)
class ExecutorBackends:
    """Constructors, not instances — the driver instantiates per lane.

    The constructor takes the lane it is being built for, because a lane-scoped executor
    generally needs it: a loader's lane reads its own row groups out of the partitioner's
    mapping. Cross-lane categories are built once per node and get `None`.
    """

    cpu: Callable[[int | None], Executor]
    gpu: Callable[[int | None], Executor] | None = None


@dataclass(frozen=True)
class NodeExecutors:
    """One of: a backend pair to select from, or a forwarder (routing has no backend)."""

    category: ExecutorCategory
    backends: ExecutorBackends | None = None
    forwarder: BatchForwarder | None = None

    def __post_init__(self):
        routing = self.category is ExecutorCategory.BATCH_FORWARDER
        if routing and (self.forwarder is None or self.backends is not None):
            raise ValueError("BatchForwarder carries a forwarder and no backends")
        if not routing and (self.backends is None or self.forwarder is not None):
            raise ValueError(f"{self.category.value} carries backends and no forwarder")


class BackendSelector(ABC):
    """One `select` per category in Rust; Python needs only the one method."""

    @abstractmethod
    def select(
        self, category: ExecutorCategory, backends: ExecutorBackends, lane: int | None
    ) -> Executor: ...


class CpuBackendSelector(BackendSelector):
    def select(
        self, category: ExecutorCategory, backends: ExecutorBackends, lane: int | None
    ) -> Executor:
        return backends.cpu(lane)


class GpuBackendSelector(BackendSelector):
    def select(
        self, category: ExecutorCategory, backends: ExecutorBackends, lane: int | None
    ) -> Executor:
        if backends.gpu is None:
            raise ValueError(f"{category.value} has no GPU backend")
        return backends.gpu(lane)


class RecipeJoinBackendSelector(BackendSelector):
    """Joins through the FlatBuffers emulation; everything else through pandas.

    The emulation (`operators/recipe_join.py`) stands in the GPU slot because it *is* the
    GPU path modelled — a recipe plan of `Cudf*` nodes addressed by seq, one
    `execute_node` call per (build, probe batch), handles consumed as the FFI consumes
    them. Only the join carries one: it is the operator whose state has to survive a call,
    so it is the operator the frozen-surface claim turns on. A join node without one is an
    error rather than a silent fall back to pandas, or the suite would report a backend it
    did not run.
    """

    def select(
        self, category: ExecutorCategory, backends: ExecutorBackends, lane: int | None
    ) -> Executor:
        if category is not ExecutorCategory.JOIN:
            return backends.cpu(lane)
        if backends.gpu is None:
            raise ValueError("a join node reached the recipe backend without one")
        return backends.gpu(lane)


class GpuNode(ABC):
    """A plan node. Immutable; all run-time state lives in the driver."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def kind(self) -> NodeKind: ...

    @abstractmethod
    def output_partitions(self) -> PartitionLayout | None:
        """Present for non-sink nodes."""

    @abstractmethod
    def output_schema(self):
        """Present for non-sink nodes. Out of scope for the prototype (T0)."""

    @abstractmethod
    def children(self) -> list["GpuNode"]: ...

    @abstractmethod
    def make_executors(self) -> NodeExecutors:
        """A fresh instance set per call, so the driver can instantiate per lane."""

    @abstractmethod
    def validate_schemas_and_partitions(self) -> None:
        """Raise `PlanError` when a child's layout does not meet this node's needs."""

    def row_interval(self):
        """`RowInterval` for a mid-plan `GpuLimit` or for a `GpuUnload` that absorbed a
        root-adjacent one; None for everything else (`limit.py`)."""
        return None
