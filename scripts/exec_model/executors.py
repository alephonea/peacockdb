"""Executor traits — one per node category.

Every state transition emits in the same call (`*_and_fetch`), so there is no wrong
interleaving to construct and output timing is a pure function of the call sequence.
Declarations only: implementations are backend code (Rust) or mocks (tests).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .batch import Batch, CallStats


class Executor(ABC):
    @abstractmethod
    def resident_bytes(self) -> int:
        """State held between calls. Accumulators report their accumulated input."""

    @abstractmethod
    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        """Pre-call model of production-time scratch; may consult self.

        Calls with no input batch (mark_done, finish) are modelled with 0 rows, 0 bytes.
        """


class ExecExecutor(Executor):
    @abstractmethod
    def exec(self, batch: Batch) -> tuple[Batch, CallStats]: ...


class BatchAccumulatorExecutor(Executor):
    @abstractmethod
    def accumulate_and_fetch(self, batch: Batch) -> tuple[list[Batch], CallStats]: ...

    @abstractmethod
    def mark_done_and_fetch(self) -> tuple[list[Batch], CallStats]: ...


@dataclass(frozen=True)
class LaneEvent:
    """One lane's event for a cross-lane accumulator: a batch, or that lane's end."""

    batch: Batch | None = None

    @classmethod
    def of(cls, batch: Batch) -> "LaneEvent":
        return cls(batch)

    @classmethod
    def done(cls) -> "LaneEvent":
        return cls(None)

    @property
    def is_done(self) -> bool:
        return self.batch is None


class PartitionAccumulatorExecutor(Executor):
    @abstractmethod
    def accumulate_and_fetch(
        self, partition: int, event: LaneEvent
    ) -> tuple[list[Batch], CallStats]:
        """One call per lane event; the call delivering the last lane's Done emits."""


class PartitionEmitterExecutor(Executor):
    @abstractmethod
    def emit(self, batch: Batch) -> tuple[list[Batch], CallStats]:
        """Exactly N outputs, some empty; the driver drops the empties."""


class JoinExecutor(Executor):
    @abstractmethod
    def set_build(self, batch: Batch) -> CallStats: ...

    @abstractmethod
    def probe_and_fetch(self, batch: Batch) -> tuple[list[Batch], CallStats]: ...

    @abstractmethod
    def finish_and_fetch(self) -> tuple[list[Batch], CallStats]: ...


class SourceExecutor(Executor):
    @abstractmethod
    def next_batch(self) -> tuple[Batch, CallStats] | None:
        """None means this lane is exhausted; it is never called again."""
