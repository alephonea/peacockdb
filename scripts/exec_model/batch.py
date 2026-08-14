"""The batch value type and the per-call stats every executor call returns.

A batch is one table's worth of rows: `num_rows()` and `byte_size()`, nothing else.

Rust takes every batch by value, so reuse after consumption is a compile error. Python
cannot express that, so the mock batch in the tests tracks consumption and raises on
the second one — the driver must never hand the same batch to two calls, because on the
GPU the handle is erased by the first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Batch(ABC):
    """Backend-independent; `CpuBatch` and `GpuBatch` in Rust, mocks here."""

    @abstractmethod
    def num_rows(self) -> int: ...

    @abstractmethod
    def byte_size(self) -> int: ...


@dataclass
class CallStats:
    """`scratch_bytes` is measured, so it is present on CPU and absent on GPU."""

    scratch_bytes: int | None = None
