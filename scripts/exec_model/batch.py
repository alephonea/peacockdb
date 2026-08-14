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

    def slice_rows(self, offset: int, length: int) -> "Batch":
        """A row range of this batch, materialized — the mid-plan limit's slice.

        Models `peacock_executor_slice_handle`: the bounds are call arguments, not plan
        constants, which is what lets `accumulators.LimitStream` stream instead of holding
        the offset prefix (`limit.py`). The root-adjacent trim never comes here — its
        range rides the `unload` call itself. A backend whose batches cannot be sliced
        cannot carry a mid-plan limit, which is what the default raise says.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot be sliced")


@dataclass
class CallStats:
    """`scratch_bytes` is the measured transient; `None` means the run was not instrumented.

    Both backends can report it — the CPU directly, the GPU through RMM allocator hooks.
    """

    scratch_bytes: int | None = None
