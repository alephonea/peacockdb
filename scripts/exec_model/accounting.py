"""Resident accounting and the enforcer, in the shape the spec's Memory accounting states.

    resident = Σ byte_size of driver-held in-flight batches
             + Σ cached resident_bytes() over live executors

Per call: pre-check `resident + scratch_bytes(n_rows, n_bytes)` against the budget;
execute; remove consumed inputs, add outputs at actual `byte_size()`, refresh the one
executor's `resident_bytes()` delta, post-check.

Two details of that are load-bearing rather than incidental. The executor total is
**cached and refreshed one instance at a time**, not summed over every live executor per
check — the sum is what a Python prototype reaches for and it is also what forces the
accountant to hold a reference to every executor, which the Rust port cannot do while the
driver holds them mutably. Caching by key removes the aliasing along with the cost. And
`CallStats.scratch_bytes` is the *measured* transient — the CPU measures directly, the GPU
through RMM allocator hooks — so the modelled figure can be compared against it. `None`
means the run was not instrumented.

**That comparison is a diagnostic, not an invariant.** `scratch_bytes` is an estimate and
estimates are wrong: a join's model rests on the optimizer's cardinality figure, a filter's
on assumed selectivity, and neither is guaranteed. Under-estimates are recorded with their
magnitude, so model quality is visible and a regression is noticeable — they are not
asserted away. The enforcer is built for exactly this: its contract is "fail cleanly when
the accounted peak exceeds the budget", not "the accounting is never wrong".
"""

from __future__ import annotations

from dataclasses import dataclass

from .batch import Batch, CallStats
from .errors import ResidentBudgetExceeded
from .executors import Executor


@dataclass(frozen=True)
class Underestimate:
    """A call whose modelled scratch came in under what it measured.

    Expected to happen; recorded so its size and frequency are visible.
    """

    label: str
    modelled: int
    measured: int

    @property
    def ratio(self) -> float:
        """How far under — 2.0 means the call used twice what was modelled."""
        return self.measured / self.modelled if self.modelled else float("inf")

    def __str__(self) -> str:
        return f"{self.label}: modelled {self.modelled} < measured {self.measured}"


class ResidentAccountant:
    """`budget=None` accounts and reports without ever tripping."""

    def __init__(self, budget: int | None = None):
        self.budget = budget
        self.in_flight_bytes = 0
        self.executor_bytes = 0
        self.peak = 0
        self.calls = 0
        #: per-executor-instance last known residency — the cache the delta refreshes
        self._cached: dict[str, int] = {}
        #: calls where the model under-predicted — a diagnostic, not a failure
        self.underestimates: list[Underestimate] = []

    # -- the formula -------------------------------------------------------------

    def resident(self) -> int:
        return self.in_flight_bytes + self.executor_bytes

    def hold(self, batch: Batch) -> None:
        """A batch enters a driver-held queue."""
        self.in_flight_bytes += batch.byte_size()
        self._observe()

    def release(self, batch: Batch) -> None:
        """A batch leaves a driver-held queue — consumed, forwarded to the caller, or dropped."""
        self.in_flight_bytes -= batch.byte_size()
        if self.in_flight_bytes < 0:
            raise AssertionError(
                f"in-flight went negative ({self.in_flight_bytes}): a batch was released "
                "without having been held. usize would panic here in Rust."
            )

    # -- per call ----------------------------------------------------------------

    def begin_call(self, label: str, executor: Executor, n_rows: int, n_bytes: int) -> int:
        """Pre-check. Returns the modelled scratch so `end_call` can compare it."""
        modelled = executor.scratch_bytes(n_rows, n_bytes)
        self._trip_if_over(self.resident() + modelled, f"{label} pre-call")
        self.calls += 1
        return modelled

    def end_call(
        self,
        label: str,
        executor: Executor,
        stats: CallStats | None = None,
        modelled: int | None = None,
    ) -> None:
        """Refresh this one executor's residency, record the measurement, post-check."""
        self.refresh(label, executor)
        if stats is not None and stats.scratch_bytes is not None and modelled is not None:
            if modelled < stats.scratch_bytes:
                self.underestimates.append(Underestimate(label, modelled, stats.scratch_bytes))
        self._observe()
        self._trip_if_over(self.resident(), f"{label} post-call")

    def worst_underestimate(self) -> float:
        """The largest measured/modelled ratio seen, or 1.0 if the model always held."""
        return max((u.ratio for u in self.underestimates), default=1.0)

    def refresh(self, label: str, executor: Executor) -> None:
        """Apply one executor's delta to the cached total — never a full re-sum."""
        current = executor.resident_bytes()
        self.executor_bytes += current - self._cached.get(label, 0)
        self._cached[label] = current

    def forget(self, label: str) -> None:
        """A finished executor stops contributing; its state is gone."""
        self.executor_bytes -= self._cached.pop(label, 0)

    # -- internals ---------------------------------------------------------------

    def _observe(self) -> None:
        self.peak = max(self.peak, self.resident())

    def _trip_if_over(self, value: int, where: str) -> None:
        if self.budget is not None and value > self.budget:
            raise ResidentBudgetExceeded(f"{where}: {value} bytes over budget {self.budget}")
