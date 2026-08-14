"""Resident accounting and the enforcer.

    resident = Σ byte_size of driver-held in-flight batches
             + Σ resident_bytes() over live executors

Prevention is a plan-time concern; this is the run-time half. The contract is "fail
cleanly when the accounted peak exceeds the budget", not "the budget is never exceeded".
"""

from __future__ import annotations

from .errors import ResidentBudgetExceeded
from .executors import Executor


class ResidentAccountant:
    """`budget=None` accounts and reports without ever tripping."""

    def __init__(self, budget: int | None = None):
        self.budget = budget
        self.in_flight_bytes = 0
        self.peak = 0
        self._executors: list[tuple[str, Executor]] = []

    def register(self, label: str, executor: Executor) -> None:
        self._executors.append((label, executor))

    def add_in_flight(self, n_bytes: int) -> None:
        self.in_flight_bytes += n_bytes
        self._observe()

    def remove_in_flight(self, n_bytes: int) -> None:
        self.in_flight_bytes -= n_bytes

    def executor_bytes(self) -> int:
        return sum(executor.resident_bytes() for _, executor in self._executors)

    def resident(self) -> int:
        return self.in_flight_bytes + self.executor_bytes()

    def check_call(self, label: str, executor: Executor, n_rows: int, n_bytes: int) -> None:
        """Pre-call check. Calls with no input batch pass 0 rows, 0 bytes."""
        projected = self.resident() + executor.scratch_bytes(n_rows, n_bytes)
        self._trip_if_over(projected, f"{label} pre-call")

    def settle(self, label: str) -> None:
        """Post-call check, once outputs are enqueued and inputs are gone."""
        self._observe()
        self._trip_if_over(self.resident(), f"{label} post-call")

    def _observe(self) -> None:
        self.peak = max(self.peak, self.resident())

    def _trip_if_over(self, value: int, where: str) -> None:
        if self.budget is not None and value > self.budget:
            raise ResidentBudgetExceeded(f"{where}: {value} bytes over budget {self.budget}")
