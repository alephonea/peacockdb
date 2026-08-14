"""Error types for the batch-partitioned execution-model prototype."""


class PlanError(Exception):
    """A plan is structurally invalid: arity, partition topology, batch layout."""


class DriverError(Exception):
    """The driver reached a state a valid plan cannot produce."""


class ResidentBudgetExceeded(Exception):
    """The accounted resident set crossed the budget; the query fails cleanly."""
