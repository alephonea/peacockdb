"""The 1:1-per-batch operators: filter, project, sort, partial aggregate, limit, unload."""

from __future__ import annotations

from ..batch import CallStats
from ..executors import ExecExecutor
from . import aggregates
from .expressions import Expr, project as project_exprs
from .frame import PandasBatch, no_scratch, scratch_of


class _Exec(ExecExecutor):
    """No state between calls, so residency is zero and scratch is the input's size."""

    def resident_bytes(self) -> int:
        return 0

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return n_bytes


class FilterExec(_Exec):
    """`cudf::apply_boolean_mask`. The mask is one expression over the input."""

    def __init__(self, predicate: Expr, name: str = "filter"):
        self.predicate = predicate
        self.name = name

    def exec(self, batch: PandasBatch):
        frame = batch.consume()
        mask = self.predicate.evaluate(frame)
        # fillna(False): a null predicate is not true, which is SQL's rule and cuDF's.
        applied = mask.fillna(False).astype(bool)
        out = frame[applied]
        return PandasBatch(out, f"{batch.tag}>{self.name}"), scratch_of(applied.to_frame())


class ProjectExec(_Exec):
    """An expression list; output column order is the list order."""

    def __init__(self, exprs: list[Expr], name: str = "project"):
        self.exprs = exprs
        self.name = name

    def exec(self, batch: PandasBatch):
        frame = batch.consume()
        out = project_exprs(frame, self.exprs)   # the columns built ARE the output
        return PandasBatch(out, f"{batch.tag}>{self.name}"), no_scratch()


class SortExec(_Exec):
    """Per-batch sort, optional per-batch top-N.

    `ascending` and `na_position` are both passed explicitly — they are `cudf::order` and
    `cudf::null_order`, two separate arguments, and the sort here must agree with the merge
    in `accumulators.py` or a k-way merge would order differently from the sort feeding it.
    """

    def __init__(self, by: list[str], ascending=None, nulls_first=False, fetch=None, name="sort"):
        self.by = by
        self.ascending = [True] * len(by) if ascending is None else list(ascending)
        self.nulls_first = nulls_first
        self.fetch = fetch
        self.name = name

    def exec(self, batch: PandasBatch):
        frame = batch.consume()
        out = frame.sort_values(
            by=self.by,
            ascending=self.ascending,
            na_position="first" if self.nulls_first else "last",
            kind="stable",
        )
        sorted_full = out
        if self.fetch is not None:
            out = out.iloc[: self.fetch]
        # A top-N sorts everything and keeps a prefix; the discarded tail is the scratch.
        return PandasBatch(out, f"{batch.tag}>{self.name}"), scratch_of(sorted_full.iloc[len(out):])


class PartialAggregateExec(_Exec):
    """One batch in, its partial state out — `GpuAggregate[final=false]`."""

    def __init__(self, keys: list[str], aggs: list[aggregates.Agg], name: str = "agg_partial"):
        self.keys = keys
        self.aggs = aggs
        self.name = name

    def exec(self, batch: PandasBatch):
        frame = batch.consume()
        out = aggregates.partial(frame, self.keys, self.aggs)
        return PandasBatch(out, f"{batch.tag}>{self.name}"), no_scratch()


class LimitExec(_Exec):
    """The mid-plan limit lowering: exact bounds over an already-coalesced input.

    Only correct on a single batch. The root-adjacent case is driver logic that counts
    rows and stops pulling — deliberately not an executor, because a per-batch call with
    frozen bounds would truncate every batch to the same interval.
    """

    def __init__(self, skip: int = 0, fetch: int | None = None, name: str = "limit"):
        self.skip = skip
        self.fetch = fetch
        self.name = name

    def exec(self, batch: PandasBatch):
        frame = batch.consume()
        stop = None if self.fetch is None else self.skip + self.fetch
        out = frame.iloc[self.skip : stop]      # a zero-copy slice
        return PandasBatch(out, f"{batch.tag}>{self.name}"), no_scratch()


class UnloadExec(_Exec):
    """`GpuBatch` in, `CpuBatch` out. Both are pandas here, so this is identity —
    the node exists because on the GPU it is the one place data crosses the boundary."""

    def __init__(self, name: str = "unload"):
        self.name = name

    def exec(self, batch: PandasBatch):
        frame = batch.consume()
        return PandasBatch(frame, f"{batch.tag}>{self.name}"), CallStats(scratch_bytes=0)
