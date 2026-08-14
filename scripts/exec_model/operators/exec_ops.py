"""The 1:1-per-batch operators: filter, project, sort, partial aggregate, limit, unload."""

from __future__ import annotations

from ..batch import CallStats
from ..executors import ExecExecutor, UnloadExecutor
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


class UnloadExec(UnloadExecutor):
    """`GpuBatch` in, `CpuBatch` out — the one place data crosses the boundary.

    `rows` is the range a root-adjacent limit narrowed this call to; on the GPU it is
    `peacock_result_from_handle`'s new arguments, so only those rows travel. Both sides are
    pandas here, so the slice costs nothing and the point is only that the call *records*
    what it was asked to move — `calls` is what the tests assert on, because a test on the
    rows that come back passes just as well when everything crossed the bus first.
    """

    def __init__(self, name: str = "unload"):
        self.name = name
        #: one entry per call: the RowRange asked for, or None for the whole batch
        self.calls: list[object] = []

    def resident_bytes(self) -> int:
        return 0

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return n_bytes

    def unload(self, batch: PandasBatch, rows=None):
        frame = batch.consume()
        self.calls.append(rows)
        if rows is not None:
            frame = frame.iloc[rows.offset : rows.stop]
        return PandasBatch(frame, f"{batch.tag}>{self.name}"), CallStats(scratch_bytes=0)
