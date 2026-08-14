"""Accumulators: the nodes that hold rows back and emit at a defined point.

These are where mandatory residency lives, so each one reports its held bytes through
`resident_bytes()` — that is what the enforcer sums, and it is the difference between a
node the memory model can see and one it cannot.
"""

from __future__ import annotations

import pandas as pd

from ..batch import CallStats
from ..executors import BatchAccumulatorExecutor, LaneEvent, PartitionAccumulatorExecutor
from . import aggregates
from .frame import PandasBatch, concatenate, no_scratch, scratch_of


class _Held:
    """Shared bookkeeping: the frames an accumulator is sitting on."""

    def __init__(self):
        self.frames: list[pd.DataFrame] = []
        self.tags: list[str] = []
        self.held_bytes = 0

    def add(self, batch: PandasBatch) -> None:
        self.held_bytes += batch.byte_size()
        self.tags.append(batch.tag)
        self.frames.append(batch.consume())

    def take(self):
        frames, tags = self.frames, self.tags
        self.frames, self.tags, self.held_bytes = [], [], 0
        return frames, tags


class CoalesceAllBatches(BatchAccumulatorExecutor):
    """`cudf::concatenate` over a partition's batches, emitted at done.

    Emits exactly one batch even when it received none — an empty one. The join's build
    lane requires precisely one batch, and a lane that produced nothing must still deliver
    that, or the driver cannot tell an empty build side from a plan error.
    """

    def __init__(self, name: str = "coalesce_all", schema: list[str] | None = None):
        self.name = name
        self.schema = schema
        self._held = _Held()

    def resident_bytes(self) -> int:
        return self._held.held_bytes

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return self._held.held_bytes + n_bytes

    def accumulate_and_fetch(self, batch: PandasBatch):
        self._held.add(batch)
        return [], CallStats(scratch_bytes=0)

    def mark_done_and_fetch(self):
        frames, tags = self._held.take()
        if frames:
            out = concatenate(frames)
        elif self.schema is not None:
            out = pd.DataFrame({column: [] for column in self.schema})
        else:
            raise ValueError(
                f"{self.name}: no batches and no declared schema, so the empty batch the "
                "join's build lane requires cannot be built"
            )
        return [PandasBatch(out, f"[{'+'.join(tags)}]>{self.name}")], no_scratch()


class AggregateBatches(BatchAccumulatorExecutor):
    """`GpuAggregateBatches` — merges pre-aggregated batches, emits at done.

    `final=False` re-partials (compacting a partition's batches without finishing them);
    `final=True` produces the declared outputs. Compaction is what keeps the held bytes
    bounded by the group cardinality rather than by the input size.
    """

    def __init__(self, keys, aggs, final: bool, name: str = "agg_batches"):
        self.keys = keys
        self.aggs = aggs
        self.final = final
        self.name = name
        self._state: pd.DataFrame | None = None
        self.tags: list[str] = []

    def resident_bytes(self) -> int:
        if self._state is None:
            return 0
        return int(self._state.memory_usage(index=False, deep=True).sum())

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return self.resident_bytes() + n_bytes

    def accumulate_and_fetch(self, batch: PandasBatch):
        self.tags.append(batch.tag)
        frame = batch.consume()
        merged = frame if self._state is None else concatenate([self._state, frame])
        # Re-partial on every arrival so the held state stays at group cardinality.
        self._state = aggregates.partial(merged, self.keys, self._restated())
        return [], CallStats(scratch_bytes=int(merged.memory_usage(index=False).sum()))

    def mark_done_and_fetch(self):
        state = self._state
        tags = self.tags
        self._state, self.tags = None, []
        if state is None:
            state = pd.DataFrame(
                {c: [] for c in self.keys + [c for a in self.aggs for c in a.state_columns]}
            )
        out = aggregates.final(state, self.keys, self.aggs) if self.final else state
        return [PandasBatch(out, f"[{'+'.join(tags)}]>{self.name}")], no_scratch()

    def _restated(self):
        """Merging state columns is a sum/min/max over the state, not over raw inputs."""
        restated = []
        for agg in self.aggs:
            if agg.func == aggregates.MEAN:
                restated.append(aggregates.Agg(aggregates.SUM, f"{agg.output}$sum", f"{agg.output}$sum"))
                restated.append(
                    aggregates.Agg(aggregates.SUM, f"{agg.output}$count", f"{agg.output}$count")
                )
            elif agg.func in (aggregates.SUM, aggregates.COUNT):
                restated.append(aggregates.Agg(aggregates.SUM, agg.output, agg.output))
            else:
                restated.append(aggregates.Agg(agg.func, agg.output, agg.output))
        return restated


class AccumulateBatchesAndSort(BatchAccumulatorExecutor):
    """Accumulates sorted batches, one merge at done — `GpuAccumulateBatchesAndSort`.

    cuDF has no streaming merge, so the whole input is resident before anything is emitted;
    the ranged alternative is #138. Inputs are already `BatchSorted`, so a stable sort over
    the concatenation is the same answer a k-way merge would give.
    """

    def __init__(self, by, ascending=None, nulls_first=False, fetch=None, name="accum_sort"):
        self.by = by
        self.ascending = [True] * len(by) if ascending is None else list(ascending)
        self.nulls_first = nulls_first
        self.fetch = fetch
        self.name = name
        self._held = _Held()

    def resident_bytes(self) -> int:
        return self._held.held_bytes

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return 2 * (self._held.held_bytes + n_bytes)  # merge materializes inputs + output

    def accumulate_and_fetch(self, batch: PandasBatch):
        self._held.add(batch)
        return [], CallStats(scratch_bytes=0)

    def mark_done_and_fetch(self):
        frames, tags = self._held.take()
        if not frames:
            return [], CallStats(scratch_bytes=0)
        out = concatenate(frames).sort_values(
            by=self.by,
            ascending=self.ascending,
            na_position="first" if self.nulls_first else "last",
            kind="stable",
        )
        merged = out
        if self.fetch is not None:
            out = out.iloc[: self.fetch]
        return ([PandasBatch(out, f"[{'+'.join(tags)}]>{self.name}")],
                scratch_of(merged.iloc[len(out):]))


class MergeSortedPartitions(PartitionAccumulatorExecutor):
    """N sorted lanes → one sorted batch — `GpuMergeSortedPartitions`.

    Input frames are passed partition-major regardless of arrival order, which is the
    determinism rule the spec pins: `cudf::merge` breaks ties by input order, so an
    arrival-ordered merge would reorder equal keys run to run.
    """

    def __init__(self, n_lanes, by, ascending=None, nulls_first=False, fetch=None, name="merge_sorted"):
        self.n_lanes = n_lanes
        self.by = by
        self.ascending = [True] * len(by) if ascending is None else list(ascending)
        self.nulls_first = nulls_first
        self.fetch = fetch
        self.name = name
        self.per_lane: dict[int, list[pd.DataFrame]] = {}
        self.tags: list[str] = []
        self.done_lanes: set[int] = set()
        self.held_bytes = 0

    def resident_bytes(self) -> int:
        return self.held_bytes

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return 2 * (self.held_bytes + n_bytes)

    def accumulate_and_fetch(self, partition: int, event: LaneEvent):
        if not event.is_done:
            self.held_bytes += event.batch.byte_size()
            self.tags.append(event.batch.tag)
            self.per_lane.setdefault(partition, []).append(event.batch.consume())
            return [], CallStats(scratch_bytes=0)

        self.done_lanes.add(partition)
        if len(self.done_lanes) < self.n_lanes:
            return [], CallStats(scratch_bytes=0)

        frames = [f for lane in sorted(self.per_lane) for f in self.per_lane[lane]]
        self.per_lane, self.held_bytes = {}, 0
        if not frames:
            return [], CallStats(scratch_bytes=0)
        out = concatenate(frames).sort_values(
            by=self.by,
            ascending=self.ascending,
            na_position="first" if self.nulls_first else "last",
            kind="stable",
        )
        merged = out
        if self.fetch is not None:
            out = out.iloc[: self.fetch]
        return ([PandasBatch(out, f"[{'+'.join(self.tags)}]>{self.name}")],
                scratch_of(merged.iloc[len(out):]))
