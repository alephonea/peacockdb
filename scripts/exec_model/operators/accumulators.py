"""Accumulators: the nodes that hold rows back and emit at a defined point.

These are where mandatory residency lives, so each one reports its held bytes through
`resident_bytes()` — that is what the enforcer sums, and it is the difference between a
node the memory model can see and one it cannot.
"""

from __future__ import annotations

import pandas as pd

from ..batch import CallStats
from ..executors import BatchAccumulatorExecutor, LaneEvent, PartitionAccumulatorExecutor
from ..limit import RowInterval, RowRange
from . import aggregates
from .frame import PandasBatch, concatenate, empty_frame, no_scratch, scratch_of

# The contract every SingleBatch accumulator here honours: exactly one batch at done, even
# when nothing was accumulated (F7 — a join's build lane cannot tell an empty build side
# from a plan error otherwise). Building the empty one takes a typed `{column: dtype}`
# schema, because an empty column still has a type; empty input with no schema is a loud
# error rather than a silently absent or mistyped batch.


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

    `schema` is a `{column: dtype}` mapping, consulted only for the empty case — see the
    module contract above.
    """

    def __init__(self, name: str = "coalesce_all", schema: dict | None = None):
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
        else:
            out = _empty_single_batch(self.name, self.schema)
        return [PandasBatch(out, f"[{'+'.join(tags)}]>{self.name}")], no_scratch()


def _empty_single_batch(name: str, schema: dict | None) -> pd.DataFrame:
    """The module contract's empty case: one typed empty frame, or a loud miss."""
    if schema is None:
        raise ValueError(
            f"{name}: accumulated nothing and has no schema, so the single empty batch "
            "its SingleBatch output owes downstream cannot be built"
        )
    return empty_frame(schema)


class LimitStream(BatchAccumulatorExecutor):
    """`GpuLimit` mid-plan: streams, holding nothing.

    Per batch it does one of three things, from a running count of the rows that have gone
    past. A batch entirely outside `start..limit` is **released without a call** — the
    unbounded saving on an offset. A batch entirely inside is **forwarded untouched**, so
    the ordinary case costs nothing at all. Only the two that straddle the interval's ends
    are **sliced**, through `peacock_executor_slice_handle`, whose bounds are call
    arguments rather than plan constants; the copy is bounded by the rows kept, which are
    the rows about to be used anyway.

    That symbol is what makes this streaming rather than accumulating. Without it the
    bounds would have to come from the frozen fb node, which is only correct against a
    table starting at row 0 of the stream — so the node would have to hold the whole
    prefix, and `OFFSET 1000000 LIMIT 10` would hold a million rows to return ten. With it,
    residency is zero and the driver still stops the scan the moment the interval is
    covered.

    Input is one partition (`GpuMergePartitions` beneath, checked on the node) and any
    number of batches. Requiring one batch is what would put a `GpuCoalesceAllBatches`
    underneath and make `... JOIN (SELECT * FROM orders LIMIT 100)` read the whole table.
    """

    def __init__(self, skip: int = 0, fetch: int | None = None, name: str = "limit"):
        self.interval = RowInterval(skip, fetch)
        self.name = name
        self.seen = 0
        #: the ranges actually passed to a slice call — two per query at most
        self.sliced: list[RowRange] = []
        #: batches released without any call, and batches forwarded untouched
        self.dropped = 0
        self.passed = 0

    def resident_bytes(self) -> int:
        return 0    # it holds nothing; that is the point

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        # A slice materializes the rows it keeps, and those are the output, not a transient.
        return 0

    def accumulate_and_fetch(self, batch: PandasBatch):
        rows = self.interval.range_of(self.seen, batch.num_rows())
        self.seen += batch.num_rows()

        if rows is None:
            self.dropped += 1
            batch.consume()
            return [], CallStats(scratch_bytes=0)

        if rows.covers(batch.num_rows()):
            self.passed += 1
            frame = batch.consume()
            return [PandasBatch(frame, f"{batch.tag}>{self.name}")], CallStats(scratch_bytes=0)

        self.sliced.append(rows)
        out = batch.slice_rows(rows.offset, rows.length)
        batch.consume()
        return [out], CallStats(scratch_bytes=0)

    def mark_done_and_fetch(self):
        return [], no_scratch()    # nothing was ever held


class ReBatchToTarget(BatchAccumulatorExecutor):
    """`GpuCoalesceBatches(target)` — [#139](../../../llm-wiki/tickets.md#t139) — with the
    splitting half added.

    The ticket's node merges only, DataFusion-style, because its purpose is compacting
    post-filter fragments. This one also cuts, because the prototype wants a node that can
    make a stream's batches *any* shape: a target above the incoming size coalesces, a
    target below it splits, and one implementation covers both since both are the same
    "fill a buffer, hand out target-sized pieces" loop.

    It is the only accumulator here that emits mid-stream — everything else holds until
    done — so it is also the one that shows a `BatchAccumulator` need not be a pipeline
    breaker. Residency is bounded by the target rather than by the input.
    """

    def __init__(self, target_rows: int, name: str = "rebatch"):
        if target_rows < 1:
            raise ValueError("target_rows must be at least 1")
        self.target_rows = target_rows
        self.name = name
        self._buffer: pd.DataFrame | None = None
        self.emitted = 0

    def resident_bytes(self) -> int:
        if self._buffer is None:
            return 0
        return int(self._buffer.memory_usage(index=False, deep=True).sum())

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return self.resident_bytes() + n_bytes

    def accumulate_and_fetch(self, batch: PandasBatch):
        frame = batch.consume()
        self._buffer = frame if self._buffer is None else concatenate([self._buffer, frame])
        return self._cut(flush=False)

    def mark_done_and_fetch(self):
        return self._cut(flush=True)

    def _cut(self, flush: bool):
        buffer = self._buffer
        if buffer is None:
            return [], CallStats(scratch_bytes=0)
        outputs = []
        while len(buffer) >= self.target_rows:
            outputs.append(self._emit(buffer.iloc[: self.target_rows]))
            buffer = buffer.iloc[self.target_rows :]
        if flush and len(buffer):
            outputs.append(self._emit(buffer))
            buffer = buffer.iloc[0:0]
        # A held remainder is residency, not scratch: it is the state of the node.
        self._buffer = None if flush else buffer
        return outputs, CallStats(scratch_bytes=0)

    def _emit(self, frame: pd.DataFrame) -> PandasBatch:
        self.emitted += 1
        return PandasBatch(frame, f"{self.name}#{self.emitted}")


#: Held bytes at which an aggregate compacts. A stand-in: the real one comes from the
#: same budget rule that sizes loader batches, which the prototype does not model.
DEFAULT_COMPACT_BYTES = 1 << 20


class AggregateBatches(BatchAccumulatorExecutor):
    """`GpuAggregateBatches` — merges pre-aggregated batches, emits at done.

    `final=False` re-partials (compacting a partition's batches without finishing them);
    `final=True` produces the declared outputs.

    **Compaction runs on a byte threshold that doubles when it fails to pay.** The two
    obvious policies are each wrong in one regime. Compacting on every arrival keeps the
    held state at group cardinality, but it re-scans that state once per batch: when the
    groups are disjoint and nothing merges, the state grows every time and the total work
    is quadratic in the batch count. Never compacting holds every partial to the end,
    which is the whole input when the group cardinality is high.

    So: hold arrivals until they cross `threshold` bytes, compact once, then raise the
    threshold to twice what the compaction left behind. A low-cardinality aggregate leaves
    a small state, so the threshold never moves and residency stays near group
    cardinality on a fraction of the calls per-arrival would make. A high-cardinality one
    leaves a state as big as its input, so the threshold doubles away and the compactions
    land at geometrically growing sizes — total re-scan work linear in the rows that pass
    through, instead of quadratic — and it stops paying for a merge that merges nothing.
    Residency then grows, which is the honest answer for that shape; the enforcer is the
    backstop (#142).
    """

    def __init__(self, keys, aggs, final_exprs=None, name: str = "agg_batches",
                 schema: dict | None = None, compact_bytes: int = DEFAULT_COMPACT_BYTES):
        self.keys = keys
        self.aggs = aggs
        #: the node's `final` list, or None to emit state. Its presence is the only thing
        #: distinguishing a merging node from a finalizing one — there is no phase flag.
        self.final_exprs = final_exprs
        self.name = name
        #: `{column: dtype}` of THIS phase's output (state columns for final=False, keys +
        #: declared outputs for final=True); consulted only for the zero-input empty case
        self.schema = schema
        self.threshold = compact_bytes
        self._state: pd.DataFrame | None = None
        #: arrivals not yet folded into `_state`
        self._pending: list[pd.DataFrame] = []
        self._pending_bytes = 0
        self.tags: list[str] = []
        #: how many times the threshold was reached — what the tests assert on
        self.compactions = 0

    def _state_bytes(self) -> int:
        if self._state is None:
            return 0
        return int(self._state.memory_usage(index=False, deep=True).sum())

    def resident_bytes(self) -> int:
        return self._state_bytes() + self._pending_bytes

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return self.resident_bytes() + n_bytes

    def accumulate_and_fetch(self, batch: PandasBatch):
        self.tags.append(batch.tag)
        self._pending_bytes += batch.byte_size()
        self._pending.append(batch.consume())
        scratch = self._compact() if self.resident_bytes() >= self.threshold else 0
        return [], CallStats(scratch_bytes=scratch)

    def _compact(self) -> int:
        """Fold every pending arrival into the state. Returns the transient's size."""
        frames = ([] if self._state is None else [self._state]) + self._pending
        merged = frames[0] if len(frames) == 1 else concatenate(frames)
        self._state = aggregates.merge(merged, self.keys, self.aggs)
        self._pending, self._pending_bytes = [], 0
        self.compactions += 1
        # A compaction that did not shrink will not shrink next time either — the keys are
        # simply distinct — so raise the bar rather than re-scanning the same state again.
        self.threshold = max(self.threshold, 2 * self._state_bytes())
        return int(merged.memory_usage(index=False, deep=True).sum())

    def mark_done_and_fetch(self):
        if self._pending:
            self._compact()
        state, tags = self._state, self.tags
        self._state, self.tags = None, []
        if state is None:
            # Zero-input lane: the schema IS the output — no phase left to run over it.
            out = _empty_single_batch(self.name, self.schema)
        elif self.final_exprs is not None:
            out = aggregates.finalize(state, self.keys, self.aggs)
        else:
            out = state
        return [PandasBatch(out, f"[{'+'.join(tags)}]>{self.name}")], no_scratch()


class AccumulateBatchesAndSort(BatchAccumulatorExecutor):
    """Accumulates sorted batches, one merge at done — `GpuAccumulateBatchesAndSort`.

    cuDF has no streaming merge, so the whole input is resident before anything is emitted;
    the ranged alternative is #138. Inputs are already `BatchSorted`, so a stable sort over
    the concatenation is the same answer a k-way merge would give.
    """

    def __init__(self, by, ascending=None, nulls_first=False, fetch=None, name="accum_sort",
                 schema: dict | None = None):
        self.by = by
        self.ascending = [True] * len(by) if ascending is None else list(ascending)
        self.nulls_first = nulls_first
        self.fetch = fetch
        self.name = name
        self.schema = schema
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
            out = _empty_single_batch(self.name, self.schema)
            return [PandasBatch(out, f"[]>{self.name}")], no_scratch()
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

    def __init__(self, n_lanes, by, ascending=None, nulls_first=False, fetch=None,
                 name="merge_sorted", schema: dict | None = None):
        self.n_lanes = n_lanes
        self.by = by
        self.ascending = [True] * len(by) if ascending is None else list(ascending)
        self.nulls_first = nulls_first
        self.fetch = fetch
        self.name = name
        self.schema = schema
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
            out = _empty_single_batch(self.name, self.schema)
            return [PandasBatch(out, f"[]>{self.name}")], no_scratch()
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
