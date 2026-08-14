"""`start..limit` — the interval, and the per-batch decision the driver makes from it.

A limit is not a node on its common path. The reason is in the spec's lowering rule: a
flatbuffer node's skip/fetch are frozen per seq, so a per-batch `GpuLimit` call would apply
the *same* bounds to every batch — two batches would yield twice the limit — and the
correct right bound for the last batch is a runtime value no frozen node can carry. Legacy
never meets this because a legacy partition is one batch.

So which lowering applies is a question about position:

- **Root-adjacent** — the limit feeds only `GpuUnload`. There is then no limit node at all:
  `skip`/`fetch` are properties of the unload, which is where they belong, because a limit
  over a stream about to leave the device is a statement about *which rows are worth moving
  across the boundary*. The driver counts rows arriving at the sink and, per batch, either
  releases the handle without unloading it, unloads a row range, or unloads the whole
  thing. That is the point of the whole arrangement: trimming after unload would ship the
  `skip` prefix across PCIe and drop it, and that prefix is unbounded.
- **Mid-plan** — the limit's output feeds further work. It stays a real node over a
  **one-partition** input (`GpuMergePartitions` beneath, which the node checks in
  `validate_schemas_and_partitions`), and it streams: outside the interval a batch is
  released without a call, inside it is forwarded untouched, and only the two that straddle
  its ends are sliced through `peacock_executor_slice_handle`
  (`accumulators.LimitStream`). Its input is emphatically *not* required to be one batch —
  that would put a `GpuCoalesceAllBatches` underneath and make
  `... JOIN (SELECT * FROM orders LIMIT 100)` read the whole of orders for a hundred rows —
  and it holds nothing, whatever the offset.

`fetch=None` is a pure offset: no stop, so it never satisfies and can only ever drop and
trim.

**Counting is across lanes.** An `Unload` executor is per lane, so only the driver can hold
the count — which is also why the range is a call argument rather than executor state. It
does mean an unordered `LIMIT` returns different rows at different partition counts; that
is settled by the spec's scope note, since the SQL does not determine those rows and every
result golden is named per plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RowRange:
    """Rows `[offset, offset+length)` of one batch.

    Straight through to `peacock_result_from_handle`'s new arguments — the export reads
    that range instead of the whole table, which is why the trim costs no device work and
    no second handle.
    """

    offset: int
    length: int

    @property
    def stop(self) -> int:
        return self.offset + self.length

    def covers(self, n_rows: int) -> bool:
        """The whole batch, so the call needs no range at all."""
        return self.offset == 0 and self.length == n_rows


@dataclass(frozen=True)
class RowInterval:
    """`skip` rows dropped, then at most `fetch` kept. `fetch=None` is unbounded."""

    skip: int = 0
    fetch: int | None = None

    def __post_init__(self):
        if self.skip < 0:
            raise ValueError("skip cannot be negative")
        if self.fetch is not None and self.fetch < 0:
            raise ValueError("fetch cannot be negative")

    @property
    def stop(self) -> int | None:
        """The row after the last one wanted, counting from the start of the stream."""
        return None if self.fetch is None else self.skip + self.fetch

    def satisfied_by(self, seen: int) -> bool:
        """True once no further row could change the answer — what `is_satisfied` asks."""
        return self.stop is not None and seen >= self.stop

    def range_of(self, seen: int, n_rows: int) -> RowRange | None:
        """Which rows of the next batch to unload, or None to release it without a call.

        `seen` is how many rows of the stream have already gone past this node — the
        driver's cross-lane count. None is the answer for every batch of the `skip`
        prefix, which is the case with no bound on how much it saves.

        A range that covers the whole batch is still a range here; the driver collapses it
        to "no range" at the call, so the ordinary case passes nothing extra to the fetch.
        """
        start = max(self.skip - seen, 0)
        stop = n_rows if self.stop is None else min(n_rows, self.stop - seen)
        if start >= stop:
            return None
        return RowRange(start, stop - start)


#: What a node with no interval hands to `unload`: the whole batch.
WHOLE_BATCH: RowRange | None = None
