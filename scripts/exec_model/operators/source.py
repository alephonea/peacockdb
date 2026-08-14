"""The loader, and the row-group → (partition, batch) policy it executes.

`partition_row_groups` is a prototype of the spec's `ParquetBatchPartitioner` (task T2):
one pure function, computed once at plan time, whose output everything else consumes
verbatim. Row groups stand in for parquet's, since there is no parquet here — the shape of
the mapping is what matters, not where the rows came from.
"""

from __future__ import annotations

import random

import pandas as pd

from ..executors import SourceExecutor
from .frame import PandasBatch, no_scratch


def split_row_groups(total_rows: int, rows_per_group: int) -> list[tuple[int, int]]:
    """Cut a table into row groups: (start, length) pairs, file order."""
    if rows_per_group <= 0:
        raise ValueError("rows_per_group must be positive")
    return [
        (start, min(rows_per_group, total_rows - start))
        for start in range(0, total_rows, rows_per_group)
    ]


def partition_row_groups(
    row_groups: list[tuple[int, int]],
    n_partitions: int,
    target_batch_rows: int | None,
) -> list[list[list[int]]]:
    """survivors → partitions → batches → row-group indices.

    Policy, matching the spec: survivors split into `n_partitions` contiguous chunks
    balanced by row count; within a chunk, consecutive row groups are packed greedily
    while the running row count stays under target. A single row group over target still
    becomes its own batch — one row group is the minimum granularity, and the planner
    always produces a plan (the enforcer owns the runtime consequence, #142).
    `target_batch_rows=None` is batching off: one batch per chunk.
    """
    if not row_groups:
        raise ValueError("empty survivor set: the caller must decide what an empty scan means")
    if n_partitions < 1:
        raise ValueError("n_partitions must be at least 1")

    total = sum(length for _, length in row_groups)
    chunks: list[list[int]] = []
    taken = 0
    index = 0
    for part in range(n_partitions):
        # Balance by rows, not by group count: the remaining rows split over the
        # remaining partitions.
        remaining_parts = n_partitions - part
        want = (total - taken + remaining_parts - 1) // remaining_parts
        chunk: list[int] = []
        got = 0
        while index < len(row_groups) and (got < want or not chunk):
            chunk.append(index)
            got += row_groups[index][1]
            index += 1
            if got >= want:
                break
        chunks.append(chunk)
        taken += got
    # Any tail from integer division lands in the last non-empty chunk.
    while index < len(row_groups):
        chunks[-1].append(index)
        index += 1

    partitions: list[list[list[int]]] = []
    for chunk in chunks:
        if target_batch_rows is None:
            partitions.append([list(chunk)] if chunk else [])
            continue
        batches: list[list[int]] = []
        current: list[int] = []
        rows = 0
        for group in chunk:
            if current and rows + row_groups[group][1] > target_batch_rows:
                batches.append(current)
                current, rows = [], 0
            current.append(group)
            rows += row_groups[group][1]
        if current:
            batches.append(current)
        partitions.append(batches)
    return partitions


def drain_lanes(
    mapping: list[list[list[int]]], lanes: "tuple[int, ...] | set[int]"
) -> list[list[list[int]]]:
    """Move the named lanes' batches elsewhere, leaving those lanes with nothing to emit.

    A lane can end up empty for reasons the planner does not control — a filter that keeps
    nothing, a hash that lands no key there, a table with fewer row groups than lanes — so
    an empty lane is an ordinary shape, not an error. This makes one on demand: the rows
    still come out, they just all come out somewhere else.
    """
    named = {lane for lane in lanes if 0 <= lane < len(mapping)}
    kept = [lane for lane in range(len(mapping)) if lane not in named]
    if not kept:
        raise ValueError("every lane cannot be empty: the rows have to go somewhere")
    out = [list(batches) for batches in mapping]
    for lane in sorted(named):
        out[kept[0]].extend(out[lane])
        out[lane] = []
    return out


class TableSource(SourceExecutor):
    """Reads one lane's batches out of a frame, per the partitioner's mapping.

    `empty_probability` makes the lane emit zero-row batches between its real ones. Those
    are not a test artefact: a filter that keeps nothing produces exactly this, and it is
    the input every downstream operator is least likely to have been written for. The
    injections are bounded (one per real batch plus one tail) so the lane still terminates,
    and driven by a seeded generator so a failure reproduces.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        row_groups: list[tuple[int, int]],
        batches: list[list[int]],
        name: str,
        lane: int,
        empty_probability: float = 0.0,
        seed: int = 0,
    ):
        self.frame = frame
        self.row_groups = row_groups
        self.batches = list(batches)
        self.name = name
        self.lane = lane
        self.emitted = 0
        self.empty_probability = empty_probability
        self.injected = 0
        self._budget = len(self.batches) + 1
        self._rng = random.Random(seed) if empty_probability > 0 else None

    def resident_bytes(self) -> int:
        return 0  # the source holds no state between calls; the table is the input

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        if self.emitted >= len(self.batches):
            return 0
        rows = sum(self.row_groups[g][1] for g in self.batches[self.emitted])
        return rows * max(1, len(self.frame.columns)) * 8

    def next_batch(self):
        if self._inject_empty():
            self.injected += 1
            tag = f"{self.name}.p{self.lane}.empty{self.injected}"
            return PandasBatch(self.frame.iloc[0:0], tag), no_scratch()
        if self.emitted >= len(self.batches):
            return None
        groups = self.batches[self.emitted]
        # Row groups within a batch are contiguous by construction, so one slice suffices —
        # the same reason cuDF's reader takes a row-group list rather than row ranges.
        start = self.row_groups[groups[0]][0]
        stop = self.row_groups[groups[-1]][0] + self.row_groups[groups[-1]][1]
        piece = self.frame.iloc[start:stop]
        tag = f"{self.name}.p{self.lane}.b{self.emitted}"
        self.emitted += 1
        return PandasBatch(piece, tag), no_scratch()   # the slice is the output

    def _inject_empty(self) -> bool:
        """Bounded, so the lane still terminates however the generator falls."""
        if self._rng is None or self.injected >= self._budget:
            return False
        return self._rng.random() < self.empty_probability
