"""The loader, and the row-group → (partition, batch) policy it executes.

`partition_row_groups` is a prototype of the spec's `ParquetBatchPartitioner` (task T2):
one pure function, computed once at plan time, whose output everything else consumes
verbatim. Row groups stand in for parquet's, since there is no parquet here — the shape of
the mapping is what matters, not where the rows came from.
"""

from __future__ import annotations

import pandas as pd

from ..executors import SourceExecutor
from .frame import PandasBatch, measured


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


class TableSource(SourceExecutor):
    """Reads one lane's batches out of a frame, per the partitioner's mapping."""

    def __init__(
        self,
        frame: pd.DataFrame,
        row_groups: list[tuple[int, int]],
        batches: list[list[int]],
        name: str,
        lane: int,
    ):
        self.frame = frame
        self.row_groups = row_groups
        self.batches = list(batches)
        self.name = name
        self.lane = lane
        self.emitted = 0

    def resident_bytes(self) -> int:
        return 0  # the source holds no state between calls; the table is the input

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        if self.emitted >= len(self.batches):
            return 0
        rows = sum(self.row_groups[g][1] for g in self.batches[self.emitted])
        return rows * max(1, len(self.frame.columns)) * 8

    def next_batch(self):
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
        return PandasBatch(piece, tag), measured(piece)
