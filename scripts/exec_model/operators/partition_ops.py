"""The hash scatter — `GpuEmitPartitions`, one call per input batch, N outputs.

The real kernel is a bit-exact Spark murmur3 at seed 42, chosen so the CPU and GPU place
every row in the same partition by construction (architecture.md, "Rehash and the comet
hash"). crc32 stands in for it here: what the prototype needs is *a* deterministic hash
with the same structural rules, not comet's bits.

The structural rule that does matter is null handling. The kernel skips null columns, so a
row whose key columns are all null hashes to the seed alone and every such row lands in the
one partition `pmod(seed, N)`. That is comet-mandated and not a knob — it is why #137 wants
the planner to filter null keys out ahead of the shuffle on sides whose unmatched rows are
never emitted, rather than teach the kernel to scatter them.
"""

from __future__ import annotations

import zlib
from typing import Callable

import pandas as pd

from ..executors import PartitionEmitterExecutor
from .frame import PandasBatch, no_scratch

SEED = 42

#: (frame, keys, n_partitions) → one partition id per row. The shape every placement
#: function has, real or deliberately degenerate.
HashFn = Callable[[pd.DataFrame, "list[str]", int], "list[int]"]


def _hash_value(value) -> bytes:
    if isinstance(value, bool):
        return b"b" + bytes([value])
    if isinstance(value, int):
        return b"i" + int(value).to_bytes(8, "little", signed=True)
    if isinstance(value, float):
        return b"f" + repr(value).encode()
    return b"s" + str(value).encode()


def row_digests(frame: pd.DataFrame, keys: list[str]) -> list[int]:
    """One digest per row, before it is reduced to a partition id.

    Split out from `partition_ids` so an alternative placement can be a function of the
    digest rather than a function of the row: whatever a test does with the result, equal
    keys must still give equal digests, and that is what keeps co-location true.
    """
    digests = []
    columns = [frame[key] for key in keys]
    for position in range(len(frame)):
        digest = SEED
        for column in columns:
            value = column.iloc[position]
            if pd.isna(value):
                continue  # comet skips null columns — all-null keys collapse to one lane
            digest = zlib.crc32(_hash_value(value), digest)
        digests.append(digest)
    return digests


def partition_ids(frame: pd.DataFrame, keys: list[str], n_partitions: int) -> list[int]:
    """One partition id per row. Null key columns are skipped, not hashed."""
    return [digest % n_partitions for digest in row_digests(frame, keys)]


# -- degenerate placements --------------------------------------------------------
#
# Every one of these is still a pure function of the key columns, which is the only
# property the engine actually depends on: a shuffle exists to co-locate equal keys, and
# nothing above it cares how lanes are loaded. Skew and collapse are therefore legal
# hashes, not corrupt ones, and a plan that only works under a well-spread hash is wrong.


def skewed_ids(frame: pd.DataFrame, keys: list[str], n_partitions: int) -> list[int]:
    """Three keys in four land in lane 0; the rest spread over the others."""
    if n_partitions == 1:
        return [0] * len(frame)
    return [
        0 if digest % 4 else 1 + (digest // 4) % (n_partitions - 1)
        for digest in row_digests(frame, keys)
    ]


def first_lane_ids(frame: pd.DataFrame, keys: list[str], n_partitions: int) -> list[int]:
    """Everything to lane 0 — the worst skew a hash can produce, and a legal one."""
    return [0] * len(frame)


def last_lane_ids(frame: pd.DataFrame, keys: list[str], n_partitions: int) -> list[int]:
    """Everything to the last lane, so lane 0 is the empty one."""
    return [n_partitions - 1] * len(frame)


class EmitPartitions(PartitionEmitterExecutor):
    """1 → N per batch. Returns exactly N frames, some empty; the driver drops those."""

    def __init__(
        self,
        keys: list[str],
        n_partitions: int,
        name: str = "emit",
        hash_fn: HashFn | None = None,
    ):
        self.keys = keys
        self.n_partitions = n_partitions
        self.name = name
        self.hash_fn = hash_fn or partition_ids

    def resident_bytes(self) -> int:
        return 0  # streaming: nothing survives the call

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return n_bytes  # the scatter builds one output the size of the input

    def emit(self, batch: PandasBatch):
        frame = batch.consume()
        ids = self.hash_fn(frame, self.keys, self.n_partitions)
        outputs = []
        for lane in range(self.n_partitions):
            mask = [pid == lane for pid in ids]
            piece = frame[pd.Series(mask, index=frame.index)] if len(frame) else frame
            outputs.append(PandasBatch(piece, f"{batch.tag}>{self.name}.p{lane}"))
        return outputs, no_scratch()
