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

import pandas as pd

from ..executors import PartitionEmitterExecutor
from .frame import PandasBatch, no_scratch

SEED = 42


def _hash_value(value) -> bytes:
    if isinstance(value, bool):
        return b"b" + bytes([value])
    if isinstance(value, int):
        return b"i" + int(value).to_bytes(8, "little", signed=True)
    if isinstance(value, float):
        return b"f" + repr(value).encode()
    return b"s" + str(value).encode()


def partition_ids(frame: pd.DataFrame, keys: list[str], n_partitions: int) -> list[int]:
    """One partition id per row. Null key columns are skipped, not hashed."""
    ids = []
    columns = [frame[key] for key in keys]
    for position in range(len(frame)):
        digest = SEED
        for column in columns:
            value = column.iloc[position]
            if pd.isna(value):
                continue  # comet skips null columns — all-null keys collapse to one lane
            digest = zlib.crc32(_hash_value(value), digest)
        ids.append(digest % n_partitions)
    return ids


class EmitPartitions(PartitionEmitterExecutor):
    """1 → N per batch. Returns exactly N frames, some empty; the driver drops those."""

    def __init__(self, keys: list[str], n_partitions: int, name: str = "emit"):
        self.keys = keys
        self.n_partitions = n_partitions
        self.name = name

    def resident_bytes(self) -> int:
        return 0  # streaming: nothing survives the call

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return n_bytes  # the scatter builds one output the size of the input

    def emit(self, batch: PandasBatch):
        frame = batch.consume()
        ids = partition_ids(frame, self.keys, self.n_partitions)
        outputs = []
        for lane in range(self.n_partitions):
            mask = [pid == lane for pid in ids]
            piece = frame[pd.Series(mask, index=frame.index)] if len(frame) else frame
            outputs.append(PandasBatch(piece, f"{batch.tag}>{self.name}.p{lane}"))
        return outputs, no_scratch()
