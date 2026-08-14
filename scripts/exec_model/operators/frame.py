"""The pandas-backed batch, and the rules that keep pandas inside cuDF's vocabulary.

pandas is the backend because it is available everywhere; cuDF is what the real executors
will call. Those two disagree in ways that would let a prototype operator "work" and its
C++ twin fail, so every operator here is written against the intersection, and the
divergences are named rather than avoided by accident.

**The subset rules.** Each one exists because breaking it produces a prototype that cannot
be ported:

1. **No index.** A `cudf::table` is an ordered collection of columns and nothing else.
   pandas carries an Index that silently aligns operands in arithmetic and reappears after
   a filter. Every frame that enters or leaves an operator here is passed through
   `normalize`, which resets it. Without that, `a[mask] + b[mask]` aligns on original row
   labels and quietly produces a correct-looking answer no cuDF kernel would give.
2. **No `apply`, no python callables.** cuDF evaluates an AST of typed operations. Anything
   expressible only as a row-wise python lambda has no counterpart, so `expressions.py`
   offers a fixed operator set and nothing else.
3. **Explicit null placement on every sort.** `cudf::order` and `cudf::null_order` are
   separate arguments; pandas defaults `na_position="last"`. Sorts here always pass both.
4. **Explicit null equality on every join.** pandas `merge` matches NaN keys to each other.
   SQL does not, and `GpuHashJoin.null_equals_null` carries the choice per join
   (architecture.md). Joins here take the flag and implement both meanings.
5. **Concatenate requires identical column names in identical order**, as
   `cudf::concatenate` requires identical types — so mismatches raise here rather than
   being reconciled by pandas' union-of-columns behaviour.

**Known divergences that remain**, because pandas cannot express them: no decimal128 (the
real engine's scale handling is the #55/#56 bug class and is out of scope here), and
integer columns holding nulls become float64 in pandas, which cuDF would keep as a
nullable integer.
"""

from __future__ import annotations

import pandas as pd

from ..batch import Batch, CallStats


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the index — rule 1. Every operator output goes through this."""
    return frame.reset_index(drop=True)


class PandasBatch(Batch):
    """One table's worth of rows. `!Clone` in Rust; consumption is one-shot here."""

    def __init__(self, frame: pd.DataFrame, tag: str = ""):
        self.frame = normalize(frame)
        self.tag = tag
        self.consumed = False

    def num_rows(self) -> int:
        return len(self.frame)

    def byte_size(self) -> int:
        return int(self.frame.memory_usage(index=False, deep=True).sum())

    def consume(self) -> pd.DataFrame:
        """Take the frame. A second call is a driver bug — on the GPU the handle is gone."""
        if self.consumed:
            raise AssertionError(f"batch {self.tag!r} consumed twice")
        self.consumed = True
        return self.frame

    def __repr__(self) -> str:
        return f"PandasBatch({self.tag!r}, rows={self.num_rows()}, cols={list(self.frame.columns)})"


def concatenate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """`cudf::concatenate` — rule 5: identical column names in identical order."""
    if not frames:
        raise ValueError("concatenate of nothing: the caller must handle the empty case")
    first = list(frames[0].columns)
    for other in frames[1:]:
        if list(other.columns) != first:
            raise ValueError(f"concatenate column mismatch: {first} vs {list(other.columns)}")
    return normalize(pd.concat(frames, ignore_index=True))


def empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    """A zero-row frame with the same columns and dtypes."""
    return frame.iloc[0:0].copy()


def measured(frame: pd.DataFrame) -> CallStats:
    """Scratch is measurable on this backend, so it is reported (`None` on the GPU)."""
    return CallStats(scratch_bytes=int(frame.memory_usage(index=False, deep=True).sum()))
