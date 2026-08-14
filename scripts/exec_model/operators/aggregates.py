"""Aggregate specs and the two-phase decomposition.

The partial/final split is not an optimization here, it is the only correct shape once a
partition holds a stream: a partial runs per batch, the results are shuffled, and the final
merges them. Two rules from the wiki are load-bearing and both are easy to get wrong:

- **A mean-type aggregate must decompose to partial SUM + COUNT and never average partial
  means** (build-test.md's multi-GPU note). The partial therefore emits two state columns
  for one output column, which is why `AggSpec` distinguishes its state schema from its
  output schema at all.
- **Grouped aggregation includes the null group.** cuDF is called with
  `null_policy::INCLUDE` and dropping it loses tpcds q15's NULL `ca_zip` row
  (architecture.md, cuDF options). pandas `groupby` drops NaN keys by default, so every
  groupby here passes `dropna=False`.

`COUNT(*)` and `COUNT(col)` are different aggregates, not one with a flag: the first counts
rows, the second counts non-nulls, and picking the wrong one is always wrong for the other
(architecture.md's rolling-count row).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .frame import normalize

SUM = "sum"
MIN = "min"
MAX = "max"
COUNT = "count"
MEAN = "mean"


@dataclass(frozen=True)
class Agg:
    """`func(column) AS output`. `column=None` with COUNT means COUNT(*)."""

    func: str
    column: str | None
    output: str

    def __post_init__(self):
        if self.func not in (SUM, MIN, MAX, COUNT, MEAN):
            raise ValueError(f"unsupported aggregate {self.func!r}")
        if self.column is None and self.func != COUNT:
            raise ValueError(f"{self.func} needs a column")

    @property
    def state_columns(self) -> list[str]:
        """The partial phase's output columns for this aggregate."""
        if self.func == MEAN:
            return [f"{self.output}$sum", f"{self.output}$count"]
        return [self.output]


def _partial_frame(group: pd.DataFrame, aggs: list[Agg]) -> dict:
    out = {}
    for agg in aggs:
        if agg.func == MEAN:
            out[f"{agg.output}$sum"] = group[agg.column].sum()
            out[f"{agg.output}$count"] = group[agg.column].count()
        elif agg.func == COUNT:
            # COUNT(*) counts rows; COUNT(col) counts non-nulls.
            out[agg.output] = len(group) if agg.column is None else group[agg.column].count()
        elif agg.func == SUM:
            out[agg.output] = group[agg.column].sum()
        elif agg.func == MIN:
            out[agg.output] = group[agg.column].min()
        else:
            out[agg.output] = group[agg.column].max()
    return out


def _final_frame(group: pd.DataFrame, aggs: list[Agg]) -> dict:
    """Merge state columns. Sums of partial sums; never a mean of partial means."""
    out = {}
    for agg in aggs:
        if agg.func == MEAN:
            total = group[f"{agg.output}$sum"].sum()
            count = group[f"{agg.output}$count"].sum()
            out[agg.output] = total / count if count else float("nan")
        elif agg.func in (SUM, COUNT):
            out[agg.output] = group[agg.output].sum()
        elif agg.func == MIN:
            out[agg.output] = group[agg.output].min()
        else:
            out[agg.output] = group[agg.output].max()
    return out


def _apply(
    frame: pd.DataFrame, keys: list[str], aggs: list[Agg], builder, out_columns: list[str]
) -> pd.DataFrame:
    if not keys:
        return normalize(pd.DataFrame([builder(frame, aggs)]))
    rows = []
    # dropna=False keeps the null group — cuDF's null_policy::INCLUDE.
    # sort=True gives a deterministic group order, which matters because the goldens
    # compare rendered output and float sums are order-sensitive.
    for key_values, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        rows.append(dict(zip(keys, key_values)) | builder(group, aggs))
    if not rows:
        # The empty frame must carry the columns THIS PHASE declares. Getting it wrong is
        # not cosmetic: a partial's state columns leaking out of a final produces a frame
        # `cudf::concatenate` would reject, and an all-empty lane is exactly where it shows.
        return pd.DataFrame({c: [] for c in out_columns})
    return normalize(pd.DataFrame(rows))


def partial(frame: pd.DataFrame, keys: list[str], aggs: list[Agg]) -> pd.DataFrame:
    """Per-batch partial aggregation; emits state columns."""
    columns = keys + [c for agg in aggs for c in agg.state_columns]
    return _apply(frame, keys, aggs, _partial_frame, columns)


def final(frame: pd.DataFrame, keys: list[str], aggs: list[Agg]) -> pd.DataFrame:
    """Merge partial state into the declared outputs."""
    return _apply(frame, keys, aggs, _final_frame, keys + [agg.output for agg in aggs])


def single(frame: pd.DataFrame, keys: list[str], aggs: list[Agg]) -> pd.DataFrame:
    """The 1-partition single-batch shortcut — and the oracle the split is checked against."""
    return final(partial(frame, keys, aggs), keys, aggs)
