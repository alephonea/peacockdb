"""`GpuJoin` — build side set once, probe side streamed, per the capability matrix.

Which join types can stream their probe side, and what each call emits:

| type | per probe call | at finish |
|---|---|---|
| INNER | matches | nothing |
| RIGHT_OUTER (build is left) | matches + unmatched probe rows, batch-locally | nothing |
| LEFT_OUTER / FULL_OUTER | matches (+ unmatched probe for FULL) | unmatched build rows, null-padded |
| LEFT_SEMI / LEFT_ANTI | nothing | build rows with / without a match |
| CROSS | build × probe batch | nothing |

**The prototype cheats where the real implementation cannot.** "Which build rows matched
at least once across all probe batches" is kept here as a boolean array over the build
frame — free in-process, and exactly the thing that never crosses the C ABI
(`peacock_executor_execute_node` returns a table plus row counts and nothing else). The
real v1 accumulates each probe batch's key columns instead and runs one anti/semi join at
finish; the alternatives are a match bitmap out-param or a persistent `cudf::hash_join`
session. All three are #136. So this file proves the *shape* of the finish pass — when it
runs, what it emits, how it composes with the driver — and not its cost.

**Null key equality is explicit**, because pandas and SQL disagree: `merge` matches NaN
keys to each other. With `null_equals_null=False` (the SQL default that
`GpuHashJoin.null_equals_null` carries per join) a row with any null key matches nothing,
which is implemented by holding those rows out of the merge and treating them as unmatched.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from ..batch import CallStats
from ..executors import JoinExecutor
from .frame import PandasBatch, concatenate, no_scratch, scratch_of, normalize


class JoinType(Enum):
    INNER = "inner"
    LEFT_OUTER = "left_outer"
    RIGHT_OUTER = "right_outer"
    FULL_OUTER = "full_outer"
    LEFT_SEMI = "left_semi"
    LEFT_ANTI = "left_anti"
    CROSS = "cross"


#: Types whose finish pass needs "did this build row ever match".
_NEEDS_FINISH = {JoinType.LEFT_OUTER, JoinType.FULL_OUTER, JoinType.LEFT_SEMI, JoinType.LEFT_ANTI}
#: Types that may not stream their probe side (matrix row: "no — single-batch probe").
_SINGLE_BATCH_PROBE: set[JoinType] = set()

#: output rows / probe rows — the ratio `CardinalityEstimator` returns in gpu_rule.rs.
#: 1.0 is what the constant estimator gives today (#19).
TRIVIAL_FANOUT = 1.0
#: the two row-id marker columns the merged frame carries, int64 each
_MARKER_BYTES_PER_ROW = 16

_BUILD_KEY = "__build_row__"
_PROBE_KEY = "__probe_row__"


class HashJoin(JoinExecutor):
    def __init__(
        self,
        join_type: JoinType,
        build_keys: list[str],
        probe_keys: list[str],
        null_equals_null: bool = False,
        name: str = "join",
        fanout: float = TRIVIAL_FANOUT,
    ):
        if join_type is not JoinType.CROSS and len(build_keys) != len(probe_keys):
            raise ValueError("join key lists must have equal length")
        self.join_type = join_type
        self.build_keys = build_keys
        self.probe_keys = probe_keys
        self.null_equals_null = null_equals_null
        self.name = name
        # A node property, supplied by the optimizer at plan time and carried into the
        # executor at construction — which is how the estimate reaches scratch_bytes
        # without changing its signature.
        self.fanout = fanout
        self.build: pd.DataFrame | None = None
        self.matched: np.ndarray | None = None
        self.probe_calls = 0

    # -- Executor ---------------------------------------------------------------

    def resident_bytes(self) -> int:
        if self.build is None:
            return 0
        return int(self.build.memory_usage(index=False, deep=True).sum())

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        """Build side (rebuilt per probe call, #136) plus the merged frame.

        The merged frame is sized by the *output* cardinality, which the signature cannot
        derive — but it does not have to. `fanout` is a node property the optimizer
        supplies at plan time and the executor is constructed with, so the estimate is
        already in `&self` by the time this is called. That is what keeps the trait
        unchanged while the model stays correct.

        `fanout` is output rows / probe rows, the same ratio `CardinalityEstimator` returns
        in `gpu_rule.rs` — greater than one for a fan-out join, less for a filtering one.
        It defaults to `TRIVIAL_FANOUT`, matching today's constant estimator (#19).
        """
        merged_rows = n_rows * self.fanout
        if n_rows == 0:
            return self.resident_bytes()
        probe_row_bytes = n_bytes / n_rows
        build_rows = max(1, len(self.build)) if self.build is not None else 1
        build_row_bytes = self.resident_bytes() / build_rows
        # Each merged row carries both sides' columns plus the two marker columns.
        per_row = probe_row_bytes + build_row_bytes + _MARKER_BYTES_PER_ROW
        return int(self.resident_bytes() + merged_rows * per_row)

    # -- JoinExecutor -----------------------------------------------------------

    def set_build(self, batch: PandasBatch) -> CallStats:
        if self.build is not None:
            raise AssertionError(f"{self.name}: set_build called twice")
        frame = batch.consume()
        self.build = frame.copy()
        self.build[_BUILD_KEY] = np.arange(len(frame))
        self.matched = np.zeros(len(frame), dtype=bool)
        return no_scratch()   # the build frame becomes residency, not scratch

    def probe_and_fetch(self, batch: PandasBatch):
        if self.build is None:
            raise AssertionError(f"{self.name}: probed before set_build")
        self.probe_calls += 1
        probe = batch.consume()
        out = self._probe(probe)
        return (
            [PandasBatch(out, f"({self.name}#{self.probe_calls}⋈{batch.tag})")],
            scratch_of(*self._transients),
        )

    def finish_and_fetch(self):
        if self.join_type not in _NEEDS_FINISH:
            return [], CallStats(scratch_bytes=0)
        build = self._build_columns()
        if self.join_type is JoinType.LEFT_SEMI:
            out = build[self.matched]
        elif self.join_type is JoinType.LEFT_ANTI:
            out = build[~self.matched]
        else:  # LEFT_OUTER / FULL_OUTER: unmatched build rows, probe columns null-padded
            out = build[~self.matched]
            for column in self._probe_only_columns():
                out = out.assign(**{column: np.nan})
        return [PandasBatch(out, f"({self.name}⋈finish)")], no_scratch()

    # -- internals --------------------------------------------------------------

    def _build_columns(self) -> pd.DataFrame:
        return normalize(self.build.drop(columns=[_BUILD_KEY]))

    def _probe_only_columns(self) -> list[str]:
        return getattr(self, "_probe_columns", [])

    def _probe(self, probe: pd.DataFrame) -> pd.DataFrame:
        build_columns = self._build_columns()
        self._probe_columns = [c for c in probe.columns if c not in build_columns.columns]

        self._transients = []
        if self.join_type is JoinType.CROSS:
            if len(probe):
                self.matched[:] = True
            return normalize(build_columns.merge(probe, how="cross"))

        # Rows whose key is null are held OUT of the merge rather than dropped, so an
        # outer type can still emit them as unmatched. That is the whole difference
        # between null_equals_null false and true.
        probe = probe.copy()
        probe[_PROBE_KEY] = np.arange(len(probe))
        if self.null_equals_null:
            build_live, probe_live = self.build, probe
        else:
            build_live = self.build[self.build[self.build_keys].notna().all(axis=1)]
            live = probe[self.probe_keys].notna().all(axis=1)
            probe_live = probe[live]

        merged = build_live.merge(
            probe_live,
            how="inner",
            left_on=self.build_keys,
            right_on=self.probe_keys,
            suffixes=("", "_probe"),
        )
        if len(merged):
            self.matched[merged[_BUILD_KEY].to_numpy()] = True

        # `merged` carries both marker columns and every matched pair; it is dropped for
        # the returned frame, so it is exactly what scratch means here.
        self._transients = [merged]
        matches = merged.drop(columns=[_BUILD_KEY, _PROBE_KEY])

        if self.join_type in (JoinType.INNER, JoinType.LEFT_OUTER):
            return normalize(matches)
        if self.join_type in (JoinType.LEFT_SEMI, JoinType.LEFT_ANTI):
            # Both emit only at finish; the probe call exists to update `matched`.
            return build_columns.iloc[0:0]

        # RIGHT_OUTER / FULL_OUTER: this batch's unmatched probe rows, build columns
        # null-padded. Batch-local, so no finish pass is needed for the probe side.
        matched_rows = set(merged[_PROBE_KEY].tolist())
        unmatched = probe[~probe[_PROBE_KEY].isin(matched_rows)].drop(columns=[_PROBE_KEY])
        if not len(unmatched):
            return normalize(matches)
        padded = unmatched.copy()
        for column in build_columns.columns:
            if column not in padded.columns:
                padded[column] = np.nan
        padded = padded[list(matches.columns)]
        return normalize(concatenate([matches, padded]))
