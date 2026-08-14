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
from .frame import PandasBatch, concatenate, measured, normalize


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
    ):
        if join_type is not JoinType.CROSS and len(build_keys) != len(probe_keys):
            raise ValueError("join key lists must have equal length")
        self.join_type = join_type
        self.build_keys = build_keys
        self.probe_keys = probe_keys
        self.null_equals_null = null_equals_null
        self.name = name
        self.build: pd.DataFrame | None = None
        self.matched: np.ndarray | None = None
        self.probe_calls = 0

    # -- Executor ---------------------------------------------------------------

    def resident_bytes(self) -> int:
        if self.build is None:
            return 0
        return int(self.build.memory_usage(index=False, deep=True).sum())

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        # The build side is rebuilt on every probe call in v1 (#136), so it is scratch
        # as well as residency.
        return self.resident_bytes() + n_bytes

    # -- JoinExecutor -----------------------------------------------------------

    def set_build(self, batch: PandasBatch) -> CallStats:
        if self.build is not None:
            raise AssertionError(f"{self.name}: set_build called twice")
        frame = batch.consume()
        self.build = frame.copy()
        self.build[_BUILD_KEY] = np.arange(len(frame))
        self.matched = np.zeros(len(frame), dtype=bool)
        return measured(frame)

    def probe_and_fetch(self, batch: PandasBatch):
        if self.build is None:
            raise AssertionError(f"{self.name}: probed before set_build")
        self.probe_calls += 1
        probe = batch.consume()
        out = self._probe(probe)
        return (
            [PandasBatch(out, f"({self.name}#{self.probe_calls}⋈{batch.tag})")],
            measured(probe),
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
        return [PandasBatch(out, f"({self.name}⋈finish)")], measured(out)

    # -- internals --------------------------------------------------------------

    def _build_columns(self) -> pd.DataFrame:
        return normalize(self.build.drop(columns=[_BUILD_KEY]))

    def _probe_only_columns(self) -> list[str]:
        return getattr(self, "_probe_columns", [])

    def _probe(self, probe: pd.DataFrame) -> pd.DataFrame:
        build_columns = self._build_columns()
        self._probe_columns = [c for c in probe.columns if c not in build_columns.columns]

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
