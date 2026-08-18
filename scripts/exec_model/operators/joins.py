"""The pandas join backend — build side set once, probe side streamed per the matrix.

Three executors, one per wire node kind: `HashJoin` (`CudfHashJoin`, nine join types),
`CrossJoin` (`CudfCrossJoin`) and `NestedLoopJoin` (`CudfNestedLoopJoin`, Inner and Left).
Which types may stream a probe side, and what each call emits, is `join_types.capability`
— one function, so this backend and the recipe emulation in `recipe_join.py` cannot hold
different opinions about the matrix.

Per call and at finish, by family:

| type | per probe call | at finish |
|---|---|---|
| INNER | matches | nothing |
| RIGHT (build is left) | matches + this batch's unmatched probe rows | nothing |
| RIGHT_SEMI / RIGHT_ANTI | this batch's probe rows with / without a match | nothing |
| LEFT / FULL | matches (+ unmatched probe rows for FULL) | unmatched build rows, null-padded |
| LEFT_SEMI / LEFT_ANTI / LEFT_MARK | nothing | build rows with / without a match, or all plus `mark` |
| CROSS, NESTED LOOP Inner | build × probe batch (predicate-filtered for the NLJ) | nothing |
| NESTED LOOP Left | single-batch probe | — |

The probe-local half of that table is not a convention: with the build side complete, a
probe row that matched nothing in this batch matched nothing anywhere, so its emission is
decided locally. The build-preserving half is the opposite claim, which is why it needs a
finish pass.

**The prototype cheats where the real implementation cannot.** "Which build rows matched at
least once across all probe batches" is kept here as a boolean array over the build frame —
free in-process, and exactly the thing that never crosses the C ABI. The v1 GPU path
accumulates each probe batch's keys and runs one anti/semi join at finish (#136); that is
what `recipe_join.py` models, and running both backends over the same plans is what makes
this file an oracle rather than a second guess.

**Null key equality is explicit**, because pandas and SQL disagree: `merge` matches NaN
keys to each other. With `null_equals_null=False` a row with any null key matches nothing,
implemented by holding those rows out of the merge and treating them as unmatched. Anti and
mark ignore the flag and stay at EQUAL, as the C++ does (#80, #59).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..batch import CallStats
from ..executors import JoinExecutor
from . import join_types as J
from .frame import PandasBatch, concatenate, no_scratch, scratch_of, normalize
from .join_types import JoinType, capability, joined_names, joined_projection

#: suffix the probe side wears inside a merge, so a shared column name cannot collapse two
#: columns into one. The wire keeps both and lets `projection` choose; so does this.
_PROBE_TAG = "__p"
#: output rows / probe rows — the ratio `CardinalityEstimator` returns in gpu_rule.rs.
#: 1.0 is what the constant estimator gives today (#19).
TRIVIAL_FANOUT = 1.0
#: the two row-id marker columns the merged frame carries, int64 each
_MARKER_BYTES_PER_ROW = 16

_BUILD_KEY = "__build_row__"
_PROBE_KEY = "__probe_row__"


def merged_scratch(resident: int, build_rows: int, fanout: float, n_rows: int, n_bytes: int) -> int:
    """Build side (rebuilt per probe call, #136) plus the merged frame.

    The merged frame is sized by the *output* cardinality, which the signature cannot
    derive — but it does not have to. `fanout` is a node property the optimizer supplies at
    plan time and the executor is constructed with, so the estimate is already in `&self`
    by the time this is called. That is what keeps the trait unchanged while the model
    stays correct. Shared with the recipe backend so both model one thing one way.
    """
    if n_rows == 0:
        return resident
    probe_row_bytes = n_bytes / n_rows
    build_row_bytes = resident / max(1, build_rows)
    per_row = probe_row_bytes + build_row_bytes + _MARKER_BYTES_PER_ROW
    return int(resident + n_rows * fanout * per_row)


class _PandasJoin(JoinExecutor):
    """What the three share: the build frame, the residency it is, and the call protocol."""

    def __init__(self, name: str, fanout: float):
        self.name = name
        self.fanout = fanout
        self.build: pd.DataFrame | None = None
        self.probe_calls = 0
        self._probe_columns: list[str] | None = None

    def resident_bytes(self) -> int:
        if self.build is None:
            return 0
        return int(self.build.memory_usage(index=False, deep=True).sum())

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        build_rows = len(self.build) if self.build is not None else 0
        return merged_scratch(self.resident_bytes(), build_rows, self.fanout, n_rows, n_bytes)

    def _capability(self) -> J.Capability:
        raise NotImplementedError

    def set_build(self, batch: PandasBatch) -> CallStats:
        if self.build is not None:
            raise AssertionError(f"{self.name}: set_build called twice")
        self.build = self._take_build(batch.consume())
        return no_scratch()   # the build frame becomes residency, not scratch

    def _take_build(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()

    def probe_and_fetch(self, batch: PandasBatch):
        if self.build is None:
            raise AssertionError(f"{self.name}: probed before set_build")
        self.probe_calls += 1
        if not self._capability().streams and self.probe_calls > 1:
            raise AssertionError(
                f"{self.name}: this join takes a single-batch probe "
                f"({self._capability().reason}); the planner inserts GpuCoalesceAllBatches "
                "under the probe side"
            )
        probe = batch.consume()
        self._probe_columns = list(probe.columns)
        self._transients = []
        frames = self._probe(probe)
        return (
            [PandasBatch(f, f"({self.name}#{self.probe_calls}⋈{batch.tag})") for f in frames],
            scratch_of(*self._transients),
        )

    def finish_and_fetch(self):
        outputs = self._finish()
        # Nothing stays resident after the finish, so the enforcer can drop this executor.
        self.build = None
        return [PandasBatch(f, f"({self.name}⋈finish)") for f in outputs], no_scratch()

    def _probe(self, probe: pd.DataFrame) -> list[pd.DataFrame]:
        raise NotImplementedError

    def _finish(self) -> list[pd.DataFrame]:
        return []


class HashJoin(_PandasJoin):
    """`CudfHashJoin`: equi-join on key pairs, with an optional residual filter."""

    def __init__(
        self,
        join_type: JoinType,
        build_keys: list[str],
        probe_keys: list[str],
        null_equals_null: bool = False,
        name: str = "join",
        fanout: float = TRIVIAL_FANOUT,
        probe_schema: list[str] | None = None,
        residual=None,
    ):
        super().__init__(name, fanout)
        if len(build_keys) != len(probe_keys):
            raise ValueError("join key lists must have equal length")
        self.join_type = join_type
        self.build_keys = build_keys
        self.probe_keys = probe_keys
        self.null_equals_null = null_equals_null
        #: the probe side's declared columns — consulted only when an outer finish must
        #: null-pad columns no probe batch ever showed it
        self.probe_schema = probe_schema
        #: a residual (non-equi) predicate over the joined row, as `CudfHashJoin.filter`
        self.residual = residual
        self.matched: np.ndarray | None = None
        #: set when a single-batch probe already emitted everything, finish included
        self._finished = False
        refusal = self._capability().refusal
        if refusal is not None:
            raise NotImplementedError(f"{name}: {refusal}")

    def _capability(self) -> J.Capability:
        return capability(self.join_type, has_filter=self.residual is not None)

    def _take_build(self, frame: pd.DataFrame) -> pd.DataFrame:
        build = frame.copy()
        build[_BUILD_KEY] = np.arange(len(frame))
        self.matched = np.zeros(len(frame), dtype=bool)
        return build

    # -- probe -------------------------------------------------------------------

    def _probe(self, probe: pd.DataFrame) -> list[pd.DataFrame]:
        frames = self._probe_output(probe)
        if self._capability().streams:
            return frames
        # A single-batch probe is the whole join in one call, which is what the legacy
        # path does: nothing waits for a finish, because there is no second batch for it
        # to wait through.
        self._finished = True
        frames += self._finish_frames()
        return [normalize(concatenate(frames))] if len(frames) > 1 else frames

    def _probe_output(self, probe: pd.DataFrame) -> list[pd.DataFrame]:
        pairs = self._matching_pairs(probe)
        self._transients = [pairs]
        if len(pairs):
            self.matched[pairs[_BUILD_KEY].to_numpy()] = True

        if self.join_type in J.FINISH_ONLY:
            return []       # the call exists to update `matched`, nothing else
        if self.join_type in (JoinType.RIGHT_SEMI, JoinType.RIGHT_ANTI):
            hit = probe.index.isin(pairs[_PROBE_KEY].to_numpy())
            keep = hit if self.join_type is JoinType.RIGHT_SEMI else ~hit
            return [normalize(probe[keep])]

        matches = self._joined(pairs, probe.columns)
        if self.join_type in (JoinType.INNER, JoinType.LEFT):
            return [matches]
        # RIGHT / FULL: this batch's unmatched probe rows, build columns null-padded.
        # Batch-local, because the build side is complete before the first probe call.
        unmatched = probe[~probe.index.isin(pairs[_PROBE_KEY].to_numpy())]
        if not len(unmatched):
            return [matches]
        padded = pd.DataFrame(
            {column: np.nan for column in self._build_columns().columns}, index=unmatched.index
        )
        for column in probe.columns:
            padded[f"{column}{_PROBE_TAG}"] = unmatched[column]
        return [normalize(concatenate([matches, self._joined(padded, probe.columns)]))]

    def _matching_pairs(self, probe: pd.DataFrame) -> pd.DataFrame:
        """Every (build row, probe row) the join matches — keys, then the residual.

        The probe side is renamed before the merge so that both sides' columns survive
        even when they share a name. That is what the wire does — a join's output is
        `[left_cols…, right_cols…]` and `TableResult` allows two columns called `k` — and
        it is the join's `projection` that decides which of them the query sees.
        """
        probe = probe.rename(columns={c: f"{c}{_PROBE_TAG}" for c in probe.columns})
        probe[_PROBE_KEY] = np.arange(len(probe))
        probe_keys = [f"{k}{_PROBE_TAG}" for k in self.probe_keys]
        # Rows whose key is null are held OUT of the merge rather than dropped, so an
        # outer type can still emit them as unmatched. That is the whole difference
        # between null_equals_null false and true. Anti and mark are EQUAL regardless.
        equal_nulls = self.null_equals_null or self.join_type in J.IGNORES_NULL_EQUALS_NULL
        if equal_nulls:
            build_live, probe_live = self.build, probe
        else:
            build_live = self.build[self.build[self.build_keys].notna().all(axis=1)]
            probe_live = probe[probe[probe_keys].notna().all(axis=1)]
        pairs = build_live.merge(
            probe_live, how="inner", left_on=self.build_keys, right_on=probe_keys
        )
        if self.residual is not None and len(pairs):
            keep = self.residual.evaluate(self._untagged(pairs, probe.columns))
            pairs = pairs[keep.fillna(False).astype(bool).to_numpy()]
        return pairs

    def _untagged(self, frame: pd.DataFrame, tagged_probe_columns) -> pd.DataFrame:
        """The joined row under its own column names — what a residual filter reads.

        On the wire the filter's `ColumnRef`s index a private intermediate schema and
        `filter_columns` maps each back to a (side, ordinal); here the same mapping is by
        name, per the prototype's expression IR (`expressions.py`).
        """
        renamed = {c: c[: -len(_PROBE_TAG)] for c in tagged_probe_columns if c.endswith(_PROBE_TAG)}
        return frame.rename(columns=renamed)

    def _joined(self, pairs: pd.DataFrame, probe_columns) -> pd.DataFrame:
        """`[build…, probe…]` cut down by the join's `projection`, named as the query sees it.

        The same rule the recipe path states as the node's `projection` field, computed by
        the same function, so the two backends produce one schema rather than two.
        """
        build_names = list(self._build_columns().columns)
        probe_names = list(probe_columns)
        keys = [
            (build_names.index(b), probe_names.index(p))
            for b, p in zip(self.build_keys, self.probe_keys)
        ]
        columns = build_names + [f"{c}{_PROBE_TAG}" for c in probe_names]
        wanted = joined_projection(build_names, probe_names, keys, self.join_type)
        out = pairs[[columns[i] for i in wanted]]
        out.columns = joined_names(build_names, probe_names, keys, self.join_type)
        return normalize(out)

    # -- finish ------------------------------------------------------------------

    def _finish(self) -> list[pd.DataFrame]:
        if self._finished:
            return []
        return self._finish_frames()

    def _finish_frames(self) -> list[pd.DataFrame]:
        """The build rows the probe side never matched — the pass #136 is about."""
        if self.join_type not in J.BUILD_PRESERVING:
            return []
        build = self._build_columns()
        if self.join_type is JoinType.LEFT_SEMI:
            return [normalize(build[self.matched])]
        if self.join_type is JoinType.LEFT_ANTI:
            return [normalize(build[~self.matched])]
        if self.join_type is JoinType.LEFT_MARK:
            return [normalize(build.assign(mark=self.matched))]
        # LEFT / FULL: unmatched build rows, probe columns null-padded.
        out = build[~self.matched]
        for column in self._pad_columns(build):
            out = out.assign(**{column: np.nan})
        return [normalize(out)]

    def _build_columns(self) -> pd.DataFrame:
        return normalize(self.build.drop(columns=[_BUILD_KEY]))

    def _pad_columns(self, build: pd.DataFrame) -> list[str]:
        """The probe columns an outer finish must null-pad.

        Learned from the first probe batch; a probe lane that never produced one falls back
        to the declared `probe_schema`. Without either the output's shape would silently
        depend on whether a probe batch happened to arrive, so the miss is loud.
        """
        probe_columns = self._probe_columns
        if probe_columns is None:
            if self.probe_schema is None:
                raise ValueError(
                    f"{self.name}: a {self.join_type.name} finish saw no probe batch and "
                    "has no probe_schema — the probe columns to null-pad are unknown"
                )
            probe_columns = list(self.probe_schema)
        keys = [
            (list(build.columns).index(b), probe_columns.index(p))
            for b, p in zip(self.build_keys, self.probe_keys)
        ]
        names = joined_names(list(build.columns), probe_columns, keys, self.join_type)
        return [c for c in names if c not in list(build.columns)]


class CrossJoin(_PandasJoin):
    """`CudfCrossJoin` — build × probe batch, one output per call, nothing at finish."""

    def __init__(self, name: str = "cross_join", fanout: float = TRIVIAL_FANOUT):
        super().__init__(name, fanout)

    def _capability(self) -> J.Capability:
        return J.Capability(streams=True, needs_finish=False)

    def _probe(self, probe: pd.DataFrame) -> list[pd.DataFrame]:
        return [normalize(self.build.merge(probe, how="cross"))]


class NestedLoopJoin(_PandasJoin):
    """`CudfNestedLoopJoin` — the cross product filtered by a predicate.

    Inner streams its probe side, as a cross join does. Left does not: its unmatched build
    rows need the finish pass, and #136's trick accumulates *keys*, which a predicate join
    has none of — so the planner gives it a single-batch probe and the whole join happens
    in one call, exactly as the legacy path does it.
    """

    def __init__(
        self,
        join_type: JoinType,
        predicate,
        name: str = "nlj",
        fanout: float = TRIVIAL_FANOUT,
    ):
        super().__init__(name, fanout)
        self.join_type = join_type
        self.predicate = predicate
        refusal = self._capability().refusal
        if refusal is not None:
            raise NotImplementedError(f"{name}: {refusal}")

    def _capability(self) -> J.Capability:
        return capability(self.join_type, has_filter=self.predicate is not None,
                          nested_loop=True)

    def _probe(self, probe: pd.DataFrame) -> list[pd.DataFrame]:
        crossed = self.build.merge(probe, how="cross")
        self._transients = [crossed]
        if self.predicate is None:
            return [normalize(crossed)]
        keep = self.predicate.evaluate(crossed).fillna(False).astype(bool)
        matches = normalize(crossed[keep])
        if self.join_type is JoinType.INNER:
            return [matches]
        # LEFT: the build rows no probe row satisfied, right columns null-padded. One
        # call, so this is not a finish pass — it is the whole join.
        hit = crossed.index.isin(crossed.index[keep])
        matched_build = set(np.repeat(np.arange(len(self.build)), len(probe))[hit].tolist())
        missing = [i for i in range(len(self.build)) if i not in matched_build]
        if not missing:
            return [matches]
        padded = self.build.iloc[missing].copy()
        for column in probe.columns:
            padded[column] = np.nan
        return [normalize(concatenate([matches, padded[list(matches.columns)]]))]
