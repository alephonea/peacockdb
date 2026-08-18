"""Join executors that go through the recipe plan — the model of the GPU backend.

`joins.py` is the pandas backend: it joins with pandas, the way DataFusion joins with
DataFusion. This is the other one. It answers every call by emitting FlatBuffers nodes and
making `execute_node(seq, handles)` calls against `recipe.NodeSession`, which runs the
cuDF the C++ runs. Both backends drive the same plans through the same driver, so a
disagreement is a real one — the mode's version of two-engine correctness, one level down.

What it is for: the batch-partitioned mode keeps the C++ side and the fbs frozen (three
additive symbols, spec's "GPU execution through the frozen FFI"). That claim is only worth
what the join can do, because the join is the operator whose state outlives a single call.
So each type here is spelled as a sequence of nodes that exist and fields that exist, and
where the frozen surface cannot express something it is named rather than worked around:

- **consume-on-use.** Every call erases the handles it reads, and a streamed probe needs
  the build side once per batch. `NodeSession.copy_handle` is off-surface and counted; the
  counts are what [#152](../../../llm-wiki/tickets.md#t152) asks for.
- **the finish pass.** A build-preserving type emits build rows that matched no probe
  batch. The probe calls therefore accumulate the probe **keys** ([#136](../../../llm-wiki/tickets.md#t136)),
  which costs a second copy — of the probe batch this time, since the join consumed it.
  Both copies disappear under [#145](../../../llm-wiki/tickets.md#t145)'s refcounted handle.

**Ordinals arrive late.** The translation layer emits these seqs at plan time, from the
schemas. The prototype has no typed schema (T7), so it resolves column names to ordinals
from the first batch it sees and emits the seqs then. The seq *set* is a function of the
join type and the filter, which is what the mapping table claims and what the tests assert.
"""

from __future__ import annotations

import pandas as pd

from ..batch import CallStats
from ..executors import JoinExecutor
from .frame import PandasBatch, no_scratch
from . import expressions
from . import join_types as J
from . import recipe as R
from .cudf_calls import Table
from .join_types import JoinType, capability, joined_projection
from .joins import TRIVIAL_FANOUT, merged_scratch


class RecipeJoin(JoinExecutor):
    """Shared plumbing: the session, the build handle, and the copies it takes to use it."""

    def __init__(self, name: str, fanout: float = TRIVIAL_FANOUT):
        self.name = name
        self.fanout = fanout
        self.session = R.NodeSession()
        self.build_handle: int | None = None
        self.build_rows = 0
        self.probe_calls = 0
        self._probe_names: list[str] | None = None

    # -- Executor ---------------------------------------------------------------

    def resident_bytes(self) -> int:
        return self.session.bytes_of(self.build_handle) if self.build_handle is not None else 0

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return merged_scratch(self.resident_bytes(), self.build_rows, self.fanout,
                              n_rows, n_bytes)

    # -- JoinExecutor -----------------------------------------------------------

    def set_build(self, batch: PandasBatch) -> CallStats:
        if self.build_handle is not None:
            raise AssertionError(f"{self.name}: set_build called twice")
        frame = batch.consume()
        self.build_handle = self.session.register(Table.from_frame(frame))
        self.build_rows = len(frame)
        return no_scratch()

    def probe_and_fetch(self, batch: PandasBatch):
        if self.build_handle is None:
            raise AssertionError(f"{self.name}: probed before set_build")
        self.probe_calls += 1
        capability_ = self._capability()
        if not capability_.streams and self.probe_calls > 1:
            raise AssertionError(
                f"{self.name}: this join takes a single-batch probe ({capability_.reason}); "
                "the planner inserts GpuCoalesceAllBatches under the probe side"
            )
        frame = batch.consume()
        self._probe_names = list(frame.columns)
        handle = self.session.register(Table.from_frame(frame))
        outputs = self._probe(handle)
        return outputs, no_scratch()

    def finish_and_fetch(self):
        outputs = self._finish()
        if self.build_handle is not None:
            self.session.release(self.build_handle)
            self.build_handle = None
        return outputs, no_scratch()

    # -- internals --------------------------------------------------------------

    def _capability(self) -> J.Capability:
        raise NotImplementedError

    def _probe(self, handle: int) -> list[PandasBatch]:
        raise NotImplementedError

    def _finish(self) -> list[PandasBatch]:
        return []

    def _build_names(self) -> list[str]:
        return list(self.session.table_for(self.build_handle).names)

    def _borrowed_build(self) -> int:
        """A handle onto the build side for one call, since the call will erase it."""
        return self.session.copy_handle(self.build_handle, "build")

    def _fetch(self, handle: int, tag: str) -> PandasBatch:
        """`peacock_result_from_handle` — the one place a table leaves the session."""
        table = self.session.table_for(handle)
        self.session.release(handle)
        return PandasBatch(table.to_frame(), tag)

    def _refuse_if_unsupported(self) -> None:
        refusal = self._capability().refusal
        if refusal is not None:
            raise NotImplementedError(f"{self.name}: {refusal}")


class RecipeHashJoin(RecipeJoin):
    """`CudfHashJoin`, plus the finish-pass seqs a build-preserving type needs."""

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
        self.probe_schema = probe_schema
        #: the residual predicate, as an expression; `filter_columns` is derived from it
        #: once the schemas name the ordinals
        self.residual = residual
        self._refuse_if_unsupported()
        #: seqs, filled in when the first batch names the ordinals
        self._probe_seq: int | None = None
        self._finish_seqs: dict = {}
        #: one handle per probe batch's projected keys — #136's accumulation
        self._key_handles: list[int] = []

    def resident_bytes(self) -> int:
        keys = sum(self.session.bytes_of(h) for h in self._key_handles)
        return super().resident_bytes() + keys

    def _capability(self) -> J.Capability:
        return capability(self.join_type, has_filter=self.residual is not None)

    # -- seq emission ------------------------------------------------------------

    def _emit_seqs(self, probe_names: list[str]) -> None:
        """The recipe plan for this join: what the mapping table's fourth column names."""
        if self._probe_seq is not None or self._finish_seqs:
            return
        build_names = self._build_names()
        join_filter = join_filter_of(self.residual, build_names, probe_names)
        keys = tuple(
            (build_names.index(b), probe_names.index(p))
            for b, p in zip(self.build_keys, self.probe_keys)
        )
        streams = self._capability().streams
        one_sided = self.join_type in J.ONE_SIDED or self.join_type is JoinType.LEFT_MARK
        projection = (
            () if one_sided else joined_projection(build_names, probe_names, keys, self.join_type)
        )

        if not streams or self.join_type in J.PROBE_LOCAL:
            # One node, one call per probe batch — and with a single-batch probe that is
            # exactly the legacy plan: `CudfHashJoin` over the whole probe side.
            self._probe_seq = self.session.add(
                R.CudfHashJoin(self.join_type, keys, join_filter, projection,
                               self.null_equals_null)
            )
            return

        if self.join_type in (JoinType.LEFT, JoinType.FULL):
            # Matches per batch; the unmatched build rows wait for the finish. A LEFT
            # emits only matches per call (INNER), a FULL also emits this batch's
            # unmatched probe rows, which are batch-local because the build is complete.
            per_call = JoinType.INNER if self.join_type is JoinType.LEFT else JoinType.RIGHT
            self._probe_seq = self.session.add(
                R.CudfHashJoin(per_call, keys, None, projection, self.null_equals_null)
            )
        self._emit_finish_seqs(build_names, probe_names, keys, projection)

    def _emit_finish_seqs(self, build_names, probe_names, keys, projection) -> None:
        """Key project → concat → anti/semi/mark join → (for an outer type) a null pad."""
        key_ordinals = [p for _, p in keys]
        self._finish_seqs["keys"] = self.session.add(
            R.CudfProject(
                tuple(R.ColumnRef(i) for i in key_ordinals),
                tuple(probe_names[i] for i in key_ordinals),
            )
        )
        self._finish_seqs["concat"] = self.session.add(R.CudfCoalescePartitions())
        # The finish join runs against the accumulated keys, whose ordinals are 0..k-1.
        finish_keys = tuple((build, i) for i, (build, _) in enumerate(keys))
        finish_type = (
            self.join_type
            if self.join_type in J.FINISH_ONLY
            else JoinType.LEFT_ANTI          # LEFT / FULL: the build rows nothing matched
        )
        self._finish_seqs["join"] = self.session.add(
            R.CudfHashJoin(finish_type, finish_keys, None, (), self.null_equals_null)
        )
        if self.join_type in (JoinType.LEFT, JoinType.FULL):
            self._finish_seqs["pad"] = self.session.add(
                self._pad_project(build_names, probe_names, projection)
            )

    def _pad_project(self, build_names, probe_names, projection) -> R.CudfProject:
        """Build columns straight through, one typed NULL literal per probe column kept.

        The anti join emits build columns only, and the output schema is the joined one, so
        the difference is exactly the probe columns the projection keeps.
        """
        exprs = [R.ColumnRef(i) for i in range(len(build_names))]
        aliases = list(build_names)
        for ordinal in projection:
            if ordinal >= len(build_names):
                exprs.append(R.NullLiteral())
                aliases.append(probe_names[ordinal - len(build_names)])
        return R.CudfProject(tuple(exprs), tuple(aliases))

    # -- calls -------------------------------------------------------------------

    def _probe(self, handle: int) -> list[PandasBatch]:
        self._emit_seqs(self._probe_names)
        if self._probe_seq is None:
            # Semi / anti / mark with a streamed probe: nothing is emitted per call. The
            # probe batch is consumed by the key project and the build side is not touched,
            # so these three stream at no copy cost at all.
            self._key_handles.append(
                self.session.execute_node(self._finish_seqs["keys"], [handle])
            )
            return []

        if self._finish_seqs:
            # LEFT / FULL: the join below consumes the probe batch, so the keys come off a
            # copy of it — the second consume-on-use cost, beside the build side's.
            self._key_handles.append(
                self.session.execute_node(
                    self._finish_seqs["keys"], [self.session.copy_handle(handle, "probe")]
                )
            )
        out = self.session.execute_node(self._probe_seq, [self._build_input(), handle])
        return [self._fetch(out, f"({self.name}#{self.probe_calls})")]

    def _build_input(self) -> int:
        """The build handle for one join call.

        A single-batch probe uses it exactly once, so it is handed over and consumed. A
        streamed probe needs it again next batch, and there is no node that duplicates a
        handle — hence the copy, and hence #152.
        """
        if self._capability().streams:
            return self._borrowed_build()
        handle, self.build_handle = self.build_handle, None
        return handle

    def _finish(self) -> list[PandasBatch]:
        if self._probe_names is None:
            # No probe batch ever arrived — an empty lane, which a shuffle produces
            # routinely. A build-preserving type still owes its whole build side, and the
            # declared probe schema is the only thing naming the columns to pad.
            if not self._capability().streams:
                raise AssertionError(
                    f"{self.name}: a single-batch probe delivered no batch — "
                    "GpuCoalesceAllBatches emits one even when it accumulated nothing (F7)"
                )
            if self.join_type not in J.BUILD_PRESERVING:
                return []
            if self.probe_schema is None:
                raise ValueError(
                    f"{self.name}: a {self.join_type.name} finish saw no probe batch and "
                    "has no probe_schema — the probe columns to null-pad are unknown"
                )
            self._probe_names = list(self.probe_schema)
            self._emit_seqs(self._probe_names)
        if not self._finish_seqs:
            return []
        keys = self._accumulated_keys()
        matched = self.session.execute_node(self._finish_seqs["join"], [self.build_handle, keys])
        self.build_handle = None
        if "pad" in self._finish_seqs:
            matched = self.session.execute_node(self._finish_seqs["pad"], [matched])
        return [self._fetch(matched, f"({self.name}⋈finish)")]

    def _accumulated_keys(self) -> int:
        """One handle for every probe key seen — `CudfCoalescePartitions` over the batch.

        With no probe batches there is nothing to concatenate and no node that makes an
        empty table, so the session is handed one directly: on the GPU the driver would
        do the same, since a zero-row key table is what "this lane saw nothing" means.
        """
        if not self._key_handles:
            return self.session.register(
                Table([pd.Series([], dtype="float64") for _ in self.probe_keys],
                      list(self.probe_keys))
            )
        if len(self._key_handles) == 1:
            return self._key_handles.pop()
        handles, self._key_handles = self._key_handles, []
        return self.session.execute_node(self._finish_seqs["concat"], handles)


class RecipeCrossJoin(RecipeJoin):
    """`CudfCrossJoin` — one call per probe batch, build × batch. No finish."""

    def __init__(self, name: str = "cross_join", fanout: float = TRIVIAL_FANOUT):
        super().__init__(name, fanout)
        self._seq = self.session.add(R.CudfCrossJoin())

    def _capability(self) -> J.Capability:
        return J.Capability(streams=True, needs_finish=False)

    def _probe(self, handle: int) -> list[PandasBatch]:
        out = self.session.execute_node(self._seq, [self._borrowed_build(), handle])
        return [self._fetch(out, f"({self.name}#{self.probe_calls})")]


class RecipeNestedLoopJoin(RecipeJoin):
    """`CudfNestedLoopJoin` — Inner streams; Left takes a single-batch probe."""

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
        self._refuse_if_unsupported()
        self._seq: int | None = None

    def _capability(self) -> J.Capability:
        return capability(self.join_type, has_filter=self.predicate is not None,
                          nested_loop=True)

    def _probe(self, handle: int) -> list[PandasBatch]:
        if self._seq is None:
            self._seq = self.session.add(
                R.CudfNestedLoopJoin(
                    self.join_type,
                    join_filter_of(self.predicate, self._build_names(), self._probe_names),
                    (),
                )
            )
        # Left is single-batch, so its one call may consume the build outright.
        streams = self._capability().streams
        build = self._borrowed_build() if streams else self.build_handle
        if not streams:
            self.build_handle = None
        out = self.session.execute_node(self._seq, [build, handle])
        return [self._fetch(out, f"({self.name}#{self.probe_calls})")]


def join_filter_of(expr, build_names: list[str], probe_names: list[str]):
    """`filter` plus the `filter_columns` map, from an expression over both sides' names.

    The wire carries the map because the filter's `ColumnRef`s index a private intermediate
    schema rather than either input; the planner builds it from the schemas, and so does
    this, at the point where the schemas are known.
    """
    if expr is None:
        return None
    columns = []
    for name in expressions.columns_of(expr):
        in_build, in_probe = name in build_names, name in probe_names
        if in_build and in_probe:
            raise ValueError(
                f"a join filter reads {name!r}, which both sides have — the wire map is "
                "per side, so the plan must rename one"
            )
        if in_build:
            columns.append(R.JoinFilterColumn(R.JoinSide.LEFT, build_names.index(name)))
        elif in_probe:
            columns.append(R.JoinFilterColumn(R.JoinSide.RIGHT, probe_names.index(name)))
        else:
            raise ValueError(f"a join filter reads {name!r}, which neither side has")
    return R.JoinFilter(expr, tuple(columns))
