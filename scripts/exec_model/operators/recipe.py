"""The recipe plan: FlatBuffers node structs, a handle registry, and the C++ that reads them.

The batch-partitioned mode does not send the C++ side its own node tree. It sends a
**recipe plan** — a structurally valid FlatBuffers plan in the legacy vocabulary whose
nodes exist to be addressed by seq — and then calls `execute_node(seq, handles)` as many
times as its own schedule wants (the spec's "GPU execution through the frozen FFI"). This
module is that layer, modelled: the structs mirror `flatbuffers/gpu_plan.fbs` field for
field, `NodeSession` mirrors `cpp/src/node_session.cpp`'s registry and dispatch, and the
`execute_*` functions mirror `cpp/src/operators/join.cpp` and `project.cpp` branch for
branch, over the primitives in `cudf_calls.py`. One fork is deliberately collapsed: the
`PEACOCK_HAVE_FILTERED_JOIN` pair, where the C++ picks `cudf::filtered_join` over the free
`left_semi_join` / `left_anti_join`. That is a cuDF-version difference and not a semantic
one — both honour `compare_nulls`, as the C++ comment says — so the model keeps one arm.

The point is not to run joins — `joins.py` already does. It is to answer whether the
frozen surface **can** run them: every capability the join executor claims has to be
spelled here as nodes that exist, fields that exist, and calls C++ already makes. Where it
cannot be, the model says so loudly rather than reaching for python.

**Ordinals, not names.** Every column reference in the IR is an ordinal into the child's
output table (architecture.md, "Column indexing"), so the structs here carry ordinals and
resolve them positionally. Only a filter's leaf lookup is by name, because the prototype's
expression IR is name-based on purpose (`expressions.py`); the intermediate table a filter
sees is still assembled by ordinal, through `filter_columns`, exactly as the C++ does.

**Handles are consumed.** `NodeSession::execute_node` erases every input handle it reads,
and so does this — that is the constraint [#152](../../../llm-wiki/tickets.md#t152) is
about, and a model that quietly kept a handle alive would answer the wrong question.
`copy_handle` is therefore **not** part of the frozen surface: it is here so the cost of
working around that can be counted instead of hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from . import cudf_calls as cudf
from .cudf_calls import NullEquality, OutOfBoundsPolicy, Table
from .join_types import JoinType


class JoinSide(Enum):
    LEFT = 0
    RIGHT = 1


@dataclass(frozen=True)
class JoinFilterColumn:
    """`struct JoinFilterColumn` — column `index` of the `side` input, in filter order."""

    side: JoinSide
    index: int


@dataclass(frozen=True)
class JoinFilter:
    """A residual filter as the wire carries it: one expression plus its column origins.

    `expr` refers to the filter's own intermediate schema, whose column i is
    `columns[i]`. Both consumers of that convention are modelled below: the equi path
    builds the intermediate over the gathered output, and the mixed / conditional paths
    evaluate it per candidate pair.
    """

    expr: object                       # expressions.Expr over the intermediate schema
    columns: tuple[JoinFilterColumn, ...]

    def intermediate(self, table: Table, left_width: int) -> pd.DataFrame:
        """The view the equi path builds: `left_width` offsets the right side's ordinals."""
        picked = [
            table.column(
                left_width + c.index if c.side is JoinSide.RIGHT else c.index
            )
            for c in self.columns
        ]
        return _named(picked, self._names(table.names, left_width))

    def _names(self, names: list[str], left_width: int) -> list[str]:
        return [
            names[left_width + c.index if c.side is JoinSide.RIGHT else c.index]
            for c in self.columns
        ]

    def evaluate(self, left: Table, right: Table, left_idx, right_idx) -> np.ndarray:
        """The pair form: `cudf::ast` over `table_reference::LEFT` / `RIGHT`.

        A conditional or mixed join evaluates the predicate for a (left row, right row)
        pair without materializing the cross product, so the model gathers only the
        candidate pairs it was handed.
        """
        picked, names = [], []
        for c in self.columns:
            side, idx = (right, right_idx) if c.side is JoinSide.RIGHT else (left, left_idx)
            source = side.column(c.index).to_numpy()
            picked.append(pd.Series(source[np.asarray(idx, dtype=np.int64)]))
            names.append(side.names[c.index])
        return self.expr.evaluate(_named(picked, names)).fillna(False).to_numpy(dtype=bool)


def _named(columns: list[pd.Series], names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(dict(enumerate(columns)))
    frame.columns = names
    return frame


# -- projection expressions --------------------------------------------------------


@dataclass(frozen=True)
class ColumnRef:
    """`ColumnRef{index}` — project.cpp's fast path, a column copy."""

    index: int

    def evaluate(self, table: Table) -> pd.Series:
        return table.column(self.index)


@dataclass(frozen=True)
class NullLiteral:
    """`LiteralExpr{ScalarValue{type, is_null: true}}` — a typed NULL of `dtype`.

    What null-pads an outer join's missing side in the finish pass. On the C++ side a
    literal-only expression goes through `compute_column`, which sizes its output from the
    table it is given, so the pad needs no input column of its own.
    """

    dtype: object = "float64"

    def evaluate(self, table: Table) -> pd.Series:
        return pd.Series(pd.array([None] * table.num_rows(), dtype=self.dtype))


# -- plan nodes --------------------------------------------------------------------
#
# One dataclass per `PlanNodeKind` the join lowering uses, carrying the fields the fbs
# declares and no others. `input` / `left` / `right` are the tree in the fbs and are
# absent here for the same reason the spec gives: in a recipe plan a node is an address,
# and the driver supplies the handles.


@dataclass(frozen=True)
class CudfHashJoin:
    join_type: JoinType
    #: `keys: [JoinKey]` as (left ordinal, right ordinal) pairs
    keys: tuple[tuple[int, int], ...] = ()
    filter: JoinFilter | None = None
    projection: tuple[int, ...] = ()
    null_equals_null: bool = False


@dataclass(frozen=True)
class CudfCrossJoin:
    pass


@dataclass(frozen=True)
class CudfNestedLoopJoin:
    join_type: JoinType
    filter: JoinFilter | None = None
    projection: tuple[int, ...] = ()


@dataclass(frozen=True)
class CudfProject:
    #: `exprs` parallel to `aliases`
    exprs: tuple[object, ...] = ()
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CudfCoalescePartitions:
    """The collapse arm: concatenate whatever k handles the call was given."""


# -- the C++ side ------------------------------------------------------------------


def execute_hash_join(node: CudfHashJoin, left: Table, right: Table) -> Table:
    """`peacock::execute_hash_join`, branch for branch.

    The NULL-equality asymmetry is the C++'s and is reproduced rather than corrected:
    semi and the equi joins take the plan's flag, while anti and mark are hardcoded EQUAL
    because `x NOT IN (…, NULL)` is neither EQUAL nor UNEQUAL (#80, #59).
    """
    left_keys = Table([left.column(l) for l, _ in node.keys],
                      [left.names[l] for l, _ in node.keys])
    right_keys = Table([right.column(r) for _, r in node.keys],
                       [right.names[r] for _, r in node.keys])
    join_nulls = NullEquality.EQUAL if node.null_equals_null else NullEquality.UNEQUAL
    jt = node.join_type

    if jt in (JoinType.LEFT_SEMI, JoinType.RIGHT_SEMI, JoinType.LEFT_ANTI, JoinType.RIGHT_ANTI):
        return _project(_semi_anti(node, left, right, left_keys, right_keys, join_nulls),
                        node.projection)

    if jt is JoinType.LEFT_MARK:
        return _project(_mark(node, left, right, left_keys, right_keys), node.projection)

    if jt is JoinType.INNER:
        left_idx, right_idx = cudf.inner_join(left_keys, right_keys, join_nulls)
    elif jt is JoinType.LEFT:
        left_idx, right_idx = cudf.left_join(left_keys, right_keys, join_nulls)
    elif jt is JoinType.FULL:
        left_idx, right_idx = cudf.full_join(left_keys, right_keys, join_nulls)
    elif jt is JoinType.RIGHT:
        # cuDF has no right_join: left_join(R, L) with the pair swapped back.
        right_idx, left_idx = cudf.left_join(right_keys, left_keys, join_nulls)
    else:
        raise NotImplementedError(f"unsupported join type: {jt}")

    # NULLIFY exactly where a sentinel can appear; DONT_CHECK elsewhere, where cuDF
    # would fault on one.
    right_policy = (
        OutOfBoundsPolicy.NULLIFY
        if jt in (JoinType.LEFT, JoinType.FULL)
        else OutOfBoundsPolicy.DONT_CHECK
    )
    left_policy = (
        OutOfBoundsPolicy.NULLIFY
        if jt in (JoinType.FULL, JoinType.RIGHT)
        else OutOfBoundsPolicy.DONT_CHECK
    )
    gathered = Table(
        cudf.gather(left, left_idx, left_policy).columns
        + cudf.gather(right, right_idx, right_policy).columns,
        list(left.names) + list(right.names),
    )

    if node.filter is not None:
        mask = node.filter.expr.evaluate(node.filter.intermediate(gathered, left.num_columns()))
        gathered = cudf.apply_boolean_mask(gathered, mask)
    return _project(gathered, node.projection)


def _semi_anti(node, left, right, left_keys, right_keys, join_nulls) -> Table:
    """Right{Semi,Anti} are Left{Semi,Anti} with the sides swapped, as in the C++."""
    jt = node.join_type
    emit_left = jt in (JoinType.LEFT_SEMI, JoinType.LEFT_ANTI)
    is_semi = jt in (JoinType.LEFT_SEMI, JoinType.RIGHT_SEMI)
    # Anti keeps EQUAL whatever the plan says; semi takes the flag.
    nulls = join_nulls if is_semi else NullEquality.EQUAL
    probe_keys, set_keys = (left_keys, right_keys) if emit_left else (right_keys, left_keys)
    probe, set_side = (left, right) if emit_left else (right, left)

    if node.filter is not None:
        if not emit_left:
            raise NotImplementedError(
                "residual filter on a Right{Semi,Anti} join: the C++ throws here — there "
                "is no swapped mixed-join path (join.cpp ~L155). Keep the emitted side as "
                "the build so the join stays a Left form."
            )
        call = cudf.mixed_left_semi_join if is_semi else cudf.mixed_left_anti_join
        indices = call(probe_keys, set_keys, left, right, node.filter, nulls)
    else:
        call = cudf.left_semi_join if is_semi else cudf.left_anti_join
        indices = call(probe_keys, set_keys, nulls)
    del set_side
    return cudf.gather(probe, indices)


def _mark(node, left, right, left_keys, right_keys) -> Table:
    """LeftMark: every left row, plus `mark` = it had ≥1 match. EQUAL on purpose (#59)."""
    if node.filter is not None:
        matched = cudf.mixed_left_semi_join(
            left_keys, right_keys, left, right, node.filter, NullEquality.EQUAL
        )
    else:
        matched = cudf.left_semi_join(left_keys, right_keys, NullEquality.EQUAL)
    target = Table([cudf.make_column_from_scalar(False, left.num_rows())], ["mark"])
    source = Table([cudf.make_column_from_scalar(True, len(matched))], ["mark"])
    marks = cudf.scatter(source, matched, target)
    return Table(list(left.columns) + marks.columns, list(left.names) + ["mark"])


def execute_cross_join(node: CudfCrossJoin, left: Table, right: Table) -> Table:
    """`peacock::execute_cross_join` — one `cudf::cross_join`, names appended."""
    return cudf.cross_join(left, right)


def execute_nested_loop_join(node: CudfNestedLoopJoin, left: Table, right: Table) -> Table:
    """`peacock::execute_nested_loop_join` — Inner and Left only, as the C++ enforces."""
    jt = node.join_type
    if jt not in (JoinType.INNER, JoinType.LEFT):
        raise NotImplementedError(
            f"CudfNestedLoopJoin: only Inner/Left join types supported (got {jt})"
        )
    if node.filter is None:
        if jt is JoinType.LEFT and right.num_rows() == 0:
            raise NotImplementedError(
                "unconditional LEFT NestedLoopJoin with an empty right side is "
                "unsupported (cross_join would drop all left rows)"
            )
        return _project(cudf.cross_join(left, right), node.projection)

    if jt is JoinType.LEFT:
        left_idx, right_idx = cudf.conditional_left_join(left, right, node.filter)
    else:
        left_idx, right_idx = cudf.conditional_inner_join(left, right, node.filter)
    right_policy = (
        OutOfBoundsPolicy.NULLIFY if jt is JoinType.LEFT else OutOfBoundsPolicy.DONT_CHECK
    )
    gathered = Table(
        cudf.gather(left, left_idx).columns + cudf.gather(right, right_idx, right_policy).columns,
        list(left.names) + list(right.names),
    )
    return _project(gathered, node.projection)


def execute_project(node: CudfProject, input_table: Table) -> Table:
    """`peacock::execute_project` — one column per expr, `aliases` naming them."""
    columns = [expr.evaluate(input_table) for expr in node.exprs]
    names = [
        node.aliases[i] if i < len(node.aliases) else f"col{i}"
        for i in range(len(node.exprs))
    ]
    return Table([c.reset_index(drop=True) for c in columns], names)


def execute_coalesce_partitions(node: CudfCoalescePartitions, inputs: list[Table]) -> Table:
    """`NodeSession`'s collapse arm — `cudf::concatenate` over the handles it was given."""
    return cudf.concatenate(inputs)


def _project(table: Table, projection: tuple[int, ...]) -> Table:
    return table.select(projection) if projection else table


# -- the session -------------------------------------------------------------------


@dataclass
class Call:
    """One `peacock_executor_execute_node` call, as a trace line."""

    seq: int
    kind: str
    in_handles: tuple[int, ...]
    out_rows: int


@dataclass
class NodeSession:
    """`begin_plan` / `execute_node` / `release`, over the registry C++ keeps.

    One session per join executor instance here, where the real one is per plan; the
    difference does not matter to what is being proven, since a seq is an address and
    the registry is flat.
    """

    plan: list = field(default_factory=list)
    _registry: dict = field(default_factory=dict, init=False)
    _next_handle: int = field(default=1, init=False)
    calls: list = field(default_factory=list, init=False)
    #: handle copies made to work around consume-on-use, by what needed one — see the
    #: module note and #152. Kept apart because the two costs are different arguments:
    #: a build copy is one per probe batch of a table sized by the build side, a probe
    #: copy is one per probe batch of that batch.
    copies: dict = field(default_factory=dict, init=False)
    copied_bytes: dict = field(default_factory=dict, init=False)

    def add(self, node) -> int:
        """Register a node in the recipe plan and return its seq."""
        self.plan.append(node)
        return len(self.plan) - 1

    def node_count(self) -> int:
        return len(self.plan)

    def register(self, table: Table) -> int:
        """A handle for a table the driver produced — the loader's output, modelled."""
        handle = self._next_handle
        self._next_handle += 1
        self._registry[handle] = table
        return handle

    def table_for(self, handle: int) -> Table:
        if handle not in self._registry:
            raise KeyError(f"unknown handle {handle}")
        return self._registry[handle]

    def release(self, handle: int) -> None:
        self._registry.pop(handle, None)

    def bytes_of(self, handle: int) -> int:
        return self.table_for(handle).byte_size()

    def copy_handle(self, handle: int, reason: str) -> int:
        """**Not on the frozen surface.** A second handle onto the same rows.

        Every `execute_node` call erases the handles it reads, so a build side probed by
        B batches is needed B times and exists once. There is no node that duplicates a
        handle: a nominal candidate — passing one handle twice to a concat — fails on the
        second read, because the first erased it. So this is the device copy #152 weighs,
        or the refcounted handle #145 proposes, and it is counted rather than assumed.
        """
        table = self.table_for(handle)
        self.copies[reason] = self.copies.get(reason, 0) + 1
        self.copied_bytes[reason] = self.copied_bytes.get(reason, 0) + table.byte_size()
        return self.register(Table([c.copy() for c in table.columns], list(table.names)))

    def execute_node(self, seq: int, input_handles: list[int]) -> int:
        """One FFI call: consume the input handles, run the node, hand back a new one."""
        node = self.plan[seq]
        inputs = [self._take(handle) for handle in input_handles]
        if isinstance(node, CudfHashJoin):
            _expect(inputs, 2, node)
            out = execute_hash_join(node, inputs[0], inputs[1])
        elif isinstance(node, CudfCrossJoin):
            _expect(inputs, 2, node)
            out = execute_cross_join(node, inputs[0], inputs[1])
        elif isinstance(node, CudfNestedLoopJoin):
            _expect(inputs, 2, node)
            out = execute_nested_loop_join(node, inputs[0], inputs[1])
        elif isinstance(node, CudfProject):
            _expect(inputs, 1, node)
            out = execute_project(node, inputs[0])
        elif isinstance(node, CudfCoalescePartitions):
            out = execute_coalesce_partitions(node, inputs)
        else:
            raise NotImplementedError(f"no dispatch for {type(node).__name__}")
        self.calls.append(
            Call(seq, type(node).__name__, tuple(input_handles), out.num_rows())
        )
        return self.register(out)

    def _take(self, handle: int) -> Table:
        """Consume-on-read, as `NodeSession::execute_node` does."""
        if handle not in self._registry:
            raise KeyError(f"unknown input handle {handle} (consumed, or never issued)")
        return self._registry.pop(handle)


def _expect(inputs: list, n: int, node) -> None:
    """`execute_one`'s consumed == provided check."""
    if len(inputs) != n:
        raise ValueError(
            f"{type(node).__name__} takes {n} inputs, was given {len(inputs)}"
        )
