"""Join types and the capability matrix, as data both backends read.

`JoinType` is `gpu_plan.fbs`'s enum, values included: one vocabulary for the pandas
executors, the recipe emulation and the spec's table, so a claim in one is the claim in
the others. A cross join is not a member — the wire has a node kind for it
(`CudfCrossJoin`) and no join type — and neither is a "nested loop" type: that is also a
node kind, carrying `Inner` or `Left` from this same enum.

**The capability question** the batch-partitioned mode asks of each type is one thing: with
the build side complete and the probe side arriving in batches, can this type answer from
one probe batch at a time? Three answers, and each is a property of where the type's
unmatched rows come from:

- **probe-local** — every output row is decided by (the whole build, this probe batch), so
  streaming needs nothing extra. Inner, Right, RightSemi, RightAnti.
- **build-preserving** — the output includes build rows that matched *nothing across every
  probe batch*, which no single batch can know. Streaming needs a finish pass, and the
  finish pass needs the probe keys kept ([#136](../../../llm-wiki/tickets.md#t136)).
  Left, Full, LeftSemi, LeftAnti, LeftMark.
- **refused** — a shape with no path on the frozen C++ surface at all.

A residual filter cuts across that: the finish pass sees accumulated *keys*, and a keys-only
table cannot evaluate a predicate over both sides' columns. So a filtered build-preserving
join does not stream, and the planner gives it a single-batch probe — which is the legacy
call, one `CudfHashJoin` over the whole probe side.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class JoinType(Enum):
    """`gpu_plan.fbs`: enum JoinType : byte."""

    INNER = 0
    LEFT = 1
    RIGHT = 2
    FULL = 3
    LEFT_SEMI = 4
    RIGHT_SEMI = 5
    LEFT_ANTI = 6
    RIGHT_ANTI = 7
    LEFT_MARK = 8


#: Decided by (whole build, this probe batch) — so a streamed probe needs no finish.
PROBE_LOCAL = frozenset(
    {JoinType.INNER, JoinType.RIGHT, JoinType.RIGHT_SEMI, JoinType.RIGHT_ANTI}
)
#: Emit build rows that matched no probe batch, so a streamed probe needs a finish pass.
BUILD_PRESERVING = frozenset(
    {JoinType.LEFT, JoinType.FULL, JoinType.LEFT_SEMI, JoinType.LEFT_ANTI, JoinType.LEFT_MARK}
)
#: Types whose entire output comes from the finish pass; the probe calls only record keys.
FINISH_ONLY = frozenset({JoinType.LEFT_SEMI, JoinType.LEFT_ANTI, JoinType.LEFT_MARK})
#: Types emitting one side's columns rather than both, so `projection` indexes that side.
ONE_SIDED = frozenset(
    {JoinType.LEFT_SEMI, JoinType.RIGHT_SEMI, JoinType.LEFT_ANTI, JoinType.RIGHT_ANTI}
)
#: Anti and mark ignore the plan's `null_equals_null` and stay EQUAL — see #80 / #59.
IGNORES_NULL_EQUALS_NULL = frozenset(
    {JoinType.LEFT_ANTI, JoinType.RIGHT_ANTI, JoinType.LEFT_MARK}
)


@dataclass(frozen=True)
class Capability:
    """What the mode may do with one join, and why."""

    #: may the probe side arrive in more than one batch
    streams: bool
    #: does a streamed probe need `finish_and_fetch` to emit anything
    needs_finish: bool
    #: set when the shape has no path at all; the planner refuses it (spec's scope rule)
    refusal: str | None = None
    #: why the probe is single-batch, when it is
    reason: str = ""


def capability(join_type: JoinType, has_filter: bool = False, nested_loop: bool = False) -> Capability:
    """The matrix, evaluated. One function so no backend can hold a different opinion."""
    if nested_loop:
        if join_type not in (JoinType.INNER, JoinType.LEFT):
            return Capability(
                False, False,
                refusal=f"CudfNestedLoopJoin supports Inner and Left only, not {join_type.name}",
            )
        if join_type is JoinType.LEFT:
            # A predicate join has no keys, so #136's accumulate-the-keys trick has
            # nothing to accumulate — the whole probe side has to be one batch.
            return Capability(False, False, reason="no keys for the finish pass")
        return Capability(True, False)

    if has_filter and join_type in (JoinType.RIGHT_SEMI, JoinType.RIGHT_ANTI):
        return Capability(
            False, False,
            refusal=(
                f"{join_type.name} with a residual filter has no cuDF path (no swapped "
                "mixed_* variant); keep the emitted side as the build so the join stays "
                "a Left form"
            ),
        )
    if has_filter and join_type in (JoinType.LEFT, JoinType.RIGHT, JoinType.FULL):
        return Capability(
            False, False,
            refusal=(
                f"{join_type.name} with a residual filter: the C++ applies the filter with "
                "apply_boolean_mask after the outer gather, so a padded row's NULL columns "
                "make the predicate NULL and the row is dropped — an ON-condition demoted "
                "to a WHERE. Latent in the legacy engine too (#153); refused at plan time "
                "rather than answered wrongly"
            ),
        )
    if has_filter and join_type in BUILD_PRESERVING:
        return Capability(False, False, reason="the finish pass sees keys, which cannot "
                                               "evaluate a residual filter")
    if join_type in BUILD_PRESERVING:
        return Capability(True, True)
    return Capability(True, False)


#: Types where every output row carries the build side's key, so the probe's copy of it is
#: redundant. Not RIGHT or FULL: there an unmatched probe row has a null build key and the
#: probe's copy is the only place the value survives.
_KEY_FROM_BUILD = frozenset({JoinType.INNER, JoinType.LEFT})


def joined_projection(build_names, probe_names, key_pairs, join_type: JoinType) -> tuple[int, ...]:
    """The `projection` an equi-join carries over `[build…, probe…]`.

    Both sides' key columns survive the join, and a query that joined `a.k = b.k` asked for
    one `k`. DataFusion drops the duplicate with the join's own projection, so the model
    does too — by name, since a name collision is what makes it a duplicate, and only for
    the types where the surviving column answers for every row.
    """
    dropped = (
        {probe for (build, probe) in key_pairs if build_names[build] == probe_names[probe]}
        if join_type in _KEY_FROM_BUILD
        else set()
    )
    keep = list(range(len(build_names)))
    keep += [len(build_names) + i for i in range(len(probe_names)) if i not in dropped]
    return tuple(keep)


def joined_names(build_names, probe_names, key_pairs, join_type: JoinType) -> list[str]:
    """The column names `joined_projection` leaves, in order."""
    all_names = list(build_names) + list(probe_names)
    return [all_names[i] for i in joined_projection(build_names, probe_names, key_pairs, join_type)]
