"""What a node declares about its output *columns*, in lieu of a full schema.

The real implementation carries column types here too (task T7) and renders them in the
plan golden. The prototype deliberately does not: types are the part a pandas stand-in
would model badly, and they are not what the aggregate sequence's validation needs. What
it needs is the semantics — which columns are group keys, and which columns are one
aggregate's state — so that a merging or finalizing node can check it is reading the state
its own partial produced rather than trusting a position.

That check is the prototype's answer to the class [#135](../../llm-wiki/tickets.md#t135)
describes: every column reference is an index into the child's output and almost nothing
verifies it, so a node reading the right *number* of columns in the wrong order produces
identical per-node statistics everywhere and surfaces only at the root. Here the index and
the name are declared together and cross-checked, which is the same pairing the spec's
node-display rule settles on for goldens.

Nodes that merely re-lane or re-batch their input forward their child's declaration
unchanged; nodes that build new columns (project, join) declare nothing, and a consumer of
an undeclared schema simply skips the check rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AggStateColumns:
    """One aggregate's state columns in a partial's output.

    `positions` are indices into the declaring node's `columns`, and `func`/`ddof` are
    carried so a merge can confirm it is merging the aggregate it thinks it is: a `sum`
    merged as if it were a `mean`'s sum-half reads the right column and computes nonsense.
    """

    output: str
    func: str
    ddof: int
    positions: tuple[int, ...]


@dataclass(frozen=True)
class Schema:
    """Column names in order — the index *is* the position — plus what they mean."""

    columns: tuple[str, ...]
    #: positions of the group-by keys, including `__grouping_id` where one was synthesized
    group_keys: tuple[int, ...] = ()
    #: one entry per aggregate whose state this output carries
    agg_state: tuple[AggStateColumns, ...] = field(default_factory=tuple)

    def position_of(self, column: str) -> int | None:
        return self.columns.index(column) if column in self.columns else None

    def state_for(self, output: str) -> AggStateColumns | None:
        return next((s for s in self.agg_state if s.output == output), None)


def aggregate_schema(keys, aggs, grouping_id: str | None = None) -> Schema:
    """What an init aggregate emits: keys, then the id if it expanded, then state.

    That order is not a convention this file invents — it is the order
    `cpp/src/operators/aggregate.cpp` builds its output names in, and it is what fixes the
    ordinals a plan's later references use.
    """
    key_columns = list(keys) + ([grouping_id] if grouping_id else [])
    columns = list(key_columns)
    state = []
    for agg in aggs:
        first = len(columns)
        columns.extend(agg.state_columns)
        state.append(
            AggStateColumns(
                output=agg.output,
                func=agg.func,
                ddof=agg.ddof,
                positions=tuple(range(first, len(columns))),
            )
        )
    return Schema(
        columns=tuple(columns),
        group_keys=tuple(range(len(key_columns))),
        agg_state=tuple(state),
    )


def finalized_schema(keys, aggs) -> Schema:
    """What a finalizing aggregate emits: the keys it grouped on, then one column per
    aggregate. No state survives, so nothing downstream can merge it — which is the point."""
    columns = tuple(list(keys) + [agg.output for agg in aggs])
    return Schema(columns=columns, group_keys=tuple(range(len(keys))))
