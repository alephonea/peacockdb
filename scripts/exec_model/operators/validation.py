"""`validate_schemas_and_partitions()` — what each node needs of its children.

Split from the whole-tree rules in `plan.py` on purpose, and the spec says why: a node
knows what it requires and can name the fix, where a generic rule can only say what is
wrong. "A limit over four lanes" should read *the planner inserts `GpuMergePartitions`
below it*, not *this category is 1:1 per lane*. `Plan.validate` runs these first for that
reason.

Two families live here.

**Layout expectations** — the parts of a child's `PartitionLayout` a node depends on:
hash distribution, sortedness, batch layout. These are exactly the properties that make a
plan silently wrong rather than loudly broken. A join whose sides are not co-partitioned
still runs and still returns rows; it returns too few, because lane p of one side was
joined against lane p of the other and the matching rows were in different lanes.

**Aggregate annotations** — the state-schema chain. A merging or finalizing aggregate
checks that the state it is about to read is the state its own partial declared: same
aggregate, same function, same positions. Without it a merge trusts a position, which is
[#135](../../../llm-wiki/tickets.md#t135)'s class of defect — right column count, wrong
column, identical per-node statistics, wrong answer at the root.

A child that declares no schema is not an error: sources, projects and joins build columns
this prototype does not model, and a check that cannot be made is skipped rather than
guessed at.
"""

from __future__ import annotations

from ..errors import PlanError
from ..layout import BatchLayout, KeyDistributionKind


def _child(node, slot: int = 0):
    return node.children()[slot]


# -- layout expectations -------------------------------------------------------------


def one_partition_in(node) -> None:
    """A limit is an interval over one stream; over N lanes it names no rows."""
    lanes = _child(node).output_partitions().n
    if lanes != 1:
        raise PlanError(
            f"{node.name()}: a limit is an interval over one stream, and its input has "
            f"{lanes} lanes — the planner inserts GpuMergePartitions below it"
        )


def prefix_is_meaningful(node) -> None:
    """A limit under a sort needs the *stream* ordered, not merely each batch.

    `BatchSorted` without `SingleBatch` means every batch is ordered and the stream is not,
    so a prefix of it is not the top-N anyone asked for — it is the first rows of whichever
    batches arrived first. Unsorted input is fine: an unordered LIMIT is allowed to return
    any rows, which the determinism scope note already covers.
    """
    layout = _child(node).output_partitions()
    if layout.sort_order.is_batch_sorted and not layout.is_stream_sorted:
        raise PlanError(
            f"{node.name()}: its input is sorted per batch but not across them, so a "
            "prefix is not a top-N — the planner puts GpuAccumulateBatchesAndSort or "
            "GpuMergeSortedPartitions below it"
        )


def sorted_input(node) -> None:
    """A k-way merge merges runs; unsorted input would make it a concatenation."""
    layout = _child(node).output_partitions()
    if not layout.sort_order.is_batch_sorted:
        raise PlanError(
            f"{node.name()}: merging sorted partitions requires BatchSorted input — the "
            "planner puts a GpuSort below it"
        )


def co_partitioned_join(build_keys, probe_keys):
    """Both sides hash-distributed on their join keys, whenever there is more than one lane.

    At one lane every row meets every other and distribution is irrelevant. Above one, a
    join runs lane-wise: rows of lane p on the left meet only rows of lane p on the right.
    That is correct exactly when equal keys hash to equal lanes on both sides, and silently
    lossy otherwise — the injector guards the same rule by convention when it decides
    whether a join's lane count may be rewritten, and this is that rule made checkable.
    """

    def check(node) -> None:
        lanes = node.output_partitions().n
        if lanes <= 1:
            return
        for side, keys, slot in (("build", build_keys, 0), ("probe", probe_keys, 1)):
            distribution = _child(node, slot).output_partitions().key_distribution
            if distribution.kind is not KeyDistributionKind.BY_HASH:
                raise PlanError(
                    f"{node.name()}: the {side} side of a {lanes}-lane join is not "
                    "hash-distributed, so lane p would be joined against rows whose "
                    "matches live in another lane — the planner shuffles both sides on "
                    "their join keys"
                )
            if len(distribution.hash_keys) != len(keys):
                raise PlanError(
                    f"{node.name()}: the {side} side is hashed on "
                    f"{len(distribution.hash_keys)} columns against {len(keys)} join keys"
                )

    return check


def single_batch_in(what: str):
    """A node that consumes its input whole in one call needs it in one batch."""

    def check(node) -> None:
        layout = _child(node).output_partitions()
        if layout.batch_layout is not BatchLayout.SINGLE_BATCH:
            raise PlanError(
                f"{node.name()}: {what} requires SingleBatch input — the planner inserts "
                "GpuCoalesceAllBatches"
            )

    return check


# -- the aggregate state chain -------------------------------------------------------


def merges_its_own_partial(keys, aggs):
    """A merge/final reads the state its own partial declared, checked rather than assumed.

    Three ways this goes wrong and all three are silent: the group keys are not the ones
    the partial grouped on, an aggregate's state is missing because the lists disagree, or
    the state is present under a different function — a `sum` read where a `mean`'s
    sum-half sits reads a real column and computes a wrong number.
    """

    def check(node) -> None:
        schema = _child(node).output_schema()
        if schema is None:
            return  # the child declares nothing this prototype can check

        positions = []
        for key in keys:
            at = schema.position_of(key)
            if at is None:
                raise PlanError(
                    f"{node.name()}: group key {key!r} is not in its input "
                    f"{list(schema.columns)}"
                )
            positions.append(at)
        if tuple(positions) != schema.group_keys:
            raise PlanError(
                f"{node.name()}: groups on {keys} at positions {positions}, but its input "
                f"declares its group keys at {list(schema.group_keys)} — the merge is not "
                "grouping on what the partial grouped on"
            )

        for agg in aggs:
            declared = schema.state_for(agg.output)
            if declared is None:
                raise PlanError(
                    f"{node.name()}: no state for {agg.output!r} in its input, which "
                    f"carries {[s.output for s in schema.agg_state]}"
                )
            if (declared.func, declared.ddof) != (agg.func, agg.ddof):
                raise PlanError(
                    f"{node.name()}: {agg.output!r} is a {agg.func}(ddof={agg.ddof}) here "
                    f"but its input declares {declared.func}(ddof={declared.ddof})"
                )
            expected = tuple(schema.position_of(c) for c in agg.state_columns)
            if None in expected:
                missing = [c for c in agg.state_columns if schema.position_of(c) is None]
                raise PlanError(
                    f"{node.name()}: state columns {missing} of {agg.output!r} are not in "
                    "its input"
                )
            if expected != declared.positions:
                raise PlanError(
                    f"{node.name()}: {agg.output!r} reads positions {list(expected)} but "
                    f"its input declares that state at {list(declared.positions)}"
                )

    return check


def hash_keys_subset_of_groups(keys):
    """`KeyDistribution.hashKeys ⊆ group columns` — subset, not equality.

    Equal group keys must land in one lane for a per-lane merge to be the global answer.
    Hashing on a subset of the group columns gives that; hashing on anything else does not,
    and grouping-set output is why the rule is a subset rather than an equality — the id is
    a group column that the shuffle deliberately does not hash.
    """

    def check(node) -> None:
        child = _child(node)
        distribution = child.output_partitions().key_distribution
        if distribution.kind is not KeyDistributionKind.BY_HASH:
            return
        schema = child.output_schema()
        if schema is None:
            return
        group_positions = {schema.position_of(k) for k in keys}
        stray = [h for h in distribution.hash_keys if h not in group_positions]
        if stray:
            raise PlanError(
                f"{node.name()}: its input is hashed on column(s) {stray}, which are not "
                f"among its group columns {sorted(p for p in group_positions if p is not None)} "
                "— equal groups could then sit in different lanes"
            )

    return check


def all_of(*checks):
    """Run several checks in order; the first to fail names the fix."""

    def check(node) -> None:
        for one in checks:
            if one is not None:
                one(node)

    return check
