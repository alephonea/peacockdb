"""The pieces every hand-lowered corpus plan is built from.

A benchmark query never lowers to one node. An aggregate is four (partial, per-batch,
collapse, finalize); a sort is two (per-lane sort-and-trim, then one k-way merge). Those
sequences are the mode, not the query, so they live here once and each builder spends its
lines on what is actually query-specific — which side of a join is the build, where the
shuffle goes, what a projection has to materialize before an aggregate can take it.

Shared by `plans_tpch.py` and the `plans_tpcds_*.py` family modules.
"""

from __future__ import annotations

import pandas as pd

from .corpus import agg_schemas
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import Alias, Binary, Col, Lit


def date(text: str) -> Lit:
    """A date literal. cuDF's own is a typed TIMESTAMP_DAYS scalar; `corpus.py` reads
    parquet `date32` into datetime64 so the comparison has something to be typed against."""
    return Lit(pd.Timestamp(text))


def all_of(*predicates):
    """`a AND b AND c` — the IR has two-operand booleans, as `cudf::ast` does."""
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = Binary("and", combined, predicate)
    return combined


def any_of(*predicates):
    """`a OR b OR c`. TPC-DS writes long `IN` lists, and an `IN` list is this."""
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = Binary("or", combined, predicate)
    return combined


def is_in(expr, values):
    """`expr IN (…)` — the OR chain a planner expands a literal `IN` list into."""
    return any_of(*[Binary("==", expr, Lit(value)) for value in values])


def between(expr, low, high):
    """`expr BETWEEN low AND high`, inclusive as SQL is."""
    return Binary("and", Binary(">=", expr, low), Binary("<=", expr, high))


def select(name, child, *names):
    """A projection that only picks columns, `Alias`ing each to itself.

    The commonest projection in these plans: joins carry every column of both sides, and
    the next operator wants four of them.
    """
    return N.project(name, child, [Alias(Col(column), column) for column in names])


def rename(name, child, pairs):
    """A projection that picks and renames — `[(from, to), …]`.

    Two uses. A self-join needs one side's columns under other names, since a batch cannot
    hold two columns called `d_year`. And the final projection of a query has to produce
    the output names the SQL declared, which is what the oracle compares against.
    """
    return N.project(name, child, [Alias(Col(source), target) for source, target in pairs])


def aggregate_to_one_row(name, child, aggs, schema_frame=None):
    """The keyless aggregate sequence: per batch, per lane, then once over the lanes.

    `schema_frame` is a zero-row frame carrying the aggregate's input columns. Without one
    a lane that filtered everything away cannot emit the batch its SingleBatch output owes
    — which is not hypothetical here: q19 keeps a few thousand of six million rows, so at
    eight lanes some lanes are legitimately empty.
    """
    schemas = agg_schemas(schema_frame, [], aggs) if schema_frame is not None else (None, None)
    partial = N.partial_aggregate(f"{name}_partial", child, [], aggs)
    per_lane = N.aggregate_batches(f"{name}_batches", partial, [], aggs, schema=schemas[0])
    return N.aggregate_batches(
        f"{name}_final", N.merge_partitions(f"{name}_merge", per_lane), [], aggs,
        A.finalize_exprs(aggs), schema=schemas[1],
    )


def aggregate_by(name, child, keys, aggs, lanes=1, schemas=(None, None),
                 schema_frame=None, grouping_sets=None):
    """The grouped aggregate sequence. With more than one lane it shuffles on the group
    keys, which is the whole point of the sequence; at one lane the merge suffices.

    `grouping_sets` is the ROLLUP the spec's reporting queries ask for: the partial emits
    one row per set per group and tags it with a grouping id, and every phase after that
    treats the id as another key.
    """
    grouped = list(keys) + ([A.GROUPING_ID] if grouping_sets else [])
    if schema_frame is not None:
        if grouping_sets:
            # The expanding init's output carries the grouping id between the keys and the
            # state, so an empty lane's batch has to as well — `agg_schemas` would derive
            # the un-expanded shape and the lane would emit a batch of the wrong width.
            state = A.partial_over_sets(schema_frame.iloc[0:0], list(keys), aggs,
                                        grouping_sets)
            schemas = dict(state.dtypes), dict(A.final(state, grouped, aggs).dtypes)
        else:
            schemas = agg_schemas(schema_frame, keys, aggs)
    state_schema, final_schema = schemas
    partial = N.partial_aggregate(f"{name}_partial", child, keys, aggs,
                                  grouping_sets=grouping_sets)
    per_lane = N.aggregate_batches(f"{name}_batches", partial, grouped, aggs,
                                   schema=state_schema)
    if lanes == 1:
        collapsed = N.merge_partitions(f"{name}_merge", per_lane)
    else:
        collapsed = N.emit_partitions(
            f"{name}_emit",
            N.coalesce_all(f"{name}_shuffle_in", N.merge_partitions(f"{name}_merge", per_lane)),
            grouped, lanes,
        )
    return N.aggregate_batches(
        f"{name}_final", collapsed, grouped, aggs, A.finalize_exprs(aggs), schema=final_schema
    )


def sorted_output(name, child, by, ascending, fetch=None, nulls_first=False):
    """The sort decomposition: each lane sorts and trims, then one k-way merge."""
    return N.merge_sorted_partitions(
        f"{name}_merge",
        N.sort(name, child, by, ascending=ascending, fetch=fetch, nulls_first=nulls_first),
        by, ascending=ascending, fetch=fetch, nulls_first=nulls_first,
    )
