"""What every TPC-DS lowering in this suite is built out of.

TPC-DS is a star schema, and a star has one shape: a fact table streaming past dimension
tables small enough to sit in one batch. That shape is the shape the batch-partitioned mode
exists for, so it is worth saying once — `fact`, `dim` and `star` below — and then spending
each query's lines on what is actually specific to it.

The family modules (`plans_tpcds_star.py` and its siblings) hold the queries; this holds
the vocabulary and `plans_tpcds.py` holds the registry.
"""

from __future__ import annotations

from . import corpus
from ..operators import nodes as N

#: Rows per batch inside a fact scan. Small enough that a probe side really streams — the
#: property under test — and large enough that per-batch overhead does not dominate a
#: 2.9M-row scan.
FACT_ROWS = 250_000

#: A dimension small enough to sit in one batch is read as one. The two that are not
#: (customer_demographics at 1.9M rows, inventory at 11.7M) are declared facts instead and
#: stream through their filter, which is what they are.
DIMENSION_ROWS = 100_000


def fact(t, name, columns, tag=None, rows=FACT_ROWS):
    """A fact table's scan: batched, so the probe side of the joins above it really streams.

    Returns `(frame, node)`. The frame is not read — the plan builders never touch a row —
    but a join needs it for the schema an empty lane would have to emit.
    """
    frame = t(name, columns)
    return frame, N.scan(tag or name, frame, 1, rows, rows)


def dim(t, name, columns, predicate=None, tag=None, rows=DIMENSION_ROWS):
    """A dimension's scan and its local predicate, as `(frame, node)`.

    One batch, because a dimension is the build side of the join it feeds and a build side
    is collected anyway. The predicate goes here rather than above the join for the reason
    every planner pushes it down: filtering 18,000 items before the hash table is built is
    free, filtering 2.9M joined rows afterwards is not.
    """
    frame = t(name, columns)
    node = N.scan(tag or name, frame, 1, rows)
    if predicate is not None:
        node = N.filter_(f"{tag or name}_where", node, predicate)
    return frame, node


def star(probe, *joins):
    """Chain one build join per dimension onto a streaming fact side.

    Each join is `(name, (frame, node), build_key, probe_key)`. Order matters only for the
    plan's readability and for which columns are in scope above — every one of these is an
    inner equi-join on a surrogate key, so the answer does not depend on it.
    """
    for name, (frame, node), build_key, probe_key in joins:
        probe = corpus.build_join(name, frame, node, probe, build_key, probe_key)
    return probe


def registry():
    """A family module's `(QUERIES, ORDER_BY, query)`.

    Each module keeps its own pair of dicts and `plans_tpcds.py` merges them, so a query
    registered in one family cannot silently land in another's registry — which it did,
    once, when the decorator was imported from a sibling module instead of made here.

    `query(name, order_by=…)` decorates a builder with the output columns its ORDER BY
    names. The oracle compares those positionally and the rows as a multiset, which is the
    pair of things a SQL query with an ORDER BY actually determines — see
    `corpus.matches_oracle`. So the ORDER BY belongs beside the plan, in the query's own
    output names.
    """
    queries: dict = {}
    ordering: dict = {}

    def query(name, order_by=()):
        def register(build):
            assert name not in queries, f"{name} registered twice"
            queries[name] = build
            ordering[name] = tuple(order_by)
            return build

        return register

    return queries, ordering, query
