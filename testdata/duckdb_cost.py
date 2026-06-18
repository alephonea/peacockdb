#!/usr/bin/env python3
"""DuckDB cost-oracle extraction (Task 5/6, generation/extraction split #70).

Two phases, two subcommands:

  generate (verda, needs duckdb + data): the gen script runs each query under JSON
    profiling and calls `normalize` to PERSIST a lean subset of the profiling tree
    to testdata/duckdb-profiles/{tpch,tpcds}/<q>.json (committed).

  extract (pure, runnable anywhere — no duckdb, no data): reads a persisted JSON
    and writes the <q>.duckdb_cost.txt golden. Because it never re-executes a
    query, golden regens for annotation/model tweaks are extraction-only.

Cost model (PROVISIONAL, directional-only — the report only displays it):
`duckdb_cost = Σ bytes materialized at pipeline breakers`.
  - Streaming operators (PROJECTION, FILTER, …) -> 0.
  - Build-from-input breakers (group-by/sort/top-n/window/merge-join) -> Σ children
    output_bytes (the INPUT they buffer, not their own tiny output).
  - Join breakers -> own output_bytes + each child's output_bytes, mirroring
    PeacockDB's Σ-over-every-node. Children that already self-count their output
    (scans via the two-part cost, nested joins) or that re-read an
    already-materialized buffer (DELIM_SCAN/CTE_SCAN/COLUMN_DATA_SCAN) are
    excluded to avoid double-counting.
  - TABLE_SCAN: two-part = bytes_read (storage, post-prune/pre-filter) +
    output_bytes (post inline-filter), mirroring PeacockDB's split scan+filter.

rows_read (operator_rows_scanned) is thread-sensitive, so goldens MUST be
generated single-threaded (`PRAGMA threads=1`); the gen script enforces that.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

SCAN_OPS = {"TABLE_SCAN"}
BUILD_INPUT_OPS = {
    "HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY", "UNGROUPED_AGGREGATE",
    "ORDER_BY", "TOP_N", "WINDOW", "PIECEWISE_MERGE_JOIN",
}
JOIN_BUILD_OPS = {
    "HASH_JOIN", "NESTED_LOOP_JOIN", "CROSS_PRODUCT",
    "RIGHT_DELIM_JOIN", "LEFT_DELIM_JOIN",
}
# Strictly binary joins (probe=left, build=right). DELIM joins carry extra
# delim-machinery children (>2), so they're handled leniently below.
BINARY_JOIN_OPS = {"HASH_JOIN", "NESTED_LOOP_JOIN", "CROSS_PRODUCT"}
# Scans that RE-READ an already-materialized buffer (CTE result, delim-dedup,
# column-data collection). Their output is already counted at the producer, so a
# parent breaker must NOT add it again (would double-count).
REREAD_OPS = {"DELIM_SCAN", "CTE_SCAN", "COLUMN_DATA_SCAN"}
STREAMING_OPS = {
    "PROJECTION", "FILTER", "UNION", "STREAMING_LIMIT", "DUMMY_SCAN",
    "CTE", "CTE_SCAN", "DELIM_SCAN", "COLUMN_DATA_SCAN", "RESULT_COLLECTOR",
}

# extra_info keys the extraction uses (for the lean persisted subset + annotations).
KEEP_EXTRA = [
    "Table", "Projections", "Filters", "Join Type", "Conditions",
    "Groups", "Aggregates", "Order By", "Top",
]


class Node:
    """One operator in the profiling tree (measured fields + extra_info + kids)."""

    def __init__(self, op: str, output_bytes: int, output_rows: int,
                 rows_scanned: int, extra: dict):
        self.op = op
        self.output_bytes = output_bytes
        self.output_rows = output_rows
        self.rows_scanned = rows_scanned  # operator_rows_scanned (scans; threads=1)
        self.extra = extra
        self.children: list["Node"] = []


def normalize(raw: dict) -> Optional[dict]:
    """Trim a raw DuckDB profiling document to the lean subset the extraction
    needs, recursively. Keeps the measured fields, the KEEP_EXTRA annotations, and
    the tree shape (incl. the top-level query wrapper, which has no operator_type
    but carries the root operator as its child)."""
    def rec(n: dict) -> Optional[dict]:
        if not isinstance(n, dict):
            return None
        out: dict = {}
        for k in ("operator_type", "operator_cardinality", "result_set_size",
                  "operator_rows_scanned"):
            if k in n:
                out[k] = n[k]
        ei = n.get("extra_info")
        if isinstance(ei, dict):
            sub = {k: ei[k] for k in KEEP_EXTRA if k in ei}
            if sub:
                out["extra_info"] = sub
        kids = [k for k in (rec(c) for c in (n.get("children") or [])) if k]
        if kids:
            out["children"] = kids
        return out or None
    return rec(raw)


def parse_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_tree(doc: dict) -> Optional[Node]:
    """Build the operator tree, dropping the query wrapper (no operator_type)."""
    def rec(raw: dict) -> Optional[Node]:
        if not isinstance(raw, dict):
            return None
        node = None
        if "operator_type" in raw:
            node = Node(
                op=raw["operator_type"],
                output_bytes=int(raw.get("result_set_size", 0) or 0),
                output_rows=int(raw.get("operator_cardinality", 0) or 0),
                rows_scanned=int(raw.get("operator_rows_scanned", 0) or 0),
                extra=raw.get("extra_info") or {},
            )
        kids = [k for k in (rec(c) for c in (raw.get("children") or [])) if k]
        if node is None:
            return kids[0] if kids else None
        node.children = kids
        return node
    return rec(doc)


def as_list(raw) -> Optional[list[str]]:
    """DuckDB's list-ish extra_info values are inconsistently typed: a bare string
    for one element, a list for 2+, and '' / missing for none. Normalise to a list
    of non-empty strings, or None. (Single-element values — e.g. a 1-column scan's
    Projections='d_date_sk' — were previously dropped.)"""
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw] if raw else None
    if isinstance(raw, list):
        names = [str(x) for x in raw if str(x)]
        return names or None
    return None


# Kept as a named alias because the extraction (and tests) refer to projection
# normalisation specifically; it's the same rule as any list-ish extra_info value.
normalize_projections = as_list


def annotation(node: Node) -> str:
    """DuckDB's own extra_info annotation for this operator, mirroring how the
    .cpu.txt nodes annotate (annotations first, cost fields after). Expressions are
    surfaced in DuckDB's representation as-is (a different engine from PeacockDB —
    not rewritten). Empty string when the operator has no annotation."""
    op, ei = node.op, node.extra
    parts: list[str] = []

    def lst(key: str, label: str) -> None:
        v = as_list(ei.get(key))
        if v is not None:
            parts.append(f"{label}=[{', '.join(v)}]")

    def scalar(key: str, label: str) -> None:
        v = ei.get(key)
        if v not in (None, ""):
            parts.append(f"{label}={v}")

    if op in SCAN_OPS:
        scalar("Table", "table")
        lst("Projections", "projections")
        filt = ei.get("Filters")
        if filt:
            if isinstance(filt, list):
                filt = " AND ".join(filt)
            parts.append(f'filters="{filt}"')
    elif op in JOIN_BUILD_OPS:
        scalar("Join Type", "join_type")
        lst("Conditions", "conditions")
    elif op in {"HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY", "UNGROUPED_AGGREGATE"}:
        lst("Groups", "groups")
        lst("Aggregates", "aggregates")
    elif op in {"ORDER_BY", "TOP_N"}:
        lst("Order By", "order_by")
        if op == "TOP_N":
            scalar("Top", "top")
    elif op == "PROJECTION":
        lst("Projections", "projections")
    return ", ".join(parts)


def scan_cost(node: Node) -> tuple[int, int, int]:
    """TABLE_SCAN two-part cost -> (bytes_read, rows_read, materialized).

    bytes_read is DERIVED (rows_read × output-row width), not a measured storage
    byte count. KNOWN LIMITATION: when output_rows == 0 (an empty post-filter
    result, e.g. a fully-pruned scan) the per-row width is unknown, so bytes_read
    collapses to 0 and the scan contributes only its (0-byte) output — i.e. a
    fully-selective scan is under-weighted here. Acceptable for a provisional,
    directional proxy; revisit with a schema-derived min width when the model is
    refined."""
    rows_read = node.rows_scanned
    per_row = (node.output_bytes / node.output_rows) if node.output_rows else 0
    bytes_read = int(rows_read * per_row)
    return bytes_read, rows_read, bytes_read + node.output_bytes


def node_materialized(node: Node, warn) -> int:
    """Bytes `node` materialises under the pipeline-breaker model (0 if streaming)."""
    op = node.op
    if op in SCAN_OPS:
        return scan_cost(node)[2]
    if op in BUILD_INPUT_OPS:
        return sum(c.output_bytes for c in node.children if c.op not in REREAD_OPS)
    if op in JOIN_BUILD_OPS:
        # Align joins with PeacockDB, which (Σ over every node's output) counts both
        # join inputs' outputs AND the join's own output. So a join contributes its
        # OWN output_bytes plus each child's output_bytes — EXCEPT children that
        # already count their own output in their own materialized (scans, via the
        # two-part cost; nested joins, via this same rule) or that re-read an
        # already-materialized buffer (REREAD_OPS, e.g. a DELIM_SCAN under a
        # RIGHT/LEFT_DELIM_JOIN) — all excluded to avoid double-counting. Build-input
        # breakers and ordinary streaming children don't self-count their output, so
        # their output is added here (once). Binary joins assert two children so a
        # profiling-shape change fails loudly; DELIM joins (>2 children) don't.
        if op in BINARY_JOIN_OPS:
            assert len(node.children) == 2, f"{op} expected 2 children, got {len(node.children)}"
        total = node.output_bytes
        for c in node.children:
            if c.op not in SCAN_OPS and c.op not in JOIN_BUILD_OPS and c.op not in REREAD_OPS:
                total += c.output_bytes
        return total
    if op not in STREAMING_OPS and node.output_bytes > 4096:
        warn(op, node.output_bytes)
    return 0


def format_node_line(node: Node, depth: int, warn) -> str:
    """One golden line: indent + op + annotation + cost fields (annotation first,
    matching .cpu.txt). `materialized` is the breaker contribution; scans also
    surface bytes_read/rows_read."""
    fields: list[str] = []
    ann = annotation(node)
    if ann:
        fields.append(ann)
    fields.append(f"output_bytes={node.output_bytes}")
    fields.append(f"output_rows={node.output_rows}")
    fields.append(f"materialized={node_materialized(node, warn)}")
    if node.op in SCAN_OPS:
        bytes_read, rows_read, _ = scan_cost(node)
        # *_est: bytes_read is DERIVED (rows_read × output-row width), not a
        # measured storage byte count. rows_read IS measured (operator_rows_scanned).
        fields.append(f"bytes_read_est={bytes_read}")
        fields.append(f"rows_read={rows_read}")
    return f"{'  ' * depth}{node.op}: " + ", ".join(fields)


def build_cost_tree(root: Node, warn) -> tuple[list[str], int]:
    """Render the tree to golden lines and return (lines, Σ materialized)."""
    lines: list[str] = []
    total = 0

    def walk(node: Node, depth: int) -> None:
        nonlocal total
        lines.append(format_node_line(node, depth, warn))
        total += node_materialized(node, warn)
        for c in node.children:
            walk(c, depth + 1)

    walk(root, 0)
    return lines, total


def extract(input_json: str, output_golden: str) -> int:
    """Read a (persisted or raw) profiling JSON; write the golden; return total."""
    warnings: set[str] = set()
    root = build_tree(parse_json(input_json))
    if root is None:
        raise SystemExit(f"no operator tree in {input_json}")
    lines, total = build_cost_tree(root, lambda op, ob: warnings.add(
        f"unclassified operator '{op}' (output_bytes={ob}) contributes 0"))
    # Explicit total footer (the per-node `materialized=` values above are the
    # contribution breakdown that sums to it). cost-report reads this line.
    lines.append(f"duckdb_cost={total}")
    with open(output_golden, "w") as f:
        f.write("\n".join(lines) + "\n")
    for w in sorted(warnings):
        print(f"warning: {w}", file=sys.stderr)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_norm = sub.add_parser("normalize", help="raw profiling JSON -> lean persisted JSON")
    p_norm.add_argument("raw_json")
    p_norm.add_argument("out_json")

    p_ext = sub.add_parser("extract", help="persisted JSON -> <q>.duckdb_cost.txt golden")
    p_ext.add_argument("input_json")
    p_ext.add_argument("output_golden")

    args = ap.parse_args()
    if args.cmd == "normalize":
        doc = normalize(parse_json(args.raw_json))
        if doc is None:
            raise SystemExit(f"empty profile {args.raw_json}")
        with open(args.out_json, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        return 0
    if args.cmd == "extract":
        print(extract(args.input_json, args.output_golden))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
