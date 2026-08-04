#!/usr/bin/env python3
"""DuckDB cost-oracle extraction (Task 5/6/#10).

THREE subcommands, two committed inputs per query:

  normalize  (pass 1, JFP OFF): raw deterministic cost profile -> lean profile,
    committed to testdata/duckdb-profiles/{tpch,tpcds}/<q>.json.
  dynfilters (pass 2, JFP ON): raw profile -> per-scan dynamic-filter min/max BOUNDS
    only (the flaky cardinalities are discarded), committed to
    testdata/duckdb-dynfilters/{tpch,tpcds}/<q>.json.
  extract: COMBINES the two inputs (+ parquet row-group stats) -> the
    <q>.duckdb_cost.txt golden. Needs the parquet present (for the pruning section);
    no duckdb / no query re-run.

Why two passes: DuckDB's join_filter_pushdown installs OPTIONAL min/max dynamic
filters on probe scans, applied opportunistically (build-vs-probe race) -> scan
cardinality is NONDETERMINISTIC even at threads=1. Pass 1 disables it for a
deterministic cost tree; pass 2 keeps it ON solely to OBSERVE the deterministic
bound VALUES (never the flaky counts).

Cost = TWO deterministic, separately-weightable components (golden footer reports
both + their sum):
  - materialization_total = Σ node_materialized (pipeline-breaker model; a SCAN
    contributes its OUTPUT only — post-static, capped at read). Streaming -> 0;
    build/join breakers -> children/own output (double-count-avoided as before).
  - storage_read_total = Σ scan bytes_read = the DECODED Arrow (in-memory) bytes of the
    surviving row groups' referenced columns (post pass1-static ∩ pass2-dynamic min/max
    pruning), in the SAME units as peacockdb's GpuScanExec output_bytes so the
    duckdb-vs-peacockdb cost ratio is apples-to-apples (IMPORTANT-1). The row-group-
    pruning section separately reports bytes_fetched in COMPRESSED parquet bytes (the
    storage/disk-IO-reduction metric).
  - duckdb_cost = materialization_total + storage_read_total.
bytes_read (decoded Arrow, read basis) and out_bytes/materialization (DuckDB
result_set_size, output basis) are DIFFERENT measurement bases, so bytes_read < out_bytes
can legitimately occur on small scans and is NOT clamped. out_rows IS
capped at rows_read (same basis = row counts). threads=1 throughout for reproducibility.
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
        self.rows_scanned = rows_scanned  # operator_rows_scanned (post-static, JFP-off)
        self.extra = extra
        self.children: list["Node"] = []
        # Filled by compute_scan_pruning (extract): the deterministic post-prune scan,
        # from pass-1 static ∩ pass-2 dynamic min/max bounds applied to parquet stats.
        # When set, the scan's read term uses rows_fetched (post-prune) — so duckdb_cost
        # reflects BOTH inputs' pruning, not just the post-static rows.
        self.pruning = None
        self.rows_fetched = None
        self.dyn_filter = ""


# --- dynamic filters (pass 2: JFP-on JSON profile) ---------------------------
# DuckDB's runtime join (dynamic) filters appear in the RAW JSON profile (with
# join_filter_pushdown ON) as a per-scan extra_info "Dynamic Filters" string/list.
# We capture only the deterministic min/max RANGE bounds (the row counts they
# reduce are the flaky, opportunistic race — never persisted). Non-range terms
# (bloom "IN BF(...)", "IN (...)") are ignored; only col >=/<=/>/</= literal.

def dynamic_filter_str(v) -> str:
    """Normalize a 'Dynamic Filters' extra_info value (str or list) to a clean,
    range-only filter string: drop the 'optional:' markers and any non-range term
    (bloom/IN), keep only `col <op> literal` clauses joined by AND."""
    import re
    if not v:
        return ""
    text = " AND ".join(str(p) for p in v) if isinstance(v, list) else str(v)
    text = text.replace("optional:", " ")
    keep = []
    for clause in re.split(r"\s+AND\s+", text):
        c = clause.strip()
        if not c or re.search(r"\bIN\b", c):
            continue
        if re.match(r"^[A-Za-z_]\w*\s*(>=|<=|>|<|=)", c):
            keep.append(c)
    return " AND ".join(keep)


def extract_dynamic_filters(raw: dict) -> list:
    """Pre-order list (one entry per TABLE_SCAN) of cleaned dynamic-filter range
    strings from a JFP-on profile; '' where a scan has none. Pre-order matches the
    cost profile's scan order (the static plan is identical JFP on/off), so the
    list indexes 1:1 with the cost-tree scans."""
    out: list = []

    def rec(n):
        if not isinstance(n, dict):
            return
        if n.get("operator_type") == "TABLE_SCAN":
            out.append(dynamic_filter_str((n.get("extra_info") or {}).get("Dynamic Filters")))
        for c in n.get("children") or []:
            rec(c)
    rec(raw)
    return out


def merge_ranges(a: dict, b: dict) -> dict:
    """Intersect two {col:{lo,hi}} range maps (static ∩ dynamic per column)."""
    out = {k: dict(v) for k, v in a.items()}
    for col, r in b.items():
        if col not in out:
            out[col] = dict(r)
            continue
        o = out[col]
        if r["lo"] is not None:
            o["lo"] = r["lo"] if o["lo"] is None else max(o["lo"], r["lo"])
        if r["hi"] is not None:
            o["hi"] = r["hi"] if o["hi"] is None else min(o["hi"], r["hi"])
    return out


def normalize(raw: dict) -> Optional[dict]:
    """Trim a raw DuckDB profiling document to the lean subset the extraction
    needs, recursively. Keeps the measured fields, the KEEP_EXTRA annotations, and
    the tree shape (incl. the top-level query wrapper, which has no operator_type
    but carries the root operator as its child).

    This is PASS 1's committed artifact (JFP-off cost profile) — pure, no derived
    pruning. The storage-pruning section is computed by `extract`, which combines
    this profile (static filters) with pass 2's dynamic-filter bounds + the parquet
    stats."""
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


# --- row-group pruning (storage-reduction) section ---------------------------
# Parquet-stats-based, deterministic, no duckdb execution. Computed at GENERATE
# time (where the parquet lives) and PERSISTED per scan into the lean JSON, so
# `extract` stays data-free (CI re-extracts without parquet/pyarrow). Reports the
# ACTUAL compressed storage bytes of the referenced (projection+filter) columns in
# the row groups that SURVIVE the pushed filter's min/max test, vs. all row groups
# — i.e. what a stats-pruning scan fetches from storage. Distinct from the cost
# tree's bytes_read_est (rows x width). Decimal columns: pyarrow can't surface
# their stats, so a predicate on one can't prune (conservative: keep the group).

def table_base(name: str) -> str:
    """Strip catalog/schema qualifiers from a DuckDB table name. 1.5.4 emits the
    fully-qualified `<catalog>.<schema>.<table>` (catalog = our cache-db filename, a
    gen artifact); the base name matches the parquet file + PeacockDB's naming."""
    return name.rsplit(".", 1)[-1] if name else name


def _parse_literal(tok: str):
    """Parse a DuckDB filter RHS literal to a comparable python value, or None if
    not a prunable type. Strips quotes and a ::TYPE cast; recognises ISO dates,
    timestamps (date part), and integers (the keys clustering uses).
    Decimals/strings/floats -> None (skip)."""
    import datetime
    t = tok.strip()
    if "::" in t:
        t = t.split("::", 1)[0].strip()
    if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
        t = t[1:-1]
    # ISO date YYYY-MM-DD, or timestamp YYYY-MM-DD HH:MM:SS (take the date part —
    # 1.5.4 rewrites `date < DATE 'x'` as `CAST(col AS TIMESTAMP) < TIMESTAMP 'x 00:00:00'`).
    if len(t) >= 10 and t[4] == "-" and t[7] == "-":
        try:
            return datetime.date.fromisoformat(t[:10])
        except ValueError:
            return None
    try:
        return int(t)
    except ValueError:
        return None


def parse_range_filters(filters) -> dict:
    """Parse a pushed Filters expression into per-column [lo, hi] inclusive bounds
    usable for row-group min/max pruning. Handles `col OP literal` clauses joined by
    AND (OP in >=,>,<=,<,=); unparseable clauses (decimal/string/OR/functions) are
    ignored so they never prune. Returns {col: {'lo':v|None, 'hi':v|None}}."""
    import re
    if not filters:
        return {}
    if isinstance(filters, list):
        filters = " AND ".join(filters)
    ranges: dict = {}
    # LHS is a bare column or `CAST(col AS type)`; the clause may be paren-wrapped
    # (1.5.4 wraps the rewritten timestamp comparison). RHS captured up to an
    # optional closing paren.
    clause_re = re.compile(
        r"^\(?\s*(?:CAST\(\s*([A-Za-z_]\w*)\s+AS\s+\w+\s*\)|([A-Za-z_]\w*))"
        r"\s*(>=|<=|>|<|=)\s*(.+?)\s*\)?$")
    for clause in re.split(r"\s+AND\s+", filters):
        m = clause_re.match(clause.strip())
        if not m:
            continue
        col, op, rhs = (m.group(1) or m.group(2)), m.group(3), m.group(4)
        val = _parse_literal(rhs)
        if val is None:
            continue
        r = ranges.setdefault(col, {"lo": None, "hi": None})
        if op in (">=", ">", "="):
            r["lo"] = val if r["lo"] is None else max(r["lo"], val)
        if op in ("<=", "<", "="):
            r["hi"] = val if r["hi"] is None else min(r["hi"], val)
    return ranges


def filter_columns(filters) -> set:
    """All column names appearing as the LHS of a comparison in a Filters string
    (incl. decimal/string cols that are read from storage but can't prune). Used to
    size the bytes fetched (projection + filter columns)."""
    import re
    if not filters:
        return set()
    if isinstance(filters, list):
        filters = " AND ".join(filters)
    cols = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:>=|<=|>|<|=)", filters))
    cols |= set(re.findall(r"CAST\(\s*([A-Za-z_]\w*)\s+AS\b", filters))  # cols inside CAST(...)
    cols.discard("CAST")
    return cols


def rowgroup_survives(stats: dict, ranges: dict) -> bool:
    """Does a row group survive the filter, given its per-column {min,max} stats?
    A group is pruned only if some ranged column has stats AND its [min,max] cannot
    intersect the predicate bound. Missing stats (e.g. decimals) => can't prune =>
    survive (conservative)."""
    for col, r in ranges.items():
        s = stats.get(col)
        if not s or s.get("min") is None or s.get("max") is None:
            continue
        lo, hi, cmin, cmax = r["lo"], r["hi"], s["min"], s["max"]
        # The bound literal and the parquet stat must be the same comparable type
        # (e.g. both int, both date). A mismatch (str-typed stat vs an int/date bound)
        # means we can't reason about this column -> can't prune -> survive.
        try:
            if lo is not None and cmax < lo:
                return False
            if hi is not None and cmin > hi:
                return False
        except TypeError:
            continue
    return True


def compute_pruning_from_rowgroups(rowgroups: list, ref_cols: set, ranges: dict) -> dict:
    """Pure core (unit-tested): given per-row-group stats, compute survivors.
    rowgroups: [{'num_rows':int, 'cols':{name:{'min','max','compressed':int}}}].
    Returns rows/bytes fetched (surviving groups) vs total over ref_cols."""
    rg_total = len(rowgroups)
    rg_kept = rows_total = rows_kept = bytes_total = bytes_kept = 0
    for rg in rowgroups:
        cols = rg["cols"]
        rg_bytes = sum(cols[c]["compressed"] for c in ref_cols if c in cols)
        survives = rowgroup_survives(cols, ranges)
        rows_total += rg["num_rows"]
        bytes_total += rg_bytes
        if survives:
            rg_kept += 1
            rows_kept += rg["num_rows"]
            bytes_kept += rg_bytes
    return {
        "row_groups_kept": rg_kept, "row_groups_total": rg_total,
        "rows_fetched": rows_kept, "rows_total": rows_total,
        "bytes_fetched": bytes_kept, "bytes_total": bytes_total,
    }


def compute_pruning(parquet_path: str, ref_cols: set, ranges: dict) -> Optional[dict]:
    """Read a parquet file's row-group stats (pyarrow) and compute pruning. Returns
    None if the file/columns are unreadable. pyarrow imported lazily so the module
    (and `extract`) work without it."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    try:
        pf = pq.ParquetFile(parquet_path)
        md = pf.metadata
    except Exception:
        return None
    names = [md.schema.column(i).name for i in range(md.num_columns)]
    idx = {n: i for i, n in enumerate(names)}
    rowgroups = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        cols = {}
        for name in ref_cols | set(ranges):
            j = idx.get(name)
            if j is None:
                continue
            col = rg.column(j)
            st = col.statistics
            # pyarrow raises ArrowNotImplementedError reading min/max of some types
            # (e.g. DECIMAL) even when has_min_max is set — treat as no stats so the
            # column simply can't prune (conservative), rather than crashing.
            try:
                mn = st.min if (st and st.has_min_max) else None
                mx = st.max if (st and st.has_min_max) else None
            except Exception:
                mn = mx = None
            cols[name] = {"min": mn, "max": mx, "compressed": col.total_compressed_size}
        rowgroups.append({"num_rows": rg.num_rows, "cols": cols})
    result = compute_pruning_from_rowgroups(rowgroups, set(ref_cols), ranges)

    # DECODED (Arrow in-memory) bytes of the surviving row groups' referenced columns —
    # the cost's storage-read term, in the SAME units as peacockdb's GpuScanExec
    # output_bytes (Arrow nbytes: Decimal128=16B, Date32=4B, int64=8B, double=8B, string
    # = offsets+content, + validity), so the duckdb-vs-peacockdb ratio is apples-to-apples
    # (IMPORTANT-1). Distinct from bytes_fetched, which stays COMPRESSED parquet bytes for
    # the storage(disk-IO)-reduction section. Read only the surviving groups' ref columns.
    ref_in_file = [c for c in ref_cols if c in idx]
    survivors = [i for i, rg in enumerate(rowgroups) if rowgroup_survives(rg["cols"], ranges)]
    decoded = 0
    if ref_in_file:
        for i in survivors:
            try:
                decoded += pf.read_row_group(i, columns=ref_in_file).nbytes
            except Exception:
                return result  # leave bytes_fetched_decoded absent -> scan_cost falls back
    result["bytes_fetched_decoded"] = decoded
    return result


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
        tb = ei.get("Table")
        if tb not in (None, ""):
            parts.append(f"table={table_base(tb)}")
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


def scan_cost(node: Node) -> dict:
    """TABLE_SCAN cost terms -> {rows_read, bytes_read, out_rows, out_bytes}.

    rows_read is the POST-PRUNE scan size: `rows_fetched` (rows in the surviving row
    groups under the pass1-static ∩ pass2-dynamic min/max bounds) when pruning was
    computed, else the post-static `operator_rows_scanned`. So duckdb_cost credits the
    deterministic row-group pruning from BOTH inputs.

    bytes_read = the DECODED Arrow (in-memory) bytes of those surviving row groups'
    referenced columns (pruning['bytes_fetched_decoded']) — the READ basis, in the SAME
    units as peacockdb's GpuScanExec output_bytes so the duckdb-vs-peacockdb cost ratio
    is apples-to-apples (IMPORTANT-1; dmitry). Falls back to the derived rows_read ×
    output-row-width only when no parquet stats are available. (NOT the compressed
    parquet bytes — pruning['bytes_fetched'] stays the storage/disk-IO-reduction metric
    in the row-group-pruning section.)

    TWO MEASUREMENT BASES — bytes_read is NOT clamped to output. bytes_read is the Arrow
    read size; out_bytes / materialization use DuckDB's result_set_size (output basis).
    Because these are DIFFERENT bases, bytes_read < out_bytes can LEGITIMATELY occur on
    small (dim-table) scans — an artifact of comparing two systems' byte accounting, NOT
    a bug, left UNCLAMPED. read>=output is a real invariant only
    WITHIN one basis, so it is NOT asserted across the read(Arrow)/output(DuckDB) seam.

    out_rows IS capped at rows_read — that cap is within ONE basis (row counts), so it's
    a real invariant: a scan can't output more rows than the surviving groups hold. It
    bites BY DESIGN for dynamic-filter facts (pass-1 JFP-off emits full cardinality while
    rows_read applies the pass-2 dynamic bounds -> output_rows > rows_read) — that case is
    silent. compute_scan_pruning WARNS only when output_rows > rows_read with NO dynamic
    filter, which means static bound-extraction over-pruned a row group DuckDB kept (a
    real bug). Returns a dict."""
    rows_read = node.rows_fetched if node.rows_fetched is not None else node.rows_scanned
    per_row = (node.output_bytes / node.output_rows) if node.output_rows else 0
    if node.pruning is not None and node.pruning.get("bytes_fetched_decoded") is not None:
        bytes_read = node.pruning["bytes_fetched_decoded"]  # DECODED Arrow read (peacockdb units)
    else:
        bytes_read = int(rows_read * per_row)          # derived fallback (no parquet)
    out_rows = min(node.output_rows, rows_read)        # row-count cap (same basis); warned when it bites
    out_bytes = int(out_rows * per_row)
    return {"rows_read": rows_read, "bytes_read": bytes_read,
            "out_rows": out_rows, "out_bytes": out_bytes}


def scan_bytes_read(node: Node) -> int:
    """Storage-read component of a scan (post-prune DECODED Arrow bytes, peacockdb
    units). SEPARATE, weightable total — NOT part of node_materialized (avoids
    double-counting)."""
    return scan_cost(node)["bytes_read"] if node.op in SCAN_OPS else 0


def node_materialized(node: Node, warn) -> int:
    """Bytes `node` materialises under the pipeline-breaker model (0 if streaming).
    For a scan this is its OUTPUT only (post-filter, capped); the storage-read term is
    tracked separately by scan_bytes_read and added to the total once."""
    op = node.op
    if op in SCAN_OPS:
        return scan_cost(node)["out_bytes"]
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
    if node.op in SCAN_OPS:
        sc = scan_cost(node)
        # output (materialization) is row-capped at the read rows; bytes_read (decoded
        # Arrow) and rows_read are the POST-PRUNE storage read (rows_fetched when pruned).
        fields.append(f"output_bytes={sc['out_bytes']}")
        fields.append(f"output_rows={sc['out_rows']}")
        fields.append(f"materialized={node_materialized(node, warn)}")  # = output (scan)
        fields.append(f"bytes_read={sc['bytes_read']}")     # storage read (post-prune, actual)
        fields.append(f"rows_read={sc['rows_read']}")
    else:
        fields.append(f"output_bytes={node.output_bytes}")
        fields.append(f"output_rows={node.output_rows}")
        fields.append(f"materialized={node_materialized(node, warn)}")
    return f"{'  ' * depth}{node.op}: " + ", ".join(fields)


def build_cost_tree(root: Node, warn) -> tuple[list[str], int, int]:
    """Render the tree to golden lines and return (lines, Σ materialized, Σ scan
    bytes_read). The two totals are the weightable cost components: materialization
    (pipeline-breaker model) and storage-read (post-prune scan bytes)."""
    lines: list[str] = []
    materialized_total = 0
    read_total = 0

    def walk(node: Node, depth: int) -> None:
        nonlocal materialized_total, read_total
        lines.append(format_node_line(node, depth, warn))
        materialized_total += node_materialized(node, warn)
        read_total += scan_bytes_read(node)
        for c in node.children:
            walk(c, depth + 1)

    walk(root, 0)
    return lines, materialized_total, read_total


def iter_nodes(root: Node):
    """Pre-order walk (matches the cost-tree line order)."""
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def dim_date_key_minmax(parquet_path: str, key_col: str, ranges: dict):
    """Reconstruct (min,max) of `key_col` (e.g. d_date_sk) over the rows of a dim
    parquet that pass `ranges` (the dim's static filter parsed to col->[lo,hi]). Used
    to CROSS-CHECK a fact scan's OBSERVED dynamic-filter date_sk bound against the
    sibling filtered date_dim. None if unreadable or no rows. pyarrow lazy-imported."""
    try:
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
    except ImportError:
        return None
    try:
        cols = list({key_col} | set(ranges))
        t = pq.read_table(parquet_path, columns=cols)
    except Exception:
        return None
    mask = None
    for col, r in ranges.items():
        if col not in t.column_names:
            return None  # can't faithfully reconstruct -> skip cross-check
        c = t[col]
        if r["lo"] is not None:
            m = pc.greater_equal(c, r["lo"])
            mask = m if mask is None else pc.and_(mask, m)
        if r["hi"] is not None:
            m = pc.less_equal(c, r["hi"])
            mask = m if mask is None else pc.and_(mask, m)
    if mask is not None:
        t = t.filter(mask)
    if t.num_rows == 0:
        return None
    k = t[key_col]
    return (pc.min(k).as_py(), pc.max(k).as_py())


def crosscheck_date_dynfilters(dyn_ranges: dict, dim_date_ranges: list, table: str, warn) -> None:
    """WARN if a fact scan's OBSERVED dynamic-filter *_date_sk bound is NOT CONTAINED in
    any sibling filtered date_dim's min/max(d_date_sk) range. Containment (not exact
    equality) because the reconstructed date_dim range is a SUPERSET — parse_range_filters
    drops clauses it can't parse (d_month_seq BETWEEN, IN-lists, …), so the true bound
    should still fall inside it. A bound that falls OUTSIDE means the dynamic filter came
    from an unexpected source. Pure (testable); never fabricates — the observed bound is
    still used, this only flags a surprise."""
    if not dim_date_ranges:
        return  # no reconstructable date_dim range in this query -> can't cross-check
    for col, r in dyn_ranges.items():
        if not col.endswith("_date_sk"):
            continue
        lo, hi = r["lo"], r["hi"]
        if lo is None or hi is None:
            continue  # need both bounds to containment-check
        contained = any(dmin <= lo and hi <= dmax for dmin, dmax in dim_date_ranges)
        if not contained:
            warn(f"{table}: dynamic-filter {col} bound ({lo},{hi}) falls OUTSIDE every "
                 f"sibling date_dim min/max(d_date_sk) {dim_date_ranges} — unexpected "
                 f"dynamic-filter source; pruning kept (bound is observed, not synthesized).")


def compute_scan_pruning(root: Node, dynamic_filters: Optional[list],
                         data_dir: Optional[str], warn) -> None:
    """Attach per-scan pruning to the tree by COMBINING the two committed inputs —
    pass 1's static `Filters` (this profile) ∩ pass 2's dynamic-filter BOUNDS
    (`dynamic_filters`, by pre-order scan index) — applied to the parquet row-group
    min/max. Sets node.pruning, node.rows_fetched (post-prune, used by the cost tree),
    node.dyn_filter. No-op without data_dir (cost then falls back to post-static)."""
    import os
    if not data_dir:
        return
    scans = [n for n in iter_nodes(root) if n.op in SCAN_OPS]
    # Bounds are aligned to scans by PRE-ORDER INDEX, which is only valid because the
    # JFP-on (pass 2) and JFP-off (pass 1) static plans are identical (verified 121/121).
    # Assert the counts match so any future plan-shape drift FAILS LOUDLY rather than
    # silently mis-attributing or dropping a scan's dynamic-filter bounds.
    if dynamic_filters is not None and len(dynamic_filters) != len(scans):
        raise SystemExit(
            f"dynamic-filter/scan count mismatch: pass2 has {len(dynamic_filters)} "
            f"bound entries but pass1 has {len(scans)} scans — plan shape drifted "
            f"between the JFP-on and JFP-off passes; bounds can't be aligned by index.")
    dynamic_filters = dynamic_filters or []
    for i, s in enumerate(scans):
        table = table_base(s.extra.get("Table", ""))
        stat = s.extra.get("Filters")
        if isinstance(stat, list):
            stat = " AND ".join(stat)
        dyn = dynamic_filters[i] if i < len(dynamic_filters) else ""
        s.dyn_filter = dyn or ""
        if not table:
            continue
        ranges = merge_ranges(parse_range_filters(stat), parse_range_filters(dyn))
        ref = (set(as_list(s.extra.get("Projections")) or [])
               | filter_columns(stat) | filter_columns(dyn))
        p = compute_pruning(os.path.join(data_dir, f"{table}.parquet"), ref, ranges)
        if p:
            s.pruning = p
            s.rows_fetched = p["rows_fetched"]   # post-prune -> feeds scan_cost
        elif ranges:
            warn(f"{table}: parquet stats unreadable -> no-prune (100%)")

        # ROW-cap anomaly warn (same basis = row counts, so a real invariant). Physical
        # truth per scan: no filter -> output_rows == read; static-only -> output_rows
        # <= read (post-static rows can't exceed the surviving groups' rows); dynamic ->
        # output_rows > read EXPECTED (pass1 JFP-off emits full cardinality while
        # rows_fetched applies the pass2 dynamic bounds) — by design, NOT warned. So
        # output_rows > read WITHOUT a dynamic filter means our static bound-extraction
        # pruned a row group DuckDB actually kept (over-prune bug) -> warn.
        eff_read = s.rows_fetched if s.rows_fetched is not None else s.rows_scanned
        if not s.dyn_filter and eff_read is not None and s.output_rows > eff_read:
            warn(f"{table}: output_rows {s.output_rows} > post-prune read rows {eff_read} "
                 f"with NO dynamic filter — static bound-extraction likely pruned a row "
                 f"group DuckDB kept (over-prune bug); out_rows capped.")

    # CROSS-CHECK: each fact scan's OBSERVED dynamic *_date_sk bound should equal the
    # min/max(d_date_sk) of a sibling FILTERED date_dim. Reconstruct the date_dim
    # range(s) independently from parquet and warn on any mismatch (defensive — the
    # bound is observed, never synthesized, so this only surfaces a surprise).
    dim_date_ranges = []
    for s in scans:
        if table_base(s.extra.get("Table", "")) != "date_dim":
            continue
        dr = parse_range_filters(s.extra.get("Filters"))
        if not dr:
            continue  # date_dim filter not parseable to a range -> can't reconstruct
        mm = dim_date_key_minmax(os.path.join(data_dir, "date_dim.parquet"), "d_date_sk", dr)
        if mm:
            dim_date_ranges.append(mm)
    for s in scans:
        # Only fact scans — the date_dim's OWN d_date_sk dynamic filter (from a
        # different join) isn't a "sibling date_dim" cross-check.
        if s.dyn_filter and table_base(s.extra.get("Table", "")) != "date_dim":
            crosscheck_date_dynfilters(parse_range_filters(s.dyn_filter), dim_date_ranges,
                                       table_base(s.extra.get("Table", "?")), warn)


def format_pruning_section(root: Node) -> list[str]:
    """Section (ii): per-scan storage retrieved (ACTUAL parquet compressed bytes of
    the referenced cols) in surviving vs all row groups, from the pruning attached by
    compute_scan_pruning. Distinct from the cost tree's bytes_read_est (rows × width)."""
    scans = [n for n in iter_nodes(root) if n.op in SCAN_OPS]
    if not scans or all(s.pruning is None for s in scans):
        return []
    out = ["", "--- row-group pruning (storage reduction) ---"]
    tot_f = tot_t = 0
    for s in scans:
        table = table_base(s.extra.get("Table", "?"))
        stat = s.extra.get("Filters")
        if isinstance(stat, list):
            stat = " AND ".join(stat)
        ann = []
        if stat:
            ann.append(f"filter=[{stat}]")
        if s.dyn_filter:
            ann.append(f"dyn_filter=[{s.dyn_filter}]")
        annstr = ("  ".join(ann) + "  ") if ann else ""
        p = s.pruning
        if not p:
            out.append(f"{table}  {annstr}(no row-group stats)")
            continue
        rgk, rgt = p["row_groups_kept"], p["row_groups_total"]
        rf, rt = p["rows_fetched"], p["rows_total"]
        bf, bt = p["bytes_fetched"], p["bytes_total"]
        tot_f += bf
        tot_t += bt
        rp = (100.0 * rf / rt) if rt else 100.0
        bp = (100.0 * bf / bt) if bt else 100.0
        out.append(f"{table}  {annstr}row_groups={rgk}/{rgt}  "
                   f"rows={rf}/{rt} ({rp:.1f}%)  bytes_fetched={bf}/{bt} ({bp:.1f}%)")
    tp = (100.0 * tot_f / tot_t) if tot_t else 100.0
    out.append(f"pruning_bytes_fetched={tot_f}/{tot_t} ({tp:.1f}%)")
    return out


def format_breakdown_section(root: Node) -> list[str]:
    """Section (iii): per-operator share of duckdb_cost (Σ materialized). Sorted by
    contribution desc; scans labelled with their table. Stays in duckdb_cost units."""
    nowarn = lambda *a: None  # noqa: E731 (already warned during the cost tree)
    items: list[tuple[str, int]] = []
    for n in iter_nodes(root):
        m = node_materialized(n, nowarn)
        if m > 0:
            label = (f"{n.op}({table_base(n.extra['Table'])})"
                     if n.op in SCAN_OPS and n.extra.get("Table") else n.op)
            items.append((label, m))
    total = sum(m for _, m in items)
    items.sort(key=lambda x: -x[1])
    out = ["", "--- cost breakdown ---"]
    for label, m in items:
        pct = (100.0 * m / total) if total else 0.0
        out.append(f"{label:32} {m:>14} {pct:5.1f}%")
    out.append(f"total={total}")
    return out


def extract(input_json: str, output_golden: str,
            dynfilters_json: Optional[str] = None, data_dir: Optional[str] = None) -> int:
    """Build the golden by COMBINING the two committed inputs:
      input_json     = pass-1 cost profile (JFP off; cost tree + static filters)
      dynfilters_json = pass-2 dynamic-filter bounds (JFP on; deterministic ranges)
    plus the parquet stats (data_dir) for pruning. The cost has TWO deterministic,
    separately-weightable components: MATERIALIZATION (Σ pipeline-breaker materialized,
    pass 1) and STORAGE_READ (Σ scan bytes_read, post-prune from pass1 static ∩ pass2
    dynamic bounds). duckdb_cost = their sum."""
    warnings: set[str] = set()
    warn = lambda m: warnings.add(m)  # noqa: E731
    root = build_tree(parse_json(input_json))
    if root is None:
        raise SystemExit(f"no operator tree in {input_json}")
    dynamic_filters = json.load(open(dynfilters_json)) if dynfilters_json else None
    # Fold the deterministic pruning into the tree FIRST, so the scan read term is
    # post-prune (pass1 static ∩ pass2 dynamic bounds + parquet).
    compute_scan_pruning(root, dynamic_filters, data_dir, warn)
    lines, materialization, storage_read = build_cost_tree(root, lambda op, ob: warn(
        f"unclassified operator '{op}' (output_bytes={ob}) contributes 0"))
    total = materialization + storage_read
    # Footer: combined cost + its two weightable component sub-totals (cost-report
    # reads `duckdb_cost=`). Both components are deterministic and summable.
    lines.append(f"duckdb_cost={total}")
    lines.append(f"materialization_total={materialization}")
    lines.append(f"storage_read_total={storage_read}")
    # NEW sections (#10): storage-pruning reduction + per-operator cost breakdown.
    lines += format_pruning_section(root)
    lines += format_breakdown_section(root)
    with open(output_golden, "w") as f:
        f.write("\n".join(lines) + "\n")
    for w in sorted(warnings):
        print(f"warning: {w}", file=sys.stderr)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # PASS 1: raw JFP-off profile -> lean cost profile (committed input #1).
    p_norm = sub.add_parser("normalize", help="raw JFP-off profiling JSON -> lean cost profile")
    p_norm.add_argument("raw_json")
    p_norm.add_argument("out_json")

    # PASS 2: raw JFP-on profile -> deterministic dynamic-filter bounds only
    # (committed input #2; the full JFP-on profile is discarded).
    p_dyn = sub.add_parser("dynfilters", help="raw JFP-on profiling JSON -> per-scan dynamic-filter bounds")
    p_dyn.add_argument("raw_json")
    p_dyn.add_argument("out_json")

    # COMBINE inputs #1 + #2 (+ parquet) -> golden.
    p_ext = sub.add_parser("extract", help="cost profile [+ dynfilters + parquet] -> golden")
    p_ext.add_argument("input_json")
    p_ext.add_argument("output_golden")
    p_ext.add_argument("--dynfilters", default=None, help="pass-2 dynamic-filter bounds JSON")
    p_ext.add_argument("--data-dir", default=None, help="parquet dir (for the pruning section)")

    args = ap.parse_args()
    if args.cmd == "normalize":
        doc = normalize(parse_json(args.raw_json))
        if doc is None:
            raise SystemExit(f"empty profile {args.raw_json}")
        with open(args.out_json, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        return 0
    if args.cmd == "dynfilters":
        bounds = extract_dynamic_filters(parse_json(args.raw_json))
        with open(args.out_json, "w") as f:
            json.dump(bounds, f, separators=(",", ":"))
        return 0
    if args.cmd == "extract":
        print(extract(args.input_json, args.output_golden, args.dynfilters, args.data_dir))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
