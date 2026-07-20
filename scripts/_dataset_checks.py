"""Shared dataset-validation checks for validate_tpch.py / validate_tpcds.py.

Designed to be TRACTABLE AT ANY SCALE FACTOR (sf200 = lineitem 1.2B rows):
  - ordering / clustering is checked from PARQUET ROW-GROUP METADATA (min/max/null_count),
    which is O(row groups) and reads no data — this is EXHAUSTIVE over the file;
  - exact within-row-group ordering is checked on a SAMPLE of row groups;
  - embedding stats stream per row group and never materialize a whole vector column
    (whole-column reads overflow pyarrow's int32 list offsets at ~32M*96 values).
Every check records whether it was EXHAUSTIVE or SAMPLED so the output can't overstate.
"""
import numpy as np
import pyarrow.parquet as pq

RESULTS = []  # (name, ok, scope, detail)   scope in {'exhaustive','sampled','meta'}


def check(name, ok, scope, detail=""):
    RESULTS.append((name, bool(ok), scope, detail))
    tag = {'exhaustive': 'EXHAUSTIVE', 'sampled': 'SAMPLED', 'meta': 'META'}.get(scope, scope)
    print(f"  [{'PASS' if ok else 'FAIL'}] ({tag}) {name}" + (f" — {detail}" if detail else ""))
    return ok


def summarize_and_exit():
    import sys
    nfail = sum(1 for _, ok, _, _ in RESULTS if not ok)
    ex = sum(1 for _, _, s, _ in RESULTS if s == 'exhaustive')
    sa = sum(1 for _, _, s, _ in RESULTS if s == 'sampled')
    print(f"\n== SUMMARY == {len(RESULTS)-nfail}/{len(RESULTS)} passed; {nfail} failed. "
          f"({ex} exhaustive, {sa} sampled)")
    sys.exit(1 if nfail else 0)


def sample_groups(ng, k=8):
    """Deterministic row-group sample: first, last, evenly spaced."""
    if ng <= k:
        return list(range(ng))
    idx = {0, ng - 1}
    step = ng / float(k)
    idx.update(int(i * step) for i in range(k))
    return sorted(i for i in idx if 0 <= i < ng)


def _rg_stats(pf, col):
    """Per-row-group (min, max, null_count, num_rows) for `col`, from METADATA only."""
    md = pf.metadata
    names = [md.schema.column(i).name for i in range(md.num_columns)]
    # leaf column index (flat columns for the sort keys we check)
    ci = names.index(col)
    out = []
    for g in range(md.num_row_groups):
        st = md.row_group(g).column(ci).statistics
        nr = md.row_group(g).num_rows
        if st is None or not st.has_min_max:
            out.append((None, None, None, nr))
        else:
            nc = st.null_count if st.has_null_count else None
            out.append((st.min, st.max, nc, nr))
    return out


def check_clustering(parquet, lead_col, label):
    """EXHAUSTIVE (metadata): per-row-group min/max on the lead sort column are
    non-decreasing and non-overlapping, and NULLS land LAST (no non-null group after a
    null-bearing one). This is the property the date-sort exists for (tight row-group
    min/max => row-group pruning)."""
    pf = pq.ParquetFile(parquet)
    st = _rg_stats(pf, lead_col)
    ng = len(st)
    prev_max = None
    seen_null = False
    ov = nn = 0
    for (mn, mx, nc, nr) in st:
        # NULLS LAST: once a group carries nulls, no later group may have non-null values
        has_nonnull = (mn is not None)
        if seen_null and has_nonnull:
            nn += 1
        if nc:
            seen_null = True
        if has_nonnull and prev_max is not None:
            if mn < prev_max:            # overlap / out of order across groups
                ov += 1
        if mx is not None:
            prev_max = mx
    check(f"{label}: row-group min/max non-decreasing & non-overlapping",
          ov == 0, 'exhaustive', f"{ng} row groups, {ov} overlaps")
    check(f"{label}: NULLS LAST (no non-null group after a null-bearing group)",
          nn == 0, 'exhaustive', f"{nn} violations")


def _key_tuple(cols_values, sort_cols, i):
    t = []
    for c in sort_cols:
        v = cols_values[c][i]
        # NULLS LAST: represent null as the maximum so it sorts after any value
        t.append((1, None) if v is None else (0, v))
    return tuple(t)


def check_within_group_order(parquet, sort_cols, label, k=6):
    """SAMPLED (exact): read the sort columns for a sample of row groups and assert the
    rows within each are ordered by `sort_cols` with NULLS LAST. Cheap columns only."""
    pf = pq.ParquetFile(parquet)
    ng = pf.num_row_groups
    groups = sample_groups(ng, k)
    bad = 0
    checked = 0
    for g in groups:
        t = pf.read_row_group(g, columns=sort_cols)
        cols = {c: t.column(c).to_pylist() for c in sort_cols}
        n = t.num_rows
        checked += n
        prev = None
        for i in range(n):
            key = _key_tuple(cols, sort_cols, i)
            if prev is not None and key < prev:
                bad += 1
                break
            prev = key
    check(f"{label}: within-row-group order {sort_cols}",
          bad == 0, 'sampled', f"{len(groups)}/{ng} groups, {checked} rows, {bad} out-of-order")


def check_row_count(parquet, expected, label, tol=0.0):
    """EXHAUSTIVE (metadata): total rows from the parquet footer (no scan)."""
    n = pq.ParquetFile(parquet).metadata.num_rows
    if tol > 0:
        ok = abs(n - expected) <= tol * expected
        det = f"{n:,} vs ~{expected:,.0f} (±{tol:.0%})"
    else:
        ok = n == expected
        det = f"{n:,} vs {expected:,}"
    check(f"{label}: row count", ok, 'meta', det)
    return n


def check_columns(parquet, expected_present, label, appended_last=None):
    """EXHAUSTIVE (schema): expected columns present; optionally that `appended_last`
    columns are the FINAL columns in order (the projection-pushdown index invariant)."""
    names = [f.name for f in pq.ParquetFile(parquet).schema_arrow]
    missing = [c for c in expected_present if c not in names]
    check(f"{label}: expected columns present", not missing, 'meta',
          f"missing {missing}" if missing else f"{len(names)} cols")
    if appended_last is not None:
        ok = names[-len(appended_last):] == list(appended_last) if appended_last else True
        check(f"{label}: embedding columns appended LAST", ok, 'meta',
              f"tail={names[-len(appended_last):]}" if appended_last else "n/a")


# ---- embedding stats: streamed per row group + sampled -----------------------

def load_vecs_sampled(parquet, col, dim, k=12):
    """Concatenate a SAMPLE of row groups' vectors into an (m, dim) array — bounded even
    at sf200 (never the whole column)."""
    pf = pq.ParquetFile(parquet)
    ng = pf.num_row_groups
    chunks = []
    for g in sample_groups(ng, k):
        a = pf.read_row_group(g, columns=[col]).column(col).chunk(0)
        chunks.append(np.asarray(a.values, dtype=np.float32).reshape(-1, dim))
    return np.vstack(chunks), ng


def stream_vector_stats(parquet, col, dim):
    """Streamed over ALL row groups: norm range, NaN/Inf, per-dim mean/variance. O(1) mem."""
    pf = pq.ParquetFile(parquet)
    S = np.zeros(dim, dtype=np.float64)
    S2 = np.zeros(dim, dtype=np.float64)
    cnt = 0
    nmin, nmax = np.inf, -np.inf
    bad = nan = 0
    for g in range(pf.num_row_groups):
        v = np.asarray(pf.read_row_group(g, columns=[col]).column(col).chunk(0).values,
                       dtype=np.float32).reshape(-1, dim)
        norms = np.linalg.norm(v, axis=1)
        nmin = min(nmin, float(norms.min())); nmax = max(nmax, float(norms.max()))
        bad += int((np.abs(norms - 1.0) > 1e-4).sum())
        nan += int((~np.isfinite(v)).sum())
        S += v.sum(0); S2 += (v.astype(np.float64) ** 2).sum(0); cnt += len(v)
    mean = S / cnt
    var = S2 / cnt - mean ** 2
    return dict(n=cnt, nmin=nmin, nmax=nmax, bad_norm=bad, nan=nan, mean=mean, var=var,
                ng=pf.num_row_groups)
