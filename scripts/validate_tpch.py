#!/usr/bin/env python3
"""validate_tpch.py — structural + embedding validation for a generated tpch.sf<N>.

Tractable at any scale factor: ordering/clustering come from parquet row-group METADATA
(exhaustive, no scan), exact ordering is checked on SAMPLED row groups, and embedding
stats stream per row group (a whole-column read overflows pyarrow int32 list offsets at
sf40+). Each line is tagged EXHAUSTIVE / SAMPLED / META so nothing overstates its scope.

Checks:
  ROW COUNTS   part 200k*SF, partsupp 800k*SF, customer 150k*SF, supplier 10k*SF,
               orders 1.5M*SF, nation 25, region 5; lineitem ~6M*SF (tolerance band).
  SCHEMA       expected columns present; embedding columns APPENDED LAST (the column-index
               invariant the projection-pushdown goldens depend on).
  SORT ORDER   orders by (o_orderdate NULLS LAST, o_orderkey); lineitem by
               (l_shipdate NULLS LAST, l_orderkey, l_linenumber). Row-group min/max
               non-decreasing/non-overlapping + NULLS LAST (exhaustive); exact within-group
               order on a sample.
  EMBEDDINGS   (external only; auto-skip on synthetic FLOAT[8]) image unit-norm 1±1e-4,
               per-dim |mean|<0.1, per-dim variance in [0.3/d,3/d]; text no NaN/Inf, bounded
               norms; NN-vs-random structural signal. Query-file (q,D) reproduce count==k-1
               only at sf1 (D is calibrated against sf1 counts).

Usage:  python3 scripts/validate_tpch.py --sf 1
"""
import os
import sys
import argparse
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_checks as dc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Row counts / column lists / embedding dims come from dataset_checks — the SINGLE
# SOURCE OF TRUTH shared with check_s3_datasets.py. Don't restate them here.
EMB_COLS = dc.TPCH_EMB_COLS


def emb_dim(parquet, col):
    a = pq.ParquetFile(parquet).read_row_group(0, columns=[col]).column(0).chunk(0)
    return len(a.values) // len(a) if len(a) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=int, default=1)
    sf = ap.parse_args().sf
    TD = os.path.join(ROOT, f"testdata/tpch.sf{sf}")
    P = lambda t: os.path.join(TD, f"{t}.parquet")
    for t in ("part", "partsupp", "orders", "lineitem"):
        if not os.path.exists(P(t)):
            print(f"error: missing {P(t)}", file=sys.stderr); sys.exit(2)

    print("-- ROW COUNTS --")
    for t in dc.TPCH_TABLES:
        exp, tol = dc.expected_rows("tpch", t, sf)
        if exp is not None:
            dc.check_row_count(P(t), exp, t, tol=tol)

    print("-- SCHEMA --")
    dims = {}
    for t in ("part", "partsupp"):
        names = [f.name for f in pq.ParquetFile(P(t)).schema_arrow]
        stock, trailing = dc.expected_columns("tpch", t)
        has_emb = all(c in names for c in trailing)
        dc.check_columns(P(t), stock, t, appended_last=trailing if has_emb else None)
        if has_emb:
            dims[t] = emb_dim(P(t), trailing[0])

    external = (dims.get("partsupp", 0) == dc.TPCH_EMB_DIMS["ps_image_embedding"]
                and dims.get("part", 0) == dc.TPCH_EMB_DIMS["p_text_embedding"])
    if not external:
        # record the skip as a check so "N/N passed" can't be misread as "embeddings validated"
        dc.check("embedding invariants", True, 'meta',
                 f"skipped: not external (dims part={dims.get('part')} image={dims.get('partsupp')}, "
                 "expected 100/96)")

    print("-- SORT ORDER --")
    dc.check_clustering(P("orders"), "o_orderdate", "orders")
    dc.check_within_group_order(P("orders"), ["o_orderdate", "o_orderkey"], "orders")
    dc.check_clustering(P("lineitem"), "l_shipdate", "lineitem")
    dc.check_within_group_order(P("lineitem"), ["l_shipdate", "l_orderkey", "l_linenumber"], "lineitem")

    if external:
        print("-- EMBEDDINGS (external) --")
        d = 96
        st = dc.stream_vector_stats(P("partsupp"), "ps_image_embedding", d)
        dc.check("image: unit-norm 1±1e-4", st["bad_norm"] == 0 and abs(st["nmax"] - 1) < 1e-4,
                 'exhaustive', f"norms [{st['nmin']:.6f},{st['nmax']:.6f}], {st['bad_norm']} bad")
        dc.check("image: no NaN/Inf", st["nan"] == 0, 'exhaustive', f"{st['nan']} bad")
        dc.check("image: per-dim |mean|<0.1", np.abs(st["mean"]).max() < 0.1,
                 'exhaustive', f"max|mean|={np.abs(st['mean']).max():.4f}")
        lo, hi = 0.3 / d, 3.0 / d
        dc.check("image: per-dim variance in [0.3/d,3/d]",
                 bool(((st["var"] >= lo) & (st["var"] <= hi)).all()),
                 'exhaustive', f"var[{st['var'].min():.5f},{st['var'].max():.5f}]")
        for c, t in [("p_text_embedding", "part"), ("ps_text_embedding", "partsupp")]:
            ts = dc.stream_vector_stats(P(t), c, 100)
            dc.check(f"{c}: no NaN/Inf, bounded norms", ts["nan"] == 0 and ts["nmax"] < 50.0,
                     'exhaustive', f"norm max {ts['nmax']:.3f}, {ts['nan']} nan")
        for parq, c, dim in [(P("partsupp"), "ps_image_embedding", 96),
                             (P("part"), "p_text_embedding", 100)]:
            V, ng = dc.load_vecs_sampled(parq, c, dim, k=6)
            m = min(len(V), 2000)
            S = V[np.linspace(0, len(V) - 1, m).astype(int)].astype(np.float64)
            sq = np.einsum("ij,ij->i", S, S)
            d2 = sq[:, None] + sq[None, :] - 2 * (S @ S.T)
            np.fill_diagonal(d2, np.inf)
            nn = float(np.sqrt(np.clip(d2.min(1), 0, None)).mean())
            d2[~np.isfinite(d2)] = np.nan
            rnd = float(np.sqrt(np.clip(np.nanmedian(d2), 0, None)))
            dc.check(f"{c}: NN << random-pair (manifold)", nn < 0.8 * rnd, 'sampled',
                     f"NN~{nn:.3f} vs rand~{rnd:.3f} (ratio {nn/rnd:.2f})")

        QP = os.path.join(ROOT, "testdata/tpch-vec-queries/query_params.jsonl")
        if sf == 1 and os.path.exists(QP):
            import json, subprocess
            entries = [json.loads(x) for x in open(QP)]
            bad = 0
            for e in entries:
                col = "ps_image_embedding" if e["modality"] == "image" else "p_text_embedding"
                pq_ = P("partsupp") if e["modality"] == "image" else P("part")
                lit = "[" + ",".join(repr(float(x)) for x in e["q"]) + "]"
                r = subprocess.run(["duckdb", ":memory:", "-noheader", "-list",
                    f"SELECT count(*) FROM '{pq_}' WHERE array_distance({col}::FLOAT[{e['dim']}], {lit}::FLOAT[{e['dim']}]) < {e['D']!r};"],
                    capture_output=True, text=True)
                if int(r.stdout.strip() or -1) != e["k"] - 1:
                    bad += 1
            dc.check("query file: (q,D) reproduce count==k-1", bad == 0, 'exhaustive',
                     f"{len(entries)} entries, {bad} bad")
        else:
            dc.check("query file: (q,D) selectivity", True, 'meta',
                     f"skipped (D is sf1-calibrated; sf={sf})")

    dc.summarize_and_exit()


if __name__ == "__main__":
    main()
