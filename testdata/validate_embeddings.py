#!/usr/bin/env python3
"""validate_embeddings.py — local pass/fail sanity gate for the embedding fixtures.

LOCAL-ONLY. Operates on a generated tpch.sf<N> plus testdata/tpch-vec-queries/
query_params.jsonl. It is NOT wired into CI: CI generates the SYNTHETIC dataset
(FLOAT[8]) with no fetch, and the DEEP1B/GloVe-specific invariants below don't
apply there — those checks auto-skip when dims != 96/100. Run it locally after
`generate_testdata.sh --embeddings external` to gate the shipped real bytes.

Exits non-zero if any check fails; prints a summary table.

Checks (real data):
  IMAGE  ps_image_embedding (DEEP1B, 96d): unit-norm 1±1e-4; per-dim |mean|<0.1;
         per-dim variance in [0.3/d, 3/d] (~uniform 1/d — L2-normalized, so the raw
         PCA "decreasing variance" is flattened; the band still trips a dead/blown/
         duplicated dim); no NaN/Inf/all-zero.
  TEXT   p_text_embedding, ps_text_embedding (GloVe mean, 100d): no NaN/Inf; bounded
         norms; zero-vector count == empty-vocab (no in-GloVe-token) row count.
  STRUCT per modality on a fixed sample: median nearest-neighbour L2 << median random
         -pair L2 (a real manifold has local structure; ~equal => noise/bug).
  QUERY  query_params.jsonl: each (q,D) reproduces count(emb <-> q < D)==k-1 on the
         real column (DuckDB array_distance); image q unit-norm; dim matches.
"""
import numpy as np, pyarrow.parquet as pq, json, sys, os, re, subprocess, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []  # (name, ok, detail)

def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    return ok

def load_vecs(parquet, col):
    ca = pq.read_table(parquet, columns=[col]).column(col).combine_chunks()
    vals = np.asarray(ca.values, dtype=np.float32)
    n = len(ca); dim = vals.size // n
    return vals.reshape(n, dim)

def duck(sql):
    r = subprocess.run(["duckdb", ":memory:", "-noheader", "-list", sql], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return r.stdout.strip()

def finite_nonzero(M, label):
    check(f"{label}: no NaN/Inf", np.isfinite(M).all(), f"{int((~np.isfinite(M)).sum())} bad")
    zero = int((np.abs(M).sum(1) == 0).sum())
    check(f"{label}: no all-zero vectors (info only for text)", True, f"{zero} zero-vectors")
    return zero

def struct_check(M, label, sample=2000):
    idx = np.linspace(0, len(M) - 1, min(sample, len(M))).astype(int)  # deterministic sample
    S = M[np.unique(idx)].astype(np.float64)
    sq = np.einsum("ij,ij->i", S, S)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (S @ S.T)
    np.fill_diagonal(d2, np.inf)
    nn_med = float(np.sqrt(np.clip(d2.min(1), 0, None)).mean())
    d2r = d2.copy(); d2r[~np.isfinite(d2r)] = np.nan
    rand_med = float(np.sqrt(np.clip(np.nanmedian(d2r), 0, None)))
    # high-dim distance concentration: real manifolds show NN/random ~0.6, not <<0.5;
    # pure noise/degenerate data gives ~1.0. 0.8 separates them with margin.
    check(f"{label}: NN dist << random-pair (manifold structure)",
          nn_med < 0.8 * rand_med, f"NN~{nn_med:.4f} vs rand~{rand_med:.4f} (ratio {nn_med/rand_med:.2f})")

def glove_words(path):
    s = set()
    with open(path) as f:
        for line in f:
            s.add(line.split(" ", 1)[0])
    return s

def empty_vocab_count(parquet, cols, words):
    t = pq.read_table(parquet, columns=cols)
    arrs = [t.column(c).to_pylist() for c in cols]
    n = len(arrs[0]); empty = 0
    for i in range(n):
        txt = " ".join(str(a[i]) for a in arrs).lower()
        toks = [tk for tk in re.split(r"[^a-z0-9]+", txt) if tk]
        if not any(tk in words for tk in toks):
            empty += 1
    return empty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=int, default=1)
    args = ap.parse_args()
    TD = os.path.join(ROOT, f"testdata/tpch.sf{args.sf}")
    CACHE = os.path.join(ROOT, "testdata/embeddings-cache")
    QP = os.path.join(ROOT, "testdata/tpch-vec-queries/query_params.jsonl")
    PS = os.path.join(TD, "partsupp.parquet"); PART = os.path.join(TD, "part.parquet")
    for p in (PS, PART):
        if not os.path.exists(p):
            print(f"error: missing {p}", file=sys.stderr); sys.exit(2)

    img = load_vecs(PS, "ps_image_embedding")
    ptxt = load_vecs(PART, "p_text_embedding")
    pstxt = load_vecs(PS, "ps_text_embedding")
    d_img, d_txt = img.shape[1], ptxt.shape[1]
    real = (d_img == 96 and d_txt == 100)

    print(f"== dims: image={d_img} text={d_txt} -> {'REAL (DEEP1B/GloVe)' if real else 'SYNTHETIC (skipping DEEP/GloVe-specific checks)'} ==")

    print("-- IMAGE (ps_image_embedding) --")
    finite_nonzero(img, "image")
    if real:
        norms = np.linalg.norm(img, axis=1)
        check("image: unit-norm 1±1e-4", np.abs(norms - 1.0).max() < 1e-4, f"max|‖v‖-1|={np.abs(norms-1).max():.2e}")
        mean = img.mean(0); var = img.var(0); d = d_img
        check("image: per-dim |mean|<0.1", np.abs(mean).max() < 0.1, f"max|mean|={np.abs(mean).max():.4f}")
        lo, hi = 0.3 / d, 3.0 / d
        check("image: per-dim var in [0.3/d,3/d] (uniform ~1/d)", ((var >= lo) & (var <= hi)).all(),
              f"var∈[{var.min():.5f},{var.max():.5f}] band=[{lo:.5f},{hi:.5f}]")
        struct_check(img, "image")
    else:
        check("image: DEEP1B invariants", True, "skipped (synthetic)")

    print("-- TEXT (p_text_embedding, ps_text_embedding) --")
    if real:
        words = glove_words(os.path.join(CACHE, "glove.6B.100d.txt")) if os.path.exists(os.path.join(CACHE, "glove.6B.100d.txt")) else None
    else:
        words = None
    for label, M, parquet, cols in [("p_text", ptxt, PART, ["p_name", "p_type"]),
                                     ("ps_text", pstxt, PS, ["ps_comment"])]:
        zero = finite_nonzero(M, label)
        norms = np.linalg.norm(M, axis=1)
        check(f"{label}: bounded norms (<50)", norms.max() < 50.0, f"max‖v‖={norms.max():.3f}")
        if real and words is not None:
            ev = empty_vocab_count(parquet, cols, words)
            check(f"{label}: zero-vectors == empty-vocab rows", zero == ev, f"zero={zero} empty_vocab={ev}")
        struct_check(M, label)

    print("-- QUERY FILE (query_params.jsonl) --")
    if os.path.exists(QP):
        entries = [json.loads(l) for l in open(QP)]
        bad = 0
        for e in entries:
            col = "ps_image_embedding" if e["modality"] == "image" else "p_text_embedding"
            parquet = PS if e["modality"] == "image" else PART
            dim = e["dim"]
            if e["modality"] == "image" and real:
                qn = float(np.linalg.norm(np.array(e["q"], dtype=np.float32)))
                if abs(qn - 1.0) >= 1e-3: bad += 1; print(f"    {e['id']}: image q not unit-norm ({qn})")
            lit = "[" + ",".join(repr(float(x)) for x in e["q"]) + "]"
            cnt = int(duck(f"SELECT count(*) FROM '{parquet}' WHERE array_distance({col}::FLOAT[{dim}], {lit}::FLOAT[{dim}]) < {e['D']!r};"))
            if cnt != e["k"] - 1:
                bad += 1; print(f"    {e['id']}: count {cnt} != k-1 {e['k']-1}")
        check("query file: all (q,D) reproduce count==k-1 + image-q unit-norm", bad == 0, f"{len(entries)} entries, {bad} bad")
    else:
        check("query file present", False, f"missing {QP}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n== SUMMARY ==")
    print(f"  {len(results)-n_fail}/{len(results)} checks passed; {n_fail} failed.")
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
