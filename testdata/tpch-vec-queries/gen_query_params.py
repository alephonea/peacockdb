#!/usr/bin/env python3
"""Generate query_params.jsonl — (q, D) pairs for the TPC-H+V vector queries.

LOCAL-ONLY (like fetch_embeddings.sh): reads the gitignored embeddings cache and
the locally generated --real-embeddings parquet. The OUTPUT query_params.jsonl IS
committed (a de-minimis sample of DEEP1B + GloVe-derived vectors — see NOTICE for
attribution); the cache and parquet are not.

Each line: {id, modality, dim, metric, k, selectivity, D, d_source, q, <provenance>}.
  metric = l2 (DuckDB array_distance = Euclidean = pgvector <->).
  D = MIDPOINT of the (k-1)-th and k-th nearest LOADED-vector distances, so
      `<embedding_col> <-> q < D` selects exactly k-1 rows. Midpoint (not the exact
      k-th distance) so the strict `< D` boundary is immune to FLOAT32<->DOUBLE
      rounding at an exact distance value. k is TIERED {10,100,1000} for selectivity
      variety; q is rounded to 6 dp and D is computed on that SAME rounded q, so
      stored (q, D) reproduce count == k-1 exactly (verified by validate_embeddings.py).
  d_source = duckdb_recalib (DEEP1B groundtruth distances are unusable here: GT is
      IDs-only over the full 1B base, but we load only the first N; see NOTICE).

Usage:  python3 testdata/tpch-vec-queries/gen_query_params.py > testdata/tpch-vec-queries/query_params.jsonl
"""
import numpy as np, subprocess, json, re, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "testdata/embeddings-cache")
TD = os.path.join(ROOT, "testdata/tpch.sf1")
KS = [10, 100, 1000]          # tiers
PER_TIER = 17                 # -> 51 image + 51 text
ROUND = 6

def err(*a): print(*a, file=sys.stderr)

def duck(sql):
    r = subprocess.run(["duckdb", ":memory:", "-noheader", "-list", sql],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return r.stdout.strip()

def lit(v):  # SQL array literal for a rounded vector
    return "[" + ",".join(repr(float(x)) for x in v) + "]"

def d_and_count(parquet, col, dim, q, k):
    """D = MIDPOINT of the (k-1)-th and k-th nearest L2 distances over `parquet`.`col`;
    and #rows strictly < D. The midpoint (a DOUBLE strictly inside the gap between two
    distinct FLOAT32 distances) makes `<embedding_col> <-> q < D` select exactly k-1 rows
    robustly — using the exact k-th FLOAT32 distance round-trips through decimal as a
    DOUBLE slightly above the FLOAT32 value, letting the k-th row slip under `< D`."""
    a = lit(q)
    out = duck(f"""
      WITH dd AS MATERIALIZED (SELECT array_distance({col}::FLOAT[{dim}], {a}::FLOAT[{dim}]) d FROM '{parquet}'),
           edge AS (SELECT d FROM dd ORDER BY d LIMIT 2 OFFSET {k-2}),
           thr AS (SELECT avg(d)::DOUBLE AS D FROM edge)
      SELECT (SELECT D FROM thr)::VARCHAR AS D, (SELECT count(*) FROM dd, thr WHERE dd.d < thr.D) AS cnt;
    """)
    D_str, cnt_str = out.split("|")
    return float(D_str), int(cnt_str)

def count_for(parquet):
    return int(duck(f"SELECT count(*) FROM '{parquet}';"))

# --- inputs present? ---
DEEP_Q = os.path.join(CACHE, "deep_query.public.10K.fbin")
for p in [DEEP_Q, os.path.join(TD, "partsupp.parquet"), os.path.join(TD, "part.parquet")]:
    if not os.path.exists(p):
        err(f"error: missing {p} — run fetch_embeddings.sh then generate_testdata.sh --real-embeddings first")
        sys.exit(1)

PS = os.path.join(TD, "partsupp.parquet"); N_PS = count_for(PS)
PART = os.path.join(TD, "part.parquet");   N_PART = count_for(PART)

# tier assignment: PER_TIER of each k, in order
tiers = [k for k in KS for _ in range(PER_TIER)]   # 51 entries

# --- IMAGE: DEEP query vectors, D over ps_image_embedding ---
with open(DEEP_Q, "rb") as f:
    nq, nd = np.frombuffer(f.read(8), dtype="<i4")
    qv = np.frombuffer(f.read(nq * nd * 4), dtype="<f4").reshape(nq, nd)
assert nd == 96, nd

entries = []
for i, k in enumerate(tiers):
    q = [round(float(x), ROUND) for x in qv[i]]        # round first; D on rounded q
    D, cnt = d_and_count(PS, "ps_image_embedding", 96, q, k)
    assert cnt == k - 1, f"img_{i:03d}: count {cnt} != k-1 {k-1}"
    entries.append({"id": f"img_{i:03d}", "modality": "image", "dim": 96, "metric": "l2",
                    "k": k, "selectivity": round(k / N_PS, 10), "D": float(D),
                    "d_source": "duckdb_recalib", "deep_query_idx": i, "q": q})
    err(f"img_{i:03d} k={k} D={D:.6f} rows<D={cnt} ok")

# --- TEXT: mean-GloVe(phrase) over TPC-H part vocab, D over p_text_embedding ---
COLORS = ["almond","antique","aquamarine","azure","beige","bisque","blanched","blue","blush",
          "brown","burlywood","chartreuse","chiffon","chocolate","coral","cornflower","cornsilk",
          "cream","cyan","dark","deep","dodger","drab","firebrick","floral","forest","frosted",
          "ghost","goldenrod","green","honeydew","hot","indian","ivory","khaki","lace","lavender",
          "lawn","lemon","light","lime","linen","magenta","maroon","medium","metallic","midnight",
          "mint","misty","moccasin","navy","olive","orange","orchid","pale","papaya","peach","pink",
          "plum","powder","puff","purple","red","rose","rosy","royal","saddle","salmon","sandy",
          "seashell","sienna","sky","slate","smoke","snow","spring","steel","tan","thistle","tomato",
          "turquoise","violet","wheat","white","yellow"]
FINISHES = ["anodized","burnished","plated","polished","brushed"]
MATERIALS = ["tin","nickel","brass","steel","copper"]
# deterministic phrases: color + finish + material (all common, in-GloVe-vocab)
phrases = [f"{COLORS[i % len(COLORS)]} {FINISHES[i % len(FINISHES)]} {MATERIALS[(i // 5) % len(MATERIALS)]}"
           for i in range(len(tiers))]

# load GloVe once (word -> 100d), OOV-excluded mean exactly like the column
glove = {}
with open(os.path.join(CACHE, "glove.6B.100d.txt")) as f:
    for line in f:
        p = line.split(" ")
        glove[p[0]] = np.asarray(p[1:], dtype=np.float32)

def mean_glove(phrase):
    toks = [t for t in re.split(r"[^a-z0-9]+", phrase.lower()) if t]
    matched = [t for t in toks if t in glove]
    v = np.mean([glove[t] for t in matched], axis=0) if matched else np.zeros(100, dtype=np.float32)
    return v, matched

for i, k in enumerate(tiers):
    v, matched = mean_glove(phrases[i])
    q = [round(float(x), ROUND) for x in v]
    D, cnt = d_and_count(PART, "p_text_embedding", 100, q, k)
    assert cnt == k - 1, f"txt_{i:03d}: count {cnt} != k-1 {k-1}"
    entries.append({"id": f"txt_{i:03d}", "modality": "text", "dim": 100, "metric": "l2",
                    "k": k, "selectivity": round(k / N_PART, 10), "D": float(D),
                    "d_source": "duckdb_recalib", "phrase": phrases[i], "matched_tokens": matched, "q": q})
    err(f"txt_{i:03d} k={k} '{phrases[i]}' matched={len(matched)} D={D:.6f} rows<D={cnt} ok")

for e in entries:
    print(json.dumps(e))
err(f"\nOK: {len(entries)} entries ({sum(e['modality']=='image' for e in entries)} image, "
    f"{sum(e['modality']=='text' for e in entries)} text); all per-entry count==k-1 asserts passed.")
