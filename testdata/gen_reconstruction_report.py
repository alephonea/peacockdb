#!/usr/bin/env python3
"""Generate reconstruction_report.html — a human-skimmable view of what the
embedding fixtures encode, so a reviewer can eyeball fidelity WITHOUT the data.

LOCAL-ONLY generation (needs the GloVe cache + the --embeddings external parquet). The
HTML OUTPUT is committed, one per dataset, at
    testdata/reports/tpch.sf<N>/reconstruction_report.html
(mirroring the testdata/goldens/tpch.sf<N>/ shape — the report describes ONE dataset,
so the path must carry the scale factor or a later --sf silently overwrites an earlier
report). It contains only public TPC-H text + public-domain GloVe words + relational
keys — NO raw DEEP1B content (image vectors are not invertible). Self-contained (inline
CSS, no external assets) and DETERMINISTIC (fixed samples) so it doesn't churn.

These reports are for humans reading the repo; they are deliberately NOT in any
build-test.sh sync KIND, so they never ship to the test hosts.

  python3 testdata/gen_reconstruction_report.py [--sf 1]
"""
import numpy as np, pyarrow.parquet as pq, json, os, re, html, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "testdata/embeddings-cache")
N_TEXT, N_IMG_PANEL, TOPW, TOPR = 20, 5, 5, 5

def load_glove():
    words, vecs = [], []
    with open(os.path.join(CACHE, "glove.6B.100d.txt")) as f:
        for line in f:
            p = line.split(" ")
            words.append(p[0]); vecs.append(np.asarray(p[1:], dtype=np.float32))
    return np.array(words), np.vstack(vecs)

def nearest_words(v, Gn, W, topn=TOPW):
    # COSINE similarity — the standard for word-vector reconstruction. (L2, the fixture's
    # <-> predicate metric, is dominated by GloVe norm outliers and decodes to junk; the
    # mean-of-word-vectors encodes DIRECTION, which cosine recovers.) Gn is L2-normalized.
    nv = np.linalg.norm(v)
    if nv == 0:
        return ["<zero-vector>"]
    sims = Gn @ (v / nv)
    idx = np.argpartition(-sims, topn)[:topn]
    return [W[i] for i in idx[np.argsort(-sims[idx])]]

def toks(s):
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]

def esc(s):
    return html.escape(str(s))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sf", type=int, default=1)
    sf = ap.parse_args().sf
    TD = os.path.join(ROOT, f"testdata/tpch.sf{sf}")
    PART, PS = os.path.join(TD, "part.parquet"), os.path.join(TD, "partsupp.parquet")
    # per-dataset output, so --sf 40 can never overwrite the sf1 report
    out_dir = os.path.join(ROOT, f"testdata/reports/tpch.sf{sf}")
    os.makedirs(out_dir, exist_ok=True)
    OUT = os.path.join(out_dir, "reconstruction_report.html")

    W, G = load_glove()
    G = G / np.linalg.norm(G, axis=1, keepdims=True)   # L2-normalized -> cosine via dot product

    def sample_by_key(parquet, key_cols, other_cols, n):
        """Deterministic 'n smallest by key' WITHOUT reading the whole table: the key
        columns alone are cheap, so find the n smallest there, then read only the row
        group(s) holding them. Reading the full table (incl. the 100-float embedding)
        would materialize >2^31 list values at sf40+ and overflow pyarrow's int32 offsets."""
        pf = pq.ParquetFile(parquet)
        keys = pq.read_table(parquet, columns=key_cols)          # narrow -> safe
        order = np.lexsort(tuple(np.asarray(keys.column(c)) for c in reversed(key_cols)))
        want = sorted(int(i) for i in order[:n])
        # map global row indices -> row groups, read only those
        bounds, acc = [], 0
        for g in range(pf.num_row_groups):
            nr = pf.metadata.row_group(g).num_rows
            bounds.append((acc, acc + nr, g)); acc += nr
        need = sorted({g for lo, hi, g in bounds for i in want if lo <= i < hi})
        cols = key_cols + other_cols
        tabs, offs = [], []
        for g in need:
            lo = next(l for l, h, gg in bounds if gg == g)
            tabs.append(pf.read_row_group(g, columns=cols)); offs.append(lo)
        out = {c: [] for c in cols}
        for i in want:
            for t, lo in zip(tabs, offs):
                if lo <= i < lo + t.num_rows:
                    for c in cols:
                        out[c].append(t.column(c)[i - lo].as_py())
                    break
        return out

    # deterministic samples: first N by primary key
    part = sample_by_key(PART, ["p_partkey"], ["p_name", "p_type", "p_text_embedding"], N_TEXT)
    pst = sample_by_key(PS, ["ps_partkey", "ps_suppkey"], ["ps_comment", "ps_text_embedding"], N_TEXT)

    def text_rows(orig_list, emb_list):
        out = []
        for orig, emb in zip(orig_list, emb_list):
            recon = nearest_words(np.asarray(emb, dtype=np.float32), G, W)
            ot = set(toks(orig)); hits = [w for w in recon if w in ot]
            out.append((orig, recon, len(hits), len(ot)))
        return out

    part_rows = text_rows([f"{n} {t}" for n, t in zip(part["p_name"], part["p_type"])], part["p_text_embedding"])
    ps_rows = text_rows([c for c in pst["ps_comment"]], pst["ps_text_embedding"])

    # query_params TEXT: phrase | matched | nearest words to q
    qp = [json.loads(l) for l in open(os.path.join(ROOT, "testdata/tpch-vec-queries/query_params.jsonl"))]
    q_text = [e for e in qp if e["modality"] == "text"][:10]
    q_text_rows = [(e["phrase"], e["matched_tokens"], nearest_words(np.asarray(e["q"], dtype=np.float32), G, W)) for e in q_text]

    # IMAGE panel: top-5 nearest partsupp rows per image query (relational only)
    # STREAM per row group, keeping a running top-K per query. Reading the whole
    # ps_image_embedding column would materialize 32M x 96 (sf40) = 3.07B values and
    # overflow pyarrow's int32 list offsets — the same wall the validator hit. This is
    # exact (every row is scored) at bounded memory.
    q_img = [e for e in qp if e["modality"] == "image"][:N_IMG_PANEL]
    qvecs = [np.asarray(e["q"], dtype=np.float32) for e in q_img]
    best = [[] for _ in q_img]          # list of (dist, partkey, suppkey)
    pf = pq.ParquetFile(PS)
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["ps_image_embedding", "ps_partkey", "ps_suppkey"])
        a = t.column("ps_image_embedding").chunk(0)
        V = np.asarray(a.values, dtype=np.float32).reshape(-1, 96)
        pk = t.column("ps_partkey").to_numpy(); sk = t.column("ps_suppkey").to_numpy()
        for qi, qv in enumerate(qvecs):
            d = np.linalg.norm(V - qv, axis=1)
            k = min(TOPR, len(d))
            idx = np.argpartition(d, k - 1)[:k]
            best[qi].extend((float(d[i]), int(pk[i]), int(sk[i])) for i in idx)
            best[qi] = sorted(best[qi])[:TOPR]
    img_panels = [(e["id"], [(pk, sk, dist) for dist, pk, sk in best[qi]])
                  for qi, e in enumerate(q_img)]

    # --- render ---
    def trow(cells): return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    def text_table(rows):
        h = "<tr><th>original text</th><th>nearest GloVe words to the row vector (lossy decode)</th><th>hits</th></tr>"
        body = ""
        for orig, recon, nh, nt in rows:
            recon_s = " ".join(f"<span class='{'hit' if w in set(toks(orig)) else 'miss'}'>{esc(w)}</span>" for w in recon)
            body += trow([esc(orig), recon_s, f"{nh}/{nt}"])
        return f"<table>{h}{body}</table>"

    qtt = "<tr><th>query phrase</th><th>matched tokens</th><th>nearest GloVe words to q</th></tr>"
    for ph, mt, nw in q_text_rows:
        qtt += trow([esc(ph), esc(", ".join(mt)), " ".join(esc(w) for w in nw)])
    imgh = ""
    for qid, rows in img_panels:
        rws = "".join(trow([esc(pk), esc(sk), f"{dist:.4f}"]) for pk, sk, dist in rows)
        imgh += f"<h4>{esc(qid)}</h4><table><tr><th>ps_partkey</th><th>ps_suppkey</th><th>image L2 dist</th></tr>{rws}</table>"

    doc = f"""<!doctype html><meta charset=utf-8><title>TPC-H+V embedding reconstruction</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}}
h1{{font-size:1.4rem}} h2{{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:.3rem}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f4f4f4}} .hit{{color:#0a7d24;font-weight:600}} .miss{{color:#888}}
.note{{background:#f8f8f2;border-left:3px solid #b8860b;padding:.6rem 1rem;font-size:13px}}
</style>
<h1>TPC-H+V embedding reconstruction report</h1>
<div class=note>
<b>What this is:</b> a fidelity spot-check of the embedding fixtures. <b>Text</b>
(p_text_embedding, ps_text_embedding) is a <i>lossy</i> mean of GloVe word-vectors —
decoding = nearest GloVe words to the row vector; green = the word is in the original
text. <b>Image</b> (ps_image_embedding, DEEP1B) content is <b>NOT reconstructable</b>
(deep CNN descriptors + PCA are not invertible), so only the relational nearest-neighbour
structure is shown. Deterministic fixed samples; no raw DEEP1B content is emitted.
</div>

<h2>Text reconstruction — part (p_name &amp; p_type), first {N_TEXT} by p_partkey</h2>
{text_table(part_rows)}
<h2>Text reconstruction — partsupp (ps_comment), first {N_TEXT} by (ps_partkey, ps_suppkey)</h2>
{text_table(ps_rows)}
<h2>query_params.jsonl — text query vectors</h2>
<table>{qtt}</table>
<h2>Image queries — nearest partsupp rows (relational only; content not reconstructable)</h2>
{imgh}
"""
    with open(OUT, "w") as f:
        f.write(doc)
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
    # stderr summary of decode hit-rate for the run log
    import sys
    ph = sum(nh for _, _, nh, _ in part_rows); pt = sum(nt for _, _, _, nt in part_rows)
    sh = sum(nh for _, _, nh, _ in ps_rows); st = sum(nt for _, _, _, nt in ps_rows)
    print(f"part decode hits {ph}/{pt} original tokens; partsupp {sh}/{st}", file=sys.stderr)

if __name__ == "__main__":
    main()
