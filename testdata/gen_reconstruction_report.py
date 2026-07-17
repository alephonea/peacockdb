#!/usr/bin/env python3
"""Generate reconstruction_report.html — a human-skimmable view of what the
embedding fixtures encode, so a reviewer can eyeball fidelity WITHOUT the data.

LOCAL-ONLY generation (needs the GloVe cache + the --real-embeddings parquet). The
HTML OUTPUT is committed (testdata/tpch-vec-queries/reconstruction_report.html);
it contains only public TPC-H text + public-domain GloVe words + relational keys —
NO raw DEEP1B content (image vectors are not invertible). Self-contained (inline
CSS, no external assets) and DETERMINISTIC (fixed samples) so it doesn't churn.

  python3 testdata/gen_reconstruction_report.py [--sf 1]
"""
import numpy as np, pyarrow.parquet as pq, json, os, re, html, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "testdata/embeddings-cache")
OUT = os.path.join(ROOT, "testdata/tpch-vec-queries/reconstruction_report.html")
N_TEXT, N_IMG_PANEL, TOPW, TOPR = 20, 5, 5, 5

def load_vecs(parquet, col):
    ca = pq.read_table(parquet, columns=[col]).column(col).combine_chunks()
    v = np.asarray(ca.values, dtype=np.float32); n = len(ca)
    return v.reshape(n, v.size // n)

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

    W, G = load_glove()
    G = G / np.linalg.norm(G, axis=1, keepdims=True)   # L2-normalized -> cosine via dot product

    # deterministic samples: first N by primary key
    part = pq.read_table(PART, columns=["p_partkey", "p_name", "p_type", "p_text_embedding"]).sort_by("p_partkey").slice(0, N_TEXT).to_pydict()
    pst = pq.read_table(PS, columns=["ps_partkey", "ps_suppkey", "ps_comment", "ps_text_embedding"]).sort_by([("ps_partkey", "ascending"), ("ps_suppkey", "ascending")]).slice(0, N_TEXT).to_pydict()

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
    img_emb = load_vecs(PS, "ps_image_embedding")
    keys = pq.read_table(PS, columns=["ps_partkey", "ps_suppkey"]).to_pydict()
    q_img = [e for e in qp if e["modality"] == "image"][:N_IMG_PANEL]
    img_panels = []
    for e in q_img:
        d = np.linalg.norm(img_emb - np.asarray(e["q"], dtype=np.float32), axis=1)
        idx = np.argpartition(d, TOPR)[:TOPR]; idx = idx[np.argsort(d[idx])]
        img_panels.append((e["id"], [(keys["ps_partkey"][i], keys["ps_suppkey"][i], float(d[i])) for i in idx]))

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
