# Minimal benchmark — H200 (cuDF) vs DuckDB, TPC-H sf40 + TPC-H+V

Per-query comparison over the sf40 dataset, for the ten queries with committed
DuckDB goldens (four TPC-H, six TPC-H+V probes). Both sides run the **same
result-defining SQL/operators** — the DuckDB side sources its SQL from
`testdata/tpch_query_sql.sh`, the same file the goldens are generated from, and the
cuDF side is the hand-written libcudf plan the GPU tests verify against those
goldens. Every DuckDB run was re-verified against the golden before timing; every
cuDF number is from a plan that passed the same golden.

## What each column measures — read this first

The two sides do **not** measure the same boundary, and the table must not be read
as if they do:

- **cuDF execute** — GPU operator time only, with all input columns **already
  resident and unpacked in VRAM**. The parquet→VRAM load is *excluded*. Ends on
  `cudaDeviceSynchronize` (drains the compute and raft streams), so it is execution,
  not kernel launch. Second-smallest of six runs, discarding first-run RMM-pool
  growth and kernel first-touch.
- **DuckDB 2nd-min** — wall time of the query over DuckDB's **own native storage,
  warm in page cache**. This *includes* DuckDB reading and decoding its columns from
  RAM — there is no separate pre-load tier to exclude. Second-smallest of six,
  measured on a 503 GB-RAM VM where the whole 32.7 GB database is 100 % resident
  (every query read **0.00 GB from the block device** on its timed run — a direct
  measurement, not a claim).

So **cuDF-execute vs DuckDB-2nd-min is GPU compute-only against DuckDB warm
end-to-end** — it flatters the GPU by excluding the load. The cuDF **load** column
is shown so a load+execute end-to-end can be read alongside for the honest picture.

## Environments

| | cuDF | DuckDB |
|---|---|---|
| Host | shad-gpu, NVIDIA H200 (140 GiB VRAM) | Nebius VM, 128 cores, 503 GB RAM |
| Version | libcudf 25.02 | DuckDB v1.5.4 |
| Storage | parquet → VRAM columns (load excluded from the number) | native `.duckdb`, 32.73 GiB, page-cache warm |
| Protocol | execute-only, device-synced, 2nd-min of 6 | warm query, 2nd-min of 6 |

## TPC-H

| Query | cuDF execute (ms) | cuDF load (ms) | DuckDB warm (ms) |
|---|--:|--:|--:|
| q6 — scan/filter/project/reduce | 43.8 | 472.7 | 137 |
| q1 — group-by + aggregates | 309.5 | 537.7 | 915 |
| q3 — 3-way join, group-by, top-N | 55.8 | 946.6 | 577 |
| q8 — 7 tables, bushy join order | 52.8 | 981.3 | 1126 |

## TPC-H+V (vector range predicate via cuVS brute-force)

cuDF splits the vector work three ways: **load** (parquet→VRAM), **index-build**
(cuVS norm precompute, built once and reused across a query's probes), and
**execute** (search → filter → the relational plan). DuckDB has no index — it
computes all distances **inline on every query** — so the cuDF quantity comparable
to DuckDB-warm is **execute + index-build/n\_probes**, not execute alone.

| Query / probe | cuDF execute (ms) | cuDF index-build (ms) | cuDF load (ms) | DuckDB warm (ms) |
|---|--:|--:|--:|--:|
| q11v / img_000 | 13.2 | 27.1 † (shared /3) | 2354.9 | 1433 |
| q11v / img_017 | 13.6 | — (shared) | — | 1501 |
| q11v / img_034 | 13.7 | — (shared) | — | 1478 |
| q12v / txt_000 | 41.4 | 1.5 | 1511.0 | 559 |
| q10v / txt_017 | 33.1 | 1.7 | 1631.9 | 960 |
| q9v / txt_034 | 62.1 | 1.7 | 1756.5 | 1631 |

† q11v's 27.1 ms index-build includes one-time cuVS/cuBLAS **first-touch** (it is the
first index built in the process). The steady-state marginal build — seen on the
text queries that run after it — is ~1.5 ms over 8M×100, and would be a few ms warm
over 32M×96. Do not read the 27 ms as pure norm-compute; amortized per-probe it is
~9 ms, but init-inflated.

## Reading the numbers

- **cuDF execute is faster than DuckDB-warm on every query**, by roughly 3–100×
  depending on the query — but that gap is partly the boundary difference: cuDF's
  inputs are pre-resident, DuckDB's warm number includes reading its columns.
- **cuDF load dominates cuDF's own end-to-end** (0.5–2.4 s vs tens of ms execute).
  A load+execute cuDF total is in the same order as, and often exceeds, DuckDB-warm
  — e.g. q6: 43.8 ms execute but 472.7 ms load → ~516 ms end-to-end vs DuckDB's
  137 ms warm. Whether the GPU wins end-to-end depends entirely on whether the load
  is amortized across many queries or paid per query.
- The vector queries are where cuDF's compute advantage is largest (13–62 ms execute
  vs 559–1631 ms DuckDB), because cuVS brute-force on the H200 parallelizes the
  distance computation that DuckDB does inline on CPU.

## Multi-GPU scaling — 1× A100 vs 2× A100 (same hardware, same pooled code)

A **same-hardware, same-code** comparison: the identical multi-GPU plan run over one
A100 (G=1: one row-group partition, one worker) vs two A100s (G=2), on verda (2×
A100-SXM4-80GB, NVLink, libcudf 26.02). Both legs use the SAME per-device RMM **pool**
allocator, so the G=1-vs-G=2 ratio isolates *parallelism* from allocator choice. Both
verified against the same committed DuckDB goldens; execute-only, all-device-synced,
2nd-min of 6. All four numbers below come from **one process per leg** — the same
`peacock_multi_gpu_tpch_tests` binary running q6→q1→q3→q8 back-to-back over a single
process-wide shared `WorkerPool` (see below). (The H200 column above is a **different
GPU** and is not a speedup baseline.)

| Query | 1× A100 execute (ms) | 2× A100 execute (ms) | speedup |
|---|--:|--:|--:|
| q6 — filter → reduce | 37.2 | 19.4 | **1.92×** |
| q1 — group-by + 8 aggregates | 632.6 | 295.5 | **2.14×** |
| q3 — 3-way join, group-by, top-N | 56.7 | 49.0 | **1.16×** |
| q8 — 7-table bushy join, group-by | 64.7 | 40.1 | **1.61×** |

Merge plans are matched to the partial's shape: **q1** gathers real partial *tables*
across the link (`pack → peer-copy → unpack → concat → merge-groupby`); **q6**'s
partial is a single decimal per GPU, so it is summed on the **host** by exact
`__int128` addition (all partials share scale −4 → bit-identical to the golden).

### Why q3 and q8 scale below the ~2× ceiling — and by different amounts

q6 and q1 are near-ideal because each GPU's work is (almost) purely a function of its own
lineitem partition. q3 and q8 add a **join tree** and a **group-by**, and the scaling is
set by two costs that do **not** shrink with G:

- **Broadcast redundancy.** Both plans replicate the small dimension tables on every GPU
  and build the dimension side *locally* — so `customer⋈orders` (q3) and the whole
  `region→nation→customer→orders`, `part`, `supplier→nation` subtree (q8) are rebuilt on
  each GPU from full-table reads. That work is constant in G (every GPU does all of it),
  so it is pure overhead against the parallel fact-side scan. It's the right trade at G=2
  (broadcasting a tiny dim beats shuffling a large fact), but it caps the ratio.
- **The group-by merge.** q3's group-by key is `l_orderkey` — **high cardinality**
  (millions of distinct keys), so partial group-bys cannot simply be gathered and merged;
  the plan pays a real **cross-GPU hash-shuffle** (`hash_partition` murmur3 →
  `pack`/`cudaMemcpyPeerAsync`/`unpack` → concat) to co-locate each key before the final
  group-by. That shuffle moves data across the link on the critical path and is why q3 is
  only **1.16×**. q8's key is `o_year` — **two groups** — so it needs **no shuffle**: each
  GPU emits a partial `sum(brazil_volume), sum(volume)` and GPU0 gathers G tiny partials
  and merge-sums them. With no shuffle, q8's parallel fact-side collapse (the
  `part='ECONOMY ANODIZED STEEL'`⋈lineitem step cuts 240M→a few hundred K *before* any
  other join) dominates, giving **1.61×** despite the heavier join tree.

So the ordering q1 (2.14×) > q8 (1.61×) > q3 (1.16×) is exactly the cardinality/shuffle
story: no join & re-aggregatable (q1) → broadcast join, no shuffle (q8) → broadcast join
**plus** a high-cardinality shuffle (q3). All four fixed costs (broadcast build, shuffle,
merge, dispatch) are constant in G, so every ratio keeps climbing toward 4/8/16 GPUs; the
shuffle in q3 is the one that also grows its *moved bytes* with data size, so q3 is the
query that most wants a smarter partition (hash-partition the fact on load) at higher G.

**The RMM pool is what makes this scale — and it changed the story.** Without it, every
transient column went through the default resource's synchronous, driver-serialized
`cudaMalloc`/`cudaFree`, which the two workers could not overlap. That fixed per-op cost
dominated the cheap query (q6 was **0.70×** — *slower* on 2 GPUs) and inflated the
single-GPU baseline of the expensive one (q1 measured a misleading **5.09×**, an
artifact of an allocation-bound G=1, not real parallelism). Installing a per-device
pool (reserved once off free VRAM; allocations become pointer bumps, no per-op
`cudaMalloc`) collapses that overhead on *both* legs: q6's G=1 dropped 66→37 ms and
q1's 2280→628 ms. With the allocator no longer the bottleneck, the honest picture
emerges — **q6 1.87× and q1 2.12×**, both near the ~2× ceiling for 2 GPUs, q1 slightly
above because its heavier per-partition compute hides the residual merge + dispatch +
boundary-sync cost that q6 (only ~20 ms) still partly shows. Those fixed costs are
constant in G, so the ratio keeps improving toward 4/8/16 GPUs.

## Multi-GPU TPC-H+V — sharded cuVS search, 1× A100 vs 2× A100 (same pooled code)

The four vector queries with the cuVS brute-force SEARCH **sharded across the GPUs**: the
embedding table (partsupp.ps_image_embedding for q11v, part.p_text_embedding for the three
text queries) is split on whole parquet row-group boundaries, each GPU builds a cuVS
brute_force index over its shard and searches the probe for the per-shard top-K, and the
G·K candidates are merged to the EXACT global result (a globally-top-K point is top-K in its
own shard, so the union contains the true top-K; `nth_element` gives the global K-th distance
for the saturation guard and a linear scan gives the rows under D — no full sort). The
relational tail that each query wraps around the hit list runs on GPU0, unchanged from the
single-GPU plan. Both legs verified against the same committed DuckDB goldens (byte-for-byte,
count-corroborated); execute-only, all-device-synced, 2nd-min of 6; both pooled. cuVS was
confirmed to allocate from the per-device RMM pool (free VRAM unchanged across a search).

| Query | 1× A100 execute (ms) | 2× A100 execute (ms) | speedup | index-build (ms, G=1→G=2) |
|---|--:|--:|--:|--:|
| q11v / img_000 | 20.6 | 16.0 | **1.29×** | 31.7 → 28.5 |
| q11v / img_017 | 21.7 | 16.6 | **1.31×** | (shared) |
| q11v / img_034 | 21.9 | 16.6 | **1.32×** | (shared) |
| q12v / txt_000 | 52.5 | 53.4 | **0.98×** | 2.3 → 1.2 |
| q10v / txt_017 | 35.6 | 36.5 | **0.97×** | 2.3 → 1.2 |
| q9v / txt_034 | 78.6 | 80.0 | **0.98×** | 2.3 → 1.2 |

**Which part dominates decides the factor — and it splits the queries cleanly in two.**

- **q11v scales (~1.3×)** because its execute IS almost entirely the search: the GERMANY
  dimension join (partsupp⋈supplier⋈nation) is probe-independent and built ONCE in setup, so
  the timed per-probe execute is search + host-merge + a small group-by. Sharding halves the
  distance scan (the parallel part) and the ~16 ms G=2 result reflects it. It is **not** 2×
  because the parallel part is only a fraction: the per-shard top-K extraction at the (very
  conservative) K=131072 is requested in FULL from every shard — mandatory for exactness — so
  that cost does not shrink with G, and the host merge (O(G·K)) and cross-GPU dispatch are
  constant-or-growing. Distance-scan parallelism buys the 1.3×; those fixed costs cap it.
- **q9v/q10v/q12v are flat (~1×)** because their execute is dominated by the RELATIONAL TAIL,
  not the search. Each joins the ~10⁵-row hit list back through the **240M-row lineitem** (plus
  orders/customer/partsupp), and that tail runs on GPU0 identically at G=1 and G=2. The search
  is only ~10–20 ms of a 35–80 ms execute, so halving it is invisible in the total and the
  small G=2 dispatch/merge overhead makes the ratio a hair below 1. The search leg itself still
  parallelizes (same mechanism as q11v) — it just isn't the bottleneck of these queries.

**The honest read:** sharding the search is EXACT (all four match the goldens at G=1 and G=2,
saturation guard active, cuVS drawing from the pool) and it lifts a real barrier — the int32
cuDF list-child ceiling that forces partsupp's 32M×96 embedding to chunk-load single-GPU is
gone once each shard's child is under 2³¹ (16M×96 at G=2, in one read). But at this K and with
the relational tail unparallelized, the win shows only where the search dominates (q11v). To
scale the *whole* query one would (a) parallelize the relational tail too (partition lineitem
as in M1/M2 and broadcast the small hit list), and/or (b) cut the per-shard K toward the true
hit count so the top-K extraction and O(G·K) merge shrink. Both are follow-on work; M3's remit
was the exact sharded search, and the search parallelism is real where it is not swamped.

## Reproduce

- DuckDB: `benchmarks/duckdb_minimal.sh --s3-direct --dir <path> --duckdb /usr/local/bin/duckdb`
  (imports sf40 from S3 into native storage, warms, verifies against goldens, times 2nd-min of 6).
- cuDF (H200, single-GPU): build the GPU test binaries and run with `PEACOCK_BENCHMARK=1` on a host with the sf40 dataset present.
- cuDF multi-GPU (both legs are the SAME binary, `peacock_multi_gpu_tpch_tests`, with
  `PEACOCK_BENCHMARK=1` — the fair same-code/same-pooled-allocator comparison):
  the 2× A100 number is a plain run on a 2-GPU host; the 1× A100 (G=1) number is that
  same binary under `CUDA_VISIBLE_DEVICES=0` (one row-group partition, one worker). All
  four queries benchmark reliably in a **single** process run (no `--gtest_filter`
  needed): the process-wide shared `WorkerPool` removed the per-test-teardown state churn
  that made multiple benchmarks-per-process flaky in M1 — the numbers above are one
  q6→q1→q3→q8 run per leg.
- cuDF multi-GPU TPC-H+V (`peacock_multi_gpu_tpchv_tests`, same `PEACOCK_BENCHMARK=1` and
  `CUDA_VISIBLE_DEVICES=0`-vs-2-GPU protocol): needs cuVS, the sf40 embedding parquets, and
  the committed `query_params.jsonl` (`PEACOCK_TPCH_VEC_PARAMS`). All four vector queries also
  benchmark reliably in one process per leg under the shared pool (cuVS included).
