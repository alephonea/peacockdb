# Minimal benchmark — cuDF vs DuckDB (TPC-H sf40 + TPC-H+V), and multi-GPU scaling

Ten queries with committed DuckDB goldens (4 TPC-H, 6 TPC-H+V probes). Both sides run the
same result-defining SQL/operators, and every timed run was verified against its golden.

## Reading the comparison honestly

The two sides do **not** measure the same boundary:

- **cuDF execute** — GPU operator time only, inputs **already resident in VRAM**
  (parquet→VRAM load excluded). Ends on `cudaDeviceSynchronize`.
- **DuckDB warm** — full query wall time over DuckDB's own storage, page-cache warm
  (includes reading/decoding its columns; there is no separate pre-load tier to exclude).

So cuDF-execute vs DuckDB-warm flatters the GPU. The **load** column is given so a
load+execute end-to-end can be read alongside: e.g. q6 is 43.8 ms execute but 472.7 ms
load → ~516 ms end-to-end vs DuckDB's 137 ms. Whether the GPU wins end-to-end depends
entirely on whether the load is amortized across queries.

Protocol everywhere: execute-only, all-device-synced, **2nd-minimum** of N runs
(`PEACOCK_BENCHMARK=1`, `PEACOCK_BENCHMARK_RUNS`), discarding first-run RMM-pool growth
and kernel first-touch.

| | cuDF | DuckDB |
|---|---|---|
| Host | shad-gpu, NVIDIA H200 (140 GiB VRAM) | Nebius VM, 128 cores, 503 GB RAM |
| Version | libcudf 25.02 | DuckDB v1.5.4 |
| Storage | parquet → VRAM (load excluded) | native `.duckdb`, 32.7 GiB, warm |

## TPC-H (sf40, H200)

| Query | cuDF execute (ms) | cuDF load (ms) | DuckDB warm (ms) |
|---|--:|--:|--:|
| q6 — scan/filter/project/reduce | 43.8 | 472.7 | 137 |
| q1 — group-by + aggregates | 309.5 | 537.7 | 915 |
| q3 — 3-way join, group-by, top-N | 55.8 | 946.6 | 577 |
| q8 — 7 tables, bushy join order | 52.8 | 981.3 | 1126 |

## TPC-H+V (vector range predicate, cuVS brute-force)

cuDF splits vector work into load, index-build (cuVS norm precompute, reused across a
query's probes) and execute; DuckDB has no index and computes distances inline per query,
so the comparable cuDF quantity is execute + index-build/n_probes.

| Query / probe | cuDF execute (ms) | index-build (ms) | load (ms) | DuckDB warm (ms) |
|---|--:|--:|--:|--:|
| q11v / img_000 | 13.2 | 27.1 † (shared /3) | 2354.9 | 1433 |
| q11v / img_017 | 13.6 | — | — | 1501 |
| q11v / img_034 | 13.7 | — | — | 1478 |
| q12v / txt_000 | 41.4 | 1.5 | 1511.0 | 559 |
| q10v / txt_017 | 33.1 | 1.7 | 1631.9 | 960 |
| q9v / txt_034 | 62.1 | 1.7 | 1756.5 | 1631 |

† includes one-time cuVS/cuBLAS first-touch; steady-state marginal build is ~1.5 ms.

Vector queries are where the GPU compute advantage is largest (13–62 ms vs 559–1631 ms),
because cuVS parallelizes the distance computation DuckDB does inline on CPU.

## Multi-GPU scaling

Same binary and same pooled per-device RMM allocator on both legs; G=1 is that binary
under `CUDA_VISIBLE_DEVICES=0`. Ratios therefore isolate parallelism, not allocator.

**2× A100** (NVLink, libcudf 26.02), 2nd-min of 6:

| Query | G=1 (ms) | G=2 (ms) | speedup |
|---|--:|--:|--:|
| q6 | 37.2 | 19.4 | 1.92× |
| q1 | 632.6 | 295.5 | 2.14× |
| q3 | 56.7 | 49.0 | 1.16× |
| q8 | 64.7 | 40.1 | 1.61× |

**8× H100** (HGX, full NVLink/NVSwitch — NV18 all-pairs, libcudf 26.02, CUDA 13),
2nd-min of 5, byte-for-byte verified against the sf40 goldens on all 8 GPUs:

| Query | G=1 (ms) | G=8 (ms) | speedup |
|---|--:|--:|--:|
| q6 | 21.8 | 3.7 | 5.82× |
| q1 | 269.5 | 32.6 | **8.27×** |
| q3 | 33.3 | 17.6 | 1.89× |
| q8 | 39.8 | 17.2 | 2.31× |
| q11v / img_000 | 11.9 | 4.8 | 2.47× |
| q11v / img_017 | 12.2 | 5.1 | 2.41× |
| q11v / img_034 | 12.5 | 5.3 | 2.38× |
| q12v / txt_000 | 32.8 | 10.2 | 3.21× |
| q10v / txt_017 | 22.6 | 12.2 | 1.85× |
| q9v / txt_034 | 47.9 | 14.2 | 3.36× |

Why the spread — every laggard is a cost that is **constant in G**:

- **q1 super-linear (8.27×)**: partials are re-aggregatable (per-GPU aggregates, then a
  merge-sum of tiny partials), so there is no cross-GPU shuffle; sharding the group-by
  working set 8 ways also relieves the memory pressure inflating the G=1 baseline.
- **q6 (5.82×)**: at 3.7 ms the fixed per-query costs (dispatch, host `__int128`
  partial-sum, boundary sync) are a visible fraction — Amdahl tail on too little work.
- **q3 (1.89× at G=8, 1.16× at G=2)**: its group-by key `l_orderkey` is high-cardinality,
  so partials can't just be gathered — the plan pays a real cross-GPU hash-shuffle
  (murmur3 → `pack`/`cudaMemcpyPeerAsync`/`unpack` → concat) on the critical path, and
  the moved bytes grow with G. This query most wants a load-time hash-partition of the
  fact table.
- **q8 (2.31×) and the vector tails**: the broadcast-dim rebuild — every GPU reconstructs
  the small dimension subtree from full-table reads — is constant work in G. (It is still
  the right trade: broadcasting a tiny dimension beats shuffling a large fact.) q10v
  (1.85×) additionally pays a serial top-N tail.

Merge plans match the partial's shape: q1 gathers real partial tables across the link
(pack → peer-copy → unpack → concat → merge-groupby); q6's partial is one decimal per
GPU, summed on the host with exact `__int128` addition (bit-identical to the golden).

## Reproduce

- **DuckDB**: `benchmarks/duckdb_minimal.sh --s3-direct --dir <path> --duckdb /usr/local/bin/duckdb`
  (imports sf40 from S3, warms, verifies against goldens, times 2nd-min of 6).
- **cuDF single-GPU**: build the GPU test binaries and run with `PEACOCK_BENCHMARK=1` on a
  host with the sf40 dataset (see `llm-wiki/build-test.md`).
- **cuDF multi-GPU**: `peacock_multi_gpu_tpch_tests` / `peacock_multi_gpu_tpchv_tests`
  (EXCLUDE_FROM_ALL — build them explicitly) with `PEACOCK_BENCHMARK=1`
  `PEACOCK_BENCHMARK_RUNS=5`; G=1 is the same binary under `CUDA_VISIBLE_DEVICES=0`. The
  vector binary additionally needs cuVS, the sf40 embedding parquets and the committed
  `query_params.jsonl` (`PEACOCK_TPCH_VEC_PARAMS`). The shared `WorkerPool` sizes a
  per-device RMM pool from each GPU's free VRAM.
