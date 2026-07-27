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

## Reproduce

- DuckDB: `benchmarks/duckdb_minimal.sh --s3-direct --dir <path> --duckdb /usr/local/bin/duckdb`
  (imports sf40 from S3 into native storage, warms, verifies against goldens, times 2nd-min of 6).
- cuDF: build the GPU test binaries and run with `PEACOCK_BENCHMARK=1` on a host with the sf40 dataset present.
