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
load+execute end-to-end can be read alongside: e.g. q6 is 19.2 ms execute but 399.9 ms
load → ~419 ms end-to-end vs DuckDB's 137 ms. Whether the GPU wins end-to-end depends
entirely on whether the load is amortized across queries — and the load is 20x the
execute, so on a single query it does not.

Protocol everywhere: execute-only, all-device-synced, **2nd-minimum** of N runs
(`PEACOCK_BENCHMARK=1`, `PEACOCK_BENCHMARK_RUNS`), discarding first-run pool growth and
kernel first-touch.

**Every benchmark on this page runs over a pooled RMM allocator, and a number taken without
one does not belong here.** rmm's default is a `cudaMalloc`/`cudaFree` per allocation, so an
unpooled run measures the driver's allocator as much as the operators — on the H200 that is
2.3x on q6, and on a unified-memory host it can be the whole query. The single-GPU binaries
install the pool themselves (`cpp/tests/gpu/rmm_pool.hpp`); the multi-GPU ones always did.
`PEACOCK_RMM_POOL=0` disables it, and exists only to measure the allocator's own share.
The engine has no pool yet — [#148](../tickets.md#t148).

| | cuDF | DuckDB |
|---|---|---|
| Host | shad-gpu, NVIDIA H200 (140 GiB VRAM) | Nebius VM, 128 cores, 503 GB RAM |
| Version | libcudf 25.02 | DuckDB v1.5.4 |
| Storage | parquet → VRAM (load excluded) | native `.duckdb`, 32.7 GiB, warm |

## TPC-H (sf40, H200)

| Query | cuDF execute (ms) | cuDF load (ms) | DuckDB warm (ms) |
|---|--:|--:|--:|
| q6 — scan/filter/project/reduce | 19.2 | 397.9 | 137 |
| q1 — group-by + aggregates | 239.9 | 505.4 | 915 |
| q3 — 3-way join, group-by, top-N | 40.1 | 741.2 | 577 |
| q8 — 7 tables, bushy join order | 47.7 | 679.8 | 1126 |

These replace the unpooled figures this table carried until 2026-08-12 (q6 43.8, q1 309.5,
q3 55.8, q8 52.8), which an unpooled re-run reproduces to within 1% — so the change is the
allocator and nothing else. The load column moves too, and by more than the execute column
on the join queries: the parquet reader allocates per column chunk, so it was paying the
same per-allocation cost.

## TPC-H+V (vector range predicate, cuVS brute-force)

cuDF splits vector work into load, index-build (cuVS norm precompute, reused across a
query's probes) and execute; DuckDB has no index and computes distances inline per query,
so the comparable cuDF quantity is execute + index-build/n_probes.

| Query / probe | cuDF execute (ms) | index-build (ms) | load (ms) | DuckDB warm (ms) |
|---|--:|--:|--:|--:|
| q11v / img_000 | 12.4 | 25.4 † (shared /3) | 2000.2 | 1433 |
| q11v / img_017 | 12.8 | — | — | 1501 |
| q11v / img_034 | 12.8 | — | — | 1478 |
| q12v / txt_000 | 33.0 | 16.3 | 1144.2 | 559 |
| q10v / txt_017 | 24.2 | 17.2 | 1421.7 | 960 |
| q9v / txt_034 | 59.5 | 17.6 | 1490.2 | 1631 |

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

## DGX Spark (GB10) — one memory pool, and what it costs to pretend otherwise

A second host, and a different question. Everything above assumes a discrete GPU with its
own VRAM, where getting data to the device is a transfer across a bus. GB10 is not that
machine, so the numbers below start by establishing what it actually is before any query
runs on it.

| | value |
|---|---|
| Device | NVIDIA GB10, sm_121, `integrated=1`, `pageableMemoryAccess=yes` |
| Memory | 121.7 GiB reported by CUDA, against 127.6 GB of system RAM — the same pool |
| Bus / L2 | 256-bit LPDDR5X, 24 MiB L2 |
| Software | driver 580.173.02, CUDA 13.0, libcudf 26.02.01 (cuda13 build), aarch64 |

`nvidia-smi` reports FB memory as `N/A` because there is no framebuffer: a 20-core Grace
CPU die and a Blackwell GPU die share one soldered LPDDR5X pool over NVLink-C2C. This is
specific to GB10 and does not generalize to the rest of the family — Grace-Hopper and
GB200-class Grace-Blackwell both pair Grace's LPDDR5X with the GPU's own HBM, two real
pools with a link between them, which is the case the rest of this page was measured on.

### RAM ↔ VRAM, GB10 against H200

`cpp/benchmarks/bench_ram_vram.cu`, 2nd-minimum of 5, GB = 1e9 bytes, 2 GiB buffers. Each
phase allocates and frees independently, so the buffer can be sized past any cache; the GB10
figures are flat from 64 MiB to 8 GiB, which is what rules out a cache explanation.

| Path | GB10 | H200 | GB10 / H200 |
|---|--:|--:|--:|
| host memcpy (1 core) | 24.3 | 11.8 | 2.1x |
| H2D `cudaMemcpy` pageable | 59.5 | 10.6 | 5.6x |
| D2H `cudaMemcpy` pageable | 59.4 | 19.0 | 3.1x |
| H2D pinned | 59.2 | 47.3 | 1.25x |
| D2H pinned | 59.2 | 47.3 | 1.25x |
| H2D pinned async, 4 streams | 59.2 | 47.3 | 1.25x |
| D2D `cudaMemcpy` | 114.5 | 1640.4 | 0.07x |
| copy kernel (r+w) | 218.7 | 3990.2 | 0.05x |
| managed prefetch H→D | 2586 | 4352 | — † |
| **kernel read of mapped host memory** | **218.1** | **35.2** | **6.2x** |
| kernel read of device memory | 239.1 | 4483.8 | 0.05x |

† Neither number is a transfer. On GB10 nothing moves because there is one pool; on H200 the
pages were already device-resident. A prefetch rate above the link rate always means no data
crossed the link.

**Pinned** means page-locked (`cudaHostAlloc`): the OS may not move or swap those pages, so
the GPU's DMA engine can read them directly over the link. Pageable memory carries no such
guarantee, so the driver first copies it into a pinned staging buffer — which is why H200's
pageable rate (10.6) tracks its single-core memcpy rate (11.8) rather than its link rate,
and why GB10, whose GPU addresses pageable host memory directly
(`pageableMemoryAccess=yes`), shows no difference at all.

Four things this settles, and only the first is obvious.

**The GPUs are not close.** H200 reads its own memory at 4484 GB/s against GB10's 239 — 19x.
Any query bound by bandwidth over device-resident data will show roughly that gap, and GB10
cannot make it up.

**The load paths are far closer than the GPUs are, and they favour GB10**: 59 vs 47 GB/s
pinned, and 5.6x on the pageable path a naive loader actually takes. GB10 is also indifferent
to pinning and to streams, because there is no bus to overlap against; H200 loses 4.4x when
the host buffer is not pinned.

**The ratio that decides plan shape is compute-read over load**: 4x on GB10 against 95x on
H200. On H200 the device is starved by its own feed, so keeping data resident across queries
is everything. On GB10 the feed nearly keeps up with the compute, so re-reading is
comparatively cheap and residency buys much less.

**Zero-copy inverts between the machines.** A kernel reading mapped host memory gets 91% of
device-memory bandwidth on GB10 and 0.8% on H200 — a 7% penalty against a 127x one. On GB10
staging a copy is close to pointless; on H200 it is mandatory. Code choosing between them
must ask `cudaDeviceProp::integrated` rather than assume.

## GB10 vs H200 — the same queries

TPC-H sf40, ms, 2nd-minimum of 5, pooled allocator on both hosts. **Whole-table** is
`test_tpch.cpp`, every column resident before the operators run. **Streamed** is
`test_tpch_streamed.cpp`, a byte-bounded chunked reader with per-batch operators, verified
against the same DuckDB goldens.

| Query | H200 exec | H200 load | GB10 whole exec | GB10 whole load | GB10 stream exec | GB10 stream load |
|---|--:|--:|--:|--:|--:|--:|
| q6 | 19.2 | 397.9 | 169.3 | 340.4 | 213.4 | 82.9 |
| q1 | 239.9 | 505.4 | 73540.4 | 294.0 ‡ | 1019.0 | 110.4 ‡ |
| q3 | 40.1 | 741.2 | 183.9 | 512.2 | 168.3 | 97.4 |
| q8 | 47.7 | 679.8 | 110.5 | 655.4 | 127.5 | 377.5 |

‡ q1's two GB10 load cells are the only cold figures on this page. q1 executes for ~75 s
there, so three warm passes cost 22 minutes to refine a number the analysis does not rest
on; its execute times are 2nd-minimum of 5 like everything else.

GB10 streamed peaks, from the allocator's own high-water mark: q6 861 MiB, q1 2340 MiB,
q3 1003 MiB, q8 939 MiB, at 512 MiB chunks. The whole-table shape needs 11-62 GiB for the
same queries, so on this host the streamed column is the one that would still run if the
machine were shared.

**Execute, q6/q3/q8: H200 leads by 2.3-8.8x**, against a 19x device-bandwidth advantage.
The gap narrows as a query does more per byte — q6 is a scan-and-reduce and lands nearest
the bandwidth ratio (8.8x), q8 is seven joins over already-small intermediates and lands
furthest from it (2.3x). Nothing here is surprising.

**Execute, q1: H200 leads by 307x, and that is not a bandwidth story.** See below.

**Load, warm on both hosts.** Every load figure here comes from running one query three
times back-to-back in its own process, so only that query's own columns are cached. That
matters: measured the lazy way — one pass of the binary, tests in declaration order — q6
warms `lineitem` for q1 and the later queries pay for tables nobody touched. Isolating them
moved H200's q3 load from 686 to 741 ms and q1's from 434 to 505 ms, in the *up* direction.

GB10 wins the TPC-H loads: q6 340 vs 398 ms, q3 512 vs 741, q8 655 vs 680. That is the
prediction the bandwidth table makes — GB10's pageable H2D path is 5.6x H200's and the
parquet reader does not pin its host buffers ([#149](../tickets.md#t149)).

**It loses the heavy vector loads, and that is the more interesting half**: q11v 2952 vs
2000 ms, q10v 2326 vs 1422, q9v 2328 vs 1490. GB10 wins where a load is dominated by moving
bytes and loses where it is dominated by *decoding* them, because parquet load is not a
transfer — it is Snappy decompression plus page decode, which is compute. The kernel
breakdown below shows the split directly. q12v sits on the line (1114 vs 1144) because its
non-vector columns compress well and decode cheaply.

**Streamed load is 3-4x cheaper than whole-table load on GB10** (q6 82 vs 340 ms for the same
four columns). An earlier version of this page said that was pool growth. **It is not.**
Sweeping the pool's initial reservation on whole-table q6:

| allocator condition | load |
|---|--:|
| pool, 25% of free initial (default, ~28 GiB) | 337-353 ms |
| pool, 60% initial — growth made impossible | 338-346 ms |
| pool, 5% initial — growth forced | 1762-1777 ms |
| no pool, `cudaMalloc` per allocation | 850-857 ms |
| streamed, 512 MiB chunks | 81-82 ms |

Reserving 60% up front changes nothing, so no growth happens at the default — q6 reads ~6.7 GB
and that already fits. Growth is punishing when forced (5x), and the pool is worth 2.5x over
none, but neither is the 4x in question.

The profiles show the two shapes doing different things. The whole-table read issues ONE
`unsnap` (33.4 ms) and one page decode (34.9 ms); the chunked reader issues 29 of each, and
its `unsnap` launches sum to 208 ms of device time inside 84.8 ms of wall time — possible only
if they overlap, so the chunked path decompresses concurrently across streams. The whole-table
load meanwhile spends 344 ms of wall time on ~90 ms of load kernels, so most of it is
host-side. Pipelining host reads against device decode is the obvious next hypothesis and is
only that; NVTX ranges around the reader's phases would settle it and have not been run. What
is established is the negative: it is not the allocator.

**Streamed execute beats whole-table on q1, q3 and q8, and loses on q6.** q6 is a single
pass with no reduction to speak of, so batching only adds per-batch launch overhead
(211 vs 170 ms). The other three all build state — a groupby or a join — and there the
smaller working set wins, decisively so on q1.

TPC-H+V, same protocol. The streamed columns are absent because the streamed binary
implements the four TPC-H queries only: a streamed brute-force top-k needs per-batch
distances and a merge of per-batch top-k, which is not written yet.

| Probe | H200 exec | H200 load | H200 setup | GB10 exec | GB10 load | GB10 setup |
|---|--:|--:|--:|--:|--:|--:|
| q11v / img_000 | 12.4 | 2000.2 | 25.4 | 65.2 | 2951.6 | 70.3 |
| q11v / img_017 | 12.8 | — | — | 66.3 | — | — |
| q11v / img_034 | 12.8 | — | — | 65.4 | — | — |
| q12v / txt_000 | 33.0 | 1144.2 | 16.3 | 118.8 | 1113.8 | 33.1 |
| q10v / txt_017 | 24.2 | 1421.7 | 17.2 | 92.1 | 2326.4 | 32.9 |
| q9v / txt_034 | 59.5 | 1490.2 | 17.6 | 126.8 | 2328.4 | 31.4 |

### What the vector loads actually move

Column sizes from the parquet footers (`total_compressed_size` / `total_uncompressed_size`
summed over row groups), for exactly the columns each probe reads:

| Probe | compressed MB | uncompressed MB | vector MB | vector share | H200 GB/s † | GB10 GB/s † |
|---|--:|--:|--:|--:|--:|--:|
| q11v | 12702.6 | 13509.8 | 12661.3 | **93.7%** | 6.75 | 4.58 |
| q12v | 6473.0 | 8192.6 | 3293.3 | 40.2% | 7.16 | 7.35 |
| q10v | 8147.8 | 11208.4 | 3293.3 | 29.4% | 7.88 | 4.82 |
| q9v | 9098.7 | 12886.5 | 3293.3 | 25.6% | 8.65 | 5.53 |

† uncompressed bytes divided by the load column, i.e. the rate at which columns are
materialized in device memory. Both columns are warm — each query was run three times
back-to-back in its own process, so only its own columns are cached and the two hosts are
measured identically.

q11v is the outlier because it is nearly all vector: `ps_image_embedding` is 96 float32 over
32M partsupp rows, 12.7 GB, against the 100-dim `p_text_embedding` over 8M part rows (3.3 GB)
that the other three read.

**The embeddings do not compress.** 12306 MB compressed against 12661 uncompressed is a ratio
of **1.03x** — float32 embeddings are high-entropy, and Snappy finds almost nothing. The
non-vector columns manage about 1.6x.

### The GPU decompresses them anyway

q11v profiled on H200 (`nsys`, CUDA kernel trace), share of total GPU kernel time:

| kernel | time | share |
|---|--:|--:|
| `gpuDecodePageDataGeneric` — parquet page decode | 634.6 ms | 42.6% |
| `nvcomp::unsnap_kernel` — Snappy decompression | 564.7 ms | 37.9% |
| `gpuComputePageSizes` | 198.6 ms | 13.3% |
| `gemv2T_kernel` — cuVS distances | 39.2 ms | 2.6% |
| `radix_kernel` — top-k select | 21.3 ms | 1.4% |

Shares are of GPU kernel time, not wall time: these sum to ~1.49 s against a 2309 ms load and
2605 ms total, the difference being host-side file reading, transfers and gaps.

The same profile on GB10, warm, same method:

| kernel | GB10 | share | H200 | GB10 / H200 |
|---|--:|--:|--:|--:|
| `decode_page_data_generic` | 590.2 ms | 35.0% | 634.6 ms | **0.93x** |
| `nvcomp::unsnap_kernel` | 419.5 ms | 24.9% | 564.7 ms | **0.74x** |
| `compute_page_sizes_kernel` | 201.2 ms | 11.9% | 198.6 ms | 1.01x |
| `gemv2T_kernel` — cuVS distances | 316.0 ms | 18.8% | 39.2 ms | **8.06x** |
| `radix_kernel` — top-k select | 51.6 ms | 3.1% | 21.3 ms | 2.42x |

Load kernels are 72.8% of GB10's GPU time against 93.8% on H200, and the search is 25.6%
against 4%. The reason is in the last two rows: GB10 matches or beats the H200 on every
*load* kernel and is 8x slower on the dense FP one. A vector query is therefore a different
shape of workload on the two machines — mostly loading on H200, a third compute on GB10.

**Those load-kernel rows are a library comparison, not a hardware one.** A GB10 does not
out-decode an H200. Two things say so. The launch counts differ over identical input — decode
is 3 launches on both but `unsnap` is 8 on H200 against 6 on GB10 — so the two libcudf
versions split the work differently and these are different implementations rather than one
kernel on two devices. And the H200 side is not noise: shad-gpu is shared with a vLLM tenant,
so the profile was repeated on a verified-idle GPU and reproduced to four significant figures
(decode 634.50 vs 634.56 ms, unsnap 564.21 vs 564.71). Contention is excluded; version is
what is left, and 26.02 bundles a newer nvcomp.

Note also that Snappy is the row where a bandwidth advantage was least likely to show:
LZ77 back-references are serially dependent inside each block, so `unsnap` is latency- and
divergence-bound rather than FLOP- or bandwidth-bound. The same experiment that attributes
the q1 cliff attributes this.

So **93.8% of the GPU work in a vector query is loading and 4% is the search**, and 565 ms of
that loading is spent decompressing data that compression shrank by 3%. Writing the embedding
columns with `compression=NONE` would delete the `unsnap` kernel entirely and cost ~3% on
disk. Nothing else happens to the vectors after decode — the float32 child of the list column
is `cudaMemcpyAsync`'d device-to-device into the search buffer, with no re-encoding, no cast
and no normalization at load time.


Vector execute is 2.1-5.4x slower on GB10, a wider spread than the TPC-H queries and for a
different reason: brute-force top-k is dense FP arithmetic over embeddings, so it leans on
FLOPs and HBM together rather than on bandwidth alone, and an H200 outclasses GB10 on both.
The index-build column widens further (87 vs 26 ms, 16 vs 1 ms) because a cuVS norm
precompute is pure compute.

The q11v load first measured 8601 ms here, and that figure was an artifact worth recording:
it was taken minutes after the 26.9 GB `partsupp` file was downloaded, with the page cache
near empty. Two re-runs on the same binary with memory free give 2989 and 3007 ms, and the
execute column does not move at all (66.3 vs 66.9 ms), so the load was reading from NVMe
and the rest was not affected. The table carries the warm figure.

The pool is NOT the explanation, though it was the first one to suggest itself: the run
reported a 24.78 GiB allocator high-water against a 28.2 GiB initial reservation, so it fit
and no pool growth happened. A load column is a disk measurement before it is anything else,
and the three remaining vector loads come from that same cold run — read them as upper
bounds.

### Per-operator, the same 50M lineitem rows

`cpp/tests/gpu/test_cudf_nodes.cpp`, ms, 2nd-minimum of 5, pooled. Real sf40 columns with
their real distributions, because cuDF's cost depends on the data and not only its size —
the two groupby rows differ only in how many distinct keys the column holds.

| Operator | GB10 | H200 | GB10 / H200 |
|---|--:|--:|--:|
| cast dec64 → dec128 | 5.42 | 0.31 | 17.5x |
| cast dec64 → float64 | 3.55 | 0.80 | 4.4x |
| expr dec128 mul | 10.75 | 0.83 | 13.0x |
| expr dec128 sub (scalar − col) | 7.05 | 0.63 | 11.2x |
| expr timestamp ≥ scalar | 1.15 | 0.32 | 3.6x |
| expr string == scalar | 1.57 | 0.51 | 3.1x |
| filter mask, 2 dec128 cols (qty<26) | 12.81 | 0.99 | 12.9x |
| filter mask, 2 dec128 cols (disc=0.05) | 3.57 | 0.42 | 8.5x |
| reduce sum dec128 | 3.40 | 0.24 | 14.2x |
| groupby 1 int64 key, 4 groups | *pending* | 6.95 | |
| groupby 1 int64 key, 1000 groups | *pending* | 8.68 | |
| groupby 1 int64 key, ~n/4 groups | 87.94 | 7.92 | 11.1x |
| groupby 1 string key, 3 groups | *pending* | 14.27 | |
| groupby 2 string keys, 4 groups | 18.28 | 32.15 | 0.6x |
| groupby q1 shape, 8 aggregates | 89.18 | 38.44 | 2.3x |
| groupby 4 dec128 sums only | 29.93 | 32.16 | 0.9x |
| groupby 3 float64 means only | 14.94 | 12.73 | 1.2x |
| hash_join build (12.5M keys) | 9.77 | 0.89 | 11.0x |
| hash_join probe | 68.47 | 15.15 | 4.5x |
| hash_join probe + gather 2 cols | 86.96 | 16.48 | 5.3x |
| sorted_order on int64 | 45.00 | 3.75 | 12.0x |
| gather 2 dec128 by a sort map | 25.36 | 1.98 | 12.8x |

**What drives a groupby is the key, not the group count.** On H200 the same aggregation over
one int64 key costs 6.95 ms at 4 groups, 8.68 at 1000 and 7.92 at 13.3M — flat across six
orders of magnitude of cardinality. Swapping that key for one string column costs 2.1x, and
for two string columns 4.6x, at identical cardinality. Variable-length key comparison is the
cost; a hash groupby does not care how many groups come out.

That is worth stating because the obvious pair of cases says the opposite. "4 groups by
(l_returnflag, l_linestatus)" against "13.3M groups by l_orderkey" varies key type, key
width, column count and cardinality together, and reads as though low cardinality were
catastrophic. It is not; the string keys are.

**Aggregation is nearly free next to grouping.** Eight aggregates cost 38.4 ms against 32.2
for one (H200), and four DECIMAL128 sums cost the same as one — 32.14 against 32.15. A plan
that worries about how many aggregates a groupby carries is worrying about the wrong term.

### The q1 cliff on GB10

Whole-table q1 costs 73.5 s on GB10 against 239 ms on H200 — 307x, where the device
bandwidth ratio is 19x. The profile says where it goes, and it is one kernel:

```
98.9%   72,860,158,304 ns   1 instance
cudf::groupby::detail::hash::single_pass_shmem_aggs_kernel(...)
```

That is the hash groupby, not a sort fallback — `mapping_indices_kernel(cuco::static_set_ref…)`
sits alongside it — and everything else in the query, casts, filter, projections and the
parquet decode together, is under 150 ms.

Timing the same eight-aggregate groupby against input size on GB10 shows it is not a slope
but a step:

| rows | q1-shape groupby | vs previous |
|---:|--:|--:|
| 25M | 41.3 ms | — |
| 50M | 89.3 ms | 2.17x |
| 100M | 199.6 ms | 2.23x |
| 200M | 12262.1 ms | **61.4x** |
| 240M | 25475.8 ms | 2.08x |

Linear at ~2 ms per million rows up to 100M, a 61x discontinuity between 100M and 200M,
linear again above it. A cost that grows smoothly with contention does not do that; a
strategy switch inside cuDF's groupby does, and the branch it takes above the threshold is
the shared-memory kernel the profile names.

H200 has no such step over the same range:

| rows | GB10 | H200 |
|---:|--:|--:|
| 25M | 41.3 ms | 20.3 ms |
| 50M | 89.3 | 38.4 |
| 100M | 199.6 | 74.9 |
| 200M | 12262.1 | 139.4 |
| 240M | 25475.8 | 164.5 |

H200 holds ~1.4 Grow/s throughout. GB10 tracks it within 2.4x to 100M and then leaves. Both
report 4 output groups at 200M and above, so the step is not a change in group count.

**Attributed: a 26.02 regression that this hardware amplifies.** The hosts differed in library
as well as hardware, so libcudf 26.02 was built on the H200 and the curve re-run there. Ratio
of 200M to 100M rows, where a linear cost gives 2.0x:

| configuration | 100M | 200M | ratio |
|---|--:|--:|--:|
| H200, libcudf 25.02 | 74.9 ms | 139.4 ms | 1.86x |
| H200, libcudf 26.02 | 60.5 ms | 279.2 ms | **4.62x** |
| GB10, libcudf 26.02 | 199.6 ms | 12262.1 ms | **61.4x** |

Same hardware, same data, different library: 25.02 is linear, 26.02 is not. The discontinuity
is 26.02's. GB10 then makes it ~13x worse again, so the device matters too — a 26.02
regression that this hardware amplifies, not one or the other.

26.02 is not uniformly worse, which is why this reads as a changed strategy rather than a
slowdown: at 50M it is 2.5x FASTER than 25.02 on the two-string-key groupby (13.1 vs 32.2 ms)
and ~5x slower on a 1000-group int64 key (45.2 vs 8.7 ms).

One axis stays confounded: shad-gpu is CUDA 12.2 and sparkdgx CUDA 13.0, and that cannot be
changed without touching a driver. Read "GB10 amplifies it" as "GB10 or CUDA 13".

This is the whole reason the streamed q1 wins by 72x (1.0 s against 73.5 s). It is not a
better algorithm: 29 batches of ~8M rows each keep every call on the linear side of the
step, and one 240M-row call does not. It also means the win is not general — it is worth
exactly as much as the distance between the batch size and the cliff, and a plan that sized
its batches at 150M rows would get none of it.


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
