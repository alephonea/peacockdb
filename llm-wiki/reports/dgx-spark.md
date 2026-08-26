# DGX Spark (GB10) for peacockdb

What this hardware is, and what it changes for an engine designed against discrete GPUs.
Query-level numbers and the GB10-vs-H200 comparison live in
[benchmark-minimal.md](benchmark-minimal.md); this page is the machine.

Host: NVIDIA GB10, sm_121, `integrated=1`, 121.7 GiB unified LPDDR5X (== system RAM, no
framebuffer), 256-bit, 24 MiB L2, CUDA 13.0, driver 580.173.02, libcudf 26.02.01, aarch64,
20-core Grace CPU. Everything below is 2nd-minimum of repeated runs on an otherwise idle host.

## 1. There is one pool, and it changes the loader's job

`cudaMemcpy` is capped at ~59 GB/s in both directions and is indifferent to pinning, to
streams and to buffer size — the signature of a copy engine, not a link. A kernel reading
*mapped host memory* gets 205-219 GB/s against 233-239 GB/s for device memory: a 7-12%
penalty where a discrete part pays 127x. Sizes from 64 MiB to 64 GiB, flat throughout; at
64 GiB that is 2730x the L2, so nothing here is a cache effect.

| path | 2 GiB | 16 GiB | 32 GiB | 64 GiB |
|---|--:|--:|--:|--:|
| H2D pageable | 59.5 | 59.3 | 59.4 | n/a † |
| H2D pinned | 59.2 | 51.9 | 48.3 | n/a † |
| D2D `cudaMemcpy` | 114.5 | 115.1 | 115.7 | n/a † |
| kernel read, mapped host | 218.1 | 209.5 | 204.1 | 205.5 |
| kernel read, device | 239.1 | 234.6 | 232.8 | 234.0 |

† two-buffer phases need 128 GiB at that size and are skipped.

**Pinning is not just useless here, it is counterproductive at scale**: pageable holds
59.4 GB/s at 32 GiB while pinned falls to 48.3. On a discrete host pinning is worth 4.4x
([#149](../tickets.md#t149)); on this one it should be off. That is a branch on
`cudaDeviceProp::pageableMemoryAccess`, not a global choice.

**Consequence for peacockdb.** Staging a copy into device memory costs ~4x what reading host
memory in place costs and buys nothing. The row-group->partition map exists to bound what
reaches VRAM; on this part there is nothing to bound.

## 2. The parquet reader, not the allocator, decides load time

Whole-table load was 4x a streamed load of the same columns. It is not the allocator —
reserving 60% of memory up front, which makes pool growth impossible, changes nothing — and
it is not chunking:

| reader | batches | q6 load |
|---|--:|--:|
| `cudf::io::read_parquet` | — | 340 ms |
| `chunked_parquet_reader`, 16 GiB chunk | 1 | **60.3-61.6 ms** |
| same, 4 GiB | 4 | 60.7-61.0 |
| same, 1 GiB | 15 | 62.5-64.4 |
| same, 512 MiB | 29 | 81.3-82.2 |
| same, 128 MiB | 115 | 240.1-245.2 |

The chunked reader with **one** chunk — identical single-shot semantics — is 5.6x faster than
`read_parquet` on the same bytes. The host floor says the fast number is the right one: `dd`
delivers this 9.0 GB file from page cache at 23.6 GB/s, so q6's ~1.5 GB of compressed columns
needs ~64 ms, which is what the chunked reader achieves and what `read_parquet` misses by 5x.
`KVIKIO_NTHREADS` explains part of the gap (1 thread 480 ms, 4 threads 341, 8 threads 323,
16 threads 323) and not the rest.

**Two consequences.** Use `chunked_parquet_reader` even where streaming is not needed. And
1-4 GiB is the chunk-size sweet spot: the streamed executor's 512 MiB default costs ~25%, and
128 MiB costs 4x.

## 3. Allocation is the other half of the load story

rmm's default is a `cudaMalloc`/`cudaFree` per allocation, and on one pool a free must return
pages to the OS. Whole-table q6 load, by allocator condition:

| condition | load |
|---|--:|
| pool, 25% of free reserved (default) | 337-353 ms |
| pool, 60% reserved (growth impossible) | 338-346 ms |
| pool, 5% reserved (growth forced) | 1762-1777 ms |
| no pool | 850-857 ms |

Pooling is worth 2.5x, and pool growth mid-query is worth 5x against you. The engine installs
no allocator at all — [#148](../tickets.md#t148).

## 4. Snappy costs more than it saves on embeddings

The sf40 embedding columns compress 1.03x (float32 is high-entropy) yet are decompressed on
the GPU anyway. In q11v, `nvcomp::unsnap_kernel` is 24.9% of GPU kernel time here and 37.9%
on the H200, recovering 3%. Loading altogether is 72.8% of GB10's GPU work in a vector query
against 25.6% for the cuVS search that is the point of it. [#150](../tickets.md#t150).

## 5. Where GB10 wins and loses against an H200

Per-kernel, q11v, same query and data:

| kernel | GB10 | H200 | ratio |
|---|--:|--:|--:|
| page decode | 590.2 ms | 634.6 ms | 0.93x |
| Snappy decompress | 419.5 | 564.7 | 0.74x |
| cuVS distances (`gemv2T`) | 316.0 | 39.2 | **8.06x** |

GB10 matches or beats the H200 on load kernels and is 8x slower on dense FP. Read the first
two rows with care: the hosts run different libcudf (26.02.01 vs 25.02), so those are
different implementations rather than one kernel on two devices. The `gemv2T` row is not
confounded that way and is the honest measure of the compute gap.

So a vector query is a different workload on the two machines: mostly loading on H200,
a quarter compute on GB10.

## 6. cuDF 26.02 has a groupby cliff, and this hardware amplifies it

An eight-aggregate groupby is linear to 100M rows and then steps. Ratio of 200M to 100M,
where linear is 2.0x:

| configuration | 100M | 200M | ratio |
|---|--:|--:|--:|
| H200, libcudf 25.02 | 74.9 ms | 139.4 ms | 1.86x |
| H200, libcudf 26.02 | 60.5 ms | 279.2 ms | 4.62x |
| GB10, libcudf 26.02 | 199.6 ms | 12262.1 ms | **61.4x** |

Same hardware, different library: the step is 26.02's. GB10 makes it ~13x worse again. The
profile names one kernel — `cudf::groupby::detail::hash::single_pass_shmem_aggs_kernel`,
72.9 s in a single launch, 98.9% of whole-table q1. CUDA version stays confounded (12.2 vs
13.0), so read "GB10 amplifies" as "GB10 or CUDA 13".

**This is why the streamed executor wins q1 by 72x** (1.0 s against 73.5 s): 29 batches of
~8M rows keep every call on the linear side of the step. The win is worth exactly the
distance between the batch size and the cliff, and a plan batching at 150M rows would get
none of it.

## 7. Grouping cost is the key, not the group count

At 50M rows on H200, one int64 key costs 6.95 ms at 4 groups, 8.68 at 1000 and 7.92 at
13.3M — flat across six orders of magnitude of cardinality. One string key costs 2.1x that,
two string keys 4.6x, at identical cardinality. Eight aggregates cost 6 ms more than one, and
four DECIMAL128 sums cost the same as one.

For the planner: the term to watch is the width and type of the group key, not how many
groups come out and not how many aggregates ride along.

## 8. One query saturates the GPU

Streamed q6, N copies at once, whole-process wall time:

| N | wall | per-query execute | throughput |
|--:|--:|--:|--:|
| 1 | 3.27 s | 211.8 ms | 0.30 q/s |
| 2 | 4.74 s | 524.4 ms | 0.42 q/s |
| 4 | 7.27 s | 791.7 ms | 0.55 q/s |

Four concurrent queries buy 1.83x throughput for 3.7x latency. There is little idle capacity
for a second query to occupy, so on this part concurrency is a throughput knob with a steep
latency price, not free parallelism — worth knowing before sizing a server around it.

## 9. No thermal throttling

55 streamed-q6 iterations back to back over 180 s: clocks pinned at 2411 MHz, 45-46 C,
11.5-11.7 W, execute steady at 210-213 ms with no drift. Nothing here is thermally limited,
and a benchmark does not need a cool-down between runs.

## 10. Uncompressed parquet reads faster than Snappy, measured

Rewriting `p_text_embedding` both ways and reading each with cuDF:

| file | size | cuDF read | rate |
|---|--:|--:|--:|
| Snappy | 3241.5 MB | 645.9 ms | 5.02 GB/s |
| uncompressed | 3311.4 MB | **446.1 ms** | 7.42 GB/s |

**1.45x faster to read while 2.2% larger** — the direct confirmation of
[#150](../tickets.md#t150), on the read path rather than inferred from a kernel share.

## 11. CPU and GPU contend for the same DRAM, and it is not small

One pool means one memory controller. Running N CPU threads each churning a 1 GiB working set
(a real memcpy loop; an earlier attempt used `dd /dev/zero`, whose 1 MiB buffer stays in cache
and showed no effect at all):

| CPU threads | host memcpy | H2D pageable | GPU read of device memory |
|--:|--:|--:|--:|
| 0 | 27.5 GB/s | 59.5 GB/s | 239.6 GB/s |
| 4 | 9.8 | 56.5 | 197.9 (−17%) |
| 12 | 4.1 | 46.9 (−21%) | **150.5 (−37%)** |

**CPU traffic takes up to 37% of the GPU's bandwidth.** On a discrete part CPU work is free
from the GPU's point of view; here it is not. That is a direct constraint on the hybrid
placement [#147](../tickets.md#t147) contemplates — moving a subtree to the CPU does not just
occupy CPU cores, it slows down whatever the GPU is doing at the same time. A placement rule
tuned on a discrete host would over-schedule the CPU here.

## 12. Allocation costs what a memset costs

| size | `cudaMalloc` | `cudaFree` | implied alloc rate |
|--:|--:|--:|--:|
| 1 MiB | 0.11 ms | 0.06 ms | 9.6 GB/s |
| 256 MiB | 9.59 | 3.16 | 28.0 GB/s |
| 1 GiB | 37.14 | 11.23 | 28.9 GB/s |
| 4 GiB | 143.36 | 43.36 | 30.0 GB/s |

Allocation runs at ~30 GB/s, i.e. the driver is zeroing pages, and free is ~3x cheaper. A
query that allocates tens of GB unpooled pays this on every intermediate — which is the
mechanism behind [#148](../tickets.md#t148) and behind the 5x that forced pool growth costs.

**Kernel launch is 1.9-2.2 us** (48 SMs, empty kernel, batched). That is the floor on batching
finer: the batch-partitioned mode issues one call per node per batch, so a 20-node plan at
1000 batches spends ~40 ms in launches alone before doing any work.

## 13. Managed memory oversubscribes, and first touch is where it hurts

`cudaMallocManaged` succeeds at **120% of total memory** — 146 GiB requested on a 121.7 GiB
machine — and the allocation itself is free (0.1 ms). Touching it is where the cost is: 1 GiB
of first touch takes 320-418 ms, about 2.5-3 GB/s against 239 GB/s for resident memory.

Two consequences. Managed memory is a real answer to [#142](../tickets.md#t142)'s "no recourse
for an oversized batch" — a plan that would die can instead run, slowly, rather than not at
all. And it must not be the default: at ~3 GB/s on first touch it is two orders of magnitude
off resident memory, so it belongs on the recovery path, not the hot one.

Separately, **it does not matter who writes a managed page first**: GPU read after a CPU write
is 161.6 GB/s and after a GPU write 162.3 GB/s. On one pool there is no migration to pay for,
so the engine need not care where a buffer was filled.

## Reproduce

- `cpp/benchmarks/bench_ram_vram.cu` — standalone, CUDA runtime only, both CUDA 12 and 13.
  `nvcc -O3 -std=c++17 -arch=native -o bench_ram_vram cpp/benchmarks/bench_ram_vram.cu`,
  then `./bench_ram_vram [buffer_MiB]`.
- `cpp/benchmarks/dgx_spark_probe.cu` — launch cost, allocation cost by size, first-touch,
  managed oversubscription.
- `cpp/tests/gpu/test_cudf_nodes.cpp` (`peacock_cudf_node_tests`) — per-operator timings;
  `PEACOCK_NODES_ROWS` drives the scale curves above.
- `cpp/tests/gpu/test_tpch_streamed.cpp` (`peacock_tpch_streamed_tests`) —
  `PEACOCK_STREAM_CHUNK_MB` / `PEACOCK_STREAM_PASS_MB` drive the chunk sweep.
- Pool sizing: `PEACOCK_RMM_POOL_INIT_PCT=<n>`. The no-pool rows were taken before the
  switch that produced them was removed; there is no unpooled configuration now.
