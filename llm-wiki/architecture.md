# peacockdb architecture

Pipeline: SQL → DataFusion logical/physical plan → `Gpu*Exec` wrapper annotation →
FlatBuffers plan IR (`flatbuffers/gpu_plan.fbs`) → C++/cuDF executor. The same IR is
consumed by both engines (two-engine correctness: no CPU-only or GPU-only plan nodes).

## Execution modes

Five executor classes in `peacockdb-core/src/executors/`, unified by two traits
(`executors/executor.rs`):

- `Executor` — `execute(sql) -> Vec<RecordBatch>`.
- `InstrumentedExecutor: Executor` — `execute_instrumented(sql)` additionally returns the
  plan and per-node `NodeMemoryStats` (post-order, aligned with the plan tree).

| Class | Traits | Driver | Partition mode / tp |
|---|---|---|---|
| `FullTableCpuExecutor` | both | own recursive streaming path | SinglePartition; tp1 or tp8 hint |
| `PartitionedCpuExecutor` | both | `execute_node_by_node` + `CpuNodeExecutor` | RealMultiPartition, tp8 |
| `FullTableGpuExecutor` | both | driver + `GpuNodeExecutor` | SinglePartition, tp1 |
| `PartitionedGpuExecutor` | both | driver + `GpuNodeExecutor` | RealMultiPartition, tp8 |
| `AllAtOnceGpuExecutor` | `Executor` only | single `peacock_execute` FFI call | SinglePartition, tp1; retiring (#110) |

Partition mode is implied by the class, never a parameter. The two GPU node-by-node
classes are thin config wrappers over the same backend — they differ only in the
constructed mode. Memory budget comes from `MemoryLimit` (micro/mini/standard/full,
`src/config.rs`); plan-level parallelism from `TargetPartitions` (Single/Multi).

## Backend-agnostic node-by-node driver

`executors/node_by_node.rs` defines the `NodeExecutor` trait (`execute_node(seq, node,
input_handles) -> (Vec<u64>, NodeMemoryStats)`, `materialize`, `release`) and the driver
`execute_node_by_node`: walk the plan in canonical post-order (children left-to-right —
matches the C++ NodeSession indexing so handles align across the FFI), feed each node its
children's **partition-handle vectors** (multi-handle model: one node → N handles, one per
partition), materialize and release only the root. Intermediates stay resident in the
backend (CPU registry / GPU VRAM); data crosses the boundary once, at the root. Two
backends implement `NodeExecutor`: `backend/cpu_node_executor.rs` (DataFusion per node)
and `backend/gpu_node_executor.rs` (FFI: `begin_plan` / `execute_node` per node /
`end_plan`; per-partition stats come back as rows + varlen bytes, and Rust recomputes
logical bytes from the schema so CPU and GPU costs are identical by construction).

## Operators

Every GPU-executable DataFusion node is wrapped in a passthrough `Gpu*Exec` (16 types,
grouped by family under `peacockdb-core/src/operators/`). The `Operator` trait
(`operators/operator.rs`) unifies them:

- `inner()` — the wrapped DataFusion node;
- `partition_topology()` — how partition handles flow: `ScanEmit` (scan emits N from its
  row-group map), `Map` (1:1 per partition), `Collapse` (M→1 concat), `KWayMerge` (M→1
  order-preserving), `RepartitionHash` (1→N), `Join` (two inputs → one);
- `strips_to_inner()` — whether CPU emulation replaces the wrapper with its inner node.

`operators/mod.rs::as_operator()` is the single downcast registry; adding an operator is
one line there. The strip set is asymmetric **and load-bearing**: 11 operators strip, 5 do
not (cross join, nested-loop join, union, global limit, window) — flipping one changes
execution substitution and reported node names. `GpuScanExec` is hand-written and carries
the explicit row-group→partition map (`build_scan_map`: survivor row-groups split into N
contiguous chunks; empty map = legacy single-partition, keeping tp1 byte-identical).

Per-operator FlatBuffers serialize/deserialize pairs are co-located in the family files.
**Statement order is the wire format**: FlatBufferBuilder is a no-interning bump arena, so
reordering writes changes bytes even with identical values. `tests/test_plan_bytes.rs` +
`goldens/plan_bytes.sha256` pin the exact bytes per query; regenerating that golden to
silence a red defeats its purpose.

## CPU emulation and the two CPU behaviors

CPU execution strips the `Gpu` wrappers (`operators::strip_target`) and runs the inner
node with DataFusion, honoring `gpu_batch_size` via a TaskContext override so the working
set stays within the same bound.

- **Full-table CPU (partition-collapsing):** `FullTableCpuExecutor`'s recursive streaming
  path (`execute_full_table*`) coalesces every node's output to a single stream — even a
  tp8-built plan executes with N→1 collapse at each node. It also owns the resident-memory
  OOM enforcement (`stream.rs::ResidentEnforcer` trips `ResourcesExhausted` when the
  modeled resident set crosses the budget mid-run).
- **Partitioned CPU (partition-preserving):** `CpuNodeExecutor` keeps N partitions alive
  across nodes as handle vectors: scans replay the row-group map via `ParquetAccessPlan`;
  map ops run once per partition; hash joins run per co-partitioned bucket;
  SortPreservingMerge does a real k-way merge; CoalescePartitions concatenates N→1.
  DataFusion's Partial/Final aggregates need no special arm — they are ordinary map nodes
  around the lowered repartition.

## Rehash and the comet hash

In RealMultiPartition mode the optimizer lowers `GpuRepartition(Hash, M→N)` into an
explicit `GpuCoalescePartitions(M→1)` feeding `GpuRepartition(Hash, 1→N)` — the shuffle's
concat becomes a visible, cost-accounted plan node, and both executors get a single
1-partition input to hash-scatter into N.

The hash is **Spark's murmur3 as implemented by comet** (seed 42), on both engines:
DataFusion's default repartition uses ahash, whose partition numbers differ from any GPU
kernel, and cuDF only exposes standard murmur3, which differs from Spark's spec
(multi-column combine, null handling). To make CPU and GPU row→partition placement
identical **by construction**, the CPU twin uses comet's `create_murmur3_hashes` and the
GPU side owns a bit-exact Spark-murmur3 kernel (`cpp/src/spark_hash_partition.cu`),
reusing cuDF only for the scatter. A live conformance gate (`peacock_spark_partition_ids`,
`test_inc2_conformance.rs`) proves GPU == comet over the same bytes; per-partition row
counts in the tp8 goldens are the murmur3-fidelity numbers.

## C++ executor layout

`cpp/src/`: `gpu_executor.cpp` (C FFI impl), `execute_plan.cpp` (all-at-once recursive
path, retires with #110), `node_session.cpp` (NodeSession: multi-partition dispatch —
scan-map emission, collapse/k-way merge, hash repartition, 1:1 map), `expr.cpp`
(expression/AST building), `operators/` (per-op `execute_*` + `dispatch.cpp` with the
`run_op` switch). Node inputs are threaded explicitly (`NodeInputs{items, idx}` — never a
thread-local; see coding-style.md), and `execute_one` enforces **consumed == provided**:
a node handed inputs must consume all of them, otherwise it would silently re-execute
child subtrees (correct answers, exponential cost). Private headers live in
`cpp/src/peacock/`; the public surface is only the C FFI (`cpp/include/peacock_gpu.h`)
plus `partitioning.hpp`.

## Multi-GPU notes (cuDF ≥26.02)

Hard-won constraints for the multi-GPU C++ path (`cpp/tests/gpu/test_multi_gpu_*`,
WorkerPool, `hash_shuffle`, `gather_here`):

- Every cudf op on a worker pinned to GPU≠0 needs a **device-local stream** —
  `cudf::get_default_stream()` is device-0-bound ("invalid device ordinal" otherwise).
- A cudf/cuVS device object must be **destroyed on its owning device's worker thread** —
  hence the worker-per-GPU pool; release partitions/results on-worker before teardown.
- Per-device **RMM pools** (with persistent per-worker streams) are what make cheap
  queries scale; pool dealloc is stream-ordered, so an object outliving a transient
  stream frees on a dead stream and crashes.
- RMM `set_per_device_resource(g, nullptr)` resets the pointer map but NOT the ref map —
  teardown must also call `reset_per_device_resource_ref(g)`.
- Benchmarking multiple queries in one process is flaky at G≥2 (process-global cudf
  stream state across WorkerPool teardowns) — one query per process (see build-test.md).

## Cost model and the DuckDB oracle

- **Peacock cost:** `.cost.txt` goldens are derived purely from the `.cpu.txt` per-node
  tree text (`tests/common/cost_model.rs::cost_text_from_cpu`): each node's
  `output_bytes` is binned into a category and multiplied by that category's weight from
  **`testdata/cost_model.conf`** (runtime-editable; format `<category> <multiplier>
  [nodes…]`). Today all 10 real categories have multiplier 1.0 (total == Σ output_bytes);
  three placeholder phases (ram_to_vram, cuda_decompress, cuda_rle_decode) sit at 0.0.
- **DuckDB oracle:** `testdata/duckdb_cost.py` runs each query through DuckDB in two
  passes (deterministic profile with join-filter-pushdown off; a second pass extracting
  only dynamic-filter min/max bounds), combines them with parquet row-group stats, and
  emits `<q>.duckdb_cost.txt` = `materialization_total` (Σ pipeline-breaker materialized
  bytes) + `storage_read_total` (decoded Arrow bytes of surviving row-groups' referenced
  columns after static ∩ dynamic pruning — deliberately the same units as `GpuScanExec`
  output_bytes so the ratio is apples-to-apples).
- **Widget:** the cost report compares peacock Σout (from the `tp8-mini` `.cost.txt`
  footer) vs duckdb Σout; ratio ≤ 1.4 renders green. Directional signal only, not a
  benchmark. Per-query mode enablement comes from `testdata/cost-registry.csv` (the
  registry the inventory tests verify), tickets from `llm-wiki/tickets.md`.
