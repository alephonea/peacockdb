# peacockdb architecture

The code is authoritative. Where this page and the code disagree, the code is right and
this page is stale — fix the page (and say so) rather than the reading.

Pipeline: SQL → DataFusion logical/physical plan → `Gpu*Exec` wrapper annotation →
FlatBuffers plan (`flatbuffers/gpu_plan.fbs`) → C++/cuDF executor. The same flat buffers are
consumed by both engines (two-engine correctness: no CPU-only or GPU-only plan nodes).

## Contents

- [Execution modes](#execution-modes)
- [Optimization rules](#optimization-rules)
  - [Where batch_size is actually honoured](#where-batch_size-is-actually-honoured)
- [tp1 / tp8](#tp1--tp8)
- [Batches and partitions](#batches-and-partitions)
  - [The six execution modes](#the-six-execution-modes)
  - [Coalescing in the full-table CPU executor at tp8](#coalescing-in-the-full-table-cpu-executor-at-tp8)
- [Operators](#operators)
  - [What each operator is](#what-each-operator-is)
  - [CPU execution](#cpu-execution)
  - [GPU execution](#gpu-execution)
  - [From flat buffer to cuDF call](#from-flat-buffer-to-cudf-call)
  - [Pipeline breakers](#pipeline-breakers)
  - [Cross-partition operators](#cross-partition-operators)
    - [GpuCrossJoin vs GpuNestedLoopJoin](#gpucrossjoin-vs-gpunestedloopjoin)
- [Traits and interfaces](#traits-and-interfaces)
  - [Rust](#rust)
  - [C++](#c)
  - [The handle registry has no type](#the-handle-registry-has-no-type)
- [Backend-agnostic node-by-node driver](#backend-agnostic-node-by-node-driver)
- [CPU emulation and the two CPU behaviors](#cpu-emulation-and-the-two-cpu-behaviors)
- [Rehash and the comet hash](#rehash-and-the-comet-hash)
- [C++ executor layout](#c-executor-layout)
- [Column indexing](#column-indexing)
  - [What actually guards it](#what-actually-guards-it)
  - [What nothing guards](#what-nothing-guards)
- [cuDF options](#cudf-options)
  - [What the Rust side puts in the flat buffers](#what-the-rust-side-puts-in-the-flat-buffers)
  - [Join types and NULL key equality](#join-types-and-null-key-equality)
- [Multi-GPU notes (cuDF ≥26.02)](#multi-gpu-notes-cudf-2602)
- [Cost model and the DuckDB oracle](#cost-model-and-the-duckdb-oracle)

## Execution modes

Five executor classes in `peacockdb-core/src/executors/`, unified by two traits
(`executors/executor.rs`):

- [`Executor`](../peacockdb-core/src/executors/executor.rs#L18) —
  `execute(sql) -> Vec<RecordBatch>`.
- [`InstrumentedExecutor: Executor`](../peacockdb-core/src/executors/executor.rs#L25) —
  `execute_instrumented(sql)` additionally returns the plan and per-node
  [`NodeMemoryStats`](../peacockdb-core/src/executors/executor.rs#L57) (post-order, aligned
  with the plan tree).

| Class | Traits | Driver | Partition mode / tp |
|---|---|---|---|
| [`FullTableCpuExecutor`](../peacockdb-core/src/executors/full_table_cpu_executor.rs#L33) | both | streaming driver (its own) | SinglePartition; tp1 or tp8 hint |
| [`PartitionedCpuExecutor`](../peacockdb-core/src/executors/partitioned_cpu_executor.rs#L24) | both | node-by-node driver + `CpuNodeExecutor` | RealMultiPartition, tp8 |
| [`FullTableGpuExecutor`](../peacockdb-core/src/executors/full_table_gpu_executor.rs#L21) | both | node-by-node driver + `GpuNodeExecutor` | SinglePartition, tp1 |
| [`PartitionedGpuExecutor`](../peacockdb-core/src/executors/partitioned_gpu_executor.rs#L24) | both | node-by-node driver + `GpuNodeExecutor` | RealMultiPartition, tp8 |
| [`AllAtOnceGpuExecutor`](../peacockdb-core/src/executors/all_at_once_gpu_executor.rs#L183) | `Executor` only | single `peacock_execute` FFI call | SinglePartition, tp1; retiring (#110) |

**The flat buffers** are the serialized physical plan (`flatbuffers/gpu_plan.fbs`) — the form
both engines consume, and the only thing the C++ side ever sees of a query. Wherever this page
says "the flat buffers", "the wire format" or "serialized", that is what it means.

Three words, three layers, so "driver" means one thing on this page. An **executor** is a
mode class implementing `Executor` / `InstrumentedExecutor` — the five above. A **driver** is
the function that walks the plan and decides what runs when, and there are exactly three: the
**node-by-node driver** (`execute_node_by_node`), the **streaming driver** (`build_stream`,
full-table CPU only) and the **recursive C++ driver** (`execute_plan`, all-at-once). A
**backend** implements `NodeExecutor` and does the per-node work for the node-by-node driver:
`CpuNodeExecutor` or `GpuNodeExecutor`. A mode is an executor plus a driver plus, for the
node-by-node ones, a backend — so "the partitioned driver" would name no single thing and is
not used below.

Partition mode is implied by the class, never a parameter. The two GPU node-by-node
classes are thin config wrappers over the same backend — they differ only in the
constructed mode. Memory budget comes from `MemoryLimit` (micro/mini/standard/full,
`src/config.rs`); plan-level parallelism from `TargetPartitions` (Single/Multi).

## Optimization rules

peacockdb adds exactly two physical optimizer rules and no logical or analyzer rules;
everything else in the plan is DataFusion's own. Both are registered in
[`lib.rs`](../peacockdb-core/src/lib.rs#L49), in this order, and both run `transform_up`.

| Rule | Rewrites | Why |
|---|---|---|
| [`gpu_execution`](../peacockdb-core/src/gpu_rule.rs#L89) | every GPU-executable DataFusion node → its `Gpu*Exec` wrapper (16 kinds), lowering `InListExpr` to an OR-chain first | annotation, not transformation: the wrapper marks intent and forwards everything, so one plan feeds both engines. IN-lists are lowered here because the serializer has no encoding for them and [panics](../peacockdb-core/src/plan_serializer.rs#L268) if one arrives |
| [`gpu_memory_budget`](../peacockdb-core/src/gpu_rule.rs#L570) | three rewrites, below | turns a byte budget into plan shape: batch size, scan partitioning, and an explicit shuffle |

The budget rule computes one number first —
`batch_size = max(1, budget / analyze_memory(plan).subtree_max_row_bytes)`
([`analyze_memory`](../peacockdb-core/src/gpu_rule.rs#L293)) — then makes three rewrites
with it:

| Rewrite | From → to | Conditions |
|---|---|---|
| [scan](../peacockdb-core/src/gpu_rule.rs#L594) | `ParquetExec` → `GpuScanExec(batch_size)`, plus a row-group→partition map when the device is genuinely partitioned | the map comes from [`surviving_row_groups`](../peacockdb-core/src/gpu_rowgroup_prune.rs#L90) (else all groups) through [`build_scan_map`](../peacockdb-core/src/operators/scan.rs#L41); off the real-partitioning device the map is empty, keeping tp1 byte-identical |
| [re-batch](../peacockdb-core/src/gpu_rule.rs#L608) | `CoalesceBatchesExec(target)` → `CoalesceBatchesExec(batch_size)` | every such node, every device |
| [shuffle](../peacockdb-core/src/gpu_rule.rs#L627) | `GpuRepartition(Hash, M→N)` → `GpuCoalescePartitions(M→1)` + `GpuRepartition(Hash, 1→N)` | real-partitioning device, `Hash`, and M>1; RoundRobin and already-1→N are untouched |

The shuffle rewrite is the one with a cost consequence: it makes the concat a visible,
cost-accounted node instead of a hidden buffering step inside the repartition, and it hands
both node executors a single input partition to hash-scatter — see
[Rehash and the comet hash](#rehash-and-the-comet-hash).

### Where `batch_size` is actually honoured

It reaches execution through two channels, and **both are CPU-only**:

- **As a `TaskContext` override at the scan.** `GpuScanExec` is the only wrapper carrying
  `gpu_batch_size`, and [`strip_target`](../peacockdb-core/src/operators/mod.rs#L150) returns
  it only for that node. Both CPU modes apply it:
  [`build_stream`](../peacockdb-core/src/executors/full_table_cpu_executor.rs#L165) in
  full-table, and
  [`execute_single_node`](../peacockdb-core/src/executors/single_node.rs#L57) in the
  partitioned mode. The partitioned scan does not skip it: `cpu_scan_partitions` rebuilds a
  per-partition `ParquetExec` and
  [re-wraps it in `GpuScanExec` with the same `gpu_batch_size`](../peacockdb-core/src/executors/backend/cpu_node_executor.rs#L179)
  precisely so the override still applies and the node still reports as `ParquetExec`,
  matching the recursive oracle.
- **Baked into a plan node.** The re-batch rewrite writes it into every
  `CoalesceBatchesExec`, and DataFusion re-batches to that target wherever it executes that
  node — which, on the CPU, is both modes.

**The GPU ignores it in all three modes**, and each channel dies for its own reason. The
scan override has nothing to act on: cuDF reads by row group (`set_row_groups`), so there is
no batch to size — reads follow the row-group layout, and the budget lands as the
row-group→partition map instead. The re-batch node is
[`execute_passthrough`](../cpp/src/operators/dispatch.cpp#L66) on the GPU: the node returns
its child's table untouched, so the target it carries is never consulted.

Both numbers are nonetheless serialized into the flat buffers — `CudfScan.batch_size`
([fbs](../flatbuffers/gpu_plan.fbs#L327)) and `CudfCoalesceBatches.target_batch_size`
([fbs](../flatbuffers/gpu_plan.fbs#L470)) — and pinned byte-for-byte by
`plan_bytes.sha256`, while `grep -rn 'batch_size' cpp/src cpp/include` finds nothing. Two
fields of wire-format contract with no consumer on the far side (#132). So the honest
summary: `batch_size` is a CPU-side bound honoured identically by both CPU modes, and the
GPU's memory shape comes from the scan map alone.

## tp1 / tp8

`tp` is DataFusion's `target_partitions`, set per session from
[`TargetPartitions`](../peacockdb-core/src/config.rs#L21) — `Single` = 1, `Multi` =
[`TARGET_PARTITIONS`](../peacockdb-core/src/config.rs#L10) = 8. It is a planning input, not
an execution switch, and it is orthogonal to the partition *mode*: which executor runs is
decided by the mode, and `tp` decides the shape of the plan handed to it.

**What tp8 changes, all of it DataFusion's own doing.**
`RepartitionExec(RoundRobinBatch(8))` appears above the scan, hash `RepartitionExec`s appear
in front of joins and aggregates, aggregates split into Partial/Final phases, joins may
choose `Partitioned` mode over `CollectLeft`, and the plan closes with a
`CoalescePartitions` or `SortPreservingMerge`. peacockdb adds none of that — it wraps
whatever DataFusion produced.

Two axes are easy to conflate here, and only one of them is `tp`:

- **How many partitions the work is divided into** is `tp`. In this corpus DataFusion does
  NOT achieve that by splitting the parquet file: every one of the 561 round-robin nodes in
  the tp8 plan goldens reads `RoundRobinBatch(8), input_partitions=1`, and every
  `GpuScanExec` in a `full_table-tp8-mini` cost golden runs at `partitions=1`. The scan stays
  single-partition and the fan-out happens above it. (DataFusion can split a file into byte
  ranges for this — `cpu_scan_partitions` clears `file.range` in case it did — but nothing in
  the corpus takes that path.) The only multi-partition scans in the tree are ours: the
  `partitioned-tp8-standard` goldens show scans at 2–8 partitions, straight from the
  row-group map.
- **How many rows land in each emitted batch** is `batch_size`, within one partition's
  stream. That is the budget rule's lever, it is orthogonal to `tp`, and on the GPU it is
  ignored entirely (see above).

**What tp8 changes in peacockdb's own rules: nothing by itself.** The gate is
[`real_partitioning`](../peacockdb-core/src/gpu_rule.rs#L554) = `tp > 1` **and**
`PartitionMode::RealMultiPartition`, and the mode does not come from `tp` — it comes from the
execution mode (full-table ⇒ `SinglePartition`, partitioned ⇒ `RealMultiPartition`). So a
tp8 plan under `SinglePartition` — the `tp8-mini` device — gets the full tp8 shape with an
empty scan map and no shuffle lowering, which is what keeps its serialized bytes and
FlatBuffer round-trip identical to tp1's. Only tp8 + `RealMultiPartition` (`tp8-standard`)
gets the row-group→partition map and the `GpuCoalescePartitions` + `GpuRepartition(1→N)`
lowering.

`tp` also moves `batch_size` indirectly: the budget rule divides the budget by
`subtree_max_row_bytes`, which is a property of the plan, so the same query at tp1 and tp8
can be assigned different batch sizes.

**At execution, tp8 means different things per mode.** The streaming driver coalesces
every node's output to one stream, so a tp8 plan runs single-partition there — the repartition
and coalesce nodes execute, they just collapse. The CPU backend keeps N alive, which is
why it requires the map that only `RealMultiPartition` produces. On the GPU the same split
holds: `full_table_tp1_standard` runs one partition per node, `partitioned_tp8_standard` runs
eight.

That is why the corpus carries both. `tp8-mini` is the determinism device: the richer plan
shape, executed collapsed, so the goldens pin the shape without depending on partition
scheduling. `tp1-standard` exists where the tp8 result is not partition-invariant — a LIMIT
with no total order returns a different row set depending on which partition wins — and it is
also the tier the GPU full-table goldens share.

Every `.plan.txt` in the tree is tp8, so the tp1 plan's cost annotation is pinned nowhere —
[#133](tickets.md#t133).

## Batches and partitions

They are perpendicular, and the engine keeps them in different places.

A **partition** is a lane. It is a plan-level property — `Partitioning` inside
`PlanProperties`, read as `output_partitioning().partition_count()` — and at execution it is
either an independent stream (CPU) or an independent resident table (GPU). In peacockdb's
node-by-node model a partition is addressed by an opaque handle, which is why
[`NodeExecutor::execute_node`](../peacockdb-core/src/executors/node_by_node.rs#L28) takes
`&[Vec<u64>]` (a handle vector per child) and returns `Vec<u64>`: one node, N handles out.
C++ mirrors it exactly — `out_handles[p]` plus one `NodeStats` per partition — and the
goldens surface it as `partitions=N` with a `p{k}:` sub-line each, from
[`PartitionStat`](../peacockdb-core/src/executors/executor.rs#L46).

A **batch** is a horizontal slice of rows *inside* one lane: an Arrow `RecordBatch`. It is an
execution-level property with no presence in the plan tree, sized by the `batch_size` on the
`TaskContext` (which is what `GpuScanExec` overrides) and re-packed by
`CoalesceBatchesExec(target)`. Nothing counts batches in a golden; the only trace is
`NodeMemoryStats.max_batch_rows`, which no test asserts.

Concretely, batches come from row groups, not from partitions: the parquet reader decodes one
row group at a time, so a partition holding k row groups yields k batches whenever
`batch_size` exceeds the row-group row count — which it usually does (q6 at mini derives
≈25.5M rows against 122,880-row groups). The tp8 goldens show partitions holding anywhere
from 1 to 7 row groups, so one batch per partition is a special case, not the rule. From
there the boundaries propagate, because `execute_single_node` replays each input partition's
`Vec<RecordBatch>` through a stub stream — and `CoalesceBatchesExec` merges toward its target
rather than splitting: DataFusion's coalescer treats `target_batch_size` as a *minimum* and
slices only at a `fetch` boundary. None of it is pinned by a golden;
`max_batch_rows` is collected and asserted nowhere.

Four behaviours cover every operator. **1:1** — filter and project emit one batch per input
batch, and union and interleave forward theirs untouched. **Split** — only a hash repartition
truly takes one batch apart, into up to N (`BatchPartitioner` builds an index list per
partition and `take`s each; our `cpu_hash_repartition` does the same and emits exactly one
batch per non-empty partition). **Re-cut** — sort, both joins, aggregate and the k-way merge
all read `batch_size` from the context and emit on their own boundaries, so input boundaries
do not survive them. **Slice** — a `fetch` truncates at the limit. So of the five operators in
the *layout-changing* block only two touch batch boundaries at all: hash repartition splits and
the k-way merge re-cuts. `GpuCoalescePartitions` and `GpuUnion` forward batches unchanged — the
union changes only how many lanes there are, by summing its inputs' counts — and
`GpuCoalesceBatches` only merges: it has an optional `fetch` that would slice, but peacockdb
always builds it with `fetch: None`.

Two consequences of that asymmetry are worth holding onto. Partition count is decided before
the query runs and is visible in every golden; batch size is decided from the memory budget
and is deliberately *not* displayed, because it would make every golden budget-dependent. And
**batches exist only where a stream exists** — which on the GPU is nowhere.

> **Naming trap.** `ScanBatch` in the flat buffers and `GpuScanExec::batches_map()` are **partitions**,
> not Arrow batches: `build_scan_map` splits the surviving row groups into `n_parts`
> contiguous chunks and entry *i* becomes partition *i*. The word "batch" there predates the
> partition vocabulary. `CudfCoalesceBatches.target_batch_size` is the other sense — real Arrow
> batches — in the same buffers.

### The six execution modes

| Mode | Partitions | Batches |
|---|---|---|
| CPU full_table, tp1 | 1 everywhere: the plan has no repartition nodes | the real unit of work — scan emits `batch_size`-bounded batches, `CoalesceBatchesExec` re-packs |
| CPU full_table, tp8 | the plan declares up to 8, but no node ever *hands* more than one to its parent: each node's output is coalesced to a single stream as it executes (below) | same as tp1, plus the coalesce interleaving streams in completion order |
| CPU partitioned, tp8 | N live across nodes; a handle is a `Vec<RecordBatch>` in the backend's registry | batches live inside each partition; map ops run once per partition and the stats sum |
| GPU all-at-once | none — one FFI call for the whole plan, one `cudf::table` out | none inside; the root is exported to Arrow IPC at the boundary |
| GPU full_table | 1 per node, one resident `cudf::table` per node in the `NodeSession` registry | none inside — a node is a whole table, not a stream |
| GPU partitioned | N per node, N resident tables, one handle each | none inside; at the root [`materialize`](../peacockdb-core/src/executors/backend/gpu_node_executor.rs#L148) walks the handles in order and concatenates the batches each IPC export yields |

So batches are a CPU-only concept in this engine, and the GPU modes differ from each other
only in partition count. That is also why `batch_size` cannot bound GPU memory: there is no
batch to bound, which is the substance of [#132](tickets.md#t132).

### Coalescing in the full-table CPU executor at tp8

The streaming driver's defining behaviour is one line in
[`build_stream`](../peacockdb-core/src/executors/full_table_cpu_executor.rs#L165):

```rust
let inner = execute_stream(node, task_ctx)?;
```

DataFusion's `execute_stream` runs `plan.execute(0, ctx)` when the node declares one output
partition and, when it declares two or more, wraps it in a `CoalescePartitionsExec` and
executes that. So the collapse is not in the plan and not in any golden — it is inserted at
execution time, per node, by the choice of entry point.

Each node is then rebuilt with its children replaced by
[`StreamSourceExec`](../peacockdb-core/src/executors/stream.rs#L119) stubs, which declare
`UnknownPartitioning(1)`. So a tp8 plan executes as a chain of single-input nodes, each
fanning out internally and collapsing again before its parent sees anything. A
`RepartitionExec(RoundRobinBatch(8))` over a one-partition child really does scatter batches
into eight channels — and the streaming driver merges them straight back. That work is pure overhead,
and it is on purpose: it keeps the executed node set identical to the tp8 plan the goldens
pin, so the same plan can back the `full_table-tp8-mini` and `partitioned-tp8-standard`
goldens.

What the collapse costs is order, not totals. `CoalescePartitionsExec` documents that no
guarantee is made about the order of the resulting partition, and it clears the input's
orderings. Row counts and byte totals are partition-invariant, so `.cpu.txt` and `.cost.txt`
stay stable — and the golden records `partitions=1` with no sub-lines, because the streaming
driver never sees more than one. Row *order* is not stable, which is why `.result.txt` comparison
renders rows sorted (`batches_to_sorted_str`) and why a LIMIT with no total order has to be
canonized at tp1: sorting cannot repair a top-N that picked different rows.

## Operators

Every GPU-executable DataFusion node is wrapped in a passthrough `Gpu*Exec` (16 types,
grouped by family under `peacockdb-core/src/operators/`). Fifteen are generated by the
[`gpu_exec_node!`](../peacockdb-core/src/operators/mod.rs#L26) macro — the wrapper carries
its inner node, forwards schema/properties/children, and delegates
[`execute`](../peacockdb-core/src/operators/mod.rs#L76) to it — and `GpuScanExec` is
hand-written because it also carries the memory-budget batch size and the explicit
row-group→partition map ([`build_scan_map`](../peacockdb-core/src/operators/scan.rs#L41):
survivor row-groups split into N contiguous chunks; an empty map means legacy
single-partition, which keeps tp1 byte-identical). The [`Operator`](../peacockdb-core/src/operators/operator.rs#L28) trait unifies them:

- `inner()` — the wrapped DataFusion node;
- [`partition_topology()`](../peacockdb-core/src/operators/operator.rs#L12) — how partition
  handles flow: `ScanEmit` (scan emits N from its row-group map), `Map` (1:1 per partition),
  `Collapse` (M→1 concat), `KWayMerge` (M→1 order-preserving), `RepartitionHash` (1→N),
  `Join` (two inputs → one). Declarative only: nothing reads it today, and the CPU backend and
  the C++ side each re-derive the same topology themselves (#130);
- `strips_to_inner()` — whether CPU emulation replaces the wrapper with its inner node
  (CPU only — see below).

[`as_operator()`](../peacockdb-core/src/operators/mod.rs#L114) is the single downcast
registry; adding an operator is one line there. Its only caller today is
[`strip_target`](../peacockdb-core/src/operators/mod.rs#L150).

Per-operator FlatBuffers serialize/deserialize pairs are co-located in the family files.
**Statement order is the wire format**: FlatBufferBuilder is a no-interning bump arena, so
reordering writes changes bytes even with identical values.
[`tests/test_plan_bytes.rs`](../peacockdb-core/tests/test_plan_bytes.rs) +
`goldens/plan_bytes.sha256` pin the exact bytes per query; regenerating that golden to
silence a red defeats its purpose.

### What each operator is

16 wrappers, 15 flat-buffer node kinds: `GpuInterleaveExec` shares `GpuUnion`'s wire form (both emit
`PlanNodeKind::GpuUnion`), and `GpuGlobalLimitExec` serializes as `GpuLimit`.

| Operator | Plan node | Semantics |
|---|---|---|
| [`GpuScanExec`](../peacockdb-core/src/operators/scan.rs#L62) | `ParquetExec` | reads the survivor row groups of one parquet table; owns the row-group→partition map and the batch-size override |
| [`GpuFilterExec`](../peacockdb-core/src/operators/filter.rs#L21) | `FilterExec` | row-wise predicate; IN-lists are lowered before wrapping |
| [`GpuProjectExec`](../peacockdb-core/src/operators/project.rs#L21) | `ProjectionExec` | expression list, rename, column pruning |
| [`GpuAggregateExec`](../peacockdb-core/src/operators/aggregate.rs#L21) | `AggregateExec` | grouped and global aggregates, in DataFusion's Partial and Final phases |
| [`GpuHashJoinExec`](../peacockdb-core/src/operators/join.rs#L21) | `HashJoinExec` | equi-join on key pairs, optional residual filter, all join types |
| [`GpuCrossJoinExec`](../peacockdb-core/src/operators/join.rs#L42) | `CrossJoinExec` | cartesian product |
| [`GpuNestedLoopJoinExec`](../peacockdb-core/src/operators/join.rs#L45) | `NestedLoopJoinExec` | non-equi join driven by a filter expression |
| [`GpuSortExec`](../peacockdb-core/src/operators/sort.rs#L22) | `SortExec` | sort by keys, optional `fetch` (top-N) |
| [`GpuInterleaveExec`](../peacockdb-core/src/operators/union.rs#L24) | `InterleaveExec` | partition-wise interleave of identically hash-partitioned inputs |
| [`GpuGlobalLimitExec`](../peacockdb-core/src/operators/limit.rs#L21) | `GlobalLimitExec` | global `skip`/`fetch`; DataFusion requires a single-partition input |
| [`GpuWindowExec`](../peacockdb-core/src/operators/window.rs#L21) | `WindowAggExec`, `BoundedWindowAggExec` | window functions; the input is hash-partitioned on the PARTITION BY keys |
| *layout-changing* | | |
| [`GpuCoalesceBatchesExec`](../peacockdb-core/src/operators/coalesce.rs#L20) | `CoalesceBatchesExec` | batch-size normalization; no data change |
| [`GpuCoalescePartitionsExec`](../peacockdb-core/src/operators/coalesce.rs#L26) | `CoalescePartitionsExec` | M→1 concat; in tp8 it is the explicit concat in front of a hash repartition |
| [`GpuRepartitionExec`](../peacockdb-core/src/operators/repartition.rs#L21) | `RepartitionExec` | hash or round-robin repartition; the Hash case scatters 1→N on Spark murmur3 (seed 42) |
| [`GpuSortPreservingMergeExec`](../peacockdb-core/src/operators/sort.rs#L34) | `SortPreservingMergeExec` | order-preserving M→1 merge of sorted inputs, optional `fetch` |
| [`GpuUnionExec`](../peacockdb-core/src/operators/union.rs#L21) | `UnionExec` | concatenate branches, normalizing each branch's decimal scale to the declared output schema |

### CPU execution

**Stripping is a CPU-emulation concept only.** Both CPU modes strip — full-table through
[`build_stream`][ft] and partitioned through [`execute_single_node`][sn], which share
[`strip_gpu`][sg] — while the GPU path never strips, because it serializes the wrapper tree
to the flat buffers and the C++ side dispatches on node kind. The strip set is asymmetric **and
load-bearing**: 11 operators strip, 5 do not (cross join, nested-loop join, union, global
limit, window). Flipping one does not change the `.cpu.txt` golden — those node names are
rendered from the plan tree, so they are always `Gpu*Exec` — it changes
`NodeMemoryStats.node_name`, and that name is what
[`resident.rs`](../peacockdb-core/src/resident.rs#L71) classifies on.

Both full-table columns run the same code, and it is not per-operator: [`build_stream`][ft]
walks the tree, [`strip_target`][st] replaces each wrapper with its inner DataFusion node
(or keeps it, for the 5 non-strippers), and `execute_stream` coalesces every node's output
to one stream. tp1 and tp8 differ in the plan, not the path — a tp8 plan carries repartition
and coalesce nodes, and each is collapsed N→1 as it is reached.

A non-stripper — cross join, nested-loop join, union, global limit, window — executes the
same arithmetic anyway: the wrapper stays in the tree and its
[`execute`](../peacockdb-core/src/operators/mod.rs#L76) forwards straight to the inner
DataFusion node, so DataFusion still does the work. The only thing that differs is the name
the node reports, and that name is not cosmetic: `resident.rs` decides how a node
contributes to the modeled resident set by matching unwrapped names, so a
`GpuCrossJoinExec` misses the join arm that a stripped `HashJoinExec` hits and is modeled as
streaming (#131).

The CPU backend is the one that does branch per operator, which is why every cell in
that column names different code. [`CpuNodeExecutor::execute_node`][pn] keeps N handles alive
across nodes and tests the node against one case after another — scan with a row-group map,
hash repartition, ordinary single-child node, co-partitioned join, SortPreservingMerge — and
if none matches, concatenates each child's partitions into one input and runs the node once.
A cell below names the case that node lands in.

| Operator | full_table tp1 | full_table tp8 | partitioned tp8 |
|---|---|---|---|
| GpuScan | [inner + batch size][st] | [inner + batch size][st] | [replay the RG map → N][scanp] |
| GpuFilter | [inner][ft] | [inner][ft] | [per-partition map][map] |
| GpuProject | [inner][ft] | [inner][ft] | [per-partition map][map] |
| GpuAggregate | [inner][ft] | [inner][ft] | [per-partition map][map] |
| GpuHashJoin | [inner][ft] | [inner][ft] | [per-bucket join][joinp], else [concat][cat] |
| GpuCrossJoin | [wrapper kept][ft] | [wrapper kept][ft] | [concat inputs][cat] |
| GpuNestedLoopJoin | [wrapper kept][ft] | [wrapper kept][ft] | [concat inputs][cat] |
| GpuSort | [inner][ft] | [inner][ft] | [per-partition map][map] |
| GpuInterleave | [substituted by `UnionExec`][il] | [substituted by `UnionExec`][il] | [concat inputs][cat] |
| GpuGlobalLimit | [wrapper kept][ft] | [wrapper kept][ft] | [per-partition map][map] |
| GpuWindow | [wrapper kept][ft] | [wrapper kept][ft] | [per-partition map][map] |
| *layout-changing* | | | |
| GpuCoalesceBatches | [inner][ft] | [inner][ft] | [per-partition map][map] |
| GpuCoalescePartitions | [inner][ft] | [inner, N→1][ft] | [concat M→1][cat] |
| GpuRepartition | [inner (ahash)][ft] | [inner (ahash), then collapsed][ft] | [comet murmur3 1→N][hashp] |
| GpuSortPreservingMerge | [inner][ft] | [inner, N→1][ft] | [real k-way merge][spmp] |
| GpuUnion | [wrapper kept][ft] | [wrapper kept][ft] | [concat inputs][cat] |

The `GpuGlobalLimit` and `GpuWindow` rows are safe only because of where the planner puts
them. Both are `Map` topology, so the CPU backend applies them per partition, which
for a *global* limit would be wrong on a multi-partition input — it holds because DataFusion
requires a single-partition input for `GlobalLimitExec`, and the tp8 goldens show it sitting
directly on a `GpuCoalescePartitionsExec`. A per-partition window is correct for the same
kind of reason: its input is hash-partitioned on the PARTITION BY keys, so each partition
holds whole groups. GPU windows at tp8 are still blocked (#32), so no test exercises that
half today.

`GpuRepartition` in full-table mode strips to DataFusion's own `RepartitionExec`, which
hashes with ahash rather than comet murmur3. That does not change results, because the
full-table path coalesces every node back to one stream — but it does mean row→partition
placement matches the GPU only in the partitioned mode, which is why the conformance gate
lives there.

### GPU execution

All three GPU modes reach the same per-operator function in `cpp/src/operators/`, through
[`run_op`](../cpp/src/operators/dispatch.cpp#L40). What differs is who supplies the inputs:
all-at-once recurses from the root ([`execute_node`][rec], retiring with #110), while both
node-by-node modes hand one node its children's handles ([`execute_one`][one]) under
[`NodeSession::execute_node`][ns].

The five operators in the last block of each table — coalesce-batches, coalesce-partitions,
repartition, sort-preserving-merge and union — change the layout rows sit in rather than the
rows themselves. Four of them `run_op` does not implement at all: it hands coalesce-batches,
coalesce-partitions, repartition and sort-preserving-merge to
[`execute_passthrough`][pass], which returns the child's table unchanged, and NodeSession owns
whatever real work they do at tp8 because only NodeSession holds the partition registry.
Coalesce-batches never has any: it is passthrough in all three GPU modes, because batching is a
DataFusion streaming concern and a GPU node is one materialized `cudf::table` — the GPU backend
hands the same N tables straight back.

Union is the exception inside that block. It has a real kernel ([`execute_union`][eun]) because
concatenating branches and normalizing each one's decimal scale is work on rows; what puts it
in the block is the layout half — its output lane count is the sum of its inputs', so it
concatenates lanes rather than mapping them. `GpuInterleave` looks like it belongs here and does
not: `can_interleave` requires every input to carry the same `Partitioning::Hash`, and output
lane p is built from lane p of each input, so no row changes lane and no lane is added — it
merges the branch axis only, which is genuine `Map` topology. On the GPU it is nonetheless the
same flat-buffer node kind as union and runs the same `execute_union`.

| Operator | all-at-once | full_table (1 partition) | partitioned (N partitions) |
|---|---|---|---|
| GpuScan | [`execute_scan`][escan] | [`execute_scan`][escan] | [N reads off the map][nscan] |
| GpuFilter | [`execute_filter`][efil] | [`execute_filter`][efil] | [per-partition][nmap] |
| GpuProject | [`execute_project`][eproj] | [`execute_project`][eproj] | [per-partition][nmap] |
| GpuAggregate | [`execute_aggregate`][eagg] | [`execute_aggregate`][eagg] | [per-partition][nmap] |
| GpuHashJoin | [`execute_hash_join`][ehj] | [`execute_hash_join`][ehj] | [child p ⋈ child p][nmap]; unequal N throws |
| GpuCrossJoin | [`execute_cross_join`][ecj] | [`execute_cross_join`][ecj] | [unequal N throws][nmap] (#97) |
| GpuNestedLoopJoin | [`execute_nested_loop_join`][enlj] | [`execute_nested_loop_join`][enlj] | [unequal N throws][nmap] (#97) |
| GpuSort | [`execute_sort`][esort] | [`execute_sort`][esort] | [per-partition][nmap] |
| GpuInterleave | [`execute_union`][eun] | [`execute_union`][eun] | [per-partition][nmap] |
| GpuGlobalLimit | [`execute_limit`][elim] | [`execute_limit`][elim] | [per-partition][nmap] |
| GpuWindow | [`execute_window`][ewin] | [`execute_window`][ewin] | [per-partition][nmap] |
| *layout-changing* | | | |
| GpuCoalesceBatches | [passthrough][pass] | [passthrough][pass] | [passthrough, once per partition][nmap] |
| GpuCoalescePartitions | [passthrough][pass] | [passthrough][pass] | [concat M→1][ncol] |
| GpuRepartition | [passthrough][pass] | [passthrough][pass] | [`spark_hash_partition` 1→N][nrep] |
| GpuSortPreservingMerge | [passthrough][pass] | [passthrough][pass] | [`cudf::merge` k-way + fetch][nmerge] |
| GpuUnion | [`execute_union`][eun] | [`execute_union`][eun] | [per-partition][nmap] |

The generic map arm pairs child `c` partition `p` with every other child's partition `p`,
and requires all children to carry the same count — a mismatch is a loud throw
("multi-partition joins are not implemented yet"), not a diagonal answer. That is what
makes the cross-join and nested-loop rows throw rather than silently return a partial
cartesian: DataFusion collects their left input to one partition while the right stays at
N, so the counts disagree by construction. The CPU backend has no such gap — a
multi-child node falls through to [concat-into-one][cat] and answers correctly, so at tp8
these two operators are a GPU-only hole, tracked in #97.

`coalesce.cpp` and `repartition.cpp` exist and are deliberately empty: those two operators
need the session's partition registry, so their bodies live in `node_session.cpp`, and the
files stay so the operator set is discoverable by name.

[ft]: ../peacockdb-core/src/executors/full_table_cpu_executor.rs#L158
[sn]: ../peacockdb-core/src/executors/single_node.rs#L52
[sg]: ../peacockdb-core/src/executors/single_node.rs#L24
[st]: ../peacockdb-core/src/operators/mod.rs#L150
[il]: ../peacockdb-core/src/executors/full_table_cpu_executor.rs#L206
[pn]: ../peacockdb-core/src/executors/backend/cpu_node_executor.rs#L300
[scanp]: ../peacockdb-core/src/executors/backend/cpu_node_executor.rs#L124
[hashp]: ../peacockdb-core/src/executors/backend/cpu_node_executor.rs#L197
[map]: ../peacockdb-core/src/executors/backend/cpu_node_executor.rs#L340
[joinp]: ../peacockdb-core/src/executors/backend/cpu_node_executor.rs#L373
[spmp]: ../peacockdb-core/src/executors/backend/cpu_node_executor.rs#L404
[cat]: ../peacockdb-core/src/executors/backend/cpu_node_executor.rs#L430
[rec]: ../cpp/src/operators/dispatch.cpp#L100
[one]: ../cpp/src/operators/dispatch.cpp#L110
[ns]: ../cpp/src/node_session.cpp#L100
[pass]: ../cpp/src/peacock/operators.h#L57
[nscan]: ../cpp/src/node_session.cpp#L123
[ncol]: ../cpp/src/node_session.cpp#L150
[nmerge]: ../cpp/src/node_session.cpp#L173
[nrep]: ../cpp/src/node_session.cpp#L225
[nmap]: ../cpp/src/node_session.cpp#L289
[escan]: ../cpp/src/operators/scan.cpp#L18
[efil]: ../cpp/src/operators/filter.cpp#L16
[eproj]: ../cpp/src/operators/project.cpp#L17
[eagg]: ../cpp/src/operators/aggregate.cpp#L128
[ehj]: ../cpp/src/operators/join.cpp#L42
[ecj]: ../cpp/src/operators/join.cpp#L390
[enlj]: ../cpp/src/operators/join.cpp#L404
[esort]: ../cpp/src/operators/sort.cpp#L17
[eun]: ../cpp/src/operators/union.cpp#L16
[elim]: ../cpp/src/operators/limit.cpp#L15
[ewin]: ../cpp/src/operators/window.cpp#L35

### From flat buffer to cuDF call

One row per wire node kind: what the plan hands the C++ side, and the cuDF it turns into.

**Two vocabularies, and the prefix is the tell.** `Cudf*` is a flat-buffer node table — the
wire form, and what the C++ side dispatches on; the root table wrapping them is still
`GpuPlan`, since it is not a node kind. `Gpu*Exec` is the DataFusion wrapper, and prose
that drops the `Exec` suffix still means the wrapper. They were the same word until T1
renamed the wire half, which is why a comment naming `GpuRepartition` beside a table row
naming `CudfRepartition` is two different things and not a typo.
The middle column is the fields that change what the call does — `input` / `left` / `right`
are the tree and are not repeated, and a field nothing reads is called out, because a wire
field with no consumer reads as a knob (#132). Line links are to the deciding call, not to
the whole function.

| Node | What steers it | The cuDF it becomes |
|---|---|---|
| [`CudfScan`](../flatbuffers/gpu_plan.fbs#L317) | `file_paths`, `projection`, `row_groups` (pruning survivors) or `batches[p]` (this partition's slice of them), or a list the call supplies instead of either (`execute_scan_rowgroups`, which is how one node loads a batch at a time), `limit`; `batch_size` **is read by nobody** (#132) | [`scan.cpp#L83`](../cpp/src/operators/scan.cpp#L83) — `cudf::io::read_parquet(opts)`, with `.columns(projected)`, `set_row_groups(...)` and `set_num_rows(limit)` set on `opts` first |
| [`CudfFilter`](../flatbuffers/gpu_plan.fbs#L346) | `predicate`, `projection` | [`filter.cpp#L25`](../cpp/src/operators/filter.cpp#L25) — `cudf::compute_column(tv, predicate)` for the mask, then `cudf::apply_boolean_mask(tv, mask->view())` |
| [`CudfProject`](../flatbuffers/gpu_plan.fbs#L358) | `exprs`, `aliases` | [`project.cpp#L49`](../cpp/src/operators/project.cpp#L49) — `cudf::compute_column(tv, ast)` per AST-able expr; a bare `ColumnRef` is a column copy, and LIKE/CASE/scalar functions take `build_column` instead |
| [`CudfAggregate`](../flatbuffers/gpu_plan.fbs#L375) | `mode` (Partial/Final/FinalPartitioned/Single/SinglePartitioned/Merge), `group_exprs`, `aggr_funcs` (each with its out decimal scale and `distinct`), `grouping_sets`, `mergeable_agg_state`, `aggr_input_schema` | [`aggregate.cpp#L666`](../cpp/src/operators/aggregate.cpp#L666) — `gb.aggregate(requests)` over [`groupby{keys, null_policy::INCLUDE}`](../cpp/src/operators/aggregate.cpp#L435); with no group keys it is [`cudf::reduce`](../cpp/src/operators/aggregate.cpp#L258) to one row |
| [`CudfHashJoin`](../flatbuffers/gpu_plan.fbs#L412) | `join_type`, `keys`, `filter` + `filter_columns` (residual), `null_equals_null`, `projection` | [`join.cpp#L290`](../cpp/src/operators/join.cpp#L290) — `cudf::inner_join` / `left_join` / `full_join(left_keys, right_keys, kJoinNulls)`; semi/anti take [`left_semi_join` / `left_anti_join`](../cpp/src/operators/join.cpp#L126), or their `mixed_*` forms when a residual filter must be evaluated during the join |
| [`CudfCrossJoin`](../flatbuffers/gpu_plan.fbs#L434) | nothing — the node is its two inputs | [`join.cpp#L394`](../cpp/src/operators/join.cpp#L394) — `cudf::cross_join(ltv, rtv)` |
| [`CudfNestedLoopJoin`](../flatbuffers/gpu_plan.fbs#L442) | `join_type`, `filter` + `filter_columns`, `projection` | [`join.cpp#L432`](../cpp/src/operators/join.cpp#L432) — `cudf::cross_join`, then [`apply_boolean_mask`](../cpp/src/operators/join.cpp#L478) over the filter evaluated on the crossed table |
| [`CudfSort`](../flatbuffers/gpu_plan.fbs#L455) | `exprs` (`asc`, `nulls_first` per key), `fetch`, `preserve_partitioning` | [`sort.cpp#L50`](../cpp/src/operators/sort.cpp#L50) — `cudf::sorted_order(keys, orders, null_orders)` then `cudf::gather`, and [`cudf::slice`](../cpp/src/operators/sort.cpp#L58) when `fetch` makes it a top-N |
| [`CudfCoalesceBatches`](../flatbuffers/gpu_plan.fbs#L469) | `target_batch_size` — **read by nobody** (#132) | [`dispatch.cpp#L66`](../cpp/src/operators/dispatch.cpp#L66) — `execute_passthrough`: the child's table, untouched. A GPU node is one materialized table, so there is no batching to do |
| [`CudfCoalescePartitions`](../flatbuffers/gpu_plan.fbs#L475) | nothing | [`node_session.cpp#L298`](../cpp/src/node_session.cpp#L298) — `cudf::concatenate(views)` over the input partitions; passthrough in the two single-partition modes, which have nothing to collapse |
| [`CudfRepartition`](../flatbuffers/gpu_plan.fbs#L486) | `kind`, `num_partitions`, `hash_exprs` (key ordinals) | [`node_session.cpp#L371`](../cpp/src/node_session.cpp#L371) — `spark_hash_partition(tv, key_cols, n)`, ours rather than cuDF's murmur3, then [`cudf::slice`](../cpp/src/node_session.cpp#L384) per partition into an owning table |
| [`CudfSortPreservingMerge`](../flatbuffers/gpu_plan.fbs#L495) | `exprs`, `fetch` | [`node_session.cpp#L286`](../cpp/src/node_session.cpp#L286) — `cudf::merge(views, key_cols, orders, null_orders)`, k-way and order-preserving; a concat fallback with no keys or one input (#118) |
| [`CudfUnion`](../flatbuffers/gpu_plan.fbs#L508) | `inputs`, `interleave`, `output_schema` | [`union.cpp#L62`](../cpp/src/operators/union.cpp#L62) — `cudf::concatenate(views)`, after [`cudf::cast`](../cpp/src/operators/union.cpp#L51) retypes each branch column to the declared output type (#41) |
| [`CudfLimit`](../flatbuffers/gpu_plan.fbs#L526) | `skip`, `fetch` | [`limit.cpp#L31`](../cpp/src/operators/limit.cpp#L31) — `cudf::slice(tv, {skip, end})`, and the whole table returned untouched when the range covers it |
| [`CudfWindow`](../flatbuffers/gpu_plan.fbs#L573) | `window_exprs` (partition keys, order keys, frame bounds, out decimal scale) | [`window.cpp#L106`](../cpp/src/operators/window.cpp#L106) — `cudf::grouped_rolling_window(keys, arg, preceding, following, min_periods, agg)`, which preserves input row order |

Two things recur. **Four nodes reach no kernel at all** — coalesce-batches always, and
coalesce-partitions, repartition and sort-preserving-merge in the modes that have one
partition — because what they change is the layout rows sit in, and a single resident
table has no layout to change.

And **three nodes need more than one call**, because cuDF has no fused form: filter
computes a mask and then applies it; sort takes `sorted_order` then `gather`, and a third
call to `slice` when a `fetch` makes it a top-N; union casts each branch column whose type
differs from the declared output, then concatenates once. Each intermediate in those
sequences exists because the pair could not be one call.

The nested-loop join is the one to read separately rather than filing beside filter. It
materialises the **full cartesian product** first and only then evaluates its predicate
over it — cross join, build the mask on the crossed table, apply it. That is three calls
whose first is the expensive one, and it is why the operator is a GPU-only hole at tp8
(#97) and why broadcast joins (#140) would change the shape rather than the constant.

### Pipeline breakers

Two independent axes, and yes — operators break one without the other in both directions.
Breaking the **partition** pipeline means downstream can no longer work lane-by-lane: the
operator changes the lane structure, so it needs every input lane before its own output is
complete. Breaking the **batch** pipeline means it must hold rows back: it cannot emit until it
has consumed its input (or a whole side of it).

The columns below describe DataFusion's streaming semantics, which is what the streaming
driver actually executes. In the three node-by-node modes the question is moot: a handle holds
a complete `Vec<RecordBatch>` or `cudf::table`, so *every* node is a materialization point
there. peacockdb's own model of this axis is
[`resident.rs::peak`](../peacockdb-core/src/resident.rs#L71), which stacks join build sides and
treats sort / aggregate / coalesce-partitions as buffering.

| Operator | Breaks partitions? | Breaks batches? |
|---|---|---|
| GpuScan | no — it is the source; emits 1, or N per its row-group map | no — decodes row group by row group |
| GpuFilter | no | no — one batch out per batch in |
| GpuProject | no | no |
| GpuAggregate | no — per-lane | **yes** — the grouped hash table consumes the whole input before emitting |
| GpuHashJoin | no — lanes preserved (co-partitioned at tp8) | **build side only** — the left is collected in full, the right streams batch by batch |
| GpuCrossJoin | no by itself, but DataFusion requires its left input at SinglePartition, so a collapse appears below it | **build side only** — left collected, right streamed |
| GpuNestedLoopJoin | as cross join | **build side only** — left collected, right streamed |
| GpuSort | no — sorts within a lane | **yes** — needs every row before the first one is ordered |
| GpuWindow | no — per-lane; input is hash-partitioned on the PARTITION BY keys | **yes** for `WindowAggExec` (`concat_batches` over the buffer); `BoundedWindowAggExec` streams a bounded frame |
| GpuGlobalLimit | no, but it requires SinglePartition input | no — streams and stops early, slicing at the boundary |
| GpuInterleave | no — same `Partitioning::Hash` in, output lane p from lane p of each input | no — forwards untouched |
| *layout-changing* | | |
| GpuCoalesceBatches | no | no — merges to a *minimum* target; a bounded buffer, not a breaker |
| GpuCoalescePartitions | **yes** — M→1; cannot finish until every lane has | no in DataFusion (`RecordBatchReceiverStream` forwards on arrival) — but the CPU backend concatenates, and `resident.rs` models it as buffering at tp>1 |
| GpuRepartition | **yes** — 1→N hash scatter, or 1→N round-robin | no in DataFusion (splits each batch on arrival); our `cpu_hash_repartition` concatenates first, so in the CPU backend it is |
| GpuSortPreservingMerge | **yes** — M→1 | no — k-way merge holds one row per stream and a builder, then re-cuts to `batch_size` |
| GpuUnion | **no**, though it changes the lane count: output lanes = Σ of the inputs', and output lane k is served by exactly one input lane | no — forwards that lane's batches untouched |

`GpuUnion` is the row that most invites the wrong answer, and the answer is no.
`UnionExec::execute(k)` walks its inputs subtracting partition counts until `k` falls inside
one of them, then returns that input's lane directly — so output lane k *is* input lane j of
child i, streaming independently, with nothing collected and no lane waiting on another. Union
renumbers lanes; it does not move a row between them. That is exactly why the block it sits in
is called *layout-changing* rather than pipeline-breaking: changing the lane count and coupling
the lanes are different things, and only `GpuCoalescePartitions`, `GpuRepartition` and
`GpuSortPreservingMerge` do the second. (Our CPU backend is stricter than DataFusion here too:
a multi-child node falls through to concat-into-one, so at tp8 a union does collapse to one
lane there.)

All three joins buffer one side and one side only. DataFusion collects the **left** —
`collect_left_input` behind a `OnceAsync`, with `Distribution::SinglePartition` required there
— and executes the **right** per partition, polling it batch by batch. So nothing on the right
is ever resident, which is also what `resident.rs` models: it stacks `children.first()`'s
output bytes and nothing of the second child. The one asymmetry worth knowing is timing rather
than memory: an outer join keeps a bitmap of matched left rows and cannot emit its unmatched
ones until the right stream ends, so the *last* output waits on the probe side even though the
probe side is never buffered.

So the two directions both occur, and each has a clean example. **Batches but not partitions:**
sort, aggregate, window, and the build side of every join — they buffer within a lane and leave
the lane structure alone. **Partitions but not batches:** coalesce-partitions,
sort-preserving-merge and repartition — they couple lanes while batches keep flowing through.
Union looks like it belongs in that list and does not, for the reason below. The only operator
that would break both axes at once is a sort feeding a collapse, which is two nodes.

One consequence worth stating: `GpuUnionExec::partition_topology()` reports `Map`, and by
DataFusion's semantics it is not — union sums its inputs' partition counts. Nothing reads that
method (#130), and the CPU backend sidesteps it by concatenating a multi-child
node's inputs into one lane, but the declaration is wrong where it is read by a human.

### Cross-partition operators

Some operators cannot answer from one lane. A grouped aggregate needs every row of a group,
and an equi-join needs both sides of a key — neither respects the row-group split the scan
partitioned on. The fix is always the same: redistribute rows by the key, which is what the
`GpuCoalescePartitions` + `GpuRepartition(Hash, 1→N)` pair does after the budget rule's
shuffle lowering. Every snippet below comes from a committed `partitioned-tp8-standard` cost golden — elided
and re-wrapped to fit, but every number is verbatim, so this is what the executor produced.

**A grouped aggregate — `shuffle_additive`, one shuffle around it.** The Partial phase runs
per lane over the scan's partitions and collapses 6M rows to 14; the shuffle then re-lands
those 14 rows by group key, and the Final phase runs per lane again:

```
GpuRepartitionExec: partitioning=Hash([l_returnflag@0, l_linestatus@1], 8), input_partitions=1, partitions=8
  p0: in_rows=4 out_rows=4    p1: in_rows=5 out_rows=5    p2: in_rows=0 out_rows=0
  p3: in_rows=4 out_rows=4    p4: in_rows=0 out_rows=0    p5: in_rows=0 out_rows=0
  p6: in_rows=0 out_rows=0    p7: in_rows=1 out_rows=1
  GpuCoalescePartitionsExec, partitions=1, output_rows=14
    GpuAggregateExec: group_by=[l_returnflag, l_linestatus], …, partitions=8
      p0: in_rows=860160 out_rows=2    p1: in_rows=737280 out_rows=2
      …
      p7: in_rows=717375 out_rows=1
      GpuScanExec: table=lineitem, …, partitions=8
        p0: row_groups=[0, 1, 2, 3, 4, 5, 6] out_rows=860160
```

Four of the eight post-shuffle lanes are empty, because TPC-H has four `(returnflag,
linestatus)` groups and murmur3 puts them where it puts them. That skew is the honest cost of
hash partitioning at small cardinality, and it is visible in the golden rather than hidden.

**A partitioned hash join — `q3`.** Each side is coalesced and hash-repartitioned on its join
key, so bucket p of the left holds exactly the rows that can match bucket p of the right, and
the join then runs eight independent times (`child0[p] ⋈ child1[p]`):

```
GpuHashJoinExec: join_type=Inner, on=[(c_custkey@0, o_custkey@1)], partitions=8
  p0: in_rows=3829 out_rows=18413    p1: in_rows=3724 out_rows=18348
  p2: in_rows=3828 out_rows=18764    …    p7: in_rows=3756 out_rows=18451
  GpuRepartitionExec: partitioning=Hash([c_custkey@0], 8), input_partitions=1, partitions=8
    GpuCoalescePartitionsExec, partitions=1, output_rows=30142        <- build side
      …
  GpuRepartitionExec: partitioning=Hash([o_custkey@1], 8), input_partitions=1, partitions=8
    GpuCoalescePartitionsExec, partitions=1, output_rows=727305       <- probe side
      …
```

Both sides shuffle on the same hash, which is why the murmur3 conformance gate is
load-bearing: if the CPU and GPU disagreed on one row's partition, that row would look for its
match in the wrong bucket and quietly vanish from the result.

**A cross join — no key to shuffle on.** `SELECT * FROM region, nation` produces no
repartition at all, because there is nothing to partition by. DataFusion instead requires the
left input at `SinglePartition` and streams the right past it:

```
GpuCrossJoinExec
  GpuScanExec: table=region, projections=[r_regionkey, r_name, r_comment]
  GpuScanExec: table=nation, projections=[n_nationkey, n_name, n_regionkey, n_comment]
```

#### `GpuCrossJoin` vs `GpuNestedLoopJoin`

Both are the wrapper over a join with no equality to hash, and which one appears is decided by
whether there is a predicate at all.

- **No join predicate ⇒ `CrossJoinExec`.** `SELECT * FROM region, nation` — a full cartesian
  product, every left row against every right row. In the corpus: the `cross-join` fixture plus
  tpcds q23, q28, q61, q77, q88, q90 — all of them pairing one-row aggregate results with no
  condition, e.g. q61 puts `sum(ss_ext_sales_price) as promotions` beside `total` so it can
  divide them.
- **A predicate that is not an equijoin ⇒ `NestedLoopJoinExec`**, carrying that predicate as
  its `filter`. `SELECT * FROM region a, nation b WHERE a.r_regionkey < b.n_regionkey` becomes
  `GpuNestedLoopJoinExec: join_type=Inner, filter=n_regionkey@1 > r_regionkey@0`. In the corpus:
  the `nested-loop-join` fixture plus tpch q11 and q22 and tpcds q9, q14, q24, q44, q54.

The tpch pair is worth recognizing, because it is a shape rather than an accident: q11's
`having sum(ps_supplycost * ps_availqty) > (select sum(…) * 0.000002 …)` plans as a
nested-loop join whose filter is the comparison against the one-row subquery —
`filter=CAST(sum(partsupp.ps_supplycost * partsupp.ps_availqty)@0 AS Decimal128(38, 15)) > …`.
A scalar threshold is a 1×N join with an inequality, so the planner has nowhere to put it but
here. (Rewriting that into a broadcast filter is the optimization #27 was archived for.)

Neither operator can run partitioned on the GPU: the map arm requires equal child partition
counts, DataFusion collapses their left input to one lane, and the mismatch throws
(#97). Both are also in the non-stripping set, which is how they miss the resident model's
join arm (#131).

## Traits and interfaces

Declarations are quoted with doc comments elided. Items marked *de facto* have no trait or
abstract base behind them, yet other code is written against them, so changing one breaks a
caller that never named it.

### Rust

**[`Executor` / `InstrumentedExecutor`](../peacockdb-core/src/executors/executor.rs#L18)** —
what a mode class offers a caller. All five modes implement the first; the four
node-visible ones implement the second, because the all-at-once GPU path makes a single FFI
call and never sees individual nodes.

```rust
pub trait Executor {
    async fn execute(&self, sql: &str) -> DfResult<Vec<RecordBatch>>;
}

pub trait InstrumentedExecutor: Executor {
    async fn execute_instrumented(
        &self,
        sql: &str,
    ) -> DfResult<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>)>;
}
```

**[`NodeExecutor`](../peacockdb-core/src/executors/node_by_node.rs#L28)** — the backend
contract behind `execute_node_by_node`. Two implementations: `CpuNodeExecutor` (DataFusion
per node) and `GpuNodeExecutor` (the FFI). Used generically, never as `dyn`.

```rust
pub trait NodeExecutor {
    async fn execute_node(
        &mut self,
        seq: usize,
        node: &Arc<dyn ExecutionPlan>,
        input_handles: &[Vec<u64>],
    ) -> DfResult<(Vec<u64>, NodeMemoryStats)>;

    async fn materialize(&mut self, handles: &[u64]) -> DfResult<Vec<RecordBatch>>;

    fn release(&mut self, handles: &[u64]);
}
```

**[`Operator` and `PartitionTopology`](../peacockdb-core/src/operators/operator.rs#L12)** —
what every `Gpu*Exec` wrapper exposes, so the engine treats them uniformly instead of
downcasting per call site. `strips_to_inner` defaults to true and five operators override
it; `partition_topology` currently has no callers (#130).

```rust
pub enum PartitionTopology {
    ScanEmit,          // the scan: N partitions from its row-group map (or 1 with no map)
    Map,               // 1:1 per partition — filter, project, and friends
    Collapse,          // M -> 1 (CoalescePartitions)
    KWayMerge,         // M -> 1 preserving order (SortPreservingMerge)
    RepartitionHash,   // 1 -> N by Spark-murmur3 hash
    Join,              // two inputs -> one output
}

pub trait Operator: ExecutionPlan {
    fn inner(&self) -> &Arc<dyn ExecutionPlan>;
    fn partition_topology(&self) -> PartitionTopology;
    fn strips_to_inner(&self) -> bool { true }
}
```

**[`GpuExtraDisplay`](../peacockdb-core/src/operators/mod.rs#L20)** — crate-private, one
method with a default. The `gpu_exec_node!` macro's `DisplayAs` impl calls it, so a family
file adds plan-text detail (`GpuScanExec: table=…`) by overriding it and nothing else.

```rust
pub(crate) trait GpuExtraDisplay {
    fn extra_display_info(&self) -> String {
        String::new()
    }
}
```

**[`SelectivityEstimator` / `CardinalityEstimator`](../peacockdb-core/src/gpu_rule.rs#L232)** —
crate-private seams the cost-annotation pass takes by reference, so the estimates can be
replaced without touching the walk. Both implementations are constants today
(`TrivialSelectivityEstimator`, `TrivialCardinalityEstimator`, both returning 1.0), which is
exactly what the plan goldens' `estimate_*` fields assume: filters pass everything, joins are
1:1. These are the hooks #19 and #73 replace.

```rust
pub(crate) trait SelectivityEstimator {
    /// Fraction of input rows passing the predicate (0.0 = none, 1.0 = all).
    fn estimate(&self, plan: &Arc<dyn ExecutionPlan>) -> f64;
}

pub(crate) trait CardinalityEstimator {
    /// output_rows / max(left_rows, right_rows): >1 fan-out, <1 filtering join.
    fn estimate(&self, plan: &Arc<dyn ExecutionPlan>) -> f64;
}
```

Traits the crate does not define but must satisfy, since they are equally part of the
surface: DataFusion's `ExecutionPlan` + `DisplayAs` (the `gpu_exec_node!` macro for 15
wrappers, hand-written for
[`GpuScanExec`](../peacockdb-core/src/operators/scan.rs#L179) and
[`StreamSourceExec`](../peacockdb-core/src/executors/stream.rs#L156)),
`PhysicalOptimizerRule` ([the wrapper-annotation and budget rules](../peacockdb-core/src/gpu_rule.rs#L89)),
`PruningStatistics` ([row-group pruning](../peacockdb-core/src/gpu_rowgroup_prune.rs#L55)),
and `Stream` + `RecordBatchStream`
([`InstrumentedStream`](../peacockdb-core/src/executors/stream.rs#L287)).

**[`ResidentEnforcer`](../peacockdb-core/src/executors/stream.rs#L42)** — *de facto*, and the
only stateful contract in the CPU path with no trait over it. `build_stream` registers each
node's accounting skeleton before that node can produce anything, every instrumented stream
reports its completion, and the enforcer recomputes the modeled peak and trips once it
crosses the budget. Four methods, and the ordering between them is the whole contract:
register before execute, or a node completes against a skeleton that does not know it.

```rust
pub(crate) struct ResidentEnforcer { /* budget, skeleton, tripped */ }

impl ResidentEnforcer {
    pub(crate) fn new(budget: usize) -> Self;
    pub(crate) fn register(&self, seq: usize, name: String, children: Vec<usize>);
    pub(crate) fn on_complete(&self, seq: usize, output_bytes: usize);
    pub(crate) fn tripped_error(&self) -> Option<DataFusionError>;
}
```

The stats types those traits pass around —
[`NodeMemoryStats` and `PartitionStat`](../peacockdb-core/src/executors/executor.rs#L46) —
are plain data, but they are contracts too: `PartitionStat` being empty is what a
single-output-partition node means, and the `.cpu.txt` golden renders sub-lines only when it
is not.

### C++

**[The C ABI](../cpp/include/peacock_gpu.h)** — the entire public surface of the C++ side,
plus `partitioning.hpp`. Everything else under `cpp/src/peacock/` is private to the library.

```c
const char* peacock_gpu_version(void);
typedef struct peacock_executor peacock_executor_t;

int  peacock_executor_create(uint64_t gpu_memory_limit, peacock_executor_t** out_executor);
void peacock_executor_destroy(peacock_executor_t* executor);
const char* peacock_last_error(peacock_executor_t* executor);

/* all-at-once: whole plan in, Arrow IPC out (retiring with #110) */
int  peacock_execute(peacock_executor_t* executor, const uint8_t* plan_bytes,
                    uint64_t plan_len, uint8_t** out_result_bytes,
                    uint64_t* out_result_len);
void peacock_result_free(uint8_t* result_bytes);

/* node-by-node: one session, one node at a time, intermediates stay resident */
typedef struct PeacockNodeStats {
  uint64_t rows; uint64_t varlen_content_bytes; uint64_t time_us;
} PeacockNodeStats;

int  peacock_executor_begin_plan(peacock_executor_t* executor, const uint8_t* plan_bytes,
                                 uint64_t plan_len, uint64_t* out_node_count);
int  peacock_executor_execute_node(peacock_executor_t* executor, uint64_t seq,
                                   const uint64_t* input_handles,
                                   const uint64_t* input_child_counts, uint64_t n_children,
                                   uint64_t* out_handles, uint64_t out_cap,
                                   uint64_t* out_count, PeacockNodeStats* out_stats);
void peacock_handle_release(peacock_executor_t* executor, uint64_t handle);
void peacock_executor_end_plan(peacock_executor_t* executor);

/* per-call entry points: what a driver decides per call — a batch's row groups, a
   limit's bounds — cannot ride a plan node, whose fields are constants.
   A row range is [offset, offset+length), UINT64_MAX meaning to the end, an offset
   past the end empty and an overrun clamped. */
int  peacock_executor_execute_scan_rowgroups(peacock_executor_t* executor, uint64_t seq,
                                             const uint32_t* row_groups, uint64_t n,
                                             uint64_t* out_handle,
                                             PeacockNodeStats* out_stats);
int  peacock_executor_slice_handle(peacock_executor_t* executor, uint64_t handle,
                                   uint64_t offset, uint64_t length, uint64_t* out_handle);
int  peacock_result_from_handle(peacock_executor_t* executor, uint64_t handle,
                                uint64_t offset, uint64_t length,
                                uint8_t** out_ipc, uint64_t* out_ipc_len);

/* benchmark instrumentation: process-global, off by default. Enabling it makes
   execute_node synchronize the default stream at every measurement boundary, so
   time_us measures execution rather than kernel submission — and serializes what
   cuDF would otherwise pipeline, which is why the correctness path never sets it.
   The floor is what an empty timed region costs; a node at or below it is
   unresolved, not cheap, and it is never subtracted. */
void     peacock_set_node_timing(int enable);
uint64_t peacock_measure_timing_floor_us(unsigned samples);

/* the conformance hook: Spark-murmur3 partition ids over one Arrow C-data batch */
int  peacock_spark_partition_ids(const void* schema, const void* array,
                                 const uint32_t* key_cols, uint64_t num_keys,
                                 uint32_t num_partitions, uint32_t seed,
                                 int32_t* out_pids, uint64_t out_cap, uint64_t* out_n);
```

**[`NodeSession`](../cpp/src/plan_executor.h#L46)** — *de facto*. No abstract base, but an
interface in every practical sense: it is what the node-by-node FFI entry points are thin wrappers over, and
the Rust `GpuNodeExecutor` is written against its shape. Nodes are addressed by canonical
post-order sequence, the same order the Rust walk uses, so child handles align across the
boundary. PIMPL, so the header exposes no cuDF internals.

```cpp
class NodeSession {
 public:
  NodeSession(const uint8_t* plan_bytes, uint64_t plan_len);
  ~NodeSession();
  NodeSession(const NodeSession&) = delete;
  NodeSession& operator=(const NodeSession&) = delete;

  size_t node_count() const;

  // Input handles are CONSUMED. out_stats is filled PER PARTITION.
  void execute_node(uint64_t seq, const uint64_t* input_handles,
                    const uint64_t* input_child_counts, size_t n_children,
                    uint64_t* out_handles, size_t out_cap, size_t* out_count,
                    NodeStats* out_stats);

  // The scan's row groups and the slice's bounds are per-call values, so they are
  // arguments here rather than fields of the node addressed by seq.
  uint64_t execute_scan_rowgroups(uint64_t seq, cudf::host_span<const uint32_t> row_groups,
                                  NodeStats* out_stats);
  uint64_t slice_handle(uint64_t handle, uint64_t offset, uint64_t length);

  const TableResult& table_for(uint64_t handle) const;
  void release(uint64_t handle);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
```

**[`TableResult` / `NodeStats`](../cpp/src/plan_executor.h#L13)** — the two value types every
C++ path returns. `NodeStats` carries only what C++ alone can measure: the byte formula
lives in Rust so the two engines cannot drift.

```cpp
struct TableResult {
  std::unique_ptr<cudf::table> table;
  std::vector<std::string> column_names;
};

struct NodeStats {
  uint64_t rows = 0;
  uint64_t varlen_content_bytes = 0;   // Σ over varlen columns of offsets[n]-offsets[0]
  uint64_t time_us = 0;                // per output partition; 0 unless timing is on
};
```

**[`NodeInputs` and the operator dispatch](../cpp/src/peacock/operators.h#L22)** — the
contract every operator translation unit shares. `NodeInputs` is passed explicitly rather
than through a thread-local, and that is deliberate: a per-translation-unit thread-local
would silently fork when the file was split and re-execute whole subtrees
(coding-style.md).

```cpp
struct NodeInputs {
  std::vector<TableResult>* items = nullptr;   // null => recursive mode
  size_t idx = 0;
};

TableResult execute_scan(const fb::CudfScan* scan,
                         const flatbuffers::Vector<uint32_t>* row_groups_override = nullptr);
TableResult execute_filter(const fb::CudfFilter* filter, NodeInputs* in);
TableResult execute_project(const fb::CudfProject* proj, NodeInputs* in);
TableResult execute_aggregate(const fb::CudfAggregate* agg, NodeInputs* in);
TableResult execute_hash_join(const fb::CudfHashJoin* join, NodeInputs* in);
TableResult execute_cross_join(const fb::CudfCrossJoin* join, NodeInputs* in);
TableResult execute_nested_loop_join(const fb::CudfNestedLoopJoin* join, NodeInputs* in);
TableResult execute_sort(const fb::CudfSort* sort, NodeInputs* in);
TableResult execute_union(const fb::CudfUnion* u, NodeInputs* in);
TableResult execute_limit(const fb::CudfLimit* limit, NodeInputs* in);
TableResult execute_window(const fb::CudfWindow* win, NodeInputs* in);

TableResult execute_node(const fb::PlanNode* node, NodeInputs* in);
TableResult execute_one(const fb::PlanNode* node, std::vector<TableResult> inputs);
inline TableResult execute_passthrough(const fb::PlanNode* input_node, NodeInputs* in);
```

**[`peacock::partitioning`](../cpp/include/peacock/partitioning.hpp)** — the second public
header: our own bit-exact Spark-murmur3, because cuDF ships only standard murmur3.

```cpp
std::unique_ptr<cudf::column> spark_partition_ids(
    cudf::table_view const& input,
    std::vector<cudf::size_type> const& key_cols,
    cudf::size_type num_partitions,
    uint32_t seed                     = 42,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> spark_hash_partition(
    cudf::table_view const& input,
    std::vector<cudf::size_type> const& key_cols, /* … same trailing defaults … */);
```

**[`ExprContext`](../cpp/src/peacock/expr.h#L25)** — *de facto*. Expression building. cuDF AST nodes hold
references, so something must own every sub-expression for the lifetime of the call; that
ownership IS the interface.

```cpp
struct ExprContext {
  std::vector<std::unique_ptr<cudf::ast::expression>> owned;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  cudf::ast::expression& keep(std::unique_ptr<cudf::ast::expression> e);
};

using JoinFilterColMap = flatbuffers::Vector<const fb::JoinFilterColumn*>;
cudf::ast::expression& build_expr(const fb::Expr* expr, ExprContext& ctx,
                                  const JoinFilterColMap* col_map = nullptr);
```

**[`GpuWorker` / `WorkerPool`](../cpp/tests/gpu/multi_gpu.hpp#L79)** — *de facto*, test-only
(`cpp/tests/gpu/`), and the one place the multi-GPU rules are encoded as a type: a cuDF or
cuVS object must be destroyed on its owning device's thread, so every device gets a worker
thread and a persistent stream, and work reaches a device only by `submit`.

```cpp
class GpuWorker {
 public:
  explicit GpuWorker(int device);
  ~GpuWorker();
  template <typename F> auto submit(F f) -> std::future<decltype(f())>;
};

class WorkerPool {
 public:
  explicit WorkerPool(int num_gpus);
  ~WorkerPool();
  int size() const;
  GpuWorker& operator[](int g);
  rmm::cuda_stream_view stream(int g) const;
};
```

### The handle registry has no type

Both node-by-node backends keep intermediates alive behind opaque `u64` handles, and on
neither side is that a class. In C++ it is two fields inside the private
[`NodeSession::Impl`](../cpp/src/node_session.cpp#L70) —
`std::unordered_map<uint64_t, TableResult> registry` and `uint64_t next_handle = 1` — with
allocation, lookup, consume-on-read and erase written inline at each of the twenty-two sites
that touch them. The Rust CPU backend has the twin arrangement — a
`HashMap<u64, Vec<RecordBatch>>` plus `next_handle`, allocated by
[`store`](../peacockdb-core/src/executors/backend/cpu_node_executor.rs#L291).

So the consume-once rule the FFI documents ("input handles are CONSUMED") holds by
convention at each site rather than by construction, and it is checked only at run time:
reading an already-consumed handle throws `unknown input handle`, and
[`execute_one`](../cpp/src/operators/dispatch.cpp#L110) throws when a node consumes a
different number of inputs than it was given. A `HandleRegistry` with `insert` / `take` /
`borrow` would put the rule in one place and make double-consumption unrepresentable rather
than merely detected. Nothing needs it yet — the two checks have held — but no type is
holding this together.

## Backend-agnostic node-by-node driver

`executors/node_by_node.rs` holds two things: the `NodeExecutor` trait and the driver
`execute_node_by_node`. The driver walks the plan in canonical post-order — children
left-to-right, then the node. That order is not just determinism: it is the order
`NodeSession` indexes by, so one sequence number means the same node on both sides of the FFI.

Each node is handed its children's partition-handle vectors and returns its own. One node maps
to N handles, one per output partition. Only the root is materialized and released — every
intermediate stays where the backend put it, the CPU registry or GPU VRAM, so data crosses the
boundary exactly once.

Two backends implement the trait. `backend/cpu_node_executor.rs` runs DataFusion per node.
`backend/gpu_node_executor.rs` drives the FFI: `begin_plan`, then one `execute_node` per node,
then `end_plan`. The GPU side returns per-partition rows and varlen bytes, and Rust recomputes
logical bytes from the schema — so the two engines' costs are identical by construction rather
than by agreement.

How that sits against everything else:

```
 caller (tests · cost goldens · CLI)
   │  execute(sql) │ execute_instrumented(sql)
   ▼
 Executor / InstrumentedExecutor ......................... the only public entry
   ├─ FullTableCpuExecutor ─────► build_stream = the streaming driver, not this one
   │                                ├─ strip_target ............... [Operator]
   │                                ├─ execute_stream → N→1 collapse per node
   │                                └─ ResidentEnforcer: register / on_complete / tripped_error
   ├─ PartitionedCpuExecutor ───┐
   ├─ FullTableGpuExecutor ─────┤ all three enter the driver; the two GPU classes are thin
   ├─ PartitionedGpuExecutor ───┘ wrappers over one GpuExecutor + a fixed PartitionMode
   └─ AllAtOnceGpuExecutor ─────► peacock_execute → the recursive C++ driver (#110)

 execute_node_by_node(root, backend)                generic over the backend, no dyn
   ├─ post_order(root) ............ children left-to-right, then the node
   ├─ per seq: backend.execute_node(seq, node, &input_handles) ..... [NodeExecutor]
   ├─ handles[seq]: Vec<u64> ....... one handle per OUTPUT partition
   └─ materialize(root) + release .. the only crossing; intermediates never leave

 NodeExecutor
   ├─ CpuNodeExecutor ... registry HashMap<u64, Vec<RecordBatch>>; per node it strips the
   │                      wrapper [Operator] and runs execute_single_node per partition
   └─ GpuNodeExecutor ... FFI: begin_plan · execute_node per node · end_plan
                            │
                            ▼   C++
                          NodeSession
                            ├─ post-order index over the FlatBuffers plan (same order)
                            ├─ registry unordered_map<uint64_t, TableResult>
                            └─ execute_one → run_op → execute_<op>
```

The driver depends on `NodeExecutor` and nothing else. It never downcasts, and never sees a
`RecordBatch` or a `cudf::table`. That is what backend-agnostic buys: the same short loop
drives a DataFusion backend and an FFI one.

`Operator` therefore never appears in the driver — it is reached one level down, through
[`strip_target`](../peacockdb-core/src/operators/mod.rs#L150), which asks `as_operator` what
kind of wrapper this is and `strips_to_inner` whether to replace it with the inner DataFusion
node. Both CPU paths call it: `build_stream` on its way down the tree, and
`execute_single_node` on each node the partitioned backend runs. The GPU backend never uses the
trait at all, because the wrapper tree is already serialized and the C++ side dispatches on the
flat-buffer node kind instead. The one wrapper it does reach into is `GpuScanExec`, for the row-group map
behind its per-partition stats, and it does that by concrete downcast rather than through
`Operator`.

`ResidentEnforcer` hangs off the streaming driver only. The node-by-node driver has no memory
enforcement of any kind: nothing registers a skeleton, nothing trips, and a partitioned run
that exceeds its budget finds out from the allocator. Porting it is half of #91.

The two post-order walks are the load-bearing coincidence. `post_order` in Rust and
`index_post_order` in `NodeSession` are separate implementations of the same rule, and every
handle the FFI exchanges is addressed by that sequence number. Nothing compares them: the
`begin_plan` call even hands Rust a node count, and `GpuNodeExecutor::new` reads it into a local
and drops it (#134). A divergence would surface indirectly — as per-node numbers that no longer
line up with the `.cpu.txt` golden, or as an unknown-handle throw — rather than as the
mismatch it is.

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
identical **by construction**, the CPU twin uses comet's `create_murmur3_hashes` — in
`peacockdb-core/src/spark_partitioning.rs`, the one spelling both CPU paths call — and the
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

## Column indexing

Nothing in the flat buffers names a column to read. Every reference is an ordinal into the child's
output table, so a node's correctness depends on the child having produced its columns in
exactly the order DataFusion assumed when it planned.

Where the ordinals come from and where they land:

| Reference | Written by | Read by |
|---|---|---|
| `ColumnRef.index` in any expression | [`plan_serializer.rs`](../peacockdb-core/src/plan_serializer.rs#L134), copied straight off DataFusion's `Column::index()` | [`build_expr`](../cpp/src/expr.cpp#L140) for the AST path, [`build_column`](../cpp/src/expr.cpp#L833) for the column path |
| `projection` index lists on filter and join | the operator's own serializer | [`filter.cpp`](../cpp/src/operators/filter.cpp#L40), [`join.cpp`](../cpp/src/operators/join.cpp#L206) — gather by ordinal, and the name list is indexed with the same ordinal |
| join key pairs, `on=[(l@0, r@0)]` | join serializer | [`join.cpp`](../cpp/src/operators/join.cpp#L58) — ColumnRef only, anything else throws |
| `JoinFilterColumn{side, index}` | join serializer | [`expr.cpp`](../cpp/src/expr.cpp#L145) — remaps a filter-schema ordinal onto the mixed join's LEFT/RIGHT tables |
| sort keys, hash keys, group keys, window args | the family serializers | [`sort.cpp`](../cpp/src/operators/sort.cpp#L38), [`node_session.cpp`](../cpp/src/node_session.cpp#L255), [`aggregate.cpp`](../cpp/src/operators/aggregate.cpp#L157), `window.cpp` |

There are 22 `->index()` reads and 30 `.column(idx)` calls on the C++ side, so this is the
engine's most common operation and the one with the least ceremony around it.

### What actually guards it

Your presumption is nearly right — there is one real backstop, and it is not ours.

- **cuDF bounds-checks column access.** `table_view::column(i)` is `_columns.at(i)`, so an
  out-of-range ordinal throws `std::out_of_range` rather than reading garbage. The FFI catches
  `std::exception` and surfaces the message, so the failure is loud. What it is not is
  *informative*: the message is `vector::at` boilerplate with no node, no operator and no
  ordinal, because the check is three layers below the code that had the context.
- **Two explicit checks, in `expr.cpp` only.** The column path
  ([#L837](../cpp/src/expr.cpp#L837)) throws with the ordinal and the column count, which is the
  message you actually want. The type-inference helper ([#L349](../cpp/src/expr.cpp#L349))
  checks the same thing and **returns `type_id::EMPTY`** — a silent fallback that turns a bad
  ordinal into an unhelpful type error further along.
- **One arity check.** The Final-stage aggregate compares its input width against the state
  arity it expects and throws when they disagree
  ([`aggregate.cpp#L505`](../cpp/src/operators/aggregate.cpp#L505)). It is the only place a
  schema-shape mismatch is deliberately caught, and it catches width, not order.
- **The FlatBuffers verifier** checks structure — that offsets and vectors are well formed. It
  has no idea what an ordinal means.

### What nothing guards

**Column names are a parallel array with no invariant.** `TableResult` is a `cudf::table` plus a
`std::vector<std::string>`, and nothing asserts the two have the same length. The six sites that
index the names do it with `operator[]`, so a names vector shorter than the table is undefined
behaviour rather than an exception — for example
[`filter.cpp#L42`](../cpp/src/operators/filter.cpp#L42), where the same loop iteration reads
`fv.column(idx)` (checked) and `input.column_names[idx]` (unchecked). Today the checked read
happens first and throws, which is luck, not design.

**Nothing checks that a child's column *order* matches what the plan assumed.** The per-node
golden records the node name, partition count, output rows and output bytes — not the column
list, not the types. And the bytes cannot help: both engines compute them from the *plan's*
schema via `logical_size_from_schema`, deliberately, so that CPU and GPU cannot drift. The same
choice means a node that emitted the right number of columns in the wrong order produces
identical per-node numbers on both engines. The divergence surfaces only at the root, in the
final result comparison, and only for a query that has a `.result.txt` golden or an oracle —
a subtree bug pinned nowhere else in the tree.

That is the honest state: the ordinal contract is enforced by cuDF's `at()` for gross violations
and by the result comparison for subtle ones, with nothing in between. Adding
`num_columns() == column_names.size()` to `TableResult`'s construction, and a per-node type
check in the GPU tiers, are the two obvious closures (#164).

## cuDF options

cuDF's defaults are not SQL's, and they are not DataFusion's. Every option below is a place
where taking the default would produce a plausible wrong answer rather than an error, so each
one is either set explicitly or carried in the flat buffers — and the ones carried in the flat buffers are carried
precisely so cuDF cannot infer something the CPU side did not.

| Option | Set at | Value | What the default would do |
|---|---|---|---|
| `parquet_reader_options` | [`scan.cpp`](../cpp/src/operators/scan.cpp#L57) | `.columns(projected)`, `set_row_groups(map ∥ pruned)`, `set_num_rows(limit)` | read every column and every row group; the row-group list is also how a partition reads only its own slice |
| `cudf::order`, `cudf::null_order` | [`sort.cpp`](../cpp/src/operators/sort.cpp#L44), [`node_session.cpp`](../cpp/src/node_session.cpp#L192) | per key from the flat buffers's `asc` / `nulls_first` | cuDF has no notion of the query's ORDER BY; the two sites must agree or a k-way merge would order differently from a sort |
| `cudf::null_equality` | [`join.cpp`](../cpp/src/operators/join.cpp#L90) ×9 | see the table below | `EQUAL` — NULL keys match, inventing rows SQL excludes |
| `cudf::out_of_bounds_policy` | [`join.cpp`](../cpp/src/operators/join.cpp#L315) | `NULLIFY` on the side that can be unmatched, `DONT_CHECK` otherwise | `DONT_CHECK` reads the `JoinNoneValue` sentinel (`INT32_MIN`) as an index and faults with `cudaErrorIllegalAddress` |
| `cudf::null_policy` (groupby) | [`aggregate.cpp`](../cpp/src/operators/aggregate.cpp#L435), [grouping sets](../cpp/src/operators/aggregate.cpp#L392) | `INCLUDE` | `EXCLUDE` silently drops the NULL group — tpcds q15's NULL `ca_zip` row disappears |
| `cudf::null_policy` (rolling count) | [`window.cpp`](../cpp/src/operators/window.cpp#L103) | `EXCLUDE` for `COUNT(col)`, `INCLUDE` for `COUNT(*)` | one of the two is always wrong: `COUNT(*)` counts rows, `COUNT(col)` counts non-nulls |
| decimal scale | [`aggregate.cpp`](../cpp/src/operators/aggregate.cpp#L212), [`union.cpp`](../cpp/src/operators/union.cpp#L45), [`window.cpp`](../cpp/src/operators/window.cpp#L86) | `data_type{id, -out_decimal_scale}` from the flat buffers | cuDF would re-derive a scale per operation and drift from DataFusion's |
| binary-op output type | [`expr.cpp`](../cpp/src/expr.cpp#L547) | boolean for predicates, else the wider input; division pre-scales the numerator to hit the flat buffers's `out_decimal_precision/scale` | cuDF promotes by its own rule, which is not SQL's decimal arithmetic |
| hash seed / algorithm | [`spark_hash_partition.cu`](../cpp/src/spark_hash_partition.cu#L198) | our own Spark-murmur3, seed 42, cuDF only for the scatter | cuDF ships standard murmur3, whose partition numbers differ from comet's — see [Rehash and the comet hash](#rehash-and-the-comet-hash) |
| IPC export | [`gpu_executor.cpp`](../cpp/src/gpu_executor.cpp#L39) | column names as `column_metadata`; DECIMAL32/64 cast up to DECIMAL128 | unnamed columns, and narrow decimals that the Rust arrow-ipc reader rejects outright |
| stream + memory resource | everywhere in the single-GPU path | `cudf::get_default_stream()`, current device resource | fine on device 0 and wrong anywhere else — the multi-GPU rules are in [Multi-GPU notes](#multi-gpu-notes-cudf-2602) |

### What the Rust side puts in the flat buffers

Half of the table above is not a choice the C++ side makes — it reads a value DataFusion
already computed and the serializer wrote down. That is deliberate: an option carried in the
flat buffers cannot be re-derived differently by the two engines, so anything where cuDF's own
inference could drift from DataFusion's is serialized rather than inferred.

| Flat-buffer field | Written by | Taken from | Becomes |
|---|---|---|---|
| `CudfHashJoin.null_equals_null` | [`join.rs#L172`](../peacockdb-core/src/operators/join.rs#L172) | `HashJoinExec::`<br>`null_equals_null()` | `cudf::null_equality` (except anti/mark, below) |
| `JoinFilterColumn{side, index}` | [`join.rs#L337`](../peacockdb-core/src/operators/join.rs#L337) | the join filter's `ColumnIndex` list | `cudf::ast::`<br>`table_reference::LEFT` / `RIGHT`,<br>plus an ordinal |
| `SortExpr.asc`, `.nulls_first` | [`sort.rs#L84`](../peacockdb-core/src/operators/sort.rs#L84) (sort), [`#L129`](../peacockdb-core/src/operators/sort.rs#L129) (merge), [`window.rs#L127`](../peacockdb-core/src/operators/window.rs#L127) | `PhysicalSortExpr::options` | `cudf::order`,<br>`cudf::null_order` |
| `CudfSort.fetch`,<br>`CudfSortPreservingMerge.fetch` | [`sort.rs#L91`](../peacockdb-core/src/operators/sort.rs#L91) | `SortExec::fetch()` | a post-sort / post-merge slice |
| `AggregateFuncNode`<br>`.out_decimal_precision/scale` | [`aggregate.rs#L125`](../peacockdb-core/src/operators/aggregate.rs#L125) | the aggregate's output `Field` type | `cudf::data_type{id, -scale}` |
| `BinaryExpr`<br>`.out_decimal_precision/scale` | [`plan_serializer.rs`<br>`#L149`](../peacockdb-core/src/plan_serializer.rs#L149) | `bin.data_type(schema)` | the binop output type, and division pre-scales to hit it |
| `WindowExpr.out_decimal_scale` | [`window.rs#L156`](../peacockdb-core/src/operators/window.rs#L156) | the window output `Field` type | `cudf::data_type` scale |
| `CudfUnion.output_schema` | [`union.rs#L61`](../peacockdb-core/src/operators/union.rs#L61) | `plan.schema()` | per-branch `cudf::cast` target before concatenate |
| `CudfAggregate.mode` | [`aggregate.rs#L73`](../peacockdb-core/src/operators/aggregate.rs#L73) for the legacy modes, [`recipe/aggregate_writer.rs`](../peacockdb-core/src/batch_partitioned/recipe/aggregate_writer.rs) for the batch-partitioned one | `AggregateExec::mode()` —<br>Partial / Final / FinalPartitioned;<br>or `Merge`, which DataFusion has no name for: merge state and emit state, so the finalize can be an expression the two engines share | which cuDF aggregation runs, whether state columns are merged, and whether the result is state or a value |
| `AggregateFuncNode.distinct` | [`aggregate.rs#L134`](../peacockdb-core/src/operators/aggregate.rs#L134) | `aggr.is_distinct()` | nothing yet — a guard throws rather than silently ignoring it (#62) |
| `CudfRepartition.hash_exprs`,<br>`num_partitions` | [`repartition.rs#L84`](../peacockdb-core/src/operators/repartition.rs#L84) | `Partitioning::Hash(exprs, n)` | key ordinals and N for<br>`spark_hash_partition` |
| `CudfScan.row_groups`,<br>`batches` | [`scan.rs#L339`](../peacockdb-core/src/operators/scan.rs#L339) | the pruning result and `build_scan_map` | `parquet_reader_options::`<br>`set_row_groups` |
| `CudfScan.limit` | [`scan.rs#L339`](../peacockdb-core/src/operators/scan.rs#L339) | a pushed-down limit | `set_num_rows` |
| `CudfScan.batch_size` | [`scan.rs#L339`](../peacockdb-core/src/operators/scan.rs#L339) | `GpuMemoryBudgetRule`'s derived batch size | **nothing** — no C++ code reads it (#132) |
| `CudfCoalesceBatches`<br>`.target_batch_size` | [`coalesce.rs#L63`](../peacockdb-core/src/operators/coalesce.rs#L63) | `CoalesceBatchesExec::`<br>`target_batch_size()` | **nothing** — the node is passthrough on the GPU (#132) |

Two shapes are worth separating here. Most rows carry a value the GPU must not recompute —
decimal scales above all, since cuDF derives its own per operation and DataFusion's is what the
result is compared against. The last three rows are different: they carry a value nothing reads,
which is the wire-format surface #132 is about.

### Join types and NULL key equality

`null_equals_null` travels in the flat buffers per join, mirroring DataFusion: `false` (the SQL default)
means a NULL key matches nothing, `true` means NULL = NULL, which is what a set operation
lowered to a join needs. Whether a join type actually honours it is the interesting part.

| Join type | cuDF call | `null_equality` | Code |
|---|---|---|---|
| Inner | `inner_join` | from the flat buffers | [join.cpp#L289](../cpp/src/operators/join.cpp#L289) |
| Left | `left_join` | from the flat buffers | [#L291](../cpp/src/operators/join.cpp#L291) |
| Full | `full_join` | from the flat buffers | [#L293](../cpp/src/operators/join.cpp#L293) |
| Right | `left_join` with sides swapped, indices swapped back | from the flat buffers | [#L295](../cpp/src/operators/join.cpp#L295) |
| LeftSemi | `left_semi_join`, `filtered_join::semi_join`, or `mixed_left_semi_join` with a residual filter | from the flat buffers | [#L116](../cpp/src/operators/join.cpp#L116) |
| RightSemi | the same, sides swapped; a residual filter is rejected | from the flat buffers | [#L153](../cpp/src/operators/join.cpp#L153) |
| LeftAnti | `left_anti_join`, `filtered_join::anti_join`, or `mixed_left_anti_join` | **hardcoded `EQUAL`** | [#L134](../cpp/src/operators/join.cpp#L134) |
| RightAnti | the same, sides swapped | **hardcoded `EQUAL`** | [#L170](../cpp/src/operators/join.cpp#L170) |
| LeftMark | `left_semi_join`-shaped, emitting one row per left row plus a boolean mark | **hardcoded `EQUAL`** | [#L224](../cpp/src/operators/join.cpp#L224) |
| Inner / Left, non-equi | `conditional_inner_join` / `conditional_left_join`, or an AST boolean mask | n/a — the predicate decides | [#L405](../cpp/src/operators/join.cpp#L405) |

Three things that table is worth reading for.

**Semi honours the flag and anti does not**, deliberately. `x IN (…)` and `EXISTS` are ordinary
three-valued predicates, so `UNEQUAL` is right and tpcds q33 needs it; a set operation lowered
to a semi join asks for `EQUAL` and gets it (q14). Anti is not symmetric: `x NOT IN (…, NULL)`
is never true for any x, which is neither `EQUAL` nor `UNEQUAL` — no cuDF setting implements
it, so anti and mark stay `EQUAL` until the planner distinguishes `NOT IN` from `NOT EXISTS`
(#80, #59).

**The equi-join default is the one that bites silently.** cuDF's `EQUAL` invents rows the SQL
oracle excludes, and the symptom is not an error but a count or sum one too large — tpcds q50,
q6 and q81 each inflated a downstream aggregate before `join_nulls` was threaded through.

**A residual filter is not optional on semi/anti.** The key-only cuDF calls ignore it, so a
LeftAnti on the key alone collapses to zero rows; those joins must take the `mixed_*` variants
that evaluate the AST during the join (TPC-H q21 is the case). RightSemi and RightAnti reject
a filter outright, because no swapped `mixed_*` variant exists.

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
- **Widget:** the cost report compares peacock Σout (from the `full_table-tp8-mini` `.cost.txt`
  footer) vs duckdb Σout; ratio ≤ 1.4 renders green. Directional signal only, not a
  benchmark. Per-query mode enablement comes from `testdata/cost-registry.csv` (the
  registry the inventory tests verify), tickets from `llm-wiki/tickets.md`.
