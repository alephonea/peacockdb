# batch-partitioned executor

A new planning and execution mode in which a GPU partition holds a *stream of batches*
instead of one resident table. The motivating pipeline: load → filter at 1% selectivity →
aggregate into few groups. Today's partitioned mode materializes the whole scan on the
GPU before the filter ever runs; with batches, small slices flow through the filter and
only the aggregate's state stays resident, so the query fits in a VRAM budget the table
does not.

Status (2026-08-07): design final — every open question from the draft review is resolved
and folded in below; the draft this supersedes lived outside the repo and is no longer
needed. Implementation has not started; the first hand-off to the developer is T0, on a
new `ENS-` branch when the human says to start. This is a large task spanning many
branches, so the spec lives on master, committed ahead of the work, rather than riding
any one task branch. Deferred work is ticketed, not latent: this file plus `tickets.md`
is the complete state.

Companion tickets: [#136](../tickets.md#t136),
[#137](../tickets.md#t137), [#138](../tickets.md#t138), [#139](../tickets.md#t139),
[#140](../tickets.md#t140), [#141](../tickets.md#t141), [#142](../tickets.md#t142),
[#143](../tickets.md#t143).

The first part of this file is the design; the second is the implementation plan, cut
into tasks that hand off to the developer one at a time.

## Contents

- [Scope and constraints](#scope-and-constraints)
- [Planning](#planning)
- [Node set](#node-set)
- [Traits](#traits)
- [Drivers](#drivers)
- [Memory accounting](#memory-accounting)
- [GPU execution through the frozen FFI](#gpu-execution-through-the-frozen-ffi)
- [Determinism rules](#determinism-rules)
- [Goldens, registry, widget](#goldens-registry-widget)
- [Implementation plan](#implementation-plan)

# Design

## Scope and constraints

**Feature scope.** Everything the legacy full_table CPU and GPU tiers run, except window
functions — window queries fail at plan time and render as plan ✗ in the new widget
tables ([#143](../tickets.md#t143)). The four TPC-DS queries that do not physically plan
on DataFusion 45 stay out until [#23](../tickets.md#t23). Unsupported shapes inside
supported features (mixed distinct + non-distinct #62, value-form CASE #57) fail at plan
time where the planner can see them, instead of throwing at run time; per-query
enablement stays in the registry as for legacy modes.

**Frozen-surface preference.** Keeping the C++ code paths, the FlatBuffers schema
(`flatbuffers/gpu_plan.fbs`) and the existing symbols in `peacock_gpu.h` unchanged is a
desired property, not an absolute: it keeps the legacy modes provably untouched and the
new mode honest about what it needs. One additive ABI entry point is planned from the
start — `peacock_executor_execute_scan_rowgroups()` (see
[GPU execution](#gpu-execution-through-the-frozen-ffi)) — because the review established
it is the single place the frozen surface actually blocks the design: the scan arm emits
every map entry in one FFI call (`cpp/src/node_session.cpp` ~L123), so incremental
loading is impossible without it, while every other operator maps onto existing arms
driven creatively. If during development the constraint proves too tight anywhere else
(candidates already known: #136's match bitmap or persistent build, #142's split entry
point), the developer does not work around it silently and does not change the surface
on their own — the coordinator raises a concrete proposal to the human naming the
blocked task, the smallest additive change that unblocks it, and what the workaround
would cost instead.

**Coexistence.** All six legacy modes and their tests stay functional throughout.
Retiring them is a separate, later decision (blocked at minimum by #143). Code reuse with
legacy is moderate and always by extracting shared helpers, never by entangling the new
planner with the wrapper/strip machinery.

## Planning

### Approach: translate the DataFusion physical plan

`plan_batch_partitioned()` runs DataFusion physical planning at the target partition
count, then a **translation layer** compiles the DataFusion tree into the new node
vocabulary. Not annotation (the legacy mistake — a 1:1 wrapper carries DataFusion's
execution semantics along), and not planning from the logical plan, which would mean
reimplementing what DataFusion does well:

- physical expression planning and type coercion — the serializers read `PhysicalExpr`
  trees and schema-derived decimal precision/scale; redoing that is the #55/#56/#63 bug
  class;
- the partial/final aggregate split with per-aggregate state schemas (sum+count for avg,
  M2 for stddev) — at tp1 DataFusion emits mode=Single aggregates, so the split only
  exists in plans made at tp>1;
- grouping-set expansion — `__grouping_id` arrives as an ordinary column of the partial,
  already in the final's group list;
- row-group pruning, which hangs off `ParquetExec` statistics.

The translation layer makes a conscious decision per DataFusion node kind — nothing is
carried over implicitly. Baseline mapping: hash `RepartitionExec` → `GpuMergePartitions` +
`GpuEmitPartitions` (the same shape the legacy budget rule lowers shuffles into);
round-robin `RepartitionExec` → dropped; `CoalesceBatchesExec` → dropped;
`SortExec` → the sort decomposition below; aggregate pairs → the aggregate sequence
below; join nodes → `GpuJoin` with side normalization; `UnionExec`/`InterleaveExec` →
driver-level relabeling plus explicit per-branch cast projects. The layer is unit-tested
node kind by node kind, and an unrecognized DataFusion node is a plan-time error naming
it — never a silent pass-through.

### Knobs

1. **Partition count**: tp1 and tp4 in tests. Small tables get 1 partition even at tp4;
   the row-count threshold is a stated planner input, not folklore.
2. **Batched loading off|on**: whether the loader emits more than one batch per
   partition. Even when off, the loader's declared layout is `MultipleBatches` — no
   downstream phase may assume one batch per partition. Only when on does the planner
   take the memory budget (micro/mini/standard/full) and size batches.

### ParquetBatchPartitioner

The row-group → (partition, batch) mapping is computed once, at plan time, by a pure
policy class, and everything else consumes its output: `GpuLoadParquet` stores it
verbatim, the plan golden's `partition_groups=[...]` renders it verbatim, the loader
executes it, validation checks the declared partition count against it. One fact, one
owner — the opposite of the #130 shape, where a fact declared in one place is re-derived
in three.

```rust
struct RowGroupMeta { index: u32, rows: u64 }        // survivors, file order
enum Batching { Off, On { target_batch_bytes: usize } }

fn partition(
    survivors: &[RowGroupMeta],
    row_width: usize,            // from the plan schema
    n_partitions: usize,         // small-table rule already applied by the caller
    batching: Batching,
) -> Vec<Vec<Vec<u32>>>          // partitions → batches → row groups
```

Policy: survivors (after pruning, same source as legacy) split into `n_partitions`
contiguous chunks balanced by row count; within a chunk, consecutive row groups are
packed greedily into batches while estimated bytes (rows × row width) stay under target;
a single row group over target still becomes its own batch — minimum granularity is one
row group, and the planner always produces a plan (the enforcer owns the runtime
consequence; recourse for oversized batches is [#142](../tickets.md#t142)). Batching off
means one batch per chunk. Contiguity is a policy choice, not a cuDF requirement —
changing it later is a golden-regenerating change and is treated as one.

### The aggregate sequence

A multi-partition multi-batch grouped aggregate plans as:

```
GpuAggregate[final=false]        per batch: partial state per input batch
GpuAggregateBatches[final=false] per partition: merge partial batches (optional, see below)
GpuMergePartitions               N → 1 stream
GpuEmitPartitions                1 → N on hash of group keys
GpuAggregateBatches[final=true]  per partition: final result
```

Shortcuts: a 1-partition single-batch input needs only `GpuAggregate[final=true]`; a
1-partition input skips Merge/Emit; a single-batch-per-partition input skips the first
`GpuAggregateBatches`. v1 skips the shuffle only for 1-partition inputs or keyless
aggregates; skipping on small key cardinality needs estimators that do not exist
([#141](../tickets.md#t141)).

Grouping sets need no special operator: the partial emits `__grouping_id` as an ordinary
column (existing C++ behavior, #65 caveats unchanged), the final groups on keys + gid
because DataFusion's final node already does, and hashing on keys alone still co-locates
correctly. The general validation rule this falls out of: the input to a final
`GpuAggregateBatches` must have `KeyDistribution.hashKeys ⊆ its group columns` — subset,
not equality.

### Null keys and the shuffle

`GpuEmitPartitions` has exactly one routing: Spark murmur3, nulls co-located (the kernel
skips null columns, comet-mandated — `cpp/src/spark_hash_partition.cu`; every all-null
key row lands in `pmod(seed, N)`). No runtime knob. The three null cases resolve
elsewhere: null=null joins require co-location (the default does it); sides whose
unmatched rows are never emitted get a planner-inserted `IS NOT NULL` filter later
([#137](../tickets.md#t137), not v1); scattering placement-free nulls is deferred with
the kernel-knob analysis in the same ticket.

## Node set

| Node | Category | Semantics |
|---|---|---|
| `GpuLoadParquet` | Source | reads survivor row groups per the partitioner's mapping; pruning as legacy; pull-based `next_batch()`; honors pushed-down limits |
| `GpuFilter`, `GpuProject` | Exec | 1:1 per batch |
| `GpuSort` | Exec | sorts each input batch independently; optional per-batch `fetch` (top-N); output `BatchSorted` |
| `GpuAccumulateBatchesAndSort` | BatchAccumulator | accumulates sorted batches, one `cudf::merge` at done; output one batch, `PartitionSorted`. No streaming emission — cuDF has no primitive; ranged emission is [#138](../tickets.md#t138) |
| `GpuMergeSortedPartitions` | PartitionAccumulator | input: N partitions, `MultipleBatches` allowed, `BatchSorted` required; all k·m sorted batches into one `cudf::merge`, `fetch` applied; output: 1 partition, one batch, `PartitionSorted` |
| `GpuCoalesceAllBatches` | BatchAccumulator | concatenates a partition's batches into one at done |
| `GpuMergePartitions` | BatchForwarder | N partition streams → 1, forwarding each batch as visited, round-robin (see [Determinism](#determinism-rules)); accumulates nothing, no backend calls |
| `GpuEmitPartitions` | PartitionEmitter | 1 → N per batch by hash scatter; streaming, one call per input batch |
| `GpuAggregate` | Exec | partial (or single-shortcut final) aggregation of one batch |
| `GpuAggregateBatches` | BatchAccumulator | merges pre-aggregated batches; emits at done |
| `GpuJoin` | Join | capability matrix below |
| `GpuCrossJoin`, `GpuNestedLoopJoin` | Join | two inputs, so `JoinExecutor`, not Exec: `set_build(left)`, probe right. Cross and inner NLJ stream the right side (same mechanics as inner hash-join probes); non-inner NLJ keeps a single-batch probe — the #136 finish trick needs keys to accumulate and a predicate join has none. Both inputs 1 partition; broadcast variants are [#140](../tickets.md#t140) |
| `GpuUnion`, `GpuInterleave` | BatchForwarder | lane relabeling; branch decimal/type normalization becomes explicit per-branch `GpuProject` casts inserted by the planner — which is what makes union pure routing. Union sums its inputs' lane counts and clears `KeyDistribution`; interleave is chosen (as in DataFusion's `can_interleave`) only when every input carries the same hash distribution — output lane p is lane p of each input, so `KeyDistribution` is preserved |
| `GpuLimit` | driver + two lowerings | start..limit over a 1-partition stream. The fb node's skip/fetch are frozen per seq, so per-batch GPU calls would truncate every batch — see the lowering rule below the recipe table |
| `GpuUnload` | Exec | `GpuBatch` in, `CpuBatch` out (Arrow IPC export per handle); 1:1 per batch, `NodeKind::Sink` at plan level |

Dropped from the draft: `GpuConcatBatchesAcrossPartitions` (subsumed by
`GpuMergePartitions`; the zip-concat adds a cross-partition barrier and copy cost for no
semantic gain) and `GpuCoalesceBatches(target)` (an optimization, not a correctness need
— [#139](../tickets.md#t139)).

### Join capability matrix

The build side is always left, single batch per partition (planner inserts
`GpuCoalesceAllBatches`). The streamable side is always right: the translation layer
swaps sides where DataFusion chose otherwise, remapping the join type
(Left ↔ Right forms) and restoring output column order with a project. Per type:

| Join type | Probe streams? | Per-probe-call emits | `finish_and_fetch()` emits |
|---|---|---|---|
| Inner | yes | matches | nothing |
| Right-outer (build=left) | yes | matches + unmatched probe rows (batch-local) | nothing |
| Left-outer / Full | yes, with finish | matches (+ unmatched probe for Full) | unmatched build rows, null-padded |
| LeftSemi / LeftAnti / LeftMark | yes, with finish | nothing | build rows with / without a match |
| CrossJoin, NestedLoop Inner | yes | left × probe batch (predicate-filtered for NLJ) | nothing |
| NestedLoop non-inner | no — single-batch probe (no keys for the #136 finish trick) | — | — |
| Any type with a residual filter | no — single-batch probe | — | — |

The finish pass needs "which build rows matched at least once", which never crosses the
current ABI; the v1 GPU implementation is the probe-key-accumulation trick, and the
per-probe-call build rebuild is an accepted v1 cost — both recorded with the ABI-change
alternatives in [#136](../tickets.md#t136). `null_equals_null` rides on `GpuJoin`
per join and applies to the finish join too; the #80 NOT IN semantics carry over
unchanged. This matrix is unit-tested: one case per (type × batch layout) asserting
stream-vs-refuse and correctness against the single-batch oracle.

## Traits

```rust
enum NodeKind { Source, Intermediate, Sink }

enum KeyDistribution { NotSpecified, ByHash { hash_keys: Vec<u32> } }   // spark murmur3, seed 42

enum SortOrder {
    NotSpecified,
    BatchSorted { columns: Vec<ColumnOrder> },      // each batch sorted; batches unordered
    PartitionSorted { columns: Vec<ColumnOrder> },  // whole stream sorted; implies BatchSorted
}
// Under SingleBatch layout the two coincide: canonicalize to PartitionSorted.
// Validation accepts BatchSorted wherever the weaker property suffices.

enum BatchLayout { SingleBatch, MultipleBatches }

struct PartitionLayout {
    n: usize,
    key_distribution: KeyDistribution,
    sort_order: SortOrder,
    batch_layout: BatchLayout,
}

trait GpuNode {
    fn kind(&self) -> NodeKind;
    fn output_partitions(&self) -> Option<PartitionLayout>;  // present for non-sink nodes
    fn output_schema(&self) -> Option<Schema>;               // present for non-sink nodes
    fn children(&self) -> Vec<&dyn GpuNode>;
    // Checks children's schemas, partition topology, key distribution, sortedness and
    // batch layout against this node's requirements; validates captured column indices
    // (group keys, aggregate lists, two-phase state columns) against child schemas.
    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError>;
}
```

`Schema` carries column types plus semantics annotations (sort column, group key,
aggregator, two-phase state) — anything a consumer can check.

**Batches.** A batch is one table's worth of rows: `num_rows()` and `byte_size()` only.
Two implementations behind the backend-independent trait: `CpuBatch` wraps an Arrow
`RecordBatch`; `GpuBatch` wraps a `u64` handle to a resident `cudf::table`, plus the
session reference its `Drop` needs.
The draft's `temporary_byte_size` is gone — production-time scratch is accounted at call
time (below), not attached to a value that outlives it. Ownership is by move:

- every executor method takes `Batch` by value — reuse after consumption is a compile
  error, not a runtime throw;
- `GpuBatch: !Clone` and `CpuBatch: !Clone` (symmetry is worth more than the free
  `RecordBatch` clone nobody needs; a future dual consumer writes an explicit `copy()` —
  [#140](../tickets.md#t140));
- `GpuBatch` holds a session reference and its `Drop` releases the handle; a handle
  consumed by an FFI call skips `Drop` (`ManuallyDrop` at the one helper that owns the
  consume boundary) because C++ erased it.

**Executors — fused call interfaces.** Every state transition emits in the same call, so
there is no wrong interleaving to construct and output timing is a pure function of the
call sequence:

```rust
struct CallStats { scratch_bytes: Option<usize> }   // measured; Some on CPU, None on GPU

trait Executor {
    fn resident_bytes(&self) -> usize;                       // state held between calls
    // pre-call model; may consult self, so accumulators include their state. Calls with
    // no input batch (mark_done, finish) are modeled with n_rows = 0, n_bytes = 0.
    fn scratch_bytes(&self, n_rows: u64, n_bytes: usize) -> usize;
}

trait ExecExecutor: Executor {
    fn exec(&mut self, batch: Batch) -> (Batch, CallStats);
}
trait BatchAccumulatorExecutor: Executor {
    fn accumulate_and_fetch(&mut self, batch: Batch) -> (Vec<Batch>, CallStats);
    fn mark_done_and_fetch(&mut self) -> (Vec<Batch>, CallStats);
}
enum LaneEvent { Batch(Batch), Done }
trait PartitionAccumulatorExecutor: Executor {
    // one call per lane event — the shape round-robin driving actually produces;
    // the call delivering the last lane's Done is the emitting call
    fn accumulate_and_fetch(&mut self, partition: usize, event: LaneEvent)
        -> (Vec<Batch>, CallStats);
}
trait PartitionEmitterExecutor: Executor {
    fn emit(&mut self, batch: Batch) -> (Vec<Batch>, CallStats);   // exactly N outputs, some empty
}
trait JoinExecutor: Executor {
    fn set_build(&mut self, batch: Batch) -> CallStats;
    fn probe_and_fetch(&mut self, batch: Batch) -> (Vec<Batch>, CallStats);
    fn finish_and_fetch(&mut self) -> (Vec<Batch>, CallStats);
}
trait SourceExecutor: Executor { fn next_batch(&mut self) -> Option<(Batch, CallStats)>; }
```

There is no sink trait: `GpuUnload` is an ordinary `ExecExecutor` whose output happens to
be a `CpuBatch` (the `Batch` trait is backend-independent, so `GpuBatch` in →
`CpuBatch` out is just a signature), and the driver collects the root node's output
batches. `NodeKind::Sink` remains a plan-level fact.

**Instantiation model.** Lane-scoped categories — Source, Exec, BatchAccumulator, Join —
get one executor instance per (node, lane), created when the driver first enters that
lane; PartitionAccumulator and PartitionEmitter instances are one per node, since they
are the cross-lane points. The enforcer's `Σ resident_bytes()` runs over instances, not
nodes.

A `PartitionAccumulator` may buffer arbitrarily many input batches internally
(`GpuMergeSortedPartitions` does) and must account them in `resident_bytes()` — buffering
is executor state, not a taxonomy change.

**From node to executor.** The driver needs to know, per node, which trait to drive and
which backends exist. Each `GpuNode` exposes its category with the backend pair inside:

```rust
enum NodeExecutors {
    Source(ExecutorBackends<dyn SourceExecutor>),
    Exec(ExecutorBackends<dyn ExecExecutor>),
    BatchAccumulator(ExecutorBackends<dyn BatchAccumulatorExecutor>),
    PartitionAccumulator(ExecutorBackends<dyn PartitionAccumulatorExecutor>),
    PartitionEmitter(ExecutorBackends<dyn PartitionEmitterExecutor>),
    Join(ExecutorBackends<dyn JoinExecutor>),
    // GpuMergePartitions, GpuUnion, GpuInterleave — routing only, no backends
    BatchForwarder(Box<dyn BatchForwarder>),
}
// GpuNode::make_executors(&self) -> NodeExecutors — a fresh instance set per call,
// so the driver can instantiate per lane.

// Routes whole batches into a new lane numbering; never touches rows, never buffers.
// No CPU/GPU backends and no CallStats — routing is driver work, and a batch's bytes
// are already accounted as driver-held in-flight.
trait BatchForwarder {
    // the (child index, child lane) pairs feeding output lane p, in service order
    fn sources_of(&self, out_lane: usize) -> Vec<(usize, usize)>;
}
// GpuMergePartitions: lane 0 ← [(0,0), (0,1), …, (0,N-1)]     (N → 1)
// GpuUnion:          lane k ← [(i,j)]                          (exactly one source)
// GpuInterleave:     lane p ← [(0,p), (1,p), …, (k-1,p)]       (child-major)
```

One driver arm serves all three: a visit to output lane p cycles `sources_of(p)` in
listed order, forwarding one batch per visit, skipping `Pending` sources, retiring
`Exhausted` ones — the merge's round-robin and the interleave's per-lane child rotation
are the same rule applied to different mappings, stated once here instead of once per
node.

`ExecutorBackends<T>` holds constructors for the CPU and GPU implementation; a
`BackendSelector` trait picks one (`CpuBackendSelector`, `GpuBackendSelector`, and mock
selectors in driver unit tests — one `select` method per category). The rust-only tier
boundary holds structurally: rust-only targets never pull the GPU backend — GPU executor
types and the selector's GPU arm are gated the same way the legacy GPU executors are.

## Drivers

> Rework pending. This section is written pull-based — nodes pulling from the nodes
> below — and the execution model is being redesigned to be DFS-like: partitions run at
> the same time within one node, and batches are pushed up the tree as far as they can
> go, rather than pulled. The T0 Python prototype exists to settle that design; the
> visit contract, backpressure rule and diagrams below are the pull-based formulation
> and will be rewritten from the prototype's findings. The parts that survive either
> model: the two-driver split and chunk boundaries, determinism requirements, bounded
> queues at the shuffle, accumulator state as the home of mandatory residency, and the
> flow-and-backpressure test surface.
>
> The prototype's scheduler and its findings so far are in
> [`scripts/exec_model/README.md`](../../scripts/exec_model/README.md): a node is
> runnable when any of its partitions can make progress, the runnable node with the
> smallest height (distance to the root) runs, ties break leftmost, and every partition
> of that node runs — plus one backpressure rule, that a join in its build phase holds
> back its whole probe subtree. Three of the findings bear directly on this section:
> queues are self-bounding at one batch per lane, so the cap-Q mechanism is unnecessary;
> `Pending` does not exist in a model where runnability is a predicate evaluated before
> the call, so the three-valued visit contract goes with it; and the join hold is what
> makes the bound unconditional, since a join cannot drain its probe side until the
> build is set.

Two drivers, both single-threaded, pull-based, deterministic, generic over
`BackendSelector`:

- **`batch_single_partition_driver`** drives a chain of non-partition-breaking operators
  within one lane, including batch accumulators. Chains are exactly linear by
  construction: after the trait pass every Source/Exec/BatchAccumulator node has at most
  one child (both join families live on `JoinExecutor`, union/interleave are
  cross-lane), so a chunk is a pipe, never a tree.
- **`batch_partitioned_driver`** breaks the tree into chunks the single-partition driver
  accepts, and owns everything cross-partition: `GpuMergePartitions` (round-robin
  polling), `GpuEmitPartitions` fan-out, joins (which pin both children to the same
  partition layout and run two-phase per lane — drain the build chain, then stream the
  probe), union/interleave relabeling. Plans are trees (CTE references are inlined,
  #101), so chunks never share an upstream and chunk sequencing cannot deadlock.

**Visit contract and backpressure.** A lane visit returns `Batch`, `Pending`, or
`Exhausted`. The third outcome plus a bounded queue per emit output lane is the one
mechanism that handles both shuffle pathologies — hash skew (a lane that receives
nothing all query cannot be allowed to drive the upstream to exhaustion filling its
siblings' queues) and accumulator-ended lanes (which yield no output until their stream
is done, so they must make progress one batch per visit, parking bytes in
`resident_bytes()`-visible executor state where partial-aggregate compaction can shrink
them). Empty scatter outputs are dropped at the emit — starvation is `Pending`'s job,
so nothing empty ever traverses a chain.

```
                 batch_partitioned_driver
   ───────────────────────────────────────────────────────────────
   pre-shuffle chunk, one lane            post-shuffle lanes (N=4)
   (run by batch_single_partition_driver)
                                          ┌▶ q0 ─▶ chain lane 0 ─▶ …
   loader ─▶ filter ─▶ … ─▶ [Merge] ─▶ Emit┼▶ q1 ─▶ chain lane 1 ─▶ …
                                          ├▶ q2 ─▶ chain lane 2 ─▶ …
             queues: cap Q batches each   └▶ q3 ─▶ chain lane 3 ─▶ …

   visit(lane p):
     q[p] non-empty ──────────▶ take one batch ─▶ push through lane p's chain
         │                                          ├─ chain yields ──▶ Batch
         │ empty                                    └─ parked in
         ▼                                             accumulator ──▶ Pending
     upstream exhausted? ─ yes ─▶ mark_done down the chain
         │ no                       ├─ flushed output ▶ Batch
         ▼                          └─ nothing left ──▶ Exhausted
     any sibling q at cap Q? ─ yes ─▶ Pending
         │ no
         ▼
     pull ONE upstream batch ─▶ scatter ─▶ drop empties ─▶ enqueue
         └▶ retry q[p] once: non-empty ▶ (take, as above) · empty ▶ Pending
```

Progress is guaranteed (every cycle either consumes a batch or pulls exactly one
upstream batch), raw queued data is bounded by N×Q batches per emit, the queues are
driver-held in-flight batches the enforcer already counts, and everything is
single-threaded deterministic. `Pending` propagates through merge visits, which skip
that input lane for the cycle. `GpuInterleave` and `GpuUnion` follow the identical rule
through `BatchForwarder::sources_of` (see [Traits](#traits)) — one
driver arm cycles a lane's declared sources in order, whatever the node.

**Flow and backpressure are a first-class test surface for both drivers**, exercised
with mock operators behind the mock `BackendSelector` (scripted batch counts, sizes,
skew patterns, accumulator behavior — no real executors). The cases that must hold, each
asserting pull counts, queue bounds, `Pending` behavior and batch/handle release:
skewed emit (starved lane, hot lane); accumulator-ended lane progress; merge-sorted over
a skewed emit (no livelock: every cycle lands a batch or pulls exactly one); limit early
exit as a release case (pulls cease through merges and emits, every in-flight batch
dropped); union with heterogeneous children (one exhausted, one `Pending`); two-phase
join with emits on both sides (probe-side queues untouched — empty, not merely bounded —
until the build drains); nested shuffles holding every emit's bound simultaneously.

Early exit: for a root-adjacent `GpuLimit` the driver counts emitted rows against the
plan's interval and stops pulling upstream the moment it is satisfied — half the point of
batched loading for limit queries. The executor never signals "done"; satisfaction is
driver logic keyed on the plan, which is why `GpuLimit` needs no executor of its own on
that path (see the limit lowering rule).

Both drivers take the resident-accounting hooks below and fail the query when the
enforcer trips. An FFI error is query-fatal: the C++ side resets the whole session and
every resident handle with it (`cpp/src/gpu_executor.cpp` ~L192) — there is no mid-flight
retry with smaller batches (that is #142's adaptive future).

## Memory accounting

Prevention lives at plan time; detection at run time.

**Plan time.** An estimator pass computes `estimated_max_resident_size` per node in
rows × row-width vocabulary (the `subtree_max_row_bytes` family), rendered in the
`.plan.mem.txt` goldens. Because `GpuMergePartitions` polls round-robin, all N lanes are
live at once, and the estimator charges the full multi-lane section — N × (per-lane
executor state + one in-flight batch) between loader/emit and the merge point. This also
makes today's estimates valid for a future parallel driver, which costs exactly that.

**Run time.** The driver keeps a running total incrementally:

```
resident = Σ byte_size of driver-held in-flight batches
         + Σ cached resident_bytes() over live executors
```

Per call: pre-check `resident + scratch_bytes(n_rows, n_bytes)` against the budget (the
model may consult `&self`, so accumulators can include their state); execute; then remove
consumed inputs, add outputs at actual `byte_size()`, refresh the one executor's
`resident_bytes()` delta, and post-check. The measured `CallStats.scratch_bytes` is
CPU-only (`Some` on CPU, `None` on GPU — GPU internals are invisible without RMM hooks)
and exists to keep the model honest: every executor's unit tests assert model ≥ measured
wherever measurement exists. The enforcer's contract is "fail cleanly when the accounted
peak exceeds budget", not "the budget is never exceeded" — same class of guarantee as the
legacy `ResidentEnforcer`.

## GPU execution through the frozen FFI

A `GpuBatch` is a `u64` handle to a resident `cudf::table`, exactly as legacy partitions
are. The review established the FFI facts that make the whole mode drivable with one
additive entry point:

- `execute_node` is stateless per seq — the only state is the handle registry, inputs are
  consumed per call, outputs get fresh handles. Calling the same seq once per batch is
  legal, and one handle per call satisfies `execute_one`'s consumed==provided check for
  single-child nodes.
- The collapse arm concatenates whatever k handles it is passed; the k-way merge arm
  merges any k>1 sorted handles and applies `fetch`; the repartition arm scatters any
  concatenated input into the plan-declared N. None of them cross-check handle counts
  against the plan tree, and `n_children` is caller-supplied and unverified.
- Stats come back per output handle per call; aggregation across calls is entirely the
  new driver's job (`NodeMemoryStats` per node = fold over its calls).

The translation layer therefore emits, alongside the `GpuNode` tree, a **recipe plan**: a
structurally valid FlatBuffers plan in the legacy vocabulary whose nodes exist to be
addressed by seq — the fbs is a menu of parameterized kernels, not the execution
structure. The mapping:

| GpuNode | fb seqs emitted | driven as |
|---|---|---|
| `GpuLoadParquet` | `GpuScan` | `peacock_executor_execute_scan_rowgroups(seq, row_groups…)` once per batch — the additive entry point, plumbing the existing `row_groups_override` parameter (`cpp/src/peacock/operators.h` ~L32) to the ABI |
| `GpuFilter` / `GpuProject` / `GpuAggregate` | same-kind node | generic map arm, one call per batch |
| `GpuSort` | `GpuSort` | map arm per batch; per-batch `fetch` for top-N |
| `GpuAccumulateBatchesAndSort` | `GpuSort` + `GpuSortPreservingMerge` | per-batch sort calls, then one merge-arm call at done |
| `GpuMergeSortedPartitions` | `GpuSortPreservingMerge` | one merge-arm call over all sorted handles, partition-major order |
| `GpuCoalesceAllBatches` | `GpuCoalescePartitions` | one collapse-arm call over the partition's batch handles |
| `GpuAggregateBatches` | `GpuCoalescePartitions` + `GpuAggregate` (final) | concat call + final-aggregate call at done |
| `GpuEmitPartitions` | `GpuRepartition(Hash, 1→N)` | repartition arm, one call per batch → N handles |
| `GpuJoin` | `GpuHashJoin`, plus finish-pass seqs (key project, concat, anti/semi join, pad project) per #136 | map arm per (partition, probe batch) |
| `GpuCrossJoin` / `GpuNestedLoopJoin` | same-kind node | one map-arm call |
| `GpuLimit` | none, or `GpuCoalescePartitions` + `GpuLimit` | see the lowering rule below |
| `GpuMergePartitions` / `GpuUnion` / `GpuInterleave` | none (union casts are `GpuProject` seqs) | `BatchForwarder` routing in the driver, zero FFI calls |
| `GpuUnload` | none | `peacock_result_from_handle` per handle |

This mapping is a first-class deliverable: documented here, unit-tested (each `GpuNode`
kind → expected seq set and call pattern), because it is the load-bearing trick that
keeps C++ frozen. The fb names in the table are the pre-T1 ones; after the T1 rename
they read `Cudf*` (`CudfScan`, `CudfRepartition`, …) and this table is updated in that
commit.

**The limit lowering rule.** A per-batch `GpuLimit` call cannot be correct: the fb node's
skip/fetch are frozen per seq, so every batch would be truncated to the same bounds
(two batches → 2× the limit), and the right bound for the last batch is a runtime value
no frozen node can carry. Legacy never sees this because a legacy partition is one batch.
Two lowerings instead: a **root-adjacent** limit (feeding only `GpuUnload` — the common
case) emits no fb seq at all: the driver counts output rows, stops pulling upstream when
the interval is satisfied, and trims skip/tail on the CPU after unload — early exit
preserved. A **mid-plan** limit whose output feeds further GPU work gets a
planner-inserted `GpuCoalesceAllBatches` and then one `GpuLimit` call with exact bounds —
correct, at the price of materializing the limit's input (no early exit on that path).

`peacock_executor_execute_scan_rowgroups` is the one C++/header change:
`(executor, seq, const uint32_t* row_groups, uint64_t n, uint64_t* out_handle,
PeacockNodeStats* out_stats)` — reads the named `GpuScan` seq's options but overrides its
row-group list for this call. Additive next to the existing symbols; legacy paths
untouched.

## Determinism rules

Batch boundaries are a pure function of the plan: the loader's come from the
partitioner's committed mapping, Exec ops are 1:1, accumulators emit at defined points.
Given that, the remaining scheduling freedoms are pinned:

- **Every `BatchForwarder` lane cycles its `sources_of` list in order**
  — for `GpuMergePartitions` that is round-robin over partitions by index; a source
  yields one batch, `Pending` (skipped this cycle — see the visit contract), or
  `Exhausted` (retired from the rotation). Emission order is arrival order under this
  schedule.
  Chosen over drain-in-partition-order deliberately: it keeps "partition = a lane that
  makes progress alongside the others" true, at the honest cost of N live lanes — which
  the estimator charges, and which matches what a parallel driver will cost anyway. If a
  driver ever goes parallel, emission order is preserved with a reorder buffer or the
  goldens regenerate deliberately.
- **`cudf::merge` tie order**: input tables are passed partition-major (partition 0's
  batches in stream order, then partition 1's, …) regardless of arrival order.
- Order pinning is part of *result* determinism, not just golden stability: float
  aggregation sums in stream order, so an unpinned order changes low bits.

## Goldens, registry, widget

**Device labels**: `bp-<tp1|tp4>-<single|batched>`, with the budget tier suffixed for
execution goldens as in legacy (`q1.bp-tp4-batched-mini.cpu.txt`). The
`partition_mode`-style label lookups stay explicit parameters at call sites, per the
coding-style case.

**Plan goldens** (4 modes: bp-tp1-single, bp-tp1-batched, bp-tp4-single, bp-tp4-batched):
one file per mode holding all queries — `goldens/<bench>/<mode>.plans.txt` — because the
per-query files would be small and numerous. Every node renders its `PartitionLayout`
(count, batch layout, key distribution, sort order); the loader renders
`partition_groups=[...]`; `estimated_max_resident_size` is attached per node. The memory
sections for all queries are pulled into a sibling `<mode>.plan.mem.txt`, in a layout
matching the new estimator's fields.

**Execution goldens** follow the legacy flow exactly: the CPU executor authors
`.cpu.txt`/`.result.txt` under `UPDATE_CANONICAL=1`, `.cost.txt` derives from `.cpu.txt`
× `cost_model.conf`, the GPU asserts read-only. New-mode `.cpu.txt` shows full
`PartitionLayout` per node and per-partition `output_batch_count`; ~10 queries across the
corpus additionally freeze `batch-info.cpu.txt` with per-batch row counts and byte sizes.
`cost_model.conf` gains the new node names (every node type appearing in a `.cpu.txt`
must be in exactly one category — the conf enforces it).

**Registry and CI**: new columns in `cost-registry.csv` for the new mode's plan/CPU/GPU
enablement, with inventory tests in both directions like the existing six; new test
targets named in `pipeline.yml` steps (the `test_ci_coverage` guard enforces this
automatically).

**Widget**: two new tables (TPC-H, TPC-DS) repeating the existing structure — plan (four
cells, one per mode), CPU (four), GPU (four) — fed from the new CSV columns. Window
queries render plan ✗ (#143). Peacock cost, DuckDB cost and ratio columns mirror the
legacy ones; when not all four modes are enabled, cost uses the last mode in the sequence
of four where CPU execution is enabled.

# Implementation plan

Tasks in dependency order; each is one developer hand-off with its own proving tests.
Legacy tests stay green throughout — every task that touches shared code runs the
affected legacy subsets (one query per mode/tier per binary plus the rust-only tier, per
build-test.md).

**T0 — Python prototype of the whole execution model.** All node types and both drivers
in Python, operators built with pandas, plans hand-built (no DataFusion, no planner) —
an emulation of tree execution whose purpose is to settle the DFS-like push model
(partitions running at the same time within one node, batches pushed up as far as they
can go) before any Rust exists. Includes the memory enforcer with the accounting formula
and a mock scratch model, so trip behavior is exercised too. Stress tests: empty
partitions; operators emitting empty batches; `GpuCoalesceBatches[target=XX]` injected
at arbitrary tree positions (the node is deferred to #139 in Rust v1, but the prototype
must prove the drivers tolerate it anywhere); skewed hashes; the full
flow-and-backpressure surface from the drivers section; determinism (two runs, identical
batch traces). Validation scope: partitioning checks and `SingleBatch` constraint checks
are in scope; schema checks are not. The hand-built corpus: 3–4 TPC-H plans (a scan
filter aggregate, a two-sided shuffle join, a top-N sort, a keyless aggregate) and ~10
interesting TPC-DS plans — unions and interleaves, rollup aggregates, semi/anti joins,
cross joins beside aggregates, multi-shuffle trees. Deliverables: the prototype under
`scripts/exec_model/` with tests wired into the cost-report job, and a rewrite of the
drivers section from its findings — the pull-based formulation there is provisional
until this lands.

Landed so far: the trait set, both drivers, the enforcer, driver tests over mock trait
implementations, and the pandas-backed operators — filter, project, sort, the aggregate
sequence with its partial/final decomposition, the accumulators, the hash scatter, the
join capability matrix, and the T2 row-group partitioning policy. Every query is checked
against a single-shot oracle at five partitioning configs, which is the prototype's
version of two-engine correctness (`scripts/exec_model/`, findings in its README). Still
open under T0: the `GpuLimit` early-exit path, the estimator, and the hand-built TPC-H /
TPC-DS plan corpus.

**T1 — flatbuffer operation-name refactor.** Nine of the fifteen legacy node-kind names
(`GpuFilter`, `GpuProject`, `GpuSort`, `GpuAggregate`, `GpuCrossJoin`,
`GpuNestedLoopJoin`, `GpuUnion`, `GpuLimit`, `GpuCoalesceBatches`) collide with the new
mode's node names. Rename the fbs tables and `PlanNodeKind` variants to a `Cudf` prefix
(`CudfScan`, `CudfFilter`, …) so the two vocabularies are visually distinct everywhere —
schema, generated code, the C++ `node_type()` switches and serializer identifiers on the
Rust side. A pure rename: FlatBuffers wire bytes carry no table names and enum ordinals
do not move, so the proof is `plan_bytes.sha256` staying byte-identical with no
regeneration, plus green legacy subsets. The same commit sweeps the llm-wiki references
(architecture.md's fb names, affected tickets, and the recipe-plan table in this spec).

**T2 — ParquetBatchPartitioner.** The pure policy class and its unit tests: fewer
survivors than N; N=3; single row group over target; batching off ⇒ one batch per chunk;
empty survivors (explicit error — the fbs "empty map means legacy single partition"
convention must not leak in); balance bound (max−min partition rows ≤ one row group);
fixed-output determinism case. No planner integration yet.

**T3 — node and trait skeleton.** `GpuNode`, `PartitionLayout` (with the three-valued
`SortOrder`), `Schema` with semantics annotations, `Batch`/`CpuBatch`/`GpuBatch` shells
with the move/`!Clone`/`Drop` rules, executor trait definitions with `CallStats`,
`BackendSelector`. Traits in their own files per coding-style. Compiles under rust-only
with the GPU side gated. Unit tests: `SortOrder` canonicalization, layout equality.

**T4 — translation layer, single-partition shapes.** DataFusion physical plan (tp1) →
`GpuNode` tree for chains: load, filter, project, sort (+fetch), limit, coalesce-all,
single/final aggregates, cross/nested-loop joins. Per-node-kind conscious mapping;
unrecognized node ⇒ plan-time error naming it; window ⇒ the #143 refusal. Unit tests
assert emitted constructs for simple queries.

**T5 — translation layer, partitioned shapes.** tp4: shuffle points → Merge+Emit, the
aggregate sequence with its shortcuts and the gid rule, join side normalization (type
remap + column-order-restoring project) and build-side coalesce insertion per the
capability matrix, union/interleave with explicit branch-cast projects. The
`hashKeys ⊆ group columns` structure is produced here (validated in T8). Unit tests per
construct in tp1 and tp4, including side-swap cases.

**T6 — estimator pass and plan goldens.** `estimated_max_resident_size` per node
(rows × width vocabulary, N-lane charging), `target_batch_bytes` derivation feeding T2's
partitioner, integration as `plan_batch_partitioned()`. Canonize all four
`<mode>.plans.txt` + `<mode>.plan.mem.txt` for TPC-H and TPC-DS (minus #23's four and
window queries, which appear as refusals).

**T7 — schema registry.** `output_schema()` on all nodes with column semantics
annotations. Unit tests: hand-crafted plans produce expected types and annotations;
decimal precision/scale fidelity through project/aggregate/union-cast paths.

**T8 — validation.** `validate_schemas_and_partitions()` on every node type: partition
topology, key-distribution subset rule, `BatchSorted`/`PartitionSorted` requirements
(merge requires ≥ BatchSorted; limit-after-sort requires PartitionSorted), `SingleBatch`
expectations (join build, cross/nlj inputs), captured-index checks. Unit tests: manually
constructed wrong combinations error, right ones pass; then run validation over every
canonized corpus plan from T6.

**T9 — additive scan ABI.** `peacock_executor_execute_scan_rowgroups` in
`gpu_executor.cpp` + `peacock_gpu.h` (the only planned C++/header change; any further
surface change goes through a proposal to the human, per the constraint section),
plumbing `row_groups_override`; Rust binding; `GpuBatch` handle plumbing (session ref,
`Drop` release, `ManuallyDrop` consume boundary). Tests: a C++ gtest in the plan-executor
suite reading disjoint row-group subsets and asserting union == whole-scan; Rust FFI
smoke on shad-gpu.

**T10 — Exec executors, CPU and GPU.** Filter, project, per-batch sort, aggregate
(partial/single), unload (`GpuBatch → CpuBatch`), the mid-plan limit call. Reuse legacy operator code by extracting
helpers — never by calling into strip/wrapper machinery. Per executor: CPU vs
hand-crafted oracle, GPU vs CPU, empty-batch cases, `CallStats` model ≥ measured on CPU.
CPU and GPU tests in separate targets so CI hosts split them.

**T11 — accumulators.** `GpuCoalesceAllBatches`, `GpuAggregateBatches` (both finals),
`GpuAccumulateBatchesAndSort`, `GpuMergeSortedPartitions`. Edge cases: zero batches, one
batch, ties for the merge (partition-major stability), fetch interaction, large batch
counts, gid-carrying aggregate merges.

**T12 — partition ops and joins.** `GpuEmitPartitions` (per-batch scatter, N=3 and large
N, empty outputs for skewed hashes), `GpuMergePartitions` round-robin rule,
`GpuJoin` with `set_build`/`probe_and_fetch`/`finish_and_fetch`, plus cross and
nested-loop joins on the same trait: the full capability matrix as a test table — per
(type × layout): stream-vs-refuse, correctness vs the single-batch oracle, the GPU
finish pass via key accumulation (#136), null_equals_null on the finish join.

**T13 — drivers and enforcer.** Both drivers with mock backends via the selector;
`batch_partitioned_driver` tested against a mocked single-partition driver; round-robin
determinism cases; the full flow-and-backpressure suite from the drivers section, with
mock operators: skewed emit (every queue ≤ cap, starved lane returns `Pending` without
draining the upstream), accumulator-ended lane (one batch per visit, bytes in
`resident_bytes()`), merge-sorted over a skewed emit (no livelock), limit early exit as
a release case (pulls cease through merges and emits, all in-flight batches dropped),
union with one child exhausted and one `Pending`, two-phase join with emits on both
sides (probe queues empty until the build drains), nested shuffles holding all bounds
simultaneously, interleave per-lane child rotation; the accounting formula with
pre/post checks; enforcer trip ⇒ clean query failure; FFI-error ⇒ query-fatal
semantics.

**T14 — recipe-plan serialization and GPU integration.** The GpuNode → fb-seq mapping
table implemented and unit-tested (expected seq sets and call patterns per node kind);
driver-side stats folding across calls into `NodeMemoryStats`; first end-to-end queries
on shad-gpu (scan → filter → aggregate; a join; a sort+limit), GPU vs CPU.

**T15 — rollout.** New macros `cpu_batch_partitioned_result_test` /
`gpu_batch_partitioned_test`; `.cpu.txt`/`.cost.txt` wiring incl. `cost_model.conf`
entries for the new node names; `batch-info.cpu.txt` for ~10 queries; registry columns +
inventory tests; `pipeline.yml` steps (satisfying `test_ci_coverage`); widget tables with
the cost-column rule and the #143 plan ✗ cells. Then query-by-query enablement across the
corpus, tickets filed per newly discovered blocker, as with the legacy rollout.
