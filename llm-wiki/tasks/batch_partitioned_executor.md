# batch-partitioned executor

A new planning and execution mode in which a GPU partition holds a *stream of batches*
instead of one resident table. The motivating pipeline: load → filter at 1% selectivity →
aggregate into few groups. Today's partitioned mode materializes the whole scan on the
GPU before the filter ever runs; with batches, small slices flow through the filter and
only the aggregate's state stays resident, so the query fits in a VRAM budget the table
does not.

Status (2026-08-11): design final, and the parts of it that could only be settled by
running are settled. **A Python prototype of the whole execution model lives in
[`scripts/exec_model/`](../../scripts/exec_model/README.md)** (task T0): the trait set,
both drivers, the scheduler, the enforcer, pandas-backed operators checked against a
single-shot oracle, and a plan rewriter that re-runs each query at every partitioning and
batching shape. Where this document and the prototype disagree the prototype is wrong and
gets fixed; where the document was written before the prototype existed, it has been
rewritten from what the prototype established — the Drivers section most of all.

This is a large task spanning many branches, so the spec lives on master, committed ahead
of the work, rather than riding any one task branch. Deferred work is ticketed, not
latent: this file plus `tickets.md` is the complete state.

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
new mode honest about what it needs. **Three additive symbols are approved and are the
working set** (see [GPU execution](#gpu-execution-through-the-frozen-ffi)):

| symbol | why the frozen surface blocks the design without it |
|---|---|
| `peacock_executor_execute_scan_rowgroups()` | the scan arm emits every map entry in one FFI call (`cpp/src/node_session.cpp` ~L123), so incremental loading is impossible |
| a row interval on `peacock_result_from_handle()` | a root-adjacent limit would otherwise export whole batches and drop rows on the CPU, shipping an unbounded `skip` prefix over PCIe to throw it away |
| `peacock_executor_slice_handle()` | a mid-plan limit would otherwise have to hold every row ahead of the ones it wants, because frozen bounds are only correct against a table starting at row 0 of the stream |

The expectation is that the rest of the work fits **within** that set: everything else maps
onto existing arms driven creatively, and a task that seems to need a fourth symbol should
first be re-examined for a way through the three. Adding one is still possible and is not a
failure — the design already grew from one to three, each time because a real shape could
not be expressed — but it is a decision, not a workaround. If development finds the
constraint too tight anywhere (candidates already known: #136's match bitmap or persistent
build, #142's split entry point), the developer does not work around it silently and does
not change the surface on their own: the coordinator raises a concrete proposal to the
human naming the blocked task, the smallest additive change that unblocks it, and what the
workaround would cost instead.

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
driver-level relabeling plus explicit per-branch cast projects;
`GlobalLimitExec`/`LocalLimitExec` → the limit lowering rule, which emits either no node at
all (the interval goes on `GpuUnload`) or a `GpuLimit` over an inserted
`GpuMergePartitions`. The layer is unit-tested
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
| `GpuAccumulateBatchesAndSort` | BatchAccumulator | accumulates sorted batches, one `cudf::merge` at done; output one batch, `SingleBatch` + `BatchSorted` (so stream-sorted). No streaming emission — cuDF has no primitive; ranged emission is [#138](../tickets.md#t138) |
| `GpuMergeSortedPartitions` | PartitionAccumulator | input: N partitions, `MultipleBatches` allowed, `BatchSorted` required; all k·m sorted batches into one `cudf::merge`, `fetch` applied; output: 1 partition, one batch, `SingleBatch` + `BatchSorted` (so stream-sorted) |
| `GpuCoalesceAllBatches` | BatchAccumulator | concatenates a partition's batches into one at done |
| `GpuMergePartitions` | BatchForwarder | N partition streams → 1, forwarding each batch as visited, round-robin (see [Determinism](#determinism-rules)); accumulates nothing, no backend calls |
| `GpuEmitPartitions` | PartitionEmitter | 1 → N per batch by hash scatter; streaming, one call per input batch |
| `GpuAggregate` | Exec | partial (or single-shortcut final) aggregation of one batch |
| `GpuAggregateBatches` | BatchAccumulator | merges pre-aggregated batches; emits at done |
| `GpuJoin` | Join | capability matrix below. Carries the optimizer's **cardinality estimate** (output rows / probe rows, as `CardinalityEstimator` in `gpu_rule.rs` returns) as a node property. The executor is constructed from the node, so the estimate reaches `scratch_bytes` through `&self` and the merged frame — sized by output cardinality, which the signature cannot derive — is modelled without changing the trait. Constant 1.0 until #19 |
| `GpuCrossJoin`, `GpuNestedLoopJoin` | Join | two inputs, so `JoinExecutor`, not Exec: `set_build(left)`, probe right. Cross and inner NLJ stream the right side (same mechanics as inner hash-join probes); non-inner NLJ keeps a single-batch probe — the #136 finish trick needs keys to accumulate and a predicate join has none. Both inputs 1 partition; broadcast variants are [#140](../tickets.md#t140) |
| `GpuUnion`, `GpuInterleave` | BatchForwarder | lane relabeling; branch decimal/type normalization becomes explicit per-branch `GpuProject` casts inserted by the planner — which is what makes union pure routing. Union sums its inputs' lane counts and clears `KeyDistribution`; interleave is chosen (as in DataFusion's `can_interleave`) only when every input carries the same hash distribution — output lane p is lane p of each input, so `KeyDistribution` is preserved |
| `GpuLimit` | BatchAccumulator — mid-plan only | start..limit over a **1-partition** stream (`GpuMergePartitions` beneath; the node checks it in `validate_schemas_and_partitions`), any number of batches. Streams and holds nothing: outside the interval a batch is released uncalled, inside it is forwarded untouched, and only the two that straddle its ends are sliced. Output layout follows the input. A **root-adjacent** limit is not a node at all: the interval becomes `GpuUnload`'s. See the lowering rule below the recipe table |
| `GpuUnload` | Unload | `GpuBatch` in, `CpuBatch` out (Arrow IPC export per handle, over a row range the driver supplies); 1:1 per batch, `NodeKind::Sink` at plan level. Carries the **root-adjacent limit's `skip`/`fetch`** as node properties — the interval belongs to the boundary crossing, because it is a statement about which rows are worth moving across it |

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
// A sink structurally has no layout and no schema; everything else always has both.
enum NodeKind {
    Source { layout: PartitionLayout, schema: Schema },
    Intermediate { layout: PartitionLayout, schema: Schema },
    Sink,
}

enum KeyDistribution { NotSpecified, ByHash { hash_keys: Vec<u32> } }   // spark murmur3, seed 42

enum SortOrder {
    NotSpecified,
    BatchSorted { columns: Vec<ColumnOrder> },      // each batch sorted; batches unordered
}
// Two-valued on purpose. A whole-stream order is BatchSorted meeting SingleBatch, derived
// as PartitionLayout::is_stream_sorted() — see below.

enum BatchLayout { SingleBatch, MultipleBatches }

struct PartitionLayout {
    n: usize,
    key_distribution: KeyDistribution,
    sort_order: SortOrder,
    batch_layout: BatchLayout,
}
impl PartitionLayout {
    // Whole stream ordered, not merely each batch. Derived, so nothing can disagree.
    fn is_stream_sorted(&self) -> bool {
        matches!(self.sort_order, SortOrder::BatchSorted { .. })
            && matches!(self.batch_layout, BatchLayout::SingleBatch)
    }
}

trait GpuNode {
    fn kind(&self) -> &NodeKind;   // layout and schema live inside it
    fn children(&self) -> Vec<&dyn GpuNode>;
    // Checks children's schemas, partition topology, key distribution, sortedness and
    // batch layout against this node's requirements; validates captured column indices
    // (group keys, aggregate lists, two-phase state columns) against child schemas.
    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError>;
    // Some for a mid-plan GpuLimit and for a GpuUnload that absorbed a root-adjacent one;
    // None everywhere else. See the limit lowering rule.
    fn row_interval(&self) -> Option<RowInterval> { None }
}
```

Node-owned validation runs **before** the generic structural rules, because a node knows
what it needs of its children and can name the fix where a generic rule can only say what
is wrong: a limit over four lanes should read "the planner inserts `GpuMergePartitions`
below it", not "this category is 1:1 per lane".

Carrying layout and schema in `NodeKind` rather than as two `Option`s that must be `None`
together removes the one thing a caller could get wrong: the prototype needed a run-time
error for a non-sink that declared no layout.

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
struct CallStats { scratch_bytes: Option<usize> }   // measured; None when not instrumented

trait Executor {
    fn resident_bytes(&self) -> usize;                       // state held between calls
    // pre-call model; may consult self, so accumulators include their state. Calls with
    // no input batch (mark_done, finish) are modeled with n_rows = 0, n_bytes = 0.
    fn scratch_bytes(&self, n_rows: u64, n_bytes: usize) -> usize;
}

// One impl per backend, naming a concrete type for the batch and for every executor
// category. The driver is generic over it, so the GPU path monomorphizes: no vtable and
// no allocation for a batch, no selector object at run time.
trait Backend: Sized {
    type Context;          // GPU: the open NodeSession; CPU: ()
    type Batch: Batch;
    type Source: SourceExecutor<Self>;
    type Exec: ExecExecutor<Self>;
    type BatchAcc: BatchAccumulatorExecutor<Self>;
    type PartAcc: PartitionAccumulatorExecutor<Self>;
    type Emitter: PartitionEmitterExecutor<Self>;
    type Join: JoinExecutor<Self>;
    type Unload: UnloadExecutor<Self>;
}

trait ExecExecutor<B: Backend>: Executor {
    fn exec(&mut self, batch: B::Batch) -> (B::Batch, CallStats);
}
trait BatchAccumulatorExecutor<B: Backend>: Executor {
    fn accumulate_and_fetch(&mut self, batch: B::Batch) -> (Vec<B::Batch>, CallStats);
    fn mark_done_and_fetch(self) -> (Vec<B::Batch>, CallStats);   // no accumulate after
}
enum LaneEvent<B: Backend> { Batch(B::Batch), Done }
trait PartitionAccumulatorExecutor<B: Backend>: Executor {
    // one call per lane event — the shape round-robin driving actually produces;
    // the call delivering the last lane's Done is the emitting call
    fn accumulate_and_fetch(&mut self, partition: usize, event: LaneEvent<B>)
        -> (Vec<B::Batch>, CallStats);
}
trait PartitionEmitterExecutor<B: Backend>: Executor {
    fn emit(&mut self, batch: B::Batch) -> (Vec<B::Batch>, CallStats);   // exactly N, some empty
}

// Join, as a typestate: build -> probe -> done, each transition consuming the last state.
trait JoinExecutor<B: Backend>: Executor {
    type Probing: ProbingJoin<B>;
    fn set_build(self, batch: B::Batch) -> (Self::Probing, CallStats);
}
trait ProbingJoin<B: Backend>: Executor {
    fn probe_and_fetch(&mut self, batch: B::Batch) -> (Vec<B::Batch>, CallStats);
    fn finish_and_fetch(self) -> (Vec<B::Batch>, CallStats);
}

// Exhaustion consumes the source, so the driver's slot IS its liveness.
enum SourceStep<B: Backend> {
    Batch { batch: B::Batch, stats: CallStats, source: B::Source },
    Exhausted,
}
trait SourceExecutor<B: Backend>: Executor {
    fn next_batch(self) -> SourceStep<B>;
}
```

**Illegal calls are unrepresentable rather than checked.** Every method that ends a
protocol takes `self: Box<Self>`, so four run-time guards the prototype needed become
compile errors: probing before `set_build`, calling `set_build` twice, probing after
`finish_and_fetch`, and accumulating after `mark_done_and_fetch`. The source's
consuming step removes a fifth thing — the driver's own `finished` flag, which today
duplicates the executor's exhaustion and can disagree with it.

**Static all the way down on the production path.** `B::Batch` is a concrete type, so a
`GpuBatch` is its `u64` handle with no box and no vtable, and `Drop` is a direct call.
Backend choice is a turbofish at the entry point — `batch_partitioned_driver::<GpuBackend>(…)`
— not a `BackendSelector` consulted per node. The whole driver is instantiated twice, once
per backend, and the mock backend the driver tests use is a third instantiation.

Going static also *simplifies* the typestate: consuming methods take a plain `self`, and
`JoinExecutor` can carry an associated `type Probing`. Both had to be spelled around when
these traits were stored as `Box<dyn …>` — `self: Box<Self>` because a bare receiver is not
`dyn`-compatible, and a returned `Box<dyn ProbingJoin>` because an associated type forces
every `dyn JoinExecutor` to name it (`E0038` and `E0191` respectively; both were compiled to
confirm). Neither constraint survives the switch, so the declarations above are the simpler
form and they compile as written.

The cost is one type per category per backend: `B::Exec` has to cover filter, project,
sort and partial aggregate, so each backend defines an enum over its operators and
dispatches with a match. Limit and unload are not among them — the mid-plan limit is a
`BatchAccumulator` and unload has its own category, for the reasons below. That is the same shape the C++ side already has in
`run_op`'s switch, and a match on a closed enum is still static dispatch.

Trait objects remain where they cost nothing: the plan tree is `dyn GpuNode`, since
planning is not hot and the tree is heterogeneous.

Two illegal states stay run-time checked on purpose: `emit` returning other than N outputs
(N is a plan value, so const generics do not apply — the count is checked once inside the
returned type rather than at each call site), and a second `Done` for one lane, which would
need per-lane state in the type for no useful gain.

`GpuUnload` needs its own category, `type Unload: UnloadExecutor<Self>`, because it is the
one operator whose output type is not `B::Batch` — and, since the row range is a call
argument, the one whose call signature the limit rule reaches into:

```rust
trait UnloadExecutor<B: Backend>: Executor {
    fn unload(&mut self, batch: B::Batch, rows: RowRange) -> (CpuBatch, CallStats);
}

/// `length: u64::MAX` means to the end. Straight through to the fetch's new arguments.
struct RowRange { offset: u64, length: u64 }
```

`rows` is where a root-adjacent limit's trim lands. It is a call argument rather than
executor state because the count it derives from is *cross-lane*, and an `Unload` instance
is per lane — only the driver can hold that count (see [Drivers](#drivers)).

Under a backend-agnostic `Batch` this could be an ordinary `ExecExecutor` with a `GpuBatch`
in and a `CpuBatch` out; once `exec` is `B::Batch -> B::Batch` it cannot. That is an
improvement rather than a tax: unload is the only place data leaves the device, and it now
says so in the type. The driver still collects the root node's output batches, and
`NodeKind::Sink` remains a plan-level fact.

**Instantiation model.** Lane-scoped categories — Source, Exec, BatchAccumulator, Join —
get one executor instance per (node, lane), created when the driver first enters that
lane; PartitionAccumulator and PartitionEmitter instances are one per node, since they
are the cross-lane points. The enforcer's `Σ resident_bytes()` runs over instances, not
nodes.

A `PartitionAccumulator` may buffer arbitrarily many input batches internally
(`GpuMergeSortedPartitions` does) and must account them in `resident_bytes()` — buffering
is executor state, not a taxonomy change.

**From node to executor.** The driver needs to know, per node, which trait to drive. The
set is an enum of concrete types — no boxes, so the match compiles to a jump and the
executor is stored inline:

```rust
enum NodeExecutors<B: Backend> {
    Source(B::Source),
    Exec(B::Exec),
    BatchAccumulator(B::BatchAcc),
    PartitionAccumulator(B::PartAcc),
    PartitionEmitter(B::Emitter),
    Join(B::Join),          // ProbingJoin comes from set_build, not from the backend
    Unload(B::Unload),
    // GpuMergePartitions, GpuUnion, GpuInterleave — routing only, no backend
    BatchForwarder(Forwarder),
}
// Backend::executors_for(ctx: &Self::Context, node: &dyn GpuNode, lane: usize)
//   -> NodeExecutors<B>. A fresh instance set per call, so the driver instantiates per
//   lane; `lane` is needed because a loader's lane picks its own row groups out of the
//   partitioner's mapping.

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
listed order, forwarding one batch per visit, skipping sources with nothing queued and
retiring those whose producer has finished — the merge's round-robin and the interleave's
per-lane child rotation are the same rule applied to different mappings, stated once here
instead of once per node.

**Construction moves off the node and onto the backend**, as
`Backend::executors_for(&dyn GpuNode)`. It cannot stay on `GpuNode` as
`make_executors<B>()`: a generic method is not `dyn`-compatible, and the plan tree is
`dyn GpuNode` (verified — `E0038`). That is the better placement anyway. A node describes
what it computes and stops knowing that backends exist; each backend owns one match from
node kind to its own operator types, which is where "does this backend implement this
node" belongs.

The rust-only tier boundary holds structurally and more cleanly than with a selector: a
rust-only target instantiates the driver only at `CpuBackend`, so the GPU backend's types
are never named and never monomorphized. `GpuBackend` itself stays gated as the legacy GPU
executors are.

## Drivers

Two drivers, both single-threaded, **push-based**, deterministic, generic over `Backend`.
Settled by the T0 prototype ([`scripts/exec_model/`](../../scripts/exec_model/README.md));
what follows is that design, not a sketch of it.

- **`batch_partitioned_driver`** owns the tree, the queues, the schedule, the three
  cross-lane categories (`PartitionEmitter`, `PartitionAccumulator`, `BatchForwarder`) and
  the one node it special-cases, a `GpuUnload` carrying a limit.
- **`batch_single_partition_driver`** drives one lane of one lane-scoped node — Source,
  Exec, BatchAccumulator, Join, Unload. It is that executor instance's state machine: it
  decides which call the lane's current input state calls for, makes exactly one call, and
  reports the outputs plus whether the lane will ever produce again.

The unit changed against the draft. A chunk is **one node's lane**, not a chain of them,
because min-height selection walks a batch up a chain node by node on its own and a
chain-walking driver would duplicate the scheduler.

### The scheduling rule

Every node carries a **height** (distance to the root, root = 0) and an **order**
(pre-order index, which in a tree is left-to-right within a level). Both are pure
functions of the tree, computed once.

A node is **runnable** when any of its partitions can make progress: a source always can,
and any other node can once that lane's inputs hold a batch or are known to be finished.
Among all runnable nodes the driver takes the **smallest height**, breaking ties
**leftmost**, and then runs **every lane** of that node.

That is the whole of it, and the push behaviour falls out rather than being programmed.
The moment a node produces a batch its parent is runnable at a strictly lower height, so
the batch is carried up before anything below produces again. It stops only at a batch
accumulator, a partition accumulator, or the sink — which is also the livelock argument:
the one thing that can block a batch is a join waiting on its other side, and orienting the
tree so the build side is always the left child removes that wait, since at equal heights
the leftmost node wins and the build subtree drains first.

**With N lanes the unit that moves is one batch per lane, not one batch.** "A batch reaches
the root before the next is produced" is the single-lane reading of a wavefront: running
every lane of the chosen node is what keeps partitions progressing together, and the
invariant is that the whole wavefront advances one level before a new one is produced
beneath it.

### There is no `Pending`

Runnability is a predicate evaluated *before* the call, so there is no third outcome for a
call to return. The draft's three-valued visit contract (`Batch` / `Pending` / `Exhausted`)
was an artefact of pulling: a puller has to ask and be told "not now". A pusher only calls
nodes it has already established can proceed. `SourceExecutor::next_batch` returning
`Option` is the whole of what remains — `None` means the lane is exhausted and it is never
called again.

### Queues need no cap

A producer's out-queue is drained by its parent before the producer runs again, since the
parent's height is strictly lower. Queues are therefore **self-bounding at one batch per
lane** and the draft's cap-Q mechanism is unnecessary — it existed to stop a puller from
running an upstream to exhaustion filling a starved sibling's queue, which min-height
scheduling makes impossible.

One shape breaks that bound on its own, and one rule closes it:

**A join in its build phase holds back its whole probe subtree.** Until `set_build` has run
the join cannot consume a probe batch, so without the hold the probe side runs anyway and
piles up in a queue nothing will drain. The hold is transitive over every edge on the path
to the root — blocking only the join's direct child would move the pile one node down
rather than remove it. It cannot deadlock: plans are trees, so a join's build subtree is
disjoint from its probe subtree and is never held by this rule, the build side therefore
always has a runnable node until it completes, and completing it is what lifts the hold.
Nested joins resolve outermost-first for the same reason.

With that in place the bound is unconditional: raw queued data is one batch per lane per
producing node, all of it driver-held in-flight batches the enforcer already counts.

Hash skew needs no mechanism at all. A lane that receives nothing is simply never runnable,
and empty scatter outputs are dropped at the emitter, so nothing empty ever traverses a
chain. An accumulator-ended lane is likewise ordinary: it consumes one batch per visit and
parks bytes in `resident_bytes()`-visible executor state, where partial-aggregate
compaction can shrink them.

```
                        batch_partitioned_driver
    ────────────────────────────────────────────────────────────────────
    heights (distance to root)          the choice each step:
                                          runnable nodes → min height,
      0            unload                 ties leftmost → run every lane
                     ▲
      1        agg_final                every edge holds ≤ 1 batch per lane,
                     ▲                   because the parent is strictly
      2          emit ─┬▶ q0 ─┐          lower and drains first
                       ├▶ q1  │
      3         merge  ├▶ q2  │  (empty scatter outputs dropped here)
                       └▶ q3 ─┘
      4      agg_batches
                     ▲
      5         filter                  join in build phase
                     ▲                    └▶ holds its whole probe subtree
      6          scan                   satisfied limit
                                          └▶ holds its whole subtree, for good
```

### Early exit at a limit

A `GpuUnload` carrying a root-adjacent limit's `skip`/`fetch` is the one node the driver
special-cases. Its executor is per lane and the count is across lanes, so the count cannot
live in the executor; and nothing can signal "done" from below, because satisfaction is a
fact about rows that have already passed, not about the next call.

`is_satisfied(node)` is that fact: the driver keeps one row count per such node — rows
arriving at it, summed over every lane — and the node is satisfied once that count reaches
`skip + fetch`. A pure offset (`fetch` absent) has no such point and is never satisfied: no
prefix determines the answer, so it can only drop and trim. The predicate feeds runnability
the way the join hold does — a satisfied node makes its **whole subtree** non-runnable,
transitively over every edge, so the scan stops being scheduled and pulls cease through
merges and emits alike. The two differ only in direction: a join's hold lifts when the
build completes, a limit's never lifts. That is why they share the release path that drops
every in-flight batch, and why a run can legitimately end with lanes not done and queues
non-empty. A satisfied node is marked done as it is held, so the hold cannot stop it from
reporting and strand its parent — the case that forces this is `LIMIT 0`, satisfied before
a single step.

Per batch the driver then decides, before calling `unload`, whether the batch falls
entirely outside the interval (release the handle, **no call**), straddles an end (call with
a row range), or lies inside it (call with the full range). A mid-plan `GpuLimit` makes the
same three-way decision inside its own executor, slicing rather than narrowing an export.
See the limit lowering rule.

### The test surface

**Flow and backpressure are a first-class test surface for both drivers**, exercised with
mock operators behind a mock `Backend` impl (scripted batch counts, sizes, skew patterns,
accumulator behaviour — no real executors). Each case asserts pull counts, queue bounds and
batch/handle release:

- skewed emit — starved lane, hot lane; every queue ≤ one batch;
- accumulator-ended lane progress — one batch per visit, bytes in `resident_bytes()`;
- merge-sorted over a skewed emit — no livelock, every step lands a batch or finalizes a
  lane;
- limit early exit as a release case — the subtree stops being scheduled, every in-flight
  batch is dropped, and `unload` is never *called* for a batch outside the interval;
- union with one child exhausted and one still producing;
- two-phase join with emits on both sides — probe-side queues **empty**, not merely
  bounded, until the build drains;
- nested shuffles holding every bound simultaneously;
- interleave per-lane child rotation;
- determinism: two runs, identical batch traces.

Both drivers take the resident-accounting hooks below and fail the query when the enforcer
trips. An FFI error is query-fatal: the C++ side resets the whole session and every
resident handle with it (`cpp/src/gpu_executor.cpp` ~L192) — there is no mid-flight retry
with smaller batches (that is #142's adaptive future).

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
`resident_bytes()` delta, and post-check. The measured `CallStats.scratch_bytes` exists so model
quality is observable: under-estimates are recorded with their magnitude. Both backends
measure — the CPU directly, the GPU through RMM allocator hooks — so `None` means this run
was not instrumented, not that the backend cannot report. **It is not an invariant that the model is never under.** `scratch_bytes` is an
estimate — a join's rests on the optimizer's cardinality figure, a filter's on assumed
selectivity — so it will sometimes come in low, and asserting otherwise would make the
suite red for something that is not a defect. The enforcer's contract is "fail cleanly when the accounted
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
| `GpuLimit` | none — `peacock_executor_slice_handle` on the two straddling batches, and nothing at all on the rest | not a seq: the bounds are runtime values. Root-adjacent there is no node either — the interval rides `GpuUnload`'s fetch |
| `GpuMergePartitions` / `GpuUnion` / `GpuInterleave` | none (union casts are `GpuProject` seqs) | `BatchForwarder` routing in the driver, zero FFI calls |
| `GpuUnload` | none | `peacock_result_from_handle` per handle, over the row range the driver supplies; batches outside a root-adjacent limit's interval are released without a call |

This mapping is a first-class deliverable: documented here, unit-tested (each `GpuNode`
kind → expected seq set and call pattern), because it is the load-bearing trick that
keeps C++ frozen. The fb names in the table are the pre-T1 ones; after the T1 rename
they read `Cudf*` (`CudfScan`, `CudfRepartition`, …) and this table is updated in that
commit.

**The limit lowering rule.** A per-batch `GpuLimit` call cannot be correct: the fb node's
skip/fetch are frozen per seq, so every batch would be truncated to the same bounds
(two batches → 2× the limit), and the right bound for the last batch is a runtime value
no frozen node can carry. Legacy never sees this because a legacy partition is one batch.

**Root-adjacent** (feeding only `GpuUnload` — the common case) **there is no limit node**.
`skip`/`fetch` become properties of `GpuUnload`, which is where they belong: a limit over a
stream about to leave the device is a statement about which rows are worth moving across
the boundary, and the boundary is the unload. It shows in the plan golden on the unload
line, so nothing is hidden in a side channel.

The driver then owns three behaviours, all keyed on that one node. A batch entirely outside
the interval is **never unloaded** — its handle is released where it stands, which is the
whole point: trimming after unload ships the `skip` prefix across PCIe and drops it, and
that prefix is unbounded. The two batches that straddle the interval's ends are unloaded
with a **row range**, so the transfer is the rows wanted rather than the batch they sit in.
And once the interval is satisfied, `is_satisfied` holds the whole plan and pulls cease.

No `GpuMergePartitions` is inserted for this path, and the count is **across lanes** —
see the determinism note below for what that does and does not decide.

**Mid-plan** — the limit's output feeds further GPU work — it is a real `GpuLimit` node
over a **one-partition** input. `GpuMergePartitions` goes beneath it, because an interval
over N lanes names no rows, and the node checks that itself in
`validate_schemas_and_partitions()`: a statement about its child's layout belongs where
the node can name the fix. Its input is **not** required to be `SingleBatch`. Requiring
that would put a `GpuCoalesceAllBatches` underneath, and

```sql
SELECT * FROM customer JOIN (SELECT * FROM orders LIMIT 100) o ON ...
```

would read the whole of `orders` to answer for a hundred rows.

It **streams and holds nothing**. Per batch, from a running count of the rows that have
gone past: a batch entirely outside `start..limit` is released without a call, a batch
entirely inside is forwarded untouched, and only the two that straddle the interval's ends
are sliced. `is_satisfied` then stops the scan exactly as it does at the sink. Output
layout follows the input — a limit is a prefix of its stream, so it neither increases the
batch count nor disturbs an order.

The slice is where the third ABI symbol goes, and holding nothing is what it buys. Frozen
skip/fetch would only be correct against a table starting at row 0 of the stream: drop the
offset prefix and every bound shifts by a runtime amount — where the batch boundaries fell,
which depends on upstream selectivity and fan-out — that a plan constant cannot carry. So
with frozen bounds the node must hold the prefix, and `OFFSET 1000000 LIMIT 10` would hold
a million rows to return ten. With call-time bounds it holds none.

Two additive C++/header changes, both in `gpu_executor.cpp` + `peacock_gpu.h`, both
additive next to the existing symbols with legacy paths untouched:

- `peacock_executor_execute_scan_rowgroups(executor, seq, const uint32_t* row_groups,
  uint64_t n, uint64_t* out_handle, PeacockNodeStats* out_stats)` — reads the named
  `GpuScan` seq's options but overrides its row-group list for this call.
- a row interval on the result fetch: `peacock_result_from_handle(executor, handle,
  uint64_t offset, uint64_t length, ...)` exports that range of the handle instead of the
  whole table. This is not a seq, so it does not reintroduce the frozen-bounds problem —
  the fetch already takes a handle and runtime arguments, which is exactly why the trim can
  ride it while the plan stays free of a limit call. `length` of `UINT64_MAX` means "to the
  end", so every existing caller is a two-argument change. Serves the **root-adjacent**
  limit, where the narrowed rows are leaving the device anyway.
- `peacock_executor_slice_handle(executor, uint64_t handle, uint64_t offset,
  uint64_t length, uint64_t* out_handle)` — a `cudf::slice` of the named handle
  materialized into a new one; consumes the input handle, as every operation on a
  `GpuBatch` does. Serves the **mid-plan** limit, whose narrowed rows feed further GPU work
  and so must be a handle rather than a result. Also not a seq, and for the same reason:
  the bounds are runtime values, and that is exactly what a frozen node cannot express.
  The copy is bounded by the rows kept — which are the rows about to be used — and at most
  two batches per limit are ever sliced.

Three additive symbols is more than the design would like, so each earns its place: the
first because incremental loading is impossible without it, and the other two because a
limit that cannot narrow a batch at runtime has to hold everything ahead of the rows it
wants. They are not interchangeable — one produces a result, the other a handle — though
the fetch range could in principle be dropped in favour of slicing and then exporting
whole, at the cost of one bounded device copy immediately before a PCIe transfer of the
same rows.

## Determinism rules

Batch boundaries are a pure function of the plan: the loader's come from the
partitioner's committed mapping, Exec ops are 1:1, accumulators emit at defined points.
Given that, the remaining scheduling freedoms are pinned:

- **Every `BatchForwarder` lane cycles its `sources_of` list in order**
  — for `GpuMergePartitions` that is round-robin over partitions by index. A source with
  a batch queued yields it; one with nothing queued is skipped this cycle; one whose
  producer has finished is retired from the rotation. Emission order is arrival order
  under this schedule.
  Chosen over drain-in-partition-order deliberately: it keeps "partition = a lane that
  makes progress alongside the others" true, at the honest cost of N live lanes — which
  the estimator charges, and which matches what a parallel driver will cost anyway. If a
  driver ever goes parallel, emission order is preserved with a reorder buffer or the
  goldens regenerate deliberately.
- **`cudf::merge` tie order**: input tables are passed partition-major (partition 0's
  batches in stream order, then partition 1's, …) regardless of arrival order.
- Order pinning is part of *result* determinism, not just golden stability: float
  aggregation sums in stream order, so an unpinned order changes low bits.
- **A root-adjacent limit counts across lanes**, so *which* rows an unordered `LIMIT`
  returns depends on the order batches reach the sink. That is settled by the scope note
  below: it is fixed for a given plan, and it differs between tp1 and tp4. Inserting a
  `GpuMergePartitions` would not change that — the merge is round-robin, so it interleaves
  lanes too — and every ordered limit is unaffected, because a sort delivers one lane and
  one batch before the sink ever sees it.

**Scope: these rules pin execution for a given plan, not across plans.** Two plans for the
same query — tp1 and tp4, batching off and on — may legitimately return different rows
where the SQL does not determine them, which is what an unordered `LIMIT` is. Nothing in
the test estate assumes otherwise. Result goldens are named
`<query>.<mode>-<tp>-<tier>`, so each plan already carries its own file, and they are
compared with `batches_to_sorted_str` — rows sorted, order-independent — so emission order
is not part of the contract either. What must hold is that one plan run twice gives one
answer, byte for byte, which is what the bullets above buy. In the corpus this touches one
query, `tpch-queries/scan-limit.sql` (`SELECT * FROM lineitem LIMIT 10`); the four other
bare-`LIMIT` queries (TPC-DS q28, q32, q38, q97) limit an already-single-row aggregate.

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
in Python, operators built with pandas, plans hand-built (no DataFusion, no planner) — an
emulation of tree execution whose purpose is to settle the push model before any Rust
exists. Lives in [`scripts/exec_model/`](../../scripts/exec_model/README.md); its tests run
in CI (cost-report, plus the TPC-H set in cpp-cpu, which has the generated sf1).

Done — struck through, and folded into this document where it changed a decision:

- ~~the trait set, both drivers, and the memory enforcer with the accounting formula~~;
- ~~the scheduling rule~~ — height, order, min-height-first with leftmost ties, every lane
  of the chosen node. Push behaviour falls out of it rather than being programmed, and the
  Drivers section is rewritten from that;
- ~~the backpressure rules~~ — a join in its build phase holds its whole probe subtree; a
  satisfied limit holds its whole subtree for good. Both were findings, not designs;
- ~~queues need no cap~~ and ~~`Pending` does not exist~~ — the draft's two flow-control
  mechanisms, both dropped, both because runnability is a predicate evaluated before the
  call;
- ~~pandas-backed operators~~ — filter, project, sort, the aggregate sequence with its
  partial/final decomposition, the accumulators, the hash scatter, the join capability
  matrix, and the T2 row-group partitioning policy, each written against the pandas/cuDF
  intersection with the divergences named;
- ~~every query checked against a single-shot oracle at five partitioning configs~~, the
  prototype's version of two-engine correctness;
- ~~both limit lowerings~~ — `GpuUnload` carries `skip`/`fetch` and is its own executor
  category taking a row range per call; the driver holds the cross-lane count, releases
  batches it wants nothing from without calling `unload`, and narrows the two that
  straddle. The mid-plan `GpuLimit` streams over a one-partition input, holding nothing.
  Tests assert on the *calls*, since only those distinguish a limit from a filter applied
  after the transfer;
- ~~the stress surface~~ — a plan rewriter (`operators/injection.py`) rather than
  hand-written variants: one plan re-run at every partitioning, batch size, empty-lane and
  hash-placement preset, with `GpuCoalesceBatches[target]` injected above every source
  (#139's node, proving the drivers tolerate it anywhere) and sources emitting zero-row
  batches at a set probability. It carries one rule worth quoting into the planner's tests
  — a join may be re-partitioned only when both sides are hash-partitioned on the join
  keys, since otherwise its lane count is load-bearing and splitting it joins matching
  slices;
- ~~empty partitions, empty batches, skewed hashes, the flow-and-backpressure surface,
  determinism (two runs, identical batch traces)~~;
- ~~validation scope~~ — partitioning and `SingleBatch` constraints in scope, schema checks
  not.

Still open:

- the **estimator** (`estimated_max_resident_size`, `target_batch_bytes`) — the prototype
  models scratch per executor but does not derive batch sizes from a budget;
- the **hand-built plan corpus**: 3–4 TPC-H plans beyond the ones `test_tpch.py` already
  runs, and ~10 interesting TPC-DS plans — unions and interleaves, rollup aggregates,
  semi/anti joins, cross joins beside aggregates, multi-shuffle trees. This is the piece
  most likely to find something, because the shapes it adds are the ones no test has run.

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
`Backend`. Traits in their own files per coding-style. Compiles under rust-only
with the GPU side gated. Unit tests: `SortOrder` canonicalization, layout equality.
**First, before anything else in this task**, compile a skeleton: the `Backend` trait with
all seven associated types, two impls whose `Batch` types differ, `NodeExecutors<B>`, and a
generic function driving one build→probe→finish transition and one source step. It
compiles with no `dyn` anywhere (verified), and it is what pins the static-dispatch
property the GPU path depends on — the mock backend the driver tests need is then a third
impl, not a special case.

**T4 — translation layer, single-partition shapes.** DataFusion physical plan (tp1) →
`GpuNode` tree for chains: load, filter, project, sort (+fetch), limit (root-adjacent ⇒
no node, `skip`/`fetch` set on `GpuUnload`; otherwise a `GpuLimit` node over a
planner-inserted `GpuMergePartitions` — never a coalesce), coalesce-all,
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

**T7 — schema registry.** The `Schema` carried in `NodeKind` populated on all nodes, with
column semantics annotations. Unit tests: hand-crafted plans produce expected types and
annotations; decimal precision/scale fidelity through project/aggregate/union-cast paths.

**T8 — validation.** `validate_schemas_and_partitions()` on every node type: partition
topology, key-distribution subset rule, sortedness requirements (merge requires
`BatchSorted`; a limit after a sort requires its input to be `is_stream_sorted()`, checked
on whichever node carries the interval — the `GpuLimit` mid-plan, the `GpuUnload`
root-adjacent), `SingleBatch`
expectations (join build, cross/nlj inputs), captured-index checks. Unit tests: manually
constructed wrong combinations error, right ones pass; then run validation over every
canonized corpus plan from T6.

**T9 — additive ABI.** Both planned C++/header changes in `gpu_executor.cpp` +
`peacock_gpu.h`; any *further* surface change goes through a proposal to the human, per
the constraint section. `peacock_executor_execute_scan_rowgroups`, plumbing
`row_groups_override`. A row interval on `peacock_result_from_handle` —
`(offset, length)`, `length = UINT64_MAX` meaning to the end, so existing callers change
by two arguments and legacy behaviour is the default. And
`peacock_executor_slice_handle`, consuming its input handle and returning a new one.
Rust bindings for all three; `GpuBatch` handle plumbing (session ref, `Drop` release,
`ManuallyDrop` consume boundary). Tests: a C++ gtest in the plan-executor suite reading
disjoint row-group subsets and asserting union == whole-scan; a gtest exporting ranges of
one handle and asserting the concatenation equals the whole, plus the empty range and the
past-the-end range; the same for slicing, plus that the input handle is released and
double-slicing it fails; Rust FFI smoke on shad-gpu. The range plumbing reaches
`UnloadExecutor::unload(batch, rows)`, so the trait's second argument lands here rather
than in T10.

**T10 — Exec executors, CPU and GPU.** Filter, project, per-batch sort, aggregate
(partial/single), unload (`GpuBatch → CpuBatch`, honouring the row range). Reuse legacy operator code by extracting
helpers — never by calling into strip/wrapper machinery. Per executor: CPU vs
hand-crafted oracle, GPU vs CPU, empty-batch cases, `CallStats` model ≥ measured on CPU.
CPU and GPU tests in separate targets so CI hosts split them.

**T11 — accumulators.** `GpuCoalesceAllBatches`, `GpuAggregateBatches` (both finals),
`GpuAccumulateBatchesAndSort`, `GpuMergeSortedPartitions`, and the mid-plan `GpuLimit`.
Edge cases: zero batches, one batch, ties for the merge (partition-major stability), fetch
interaction, large batch counts, gid-carrying aggregate merges. For the limit specifically:
that `resident_bytes()` stays zero whatever the offset, that at most two batches per query
are sliced, and that the scan stops — each asserted as a call or pull count, since a test
on the rows returned passes just as well when the whole input was read and held.

**T12 — partition ops and joins.** `GpuEmitPartitions` (per-batch scatter, N=3 and large
N, empty outputs for skewed hashes), `GpuMergePartitions` round-robin rule,
`GpuJoin` with `set_build`/`probe_and_fetch`/`finish_and_fetch`, plus cross and
nested-loop joins on the same trait: the full capability matrix as a test table — per
(type × layout): stream-vs-refuse, correctness vs the single-batch oracle, the GPU
finish pass via key accumulation (#136), null_equals_null on the finish join.

**T13 — drivers and enforcer.** Both drivers with mock backends via the selector;
`batch_partitioned_driver` tested against a mocked single-partition driver; round-robin
determinism cases; the full flow-and-backpressure suite from the drivers section, with
mock operators: skewed emit (every queue ≤ one batch per lane, starved lane never
runnable), accumulator-ended lane (one batch per visit, bytes in `resident_bytes()`),
merge-sorted over a skewed emit (no livelock), limit early exit as a release case
(`is_satisfied` makes the whole subtree non-runnable, pulls cease through merges and
emits, all in-flight batches dropped; `unload` is never *called* for a batch outside
`start..limit` and is called with the right range for the two that straddle — asserted
call by call, since a test on the returned rows alone passes just as well when every batch
crossed the bus first), union with one child exhausted and one still producing, two-phase
join with emits on both sides (probe queues empty until the build drains), nested shuffles
holding all bounds simultaneously, interleave per-lane child rotation; the accounting
formula with pre/post checks; enforcer trip ⇒ clean query failure; FFI-error ⇒
query-fatal semantics.

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
