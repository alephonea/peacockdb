# batch-partitioned executor

A new planning and execution mode in which a GPU partition holds a *stream of batches*
instead of one resident table. The motivating pipeline: load → filter at 1% selectivity →
aggregate into few groups. Today's partitioned mode materializes the whole scan on the
GPU before the filter ever runs; with batches, small slices flow through the filter and
only the aggregate's state stays resident, so the query fits in a VRAM budget the table
does not.

Status (2026-08-11): design final, including the parts only a running model could settle.
**A Python prototype of the whole execution model lives in
[`scripts/exec_model/`](../../scripts/exec_model/README.md)** (task T0): the trait set,
both drivers, the scheduler, the enforcer, pandas-backed operators checked against a
single-shot oracle, and a plan rewriter that re-runs each query at every partitioning and
batching shape. Where the two disagree the prototype is wrong and gets fixed; where this
document predates it, it has been rewritten from what the prototype established — the
Drivers section most of all.

The spec lives on master, committed ahead of the work, because the task spans many
branches. Deferred work is ticketed rather than latent: this file plus `tickets.md` is the
complete state.

Companion tickets: [#136](../tickets.md#t136),
[#137](../tickets.md#t137), [#138](../tickets.md#t138), [#139](../tickets.md#t139),
[#140](../tickets.md#t140), [#141](../tickets.md#t141), [#142](../tickets.md#t142),
[#143](../tickets.md#t143).

## Contents

- [Scope and constraints](#scope-and-constraints)
- [Planning](#planning)
  - [Approach: translate the DataFusion physical plan](#approach-translate-the-datafusion-physical-plan)
  - [Knobs](#knobs)
  - [ParquetBatchPartitioner](#parquetbatchpartitioner)
  - [The sort decomposition](#the-sort-decomposition)
  - [The aggregate sequence](#the-aggregate-sequence)
  - [Aggregates](#aggregates)
  - [Grouping sets](#grouping-sets)
  - [Compaction runs on a doubling threshold](#compaction-runs-on-a-doubling-threshold)
  - [Implicit casts become explicit](#implicit-casts-become-explicit)
  - [DISTINCT lowers to grouping](#distinct-lowers-to-grouping)
  - [Null keys and the shuffle](#null-keys-and-the-shuffle)
- [Node set](#node-set)
  - [Join capability matrix](#join-capability-matrix)
- [Traits](#traits)
  - [Aggregators](#aggregators)
- [Drivers](#drivers)
  - [The scheduling rule](#the-scheduling-rule)
  - [There is no `Pending`](#there-is-no-pending)
  - [Queues need no cap](#queues-need-no-cap)
  - [Early exit at a limit](#early-exit-at-a-limit)
  - [The test surface](#the-test-surface)
- [Memory accounting](#memory-accounting)
  - [What the corpus rollout measured](#what-the-corpus-rollout-measured)
- [GPU execution through the frozen FFI](#gpu-execution-through-the-frozen-ffi)
- [What the frozen surface costs, and what unfreezing would buy](#what-the-frozen-surface-costs-and-what-unfreezing-would-buy)
- [Determinism rules](#determinism-rules)
- [Goldens, registry, widget](#goldens-registry-widget)
  - [Node display](#node-display)
- [Implementation plan](#implementation-plan)

# Design

## Scope and constraints

**Feature scope.** Everything the legacy full_table CPU and GPU tiers run, except window
functions — window queries fail at plan time and render as plan ✗ in the new widget
tables ([#143](../tickets.md#t143)). The four TPC-DS queries that do not physically plan
on DataFusion 45 stay out until [#23](../tickets.md#t23). Unsupported shapes inside
supported features (mixed distinct + non-distinct #62, value-form CASE #57) fail at plan
time where the planner can see them, instead of throwing at run time; per-query
enablement stays in the registry as for legacy modes. The corpus this mode covers is
therefore slightly larger than legacy's rather than a subset: tpcds q38, q76 and q87 plan in
all five modes and were each measured to run correctly on the shared GPU operators, while the
legacy tiers leave them disabled by choice
([#115](../archive/archived-tickets.md#t115), closed).

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

A task that seems to need a fourth symbol (candidates: #136, #142) goes to the human as a
coordinator proposal — the blocked task, the smallest additive change, the workaround's cost.

**Two additive changes to the schema are approved on the same terms**, both appended so no
ordinal moves and no existing plan's bytes change: `UnaryOp` gains `Sqrt` and `AggregateMode`
gains `Merge`. They are the two halves of one decomposition — a merge that only merges, and a
finalize written as an expression — and without them the C++'s hardwired arithmetic is the only
one available, which leaves the two engines agreeing because two implementations happen to match.
Neither is a Welford accommodation: any aggregate whose state is composite meets the same wall,
and Welford is the first one this corpus contains. Both follow the protocol #128 set: settled here
before they are written, legacy paths untouched, `plan_bytes.sha256` proving the legacy wire form
did not move, and each new arm exercised by a gtest in the plan-executor suite on shad-gpu — an
arm nothing runs is a claim nobody has checked.

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
  class. All three surface as run-time throws deep in the C++ expression path, never at
  plan time: the aggregate materializes its serialized argument expression against
  whatever table its phase holds (`cpp/src/operators/aggregate.cpp` ~L194), so a
  partial-phase divisor cast re-evaluated on final-phase inputs fails to cast (#55), a
  string comparand in that argument hits cuDF binaryop "Unsupported operator" (#56), and
  a one-row scalar-subquery CASE branch dies in `build_column_case`'s `copy_if_else`
  fold on a row-count mismatch (`cpp/src/expr.cpp` ~L797) (#63);
- per-aggregate state schemas (sum+count for avg, M2 for stddev), read off
  `AggregateExpr::state_fields()` rather than restated by us. The *split* is ours, though,
  and has to be: at tp1 DataFusion emits mode=Single, while a batched lane needs a per-batch
  init and a merge whatever the partition count — see [the aggregate
  sequence](#the-aggregate-sequence);
- grouping-set expansion — `__grouping_id` arrives as an ordinary column of the partial,
  already in the final's group list;
- row-group pruning, which hangs off `ParquetExec` statistics.

The translation layer makes a conscious decision per DataFusion node kind — nothing is
carried over implicitly. Baseline mapping: hash `RepartitionExec` → `GpuMergePartitions` +
`GpuEmitPartitions` (the same shape the legacy budget rule lowers shuffles into);
round-robin `RepartitionExec` → dropped; `CoalesceBatchesExec` → no node, since this model
has none (`GpuCoalesceBatches(target)` was dropped from the node set — [#139](../tickets.md#t139)),
**except** its `fetch`, which lowers through the limit rule: DataFusion's limit pushdown parks
a limit on that node and removes the limit node, so consuming it without its fetch answers a
different question (`SELECT count(*) FROM (… LIMIT 3)` counts every row);
`SortExec` → [the sort decomposition](#the-sort-decomposition); aggregate pairs → the aggregate sequence
below; join nodes → `GpuJoin` with side normalization; `UnionExec`/`InterleaveExec` →
driver-level relabeling plus explicit per-branch cast projects;
`GlobalLimitExec`/`LocalLimitExec` → the limit lowering rule, which emits either no node at
all (the interval goes on `GpuUnload`) or a `GpuLimit` over an inserted
`GpuMergePartitions`. The layer is unit-tested
node kind by node kind, and an unrecognized DataFusion node is a plan-time error naming
it — never a silent pass-through.

**Targeted unit tests are the coverage; the corpus plan goldens are not.** One test per node
kind, per expression kind, and per planner rule — the shuffle insertion, the limit lowering,
the aggregate split, the partitioner's mapping, the small-table drop to one lane — each
built from the smallest plan that exercises it and asserting the tree it produces. A golden
over a whole TPC-DS query cannot do that job: it goes red when anything upstream moves, it
names a file rather than a rule, and a query that happens not to carry a shape leaves that
shape untested with nothing to show it. The goldens are the regression net over
whole plans and the readable record of what the mode emits, which is a different question
from whether each rule is right.

**Expressions are translated the same way**, and the layer is not done when the node kinds
map. Every node that carries one — a filter's predicate, a project's list, join keys and a
residual, sort and hash keys, an aggregate's arguments and its `final` list — holds a
`PhysicalExpr` tree that has to become the mode's own expression IR, one conscious decision
per expression kind, with an unrecognized kind a plan-time error naming it. What is reused
is DataFusion's *planning* of those trees: the coercions it already resolved and the
precision and scale it already derived. What is not reused is the shape, because a column
reference is an ordinal into a child whose column order this mode decides.

Two consequences worth stating before the work starts. Ordinals are rebased at every node
the layer inserts, so a per-branch cast project or an inserted merge shifts every reference
above it — this is where [#135](../archive/archived-tickets.md#t135)'s ordinal class becomes a plan-time
concern rather than a runtime one. And an aggregate's argument expression is evaluated
against a different table in each phase, which is the whole of #55/#56/#63: the init sees
input columns, the merge sees state columns, and an expression written for one is not valid
against the other.

### Knobs

1. **Partition count**: tp1 and tp4 in tests. The threshold below which a source drops to one
   lane is a stated planner input, not folklore, and it is measured in **bytes actually read**
   — the parquet column-chunk total over the projected columns of the surviving row groups,
   which is the `bytes` the partitioner already carries. Rows would misjudge both directions:
   a narrow table of many rows reads less than a wide table of few, and a query projecting two
   columns of a large table reads a fraction of one projecting twenty. So the threshold is a
   property of the scan and not of the table, and the same table can be above it in one query
   and below it in another.
2. **Batched loading off|on**: whether the loader emits more than one batch per
   partition. Even when off, the loader's declared layout is `MultipleBatches` — no
   downstream phase may assume one batch per partition. The count is known — the partitioner
   fixes every batch boundary at plan time — so that is incremental simplicity, and
   [#170](../tickets.md#t170) is what saying so would buy. Only when on does the planner take the memory budget
   (micro/mini/standard/full) and size batches, and only then does the threshold above bite.

   Batching has three forms, and they are an enum rather than a target value that means
   something special at one end of its range: `Off` gives one batch per chunk, `PerRowGroup`
   gives one batch per row group, and `Sized { target_batch_bytes }` takes the estimator's
   number. **Five planning modes cover them**, each label naming its form:

   | Mode | Lanes | Batching | Budget |
   |---|---|---|---|
   | `bp-tp1-single` | 1 | `Off` | no |
   | `bp-tp1-rowgroup` | 1 | `PerRowGroup` | no |
   | `bp-tp4-single` | 4 | `Off` | no |
   | `bp-tp4-rowgroup` | 4 | `PerRowGroup` | no |
   | `bp-tp4-sized` | 4 | `Sized` | **yes** |

   There is no `bp-tp1-sized`: at one lane a source takes essentially the whole budget, so the
   sized form collapses to `Off` on anything smaller than the budget and the mode carries no
   signal. So **one mode takes a budget**, `bp-tp4-sized`, whose `partition_groups` and
   estimates move with the tier; it records the tier in-band on its `--- memory ---` summary
   line, since the label does not carry it. The other four are budget-independent and
   reproduce from the data alone.

   The threshold's effect: a source reading fewer bytes than it drops to one lane even at
   tp4, and the nodes a one-lane region does not need are then not emitted at all — the
   `GpuMergePartitions` + `GpuEmitPartitions` pair above it, and the per-lane half of an
   aggregate sequence that would have merged what one lane never split. So the two batching
   modes differ in plan shape, not only in the loader's mapping, and a small dimension table
   is where to look for the difference.

### ParquetBatchPartitioner

The row-group → (partition, batch) mapping is computed once, at plan time, by a pure
policy class, and everything else consumes its output: `GpuLoadParquet` stores it
verbatim, the plan golden's `partition_groups=[...]` renders it verbatim, the loader
executes it, validation checks the declared partition count against it. One fact, one
owner — the opposite of the #130 shape, where a fact declared in one place is re-derived
in three.

```rust
struct RowGroupMeta { index: u32, rows: u64, bytes: u64 }   // survivors, file order
enum Batching { Off, On { target_batch_bytes: usize } }

fn partition(
    survivors: &[RowGroupMeta],
    n_partitions: usize,         // lane count for this source's region, already decided
    batching: Batching,
) -> Vec<Vec<Vec<u32>>>          // partitions → batches → row groups
```

Policy: survivors (after pruning, same source as legacy) split into `n_partitions`
contiguous chunks balanced by row count — a chunk ends where taking the next group would
land further from its share than stopping does, rather than where the share is first
reached, which overshoots by a whole group; within a chunk, consecutive row groups are packed
greedily into batches while bytes stay under target; a single row group over target still
becomes its own batch — minimum granularity is one row group, and the planner always
produces a plan (the enforcer owns the runtime consequence; recourse for oversized batches
is [#142](../tickets.md#t142)). Batching off means one batch per chunk, and `PerRowGroup` means one batch per row group —
the floor the paragraph above already sets, named rather than reached by passing a target of
one byte. Contiguity is a
policy choice, not a cuDF requirement — changing it later is a golden-regenerating change
and is treated as one.

**The balance bound holds for uniform row groups, and contiguity is why it is not
universal.** Max−min partition rows ≤ one row group is true of what a parquet writer emits —
one group size per file, a short last group — and it is tight there rather than loose,
checked over 200k generated shapes. Row groups differing by orders of magnitude within one
file beat it, because a contiguous chunk cannot step over a large group to balance around
it. Contiguity is the stronger rule and stays; the bound is the property it buys on real
files, so it is asserted with that qualifier rather than unconditionally.

`bytes` is the parquet column-chunk total over the columns the scan projects, not rows ×
a width derived from types. A varchar's width is a property of the data and the file
metadata already holds the answer; type widths are for columns the plan creates rather
than reads.

**`target_batch_bytes` is derived, not configured.** The budget is what the hardware
fixes; batch size is how the planner spends it, so an estimator pass solves for it per
source and the budget tier survives only as the fallback when statistics are missing.

The walk starts at each source and follows its batch upward until it reaches an
accumulator. Every node in between is a per-batch transform, so what it holds is
proportional to the source batch, and the amplification is a product along the path:
filter selectivity, the width change across a project, a join's cardinality times the
bytes the other side contributes, and the lane count in force at that point. A source's
figure is the maximum over its path, not the value at either end — a batch is rarely
widest where it starts, and above a merge point the same batch costs one lane's worth
rather than n.

Accumulators end the walk because they are exactly where resident stops scaling with batch
size: a join's build side holds a whole relation, an aggregate's state one row per group.
Those are constants, so they come off the budget before anything is divided.

```
Σ_sources  amplification_s × lanes_s × batch_bytes_s   ≤   budget − Σ held_by_accumulators
```

The remainder is split equally across sources, each dividing by its own amplification.
Equal shares rather than proportional ones: a proportional split hands the most budget to
the source already producing the most bytes.

Four limits on the number it produces.

The sum over sources is an upper bound rather than a peak. The driver runs one node at a
time, so two sources are rarely at their widest in the same instant; enumerating the
reachable states would be tighter and is not worth what it costs.

If the constants alone exceed the budget no batch size helps — but the planner refuses only
where the constant **cannot be an overestimate**. A build side is its input's rows and that
argument was written for it; an aggregate's state rests on the cardinality estimate, and with
today's trivial estimators (#19) that is one row per input row. tpch q1 groups 6M lineitem
rows into four, so its state is modelled at 3.8 GB against a 2 GiB budget and the plan is
refused — wrong by six orders of magnitude, and everything above the aggregate inherits it.
Refusing there turns "we do not know" into "you cannot run this". Until an estimate is a
floor rather than a guess, the number goes in the `--- memory ---` section as the worst case
it is and the enforcer decides at run time. That is the page's own division: prevention at
plan time, detection at run time.

The output is a target and not a bound. The mapping is quantized to whole row groups and
an oversized row group is still its own batch, so what the planner guarantees is that it
aimed at the budget.

The inputs are the optimizer's estimates, and on 55 of 103 TPC-DS joins the cardinality is
missing outright ([#19](../tickets.md#t19)). Underestimating amplification produces a batch
too large and a query the enforcer kills; overestimating produces one too small and a
query that is merely slower, so the derived target rounds down, onto a coarse grid — an
estimate that drifts slightly should not regenerate every golden.

**The small-table rule is per region, and a region ends at the nearest shuffle.** A source
arrives with `KeyDistribution::NotSpecified`, so a co-partitioned join has to shuffle both
sides on the join keys whatever the scans were partitioned into; the Merge and the Emit
between each side and the join re-establish the lane count, and the sides agree at the
join rather than at the scan. Below that shuffle a source's lane count is its own business.
Regions reach across more than one source only where lanes are combined positionally, as
in `GpuInterleave`'s lane p ← [(0,p), (1,p), …], or under the streamed join, which has no
shuffle to re-establish anything and so is one lane-count decision for its whole subtree.

Demoting a region is a change of lowering, not of a number, and the two halves of a
shuffle answer to different things. The Merge goes, because there is nothing to merge: a
single-lane input feeds `GpuEmitPartitions` directly. The Emit stays whenever its consumer
still wants n lanes hashed on its keys — a small dimension table joins a four-lane fact
table by emitting into four lanes, having merged nothing first — and goes only when the
consumer is itself at one lane, which is the aggregate shortcut already stated. A
stream-sorted result becomes `GpuAccumulateBatchesAndSort` rather than sort-then-merge.
That is why the decision belongs in translation and not after it: a one-lane Merge is not
a harmless no-op but a different plan, and it renders as one in the golden.

### The sort decomposition

A `SortExec` becomes a per-batch `GpuSort`, and a stream-sorted result needs an accumulator
above it, because sorting each batch leaves the batches individually ordered and
collectively not. Which accumulator depends on what the parent needs: one lane's stream
sorted is `GpuAccumulateBatchesAndSort`, and a `SortPreservingMergeExec`'s N-into-1 is
`GpuMergeSortedPartitions`. Both merge everything they have received at done, so both are
pipeline breakers ([#138](../tickets.md#t138) would relax the first).

**Where the `fetch` comes from.** DataFusion alone; peacock never derives one. A
`SELECT … ORDER BY … LIMIT n` is planned with the limit pushed *into* the sort, so
`SortExec::fetch()` returns `Some(n)`, and a `SortPreservingMergeExec` above it carries
`Some(n)` as well; the serializer copies both onto the wire and the deserializer restores
them (`peacockdb-core/src/operators/sort.rs`). The consequence is that a top-N usually
reaches us with **no limit node in the plan at all** — 28 `GpuSortExec`s in the corpus carry
a `fetch` against 24 `GpuGlobalLimitExec`s, and tpch q3's `ORDER BY revenue DESC,
o_orderdate LIMIT 10` is entirely fetch-carried:

```
GpuSortPreservingMergeExec: [revenue@1 DESC, o_orderdate@2 ASC], partitions=1, output_rows=10
  GpuSortExec: expr=[revenue@1 DESC, o_orderdate@2 ASC], fetch=10, partitions=8, output_rows=80
    p0: in_rows=1463 out_rows=10   p1: in_rows=1488 out_rows=10   …
```

Eight lanes of ten rows enter the merge and ten leave, so the merge is applying a `fetch` of
its own. The line does not say so because `GpuSortPreservingMergeExec`'s `extra_display_info`
renders only the sort keys while `GpuSortExec`'s appends `fetch=` — peacock's display gap,
not DataFusion's, which is why the number has to be inferred from the row counts. The new
mode's renderer closes it; see [Node display](#node-display).

**Whether the GPU honours it today.** In the partitioned node-by-node mode yes — the merge
arm runs `cudf::merge` over the k inputs, then `cudf::slice`s to `spm->fetch()`
(`cpp/src/node_session.cpp` ~L196). Two paths do not. The same arm's fallback — no sort
keys, or a single input partition — is a plain `cudf::concatenate` with no slice, which is
[#118](../tickets.md#t118); and in full-table and all-at-once mode a
`GpuSortPreservingMerge` is `execute_passthrough` (`cpp/src/operators/dispatch.cpp` ~L71),
so its fetch is dropped there too. Both are latent for one reason: they are taken only when
the input is a single partition, and then the `SortExec` beneath has already trimmed to n.
Across all 112 SPM nodes in the corpus exactly one — q3 at `partitioned-tp8-standard` —
receives more rows than its fetch, which is the evidence #118 asks for and the reason its
severity stays low.

**What the new mode emits.** The `fetch` is replicated onto every stage of the
decomposition: `GpuSort(fetch=n)` per batch, then `GpuAccumulateBatchesAndSort(fetch=n)`
within a lane, then `GpuMergeSortedPartitions(fetch=n)` across lanes. That is sound because
top-n distributes over concatenation — the top n of a union is the top n of the union of
each part's top n — and it is what makes a top-N memory-bounded rather than a full sort with
a slice at the end: each stage holds at most n rows per live batch instead of its whole
input. Skipping it on the accumulator would mean a one-partition `ORDER BY … LIMIT 10`
accumulates and sorts the entire stream to return ten rows, which is precisely the failure
the limit lowering rule exists to avoid elsewhere. A `fetch` is
therefore a node property in this mode, printed by the golden on every node that carries
one.

One coverage note falls out of the same audit: no benchmark query has an `OFFSET`, so every
`GpuGlobalLimitExec` in the corpus has `skip=0`. `nested-limits.sql` is the query that reaches
the offset half of both lowerings, at tp1 and tp4 alike. What it cannot reach is a mid-plan
`GpuLimit` over more than one lane: a join between the two limits lets DataFusion push the
interval into the scan, and this mode plans a limit-carrying source single-lane, while an
aggregate between them loses the interval outright at tp4 ([#166](../tickets.md#t166)). So a
multi-lane mid-plan limit is a case T15 constructs rather than canonizes.

### The aggregate sequence

**DataFusion's partial-aggregation probe must be off, and the reason is structural.** An
`AggregateExec` in Partial mode measures its own aggregation ratio after
`skip_partial_aggregation_probe_rows_threshold` rows and, where the groups are nearly as numerous
as the rows, stops grouping and passes its input through as state. That is sound in a DataFusion
plan because a Final stage regroups downstream. Here nothing does: the init emits state and the
merge is a Partial too, so duplicate keys survive to the finalize and come out as extra rows.
Both take a context with the threshold at `usize::MAX` — stated as the probe never happening
rather than as a ratio that cannot be met, which is a bound the next reader would have to check.

It is a silent wrong answer, not a failure, and only over a wide high-cardinality group by: one key
over `store_sales` is exact, `date_dim` is exact, and `GROUP BY ss_customer_sk, ss_item_sk` returns
2,797,913 against DataFusion's 2,764,744. A device never skips, so this is also what keeps the two
engines answering the same rather than only this one answering correctly.


**An aggregate node carries no phase.** It declares what it computes — a list of
aggregators over its own input, and optionally a list of finalizing expressions over the
results — and the planner emits the parts each position needs. This replaces the legacy
`AggregateMode`, whose Partial/Final flag left the executor to reconstruct by inference
what the planner already knew: the width of each state (`avg_state_2col`'s residual
arithmetic, and the q18/q22 out-of-bounds read it caused), its shape (the
`mergeable_agg_state` wire flag), and how to finish it (the hardwired `avg_div` and
`std_finalize` arms in `cpp/src/operators/aggregate.cpp`).

Every aggregate decomposes into three declared parts, each of them ordinary IR:

- **init** — aggregators over raw input rows, emitting *state* columns. One aggregate may
  emit several: `avg` emits `sum` and `count`, `stddev` emits Welford's `count`, `mean`
  and `m2`.
- **merge** — aggregators over state columns, emitting the same state schema. Not the same
  functions as init: a `count` merges by `sum`, and Welford state merges by `merge_m2`,
  which is nameable in the IR precisely so that no aggregate needs a bespoke finish arm.
- **finalize** — one expression per output column, over the merged state. `avg` becomes a
  divide, `stddev` a `CASE` over a `sqrt`, `count` and `sum` a rename.

A node with no `final` list emits its state; a node with one emits the finalized columns.
Nothing else distinguishes the positions, so the single-node shortcut is not a third case:
it is init aggregators and finalize expressions on the same node.

Rendered in a plan golden — the shuffled four-lane form of

```sql
SELECT l_returnflag, count(*) AS n, sum(l_quantity) AS qty,
       avg(l_extendedprice) AS avg_price, stddev(l_discount) AS sd_disc
FROM lineitem GROUP BY l_returnflag
```

with the layout fields elided to everything but the lane count:

```
GpuAggregateBatches: group_by=[l_returnflag@0], partitions=4,          <- merge + finalize
    aggs=[sum(n$count@1) as n$count, sum(qty$sum@2) as qty$sum,
          sum(avg_price$sum@3) as avg_price$sum, sum(avg_price$count@4) as avg_price$count,
          merge_m2(sd_disc$count@5, sd_disc$mean@6, sd_disc$m2@7) as sd_disc$*],
    final=[n$count@1 as n,
           qty$sum@2 as qty,
           cast(avg_price$sum@3 as Decimal128(38, 6))
             / cast(avg_price$count@4 as Decimal128(38, 0)) as avg_price,
           CASE WHEN sd_disc$count@5 - 1 <= 0 THEN NULL
                ELSE sqrt(sd_disc$m2@7 / (sd_disc$count@5 - 1)) END as sd_disc]
  GpuEmitPartitions: hash=[l_returnflag@0], 1 -> 4                     <- one scatter call
    GpuCoalesceAllBatches: partitions=1
      GpuMergePartitions: 4 -> 1
        GpuAggregateBatches: group_by=[l_returnflag@0], partitions=4,  <- merge only
            aggs=[sum(n$count@1) as n$count, sum(qty$sum@2) as qty$sum,
                  sum(avg_price$sum@3) as avg_price$sum,
                  sum(avg_price$count@4) as avg_price$count,
                  merge_m2(sd_disc$count@5, sd_disc$mean@6, sd_disc$m2@7) as sd_disc$*]
          GpuAggregate: group_by=[l_returnflag@0], partitions=4,       <- init, per batch
              aggs=[count(*) as n$count, sum(l_quantity@1) as qty$sum,
                    sum(l_extendedprice@2) as avg_price$sum,
                    count(l_extendedprice@2) as avg_price$count,
                    count(l_discount@3) as sd_disc$count,
                    mean(l_discount@3) as sd_disc$mean,
                    m2(l_discount@3) as sd_disc$m2]
            GpuLoadParquet: table=lineitem, partitions=4, partition_groups=[…]
```

The two `GpuAggregateBatches` are the same node with and without `final`, which is the
point of dropping the flag. `sd_disc$*` abbreviates the three columns `merge_m2` returns
together; the golden spells them out.

**References are ordinals, displayed as `name@ordinal`.** Inside `aggs` they index the
node's input; inside `final` they index the node's own intermediate table, `[group keys…,
state columns…]` — a table the node materializes rather than a private numbering, so a
finalize can reach a group key or the rollup `__grouping_id` when it needs one. The name is
carried in `ColumnRef` beside the index, as it already is throughout the IR, and it is
checked against the intermediate schema at that position rather than merely displayed: a
mismatch throws, which is what makes the redundancy worth its bytes
([#135](../archive/archived-tickets.md#t135) is the general case). The state schema is declared with types,
so nothing downstream infers a width.

Shortcuts: a 1-partition single-batch input needs one `GpuAggregate` carrying both `aggs`
and `final`; a 1-partition input skips Merge/Emit; a single-batch-per-partition input skips
the first `GpuAggregateBatches`. v1 skips the shuffle only for 1-partition inputs or keyless
aggregates; skipping on small key cardinality needs estimators that do not exist
([#141](../tickets.md#t141)).

Grouping sets need no special operator: the init node emits `__grouping_id` as an ordinary
column (existing C++ behavior, #65 caveats unchanged), every node downstream groups on keys
+ gid, and hashing on keys alone still co-locates correctly. The general validation rule
this falls out of: the input to a finalizing `GpuAggregateBatches` must have
`KeyDistribution.hashKeys ⊆ its group columns` — subset, not equality.

Three costs, all small. Two are appended fbs values, one per half of the sequence: `sqrt` becomes
a `UnaryOp` on both sides — our IR and the fbs — so a finalize can be written (cuDF's
`unary_operator::SQRT` is what the hardwired finalize already calls), and `AggregateMode` gains
`Merge`, merge state and emit state, so a merge can be only a merge. The fbs had no such mode
because the C++ merged and finalized on one call, which is precisely the coupling this
decomposition undoes. Third, the translation layer gains a decomposition registry of about six
entries, whose state names
and types come from DataFusion's `AggregateExpr::state_fields()` so our split cannot drift
from the split DataFusion planned. Adding an aggregate is then a row in that registry
rather than an arm in C++. What stays on the C++ side is the cuDF calling convention alone:
`merge_m2` packs its three inputs into the struct cuDF wants, with the INT32 count that
25.02 requires ([#94](../tickets.md#t94)) — a version-specific detail that must not reach
the wire format.

### Aggregates

Every aggregate the corpus uses, decomposed. The counts are uses across all `.cpu.txt`
goldens; `x` is the aggregate's argument and `$` names a state column belonging to output
`o`. This is the decomposition registry the translation layer owns — the prototype's copy
is `finalize_exprs` in `scripts/exec_model/operators/aggregates.py`, and adding an
aggregate is a row here and there rather than an arm in C++.

| Aggregate | Uses | init (over rows) | state | merge (over state) | finalize |
|---|--:|---|---|---|---|
| `sum(x)` | 1010 | `sum(x)` | `o` | `sum(o)` | `o` |
| `avg(x)` | 204 | `sum(x)`, `count(x)` | `o$sum`, `o$count` | `sum`, `sum` | `o$sum / o$count` |
| `count(x)` | 190 | `count(x)` — non-nulls | `o` | **`sum(o)`** | `o` |
| `count(*)` | — | `count(*)` — rows | `o` | **`sum(o)`** | `o` |
| `stddev(x)`, `stddev_pop(x)` | 29 | `count(x)`, `mean(x)`, `m2(x)` | `o$count`, `o$mean`, `o$m2` | **`merge_m2`** | `CASE WHEN o$count − ddof <= 0 THEN NULL ELSE sqrt(o$m2 / (o$count − ddof)) END` |
| `var(x)`, `var_pop(x)` | 10 | as stddev | as stddev | **`merge_m2`** | the same without the `sqrt` |
| `max(x)` | 22 | `max(x)` | `o` | `max(o)` | `o` |
| `min(x)` | 12 | `min(x)` | `o` | `min(o)` | `o` |

Three rows carry the weight. **`count` merges by `sum`** — the one place where naming the
merge separately from the init is not bookkeeping but the difference between a right and a
wrong answer, and the reason DataFusion's own distinct rewrite has to exclude `count`.
**`avg` is two state columns**, never a mean of means, which is the multi-GPU rule in
build-test.md. And **the Welford pair merges by `merge_m2`**, an aggregator the IR names
because the combine is not a per-column reduction: it needs the count-weighted mean and the
cross term, which is what cuDF's `MERGE_M2` computes and what the C++ packs a three-child
struct for. `ddof` is 1 for the sample forms and 0 for the population ones, matching
`stddev_ddof` in `cpp/src/operators/aggregate.cpp`.

The finalize column is what replaces the hardwired `avg_div` and `std_finalize` arms: for
the five simple aggregates it is a rename, for `avg` a divide, and for the Welford pair a
`CASE` over a `sqrt` — all of them ordinary expressions the planner emits, which is why
`sqrt` joins `UnaryOp`. A group with `count <= ddof` has no dispersion to report, so the
`CASE` yields NULL rather than dividing by zero or rooting a negative.

### Grouping sets

**What the corpus has.** Ten TPC-DS queries write `ROLLUP`; neither benchmark contains a
`CUBE` or an explicit `GROUPING SETS` clause anywhere. They do not all reach the feature.
Seven — q5, q14, q18, q22, q67, q77, q80 — get DataFusion's grouping-set expansion inside a
single aggregate, visible as `__grouping_id` in the final's group list. q36 hand-writes its
rollup as a `UNION` of three ordinary aggregates in SQL, so it never touches the path at
all. q70 and q86 do not physically plan on DataFusion 45 ([#23](../tickets.md#t23)) — and
they are also the only two queries that project `GROUPING()`, which is precisely what
[#65](../tickets.md#t65) is about. So the feature is live in seven queries and its one known
defect is latent behind the two that cannot plan.

**No new node type.** The set masks ride on the init aggregate exactly as they do in the
legacy IR (`grouping_sets`, `null_exprs`, `null_names`), and everything downstream sees
`__grouping_id` as an ordinary group column. So merge and finalize need to know nothing
about sets; the merge groups on keys + gid, and the shuffle still hashes the keys alone
because `hashKeys ⊆ group columns` is the rule it has to satisfy. Taking

```sql
SELECT l_returnflag, l_linestatus, sum(l_quantity) AS qty
FROM lineitem GROUP BY ROLLUP(l_returnflag, l_linestatus)
```

at four lanes, with the layout fields elided to the lane count as before:

```
GpuProject: expr=[l_returnflag@0 as l_returnflag, l_linestatus@1 as l_linestatus, qty@3 as qty]
  GpuAggregateBatches: group_by=[l_returnflag@0, l_linestatus@1, __grouping_id@2],
      partitions=4, aggs=[sum(qty$sum@3) as qty$sum],
      final=[qty$sum@3 as qty]
    GpuEmitPartitions: hash=[l_returnflag@0, l_linestatus@1], 1 -> 4
      GpuCoalesceAllBatches: partitions=1
        GpuMergePartitions: 4 -> 1
          GpuAggregateBatches: group_by=[l_returnflag@0, l_linestatus@1, __grouping_id@2],
              partitions=4, aggs=[sum(qty$sum@3) as qty$sum]
            GpuAggregate: group_by=[l_returnflag@0, l_linestatus@1], partitions=4,
                grouping_sets=[(l_returnflag@0, l_linestatus@1), (l_returnflag@0), ()],
                aggs=[sum(l_quantity@2) as qty$sum]
              GpuLoadParquet: table=lineitem, partitions=4, partition_groups=[…]
```

Two lines differ from a plain `GROUP BY l_returnflag, l_linestatus`. The init gains
`grouping_sets`, and its output gains `__grouping_id@2` which every node above treats as one
more group column; and the root gains a `GpuProject` to drop that column again, because the
finalizing aggregate emits `[keys…, final…]` and the gid is one of its keys — without the
project the query returns a column it never asked for. The hash covers the two user keys and
not the gid: the subset rule permits it, and equal group keys always carry equal user keys,
so co-location holds.

The ids are the bitmask of each set's **masked** positions, so a two-key rollup gives 0, 2
and 3 rather than 0, 1, 2 — distinct per set, which is all the merge needs, and not
DataFusion's `GROUPING()` encoding, which is [#65](../tickets.md#t65).

The gid is a real column rather than a plan-level annotation: the expansion materializes an
INT32 constant per set and appends it **after the group keys and before the aggregate
outputs**, which is why it is `@2` here and the sum is `@3`. Its rendering is asymmetric,
and deliberately so — the init's `group_by` does not list it, because at that node it is a
tag being synthesized rather than a key being grouped on, while every node above does list
it, because there it is an ordinary group key. It leaves again at the projection over the
final, which selects the user's columns and drops it. So a reader following only the
execution golden sees the column appear between two nodes with nothing to explain it; the
plan golden is where the init's declared output schema shows it being introduced.

One batch out of that init, with all three sets in it, is what the "one `cudf::table`"
claim looks like concretely:

```
l_returnflag  l_linestatus  __grouping_id  qty$sum
A             F             0              37.0     <- set (returnflag, linestatus)
N             O             0              12.0
A             NULL          2              37.0     <- set (returnflag), linestatus masked
N             NULL          2              12.0
NULL          NULL          3              49.0     <- the grand total, both masked
```

A masked column is a typed NULL rather than an absent one — chosen to be
concatenate-compatible with the real column at that position — which is what lets every set
share a schema and sit in one `cudf::table`, distinguished by the gid. That is already what
the `cudf::concatenate` at the end of the expansion produces.

Not a Spark-style expand, either. Spark multiplies rows *before* aggregating — k×N rows
materialized for k sets — whereas the C++ here runs k groupbys over the same input and
concatenates the k results (`cpp/src/operators/aggregate.cpp` ~L334-427), so the peak is the
input plus the sum of the per-set outputs rather than k times the input. In a mode built to
bound residency that is the difference that matters, and a row-multiplying operator would be
a new category (1 → k rows) where the current shape is an ordinary Exec.
[#144](../tickets.md#t144)'s multi-distinct lowering is the one that genuinely needs such a
node: both use a gid, only one needs the rows multiplied.

**It must be one batch rather than k**, and the reason is the driver rather than cuDF: **no
executor may return more than one batch per call per output lane.** That is the queue bound
the whole flow-control argument rests on, and the prototype keeps a deliberately
non-conforming accumulator around solely as the input that turns the assertion red. An init
aggregate emitting one batch per grouping set would put k batches into a one-lane queue and
break it. So the shape is forced: k groupbys per input batch, concatenated, one batch out.
What grows is the row count and not the batch count — the init emits up to k×G rows where a
plain aggregate emits G — and that lands on the compaction threshold above, which is a
larger state compacted by the same rule, not a new problem.

**Skew is a non-issue here despite looking like one.** A rollup's last set masks every key,
so those rows hash on nothing and land in the single lane `pmod(seed, N)` —
[#137](../tickets.md#t137)'s shape exactly. It is one row, the grand total, so there is
nothing to mitigate; the intermediate sets still spread on whichever keys survive their mask.

### Compaction runs on a doubling threshold

`GpuAggregateBatches` holds pre-aggregated state, and when it folds that state down decides
whether the sequence is memory-bounded or not. Both obvious policies are wrong in one
regime. Compacting on every arrival keeps the state at group cardinality, but it re-scans
the state once per batch: where the groups are disjoint the state grows every time and the
total work is quadratic in the batch count. Never compacting — holding every partial for
one concat at done — is the whole input whenever the group cardinality is high.

So the node holds arrivals until they cross a byte threshold, compacts once, and then sets
the threshold to twice what that compaction left behind. A low-cardinality aggregate leaves
a small state, the threshold never moves, and residency stays near group cardinality at a
fraction of the calls per-arrival would cost. A high-cardinality one leaves a state the size
of its input, so the threshold doubles away: compactions land at geometrically growing
sizes, total re-scan work is linear in the rows that pass through rather than quadratic, and
the node stops paying for a merge that merges nothing. Residency then grows, which is the
honest answer for that shape — the enforcer is the backstop
([#142](../tickets.md#t142)) — and the threshold itself comes from the same budget rule that
sizes loader batches. The prototype implements this
(`operators/accumulators.py`), and its two regimes are pinned by tests that go red on the
mutation of dropping the doubling: 40 disjoint arrivals compact 3 times with it and 32 times
without.

**The shuffle beneath a final aggregate is coalesced first.** `GpuMergePartitions` forwards
its L lanes' batches without concatenating, so without help `GpuEmitPartitions` would
scatter once per arriving batch and emit L×N batches in total. The planner therefore puts a
`GpuCoalesceAllBatches` between the two: the emit then sees one batch, makes one scatter
call, and emits N.

The reason is batch *shape*, not residency. All L pre-shuffle batches are resident either
way — the driver runs every lane of a node in one step, so they land in the producing node's
out-queues together, and the coalesce moves them from L queues into one table rather than
turning L into 1. What it buys is scatter outputs of N batches at about G/N rows instead of
L×N at about G/(L·N): every downstream operator pays per batch, and the repartition arm
allocates one owning table per output partition (`cpp/src/node_session.cpp` ~L270), so L
scatter calls make L×N allocations where one makes N. [#145](../tickets.md#t145) removes those copies altogether — `cudf::partition` already
returns the N partitions contiguous in one table, and only the handle model's insistence
that each own its memory forces the deep copy — but until it lands their count is worth
minimizing.

It costs one concat, which the per-batch path would not pay at all — a single-handle call
takes the `std::move` side of `owned.size() == 1 ? … : cudf::concatenate(views)` at ~L248,
and that `concatenate` is a defensive branch no lowering reaches. The concat is bounded by
the smallest data in the plan, since this point is post-partial-aggregate at group
cardinality — which is why the coalesce is affordable here and nowhere else. A join's probe-side shuffle is **not** coalesced — its
input is unbounded and streaming past the build side is the whole point.

### Implicit casts become explicit

The legacy executor inserts type coercions the plan never mentions. An audit of `cpp/src`
finds nine, of which the first seven become plan nodes:

| Cast | Site | What it does, and why it is there |
|---|---|---|
| `avg`'s decimal input | [aggregate.cpp ~L214](../../cpp/src/operators/aggregate.cpp#L214), again at [window.cpp ~L88](../../cpp/src/operators/window.cpp#L88) | The input is cast up to DataFusion's declared out scale (s+4) before the mean, because cuDF's mean keeps the input scale s. Written twice — the window copy's comment says it mirrors the aggregate's |
| `avg`'s finalize | [aggregate.cpp ~L722](../../cpp/src/operators/aggregate.cpp#L722) | Σsum is cast to the out scale and Σcount to DECIMAL128 scale 0, so cuDF's DIV — whose result scale is lhs.scale − rhs.scale — lands on the declared scale. The non-decimal branch casts both to FLOAT64 |
| `count`'s width | [aggregate.cpp ~L413, ~L738](../../cpp/src/operators/aggregate.cpp#L738) | cuDF's count returns INT32 and SQL BIGINT wants INT64, so every count result is widened — in the grouping-set path and the ordinary one separately |
| `stddev`/`var` operands | [aggregate.cpp ~L580, ~L695](../../cpp/src/operators/aggregate.cpp#L695) | The value is cast to FLOAT64 before Welford accumulation, and the merged count to FLOAT64 for the `m2/(count−ddof)` divide |
| union branch types | [union.cpp ~L51](../../cpp/src/operators/union.cpp#L51) | Branches are planned independently, so one column can arrive as a different cuDF type per branch; each numeric/decimal column is retyped to the union's declared output type or `cudf::concatenate` throws ([#41](../tickets.md#t41)) |
| a decimal divide's numerator | [expr.cpp ~L591](../../cpp/src/expr.cpp#L591) | cuDF's fixed-point DIV yields scale lhs−rhs, so the numerator is pre-scaled to `e_o + e_r` to make the result carry the declared out scale |
| `round`'s operand | [expr.cpp ~L715](../../cpp/src/expr.cpp#L715) | `cudf::round` is called on FLOAT64, so a non-float operand is cast first |
| *stays in C++* | | |
| the loader's decimal width | [scan.cpp ~L112](../../cpp/src/operators/scan.cpp#L112), repeated at the IPC boundary in [gpu_executor.cpp ~L53](../../cpp/src/gpu_executor.cpp#L53) | cuDF's parquet reader picks the narrowest fixed_point width, while DataFusion uses Decimal128 throughout and `binary_operation` rejects mixed widths. Not a coercion of its own: the source honouring the output schema it already declares |
| hash key normalization | [spark_hash_partition.cu ~L148](../../cpp/src/spark_hash_partition.cu#L148) | INT8/INT16 are value-cast to INT32 and TIMESTAMP_DAYS bit-cast, so the 4-byte kernel hashes Spark-identical bytes. Feeds the hash alone and never reaches a returned value, like `merge_m2`'s INT32 count ([#94](../tickets.md#t94)) |

All seven are invisible in today's goldens, so a wrong coercion presents as a wrong number
rather than a wrong plan, and the two engines can disagree about a cast neither plan states.
In the new mode each is a `CastExprNode` the planner emits — the aggregate ones inside the
`final` expressions above, the union ones as the per-branch `GpuProject` casts the node set
already calls for, the expression ones at the point of use.

**The rule is general, and the table is only the audit that produced it: every cast is
explicit.** No executor may change a type the plan did not ask it to. A type appearing at a
node's output that its input and its expressions do not account for is a defect whichever
side of the boundary invented it, and it is one a reader can now see, because the plan
golden prints the declared schema per node beside the expressions. Nine were found by
reading `cpp/src`; the count is a property of today's operator set, not a budget, and an
operator added later inherits the rule rather than extending the table.

One of the seven is usually a no-op, and it is worth knowing which. DataFusion's coercion
equalizes a union's branch types at planning, so a `UnionExec`'s branches already match its
declared output in the plans this corpus produces — the divergence [#41](../tickets.md#t41)
was filed for is cuDF's parquet reader narrowing decimals, which stays in C++. The planner
inserts the per-branch casts anyway, because the rule is the rule and a branch that does
differ must not reach `concatenate`; what goes red on a mismatch is the union node's own
guard.

The two rows that stay in C++ are exceptions with a reason, not omissions. The loader's
decimal width is the source honouring the output schema it already declares, so the plan
does state it. Hash key normalization feeds the hash alone and never reaches a returned
value — a cast that cannot change an answer is not one the plan needs to carry.

### DISTINCT lowers to grouping

`DISTINCT` is never a property of an aggregator in the new mode — no `aggs` entry carries a
flag, and the legacy `AggregateFuncNode.distinct` is never set. It is a planning-time input
that lowers to grouping, for a reason that follows from the decomposition above: the state
of a distinct aggregate is *the set of its distinct values*, and the only way this IR
represents a set of values is as the rows of a grouped table. So the distinct argument
becomes an extra group key on an inner aggregate, and the aggregate that consumed it becomes
an ordinary one over the deduplicated rows. Three shapes cover the corpus, and the first two
need no new machinery at all:

| Shape | Corpus carriers | What arrives from DataFusion | What the mode emits |
|---|---|---|---|
| `SELECT DISTINCT`, and the set ops that lower to dedup | q6, q41, q54, plus q38/q87's INTERSECT/EXCEPT ([#115](../archive/archived-tickets.md#t115)) | an aggregate with group keys and **no** aggregators — `group_by=[c_customer_sk, c_current_addr_sk], aggr=[]` in q54's golden | the ordinary sequence with an empty `aggs` list: dedup per batch, dedup per lane, shuffle on the keys, dedup again. Correct because dedup is idempotent and associative, so no `final` list is needed either |
| one distinct argument, companions limited to `sum`/`min`/`max` | q16, q94, q95 | already two aggregates: DataFusion's `SingleDistinctToGroupBy` fired at the logical level, so no flag survives. q16's golden shows the pair — inner `group_by=[alias1], aggr=[alias2, alias3]`, outer `aggr=[count(alias1), sum(alias2), sum(alias3)]` | two ordinary sequences, each decomposing as any other aggregate. This is why q16/q94/q95 are green today |
| one distinct argument, any other companion | **q28 only** — `aggr=[avg(ss_list_price), count(ss_list_price), count(DISTINCT ss_list_price)]` | the rule refuses and `distinct: true` reaches the executor, where a guard throws ([#62](../tickets.md#t62)) | v1 refuses at plan time, per [Scope](#scope-and-constraints). The lowering below is the fix, and it needs no new node |

q28 is the only aggregate in either benchmark that carries the flag. Grepping the goldens
for `DISTINCT` finds q16, q94 and q95 as well, but there the word survives only inside an
output column *name* — `count(alias1)@0 as count(DISTINCT cs1.cs_order_number)`, where the
alias is the original logical expression's display name over an aggregate that has already
been rewritten. A flag and a name read the same to `grep` and not to the executor.

Why DataFusion refuses q28 is worth stating, because it is a limitation of its rewrite
rather than of the shape. Its rule re-applies *the same function* at the outer level, so it
is only sound for functions where `f(f(x))` is `f(x)` — hence the restriction of non-distinct
companions to `sum`, `min` and `max`, and hence q16's outer `sum(alias2)` over an inner
`alias2`. `count` and `avg` fail that test: counting a column of per-group counts is not
summing them. Our decomposition has already separated the two, because a `count`'s merge
aggregator *is* `sum` — so the outer level applies the merge aggregators the registry
provides, and the restriction lifts. q28 then lowers to an inner aggregate grouping on
`ss_list_price` with `aggs=[sum(ss_list_price), count(ss_list_price)]` and an outer one
computing `count(ss_list_price)` for the distinct count beside the merges of those two,
with `avg` finalized from them as usual. That closes #62 with a planner rewrite rather than
a distinct-aware kernel: no `nunique`, no flag on the wire, and the C++ guard stays a guard
that the new mode can no longer reach.

Nulls fall out correctly in both directions, which is worth checking rather than assuming
because the two cases want opposite things. `SELECT DISTINCT` must keep a null as a value,
and does: the dedup groups it like any other key (`null_policy::INCLUDE`, the rule
architecture.md's cuDF options table states). `count(DISTINCT x)` must *not* count it, and
does not: the inner dedup produces one null row, and the outer `count(x)` counts non-nulls
and skips it. Multiple distinct arguments over *different* expressions are the one shape
this lowering cannot express — they need a gid-multiplying expand step, as Spark's
`RewriteDistinctAggregates` does. No query in either benchmark has one; the planner refuses
the shape at plan time and the lowering is [#144](../tickets.md#t144), whose gid is not
[#65](../tickets.md#t65)'s ROLLUP `__grouping_id`.

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
| `GpuAccumulateBatchesAndSort` | BatchAccumulator | accumulates sorted batches, one `cudf::merge` at done, `fetch` applied; output one batch, `SingleBatch` + `BatchSorted` (so stream-sorted). No streaming emission — cuDF has no primitive; ranged emission is [#138](../tickets.md#t138) |
| `GpuMergeSortedPartitions` | PartitionAccumulator | input: N partitions, `MultipleBatches` allowed, `BatchSorted` required; all k·m sorted batches into one `cudf::merge`, `fetch` applied; output: 1 partition, one batch, `SingleBatch` + `BatchSorted` (so stream-sorted) |
| `GpuCoalesceAllBatches` | BatchAccumulator | concatenates a partition's batches into one at done |
| `GpuMergePartitions` | BatchForwarder | N partition streams → 1, forwarding each batch as visited, round-robin (see [Determinism](#determinism-rules)); accumulates nothing, no backend calls |
| `GpuEmitPartitions` | PartitionEmitter | 1 → N per batch by hash scatter; streaming, one call per input batch |
| `GpuAggregate` | Exec | aggregates one batch: init aggregators, plus finalize expressions when it is also the single-node shortcut |
| `GpuAggregateBatches` | BatchAccumulator | merges pre-aggregated batches; emits at done. Compacts on a byte threshold that doubles when a compaction fails to shrink — see [compaction](#compaction-runs-on-a-doubling-threshold) |
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
(Left ↔ Right forms) and restoring output column order with a project.

Three tables, one row per join mode in each. The **first** is what the mode is and what it
becomes in this vocabulary; the **second** is what crosses the wire and what cuDF runs,
which is where the frozen-surface claim is either true or not; the **third** works one
example per mode end to end, from SQL to what a single lane actually does.

None of it is paper, and none of it is a device either. `scripts/exec_model` runs every row on
two backends that share no join code — one joining with pandas, one emitting these seqs and
interpreting them with these calls — at every batching and partitioning shape the layout injector
can produce, and the seq sequences in the second table are asserted per mode. What a **device**
runs is a smaller set, because a handle is erased by its reader and nothing copies one: the second
table's last column says which, and [#145](../tickets.md#t145) is what grows it.

**Mode, and the plan it becomes.**

| Mode | Requires `GpuMergePartitions`? | Also covers | Batch-partitioned nodes |
|---|---|---|---|
| **Inner** | no — not by the join | multi-key and composite keys; `null_equals_null=true` (INTERSECT-derived); a residual filter — which still streams, since every emitted row is decided by (build, this batch) | `GpuJoin{Inner}`, probe streams, no finish |
| **Right outer** (probe side preserved) | no — not by the join | a DataFusion Left-outer whose build side the swap moved left | `GpuJoin{Right}`, probe streams, no finish — a probe row unmatched in this batch is unmatched everywhere, because the build side is complete before the first call |
| **Left outer** (build side preserved) | no — not by the join | a DataFusion Right-outer after the swap | `GpuJoin{Left}`, probe streams **with finish**; the accumulated probe keys are resident until it runs |
| **Full outer** | no — not by the join | — | `GpuJoin{Full}` — Left's finish, Right's per-call emission |
| **Build-side semi family** — `LeftSemi` | no — not by the join | `LeftAnti`, `LeftMark`; IN/EXISTS semi (UNEQUAL) against INTERSECT semi (EQUAL); the filtered forms, which take a single-batch probe and are then one legacy call | `GpuJoin{LeftSemi\|LeftAnti\|LeftMark}`, probe streams with finish. **The per-call join disappears**: a probe call is only the key project, so the build side is not touched until the finish consumes it |
| **Probe-side semi family** — `RightSemi` | no — not by the join | `RightAnti` | `GpuJoin{RightSemi\|RightAnti}`, probe streams, no finish — membership in a complete build side is a per-row question |
| **Cross join** | **yes**, both sides | — | `GpuCrossJoin`, probe streams, no finish; both inputs one lane (broadcast is [#140](../tickets.md#t140)) |
| **Nested-loop Inner** | **yes**, both sides | a predicate that is not AST-able (tpch q11/q22), which takes the cross-then-mask path | `GpuNestedLoopJoin{Inner}`, probe streams, no finish |
| **Nested-loop Left** | **yes**, both sides | — | `GpuNestedLoopJoin{Left}` with a **single-batch probe**: `GpuCoalesceAllBatches` under the probe side too, since #136's finish trick accumulates keys and a predicate join has none |

**On that second column.** A merge is not something a join asks for, it is what a *shuffle*
needs: `GpuEmitPartitions` takes a single-lane input, so any plan that hash-partitions a
side puts a `GpuMergePartitions` under the emitter. A side already co-located on the join
keys — a scan partitioned that way, or the output of an earlier join on the same key —
needs neither node. Cross and nested-loop joins are the exception, and it is the join
itself asking: with no key to co-locate on, both inputs must be one lane.

**An interleave needs its branches to agree on lane count**, and they may not. `can_interleave`
takes output lane p from lane p of each input, so a union whose branches carry different lane
counts has no such correspondence and becomes a `GpuUnion` instead — interleaving would
preserve a distribution the branches no longer share. tpcds q77 is the case: its store and web
branches join on their channel key and stay four lanes hashed on it, while the catalog branch is
`FROM cs, cr` — a cross join, which asks both its inputs onto one lane — so that branch arrives
with one lane and no distribution at all. The union then declares 4+1+4, and the partial rollup
above it runs in nine lanes rather than four.

**Equal lane counts are not co-partitioning**, which is what the column's `no` hides.
DataFusion picks `CollectLeft` for two small tables and emits no repartition at all, while
both loaders still produce N lanes — so lane p of one side holds nothing that must match
lane p of the other, and joining lane-wise would silently drop matches. The translation
therefore checks the hash and not the count: unless both sides are scattered on their own
join keys, in key order, both merge to one lane. That is the broadcast shape
[#140](../tickets.md#t140) removes. Until then read the column as "no merge beyond the one
the shuffle already did", not "no merge".

**What crosses the wire, and what cuDF runs.**

`Runs on` is what is proved, not what is possible: **device** means a device case runs it,
**CPU** that only the CPU backend does, **refused** that the ABI cannot express it and the
refusal is what the device test asserts.

| Mode | fb seqs: per probe call / at finish | cuDF underneath | Runs on |
|---|---|---|---|
| **Inner** | `CudfHashJoin{Inner, keys, filter, projection, null_equals_null}` / — | `inner_join(bk, pk, nulls)` → two gather maps → `gather(build, ·, DONT_CHECK)` + `gather(batch, ·, DONT_CHECK)` → column concat → residual: `build_column` + `apply_boolean_mask` → `projection` | device, one probe batch ([#152](../tickets.md#t152)) |
| **Right outer** | `CudfHashJoin{Right}` / — | `left_join(pk, bk, nulls)` with the pair swapped back → `gather(build, ·, NULLIFY)` + `gather(batch, ·, DONT_CHECK)` | CPU; a device would take one probe batch (#152) |
| **Left outer** | `CudfHashJoin{Inner}` + `CudfProject`(key ordinals) / `CudfCoalescePartitions` → `CudfHashJoin{LeftAnti}` → `CudfProject`(build columns + typed NULL literals) | per call as Inner; at finish `left_anti_join(bk, accumulated_keys, EQUAL)` → `gather` → one literal-only `compute_column` per padded column | refused — no device path (#152) |
| **Full outer** | `CudfHashJoin{Right}` + `CudfProject`(keys) / as Left outer | per call as Right outer; finish as Left outer | refused — no device path (#152) |
| **Build-side semi family** | `CudfProject`(key ordinals) / `CudfCoalescePartitions` → `CudfHashJoin{LeftSemi\|LeftAnti\|LeftMark}` | finish only: `left_semi_join` / `left_anti_join` (`filtered_join::semi_join` / `anti_join` where the header exists) → `gather`; mark scatters `true` into an all-false column and appends it | device — LeftSemi, LeftAnti and LeftMark, each streamed and answered at done |
| **Probe-side semi family** | `CudfHashJoin{RightSemi\|RightAnti}` / — | `left_semi_join(pk, bk, nulls)` with the sides swapped, as the C++ does → `gather(batch, ·)` | CPU; a device would take one probe batch (#152) |
| **Cross join** | `CudfCrossJoin` / — | `cross_join(build, batch)` | CPU; a device would take one probe batch (#152) |
| **Nested-loop Inner** | `CudfNestedLoopJoin{Inner, filter, filter_columns}` / — | `conditional_inner_join(build, batch, ast)` → `gather` ×2; or `cross_join` → `build_column` → `apply_boolean_mask` when the predicate is not AST-able | CPU; a device would take one probe batch (#152) |
| **Nested-loop Left** | one `CudfNestedLoopJoin{Left, filter}` / — | `conditional_left_join` → `gather(build, ·)` + `gather(probe, ·, NULLIFY)` | CPU; a device would take one probe batch (#152) |

**One worked example per mode.** `dim(k, label)` is the build side and `fact(fk, v)` the
probe throughout, so the shapes are comparable; the plan column shows the join subtree only,
since everything above and below it is the same in every row. Read the fourth column as one
lane: `p` is any lane, and every lane does this.

| Mode | SQL | batch-partitioned plan | inside lane `p`, in cuDF |
|---|---|---|---|
| **Inner** | `SELECT * FROM dim d JOIN fact f ON d.k = f.fk` | `GpuJoin{Inner, on d.k@0 = f.fk@0}`<br>`├─ build: GpuEmitPartitions(k) → GpuCoalesceAllBatches`<br>`└─ probe: GpuEmitPartitions(fk)` | build lands as one resident `cudf::table`; then per probe batch: the build handle is consumed and would have to be copied for the next batch (#152), `inner_join` → two gather maps, `gather` each side, concat the columns, slice to the `projection`, hand the output up as a handle. Nothing at done |
| **Right outer** | `SELECT * FROM dim d RIGHT JOIN fact f ON d.k = f.fk` | same, `GpuJoin{Right}` | as Inner, but the map for the build side carries `JoinNoneValue` where the batch had no match, and that gather is `NULLIFY` — so this batch's unmatched probe rows come out with NULL build columns. Correct batch-locally: the build side was complete before the first probe call |
| **Left outer** | `SELECT * FROM dim d LEFT JOIN fact f ON d.k = f.fk` | `GpuJoin{Left, on k@0 = fk@0}`<br>`├─ build: … → GpuCoalesceAllBatches`<br>`└─ probe: GpuEmitPartitions(fk)` | per probe batch, two calls: `CudfProject` off the batch keeps its key column (that is the lane's growing key table), and `CudfHashJoin{Inner}` off the build emits this batch's matches. Both reads want a copy that the surface cannot give, which is why this row is the one with no device path at all. At done: concat the key tables, `left_anti_join(build, keys)` → the build rows nothing ever matched, then one project that appends a typed NULL per probe column |
| **Full outer** | `SELECT * FROM dim d FULL OUTER JOIN fact f ON d.k = f.fk` | same, `GpuJoin{Full}` | Right outer's per-batch call, Left outer's done call, and nothing else — the two halves are independent because unmatched probe rows are batch-local and unmatched build rows are not |
| **Build-side semi family** | `SELECT * FROM dim d WHERE EXISTS (SELECT 1 FROM fact f WHERE f.fk = d.k)` | `GpuJoin{LeftSemi, on k@0 = fk@0}`<br>`├─ build: … → GpuCoalesceAllBatches`<br>`└─ probe: GpuEmitPartitions(fk)` | per probe batch: one `CudfProject`, keeping the key column. No join call, no build copy, no output. At done: concat the key tables and run `left_semi_join(build, keys)` — `anti_join` for NOT EXISTS, and the mark form scatters `true` into an all-false column and appends it. The whole lane's output is that one table |
| **Probe-side semi family** | `SELECT * FROM fact f WHERE EXISTS (SELECT 1 FROM dim d WHERE d.k = f.fk)` | `GpuJoin{RightSemi, on k@0 = fk@0}`<br>`├─ build: … → GpuCoalesceAllBatches`<br>`└─ probe: GpuEmitPartitions(fk)` | per probe batch: copy the build handle, `left_semi_join(batch keys, build keys)` — the sides swapped, as the C++ does — then `gather` the batch by the returned indices. The batch's surviving rows leave immediately; nothing accumulates and there is no done call |
| **Cross join** | `SELECT * FROM region, nation` | `GpuCrossJoin`<br>`├─ build: GpuMergePartitions → GpuCoalesceAllBatches`<br>`└─ probe: GpuMergePartitions` | one lane only, since there is no key to co-locate on. Per probe batch: copy the build handle, `cross_join(build, batch)` — every build row against every row of this batch — out as a handle |
| **Nested-loop Inner** | `SELECT * FROM dim d, fact f WHERE d.k < f.fk` | `GpuNestedLoopJoin{Inner, filter=k@0 < fk@0}`<br>`├─ build: GpuMergePartitions → GpuCoalesceAllBatches`<br>`└─ probe: GpuMergePartitions` | per probe batch: copy the build handle, then `conditional_inner_join(build, batch, ast)` evaluates the predicate per pair without materializing the product, and two gathers build the output. Where the predicate is not AST-able the shape is instead `cross_join` → mask → `apply_boolean_mask`, which does materialize it |
| **Nested-loop Left** | `SELECT * FROM dim d LEFT JOIN fact f ON d.k < f.fk` | `GpuNestedLoopJoin{Left, filter=k@0 < fk@0}`<br>`├─ build: GpuMergePartitions → GpuCoalesceAllBatches`<br>`└─ probe: GpuMergePartitions → GpuCoalesceAllBatches` | the probe side arrives as one batch, so this is a single call and the build handle is handed over rather than copied: `conditional_left_join` → `gather(build)` + `gather(probe, NULLIFY)`, which emits the unmatched build rows in the same pass. No done call, because there is no second batch for one to wait through |

**A single-batch probe is always the legacy call.** Where the matrix refuses to stream, the
planner puts a `GpuCoalesceAllBatches` under the probe side and the join is one
`CudfHashJoin` / `CudfNestedLoopJoin` over the whole of it — the same node the legacy modes
emit, doing the whole join in one call with no finish pass. That is what makes the fallback
cheap to trust.

**Three shapes are refused at plan time**, per the scope rule for an unsupported shape
inside a supported feature. Two more are refused at run time on a device, for the reason
below rather than for anything about the shape: Left and Full outright, and a second probe
batch on every family that needs the build side per call.

- **Left, Right or Full with a residual filter — [#153](../tickets.md#t153), and it is a
  live defect in the shipping engine, not a limitation of this mode.**
  `execute_hash_join` applies the filter with `apply_boolean_mask` *after* the outer
  gather, so a padded row's NULL columns make the predicate NULL and the row is dropped:
  `LEFT JOIN b ON a.k = b.k AND b.v > a.v` returns an inner join. Latent because
  DataFusion pushes an ON predicate that reads only the nullable side below the join, so
  what is left on a join is a predicate reading both sides — and no corpus query has one.
  Refused here until #153 lands; nothing in this mode's design depends on the outcome.
- **RightSemi or RightAnti with a residual filter**: no swapped `mixed_*` variant exists and
  the C++ throws. The fix is orientation rather than code — keep the emitted side as the
  build, and the join stays a Left form, whose `mixed_left_semi_join` path works.
- **A nested-loop join that is neither Inner nor Left**, which `execute_nested_loop_join`
  rejects outright.

**What a streamed probe costs on the frozen surface.** `execute_node` erases every handle it
reads, and no node duplicates one — passing the same handle twice to a concat fails on the
second read, because the first erased it. So a build side probed by B batches is needed B
times and exists once, which is [#152](../tickets.md#t152) exactly. Per family, per probe
batch: the probe-local types (Inner, Right, RightSemi, RightAnti) would need **one build-side
copy**; Left and Full **that plus a copy of the probe batch**, because the join consumes the
batch and the key accumulation needs it too; the build-side semi family **none at all**, since
its probe calls never touch the build side. None of those copies can be taken — the surface has
no symbol for one, and `slice_handle` moves rather than copies — so what the count actually
predicts is where a device refuses: the probe-local types after one batch, Left and Full
outright. [#145](../tickets.md#t145)'s refcounted handle is what turns each of them from a
refusal into a copy, and then into no copy at all. The prototype counts every copy it takes, so the figure for
a given plan is a test assertion rather than an estimate.

**The `CudfCoalescePartitions` in those finish sequences is a concat, not a partition
operation.** Left outer, Full outer and the whole build-side semi family carry one.

It concatenates its inputs **whole** — it has no idea what a key is. What makes it cheap is
what it is handed. The `CudfProject` beside it runs once per probe batch and writes its
result to a *new handle*: a table of the key columns and nothing else. The probe batch's
own handle goes to the join (or, for the semi family, is consumed by that project and goes
nowhere else). So a lane arrives at its finish holding k tables that are already key-only,
and the concat does the ordinary thing to them. Nothing selective happens anywhere: the
narrowing is the project's, one batch at a time, and the concat merely inherits it.

Three things follow. It is *that* node because its arity is a call argument rather than a plan
constant: the collapse arm concatenates whatever k handles the call hands it, and k here is
a runtime number, how many batches this lane happened to see. (`CudfUnion` also
concatenates, but declares its inputs in the node.) It fires only where k > 1 — one probe
batch hands its key table straight to the finish join. **What k = 0 does is the executor's
answer, not the concat's**: a lane that saw no probe batch has no key table to hand over, and
since a concat of nothing throws ([#173](../tickets.md#t173)) the finish has to answer from the
build side alone — LeftAnti over an empty key table is every build row, which is what the CPU
does and what the device must be made to do. And the word "partitions" in its name is the wire's
vocabulary, not this mode's; `ScanBatch` sets the same trap in architecture.md.

**Why keys and not the probe rows.** The alternative to accumulating keys is not "a smaller
concat" — it is the single-batch probe, a `GpuCoalesceAllBatches` under the probe side, and
that changes the join rather than its inputs. One probe batch means one join call, and one
call means the whole *join result* materializes as one table; on a fan-out join that result
is the biggest thing in the lane. Accumulating keys keeps the output leaving a batch at a
time and holds only the key columns between calls.

Held bytes alone would not settle it: on tpch q3 the keys are about a third of the probe
side (8 B/row against 24.5, from the `.cpu.txt` goldens #152 quotes), and Left and Full pay
that back in the per-batch probe copy. The output is what settles it. For the semi family
it is not close in any case — those lanes make no per-batch join call at all, so neither
the probe side nor the join result ever exists as a table.

**What the finish pass computes.** "Which build rows matched at least once" is the one fact
a streamed probe loses, and it never crosses the current ABI:
`peacock_executor_execute_node` returns a table and row counts. Hence the trick — keep the
probe keys, and at done let one `left_anti_join` (or `left_semi_join`) against them answer
the question in a single call. The build side is therefore built once more at finish, an
accepted v1 cost recorded with the ABI-change alternatives in
[#136](../tickets.md#t136).

**And it computes it with the legacy semantics, not a variant.** `null_equals_null` rides
on `GpuJoin` per join and reaches the finish join too, hardcoded `EQUAL` for anti and mark
included — so a build row with a NULL key is matched by a NULL probe key at finish exactly
as it would be inside a legacy single call, three-valued `NOT IN` trap and all (#80, #59).
That equivalence is the point: the finish pass has to be a substitute for the legacy join,
and a lowering that quietly fixed the null semantics would not be one.

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

These types live in `peacockdb-core/src/batch_partitioned/` as of T3; the sketch here is
the argument for their shape, and the code is what they are. Four the sketch left implicit
are settled there: `PlanError` is `Unsupported` (a shape refused at plan time) or `Invalid`
(a node's requirement on its children); `RowInterval` is `{skip, fetch}`; `Forwarder` is a
three-variant enum over the mappings; and a `Schema`'s column types are an arrow
`SchemaRef`, so decimal precision and scale stay the planner's own rather than a copy.

`UniqueKeys` / `UniqueScope` are deliberately not here, though `scripts/exec_model/layout.py`
declares them. Nothing reads them, by the prototype's own account — they are for later work
— and a field declared everywhere and read nowhere is the [#130](../tickets.md#t130) shape
this spec exists to avoid. The aggregate sequence knows each scope for free, so the
declaration costs nothing to add on the day something consumes it.

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
    type Context;          // GPU: the session pointer and the recipe plan; CPU: a TaskContext
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

**A call can fail, and failing ends the query.** Every executor method returns
`Result<(…, CallStats), BackendError>`, and `BackendError` carries a message and no kind,
because there is one response to all of them: stop. The driver adds the node and lane it was
calling and fails the query — no retry with a smaller batch, which is [#142](../tickets.md#t142)'s
adaptive future and not this design.

What the C ABI actually guarantees, since the rule follows from it.
`peacock_executor_execute_node` calls `session.reset()` on any exception
([`gpu_executor.cpp`](../../cpp/src/gpu_executor.cpp#L219)), dropping the plan and every
resident intermediate — so after that failure no handle is usable, and the next call reports
"no plan loaded" rather than misbehaving. `peacock_result_from_handle` does *not* reset, so a
failed export leaves the session and its handles intact. And `peacock_handle_release` is
null-guarded, which makes releasing a handle into a reset session a silent no-op rather than a
fault.

So the driver needs no separate teardown: on a failure it stops scheduling and releases what it
holds, which keeps holds equal to releases. That is one line more than it sounds, and the line
is not in the accountant — the early exit walks the queues, and the batch handed to the call
that failed is not in a queue, it went into the call. **The failure site releases it**, where
the successful path releases the same batch. The rest of the rule is about use rather than
release: **no handle is touched after a failure** — no unload, no further call, no result read.

A trip says which check produced it, as data rather than as words: `RunError::BudgetExceeded`
carries the phase, and the sentence carries it too, for a reader. Today both phases end the
query. Whether a pre-call trip should instead be recorded and the call allowed — the model is an
estimate, and refusing on an estimate is a policy rather than a fact — is deliberately open; the
type is what leaves room for it, and #142 is where the recourse question lives.

`resident_bytes` and `scratch_bytes` stay infallible, and the line is between a method that does
work and one that reports a number the executor already holds. That is forced rather than
chosen: an accountant handed a failure instead of a figure has nothing to do with it — zero stops
the enforcer enforcing, unbounded kills a query over a reporting hiccup, and skipping the check
disables the guard silently. `CallStats` is the visible tell, since every fallible method returns
one and neither of these does. A mock backend scripts a failure at a chosen call, which is how
the path is tested without a GPU.

**Illegal calls are unrepresentable rather than checked.** Every method that ends a
protocol consumes `self`, so four run-time guards the prototype needed become
compile errors: probing before `set_build`, calling `set_build` twice, probing after
`finish_and_fetch`, and accumulating after `mark_done_and_fetch`. The source's
consuming step removes a fifth thing — the driver's own `finished` flag, which today
duplicates the executor's exhaustion and can disagree with it.

**Static all the way down on the production path.** `B::Batch` is a concrete type, so a
`GpuBatch` is its `u64` handle with no box and no vtable, and `Drop` is a direct call.
Backend choice is a turbofish at the entry point — `batch_partitioned_driver::<GpuBackend>(…)`
— not a `BackendSelector` consulted per node. The whole driver is instantiated twice, once
per backend, and the mock backend the driver tests use is a third instantiation.

Going static also *simplifies* the typestate. Stored as `Box<dyn …>`, consuming methods
needed `self: Box<Self>` (a bare receiver is not `dyn`-compatible), and `JoinExecutor` could
not carry an associated `type Probing` — it had to return a `Box<dyn ProbingJoin>`, since an
associated type forces every `dyn JoinExecutor` to name it. `E0038` and `E0191`
respectively, both compiled to confirm. Neither constraint survives the
switch, so the declarations above are both the simpler form and the one that compiles.

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
in and a `CpuBatch` out; once `exec` is `B::Batch -> B::Batch` it cannot. An improvement
rather than a tax: unload is the only place data leaves the device, and the type now says
so. The driver still collects the root node's output batches, and `NodeKind::Sink` remains a
plan-level fact.

**Instantiation model.** Lane-scoped categories — Source, Exec, BatchAccumulator, Join,
Unload — get one executor instance per (node, lane), created when the driver first enters
that lane; PartitionAccumulator and PartitionEmitter instances are one per node, since they
are the cross-lane points. A BatchForwarder is neither: it has no backend and no executor
at all, so the driver owns its rotation directly. The enforcer's `Σ resident_bytes()` runs
over instances, not nodes.

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
// Backend::executors_for(ctx: &Self::Context, node: &dyn GpuNode, post_order: usize,
//   lane: usize) -> Result<NodeExecutors<B>, PlanError>. A fresh instance set per call, so
//   the driver instantiates per lane; `lane` is needed because a loader's lane picks its own
//   row groups out of the partitioner's mapping. `post_order` is how a GPU executor finds
//   its node's recipe, which is keyed that way because the FFI addresses nodes that way —
//   neither `&dyn GpuNode` nor an identity map is an address. It returns a Result because
//   this match is where "does this backend implement this node" is answered, and that
//   question has a no — a GPU window (#32) is the standing case.

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

### Aggregators

Two enums, not one trait. The [Aggregates](#aggregates) table becomes a `const`
declaration keyed by what sql asked for; nothing dispatches on an aggregate at execution
time, because the plan already names what to run.

```rust
enum AggFunc { Sum, Min, Max, Count, Avg, Stddev, Var }        // what sql asked for
enum PlanAgg { Sum, Min, Max, Count, Mean, M2, MergeM2 }       // what a node runs

struct Decomposition {
    state: &'static [(&'static str, PlanAgg)],   // suffix and the aggregator producing it
    merge: Merge,
}
enum Merge {
    PerColumn(&'static [PlanAgg]),   // positionally, one per state column
    Combined(PlanAgg),               // merge_m2: three columns in, three out
}

const fn decomposition(f: AggFunc) -> Decomposition { /* the Aggregates table, verbatim */ }
fn finalize(f: AggFunc, call: &AggCall, state: &[ColumnRef]) -> Expr;
```

`state` is pairs rather than two parallel lists so a column and the aggregator producing
it cannot desync. `merge` is listed rather than derived from `state`: the rule would be
"same aggregator, except count merges by sum", and that exception is the whole content.
`Combined` exists only because `merge_m2` is not a per-column reduction; a general
(inputs, aggregator, outputs) form would cover it and buy nothing.

Finalize stays a function. It needs `ddof`, the declared decimal type and the null guard,
none of them per-aggregator constants — as data it would need an expression dsl in a
`const`.

The two enums overlap on four names and that is not a reason to merge them: `Avg` is never
a `PlanAgg` (decomposing it is the point) and `MergeM2` is never an `AggFunc`. One enum
would make some variants illegal by convention in each position, which is the defect this
design removes.

`PlanAgg` is declared in `gpu_plan.fbs` so flatc generates it for both languages and the
C++ copy is not hand-maintained; the wire cost is paid regardless, since
`AggregateFuncNode.name` stops meaning the sql function. The fbs trades `AggregateMode`
for it. What remains in C++ is two exhaustive matches over the generated enum — groupby
and reduce — replacing today's two independent string chains in
`cpp/src/operators/aggregate.cpp`, which already disagree about `count`. Note that
exhaustiveness there is `-Wswitch`, a warning: the compile-time guarantee is real only on
the rust side unless the build promotes it.

The table grows a row per aggregate. It does not grow a variant for an aggregate that
cannot be decomposed at all (a true median): that is an absent `Decomposition` and a
planner that declines to split the phase.

## Drivers

Two drivers in `batch_partitioned/driver/`, both single-threaded, push-based, deterministic
and generic over `Backend`, with the schedule and the accounting as units of their own.

| File | What it owns |
|---|---|
| `partitioned.rs` | the tree, the queues, the run loop, the three cross-lane categories (`PartitionEmitter`, `PartitionAccumulator`, `BatchForwarder`), and the row-range decision at a `GpuUnload` carrying a limit |
| `single_partition.rs` | one lane of one lane-scoped node — Source, Exec, BatchAccumulator, Join, Unload — as that executor instance's state machine: which call the lane's input state calls for, one call, then the outputs and whether the lane can produce again |
| `scheduler.rs` | which node runs next, from plain numbers: no backend, no batch, no executor |
| `index.rs` | the tree as the schedule and the accountant see it — heights, pre-order subtree ranges, per-node category, executor slots |
| `accounting.rs` | the resident total, the per-slot executor cache, the budget checks |
| `mock.rs`, `plans.rs` | test-only, and outside `tests/` because both test suites share them: a scripted backend, and plan builders that reach shapes no query produces |

A chunk is **one node's lane**, not a chain of them: min-height selection walks a batch up a
chain node by node on its own, so a chain-walking driver would duplicate the scheduler.

Three things the drivers refuse rather than absorb. A tree whose interval-carrying node feeds
only the sink is not canonical — the planner puts that interval on `GpuUnload` — and
`Driver::new` asks `validate.rs` rather than carrying its own copy of the rule, so a mock plan
meets the same refusal a planned one does. A backend that returns executors of a different
category than the node is fails where it was built, not at the first call that finds the wrong
method — `NodeExecutors::category()` against `category_of` off the `NodeRef`
registry. And an `emit` returning other than the plan's lane count is a run-time error, as
the trait's own note says it must be. Both are `RunError`, which is where run-time failure
lives now that `PlanError` is plan time only: a budget trip, a protocol violation, a backend
that has no executor for a node.

`Driver::new` runs the rules its own correctness rests on rather than the whole of `validate()`:
the planner owns the rest, and a rule moves into `check_canonical_form` when a driver need is
shown for it, as the limit-position rule did.

### The scheduling rule

Every node carries a **height** (distance to the root, root = 0) and an **order**
(pre-order index, which in a tree is left-to-right within a level). Both are pure
functions of the tree, computed once.

A node is **runnable** when any of its partitions can make progress: a source always can,
and any other node can once that lane's inputs hold a batch or are known to be finished.
Among all runnable nodes the driver takes the **smallest height**, breaking ties
**leftmost**, and then runs **every lane** of that node.

The unit that becomes ready is not always an output lane, which the schedule has to carry
separately: a `PartitionAccumulator` has one output lane and becomes ready one *input* lane
at a time, and an emitter reads a single input lane whatever it emits. Counting output lanes
there gives a driver that never schedules a merge's later lanes.

The schedule is maintained incrementally rather than rescanned: a rank order from
(height, order) computed once, a ready bitset over it, per-node ready-lane counters, and hold
counters — counters, not flags, since one node can sit in two joins' probe subtrees at once.
What that costs is worth stating, since it is the reason for the shape. A pick is the lowest
set bit — one word scan per 64 nodes. A step re-checks the node that ran and its parent, and
nothing else, because nothing else can have changed: a node's readiness is a fact about its
inputs, so consuming from a child does not move the child's own. Each join stamps its probe
subtree once as the schedule is built and once as the hold lifts, and a satisfied limit stamps
its subtree once and never lifts — so the holds are O(nodes) per join and per limit across the
whole run rather than anything per step. The prototype instead re-derives the entire predicate
every step: every node, every lane, and both hold chains to the root.

A naive rescan survives as a test-only oracle
([`the_incremental_schedule_picks_what_a_full_rescan_would`](../../peacockdb-core/src/batch_partitioned/driver/scheduler/tests.rs#L365)),
compared pick by pick over seeded random shapes, because an incremental schedule that
disagrees with it is wrong by definition.

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

Per batch the driver then makes the three-way decision the limit lowering rule states —
outside the interval, straddling an end, inside it — before calling `unload`. A mid-plan
`GpuLimit` makes the same decision inside its own executor, slicing rather than narrowing an
export.

### The test surface

Both drivers are tested against a mock `Backend` — the third instantiation, in
`driver/mock.rs` — whose operators are scripted rather than computed: batch counts and
sizes per source, an `ExecRule`/`AccRule`/`EmitRule`/`JoinRule` per category, a miswired
variant that returns the wrong executor kind, and instrumented and uninstrumented forms so
`CallStats::scratch_bytes` is present in one and `None` in the other. Plans are built from
the helpers in `driver/plans.rs`, not from sql, so a shape no query produces is one
line.

The tests assert calls, queue bounds and release rather than returned rows: a test on the
rows passes just as well when the whole input was read and held.

| Where | What it holds |
|---|---|
| `scheduler/tests.rs` | the pick is min height and ties break leftmost; a node is ready while any of its lanes is, and readiness comes and goes; a join holds its probe subtree from time zero and lifts only when every lane has left build; a node under two builds stays held when the inner one lifts; a join inside another's build subtree is not held; a limit's hold and a join's are independent; `LIMIT 0` is satisfied before a step; satisfying twice holds once; and the incremental schedule picks what a full rescan would, over seeded random shapes |
| `tests/flow.rs` | a batch reaches the root before the next is produced; every lane of the chosen node runs; queues stay at one batch per lane with no cap; a dry lane does not stall its siblings; empty batches are carried and empty scatter outputs are dropped at the emit; probe-side queues stay **empty** until the build is set, transitively, with nested joins resolving outermost-first; both build-side protocol violations; merge, union, interleave and cross-lane accumulator routing; a join's finish pass reaching the root; two runs, identical traces |
| `tests/limit.rs` | early exit and its release path; the skip prefix never unloaded; only straddling batches narrowed; the count taken across lanes, not per lane; zero fetch, offset with no fetch, skip past the end; the exit reaching through a shuffle; a mid-plan limit stopping its own subtree and holding nothing; a satisfied limit reporting done so the node above it can finish |
| `single_partition/tests.rs` | the lane driver as a state machine, with no tree and no schedule: a source runs to exhaustion, exec is one call per batch, an accumulator emits only at done, a join sets build before it probes, a finished lane refuses another call, the executor is built on the first step rather than at construction, and `can_step` agreed with what `step` does over every category crossed with every input availability — an accepted state yields a call or a named violation, never a silence |
| `tests/stress.rs` | one plan re-run at five shapes — one lane one batch, one lane many, four lanes one each, four lanes many, every key into one lane — each asserting rows delivered equals rows given, queues at one batch per lane, and holds equal releases; then the same five with a node the plan did not ask for injected above every source |
| `tests/budget.rs` | the accountant's decisions at their boundary, every budget derived from an unbudgeted run's peak rather than chosen: equality completes and one byte below trips, naming the node the peak occurs at; a priced emission refuses the call before it runs; a silent model lets the same peak pass a budget nothing checked it against; and a limit above an accumulator saves nothing, while the same interval over the scan does |
| `tests/failure.rs` | a backend failure at each call shape — a source step, a mid-chain exec, an emit, a join probe, and the two that consume their executor — each asserting the query fails with the node, the lane and the backend's own words, that scheduling stops, and that holds still equal releases across both release paths |
| `accounting/tests.rs` | resident is in-flight plus executor state; the cached total tracks a live sum; forget on a consuming call; release without hold is an error; both checks trip, pre-call and post-call; the peak is a high-water mark; an under-predicting model is recorded with its magnitude and never enforced; an absent measurement is not an underestimate |
| `tests/memory.rs` | the same properties through a whole run: a tight budget fails the query cleanly and a generous one records a peak; a consumed input stays accounted through its call; an accumulator's residency is visible while it holds rows; a join reports its build side while probing; one bound asserted at two partitionings, since a residency defect can be invisible at one |
| `driver/index/tests.rs` | what `PlanIndex` derives rather than what a plan happens to produce: pre-order numbering with `parent` and the snapshotted children, the contiguous subtree range every hold rests on and a join's probe range read off it, the three counts a category changes, and `slot_base` where it is lane-scoped against where it is not |
| `driver/tests/wiring.rs` | the post-order each backend is handed, recorded by the mock and checked against a children-first walk written in the test — over a plan with a scatter and a sorted merge under a join's probe side, so all three `executors_for` call sites are reached and not only the lane one |
| `driver/tests/instrument.rs` | the instrument against its own script, since every assertion in `driver/tests/` is measured by it: per-lane batch counts and sizes, the lanes an emitter's skew fills and the ones it leaves empty rather than absent, and an accumulator emitting where its rule says |

Both drivers take the resident-accounting hooks below and fail the query when the enforcer
trips. A backend failure ends it the same way and by the same path — the convention is with the
[executor traits](#traits), since it is their return type that carries it.

## Memory accounting

Prevention lives at plan time; detection at run time.

**Plan time.** An estimator pass computes `estimated_max_resident_size` per node in
rows × row-width vocabulary (the `subtree_max_row_bytes` family), rendered in each plan
golden's `--- memory ---` section. Because `GpuMergePartitions` polls round-robin, all N lanes are
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
quality is observable: under-estimates are recorded with their magnitude. The three calls that
consume their executor — `mark_done_and_fetch`, `finish_and_fetch`, an exhausted `next_batch` —
skip the post-call residency read and forget the slot instead, a consumed executor holding
nothing; the budget check still runs. `set_build` needs no such case, since the successor
reports for the same slot. Both backends
measure — the CPU directly, the GPU through RMM allocator hooks — so `None` means this run
was not instrumented, not that the backend cannot report.

Four things about `driver/accounting.rs` are load-bearing rather than incidental.

- **The executor total is a cache refreshed one slot at a time, never a sum over live
  executors.** A slot is a dense index per executor instance; the sum is what a prototype
  reaches for, and it is also what would force the accountant to hold references to executors
  the driver owns mutably.
- **A batch's size is read once, when it is held, and the same figure is released.** An arrow
  batch recomputes its size by walking every array, so a second read is a second chance to
  disagree — and a released figure that differs from the held one drifts the total with nothing
  going red.
- **Holds and releases are counted, not netted.** A total back at zero is also what releasing
  something never held would leave behind, so the pair is the check and the total is not. This
  is the invariant on the early-exit path, where a satisfied limit ends the run with queues
  still full and every one of those batches released rather than drained.
- **A trip carries no names.** It is a slot and two figures until the driver formats a message
  on the path that ends the query, so the check costs no formatting per call. A budget of `None`
  accounts and reports without ever tripping, which is what the flow tests run under.
- **The peak is an observation, and the checks are the enforcement.** They do not see the same
  total, and T13 measured the gap: the peak is raised in `hold`, where a batch enters a queue,
  while the post-check runs after the emitting executor's slot has been refreshed or forgotten.
  A buffering node holds its state and its output alive together for the length of one call —
  precisely what `cudf::concatenate` does — so its transient raises the peak and no check sees
  it. A budget below a reported peak can therefore complete, which is a fact about where the
  two numbers are taken and not a hole to be patched by re-ordering them: after the call, the
  state really is gone.

**What prices a transient is the pre-check, so pricing it is an obligation.** An accumulator's
`scratch_bytes` on its emitting call must include the output it is about to build; the model may
consult `&self`, which is what that permission is for. A model that returns zero there is not a
cheap call, it is a guard switched off — and it fails open, since nothing goes red. The driver
tests pin both halves: a priced emission refuses the call before it runs, and a silent one lets
the same peak pass a budget it was never checked against.

**Model ≥ measured is not an invariant.** `scratch_bytes` rests on the optimizer's
cardinality figure for a join and on assumed selectivity for a filter, so it will sometimes
come in low, and asserting otherwise would make the suite red for something that is not a
defect. The enforcer's contract is "fail cleanly when an accounted total at a check point
exceeds budget", not "the budget is never exceeded" and not "the peak stays under it" — the
same class of guarantee as the legacy `ResidentEnforcer`.

**Both checks end the query today**, and the pre-call one does so on an estimate rather than on
a fact. Whether it should instead record and let the call proceed is open rather than settled:
the error carries which check tripped, so something can branch on it, but there is nowhere to
record into — `RunReport` has no trip log, and `Underestimate` is the precedent for what one
would look like. [#142](../tickets.md#t142) holds that question with the other recourses.

### What the rollout left binding

The T0 prototype ran the whole corpus under a 2 GiB accountant — 558 runs, three layouts, both
join backends — and the formula survived it. Four rules did not come out of the design and had
to be measured; each one holds for the Rust implementation and the case behind it is in
[`archive/designs.md`](../archive/designs.md).

- **`resident_bytes()` is a total for the enforcer to check, never a numerator for a per-row
  cost.** Anything dividing it wants the part that scales with build rows, and only the
  executor knows which part that is. This mispriced one call at 2.0 TB and declined a query
  whose whole run peaked at 11.5 MB.
- **A build-preserving join's residency grows with the probe side**, not the build side, since
  it holds key columns for every probe row it has seen. Plan time must charge that per lane,
  for all lanes live at once — and the CPU backend never pays it, so it cannot be used to
  price it.
- **A memory bound asserted at one partitioning is asserting about one shape of arrival.**
  Only a streamed probe accumulates; a single-batch probe accumulates once and finishes. Two
  corpus queries passed at one layout and failed at two others.
- **Zero rows is not zero bytes, and a zero peak is a defect.** A batch of no rows still costs
  its schema; an empty lane emits no batch at all, on either backend, which is a different
  thing and settled in T16. Every run asserts a peak above zero and
  `in_flight` back to zero at the end. `peak <= budget` is *not* among them, and T13 measured
  why: the peak is observed where a batch is held, and the post-check runs after the emitting
  executor's state has been forgotten, so a buffering node's transient — its state and its
  output alive together, which is exactly what `cudf::concatenate` holds — raises the peak
  without any check seeing it. What prices that transient is the pre-check, so an accumulator
  **must** include the output it is about to build in its `scratch_bytes`; a silent model is a
  guard switched off, not a cheap call. That obligation is **T17's**, with the rest of the
  accounting: `scratch_bytes` belongs to `Executor`, which no task before it implements.

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
structure.

A seq is therefore a construction input to every GPU executor, and where it comes from splits
the work: in T10 the test hand-builds a one-node recipe plan per executor, which is what the
C++ plan-executor suite already does and is enough to prove a kernel; the general mapping
below, over a whole tree, is T14's. A GPU executor cannot be written before something hands
it a seq, and nothing before T10 does.

The mapping:

| GpuNode | fb seqs emitted | driven as |
|---|---|---|
| `GpuLoadParquet` | `CudfScan` | `peacock_executor_execute_scan_rowgroups(seq, row_groups…)` once per batch — the additive entry point, plumbing the existing `row_groups_override` parameter (`cpp/src/peacock/operators.h` ~L32) to the ABI |
| `GpuFilter` / `GpuProject` / `GpuAggregate` | same-kind node (`CudfFilter` / `CudfProject` / `CudfAggregate`) | generic map arm, one call per batch |
| `GpuSort` | `CudfSort` | map arm per batch; per-batch `fetch` for top-N |
| `GpuAccumulateBatchesAndSort` | `CudfSort` + `CudfSortPreservingMerge` | per-batch sort calls, then one merge-arm call at done |
| `GpuMergeSortedPartitions` | `CudfSortPreservingMerge` | one merge-arm call over all sorted handles, partition-major order |
| `GpuCoalesceAllBatches` | `CudfCoalescePartitions` | one collapse-arm call over the partition's batch handles |
| `GpuAggregateBatches` | `CudfCoalescePartitions` + `CudfAggregate` in merge mode, plus a `CudfProject` carrying the finalize where it finalizes | one concat + one aggregate call per compaction, and again at done; the project runs once, at done |
| `GpuEmitPartitions` | `CudfRepartition(Hash, 1→N)` | repartition arm, one call per batch → N handles |
| `GpuJoin` | `CudfHashJoin`, plus finish-pass seqs (key project, concat, anti/semi join, pad project) per #136 — per type in the [capability matrix](#join-capability-matrix), which is where the seq sequence for each join mode is spelled out and tested | map arm per (partition, probe batch), plus a copy of the build handle before each, since the call consumes it (#152) |
| `GpuCrossJoin` / `GpuNestedLoopJoin` | same-kind node (`CudfCrossJoin` / `CudfNestedLoopJoin`) | one map-arm call |
| `GpuLimit` | none — `peacock_executor_slice_handle` on the two straddling batches, and nothing at all on the rest | not a seq: the bounds are runtime values. Root-adjacent there is no node either — the interval rides `GpuUnload`'s fetch |
| `GpuMergePartitions` / `GpuUnion` / `GpuInterleave` | none (union casts are `CudfProject` seqs) | `BatchForwarder` routing in the driver, zero FFI calls |
| `GpuUnload` | none | `peacock_result_from_handle` per handle, over the row range the driver supplies; batches outside a root-adjacent limit's interval are released without a call |

This mapping is a first-class deliverable: documented here, unit-tested (each `GpuNode`
kind → expected seq set and call pattern), because it is the load-bearing trick that
keeps C++ frozen. The fb names in the table are the post-T1 ones (`CudfScan`, `CudfRepartition`, …); the
`GpuNode` column is this mode's own vocabulary, and keeping the two visually apart is what
the rename bought.

**Every aggregate merges as state and finalizes in a project.** A `GpuAggregateBatches` lowers to
`CudfCoalescePartitions` plus `CudfAggregate` in merge mode, and where it finalizes, a
`CudfProject` carrying the finalize expression. One rule with no exception is the point: both
engines evaluate the same expression, so they agree by construction rather than by two
implementations happening to match.

The two appended fbs values this needs, `UnaryOp.Sqrt` and `AggregateMode.Merge`, are
[the aggregate sequence](#the-aggregate-sequence)'s to explain and
[Scope and constraints](#scope-and-constraints)' to approve. Worth knowing here is only why a
merge mode had to be added rather than found: cuDF's `MERGE_M2` is reachable only from an arm that
finalizes on the same call, and these plans stack two merges, per lane and then across lanes, so
the lower one would have nothing to hand upward.

The casts are the part to get right: cuDF's divide takes its scale from its operands, so `avg`'s
denominator goes to `Decimal128(p, 0)` and its numerator to the declared type inside the project —
the rule [Implicit casts become explicit](#implicit-casts-become-explicit) already states, now
reaching a finalize. The fb aggregate takes SQL names, so the writer reconstructs `stddev` from
the schema's `agg_state` annotations rather than from the three aggregators; that is the only
aggregate needing it, since every other one merges per column under its own tag.

**The limit lowering rule.** A per-batch `GpuLimit` call cannot be correct: the fb node's
skip/fetch are frozen per seq, so every batch would be truncated to the same bounds
(two batches → 2× the limit), and the right bound for the last batch is a runtime value
no frozen node can carry. Legacy never sees this because a legacy partition is one batch.

**A scan carrying a pushed-down limit plans one lane and one batch.** Where DataFusion can push
the bound all the way into the source it erases the limit node — `SELECT * FROM nation LIMIT 3`
is a `ParquetExec{limit: 3}` and nothing above it. DataFusion is safe because its scan is one
partition; our lane count is our own decision, so four lanes each honouring `limit=3` answer
with twelve rows. The batch count is the same defect one level down, and the ABI is what makes
it one: `CudfScan.limit` becomes `set_num_rows` on every `execute_scan_rowgroups` call
(`cpp/src/operators/scan.cpp` ~L61), so B batches answer with B × limit rows, exactly as a
frozen `GpuLimit` node would. One lane and one batch make the loader's own limit the whole
answer, and the limit bounds what that batch reads, so a source is never sized by more than
what was asked of it. Same shape as the small-table rule, and no new node. The corpus case is
`scan-limit`, whose plan is a bare scan at ten rows with nothing above it.

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

**Intervals nest.** Two on one root-to-leaf path are legal, and each counts the stream it is
handed rather than the one below it — which is what nesting means, not a defect. DuckDB plans a
nested pair as two stacked limits, and DataFusion composes rather than refuses: `combine_limit`
merges them where the child is the immediate input and both bounds are literals, so only the
non-adjacent form — a limited subquery under a join, a limit at the root — reaches this layer
as two intervals. Nothing here rejects it.

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
skip/fetch are correct only against a table starting at row 0 of the stream: drop the offset
prefix and every bound shifts by where the batch boundaries fell, a runtime amount depending
on upstream selectivity and fan-out. So a frozen-bounds node must hold the prefix, and
`OFFSET 1000000 LIMIT 10` would hold a million rows to return ten.

Three additive C++/header changes, all in `gpu_executor.cpp` + `peacock_gpu.h`, all
beside the existing symbols with legacy paths untouched:

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

The two limit symbols are not interchangeable — one produces a result, the other a handle —
though the fetch range could in principle be dropped in favour of slicing and then exporting
whole, at the cost of one bounded device copy immediately before a PCIe transfer of the same
rows.

## What the frozen surface costs, and what unfreezing would buy

The frozen-surface preference is a choice, not a law ([Scope](#scope-and-constraints)), and
the join lowering is where its bill comes due — the join is the only operator whose state
has to outlive a call. Five costs, each with the smallest unfreeze that removes it.

**A sixth arrived with T17, and it is a different kind: three refusals, one wall.** Nothing on
the surface makes a table out of nothing, so a collapse of no handles, a merge of no runs and a
finish whose probe produced no keys each refuse by name ([#173](../tickets.md#t173)); a Right,
Full or RightAnti lane whose build side was empty owes its probe rows padded and cannot make them
([#175](../tickets.md#t175)); and `PlaceholderRowExec`, an aggregate DataFusion answers from
parquet statistics, is a table of literals with no input at all ([#158](../tickets.md#t158)).
The five above are costs — the engine runs and pays. These are shapes it declines, and two of
them are reachable from the corpus today: tpcds q77 is out of the end-to-end list for the second,
and `SELECT count(*) FROM nation` is the third.

The unfreeze is one call — make a table of a schema and a literal row count — and it is the
cheapest on this page after the fbs semantics change below. What makes it worth deciding rather
than deferring is that a refusal here is not a slow path: the CPU can answer all three, so every
one of them is a shape where implementing the obvious thing would make the oracle disagree with
the engine it is checking. They
are ordered by what they buy, and the last one is not an ABI matter at all. Deciding them
is [#155](../tickets.md#t155), which exists because they overlap: three of the five are
removed by more than one of the changes, so taken one at a time they buy the same thing
twice.

| Cost | Why the frozen surface causes it | The unfreeze that removes it | What that costs |
|---|---|---|---|
| **A build-side copy per probe batch** (#152) | `execute_node` erases the handles it reads and nothing duplicates one, so a build side probed by B batches is needed B times and exists once | [#145](../tickets.md#t145): `TableResult` becomes a shared owner plus a view, so sibling handles share one table | **no ABI change** — a handle stays a `u64`. 35 call sites across 11 C++ files; the frozen thing here is the *implementation*, not the surface |
| **A probe-batch copy per batch on Left/Full** (#152) | two consumers, one handle: the join needs the batch and the key project needs it too | the same refcount, or a node allowed to return its input alongside its output | as above; the second form is an fbs *semantics* change with no ABI change (below) |
| **The build side re-hashed per probe batch** (#136) | `CudfHashJoin` is stateless per call — every call builds a fresh `cudf::hash_join` from the build table, so B batches means B builds. Refcounting removes the copy and not this | a join session: `join_begin(seq, build) → id`, `join_probe(id, batch)`, `join_finish(id)`, holding one `cudf::hash_join` | three new symbols and session state keyed by id. The largest of these changes, and the one that also removes the next row |
| **Probe keys held resident, plus an extra join at finish** (#136) | "which build rows matched at least once" cannot cross the ABI: `execute_node` returns a table and row counts | either a match bitmap out-param on the probe call, or the join session above, which tracks it internally | the bitmap is a one-argument ABI delta; the session is the fuller answer. Both delete the key accumulation, the finish concat, and the extra anti/semi join |
| **A new symbol per runtime-varying parameter** | an fb node's fields are plan constants, so anything the driver decides per call — a limit's bounds, a scan's row groups — cannot ride the node. Three additive symbols exist for exactly this | per-call parameter overrides: one `execute_node` variant taking a small override struct | one symbol instead of three, and the next runtime-varying field costs nothing. The generalization is the point: without it every such field is a new entry point |

**The ABI is already more general than the node semantics**, which is worth knowing before
adding to it. `peacock_executor_execute_node` writes into `uint64_t* out_handles` with an
`out_cap` and an `out_count` — k outputs are expressible today, and the repartition arm
already uses that. What forbids a node from emitting two things is the fbs vocabulary,
where every node kind means one output. So "return the input beside the output", which
would remove the Left/Full probe copy, is an fbs semantics change with **no ABI change at
all** — the cheapest unfreeze on this page, and the one nobody has costed.

**One cost on this list is not about the surface.** Every operator exit path deep-copies its
columns into a fresh table where `release()` would move them
([#154](../tickets.md#t154)) — for a join with a projection, twice over. Legacy pays that
once per node per query; this mode pays it once per node per *batch*. No ABI, no fbs and no
golden moves, which makes it independent of everything above it rather than trivial: five of
the ten join sites are a mechanical swap and the rest turn on dangling views, projection
ordinals and who owns a sliced column, which is what the ticket is for.

**What none of this changes** is the capability claim. The
[capability matrix](#join-capability-matrix) is what the frozen surface can already run,
demonstrated rather than argued; everything above is the price of running it that way, and
each row is a decision that can be taken later, on its own, without disturbing the
lowering.

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
- **No sort here preserves tie order.** DataFusion sorts through `sort_unstable_by`
  (`lexsort_to_indices`) and cuDF's `sorted_order` is unstable too, so which of two tied rows an
  ordered `LIMIT` returns is decided by neither engine's contract. Each is reproducible for a
  given plan and input, which is what the scope note below asks for; equal answers across
  batchings under ties is a stronger claim, and the tie-break that would buy it belongs with the
  task that needs it. The accumulating sort still orders and slices rather than taking a top-N,
  because a heap's selection moves with arrival and a slice's does not.
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

**Device labels**: `bp-<tp1|tp4>-<single|rowgroup|sized>`, minus `bp-tp1-sized` which does
not exist, with the budget tier suffixed for execution goldens
(`bp-tp4-sized-mini.cpu.txt`, one file holding every query in sections — see T18). The label names the batching form rather than a
single/batched pair, because `batched` meant `PerRowGroup` at tp1 and `Sized` at tp4 — one
word for two behaviours, which is the shape coding-style.md warns about. The
`partition_mode`-style label lookups stay explicit parameters at call sites, per the
coding-style case.

### Node display

The legacy per-node line is `<Name>: <node fields>, partitions=N, output_rows=R,
output_bytes=B`, with a `pK: in_rows=… out_rows=… out_bytes=…` sub-line per partition. That
skeleton is kept — it is what makes the two mode families comparable by eye — and four
things change inside it.

**A `FilterExec` projects as well as filters**, and its `projection` is part of the node —
`GpuFilter` carries it, rebases its layout through it, and prints it, exactly as the fbs's
`CudfFilter` already does. Dropping it declares the child's columns while emitting fewer, so
every ordinal above the node is off by one.

**Every column reference renders `name@ordinal`.** Three conventions coexist today, all
three ours — each wrapper's `extra_display_info` composes its own text, so none of it is
DataFusion's to fix. Filter predicates, project expressions, sort keys, join keys and
hash keys already print `name@ordinal` (`on=[(c_custkey@0, o_custkey@1)]`,
`partitioning=Hash([c_count@0], 8)`). Against that, `group_by=[c_count]` and the scan's
`projections=[c_custkey, c_mktsegment]` print a bare name with no position
(`operators/aggregate.rs` ~L25, `operators/scan.rs` ~L157), the post-filter and post-join
`projection=[0, 1]` prints bare positions with no name, and `aggr=[sum(lineitem.l_quantity)]`
prints a fully-qualified *logical* name that is neither — in a final aggregate it names a
column that is not even in the input, since the argument there is positional state. The new
renderer has one rule, the one the aggregate sequence already relies on: the ordinal is
authoritative, the name comes from the declared schema at that position, and they are
printed together. A reader can then follow a reference without holding the child's column
order in their head, and a name that disagrees with its ordinal is a visible defect rather
than an invisible one.

**Layout replaces the lane count.** `partitions=N` becomes the whole `PartitionLayout`,
because in this mode it decides what a parent may assume: `lanes=N, batches=single|multiple`
on every node, plus `hashed_on=[…]` and `sorted_on=[…]` where the layout carries them.
Those two print only when specified, so their absence is `NotSpecified` and reads as the
fact it is. The execution goldens drop legacy's per-partition sub-line, whose numbers the
per-batch lists below carry per batch rather than per lane — that half arrives with T18, since
nothing executes this mode yet.

**Every node that carries a `fetch` prints it.** A merge that turns 80 rows into 10 must say
so on its own line; today only `GpuSortExec` does, and the merge above it is silent (see
[the sort decomposition](#the-sort-decomposition)). Same for the aggregate's `aggs` and
`final` lists, the loader's `partition_groups`, and a limit's interval wherever it lives —
on the `GpuLimit` mid-plan, on the `GpuUnload` root-adjacent.

**Types move into the plan golden, not the execution golden.** The declared output schema
per node — `name:type` per column — is a plan fact, so it belongs beside the layout in
`<mode>.plans.txt` and is not repeated per query in `.cpu.txt`. Both files carry the node line
and its fields — `.cpu.txt` is a plan golden that also holds what execution produced, which is
what lets one comparison catch a plan that moved and a row count that moved, as legacy's does. Printing it there is what makes the explicit casts legible: a
`Decimal128(38, 6)` in a `final` expression means nothing without the state column's declared
scale beside it. What it does not do is check anything — a golden records what the planner
declared, and the declaration is exactly what a wrong type would move. The check that a
reference's name matches the field at its position is what closed [#135](../archive/archived-tickets.md#t135)
on the rust side; comparing a declared type against the expression that produces it is
[#163](../tickets.md#t163), and the C++ half is [#164](../tickets.md#t164).

Node names lose the `Exec` suffix, since these are not DataFusion nodes — `GpuLoadParquet`,
`GpuAggregateBatches`, `GpuEmitPartitions` — and after the T1 rename the legacy vocabulary
reads `Cudf*`, so a line from either family says which mode produced it without a caption.
What deliberately does not change: the indentation-as-tree shape, `output_rows` and
`output_bytes` (they are the CPU/GPU cross-check, and their meaning is unchanged), and
`batches_to_sorted_str` result comparison.

**Plan goldens** (5 modes: bp-tp1-single, bp-tp1-rowgroup, bp-tp4-single, bp-tp4-rowgroup,
bp-tp4-sized): one file per mode holding all queries — `goldens/<bench>/<mode>.plans.txt` —
because the per-query files would be small and numerous. A query's section is `== <query>`
followed by the tree and its `--- memory ---`, or by a single `refused:` line where the
planner declined the shape and a `refused by datafusion:` one where DataFusion did. A
refusal is a golden like any other: it names its ticket, the meta tier checks that the
ticket exists, and the registry's cell for that mode has to agree with it in both
directions.

A parquet source renders its whole mapping as one nested structure on one line —
`partition_groups=[[[0,1],[2,3]],[[4],[5,6,7]]]`, partitions outermost, batches within
them, row groups innermost — which is verbatim the `Vec<Vec<Vec<u32>>>` the partitioner
returned. Not a partition count beside a batch count: the two are not independent, and
every property worth reading off a source line is about which batch sits in which
partition — the balance bound, an oversized row group standing alone, a partition whose
batches all came from one file region.

**Estimates go in a `--- memory ---` section per query, not on the node line**, as the legacy
`.plan.txt` already does. The section opens with the run's own inputs —
`budget=…, accumulators=…, certain=…` — and then repeats the tree with one
`estimated_max_resident_size` per node, so a number is always read against the shape it was
computed for. They churn where plan shapes do not — an estimator change, then
#19's statistics, then #147's refinement in flight — so on the node line every such change
rewrites every line and a reader cannot tell a shape change from a number. In their own
section the tree stays byte-identical and the diff says which it was. A section rather than a
sibling file because nothing consumes the memory data on its own, and cross-referencing two
files to ask whether a node's estimate suits its layout is worse than scrolling.

**Execution goldens** keep the legacy roles and not its file layout: the CPU executor authors
`.cpu.txt`/`.result.txt`, `.cost.txt` derives from `.cpu.txt` × `cost_model.conf`, and the GPU
asserts read-only — over one file per mode rather than one per query, which is T18's. New-mode `.cpu.txt` shows full
`PartitionLayout` per node, and under each node one line carrying the rows and bytes of every
batch it emitted, per lane — see T18, which is also where the record stopped being a separate
file for a chosen few queries.
`cost_model.conf` gains the new node names (every node type appearing in a `.cpu.txt`
must be in exactly one category — the conf enforces it).

**Registry and CI**: new columns in `cost-registry.csv` for the new mode's plan/CPU/GPU
enablement, with inventory tests in both directions like the existing six; new test
targets named in `pipeline.yml` steps (the `test_ci_coverage` guard enforces this
automatically).

**Widget**: two new tables (TPC-H, TPC-DS) repeating the existing structure — plan (five
cells, one per mode), CPU (five), GPU (five) — fed from the new CSV columns, and rendered
into both of `cost-report`'s outputs: the markdown PR comment and the html site. Window
queries render plan ✗ (#143). Peacock cost, DuckDB cost and ratio columns mirror the
legacy ones; when not all five modes are enabled, cost uses the last mode in the sequence
of five where CPU execution is enabled.

# Implementation plan

Tasks in dependency order, and the numbers now ascend with it. T13 is the one that does not:
it landed early, because both drivers over a mock backend needed none of T9–T12, and it keeps
its number because commits and reviews already name it. T21 sits out of order for the same
reason — it was split off T14 after it had been narrowed — and the tail of the list runs
T20, T22 because 21 is spent. T11 and T12 were retired in the same
renumbering — their work is T15 and T16 — so a number is never reused and an older reference
still resolves. Each task is one developer hand-off with its own proving tests.
Legacy tests stay green throughout — every task that touches shared code runs the
affected legacy subsets (one query per mode/tier per binary plus the rust-only tier, per
build-test.md).

~~**T0 — Python prototype of the whole execution model**~~ (done). All node types and both drivers
in Python, operators built with pandas, plans hand-built (no DataFusion, no planner) — an
emulation of tree execution whose purpose is to settle the push model before any Rust
exists. Lives in [`scripts/exec_model/`](../../scripts/exec_model/README.md); its tests run
in CI (cost-report, plus the TPC-H set in cpp-cpu, which has the generated sf1).

Done — struck through, and folded into this document where it changed a decision:

- ~~the trait set, both drivers, and the memory enforcer with the accounting formula~~;
- ~~the scheduling rule~~ — height, order, min-height-first with leftmost ties, every lane
  of the chosen node; the Drivers section is rewritten from it;
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
- ~~both limit lowerings~~, as the limit rule now states them. One finding survives here:
  the tests must assert on the *calls*, since only those distinguish a limit from a filter
  applied after the transfer;
- ~~the stress surface~~ — a plan rewriter (`operators/injection.py`) rather than
  hand-written variants: one plan re-run at every partitioning, batch size, empty-lane and
  hash-placement preset, with `GpuCoalesceBatches[target]` injected above every source
  (#139's node, proving the drivers tolerate it anywhere) and sources emitting zero-row
  batches at a set probability. It carries one rule the planner's tests should quote: a
  join may be re-partitioned only when both sides are hash-partitioned on the join keys,
  since otherwise its lane count is load-bearing and splitting it joins matching slices;
- ~~empty partitions, empty batches, skewed hashes, the flow-and-backpressure surface,
  determinism (two runs, identical batch traces)~~;
- ~~validation scope~~ — partitioning and `SingleBatch` constraints in scope, schema checks
  not.

- ~~the hand-built plan corpus~~ — 22 TPC-H and 71 TPC-DS query texts rather than the
  3–4 and ~10 the plan asked for, each at three layouts and on both join backends. It was
  the piece most likely to find something and it did: [what the corpus rollout
  measured](#what-the-corpus-rollout-measured) is the section it produced, and every item
  there is a property of the design rather than of the prototype.

Closed without the **estimator** (`estimated_max_resident_size`, `target_batch_bytes`).
The prototype models scratch per executor and never derived batch sizes from a budget,
and T6 derives both in Rust directly — a prototype estimator would be a second model to
keep true against the one that ships. The corpus is what T6 will calibrate against.

~~**T1 — flatbuffer operation-name refactor**~~ (done). Nine of the fifteen legacy node-kind names
(`GpuFilter`, `GpuProject`, `GpuSort`, `GpuAggregate`, `GpuCrossJoin`,
`GpuNestedLoopJoin`, `GpuUnion`, `GpuLimit`, `GpuCoalesceBatches`) collide with the new
mode's node names. Rename the fbs tables and `PlanNodeKind` variants to a `Cudf` prefix
(`CudfScan`, `CudfFilter`, …) so the two vocabularies are visually distinct everywhere —
schema, generated code, the C++ `node_type()` switches and serializer identifiers on the
Rust side. A pure rename: FlatBuffers wire bytes carry no table names and enum ordinals
do not move, so the proof is `plan_bytes.sha256` staying byte-identical with no
regeneration, plus green legacy subsets. The same commit sweeps the llm-wiki references
(architecture.md's fb names, affected tickets, and the recipe-plan table in this spec).

Landed on master as PR #120, with `plan_bytes.sha256` byte-identical and no golden
regenerated, which is the proof the rename asked for.

~~**T2 — ParquetBatchPartitioner.**~~ The pure policy class and its unit tests: fewer
survivors than N; N=3; single row group over target; batching off ⇒ one batch per chunk;
empty survivors (explicit error — the fbs "empty map means legacy single partition"
convention must not leak in); the balance bound on uniform row groups (max−min partition
rows ≤ one row group);
fixed-output determinism case. No planner integration yet.

~~**T3 — node and trait skeleton.**~~ `GpuNode`, `PartitionLayout` (with the two-valued
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

~~**T4 — translation layer, single-partition shapes.**~~ DataFusion physical plan (tp1) →
`GpuNode` tree for chains: load, filter, project, sort (+fetch), limit (root-adjacent ⇒
no node, `skip`/`fetch` set on `GpuUnload`; otherwise a `GpuLimit` node over a
planner-inserted `GpuMergePartitions` — never a coalesce), coalesce-all,
single/final aggregates, cross/nested-loop joins. Per-node-kind conscious mapping;
unrecognized node ⇒ plan-time error naming it; window ⇒ the #143 refusal. Unit tests
assert emitted constructs for simple queries.

~~**T5 — translation layer, partitioned shapes.**~~ tp4: shuffle points → Merge+Emit, the
aggregate sequence with its shortcuts and the gid rule, join side normalization (type
remap + column-order-restoring project) and build-side coalesce insertion per the
capability matrix, union/interleave with explicit branch-cast projects. The
`hashKeys ⊆ group columns` structure is produced here (validated in T8). Unit tests per
construct in tp1 and tp4, including side-swap cases.

~~**T6 — estimator pass and plan goldens.**~~ `estimated_max_resident_size` per node
(rows × width vocabulary, N-lane charging), `target_batch_bytes` derivation feeding T2's
partitioner, integration as `plan_batch_partitioned()`. Canonize all four
`<mode>.plans.txt`, memory sections included, for TPC-H and TPC-DS (minus #23's four and
window queries, which appear as refusals).

~~**T7 — schema registry.**~~ (done). The `Schema` carried in `NodeKind` populated on all
nodes, with column semantics annotations. Unit tests: hand-crafted plans produce expected
types and annotations; decimal precision/scale fidelity through project/aggregate/union-cast
paths.

The type and the annotations landed with T3-T6; the tests are PR #126. They assert on the tree
rather than on rendered text, since both engines derive their per-node bytes from the same
declared schema and a wrong type moves no golden byte — `agg_state` at the init, the per-lane
merge and the finalizing merge, and `avg`'s state columns typed by what they hold rather than
by position, which is the case the task existed for.

~~**T8 — validation.**~~ (done). `validate_schemas_and_partitions()` on every node type: partition
topology, key-distribution subset rule, sortedness requirements (merge requires
`BatchSorted`; a limit after a sort requires its input to be `is_stream_sorted()`, checked
on whichever node carries the interval — the `GpuLimit` mid-plan, the `GpuUnload`
root-adjacent), `SingleBatch`
expectations (join build, cross/nlj inputs), captured-index checks. Unit tests: manually
constructed wrong combinations error, right ones pass; then run validation over every
canonized corpus plan from T6.

Node-local validation landed with T4/T5 and is called from `plan_batch_partitioned`; the
generic pass is `batch_partitioned/validate.rs`, PR #126. It runs over every canonized plan
because the planner calls it, so a rejection renders as a `refused:` section and fails both the
golden compare and the registry cross-check. Three planner defects were found this way and
fixed there rather than ticketed: `NestedLoopJoinExec`'s dropped projection, `GpuJoin` minting
a key distribution instead of carrying one, and a non-exhaustive match that dropped the claim a
mark join earns.

~~**T13 — drivers and enforcer.**~~ (done). Both drivers over a mock `Backend` impl — the
third instantiation, alongside CPU and GPU — with the schedule and the accountant as units of
their own, and the accounting formula with its pre/post checks. What the task settled is in
[Drivers](#drivers) and [Memory accounting](#memory-accounting); what it left for T14 is every
real executor, since nothing here computes a row.

~~**T9 — additive ABI.**~~ (done). The three approved symbols in `gpu_executor.cpp` + `peacock_gpu.h`,
signatures as [GPU execution](#gpu-execution-through-the-frozen-ffi) gives them; any
*further* surface change goes through a proposal to the human, per the constraint section.
Rust bindings for all three; `GpuBatch` handle plumbing (session ref, `Drop` release,
`ManuallyDrop` consume boundary). Tests: a C++ gtest in the plan-executor suite reading
disjoint row-group subsets and asserting union == whole-scan; a gtest exporting ranges of
one handle and asserting the concatenation equals the whole, plus the empty range and the
past-the-end range; the same for slicing, plus that the input handle is released and
double-slicing it fails; Rust FFI smoke on shad-gpu. The range plumbing reaches
`UnloadExecutor::unload(batch, rows)`, so the trait's second argument lands here rather
than in T10.

Landed with two shapes worth knowing. The row range is one function, `clamp_row_range`, that
the export and the slice share, so the two cannot disagree about an overrun; and the row-group
override reaches `execute_scan` as a `cudf::host_span`, so the node's own vector and a caller's
array take one path. `RowRange` and `unload(batch, rows)` were already in from T13, so the
trait needed nothing. One test moved tier against the list above: two IPC streams do not
concatenate, so "the ranges are the whole" is asserted in `test_gpu_abi`, where arrow-rs decodes
them, and the gtest holds the contract edges instead.

~~**T14 — recipe-plan serialization.**~~ The `GpuNode` → fb-seq mapping implemented, canonized and
unit-tested. `attach_recipes()` runs after the plan is complete and hangs a recipe on each node
that drives the GPU ABI; a node that makes no ABI call gets none, which is a fact about the node
and so is worth reading off the plan. One function per node kind produces that node's recipe from
**that node alone** — no child, no parent, no tree walk. The mapping table is a per-node statement
and a function that can reach a child would let it stop being one, so the restriction is the
design rather than an economy.

A recipe is a sequence of ABI calls, each carrying the built FlatBuffers node it addresses — the
payload, not a reference to where the fields live. Two renderings, one function taking an enum:
without payloads it is a section in every `<mode>.plans.txt`, between the plan tree and
`--- memory ---`, keeping the tree shape and repeating nothing the tree already shows except the
lane count; with payloads it is a golden of its own, holding the recipes alone — no plan tree, no
memory — for a subset of queries chosen to reach every fb kind and every call shape longer than
one call. A digest of the serialized bytes rides beside the payload text, since text and bytes can
disagree and `plan_bytes.sha256` is the precedent for pinning the wire form rather than a
description of it. Unit tests cover the kinds whose recipe is more than one call, `GpuJoin` first:
the seq set and call pattern per join type, against the
[capability matrix](#join-capability-matrix).

Dense seqs are impossible, and that is a property of the fbs rather than a choice. Children are
nested (`input`, `left`/`right`, `inputs`) and `CudfScan` is the only leaf table, so a set of
addressed nodes whose arities exceed its own edge count has to be padded with stub scans — and
every stub takes a post-order slot, so it moves the seqs above it. Three rules follow: stubs
rather than a re-hung child, since a shared offset is a DAG and gets indexed twice; a call whose
input is a runtime handle hangs off the previous fb node of its own recipe, or a stub where there
is none; and a forwarder's unconsumed branch is gathered by a structural `CudfUnion`, because an
orphan is never indexed and its shift has no visible cause. The pass and the serializer are
therefore one walk: a seq is the post-order index of what was built, so it cannot be counted
before the building.

The `Expr` -> `fb::Expr` writer is its own file and is where the unit tests concentrate: every
expression variant and operator, nesting, and each scalar kind the corpus produces — decimals
with their precision and scale first, since a wrong write there is invisible in plan text and
wrong on a device. `plan_serializer.rs` serializes a DataFusion plan and keeps that one job;
`serialize_scalar_value` and `serialize_schema` are reused, `serialize_expr` cannot be, since it
downcasts `PhysicalExpr` and this IR is our own.

Post-order is the agreement to assert rather than assume: `begin_plan` indexes post-order, so the
emitted tree has to number exactly as the recipes say, and a plan simple enough to check by its
answer would answer correctly while addressing the wrong node.

Nothing executes here and nothing new runs on a GPU. `scripts/exec_model/operators/recipe.py` and
`recipe_join.py` are the starting point for the join sequences — a model, not a spec, and the fbs
and `cpp/src/operators/join.cpp` are what settle a disagreement.

The proving set is the new unit tests and the golden target, and no legacy subsets — the human
scoped it that way because the change is additive: a recipe is attached to a plan nothing reads
yet, so the only tests whose result can move are the ones that read it.

~~**T21 — a recipe plan on a live GPU, driven by hand.**~~ It needs T14 and nothing else: no driver,
no executors, no scheduling. A new test file on the shad-gpu tier plans a query over TPC-H sf1,
loads the recipe plan T14 already built, and makes exactly the calls the recipes name, threading
each call's output handle into the next one's input and exporting at the root. One helper does the
whole walk; one test per query calls it, so a failure names the query rather than a stage.

`begin_plan`'s `out_node_count` is asserted equal to the number of fb nodes the writer created —
not to the `GpuNode` count, which is a different number: stubs, structural unions and any node
with more than one call all separate the two, in both directions and in most plans. It is the
first thing this task can settle that nothing before it can: every seq indexes a post-order the
C++ builds in `index_post_order`, and until a device has parsed a buffer we wrote, our agreement
with that walk rests on two child-order functions having been read side by side. One assertion, at
the first call, in the first place both numbers exist at once.

Shapes, chosen so each call is unambiguous. Everything but the aggregates plans one partition and
one batch, which makes every recipe a single call per node and the walk a straight line: a bare
scan, a filter, a project over a filter, and the joins — inner, and one build-preserving type,
whose single probe batch takes the legacy one-call form. The aggregates plan one batch and **two**
partitions, because a merge is the operator this mode adds and one partition never performs one:
two lanes each merge their own state, the cross-lane merge folds them, and the finalize project
runs once. That is the first time `AggregateMode::Merge` and the finalize expression meet a
device.

`avg` is the case worth a test of its own. Its finalize divides a decimal by a count, and cuDF
derives a divide's result scale from its operands where arrow takes it from the declared output
type — so a wrong cast is invisible on a CPU host and wrong on a GPU, in a column whose type reads
correctly either way. Assert the digits, not the type.

The oracle is DataFusion on the same SQL — `data_fusion_exact`, the CPU tier's own vocabulary —
and deliberately neither a result golden nor our CPU executor. A golden records what the first run
produced, so a finalize whose scale is wrong from the start is pinned rather than caught; and our
CPU executor evaluates the same finalize expression the device is sent, so it agrees with a wrong
one. DataFusion computes `avg` without a Welford triple, a merge mode or cuDF's divide-scale rule,
which is what makes agreement with it evidence. Joins compare as sorted multisets, since a GPU
join's output order is not deterministic. What it deliberately leaves out is everything the
driver decides — batching, backpressure, arrival order — since every shape here is one batch;
those arrive with the executors, and the driven end-to-end over every layout is T17's.

What has a device behind it. Ten fb kinds have run on one; every other is refused by name in a
`match` over every `FbKind`, so a variant added later stops the file compiling rather than going
quietly unclassified. Still unproven on hardware: [#136](../tickets.md#t136)'s finish pass — probe
keys per batch, the concat at done, the finish join, the pad project — the whole Right family,
cross and nested-loop joins, both sort nodes, `slice_handle` and a ranged export.

~~**T10, T15 and T16 — the executors, as one task.**~~ All three land together on one branch, because
they are one question asked of three node families: what does an executor do when the recipe
already says which calls to make. Ordered inside the task as T10 then T15 then T16, since the
accumulators and the joins are the Exec executors' shapes with state added.

~~**T10 — Exec executors.**~~ Filter, project, per-batch sort, aggregate (partial/single), unload
(`GpuBatch → CpuBatch`, honouring the row range). The **GPU executor runs the recipe attached to
its node** — the calls, in order, with the handles threaded — and reuses no legacy operator code:
the recipe is the instruction set, and reaching into legacy operator internals would be a second
path to the same kernels. The **CPU executor relays to DataFusion**, where reuse with legacy is
expected rather than avoided, since both are asking DataFusion for the same operator.

~~**T15 — accumulators.**~~ `GpuCoalesceAllBatches`, `GpuAggregateBatches` (merge-only and finalizing),
`GpuAccumulateBatchesAndSort`, `GpuMergeSortedPartitions`, and the mid-plan `GpuLimit`. Edge cases:
zero batches, one batch, ties for the merge (partition-major stability), fetch interaction, large
batch counts, gid-carrying aggregate merges.

~~**T16 — partition ops and joins.**~~ `GpuEmitPartitions` (per-batch scatter at a small N and a
large one, empty outputs for skewed hashes, and the lane each key lands in, asserted on both
backends — co-partitioning is what every partitioned join rests on). `GpuMergePartitions` is not
here: its mapping is `Forwarder`'s, from T13, and its service order is the driver's. `GpuJoin` with
`set_build`/`probe_and_fetch`/`finish_and_fetch`, plus cross and nested-loop joins on the same
trait. The [capability matrix](#join-capability-matrix) is emulated as a test table — per
(type × layout): stream-vs-refuse, correctness against a hand-built oracle, the GPU finish pass via
key accumulation ([#136](../tickets.md#t136)), `null_equals_null` on the finish join. That finish
pass is the one shape this mode invented with no device behind it after T21, which is why the
matrix is emulated here rather than assumed.

**What a copy costs decides the matrix, and the copy does not exist yet.** Every handle is erased
by its reader, and the frozen surface has no copy symbol, so a shape whose recipe names
`BuildSideCopy` meets a second probe batch with a dead handle. The question T16 had to settle was
whether to keep claiming those shapes stream and refuse until [#145](../tickets.md#t145), or make
a single-batch probe the matrix's permanent rule. Over the 37 hash joins in the
partitioned-tp8-standard goldens a copy would cost 0.08 of the probe stream at the median and more
than it for 12, so a permanent single-batch rule would price every join at the worst one: the
claims stand and the device refuses, naming [#152](../tickets.md#t152), with a test on the
refusal.

Left and Full outer go further and have **no device path at all** until then, which is #152's
second row rather than its first: their key project and their per-call join read the same probe
batch, so no ordering of the two leaves both an input. The finish pass's pad is therefore proved
on the CPU alone, and the device test asserts the refusal.

**How everything here is tested.** Small synthetic data, never the corpus; plans hand-constructed
rather than planned, so a test names the shape it means instead of hoping a query produces it;
`attach_recipes()` is fair game, since the recipe is what a GPU executor consumes. The oracle is
hand-constructed too: an expected result written down, not derived by the code under test. CPU and
GPU tests in separate targets so CI hosts split them. A device test writes its own parquet, which
is the ABI's doing rather than an exception: the four entry points load a table only by reading
one, so a device test's input is a scan or nothing. What the rule excludes is tpch.minimal and the
generated sf1, whose values nobody chose.

**What this task does not do.** No driver: nothing here is hooked into the schedule, and every
assertion is about one executor answering one call. That defers the whole class of claims that
read as call counts and pull counts — a limit holding nothing whatever the offset, at most two
batches sliced per query, the scan stopping — to T17, which is where a driver exists to make them.
`PlaceholderRowExec` ([#158](../tickets.md#t158)) waits for the same reason: it is a source, and a
source proves itself by what the driver pulls from it. T17 then found it cannot be discharged at
all while the surface is frozen — see the sixth entry under
[What the frozen surface costs](#what-the-frozen-surface-costs-and-what-unfreezing-would-buy).

~~**T17 — the whole path, under injection.**~~ The first task in which SQL goes in and rows come out:
planning, the recipes, the executors and both drivers running together, rather than each proved
against a fixture of the last one's shape. Every test starts from a query's text and ends at its
results, so what is under test is the join between the pieces — which is the only part four tasks
of separate proofs cannot reach.

The oracle is DataFusion on the same SQL. Not the legacy CPU executor, as this entry said before
T21: a second engine of our own agrees with us wherever we are consistently wrong, and by the time
this task runs, the finalize expression it evaluates is the one we also send to the device. The
one independent implementation in reach is the one that decomposed the aggregate differently.

Queries chosen to be interesting rather than representative, over the sf1 corpus text, and
between them covering the [join capability matrix](#join-capability-matrix): every join type this
mode claims, crossed with the layouts that make each one stream or refuse. The matrix is emulated
on synthetic data in the executors task, where each type is one executor answering one call; here
it is planned from SQL and run through the drivers, which is the first time a type's claim is
tested as the thing a user gets rather than as the thing an operator returns.

Four shapes are not join cells, so no join cover reaches them, and each is named by the query that
carries it:

| Shape | Query | Why that one |
|---|---|---|
| union lowered to an interleave | tpcds q33, q56, q60 or q66 | the claim is output lane p from lane p of each branch, so it needs four lanes; q14 also interleaves and is the trap, since its is `lanes=1` |
| union that cannot interleave | tpcds q77 | its branches disagree on lane count — 4+1+4, and the golden says `lanes=9` — which is the case [Node set](#node-set) argues in prose and nothing executes |
| both row-interval lowerings | tpch nested-limits | the root-adjacent interval becomes `GpuUnload`'s skip/fetch and the mid-plan one a `GpuLimit` over the scan; the only `OFFSET`s in either corpus, and it has no `.cpu.txt`, so this is its first execution |
| a merge with state worth merging | tpch shuffle-stddev | `GpuAggregateBatches` rides in most of the join queries as a sum; this is the Welford init, both merges and the finalize project |

Nested-loop Left is the one matrix cell no corpus query reaches, and its shape — a single-batch
probe, since #136's finish trick accumulates keys and a predicate join has none — is reachable
from no other row, so this task writes the query. The other uncovered cells stand: an Inner with
`null_equals_null` (the flag rides an INTERSECT lowering, and every corpus INTERSECT lands as a
semi form) and the three plan-time refusals, which a corpus query cannot provoke by construction.

Each query is re-run under injection, several modes rather than one, with the same answer demanded
every time. The prototype's [`LayoutInjector`](../../scripts/exec_model/operators/injection.py) is
where to look for modes worth having — layouts re-planned rather than edited, a rebatcher above
every source, sources emitting zero-row batches at a set probability — and it is a model rather
than a specification, so a mode it lacks and this path needs is a mode to add. Rebuild rather than
edit, for the reason the prototype records: a node's partitioning is not a field, so a rewrite
re-runs the planner at a chosen `(target_partitions, batching, small_table_bytes)` and the shapes
come out consistent.

Two rules the injector carries and this one must too: a join may be re-partitioned only when both
sides are hash-partitioned on the join keys, since otherwise its lane count is load-bearing and
splitting it joins matching slices; and a degenerate hash — every key into one lane — is a legal
hash, because a shuffle's contract is co-location and nothing above it may depend on how evenly
the lanes were loaded.

It also inherits what the executors task could not assert without a driver: a limit holding
nothing whatever the offset, at most two batches sliced per query, the scan stopping — each a call
or pull count — and `PlaceholderRowExec` ([#158](../tickets.md#t158)), which is a source and so
proves itself by what a driver pulls from it.

`PlanIndex` gets the unit tests it has never had, and they belong here because this is the first
task whose failures would be read through it. Nothing tests it directly today: `PlanIndex::build`
has one caller, and the scheduler tests derive their own subtree ranges from a parents array rather
than taking the index's. Assert what the derivation decides rather than what a plan happens to
produce — pre-order numbering and the contiguous subtree range that every hold rests on, `parent`
and the snapshotted children, and the three counts a category changes: `ready_lanes` against
`lanes` for a cross-lane accumulator and an emitter, `input_lanes` for the `Done` events a
partition accumulator owes, and `slot_base` where it is lane-scoped against where it is not. Each
of those is wrong far from where it shows.

**Two numberings meet in the driver, and one walk should compute both.** `PlanIndex` is
pre-order — a subtree is a contiguous range, which is what every hold rests on — and a recipe is
keyed by post-order, because that is how the FFI addresses a node. So the index records each
node's post-order position beside its pre-order one, from the walk it already makes, and
`executors_for` takes it. What must not happen is a third derivation: `attach_recipes` numbers at
plan time and the index numbers at run time, so a test asserts the two agree over the corpus —
[#134](../tickets.md#t134) is the same pair one boundary over, and it is unchecked there.

A source executor is this task's to write, and so is the answer to a lane with no build batch: a
`set_build` that never happens because `GpuCoalesceAllBatches` emitted nothing is a driver
decision, not an executor one, and T16 left it here deliberately (the finish's own zero-key answer
is already settled).

The row range the driver hands an unload is asserted before the call. `clamp_row_range` absorbs
an offset past the end and a length past it, because a C ABI has to be total — but
`RowInterval::range_of` cannot produce either, so the tolerance can only be reached by a driver
whose `rows_seen` has drifted, and what that looks like is a `LIMIT` quietly returning short.
Assert non-empty and within the batch where the driver builds the range, so the arithmetic names
itself rather than being absorbed.

The mock backend gets a handful of its own for the same reason one level up. Every assertion in
`driver/tests/` is measured against it, so a mock that miscounts is 1255 lines of tests agreeing
with the wrong answer and staying green. Pin what a script says against what the mock does — the
scripted batch counts and sizes per source and lane, the skew pattern an emitter fills its lanes
by, and an accumulator emitting where the script says it emits. A few cases, not a suite: what is
being checked is that the instrument reads what it was set to.

It starts by wiring the executors to their traits. `Backend` names seven associated types, so no
earlier task can implement it — T16 finishes the last of them and none of them owns a source — and
`Executor`'s `resident_bytes`/`scratch_bytes` are the memory accounting this task adds. So the
executors arrive as inherent methods in the trait's shapes, and the first commit here is the one
that makes the compiler check that.

**Defects found here are fixed here.** Every task before this one proved its own layer against a
fixture; the first thing to run all of them at once will find things about their joins, and
parking those behind tickets would leave the path unproven in exactly the way this task exists to
end. T21 is the precedent: it was meant to be one test file and it found four defects, each a
rule held by a doc comment with nothing reading it, and one guard that could not go red.

~~**T17a — layout injection over corpus queries.**~~ (done). T17 runs seventeen queries at five modes and
calls it injection; it is not. Each of the five is a plan the planner would have chosen anyway —
`(target_partitions, sizing)` re-planned, with `small_table_bytes` and the budget constant. The
prototype's [`LayoutInjector`](../../scripts/exec_model/operators/injection.py) does a different
thing: it takes one plan and rewrites it into layouts no planner would emit — lanes deliberately
drained, a degenerate hash, a rebatcher above every source cutting against the grain, sources
emitting zero-row batches at a probability. Four dimensions, none of them reachable from SQL, and
none of them exercised by any real query today. They live in `driver/tests/stress.rs` over a mock
whose answer is a script.

**Why T17 could not do it, and what this task has to build.** `GpuNode` exposes `children()` and
nothing that reconstructs a node — no `with_children`, no `rebuild` — which is why T17 re-planned
instead. But the rewrite is writable without touching production: `as_node_ref` is a public
exhaustive `NodeRef` over all eighteen kinds, each node's steering fields are `pub`, and each kind
has a public constructor. So a test-side `rebuild(node, new_children)` is eighteen arms that read
the fields and call the constructor, and because the match is exhaustive a nineteenth kind fails to
compile rather than being silently un-rewritten — the same guard `node_kind()` and `driven` already
rest on.

**Prove the rewrite before using it, and a unit test is the right size.** A rebuild that drops a
field is a plan that differs from the one under test for a reason nobody chose, and every result
after it is then about a different query. The case is small: hand-built plans rebuilt with their own
children, identical in **debug output** — not in the rendered plan. The renderer is what a golden
reads and it does not print everything: a loader's survivors and `can_be_null`, and an aggregate's
intermediate schema, reach no plan line, so a rebuild that drops `can_be_null` renders identically
and passes. Those are exactly the fields a corpus plan never varies, which is what would have made
the rendering-only form a guard that cannot go red. Debug prints every field including the private
ones and is the identity. Nothing in this case reads a committed file: the comparison is a plan
against its own rebuild, so it holds without goldens and moves when neither of them does.

**Make the field cover exhaustive rather than claimed.** Fixtures that populate every optional
field are a hand-maintained list, and the failure it misses is the one that will happen: a field
added to a node next quarter, no fixture varying it, the arm free to drop it, the test green.
Three levels, and the third is the one worth building:

- *arms* — the match over `NodeRef` is exhaustive, so a nineteenth kind fails to compile;
- *kinds* — the fixture set is asserted to reach every variant, so a kind with no fixture is red;
- *fields* — derive them from the debug output rather than listing them. Every field name a node's
  debug prints must take **at least two distinct values across the fixture set**, because a field
  constant everywhere is a field whose loss no fixture can detect. It goes red naming the field,
  and it goes red the day the field is added rather than the day something depends on it.

That last one is what makes the case exhaustive without reflection and without a maintained list.
It is also a guard that can go red on its own, which on this chain has been the exception. Not through
`driver/plans.rs` — it is `#[cfg(test)] mod plans`, so an integration test cannot see it, and its
builders leave every optional field `None`, which is the half the property needs. The fixtures
live beside `rebuild` and give each node a value in every optional field: `Filter` with a
projection as well as without, `Limit` with a fetch, `Join` with a residual. A corpus plan covers
only the combinations its queries happen to produce, and a fixture set of all-`None` nodes would
let a dropped field through in silence.

Cover the arms the way `every_node_kind_builds_the_executor_its_category_names` covers its own:
assert the fixtures reach every `NodeRef` variant, so a kind added later is a compile error in the
match and a red count here. It goes red by dropping one field from one arm. Nothing is injected
until it passes, and no corpus query is needed to prove it.

**Two mechanisms, two dimensions each, because the dimensions are not the same kind of thing.** A rebatcher is a node —
it changes the tree, and it must be placeable anywhere a batch flows, not only above a source. The
other three are behaviour a node does not carry: this engine's hash is Spark-murmur3 fixed in the
emitter, and neither an empty lane nor a zero-row batch is a field of `GpuLoadParquet`.

| Dimension | Mechanism | Why not the other one |
|---|---|---|
| a rebatcher, at any edge that still validates | tree rewrite: insert a node above the chosen child. One direction only — nothing below the loader splits a batch ([#142](../tickets.md#t142)), so the node is `GpuCoalesceAllBatches` merging a lane to one, and the finer direction is the mode axis already. A coalesce clears the sort order by construction, so an edge under a `GpuMergeSortedPartitions` or a limit-after-sort is refused at validation: assert the refusals are exactly that class rather than skipping those edges quietly, since a skipped edge is a hole nothing reports | wrapping an executor cannot do it — an exec is one batch in, one out, and a merging rebatch must hold batches across calls, which is an accumulator's job. A wrapper that held them would lie to the accountant and break the queue bound the driver guarantees |
| a drained lane | tree rewrite: move that lane's row groups into another lane | **not** a wrap — a source producing nothing for a lane loses its rows, so every oracle comparison fails and the dimension is untestable. `partition_groups` is the mapping, so moving groups keeps the lane count and every row. Safe at every plan this mode produces: `co_partitioned()` requires one lane or a ByHash distribution on the join keys, so no join here rests on scan lane alignment — the shape the prototype needed `_is_shuffled_join` for and this mode does not have |
| zero-row batches at a probability | wrap the source executor: emit an empty batch instead of advancing | not a node field; the driver already carries empty batches, so this exercises a path the operators have and this one does not |
| a degenerate hash | wrap the emit executor: route every key to one lane. **Not compatible with a plan carrying Right, Full or RightAnti**: one lane holding every key leaves the others with an empty build side, and those three owe their probe side against one, which is [#175](../tickets.md#t175)'s refusal. Drop the dimension for those plans and pin the refusal with a case, so the drop is a rule a reader meets rather than an abort they rediscover | the emitter carries key *expressions*, not a hash function — ours is fixed |

Lane counts are deliberately absent: `target_partitions` already varies them through the planner,
and a lane count no planner would choose is the one dimension whose correctness rule the prototype
had to encode (`_is_shuffled_join`, `_LaneOrigin`). Buying that rule to vary something the mode
axis already varies is not worth it here.

**Three phases, and the middle one is the point.** Plans first, selection second, execution third,
because 10 queries × 5 modes × the injection crossing is more runs than a CI tier can hold, and
choosing *which* to run is a claim that has to be visible rather than a `take(30)`.

1. **Plan.** For each enabled query, plan at all five modes. Five plans, no injection yet, and a
   plan that fails to build is a failure rather than a skip.
2. **Select.** Derive the candidate set — each plan crossed with the injection settings — and
   choose at most **30 per query**. Representative means the selection covers each dimension at its
   boundaries and each mode at least once, not a sample: the rebatcher in both directions, the
   empty-batch probability at zero and at its high setting, a drained lane where the mode has more
   than one, the degenerate hash where the plan shuffles. The chosen set is **asserted, not
   trusted** — a test over the selector alone, with no queries, that a known candidate set yields
   a cover, and that dropping a dimension from the settings makes it go red. Deterministic and
   seeded: two runs choose the same 30, or a failure is not reproducible.
3. **Run.** Each selected plan through the driver, every answer against the same oracle.

**Render the oracle once per query.** Comparing renders both sides, so an oracle rendered per
variant makes a query's cost grow with the number of variants for no reason — at thirty variants
that is most of the tier. Render once, hold the text, compare every variant against it.

**The oracle does not change and must not.** It is DataFusion on the same SQL, planned and
collected once per query at `target_partitions=1`, compared against every variant. That is what
makes injection meaningful: the answer is fixed by something that never saw the layout, so a
layout that changes the answer is a defect rather than a disagreement between two of our own
shapes. One oracle per query, not per variant — it is the expensive half and it is invariant.

**Eleven queries, and the cost axis is not the obvious one.** Injection multiplies runs, so it goes
on the cheap end of the list — but *cheap* here is the size of the result, not of the scan. A run's
cost is dominated by the oracle comparison, so `anti-join`, a `SELECT *` over 1.2M rows, is the
most expensive member of a set picked by rows scanned. Scanned rows at sf1, which is the proxy the
set was first chosen by and is kept here so the correction is legible:

| Query | rows scanned | what it carries |
|---|---:|---|
| tpch `nested-loop-join` | 40,000 | nested-loop Inner |
| tpch `nested-loop-left-join` | 40,000 | nested-loop Left, single-batch probe |
| tpch `nested-limits` | 220,000 | both row-interval lowerings, cross join |
| tpcds q45 | 899,384 | LeftMark |
| tpch `anti-join` | 1,600,000 | RightAnti |
| tpcds q8 | 3,060,404 | Inner multi-key, LeftSemi with `null_equals_null` |
| tpcds q16 | 3,087,163 | LeftAnti, LeftSemi with a filter, a mid-plan limit |
| tpcds q93 | 3,187,918 | Right outer, multi-key |
| tpcds q97 | 4,361,952 | Full outer |
| tpcds q2 | 4,401,864 | the union that cannot interleave |
| tpcds q33 | 5,281,336 | the four-lane interleave — eleventh on cost, first on merit |

q33 is in on merit, and it happens to cost little: eleventh of the seventeen, six heavier ones
left out. It is the only four-lane interleave in the
list, and an interleave is the one operator whose correctness *is* a lane correspondence — output
lane p from lane p of each branch. Excluding the one shape a perturbed lane could break, to save
0.9M rows over the tenth, would be picking the cheap set over the point of the exercise.

The six left out are the heavy end — q38, q87, `shuffle-stddev`, q20, `left-join`, and q21 at
19.5M rows, three `lineitem` scans in one query. If the budget turns out to allow more, `left-join`
is the next one worth having: a Left outer's finish pass accumulates probe keys, so its residency
is a function of how the batches arrive.

**Shape of the change.** One new test library file carrying the wrappers, the settings, the
candidate derivation, `rebuild` and the selector — `peacockdb-core/tests/common/injection.rs`,
split into a second file only if it passes the 1000-line bar. The **tests** go in
`test_cpu_batch_partitioned.rs`, not beside the mechanism: a `#[test]` under `tests/common/`
compiles into every one of the 22 binaries that declare `mod common`, so it would run 22 times and
be counted 22 times. A target of its own would avoid that and costs a `pipeline.yml` step and a
`test_ci_coverage` entry, which is not worth it for two cases. The eleven
fixtures are declared as one list: `injected_queries!` expands it into both the `INJECTED` const
and the fixtures, so a query leaves the set only by leaving the list, and everything else is
unchanged. One thing in `peacockdb-core/src` changes, and only its visibility: `validate()` becomes `pub`.
The driver validates nothing — `check_canonical_form` is limit positions and no more — so an
injected tree would have run unchecked, and a rewrite that broke a node's requirements would have
answered rather than been refused. That is the failure this task must not have: an injector
quietly generating plans the planner would never emit. It is also what makes the rebatcher's
refusal a demonstration rather than a prediction, since `validate()` is what names the node and
the order it broke.

The cases split by dataset rather than by subject. Four need none — the identity case, the
selector's cover, the rebatcher refusal and the degenerate-hash emit — and go in
`tests/test_batch_partitioned_injection.rs` with its own `pipeline.yml` step and
`test_ci_coverage` entry, so they run in seconds on any host. The two that need a query — the dimension demonstration, which plans
`nested-loop-join` over sf1, and the injected corpus set — stay in the end-to-end file. A mechanism proof buried in a tier that takes minutes is one nobody runs
while iterating.

**Measure before capping, one run at a time.** T17's seventeen queries at five modes are 85 runs in
4m39s at four threads. Eleven queries at up to 30 is 330, on the cheap end of the corpus — a number
to measure, not to assume.

The measuring pass runs **serially**, `--test-threads=1`, and times each run on its own. Four
threads contending for the same host is what the correctness tier wants and the opposite of what a
timing wants: a number taken under contention cannot be compared with another taken under different
contention, so a table built from a parallel run would rank the wrong things. Time each
(query, mode, injection setting) individually and report every one — not a total, since a total
cannot say which row to cut.

The table is the deliverable of that pass and belongs in `llm-wiki/reports/`, since it is a
measurement of a host rather than a fact about the code: one row per query, the per-setting times
across it, its total, and the grand total, with the host and thread count at the top the way the
benchmark records carry `build_profile`. Then the cap is chosen against it.

**What may be trimmed, and what may not.** If it does not fit the tier, cut runs and not cover.
The selection rule already guarantees each dimension at its boundaries and each mode at least once,
so a smaller cap is still a cover — that is what the rule is for. What must survive any trim is one
carrier per injection dimension and the shapes that only one query has: q33's interleave, the two
nested-loop forms, `nested-limits`' two row-interval lowerings. Trimming to the cheapest eleven
minus q33 would be the obvious cut and the wrong one, for the reason q33 is in the list at all.
Where a query is dropped entirely, say which dimension lost a carrier and why the remaining ones
cover it.

**T18 — infra for running corpus queries.** T17 proved the path on seventeen queries and T17a
injected layouts into eleven; the corpus is 39 tpch and 99 tpcds, each at five modes on two
engines. What stands between the two is not more test cases but a way to declare one. This task
builds that. It was split out of T20, which carried both this and the new query shapes, once it
was clear the rollout needed the infra before it needed either.

**One declaration per query.** `corpus_query!(dataset, sf, query, cpu_modes, gpu_modes,
cpu_oracle, gpu_oracle)`, where the two mode arguments are a bitwise or over the five planning
modes the plan goldens already carry. A query's whole coverage is then one line that can be read
and diffed, rather than up to ten macro invocations that can disagree with each other. It expands
to one test case per (query, enabled mode) on each engine, so `cargo test <query>` names exactly
the runs that query has — which is what makes the filtered regeneration below possible at all.

The CPU and GPU cases go in **different binaries**, as the legacy pair does: the GPU targets are
staged to shad-gpu by the gpu-tests job, and `test_ci_coverage` verifies that job's array names
them. One macro writing into two binaries is the reason the mode arguments are separate — a query
can be enabled on the CPU at four modes and on a device at one.

**The oracle keywords already exist and are taken unchanged.** `cpu_oracle` is
`data_fusion_exact | data_fusion_approximate | data_fusion_subset`: one oracle, plain DataFusion
at `target_partitions = 1`, asked three ways — the whole answer, the whole answer at a 1e-12
relative tolerance where the sole divergence is float summation reassociation, and the count and
containment alone where the SQL does not determine which rows (see the unordered `LIMIT` below).
The third is new; the first two are legacy's, unchanged. `gpu_oracle` is
`golden_exact | golden_approx | golden_approx_std | live_cpu | skip`: the frozen result compared
exactly, at 1e-12, at 1e-11 where cuDF's variance diverges further than the convention allows,
against a live CPU run where the result is too large to commit, and not at all for a query whose
row order is undetermined. A second vocabulary for the same choice is how two families drift, so
these are the legacy sets or they are a rename of them, never a parallel set — and `live_cpu` is
that rename, applied to legacy's ten call sites in the same commit. Legacy spells it `oracle`,
which inside an argument named `gpu_oracle` says only that an oracle is an oracle, while the
choice it actually makes is between a frozen result and a live one.

There is no seventh argument for whether the run writes `.result.txt`. Legacy carries one
(`result_golden` / `no_result_golden`) because its producer and consumer are declared in
different files, and an orphan golden — written by a CPU case no GPU case reads — is silent
where a missing one is loud. Here both sides are in one declaration, so the predicate is
`gpu_oracle` naming a golden at any enabled mode, and the pairing cannot be stated wrongly.

**Goldens stop being one file per query.** Today each query carries its own `.cpu.txt`,
`.cost.txt` and `.result.txt` per mode, which is how `testdata/goldens/` reached 277 files for
tpch.sf1 and 625 for tpcds.sf1. This mode takes the shape its plan goldens already took: one
`.cpu.txt` and one `.cost.txt` per mode, each holding every query in `== <query>` sections, and
one `.result.txt` across all modes. The comparator is the plan goldens' own `section_differences`
— it names what moved, what is missing and what is out of order, and it has unit tests — so this
is a second caller rather than a second differ.

A query whose bit is clear for a mode still has a section in that mode's files, carrying a marker
that says it was skipped. An absent section and a skipped one are different facts and the file has
to hold both: a query that stopped planning is a regression, a query never enabled at that mode is
a decision, and a format that renders them alike loses the only artifact that could tell them
apart.

**`.result.txt` is one entry per query**, keyed by the query alone. The modes are supposed to
agree on results, so one of them authors it: the last mode the query declares in the fixed
sequence of five — `bp-tp4-sized` for most. Its authority comes from the declaration and not from
what happened to run, which is what keeps it well defined under a filtered regeneration. A run
that does not include the authoritative mode leaves the section untouched; it is the one golden
whose key carries no mode, so it is the one a partial regen could otherwise re-author from a mode
that is not the authority, with the body's own line moving to say so and nobody reading it. The
section still records which mode produced it: the key carries no mode because there is one entry,
and the body names one because where the modes disagree, that disagreement is what the file exists
to make visible. A result at or above `RESULT_GOLDEN_MAX_BYTES` (256 KB) keeps its section and
carries a marker saying so, rather than being deleted — the legacy path deletes it, which reads as
"no golden" and as "golden not applicable" identically, and `build-test.md` states the old rule
and is corrected in the same commit.

**`live_cpu` is what the device uses when no frozen result can serve it, and there are two such
cases rather than one.** The first is legacy's: a result over the cap has no golden to compare
against, so the device is held to a live cpu run instead. The second falls out of this mode and
has no legacy counterpart — `.result.txt` holds one entry per query, authored by one mode, while
the device runs at every mode the query enables. Where the rows are the same at every mode that
is one golden serving five runs, which is the point of the single entry. Where they are not, it
is one golden that only one of them can match: `scan-limit` at `bp-tp1-single` returns different
rows from the `bp-tp4-sized` section, and `golden_exact` would fail on a correct device. So a
query whose `cpu_oracle` is `data_fusion_subset` takes `live_cpu` on the device, compared against
a cpu run at the *same* mode — where both walk one driver over one plan and the rows are
determined again.

Both conditions are derivable — one from the golden's own marker, one from the declaration two
arguments to its left — so neither is a seventh argument, and both are asserted rather than
trusted. A `golden_exact` where the cap or the subset oracle applies is a test that fails on
correct behaviour; a `live_cpu` where neither does is a device run spending a live oracle on a
comparison a committed file would have made faster and harder.

A section can therefore turn from content into a marker, and back, as a result crosses the
threshold. Nothing guards that transition and nothing should: at a fixed scale factor a result's
size moves only when its answer moves, which is the thing every other check in the tier is
already watching. The marker is there so the file says why a result is absent, not to absorb a
size that oscillates.

**A run that stopped early says so, on its own line.** A satisfied limit ends the run with work
undone: lanes that were never pulled, row groups never read, batches produced and never consumed.
Every one of those shows up in the annotations as a smaller number, and a smaller number with no
stated cause is indistinguishable from a plan that produced less — which is the absent-versus-
skipped confusion this tier exists to prevent, one level down. So each section opens with
`early_exit=<node>@<ordinal>` naming the limit that was satisfied, or `early_exit=none`. Always
present, never inferred from absence: a query that ran to completion is a fact the file states.
It is stable, because [the determinism rules](#determinism-rules) pin the schedule — which
batches were consumed before the limit was met is fixed for a plan.

Beside it, `rows_skipped=N` on each node that has any: rows released without an unload call,
which the driver already counts per node precisely because the rows returned look identical
either way. It is the saving a limit buys, and the golden is where it becomes visible.

**A batch emitted and never consumed is still emitted.** It is counted where it enters a queue,
so it is in its producer's `batch_rows` and `batch_bytes` whether or not the parent ever took it;
what the parent took is its `in_rows`. The gap between the two is the work a limit threw away,
per child and per lane, and it is the quantity `early_exit` exists to explain — without the
marker it reads as a node that emitted more than its parent wanted, which is a defect
everywhere else.

The same rule decides the cost, and the answer is not the intuitive one: `.cost.txt` prices what
was produced, not what was used, because the device did that work and holding those bytes is what
the budget was spent on. A query that exits early is genuinely cheaper than one that does not,
and the batches it produced before exiting are genuinely not free.

Where the drop begins is bounded rather than recorded, and does not need a third list. Queues are
FIFO and arrival order is pinned, so what a parent took is a prefix of the child's lane — the
cumulative sum of `batch_rows[j]` reaching `in_rows`, ending mid-batch only where a limit sliced a
straddling one. So `in_rows[child][j]` lies between two adjacent prefix sums of that child's lane,
which is a tighter check than `<=` on the total and is the form to assert.

**Two checks change shape under it, and both get stronger rather than weaker.** The loader
identity — `batch_rows` having `partition_groups`' exact shape — is a plan-against-run
comparison, and early exit is exactly when the run does less than the plan. Per lane it becomes a
prefix: `batch_rows[j]` has at most as many entries as `partition_groups[j]`, aligned from the
start because a scan reads its groups in order, and equal on any run whose marker says `none`.
And the `in_rows` exceptions stop being permanent exemptions: a limit-satisfied node and an
empty-build-side join may consume less than they were offered *when the marker says the run
stopped early*, and must consume everything when it says `none`. A conditional weakening the
file itself justifies, rather than two node kinds exempted for good.

**Several test cases now write one file, and that is the task's one real hazard.** A whole-file
write is last-writer-wins, which would drop every other query's section and leave a green run. So
a regenerating write takes an advisory lock on the file, merges its section, and publishes by
writing a sibling and renaming onto the name — the rename because a crash mid-write must not
leave a truncated golden, which `point_canonical_root` already does for the same reason.

**The lock is on the file, not in the process, and the distinction is load-bearing.** Libtest runs
a binary's cases as threads in one process, so a `Mutex` would serialize them — but that is a
guarantee about one binary, and what makes it sufficient here is the separate fact that only the
CPU binary writes. An invariant, not a property of the language: the next writing binary breaks it
with nothing red, `cargo nextest` runs each case in its own process and would silently reduce a
mutex to no lock at all, and two shells regenerating at once are outside any of it. The
`canonical_root` comment records the same surprise one level down — two binaries reaching one path
at the same time, fixed with an atomic rename rather than a lock, because a lock in one process
could not see the other. `std::fs::File::lock` is stable on the toolchain in use, so this costs no
dependency.

**Partial regeneration follows from that.** A filtered run regenerates only the sections its cases
produced and leaves every other section as it is, under an environment variable of its own rather
than by widening `UPDATE_CANONICAL`, whose contract is a whole file. `PCK_TEST_FILTER` already
scopes which cases run, so the mechanism has a caller before it has a second one. What must
survive is the distinction above: a filtered regen that cannot tell "did not run" from "stopped
planning" deletes coverage silently, which is exactly what a golden exists to prevent.

**`.cost.txt` stays a pure function of its `.cpu.txt` and `cost_model.conf`**, so the derivation
is per section rather than per file, and `test_cost_model` re-derives every one.

**The GPU side reads what the CPU side wrote**, as in legacy: per-node plan shape and the
input/output statistics against that mode's `.cpu.txt` always, and `.result.txt` where the
`gpu_oracle` names a golden. It never writes either, and ignores the regeneration variable
rather than honouring it — a device that can author its own golden proves nothing against it.

One thing is easier here than in legacy, and worth not spending twice. Legacy's two engines run
separate executors, so the two rendering the same tree is a coincidence the golden exists to
check; here `batch_partitioned_driver` is generic over the backend, so both engines walk one
driver and report through one `RunReport`. The rendering is therefore written once, and what the
golden still checks is the answer the device gave, not the shape of the report it came in.

**Registry.** The five `bp_*` columns exist and mean plan enablement, declared by the golden's
section rather than by a macro — `test_batch_partitioned_plans` holds the two to each other in
both directions. Execution needs its own columns, five per engine, and those are macro-declared
through the existing link-time inventory, so the widget's three groups each have a source that
something checks. Adding ten columns to a seventeen-column csv is the point to decide whether the
row stays flat or the modes become a repeated group; the inventory tests are what must keep
working either way.

**Widget.** A batch-partitioned table per bench, repeating the legacy structure — peacockdb cost,
duckdb cost, ratio, features, tickets — with the four mode columns replaced by three: planning,
cpu execution, gpu execution. Each holds one cell per mode. A planning cell links to that query's
section in `<mode>.plans.txt`, which is also where its refusal is, so an enabled query and a
refused one link to the same file and differ only in what the reader lands on. A cpu cell links to
the query's section in that mode's `.cpu.txt`. Where not all modes are enabled, the cost columns
use the last mode in the sequence for which cpu execution is enabled, as the legacy rule does for
its own last mode.

**The cost-regression gate is a third rendering, and it degrades silently unless this task
changes it.** `--cost-diff` compares each `.cost.txt` against the same path at the PR's base and
upserts its own PR comment: improvements green, regressions red, and a non-zero exit that fails
the build. It is where a cost win becomes visible, so the new mode belongs in it. But it is
written for one file per query — `collect_cost_goldens` globs `*.cost.txt`, `read_total` takes
the *first* `peacockdb_cost=` line in a file, and `diff_label` takes the filename's first
dot-segment as the query name. Point that at `bp-tp4-sized-mini.cost.txt` and it picks the file
up, reads whichever query's section happens to be first, and labels the number with the mode: a
wrong row, rendered confidently, gating the build. The glob is what keeps it quiet — the files
are found, so nothing reports them missing.

So the differ becomes section-aware in the same way the comparator did: a row per (query, mode),
totals read per section, the label carrying both. `cost_diff` itself is unchanged, since it
already works over a map of label to number; what changes is what fills the map. Its unit tests
are the pattern to extend, and the red case is a two-section file whose second section moved —
the per-file reader reports no change, the section reader reports one.

**There are two widgets, and both get the table.** `cost-report` renders the same data twice —
`--md`, a markdown blob upserted as one PR comment keyed on a sentinel, and `--html`, the site
published to Pages from master. Close to identical rather than identical, and the differences
are the ones a format forces: markdown cannot set a row background, so a row over the ratio
threshold is flagged in the cell instead, and the mode cells have a markdown counterpart to
`mode_cells_html` rather than sharing it. A table added to one and not the other is the failure
to expect here, because the html is what a person looks at and the markdown is what the review
actually reads.

**No production change that alters what the engine computes, here or in T19.** A query that
does not run is disabled with a ticket, never fixed in passing: a fix made while enabling one
query is a fix nothing else in the branch proves, and the diff under review stops being the
infra. What this task does touch in `peacockdb-core/src` is the goldens' own surface and nothing
else — the node renderer per [Node display](#node-display) (`name@ordinal` references, layout in
place of the lane count, `fetch` and the aggregate lists wherever carried, schema in the plan
golden only), the per-batch line below, `cost_model.conf` entries for all eighteen node kinds at
once rather than as each is first seen — held there by a case red on a nineteenth, derived from
the exhaustive `as_node_ref` the way the writer's field cover is, since entered-at-once is a
moment and T19's "no `cost_model.conf`" rule needs a property — and
per-node emitted rows and bytes on `RunReport` — the one thing the engine does not already
report. Everything else the
comparison needs is there: `Batch` gives rows and bytes on both backends, the device's coming
through the frozen ABI's `PeacockNodeStats` priced against the declared schema, and
`trace`'s per-event `outputs` sums to `out_batches` per lane. What is absent is a per-node total
of what each node emitted; the driver reads both numbers already, for the accountant and the
limit interval, and keeps only aggregates. `rows_seen` is not the emitted total and must not be
reused as one — it counts rows arriving at a node for the limit rule, which is `in_rows` summed
over lanes rather than anything a node produced. Plus the
`pipeline.yml` steps `test_ci_coverage` requires, which are not engine code at all. Naming that
surface is what gives the rule an edge: T19 has none of it left to touch, and T20 is where the
engine moves again.

**Every node carries what it consumed and the size of every batch it emitted**, as parallel
structures on one continuation line under the node:

    GpuLoadParquet: table=lineitem, partition_groups=[[[0,1],[2,3]],[[4],[5,6,7]]], lanes=2, …
      in_rows=[] batch_rows=[[1500304, 1500303], [1500304, 1500304]] batch_bytes=[[31881456, 31881440], [31881472, 31881455]]

Lanes outermost and batches within, which is `partition_groups`' own nesting — so on a loader,
element `i` of lane `j` in all three lines is one batch, and the row groups that produced it sit
at the same index as its size.

`in_rows` is what the node consumed, per lane, nested by child rather than flat — a filter reads
`in_rows=[[860160, 737280, …]]` and a join reads both of its sides. Legacy prints one `in_rows`
per partition and it is the first child's, so a join's build side appears nowhere on the join's
own line; here the two sides differ in kind and the capability matrix turns on which is which,
so a format that can only show one of them is one that hides the interesting half. A source has
no children and prints `[]`, which is not the same as a child that consumed nothing. It is the
one figure the batch lists cannot imply, and the reason legacy's sub-line is dropped rather than
merely thinned: rows in against rows out, on the node's own line, is what makes a selectivity or
a skew visible without walking the tree.

Rows and bytes are separate lists rather than pairs because each is then a column a reader scans
down and a diff reports as one moved number; the cost is that a batch's two figures are not
adjacent, which the shared indexing is what makes navigable. All on one line, so a node stays one
entry however many batches it has.

This is where the per-batch record lives, rather than in a `batch-info.cpu.txt` for a chosen ten:
a separate file for a subset was worth it only while the figures were too bulky for every node,
and at a median of one batch and a p90 of six they are not. The tail is what to watch — the
worst source in the corpus is tpcds `inventory` at `bp-tp1-rowgroup` with 96 batches in one
lane, about two kilobytes across the two lists — so the comparator reports a moved section by
name and never dumps the line.

**A failing query is disabled with a ticket in [`bp-tickets.md`](bp-tickets.md)**, not in the
main list. Rollout tickets arrive in bulk when a sweep hits a wall and close in bulk when it is
cleared, which is not what a triage pass reads for. The ID space is shared, so a number is never
two things. `TicketIndex::load` reads two files and gains a third here, and it is
not cosmetic: `cost-report` already exits 1 when a ticket the registry names resolves in neither
file, so the first rollout ticket filed in `bp-tickets.md` fails the cost-report job until the
index reads it.

**What proves the infra, since every way it fails is quiet.** A dropped section, a filtered
regeneration that deletes coverage, a skipped marker a reader cannot tell from an absent one — none
of these turn a run red on their own, so each gets a case that goes red on demand, built from
strings rather than by editing a golden and undoing it:

- two writers interleaved at the merge keep both sections, and the same case with the lock
  removed loses one — a lock nothing can be shown to need is a lock the next refactor drops.
  Forced rather than raced: two threads that merely start together prove nothing on a fast
  machine and fail on a loaded one, so the overlap is imposed — a barrier between read and write,
  or the merge driven directly with a second writer's section already on disk. Every case here
  is deterministic or it is not a case: this tier's whole claim is that a golden means something,
  and a test that passes on the third run is a golden that means nothing;
- a filtered regeneration rewrites its own sections and leaves every other byte identical;
- a section absent because its case did not run and one absent because the query stopped planning
  are distinguished, and the run that confuses them fails;
- a skipped marker round-trips: a query whose bit is clear writes one, and clearing a bit that was
  set turns a real section into a marker rather than deleting it;
- both of `cost-report`'s renderings carry the new table, asserted per rendering rather than once;
- the section reader and the per-file reader disagree on a two-section file whose second section
  moved, which is the cost gate's red case.

**Three things have to agree, and the inventory test becomes the place they do.** Legacy holds
the macro invocations and `cost-registry.csv` to each other in both directions, per binary,
because `inventory` collects per linked binary. The corpus tier adds a third leg: the goldens.
Every `corpus_query!` registration means an enabled cell, every enabled cell means a
registration, and every enabled (query, mode) means a golden section carrying content rather than
a skipped marker — with the converse holding too, since a `disabled` cell whose section is full
is coverage nobody is reading. The plan columns stay declared by the goldens alone, as they are
today; what is new is the execution columns being declared by the macro and checked against both.
One consequence to state rather than discover: the gpu half of this runs only on the gpu host,
because that is where its registrations are linked, so a gpu-column drift does not go red on the
cpu leg.

**Soundness is checked against the file's own redundancy, not with regular expressions.** A
regexp says a line looks like a line. What is wanted is whether the numbers mean anything, and
the format carries the same quantity more than once by design, so the checks are arithmetic and
cheap:

- a node's `out_rows` is the sum of its `batch_rows` lanes, and `out_bytes` of its `batch_bytes`.
  A batch is counted where it enters a queue, so a scatter output dropped for being empty is not
  one — the sums are unaffected either way, and the batch count then means what flowed;
- the lane count of those lists is the `lanes=N` on the node line;
- a node's `in_rows` for child *k* equals that child's `out_rows`, **per lane and not only in
  total**: the index is the *child's* lane, so it lines up with that child's own `batch_rows`
  entry for entry. Indexing by the consuming node's lanes would make the identity checkable only
  in aggregate wherever the two counts differ, which is every emitter, merge and accumulator —
  the nodes it is most worth checking. An equality, with the exceptions named rather than
  softened away: a node whose limit is satisfied stops pulling, and a join with an empty build
  side ends its lane with the probe's batches unconsumed ([#175](../tickets.md#t175) is the same
  shape from the other side), so those two assert `in_rows <= out_rows` of the child and every
  other node asserts equality. `<=` everywhere is the repair to refuse: it passes for almost
  anything, and this is the identity most worth having;
- on a loader, `batch_rows` has `partition_groups`' exact shape, which is the correspondence that
  nesting was chosen for;
- the root's `out_rows` is the row count in `.result.txt`, which the oracle checked.

That last one is the interesting one, and the rest are worth more than they look. The cost tree
has no external oracle — the golden is written by the run it will later check — but a file that
contradicts itself is a file a renderer got wrong, and a renderer is most of what could be wrong.
It is a weak oracle rather than none, and it is free: the redundancy is already committed.

**Reading it back needs one parser, and there are already two and a half.** What exists:
`ordered_sections` and `section_differences`, which split on `== <query>` and report what moved,
private to `test_batch_partitioned_plans.rs`; `parse_node_line` in `common/cost_model.rs`, which
finds `output_bytes=` and takes the leading identifier, private, and enough for the cost
derivation and nothing else; and `read_total_str` in `cost-report`, another crate reading a
`key=` off a line. None of them knows a tree, a lane, or a nested list, which is what the
arithmetic above asks for.

So: the comparator moves into `tests/common/`, where both corpus binaries can reach it — and that
move is owed anyway, since `test_batch_partitioned_plans.rs` is at 1418 lines against a
thousand-line cap. `parse_node_line` grows from `(type, bytes)` into the node's fields and its
indent depth, and `cost_text_from_cpu` becomes a caller, so the cost derivation and the soundness
checks read one structure rather than two that agree by luck. `cost-report` stays separate, since
it reads a total out of `.cost.txt` and never a node line; what must not drift between the crates
is the `== <query>` convention itself, which is why it is written here rather than inferred from
whichever file a reader opens first.

A fourth parser is the thing to refuse. Three readers of one format already disagree about what a
node line is, and the format is about to carry four more fields.

**Consolidating them saves no code, and that is not the reason to do it.** All the parsing in the
tree is about ninety lines: `ordered_sections` is fifteen, `parse_node_line` seventeen,
`cost-report`'s `field` and `read_total_str` thirteen between them, and the rest is
`section_differences` reporting. A parser that knows a tree, a lane and a nested list is larger
than the three it replaces, so the change adds lines. What it buys is one definition of the
format at the moment the format gains four fields — the divergence is the cost, never the
duplication.

**The crate boundary is where consolidation stops, on purpose.** `cost-report`'s `[dependencies]`
is empty, and deliberately: it builds in seconds in the cpu tier because it pulls in neither the
executor nor a device. Making it depend on `peacockdb-core` to share thirteen lines of `key=`
reading trades a stated property for nothing, and a new workspace crate for the same thirteen
lines is worse. It needs the convention, not the code — `== <query>` sections, one
`peacockdb_cost=` per section.

So pin the convention with a fixture rather than with shared code: one committed two-section
sample that `cost-report`'s unit tests and the test-side parser's unit tests both read, each
asserting the same extracted values. The two readers then diverge red without either crate
depending on the other, which is the property that was actually wanted.

**An unordered `LIMIT` runs at every mode, and drops the DataFusion oracle rather than the
mode.** Legacy canonizes `scan_limit` at tp1 because at tp>1 its rows and its per-node bytes vary
run to run. That is a property of legacy's executor, not of this one:
[the determinism rules](#determinism-rules) pin the schedule and require that one plan run twice
gives one answer byte for byte, so a section here is stable at every mode. What those rules
explicitly do not promise is agreement *across* plans — tp1 and tp4 may return different rows
where the SQL does not determine which, which is what an unordered `LIMIT` is.

So the mode is fine and the comparison is what narrows: DataFusion single-stream is not an
authority on *which* ten rows a four-lane plan returns. It is still an authority on the rest, and
an unordered limit determines more than it looks. The count is `max(0, min(n, |unlimited| - m))`
for `LIMIT n OFFSET m` — written with the offset from the start, because the zero-skip form is a
rule that goes wrong on the first query that carries one and T20 adds those deliberately. And the
rows are a sub-*multiset* of the unlimited result, compared as a multiset and not as a set: set
membership passes a run that returned one row twice where the oracle has it once, which is a
live failure mode for a limit over a join. Both are asked of the session that
already runs, neither needs the two plans to agree on which rows, and together they catch what a
frozen golden cannot: a limit dropped, an offset ignored, rows invented, the wrong table sliced.

That is the third `cpu_oracle` value, and it is named for what it checks rather than for what it
declines to — the count and the containment, against `data_fusion_exact`'s whole answer. So no
query's answer is frozen with no external check, `scan-limit` included. The device is held to the
cpu's section on top of that, which is a real check rather than a fallback: both engines walk one
driver over one plan with emission order pinned, so disagreeing means one of them broke the
schedule.

Two limits to state with it. An *ordered* `LIMIT` with ties at the boundary is not covered:
neither `sort_unstable_by` nor cuDF's `sorted_order` is stable, so which tied row survives is
decided by neither engine's contract and cpu-gpu agreement is not owed. And the scope is small —
the determinism section counts one corpus query with a bare `LIMIT` whose rows are undetermined,
`tpch-queries/scan-limit.sql`; the other four limit an already-single-row aggregate.

**A query carries every mode it is correct at, and a ticket for the ones it is not.** Not a
default to be tuned: all five where all five run and agree with the oracle, and where some do
not, the ones that do are enabled and the rest are `disabled` against a ticket naming which modes
and why. That is what makes the mode arguments a set rather than a switch, and it is the same
rule on both engines — a query can be correct at five modes on the cpu and at two on a device,
which is two tickets and not one.

**The cost of running all of it is accepted for now.** 119 queries at five modes is about 595 cpu
runs and as many on a device, against a corpus measurement nobody has: T17a's report covers
eleven queries chosen for being the cheapest. Measuring it the way T17a did — serially, one row
per query — is still worth doing, because a number nobody has is a number nobody can decide
against later. What it does not do is gate the rollout. The one thing that would force this open
again is a job that stops finishing: pipeline.yml's cpp-cpu leg has already been lost once at
fifty-eight minutes on a tier a fraction of this size, and the device leg runs
`--test-threads=1` on one host.

**A first enablement freezes whatever the run produced, and only half of it has an oracle.** The
answer does: `assert_cpu_results_match_datafusion` builds a separate plain-DataFusion session at
`target_partitions = 1` and compares against it unconditionally, before and regardless of any
golden write, so a wrong answer cannot be frozen — and that comparison is the standing check on
every later run too, not a first-enablement rite. The `.cpu.txt` does not: under a regeneration
it is written with no comparison, and nothing independent says the plan should have that shape or
that an interior node emitted that many rows. The root is anchored — its `out_rows` is the
result's row count, which the oracle checked — and every node below it is taken on trust.

What later covers some of it is the device asserting against the same file read-only, and it is
worth being exact about which half of that assertion carries information. The tree carries none:
`batch_partitioned_driver` is generic over the backend, so both engines walk one driver over one
plan and produce the same shape by construction rather than by agreeing. That is the design and
not a weakness of it — the walk being identical is what makes the two engines comparable at all.
The evidence is the rows and the bytes: the same walk, and the device's numbers against the
CPU's. The tree assertion stays anyway, because it costs nothing and goes red on the day that
construction stops holding, which is the only day it could ever say anything.

So enabling a hundred queries is an ordinary act for the answers, and for the interior of the
cost tree it rests on two engines producing the same counts through one walk. The mitigation is
the one T19 already has: batches of five, where a tree that looks wrong is still attributable.

**`build-test.md` gains rows, not only counts**: one per new tier in the category table, and
three in the golden table — the per-mode `.cpu.txt` and `.cost.txt` and the one `.result.txt` —
each naming the partial-regeneration variable beside `UPDATE_CANONICAL`. That table's existing
`.result.txt` row says the golden is deleted above 256 KB, which is the rule this task replaces,
so the row is split rather than edited: legacy keeps its sentence and the new mode states its own.

**And enough queries to prove the infra**: about ten tpch and about ten tpcds at sf1, enabled on
the macro as it is built. They are chosen by plan size off the committed goldens rather than by
taste, because the constraint is not what can plan — all 22 tpch benchmark queries already plan
and attach recipes at all five modes, as do 81 of the 99 tpcds — but what is small enough that a
failure is legible. The tpch ten are the smallest carrying no ticket: q6, q1, q14, q19, q12, q13,
q15, q17, q3, q10. The tpcds ten are q41, q42, q3, q43, q52, q55, q96, q15, q37, q82. A legacy
ticket on a row is not automatically a blocker here — [#97](../tickets.md#t97) is the real-8-way
join blocker and this is a different executor — so those rows are held back as unknown rather
than as known bad, and T19 is where that is settled query by query.

| bench | enabled here |
|---|---|
| tpch | `q6` `q1` `q14` `q19` `q12` `q13` `q15` `q17` `q3` `q10` |
| tpcds | `q41` `q42` `q3` `q43` `q52` `q55` `q96` `q15` `q37` `q82` |

**T19 — rollout.** Query-by-query enablement across the rest of the corpus on T18's macro,
starting from the twenty it already carries. No production code changes, per T18: a query that
does not run is disabled with a ticket in [`bp-tickets.md`](bp-tickets.md), which is the whole
output of this task besides the enabled rows. Not one line of `peacockdb-core/src` moves here:
T18 left the goldens' own surface finished, so anything this task would have to change is by
definition a query's blocker rather than the rollout's.

**Three paths are edited by hand, two kinds of file move without being written, and a commit
touching anything else has stopped being a rollout.** By hand:

- the shared `corpus_query!` list, one file that every binary includes and reads through its own
  arm, as `gpu_cases.inc` is read today — so a query is enabled once, not once per engine;
- `testdata/cost-registry.csv`, cells rather than rows: all 138 corpus queries already have one;
- [`bp-tickets.md`](bp-tickets.md), for what a query is disabled on — a whole query, or the
  subset of modes it is wrong at, which is the commoner case and the one whose ticket has to name
  the modes or the next reader re-derives them.

Every cell this task turns off carries a ticket, which is an invariant the registry already
holds: no row in it today has a `disabled` cell and an empty ticket list, and the four queries
DataFusion cannot plan at all carry [#23](../tickets.md#t23) against an `na`. A rollout is the
one thing that could break it, since it is the only task that turns cells off in bulk.

Then the **golden sections** fill in: a query enabled at a mode stops carrying that mode's
skipped marker and starts carrying its plan, its per-batch sizes and its costs. That is eleven
files for the bench it belongs to — a `.cpu.txt` and a `.cost.txt` per mode, plus the one
`.result.txt` across modes — and twenty-two for a commit spanning both. The number is the point:
it is modes × benches and does not grow with the corpus, so the hundredth query enabled moves
the same eleven files as the first, where legacy would have added three more per (query, label)
and did, 277 of them for tpch.sf1 alone. And **`build-test.md`'s counts** move with the case
count, in the commit that moves them rather than in a later sweep.

**What it enables, and what stays out.** 119 of the corpus's 138 queries plan at all five modes
today, which is the eligibility test — not legacy enablement, since a query legacy disabled may
have been fixed since and the goldens are what know. Twenty are T18's, so a hundred are here:

| bench | enabled here |
|---|---|
| tpch (29) | `aggregate-groupby` `anti-join` `cross-join` `filter-project` `hash-join` `join-int` `left-join` `mixed-join` `nested-limits` `nested-loop-join` `nested-loop-left-join` `q11` `q16` `q18` `q2` `q20` `q21` `q22` `q4` `q5` `q7` `q8` `q9` `rollup-over-join` `scan-limit` `semi-join` `shuffle-additive` `shuffle-additive-avg` `shuffle-stddev` |
| tpcds (71) | `q1` `q10` `q11` `q13` `q14` `q16` `q17` `q18` `q19` `q2` `q21` `q22` `q23` `q24` `q25` `q26` `q29` `q30` `q31` `q32` `q33` `q34` `q35` `q38` `q39` `q4` `q40` `q45` `q46` `q48` `q5` `q50` `q54` `q56` `q58` `q59` `q6` `q60` `q61` `q62` `q64` `q65` `q66` `q68` `q69` `q7` `q71` `q73` `q74` `q75` `q76` `q77` `q78` `q79` `q8` `q80` `q81` `q83` `q84` `q85` `q87` `q88` `q9` `q90` `q91` `q92` `q93` `q94` `q95` `q97` `q99` |

`tpch/mixed-join` is CPU-only and the one query whose two engines differ: it plans, validates and
runs, and its recipes do not attach — [#168](../tickets.md#t168)'s interval `ScalarValue`, so the
crossing to a device is what fails, not the plan. `cpu_modes` carries all five and `gpu_modes`
none, which is the case the two mode arguments exist for.

The nineteen that stay out, every one against a ticket that already exists:

| held back | why | ticket |
|---|---|---|
| tpcds `q12` `q20` `q36` `q44` `q47` `q49` `q51` `q53` `q57` `q63` `q67` `q89` `q98` | the planner refuses `WindowAggExec` and `BoundedWindowAggExec` — the one feature regression against legacy, which plans all thirteen | [#143](../tickets.md#t143) |
| tpcds `q27` `q70` `q72` `q86` | DataFusion 45 does not physical-plan them at all, so they are dead in legacy too (`plan_status=fail`) | [#23](../tickets.md#t23) |
| tpcds `q28` | a `DISTINCT` inside `count(DISTINCT …)`, refused by name | [#62](../tickets.md#t62) |

Window functions stay disabled and are not this task's to fix — [#143](../tickets.md#t143) calls
it the blocker for ever retiring the legacy modes, which is a decision above a rollout.

**Seventeen of these hundred are already run by T17's tier, and the overlap is deliberate.**
`test_cpu_batch_partitioned` runs six queries at the five modes and eleven more at the modes plus
the injected shapes; every one of the seventeen falls in this task's list and none in T18's —
tpch `left-join` `q20` `q21` `shuffle-stddev` `anti-join` `nested-limits` `nested-loop-join`
`nested-loop-left-join`, tpcds `q38` `q87` `q2` `q8` `q16` `q33` `q45` `q93` `q97`. They are not
excluded, and a reader who finds the duplication should leave it.

The two tiers ask different questions of the same query: the injected one asks whether the
drivers tolerate a layout no planner would emit, over one plan; the corpus one asks whether the
answer, the plan and the costs match a golden, across the whole bench. Neither answers for the
other. And a corpus list defined by subtracting another list is a set held in prose — which is
exactly the defect T17a's completeness pass found in its own fixture list and closed by
generating `INJECTED` from one declaration. Reintroducing it one level up, so that removing a
query from the injected tier would silently need a matching addition here, buys a duplicate mode
run and costs the property that made the smaller list trustworthy.

**Roll out in batches of about five, not in one sweep.** A hundred queries enabled at once is a
regeneration whose diff nobody reads and a failure nobody can attribute: eleven golden files move
either way, so the batch size is the only thing that says which query moved which section. Five
is small enough that a red run names its cause without bisecting and large enough that the
regeneration cost is amortised. Each batch is its own commit — the five queries, their registry
cells, the sections they filled, and whatever went to `bp-tickets.md` — so the history reads as
the rollout it was, and a batch that goes wrong is reverted without taking the ninety-five with
it.

Nothing else, and the list is still short enough to read off a diff: no `cost_model.conf`,
because T18 enters all eighteen node kinds at once rather than as each is first seen — an
exhaustive set entered piecemeal is one whose next gap is a rollout's problem; no `pipeline.yml` or `test_ci_coverage`,
because the targets exist by then; no `registry.rs`, whose columns are T18's.

That sweep is where T17's `nested-loop-left-join` gets its GPU columns. They ship `na` because
T17 could not commit a device run, which is [#116](../tickets.md#t116)'s shape — a cell with no
coverage and no blocker — so the reason is recorded here rather than left for a reader to
reconstruct from an empty column.

**T20 — corpus shapes the benchmarks do not have.** The corpus is numeric-aggregate heavy, and
this mode's risk sits in what it never sees: an audit of every `.cpu.txt` finds zero `OFFSET`s,
zero `min`/`max` over a non-numeric column, and no query at all for several shapes the tickets
already describe as unexercised. Add a small set of hand-written queries over the **existing**
tables — no new dataset, no new scale factor — each justified by naming the feature it reaches and
the ticket or code path that cares. Where a query needs engine work to run at all, that work is
this task's too, which is what makes this the one task after T19 that still moves the planner.

| Query shape | Why it is worth a query |
|---|---|
| `ORDER BY … LIMIT n OFFSET m` | every `GpuGlobalLimitExec` in the corpus has `skip=0`, so the offset half of both lowerings — the released prefix, the two straddling slices, and a pure offset that never satisfies — runs only in synthetic tests. T17 executes `nested-limits`, so what is left here is the legacy tiers; shape it away from [#166](../tickets.md#t166)'s two droppers, an outer limit above an aggregate and a limit inside a `UNION ALL` branch |
| `min`/`max` over a string or date column | zero uses today; the merge is a string reduce, and nothing exercises one. Needs nothing new: `aggregates.rs` decomposes `Min`/`Max` without reference to type, and the generic reduce arm takes the input's |
| a join on a nullable key | [#59](../tickets.md#t59) and [#80](../tickets.md#t80) both record that no corpus query has one, and this mode adds [#137](../tickets.md#t137)'s skew on top — every all-null key lands in one lane. #137's planner half comes with it: the translation inserts `GpuFilter(<key> IS NOT NULL)` under the feeding `GpuEmitPartitions`, on the sides whose unmatched rows are never emitted. Expect the query red on the anti-join's three-valued logic until #80 |
| a shuffle keyed on a decimal | [#95](../tickets.md#t95)'s kernel throws on decimal keys; the batched mode shuffles more often than legacy, so this stops being latent. The largest item here, and not a query: dispatch by logical precision (≤18 → the low 8 LE bytes of the int128, >18 → the raw 16), the precision threaded through the partition FFI, and the murmur3 conformance gate extended to cover it — that gate is what makes CPU and GPU place a row in the same lane, so a decimal key it does not cover is a key the two engines may disagree on silently |
| two `DISTINCT` arguments over different expressions | [#144](../tickets.md#t144) has no refusal of its own — `translate/aggregate.rs` refuses every distinct aggregate naming [#62](../tickets.md#t62) — so the translator learns to tell the two shapes apart and the query provokes a refusal that names its own ticket. A refusal pointing at the wrong ticket sends its reader to the wrong fix |
| a wide `SELECT DISTINCT` | dedup whose state is the whole row rather than a few keys — the compaction threshold's worst case, where nothing merges and the doubling has to earn its keep |

Each query lands in `testdata/{tpch,tpcds}-queries/` with registry rows and goldens through the
existing generators. Keep the set small: every query multiplies across enabled modes and tiers,
which is why the corpus grew the way it did — and why T18 comes first, so that the multiplication
lands in sections of a file rather than in files.

**T22 — per-node benchmarks for the new mode.** Port `peacock_gpu_benchmarks` to the
batch-partitioned executor, keeping the protocol that makes its numbers comparable: one
discarded warm-up, ten measured runs, the **2nd-smallest by `total_us`** reported whole, and
the floor measured over 200 samples. The run counts stay compile-time constants
(`tests/common/mod.rs`) so every record in the tree was taken at the same ones.

**A node is called many times here, and that is the port's one real design question.** The
legacy record carries one `time_us` per node, or one per partition — a node runs once. In
this mode a node runs once per batch per lane, so a per-node figure is a *sum over calls*
and the call count belongs beside it; a node at 40 ms over 200 calls and one at 40 ms over
two are different findings, and a record that cannot tell them apart measures nothing useful.
`CallStats` is already returned per call and is where the per-call figures come from.

The record gains the mode and the tier — `<query>.<mode>.benchmark.txt` alongside the legacy
`<query>.<mode>-<tp>-<tier>` — and keeps `build_profile`, `sync_floor_us`,
`nodes_at_or_below_floor` and the rest, since they mean the same thing. It also carries which
allocator measured the run: the pool landed with
[#151](../archive/archived-tickets.md#t151), [#148](../tickets.md#t148) is still open, and a
number taken without one is not comparable with a number taken with one.

Case list as the correctness tiers use, so the measured set cannot drift from the verified
one, and `test_ci_coverage` exempts the target explicitly because it asserts nothing.
