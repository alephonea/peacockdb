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
   downstream phase may assume one batch per partition. Only when on does the planner take
   the memory budget (micro/mini/standard/full) and size batches, and only then does the
   threshold above bite.

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
multi-lane mid-plan limit is a case T11 constructs rather than canonizes.

### The aggregate sequence

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

Two costs, both small. The expression IR gains `sqrt` (a `UnaryOp` variant —
cuDF's `unary_operator::SQRT` is what the hardwired finalize already calls), and the
translation layer gains a decomposition registry of about six entries, whose state names
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

None of it is paper. `scripts/exec_model` runs every row on two backends that share no join
code — one joining with pandas, one emitting these seqs and interpreting them with these
calls — at every batching and partitioning shape the layout injector can produce, and the
seq sequences in the second table are asserted per mode.

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
preserve a distribution the branches no longer share. tpcds q77 is the case: DataFusion hashes
three branches into four lanes each, and the small-source rule puts the middle one on one lane.

**Equal lane counts are not co-partitioning**, which is what the column's `no` hides.
DataFusion picks `CollectLeft` for two small tables and emits no repartition at all, while
both loaders still produce N lanes — so lane p of one side holds nothing that must match
lane p of the other, and joining lane-wise would silently drop matches. The translation
therefore checks the hash and not the count: unless both sides are scattered on their own
join keys, in key order, both merge to one lane. That is the broadcast shape
[#140](../tickets.md#t140) removes. Until then read the column as "no merge beyond the one
the shuffle already did", not "no merge".

**What crosses the wire, and what cuDF runs.**

| Mode | fb seqs: per probe call / at finish | cuDF underneath |
|---|---|---|
| **Inner** | `CudfHashJoin{Inner, keys, filter, projection, null_equals_null}` / — | `inner_join(bk, pk, nulls)` → two gather maps → `gather(build, ·, DONT_CHECK)` + `gather(batch, ·, DONT_CHECK)` → column concat → residual: `build_column` + `apply_boolean_mask` → `projection` |
| **Right outer** | `CudfHashJoin{Right}` / — | `left_join(pk, bk, nulls)` with the pair swapped back → `gather(build, ·, NULLIFY)` + `gather(batch, ·, DONT_CHECK)` |
| **Left outer** | `CudfHashJoin{Inner}` + `CudfProject`(key ordinals) / `CudfCoalescePartitions` → `CudfHashJoin{LeftAnti}` → `CudfProject`(build columns + typed NULL literals) | per call as Inner; at finish `left_anti_join(bk, accumulated_keys, EQUAL)` → `gather` → one literal-only `compute_column` per padded column |
| **Full outer** | `CudfHashJoin{Right}` + `CudfProject`(keys) / as Left outer | per call as Right outer; finish as Left outer |
| **Build-side semi family** | `CudfProject`(key ordinals) / `CudfCoalescePartitions` → `CudfHashJoin{LeftSemi\|LeftAnti\|LeftMark}` | finish only: `left_semi_join` / `left_anti_join` (`filtered_join::semi_join` / `anti_join` where the header exists) → `gather`; mark scatters `true` into an all-false column and appends it |
| **Probe-side semi family** | `CudfHashJoin{RightSemi\|RightAnti}` / — | `left_semi_join(pk, bk, nulls)` with the sides swapped, as the C++ does → `gather(batch, ·)` |
| **Cross join** | `CudfCrossJoin` / — | `cross_join(build, batch)` |
| **Nested-loop Inner** | `CudfNestedLoopJoin{Inner, filter, filter_columns}` / — | `conditional_inner_join(build, batch, ast)` → `gather` ×2; or `cross_join` → `build_column` → `apply_boolean_mask` when the predicate is not AST-able |
| **Nested-loop Left** | one `CudfNestedLoopJoin{Left, filter}` / — | `conditional_left_join` → `gather(build, ·)` + `gather(probe, ·, NULLIFY)` |

**One worked example per mode.** `dim(k, label)` is the build side and `fact(fk, v)` the
probe throughout, so the shapes are comparable; the plan column shows the join subtree only,
since everything above and below it is the same in every row. Read the fourth column as one
lane: `p` is any lane, and every lane does this.

| Mode | SQL | batch-partitioned plan | inside lane `p`, in cuDF |
|---|---|---|---|
| **Inner** | `SELECT * FROM dim d JOIN fact f ON d.k = f.fk` | `GpuJoin{Inner, on d.k@0 = f.fk@0}`<br>`├─ build: GpuEmitPartitions(k) → GpuCoalesceAllBatches`<br>`└─ probe: GpuEmitPartitions(fk)` | build lands as one resident `cudf::table`; then per probe batch: copy the build handle (it is consumed), `inner_join` → two gather maps, `gather` each side, concat the columns, slice to the `projection`, hand the output up as a handle. Nothing at done |
| **Right outer** | `SELECT * FROM dim d RIGHT JOIN fact f ON d.k = f.fk` | same, `GpuJoin{Right}` | as Inner, but the map for the build side carries `JoinNoneValue` where the batch had no match, and that gather is `NULLIFY` — so this batch's unmatched probe rows come out with NULL build columns. Correct batch-locally: the build side was complete before the first probe call |
| **Left outer** | `SELECT * FROM dim d LEFT JOIN fact f ON d.k = f.fk` | `GpuJoin{Left, on k@0 = fk@0}`<br>`├─ build: … → GpuCoalesceAllBatches`<br>`└─ probe: GpuEmitPartitions(fk)` | per probe batch, two calls: `CudfProject` off a copy of the batch keeps its key column (that is the lane's growing key table), and `CudfHashJoin{Inner}` off a copy of the build emits this batch's matches. At done: concat the key tables, `left_anti_join(build, keys)` → the build rows nothing ever matched, then one project that appends a typed NULL per probe column |
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
inside a supported feature:

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
batch: the probe-local types (Inner, Right, RightSemi, RightAnti) need **one build-side
copy**; Left and Full need that **plus a copy of the probe batch**, because the join
consumes the batch and the key accumulation needs it too; the build-side semi family needs
**none at all**, since its probe calls never touch the build side. [#145](../tickets.md#t145)'s
refcounted handle removes all of them, and it is the one change that would let this mode
stream a probe without a copy. The prototype counts every copy it takes, so the figure for
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
batch hands its key table straight to the finish join, none registers an empty one, which
is what an empty lane means. And the word "partitions" in its name is the wire's
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
// Backend::executors_for(ctx: &Self::Context, node: &dyn GpuNode, lane: usize)
//   -> Result<NodeExecutors<B>, PlanError>. A fresh instance set per call, so the driver
//   instantiates per lane; `lane` is needed because a loader's lane picks its own row
//   groups out of the partitioner's mapping. It returns a Result because this match is
//   where "does this backend implement this node" is answered, and that question has a
//   no — a GPU window (#32) is the standing case.

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

A chunk is **one node's lane**, not a chain of them: min-height selection walks a batch up a
chain node by node on its own, so a chain-walking driver would duplicate the scheduler.

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

Per batch the driver then makes the three-way decision the limit lowering rule states —
outside the interval, straddling an end, inside it — before calling `unload`. A mid-plan
`GpuLimit` makes the same decision inside its own executor, slicing rather than narrowing an
export.

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
quality is observable: under-estimates are recorded with their magnitude. Both backends
measure — the CPU directly, the GPU through RMM allocator hooks — so `None` means this run
was not instrumented, not that the backend cannot report.

**Model ≥ measured is not an invariant.** `scratch_bytes` rests on the optimizer's
cardinality figure for a join and on assumed selectivity for a filter, so it will sometimes
come in low, and asserting otherwise would make the suite red for something that is not a
defect. The enforcer's contract is "fail cleanly when the accounted peak exceeds budget",
not "the budget is never exceeded" — the same class of guarantee as the legacy
`ResidentEnforcer`.

### What the corpus rollout measured

The T0 prototype now runs 22 TPC-H and 71 TPC-DS query texts through this node set, at
three partition layouts and on both join backends, under a 2 GiB accountant — 558 runs over
the sf1 tables (`scripts/exec_model/tests/`). The formula above survived that; the
following is what it got wrong first, and every item is a property of the design rather
than of the prototype, so the Rust implementation inherits all of it.

**An executor's residency is not one number.** `merged_scratch` prices an output row as
`probe_row_bytes + build_row_bytes + 16`, and derives `build_row_bytes` by dividing the
executor's residency by its build row count. That identity holds only while residency *is*
the build side. Under #136 a `LEFT_SEMI`/`LEFT_ANTI`/`LEFT_MARK` join's residency also holds
every probe batch's projected keys, waiting for the finish pass — so the division charges
the accumulation to each output row. Measured on TPC-DS q37: a build side of **one row**,
8.0 MB of accumulated keys, a 250k-row probe batch, and a modelled scratch of **2.0 TB** for
a call whose entire query peaks at 11.5 MB. The enforcer refused the query at 13.8 GB
against a 2 GiB budget — a correct plan, declined. The fix is an accessor for the build side
alone (`RecipeJoin.build_bytes()`), which `scratch_bytes` divides by while `resident_bytes`
goes on reporting the whole. The trait needs no change — the split is internal to the
executor — but the rule it encodes is general: **`resident_bytes()` is a total for the
enforcer to check, never a numerator for a per-row cost.** Anything that divides it wants
the part that scales with build rows, and only the executor knows which part that is.

**The finish-pass accumulation is a residency term that grows with the probe side.** The
estimator's `subtree_max_row_bytes` vocabulary charges a join in build-side terms, which is
right for the hash table and wrong for this. A build-preserving join on the frozen surface
holds key columns for every probe row it has seen — O(probe rows in the lane × key width),
unbounded in the build side, and precisely the term that decides whether a plan fits. Plan
time must charge it per lane, for all lanes live at once.

**The CPU backend does not pay it, so it cannot be used to price it.** The pandas backend
keeps "which build rows matched" as a boolean array over the build frame: free in-process,
and exactly the thing that never crosses the C ABI. On q37 that is the difference between a
6.0 MB peak and an 11.5 MB one, for the same plan and the same answer. A residency model
calibrated on the CPU path will under-charge every build-preserving join on the GPU path.

**A residency defect can be invisible at one layout.** q37 and q82 passed at
`one_partition_one_batch` and failed at `default` and `many_small_partitions`: only a
*streamed* probe accumulates, and a single-batch probe accumulates once and then finishes.
Anything that asserts a memory bound has to run at more than one partitioning, or it is
asserting about one shape of arrival.

**The layout that avoids the accumulation is the expensive one.** Coalescing a probe into a
single batch to skip the finish pass moves the cost into the batch: q3's peak goes from
6.2 MB at `default` to 69.3 MB, q7's from 53.3 MB to 367.7 MB. Streaming a probe versus
coalescing it is a residency trade and not a correctness one, and neither side is free —
the planner needs both numbers to choose.

**Build bytes are counted twice, and that is right.** #136 rebuilds and gathers against the
build side on every probe call, so the same bytes are in the resident total and in each
call's transient. `merged_scratch` returning `resident + …` rather than a delta is
deliberate, not double counting.

**Zero rows is not zero bytes, and a zero peak is a defect.** `merged_scratch` returns the
residency unchanged for an empty batch: an empty lane still owes a typed batch, which costs
schema and no rows. Every corpus run asserts `0 < peak <= budget` and `in_flight_bytes == 0`
at the end for the matching reason — an accountant that finished at zero peak observed
nothing, and a non-zero in-flight total means a batch was held and never released. Both
checks are free and both have caught real breakage.

**The measured-versus-modelled diagnostic has a hole exactly where it is needed.** Joins
return `no_scratch()` on both backends, so `CallStats.scratch_bytes` is 0 for them,
`Underestimate` never fires for the node whose model is least certain, and the 2 TB
mis-pricing above passed the diagnostic in silence — what caught it was the enforcer
tripping on a query that fits. On the GPU path joins are the first nodes that must be
instrumented through the RMM hooks, not the last.

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
| `GpuLoadParquet` | `CudfScan` | `peacock_executor_execute_scan_rowgroups(seq, row_groups…)` once per batch — the additive entry point, plumbing the existing `row_groups_override` parameter (`cpp/src/peacock/operators.h` ~L32) to the ABI |
| `GpuFilter` / `GpuProject` / `GpuAggregate` | same-kind node (`CudfFilter` / `CudfProject` / `CudfAggregate`) | generic map arm, one call per batch |
| `GpuSort` | `CudfSort` | map arm per batch; per-batch `fetch` for top-N |
| `GpuAccumulateBatchesAndSort` | `CudfSort` + `CudfSortPreservingMerge` | per-batch sort calls, then one merge-arm call at done |
| `GpuMergeSortedPartitions` | `CudfSortPreservingMerge` | one merge-arm call over all sorted handles, partition-major order |
| `GpuCoalesceAllBatches` | `CudfCoalescePartitions` | one collapse-arm call over the partition's batch handles |
| `GpuAggregateBatches` | `CudfCoalescePartitions` + `CudfAggregate` (merge aggregators, with `final` when it finalizes) | one concat + one aggregate call per compaction, and again at done |
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

**The limit lowering rule.** A per-batch `GpuLimit` call cannot be correct: the fb node's
skip/fetch are frozen per seq, so every batch would be truncated to the same bounds
(two batches → 2× the limit), and the right bound for the last batch is a runtime value
no frozen node can carry. Legacy never sees this because a legacy partition is one batch.

**A scan carrying a pushed-down limit plans one lane.** Where DataFusion can push the bound
all the way into the source it erases the limit node — `SELECT * FROM nation LIMIT 3` is a
`ParquetExec{limit: 3}` and nothing above it. DataFusion is safe because its scan is one
partition; our lane count is our own decision, so four lanes each honouring `limit=3` answer
with twelve rows. One lane makes the loader's own limit the whole answer. Same shape as the
small-table rule, and no new node.

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
has to outlive a call. Five costs, each with the smallest unfreeze that removes it. They
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
not exist, with the budget tier suffixed for execution goldens as in legacy
(`q1.bp-tp4-sized-mini.cpu.txt`). The label names the batching form rather than a
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

**Layout replaces the lane count.** `partitions=N` becomes the full `PartitionLayout` —
lane count, batch layout, key distribution, sort order — because in this mode those decide
what a parent may assume, and three of the four are invisible today. The per-partition
sub-line gains `out_batches` beside its row and byte counts.

**Every node that carries a `fetch` prints it.** A merge that turns 80 rows into 10 must say
so on its own line; today only `GpuSortExec` does, and the merge above it is silent (see
[the sort decomposition](#the-sort-decomposition)). Same for the aggregate's `aggs` and
`final` lists, the loader's `partition_groups`, and a limit's interval wherever it lives —
on the `GpuLimit` mid-plan, on the `GpuUnload` root-adjacent.

**Types move into the plan golden, not the execution golden.** The declared output schema
per node — `name:type` per column — is a plan fact, so it belongs beside the layout in
`<mode>.plans.txt` and is not repeated per query in `.cpu.txt`, which stays a record of what
execution produced. Printing it there is what makes the explicit casts legible (a
`Decimal128(38, 6)` in a `final` expression means nothing without the state column's
declared scale beside it) and it closes the second half of [#135](../archive/archived-tickets.md#t135): a
node emitting the right column count in the wrong order is today invisible until the root,
because per-node bytes are derived from the plan's schema on both engines and so agree by
construction.

Node names lose the `Exec` suffix, since these are not DataFusion nodes — `GpuLoadParquet`,
`GpuAggregateBatches`, `GpuEmitPartitions` — and after the T1 rename the legacy vocabulary
reads `Cudf*`, so a line from either family says which mode produced it without a caption.
What deliberately does not change: the indentation-as-tree shape, `output_rows` and
`output_bytes` (they are the CPU/GPU cross-check, and their meaning is unchanged), and
`batches_to_sorted_str` result comparison.

**Plan goldens** (5 modes: bp-tp1-single, bp-tp1-rowgroup, bp-tp4-single, bp-tp4-rowgroup,
bp-tp4-sized):
one file per mode holding all queries — `goldens/<bench>/<mode>.plans.txt` — because the
per-query files would be small and numerous. Every node renders its `PartitionLayout`:
count, batch layout, key distribution, sort order.

A parquet source renders its whole mapping as one nested structure on one line —
`partition_groups=[[[0,1],[2,3]],[[4],[5,6,7]]]`, partitions outermost, batches within
them, row groups innermost — which is verbatim the `Vec<Vec<Vec<u32>>>` the partitioner
returned. Not a partition count beside a batch count: the two are not independent, and
every property worth reading off a source line is about which batch sits in which
partition — the balance bound, an oversized row group standing alone, a partition whose
batches all came from one file region.

**Estimates go in a `--- memory ---` section per query, not on the node line**, as the legacy
`.plan.txt` already does. They churn where plan shapes do not — an estimator change, then
#19's statistics, then #147's refinement in flight — so on the node line every such change
rewrites every line and a reader cannot tell a shape change from a number. In their own
section the tree stays byte-identical and the diff says which it was. A section rather than a
sibling file because nothing consumes the memory data on its own, and cross-referencing two
files to ask whether a node's estimate suits its layout is worse than scrolling.

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

**Widget**: two new tables (TPC-H, TPC-DS) repeating the existing structure — plan (five
cells, one per mode), CPU (five), GPU (five) — fed from the new CSV columns. Window
queries render plan ✗ (#143). Peacock cost, DuckDB cost and ratio columns mirror the
legacy ones; when not all five modes are enabled, cost uses the last mode in the sequence
of five where CPU execution is enabled.

# Implementation plan

Tasks in dependency order; each is one developer hand-off with its own proving tests.
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

**T9 — additive ABI.** The three approved symbols in `gpu_executor.cpp` + `peacock_gpu.h`,
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

**T10 — Exec executors, CPU and GPU.** Filter, project, per-batch sort, aggregate
(partial/single), unload (`GpuBatch → CpuBatch`, honouring the row range). Reuse legacy operator code by extracting
helpers — never by calling into strip/wrapper machinery. Per executor: CPU vs
hand-crafted oracle, GPU vs CPU, empty-batch cases, `CallStats` model ≥ measured on CPU.
CPU and GPU tests in separate targets so CI hosts split them.

Also here: **`PlaceholderRowExec` is a source that emits its constant rows**, not a shape to
refuse ([#158](../tickets.md#t158)). DataFusion's `AggregateStatistics` rule answers an
unfiltered `count(*)` from parquet metadata and leaves this node holding the result, so the
plan carries the answer and the executor has only to produce it — one lane, one batch, the
rows the node already holds. The translation layer maps it like any other source; the work
is the executor, which is why it lands in this task rather than in T4.

**T11 — accumulators.** `GpuCoalesceAllBatches`, `GpuAggregateBatches` (merge-only and
finalizing),
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

**T13 — drivers and enforcer.** Both drivers over a mock `Backend` impl — the third
instantiation, alongside CPU and GPU; `batch_partitioned_driver` tested against a mocked
single-partition driver; round-robin determinism cases; [the test
surface](#the-test-surface) in full, with mock operators, each case asserting pull counts,
queue bounds and batch release rather than returned rows. Plus the accounting formula with
pre/post checks, enforcer trip ⇒ clean query failure, and FFI-error ⇒ query-fatal
semantics.

**T14 — recipe-plan serialization and GPU integration.** Re-check `avg`'s finalize casts
here against a real GPU result: the denominator goes to `Decimal128(p, 0)` and the numerator
to the declared type so cuDF's own divide scale lands where DataFusion declared it, and no
CPU-side test can prove that arithmetic. The GpuNode → fb-seq mapping
table implemented and unit-tested (expected seq sets and call patterns per node kind);
driver-side stats folding across calls into `NodeMemoryStats`; first end-to-end queries
on shad-gpu (scan → filter → aggregate; a join; a sort+limit), GPU vs CPU.

**T15 — corpus shapes the benchmarks do not have.** The corpus is numeric-aggregate
heavy, and this mode's risk sits in what it never sees: an audit of every `.cpu.txt` finds
zero `OFFSET`s, zero `min`/`max` over a non-numeric column, and no query at all for several
shapes the tickets already describe as unexercised. Add a small set of hand-written queries
over the **existing** tables — no new dataset, no new scale factor — each justified by
naming the feature it reaches and the ticket or code path that cares:

| Query shape | Why it is worth a query |
|---|---|
| `ORDER BY … LIMIT n OFFSET m` | every `GpuGlobalLimitExec` in the corpus has `skip=0`, so the offset half of both lowerings — the released prefix, the two straddling slices, and a pure offset that never satisfies — runs only in synthetic tests |
| `min`/`max` over a string or date column | zero uses today; the merge is a string reduce, and nothing exercises one |
| a join on a nullable key | [#59](../tickets.md#t59) and [#80](../tickets.md#t80) both record that no corpus query has one, and this mode adds [#137](../tickets.md#t137)'s skew on top — every all-null key lands in one lane |
| a shuffle keyed on a decimal | [#95](../tickets.md#t95)'s kernel throws on decimal keys; the batched mode shuffles more often than legacy, so this stops being latent |
| two `DISTINCT` arguments over different expressions | [#144](../tickets.md#t144)'s plan-time refusal has nothing to refuse today |
| a wide `SELECT DISTINCT` | dedup whose state is the whole row rather than a few keys — the compaction threshold's worst case, where nothing merges and the doubling has to earn its keep |
| `LIKE` with a non-prefix pattern | 21 `LIKE`s exist and all are simple; the non-prefix form is a different cuDF path |

Each query lands in `testdata/{tpch,tpcds}-queries/` with registry rows and goldens through
the existing generators. Keep the set small: every query multiplies across enabled modes and
tiers, which is why the corpus grew the way it did.

**T16 — rollout.** New macros `cpu_batch_partitioned_result_test` /
`gpu_batch_partitioned_test`; the node renderer per [Node display](#node-display)
(`name@ordinal` references, layout in place of the lane count, `fetch` and the aggregate
lists wherever carried, schema in the plan golden only); `.cpu.txt`/`.cost.txt` wiring incl.
`cost_model.conf` entries for the new node names; `batch-info.cpu.txt` for ~10 queries;
registry columns +
inventory tests; `pipeline.yml` steps (satisfying `test_ci_coverage`); widget tables with
the cost-column rule and the #143 plan ✗ cells. Then query-by-query enablement across the
corpus, tickets filed per newly discovered blocker, as with the legacy rollout.

**T17 — per-node benchmarks for the new mode.** Port `peacock_gpu_benchmarks` to the
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
