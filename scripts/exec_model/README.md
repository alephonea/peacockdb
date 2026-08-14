# batch-partitioned execution model — Python prototype

An emulation of the execution model in
[`llm-wiki/tasks/batch_partitioned_executor.md`](../../llm-wiki/tasks/batch_partitioned_executor.md),
built to settle the scheduling rule and the trait set before any Rust exists (task T0).
Not production code. The tests run in CI, in the `cost-report` job's cheap python tier.

**Two halves.** The scheduler — plan, drivers, traits — is stdlib only and uses mock
executors. The operators under `operators/` are pandas-backed, so `test_operators.py`,
`test_end_to_end.py`, `test_accounting.py` and `test_tpch.py` need pandas (and pyarrow for
the last) and **fail rather than skip** without them: a skipped operator suite reads
exactly like a passing one.

`test_tpch.py` runs simple hand-built plans over real TPC-H tables. It runs in CI's
**cpp-cpu** job, after that job generates sf1; every other file runs in cost-report, which
has no dataset. Locally it falls back to the committed `testdata/tpch.minimal` when sf1 was
never generated — for the four tables it uses the two are the same data, same schema and
same row counts, so the assertions hold either way. Every plan there and in
`test_end_to_end.py` runs under a real resident budget, so the enforcer is engaged rather
than dormant.

Every test file runs on its own with the stock python, no pytest:

```
python3 scripts/exec_model/tests/test_determinism.py
for f in scripts/exec_model/tests/test_*.py; do python3 "$f" || break; done
```

pytest collects the same files if it is available, and is the better loop when it is:

```
python3 -m pytest scripts/exec_model/tests -q
python3 -m pytest scripts/exec_model/tests/test_plan.py::test_height_is_distance_to_root -q
```

`tests/harness.py` is what makes both work: a `raises` context manager and a runner over
the module's `test_*` functions, which is all this suite ever used pytest for. Each test
file carries a four-line header that puts the repo root on `sys.path` and names its
package when it is run as a script, so the relative imports resolve either way.

## What is here

| File | Holds |
|---|---|
| `layout.py` | `NodeKind`, `KeyDistribution`, `SortOrder`, `BatchLayout`, `PartitionLayout` |
| `batch.py` | the `Batch` value type and `CallStats` |
| `executors.py` | `Executor` plus the seven executor traits and `LaneEvent` |
| `forwarder.py` | `BatchForwarder` and the merge / union / interleave mappings |
| `node.py` | `GpuNode`, `NodeExecutors`, `ExecutorBackends`, `BackendSelector` |
| `plan.py` | heights, left-to-right order, structural validation, which limit lowering applies |
| `accounting.py` | the resident formula, the cached-delta executor total, and the enforcer |
| `limit.py` | `RowInterval`, `RowRange`, and the per-batch decision behind the two lowerings |
| `runtime.py` | per-node queue state and one lane's view of its inputs |
| `batch_single_partition_driver.py` | one lane of one lane-scoped node |
| `batch_partitioned_driver.py` | the scheduler and everything cross-partition |
| `operators/frame.py` | the pandas batch, and the rules that keep pandas inside cuDF's vocabulary |
| `operators/expressions.py` | the expression IR — what `cudf::ast` accepts, nothing more |
| `operators/aggregates.py` | aggregate specs and the partial/final decomposition |
| `operators/source.py` | the loader and the row-group → (partition, batch) policy (T2) |
| `operators/exec_ops.py` | filter, project, sort, partial aggregate, limit, unload |
| `operators/accumulators.py` | coalesce-all, aggregate-batches, accumulate-and-sort, merge-sorted |
| `operators/partition_ops.py` | the hash scatter |
| `operators/joins.py` | the join capability matrix |
| `operators/nodes.py` | `GpuNode` implementations wiring the operators into plans |
| `operators/injection.py` | `LayoutInjector` — rewrite a plan's partitioning, batching and hash placement |

Traits are declarations only. The driver tests drive mocks (`tests/mocks.py`) because the
strategy under test is *which node runs when*; the operator tests drive the real thing.
Both go through the same two drivers, selected by `BackendSelector` — which is the point
of the selector existing.

### Keeping pandas inside cuDF's vocabulary

pandas is the backend because it is everywhere; cuDF is what the real executors call. They
disagree in ways that would let a prototype operator work and its C++ twin fail, so the
operators are written against the intersection and the divergences are named in
`operators/frame.py`. The five rules: no index (a `cudf::table` is columns and nothing
else); no `apply` or python callables; explicit null placement on every sort; explicit
null equality on every join; concatenate requires identical columns. Each has a test in
`test_operators.py` that fails if the pandas default is allowed to stand in.

## The strategy

Plans are trees — joins take exactly two children, forwarders any number — oriented so a
join's build side is always the left child. Each node carries a **height** (distance to
the root, root = 0) and an **order** (pre-order index, which in a tree is left-to-right
within a level). Both are computed once.

A node is **runnable** when any of its partitions can make progress: a source always can,
and any other node can once that lane's inputs hold a batch or are known to be finished.
Among all runnable nodes the driver takes the smallest height, breaking ties leftmost,
and then runs **every** lane of that node.

Min-height-first is what makes this a push model. The moment a node produces a batch its
parent is runnable at a strictly lower height, so the batch is carried up before anything
below produces again — it stops only at a batch accumulator, a partition accumulator, or
the sink. That is also the livelock argument: the one thing that can block a batch is a
join waiting on its other side, and putting the build side on the left removes that wait.
`run()` fails loudly if it ever ends with nothing runnable and a queue non-empty.

One rule sits on top of the height rule: **a join in its build phase holds back its whole
probe subtree**. Until `set_build` has run the join cannot consume a probe batch, so
without the hold the probe side runs anyway and piles up in a queue nothing will drain.
The hold is transitive over every edge on the path to the root — blocking only the join's
direct child would move the pile one node down rather than remove it. It cannot deadlock:
plans are trees, so a join's build subtree is disjoint from its probe subtree and is never
held by this rule, and completing the build is what lifts the hold. Nested joins resolve
outermost-first for the same reason.

The driver split follows the spec: `batch_partitioned_driver` owns the tree, the queues,
the schedule and the three cross-lane categories, and delegates each lane-scoped call to
`batch_single_partition_driver`. What changed against the spec's formulation is the
*unit*: a chunk is one node's lane rather than a chain of them, because min-height
selection walks a batch up a chain node by node on its own.

## Layout injection

A query's answer is a function of its rows, not of how they were divided. `LayoutInjector`
takes that seriously: give it a plan and it hands back an equivalent one whose lanes, row
groups, batch sizes and hash placement are whatever preset you name — one lane and one
batch, a few of each, many small lanes, lanes with nothing in them, re-cut batch
boundaries — with sources injecting zero-row batches at a given probability, and with
`GpuEmitPartitions` placing rows by a hash that ranges from well spread to
everything-in-one-lane. `test_tpch.py` writes each plan once and runs it at every shape.

Rebuilding, not editing: a node's partitioning is baked into a closure at build time, so
every builder in `nodes.py` records its call and a rewrite re-runs it. Two rules keep the
rewrite honest. A join may only be re-partitioned when both its sides are hash-partitioned
on the join keys — otherwise its lane count is load-bearing and splitting it would join
matching slices and silently return too few rows. And every placement is a pure function
of the key columns: a shuffle's contract is co-location and nothing above it may depend on
how evenly the lanes were loaded, so all-to-one-lane is a legal hash and a plan that only
works under a well-spread one is broken rather than unlucky.

## The limit

`start..limit` is decided by where it sits, because a per-batch call with frozen skip/fetch
cannot be correct — two batches would yield twice the limit.

Feeding only the sink it is **not a node at all**: `skip`/`fetch` are `GpuUnload`'s, which
is where they belong, since a limit over a stream about to leave the device is a statement
about which rows are worth moving across the boundary. The driver counts rows **across
lanes** — an unload executor is per lane, so only the driver can — and per batch either
releases the handle without a call, narrows the call to a row range, or passes it whole.
Once `is_satisfied` holds, the node's whole subtree stops being runnable, the same shape as
the join's build hold but never lifting; the run then ends with lanes not done and queues
non-empty, which is what the in-flight release is for.

Anywhere else it **stays a node**, over a one-partition input the planner guarantees
(`GpuMergePartitions` beneath it — an interval over N lanes names no rows) and any number
of batches. It streams, holding nothing: a batch outside the interval is released without
a call, one inside is forwarded untouched, and only the two that straddle its ends are
sliced, through `peacock_executor_slice_handle`, whose bounds are call arguments rather
than plan constants. Once satisfied it is held exactly as the sink's is — and marked done
as it is held, so its parent is not left waiting for a lane that has in fact finished.

`limit.py` holds `RowInterval` and `RowRange`; on the GPU the range is
`peacock_result_from_handle`'s new arguments, so a trimmed unload moves only the rows
wanted and allocates nothing. `test_limit.py` asserts on the **unload calls**, not on the
rows that come back — both look the same for a correct implementation, but only the calls
can tell a limit from a filter applied after the transfer, and that difference is the whole
feature.

## Findings

Numbered because the spec's Drivers section is to be rewritten from them. **Cite them by
title, not by number** — the numbers have already shifted once as findings merged, and
three of these are meant to be quoted into `llm-wiki/tasks/batch_partitioned_executor.md`.
F2, F3 and F5 are the three that bear on that rewrite.

F1. **The push behaviour falls out of the height rule alone.** No chain-walking logic is
   needed — `test_a_batch_is_carried_to_the_root_before_the_next_one_is_produced`.

F2. **Queues are self-bounding; the spec's cap-Q is unnecessary.** A producer's out-queue
   is drained by its parent before the producer can run again, because the parent's height
   is strictly lower. No node holds more than one batch per lane, on any shape — the check
   runs inside the `run()` helper of `test_partitioned_driver.py`, so every plan the suite
   exercises asserts it.
   The corollary matters for reading that check: with F3's hold in place the *scheduling*
   half can no longer go red, and no shape appears to exist that violates it. What the
   assertion still guards is **executor emission discipline** — no executor may return
   more than one batch per call per output lane. A partition accumulator emitting per lane
   event trips it, since the driver feeds it every input lane in one step and the node
   declares one output lane; [#138](../../llm-wiki/tickets.md)'s ranged merge emission is
   the same shape arriving from real code. `test_the_queue_bound_assertion_is_live`.

F3. **The bound needed one rule to become unconditional: the join hold.** A join in its
   build phase cannot consume a probe batch, and the planner's build-side
   `GpuCoalesceAllBatches` makes the build subtree one level *deeper* than the probe's —
   so min-height handed the probe every choice, and in a two-sided shuffle join the entire
   probe input (32 batches) went resident before the first `set_build`. Left-orienting the
   build removes the deadlock, not that. Holding the whole probe subtree until the build
   is set brings it back to 4 (the lane count):
   `test_probe_side_queues_stay_empty_until_the_build_is_set`,
   `test_a_join_in_its_build_phase_holds_back_its_probe_subtree`,
   `test_the_hold_is_transitive_over_the_whole_probe_subtree`,
   `test_the_hold_stays_on_while_any_join_lane_is_still_building`,
   `test_nested_joins_resolve_outermost_first_without_deadlocking`.
   Each of those is mutation-checked: dropping the hold, restricting it to the join's
   direct child, or weakening `_awaits_build` from any-lane to all-lanes each turns one of
   them red. The shape matters more than it looks — an earlier version of the transitivity
   test gave the probe the *deeper* subtree, so min-height preferred the build unprompted
   and the test passed with no hold at all.

F4. **Multi-child forwarders degenerate to left-first.** A source with nothing available is
   skipped rather than waited on (as the spec requires), and under min-height the leftmost
   child is always the one holding a batch — so a union or interleave drains its left
   child first instead of alternating. Deterministic, but not round-robin. Only
   `GpuMergePartitions` genuinely rotates, because it has one child and "run every
   partition" fills all its lanes in a single step —
   `test_a_multi_child_forwarder_skips_a_pending_source_instead_of_waiting`.

F5. **`Pending` does not exist in this model.** The `Batch` / `Pending` / `Exhausted`
   visit contract was a pull-model artefact: a puller has to be told "nothing yet",
   because it asked. Here runnability is a predicate over queue and done state evaluated
   *before* any call, so a node that has nothing to do is simply not chosen — there is no
   third outcome to return, and nothing to propagate through merge visits. `Exhausted`
   survives only as the per-lane `finished` flag the driver already keeps.

F6. **Executor constructors need their lane.** `ExecutorBackends` holds
   `Callable[[lane], Executor]`, not `Callable[[], Executor]` — a loader's lane cannot
   otherwise find its row groups in the partitioner's mapping. Cross-lane categories
   (`PartitionAccumulator`, `PartitionEmitter`) are built once per node and get `None`.

F7. **A join's build lane must deliver exactly one batch — including an empty one.**
   `GpuCoalesceAllBatches` therefore emits one batch even when it accumulated nothing;
   zero and two are both plan errors, and the driver says which.

F8. **Partition and lane validation is naturally central, not per node.** Arity, lane-count
   agreement, the emitter's single-lane input and the build side's `SingleBatch` are all
   whole-tree facts, so they live in `plan.py` — one owner.
   `validate_schemas_and_partitions()` is left to schema checks, which are out of T0 scope.

F9. **A cardinality estimate belongs on the join node, not in the model's signature.** A
   join's transient is sized by the *output* cardinality — matched rows × the combined
   width, build side replicated per match — which `scratch_bytes(n_rows, n_bytes)` cannot
   derive. It does not have to: the executor is constructed from the node, so an estimate
   the optimizer attaches at plan time reaches the model through `&self`, and the trait is
   unchanged. `GpuJoin` therefore carries a fan-out figure (output rows / probe rows, the
   ratio `CardinalityEstimator` already returns), constant 1.0 until
   [#19](../../llm-wiki/tickets.md). Corollary: **model ≥ measured is not an invariant.**
   The estimate can be wrong, so the model can come in under, and the enforcer is built for
   that — its contract is "fail cleanly when the accounted peak exceeds the budget". The
   comparison is recorded with its magnitude and never asserted away.

F10. **The executor total must be cached, not summed.** The spec says "Σ cached
   `resident_bytes()`" and the caching is not an optimization: summing live is what forces
   the accountant to hold a reference to every executor, which the Rust port cannot do
   while the driver holds them mutably. Refreshing one instance's delta per call removes
   the aliasing along with the cost.

## Not in this cut

The plan-time `estimated_max_resident_size` estimator, and the hand-built TPC-H / TPC-DS
plan corpus. Window functions are refused by the design (#143). The enforcer is here with
the accounting formula, and trips cleanly on a tight budget.
