# batch-partitioned execution model — Python prototype

An emulation of the execution model in
[`llm-wiki/tasks/batch_partitioned_executor.md`](../../llm-wiki/tasks/batch_partitioned_executor.md),
built to settle the scheduling rule and the trait set before any Rust exists (task T0).
Not production code. The tests run in CI, in the `cost-report` job's cheap python tier.

**The coordinator owns this prototype.** Changes here are made by the coordinator directly
rather than delegated, because what it settles is design — which is the same reason the
spec it models is coordinator-owned.

**Two halves.** The scheduler — plan, drivers, traits — is stdlib only and uses mock
executors. The operators under `operators/` are pandas-backed, so `test_operators.py`,
`test_end_to_end.py`, `test_accounting.py` and the TPC-H/TPC-DS files need pandas (and pyarrow for
the last) and **fail rather than skip** without them: a skipped operator suite reads
exactly like a passing one.

`test_tpch.py` runs simple hand-built plans over real TPC-H tables. It runs in CI's
**dataset-matrix** job, after that job generates sf1; every other file runs in cost-report, which
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
the module's `test_*` functions, which is all this suite ever used pytest for. Each test file carries a
five-line header that puts the repo root on `sys.path` and names its package when it is run
as a script, so the relative imports resolve either way.

## What is here

| File | Holds |
|---|---|
| `layout.py` | `NodeKind`, `KeyDistribution`, `SortOrder`, `BatchLayout`, `PartitionLayout` |
| `errors.py` | `PlanError`, `DriverError`, `ResidentBudgetExceeded` — what each failure class means |
| `batch.py` | the `Batch` value type and `CallStats` |
| `executors.py` | `Executor` plus the seven executor traits and `LaneEvent` |
| `forwarder.py` | `BatchForwarder` and the merge / union / interleave mappings |
| `node.py` | `GpuNode`, `NodeExecutors`, `ExecutorBackends`, `BackendSelector` |
| `plan.py` | heights, left-to-right order, whole-tree structural validation, which limit lowering applies |
| `schema.py` | what a node declares about its columns — annotations, not types |
| `accounting.py` | `ResidentAccountant` — the resident formula, the cached-delta executor total, and the budget trip (`ResidentBudgetExceeded`) |
| `limit.py` | `RowInterval`, `RowRange`, and the per-batch decision behind the two lowerings |
| `runtime.py` | per-node queue state and one lane's view of its inputs |
| `batch_single_partition_driver.py` | one lane of one lane-scoped node |
| `batch_partitioned_driver.py` | the scheduler and everything cross-partition |
| `operators/frame.py` | the pandas batch, and the rules that keep pandas inside cuDF's vocabulary |
| `operators/expressions.py` | the expression IR — what `cudf::ast` accepts, nothing more |
| `operators/aggregates.py` | aggregate specs and the init / merge / finalize decomposition, and the registry that emits each one's finalize expressions |
| `operators/source.py` | the loader and the row-group → (partition, batch) policy (T2) |
| `operators/exec_ops.py` | filter, project, sort, partial aggregate, limit, unload |
| `operators/accumulators.py` | coalesce-all, aggregate-batches, accumulate-and-sort, merge-sorted |
| `operators/partition_ops.py` | the hash scatter |
| `operators/join_types.py` | `JoinType` in the fbs vocabulary, and the capability matrix as a function both join backends read |
| `operators/joins.py` | the pandas join backend — nine hash-join types, cross, nested loop |
| `operators/cudf_calls.py` | the cuDF calls `cpp/src/operators/join.cpp` makes, modelled: joins that return gather maps, `gather` with its out-of-bounds policy, `scatter`, `apply_boolean_mask` |
| `operators/recipe.py` | the FlatBuffers node structs, the handle registry with consume-on-use, and the C++ that reads them |
| `operators/recipe_join.py` | the second join backend: answers every call by emitting fb nodes and making `execute_node` calls |
| `operators/nodes.py` | `GpuNode` implementations wiring the operators into plans |
| `operators/validation.py` | the checks a node's `_validator` is composed from with `all_of` — layout expectations and the aggregate state chain. The method is abstract on `GpuNode` (`node.py`) and implemented once, on `PandasNode` (`operators/nodes.py`), which just runs that validator |
| `operators/injection.py` | `LayoutInjector` — rewrite a plan's partitioning, batching and hash placement |
| `plan_text.py` | rendering a plan as text, for the corpus plan goldens |
| `tests/corpus.py` | reading the generated datasets, the corpus budget, the layout runner, and the plan-golden check |
| `tests/plans_tpch*.py`, `tests/plans_tpcds*.py` | the corpus query lowerings — builders over a table provider, so a plan needs no data |
| `tests/plan_helpers.py` | the aggregate and sort sequences every lowering is built from |
| `tests/plans.py` | render or rewrite the plan goldens without executing anything |

Traits are declarations only. The driver tests drive mocks (`tests/mocks.py`) because the
strategy under test is *which node runs when*; the operator tests drive the real thing.
Both reach the same two drivers through `BackendSelector`, which is what the selector is
for.

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
probe subtree**, transitively to the root. F3 below has the reasoning and the evidence.

`batch_partitioned_driver` owns the tree, the queues, the schedule and the three cross-lane
categories, and delegates each lane-scoped call to `batch_single_partition_driver`. The unit
is one node's lane rather than a chain of them: min-height selection walks a batch up a
chain node by node on its own.

## The two join backends

Every join test runs twice, on backends that share no join code.

`operators/joins.py` joins with pandas, the way DataFusion joins with DataFusion.
`operators/recipe_join.py` answers each call the way the GPU will: it emits a **recipe
plan** — `Cudf*` nodes in the legacy vocabulary, addressed by seq — and makes
`execute_node(seq, handles)` calls against `operators/recipe.py`, whose registry consumes
handles exactly as `NodeSession` does and whose node implementations mirror
`cpp/src/operators/join.cpp` branch for branch over the primitives in
`operators/cudf_calls.py`. `RecipeJoinBackendSelector` picks it for joins and leaves every
other category on pandas.

`tests/test_join_capability.py` is where both are driven: one case per join mode against a
SQL oracle, on both backends, at every batching config and layout preset, plus the emitted
seq sequence, the copy counts, the null-key asymmetry, and two guards that read
`cpp/src/operators/join.cpp` and fail when it names a join type or calls a cuDF function
the model has never heard of. The model mirrors that file by hand and no import links them,
so those two are the only thing standing between a C++ change and silent drift.

The question it exists to answer is the mode's load-bearing one: can the **frozen** fbs and
C++ execute every join type with a probe side arriving in batches? Three answers came back,
and the spec's join tables carry all of them.

- **Yes, for every type**, with the lowering in that table: probe-local types are one node
  per batch, build-preserving types add the #136 finish sequence, and where the matrix
  refuses to stream, the single-batch fallback is precisely the legacy call.
- **At a cost the frozen surface makes unavoidable**: a handle is consumed by the call that
  reads it and nothing duplicates one, so a streamed probe copies the build side once per
  batch — [#152](../../llm-wiki/tickets.md#t152), and Left/Full copy the probe batch as
  well. The session counts every copy, so the figure is asserted rather than estimated.
- **Except one shape, which turned out to be a defect rather than a limit.** An outer join
  with a residual filter is wrong in the shipping C++ — the filter is applied after the
  outer gather, dropping the rows it was meant to preserve
  ([#153](../../llm-wiki/tickets.md#t153)). Latent, since no corpus query has that shape.

## The corpus

`tests/test_tpch_corpus.py` and `tests/test_tpcds.py` run real benchmark queries — the query
text lowered by hand into the mode's nodes, over **whole** sf1 tables with the spec's own
parameters. 22 TPC-H queries and 71 TPC-DS ones: every TPC-DS query the engine already runs
in `full_table` mode (`testdata/cost-registry.csv`, `ftc_tp1 = enabled`) except the seven
that need a window function, which the mode has no node for.

**Two oracles, on purpose.** TPC-H has a hand-written pandas equivalent per query: it states
what the query means in a second language, and catches a lowering that answered a
differently-shaped question. TPC-DS runs the query's **own text** through DuckDB over the
same parquet files: that catches a *reading* of the SQL, which a hand-written pair cannot,
since both halves would share one reading — the circularity #80 complains of. It also pins
the output column names, which come from the query's aliases and from nowhere else.
Seventy-one hand-written oracles would have been seventy-one new places to be wrong.

What a comparison asserts is what SQL determines: the rows as a **multiset**, and the ORDER
BY columns **positionally** (`corpus.matches_oracle`, shared by both benchmarks). Comparing
whole rows positionally makes a tie into a failure, which is how TPC-H q11 failed — two
German parts come to 223626.0 exactly, and which of them is printed first is not the
query's to say. The rest of the TPC-H corpus is still compared positionally throughout,
because its sort keys are unique in this data; a query that starts failing there has found
a tie, and the fix is to name its `order_by`, not to re-sort the oracle until they agree.

**Manual dispatch only** (`.github/workflows/exec-model-corpus.yml`): six million lineitem
rows through a pandas operator chain is minutes, not seconds. The queries are independent,
so `PCK_SHARD=k/n` splits a file across n processes and `PCK_LAYOUT` picks one of the three
layouts each query runs at. `test_tpch.py`'s short plan-shape tests are a separate file and
still run on every push in dataset-matrix.

**`PCK_BACKEND=recipe`** re-runs the whole corpus with every join going through the
FlatBuffers emulation instead of pandas — the same `Cudf*` node sequence the C++ reads off
the wire, interpreted by a python model of the cuDF calls. That is the claim the join
capability work made, checked against real queries rather than against a synthetic matrix.

**Plans are separate from tests.** A plan is a function of the schemas, so the builders in
`plans_tpch*.py` and `plans_tpcds*.py` take a table provider and never read a row; `plans.py`
renders every plan from parquet footers in a fraction of a second and writes
`tpch.plans.txt` / `tpcds.plans.txt`, which the corpus tests then check the default layout
against. Regenerating a plan golden does not cost a corpus run — only verifying it does.

The TPC-DS lowerings are grouped by what the lowering has to do rather than by number —
stars, bucketed reports, baskets, correlated subqueries, banded disjunctions, semi/anti/mark
joins, channel unions, year-over-year self-joins — and `plans_tpcds.py` is the registry that
names them all.

**Sampling was tried and abandoned.** Both benchmarks are written clustered by date, so a
row prefix is one quarter of 1992, a row-group sample is a set of date windows, and two
tables sampled independently join to nothing. Every one of those bit before whole tables
settled it; `corpus.py` records the reasoning.

## Layout injection

A query's answer is a function of its rows, not of how they were divided. `LayoutInjector`
takes that seriously: give it a plan and it hands back an equivalent one whose lanes, row
groups, batch sizes and hash placement are whatever preset you name — one lane and one
batch, a few of each, many small lanes, lanes with nothing in them, re-cut batch
boundaries — with sources injecting zero-row batches at a given probability, and with
`GpuEmitPartitions` placing rows by a hash that ranges from well spread to
everything-in-one-lane. Each corpus plan is written once and run at every shape.

Rebuilding, not editing: a node's partitioning is baked into a closure at build time, so
every builder in `nodes.py` records its call and a rewrite re-runs it. Two rules keep the
rewrite honest. A join may only be re-partitioned when both its sides are hash-partitioned
on the join keys — otherwise its lane count is load-bearing and splitting it would join
matching slices and silently return too few rows. And every placement is a pure function
of the key columns: a shuffle's contract is co-location and nothing above it may depend on
how evenly the lanes were loaded, so all-to-one-lane is a legal hash and a plan that only
works under a well-spread one is broken rather than unlucky.

## The limit

Both lowerings are implemented, as the spec's limit rule states them.

Feeding only the sink it is **not a node at all**: `skip`/`fetch` are `GpuUnload`'s. The
driver counts rows **across lanes** — an unload executor is per lane, so only the driver
can — and per batch either releases the handle without a call, narrows the call to a row
range, or passes it whole. Once `_is_satisfied` holds, the node's whole subtree stops being
runnable: the join hold's shape, but never lifting, so the run ends with lanes not done and
queues non-empty and the in-flight release is what cleans up.

Anywhere else it **stays a node**, over the one-partition input the planner guarantees and
any number of batches, streaming and holding nothing: only the two batches straddling
`start..limit` are sliced, through `peacock_executor_slice_handle`, whose bounds are call
arguments rather than plan constants. Once satisfied it is held exactly as
the sink's is — and marked done as it is held, so its parent is not left waiting for a lane
that has in fact finished.

`limit.py` holds `RowInterval` and `RowRange`; on the GPU the range is
`peacock_result_from_handle`'s new arguments, so a trimmed unload moves only the rows
wanted and allocates nothing. `test_limit.py` asserts on the **unload calls**, not on the
rows that come back — both look the same for a correct implementation, but only the calls
can tell a limit from a filter applied after the transfer, and that difference is the whole
feature.

## Findings

The spec's Drivers section was rewritten from these. **Cite them by title, not by
number** — the numbers have already shifted once as findings merged. F2, F3 and F5 are the
three that rewrite carried.

F1. **The push behaviour falls out of the height rule alone.** No chain-walking logic is
   needed — `test_a_batch_is_carried_to_the_root_before_the_next_one_is_produced`.

F2. **Queues are self-bounding; the draft's cap-Q is unnecessary.** A producer's out-queue
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
   `GpuMergePartitions` genuinely rotates, because its one child feeds all of its input
   lanes and "run every partition" fills them in a single step —
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

F8. **Validation splits by what the rule is about, not by convenience.** Whole-tree facts —
   arity, lane-count agreement, the emitter's single-lane input, the build side's
   `SingleBatch` — live in `plan.py`, one owner. What a node needs *of its children* lives
   in its `_validator`, composed from the checks in `operators/validation.py` and run by
   `validate_schemas_and_partitions()`, because only there can the message name the fix: "the planner inserts GpuMergePartitions below it" rather
   than "this category is 1:1 per lane". That half covers hash distribution, sortedness,
   batch layout, and the aggregate state chain — a merge checks that the state it reads is
   the state its own partial declared, same aggregate and same positions, which is
   [#135](../../llm-wiki/tickets.md#t135)'s class of defect made checkable.
   Full column types stay out of the prototype (`schema.py` carries annotations only);
   they are the real implementation's T7.

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

F11. **A residual filter does not by itself force a single-batch probe.** The draft matrix
   refused a streamed probe to any filtered join. What actually forbids it is the *finish
   pass*: it sees accumulated keys, and a keys-only table cannot evaluate a predicate over
   both sides. A filtered Inner — or any probe-local type — streams, because each output
   row is decided by (the whole build, this batch) and the filter is part of that decision.
   The matrix in the spec was narrowed to say so.
   `test_a_residual_filter_rides_the_join_on_both_backends`.

F12. **Modelling the wire found a defect in the shipping engine, not in the model.** The
   recipe backend reproduces `execute_hash_join` branch for branch, so where it disagreed
   with the SQL oracle the disagreement belonged to the C++: an outer join's residual
   filter is applied after the outer gather and drops the rows the outer join exists to
   keep ([#153](../../llm-wiki/tickets.md#t153)). Reproducing a code path faithfully is
   worth more than implementing it correctly — a correct model would have hidden this.
   `test_an_outer_join_with_a_residual_filter_is_refused_not_answered_wrongly`.

## The prototype is a model, not a specification

Where the Rust implementation and this prototype disagree and the prototype is the one that
is wrong, the Rust is right and the divergence is the outcome, not a drift to reconcile.
This happened first with row-group chunking: the prototype takes groups until a lane's share
is reached and overshoots by a whole group, while `ParquetBatchPartitioner` stops where
stopping is closer, which is what holds the balance bound. Do not file that class as a
defect against the engine, and do not change the engine to match a model it has outgrown.

## Not in this cut

The plan-time `estimated_max_resident_size` estimator, which T6 derives in Rust directly
rather than here. Window functions are refused by the design (#143). The enforcer is here
with the accounting formula, and trips cleanly on a tight budget.
