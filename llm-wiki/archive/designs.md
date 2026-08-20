# archived design artifacts

Findings from work that has landed, kept because the numbers behind them are not
reproducible from the code that shipped: a prototype's measurements, a comparison whose
inputs are gone, a case that decided a shape. The rules they produced live where they bind —
in `architecture.md`, in a task spec, or in the code — and this page is what those rules were
argued from. Newest first.

## What the batch-partitioned corpus rollout measured (T0, 2026-08)

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

The rules this left binding on the Rust implementation are in the Memory accounting section of
[`tasks/batch_partitioned_executor.md`](../tasks/batch_partitioned_executor.md); what is kept
here is the measurement, the case and the numbers.
