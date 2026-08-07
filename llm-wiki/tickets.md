# peacockdb tickets

Migrated from GitHub issues on 2026-07-31; GitHub issues are closed and this file is the
registry. The number is the permanent ticket ID — each ticket carries an `<a id="tNN">`
anchor that the cost widget links to. Device labels are `tp<N>-<tier>` (micro=100MiB,
mini=2GiB, standard=12GiB).

A ticket carries a **Priority** line only when it is not medium; medium is the default.
New tickets take the next free number (currently 144). Finished and lapsed tickets move to
`llm-wiki/archive/archived-tickets.md` (Done / Stale) — numbers are never reused, so an old
reference still resolves there.

## Contents

| Section | Open | Tickets |
|---|--:|---|
| [Critical correctness](#critical-correctness) | 14 | #103 #80 #59 #46 #47 #60 #121 #122 #123 #118 #119 #120 #117 #41 |
| [Blockers for disabled coverage](#blockers-for-disabled-coverage) | 15 | #97 #23 #32 #65 #62 #91 #95 #57 #45 #63 #56 #55 #115 #96 #143 |
| [Performance / architecture](#performance--architecture) | 15 | #19 #16 #20 #71 #101 #73 #110 #75 #136 #137 #138 #139 #140 #141 #142 |
| [Infrastructure / process](#infrastructure--process) | 16 | #113 #114 #116 #126 #132 #131 #130 #129 #128 #127 #124 #125 #13 #94 #69 #49 |

## Critical correctness

<a id="t103"></a>
### #103 — GPU SIGSEGV: shuffle_stddev tp8-standard (Welford N-way merge)
`gpu_partitioned_tpch_sf1_shuffle_stddev_partitioned_tp8_standard` segfaults (139) or fails as a contained
`vector::reserve` Err on shad-gpu. Nondeterministic; reproduces at 12 GiB and 120 GiB, so
not budget-related. Inside `execute_instrumented`, upstream of golden compares; tp1 never
crashes. Suspect the 8-way Welford M2 merge (`cpp/src/operators/aggregate.cpp`).
**Test quarantined** 2026-07-31 (commented out in `test_gpu_partitioned.rs`) — it was the only
coverage of the 8-way M2 merge; tp8 goldens kept. Has already caused one wrongly
diagnosed "regression" + rollback: check this ticket before blaming your change.

<a id="t80"></a>
### #80 — Anti-join NOT IN three-valued logic + independent DuckDB result oracle
`NOT IN` with any NULL in the build side must yield the empty set; neither
`null_equality::EQUAL` nor `UNEQUAL` implements that, so ANTI/mark joins stay hardcoded
EQUAL (`cpp/src/operators/join.cpp`). Needs a planner/serializer flag distinguishing
NOT IN from NOT EXISTS, a nullable-anti-key test (no corpus query exercises it), and a
DuckDB final-result oracle — today's validation is circular (goldens vs DataFusion, GPU
vs CPU). Semi half done (q33; semi honors per-join `null_equals_null`).

<a id="t59"></a>
### #59 — Nullable-key semantics for semi/anti/mark joins
Anti/mark keep `null_equality::EQUAL` deliberately; a blind UNEQUAL flip is wrong for
`NOT IN`. Latent — no enabled query has a nullable anti/mark key. Wants a dedicated
analysis plus expr/join goldens covering nullable IN / NOT IN / EXISTS before defaults
change. Anti remainder overlaps #80.

<a id="t46"></a>
### #46 — q61 GPU: 'promotions' sum subtree returns the wrong value
Cross join is correct; `promotions` sum returns 2855378.83 vs CPU 2894907.87 — an
upstream filtered-sum / projection / aggregate bug. Bisect that subtree node-by-node,
CPU vs GPU. q61 full_table_gpu stays off.

<a id="t47"></a>
### #47 — q77 GPU returns 40 rows vs CPU 45
Cross join correct; an upstream outer-join/aggregate branch drops 5 rows. Bisect
per-node row counts. q77 full_table_gpu stays off.

<a id="t60"></a>
### #60 — q78 GPU diverges in anti-join + top-N; possibly memory-borderline
3-CTE anti-join (`LEFT JOIN … IS NULL`) + multi-key DESC LIMIT 100. `round` is proven
fine (q54). The isolated diff harness segfaults under GPU memory pressure on a shared
H200 — re-check on a free GPU to separate wrong-result from memory-induced.

<a id="t121"></a>
### #121 — Multi-GPU q3 frees GPU-p memory on the calling thread
`cpp/tests/gpu/test_multi_gpu_tpch.cpp` (~L634): `shuffled` (from `hash_shuffle`) is a
local of the `execute` lambda, so it is destroyed on the **calling** thread. Each
`shuffled[p]` for p≠0 is a `cudf::table` from GPU p's RMM pool, so this frees device-p
memory while device 0 is current — the exact worker-per-GPU destruction rule this file
follows everywhere else (every other buffer is explicitly released on its worker). Pool
deallocation is stream-ordered per device, so this is the class of mistake that shows up
later as a corrupted pool or a teardown crash, not immediately. Release each
`shuffled[p]` on `pool[p]` like the neighbouring `local_top` block.

<a id="t122"></a>
### #122 — Multi-GPU q6 host merge doesn't check partial scales agree
`cpp/tests/gpu/test_multi_gpu_tpch.cpp` (~L198): the host `__int128` sum overwrites
`scale` with each partial's scale without verifying they match. All partials share a
scale today, so the result is correct — but a future divergence would silently produce a
wrong sum instead of failing. Assert equality.

<a id="t123"></a>
### #123 — Unused static helper in test_plan_executor.cpp
`cpp/tests/gpu/test_plan_executor.cpp:53`: `make_float64_literal(...)` is defined and
never called — a `-Wunused-function` warning waiting on the next warning-level bump.
Remove it or use it.

<a id="t118"></a>
### #118 — SortPreservingMerge concat fallback ignores fetch (LIMIT dropped)
`cpp/src/node_session.cpp` (~L208): the k-way-merge branch applies `spm->fetch()` after
merging, but the fallback branch — taken when the SPM has no sort keys **or only one
input partition** — is a plain `cudf::concatenate` with no fetch applied. A
single-partition SortPreservingMerge carrying a fetch therefore returns **all** rows
instead of the top-N, i.e. a silently dropped LIMIT. Apply the same slice in both
branches. Found during the comment audit; not yet reproduced against a corpus query
(most SPMs arrive multi-partition), so severity depends on whether any enabled plan hits
the single-partition path.

<a id="t119"></a>
### #119 — Unchecked cudaMemcpy in peacock_spark_partition_ids
`cpp/src/gpu_executor.cpp` (~L250): the device→host `cudaMemcpy` return value is
discarded, so a failed copy still returns success with `out_pids` holding garbage. Check
it and surface the error.

<a id="t120"></a>
### #120 — peacock_spark_partition_ids' documented error path is unreachable
`cpp/include/peacock_gpu.h` documents retrieving the failure message via
`peacock_last_error(NULL)`, but `gpu_executor.cpp` returns `""` for a null executor and
the implementation only `fprintf`s to stderr — the message cannot be obtained through the
documented API. Either store it where `peacock_last_error(NULL)` can return it, or fix
the doc.

<a id="t117"></a>
### #117 — register_tables_for silently mis-handles non-parquet files
`peacockdb-core/src/lib.rs` (~L87): the extension check `if path.extension() != Some("parquet") { () }`
is a no-op, so non-parquet dir entries are not skipped; and `read_table` never returns
`Err` (failures panic), making the `else { continue }` dead. A stray non-parquet file in
a data dir panics instead of being skipped. Found during the comment audit.

<a id="t41"></a>
### #41 — Standing test for GpuUnion branch-type normalization cast
Nothing is unimplemented here, which is why the title says *test*. The cast exists and works:
`execute_union` retypes every branch column to the union's declared output type before
`cudf::concatenate`, because branches are planned independently and one column can land a
different cuDF type per branch (`cpp/src/operators/union.cpp` ~L49). Without it concatenate
throws on mismatched types.

What is open is that the coverage is *incidental*, and unverified. The implementation comment names the case it
was written for — tpcds q5, which pairs a decimal measure against a `cast(0 AS decimal(7,2))`
literal that materializes as FLOAT64, plus cuDF's SUM drifting fixed_point scale per branch —
and tpcds q5 does run `full_table_tp1_standard` GPU at `golden_exact` (tpch q5 is irrelevant:
its plan has no `GpuUnionExec` at all). So the branch is exercised today, and breaking the
cast would go red. But it is red by luck: nothing pins that q5 keeps a union with mismatched
branch types, and if that shape changes the cast becomes untested silently — the exact
situation this ticket was filed to end.

Fix: a focused gtest that builds a two-branch union with FLOAT64 against DECIMAL128 and
asserts the concatenate succeeds with the declared type. Cheap, and independent of any corpus
query.

## Blockers for disabled coverage

<a id="t97"></a>
### #97 — Semi/anti/outer joins at real-8-way
The widest tp8 blocker (~25 registry rows). Per-partition semi/anti/outer landed
(semi_join / anti_join / left_join green at tp8-standard). Remaining: (a) broadcast /
CollectLeft semi-anti-outer need a cross-probe-partition build-match bitmap reduction;
(b) the NOT-IN global "any build partition holds a NULL key" check (#80). Carriers:
tpch q4/q16/q18/q20/q21/q22 plus the tpcds outer/semi set.

<a id="t23"></a>
### #23 — Upgrade DataFusion 45→46+ to unblock q27/q70/q72/q86
Four TPC-DS queries fail to physical-plan on DataFusion 45 (`plan_status=fail`): q27
SanityCheckPlan vs ROLLUP SortPreservingMerge ordering; q70/q86 `GROUPING()` aggregate
not planned; q72 `Date32 + Int64` coercion. Whole rows dead until the upgrade. See #114.

<a id="t32"></a>
### #32 — GPU window functions / PARTITION BY
12 registry rows. Whole-partition aggregate windows work
(`cpp/src/operators/window.cpp`; q12/q51/q53/q63/q89 green at tp1). Remaining:
`rank()`/`dense_rank()` (`StandardWindowExpr`) for q36/q44/q47/q57/q67; q49 secondary
blocker; q20 LIMIT-boundary NULL-ordering tiebreak; q98 OOM on a shared GPU (now
`oracle` mode at tp1). Windows also gate the tp8 rollout for those rows.

<a id="t65"></a>
### #65 — __grouping_id encoding doesn't match DataFusion's GROUPING()
Grouping-set expansion (`cpp/src/operators/aggregate.cpp`) emits a gid that is
distinct-per-set but not DataFusion's positional bitmask. Safe while no enabled query
projects or sorts `GROUPING()`; must be fixed before one does (q70/q86 after #23).
9 rollup rows carry this ticket.

<a id="t62"></a>
### #62 — count(DISTINCT) ignores the DISTINCT flag in GpuAggregate
`cpp/src/operators/aggregate.cpp` ignores `AggregateFuncNode.distinct`; a guard now
throws rather than silently miscomputing. Standalone count-distinct works via
DataFusion's `SingleDistinctToGroupBy` rewrite (q16×2/q94/q95 green); the flag survives
only for mixed distinct + non-distinct, blocking q28. Fix: map count+distinct → cuDF
`nunique` (`null_policy::EXCLUDE`) in the grouped and global paths without regressing
the rewrite queries.

<a id="t91"></a>
### #91 — Memory-aware GPU hash-repartition + resident-OOM in the partitioned executor
Real 8-way is only validated at generous budgets. (1) Repartition is concat-first
(`GpuCoalescePartitions` full concatenate, then scatter), spiking peak memory and
defeating the point of partitioning. (2) The resident-OOM enforcer
(`executors/stream.rs`) hangs off the streaming driver only, not ported to the CPU backend.
Payoff: a genuine tp8-mini (2 GiB real-8-way) device, inexpressible today.

<a id="t95"></a>
### #95 — Decimal partition keys for real-8-way murmur3
murmur3 covers int/date/timestamp/composite/null; decimal deferred (float indefinitely).
Needed once the tp8 rollout reaches decimal-keyed repartitions (tpch q18 `o_totalprice`,
q22 `c_acctbal`, tpcds `i_current_price`). Dispatch by *logical* precision (≤18 → low 8 LE
bytes of int128; >18 → raw 16B LE) and thread precision through the partition FFI. Until
then `spark_hash_partition.cu` throws a loud "decimal partition key unsupported".

<a id="t57"></a>
### #57 — Value-form CASE produces wrong results on the GPU column path
`CASE x WHEN v THEN …` in `build_column_case` (`cpp/src/expr.cpp`) returned all-0/null
(the per-branch EQUAL condition comes back all-null through `copy_if_else`); reverted to
a throw guard. Sole corpus user is q39 — this is what keeps q39 off the GPU. Fix plus a
direct gtest, then enable q39 (stddev itself works; q17 passes).

<a id="t45"></a>
### #45 — q24 GpuHashJoin: 'Unary cast type must be fixed-width'
A join-key cast targets string; cuDF's unary cast rejects non-fixed-width. Diagnose the
emitted cast and handle string keys by hashing rather than casting
(`cpp/src/operators/join.cpp`).

<a id="t63"></a>
### #63 — q9 GpuProject copy_if_else size mismatch (CASE over scalar subqueries)
A top-level CASE over ~15 scalar-subquery comparisons fails in cuDF `copy_if_else`
(1-row scalar branch vs other-sized branch). Needs scalar-subquery branches broadcast to
the row count, or a different CASE lowering (`cpp/src/expr.cpp`).

<a id="t56"></a>
### #56 — q2: CASE-over-string-equality inside a partial-phase sum
Partial GpuAggregate builds an AST for `sum(CASE WHEN <string equality> …)` → cuDF
binaryop "Unsupported operator" (string comparand in the aggregate AST path). Support it
or lower it before the aggregate (`cpp/src/operators/aggregate.cpp`).

<a id="t55"></a>
### #55 — q66: two-phase decimal aggregate ignores the partial-phase divisor cast
DataFusion evaluates `sum(decimal/int)`'s division (divisor cast to Decimal128) only in
the Partial phase; GpuAggregate re-evaluates against Final-phase inputs → cast failure.
Honor the partial-phase operand cast / state schema.

<a id="t115"></a>
### #115 — q38/q76/q87 set-op & union-count divergences — retriage
The remainder of the old bucket-I set: q38 (INTERSECT ×3), q87 (EXCEPT ×2 — DISTINCT
sets lowered to semi/anti joins with `null_equals_null=true` feeding count(*)), q76
(UNION ALL + IS NULL filters + grouped count(*)). Diverged historically; the fourth
member (q75) now passes and semi joins now honor per-join null-equality — so first rerun
on H200, enable what passes, root-cause the rest. Anti/EXCEPT null handling overlaps #80.

<a id="t96"></a>
### #96 — GPU real-8-way per-partition JOIN execution
Largely landed: the CPU oracle and the GPU map arm both run partitioned joins
per-partition (child0[p] ⋈ child1[p]); q17 green at tp8-standard plus
q3/q5/q7/q8/q9/q12/q13/q19. Before closing, confirm nothing remains beyond the
broadcast/CollectLeft and non-inner surfaces now tracked in #97.

<a id="t143"></a>
### #143 — window functions in batch-partitioned mode
The batch-partitioned planner refuses window queries at plan time (rendered as plan ✗ in
the new widget tables) — the one feature regression against legacy full_table coverage,
and a blocker for ever retiring the legacy modes. Direction: a window is a per-partition
op once the input is hash-partitioned on the PARTITION BY keys (the legacy invariant);
whole-partition aggregate windows need a single batch (coalesce-all first), while
`BoundedWindowAggExec`-class frames could stream as a `BatchAccumulator`. The
rank/dense_rank gaps of #32 carry over unchanged.

## Performance / architecture

<a id="t19"></a>
### #19 — Stats propagation: 55 TPC-DS hash joins blind to cardinality
`num_rows` doesn't survive descent through the `Gpu*`-wrapped tree (`statistics()` impls
in `gpu_rule.rs`; GpuScanExec not forwarding ParquetExec row counts), so JoinSelection
can't flip build/probe on 55 of 103 joins across 38 queries. One confirmed mis-order: q45
build 9.95× larger than probe. Foundation for all CBO work (#73, #20).

<a id="t16"></a>
### #16 — Dynamic / runtime filters: build-side keys → probe-side scan
Star-schema fact scans read 100% of rows while the joined dimension is filtered to ~30.
Build an IN-set / min-max (later Bloom) at hash-join build completion and feed the
probe-side GpuScan for row-group pruning and pre-filtering. Applies to 76/99 TPC-DS
queries; validate on q3/q19/q33. Best after #19.

<a id="t20"></a>
### #20 — Join enumeration: DPccp/DPhyp cost-based tree reshaping
DataFusion 45 has no join enumerator — trees come out in FROM-clause order, and ~70/99
TPC-DS queries have 4+ joins (q64 ≈ 18 tables). Implement DPccp (extend to DPhyp, IKKBZ
fallback beyond 14 tables) as a logical rule after PushDownFilter, cost = Σ intermediate
cardinality. Blocked by #19. Landing rewrites all plan goldens.

<a id="t71"></a>
### #71 — GPU scan: no predicate pushdown into the cuDF read
Partly addressed: stats-based row-group pruning exists (`gpu_rowgroup_prune.rs` → cuDF
`set_row_groups`, parity with ParquetExec). Remaining: serialize the predicate itself
into the cuDF `read_parquet` filter AST (page pruning / pre-filter during decode),
multi-file scans, dynamic ranges (#16). Cause of red widget ratios on selective queries.

<a id="t101"></a>
### #101 — No CSE / CTE materialization: identical subexpressions recomputed N times
**Priority: post-MVP**

DataFusion inlines every CTE reference and peacock re-scans each copy. Worst: tpcds q23
(CTEs ×5/×3/×2 → 6 scans), q4/q11 (`year_total` ×6/×4), q31; tpch q15 (`revenue0` ×2),
q2. Direction: CTE materialization or physical CSE; at minimum make the cost model aware.

<a id="t73"></a>
### #73 — Cost-based optimizer (CBO) umbrella
Move physical planning from static heuristics to cost-based, validated against the DuckDB
cost oracle and goldens. In scope: adaptive filter placement, join enumeration (#20),
stats (#19, the load-bearing prereq), runtime filters (#16).

<a id="t110"></a>
### #110 — Retire the all-at-once GPU executor
Remove `executors/all_at_once_gpu_executor.rs` → `peacock_execute` FFI (whole-plan, no
per-node stats), migrate or drop its 5 smoke tests plus the lifecycle test, and drop the
FFI symbols from `cpp/include/peacock_gpu.h` / `cpp/src/gpu_executor.cpp` (and
`execute_plan.cpp`). Its stated blocker — a common `Executor` trait — has landed, so this
is unblocked pending full_table/partitioned covering all needs.

<a id="t75"></a>
### #75 — Refactor duckdb_cost.py: separate cost formula from extraction
**Priority: low** — not started (867 lines, 29 top-level defs as of 2026-08-05), and it buys
readability, not behaviour.

~870 lines where the ~80-line cost formula hides inside JSON parsing, predicate parsing
and row-group pruning. Shape: preprocessor → flat per-node intermediate
(op/rows_read/bytes_read/out_rows/out_bytes/breaker) → one-line cost. Pure refactor:
`.duckdb_cost.txt` numbers must not move.

<a id="t136"></a>
### #136 — batch-partitioned GpuJoin: build-side match tracking when the probe side streams

The batch-partitioned executor design streams a join's probe side in batches. Inner joins
compose per-batch, and right-outer emits its unmatched probe rows batch-locally — but
left-outer/full/semi/anti/mark need "which build rows matched at least once across all
probe batches", and that information never crosses the current ABI:
`peacock_executor_execute_node` returns only the joined table plus rows/varlen stats
(`cpp/include/peacock_gpu.h`), and every call rebuilds the join from scratch — there is no
persistent `cudf::hash_join` anywhere in `cpp/src/operators/join.cpp`.

No-ABI-change plan (v1): accumulate each probe batch's key columns only (a project node
plus the `GpuCoalescePartitions` concat arm — keys are small next to full rows), and at the
join's finish call run one `left_anti_join(build, accumulated_keys)` — semi form for
semi/mark — then null-pad the probe columns with a synthesized project. Correct for pure
equi-joins; the per-join `null_equals_null` flag applies to the finish join too, and the
#80 NOT IN caveat carries over unchanged. Not usable with a residual filter (a keys-only
input cannot evaluate it), so filtered semi/anti keep a single-batch probe side. Cost:
the accumulated keys stay resident, and the build side is built one extra time at finish.

ABI-change options if that cost bites: (a) return a build-side match bitmap per probe
call (new out-param, or a handle to a bool column) and OR the bitmaps in Rust; (b) a
persistent per-seq `cudf::hash_join` session object with internal matched-row tracking —
which also removes the per-probe-call build rebuild, the standing perf cost of streamed
probes. Either would be additive, alongside the planned scan-with-row-groups entry point.

<a id="t137"></a>
### #137 — batch-partitioned planner: drop null join keys before the shuffle

With `null_equals_null=false`, a row whose join key columns are all null matches nothing —
and `spark_hash_partition.cu` skips null columns (comet conformance), so every such row
lands in the one fixed partition `pmod(seed, N)`. On a null-dominated input that is pure
shuffle skew carrying rows the join will discard anyway.

Fix, for the sides whose unmatched rows are never emitted (both sides of an inner join;
the probe side of left-outer/semi/anti — the join-capability matrix knows which): the
translation layer inserts `GpuFilter(<key> IS NOT NULL)` under the `GpuEmitPartitions`
feeding that side. Existing filter node, existing serialization, visible and
cost-accounted in the plan; the shuffle shrinks and the skew case never reaches the
kernel. `GpuEmitPartitions` itself keeps exactly one routing (hash, nulls co-located) —
no runtime knob.

Two deliberately-out-of-scope halves. Scattering null-keyed rows on placement-free sides
(the preserved side of outer/anti, where rows must be emitted but cannot match) needs a
C++ kernel knob plus a conformance-gate extension, and no corpus query has a
null-dominated join key to exercise it. And the adaptive form — partially execute a node,
observe a null-dominated intermediate, insert the filter at replan time — waits on
adaptive replanning existing at all; this ticket is the static planner tweak that the
adaptive path would later reuse.

<a id="t138"></a>
### #138 — batch-partitioned sort: ranged merge emission
`GpuAccumulateBatchesAndSort` and `GpuMergeSortedPartitions` run one `cudf::merge` over
all sorted input batches and materialize the full output, so the local peak is inputs +
output (~2× the data). cuDF has no streaming merge; the hand-rolled alternative is a
ranged merge — pick split keys, `cudf::slice` each sorted batch at `upper_bound`
boundaries (zero-copy views), merge range by range, emit and release each range. Bounds
the output term to one range; the inputs stay resident either way, so the win is at most
~2× on the sort's local peak. Do it only if sort peaks bind after the mode ships; it also
unlocks multi-batch output from merge nodes.

<a id="t139"></a>
### #139 — batch-partitioned GpuCoalesceBatches(target): compact post-filter fragments
Dropped from v1. After a selective filter, batches shrink to a few rows and every
downstream kernel pays per-launch overhead on each fragment. A `BatchAccumulator` that
concatenates to a minimum target size (DataFusion semantics: merge only, never split),
streaming out one batch whenever the threshold is crossed. `cudf::concatenate` via the
existing collapse arm — no C++ change; target size from the same budget rule that sizes
loader batches.

<a id="t140"></a>
### #140 — batch-partitioned broadcast joins (1:N partition broadcast)
Deferred by the design. Lets one partition (small dimension side) be broadcast to all N
partitions of the other side without shuffling the big side; also unblocks partitioned
cross/nested-loop joins. The blocker is consume-once: a GPU handle feeds exactly one
call, so a broadcast build needs either an explicit device copy (`GpuBatch::copy()` — 
keep `!Clone` so the cost stays visible at call sites) or a C++-side non-consuming/
refcounted handle. Interacts with #136's persistent-build option, which would solve both
at once.

<a id="t141"></a>
### #141 — batch-partitioned planner: skip the shuffle for small group-key sets
v1 skips `GpuMergePartitions` + `GpuEmitPartitions` around an aggregate only when the
input is already one partition or the aggregate is keyless. Skipping when the key set is
merely small (collapse to one partition, run `GpuAggregateBatches[final]` once, avoid the
shuffle) needs a cardinality estimate that does not exist — the estimators are constants
(#19). When stats land, add the rule and regenerate the affected plan goldens.

<a id="t142"></a>
### #142 — batch-partitioned: no recourse for oversized batches
Nothing downstream of the loader can split a batch: minimum load granularity is one row
group, `GpuCoalesceAllBatches` before a join build side can exceed any budget, and the
planner deliberately still produces a plan — the enforcer then trips at run time and the
query dies cleanly. Recourse options, deferred until better estimators and adaptive
execution: a split operator (needs a C++ slice-to-handles entry point), or adaptive
replanning on trip (re-plan with more partitions or smaller batches). Related: #91.

## Infrastructure / process

<a id="t113"></a>
### #113 — Provision GPU-host testdata by sweeping git-tracked files
Hand-maintained rsync lists (`pipeline.yml` gpu-tests, `scripts/build-test-shadgpu.sh`)
bit four times during the refactor. Fix: sweep
`git ls-files --cached --others --exclude-standard testdata`. Lessons from the parked
draft: `--exclude=*.parquet` is wrong (tpch.minimal commits 5 needed .parquet files);
tracked-only misses new uncommitted fixtures. Needs its own clean verification run.

<a id="t114"></a>
### #114 — plan_status is shape-validated but never truth-verified
`plan_status` (ok/fail) is shape-checked only; no test attempts to plan the `fail` rows.
When #23 lands and q27/q70/q72/q86 start planning, the widget keeps rendering `plan ✗`
with nothing going red. Fix: a permanent plan-attempt probe in `test_query_plan.rs`
asserting each `fail` row still fails to physically plan (and `ok` rows plan) — that tier
already provisions parquet. Accepted risk until then: stale ✗ cells after an upgrade.

<a id="t116"></a>
### #116 — Registry rows with no GPU coverage and no blocker
`hash_join` and `mixed_join` (full_table_gpu=na, no `gpu_full_table_test!` entry, no known blocker —
plain inner joins plus an aggregate; mixed_join adds a residual range filter) and
`join_int` (tp8-only oracle test, no tp1 row). Either add the missing GPU test rows or
mark the cells intentionally-na with a reason.

<a id="t126"></a>
### #126 — maybe_write_result_golden discards its removal result
`peacockdb-core/tests/common/mod.rs` (~L566): when a result exceeds
`RESULT_GOLDEN_MAX_BYTES` the golden is deleted so the GPU test falls back to the live
oracle, but `let _ = std::fs::remove_file(...)` discards the outcome and the message
prints unconditionally. A failed removal is therefore reported as "no golden" while the
stale golden is still on disk, and the message cannot distinguish "deleted a stale one"
from "there was nothing here". Narrow, but it is a regen path: the operator's next move
is to trust the log and commit. Same shape as #119. Check the result, and say which of
the two things happened.

<a id="t135"></a>
### #135 — Column ordinals are enforced only by cuDF's `at()` and the final result
Every column reference in the IR is an ordinal into the child's output table, and almost nothing
checks them. Two concrete gaps. (a) `TableResult` is a `cudf::table` plus a
`std::vector<std::string>` of names with no invariant that the two have the same length, and the
six sites that index the names use `operator[]` — so a short names vector is undefined behaviour,
not an exception (`filter.cpp` ~L42 reads `fv.column(idx)` and `input.column_names[idx]` in one
loop iteration; only the first is checked, and it happens to run first). Assert
`num_columns() == column_names.size()` where `TableResult` is built. (b) Nothing checks that a
child produced its columns in the ORDER the plan assumed: the per-node golden records name,
partitions, rows and bytes, and the bytes come from the plan's schema on both engines by design
(`logical_size_from_schema`, single-sourced so CPU and GPU cannot drift) — so a node emitting the
right column count in the wrong order yields identical per-node numbers everywhere and surfaces
only at the root, in a `.result.txt`/oracle comparison. A per-node type or column-name check in
the GPU tiers would close it. Also worth fixing while there: `expr.cpp` ~L349 bounds-checks a
ColumnRef and returns `type_id::EMPTY` instead of throwing, converting a bad ordinal into a
confusing type error later. Found auditing indexing for architecture.md.

<a id="t134"></a>
### #134 — begin_plan's node count is returned and discarded
`peacock_executor_begin_plan` fills `out_node_count`, and
`GpuNodeExecutor::new` (`backend/gpu_node_executor.rs`) reads it into a local and drops it. That
number is the one free cross-check on the invariant the whole node-by-node FFI rests on: Rust's
`post_order` and C++'s `index_post_order` are separate implementations of the same rule, and
every handle is addressed by the resulting sequence number. Today a divergence surfaces
indirectly — per-node stats that stop matching the `.cpu.txt` golden, or an unknown-handle throw
— i.e. as wrong-looking results rather than as the structural mismatch it is. Fix: compare it
with the Rust walk's node count in `GpuNodeExecutor::new` and error naming both numbers. Two
lines, and it converts a confusing failure into an obvious one. Found while diagramming the
node-by-node driver's collaborators.

<a id="t133"></a>
### #133 — No tp1 plan golden: tp1 cost annotation is unpinned
Every `.plan.txt` in the tree is tp8 — 130 at `tp8-mini` plus `shuffle_additive` at
`tp8-standard` — so the plan tier never renders a tp1 plan. The tp1 node TREE is still pinned,
by the 110 `tp1-standard` `.cpu.txt` goldens, which carry each node's exprs, predicates and
projections. What is pinned nowhere is the plan-only annotation at tp1: `row_width`,
`subtree_max_row_bytes`, `estimate_input_bytes`, `estimate_output_bytes`, `estimate_cost`
appear only in `.plan.txt`. A change to the memory/cost model that moved those numbers at tp1
while leaving tp8 unchanged would go green — and tp1 is the shape the GPU full-table tier
runs, so it is not a corner. Fix: canonize a handful of tp1 `.plan.txt` goldens (the same
queries the tp1-standard cpu tier already uses), or teach the cpu golden to carry the
annotation. Same class as #114 — a cell nothing verifies. Found auditing tp1/tp8 planning
impact.

<a id="t132"></a>
### #132 — Two batch-size fields cross the IR and nothing on the C++ side reads them
`GpuScan.batch_size` and `GpuCoalesceBatches.target_batch_size` are both computed by
`GpuMemoryBudgetRule`, serialized by the plan serializer, and pinned byte-for-byte by
`goldens/plan_bytes.sha256` — and `grep -rn 'batch_size' cpp/src cpp/include` returns
nothing. Neither is read: the GPU scan reads by row group (`set_row_groups`), so there is no
batch to size, and `GpuCoalesceBatches` is `execute_passthrough` in `dispatch.cpp`, so the
target it carries is never consulted. Harmless today, and the wire format is deliberately
frozen, but it reads as a memory bound the GPU respects when the GPU's only memory lever is
the row-group→partition map. Decide which it is: either the GPU should honour a batch/chunk
bound (overlaps #91's memory-aware repartition and #34's single-table constraint) or the two
fields should be dropped from the IR in a commit that regenerates the digest deliberately.
Until then, do not read these fields as evidence that GPU execution is batch-bounded. Found
testing the assumption that batch_size is respected only in full-table CPU execution — it is
respected in BOTH CPU modes and in neither GPU mode.

<a id="t131"></a>
### #131 — resident model never accounts for cross / nested-loop join build sides
`resident.rs::peak()` stacks a join's build side by matching `stat.node_name` against
`"HashJoinExec" | "CrossJoinExec" | "NestedLoopJoinExec"`, and its own doc comment names all
three. But `GpuCrossJoinExec` and `GpuNestedLoopJoinExec` are two of the five operators that
do NOT strip, so the name reaching the classifier is `GpuCrossJoinExec` /
`GpuNestedLoopJoinExec`, which matches nothing and falls into the streaming arm: the build
side contributes zero and never stacks with the probe. `HashJoinExec` works only because its
wrapper strips. So the resident-OOM enforcer under-estimates every plan containing a cross
or nested-loop join, in the direction that lets a query through that should have tripped.
Latent today — the tight-budget set (`test_cpu_oom`: tpcds q78, tpch q7/q18) has no such
join — and unreachable by the existing unit tests, which construct names by hand
(`node("HashJoinExec", …)`) and so cannot see the mismatch: a guard that cannot go red.
Fix: classify on a type, not a rendered name (`as_operator` + the wrapped node's identity),
or normalize the `Gpu` prefix at the one place the name is recorded; then add a case built
from a real wrapped plan rather than a hand-written name. Found auditing the operator
tables for architecture.md.

<a id="t130"></a>
### #130 — `partition_topology()` is implemented 16 times and read by nobody
`Operator::partition_topology()` (`peacockdb-core/src/operators/operator.rs`) returns each
operator's partition behaviour — ScanEmit / Map / Collapse / KWayMerge / RepartitionHash /
Join — and has *no* callers: not in `src`, not in `tests`. The CPU backend and the C++ side
each re-derive the same
classification independently: the CPU partitioned backend by ad-hoc predicates
(`collapses_partitions`, `hash_repartition_of`, `partitioned_join_arity` in
`backend/cpu_node_executor.rs`) and the C++ side by `node_type()` switches in
`node_session.cpp`. That is the duplicated-rule antipattern coding-style.md names: three
copies of one fact, and the trait copy — the one a reader would trust, since it is the
declared interface — is the copy nothing can prove right, because no test can make it go
red. Either drive both off it (the CPU predicates become one match on the topology)
or delete it and stop implying an abstraction the code does not use. Found auditing the
operator tables for architecture.md.

<a id="t129"></a>
### #129 — The "26.02" CI leg builds against a 25.10a image; the GPU job has no fork guard
Two unrelated smells in `pipeline.yml`, both found auditing the CI section of
build-test.md. (a) The cpp-cpu matrix leg labelled `cudf: "26.02"` runs
`rapidsai/base:25.10a-cuda12-py3.12`, so the compile-only 26.02 coverage the wiki and
`#94` both rely on is actually 25.10a coverage; the label is the only place 26.02 appears.
Either bump the image or rename the leg — as it stands, "26.02 compiles" is a claim no job
makes. (b) `s3-datasets` explains its fork guard as mirroring "the GPU job's fork guard",
but `gpu-tests` has no job-level `if:` — on a fork PR `secrets.SHAD_GPU_SSH_KEY` is empty,
so Setup SSH writes an empty key and the job goes red on ssh instead of skipping. Moot
while the repo has no forks, which is exactly why it will bite later.

<a id="t128"></a>
### #128 — Doctests run nowhere, and the meta guard cannot see them
`peacockdb-core` has one doctest (`CpuExecutor`, `src/lib.rs` ~L166). No step in
`pipeline.yml` passes `--doc`, so it executes only in a local `cargo test`. Worse than a
missing step: `test_ci_coverage.rs` enumerates `--test` targets plus `--lib`, so a
doctest is invisible to the guard whose whole job is finding targets CI does not run —
this one is unlisted, not exempted, and adding a second doctest would go equally
unnoticed. Fix: run `cargo test --features rust-only -p peacockdb-core --doc` in the
cpp-cpu tier and teach the guard that `--doc` is a target class it must see named
(the `--lib` check at `line_runs_lib_tests` is the pattern). Found during the
test-category census (2026-08-04).

<a id="t127"></a>
### #127 — Unowned testdata dirs on shad-gpu
shad-gpu carries `testdata/plans.sf1` (10 files) and `testdata/plans` (3) that no local
checkout has, that the git-derived fixture sweep does not produce, and that nothing in
the test suite references. Same shape as the binary orphans that `--delete` just closed:
state on a remote host no provisioning path owns, so nobody can say whether it is stale
or load-bearing. Not deleted, because something outside this repo may read it. Establish
what wrote them and either bring them under the sweep or remove them.

<a id="t124"></a>
### #124 — common/mod.rs is over the 1000-line bar
`peacockdb-core/tests/common/mod.rs` is 1410 lines against coding-style.md's "under 1000".
Pre-existing; the exec-mode refactor moved it the right way (`common/exec_mode.rs` split
out) but added net lines. Next extraction is the obvious one: ~250 lines of
`macro_rules!` (the test-macro definitions) into `common/macros.rs`, which puts the file
under on its own. Deliberately not done inside the refactor — the same page forbids
scope-creep refactors.

<a id="t125"></a>
### #125 — `elsewhere` parameter in assert_registry_matches_csv is dead
`peacockdb-core/tests/common/registry.rs`: after the by-mode file split every CSV column
is owned wholly by one binary, so all four callers pass `&[]` and the ~20 lines of
staleness checking can no longer fire. Reviewer's read is *delete* (coding-style.md: no
fallbacks the task didn't ask for); the parameter was kept only to hold branch scope, and
its doc paragraph was rewritten to stop justifying it with a now-false example. Re-add it
in the commit that actually splits a column across binaries again.

<a id="t13"></a>
### #13 — Hermetic builds: system-library whitelist + CI audit
`ld` silently prefers system libs over the conda env (seen as
`libarrow.so.2300: undefined reference to curl_easy_getinfo@CURL_OPENSSL_4`). Whitelist
glibc/libgcc_s/libcuda only; everything else from `$CUDF_ROOT`. Enforce via CMake
find-root pinning, build.rs link-search order, and a post-link `ldd` audit that fails CI.

<a id="t94"></a>
### #94 — MERGE_M2 count-child type is cuDF-version-specific
The stddev/var Final path (`cpp/src/operators/aggregate.cpp`) hardcodes INT32
`valid_count` for the 25.02 GPU runtime; 25.10/26.02 want INT64/FLOAT64. The 26.02 CI leg
is build-only so it stays green — this bites at the next GPU-remote cuDF bump. Switch or
version-gate the type then; the comment marks the site.

<a id="t69"></a>
### #69 — DuckDB cost oracle: multi-threaded golden generation for larger SF
`gen_duckdb_cost.sh` pins `PRAGMA threads=1` because `operator_rows_scanned` scales with
thread count (`output_bytes`/`output_rows` are thread-invariant). Fine at sf1, too slow
at sf10/sf100. Simplest fix: parallelize across queries, keep per-query threads=1; or
drop/normalize the thread-sensitive field.

<a id="t49"></a>
### #49 — Test crates bake CARGO_MANIFEST_DIR for testdata
**Priority: low**

Why it matters: a test binary is built on one host and run on another. Remote CPU runs ship
binaries, goldens and data — never source — so any path baked in at compile time is a path
that does not exist on the remote. `tests/common/mod.rs testdata_root()` solves that by
honouring `PEACOCK_TESTDATA_DIR` first, and `build-test.sh` sets it for remote runs.

The residual is three test files that read testdata through
`env!("CARGO_MANIFEST_DIR")` directly instead of `testdata_root()`:
`test_node_executor.rs` and `test_plan_serialiser.rs` (both for `tpch.minimal`) and
`diag_flip_audit.rs`. Those binaries therefore only find their data where the build tree
stood — which is exactly why a remote CPU host needs a `/media/data/peacockdb` symlink, and
why `--gpu` runs, which set the env var, do not.

`test_ci_coverage.rs` is on the old list but should come off it rather than be swept: its
`CARGO_MANIFEST_DIR` use resolves the REPO root to read `.github/workflows/pipeline.yml`, not
testdata, and no env var should redirect that. Sweep the three, drop the fourth from the
list, then close.
