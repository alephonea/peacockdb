# peacockdb tickets

Migrated from GitHub issues on 2026-07-31; GitHub issues are closed and this file is the
registry. The number is the permanent ticket ID — each ticket carries an `<a id="tNN">`
anchor that the cost widget links to. Device labels are `tp<N>-<tier>` (micro=100MiB,
mini=2GiB, standard=12GiB).

A ticket carries a **Priority** line only when it is not medium; medium is the default.
New tickets take the next free number (currently 156). Finished and lapsed tickets move to
`llm-wiki/archive/archived-tickets.md` (Done / Stale) — numbers are never reused, so an old
reference still resolves there.

## Contents

| Section | Open | Tickets |
|---|--:|---|
| [Critical correctness](#critical-correctness) | 16 | #157 #153 #103 #80 #59 #46 #47 #60 #121 #122 #123 #118 #119 #120 #117 #41 |
| [Blockers for disabled coverage](#blockers-for-disabled-coverage) | 16 | #158 #97 #23 #32 #65 #62 #91 #95 #57 #45 #63 #56 #55 #115 #96 #143 |
| [Performance / architecture](#performance--architecture) | 26 | #155 #154 #152 #151 #150 #149 #148 #147 #146 #145 #19 #16 #20 #71 #101 #73 #110 #75 #136 #137 #138 #139 #140 #141 #144 #142 |
| [Infrastructure / process](#infrastructure--process) | 19 | #113 #114 #116 #126 #135 #134 #133 #132 #131 #130 #129 #128 #127 #124 #125 #13 #94 #69 #49 |

## Critical correctness

<a id="t157"></a>
### #157 — the budget rule drops a CoalesceBatchesExec's fetch, and the wire cannot carry one
**Priority: high**

`gpu_rule.rs` ~L611 rebuilds the node as `CoalesceBatchesExec::new(input, batch_size)`, which
sets `fetch: None`. Any limit DataFusion pushed onto that node is gone.

DataFusion's limit pushdown does park one there: `SELECT count(*) FROM (SELECT * FROM nation
WHERE n_regionkey > 1 LIMIT 3)` plans as an aggregate over a `CoalesceBatchesExec{target,
fetch: 3}`. Losing the fetch counts every row instead of three. The GPU half cannot carry it
regardless — `CudfCoalesceBatches` has only `target_batch_size` ([fbs](../flatbuffers/gpu_plan.fbs#L469)),
and the node is `execute_passthrough` — so a fetch surviving the rule would still die at the
wire. Nor would the plan text show it: the node's display prints estimates and never a
`fetch`, so no golden can be checked for this and corpus reachability is unknown rather than
ruled out. Fix: preserve `fetch` through the rebuild, carry it in the fbs, read it, and print
it. Found building the batch-partitioned layer, whose spec had the same "drop it" mapping.

<a id="t103"></a>
### #103 — GPU SIGSEGV: shuffle_stddev tp8-standard (Welford N-way merge)
`gpu_partitioned_tpch_sf1_shuffle_stddev_partitioned_tp8_standard` segfaults (139) or fails as a contained
`vector::reserve` Err on shad-gpu. Nondeterministic; reproduces at 12 GiB and 120 GiB, so
not budget-related. Inside `execute_instrumented`, upstream of golden compares; tp1 never
crashes. Suspect the 8-way Welford M2 merge (`cpp/src/operators/aggregate.cpp`).
**Test quarantined** 2026-07-31 (commented out in `test_gpu_partitioned.rs`) — it was the only
coverage of the 8-way M2 merge; tp8 goldens kept. Has already caused one wrongly
diagnosed "regression" + rollback: check this ticket before blaming your change.

<a id="t153"></a>
### #153 — equi-join residual filter is applied after the outer gather
**Priority: high**

`join.cpp` (~L353) masks a `CudfHashJoin.filter` over the gathered table *after* the join
null-padded its unmatched rows, so Left, Right and Full demote the ON condition to a WHERE.

`… LEFT JOIN b ON a.k = b.k AND b.v > 50` returns an inner join: a padded row's NULLs make the
predicate NULL, and a left row whose only matches fail is dropped rather than null-padded.
`execute_nested_loop_join` (~L453) refuses this exact shape with the argument written out, so
the fix direction is settled here rather than proposed; that path is unaffected. Latent because
DataFusion pushes an ON predicate reading one side below the join (tpch q13) — it needs one
referencing both sides, which no corpus query has. Fix: evaluate the residual during the join
(`mixed_*` covers Left, Right is its swap, Full needs inner-plus-re-add), plus a `PlanExecutor`
gtest with a filtered Left join. The same commit drops the prototype's deliberate reproduction.

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
Nothing is unimplemented — the title says *test*. `execute_union` retypes every branch column to
the declared output type before `cudf::concatenate` (`union.cpp` ~L49), or it throws.

What is open is that the coverage is incidental. The implementation comment names the case it
was written for — tpcds q5, pairing a decimal measure against a `cast(0 AS decimal(7,2))`
literal that materializes as FLOAT64, plus cuDF's SUM drifting fixed_point scale per branch —
and tpcds q5 does run `full_table_tp1_standard` GPU at `golden_exact`, so breaking the cast goes
red today. But it is red by luck: nothing pins that q5 keeps a union with mismatched branch
types, and if that shape changes the cast becomes untested silently, which is the situation this
ticket was filed to end. (tpch q5 is irrelevant — its plan has no `GpuUnionExec`.) Fix: a
focused gtest building a two-branch union with FLOAT64 against DECIMAL128, asserting the
concatenate succeeds with the declared type. Cheap, and independent of any corpus query.

## Blockers for disabled coverage

<a id="t158"></a>
### #158 — an aggregate DataFusion answers from statistics does not plan batch-partitioned
`SELECT count(*) FROM nation` never reaches an `AggregateExec`: DataFusion's
`AggregateStatistics` rule answers it from parquet metadata and emits `PlaceholderRowExec`
plus a projection.

The batch-partitioned translation layer refuses that node by name, which is the right v1
behaviour — inventing a one-row source is a node-set change, not a mapping. But it is a
coverage cliff rather than a curiosity: any corpus query of that shape simply has no plan in
the new mode, and it will present at T16 as a query that cannot be enabled. Two ways out
when it matters: a `GpuLiteralRows` source node holding the constant rows, or disabling the
rule for this planner so the aggregate survives to be lowered normally. The second keeps the
node set closed and costs a scan the first avoids. Audit the corpus for the shape before
T16 so the count is known rather than discovered.

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

<a id="t155"></a>
### #155 — umbrella: join execution through a wider C and FlatBuffers API
Every join mode already runs on the frozen surface (`scripts/exec_model`, and the capability
matrix in `tasks/batch_partitioned_executor.md`); open is what running it there costs.

| Cost | Ticket | What removes it | Surface change |
|---|--:|---|---|
| build side copied per probe batch | [#152](#t152) | refcounted handles | none — a handle stays a `u64` |
| probe batch copied per batch (Left/Full) | [#152](#t152) | refcount, or a node returning its input | none / fbs semantics |
| build side re-hashed per probe batch | [#136](#t136) | a stateful join session | 3 new symbols |
| probe keys resident + a finish join | [#136](#t136) | a match bitmap out-param, or that session | 1 argument / 3 symbols |
| a symbol per runtime-varying field | — | per-call overrides on `execute_node` | 1 symbol, replacing 3 |

Each follows from a handle being consumed by its reader, or an fb node's fields being plan
constants, and they overlap. The session subsumes the bitmap; the top two rows need no ABI
change. Land [#154](#t154) first, or the numbers are inflated by per-call copies.

<a id="t154"></a>
### #154 — every operator exit path deep-copies its output into a fresh table
`std::make_unique<cudf::column>(view)` deep-copies the device buffer, and 18 sites under
`cpp/src/` do it — 10 in `join.cpp` — mostly to a table the same function just produced.

`execute_hash_join` is worst: `cudf::gather` returns an owning table, the code copies each
column into `all_cols` (~L337, ~L342), then copies the kept ones again if the node projects
(~L376). `release()` moves instead; `scan.cpp` L108, `union.cpp` L38, `join.cpp` L254 are the
pattern, and it is C++-internal — no header, fbs, Rust or golden moves. Four kinds: whole table
freshly produced (`join.cpp` 202, 337, 342, 512, 515), mechanical; ordinal subset (`join.cpp`
211, 270, 376, 525, `filter.cpp` 41), needing an assert the ordinals are distinct; a column of
an **input** table (`join.cpp` 259, `project.cpp` 44, `window.cpp` 46, `expr.cpp` 850), changing
who destroys what under `NodeInputs`; and `aggregate.cpp` 407, 602, 604, 682, unresolved without
reading. Traps: a view taken before the release dangles (`ftv` ~L372), and a repeated projection
ordinal moves one column twice leaving a hole — a wrong answer, not a throw, which is why it
needs the assert and not the observation. Land before [#155](#t155).

<a id="t152"></a>
### #152 — batch-partitioned GpuJoin: the build handle does not survive a streamed probe
`NodeSession::execute_node` erases every input handle it reads (`node_session.cpp` ~L250, ~L339,
~L427), but a streamed probe calls the join seq once per batch and needs it B times.

[#136](#t136) assumes the table is there and so does the recipe mapping in
`tasks/batch_partitioned_executor.md`; neither says how. [#140](#t140) records the same
constraint one axis over — one build feeding N lanes — and [#145](#t145) is the mechanism both
want. Until then the choices are a device copy of the build side per probe batch, the cost
streaming exists to avoid, or a single-batch probe, which the capability matrix already forces
for filtered and non-inner-NLJ joins. The finish pass (#136) is unaffected either way.

Whether the copy is tolerable is answerable from the goldens: each join's two
`GpuCoalescePartitionsExec` lines carry both sides' `output_bytes`, and B copies cost `B ×
build_bytes` against one probe stream. Take the ratio on **bytes, not rows** — tpch q3 is 24:1
by rows and 73:1 by bytes. Decide before T12, under [#155](#t155).

<a id="t151"></a>
### #151 — the per-node benchmarks measure the engine with no RMM pool
`peacock_gpu_benchmarks` times the engine through the FFI and the engine installs no device
resource ([#148](#t148)), so every intermediate is a `cudaMalloc`/`cudaFree` inside `time_us`.

The C++ gtest binaries stopped measuring that — `cpp/tests/gpu/rmm_pool.hpp` installs a pooled
resource in `main()`, `PEACOCK_RMM_POOL=0` turns it off — so the tree's two families of numbers
are taken under different allocators, and the per-node records charge a node for the default
resource worst where its output is largest. Fix by landing #148, so the benchmark inherits the
shipping engine's pool; failing that, an install hook sharing `rmm_pool.hpp`'s sizing rule.
Either way the record must name its allocator beside `build_profile` and `sync_floor_us`.

The committed records' shape is current (all 127 verified 2026-08-14); their values are not —
they predate the scatter's timed region, whose stream sync barriered execution as well as
costing a floor. Re-running put 66 of 88 repartition nodes faster, median -2.5%, and nothing
asserts the records, so nothing is red.

<a id="t150"></a>
### #150 — store the embedding columns uncompressed; Snappy costs a third of a vector query to save 3%
The sf40 embedding columns are written SNAPPY and do not compress: `ps_image_embedding`
12306/12661 MB and `p_text_embedding` 3205/3293 MB, both 1.03x against ~1.6x elsewhere.

Float32 embeddings are high-entropy, so that is the data rather than the writer — and the GPU
decompresses them anyway. On q11v (`nsys`, share of GPU kernel time) `nvcomp::unsnap_kernel` is
564.7 ms / 37.9% on H200 and 419.5 ms / 24.9% on GB10, against 60.5 ms / 4.0% for the cuVS
distances and top-k the query exists to do; loading is 93.8% of H200 kernel time.

The change is `compression=NONE` for those two columns in `testdata/generate_testdata.sh`.
Parquet compression is lossless, so no value changes and no golden moves — only file size (~440
MB more) and the load path. Not free to do, though: sf40 is generated, uploaded and mirrored to
shad-gpu, so it means re-uploading 40 GB and re-verifying the 16 sf40 goldens. Measure with
`load_ms` per vector probe and the `unsnap` line from `nsys stats`.

<a id="t149"></a>
### #149 — the parquet load must use pinned host memory
**Priority: high**

Nothing in the engine sets a host memory resource for IO, so parquet loads from pageable host
memory — 10.6 GB/s H2D on the H200 against 47.3 GB/s pinned, 2 GiB buffers, 2nd-min of 5.

A discrete GPU's DMA engine transfers by physical address and cannot be handed a page the OS may
move, so a pageable source is bounced through an internal pinned staging buffer: a host memcpy
of every byte, which is what bounds the rate — H200's 10.6 GB/s sits just under its 11.8 GB/s
single-core memcpy, nowhere near its link rate. That is 4.4x on every byte the loader moves, and
the load dominates: 400-690 ms against 19-48 ms of execute on sf40. cuDF exposes
`cudf::io::set_host_memory_resource` and defaults to pageable — [#148](#t148)'s shape one side
over. Condition it on the device: GB10 shows 59.5 vs 59.2 because it has one physical pool, so
`pageableMemoryAccess` is the branch. Tests: compare the existing `[bench] … load_ms=` on both
hosts, asserting the discrete host improves and the integrated one does not regress.

<a id="t148"></a>
### #148 — the engine installs no RMM allocator, and `gpu_memory_limit` is accepted and ignored
**Priority: high**

Nothing under `cpp/src/` or `cpp/include/` calls `set_current_device_resource`, so every cuDF
intermediate takes rmm's default: a `cudaMalloc`/`cudaFree` driver round trip each.

Measured: TPC-H q1 over sf40 whole-table on GB10 is 76.5 s execute (2nd-min of 5, all runs
inside [75.4, 78.8], so steady state) against 3.9 s streamed through bounded batches for the
same answer. The gap was fixed twice already, in `multi_gpu.cpp` and `rmm_pool.hpp`; the engine
is the only one of the three that ships. The second half is the same fix: `gpu_memory_limit` is
documented as a bound, stored at `gpu_executor.cpp:99` and never read — the #132 shape one level
up — and a pool's `maximum` IS that bound. Care: install per device before any cuDF call, tear
down on the owning thread (`set_per_device_resource(id, nullptr)` misses the ref map), and size
by host kind, an integrated part's reservation sharing the page cache's pool. Tests: the GPU
tiers stay byte-identical, plus a case asserting a small limit is honoured.

<a id="t19"></a>
### #19 — Stats propagation: 55 TPC-DS hash joins blind to cardinality
`num_rows` doesn't survive descent through the `Gpu*`-wrapped tree (`statistics()` impls
in `gpu_rule.rs`; GpuScanExec not forwarding ParquetExec row counts), so JoinSelection
can't flip build/probe on 55 of 103 joins across 38 queries. One confirmed mis-order: q45
build 9.95× larger than probe. Foundation for all CBO work (#73, #20).

<a id="t16"></a>
### #16 — Dynamic / runtime filters: build-side keys → probe-side scan
Star-schema fact scans read 100% of rows while the joined dimension is filtered to ~30%. Build
an IN-set / min-max (later Bloom) at build completion and feed the probe-side GpuScan.

Applies to 76/99 TPC-DS queries; validate on q3/q19/q33. Best after #19. **Design it as the
groundwork for CTEs, not a join-to-scan special case.** A dynamic filter is the first thing here
whose producer has two consumers: the build side feeds the join, and it also feeds a replanning
consumer that turns those keys into a predicate on a scan below. That is a diamond, and every
plan model here is a tree. What serves it — a fork handing one batch stream to N consumers, plus
a consumer that plans rather than executes — is what a materialized CTE needs ([#101](#t101))
and what [#147](#t147) calls refinement in flight. Do it after the batch-partitioned mode, which
suits the shape: with refcounted handles ([#145](#t145)) a tee costs nothing on the device, the
accountant already models a fork's residency as the slowest consumer's backlog, and a consumer
blocking its producer is the join hold, one rule already mutation-tested. A diamond in the plan
is then routing rather than scheduling.

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

The mechanism a materialized CTE needs — one producer, N consumers of the same batch
stream — is the one [#16](#t16) has to build first for dynamic filters, and it is cheap in
the batch-partitioned model and expensive in a single-resident-table one. Sequence them
that way round.

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
Left-outer, full, semi, anti and mark need "which build rows matched across all probe batches",
and that never crosses the ABI — every call rebuilds the join from scratch.

`peacock_executor_execute_node` returns only the joined table plus rows/varlen stats, and there
is no persistent `cudf::hash_join` in `join.cpp`. Inner composes per-batch and right-outer emits
its unmatched probe rows batch-locally, so those are unaffected. No-ABI-change plan (v1):
accumulate each probe batch's key columns only, keys being small next to rows, and at the finish
call run one `left_anti_join(build, accumulated_keys)`, semi form for semi/mark, then null-pad
the probe columns with a synthesized project. Correct for pure equi-joins, `null_equals_null`
applying to the finish join too and #80's NOT IN caveat carrying over. A keys-only input cannot
evaluate a residual filter, so filtered semi/anti keep a single-batch probe. Cost: the keys stay
resident and the build side is built once more at finish. If that bites, the ABI options are a
per-call match bitmap out-param or a per-seq join session that also removes the rebuild — weigh
them with the rest of the join surface, [#155](#t155).

<a id="t137"></a>
### #137 — batch-partitioned planner: drop null join keys before the shuffle
With `null_equals_null=false` an all-null key matches nothing, and `spark_hash_partition.cu`
skips null columns, so every such row lands in the one partition `pmod(seed, N)`.

On a null-dominated input that is pure shuffle skew carrying rows the join discards anyway. Fix,
for the sides whose unmatched rows are never emitted — both sides of an inner join, the probe
side of left-outer/semi/anti, and the capability matrix knows which: the translation layer
inserts `GpuFilter(<key> IS NOT NULL)` under the feeding `GpuEmitPartitions`. Existing node and
serialization, cost-accounted in the plan, and `GpuEmitPartitions` keeps its one routing.

Two halves are out of scope deliberately. Scattering null-keyed rows on placement-free sides
(outer/anti's preserved side) needs a kernel knob and a conformance-gate extension, and no
corpus query exercises it. The adaptive form — insert the filter at replan time — waits on
adaptive replanning existing at all.

<a id="t138"></a>
### #138 — batch-partitioned sort: ranged merge emission
`GpuAccumulateBatchesAndSort` and `GpuMergeSortedPartitions` run one `cudf::merge` over all
sorted inputs and materialize the whole output, so the local peak is inputs + output.

cuDF has no streaming merge. The hand-rolled alternative is a ranged merge: pick split keys,
`cudf::slice` each sorted batch at `upper_bound` boundaries (zero-copy views), merge range by
range, emit and release each. That bounds the output term to one range; the inputs stay resident
either way, so the win is at most ~2x on the sort's local peak. Do it only if sort peaks bind
after the mode ships — it also unlocks multi-batch output from merge nodes.

Landing it re-introduces a third `SortOrder` state. The enum is two-valued today because every
node that orders a whole stream emits one batch, so "stream sorted" is `BatchSorted` meeting
`SingleBatch` and is derived rather than declared. Ranged emission produces the one shape that
breaks that — a stream ordered across several batches — so it must add `PartitionSorted` back
and teach the limit-after-sort validation to accept it alongside the derived form.

<a id="t139"></a>
### #139 — batch-partitioned GpuCoalesceBatches(target): compact post-filter fragments
Dropped from v1. After a selective filter, batches shrink to a few rows and every
downstream kernel pays per-launch overhead on each fragment. A `BatchAccumulator` that
concatenates to a minimum target size (DataFusion semantics: merge only, never split),
streaming out one batch whenever the threshold is crossed. `cudf::concatenate` via the
existing collapse arm — no C++ change; target size from the same budget rule that sizes
loader batches. The T0 prototype has the node
(`scripts/exec_model/operators/accumulators.py`, `ReBatchToTarget`) so the drivers are
shown to tolerate one at any tree position; it also splits, which the ticket's node does
not need, because the prototype uses it to make a stream's batches any shape.

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

<a id="t147"></a>
### #147 — PlanEstimates: a tree the planner emits and the runtime refines
The planner's `target_batch_bytes` walk already computes a per-node maximum resident size and
throws all but one number away. Keep it, as a tree shaped like the plan, one estimate per node.

`ParquetBatchPartitioner` emits it beside the row-group mapping. Nothing in the plan's
executability depends on it, but a wrong estimate is not free: too low and the query dies at the
enforcer's `scratch_bytes` pre-check, or as a cuda OOM below that. Neither is a wrong answer,
and #142 handles both gracefully later; a better estimate makes fewer queries reach either. Two
consumers, neither existing yet. **Placement** moves subtrees onto the CPU where the GPU cannot
hold them — the `Backend` trait already makes that a matter of choosing per node. **Refinement
in flight** extrapolates from one batch actually read, since the estimates otherwise rest on
constants (#19). The first version rewrites only what needs no replanning, a still-reading
loader's remaining batch sizes; later revisions may replace the plan outright, killing
in-progress GPU work and rebuilding the driver rather than editing the running tree — which is
why the driver owns no state a caller must survive it.

<a id="t146"></a>
### #146 — aggregate shaping beyond the fixed sequence
**Priority: low** — each part optimizes an already-correct plan and needs the same estimate.

The batch-partitioned aggregate applies one shape everywhere — per-batch init, merge per lane,
shuffle, finalizing merge — right where group cardinality is far below row count.

Three shapes it cannot express or choose. **(a) A merge accepting raw rows.**
`GpuAggregateBatches` requires pre-aggregated state, so the per-batch `GpuAggregate` is
mandatory: a lane with few batches pays B groupbys where one over the concatenation would do,
and a non-reducing aggregate pays an init that shrinks nothing. A raw-accepting merge holds
rows where a state merge holds groups, so it fits only the non-reducing case. **(b) A loader
emitting one batch per partition**, where its consumer materializes the whole partition anyway
— decide it from the subtree beneath the loader. **(c) The cardinality estimate all three
want** — does this aggregate reduce? — which the constant estimators (#19) cannot answer and
which gates [#141](#t141). Land after #19.

<a id="t145"></a>
### #145 — Refcounted handles: stop copying every partition out of a scatter
`spark_hash_partition` returns one table whose N partitions are already contiguous, and
`node_session.cpp` (~L265-272) deep-copies each range out, because a handle owns its memory.

So every shuffle copies its whole input a second time and peaks at twice the data — the concrete
form of [#91](#t91)'s repartition spike, on the batch-partitioned hot path, once per aggregate
and once per join side. The change: `TableResult` (`plan_executor.h:13`) becomes a
`shared_ptr<cudf::table> owner` plus a `cudf::table_view view` beside the names, and the scatter
registers N handles sharing one owner. Mechanical but wide — 35 sites across 11 files touch
`.table` / `->table`. **No ABI change**: a handle stays a `u64` and consume-on-use is unchanged,
so the three-symbol budget is untouched. The cost to weigh: a slice pins its whole parent, so a
skewed hash leaves one hot lane holding the pre-scatter table after the empty lanes finished —
the peak halves, the tail lengthens, and a materialize-if-small threshold is the local fix. Also
unlocks [#140](#t140). Tests: the GPU tiers stay byte-identical, plus a gtest releasing N−1
handles and checking the survivor still reads.

<a id="t144"></a>
### #144 — multiple DISTINCT arguments need a gid-multiplying expand
**Priority: low** — no query in either benchmark has this shape.

`count(DISTINCT a), count(DISTINCT b)` over different expressions is the one distinct shape the
batch-partitioned lowering cannot express.

Single-distinct lowers to grouping on the distinct argument, and non-distinct companions ride
along because Σ over the inner groups recovers each total. Two distinct arguments break it: one
grouping can dedup `a` or `b`, not both, and grouping by `(a, b)` dedups neither. The standard
fix is Spark's `RewriteDistinctAggregates` — an expand multiplies each row into one per distinct
argument tagged with a group id, the inner aggregate groups by `(keys, gid, args)`, and the
outer computes each count from the rows carrying its own gid. That needs a new row-multiplying
node, which is why it is a ticket rather than a planner tweak. Not [#65](#t65), whose gid is the
ROLLUP/CUBE `__grouping_id` — the two would coexist as separate columns. Until it lands the
planner refuses the shape at plan time.

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
checks them. Two concrete gaps.

(a) `TableResult` is a `cudf::table` plus a name vector with no invariant that the two have the
same length, and the six sites indexing names use `operator[]` — so a short vector is undefined
behaviour, not an exception. `filter.cpp` ~L42 reads `fv.column(idx)` and
`input.column_names[idx]` in one iteration and only the first is checked; it happens to run
first. Assert `num_columns() == column_names.size()` where `TableResult` is built. (b) Nothing
checks a child produced its columns in the order the plan assumed. Per-node bytes come from the
plan's schema on both engines by design (`logical_size_from_schema`, single-sourced so they
cannot drift), so a node emitting the right count in the wrong order yields identical numbers
everywhere and surfaces only at the root; a per-node type check in the GPU tiers closes it. Also
there: `expr.cpp` ~L349 returns `type_id::EMPTY` for an out-of-range ColumnRef instead of
throwing, turning a bad ordinal into a confusing type error further along.

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
`CudfScan.batch_size` and `CudfCoalesceBatches.target_batch_size` are both computed by
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
`resident.rs::peak()` stacks a build side by matching `stat.node_name` against `"HashJoinExec" |
"CrossJoinExec" | "NestedLoopJoinExec"`, and two of the three never match.

`GpuCrossJoinExec` and `GpuNestedLoopJoinExec` are two of the five operators that do not strip,
so the name reaching the classifier keeps its `Gpu` prefix, matches nothing and falls into the
streaming arm: the build side contributes zero and never stacks with the probe. `HashJoinExec`
works only because its wrapper strips. So the resident-OOM enforcer under-estimates every plan
containing a cross or nested-loop join, in the direction that lets a query through that should
have tripped. Latent — the tight-budget set (`test_cpu_oom`: tpcds q78, tpch q7/q18) has no such
join — and unreachable by the existing unit tests, which construct names by hand
(`node("HashJoinExec", …)`) and so cannot see the mismatch: a guard that cannot go red. Fix:
classify on a type rather than a rendered name (`as_operator` plus the wrapped node's identity),
or normalize the prefix where the name is recorded, then add a case built from a real wrapped
plan.

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
`peacockdb-core/tests/common/mod.rs` is 1359 lines against coding-style.md's "under 1000".
Pre-existing, and each refactor has moved it the right way while leaving it over:
`common/exec_mode.rs` first, then `common/benchmark.rs` (the GPU benchmark harness, 294
lines, split out as it was written rather than after). The obvious next extraction is
~250 lines of `macro_rules!` into `common/macros.rs` — which no longer suffices on its
own, so whoever takes this should expect a second cut, most likely the golden
assert/compare helpers. Accepted at 1359 for now.

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

Three test files read testdata through `env!("CARGO_MANIFEST_DIR")` instead of
`testdata_root()`, so they only find their data where the build tree stood.

It matters because a test binary is built on one host and run on another: remote CPU runs ship
binaries, goldens and data but never source, so a compile-time path is a path the remote does
not have. `tests/common/mod.rs testdata_root()` solves that by honouring `PEACOCK_TESTDATA_DIR`
first, which `build-test.sh` sets for remote runs. The residual is `test_node_executor.rs`,
`test_plan_serialiser.rs` (both `tpch.minimal`) and `diag_flip_audit.rs` — which is exactly why
a remote CPU host needs a `/media/data/peacockdb` symlink and why `--gpu` runs, which set the
env var, do not. `test_ci_coverage.rs` is on the old list but should come off rather than be
swept: its `CARGO_MANIFEST_DIR` resolves the repo root to read `pipeline.yml`, not testdata, and
no env var should redirect that. Sweep the three, drop the fourth, close it.
