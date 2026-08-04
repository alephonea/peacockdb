# peacockdb tickets

Migrated from GitHub issues on 2026-07-31; GitHub issues are closed and this file is the
registry. The number is the permanent ticket ID — each ticket carries an `<a id="tNN">`
anchor that the cost widget links to. Names reflect the post-refactor tree: device labels
`tp<N>-<tier>` (micro=100MiB, mini=2GiB, standard=12GiB); C++ in `cpp/src/operators/*.cpp`,
`expr.cpp`, `node_session.cpp`, `dispatch.cpp`; Rust modes in `peacockdb-core/src/executors/`.

New tickets take the next free number (currently 124).

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
(`executors/stream.rs`) is full-table-only, not ported to the partitioned driver.
Payoff: a genuine tp8-mini (2 GiB real-8-way) device, inexpressible today.

<a id="t95"></a>
### #95 — Decimal partition keys for real-8-way murmur3
murmur3 covers int/date/timestamp/composite/null; decimal deferred (float indefinitely).
Needed once the tp8 rollout reaches decimal-keyed repartitions (tpch q18 `o_totalprice`,
q22 `c_acctbal`, tpcds `i_current_price`). Dispatch by LOGICAL precision (≤18 → low 8 LE
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
DataFusion inlines every CTE reference and peacock re-scans each copy. Worst: tpcds q23
(CTEs ×5/×3/×2 → 6 scans), q4/q11 (`year_total` ×6/×4), q31; tpch q15 (`revenue0` ×2),
q2. Direction: CTE materialization or physical CSE; at minimum make the cost model aware.

<a id="t73"></a>
### #73 — Cost-based optimizer (CBO) umbrella
Move physical planning from static heuristics to cost-based, validated against the DuckDB
cost oracle and goldens. In scope: adaptive filter placement, join enumeration (#20),
stats (#19, the load-bearing prereq), runtime filters (#16).

<a id="t34"></a>
### #34 — Multi-partition table results in the GPU executor
The C++ executor models every node as one materialized `cudf::table`, so
GpuUnion/GpuInterleave must concatenate (peak = Σ inputs) and
Coalesce/Repartition/SortPreservingMerge are single-input pass-throughs. Introduce a real
multi-partition result type and move the concat up to GpuCoalescePartitions / final
collect. Partly overtaken by the multi-handle partitioned path — rescope to the
full-table executor before building.

<a id="t110"></a>
### #110 — Retire the all-at-once GPU executor
Remove `executors/all_at_once_gpu_executor.rs` → `peacock_execute` FFI (whole-plan, no
per-node stats), migrate or drop its 5 smoke tests plus the lifecycle test, and drop the
FFI symbols from `cpp/include/peacock_gpu.h` / `cpp/src/gpu_executor.cpp` (and
`execute_plan.cpp`). Its stated blocker — a common `Executor` trait — has landed, so this
is unblocked pending full_table/partitioned covering all needs.

<a id="t75"></a>
### #75 — Refactor duckdb_cost.py: separate cost formula from extraction
~870 lines where the ~80-line cost formula hides inside JSON parsing, predicate parsing
and row-group pruning. Shape: preprocessor → flat per-node intermediate
(op/rows_read/bytes_read/out_rows/out_bytes/breaker) → one-line cost. Pure refactor:
`.duckdb_cost.txt` numbers must not move.

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

<a id="t77"></a>
### #77 — Cost report: publish per-SHA history on master
Pages deploy overwrites the latest report each run — no history, no trend. Deploy under
`/<sha>/`, keep `/index.html` as latest, add a lightweight index + `history.json`. Open:
retention policy, gh-pages branch vs deploy-pages artifact model.

<a id="t69"></a>
### #69 — DuckDB cost oracle: multi-threaded golden generation for larger SF
`gen_duckdb_cost.sh` pins `PRAGMA threads=1` because `operator_rows_scanned` scales with
thread count (`output_bytes`/`output_rows` are thread-invariant). Fine at sf1, too slow
at sf10/sf100. Simplest fix: parallelize across queries, keep per-query threads=1; or
drop/normalize the thread-sensitive field.

## Backlog / stale

<a id="t29"></a>
### #29 — Track skipped TPC-DS GPU execution tests
Superseded: enablement now lives in `test_gpu_full_table.rs` / `test_gpu_partitioned.rs`
plus `testdata/cost-registry.csv`, and
each surviving bucket has its own ticket (#32, #62, #57, #55, #56, #63, #45, #46, #47,
#60, #115). Close.

<a id="t27"></a>
### #27 — TPC-H q11/q22 scalar-threshold NLJ → broadcast filter
Likely stale: q11 and q22 now run full_table_gpu at tp1-standard and
nested-loop/cross-join operators exist. The 1×N broadcast-filter rewrite may still be a
perf win — verify, then close or rescope as an optimization.

<a id="t41"></a>
### #41 — Standing test for GpuUnion branch-type normalization cast
Premise was "no enabled test exercises the cast until q5 is re-enabled" — q5 is now
enabled. Verify q5 actually drives the `type()!=want` branch in
`cpp/src/operators/union.cpp`; if yes close, else add the focused FLOAT64-vs-DECIMAL128
fixture.

<a id="t53"></a>
### #53 — Deterministic multi-partition CPU node-by-node execution
Largely superseded: `partitioned_cpu` produces deterministic tp8-standard `.cpu.txt`
goldens and `full_table_cpu` runs tp8-hinted plans single-partition. Residual: whether
recursive ftc_tp8 byte-accounting determinism still matters for any golden. Re-evaluate,
likely close.

<a id="t49"></a>
### #49 — Test crates bake CARGO_MANIFEST_DIR for testdata
Mostly done: `tests/common/mod.rs testdata_root()` honors `PEACOCK_TESTDATA_DIR`.
Residual direct uses: `test_node_executor.rs`, `test_plan_serialiser.rs` (tpch.minimal),
`diag_flip_audit.rs`, `test_ci_coverage.rs`. Sweep those, then close.

<a id="t18"></a>
### #18 — Add DuckDB cost estimates to canonical plan goldens
Appears delivered: `*.duckdb_cost.txt` goldens exist under `testdata/goldens/*.sf1/`,
generated by `gen_duckdb_cost.sh` / `duckdb_cost.py`. Verify the remaining checklist
items (regeneration wiring, diff display), then close.

<a id="t4"></a>
### #4 — Make the build process more understandable and transparent
Empty-body stub from project start. Largely answered by `llm-wiki/build-test.md`; fold
the rest into #13 or close.

<a id="t3"></a>
### #3 — CI is too long
Empty-body stub from project start. Superseded by the tiered pipeline; close or re-file
with concrete targets.
