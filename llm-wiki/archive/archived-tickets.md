# peacockdb archived tickets

Tickets that are finished or that the tree outgrew. Numbers stay permanent and are never
reused, so a commit message or comment naming an old ticket still resolves — here.
`llm-wiki/tickets.md` holds only open work.

**Numbers spent without a ticket.** A number withdrawn before it described anything real is
recorded here and nowhere else, so the counter never walks back over it:

- **#165** — filed and withdrawn 2026-08-20. It reported a dead widget link — `cost-registry.csv`
  naming #115 after #115 was archived — and there was no such link: the widget already follows a
  number to the file holding its anchor, with #115 as that test's own case. What was wrong was
  the note below, which predated that code and now says what holds. Named by commit 7b98a99.
- **#156** — filed and withdrawn 2026-08-19. It called the exec-model prototype's row-group
  chunking a defect against the engine; the prototype is a model, not a specification, so
  diverging from it where it is wrong is the outcome rather than a drift to reconcile
  (`scripts/exec_model/README.md`). Named by commit 4c89d91.

Archiving a ticket the registry names is safe: the cost widget resolves a number to
whichever of the two files holds its `<a id="tNN">` anchor (`TicketIndex::path_for` in
`cost-report/src/main.rs`), and refuses to render a link for a number in neither, failing
the report rather than emitting one that goes nowhere.

## Done

<a id="t103"></a>
### #103 — GPU SIGSEGV: shuffle_stddev tp8-standard (Welford N-way merge)
`gpu_partitioned_tpch_sf1_shuffle_stddev_partitioned_tp8_standard` segfaults (139) or fails as a contained
`vector::reserve` Err on shad-gpu. Nondeterministic; reproduces at 12 GiB and 120 GiB, so
not budget-related. Inside `execute_instrumented`, upstream of golden compares; tp1 never
crashes. Suspect the 8-way Welford M2 merge (`cpp/src/operators/aggregate.cpp`).
**Fixed and the test is back**, 2026-08-21. The cause was `OutBuild` in
`cpp/src/operators/aggregate.cpp` carrying default initializers on its last fields only: the
Final-mergeable arm filled the struct by name and left `res` — an index into the results vector —
uninitialized, so it read whatever the stack held. That accounts for every symptom: two
signatures from one input, a budget that changed nothing, and tp1's immunity, since tp1 plans
`Single` and never enters that arm. Found when a new merge arm wrote the same shape and turned
the latent read into a segfault on its first call.

`gpu_case!(tpch, 1, shuffle_stddev, partitioned_tp8_standard, golden_approx_std)` is enabled
again after 20 consecutive green runs on shad-gpu, and `common/gpu_cases.inc` names this ticket
beside it: if it flakes again, reopen this rather than filing a new number.
<a id="t151"></a>
### #151 — the per-node benchmarks measured the engine with no RMM pool

`peacock_gpu_benchmarks` times the engine through the FFI, and the engine installs no device
resource ([#148](../tickets.md#t148)) — so every intermediate it allocated was a
`cudaMalloc`/`cudaFree` round trip, and a node's `time_us` included that. The C++ gtest binaries had stopped
measuring it, so the two families of numbers in the tree were taken under different
allocators, and the per-node records charged a node for the default resource — worst exactly
where its output is largest.

**Landed as the fallback, not the fix.** [#148](../tickets.md#t148) stays open on its own
terms: it also makes a shipping query faster, and it carries a second decision about
`gpu_memory_limit`. Instead:

- The sizing rule moved to `cpp/include/peacock/rmm_pool.hpp`, shared rather than copied —
  `multi_gpu.cpp` takes its 85/95 from the same constants. Header-only, because four gtest
  targets link `cudf::cudf` and gtest but not `peacock_gpu`.
- `peacock_install_rmm_pool` reports the OUTCOME, not success: a pool, or a pool that could
  not be built. Before it a failed reservation was indistinguishable from an ordinary run.
- It lives in `libpeacock_gpu.so` though no shipping query calls it. A test-only shim DSO
  would leave production untouched, but the build sets no `-fvisibility=hidden`, so rmm's
  current-device-resource state could be duplicated across DSOs — a green run that fixes
  nothing.
- Idempotency is in C++ alone, not a Rust `OnceLock`: one guard, on the side that owns the
  resource. Rebuilding would drop a resource live allocations still point into, and 127
  `#[test]`s in one process is the shape that finds it.
- Records carry `allocator=` beside `build_profile=` and `sync_floor_us=`, and the harness
  asserts both before it measures anything: a time taken without a pool, or from a
  non-release build, is an invalid comparison, so it is refused rather than written.

**Re-swept, all 127 records, every one faster**: median -9.6% on `total_us`. The move is not
uniform, which is the whole point — `GpuScanExec` -5.3% (parquet read, few intermediates)
against `GpuProjectExec` -53%, `GpuHashJoinExec` -51%, `GpuFilterExec` -46%,
`GpuAggregateExec` -30%. The old records were wrong about the SHAPE of a plan's cost, not
only its scale.

They were also stale for the smaller second reason this ticket carried — measured before the
scatter stopped opening a timed region of its own — so a part of that move is not the
allocator. Measured then at -2.5% median on repartition nodes and -0.6% on whole records, it
cannot account for -50% on joins. Both halves are now settled by one sweep.

**Done.** The pool is now the only way the benchmarks run — there is no switch to turn it
off and no outcome but a pool that the harness will record.

<a id="t173"></a>
### #173 — a collapse of nothing answers with a table that has no columns

`GpuCoalesceAllBatches` over a lane that received no batch returns a handle whose table carries
zero columns, so the export decodes a batch whose first column is out of bounds.
`cudf::concatenate` of an empty view list has no schema to preserve, and the node's declared one
is not reachable: `PlanNode.output_schema` exists on the wire but the recipe writer leaves it
`None` (`recipe/writer.rs:102`, `:131`), which `WRITER_DIFFERENCES` records with the reason that
nothing on the C++ side reads it.

Decided rather than filled in: an empty lane emits nothing, on both backends, so the arm is
unreachable from a correct driver and `node_session.cpp:302` throws on a zero-input collapse
instead of answering with a schemaless table — the same shape as `execute_one`'s
consumed-equals-provided check. Putting the schema on the wire for this one arm would move all 16
digests in the payload golden to serve a case that no longer happens.

Closes when the throw is in, a device test proves it goes red, the CPU executor emits nothing for
an empty lane as the GPU one does, and the `WRITER_DIFFERENCES` reason says the arm that would
have read a schema is now a refusal.

**Done 2026-08-25**, by 462e018: the zero-input collapse throws, a device test constructs it
through the ABI and shows it red, the CPU executor emits nothing for an empty lane as the GPU one
does, and `WRITER_DIFFERENCES` records that the arm which would have read a schema is now a
refusal.

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

Closed 2026-08-19. The batch-partitioned planner now checks every column reference's name against the field at its position, so the class is caught at plan time for that mode; the C++ items above are carried by [#164](../tickets.md#t164).

<a id="t115"></a>
### #115 — q38/q76/q87 set-op & union-count divergences — retriage
q38 (INTERSECT ×3), q76 (UNION ALL + IS NULL filters + grouped `count(*)`) and q87 (EXCEPT ×2)
diverged on the GPU historically, and the ticket's instruction was to rerun on an H200, enable
what passes, and root-cause the rest.

**Done 2026-08-19.** All three rerun on shad-gpu in `golden_exact` — per-node rows and cost
against a freshly generated `.cpu.txt`, plus the whole result. None diverges in any way: not a
row count, not a value, not a per-node mismatch. q38 answers 107, q87 47298, and q76 matches
its 103-line grouped output exactly. Whatever the bucket-I triage saw was fixed somewhere
between it and today; q87's named suspect — anti/EXCEPT null handling overlapping
[#80](../tickets.md#t80) — was never the cause, since `EXCEPT` wants the `EQUAL` that anti
hardcodes.

**The three stay disabled in the legacy modes deliberately**, which is why this is closed
rather than acted on: they are `full_table_gpu=na` by choice now, not by defect, and the
registry keeps `115` in their `tickets` column so the widget still says which decision the
cells rest on. The batch-partitioned mode plans all three; q87 is recorded in
`llm-wiki/tasks/batch_partitioned_executor.md` as a query whose corpus is larger than legacy's.

<a id="t77"></a>
### #77 — Cost report: publish per-SHA history on master
Pages deploy overwrites the latest report each run — no history, no trend. Deploy under
`/<sha>/`, keep `/index.html` as latest, add a lightweight index + `history.json`. Open:
retention policy, gh-pages branch vs deploy-pages artifact model.

**Done 2026-08-05.** Implemented and labelled as such in the code
(`cost-report/src/main.rs`, "page-per-sha Pages site (ticket #77)"): `<dir>/index.html` is
the latest report, `<dir>/<sha>/index.html` the same run addressable by commit,
`history.tsv` the newest-first manifest and `history.html` the rendered index, with the
report footer linking *latest · all reports*. The Pages deploy replaces the whole site, so
the CI step curls the prior manifest and each prior `<sha>/index.html` forward from the live
site before generating. Both open questions resolved: the deploy-pages artifact model (not a
gh-pages branch), and retention = whatever the manifest lists.

<a id="t27"></a>
### #27 — TPC-H q11/q22 scalar-threshold NLJ → broadcast filter
Likely stale: q11 and q22 now run full_table_gpu at tp1-standard and
nested-loop/cross-join operators exist. The 1×N broadcast-filter rewrite may still be a
perf win — verify, then close or rescope as an optimization.

**Done 2026-08-05.** The blocker is gone: tpch q11 and q22 both run
`full_table_tp1_standard` on the GPU (`test_gpu_full_table.rs` — q11 as `oracle` because its
result exceeds the golden size cap, q22 `golden_exact`), and both plans carry NestedLoopJoin
nodes that execute there. What remains is a pure optimization — rewriting the 1×N
scalar-threshold nested-loop join into a broadcast filter — and that belongs under the CBO
umbrella (#73) or runtime filters (#16), not as a standing blocker on two queries that
pass.

<a id="t18"></a>
### #18 — Add DuckDB cost estimates to canonical plan goldens
Appears delivered: `*.duckdb_cost.txt` goldens exist under `testdata/goldens/*.sf1/`,
generated by `gen_duckdb_cost.sh` / `duckdb_cost.py`. Verify the remaining checklist
items (regeneration wiring, diff display), then close.

**Done 2026-08-05.** The `*.duckdb_cost.txt` goldens exist for the whole corpus
(22 tpch + 99 tpcds under `testdata/goldens/*.sf1/`), `gen_duckdb_cost.sh` regenerates them
in two modes (`--gen` needs DuckDB 1.5.4; `--extract-only` rebuilds from the committed
profiles), and the cost report renders peacock Σout against duckdb Σout with the ratio
bucket. Nothing on the original checklist is outstanding.

<a id="t4"></a>
### #4 — Make the build process more understandable and transparent
Empty-body stub from project start. Largely answered by `llm-wiki/build-test.md`; fold
the rest into #13 or close.

**Done 2026-08-05.** Answered by `llm-wiki/build-test.md`, which now carries the local
build workflows and their separate cargo/C++ target dirs, what `rust-only` selects, the CI
job graph, and the remote hosts. The hermetic-build remainder it also gestured at is #13.

<a id="t3"></a>
### #3 — CI is too long
Empty-body stub from project start. Superseded by the tiered pipeline; close or re-file
with concrete targets.

**Done 2026-08-05.** Superseded by the tiered pipeline: cpp-cpu (two cuDF legs),
cpp-build-2502, gpu-tests and cost-report run as independent chains, the GPU leg ships
prebuilt binaries to shad-gpu rather than building there, and the golden/meta tier runs
under `rust-only` with no C++ at all. Re-file with concrete time targets if CI length
becomes a problem again.

## Stale

<a id="t157"></a>
### #157 — legacy: the budget rule drops a CoalesceBatchesExec's fetch, and the wire cannot carry one
**Priority: low** — legacy planning only; the batch-partitioned model plans from scratch and
has no such node.

`gpu_rule.rs` ~L611 rebuilds the node as `CoalesceBatchesExec::new(input, batch_size)`, which
sets `fetch: None`, so a limit DataFusion pushed onto it is gone.

DataFusion's limit pushdown does park one there and removes the limit node once it has:
`SELECT count(*) FROM (SELECT * FROM nation WHERE n_regionkey > 1 LIMIT 3)` plans as an
aggregate over `CoalesceBatchesExec{target, fetch: 3}`. The GPU half could not carry it
regardless — `CudfCoalesceBatches` has only `target_batch_size` and the node is
`execute_passthrough`. No golden can show it either: the node's display prints estimates and
never a `fetch`, so corpus reachability is unknown rather than ruled out, and the corpus
limits that were checked survive as their own nodes. Fix is three parts — `with_fetch`
through the rebuild, an fbs field the C++ reads, and the fetch in the node display.

**Stale 2026-08-20.** Legacy planning only, and the legacy modes are not being fixed. The
three-part fix — `with_fetch` through the rebuild, an fbs field the C++ reads, and the fetch
in the node display — would extend the wire format for a planner the batch-partitioned mode
replaces, which plans from scratch and emits no such node. Whether the corpus reaches it was
never established either way.

<a id="t171"></a>
### #171 — shad-gpu's benchmark tree is in an instrumentation this repo does not use

Every record on the host carries per-node `setup_us`, `submit_us` and `device_us`; the committed
form is a single `time_us`. Someone's instrumentation change ran there and its output stayed.
Since `--pull-benchmarks` deletes nothing but overwrites everything it finds, any pull — including
one after a single filtered case — rewrites all 127 committed files into that other format, which
is what happened on 2026-08-21 and was reverted by hand.

Two ways out and they are not equivalent. Either the split instrumentation is wanted, in which
case it belongs in the repo with the goldens regenerated on purpose and the reader taught the new
lines; or it is not, in which case the host's tree should be cleared so the next pull cannot
resurrect it. Nobody has decided, and the cost of not deciding is paid by whoever pulls next
without reading `build-test.md`.

**Stale 2026-08-25.** Decided the other way round from the ticket's two options: shad-gpu is
scratch space for benchmark experiments, and the committed records in git are the canonical ones.
So neither the host's tree has to be cleared nor its instrumentation adopted — a pull brings back
whatever the host holds, and what to keep is the puller's to read before committing, which
`build-test.md` says at `--pull-benchmarks`.

<a id="t53"></a>
### #53 — Deterministic multi-partition CPU node-by-node execution
Largely superseded: `partitioned_cpu` produces deterministic tp8-standard `.cpu.txt`
goldens and `full_table_cpu` runs tp8-hinted plans single-partition. Residual: whether
recursive ftc_tp8 byte-accounting determinism still matters for any golden. Re-evaluate,
likely close.

**Stale 2026-08-05.** Largely superseded: `partitioned_cpu` produces deterministic
tp8-standard `.cpu.txt` goldens and `full_table_cpu` runs tp8-hinted plans single-partition.
Filed stale rather than done because the residual question — whether recursive ftc_tp8
byte-accounting determinism still matters for any golden — was never answered; it has simply
never failed.

<a id="t34"></a>
### #34 — Multi-partition table results in the GPU executor
The C++ executor models every node as one materialized `cudf::table`, so
GpuUnion/GpuInterleave must concatenate (peak = Σ inputs) and
Coalesce/Repartition/SortPreservingMerge are single-input pass-throughs. Introduce a real
multi-partition result type and move the concat up to GpuCoalescePartitions / final
collect. Partly overtaken by the multi-handle partitioned path — rescope to the
full-table executor before building.

**Stale 2026-08-05.** Overtaken in the part that mattered: the partitioned GPU path
keeps one `cudf::table` per output partition handle in the NodeSession registry, so a node
there is N tables, not one. The single-table constraint survives only in the full-table and
all-at-once paths — where it is why `GpuUnion`/`GpuInterleave` must concatenate — and #110
retires the all-at-once one. Rescope to the full-table executor if that path is still around
when peak memory there matters.

<a id="t29"></a>
### #29 — Track skipped TPC-DS GPU execution tests
Superseded: enablement now lives in `test_gpu_full_table.rs` / `test_gpu_partitioned.rs`
plus `testdata/cost-registry.csv`, and
each surviving bucket has its own ticket (#32, #62, #57, #55, #56, #63, #45, #46, #47,
#60, #115). Close.

**Stale 2026-08-05.** Superseded by structure rather than by work: enablement lives in
`test_gpu_full_table.rs` / `test_gpu_partitioned.rs` and `testdata/cost-registry.csv`, whose
inventory tests check both directions, so a skipped query cannot go untracked. Every
surviving bucket has its own ticket (#32, #45, #46, #47, #55, #56, #57, #60, #62, #63,
#115).
