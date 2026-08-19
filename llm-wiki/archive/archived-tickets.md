# peacockdb archived tickets

Tickets that are finished or that the tree outgrew. Numbers stay permanent and are never
reused, so a commit message or comment naming an old ticket still resolves — here.
`llm-wiki/tickets.md` holds only open work.

**Numbers spent without a ticket.** A number withdrawn before it described anything real is
recorded here and nowhere else, so the counter never walks back over it:

- **#156** — filed and withdrawn 2026-08-19. It called the exec-model prototype's row-group
  chunking a defect against the engine; the prototype is a model, not a specification, so
  diverging from it where it is wrong is the outcome rather than a drift to reconcile
  (`scripts/exec_model/README.md`). Named by commit 4c89d91.

One caveat before archiving anything else: the cost widget links tickets as
`llm-wiki/tickets.md#tNN` (`cost-report/src/main.rs`), so a ticket named in the `tickets`
column of `testdata/cost-registry.csv` must not be moved here without also updating that
link.

## Done

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
