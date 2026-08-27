# Archived task specs

Specs for tasks whose PR has merged, newest first. Each is the contract the work was
done against, kept verbatim -- including the amendments and corrections made mid-task,
since those are the part a later reader cannot reconstruct from the diff.

T15, T16, T17, T17a, T18 and T19 merged 2026-08-27 as PRs #131 through #135 and are not
here: their specs are sections of
[`batch_partitioned_executor.md`](../tasks/batch_partitioned_executor.md), which is still
in-flight for T20 and T22, and are struck through in its implementation plan instead.

Merged 2026-08-20 as PR #126, opened against ENS-bp-plan-skeleton and retargeted to
master when that base merged.


---

<!-- archived from llm-wiki/tasks/schema_and_validation.md -->

# schema registry and validation (T7/T8 remainder)

**Goal.** Finish the two tasks whose implementations landed early on
`ENS-bp-plan-skeleton` but whose test surfaces did not: prove the `Schema` carried on every
node is right, and make `validate_schemas_and_partitions` a check that can go red for the
reasons it claims rather than only for the ones a real plan happens to hit.

**What already landed, so nobody rebuilds it.** `Schema` exists with `group_keys` and
`agg_state`, every node carries a populated one, the plan goldens print the declared type per
column, and node-local validation is called from `plan_batch_partitioned` with all ten
goldens passing it node by node. That half was pulled forward by a review finding — ten
guards existed and none ran on the live path.

## T7 — what the schema tests must show

- A hand-built plan carrying a project, an aggregate and a union produces the expected types
  and the expected semantics annotations, asserted on the tree rather than on rendered text.
- The annotations survive the aggregate sequence: `agg_state` is right at the init, at the
  per-lane merge and at the finalizing merge, which are three schemas for one logical
  aggregate.
- **Decimal precision and scale through project, aggregate and union-cast.** This is the
  one that earns the task: `avg`'s state columns were once typed backwards and per-node
  bytes could not show it, because both engines derive them from the same plan schema — so
  CPU and GPU agreed on the same wrong number and only a real divide would have diverged.

## T8 — what validation still owes

- The generic structural pass, over and above the per-node checks that exist.
- Manually constructed wrong combinations: each rule turned red by an input built to break
  it, per the reviewer's anchor that a guard which cannot go red is not a guard.
- Validation run over every canonized corpus plan as a standing check, not as a one-off.
- Defects in the checks themselves, including any the reviewer reported against
  `ENS-bp-plan-skeleton` and I deferred here.

**Constraints.** Every committed golden plan passes the validation this task adds. A
rejection is a planner defect until shown otherwise: stop, report it, and fix planning —
never weaken the check to fit the plan, and never regenerate a golden to silence one. The
expectation is still that no plan moves, so a golden that does move is a deliberate decision
taken with the human rather than a side effect of the regen. A test that only passes is worth
less than one shown to fail on the defect it guards.

**Verification bar.** Every rule in `validate_schemas_and_partitions` has an input that turns
it red. Decimal fidelity is asserted at each step of project → aggregate → union-cast. Every
committed golden plan passes validation, and a golden moves only where a planner fix required
it. `test_ci_coverage` names whatever targets appear.


Merged 2026-08-18 as PR #121.


---

<!-- archived from llm-wiki/tasks/join_capability_recipe.md -->

# join capability through the recipe plan (T0 extension)

**Goal.** Establish, by execution rather than argument, that the frozen FlatBuffers schema
and C++ operators can run **every** join mode in the batch-partitioned model — with the
build side complete and the probe side arriving in batches — and write the per-mode
lowering down where the implementation tasks will read it.

**What was built** (all in `scripts/exec_model/`, coordinator-owned):

- `operators/join_types.py` — `JoinType` in the fbs vocabulary and the capability matrix as
  one function, so no backend can hold a different opinion of it.
- `operators/cudf_calls.py` — the cuDF calls `cpp/src/operators/join.cpp` makes, at their
  own signatures: joins that return gather maps, `gather` with its out-of-bounds policy,
  `scatter`, `apply_boolean_mask`, `cross_join`.
- `operators/recipe.py` — the fb node structs, a handle registry that consumes on read as
  `NodeSession` does, and the node implementations mirroring join.cpp branch for branch.
- `operators/recipe_join.py` — the second join backend: every call answered by emitting fb
  seqs and making `execute_node` calls.
- `operators/joins.py` — the pandas backend, widened from five join types to all nine plus
  cross and nested loop.

**Constraints.** The two backends share no join code; agreement between them is the
evidence. The recipe backend may not reach for python where the frozen surface has no
answer — it names the gap and counts what working around it costs (`copy_handle`).

**Verification bar.** `scripts/exec_model/tests/test_end_to_end.py`: every join mode
against a SQL oracle, on both backends, at five batching/partitioning configs and across
the layout injector's presets; the emitted seq sequence asserted per mode against the
spec's table; the per-batch copy counts asserted per family; every refused shape refused
loudly on both backends. Whole prototype suite green, ~90 s.

**Outcome.** Recorded in the spec's [join capability
matrix](../tasks/batch_partitioned_executor.md#join-capability-matrix): every mode is expressible,
the streamed-probe copy cost is [#152](../tickets.md#t152) quantified per family, and one
shape turned out to be a defect in the shipping engine rather than a limit of the mode —
[#153](../tickets.md#t153).


---

Merged 2026-08-04 as PRs #112 / #115 / #114.


---

<!-- archived from llm-wiki/tasks/build-test-flags.md -->

# Task: build-test.sh flag surface, failure semantics, and the regen guard

On `ENS-test-exec-mode`. Touches `scripts/build-test.sh`, `scripts/build-test-shadgpu.sh`,
`peacockdb-core/tests/test_plan_bytes.rs`, `llm-wiki/build-test.md`. The rules this
implements are already in `coding-style.md` ("Bash: the flag set is an interface, and
failure is fatal", b05a631) — that bullet was written from this script's defects.

No golden may move. No test may be added, removed or re-tiered.

## A — flag surface

1. **Delete `--cpu`.** It sets the default and has no callers anywhere in the repo.

2. **`--gpu` and `--rust-only` are mutually exclusive** and must be *rejected*, not
   resolved by argument order. Today:
   - `--rust-only --gpu` → `MODE=gpu` with `RUST_ONLY=1` still live, so the run branch
     sets `LD_ENV=":"` and the GPU binaries never get `LD_LIBRARY_PATH` — they fail to
     resolve `libpeacock_gpu` and it reads as a product fault.
   - `--gpu --rust-only` → the reverse, GPU silently ignored.
   Same two flags, opposite outcomes, no warning. Error naming the contradiction.

3. **Replace `--push-testdata` / `--pull-testdata KIND[,KIND]`** with per-kind flags:

       --push-{parquet,queries,goldens,duckdb-profiles,duckdb-dynfilters}
       --pull-{parquet,queries,goldens,duckdb-profiles,duckdb-dynfilters}

   The point is structural, not cosmetic: the argument parser *becomes* the validator.
   No comma splitting, no kind lookup at the call site, and an unknown kind is just an
   unknown flag caught by the existing `*) usage` arm before any side effect. The
   multi-kind form is generality nobody uses — every documented invocation moves exactly
   one kind. `testdata_dirs_for_kind` stays as the kind→dirs map; only its use as a
   validator goes.

4. **`--fetch-goldens` → `--pull-goldens`.** A rename, not an alias. Today it appends
   `goldens` to `PULL_TESTDATA`, so passing it alongside `--pull-testdata goldens` pulls
   twice. Keep the current ordering property: the flag must be resolved *before* the
   `--host` requirement check, so `--pull-goldens` alone still demands `--host`.

5. **`--rsync` → `--push-binaries`, in BOTH scripts.** `--rsync` names the tool rather
   than the intent. `build-test.sh`'s own first line says it mirrors
   `build-test-shadgpu.sh`, so renaming one and not the other breaks the parallel that
   makes either readable after time away.
   - It **still pushes goldens.** They are part of the payload, not an optional kind:
     binaries shipped without the fixtures they assert against is the trap that produced
     110/110 "canonical file not found". Requiring `--push-goldens` alongside would
     replace a footgun with a guarded footgun.
   - `--push-goldens` remains useful and is not redundant with it — it is the subset
     operation, refreshing fixtures without rebuilding or reshipping binaries.

6. **Require an action.** `--host x` with no `--build`/`--push-binaries`/`--run`/push/pull
   currently does nothing and exits 0.

7. **Value-taking flags check their value exists.**

8. **`usage()` states what is deliberately absent**, so the next reader does not file it
   as a gap:
   - no `--pull-binaries` — binaries flow one way, built locally and shipped;
   - `embeddings-cache` is a per-host intermediate for the tpch vector datasets
     (`fetch_embeddings.sh`, ~1.8 GB, gitignored) and is deliberately not syncable.
   Also state the mode ladder: rust-only ⊂ cpu ⊂ gpu, and that a mode which builds more
   never runs less.

## B — failure semantics

1. **`set -euo pipefail`.** `pipefail` is load-bearing: `cargo test --no-run … | python3`
   currently takes python's status, so a cargo failure is caught only by the explicit
   emptiness check afterwards.

2. **An empty derived suite is an error.** Verified: `mapfile -t A < <(helper)` with a
   helper that outputs nothing yields a zero-length array, the `for` body never runs, and
   the script exits 0. A typo in the derivation would silently run no tests and report
   success.

3. **All validation before the first side effect** — a bad flag must fail before anything
   is built, shipped or deleted.

4. **The remote heredoc keeps its deliberate `set -e` omission.** Running every test
   binary and accumulating `rc` is correct: a failing C++ test must not skip the Rust
   ones. This is the stated exception `coding-style.md` allows; keep the comment that
   says why, and keep the non-zero exit at the end.

## C — deduplicate the sync layer

1. **One `sync_goldens()`.** "Push goldens" exists twice with different flags: the
   `--push-binaries` block uses `rsync -r --delete`, `--push-goldens` uses
   `rsync -a --delete`. Same intent, different metadata handling, no shared code.

2. **Push mirrors (`--delete`) uniformly; pull is additive uniformly** — and the
   asymmetry is deliberate, so document it rather than "fixing" it. The remote is a
   *partial* mirror: `testdata/goldens/` contains `tpch.sf40/` (16 CSVs) and sf40 lives
   on shad-gpu, so mirroring downward from verda would delete fixtures that host never
   had. The destination is a git working tree.

   Known consequence, accepted: a regen deletes a `.result.txt` when a result exceeds
   256 KB (`maybe_write_result_golden`), and an additive pull cannot propagate that. The
   deletion is already announced on stderr and reaches the operator through the ssh
   heredoc — that is the handling, not a `--delete` flag armed for one rare case.

3. **Drop the `*.txt` filter on the goldens pull** — after A/B and D, not before. Its real
   job is keeping `plan_bytes.sha256` out of the round trip, which becomes the self-guard's
   job in D; keeping both would be two mechanisms for one invariant with the weaker one in
   the wrong place. Removing it also stops silently dropping the 16 sf40 CSVs.

4. **Clear `cpp/install/rust-tests` before staging**, matching `build-test-shadgpu.sh`.
   `build-test.sh` does not, so orphaned binaries from a previous mode accumulate;
   today that is mitigated only by running binaries by explicit name.

## D — move the regen guard into the test

`--update-canonical` exports `UPDATE_CANONICAL=1` to every staged binary, so the run set
doubles as the regen set and `regen_excluded()` subtracts `test_plan_bytes` back out.
That protects one invocation path only: `UPDATE_CANONICAL=1 cargo test --features
rust-only -p peacockdb-core --test test_plan_bytes` — the exact command the golden's own
header prints — still rewrites `plan_bytes.sha256` silently.

Move the refusal into `test_plan_bytes.rs`: under `UPDATE_CANONICAL`, refuse unless a
dedicated override (`PEACOCK_REGEN_PLAN_BYTES=1`) is also set, panicking with the reason —
the digests are the wire-format guard, the C++ side reads those bytes, and regenerating
rewrites the evidence instead of failing. Then **delete `regen_excluded()`**; the script
stops carrying knowledge about a test's internals.

`test_cost_model` stays in the regen set. That inclusion is a fix, not a risk: `.cost.txt`
derives from `.cpu.txt`, which a regen rewrites, so the old six-target list left every
`.cost.txt` stale and `test_cost_model` went red immediately after a "successful" regen.

## E — comments and docs

- Header comment says "Two suites" and then lists three.
- The `PUSH_TESTDATA` comment lists four kinds; `usage()` lists five (`duckdb-dynfilters`).
- The goldens-push comment justifies itself entirely in verify terms and disposes of the
  regen case in a parenthetical, which reads as "this push is redundant when
  regenerating" — the opposite of what is true. State both reasons: for a verify run the
  binaries assert against these files; for a regen the push establishes the baseline, so
  the pulled-back set is local-committed ∪ regenerated rather than remote-leftovers ∪
  regenerated, and `--delete` is the mechanism.
- `build-test-shadgpu.sh:176` references "the goldens that build-test.sh's --rust-only
  mode used to skip" — check it still says something true.
- `llm-wiki/build-test.md`: the verda row, the shad-gpu row (`--rsync` → `--push-binaries`),
  the golden-regen bullet, and a line recording that `embeddings-cache` is not syncable.

## Sequencing

**A+B → D → C → E.** A/B is the interface and the failure model and touches everything
later; D must land before C3; C is mechanical once D is in; E last, so the docs describe
the final state rather than an intermediate one.

## Verification

- `bash -n` on both scripts.
- A flag matrix: for each mode, print the derived suite and assert it matches; for each
  rejected combination, assert it errors non-zero. Include `--gpu --rust-only` in both
  orders, `--host` with no action, and a value-taking flag with no value.
- Prove the empty-suite error fires (temporarily break the derivation, confirm non-zero,
  restore).
- One real `--rust-only --build` to prove staging still produces binaries.
- `git status --porcelain testdata/goldens` must be empty at the end.

## Out of scope

`maybe_write_result_golden`'s discarded `remove_file` result and its unconditional "no
golden" message — a ticket, not this task. The `--cpu`/`--rust-only` naming axis is
resolved by A1/A2 and needs no further rename.


---

<!-- archived from llm-wiki/tasks/widget-golden-links.md -->

# Task: link CPU ✓ cells to their goldens; small-font Query and Σout columns

Branch `ENS-widget-golden-links` (off `ENS-test-exec-mode`). Cost-report widget only —
`cost-report/src/main.rs`. No changes to the registry CSV, the test suite, or any golden.

## 1. The premise, verified

Every `enabled` cell in `ftc_tp1`, `ftc_tp8` and `partitioned_cpu` **has a committed
`.cpu.txt`**. `assert_cpu_cost_canonical` runs unconditionally on every CPU macro
invocation (`common/mod.rs:847`), before and independent of the `ResultGolden` keyword —
that keyword gates only `.result.txt`. So a `✓` in those three columns always has a
golden to point at.

This does *not* extend to the GPU columns: those read the CPU golden rather than owning
one, so leave `full_table_gpu` / `partitioned_gpu` cells alone.

## 2. Which golden each ✓ points at

The device label is **not** in the CSV, and it is not one-per-column:

| Column | Golden label |
|---|---|
| `ftc_tp8` | `full_table-tp8-mini` |
| `partitioned_cpu` | `partitioned-tp8-standard` |
| `ftc_tp1` | `full_table-tp1-standard` |

**Corrected 2026-08-04.** This section originally said `ftc_tp1` is tp1-standard "except
`scan_limit`, which is `full_table-tp1-mini`". That is wrong, and the developer caught it:
`scan_limit` has **both** goldens. It is registered twice — `tp1_mini` at
`test_cpu_full_table.rs:24` and `tp1_standard` at `:188` — and `column_for` keys on the tp
count, not the memory tier, so both land in the single `ftc_tp1` cell. It is not the
exception to the column's label; it is the one query whose cell aggregates two runs.

**Decision: one label per column, `full_table-tp1-standard`.** It is the label every tp1
row uses including `scan_limit`, so the link target is predictable from the column alone.
Drop the tp1-mini candidate rather than carrying config that nothing can reach — this repo
already treats unreachable config as a hazard in its own right (`GOLDEN_INVARIANT_EXEMPT`
and `INTENTIONALLY_NOT_IN_CI` both carry staleness assertions for exactly that).

What a cell aggregating two runs should render is a real question, but a hyperlink cannot
express it and the answer would change the cell shape. Out of scope here; raise it as its
own task if the tp1-mini run ever needs to be reachable from the widget.

The fail-loud check below is what makes dropping the candidate safe: a future query
registered ONLY at tp1-mini has no golden under the single label, so the widget fails
naming it instead of rendering a dead link. That is strictly better than a silent second
candidate, because it forces the decision rather than guessing.

Link target: the same `links.golden_url(canon_rel, stem, "<label>.cpu.txt")` helper the
Σout cell uses, so dry runs with no sha degrade to plain text exactly as they do today.

Only `✓` becomes a link. `~`, `✗` and `—` stay plain — there is no golden behind them.

## 3. Small font

- **Query column:** non-numeric queries only — `aggregate_groupby`, `scan_limit`,
  `shuffle_stddev`, `hash_join`, … The discriminator already exists: `Row::number` is
  `None` for exactly these (it is `Some(n)` for `q<N>`). Numbered queries keep today's
  size.
- **PeacockDB Σout and DuckDB Σout:** small font for **both the header and the values**.
  The Ratio column is not in scope.

Both renders. They use different mechanisms and the difference is load-bearing: the HTML
report can use a CSS class (`th.modeh` is the precedent), while the PR-comment table must
use `<sub>` because GitHub strips `class`/`style` — see the `mode_cells_md` doc comment.

## 4. Watch for

- `MODE_COLUMNS` and `registry.rs::COLUMNS` are the CSV header contract. This task is
  display-only: do not touch either, and do not change any cell value.
- `ftc_cell()` currently renders `tp1✓ tp8✓` as one string inside a single `<td>`. Both
  glyphs need to become independently linkable, so that function has to return markup
  rather than a plain label — check its callers in both renders before changing its shape.
- `CPU_DEVICE` is the const the Σout cells resolve through. The new per-column labels are
  related but not the same thing; do not fold them together in a way that makes a Σout
  change silently move a mode link.

## 5. Verification

- `cargo test -p cost-report` — the widget's own unit tests, including the
  `mode_cells_md` shape asserts.
- `scripts/cost-report-preview.sh` and eyeball the generated HTML: a linked `✓` per
  enabled CPU cell, `scan_limit`'s tp1 link resolving to `full_table-tp1-mini`, small-font
  micro-query names, small-font Σout headers and values.
- Confirm the PR-comment render still parses as HTML on GitHub (`<sub>`, no class/style).
- No golden, CSV or test-suite file appears in `git status`.


---

<!-- archived from llm-wiki/tasks/test-exec-mode.md -->

# Task: execution mode explicit in test macros and golden filenames

Branch `ENS-test-exec-mode` (off `ENS-llm-wiki`). Pure refactor: **no behavior change, no
golden regeneration, no coverage change**. Every test that runs today must still run, with
the same assertions, reading the same bytes from renamed files.

## Why

Two things are inferred today that should be stated:

1. `common/mod.rs::partition_mode(device)` maps the string `"tp8-standard"` to
   `PartitionMode::RealMultiPartition` and everything else to `SinglePartition`. A device
   label silently picks an executor. Adding a device (say `tp8-mini` as a real 8-way tier,
   which #91 wants) would route it to the wrong executor with no diff to the routing code.
2. Golden filenames carry only `tp<N>-<tier>`, so `q15.tp8-mini.cpu.txt` does not say which
   CPU executor produced it. The mode is real: the same plan at tp8 produces a different
   per-node cost tree under full-table vs partitioned execution.

`node13` is an obsolete task number used as an executor name. It goes.

## 1. Macro renames

| Old | New |
|---|---|
| `cpu_result_test!` | `cpu_full_table_result_test!` |
| `cpu_result_approx_test!` | `cpu_full_table_result_approx_test!` |
| `cpu_node13_result_test!` | `cpu_partitioned_result_test!` |
| `cpu_node13_result_approx_test!` | `cpu_partitioned_result_approx_test!` |
| `gpu_test!` | `gpu_full_table_test!` **and** `gpu_partitioned_test!` |

`cpu_result_error_test!` / `cpu_result_fits_test!` keep their names — decision, not
oversight: they take a raw budget rather than a device, read no mode-tagged golden, and the
resident-OOM enforcer they drive is full-table-only by construction (#91 tracks porting it).
Say that in a one-line comment where they are defined so the asymmetry doesn't read as a
miss.

Generated test-fn names include the mode, so two modes at the same device can never collide:
`cpu_full_table_tpch_sf1_q1_tp8_mini`, `cpu_partitioned_tpch_sf1_q6_tp8_standard`,
`gpu_full_table_tpch_sf1_q1_full_table_tp1_standard`. Grep `pipeline.yml` and the
`build-test*.sh` scripts for `--exact` / filter strings built from the old names and fix them.

## 2. Call-site shapes

CPU — mode from the macro name, device stays `tp<N>_<tier>`:

    cpu_full_table_result_test!(tpch, 1, q1, tp8_mini, no_result_golden);
    cpu_partitioned_result_test!(tpch, 1, q6, tp8_standard, result_golden);

GPU — mode from the macro name; the device argument is the **combined golden label**, which
is the golden filename component verbatim:

    gpu_full_table_test!(tpch, 1, q1, full_table_tp1_standard, golden_exact);
    gpu_partitioned_test!(tpch, 1, q3, partitioned_tp8_standard, golden_exact);

So `gpu_full_table_test!(tpch, 1, q1, full_table_tp1_standard, …)` reads
`goldens/tpch.sf1/q1.full_table-tp1-standard.{cpu,result}.txt` — reconstructible from the
invocation with no lookup. The GPU run config splits cleanly: `PartitionMode` comes from the
macro name, `tp` + budget are parsed out of the label. A crossed pair
(`gpu_full_table_test!` with a `partitioned_…` label) is visible at the call site; assert it
cannot pass silently — the label's mode prefix must equal the macro's mode.

## 3. Last parameter: bool → enum

The trailing `$gen:literal` bool becomes an ident backed by a real enum in `common/mod.rs`,
so the call site names the artifact it produces:

    pub enum ResultGolden { Write, Skip }   // `result_golden` | `no_result_golden`

Mirror `gpu_result_mode`'s shape (keyword → enum, unknown keyword panics with the accepted
set). Keep the existing invariant comment about when writing is correct — it is the useful
part of that parameter.

## 4. Delete the implicit routing

- Delete `partition_mode(device: &str)` outright.
- `assert_cpu_results_match_datafusion`: replace `use_node13: bool` with an explicit
  execution-mode parameter that carries both the executor choice and the `PartitionMode`.
- `assert_gpu_query` / `assert_gpu_nodes_match_golden`: take the `PartitionMode` from the
  caller (the macro), never from a label.
- `plan_is_node13_executable` → `plan_is_partitioned_executable`, unchanged behavior; it
  stays the safety assert on the partitioned path.
- `registry.rs::column_for`: kind `"node13"` → `"partitioned"`; kind `"gpu"` splits into
  `"gpu_full_table"` / `"gpu_partitioned"` mapping straight to their columns instead of
  sniffing `device.starts_with("tp1")`. The `ftc` arm keeps its tp1/tp8 split — that one
  reads a `tp` count out of a `tp` label, which is parsing, not routing.

### The plan tier keeps a label → mode lookup (amended 2026-08-04)

"Delete `partition_mode` outright" was wrong about the plan tier, and the developer caught
it. `test_plan_bytes.rs::corpus()` builds its entire corpus by reading `.plan.txt` filenames
off disk and splitting the device out of the stem — there is no call site to state a mode at,
and `plan_bytes.sha256` keys on `<query>.<device>`, which this task freezes. The mode is
load-bearing there: `shuffle-additive` @ tp8-standard is the one plan golden whose shape
depends on `RealMultiPartition`.

So: `partition_mode` is deleted from the **execution** path — CPU results, GPU query, GPU
nodes, and the oracle `CpuExecutor` all take the mode from the macro name. One narrowly
scoped `plan_partition_mode(device)` survives for the plan tier, documented as that tier's
label contract and made **exhaustive** — an unknown label panics instead of falling through
to `SinglePartition`. That catch-all is the actual hazard this task's "Why" names, and
because `corpus()` reads labels off disk, exhaustiveness is self-enforcing: a new plan golden
with an unlisted device fails loudly instead of silently planning single-partition.

Keep it to **one** mechanism. Do not also add an explicit mode argument to `query_plan_test!`
or `plan_for` — two sources for the same fact is how a plan golden gets generated under one
mode and its byte digest under the other. Concretely: `test_query_plan_misc.rs:27`
(`plan_for(…, "tp8-standard")`) is unchanged, while `test_cpu_executor_misc.rs:41` is an
execution-tier site and does pass `PartitionMode::RealMultiPartition` explicitly.

The `coding-style.md` entry names this as a deliberate exception, so the surviving function
reads as scoped rather than missed.

**CSV columns and `testdata/cost-registry.csv` do not change.** `COLUMNS` is the committed
fixture's header contract (`load_csv` asserts on it) and `MODE_COLUMNS` in cost-report keys
off the same names. Renaming `ftc_tp1` is a separate change; not this one.

## 5. Golden filenames

New: `<query>.<mode>-<tp>-<tier>.{cpu,cost,result}.txt`, e.g.
`q15.full_table-tp8-mini.cpu.txt`.

**Unchanged:** `.plan.txt` (plan shape is executor-independent, and
`goldens/plan_bytes.sha256` pins those names) and `.duckdb_cost.txt` (oracle, no peacock
executor involved).

`git mv` only — do not regenerate. The mapping is total and collision-free because no device
label is used by both modes today (verified: `tp8_standard` appears only under
`cpu_node13_*`; every other label only under `cpu_result_*`):

| Old | New | files |
|---|---|---|
| `*.tp8-mini.{cpu,cost}.txt` | `*.full_table-tp8-mini.…` | 129 + 129 |
| `*.tp1-mini.{cpu,cost}.txt` | `*.full_table-tp1-mini.…` | 1 + 1 |
| `*.tp1-standard.{cpu,cost,result}.txt` | `*.full_table-tp1-standard.…` | 110 + 110 + 104 |
| `*.tp8-standard.{cpu,cost,result}.txt` | `*.partitioned-tp8-standard.…` | 18 + 18 + 14 |

634 files total across `testdata/goldens/{tpch.sf1,tpcds.sf1}`. Counts are mine and worth
re-deriving — if yours differ, say so before renaming rather than after. Verify content is
untouched: the multiset of file *contents* must be identical before and after (e.g. compare
sorted `sha256sum` values of the renamed set).

## 6. File reorganization — by mode, not by memory tier

| Now | Becomes |
|---|---|
| `test_cpu_executor.rs` (tp8-mini + tp1-mini + tp8-standard) | `test_cpu_full_table.rs` (tp8-mini, tp1-mini, tp1-standard) |
| `test_cpu_h200.rs` (tp1-standard) | `test_cpu_partitioned.rs` (tp8-standard) |
| `test_gpu.rs` | `test_gpu_full_table.rs` + `test_gpu_partitioned.rs` |

`test_cpu_executor_misc.rs` / `test_gpu_executor_misc.rs` keep their names; they only need
their golden paths updated.

Registry ownership follows the files — update both the `registry_matches_csv_*` fns and their
doc comments:

- `test_cpu_full_table.rs` → `ftc_tp1` + `ftc_tp8`
- `test_cpu_partitioned.rs` → `partitioned_cpu`
- `test_gpu_full_table.rs` → `full_table_gpu`
- `test_gpu_partitioned.rs` → `partitioned_gpu`

The cross-binary caveat in `test_cpu_h200.rs`'s doc comment (scan_limit registered at
tp1-mini in one binary, tp1-standard in another) is moot once both live in
`test_cpu_full_table.rs` — delete it rather than carrying it forward. The note at the foot of
`test_gpu.rs` explaining why the cross-mode invariant lives in `test_query_plan.rs` is still
load-bearing; keep it on whichever GPU file you consider primary.

`test_gpu.rs`'s file-level doc explains the merged one-run-asserts-both design — that belongs
in both new GPU files or in the macro doc, not dropped.

## 7. Consumers that read the old names

- `cost-report/src/main.rs`: `CPU_DEVICE = "tp8-mini"` → `"full_table-tp8-mini"`. Check the
  surrounding comment about scan_limit being tp1-mini-only (~L345) — it still holds, but its
  wording names devices.
- `registry.rs::assert_cross_mode_golden_invariant`: the `("full_table_gpu", "tp1-standard")`
  / `("partitioned_gpu", "tp8-standard")` pairs become the new labels. This one is
  load-bearing — it is what catches an enabled GPU mode with no CPU golden.
- `test_ci_coverage.rs`: four new test-target names in, two out. This gate is the reason a
  renamed binary can't silently drop out of CI.
- `.github/workflows/pipeline.yml`: rust test target lists in the cpu-cpu tier and the
  GPU-remote job; `scripts/build-test.sh` and `scripts/build-test-shadgpu.sh` build/stage
  test binaries one `--test` at a time and name them.
- `test_cost_model.rs` globs `*.cpu.txt` and derives the sibling `.cost.txt` by string
  replace — should need no change; confirm rather than assume.
- `llm-wiki/build-test.md` and `architecture.md`: test-file names, device labels, golden
  naming. Same commit.

## 8. Wiki changes owed by this task

`coding-style.md` — new antipattern entry, in the voice of the existing thread-local one:
implicit routing from a label. `partition_mode(device)` turned the string `"tp8-standard"`
into `RealMultiPartition`; the executor a test ran was a side effect of how its golden was
named, and a new device label would have silently taken the wrong path with no diff to the
routing code. State the mode at the call site and pass it as a parameter.

`build-test.md` — one sentence: a refactor that must not change behavior is verified with a
representative subset (one query per mode/tier per binary) plus the full rust-only tier, not
a full CPU/GPU suite run; the goldens are the invariant.

## 9. Verification — subset only, explicitly

Do **not** run the full CPU or GPU suite.

1. The rust-only **golden/meta gates**, must be green — named by target, not by package:

       cargo test --features rust-only -p peacockdb-core \
         --test test_plan_bytes --test test_cost_model --test test_ci_coverage

   plus the registry tests, which live inside the execution targets and must be filtered to
   (`--test test_query_plan -- registry_`, and the same for `test_cpu_full_table` /
   `test_cpu_partitioned`). Together: the CSV contract, the cross-mode golden invariant, and
   the CI-coverage gate. Cheapest and highest-value gate for this change; seconds, not
   minutes.

   **Corrected 2026-08-04** — this item originally read `cargo test --features rust-only -p
   peacockdb-core` with no `--test`, which contradicted this section's own headline: nothing
   cfg's the CPU execution targets out of the rust-only build, so package-wide sweeps all 241
   + 18 of them. `--features rust-only` selects a *build*, not a tier; only `--test` selects
   the tier. `build-test.md`'s table lists the bare package command as the rust-only loop,
   which is what made this easy to mis-transcribe — the §8 sentence owed to `build-test.md`
   should make the distinction explicit rather than just saying "run a subset".
2. `cargo build --tests` clean, no new warnings.
3. CPU subset — a few per binary, chosen to cover each device label:
   `test_cpu_full_table` at tp8-mini, tp1-mini (scan_limit), tp1-standard;
   `test_cpu_partitioned` at tp8-standard (include one approx: tpcds q17).
4. GPU subset on shad-gpu — one per new binary is enough (e.g. tpch q1 full-table, tpch q6
   partitioned). This proves golden-path resolution on both binaries; it is not a
   correctness run.
5. **Test-count invariant**: capture `--list` output for the affected binaries before and
   after and show the sets correspond 1:1 under the intended renames. A refactor that
   silently drops a test is the failure mode here, and neither a green subset nor
   `test_ci_coverage` alone would catch a dropped `gpu_test!` line.

## Out of scope

CSV column renames; `ftc` as a kind name; porting the resident-OOM enforcer to the
partitioned driver (#91); regenerating any golden; re-tiering any test.

## 10. Follow-up (added 2026-08-04): fold the approx variants into an oracle argument

Human's instruction, on this same task rather than a new one.

The `_approx_` macros exist only to pass `Some(1e-12)` instead of `None` for `rel_tol`.
That is a property of how the result is compared, not a different kind of test, and
spelling it in the macro name means two names per mode where one plus an argument says
more. Delete `cpu_full_table_result_approx_test!` and `cpu_partitioned_result_approx_test!`
and add a **second-to-last** argument to the two surviving macros:

    cpu_full_table_result_test!(tpch, 1, q1, tp8_mini, data_fusion_exact, no_result_golden);
    cpu_partitioned_result_test!(tpcds, 1, q17, tp8_standard, data_fusion_approximate, result_golden);

Backed by a real enum in `common/exec_mode.rs`, keyword-mapped like `ResultGolden` and
`gpu_result_mode` (unknown keyword panics naming the accepted set):

    pub enum CpuOracle { DataFusionExact, DataFusionApproximate }

`DataFusionExact` → `rel_tol = None`, `DataFusionApproximate` → `Some(1e-12)`.

The name states what the oracle IS, which the old name did not: **both** variants compare
against a live plain-DataFusion run at `target_partitions = 1` (`build_session_state(1)`);
only the float tolerance differs. Nothing about the oracle changes — this is a rename of
an existing bool-in-disguise.

Move the 1e-12 rationale — float summation reassociates across partitions at tp>1, ~1 ULP,
while the `output_bytes` cost golden stays exact because a ULP does not change byte width —
onto the `DataFusionApproximate` variant, where `ResultGolden` and `GpuResultMode` keep
theirs. Do not leave it stranded on a deleted macro.

**The five call sites that become `data_fusion_approximate`** (every other CPU call site
takes `data_fusion_exact`):

| File | Query |
|---|---|
| `test_cpu_full_table.rs:65` | tpch `shuffle_stddev` @ tp8-mini |
| `test_cpu_full_table.rs:85` | tpcds `q14` @ tp8-mini |
| `test_cpu_full_table.rs:109` | tpcds `q39` @ tp8-mini |
| `test_cpu_full_table.rs:226` | tpch `shuffle_stddev` @ tp1-standard |
| `test_cpu_partitioned.rs:38` | tpcds `q17` @ tp8-standard |

Note `tpcds q14 @ tp1-standard` is **exact** today and stays exact — at tp1 there is no
reassociation. Converting it along with its tp8-mini sibling would silently loosen a check.

**Test-fn names do not change.** The approx macros already generated the same
`cpu_<mode>_<ds>_sf<sf>_<query>_<device>` pattern as the exact ones, so the 259/259
correspondence must still hold exactly. Re-run the `--list` comparison and say so; a
changed count here means something other than the intended edit happened.

Verification is §9 unchanged, and no golden may move: `rel_tol` affects only the result
compare, never `assert_cpu_cost_canonical`.
