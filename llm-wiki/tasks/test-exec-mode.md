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
