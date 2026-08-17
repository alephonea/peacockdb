# peacockdb build & test

Code and tests are authoritative; this page maps them.

## Test categories

**Grand total: 1018 test cases — Rust 733, C++ 37, Python 248.**

**Runs** — `cpp-cpu` = pipeline.yml's cpp-cpu job, both cuDF legs · `cost-report` = the
cost-report job · `shad-gpu` = CI GPU job on the remote host, `--test-threads=1` ·
`manual` = no CI step runs it · `2gpu` = manual, host with ≥2 GPUs (verda-gpu) ·
`validate-large` = validate-large.yml manual dispatch. Everything except the `shad-gpu`
and `2gpu` rows also runs locally; large CPU batches go to verda.

| Category (lang) | Why | Examples | Runs | N |
|---|---|---|---|---|
| Plan goldens, tp8 (Rust) | plan text vs `.plan.txt` — planner/rule drift, no execution; every one is tp8 (#133) | [scan_limit](../peacockdb-core/tests/test_query_plan.rs#L9), [shuffle_additive](../peacockdb-core/tests/test_query_plan.rs#L46) | cpp-cpu | 128 |
| Plan wire format (Rust) | FlatBuffers bytes are the C++ contract | [serialized_plan_bytes_are_stable](../peacockdb-core/tests/test_plan_bytes.rs#L129) | cpp-cpu | 1 |
| Plan serializer (Rust) | per-operator serialize/deserialize round trip | [left_mark_join](../peacockdb-core/tests/test_plan_serialiser.rs#L445), [scan_row_group_pruning](../peacockdb-core/tests/test_plan_serialiser.rs#L289) | cpp-cpu | 12 |
| Plan/rule bespoke (Rust) | planner + memory-model cases the macros can't express | [budget_reduces_batch_size](../peacockdb-core/tests/test_query_plan_misc.rs#L169), [mergeable_agg_state](../peacockdb-core/tests/test_query_plan_misc.rs#L26) | cpp-cpu | 10 |
| CPU exec, full_table tp8-mini (Rust) | tp8 plan collapsed to one stream at 2 GiB: result + per-node cost tree vs DataFusion | [hash_join](../peacockdb-core/tests/test_cpu_full_table.rs#L16), [left_join](../peacockdb-core/tests/test_cpu_full_table.rs#L17) | cpp-cpu | 129 |
| CPU exec, full_table tp1-standard (Rust) | same at 12 GiB, single-partition — the tier the GPU shares goldens with | [scan_limit](../peacockdb-core/tests/test_cpu_full_table.rs#L188), [aggregate_groupby](../peacockdb-core/tests/test_cpu_full_table.rs#L190) | cpp-cpu | 110 |
| CPU exec, full_table tp1-mini (Rust) | one probe that tp1 also holds at 2 GiB | [scan_limit](../peacockdb-core/tests/test_cpu_full_table.rs#L24) | cpp-cpu | 1 |
| CPU exec, partitioned tp8-standard (Rust) | real 8-way: N partitions stay live across nodes; the GPU tier's CPU oracle | [q6](../peacockdb-core/tests/test_cpu_partitioned.rs#L17), [shuffle_additive](../peacockdb-core/tests/test_cpu_partitioned.rs#L21) | cpp-cpu | 17 |
| CPU exec, resident OOM micro (Rust) | the enforcer trips at a tight budget, and the boundary is real | [q78 oom](../peacockdb-core/tests/test_cpu_oom.rs#L24), [q7 fits](../peacockdb-core/tests/test_cpu_oom.rs#L27) | cpp-cpu | 3 |
| CPU exec bespoke (Rust) | wrapper stripping, routing predicate, instrumented stats | [test_execution_strips_gpu_nodes](../peacockdb-core/tests/test_cpu_executor_misc.rs#L102) | cpp-cpu | 5 |
| CPU node-by-node parity (Rust) | the unified walk must equal the recursive path, byte for byte | [cpu_node_executor_matches_recursive](../peacockdb-core/tests/test_node_executor.rs#L16) | cpp-cpu | 1 |
| GPU exec, full_table tp1-standard (Rust) | one run asserts per-node rows+cost vs `.cpu.txt` AND the final result | [scan_limit](../peacockdb-core/tests/common/gpu_cases.inc#L29) | shad-gpu | 110 |
| GPU exec, partitioned tp8-standard (Rust) | same on the real 8-way path, with per-partition asserts | [q6](../peacockdb-core/tests/common/gpu_cases.inc#L154) | shad-gpu | 17 |
| GPU↔comet murmur3 (Rust) | the linchpin gate: both sides place every row in the same partition, bit-exact | [gpu_spark_partition_ids_match_comet_live](../peacockdb-core/tests/test_inc2_conformance.rs#L131) | shad-gpu | 10 |
| GPU all-at-once smoke (Rust) | whole-plan `peacock_execute` FFI; retires with [#110](tickets.md) | [scan_nation](../peacockdb-core/tests/test_gpu_executor_misc.rs#L18) | manual | 6 |
| GPU per-node timing (Rust) | measures, asserts nothing: one `.benchmark.txt` per case, same list as the two GPU tiers ([`gpu_cases.inc`](../peacockdb-core/tests/common/gpu_cases.inc)) | [run_gpu_benchmark](../peacockdb-core/tests/peacock_gpu_benchmarks.rs) | manual | 127 |
| Cost-model goldens (Rust) | `.cost.txt` derivation from `.cpu.txt` × `cost_model.conf` | [cost_goldens_match_and_total_is_byte_identical](../peacockdb-core/tests/test_cost_model.rs#L36) | cost-report | 2 |
| Registry ↔ CSV (Rust) | each `cost-registry.csv` mode column matches the tests that exist, both directions | [full_table_columns](../peacockdb-core/tests/test_cpu_full_table.rs#L313), [partitioned_gpu_column](../peacockdb-core/tests/test_gpu_partitioned.rs#L46) | cpp-cpu ×4, shad-gpu ×2 | 6 |
| CI wiring guard (Rust) | every Rust target must be named by a CI step — CI does not glob | [every_rust_test_target_is_named_by_ci](../peacockdb-core/tests/test_ci_coverage.rs#L284) | cost-report | 2 |
| tp8 flip diagnostic (Rust) | prints would-be flips; a printer, no assertions | [diag_flip_audit](../peacockdb-core/tests/diag_flip_audit.rs#L135) | manual | 1 |
| Lib unit (Rust) | config tiers, batch-size rule, resident model | [tiers_are_strictly_increasing](../peacockdb-core/src/config.rs#L119), [nested_join_build_sides_stack](../peacockdb-core/src/resident.rs#L165) | cpp-cpu | 9 |
| Doctest (Rust) | the `CpuExecutor` rustdoc example still compiles | [CpuExecutor example](../peacockdb-core/src/lib.rs#L166) | manual — unlisted, see [#128](tickets.md) | 1 |
| FFI smoke (Rust) | the crate links; executor lifecycle | [test_executor_lifecycle](../peacockdb-ffi/tests/test_ffi.rs#L17) | cpp-cpu | 2 |
| Cost-report renderer (Rust) | glyphs, links, ratio bucket, regression gate, history | [bucket_threshold_is_1_4](../cost-report/src/main.rs#L1552), [regression_count_drives_exit_decision](../cost-report/src/main.rs#L1623) | cost-report | 23 |
| DuckDB cost extraction (Python) | classifier / pruning / dynamic-filter logic — fails CI before generation | [scan_count_mismatch_fails_loud](../testdata/test_duckdb_cost.py#L280), [compute_pruning_from_rowgroups](../testdata/test_duckdb_cost.py#L226) | cost-report | 41 |
| Exec-model prototype (Python) | the batch-partitioned scheduler over mock traits, plus pandas-backed operators checked against a single-shot oracle at five partitioning configs, and both limit lowerings — no project code | [test_a_join_in_its_build_phase_holds_back_its_probe_subtree](../scripts/exec_model/tests/test_scheduling.py), [test_a_root_adjacent_limit_stops_the_run_early](../scripts/exec_model/tests/test_limit.py) | cost-report | 188 |
| Exec-model prototype, TPC-H (Python) | the same drivers over real sf1 tables under a live resident budget, each plan re-run at every layout `LayoutInjector` can produce; needs the generated dataset, so it rides cpp-cpu rather than cost-report | [test_the_accumulator_is_what_makes_the_budget_bind](../scripts/exec_model/tests/test_tpch.py), [test_every_layout_gives_the_same_shuffled_join](../scripts/exec_model/tests/test_tpch.py) | cpp-cpu (25.02 leg) | 19 |
| C++ CPU/FFI unit | decimal binop typing, AST routability, lifecycle; no GPU needed | [DecimalScale.BinopOutputType](../cpp/tests/cpu/test_executor.cpp#L26), [AstRouting.IsAstAble](../cpp/tests/cpu/test_executor.cpp#L81) | cpp-cpu (`ctest -L cpu`) + shad-gpu | 5 |
| cuDF GPU smoke (C++) | the GPU is alive; the Spark-murmur3 kernel matches comet in C++ | [CudfGpu.SparkPartitionIdsMatchComet2ColWithNulls](../cpp/tests/gpu/test_cudf.cpp#L84) | shad-gpu | 3 |
| Plan-executor (C++) | hand-built plan IR through the C++ executor, node by node | [PlanExecutor.HashJoinNationRegion](../cpp/tests/gpu/test_plan_executor.cpp#L255) | shad-gpu | 12 |
| TPC-H sf40 bare-cuDF (C++) | hand-written cuDF pipelines vs DuckDB sf40; the benchmark vehicle | [TpchSf40.Q1GroupByAggregates](../cpp/tests/gpu/test_tpch.cpp#L216), [Q3JoinsGroupByTopN](../cpp/tests/gpu/test_tpch.cpp#L376) | shad-gpu (sf40 is a hard precondition) | 4 |
| TPC-H+V sf40 bare-cuDF (C++) | the same for the vector-embedding queries | [TpchSf40.Q11VectorBruteForce](../cpp/tests/gpu/test_tpchv.cpp#L326) | shad-gpu | 4 |
| Multi-GPU TPC-H (C++) | WorkerPool, hash_shuffle, per-device RMM pools across GPUs | [TpchSf40.Q3MultiGpu](../cpp/tests/gpu/test_multi_gpu_tpch.cpp#L460) | manual, 2gpu | 4 |
| Multi-GPU TPC-H+V (C++) | the same for the vector queries | [TpchSf40.Q10VectorCustomerTopNMultiGpu](../cpp/tests/gpu/test_multi_gpu_tpchv.cpp#L920) | manual, 2gpu | 4 |
| Multi-GPU basics (C++) | device-local streams + destruction on the owning worker | [BasicMultiGpu.CudfAndCuvsAcrossTwoGpus](../cpp/tests/gpu/test_basic_multi_gpu.cpp#L228) | manual, 2gpu | 1 |
| Dataset validators (Python) | row counts / clustering / embedding stats over whatever dataset they are pointed at; each check tags itself `EXHAUSTIVE` or `SAMPLED` | [validate_tpch.py](../scripts/validate_tpch.py), [check_s3_datasets.py](../scripts/check_s3_datasets.py) | cpp-cpu (sf1) · validate-large (sf40/sf200) | n/a¹ |

¹ Data-driven — the check count depends on the dataset and SF, so these are excluded from
the 891. Everything else in the repo that can be enumerated as a test case is counted.

Notes

- Execution mode is `full_table` | `partitioned` and comes from the macro name, never from
  the device label. Device labels: `tp1-standard`, `tp8-standard`, `tp8-mini`, `tp1-mini`;
  OOM tests take a raw budget (micro tier), not a label.
- CPU exec goldens: `<query>.<mode>-<tp>-<tier>.cpu.txt`, with derived `.cost.txt` and
  frozen `.result.txt` riding alongside. The GPU macros take the combined label
  (`full_table_tp1_standard`), so the golden filename is reconstructible from the call
  site. GPU result validation is `golden_exact` | `golden_approx` | `golden_approx_std` |
  `oracle` | `skip`.
- One test is quarantined, not deleted: the tp8-standard `shuffle_stddev` GPU case
  (#103), commented out in `test_gpu_partitioned.rs` with its goldens kept. It is not in
  the counts above.
- Why the three `manual` Rust targets run nowhere. `test_gpu_executor_misc` needs the linked
  C++/CUDA executor, so the CPU tiers cannot build it — and it is not staged for the GPU
  job either, which is the part nobody chose deliberately: it drives the all-at-once
  executor that #110 retires, so its 6 smoke tests are already scheduled to be migrated or
  dropped. `diag_flip_audit` is a diagnostic printer with no assertions, run by hand while
  #97/#95 gate the tp8 rollout; wiring it to CI would add a step that cannot fail.
  `peacock_gpu_benchmarks` measures rather than asserts, so there is nothing for a gate to
  go red on; correctness for the same case list belongs to the two GPU tiers, which share
  `gpu_cases.inc` with it.
- The doctest is different in kind from those three: they are exempted with a stated reason,
  it is merely unlisted. No CI step passes `--doc` and the meta guard enumerates only
  `--test` targets plus `--lib`, so nothing would go red if it broke (#128).

## Datasets

Host columns are audited, not assumed: they say whether the bytes were actually found
there (`ls` + `sha256sum`, 2026-08-05). Only tpch.minimal is in git; everything else is
generated or fetched per host. S3 is Nebius object storage, endpoint
`https://storage.eu-north1.nebius.cloud:443`, region `eu-north-1`.

| Dataset | Shape | <sub>local</sub> | <sub>verda</sub> | <sub>shad-gpu</sub> | S3 bucket | Used by |
|---|---|:-:|:-:|:-:|---|---|
| tpch.minimal | 5 tables, 19 MB, git-committed | ✓ | ✓ | ✓ | — | C++ plan tests, plan serializer, node executor |
| TPC-H+V sf1, external vectors | 8 tables, 987 MB; GloVe 100-d + DEEP1B 96-d | ✓ | ✓ | ✓ | — | most CPU/GPU rust tests |
| TPC-H+V sf1, synthetic vectors | same tables, hash-generated FLOAT[8] | — | — | — | — | CI only — regenerated every run |
| TPC-DS sf1 | 24 tables, 764 MB | ✓ | ✓ | ✓ | — | plan + CPU + GPU subsets |
| TPC-H+V sf40 | 8 tables, 40 GB | — | — | ✓ | `tpch-sf40` | `peacock_tpch(v)_tests`, multi-GPU (manual) |
| TPC-DS sf200 | 24 tables, 80 GB | — | — | ✓ | `tpcds-sf200` | no tests — S3 check, validate-large |
| TPC-H sf200 | not generated yet | — | — | — | `tpch-sf200` | nothing yet |
| embeddings cache | 1.8 GB local / 129 GB shad-gpu | ✓ | — | ✓ | — | generator input, not a test input |

Paths: `testdata/{tpch.minimal,tpch.sf1,tpcds.sf1,embeddings-cache}`; sf40 and sf200 live
outside the repo on shad-gpu, under `/home/info/peacock-datasets/testdata/`. **+V** means
the TPC-H tables carry vector columns — `part.p_text_embedding` and
`partsupp.ps_image_embedding` + `ps_text_embedding`; TPC-DS does not have embeddings.

The three sf1-class datasets are byte-identical on all three hosts (aggregate parquet
sha256 matches local). Notes on the rows above:

- **The sf1 parquet is generated** — `testdata/generate_testdata.sh` drives DuckDB, and
  `/tpch.sf*/`, `/tpcds.sf*/` are gitignored, so CI regenerates it every run. That is why
  the two sf1 rows differ: `--embeddings synthetic` is the default, so a vector query in
  CI runs against hash-generated FLOAT[8], not the vectors a dev host has.
- **The embeddings cache is a per-host intermediate and is NOT syncable** — no
  `--push`/`--pull` kind covers it; `fetch_embeddings.sh` is local-only and hard-guards
  against running on CI/verda/shad-gpu. Regenerate it where you need it, or ship the
  augmented parquet instead.
- **sf40 lives only on shad-gpu**, and its presence there is a hard precondition of the
  GPU job; the sf40 goldens are committed (see below), so CI compares without touching
  40 GB.
- verda reaches the tree through a `/media/data/peacockdb` symlink to
  `~/peacockdb` — the CPU test crates bake the testdata path at compile time (#49).

## Golden files

All goldens are committed, under `testdata/goldens/`: `tpch.sf1` (266 files), `tpcds.sf1`
(620), `tpch.sf40` (16), plus the single-file `plan_bytes.sha256` (140 digests). The
committed DuckDB profile inputs live beside them in `testdata/duckdb-profiles/{tpch,tpcds}`
(22 + 99) and `testdata/duckdb-dynfilters/{tpch,tpcds}` (22 + 99).

In the Golden column `…` stands for `<query>.<mode>-<tp>-<tier>`; the generator scripts
live in `testdata/`.

| Golden | Produced by | Depends on | Asserted by |
|---|---|---|---|
| `<q>.<device>.plan.txt` | plan tier with `UPDATE_CANONICAL=1` | sf1 parquet (schemas + row-group stats reach the plan) | plan goldens tier |
| `plan_bytes.sha256` | `test_plan_bytes` with `UPDATE_CANONICAL=1`<br>**and** `PEACOCK_REWRITE_PLAN_BYTES=1` | the same physical plans, serialized | <sub>serialized_plan_<br>bytes_are_stable</sub> |
| `….cpu.txt` | the CPU executor under `UPDATE_CANONICAL=1` — never the GPU | sf1 parquet | CPU exec tiers; the GPU tiers assert against it read-only |
| `….cost.txt` | derived from the sibling `.cpu.txt` text × `cost_model.conf` | `.cpu.txt`, `cost_model.conf` | CPU exec tiers + `test_cost_model` (re-derives and compares) |
| `….result.txt` | the CPU oracle under `UPDATE_CANONICAL=1`; deleted above 256 KB so the GPU test falls back to a live oracle | sf1 parquet | CPU exec tiers; GPU tiers in `golden_exact`/`golden_approx` mode |
| `<q>.duckdb_cost.txt` | `gen_duckdb_cost.sh --gen`<br>(DuckDB 1.5.4, `threads=1`, pyarrow 19.0.1) | committed pass-1 profiles ∩ pass-2 dynamic-filter bounds ∩ parquet row-group stats | the cost-report widget (directional signal, not a test) |
| `tpch.sf40/duckdb_<q>.csv`, `.count.csv` | `gen_duckdb_goldens.sh --sf 40`<br>on shad-gpu | sf40 parquet, query text from `tpch_query_sql.sh` | `peacock_tpch_tests` / `peacock_tpchv_tests` |

How they hang together — parquet at the top, goldens derived left to right:

```
testdata/{tpch,tpcds}-queries/*.sql
        │
        └── generate_testdata.sh (DuckDB, --embeddings synthetic|external)
                 ▼
        tpch.sf1 / tpcds.sf1   (parquet, gitignored)
          │
          ├── plan tier, UPDATE_CANONICAL=1
          │     ├──► <q>.<device>.plan.txt
          │     └──► plan_bytes.sha256     [also needs PEACOCK_REWRITE_PLAN_BYTES=1]
          │
          ├── CPU executor, UPDATE_CANONICAL=1   (the oracle; the GPU never writes)
          │     ├──► <q>.<mode>-<tp>-<tier>.cpu.txt
          │     │       └──× cost_model.conf ──► <q>.<mode>-<tp>-<tier>.cost.txt
          │     └──► <q>.<mode>-<tp>-<tier>.result.txt    [dropped if ≥ 256 KB]
          │
          └── gen_duckdb_cost.sh --gen  (DuckDB 1.5.4, threads=1, pyarrow 19.0.1)
                ├── pass 1, JFP off ──► duckdb-profiles/<bench>/<q>.json     (committed)
                ├── pass 2, JFP on  ──► duckdb-dynfilters/<bench>/<q>.json   (committed)
                └── duckdb_cost.py extract
                      (pass-1 profile ∩ pass-2 bounds ∩ parquet row-group stats)
                        └──► <q>.duckdb_cost.txt

embeddings-cache ──► generate_testdata.sh --embeddings external   (local only, per host)

tpch.sf40 (shad-gpu only) + testdata/tpch_query_sql.sh
    └── gen_duckdb_goldens.sh --sf 40 ──► goldens/tpch.sf40/duckdb_<q>.csv (+ .count.csv)
```

Consequences worth knowing before you regenerate:

- **`.cost.txt` is a pure function of `.cpu.txt` and `cost_model.conf`** — regenerating one
  `.cpu.txt` obliges the sibling `.cost.txt`, and `test_cost_model` re-derives every one of
  them, so a hand-edited `.cost.txt` goes red.
- **The GPU never authors a golden.** Both GPU tiers read the CPU-authored `.cpu.txt` and
  `.result.txt`, which is what makes a GPU-vs-CPU divergence a red test rather than a
  quietly rewritten expectation.
- **`plan_bytes.sha256` needs a second, deliberate variable.** Under plain
  `UPDATE_CANONICAL=1` the test verifies instead of rewriting, and says so — a bulk regen that moved the wire
  format goes red during the regen, before the goldens are pulled home. It is also the only
  thing pinning the FlatBuffers union ordinals: `test_plan_serialiser` checks `node_type()`
  against the enum, but both come from the same generated code, so a reordered union agrees
  with itself and passes. It pins a kind only where the corpus plans one — true for all
  fifteen today (thinnest: cross join and interleave at 7 goldens each), and not structural,
  so a sixteenth kind no query plans would be unpinned with nothing going red.
- **The `.duckdb_cost.txt` path is re-runnable without DuckDB**: `--extract-only` rebuilds
  the goldens from the committed profiles plus the parquet, so only a genuine oracle change
  needs the 1.5.4 pin.

## CI structure (`.github/workflows/pipeline.yml`)

`pipeline.yml` runs on pushes to master and on every PR, but not on documentation —
`**.md` and `llm-wiki/**`, which nothing builds, tests or reads. It takes two layers:
`paths-ignore` skips a wholly-documentation diff, and the **changes** job skips a
documentation-only push to a PR that carries code, which `paths-ignore` cannot see because
it judges the whole PR diff. Both fail open, and a doc file that ever becomes an input to
something has to come off both lists.

Five independent job chains:

```
changes ──► everything below
cpp-cpu (2 legs)          cpp-build-2502 ──► gpu-tests
cost-report ──► deploy-pages (master push only)          s3-datasets
```

- **changes** — the documentation gate above; every other job carries `needs: changes`
  and runs only on its `code == 'true'`.
- **cpp-cpu** — two matrix legs, each in a RAPIDS container: `cudf: 25.02`
  (`rapidsai/base:25.02-cuda12.0-py3.12`) and `cudf: 26.02`
  (`rapidsai/base:25.10a-cuda12-py3.12` — the leg's label is ahead of its image, #129).
  Both legs do the same work: C++ build through the shared `.github/actions/cpp-build`
  composite (conda gcc + ccache + pinned cmake/ninja), `ctest --test-dir cpp/build -L cpu`,
  generate + validate sf1 (pinned DuckDB — the TPC-DS dsdgen column types drift between
  releases and would move every plan golden), then the CPU rust tiers:
  `test_plan_serialiser`, `test_query_plan`, `test_cpu_full_table`,
  `test_cpu_executor_misc`, `test_cpu_oom`, `test_node_executor`, `test_query_plan_misc`,
  `test_cpu_partitioned`, `peacockdb-ffi --test test_ffi`, plus rust-only
  `test_plan_bytes` and `--lib`. Build and run are separate steps whose env blocks must
  stay byte-identical, or the run step recompiles.
- **cpp-build-2502** — builds the 25.02 C++ side, bundles the Arrow/Parquet runtime libs,
  and stages `test_gpu_full_table`, `test_gpu_partitioned`, `test_inc2_conformance` as the
  `cpp-install-25.02` artifact. Separate from cpp-cpu so the GPU job can start without
  waiting for the CPU tests.
- **gpu-tests** (needs cpp-build-2502) — ssh to **shad-gpu** into a per-run `REMOTE_DIR`:
  rsync artifact + testdata, patch the binaries for glibc 2.35, generate sf1 on the host
  if absent, then run. Three guards, each closing a hole that shipped: sf40 presence is
  asserted by CI (`tpch.sf40/lineitem.parquet`) because the binaries themselves skip and
  exit 0, which would be green having verified nothing; every `peacock_*_tests` binary runs
  by glob with a ran-any assertion (a hand-written list once let `peacock_tpchv_tests` be
  built, shipped and patched but never run); and any binary reporting `PASSED 0 tests` is
  an error. The staged rust binaries then run with `--test-threads=1` (cuDF/RMM share one
  process-wide pool). No `set -e` — statuses are OR'd so one failure cannot skip the rest —
  and `REMOTE_DIR` is removed on `always()`.
- **cost-report** — the golden/meta tier lives here, not in cpp-cpu: python
  `testdata/test_duckdb_cost.py`, the `scripts/exec_model/tests/test_*.py` prototype set,
  `cargo test -p cost-report`, and rust-only
  `test_cost_model` + `test_ci_coverage`. Then report generation, a PR-comment upsert, and
  the cost-regression gate against the base SHA, which fails the job on a regression. On a
  master push it uploads the Pages artifact.
- **deploy-pages** (needs cost-report; master pushes only) — publishes that artifact.
- **s3-datasets** — reads parquet footers of the large S3 datasets over ranged GETs
  (schema, embedding dims, row counts). Skipped on fork PRs, where the secrets do not
  exist, and exits 0 loudly if the credentials are absent anyway.
- **validate-large.yml** — `workflow_dispatch` only, gated on the same repo: validates the
  named datasets in place on shad-gpu with the scale-safe validators, then runs the S3
  metadata check. The `datasets` input defaults to `tpch.sf40 tpcds.sf200`; `tpch.sf200` is
  allow-listed but deliberately not defaulted, because it does not exist yet.

Two traps before adding a step: a container job defaults `run:` to `sh`, not bash, so the
two of them declare `defaults.run.shell: bash` to get `-o pipefail` at all; and the python
steps install the current pandas — 3 on the runner's 3.12, 2.3 on a dev box's 3.10 — so the
prototype has to hold across that boundary, unpinned on purpose.

Which Rust targets run where is not folklore — `test_ci_coverage` fails when a target
exists that no workflow step names. Doctests are the one class outside that guard (#128).
The two python steps need no such guard because neither names files: the extractor test is
one file, and the prototype step globs `test_*.py` and errors when the glob matches
nothing. Inside a prototype file the equivalent hole — a test defined below the
`__main__` footer, which pytest collects and direct execution would not — is closed by
`tests/harness.py`, which reads the source back and fails naming what it missed.

## What `rust-only` means

A cargo feature (`peacockdb-core/rust-only` → `peacockdb-ffi/rust-only`, an empty marker),
and the definition lives in `peacockdb-ffi/build.rs`: under it the build script **skips
cmake entirely**, so nothing links `libpeacock_gpu` and neither cuDF nor a CUDA toolchain
is needed. Everything that reaches the FFI is compiled out behind
`#[cfg(not(feature = "rust-only"))]` — the GPU executors and backend, the extern
declarations, and the three GPU test files, which are gated at file level because they
would have nothing to call.

So it is the Rust half built against DataFusion alone. That is why it is the fast loop,
and why anything a rust-only binary can do is by definition CPU-only.

**It selects a *build*, not a set of tests.** `cargo test --features rust-only -p
peacockdb-core` runs every target that compiles under it — including the full CPU
execution suite — not just the golden/meta tier. Naming a tier takes `--test`. This has
been mis-transcribed at least once; see the "refactor is verified with a subset" rule
below.

## Local build workflows and caches

One cargo target dir per workflow — this is what prevents cache thrashing (feature flags
and `cudf_ROOT` changes bust fingerprints; sharing a dir means recompiling the DataFusion
stack on every switch):

| Workflow | Command | C++ build dir | Cargo target dir |
|---|---|---|---|
| rust-only (golden regen/verify, fastest loop) | `cargo test --features rust-only -p peacockdb-core --test <target>` (drop `--test` and it runs every target that compiles, not a tier — see above) | — | `target/` |
| cost-report | `cargo test -p cost-report`; preview: `scripts/cost-report-preview.sh` | — | `target/` |
| C++ only, cudf 25.02 | `scripts/build.sh --cudf_ROOT <rapids-cuda-12.2> --gcc-version 12 ...` | `cpp/build` | — |
| C++ + staged Rust bins, cudf 26.02 | `scripts/build-test.sh --build` (drives cmake directly) | `cpp/build26` | `target-cudf-<basename cudf_ROOT>` |
| Rust + cudf (FFI) | via build-test scripts, or manually with `scripts/cargo-cudf.sh` | cargo OUT_DIR | `target-cudf-<basename cudf_ROOT>` |
| GPU benchmarks (`peacock_gpu_benchmarks`) | `scripts/build-test-shadgpu.sh --build-benchmarks` | cargo OUT_DIR | `target-cudf-<basename cudf_ROOT>/benchmarks/` |
| Any of the above, containerized | `scripts/docker-build.sh [--no-image] -- <command>` | `<cache-dir>/cpp-build` | `<cache-dir>/cargo-target` |

The benchmark row shares the target *root* with the correctness builds on purpose:
`[profile.benchmarks]` (`inherits = "release"`) already separates the artifacts into
`benchmarks/`, and a second root would fork the `.peacock-ffi-cudf-root` stamp — a
spurious `cargo clean -p peacockdb-ffi` on first use — without saving a rebuild. Why the
profile exists at all is argued once, in `Cargo.toml`; the consequence here is that a
record's `total_us − nodes_total_us` is only a measurement when it was built under it,
which is why `build_profile` is in the record.
Two costs, one-time per profile: the first `--build-benchmarks` cold-compiles the whole
DataFusion stack **and** builds `libpeacock_gpu.so` a third time (peacockdb-ffi's cmake
runs in `OUT_DIR`, which lives inside the profile dir). It leaves the correctness caches
untouched, which is the trade. Every record carries `build_profile=` so numbers from
different profiles are never silently compared.

The container and the native path deliberately do **not** share a cargo cache
(`/cache/cargo-target` + `RUSTFLAGS=-C debuginfo=0` vs `$PWD/target-cudf-*`), so
alternating between them costs a cold rebuild each way. That is the price of having
both entry points, not thrash to be diagnosed.

ccache is auto-enabled for both C++ workflows when the binary is present (host
compilers only — ccache + nvcc is unreliable). Rust caching is the per-workflow target
dir below.

Note the C++ side is built *twice* on the cudf path, into two dirs that are both used:
`build-test.sh` drives cmake into `cpp/build26` for the installable libs and the C++
test binaries it ships, while `peacockdb-ffi/build.rs` runs cmake again into cargo's
`OUT_DIR` for the copy the Rust binaries link against. The `.peacock-ffi-cudf-root`
stamp is what keeps the second one from rebuilding on every `--build`.

Rules that keep this healthy:

- **A refactor that must not change behavior is verified with a representative subset**
  — one query per mode/tier per binary, plus the full rust-only tier — not a full
  CPU/GPU suite run: the goldens are the invariant, so unchanged golden bytes and a
  green rust-only tier prove more per minute than re-running everything.
- **Day-to-day iteration is the rust-only loop** — plain cargo into `./target`, no
  wrapper, no C++/CUDA:
  `cargo test --features rust-only -p peacockdb-core --test test_query_plan`
- **Never run cudf-feature cargo builds in `./target`** — they would evict the rust-only
  cache and vice versa (the `ffi` feature and `cudf_ROOT` both change fingerprints, and
  the cudf side recompiles the DataFusion stack at opt-3). For one-off cudf/FFI cargo
  commands use `scripts/cargo-cudf.sh`, which requires `CUDF_ROOT` and derives *both*
  `CARGO_TARGET_DIR=target-cudf-$(basename "$CUDF_ROOT")` AND `CC`/`CXX` from that same
  basename (25.02 → gcc-12, 26.02 → gcc-14; an unknown root fails and asks for
  `GCC_VERSION`) — the same dir and the same compiler the build-test scripts use, so it
  shares their warm cache. Both halves matter: cc-rs emits `rerun-if-env-changed` on
  `CC`/`CXX`, so entering a target dir with a different C compiler re-runs every native
  build script (zstd-sys, bzip2-sys, lzma-sys, psm, blake3) and rebuilds the whole
  DataFusion stack above them — before this, alternating between the two entry points
  thrashed the cache each way:
  `CUDF_ROOT=~/data/miniforge3/envs/rapids scripts/cargo-cudf.sh test -p peacockdb-core --test test_gpu_full_table --no-run`
  For anything more than a one-off command, use `build-test.sh` / `build-test-shadgpu.sh`
  instead — they handle build, staging, shipping and running.
- The FFI crate caches its cmake `cudf_DIR` in OUT_DIR; both build-test scripts clean it
  **only when the cuDF root changed** (stamp file `.peacock-ffi-cudf-root`);
  `PEACOCK_FFI_CLEAN=1` forces.
- `cpp/build` stays 25.02, `cpp/build26` stays 26.02 — separate dirs also dodge cmake's
  stale `cudf_DIR` cache. Keep both even when stale.
- cuDF version scope: a functional GPU run on **25.02** (shad-gpu) is sufficient
  verification; **26.02 needs only to compile** (CI builds both legs).

## Memory constraints (15 GiB-class hosts)

- **At most one binary links at a time**: the cmake configures set a Ninja
  `link_pool=1` job pool (parallel links OOM the host; compiles stay parallel). The
  build-test scripts also serialize rust test-binary builds one `--test` per invocation.
- Cold cudf cargo builds are throttled to `CARGO_BUILD_JOBS=3` on <20 GiB hosts
  (override via env).
- Full local CPU suite: run with `-- --test-threads=2` or less; the heavy golden suites
  OOM at default parallelism.

## Remote hosts

| Host | Use | Managed by |
|---|---|---|
| **shad-gpu** (most used) | GPU test suite (cudf 25.02, H200-class; old glibc → patch step) | `scripts/build-test-shadgpu.sh` (`--build --push-binaries --patch --run[-detached]` / `--all`, `--run-status`); same script measures, under separate flags (`--build-benchmarks`, `--run-benchmarks[-detached]`, `--benchmark-status`, `--pull-benchmarks`) that `--all` deliberately does not imply — one exit code cannot mean both "correctness passed" and "measurement completed". Resilient rsync + retries — the link is flaky |
| **verda** (when available) | large CPU runs, golden regen | `scripts/build-test.sh --host verda --all` (add `--rust-only` to skip the C++/FFI half) |
| **verda-gpu** (least used) | same root volume as verda, with a GPU attached | `scripts/build-test.sh --host verda-gpu --gpu --all` |
| **nebius** | large CPU-only VM for benchmarking | manual |

- **Testing the regen *mechanism* is not a full regen.** Scope it with `PCK_TEST_FILTER`
  (forwarded to every staged binary) — two or three queries prove the path as well as
  903 do, and a full `--update-canonical` run rewrites every golden on the remote and
  pulls the whole set back into a git working tree, where an unrelated diff can ride
  home in an unrelated commit. Note that a binary whose tests all filter out runs zero
  tests and passes, so say which binaries actually executed; and a query-name filter
  never matches `test_plan_bytes` (its only test is `serialized_plan_bytes_are_stable`),
  so exercising that guard takes a filter that matches it or a separate run.
- **Prefer verda for large CPU runs** (whole suite or big selections). It is not always
  up (the human starts it manually) — falling back to a local run is completely fine.
- **Comparing files across hosts: use checksums, and `LC_ALL=C sort` for any listing.**
  Two hosts collate `ls`/`sort` differently, so a manifest diff reports differences that
  are pure collation — this produced two false alarms in one session before checksums
  settled it. Compare `sha256sum`/`md5sum` output, not directory listings.
- Rented hosts change SSH host keys on reprovision: `ssh-keygen -R <host>` + re-keyscan
  rather than fighting the mismatch.
- **Golden regen**: on verda via `scripts/build-test.sh --host verda --rust-only
  --update-canonical`, then `--pull-goldens` to bring regenerated goldens back; or
  locally with `UPDATE_CANONICAL=1 cargo test --features rust-only ...`. Sync is one
  flag per kind per direction — `--push-<kind>` / `--pull-<kind>` for
  `parquet | goldens | duckdb-profiles | duckdb-dynfilters | queries`.
  **Pushes mirror (`--delete`), pulls are additive**, deliberately: the remote is a
  *partial* mirror (`testdata/goldens/` holds `tpch.sf40/`, which lives only on shad-gpu),
  so mirroring downward would delete fixtures the source host never had, out of a git
  working tree. (The sf40 *dataset* is what lives only there — its 16 CSV goldens are
  committed.) `plan_bytes.sha256` rides along safely because `test_plan_bytes` itself
  refuses to regenerate without `PEACOCK_REWRITE_PLAN_BYTES=1`.
  The tpch **embeddings cache is NOT syncable** — a per-host intermediate
  (`fetch_embeddings.sh`, ~1.8 GB, gitignored); regenerate it where you need it.
- Remote CPU runs ship built binaries + goldens + data only — never source. The CPU test
  crates bake the testdata path at compile time (#49), so the remote needs a
  `/media/data/peacockdb` symlink.
- **Large runs (test or regen): arm a monitor that reports progress every 5 minutes** —
  progress may stall (flaky links, OOM kills), and a silent stall looks identical to a
  long run.
- **Large CPU + GPU batches run in parallel** — kick off the shad-gpu run and the
  verda/local CPU run concurrently; neither waits for the other.

## Benchmarks

### Per-node GPU timing — `peacock_gpu_benchmarks` (scripted)

Same case list as the GPU correctness gate (all three targets `include!`
`peacockdb-core/tests/common/gpu_cases.inc`, so the measured set cannot drift from
the verified one), different question: how long did each plan node take. It asserts
nothing, so it can never gate a merge — `test_ci_coverage.rs` exempts it explicitly.

```
scripts/docker-build.sh --no-image -- ./scripts/build-test-shadgpu.sh --build-benchmarks
./scripts/build-test-shadgpu.sh --push-binaries --patch --run-benchmarks --pull-benchmarks
# or, for a run that outlives your ssh session (the suite takes tens of minutes):
./scripts/build-test-shadgpu.sh --push-binaries --patch --run-benchmarks-detached
./scripts/build-test-shadgpu.sh --benchmark-status     # going? finished? log tail
./scripts/build-test-shadgpu.sh --pull-benchmarks      # once it reports finished
```

The correctness gate has the same pair — `--run-detached` and `--run-status` — through the
one launcher both phases go through; `--all` stays the attached form. Either status flag
exits 0 **only** when the latest run of that phase finished with 0: still going, died
without writing its code, and a completion belonging to an earlier run are all non-zero,
because the alternative is a status command that reports someone else's success as yours.
Run state lives in `$REMOTE_REPO/.run-state/<phase>.{sh,log,rc,id}`, deliberately outside
`cpp/install/` — that tree is mirrored with `--delete`, so a marker kept there is erased by
the next push.

One record per case, at
`testdata/benchmark-results/<dataset>.sf<sf>/<query>.<label>.benchmark.txt` —
`<label>` being the `<mode>-<tp>-<tier>` component the `.cpu.txt` goldens carry,
because 17 of the cases are measured at both `full_table-tp1-standard` and
`partitioned-tp8-standard`: same query, different plan, different time. The tree is
the plan with `time_us` per node (and `p<k>:` sub-lines where N>1), then a trailer:

| Field | Reading |
|---|---|
| `build_profile` | how the harness was compiled. `total_us − nodes_total_us` is that Rust; records from different profiles are not comparable on it |
| `allocator` | which rmm device resource the node times were taken under — `rmm-pool …` with the sizes it was built with, or `rmm-default …` with the reason. Without a pool every cuDF intermediate is a `cudaMalloc`/`cudaFree` round trip billed to whichever node allocated it, so it inflates the largest-output nodes hardest: records under different allocators differ in **profile**, not just scale |
| `shared_work_charged_to` | which `p<k>` sub-line carries work a node does once for all its partitions — the hash scatter concatenates and scatters in one operation and bills p0, so a p0 far above its siblings is the accounting, not skew. Written whether or not the plan has a repartition, so absence means only "written before the field" |
| `sync_floor_us` | what the timed region costs around no work. **Every node time includes one; do not subtract it** — a node at or below it is unresolved, not cheap |
| `nodes_at_or_below_floor` | how much of the tree this file cannot resolve. 2/40 is a profile; 35/40 measured mostly its own instrument |
| `nodes_total_us` | Σ of the node times |
| `total_us` | the whole query end to end — parse, plan, serialize, node walk, materialize |

Reported run is the **2nd-smallest by `total_us`** of ten measured runs, after one
discarded warm-up: the fastest run is the one most likely to have caught a
favourable scheduling accident, and the whole run is reported rather than a per-node
minimum, which would produce a tree belonging to no single execution. Not `--delete`d
by any push (see `benchmark_result()` in `tests/common/mod.rs` for why the tree is not
called `benchmark-goldens`); `--pull-benchmarks` is additive.

The run counts are compile-time constants in
[`tests/common/mod.rs`](../peacockdb-core/tests/common/mod.rs#L1290) — warm-up 1,
measured 10, floor samples 200. No environment variable moves them, unlike
`PCK_TEST_FILTER` or the C++ suites' `PEACOCK_BENCHMARK_RUNS`: changing one is an edit
and a rebuild, so every record in the tree was taken at the same counts.

`PEACOCK_GPU_DEBUG` is deliberately **not** forwarded to this run — it adds a
`cudaStreamSynchronize` after every operator, which changes exactly the thing being
measured.

### Wall-time C++ suites (currently unscripted)

Wall-time runs are manual; the protocol: `PEACOCK_BENCHMARK=1`
(`PEACOCK_BENCHMARK_RUNS=5`), execute-only timing, all-device-synced, report the
**2nd-minimum** of the runs. Current numbers: `llm-wiki/reports/benchmark-minimal.md`.

- **TPC-H / TPC-H+V, single GPU**: build `peacock_tpch_tests` / `peacock_tpchv_tests`
  (25.02 leg) and run on shad-gpu with the sf40 env vars
  (`PEACOCK_TPCH_SF40_DIR`, `PEACOCK_TPCH_GOLDEN_DIR`, `PEACOCK_TPCH_VEC_PARAMS`).
- **Multi-GPU**: build `peacock_multi_gpu_tpch_tests` / `peacock_multi_gpu_tpchv_tests`
  explicitly (EXCLUDE_FROM_ALL) from `cpp/build26` on a multi-GPU host; G=1 baseline via
  `CUDA_VISIBLE_DEVICES=0`. Benchmark **each query in its own process**
  (`--gtest_filter=...`) — multiple benchmark queries in one process are flaky at G≥2
  (process-global cudf stream state across WorkerPool teardowns). Correctness (single
  execute) is fine in one process. Mean-type aggregates must decompose to partial
  SUM+COUNT — never average partial means.
