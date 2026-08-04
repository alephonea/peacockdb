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
