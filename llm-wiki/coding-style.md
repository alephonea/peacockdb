# peacockdb coding style

- **Create useful abstractions.** New code should introduce (or reuse) abstractions that
  make later reuse easy — a shared driver, a trait, a helper — rather than copies of
  similar logic.
- **Small files:** under 1000 lines. Split by responsibility (the operator family files
  and executor modules are the pattern).
- **Interfaces/traits in separate files** from their implementations (`executors/executor.rs`,
  `executors/node_by_node.rs`, `operators/operator.rs` are the models).
- **Short functions:** under 150 lines in most cases.
- **Comments say WHY, briefly.** Only non-obvious constraints, invariants, and gotchas —
  never what the next line does, never process history. If a comment documents an
  important historical decision, a distilled version may go to
  `llm-wiki/archive/historical-comments.md` instead of living in the code.
- **Antipattern — thread-local as an output/side-channel argument.** Found during the
  executors refactor: `execute_one`/`execute_node` passed node inputs through an
  anonymous-namespace `thread_local`, which is per-translation-unit — splitting the file
  would have silently forked the variable and re-executed whole subtrees (correct
  answers, exponential cost, invisible to correctness tests). Pass inputs/outputs
  explicitly through parameters; never smuggle them through thread-locals or globals.
- **Antipattern — routing on a label.** The test harness derived execution mode from a
  device string: `partition_mode("tp8-standard")` returned `RealMultiPartition` and
  every other label fell through to `SinglePartition`, so which executor a test ran was
  a side effect of how its golden file happened to be named. Adding a device — a
  memory-constrained genuine-8-way tier (#91) — would have routed it to the wrong
  executor with no diff to the routing code and nothing failing. State the mode at the
  call site and pass it as a parameter. Where a name genuinely must be decoded back into
  config (the plan tier: `.plan.txt` golden names are frozen and `test_plan_bytes`
  derives its whole corpus by parsing them off disk, so there is no call site to state
  it at), keep the lookup EXHAUSTIVE — an unlisted label panics naming its fix instead
  of defaulting to a mode nobody chose.
- **Match surrounding idiom** (naming, comment density, error handling). Trust rustfmt;
  don't hand-format.
- **Python:** plain module filenames — no leading underscores.
- **No defensive code for impossible scenarios**; trust internal invariants and framework
  guarantees. No fallbacks or feature flags the task didn't ask for.
- **No scope-creep refactors**: a bug fix doesn't need surrounding cleanup.
