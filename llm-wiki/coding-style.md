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
- **Match surrounding idiom** (naming, comment density, error handling). Trust rustfmt;
  don't hand-format.
- **Python:** plain module filenames — no leading underscores.
- **No defensive code for impossible scenarios**; trust internal invariants and framework
  guarantees. No fallbacks or feature flags the task didn't ask for.
- **No scope-creep refactors**: a bug fix doesn't need surrounding cleanup.
