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
- **Match surrounding idiom** (naming, comment density, error handling). Trust rustfmt;
  don't hand-format.
- **Python:** plain module filenames — no leading underscores.
- **No defensive code for impossible scenarios**; trust internal invariants and framework
  guarantees. No fallbacks or feature flags the task didn't ask for.
- **No scope-creep refactors**: a bug fix doesn't need surrounding cleanup.

## Antipatterns

Each of these shipped here and cost something. They are recorded with the case that
revealed them, because the general rule is easy to nod along to and hard to recognize in
your own diff.

### A large behavior change triggered implicitly by the arguments

A function that switches to fundamentally different behavior based on some combination of
its inputs — a magic value, a string it parses, a pair of flags read together — hides the
most important thing it does. The caller reads one call and cannot tell which behavior it
gets. Worse, the switch has no natural place to fail: feed it an input nobody anticipated
and it picks a branch silently, with no diff to the switching code and nothing going red.

Make the behavior an explicit parameter with a name — an enum, not a bool and not a string
— so the call site states which one it wants. Where an input genuinely must be decoded
back into behavior (a frozen file format, an external contract, anywhere there is no call
site to state it at), keep the mapping **exhaustive**: an unlisted value panics naming its
fix rather than falling through to a default nobody chose.

The case that revealed it, in the test harness: `partition_mode("tp8-standard")` returned
`RealMultiPartition` and every other device label fell through to `SinglePartition`, so
which executor a test ran was a side effect of how its golden file happened to be named.
Adding a device — a memory-constrained genuine-8-way tier (#91) — would have routed it to
the wrong executor with nothing failing. The mode is now a parameter stated at the call
site. One lookup survives, for the plan tier, whose `.plan.txt` names are frozen and whose
corpus is built by parsing them off disk: it is exhaustive, so a new label fails loudly
instead of planning single-partition by default.

The same shape appears as a bool that means two unrelated things, a trailing `Option`
whose `None` selects a different algorithm, and an argument order that is not type-checked
because both parameters are the same type.

### A thread-local as an output or side-channel argument

Found during the executors refactor: `execute_one`/`execute_node` passed node inputs
through an anonymous-namespace `thread_local`, which is per-translation-unit — splitting
the file would have silently forked the variable and re-executed whole subtrees (correct
answers, exponential cost, invisible to correctness tests). Pass inputs and outputs
explicitly through parameters; never smuggle them through thread-locals or globals.
