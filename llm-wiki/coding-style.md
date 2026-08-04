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
- **C++ formatting** is defined by `.clang-format` at the repo root. Apply it to the
  lines you changed — `git clang-format` — never to whole files: the tree was never
  machine-formatted, so reformatting one file to fix one function buries a three-line
  change in three hundred.
- **Python:** plain module filenames — no leading underscores.
- **Bash: the flag set is an interface, and failure is fatal.** No flag that another
  flag already implies, and none whose only effect is the default. Contradictory
  combinations are rejected with a message naming the contradiction — never resolved by
  argument order, which makes the same two flags mean different things depending on how
  they were typed. Validate every argument, and that the run will actually do something,
  *before* the first side effect: a typo should fail before it ships files, not halfway
  through. Prefer `set -euo pipefail`, and remember what it does not cover: an `exit`
  inside `$(…)` ends the subshell only; an empty list makes a `for` body vanish
  silently, so a derived-but-empty work set must be an explicit error rather than a
  green no-op; and a failing `&&` list is ignored at statement level but becomes the
  return value when it is a function's LAST command, so `[ -f x ] && do_thing` written
  at the end of a function silently fails the caller. Where execution genuinely must continue past a failure (running every
  test binary before reporting), say why at the site and accumulate the status so the
  script still exits non-zero.
- **No defensive code for impossible scenarios**; trust internal invariants and framework
  guarantees. No fallbacks or feature flags the task didn't ask for.
- **No scope-creep refactors**: a bug fix doesn't need surrounding cleanup.

## Antipatterns

Most of these shipped here and cost something, and are recorded with the case that
revealed them, because the general rule is easy to nod along to and hard to recognize in
your own diff. An entry with no case attached is stated generically on purpose; add the
case when one turns up.

### Encapsulation violations

Reaching past an interface into what it was meant to hide — reading or writing private
state, depending on a representation its owner is free to change, re-implementing a rule
that lives inside the boundary, or letting a caller assemble something only the owner
should assemble.

It compiles and it passes, which is why it survives review. The cost comes later: the
owner can no longer reason about its own invariants, because correctness now depends on
code it cannot see, and a change that is local by every reasonable reading breaks
something far away. The rule gets duplicated rather than moved, so the two copies drift
and the one that is wrong is whichever the reader did not open.

Fix the interface rather than the caller: add the operation the caller actually needs, and
keep each invariant on the side of the boundary that owns it.

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
