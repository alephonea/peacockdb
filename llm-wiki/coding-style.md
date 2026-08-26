# peacockdb coding style

- **Create useful abstractions.** New code should introduce (or reuse) abstractions that
  make later reuse easy — a shared driver, a trait, a helper — rather than copies of
  similar logic.
- **Small files:** under 1000 lines. Split by responsibility (the operator family files
  and executor modules are the pattern).
- **Interfaces/traits in separate files** from their implementations (`executors/executor.rs`,
  `executors/node_by_node.rs`, `operators/operator.rs` are the models).
- **Short functions:** under 150 lines in most cases.
- **Comments say *why*, briefly.** Only non-obvious constraints, invariants, and gotchas —
  never what the next line does, never process history. If a comment documents an
  important historical decision, a distilled version may go to
  `llm-wiki/archive/historical-comments.md` instead of living in the code.
- **Comment length is capped.** Four lines for a comment inside a function body, ten for
  one above a declaration or at the top of a file. A comment past its cap has stopped
  annotating and started explaining, and an explanation nobody can find unless they are
  already reading this function is one nobody reads — move it to `llm-wiki/` and leave
  the line that points there.
- **Match surrounding idiom** (naming, comment density, error handling). Trust rustfmt;
  don't hand-format — and run it over the files you touched, never the crate, which
  predates the installed rustfmt and reformats 49 of them. A `mod.rs` is not one file for
  this purpose: rustfmt follows `mod` declarations, so formatting one reformats every file
  below it. Name the leaves instead.
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
  return value when it is a function's *last* command, so `[ -f x ] && do_thing` written
  at the end of a function silently fails the caller. Where execution genuinely must continue past a failure (running every
  test binary before reporting), say why at the site and accumulate the status so the
  script still exits non-zero.
- **No defensive code for impossible scenarios**; trust internal invariants and framework
  guarantees. No fallbacks or feature flags the task didn't ask for.
- **No scope-creep refactors**: a bug fix doesn't need surrounding cleanup.

## Names

Kernighan's rules, written down after the fact rather than followed from the start:
`test_inc2_conformance` is named after an increment, which the second bullet forbids, and
renaming it would move the staging array, the exemption list and two pages — so it stands as
the exception rather than as the example.

- **Length is proportional to scope** (K&R §2.1). A loop index is `i`; a name crossing a
  module, a trait or the FFI earns words. Both halves bite: a paragraph-long name in a
  three-line body is noise, and a terse one in an exported signature is a puzzle.
- **A name says what a thing is, never how it came to be.** No `new` in the sense of the
  newer one — `ExecutorNew`, `process_v2` — no `additive`, no ticket number, no task id.
  Rust's `Type::new` is a constructor, not a version marker. The manner of a change is the
  shortest-lived fact about it, git holds it already, and the reader who greps a year later
  is looking for behaviour. The case: T9's gtest suite shipped as `AdditiveAbi` — a suite
  about per-call scan reads and row ranges, named after the fact that it was added without
  breaking anything.
- **Functions get active names, and an inaccurate name is worse than a vague one**
  (Kernighan and Pike, *The Practice of Programming*). A `check_` that also repairs has
  misled every reader who trusted it.
- **The same thing carries the same name everywhere, and one name means one thing** —
  across the FFI most of all, where two names for one value is how the two sides drift
  without either being wrong. The inverse costs as much: `ScanBatch` in the flat buffers
  means partitions while `CudfCoalesceBatches.target_batch_size` in the same buffers means
  Arrow batches, which architecture.md has to carry a naming trap for.

## Length limits

- **A ticket is at most 15 lines**: one header, at most two stating the problem, the rest
  describing it. What runs longer is a design document wearing a ticket's number — put it
  in `llm-wiki/tasks/` and let the ticket point at it. The cap is also what keeps the list
  usable: a reader triaging 75 tickets reads headers and first lines, so a ticket that
  buries its problem statement on line 20 is not being read at all.
- **A ticket is about code, never about documentation.** A stale sentence, a dead link, a
  count that no longer adds up, a widget rendering any of them — fix it in the commit that
  found it. Documentation is anything whose product is prose for a reader: `llm-wiki/`, code
  comments, and the rendered cost report. Filing costs a number, a triage pass and a reader's
  attention, and the page stays wrong for as long as it sits in the list. A ticket asking that
  a page say more is the same shape — pages grow when a human asks, not when a ticket does.
- **Architecture pages describe the current state, not the route to it.** No "was X, now
  Y", no account of what an earlier attempt did or why it was abandoned. `architecture.md`
  and `build-test.md` answer what is true today; git holds the sequence, and a decision
  worth carrying forward goes to `llm-wiki/archive/`. A page that narrates its own history
  makes the reader work out which sentence is still in force.
- **A commit message, a PR description or a PR comment is at most 10 lines**, subject line
  included. All three are read in a narrow column next to the thing they describe, and the
  diff is the detail — a message that restates the diff is read by nobody, and one that
  argues a design is in the wrong place. Point at the ticket or the task spec instead, which
  is where a later reader will look anyway.

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

### A doc comment reassigned by an insertion

A doc block belongs to the declaration below it, so a type inserted above an existing one
takes its documentation and leaves it with none. Nothing reads as missing and every cap
still holds, which is why review passes it: `When` above `RunError` took `RunError`'s
block, whose opening sentences described `RunError` and stated a contract the same task
had disproved. After inserting a declaration, read each doc block's first sentence.

### A thread-local as an output or side-channel argument

Found during the executors refactor: `execute_one`/`execute_node` passed node inputs
through an anonymous-namespace `thread_local`, which is per-translation-unit — splitting
the file would have silently forked the variable and re-executed whole subtrees (correct
answers, exponential cost, invisible to correctness tests). Pass inputs and outputs
explicitly through parameters; never smuggle them through thread-locals or globals.
