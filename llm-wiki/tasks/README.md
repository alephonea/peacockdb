# Task specs

Working specs for in-flight tasks live here.

- A task whose work outlives one commit gets its spec committed here, and the spec moves
  to `llm-wiki/archive/` when the task completes.
- A spec for work that fits in one commit is deleted when the task is done.
- Keep specs short: goal, constraints, verification bar. Long-lived design facts belong in
  `architecture.md` / `build-test.md`, not here.
- **A spec that outlives its commit ends with what the work measured**, before it is archived:
  what closed, the number that says so, and what the task cannot prove. Three tasks in a row
  reached their completeness pass without one, each time with the outcome sitting in msgq
  instead — which is where a reader of the ticket it closed will not look.
