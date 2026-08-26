# batch-partitioned tickets

Queries the batch-partitioned rollout disabled, and what has to change before each comes back.

Separate from [`../tickets.md`](../tickets.md) because these are a rollout's worklist rather than
the engine's: they arrive in bulk when a sweep hits a wall and close in bulk when it is cleared,
and a list read by triage is not the place for that. **The ID space is shared** — the counter
lives in `../tickets.md`, so a number is never two things and an older reference resolves
wherever it points.

Otherwise the same rules, in `coding-style.md`: at most fifteen lines, at most two stating the
problem, about code and never about documentation. Each ticket carries an `<a id="tNN">` anchor
above its own header, which is what the cost widget's index resolves a link against.

No tickets yet — the rollout starts with T18.
