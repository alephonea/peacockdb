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

<a id="t180"></a>
### #180 — a shuffled count(\*) merges to nullable against a non-nullable declaration

`tpcds/q96` at `bp-tp4-single`, `bp-tp4-rowgroup` and `bp-tp4-sized`: "Column 'count(\*)' is
declared as non-nullable but contains null values". Both tp1 modes are clean and stay enabled.

A shuffle is what puts a state merge under the aggregate, so the three tp4 modes reach a path the
tp1 ones do not — which is why this is scoped to modes rather than to the query. The declaration
comes from DataFusion's schema for the final aggregate; what the merge produces is the engine's
own answer, and the two disagree only on nullability, so the same width and the same bytes make it
invisible to every golden that is not a run.

Neighbour to [#163](../tickets.md#t163) rather than the same thing: that one is a declared type
never checked against the expression producing it, this one is a declared *nullability* the merge
contradicts at run time. Found by T18's stage-4 enablement, disabled at three of five modes.
