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

<a id="t181"></a>
### #181 — the GPU backend reaches `unreachable!` on every Inner join

`gpu_backend/backend.rs:136` calls `per_call_join_type` for every `NodeRef::Join`, and that
function is total only over the streaming family, so an Inner join lands on
`unreachable!("Inner needs no finish pass")` at `nodes/join.rs:586`.

The CPU backend does not, and the difference is one guard: `cpu_backend/join.rs:76` returns on
`capability.answers_in_one_call()` before asking which per-call type a finish pass would need. A
join that answers in one call has no finish pass to type, so asking is the bug rather than the
answer being missing.

It is why no ordinary join runs on a device in this mode: 72 of T18's 88 device cases, over 15
queries at all five modes — tpch q3 q10 q12 q14 q15 q19, tpcds q3 q15 q37 q42 q43 q52 q55 q82
q96. Those rows carry `#181` on their gpu columns.

<a id="t182"></a>
### #182 — after logical pricing, a peak cannot depend on batch shape

Pricing a batch from the plan's schema made the two engines agree and took two properties with
it, both of which were asserted and are now disabled at their sites with their numbers.

`a_query_has_a_smallest_budget_that_fits_and_trips_a_byte_below_it`: 10058 fits and so does
10057, so the budget is no longer a boundary. And
`an_injected_shape_moves_what_the_query_holds` loses its premise rather than its number — a
rebatcher cannot move a peak built from logical bytes, since those are a function of rows and
var-length content and neither depends on how the rows were batched.

The second is the question worth answering: T17a's rebatcher evidence rested on a peak that moved,
and after this it provably cannot. Either the peak should be priced on what is resident rather
than on what is declared, or the accounting is right and the rebatcher was never going to move it
and T17a measured an artefact of arrow's allocator. Deferred deliberately — T18's bar is results
and node stats, not memory.

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
