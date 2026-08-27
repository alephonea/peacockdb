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
### #182 — two accounting properties are out of reach, and no budgeted run survives

Pricing a batch from the plan's schema disabled two cases. Neither property stopped being true,
and with them off, those two sites were the only ones in the tree passing `Some(budget)` over a
planned query — so the accountant's enforcement path is now exercised by unit cases over the mock
backend and by nothing else, which is the state the budget case was written to end.

The budget case is [#179](../tickets.md#t179): `boundary()` searches upward from the observed peak,
so "10057 also fits" says the peak is 10,057 and the trip is below it. Logical pricing raised the
peak (8,222 to 10,057, a bitmap arrow never allocated) without raising the modelled transient as
much, so a downward search settles it.

The rebatcher case loses its query, not its premise. A rebatcher cannot move a *total* built from
logical bytes, but a peak is what is resident at once and `GpuCoalesceAllBatches` holds its whole
lane before emitting one batch — both live at the emit, a rows fact logical bytes carry.
`nested-loop-join` has one batch per lane and its old 318-byte move was arrow reallocating it;
`tpch/nested-limits` at `bp-tp4-rowgroup` can show it, at 3 ms a run. T17a's drain half is
untouched — a drained lane changes rows per lane, so q16's 104.7 MB against 77.9 MB stands.

<a id="t183"></a>
### #183 — the device exports Utf8 where the sink declares Utf8View

`GpuUnload lane 0: the exported stream is not the sink's rows: column types must match schema
types, expected Utf8View`. The sink's schema comes from DataFusion, which uses `Utf8View`; the
device's IPC export produces `Utf8`. Same values, different arrow type.

Twelve of T18's device cases over eleven queries — tpcds q3 q15 q37 q42 q43 q52 q55 q82, tpch q10
q12 q15 — with `#183` on their gpu columns.

The same divergence bit the digest comparator one layer up, where hashing the column type reddened
eight legacy gpu cases whose rendered comparison had never looked at types. That one was a
comparison artefact and was fixed by hashing names; this one is the export genuinely disagreeing
with the schema the plan declared, and no comparison choice makes it go away.

<a id="t184"></a>
### #184 — a hash repartition of one lane into four fails in cuDF

`GpuEmitPartitions: execute_node(#11 CudfRepartition{Hash, 1 -> 4}): CUDF failure at
cpp/src/spark_hash_partition.cu:179`. Four of T18's device cases, `tpch/q15` at every mode, and
no other query reaches it.

One lane in and four out is the shape, which the batched mode produces far more often than legacy
does — a shuffle above a single-lane source. Whether the kernel refuses the 1-to-N case or the
input carries something it will not take is the first thing to establish.

<a id="t185"></a>
### #185 — the device reports one input row where the CPU reports the group count

The two engines' sections differ in `in_rows` and in nothing else. `tpcds/q96` expects
`in_rows=[[34]]` and the device reports `[[1]]`; `tpch/q14` expects `[[10]]` against `[[1]]`, and
at `bp-tp4-sized` `[[3,3,3,3]]` against `[[1,1,1,1]]`. Every `batch_rows` entry and every byte
agrees, on every node.

So it is an aggregate's state being counted as one row rather than as its groups, on the consuming
side. Four device cases over tpcds q96 and tpch q3 q14.

Worth the note that this is the failure the per-node golden exists for: the answers match, the
bytes match, and no result comparison at any tolerance could see it. It was found on the first
device run of the corpus tier, which is the argument for the tier.

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
