# batch-partitioned tickets

Queries the batch-partitioned rollout disabled, and what has to change before each comes back.

Separate from [`../tickets.md`](../tickets.md) because these are a rollout's worklist rather than
the engine's: they arrive in bulk when a sweep hits a wall and close in bulk when it is cleared,
and a list read by triage is not the place for that. **The ID space is shared** — the counter
lives in `../tickets.md`, so a number is never two things and an older reference resolves
wherever it points.

Otherwise the same rules, in `coding-style.md`: at most fifteen lines, at most two stating the
problem, about code and never about documentation. The deferred-fix exception to the
fifteen-line cap is stated there and applies here. Each ticket carries an `<a id="tNN">` anchor
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

Pricing a batch from the plan's schema disabled two cases in `test_cpu_batch_partitioned.rs`,
both `#[ignore]`d rather than deleted so they stay in `--list`. Neither property stopped being
true. Deferred deliberately: T18's bar was results and node stats, not memory.

**What it costs meanwhile.** Those two sites were the only ones in the tree passing `Some(budget)`
over a planned query, so the accountant's enforcement path is now exercised by unit cases over the
mock backend and by nothing else — eight `Executor` impls report a residency and a transient on
real plans and nothing budgeted reads them. That is the state the budget case was written to end,
and it is the reason to pick this up rather than either property on its own.

**The budget case is [#179](../tickets.md#t179).** `boundary()` sets `(low, high) = (peak, peak * 8)`
and searches upward, so its byte-below arm fails whenever `fits(peak)` — and "10058 fits, 10057
also fits" says the peak is 10,057 and the trip is somewhere below it. Logical pricing raised the
peak (8,222 to 10,057, a validity bitmap arrow never allocated) without raising the modelled
transient as much, so the boundary fell under the floor the search starts from.

Fix: when `fits(peak)`, search `(0, peak)` instead. Same bisection, same assertions, and the
byte-below arm then means what it says. `boundary()`'s doc claims the peak is a floor because a
pre-call check tests a modelled transient that can exceed what was held — true, and this is the
case where it does not, so the doc wants the other direction named beside it.

**The rebatcher case lost its query, not its premise.** A rebatcher cannot move a *total* built
from logical bytes, but a peak is what is resident at once, and `GpuCoalesceAllBatches` holds its
whole lane before emitting one batch — both live at the emit, a rows fact logical bytes carry.
`nested-loop-join` cannot show it: one batch per lane, so the rebatcher merges one into one, and
its old 318-byte move was arrow reallocating a single batch.

Fix: repoint the second pair at `tpch/nested-limits` at `bp-tp4-rowgroup` (`BP_MODES[3]`), whose
`part` loader is `partition_groups=[[[0],[1]]]` — two batches in one lane — and is the plan's peak
node at 983,071 estimated against 1,600,062 source bytes. Three milliseconds a run. It is already
in the corpus and in the injected set, so nothing new is declared.

T17a's drain half is untouched: a drained lane changes rows per lane, so q16's 104.7 MB against
77.9 MB stands.

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
### #185 — `GpuAggregateBatches` reports its own output as `in_rows`

The two engines' sections differ in `in_rows` and in nothing else, at that node and no other. Every
`batch_rows` entry and every byte agrees, both runs complete, and no result comparison at any
tolerance could see it — which is the failure the per-node golden exists for.

The rule is the node reporting what it emitted where it should report what it consumed. It read as
"one row instead of the group count" for three batches, because every early case was an aggregate
with no group keys, whose output is one row: `tpcds/q96` expected 34 and reported `[[1]]`,
`tpch/q14` expected 10 and `[[3,3,3,3]]` at `bp-tp4-sized` against `[[1,1,1,1]]`. T19's third batch
is what showed the general shape — `tpcds/q93` expects `[[7486]]` and reports `[[7169]]`, which is
neither one nor a group count but is exactly that node's own output.

So a fix keyed on the no-group-key case would have passed q93 and been wrong; the number to correct
is what the consuming side records, not what the aggregate emits.

Confirmed out of sample: `tpcds/q38`, found seven batches after this was rewritten, reports
`[[11788]]` against the CPU's `[[12446]]` — again the node's own output, and again neither one nor a
group count. Eight device cells: `tpcds` q96 q48 q93 q38, `tpch` q3 q14.
`q48` and `q93` are also the first cells in this rollout where a device COMPLETED a plan and the
golden caught the disagreement — every other device failure so far has been a refusal.

<a id="t187"></a>
### #187 — the device widens a decimal the plan declared narrow

`tpch/filter-project` at all five modes: `expected Decimal128(15, 2) but found Decimal128(38, 2)`
at the unload. The query is `SELECT l_orderkey, l_quantity FROM lineitem WHERE l_quantity > 30` —
no arithmetic anywhere, so this is the export or the filter widening the column rather than a scale
rule at a binop.

The CPU has a rule for this: `widened_decimal` (`cpu_backend.rs:499`) accepts a produced decimal
wider than the declared one at equal scale, and `declared_as` casts it back — over every exec
stage's output, not only a merged state, though its doc argues the merge that motivated it. The
device's unload has no such rule: `gpu_backend.rs:166` concats against the sink schema and arrow
refuses the type outright.

The value in question is the device's own export; DataFusion produces `(15,2)` throughout and the
CPU tier is green. So this is two rules for produced-against-declared, one per engine, disagreeing
on the same bytes — one tolerating and casting back, the other refusing. Whichever is right, one of
them is wrong. Neighbour of [#163](../tickets.md#t163) for that reason, where
[#183](bp-tickets.md#t183) is two representations of one value rather than two verdicts on it.

First plain decimal projection to reach a device in the corpus: q6's decimals are sums, whose
declared type is already wide, which is why twenty queries went past this and the twenty-first did
not.

Two sightings, and the pair narrows it: `filter-project`'s projected column declares `(15,2)` and
`hash-join`'s sum declares `(25,2)`, and both are found as `(38,2)`. Same scale, same 38, two
different declarations — so the export appears to produce one width rather than widening each
value by a step, which is a different fix from a scale rule and points at the export rather than at
anything upstream of it. Six device cells across T19's first two batches.

<a id="t188"></a>
### #188 — the device refuses a read with row groups and a limit together

`tpch/scan-limit` at three modes: `CUDF failure … row_groups can't be set along with skip_rows and
num_rows`. The plan puts the interval in the scan, so the recipe carries both a row-group list and
a row range, and the reader takes one or the other.

The same plan shape as [#186](bp-tickets.md#t186) from the other side: where the interval sits in
the scan, the CPU ignores it and answers six million rows and the device refuses the read outright.
Neither engine runs it and they fail differently, so a fix for either has to decide what that shape
means — push the limit into the reader, or keep the interval on the unload at every mode as the
three tp4 ones already do.

<a id="t195"></a>
### #195 — the two engines cut a node into different batches, and the byte total follows

Six device cells, all reached for the first time now that [#183](#t183)'s cast lets a string sink
complete: the per-node `output_bytes` disagree, and so do the per-batch lists beside them.

The delta is exactly the per-batch structural overhead of the boundaries the cpu has and the device
does not — `(rows+7)/8` per column plus `(rows+1)*4` per var-length column, charged once per batch.
`tpch/cross-join` at bp-tp1-single: five cpu batches of 25 totalling 22990 against the device's
22898, and 92 is what five charges over 25 rows cost above one charge over 125.
`nested-loop-left-join`: two batches, 9477 against 9454, 23 to the byte. Recomputed from the parquet
the whole-node figure is the device's, so neither side's arithmetic is wrong — they are summing
different numbers of batches.

Which side moves is the open half: the golden asserts the per-batch lists, so a device that
legitimately batches differently needs the golden to say so, and one that should batch as the cpu
does needs the driver to. Cells: `tpch` cross-join, nested-loop-left-join, q15, q4, q20, q21.

<a id="t192"></a>
### #192 — tpcds/q64 needs 13 GB of host memory and no budget stops it

ONE mode measured — 11.2, 12.3, 12.7, 12.9, 13.1 GB on a 15 GiB host, alone at `--test-threads=1`,
stopped rather than finished, and which mode is not recorded. All five are disabled by decision
rather than by measurement, and not re-run.

Context rather than the reason for that scope: the plan is the corpus's largest at 399 nodes and 102
`GpuCoalesceAllBatches` over 37 Inner joins, and its `estimated_max_resident_size` sums per mode to
35.5 and 35.9 GB at the tp1 modes against 64.8 to 65.4 GB at the tp4 ones. That model counts
DEVICE-resident bytes where 13.1 GB is host RSS from a `CpuBackend` run — not the same quantity.

Nothing refused it because the corpus tier passes no budget: `run_cpu` calls
`batch_partitioned_driver(tree, &ctx.task_ctx(), None)`. Enabling it is a decision about every run
of that tier — 13 GB SIGKILLs a runner as an infrastructure failure, not a test failure.

<a id="t191"></a>
### #191 — the device exports Int16 for an extracted year the plan declared Int32

`tpch/q8` at `bp-tp1-single`: `the exported stream is not the sink's rows: expected Int32 but found
Int16 at column index 0`. That column is `o_year`, an `extract(year from o_orderdate)` — DataFusion
types it `Int32` and the device answers `Int16`.

**Not [#187](bp-tickets.md#t187), and merging them would lose the distinction.** That one is the
device *widening* a decimal, to 38 whatever the declaration says. This is the device *narrowing* an
integer, to the natural width for a year rather than to a maximum. Opposite direction, different
type family, and a fix for either says nothing about the other.

Not new behaviour either, only newly reached: `diag_flip_audit`'s header already records
`extract_year -> INT16` as a place where the DataFusion type is an imperfect proxy for the cuDF one.
What is new is a corpus query whose unload sees it.

One cell, `tpch/q8` at `bp-tp1-single` — which is the only mode that gets far enough to reach the
unload, the other four stopping at [#152](../tickets.md#t152).

<a id="t190"></a>
### #190 — the CPU backend drops a nested-loop join's projection

`tpch/q11` at all five modes: `the node declares Schema { … 2 fields } and DataFusion answered with
Schema { … 3 fields }`. The extra column is the build side's scalar, which the node's projection
drops.

`cpu_backend/join.rs:140` builds `NestedLoopJoinExec::try_new(build, probe, Some(filter),
&join_type, None)` — that last argument is DataFusion's projection, passed `None`. Forty lines
down, the hash-join path at :302 reads `node.projection` and passes it. One join family applies the
projection the plan declares and the other ignores it.

The projection is not missing from the plan: `check_projection` validates it, the plan golden
carries it, and the node's declared schema is derived from it. Only the executor ignores it.

**It refuses rather than answering wrongly by luck.** `declared_as` compares column counts before
anything reads a value, so a projection that drops a column changes the count and is caught. A
projection that reorders columns, or drops one and leaves the same count, would have produced a
wrong answer with matching shapes and nothing to catch it.

Why the corpus took until T19's sixth batch to reach it: `q11` is the first query whose nested-loop
join projects at all. `nested-loop-join`, `nested-loop-left-join` and `cross-join` are `SELECT *`,
so their projection is `None` and passing `None` is correct for every one of them.

Device half untested — the CPU refuses first, as with [#189](bp-tickets.md#t189).

<a id="t189"></a>
### #189 — the shuffle cannot hash a rollup's grouping-set id

`tpch/rollup-over-join` at the three tp4 modes: `GpuEmitPartitions lane 0: assigning the scatter's
lanes: External error: comet murmur3: Internal error: Unsupported data type in hasher: UInt8`. Both
tp1 modes pass — no shuffle, no hash, no failure.

A `ROLLUP`'s grouping-set id column is `UInt8` and the scatter hashes every group key including it.
A refusal rather than a wrong answer.

It lands on the linchpin. That hasher is what makes the CPU and the device place a row in the same
lane, so a type it cannot hash is a type neither engine can shuffle on — and the device half here is
untested, because the CPU refuses before a device sees it. Widening the supported set therefore
means widening the murmur3 conformance gate with it, which is why this is not the one-line arm it
looks like.

First cause in T19's rollout that is not a device cause: the six before it were the device refusing
or disagreeing. Three cells, `tpch/rollup-over-join` at `bp-tp4-single`, `bp-tp4-rowgroup` and
`bp-tp4-sized`.

<a id="t186"></a>
### #186 — the CPU backend ignores a limit pushed into the scan

`SELECT * FROM lineitem LIMIT 10` returns 6,001,215 rows at `bp-tp1-single` and
`bp-tp1-rowgroup`. A wrong answer, not a refusal.

At tp1 the planner puts the interval in the scan alone — `GpuLoadParquet: … limit=10` under a bare
`GpuUnload` — and `CpuSource::new` (`cpu_backend/source.rs`) never reads `node.limit`. At the three
tp4 modes the interval lands on the unload instead and all three are correct, which is why every
other limit query in the corpus passes.

The device does not share the hole: `recipe/node_writer.rs:93` writes `limit: node.limit
.unwrap_or(0)` into the scan's recipe. So the two engines disagree on this plan by construction,
and only on the plans where the interval sits in the scan.

Found by T19's first batch, and only because `scan-limit` declares `data_fusion_subset` — the
count rule is what caught it, where an exact compare against a frozen golden would have frozen the
wrong answer. `tpch/scan-limit` is disabled at the two tp1 modes on this.

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
