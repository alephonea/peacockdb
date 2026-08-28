# The declared schema on the wire, and the precision the export is never told

**This task closes [#187](bp-tickets.md#t187)** — the device widens a decimal the plan declared
narrow — by giving the export the precision it currently defaults to 38 for, rather than by casting
the result back.

Sibling of [`casts.md`](casts.md), which predicts the divergence and deliberately does not fix this
half. **Do that task first**: it writes down what the device returns for a declared type, and this
one changes the answer for decimals.

## Why it happens

cuDF's decimal type carries **scale but not precision**. `export_table_to_ipc` builds its metadata as
`col_meta.push_back({name})` — name only — so `decimals_to_arrow` falls back to
`metadata.precision.value_or(max_precision<__int128_t>())`, which is
`floor(128 · ln2 / ln10) = 38`. Every decimal exports at precision 38 with scale preserved. That is
why `tpch/filter-project` declares `(15,2)` and receives `(38,2)`, and why `q6` never hit it: its
sums declare `Decimal128(38,4)` already.

**This is a missing argument, not two engine rules disagreeing.** #187's current text frames it as
the CPU's `widened_decimal` and the device's concat reaching different verdicts on the same bytes.
That framing is wrong and the ticket should be corrected as it closes.

The Rust side already sends the value: `Field.decimal_precision` exists in `gpu_plan.fbs` and
`serialize_schema` fills it from `Decimal128(p, s)`. It is dropped on the C++ side, after arrival —
`TableResult` (`plan_executor.h:15`) is `table` + `column_names`, and once a column is in one, the
precision exists nowhere.

## The work

### 1. Write `output_schema` on the wire

`PlanNode.output_schema: Schema` already exists in `gpu_plan.fbs` and is documented "Output schema of
this node". The bp writer never fills it: `fb::PlanNode::create` appears once on this path, in
`Writer::push` (`recipe/writer.rs:97`), with `output_schema: None`. `serialize_schema` already fills
`decimal_precision`/`decimal_scale` from `Decimal128(p, s)`, so the field is the only thing missing.

**The schema is per fb node, and a plan node's schema is not it.** An earlier draft of this spec said
`GpuNode::schema()` is node-local with nothing to derive; that is true only of the nodes emitting one
seq. `GpuAggregateBatches` emits `CudfCoalescePartitions`, `CudfAggregate{Merge}` and
`CudfProject{finalize}`, and the first two output aggregate state rather than the node's columns.
`GpuJoin` emits a key project, the join, and a pad or narrow project, and the key project outputs the
probe's keys alone. Writing the plan node's schema on all of them declares something false for every
seq but the last, which is worse than the `None` there today — `union.cpp:35` reads this field, so a
consumer would believe it.

So the schema rides on `Payload`: the closure that builds a payload is the only thing that knows what
that payload outputs, and `Writer::push` — which takes a `Payload` and nothing else — stays the one
funnel. Most arms already have it: five emit the rows they were given, so their output is their
input, and the two aggregates carry `intermediate()`, the state schema the planner computed once.

`GpuJoin` carries no equivalent, and its join seq needs one — for Left and Full the join call emits
the per-call shape and the pad project is what makes the output the node's schema. **It gains an
`intermediate()` too**, computed in `translate/mod.rs` where `join.left().schema()` and
`join.right().schema()` are already read for the key ordinals, so the value is a concatenation of
things in scope rather than new work. The writer must not derive it instead: the C++ computes this
shape and the pad project assumes it, and a third guess in a task about deriving a value once would
be the wrong direction. **The pad project reads the same `intermediate()`**, which is what makes this
one value replacing an assumption rather than one more value to keep true.

Write it for **every** node, not only where a decimal appears: `fb_text.rs`'s own header warns that
"not set" and "set to zero" are different instructions to the executor, and a conditional wire format
is a worse thing to own than a slightly larger buffer.

The C++ then reads it where `TableResult` is built (`plan_executor.h:15`, today `table` +
`column_names` only) and carries a per-column precision alongside the name. `operators/union.cpp:35`
already reads an `output_schema` for the neighbouring problem — branches landing different
fixed_point scales — so this is an existing mechanism reaching one more node kind rather than a new
one.

**The precision is set on the Arrow side, not on `column_metadata`.** An earlier draft said
`export_table_to_ipc` sets `column_metadata.precision`; that field does not exist in 25.02, whose
`column_metadata` is `name` + `children_meta` (`interop.hpp:108`). It arrives in 26.02. Since 25.02
is the leg that runs — shad-gpu and `cpp-build-2502` both build it, and build-test.md's rule is that
a functional run there is the verification while 26.02 need only compile — writing that field would
compile on the leg that only compiles and break the one that runs, taking every device test with it.

`export_table_to_ipc` already goes `to_arrow_schema` → `arrow::ImportSchema` → `to_arrow_host` →
`ImportRecordBatch(array, schema)`. Rebuild the imported schema with `arrow::decimal128(p, s)` from
the declared fields before the batch import. An Arrow decimal128 is 16 bytes whatever precision it
declares — which is why cuDF can default it to 38 in the first place — so this is metadata and not a
cast, it needs no `#if CUDF_VERSION`, and it does not rest on a field one leg lacks.

25.02's own header documents #187 as intended: "since the precision is not stored for them in
libcudf, decimals will be converted to an Arrow decimal128 which has the widest precision that cudf
supports". That is the sentence this task makes untrue for our exports.

The open question is whether `arrow::ImportRecordBatch` accepts a schema whose decimal precision
differs from what `to_arrow_host` produced, or validates the two against each other. If it validates,
the fallback is a post-import cast on the Arrow arrays, which is heavier and is a decision to bring
back rather than take.

With precision on the wire, `export_type_for`'s `Decimal128(p,s) → Decimal128(38,s)` row stops being
true: the divergence is removed rather than absorbed. Update the row in
[`casts.md`](casts.md) and the prediction test that records it — the cast list never gains a decimal
arm, because pinning a fixable omission into a golden would make it look inherent.

**The C++ half was the open risk and it is closed.** The worry was that
`node_session.cpp:517` does `result.column_names = input.column_names` — names propagating from
*inputs* rather than from each node's declared output — which would mean precision had to be
threaded from an origin. It does not. `execute_node` holds the node it is executing:

```cpp
const fb::PlanNode* node = impl_->post_order[seq];   // node_session.cpp:190
```

and every `TableResult` built inside it (`:267` collapse, `:355`/`:399` repartition) is in that
scope, so `node->output_schema()` is directly readable once step 1 populates it. Line `:517` is
`NodeSession::slice_handle`, which has no node because it slices an existing handle — it copies
`input.column_names` from a `TableResult` that already exists, and precision rides along the same
way. **No threading from an origin is required, and no signature changes.**

### 2. The payload golden shows what step 1 writes

`fb_text.rs:229` formats fields as `{}:{:?}` over `f.data_type()`, so `bp-recipe-payloads.txt` prints
bare `Decimal128` while expressions on the same page print `Decimal128(23, 2)`. Two fields that are
on the wire are invisible to the golden whose job is to pin the wire — a change to either, including
one that broke step 1, would not move it. Render them for `Decimal128` fields.

That is not enough on its own, and the gap only shows when step 1 lands: rendering precision reaches
the schemas the file already prints — a scan's `file_schema`, a union's own `output_schema` — and
`PlanNode.output_schema` was not one of them. Twenty digests move and nothing in the text says why.
So `payload_text` prints `declares:` for **every** node, `unset` where the field is absent, which is
what makes the absence on the structural union a thing a reader notices rather than a blank.

### 3. Unit tests

Same idiom as [`casts.md`](casts.md) — build the node, run its recipe fn, `writer.finish()`,
`flatbuffers::root::<fb::GpuPlan>`, `node_at(seq)`, assert on payload fields.

- **precision reaches the payload** — assert `output_schema().fields()[i].decimal_precision()` is
  the declared 15 rather than 0. Reuses the `(name, precision, scale)` schema helper `casts.md`
  adds. **Red before step 1 lands**, which is the order `coding-style.md` asks for.
- **every node carries a schema, not only the ones with decimals** — a node of plain `Int64`
  columns still has `output_schema` set, since a conditional wire format is the thing step 1
  declines to own.

## What this task should also carry

`casts.md` leaves `exports=` predicting types that nothing compares against: the golden is checked
by eye and the runtime check at `unload` sees only the divergences the cast absorbs. The device tier
already holds both sides — it reads the cpu-authored section and has the exported schema in hand — so
"the types the device exported are the ones `exports=` predicted" is one comparison in a place that
already runs. It also covers the `StringType` arm, which the cast otherwise hides from every
observer: the assertion sees the export before the absorption, where a reader of results cannot.

## Restriction

**Code and test changes are limited to what is written above.** No refactor of `node_session.cpp`
beyond carrying one field alongside `column_names`, no generalizing the export metadata past
precision, no cleanup of `TableResult`'s neighbours. Anything else found on the way is a ticket.

## Goldens, and how each moves

| golden | how it moves | why |
|---|---|---|
| `bp-recipe-payloads.txt` | **bytes** change on every node; text changes on decimal fields | step 1 adds `output_schema` everywhere, step 2 renders precision |
| `bp-*.plans.txt` | **no change** | plan text renders the Rust tree, which already knew the precision |
| `<mode>-<tier>.cpu.txt`, `.cost.txt`, `.result.txt` | **no change** | values and byte pricing are unaffected; only a declared type moves |
| `testdata/cost-registry.csv` | device cells move off #187 | see below |

The payload golden is the one to review rather than accept: its bytes move for every node in every
plan, and step 2 is what makes that diff legible instead of opaque.

## Device workflow

1. Run the **39 queries carrying #187** on `shad-gpu` with `build-test-shadgpu.sh`, in batches of
   about five, as T19 does. It was ten when this spec was written; `casts.md`'s rollout moved
   twenty-nine cells here from #183, which is the bulk of what that task produced. The list is
   `awk -F, 'NR>1 && $NF ~ /(^| )187( |$)/' testdata/cost-registry.csv`, and it is the list rather
   than a copy of it here, because it moves as the tickets do.
2. For each, **either enable the device cells or update the ticket**. The causes are ordered, so a
   cell that stops failing on #187 lands on whatever refuses next rather than going green. `casts.md`
   expected #152 and got #185 and #195 instead — thirteen cells each, both goldens disagreeing about
   a number rather than a crossing failing, so expect those here too. A cell whose cause changed is a ticket edit, not a cell that stays
   where it was.
3. Close #187 only when its cells are gone from the registry, and correct its text as it closes.
