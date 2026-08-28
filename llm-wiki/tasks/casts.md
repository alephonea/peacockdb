# Export types: predicted at plan time, carried on the wire

Two device cells fail on the same class of thing — the device hands back a column whose Arrow type
is not the one the plan declared — and each is currently discovered at the boundary rather than
predicted before it. This task writes the prediction down where it can be checked, and closes both.

**This task closes [#183](bp-tickets.md#t183)** — the device exports `Utf8` where the sink declares
`Utf8View` — by predicting the export type at plan time and casting the one divergence that is
inherent.

Its sibling [`wire-schema.md`](wire-schema.md) closes [#187](bp-tickets.md#t187) with the same
derivation and the C++ half this task deliberately leaves out. Do this one first: it is Rust-only,
and the prediction it writes down is what the other one then makes true for decimals.


## Why they happen

`unload` concatenates the decoded IPC batches against the sink's declared schema
(`gpu_backend.rs:166`), and `concat_batches` requires exact type equality. Neither side is
misbehaving:

- cuDF has exactly one string type. `expr.cpp:74` maps the `Utf8View` tag to `type_id::STRING` and
  `cudf::to_arrow_schema` maps that back to `arrow::utf8()`. The divergence is inherent and the
  cast is the only place to absorb it.
- cuDF's decimal type carries **scale but not precision**. `export_table_to_ipc` builds its metadata
  as `col_meta.push_back({name})` — name only — so `decimals_to_arrow` falls back to
  `metadata.precision.value_or(max_precision<__int128_t>())`, which is 38. Every decimal exports at
  precision 38 with scale preserved. That is why `tpch/filter-project` declares `(15,2)` and receives
  `(38,2)`, and why `q6` never hit it: its sums declare `Decimal128(38,4)` already.

The second is a missing argument, not a disagreement between two engine rules. #187's current text
frames it as the CPU's `widened_decimal` and the device's concat disagreeing about the same bytes;
that framing is wrong and the ticket should be corrected when it is closed.

## The work

### 1. `export_type_for` — the composition, written down once

The round trip Arrow → fb → cuDF → Arrow lives in three files and two languages today, so nobody can
answer "what type will the device hand back for this column?" without reading `expr.cpp`. Make it a
Rust function. It is total over the declared type for every case below.

| declared | fb tag | cuDF | exported | |
|---|---|---|---|---|
| `Boolean`, `Int8`–`Int64`, `UInt8`–`UInt64`, `Float32/64` | direct | direct | same | identity |
| `Utf8` | `Utf8` | `STRING` | `Utf8` | identity |
| `Date32` | `Date32` | `TIMESTAMP_DAYS` | `Date32` | identity, via `to_arrow_schema`'s `default:` arm |
| `Utf8View` | `Utf8View` | `STRING` | `Utf8` | **#183** |
| `LargeUtf8` | `LargeUtf8` | `STRING` | `Utf8` | same shape, no corpus query reaches it |
| `Date64` | `Date64` | `TIMESTAMP_MILLISECONDS` | `Timestamp(ms, None)` | no corpus query declares it |
| `Decimal128(p,s)` | `Decimal128` | `DECIMAL128` | `Decimal128(38,s)` | **#187** — predicted here, **not cast here**; [`wire-schema.md`](wire-schema.md) removes the divergence instead |
| `Null`, `Float16`, `Binary`, `LargeBinary`, `BinaryView` | mapped | **`EMPTY`** | — | **refuse** |

The last row is not a cast. `convert_data_type` serializes all five and `fb_to_type_id` has no case
for any of them, so they reach the device as a typeless column. `export_type_for` returns `Err` and
the plan is refused at planning time, which is what happens to them today only by accident.

### 2. Derive `exports` inside `attach_recipes`

Not a separate pass and not in `plan_serializer.rs`, which is the wire codec rather than a planning
phase. The recipe walk already delivers the input:

```rust
fn unload(
    _node: &GpuUnload,
    _inputs: &[&Schema],     // the sink's declared schema, already here, unused
    _writer: &mut Writer,
) -> Result<Option<Recipe>, PlanError>
```

`_inputs` is exactly what `export_type_for` needs, handed to the one function already positioned at
the sink. Deriving it there costs no second traversal and creates no second place that has to agree
with the first — the hazard `driver/index.rs:151` exists to guard for node numbering.

The result is a per-column identity-or-cast list, carried to `GpuSink::new`
(`gpu_backend.rs:124`) alongside the schema the driver already passes it.

**The cast at `unload` must be narrow.** `concat_batches` against the declared schema is today the
only check that the device produces what the plan says it produces; it is what surfaced #187. A
blanket cast-to-schema fixes #183 and destroys that. Only the arms the table marks as divergent are
cast; every other mismatch still fails, as `export_table_to_ipc`'s `DECIMAL32/64 → DECIMAL128`
widening is a named normalization rather than a general one.

Two review points, both of which lose a check if missed:

- `self.schema.fields().zip(batch.columns())` truncates silently on a column-count mismatch, which
  `concat_batches` catches today. Check the length explicitly.
- Non-string divergences would fail at `RecordBatch::try_new` rather than at `concat_batches`, so
  `"the exported stream is not the sink's rows: {error}"` moves with them or #187-class failures lose
  the message the tickets quote.

### 3. Unit tests

`recipe/tests.rs` already asserts on the buffer where the recipe cannot answer, with the idiom at
`a_finalize_project_emits_the_group_keys_the_finalize_list_leaves_out`: build the node, run its
recipe fn, `writer.finish()`, `flatbuffers::root::<fb::GpuPlan>`, `node_at(seq)`, assert on payload
fields. No GPU, no session, no plan load.

- **exports at the unload** — pure function of the input schema, so it needs neither the buffer nor
  `finish()`. A `GpuUnload` over `Utf8View` + `Decimal128(15,2)` + `Int64` derives a list naming the
  divergent columns and omitting the identity ones.
- **the `EMPTY` class refuses** — `Binary`, `Null`, `Float16` return `Err` rather than passing
  silently. This is the arm most likely to rot, since no corpus query reaches it.
- **a decimal predicts but does not cast** — `export_type_for` reports `Decimal128(15,2) →
  Decimal128(38,s)` and the unload's cast list omits it, so the prediction is recorded before its
  fix exists. `columns_of` hardcodes `DataType::Int64`, so this needs a sibling taking
  `(name, precision, scale)` — which [`wire-schema.md`](wire-schema.md) reuses.

`recipe/tests.rs` is at 887 lines against the 1000-line cap. These land around 930 — under, but the
next addition to that file forces the split rather than this one.

## Restriction

**Code and test changes are limited to what is written above.** No refactor of the surrounding
writer, no generalizing `export_type_for` beyond the table, no second cast site, no cleanup of
`plan_serializer.rs` while passing through it. Anything else found on the way is a ticket.

## Goldens, and how each moves

| golden | how it moves | why |
|---|---|---|
| `bp-*.plans.txt` (10 files) | every `GpuUnload` line gains `exports=` | new field; `GpuUnload` renders bare today |
| `<mode>-<tier>.cpu.txt` | **no change** | `memory.rs:43` prices `Utf8View` and `Utf8` identically at `(rows+1)*4`, and content size is Σ value lengths for both |
| `<mode>-<tier>.cost.txt` | **no change** | derived from the `.cpu.txt` sections, which do not move |
| `bp-<tier>.result.txt` | **no change** | values are unaffected; only types were ever in question |
| `testdata/cost-registry.csv` | device cells move off #183/#187 | see the device workflow below |

`exports=` renders on every unload, including `exports=none` where nothing diverges. Omitting the
attribute would make "nothing diverges" and "the list was never computed" identical in the golden,
which is the invisible-absence shape `coding-style.md` records twice.

`exports=` is a Rust-side prediction that the C++ never receives: the unload writes no payload, so
nothing on the wire describes the sink's columns. The golden is documentation and a tripwire, and
the runtime check at `unload` is the only thing that enforces it. [`wire-schema.md`](wire-schema.md)
is what makes it checkable against the buffer.

The tripwire fired during the rollout: the attribute predicted the decimal divergence on six rows
before they ran, and the device agreed on each. That is one arm of the table confirmed six times,
not the table confirmed — `StringType` is cast at the unload and so can never be observed as a
landing, and `DateAsTimestamp` has no corpus query. More batches add instances of the same arm.
Turning it into a claim about the table needs an assertion rather than more rows, and that is
[`wire-schema.md`](wire-schema.md)'s, not this task's.

## Device workflow

The cast and the precision are not proved by a green CPU tier — every cell they exist for is a device
cell that is currently disabled. After the code lands:

1. Run the affected corpus queries on `shad-gpu` with `build-test-shadgpu.sh`, in batches of about
   five, as T19 does. **59 queries carry #183** — every query with a string in its sink schema,
   which is derivable from `bp-tp1-single.plans.txt` without running anything, and was: no query
   without a string in its sink carries #183, over 82 checked, with no exceptions.
2. For each, **either enable the device cells or update the ticket**. A cell that now reaches a
   different cause is a cell whose ticket changes, not a cell that stays where it was — and the
   causes are ordered, so fixing #183 will move cells onto whatever refuses next rather than turning
   them all green. Expect #152 to be the common landing place.
3. Close #183 only when its cells are gone from the registry, not when the code lands. A ticket
   whose cells are still disabled against it is not closed.

What the rollout found: #152 was not the common landing place. The landings are of two kinds, and
neither is a refusal to cross. A cell whose sink carries a decimal still fails at the unload, because
`DecimalPrecision` is the arm this task predicts and declines to cast — [`wire-schema.md`](wire-schema.md)
removes it by putting the precision on the wire, so every one of those has its fix already specified.
Every other cell now completes its plan on the device and is held by a golden disagreeing about a
number: #185 where a node reports its own output as `in_rows`, #195 where the two engines cut a node
into different batches. The commit message on 4f6138b says the first kind stopped as well. It did
not, and #187's first line is the counterexample.
