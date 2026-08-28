# Answering with nothing: the table no call can build

Two refusals, one wall. A node owes rows it did not receive, and every entry point on the surface
loads a table by reading one — so there is no call to make.

- [#173](../tickets.md#t173) — a collapse of no handles, a merge of no runs, and a finish whose
  probe produced no keys.
- [#175](../tickets.md#t175) — a join whose build side produced no batch, where `Right`, `Full` and
  `RightAnti` owe their probe side. #175 says it itself: *"the same wall as #173, reached from the
  join instead of the accumulator."*

**Both close here.** They are one task because they are one missing capability seen from two nodes,
and separating them would mean building it twice.

**Depends on [`wire-schema.md`](wire-schema.md), and that dependency is the whole reason this is
small.** Both tickets say "unfreezing buys a make-empty-of-schema call and the refusals go", written
when nothing on the wire carried a bp node's schema. Once `PlanNode.output_schema` is populated, the
call is not needed at all: `execute_node` already holds the node
(`node_session.cpp:190`), so it can answer with an empty table rather than throw. **No new ABI
symbol.** The C++ said so before either of us did — `node_session.cpp:257`:

> *"A collapse of nothing has no schema to answer with: **the node's own `output_schema` is absent on
> a recipe plan**, and concatenating no views gives a table of no columns, which is not a batch
> anything above can read."*

## 1. #173 — an empty table of the declared schema

**One site refuses, not three.** This spec listed three, taking two of them from doc comments in
`gpu_backend/accumulate.rs` that say "the device refuses that (#173)". The code there has never
refused: both arms return `Ok((Vec::new(), ..))` and emit nothing, exactly as their CPU counterparts
at `cpu_backend/accumulate.rs:139`, `:197` and `:270` do. Making them call the device — which now
answers with an empty table — would have the device emit one empty batch where the CPU emits none: a
disagreement introduced by a fix for a disagreement. The comments are corrected here; the code is not
touched.

| site | what owes rows |
|---|---|
| `cpp/src/node_session.cpp:261-266` | a collapse with no input handles |
| `gpu_backend/join.rs:236` | a finish whose probe was empty, so it has no keys to join against |

The join's finish is where the two engines genuinely differ. `cpu_backend/join.rs:257` concats the
accumulated keys against an explicit `key_schema`, so with none accumulated it gets an empty keys
batch rather than an error, runs the finish against it, and answers; the device refuses.

In `execute_node`, build the answer from `node->output_schema()`: one `cudf::make_empty_column` per
field (`cudf/column/column_factories.hpp:43`), assembled into a table with the declared names.

The join's finish then needs no case per join type. The keys it joins against come from a collapse
over the accumulated key batches, and a collapse of no handles is what now answers with an empty
table of its declared schema — so the device makes the same two calls the CPU makes and gets the same
answer. `finish_without_keys`'s five arms, which describe what cannot be built rather than building
it, collapse into making the calls.

**The global aggregate is the exception and must stay one.** #173: *"a global aggregate owes its
identity row whatever arrived"* — `count` is 0, not absent. An empty-of-schema answer there would
drop a row DataFusion produces, so the aggregate arm keeps its own path and this task must not
collapse the two into "empty input, empty output".

**The CPU backend already emits nothing in the same places**, deliberately, so the two engines agree.
Check each site against its CPU counterpart as it changes: the point is not that the device stops
refusing, it is that both engines answer the same thing.

## 2. #175 — the probe side, padded or not

`empty_build_answers_nothing` (`nodes/join.rs:563`) is already the right decision in the right place;
what is missing is what to do when it returns false. Its own doc names the split:

> *"what they owe is the probe side, **padded or not**"*

- **`RightAnti` owes the probe side unpadded.** Its output is the probe columns alone, and an empty
  build side makes every probe row unmatched — so the answer *is* the probe batch. **Route it, do
  not call.** No pad, no kernel, no handle beyond the one the driver holds.
- **`Right` and `Full` owe it padded**, with typed NULLs in the build columns. That is the mirror of
  the pad that already exists: `ProjectRole::NullPad { nulls }` with `pad_project` and
  `padded_columns` (`recipe/join.rs:114`, `:261`, `:423`) appends one NULL per **probe** column a
  build-preserving join's projection keeps. This needs the same shape counting **build** columns.

So the recipe gains one role, not a mechanism. Whether that is a second `ProjectRole` variant or a
side on the existing one is the author's call — but the name must say which side is being padded,
since a reader who assumes the existing direction gets a plan that type-checks and pads the wrong
columns.

**Two corpus queries are waiting on it**, both found by T19: `tpch/q21` at `bp-tp4-single`, and
`tpcds/q77`, whose `Right` outer at four lanes gets no build side. q77 is currently out of the
end-to-end list with `tpch/q2` carrying its claim, because writing the CPU pad alone would make the
oracle answer a query the device refuses. **Put q77 back on that list as part of this**, or the
reason it was removed outlives the reason.

## 3. Tests

### The contract both engines owe

Deleting `finish_without_keys` gives four join types an answer they did not have — LeftAnti,
LeftSemi, LeftMark and Left/Full — and no golden can catch a wrong one, because the old code refused
rather than answering differently. The two engines reach these answers through separate code:
`cpu_backend/join.rs` has its own pad beside `recipe/join.rs`'s, so "the cpu does the same" is the
claim needing proof rather than the proof.

Four rows in `executor_cases.inc`, one per type, each a finish whose probe produced no keys. The cpu
backend runs them in `test_cpu_executors` and the device runs the same rows in `test_gpu_executors`,
so a disagreement between the engines is what goes red.

**This costs more than four rows and is authorised anyway.** Every existing case is one node with one
input, and neither harness mentions `GpuJoin`; a finish-with-no-probe needs a build side, a probe
that produced nothing, and `finish_and_fetch` driven at done — a two-input node with a lifecycle, so
both `emitted` arms grow. Estimated 150-250 lines across three files plus a device cycle. It lands
**before** section 2: #175 puts a second pad into one of the two engines, and the per-type contract is
what that pad then rests on. Written afterwards it is the same work with the pad already on top of it.

If it runs materially past that estimate, four gtests plus a ticket is the honest fallback — and the
ticket is **the two pads are compared by nothing**, not "executor_cases rows are owed". The first
names the shape this chain has met four times and can be prioritised against
[#198](../tickets.md#t198) and [#199](../tickets.md#t199); the second reads as a chore and is
deferred forever. Written here because a ticket filed at the moment of abandoning a harness is the
one most likely to describe the harness instead of the defect.

### Unit, Rust

- **the empty-build decision by type** — `empty_build_answers_nothing` returns true for the six that
  end the lane and false for `Right`, `Full`, `RightAnti`. It exists; assert it names all nine so a
  tenth type cannot be added silently.
- **`RightAnti` routes rather than calls** — a lane with an empty build side and a probe batch emits
  that batch and makes no ABI call. The absence of the call is the assertion, not the rows.
- **the build-side pad counts build columns** — the mirror of
  `the_pad_project_appends_one_null_per_probe_column_the_projection_keeps`, written the same way and
  next to it, so the two directions are read together. A projection keeping no build column pads
  nothing.
- **a pad in the wrong direction is caught** — assert the emitted NULL count against a join whose
  build and probe widths **differ**. With equal widths both directions pass, which is how this ships
  wrong.

### Unit, gtest

- **a collapse of no handles answers an empty table of the declared schema** — column count, names
  and types from `output_schema`, zero rows. The test that pins the fix.
- **the types are the declared ones, not defaults** — a schema with a `Decimal128(15,2)` and a
  string, asserting both come back as themselves. An empty table of the wrong types is the failure
  this cannot afford, since nothing downstream has rows to notice with.
- **a global aggregate with no input still emits its identity row** — the exception, asserted rather
  than assumed, because the natural implementation of everything above deletes it.

### Recipe walk

- **a query with an empty lane runs end to end** — the existing harness with a shape that leaves one
  lane with no build side, driven at `target_partitions` > 1. Today it refuses; this is the first
  test where an empty lane reaches the unload.

## 4. Goldens and the device workflow

| golden | moves | why |
|---|---|---|
| `bp-*.plans.txt` | **yes**, where a Right/Full join gains a pad | a new recipe call renders in the node's recipe line |
| `bp-recipe-payloads.txt` | **yes** | the pad project is a payload |
| `testdata/cost-registry.csv` | **yes** | cells move off #173 and #175 |
| `<mode>-<tier>.cpu.txt`, `.cost.txt` | **yes**, for the queries that newly run | new sections, not changed ones |

Device work, in batches of about five on `shad-gpu` with `build-test-shadgpu.sh`, as T19 does: the
two known sightings first, then any cell whose ticket names #173 or #175. Expect the freed cells to
land on whatever refuses next rather than going green — the causes are ordered, and since `casts.md`
and `wire-schema.md` closed #183 and #187, what sits in front of most of the corpus is
[#185](bp-tickets.md#t185) and [#195](bp-tickets.md#t195), which hold 60 of the 63 device cells still
disabled. **Close #173 and #175 when their cells are gone from the registry**, not when the code
lands.

The registry is a smaller worklist than this spec assumed: no row cites #173 and two cite #175. #173
is reached from the accumulator and the collapse rather than from a disabled cell, so its proof is
the gtest and the recipe-walk case rather than a rollout.

## 5. Out of scope

The other frozen-surface refusals. This task buys exactly two capabilities — an empty table of a
declared schema, and a probe side passed through padded or bare — and every other "the surface
cannot express this" stays where it is. If a third refusal looks like it would fall out for free,
that is a ticket, not an addition.
