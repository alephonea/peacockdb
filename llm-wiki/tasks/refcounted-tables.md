# Refcounted tables: one allocation, several handles

A handle owns its table exclusively — `TableResult` (`cpp/src/plan_executor.h:15`) holds a
`std::unique_ptr<cudf::table>` — and `execute_one` takes its inputs **by value**
(`cpp/src/operators/dispatch.cpp:110`), so the session moves a table out of its registry to run a
call. Two consequences, and this task is both of them:

- [#152](../tickets.md#t152) — a streamed probe calls a join once per batch and the first call takes
  the build side away. Both halves: the build side across batches, and a Left or Full join whose key
  project and per-call join read the same probe batch.
- [#145](../tickets.md#t145) — the scatter deep-copies every partition out of one contiguous table,
  so a shuffle copies its input twice and peaks at twice the data ([#91](../tickets.md#t91)).

**This task closes #145 and #152 and unblocks [#140](../tickets.md#t140).** #152 is the rollout's
largest single cause: 75 corpus queries carry it.

**Out of scope, deliberately: memory accounting.** Sharing changes what a release frees, so the
driver's resident model will diverge from the device — see the last section. Nothing here fixes it.

## 1. `TableResult` shares its owner

```cpp
struct TableResult {
  std::shared_ptr<cudf::table> owner;
  cudf::table_view view;
  std::vector<std::string> column_names;
};
```

**39 sites across 11 files** touch `.table` / `->table`: `node_session.cpp` holds 17,
`operators/join.cpp` 6, `filter.cpp` 4, `union.cpp` 3, `project.cpp` and `gpu_executor.cpp` 2 each,
and one apiece in `window`, `sort`, `limit`, `dispatch` and `aggregate`. Mechanical but wide.

`execute_one`'s signature does **not** change. Operators keep receiving owned inputs; those inputs
simply stop owning exclusively. That is the point of doing this at the member rather than at the
interface: the alternative — passing views so the registry can retain its entry — is the same 11
files with an interface change instead of a type change, and every operator has to be re-read
rather than re-typed.

Erase-on-read also does **not** change. The legacy driver depends on it: `Backend::release` has one
caller, `node_by_node.rs:87`, releasing only the root handles, so every intermediate on that path is
reclaimed by being read. What changes is what erasing costs — dropping a reference, not destroying a
table.

## 2. `peacock_handle_retain` — the one new ABI symbol

Refcounting makes a second handle cheap; it does not mint one. The driver still holds a single build
handle, the call still consumes it, and nothing else names the table. So the surface gains one
symbol, shaped like `peacock_executor_slice_handle`, which is the existing operation that takes a
handle and produces another:

```c
/// Mint a second handle on the same resident table. The two are independent names: consuming
/// or releasing either leaves the other valid, and the table lives while any handle on it
/// does. O(1) — no rows are copied. A failure leaves the session standing.
/// @return 0 on success, non-zero on failure (an unknown handle is a failure, not a new
/// handle to nothing).
int peacock_handle_retain(peacock_executor_t* executor, uint64_t handle, uint64_t* out_handle);
```

**It returns a new id rather than bumping a count on the old one**, because
`peacock_handle_release` is documented idempotent. If release decremented, calling it twice would
drop two references and that guarantee would be gone. Under a new-id design, releasing an
already-consumed id stays a no-op and the existing contract holds unchanged.

So the pairing is not retain/release on one id. **Every handle — original or retained — is disposed
of exactly once**, by being consumed by a call or by `peacock_handle_release`. Retain adds one
obligation; it attaches none to the original.

Correct [#145](../tickets.md#t145) as this lands: its "No ABI change: a handle stays a `u64`" is
true of the scatter and not of #152. The handle stays a `u64`; the surface gains a symbol.

## 3. How the driver uses it

**The probe batch count is not known in advance** — the driver pulls batches until the source is
exhausted — so "give the last call the original handle" is not implementable. Retain before every
call, release the original once at the end:

```rust
let build = /* one handle, from the build side */;          // H1
loop {
    let probe = match source.next_batch()? {
        SourceStep::Batch { batch, .. } => batch,
        SourceStep::Done => break,
    };
    let build_for_this_call = retain(build)?;               // H7, H8, H9, …
    execute_node(join_seq, &[vec![build_for_this_call], vec![probe.handle()]]);
}
release(build);                                             // H1, never consumed
```

Three probe batches: `H1` retained three times, each retained handle consumed by its call, `H1`
released. Four handles, one table, no copies. The `B - 1` variant — last call takes the original —
is unimplementable for a stream and an off-by-one either way: one short is a use-after-erase, one
over is a handle resident until `end_plan`.

The probe-batch case is the same symbol inside one batch's chain:

```rust
let keys_input = retain(probe)?;
let keys = execute_node(key_project_seq, &[vec![keys_input]]);
let out  = execute_node(join_seq, &[vec![keys], vec![probe]]);
```

Why the driver and not the session: the C++ is sent **a menu of parameterized kernels, not a
schedule** (`recipe/mod.rs` header). Two seqs receiving one handle is a fact about the driver's
chain, absent from the wire by design, so no node-kind rule in `execute_node` can derive it. A
session-side rule keyed on node type would also be behaviour switched implicitly by which node you
happen to be, which `coding-style.md` records as an antipattern.

## 4. Recipes, and the refusals that go away

`Input::BuildSideCopy` and `Input::BatchCopy` stop describing a copy and start describing a second
name for the same table. Their docs currently say "The surface has no copy symbol, so a recipe
naming this is one an executor refuses until #145" — that sentence is the change.

**Rename both.** `coding-style.md`: an inaccurate name is worse than a vague one, and after this
there is no copy. `BuildSideAgain` / `BatchAgain` say what the call needs — this input, once more,
and a later call still will — without naming a mechanism that no longer exists.

The runtime refusals go entirely. `gpu_backend/join.rs`:

- `copy_of` (`:292`) — *"this join's recipe copies its probe batch … and the ABI has no copy, so
  neither call can run without erasing the other's input (#152)"* — deleted; it retains.
- `build_copy` (`:302`) — *"probe batch N has no build side left, since the call for batch 1 erased
  it"* — deleted; it retains.

Both messages name the absent symbol as the cause. Leaving them would point the next reader at the
wrong fix; deleting them is part of the work rather than tidy-up after it.

`recipe/tests.rs:143` keeps asserting the recipe names the input — the recipe was always right, and
what changes is only the name it uses.

## 5. The scatter

`node_session.cpp:398-400` materialises each partition:

```cpp
// One owning table per partition (slice → deep copy so each handle owns memory).
cudf::table_view slice = cudf::slice(pv, {start, end}).front();
part.table = std::make_unique<cudf::table>(slice);
```

becomes one owner and N views:

```cpp
auto owner = std::shared_ptr<cudf::table>(std::move(parted));
for (size_t p = 0; p < n; ++p) {
  TableResult part;
  part.column_names = column_names;
  part.owner = owner;
  part.view  = cudf::slice(pv, {start, end}).front();
  …
}
```

Retire the comment with the copy: *"so each handle owns memory"* is the invariant being dropped, and
a comment left describing code that no longer does that is worse than none.

**The scatter needs no `retain`.** Its N names are minted by the session inside one call and handed
back through `out_handles`; the driver consumes or releases each exactly once, as it does today.
This is the half of #145 for which "no ABI change" is straightforwardly true.

What is pinned is `parted`, the **post**-scatter table, so a surviving partition holds every
partition's rows. That is #145's "a slice pins its whole parent" in concrete form: the peak halves
and the tail lengthens.

**One consequence that is not memory.** The per-partition timers measure exactly the copies being
deleted — the function says so: *"Only the per-partition slice copies below are separable."* After
this, partitions 1..N-1 do a `shared_ptr` copy and a `cudf::slice`, and their timed regions collapse
toward zero. The structure holds (N partitions still cost N regions, which is what that comment's
warning protects), but the per-partition times in the benchmark tree drop toward zero. That is
true signal — the work genuinely went — but it lands in T22's output as a jump with no visible
cause, so it belongs in #145's text.

## 6. Tests

### The ABI symbol (gtest, `cpp/tests/gpu/`)

- **retain then consume one**: retain a handle, run a call consuming the retained id, read the
  original — the rows are still there. The single fact the symbol exists for.
- **release N−1, read the survivor**: #145's own proposed test, and the scatter's invariant.
- **retain an unknown handle fails**: non-zero return and no handle minted, matching
  `slice_handle`'s treatment of the same mistake.
- **the original outlives a consumed retain, and vice versa**: both orders, since "either leaves the
  other valid" is a claim about both.

### Recipe walk, joins over multiple batches (`test_gpu_recipe_walk.rs`)

The harness already names this ticket as its stopping point (`:441`):

> `"{}: {} probe batches, and the call consumes the build handle with no ABI symbol to copy it
> (#152) — every shape here plans one probe batch"`

- **A third knob set** beside `ONE_LANE` and `TWO_LANES`: `target_partitions: 1` with
  `BatchSizing::OneBatchPerRowGroup`, which gives many batches in one partition — multiple probe
  batches with no scheduling to reason about.
- **`Walker::join` loops** over the probe batches instead of asserting one, holding `At.build`
  across them and retaining per call.
- **`resolve` (`:212`) stops collapsing the two**: `Input::BuildSide | Input::BuildSideCopy =>
  held(at.build)` hands the same handle to both today, which is why a second call passes an id the
  first erased. The renamed inputs get distinct arms.

Write the loop **before** the fix if you can: it fails with `NodeSession::execute_node: unknown
input handle` from `node_session.cpp:252`, which is a minimal reproduction of #152 — one query, one
partition, no driver, no corpus. Today the defect is reachable only through 75 corpus queries'
device cells. Red before the fix is what `coding-style.md` asks for, and this is the first place it
has been cheap.

The file is at 852 lines against the 1000-line cap; `recipe/tests.rs` is at 887.

### The scatter, targeted

- **N handles, one allocation**: after a repartition, releasing N−1 leaves the survivor readable and
  correct — the gtest above, reached through a real scatter rather than by hand.
- **The partitions still hold what they held**: a repartition's N outputs concatenate back to the
  input's rows, which is the property the deep copy was incidentally guaranteeing.
- **Per-partition stats are unchanged**: `out_stats[p]` rows and `varlen_content_bytes` come from the
  view, and must not move when the copy goes. This is the assertion that catches a slice taken
  against the wrong offsets — the failure mode a deep copy made impossible.

## 7. Goldens

| golden | moves | why |
|---|---|---|
| `bp-*.plans.txt` (10 files) | **yes**, if the inputs are renamed | `Input` renders into the recipe line — `execute_node(#N CudfNestedLoopJoin, build copy, batch)` — via `recipe/types.rs:143`. Mechanical, every join with a streamed probe |
| `testdata/cost-registry.csv` | **yes** | device cells move off #152 — see below |
| `bp-recipe-payloads.txt` | no | no flatbuffer change; the wire is untouched |
| `<mode>-<tier>.cpu.txt`, `.cost.txt` | no | the corpus cpu tier runs `CpuBackend`, which holds no handles |
| `bp-<tier>.result.txt` | no | no value changes |
| benchmark output | **yes** | per-partition times collapse, per the scatter section |

**The registry is the slow part, not the code.** 75 queries carry #152. Each needs a device run and
then either an enable or a ticket edit, in batches of about five on `shad-gpu` with
`build-test-shadgpu.sh`, as T19 does. The causes are ordered, so a cell that stops failing on #152
lands on whatever refuses next rather than going green — expect [#183](bp-tickets.md#t183) at the
unload, which is what `casts.md` closes. **Close #152 and #145 when their cells are gone from the
registry, not when the code lands.**

## 8. Memory accounting — out of scope, and why it will need doing

The driver's model (`driver/accounting.rs`) is

```text
resident = Σ byte_size of driver-held in-flight batches
         + Σ cached resident_bytes() over live executors
```

and it prices each batch independently from its schema. After this change a released handle's bytes
leave the model while the memory stays on the device behind a sibling — so the model **under**-reports
residency, and the enforcer at `driver/partitioned.rs:890` reads it.

The eventual answer is that the summing belongs in the backend: a device-reported figure is exact
however sharing arises, and `GpuBackend::resident_bytes()` already exists as the hook. What cannot
move is the rest — `begin_call`'s pre-call model refuses **before** allocating, `Underestimate`
compares model against measurement, and `hops()` proves the driver's own ledger balances. A measured
number answers none of those, and `memory.rs:128` forbids an allocation-dependent figure in a golden
at all.

**Do none of it here.** Take the divergence as a ticket when this lands. It is the first time the two
numbers disagree for a legitimate reason, and nothing today would notice.
