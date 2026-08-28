# Task sequence

## 1. [`casts.md`](casts.md) — closes [#183](../archive/archived-tickets.md#t183)

Rust only, no FFI, no C++. Predict the export type at plan time in `attach_recipes`, render it as
`exports=`, and cast the one divergence that is inherent — cuDF has a single string type. The
decimal is predicted here and deliberately not cast. Moves ten `.plans.txt`; 59 queries carry it.

## 2. [`wire-schema.md`](wire-schema.md) — closes [#187](bp-tickets.md#t187)

Crosses the FFI but small: `Writer::push` is the one funnel that must fill `PlanNode.output_schema`,
and the C++ carries a per-column precision into `column_metadata`, which cuDF currently defaults to
38 because nobody passes it. Removes the divergence rather than casting it. Payload golden bytes
move, so `schema_text` must render precision in the same change or the diff shows nothing.

## 3. [`empty-answers.md`](empty-answers.md) — closes [#173](../tickets.md#t173), [#175](../tickets.md#t175)

Needs task 2 first, and that is what makes it small: with `output_schema` on the wire, `execute_node`
answers with an empty table instead of throwing, and no ABI symbol is needed. `RightAnti` routes its
probe side rather than calling; `Right` and `Full` want the mirror of a pad that already exists. The
global aggregate keeps its identity row — the natural implementation deletes it.

## 4. [`refcounted-tables.md`](refcounted-tables.md) — closes [#145](../tickets.md#t145), [#152](../tickets.md#t152)

The largest and the one that frees the most: 39 `.table` sites across 11 files, plus
`peacock_handle_retain`, the first new ABI symbol — needed because `execute_one` takes its inputs by
value, so the registry cannot keep a handle and let an operator own its input unless the owner is
shared. 75 queries carry #152. Memory accounting is deliberately out of scope and will diverge.
