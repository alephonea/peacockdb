# join capability through the recipe plan (T0 extension)

**Goal.** Establish, by execution rather than argument, that the frozen FlatBuffers schema
and C++ operators can run **every** join mode in the batch-partitioned model — with the
build side complete and the probe side arriving in batches — and write the per-mode
lowering down where the implementation tasks will read it.

**What was built** (all in `scripts/exec_model/`, coordinator-owned):

- `operators/join_types.py` — `JoinType` in the fbs vocabulary and the capability matrix as
  one function, so no backend can hold a different opinion of it.
- `operators/cudf_calls.py` — the cuDF calls `cpp/src/operators/join.cpp` makes, at their
  own signatures: joins that return gather maps, `gather` with its out-of-bounds policy,
  `scatter`, `apply_boolean_mask`, `cross_join`.
- `operators/recipe.py` — the fb node structs, a handle registry that consumes on read as
  `NodeSession` does, and the node implementations mirroring join.cpp branch for branch.
- `operators/recipe_join.py` — the second join backend: every call answered by emitting fb
  seqs and making `execute_node` calls.
- `operators/joins.py` — the pandas backend, widened from five join types to all nine plus
  cross and nested loop.

**Constraints.** The two backends share no join code; agreement between them is the
evidence. The recipe backend may not reach for python where the frozen surface has no
answer — it names the gap and counts what working around it costs (`copy_handle`).

**Verification bar.** `scripts/exec_model/tests/test_end_to_end.py`: every join mode
against a SQL oracle, on both backends, at five batching/partitioning configs and across
the layout injector's presets; the emitted seq sequence asserted per mode against the
spec's table; the per-batch copy counts asserted per family; every refused shape refused
loudly on both backends. Whole prototype suite green, ~90 s.

**Outcome.** Recorded in the spec's [join capability
matrix](batch_partitioned_executor.md#join-capability-matrix): every mode is expressible,
the streamed-probe copy cost is [#152](../tickets.md#t152) quantified per family, and one
shape turned out to be a defect in the shipping engine rather than a limit of the mode —
[#153](../tickets.md#t153).
