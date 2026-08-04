# Archived historical comments

Decisions that were once documented in code comments and removed during the 2026-07-31
comment audit. Kept because the *reasoning* still explains why the current code looks the
way it does; removed from the source because the code no longer matches the narration.

## batch_size was budget-derived and rendered into goldens (removed 2026-07-31)

`GpuScanExec.batch_size` / `GpuCoalesceBatchesExec.target_batch_size` were computed as
`gpu_memory_budget / subtree_max_row_bytes` with **no data cap** — a 300-row table
carried `batch_size=619466436` — and were rendered into every `.cpu.txt` / `.plan.txt`
golden, pinning the goldens to the memory-budget value.

They were proven **vestigial for observable output**: the field only sets the parquet
reader's chunk size, never *which* rows are read (the row-group→partition map comes from
`build_scan_map(rgs, n_parts)`, which the budget is not an input to), and every golden
quantity is a per-node aggregate, so chunking cancels. Verified empirically at tp1 and
tp8: remapping the standard tier 120 GiB → 12 GiB left every `output_rows`,
`output_bytes`, per-partition line and plan shape byte-identical.

Resolution: stop rendering both tokens and regenerate goldens from the renderer (never
hand-edit goldens). The field is still **serialized** — this was a display-only change.

## The pre-PartitionMode budget threshold (removed 2026-07-31)

Before `PartitionMode` existed, real multi-partition scans were gated on a minimum GPU
memory budget (a 16 GiB `REAL_PARTITION_MIN_BUDGET`): below it a device was treated as
memory-constrained and the scan stayed single-partition so the plan serialized
byte-identically to the legacy scan and the FlatBuffer round-trip stayed stable
(deserialize does not reconstruct N partitions, so `GpuRepartitionExec.input_partitions`
would not flip 1→8). That budget threshold was replaced by the explicit `PartitionMode`
enum, which is now the sole discriminator.

## thread_local as an output side-channel (eliminated 2026-07-31)

`execute_one` / `execute_node` passed node inputs through an anonymous-namespace
`thread_local` pair (`g_node_inputs`, `g_node_input_idx`) plus an RAII `Restore`. Because
an anonymous namespace gives **every translation unit its own copy**, splitting the C++
monolith would have silently forked the variable: `execute_node` would read a
permanently-null copy and fall through to re-executing whole child subtrees from parquet
— correct answers, exponential cost, and invisible to goldens, byte digests, round-trip
and result comparison alike.

Replaced by an explicit `NodeInputs{items, idx}` parameter threaded through
`execute_one → run_op → execute_node → every execute_*`, plus the **consumed == provided**
invariant in `execute_one` (a node handed inputs must consume all of them), which turns
that whole bug class into a loud failure in every NodeSession-driving test. See
`llm-wiki/coding-style.md`.

## strip asymmetry (kept in code, recorded here for context)

`strip_gpu` strips 11 of the 16 operator wrappers; 5 deliberately do not strip
(cross join, nested-loop join, union, global limit, window). This is **load-bearing**:
flipping one changes execution substitution and the reported `NodeMemoryStats.node_name`.
The reason lives on `GpuCrossJoinExec::strips_to_inner` in `cpp`-adjacent Rust
(`operators/join.rs`); the others reference it. Contrast `GpuInterleaveExec`, which *is*
stripped so `build_stream` can rebuild it as an equivalent `UnionExec` (its
single-partition stubs mean `InterleaveExec::try_new` cannot interleave).
