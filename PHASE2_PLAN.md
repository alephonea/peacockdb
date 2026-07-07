# Task #13 Phase 2 — Implementation Plan (real GPU tp8 partitioning)

Branch: off `ENS-unified-node-exec` (so #79/Phase-1 stays independently mergeable).
Coordinator handles ALL git; dev holds commits. Each increment ends with a
GPU==golden proof point + a report before scaling.

## Goal
Real GPU 8-way (tp8) partitioning so the H200/tp8 **node-by-node** CPU-emulated
output == real-GPU output **exactly** (per-node rows + rows/schema cost), then add
the H200/tp8 goldens and verify. tp1 goldens stand unchanged.

## Two new dmitry constraints absorbed
- **C1 — ALLOC-OP = NODE.** Every cuDF op that ALLOCATES or TRANSFORMS a table is
  an explicit plan node: visible in the golden, cost+memory-budgeted, CPU-emulated.
  In Phase 2 that means THREE op-nodes:
  - (a) **CONCAT** for hash-repartition M→1 = `GpuCoalescePartitions` (BUFFERING in
    resident.rs; output_bytes/rows = Σ inputs).
  - (b) **cuDF hash_partition** 1→N = `GpuRepartitionExec(hash)` node.
  - (c) **scan-batch read** = the scan node emitting the row-group→batch→partition
    MAP (`read_parquet(set_row_groups)` GPU / `with_row_groups` CPU).
  NOT a node: the semi-join `null_equality` flag — it's a PARAMETER of the existing
  GpuHashJoin (no allocation/transform), so no new node (dmitry confirmed).
- **C2 — ONE merged GPU test.** Collapse `test_gpu_node` + `test_gpu_executor` into
  a single per-query GPU test: one GPU execution asserts BOTH (a) per-node exact
  rows + rows/schema cost vs the `.cpu.txt` golden AND (b) the final RESULT. Keep
  the CI gate (fail on per-node OR final-result divergence). pipeline.yml → one bin.

## Settled design (recap, unchanged)
batch = optimizer-chosen group of WHOLE row groups per table (boundaries = RG
boundaries, deterministic, floor 1 RG). Explicit RG→batch→partition MAP computed
from #12 survivors + budget, CANONIZED in the tp8 `.cpu.txt` golden; both backends
replay it. RoundRobin(8) (all input_partitions=1 in our plans) DISSOLVES into the
scan read (rg-batch i → partition i mod N) — no separate redistribution op.
Hash repartition = cuDF hash_partition, M→N CONCAT-FIRST. Merge-correct Final agg:
SUM/COUNT/MIN/MAX trivial, AVG=SUM+COUNT (#25), STDDEV/VAR via additive M2 moments;
REMOVE has_singleton_final / has_avg_final / positional-singleton guards. Cost via
logical_size_from_schema. tp8 goldens exclude non-deterministic LIMIT queries,
approx-float q14/q39 (#11 policy). Verify INCREMENTALLY in buckets, never all at once.

## Ordered increments (each = a GPU==golden proof point)

### Inc 0 — Merge the two GPU tests (C2), prove on existing tp1
- Collapse into one per-query GPU test (common/mod.rs macro): one GPU run →
  assert per-node rows+cost vs `.cpu.txt` AND the final result.
- pipeline.yml: stage/run ONE bin; keep the gate (rc + exit $rc).
- **Proof:** tp1 merged test 107/107, both checks asserted; CI still gates.
- Independent of tp8 → do FIRST; halves GPU run cost for every later increment.

### Inc 1 — Scan-batch read as a node + RG→batch→partition MAP (C1c), tp8, ONE query (q1 or q6)
- Add the optimizer step computing the RG→batch→partition MAP (from #12 survivors
  + budget grouping of whole RGs). Surface it on the scan node (flatbuffer field).
- GPU scan: `read_parquet(set_row_groups(batch RGs))` per partition. CPU emulation:
  `ParquetRecordBatchReaderBuilder::with_row_groups(same)` → 1 RecordBatch/partition.
- RoundRobin(8) dissolves into the scan read (rg-batch i → partition i mod N).
- Canonize the MAP in the tp8 `.cpu.txt` golden.
- **Proof:** tp8 node-by-node match on q1/q6 from scan up to (not incl) the first
  repartition, GPU==golden. REPORT before scaling.

### Inc 2 — Hash repartition: CONCAT node + hash_partition node (C1a+C1b), tp8, q1/q6
- Lower multi-input `GpuRepartition(Hash, input_partitions=M)` into:
  `GpuCoalescePartitions(M→1)` [CONCAT, BUFFERING] + `GpuRepartitionExec(hash, 1→N)`
  [cuDF hash_partition slice]. Both explicit nodes, visible in golden, CPU-emulated.
- resident.rs: classify THIS concat as BUFFERING (full output resident), not
  streaming(0) like normal Coalesce* — so the tp8 OOM peak sees the shuffle.
- **Proof:** tp8 node-by-node match through the repartition on q1/q6, GPU==golden.
- **RISK/GATE:** cuDF murmur3 ≠ DataFusion ahash → different partition NUMBER
  (harmless, count-preserving) BUT a downstream partial-agg grouped by a non-partition
  high-card key can get hash-distribution-dependent cardinality → mismatch. Low-card
  TPC-H keys match; ESCALATE if a high-card query diverges (fix: shared hash / extend
  map / skip-verify that node — coordinator's call).

### Inc 3 — Merge-correct Final agg: SUM/MIN/MAX/COUNT, tp8, q1/q6
- Two-phase agg: partial emits real STATE; final merges (SUM=Σ, COUNT=Σ, MIN/MAX trivial).
- **Proof:** q1/q6 tp8 final-agg node + final result == golden.

### Inc 4 — AVG = SUM + COUNT (#25)
- Decompose AVG in the final merge; remove `has_avg_final` guard.
- **Proof:** an AVG-at-tp8 query (q1 has avg) node + result == golden.

### Inc 5 — STDDEV/VAR additive M2 + remove remaining guards
- Additive moments (n, mean, M2; M2_AB = M2_A+M2_B+δ²·nA·nB/nAB). Remove
  has_singleton_final + the Final-stage positional-singleton shortcut.
- **Proof:** a STDDEV/VAR-at-tp8 query node + result == golden.

### Inc 6 — Full tp8 TPC-H bucket
- Generate + verify all tpch tp8 goldens (exclude non-det LIMIT; approx-float q14/q39).
- **Proof:** tpch tp8 merged test green (per-node + result).

### Inc 7 — Full tp8 TPC-DS buckets (incremental)
- Same for tpcds, in buckets (large surface — never all at once).
- **Proof:** tpcds tp8 merged test green, bucket by bucket.

## Cross-cutting touch points
- flatbuffer (gpu_plan.fbs): scan node RG→batch→partition map; GpuRepartition hash
  mode + partition count; ensure GpuCoalescePartitions can carry the BUFFERING concat
  role. (Schema additions = codegen auto-regen Rust+C++.)
- plan_serializer.rs: serialize/deserialize the map + repartition params.
- cpu_executor.rs / node_executor.rs: CPU emulation of the 3 op-nodes (controlled
  with_row_groups scan, concat M→1, hash_partition 1→N) so the oracle replays the map.
- plan_executor.cpp: real cuDF hash_partition; two-phase merge-correct final agg;
  remove the 3 guards.
- resident.rs: BUFFERING classification for the shuffle concat.
- common/mod.rs + pipeline.yml: merged test + one-bin CI gate.

## Build/run discipline (unchanged)
Build LOCAL; ship binaries/data/goldens to shad-gpu (re-patch after rsync); GPU
runs on shad-gpu; canon golden verification on the remote with shipped goldens.
Coordinator commits per milestone.

## Coordinator refinements (greenlight 364) — baked in
- **R1 (murmur3, DECIDED):** CPU-emulate `GpuRepartitionExec(hash)` with cuDF
  murmur3 (same seed, null-handling, modulo) in the node-executor — NOT DataFusion
  ahash. Entailed by C1 (the hash_partition node is an explicit alloc-op the CPU
  must faithfully emulate → same hash → per-node counts match GPU by construction;
  final agg result stays oracle-correct since partition layout is merge-invariant).
  totals/multiset-relaxation is a FALLBACK ONLY if murmur3-in-Rust is infeasible →
  **CONFIRM murmur3-feasibility at Inc2 start; ESCALATE rather than silently relax.**
- **R2 (guard blast radius):** re-run the FULL tp1 107 corpus (node+result) after
  EACH of Inc4 and Inc5 — the removed guards fire for every agg/singleton-final
  query, not just q1/q6. Cheap post-Inc0.
- **R3 (IR round-trip):** every new gpu_plan.fbs field (scan RG→batch→partition map;
  repartition hash params) gets a Tier-1a round-trip test with NON-DEFAULT/POPULATED
  values, in the SAME increment that adds the field (no deferring to the GPU harness).
- **R4 (merged test = UNION):** the single GPU bin covers the UNION of
  test_gpu_node + test_gpu_executor query sets, per-query assertion config — exact
  per-node cost ALWAYS; result = exact / approx(q14,q39) / skip(non-det LIMIT). No
  query loses its node-assertion OR its result-assertion. Preserve --test-threads=1,
  rc/exit-$rc propagation, fail-closed-on-missing-golden.
- **resident.rs (C1a impl note):** concat currently buckets as STREAMING; add a
  BUFFERING arm for GpuCoalescePartitions(concat) mirroring the SortExec|AggregateExec arm.
