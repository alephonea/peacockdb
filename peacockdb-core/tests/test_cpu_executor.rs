//! Parameterized CPU-executor result + cost tests for TPC-H and TPC-DS. Each
//! `cpu_result_test!(dataset, sf, query, device)` runs <dataset>-queries/<query>.sql
//! through plain DataFusion (ground truth) and the CPU executor, asserting the
//! results match and the cost tree matches testdata/goldens/<dataset>.sf<sf>/<query>.<device>.cpu.txt.
//! Helpers + the macro live in common/mod.rs; bespoke tests in test_cpu_executor_misc.rs.
#[macro_use]
mod common;

// ── TPC-H ─────────────────────────────────────────────────────────────────
cpu_result_test!(tpch, 1, hash_join, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, left_join, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, mixed_join, tp8_mem2gib, false);
// LIMIT without a total order (no/under-specified ORDER BY → ties at the LIMIT
// boundary) has no partition-invariant row set: at tp>1 both the result rows and
// the per-node output_bytes vary run-to-run. These queries are therefore canonized
// at tp1 (deterministic single-stream) for the result-assert AND the cost golden.
// Their .plan.txt (plan-shape) goldens stay tp8 — only the cpu device moves.
cpu_result_test!(tpch, 1, scan_limit, tp1_mem2gib, false);
cpu_result_test!(tpch, 1, filter_project, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, aggregate_groupby, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, semi_join, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, anti_join, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, nested_loop_join, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, cross_join, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q1, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q2, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q3, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q4, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q5, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q6, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q7, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q8, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q9, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q10, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q11, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q12, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q13, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q14, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q15, tp8_mem2gib, false);  // view inlined as a CTE (see q15.sql)
cpu_result_test!(tpch, 1, q16, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q17, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q18, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q19, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q20, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q21, tp8_mem2gib, false);
cpu_result_test!(tpch, 1, q22, tp8_mem2gib, false);
// shuffle_additive (GROUP BY rf,ls; count/sum only) — the Inc2 hash-shuffle proof
// query. At tp8-mem2gib it stays on #11 (single-partition-coalesced, partitions=1);
// the real 8-way #13 golden is the tp8-mem120gib entry below.
cpu_result_test!(tpch, 1, shuffle_additive, tp8_mem2gib, false);
// shuffle_additive_avg (GROUP BY rf,ls; count, avg, sum — avg BEFORE sum) — the Inc4
// AVG proof query. tp8-mem2gib stays on #11; the real 8-way #13 golden is below.
cpu_result_test!(tpch, 1, shuffle_additive_avg, tp8_mem2gib, false);
// shuffle_stddev (GROUP BY rf,ls; stddev_samp/pop + var_samp/pop) — the Inc5 STDDEV/VAR
// proof. tp8-mem2gib stays on #11 (SinglePartition make_std/make_variance singleton);
// the real 8-way #13 Welford-M2-merge golden is at tp8-mem120gib. Approx: the M2
// summation reassociates across partitions (~1 ULP), so float compare is tolerant.
cpu_result_approx_test!(tpch, 1, shuffle_stddev, tp8_mem2gib, false);

// ── H200/tp8 (Phase 2 Inc1): real 8-way partitioning ────────────────────────
// The scan's RG→partition map drives this through the #13 CpuNodeExecutor, so
// per-node stats are Σ-over-8 partitions (partial-agg = 8 rows, CoalescePartitions
// concat 8→1) — the SAME the real 8-way GPU produces. Cross-checks #13-CPU vs
// DataFusion on the CPU tier; the GPU verifies against this device's .cpu.txt.
// q6 = scan→filter→partial-agg→CoalescePartitions→final-agg (no hash; Inc1 proof).
// EXPLICIT #13 opt-in: the SAME plan at tp8-mem2gib stays on #11 (above); only this
// H200/tp8 device runs the real N-partition #13 executor (Σ-over-8 golden).
cpu_node13_result_test!(tpch, 1, q6, tp8_mem120gib, true);
// shuffle_additive: scan→partial-agg(8)→GpuCoalescePartitions(8→1)→GpuRepartition
// Hash(1→8, Spark-murmur3)→final-agg(8). The Inc2 hash-shuffle proof — per-partition
// out_rows on the repartition node are murmur3-fidelity numbers the GPU must match.
cpu_node13_result_test!(tpch, 1, shuffle_additive, tp8_mem120gib, true);
// shuffle_additive_avg: real 8-way with an AVG (state = sum,count merged Σsum/Σcount
// per hash bucket). Inc4 AVG proof. Minimal single-avg carrier (de-risks before q1).
cpu_node13_result_test!(tpch, 1, shuffle_additive_avg, tp8_mem120gib, true);
// q1: the canonical AVG carrier (3 avgs + 4 sums + count, GROUP BY rf,ls) — now real
// 8-way #13 at tp8-mem120gib (was #11-only until Inc4). tp8-mem2gib q1 stays #11.
cpu_node13_result_test!(tpch, 1, q1, tp8_mem120gib, true);
// join-int (#96): minimal 2-table single-INT-key inner join (orders⋈customer on custkey)
// + GROUP BY count — the smallest real-8-way carrier for per-partition INNER JOIN
// execution, isolating it from q17's 15-join complexity. gen=false until the GPU join
// fix lands (then a gpu_test! consumes its .result.txt).
cpu_node13_result_test!(tpch, 1, join_int, tp8_mem120gib, false);
// q17 (tpcds): the CPU-side real-query STDDEV proof — per measure count(1)+avg(2)+
// stddev(3) state, ×3 measures — exercising the #13 mixed count/avg/stddev Final
// width-detect at real 8-way (comet hashes the composite int join keys). gen=true
// (#96): the CPU oracle now runs q17's 7 Partitioned joins per-partition (8-way),
// and the tp8 gpu_test! (golden_approx_std) consumes this owned .result.txt.
cpu_node13_result_approx_test!(tpcds, 1, q17, tp8_mem120gib, true);
// semi/anti/left joins (#97-a): per-partition NON-inner Partitioned joins at real 8-way.
// DataFusion runs all three Partitioned 8→8 (semi-join=RightSemi, anti-join=RightAnti,
// left-join=Left); the #96 MAP arm admits every join type, so this is the conformance
// proof — each asserts the #13 per-partition result against VANILLA DataFusion, so a
// wrong per-partition anti (NOT-IN global-null edge) fails LOUD here. gen=false: the
// GPU tp8 tests use oracle mode (>256KB result), so no .result.txt is consumed. The
// legacy tp8-mem2gib cpu_result_test carriers above stay (exercise the #11 executor).
cpu_node13_result_test!(tpch, 1, semi_join, tp8_mem120gib, false);
cpu_node13_result_test!(tpch, 1, anti_join, tp8_mem120gib, false);
cpu_node13_result_test!(tpch, 1, left_join, tp8_mem120gib, false);
// Real TPC-H join-query flip batch 1 (post-#96/#97-a): all Partitioned Inner joins,
// Int32/Int64 keys (q5 composite), all-sum mergeable aggs, scan-map present, zero
// CollectLeft/decimal/distinct — node13-executable at real 8-way. gen=true: each owns
// the small .result.txt the tp8 gpu_test!(golden_exact) consumes (result is
// partition-independent). Legacy tp8-mem2gib #11 carriers above stay.
// q3 (#99): ORDER BY revenue ... LIMIT 10 — the SPM/TopK gate is now FIXED (the #13
// SortPreservingMerge k-way-merges the 8 sorted partitions + applies fetch instead of
// concatenating), so the global top-10 == DataFusion. Re-flipped.
cpu_node13_result_test!(tpch, 1, q3, tp8_mem120gib, true);
cpu_node13_result_test!(tpch, 1, q5, tp8_mem120gib, true);
// q7/q8/q9 FLIPPED via the GPU repartition key-type extensions: (1) dict-encoded string
// keys decode to STRING (DF45 dict-encoding); (2) their GROUP-BY o_year/l_year reaches
// the kernel as cuDF INT16 (cudf::extract_year → INT16, NOT the DataFusion Int32 the
// audit saw — the cuDF-vs-DF type gap), now widened to INT32; (3) q3-class DATE keys
// (TIMESTAMP_DAYS) bit-cast to INT32. All hash-only (scattered output unchanged → no
// golden regen). Legacy tp8-mem2gib #11 lines kept.
cpu_node13_result_test!(tpch, 1, q7, tp8_mem120gib, true);
cpu_node13_result_test!(tpch, 1, q8, tp8_mem120gib, true);
cpu_node13_result_test!(tpch, 1, q9, tp8_mem120gib, true);
// Flip batch 2: q12/q19 (Partitioned Inner, int keys, mergeable sum, no
// LIMIT/decimal/distinct → no gate). Legacy tp8-mem2gib #11 lines kept.
cpu_node13_result_test!(tpch, 1, q12, tp8_mem120gib, true);
cpu_node13_result_test!(tpch, 1, q19, tp8_mem120gib, true);
// q13 (bucket-2 addendum): grouped count over a Partitioned LEFT-outer join, ORDER BY
// without LIMIT. Un-gated once #100 clarified the LEFT-outer is correct (only global
// count(*) mis-merged; q13's counts are GROUPED). No LIMIT (no #99), no global count
// (no #100). Legacy tp8-mem2gib #11 line kept.
cpu_node13_result_test!(tpch, 1, q13, tp8_mem120gib, true);

// ── TPC-DS ────────────────────────────────────────────────────────────────
cpu_result_test!(tpcds, 1, q1, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q2, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q3, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q4, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q5, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q6, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q7, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q8, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q9, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q10, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q11, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q12, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q13, tp8_mem2gib, false);
// q14/q39: float (avg/stddev) summation reassociates across partitions at tp8 →
// ~1 ULP result drift vs the single-partition DataFusion oracle. Tolerant result
// compare (rel≤1e-12); output_bytes golden stays exact (float value doesn't change
// byte width), so these remain tp8.
cpu_result_approx_test!(tpcds, 1, q14, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q15, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q16, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q17, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q18, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q19, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q20, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q21, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q22, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q23, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q24, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q25, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q26, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q28, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q29, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q30, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q31, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q32, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q33, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q34, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q35, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q36, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q37, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q38, tp8_mem2gib, false);
cpu_result_approx_test!(tpcds, 1, q39, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q40, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q41, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q42, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q43, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q44, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q45, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q46, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q47, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q48, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q49, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q50, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q51, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q52, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q53, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q54, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q55, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q56, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q57, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q58, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q59, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q60, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q61, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q62, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q63, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q64, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q65, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q66, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q67, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q68, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q69, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q71, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q73, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q74, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q75, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q76, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q77, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q78, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q79, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q80, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q81, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q82, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q83, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q84, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q85, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q87, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q88, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q89, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q90, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q91, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q92, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q93, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q94, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q95, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q96, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q97, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q98, tp8_mem2gib, false);
cpu_result_test!(tpcds, 1, q99, tp8_mem2gib, false);

// ── TPC-DS disabled — blocked on DataFusion 46+ upgrade (issue #23) ──────────
// These four don't physical-plan under DataFusion 45, so they're also disabled
// in the plan and gpu suites. Re-enable once the DataFusion 46+ upgrade (#23,
// which names these exact queries) lands.
//
// q27: SanityCheckPlan rejects the SortPreservingMergeExec ordering for ROLLUP.
// cpu_result_test!(tpcds, 1, q27, tp8_mem2gib, false);
// q70: GROUPING() aggregate has no physical-plan support.
// cpu_result_test!(tpcds, 1, q70, tp8_mem2gib, false);
// q72: Date32 + Int64 type-coercion not supported.
// cpu_result_test!(tpcds, 1, q72, tp8_mem2gib, false);
// q86: GROUPING() aggregate has no physical-plan support.
// cpu_result_test!(tpcds, 1, q86, tp8_mem2gib, false);
