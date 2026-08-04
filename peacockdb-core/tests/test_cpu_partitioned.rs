//! Real 8-way partitioned CPU execution (`cpu_partitioned_result_test!`).
//!
//! The `CpuNodeExecutor` maintains N partitions across nodes (partial-agg =
//! Σ-over-partitions, CoalescePartitions concat N→1) — the same shape the real 8-way
//! GPU produces, which is why `test_gpu_partitioned.rs` verifies against the
//! `partitioned-tp8-standard.cpu.txt` goldens these tests write. Mode comes from the
//! macro name, not from the device label: the SAME plan at tp8-mini runs full-table
//! in test_cpu_full_table.rs.
#[macro_use]
mod common;

// ── tp8-standard: real 8-way partitioning ───────────────────────────────────
// The scan's RG→partition map drives the per-partition execution, so per-node stats
// are Σ-over-8 partitions. Cross-checks partitioned-CPU vs DataFusion on the CPU
// tier; the GPU verifies against this device's .cpu.txt.
// q6 = scan→filter→partial-agg→CoalescePartitions→final-agg (no hash).
cpu_partitioned_result_test!(tpch, 1, q6, tp8_standard, result_golden);
// shuffle_additive: scan→partial-agg(8)→GpuCoalescePartitions(8→1)→GpuRepartition
// Hash(1→8, Spark-murmur3)→final-agg(8). The hash-shuffle proof — per-partition
// out_rows on the repartition node are murmur3-fidelity numbers the GPU must match.
cpu_partitioned_result_test!(tpch, 1, shuffle_additive, tp8_standard, result_golden);
// shuffle_additive_avg: real 8-way with an AVG (state = sum,count merged Σsum/Σcount
// per hash bucket). Minimal single-avg carrier.
cpu_partitioned_result_test!(tpch, 1, shuffle_additive_avg, tp8_standard, result_golden);
// q1: the canonical AVG carrier (3 avgs + 4 sums + count, GROUP BY rf,ls) — real
// 8-way at tp8-standard. tp8-mini q1 stays full-table.
cpu_partitioned_result_test!(tpch, 1, q1, tp8_standard, result_golden);
// join-int (#96): minimal 2-table single-INT-key inner join (orders⋈customer on custkey)
// + GROUP BY count — the smallest real-8-way carrier for per-partition INNER JOIN
// execution, isolating it from q17's 15-join complexity. no_result_golden until the GPU
// join fix lands (then a gpu_partitioned_test! consumes its .result.txt).
cpu_partitioned_result_test!(tpch, 1, join_int, tp8_standard, no_result_golden);
// q17 (tpcds): the CPU-side real-query STDDEV proof — per measure count(1)+avg(2)+
// stddev(3) state, ×3 measures — exercising the mixed count/avg/stddev Final
// width-detect at real 8-way. result_golden (#96): the CPU oracle runs q17's 7
// Partitioned joins per-partition (8-way), and the tp8 gpu_partitioned_test!
// (golden_approx_std) consumes this owned .result.txt.
cpu_partitioned_result_approx_test!(tpcds, 1, q17, tp8_standard, result_golden);
// semi/anti/left joins (#97-a): per-partition NON-inner Partitioned joins at real
// 8-way (semi=RightSemi, anti=RightAnti, left=Left, each 8→8). Each asserts the
// per-partition result against VANILLA DataFusion, so a wrong per-partition anti
// (NOT-IN global-null edge) fails LOUD here. no_result_golden: the GPU tp8 tests use
// oracle mode (>256KB result), so no .result.txt is consumed.
cpu_partitioned_result_test!(tpch, 1, semi_join, tp8_standard, no_result_golden);
cpu_partitioned_result_test!(tpch, 1, anti_join, tp8_standard, no_result_golden);
cpu_partitioned_result_test!(tpch, 1, left_join, tp8_standard, no_result_golden);
// TPC-H join queries at real 8-way: all Partitioned Inner joins, Int32/Int64 keys
// (q5 composite), all-sum mergeable aggs, scan-map present, zero CollectLeft/
// decimal/distinct — partitioned-executable. result_golden: each owns the small
// .result.txt the tp8 gpu_partitioned_test!(golden_exact) consumes (result is
// partition-independent). Legacy tp8-mini full-table carriers stay in
// test_cpu_full_table.rs.
// q3 (#99): ORDER BY revenue ... LIMIT 10 — SortPreservingMerge k-way-merges the 8
// sorted partitions + applies fetch (not concat), so the global top-10 == DataFusion.
cpu_partitioned_result_test!(tpch, 1, q3, tp8_standard, result_golden);
cpu_partitioned_result_test!(tpch, 1, q5, tp8_standard, result_golden);
// q7/q8/q9 need the GPU repartition key-type extensions: (1) dict-encoded string
// keys decode to STRING; (2) GROUP-BY o_year/l_year reaches the kernel as cuDF
// INT16 (cudf::extract_year → INT16, not DataFusion's Int32), widened to INT32;
// (3) DATE keys (TIMESTAMP_DAYS) bit-cast to INT32.
cpu_partitioned_result_test!(tpch, 1, q7, tp8_standard, result_golden);
cpu_partitioned_result_test!(tpch, 1, q8, tp8_standard, result_golden);
cpu_partitioned_result_test!(tpch, 1, q9, tp8_standard, result_golden);
// q12/q19: Partitioned Inner, int keys, mergeable sum, no LIMIT/decimal/distinct.
cpu_partitioned_result_test!(tpch, 1, q12, tp8_standard, result_golden);
cpu_partitioned_result_test!(tpch, 1, q19, tp8_standard, result_golden);
// q13: grouped count over a Partitioned LEFT-outer join, ORDER BY without LIMIT.
// Safe per #100: the LEFT-outer is correct — only global count(*) mis-merges, and
// q13's counts are GROUPED. No LIMIT (no #99 exposure).
cpu_partitioned_result_test!(tpch, 1, q13, tp8_standard, result_golden);

// ── registry verification ───────────────────────────────────────────────────
/// Owns the `partitioned_cpu` column outright — every `cpu_partitioned_*`
/// invocation lives in this binary, so both directions of the check are complete
/// here.
#[test]
fn registry_matches_csv_partitioned_column() {
    common::registry::assert_registry_matches_csv(&["partitioned_cpu"], &[]);
}
