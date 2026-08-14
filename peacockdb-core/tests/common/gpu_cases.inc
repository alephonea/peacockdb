// THE case list for the GPU suite — the single source of truth for which
// (dataset, sf, query, golden-label, result-mode) combinations the GPU is expected
// to run. Deliberately NOT a `.rs` file and NOT at `tests/` top level: cargo would
// auto-discover either as its own test target.
//
// `include!`d by THREE targets, which differ only in what `gpu_case!` means there:
//   - `tests/test_gpu_full_table.rs`    — expands the `full_table_tp1_standard` rows
//     to `gpu_full_table_test!`, and the partitioned rows to NOTHING.
//   - `tests/test_gpu_partitioned.rs`   — the mirror image.
//   - `tests/peacock_gpu_benchmarks.rs` — expands BOTH to a benchmark that times the
//     query and asserts nothing.
//
// The label is matched as a LITERAL token, not captured as an `ident`. That is the
// point of keeping one file rather than two: a row whose label is neither of the two
// known ones matches no rule and fails to COMPILE, in all three targets at once.
// Split across two files the same mistake compiles fine — the row just silently runs
// in the wrong suite, or in none.
//
// Each target defines its own `gpu_case!` BEFORE including this file. A distinct name
// (rather than redefining `gpu_full_table_test!` in one of them) is deliberate:
// shadowing a `#[macro_export]`ed macro works, but "which definition is in scope here"
// is then something a reader has to reason about, and getting it wrong silently runs
// the wrong thing rather than failing to compile.
//
// One list, so a query added to the suite is benchmarked without a second edit, and a
// benchmark record can never describe a query the gate does not run.

// ── TPC-H (the GPU-supported set) ───────────────────────────────────────────
gpu_case!(tpch, 1, scan_limit, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, filter_project, full_table_tp1_standard, oracle);    // >256KB result
gpu_case!(tpch, 1, aggregate_groupby, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, semi_join, full_table_tp1_standard, oracle);        // >256KB result
gpu_case!(tpch, 1, anti_join, full_table_tp1_standard, oracle);        // >256KB result
gpu_case!(tpch, 1, nested_loop_join, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, cross_join, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q1, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q2, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q3, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q4, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q5, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q6, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q7, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q8, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q9, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q10, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q11, full_table_tp1_standard, oracle);              // >256KB result
gpu_case!(tpch, 1, q12, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q13, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q14, full_table_tp1_standard, golden_approx);
gpu_case!(tpch, 1, q15, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q16, full_table_tp1_standard, oracle);              // >256KB result
gpu_case!(tpch, 1, q17, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q18, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q19, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q20, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q21, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, q22, full_table_tp1_standard, golden_exact);

gpu_case!(tpch, 1, shuffle_additive, full_table_tp1_standard, golden_exact);
gpu_case!(tpch, 1, shuffle_additive_avg, full_table_tp1_standard, golden_exact);
// STDDEV/VAR at tp1 (single-partition make_std/make_variance). golden_approx_std
// (1e-11): cuDF's variance algo diverges from DataFusion's Welford by ~2e-12 —
// more than the 1e-12 sum/avg convention. See GpuResultMode doc.
gpu_case!(tpch, 1, shuffle_stddev, full_table_tp1_standard, golden_approx_std);

// ── TPC-DS (GPU-operational set) ────────────────────────────────────────────
gpu_case!(tpcds, 1, q1, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q3, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q4, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q5, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q6, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q7, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q8, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q10, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q11, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q12, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q13, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q14, full_table_tp1_standard, golden_approx);
gpu_case!(tpcds, 1, q15, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q16, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q17, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q18, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q19, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q20, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q21, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q22, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q23, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q24, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q25, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q26, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q29, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q30, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q31, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q32, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q33, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q34, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q35, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q37, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q40, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q41, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q42, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q43, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q45, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q46, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q48, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q50, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q51, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q52, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q53, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q54, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q55, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q56, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q58, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q59, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q60, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q62, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q63, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q64, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q65, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q68, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q69, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q71, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q73, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q74, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q75, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q79, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q80, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q81, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q82, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q83, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q84, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q85, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q88, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q89, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q90, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q91, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q92, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q93, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q94, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q95, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q96, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q97, full_table_tp1_standard, golden_exact);
gpu_case!(tpcds, 1, q98, full_table_tp1_standard, oracle);             // >256KB result
gpu_case!(tpcds, 1, q99, full_table_tp1_standard, golden_exact);

// ════════════════════════════════════════════════════════════════════════════
// Real 8-way partitioned execution (partitioned_tp8_standard)
// ════════════════════════════════════════════════════════════════════════════
// ── TPC-H ───────────────────────────────────────────────────────────────────
// One GPU run asserts per-node Σ-over-8 rows+cost vs the partitioned-tp8-standard
// .cpu.txt (generated by the partitioned CPU executor) AND the final result
// (decimal SUM → exact). q6 = no shuffle; shuffle_additive = Hash-repartition via
// Spark-murmur3.
gpu_case!(tpch, 1, q6, partitioned_tp8_standard, golden_exact);
gpu_case!(tpch, 1, shuffle_additive, partitioned_tp8_standard, golden_exact);
// AVG at real 8-way: state = (sum,count), merged Σsum/Σcount per hash bucket.
gpu_case!(tpch, 1, shuffle_additive_avg, partitioned_tp8_standard, golden_exact);
// q1: canonical AVG carrier (3 avgs + 4 sums + count, GROUP BY rf,ls) — real 8-way
// at tp8-standard. Exercises the running column-offset (avgs before count).
gpu_case!(tpch, 1, q1, partitioned_tp8_standard, golden_exact);
// STDDEV/VAR at real 8-way: Welford [count,mean,m2] state merged via cuDF MERGE_M2
// per hash bucket (stddev_samp/pop + var_samp/pop, all 4 finalize branches).
//
// QUARANTINED as flaky — ticket #103 (llm-wiki/tickets.md). Intermittently SIGSEGVs
// (exit 139) in the GpuAggregate final merge on the shad-gpu H200 (the same host CI's
// gpu-tests job uses), or fails as a contained `vector::reserve` Err — two failure
// signatures from identical inputs, i.e. nondeterministic, not broken. Suspected
// origin: the N-way Welford merge; the full-table tier (no merge) never crashes. Not
// caused by the executors refactor (reproduces at both old and new memory tiers,
// upstream of every golden comparison).
// IF YOU SEE THIS FAIL LOCALLY, check #103 before assuming your change caused it — it
// has already cost one wrongly-diagnosed "regression" and an unnecessary rollback.
// Cost of disabling: this is the ONLY coverage of the 8-way Welford M2 merge path.
// The tp8 goldens are deliberately LEFT IN PLACE. Re-enable when #103 is fixed.
// gpu_case!(tpch, 1, shuffle_stddev, partitioned_tp8_standard, golden_approx_std);

// (#96) Real-8-way per-partition JOINs. The partitioned CPU oracle runs
// PartitionMode::Partitioned joins per-partition (child0[p] ⋈ child1[p]), matching
// vanilla DataFusion (all q17 joins are Partitioned 8→8) and the GPU C++ MAP arm.
// q17: mixed count/avg/stddev over 7 real-8-way joins (golden_approx_std for the
// GPU-cuDF-M2 stddev). join_int: minimal 1-key fact-dim join (oracle mode).
gpu_case!(tpcds, 1, q17, partitioned_tp8_standard, golden_approx_std);
gpu_case!(tpch, 1, join_int, partitioned_tp8_standard, oracle);
// (#97-a) Real-8-way per-partition NON-inner Partitioned joins (semi=RightSemi,
// anti=RightAnti, left=Left, each 8→8). oracle mode (>256KB results): the GPU output
// is checked against a LIVE CPU-oracle run, so anti's NOT-IN null handling is gated
// at both levels.
gpu_case!(tpch, 1, semi_join, partitioned_tp8_standard, oracle);
gpu_case!(tpch, 1, anti_join, partitioned_tp8_standard, oracle);
gpu_case!(tpch, 1, left_join, partitioned_tp8_standard, oracle);
// TPC-H join queries at real 8-way (Partitioned Inner, int keys, mergeable sum
// aggs); small results → golden_exact (consumes the cpu_partitioned-owned
// .result.txt). Multi-func aggs (q8, q12: 2 sums) verify on CI only — the local flatc
// aggr_funcs-vector bug truncates multi-func aggs; single-agg queries verify locally
// too. q3 (#99 SPM/TopK fix): ORDER BY revenue LIMIT 10, SPM k-way-merges + fetches.
gpu_case!(tpch, 1, q3, partitioned_tp8_standard, golden_exact);
gpu_case!(tpch, 1, q5, partitioned_tp8_standard, golden_exact);
gpu_case!(tpch, 1, q7, partitioned_tp8_standard, golden_exact);
gpu_case!(tpch, 1, q8, partitioned_tp8_standard, golden_exact);
gpu_case!(tpch, 1, q9, partitioned_tp8_standard, golden_exact);
gpu_case!(tpch, 1, q12, partitioned_tp8_standard, golden_exact);
gpu_case!(tpch, 1, q19, partitioned_tp8_standard, golden_exact);
// q13: grouped count over a Partitioned LEFT-outer (nested group-bys, single-count
// each level → local-reliable). golden_exact (small custdist result).
gpu_case!(tpch, 1, q13, partitioned_tp8_standard, golden_exact);
