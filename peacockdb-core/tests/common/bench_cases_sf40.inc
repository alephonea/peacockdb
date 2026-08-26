// The sf40 case list — measurement only, and the only case list that is not shared.
//
// `gpu_cases.inc` is included by three targets and every row there is a correctness
// test in two of them. sf40 cannot be a row there: there are no CPU goldens at that
// scale and there will not be, because producing them means running the query on the
// CPU over 42 GB of parquet. A row moved from this file into that one would silently
// enrol sf40 in two correctness suites that would then fail for want of files nobody
// intends to generate.
//
// Hence a second file, and hence `bench_only_case!` rather than `gpu_case!`: the macro
// is defined in `peacock_gpu_benchmarks.rs` and nowhere else, so including this file
// anywhere else is a compile error naming an undefined macro rather than a suite that
// quietly grew. Same reasoning as the literal label token in `gpu_cases.inc` — the
// mistake has to fail to build, not to run.
//
// No result-mode argument, unlike `gpu_case!`. Nothing here compares a result against
// anything, and the absence of the argument is what says so.
//
// The dataset is read in place from outside the repo; `build-test-shadgpu.sh` links it
// to <testdata>/tpch.sf40 before the run, which is the only name the rust side resolves.

// ── TPC-H sf40, full-table, 70 GiB ──────────────────────────────────────────
// The 32 tpch queries the gate runs full-table at sf1, at the scale the cost model is
// meant to extrapolate to. tp1 because a region is (node, output partition), and a
// partitioned run is not the same plan split eight ways: it carries repartition and
// coalesce-partitions nodes that a tp1 plan never produces. Measuring it here would put
// two different questions in one file instead of answering this one at scale.
bench_only_case!(tpch, 40, scan_limit, full_table_tp1_full);
bench_only_case!(tpch, 40, filter_project, full_table_tp1_full);
bench_only_case!(tpch, 40, aggregate_groupby, full_table_tp1_full);
bench_only_case!(tpch, 40, semi_join, full_table_tp1_full);
bench_only_case!(tpch, 40, anti_join, full_table_tp1_full);
// region ⋈ nation only, and both tables are fixed-size in TPC-H — the sf1 and sf40
// files are byte-identical. Kept because a node type measured at one size is still a
// node type measured, and these two are the only place GpuCrossJoinExec and
// GpuNestedLoopJoinExec appear at all. They carry no scale information: read their rows
// as one more sample at sf1's size, not as a second point on the byte axis.
bench_only_case!(tpch, 40, nested_loop_join, full_table_tp1_full);
bench_only_case!(tpch, 40, cross_join, full_table_tp1_full);
bench_only_case!(tpch, 40, q1, full_table_tp1_full);
// q2, q9, q11, q16, q20 read partsupp — 26.9 GB, of which the psimage/ptext blobs are
// most and which no tpch query names. They are affordable only if column pruning keeps
// those columns unread; that it does is what the whole 32-case list finishing in about
// eight minutes says.
bench_only_case!(tpch, 40, q2, full_table_tp1_full);
bench_only_case!(tpch, 40, q3, full_table_tp1_full);
bench_only_case!(tpch, 40, q4, full_table_tp1_full);
bench_only_case!(tpch, 40, q5, full_table_tp1_full);
bench_only_case!(tpch, 40, q6, full_table_tp1_full);
bench_only_case!(tpch, 40, q7, full_table_tp1_full);
bench_only_case!(tpch, 40, q8, full_table_tp1_full);
bench_only_case!(tpch, 40, q9, full_table_tp1_full);
bench_only_case!(tpch, 40, q10, full_table_tp1_full);
bench_only_case!(tpch, 40, q11, full_table_tp1_full);
bench_only_case!(tpch, 40, q12, full_table_tp1_full);
bench_only_case!(tpch, 40, q13, full_table_tp1_full);
bench_only_case!(tpch, 40, q14, full_table_tp1_full);
bench_only_case!(tpch, 40, q15, full_table_tp1_full);
bench_only_case!(tpch, 40, q16, full_table_tp1_full);
bench_only_case!(tpch, 40, q17, full_table_tp1_full);
bench_only_case!(tpch, 40, q18, full_table_tp1_full);
bench_only_case!(tpch, 40, q19, full_table_tp1_full);
bench_only_case!(tpch, 40, q20, full_table_tp1_full);
bench_only_case!(tpch, 40, q21, full_table_tp1_full);
bench_only_case!(tpch, 40, q22, full_table_tp1_full);
bench_only_case!(tpch, 40, shuffle_additive, full_table_tp1_full);
bench_only_case!(tpch, 40, shuffle_additive_avg, full_table_tp1_full);
bench_only_case!(tpch, 40, shuffle_stddev, full_table_tp1_full);
