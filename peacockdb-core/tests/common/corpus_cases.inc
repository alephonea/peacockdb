// The corpus, one line per query — read and diffed as a line rather than as up to ten
// macro invocations that can disagree with each other. Included by both corpus binaries,
// which define `corpus_query!` differently: the cpu one expands `cpu_modes`, the device one
// `gpu_modes`, so a query enabled at four modes on the cpu and one on a device is still one
// declaration.
//
//   corpus_query!(dataset, sf, query, cpu_modes, gpu_modes, cpu_oracle, gpu_oracle)
//
// The mode arguments are the five planning modes the plan goldens carry, or'd together, or
// `none`. Every argument is written rather than deduced from the one beside it: a query's
// whole coverage reads off the line.
//
// The twenty here are chosen by plan size off the committed goldens rather than by taste —
// the smallest carrying no ticket — because the constraint is not what can plan but what is
// small enough that a failure is legible. The rest of the corpus is T19.

// --- tpch ---------------------------------------------------------------------
corpus_query!(tpch, 1, q6, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, data_fusion_exact, golden_exact);
// q1 and q17 carry an `avg`, whose count state the planner declares UInt64 and
// DataFusion's accumulator produces Int64 — #163, at all five modes, so neither runs
// here at all. Disabled rather than fixed: a fix made while enabling a query is a fix
// nothing else in this branch proves.
corpus_query!(tpch, 1, q1, none, none, data_fusion_exact, golden_exact);
// Every query below carrying a join is #181 on a device: `per_call_join_type` is called for
// every join by the gpu backend and is total only over the streaming family, so an Inner
// join reaches its `unreachable!`. Fifteen queries, all five modes, one defect. q6 is the
// only query here with no join, which is why it is the only device column left.
corpus_query!(tpch, 1, q14, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q19, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q12, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
// q13 and tpcds/q41 are #152 on a device at every mode: the join's recipe copies its probe
// batch and the frozen ABI has no copy, so neither call can run without erasing the other's
// input. The refusal is by name, which is the planner saying so rather than a crash.
corpus_query!(tpch, 1, q13, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q15, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q17, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q3, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q10, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// --- tpcds --------------------------------------------------------------------
corpus_query!(tpcds, 1, q41, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q42, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q3, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q43, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q52, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q55, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
// q96's `count(*)` is declared non-nullable and the state merge answers nullable, at the
// three tp4 modes only — the shuffle is what puts a merge under it. The two tp1 modes
// run, so this is a mode-scoped disablement rather than a query-scoped one.
corpus_query!(tpcds, 1, q96, bp_tp1_single | bp_tp1_rowgroup, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q15, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q37, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q82, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
