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
corpus_query!(tpch, 1, q14, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q19, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, bp_tp1_single, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q12, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
// THE DEVICE COLUMN, after the first two cycles. q6 runs at all five modes and q19 at
// bp-tp1-single; every other device cell is off against a ticket. Four causes, none of them
// this task's to fix: #152 refuses a join whose recipe copies a side the ABI cannot copy (52
// cases, by name rather than by crash), #183 is an unload declaring Utf8View where the device
// exports Utf8, #184 is a cuDF failure in the hash scatter, and #185 is the device reporting
// one input row where the cpu reports the group count — which is the golden catching exactly
// what it is for. T19 is where these are settled query by query.
corpus_query!(tpch, 1, q13, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q15, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q17, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q3, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q10, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 1, the five smallest tpch plans the rollout list carries.
// scan-limit's rows are not determined — an unordered LIMIT over lineitem — so its modes
// need not agree and the oracle is the subset one; filter-project returns most of lineitem,
// which is over the result cap. Both take a live cpu run on the device for that reason.
// scan-limit runs at the tp4 modes only: at the two tp1 modes the planner puts the limit
// in the scan alone, and the cpu backend's reader does not apply it — six million rows
// where ten were asked for, #186.
// No device cell in this batch. All 23 failed on the first cycle, on three causes: #183's
// Utf8View export for the three string-keyed queries, #187's widened decimal for
// filter-project, and #188 for scan-limit, where cuDF refuses row groups and a row count in
// one read — the same limit-in-the-scan shape #186 is about, erroring rather than ignored.
corpus_query!(tpch, 1, scan_limit, bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_subset, live_cpu);
corpus_query!(tpch, 1, filter_project, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, live_cpu);
corpus_query!(tpch, 1, aggregate_groupby, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, cross_join, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, shuffle_additive, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 2, the five smallest joins. semi-join and anti-join return whole `orders` rows,
// which is over the result cap, so the device compares against a live cpu run.
// No device cell here either, and no new cause: 25 cells over three known tickets. #183 for the
// three that export a string, #152 for both copy shapes — left-join copies its probe batch and
// hash-join its build side — and #187 once, hash-join at bp-tp1-single, where the sum's declared
// Decimal128(25,2) comes back 38.
corpus_query!(tpch, 1, hash_join, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, left_join, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, semi_join, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, live_cpu);
corpus_query!(tpch, 1, anti_join, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, live_cpu);
corpus_query!(tpch, 1, nested_loop_join, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, data_fusion_exact, golden_exact);

// T19 batch 4, the small end of the tpch list emptied, and two firsts.
// shuffle-additive-avg is out at every mode on BOTH engines — #163's UInt64 count state against
// DataFusion's Int64 — which is the rollout's first whole-query cpu disablement rather than a
// subset of modes.
// shuffle-stddev is the first corpus query to need the tolerance oracle, and the tell is which
// mode is exact: bp-tp1-single matches DataFusion to the digit, and it is the mode with a single
// lane and so nothing to reassociate. The other four differ by ~3e-15, which makes the difference
// the association order of the Welford merge rather than its arithmetic — a wrong merge would not
// come right at one lane.
// No device cell again: #183 for shuffle-stddev and nested-loop-left-join, #188 for
// nested-limits, whose two intervals on one path meet the same refusal scan-limit does.
// mixed-join's `none` is the only one in this file that means "cannot cross to a device"
// rather than "crossed and was refused" — #168's interval `ScalarValue`, and the plan golden
// carries the reason at every mode.
corpus_query!(tpch, 1, shuffle_additive_avg, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, shuffle_stddev, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, data_fusion_approximate, golden_approx_std);
corpus_query!(tpch, 1, mixed_join, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, nested_loop_left_join, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, nested_limits, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 5, the tpch list from here being aggregates over joins, and the cpu is where it
// showed. Three of the five are cut before a device ever sees them: q22 carries an `avg` and is
// out at every mode on #163; q16 and rollup-over-join run at the two tp1 modes and not at the
// three tp4 ones, which is the shuffle rather than the query — #175's empty build side for q16,
// and #189 for the rollup, whose grouping-set id is UInt8 and whose hasher has no arm for it.
// The fourteen cells that did reach a device split evenly, seven and seven, between #152's
// build-side copy and #183 — every query here aggregates over a join, so which of the two it
// meets is decided by whether the plan reaches the join before the unload refuses the export.
corpus_query!(tpch, 1, rollup_over_join, bp_tp1_single | bp_tp1_rowgroup, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q4, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q16, bp_tp1_single | bp_tp1_rowgroup, none, data_fusion_exact, live_cpu);
corpus_query!(tpch, 1, q18, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q22, none, none, data_fusion_exact, golden_exact);

// T19 batch 6, the rest of tpch bar q7, q8 and q2. q11 is out at every mode on both engines:
// its nested-loop join carries a projection and the cpu backend passes `None` where DataFusion
// takes one, so the node declares two columns and the join answers three — #190. The first
// corpus query whose nested-loop join projects, which is why four batches went past it.
// The twenty device cells fell in one pattern, four times: #183 at bp-tp1-single and #152 at the
// other four modes, per query. bp-tp1-single is the only mode whose join gets a single probe
// batch, so it is the only one that reaches the unload — which is the ordering of the causes
// showing up as a property of the mode rather than of the query.
corpus_query!(tpch, 1, q20, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q5, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q9, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q11, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q21, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 7, the end of tpch: three queries rather than five, because these are its deepest
// plans and the regeneration cost has moved into the queries rather than their number.
// #152 at four modes each, and at bp-tp1-single — the mode that always reaches the unload —
// three different type disagreements at that one node: #183's string for q7, #187's decimal for
// q2, and #191 for q8, where the device exports Int16 for an extracted year the plan declared
// Int32. Same site, three tickets, decided by which column type differs first.
corpus_query!(tpch, 1, q7, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q8, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpch, 1, q2, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

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

// T19 batch 3, tpcds, chosen for feature spread rather than size: q48 carries no feature code,
// q21 and q84 are top_n, q93 adds an outer join and q34 a correlated subquery. The three smaller
// candidates all carry `avg`, which #163 disables at every mode, so they would produce no device
// cell to learn from.
// No device cell, and still no seventh cause: #152's build-side copy 20 times, #183 three times,
// and #185 twice — q48 and q93 at bp-tp1-single, where the device RUNS and the golden catches it.
// Those two widen #185: q93 reports in_rows 7169 against the cpu's 7486, so the rule is the merge
// counting its own output rather than its input, and `[[1]]` is that rule with no group keys.
corpus_query!(tpcds, 1, q48, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q21, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q93, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q34, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q84, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 8, tpcds, chosen by expected executor calls rather than plan size: row groups scanned
// times joins, off the bp-tp4-rowgroup plans. q32 and q26 score as low and are skipped for the
// `avg` #163 takes out at every mode.
// Batch 6's pattern again — #152 at four modes, the unload's type at bp-tp1-single — with q97 as
// its exception: its outer join copies the PROBE batch, which #152 refuses at any batch count, so
// the one mode that usually gets far enough does not.
corpus_query!(tpcds, 1, q45, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q40, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q97, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q50, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q62, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 9, the next five by expected calls. q1 and q7 carry an `avg` and are out at every
// mode on #163, which the feature column says and I did not read before choosing them.
// The fifteen device cells are the settled pattern: #152's build copy at four modes, the unload's
// type at bp-tp1-single — a string for q73 and q79, a decimal for q94.
corpus_query!(tpcds, 1, q73, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q79, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q1, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q7, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q94, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 10, by expected calls with the `avg` #163 disables filtered out in the same pass.
// Twenty-five device cells in the settled pattern without a single exception: #152's build copy at
// four modes and #183 at bp-tp1-single, five times over.
corpus_query!(tpcds, 1, q19, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q8, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q99, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q91, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q46, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 11, by expected calls. q71 is #152 at bp-tp1-single too, where every other query so
// far has reached the unload: its probe side is three fact tables unioned, and a fact table
// streams in 8192-row batches whatever the mode. `single` is a scan's partitioning, not a
// promise of one batch — what decides #152's build half is the probe side's SIZE.
corpus_query!(tpcds, 1, q68, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q71, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q66, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q76, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q16, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 12, by expected calls. q90 runs at the two tp1 modes and not the three tp4 ones:
// #180's shuffled `count(*)` merging to nullable against a non-nullable declaration, the second
// query to reach it after q96 and mode-scoped for the same reason — a shuffle is what puts a
// state merge under the aggregate.
// q2 is #152 at bp-tp1-single as well, and it is the first query the mechanism rule predicted
// rather than explained: its probe side unions web_sales and catalog_sales, so the join sees
// several batches at every mode.
corpus_query!(tpcds, 1, q25, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q29, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q90, bp_tp1_single | bp_tp1_rowgroup, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q2, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q59, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 13, by expected calls, and both of the rollout's cpu-side tickets get a second query.
// q5 carries a rollup, so #189's UInt8 grouping-set id meets the hasher at the three tp4 modes and
// the two tp1 ones are clean — mode-scoped exactly as rollup-over-join was. q54's nested-loop join
// projects, so #190 drops it and the node declares three columns against DataFusion's four.
// On the device, q38 is the third cell in the rollout where the device RAN and the golden caught
// it rather than the run refusing — #185 again, in_rows 11788 against the cpu's 12446, which is
// the merge counting its own output.
corpus_query!(tpcds, 1, q78, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q5, bp_tp1_single | bp_tp1_rowgroup, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q54, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q95, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q38, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 14, by expected calls, and the batch-count rule LOST to the query written to test
// it. q11's eleven joins all copy their build side and four take 336 or 88 probe batches at
// bp-tp1-single, where the rule says #152 must fire; the device reached the unload and failed
// on #183 instead. What separates it from q71 and q2, which do refuse there, is not in these
// goldens — see #152. q87's RightAnti-against-#175 race resolved as neither: it ran, and the
// golden caught it on #185.
corpus_query!(tpcds, 1, q87, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q11, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q83, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q61, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q10, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 15, the expensive end of the pool: four semi-joins and a rollup over an outer join.
// q77 runs at the two tp1 modes only: #175's empty build side, the second query to reach it after
// q16 and mode-scoped for the same reason. Whether its rollup would also meet #189 at those modes
// is unknown — #175 refuses first, and the ordering is not something a disabled cell can show.
corpus_query!(tpcds, 1, q33, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q56, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q60, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q69, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q77, bp_tp1_single | bp_tp1_rowgroup, none, data_fusion_exact, golden_exact);

// T19 batch 16, the top of the cost ranking. q80 is q77's shape — a rollup over an outer join —
// and it refuses at the three tp4 modes on #189's UInt8 grouping-set id rather than on #175's
// empty build side, which is the ticket that took q77 there. Same two candidates, opposite winner,
// so which one a query meets is a property of the query and not of the shape.
corpus_query!(tpcds, 1, q74, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q58, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q80, bp_tp1_single | bp_tp1_rowgroup, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q4, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q31, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);

// T19 batch 17, the four largest plans in the corpus — 324 to 447 nodes against 267 for the
// largest already enabled, and no overlap between the two sets. Their node KINDS are ordinary:
// two of the four sit below the corpus median of nine, so this batch varies plan size and
// nothing else. q88 runs at the two tp1 modes only — #180's shuffled `count(*)` against a
// non-nullable declaration, the third query to reach it after tpcds q96 and q90.
// q64 is out at every mode on both engines and is not re-run: it takes 13 GB on a 15 GiB host,
// which is #192 — the corpus tier passes no budget, so nothing refuses a plan whose accumulators
// are estimated at 35.9 GB in total.
// Twelve device cells and no new cause: #152 nine times, #183 twice, and q88 at bp-tp1-single is
// #185 again — the fifth cell in the rollout where the device ran and the golden caught it.
corpus_query!(tpcds, 1, q75, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q23, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q88, bp_tp1_single | bp_tp1_rowgroup, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q64, none, none, data_fusion_exact, golden_exact);

// T19 batch 18, the last query outside `avg` and the first four inside it. Each is RUN rather than
// declared: a feature code is a selection filter, not evidence of an outcome. All four refuse on
// #163 at every mode, each naming the UInt64 count state against DataFusion's Int64, and the
// column it lands on differs per query — 2 for q6, 1 for the rest.
// join_int's five device cells are #152 at four modes and #185 at bp-tp1-single, the sixth cell in
// the rollout the device ran and the golden caught.
corpus_query!(tpch, 1, join_int, bp_tp1_single | bp_tp1_rowgroup | bp_tp4_single | bp_tp4_rowgroup | bp_tp4_sized, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q6, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q9, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q13, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q14, none, none, data_fusion_exact, golden_exact);

// T19 batch 19, the first seven of the thirteen `avg` queries, run rather than declared. All seven
// refuse on #163 at every mode, and the column runs 1, 2, 2, 2, 5, 6, 6 across them — the count
// state's position in the state row rather than a fixed place.
corpus_query!(tpcds, 1, q17, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q18, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q22, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q24, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q26, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q30, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q32, none, none, data_fusion_exact, golden_exact);

// T19 batch 20, the last six, run rather than declared, and the rollout's end. All six refuse on
// #163 at all five modes. Across the seventeen `avg` queries the refusal's column runs 1 to 10,
// which is the count state's position in a differently-shaped state row each time rather than a
// place a narrower fix could special-case.
corpus_query!(tpcds, 1, q35, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q39, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q65, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q81, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q85, none, none, data_fusion_exact, golden_exact);
corpus_query!(tpcds, 1, q92, none, none, data_fusion_exact, golden_exact);
