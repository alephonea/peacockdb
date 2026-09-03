// The benchmark corpus, one line per query. Included by `tests/peacock_gpu_benchmarks.rs`
// and by nothing else, which is what makes the list below a statement about timing rather
// than about coverage.
//
//   corpus_query_benchmark!(dataset, sf, query, modes)
//
// `modes` are the batch-partitioned planning modes of `common/bp_mode.rs`, or'd together,
// or `none`. Two of the five are timed — `bp_tp1_single` and `bp_tp4_sized` — the ends of
// the range: one lane and one batch against four lanes sized by the estimator. The three
// between them differ from these in the loader's mapping, not in the kind of work a node
// does, so timing them would cost four more runs per query to re-measure the same shapes.
//
// SEPARATE FROM `corpus_cases.inc`, which is the correctness list. Two reasons, and the
// second is the load-bearing one:
//
//   - A benchmark row carries neither of the two oracles that file's rows carry, because
//     nothing here compares an answer. Reusing that row shape would mean writing an oracle
//     beside every timing declaration and then ignoring it.
//   - The two lists disagree about sf on purpose. Correctness runs at sf1, where a wrong
//     answer is legible; a benchmark at sf1 measures mostly the host prologue, so the
//     rows worth timing are at sf40 and the rows worth checking are not. One file cannot
//     hold both without a column saying which tier each row is for, and that column is
//     this file.
//
// A query timed here must still be enabled on a device in `corpus_cases.inc` — the two
// binaries plan through the same `plan_at`, and a mode this one names for a query that
// mode refuses will fail at plan time rather than silently measure nothing.

// ── TPC-H, sf40 ─────────────────────────────────────────────────────────────
// sf40 AND NOT sf1, which is the task's own instruction and not a preference: at sf1 a
// query is mostly the host prologue — measured, 0..4us per call, below the resolution of
// the clock — so a fit over sf1 rows would fit the instrument. The correctness corpus
// stays at sf1 for the mirror-image reason: a wrong answer is legible in six million rows
// and not in six hundred million.
//
// The dataset lives outside the repo and `build-test-shadgpu.sh` symlinks it in as
// `testdata/tpch.sf40` before the run, so the path a case resolves is the same
// `<testdata>/<dataset>.sf<sf>` every other tier uses. There is no second variable.
corpus_query_benchmark!(tpch, 40, q6, bp_tp1_single | bp_tp4_sized);
// One mode, not two: q19's device cell in `corpus_cases.inc` is `bp_tp1_single` alone, and
// naming a mode a query is not enabled at buys a plan-time failure rather than a number.
// Which ticket holds the other four is recorded there, beside the cell — not restated here,
// where a second copy would outlive the first. That cell is written at sf1 and read here at
// sf40 because what it refuses is a (query, mode) pair: the mode declines the query's SHAPE,
// which no amount of data changes.
corpus_query_benchmark!(tpch, 40, q19, bp_tp1_single);
