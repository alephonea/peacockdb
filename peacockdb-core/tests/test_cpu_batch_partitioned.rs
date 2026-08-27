//! SQL in, rows out: the batch-partitioned mode end to end on the CPU backend, at the five
//! modes and at the injected shapes.
//!
//! Every other test of this mode proves one layer against a fixture of the last one's
//! shape. This one starts at a query's text and ends at its rows, so what it tests is the
//! join between the pieces — and the oracle is DataFusion on the same SQL rather than our
//! own legacy executor, which would agree with us wherever we are consistently wrong.
//!
//! Two axes vary: the five modes, which are plans the planner would have chosen, and the
//! injected shapes, which are ones it never would. `small_table_bytes` is constant across
//! both, so a join's co-partitioning is the knob neither turns.

mod common;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::DataType;
use datafusion::execution::context::SessionContext;

use peacockdb_core::batch_partitioned::cpu_backend::backend::CpuBackend;
use peacockdb_core::batch_partitioned::driver::{RunReport, batch_partitioned_driver};
use peacockdb_core::batch_partitioned::plan::plan_batch_partitioned;
use peacockdb_core::batch_partitioned::{GpuNode, RunError, When};

use common::injection::{
    CAP, Dimensions, Drain, Empties, Injected, InjectedContext, Injection, PlannedMode, Rebatch,
    SEED, apply, node_count, planned_mode, select,
};
use common::bp_mode::{BP_MODES, BpMode};
use common::{assert_results_match, batches_to_sorted_str, data_dir_for, queries_dir_for};

/// Where a Welford merge is the only divergence: this mode decomposes the aggregate into
/// an init, two merges and a finalize, and DataFusion computes it in one pass, so the last
/// digits differ by reassociation. The legacy GPU tier uses the same figure for the same
/// reason (`golden_approx_std`).
const WELFORD_TOLERANCE: f64 = 1e-11;

/// One query at every mode, against DataFusion on the same text.
/// Whether a query also runs the injected shapes. An enum rather than a flag, because
/// which of the two a call gets is the difference between five runs and up to thirty-five.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coverage {
    ModesOnly,
    ModesAndInjection,
}

async fn answers_match_datafusion(
    dataset: &str,
    query: &str,
    tolerance: Option<f64>,
    coverage: Coverage,
) {
    let sql_path = queries_dir_for(dataset).join(format!("{query}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));
    sql_answers_match_datafusion(dataset, query, &sql, tolerance, coverage).await;
}

async fn sql_answers_match_datafusion(
    dataset: &str,
    query: &str,
    sql: &str,
    tolerance: Option<f64>,
    coverage: Coverage,
) {
    let data_dir = data_dir_for(dataset, "1");

    let oracle_ctx =
        peacockdb_core::register_tables_for(peacockdb_core::build_session_state(1), &data_dir)
            .await
            .expect("register the tables");
    let expected = oracle_ctx
        .sql(sql)
        .await
        .expect("the oracle plans the query")
        .collect()
        .await
        .expect("the oracle runs the query");

    // Encoded once: `assert_results_match` renders both sides per call, so one oracle
    // against thirty runs is rendered thirty times, and on a million-row result that is
    // most of the tier. The tolerance path indexes rather than renders and keeps its own.
    let expected_rows = tolerance.is_none().then(|| sorted_rows(&expected)).flatten();
    let mut planned = Vec::new();
    for mode in &BP_MODES {
        let name = mode.name;
        let ctx = peacockdb_core::register_tables_for(
            peacockdb_core::build_session_state(mode.target_partitions),
            &data_dir,
        )
        .await
        .expect("register the tables");
        let plan = ctx
            .sql(sql)
            .await
            .expect("the query plans")
            .create_physical_plan()
            .await
            .expect("the query has a physical plan");
        let (tree, _memory) = plan_batch_partitioned(&plan, mode.knobs())
            .unwrap_or_else(|error| panic!("{dataset}/{query} at {name}: {error}"));
        run_and_check(
            tree.as_ref(),
            &ctx.task_ctx(),
            Injection::NONE,
            &Oracle {
                batches: &expected,
                rows: expected_rows.clone(),
                tolerance,
            },
            &format!("{dataset}/{query} at {name}"),
        );
        planned.push((planned_mode(name, tree.as_ref()), tree, ctx));
    }
    if coverage == Coverage::ModesOnly {
        return;
    }

    // One oracle for every shape below, computed above: it is the expensive half and it is
    // invariant, which is what makes a layout that changes the answer a defect rather than
    // a disagreement between two shapes of ours.
    let modes: Vec<PlannedMode> = planned.iter().map(|(mode, _, _)| *mode).collect();
    // Beside the times, because what an injected run costs is the size of the answer it
    // has to compare rather than the rows it scanned.
    eprintln!(
        "[injection] {dataset}/{query} oracle rows={}",
        expected.iter().map(RecordBatch::num_rows).sum::<usize>()
    );
    for candidate in select(&modes, &Dimensions::default(), CAP, SEED) {
        let (_, tree, ctx) = &planned[candidate.mode];
        let injected = apply(tree.as_ref(), candidate.injection, SEED);
        let what = format!("{dataset}/{query} at {}", candidate.label(&modes));
        // A rebatcher that found no edge to take injects nothing, and the run would then
        // be the as-planned one under a label saying otherwise — the same shape as a
        // dimension whose carrier was quietly dropped.
        if candidate.injection.rebatch != Rebatch::None {
            assert!(
                node_count(injected.as_ref()) > node_count(tree.as_ref()),
                "{what}: the rebatcher found no edge to take"
            );
        }
        let started = std::time::Instant::now();
        run_and_check(
            injected.as_ref(),
            &ctx.task_ctx(),
            candidate.injection,
            &Oracle {
                batches: &expected,
                rows: expected_rows.clone(),
                tolerance,
            },
            &what,
        );
        // Swallowed by libtest unless the run is a failing one or `--nocapture` is passed,
        // which is how the timing table in llm-wiki/reports/ was taken.
        eprintln!("[injection] {what} {}us", started.elapsed().as_micros());
    }
}

/// The answer every shape of one query is measured against: the rows, and — where the
/// comparison is exact — their arrow row encoding, sorted, held rather than rebuilt per
/// run.
struct Oracle<'a> {
    batches: &'a [RecordBatch],
    rows: Option<Vec<Vec<u8>>>,
    tolerance: Option<f64>,
}

/// The columns an answer declares. Names are not an input to the row encoding, and the
/// rendered comparison it replaced carried them in its header — so they are asserted here
/// rather than dropped. This mode's own defect class is what makes it worth an assert: a
/// finalize project emitting the right values under the wrong names ([#163]), which the
/// oracle cannot get wrong the same way.
///
/// [#163]: ../../llm-wiki/tickets.md#t163
fn columns_of(batches: &[RecordBatch]) -> Vec<(String, DataType)> {
    batches
        .first()
        .map(|batch| {
            batch
                .schema()
                .fields()
                .iter()
                .map(|field| (field.name().clone(), field.data_type().clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Every row as its arrow row encoding, sorted — a multiset of rows, compared in bytes.
///
/// Values and types only — a column's NAME is not an input to the encoding, so
/// [`columns_of`] is asserted beside this rather than folded into it. On values it is not
/// looser than the rendered text it replaced: arrow renders a float at round-trip
/// precision, so two values that render alike are the same bits. What it drops is the
/// rendering, which one oracle against thirty runs otherwise pays for thirty times —
/// `SELECT *` over a million rows renders for seconds. A mismatch falls back to the
/// rendered form, so the message still says how they differ.
fn sorted_rows(batches: &[RecordBatch]) -> Option<Vec<Vec<u8>>> {
    use datafusion::arrow::row::{RowConverter, SortField};

    let schema = batches.first()?.schema();
    let fields: Vec<SortField> = schema
        .fields()
        .iter()
        .map(|field| SortField::new(field.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields).ok()?;
    let mut rows = Vec::new();
    for batch in batches {
        let encoded = converter.convert_columns(batch.columns()).ok()?;
        rows.extend(encoded.iter().map(|row| row.as_ref().to_vec()));
    }
    rows.sort_unstable();
    Some(rows)
}

/// One run of one shape, against the one oracle. The two accounting assertions ride here
/// rather than at the call sites: an injected run leaks exactly as visibly as a planned
/// one, and a batch held and never released shows in neither's rows.
fn run_and_check(
    tree: &dyn peacockdb_core::batch_partitioned::GpuNode,
    task: &std::sync::Arc<datafusion::execution::TaskContext>,
    injection: Injection,
    oracle: &Oracle<'_>,
    what: &str,
) {
    let ctx = InjectedContext::new(task.clone(), injection, SEED);
    // The check the planner made, made again: an injected tree is one no planner emitted,
    // and the driver asks only for canonical form. Without this a rewrite that broke a
    // node's requirements would run and answer, which is the failure this whole tier is
    // about.
    peacockdb_core::batch_partitioned::validate::validate(tree)
        .unwrap_or_else(|error| panic!("{what} is not a plan: {error}"));
    let report = batch_partitioned_driver::<Injected>(tree, &ctx, None)
        .unwrap_or_else(|error| panic!("{what}: {error}"));
    let actual: Vec<RecordBatch> = report
        .batches
        .iter()
        .map(|batch| batch.record_batch().clone())
        .collect();
    assert_eq!(
        columns_of(&actual),
        columns_of(oracle.batches),
        "{what} answers different columns from the oracle"
    );
    match &oracle.rows {
        // The rendered comparison is what says *how* they differ, so a mismatch pays for it
        // once rather than every run paying in case one does.
        Some(rows) if sorted_rows(&actual).as_ref() != Some(rows) => assert_eq!(
            batches_to_sorted_str(&actual),
            batches_to_sorted_str(oracle.batches),
            "result for {what} differs from oracle (exact compare)"
        ),
        Some(_) => {}
        None => assert_results_match(oracle.batches, &actual, oracle.tolerance, what),
    }
    assert_eq!(report.in_flight_bytes, 0, "{what} ended holding batches");
    assert_eq!(
        report.holds, report.releases,
        "{what} held {} batches and released {}",
        report.holds, report.releases
    );
    // Counted where it happens rather than inferred from the answer, which is unchanged
    // either way: a seed under which no call fires would leave the setting carried in the
    // label and in nothing else.
    if injection.empties != Empties::Never {
        assert!(
            ctx.empty_batches() > 0,
            "{what}: no source call emitted an empty batch"
        );
    }
}

/// `end_to_end!(dataset, query)` — one test per query, so a failure names it. Two optional
/// arguments follow: a tolerance, for a query whose answer is compared approximately, and
/// the coverage, which decides whether the query runs the five modes alone or the injected
/// shapes as well. `injected_queries!` is the only caller passing the second.
macro_rules! end_to_end {
    ($dataset:ident, $query:ident) => {
        end_to_end!($dataset, $query, None);
    };
    ($dataset:ident, $query:ident, $tolerance:expr) => {
        end_to_end!($dataset, $query, $tolerance, Coverage::ModesOnly);
    };
    ($dataset:ident, $query:ident, $tolerance:expr, $coverage:expr) => {
        paste::paste! {
            #[tokio::test]
            async fn [<bp_ $dataset _ $query>]() {
                answers_match_datafusion(
                    stringify!($dataset),
                    &stringify!($query).replace('_', "-"),
                    $tolerance,
                    $coverage,
                )
                .await;
            }
        }
    };
}

/// The injected set, and the tests for it, from one list — so a query leaves the set only
/// by leaving the list, which `the_injected_set_keeps_the_shapes_only_one_query_has` reads.
/// The two forms differ by one word otherwise, and turning one back is invisible.
macro_rules! injected_queries {
    ($($dataset:ident / $query:ident),+ $(,)?) => {
        /// The queries that run the injected shapes as well as the five modes. Four of
        /// them carry a shape no other query here has, and a trim that drops one of those
        /// cost coverage rather than runs: `tpcds/q33` is the only four-lane interleave,
        /// `tpch/nested_loop_join` and `tpch/nested_loop_left_join` are the two nested-loop
        /// forms, and `tpch/nested_limits` carries both row-interval lowerings.
        const INJECTED: &[&str] = &[$(concat!(
            stringify!($dataset),
            "/",
            stringify!($query)
        )),+];
        $(end_to_end!($dataset, $query, None, Coverage::ModesAndInjection);)+
    };
}

// ── the join capability matrix ──────────────────────────────────────────────
// Chosen by cover over the matrix rather than by taste, off the mode goldens: every join
// type this mode claims, crossed with a residual filter, null_equals_null and multi-key.
// Eleven of the seventeen carry a cell no other query here does.
//
// nested-loop Inner, and the smallest plan in the corpus that carries a join at all.
// injected: tpch/nested-loop-join
// nested-loop Left: the one mode whose probe side is a single batch, so its call takes the
// build side rather than a copy of it.
// injected: tpch/nested-loop-left-join
// RightAnti — the probe-side semi family, answered per batch with no finish pass.
// injected: tpch/anti-join
// Left outer: the finish pass, and the only cell with no device path at all (#152).
end_to_end!(tpch, left_join);
// Inner, multi-key Inner, Inner with a residual filter, LeftSemi and RightSemi in one plan.
end_to_end!(tpch, q20);
// LeftAnti with a residual filter, which the matrix answers in one legacy call over a
// probe side the planner made single-batch.
end_to_end!(tpch, q21);
// Full outer: Right's per-batch call and Left's finish, in one node.
// injected: tpcds/q97
// LeftAnti, and a LeftSemi carrying a residual filter.
// injected: tpcds/q16
// LeftMark, which scatters a boolean into an all-false column.
// injected: tpcds/q45
// LeftSemi with null_equals_null — a set operation lowered to a join, where NULL = NULL.
// injected: tpcds/q8
// RightSemi with null_equals_null.
end_to_end!(tpcds, q38);
// RightAnti with null_equals_null.
end_to_end!(tpcds, q87);
// Right outer, and its keys are composite — the only multi-key outer in either corpus.
// injected: tpcds/q93

// ── the shapes a join cover does not reach ──────────────────────────────────
// A union executed as an INTERLEAVE at four lanes: output lane p is built from lane p of
// each branch, which is the whole claim and is invisible at one lane.
// injected: tpcds/q33
// A union that cannot interleave: its two unions carry branches that disagree on lane
// count, which is what makes them unions rather than interleaves, and the lanes above are
// their sum. q77 is the shape this claim was written for — 4+1+4 — and it is not here
// because it is refused rather than untested: one of its Right outers gets a lane whose
// build side is empty, and what that owes is its probe side padded, which takes a call the
// recipe does not publish ([#175](../../llm-wiki/tickets.md#t175)).
// injected: tpcds/q2
// Both row-interval lowerings on one root-to-leaf path — the root-adjacent one becoming
// the unload's skip/fetch and the mid-plan one a limit over the scan — and the only
// OFFSETs in either corpus. It is also where the matrix's cross join is covered: what
// connects the two intervals has to be one, so the cell has no query of its own here.
// injected: tpch/nested-limits
// A merge over state worth merging: the Welford init, both merges and the finalize
// project. Every other aggregate here merges a sum.
end_to_end!(tpch, shuffle_stddev, Some(WELFORD_TOLERANCE));

// ── the injected set ────────────────────────────────────────────────────────

// Eleven queries, four of which carry a shape no other query here has — see INJECTED.
injected_queries!(
    tpch/nested_loop_join,
    tpch/nested_loop_left_join,
    tpch/anti_join,
    tpcds/q97,
    tpcds/q16,
    tpcds/q45,
    tpcds/q8,
    tpcds/q93,
    tpcds/q33,
    tpcds/q2,
    tpch/nested_limits
);

// ── the aggregate that stops aggregating ────────────────────────────────────

/// Groups nearly as many as rows, over more than 100,000 of them: the shape that makes
/// DataFusion's partial aggregate give up on grouping.
///
/// `AggregateExec` in Partial mode probes its own aggregation ratio and, where grouping is
/// not paying, passes its input through as state — sound in a DataFusion plan because a
/// Final stage regroups downstream, and wrong here, where the init emits state and the
/// merge is a Partial too. The duplicate keys reach the finalize and come out as extra
/// rows: 2,797,913 groups against 2,764,744 before the fix, and tpcds q97 counted them.
///
/// One key does not reach it — the ratio is far below the threshold — which is why this
/// case has two.
#[tokio::test]
async fn a_two_key_group_by_over_many_rows_does_not_emit_a_group_twice() {
    sql_answers_match_datafusion(
        "tpcds",
        "two-key group by",
        "SELECT count(*) FROM (SELECT ss_customer_sk, ss_item_sk FROM store_sales GROUP BY 1, 2)",
        None,
        Coverage::ModesOnly,
    )
    .await;
}

// ── what a limit costs, in calls rather than in rows ────────────────────────

/// The claims a result comparison cannot make: the same rows come back whether the plan
/// read the whole table or stopped, so what a limit buys is only visible as calls not made.
///
/// `nested-limits` carries both lowerings on one root-to-leaf path — the root-adjacent one
/// as the unload's skip/fetch, and the mid-plan one as a `GpuLimit` over the scan — so one
/// query holds every count below.
#[tokio::test]
async fn a_limit_slices_at_most_two_batches_and_stops_the_scan() {
    use peacockdb_core::batch_partitioned::driver::CallKind;
    use peacockdb_core::batch_partitioned::nodes::{NodeRef, as_node_ref};

    let data_dir = data_dir_for("tpch", "1");
    let sql = std::fs::read_to_string(queries_dir_for("tpch").join("nested-limits.sql"))
        .expect("the query text");
    let mut most_offered = 0;
    for mode in &BP_MODES {
        let name = mode.name;
        let ctx = peacockdb_core::register_tables_for(
            peacockdb_core::build_session_state(mode.target_partitions),
            &data_dir,
        )
        .await
        .expect("register the tables");
        let plan = ctx
            .sql(&sql)
            .await
            .expect("the query plans")
            .create_physical_plan()
            .await
            .expect("the query has a physical plan");
        let (tree, _memory) = plan_batch_partitioned(&plan, mode.knobs())
            .unwrap_or_else(|error| panic!("nested-limits at {name}: {error}"));
        let offered = batches_offered(tree.as_ref());
        let report = batch_partitioned_driver::<CpuBackend>(tree.as_ref(), &ctx.task_ctx(), None)
            .unwrap_or_else(|error| panic!("nested-limits at {name}: {error}"));
        let calls = |kind: CallKind| report.trace.iter().filter(|e| e.call == kind).count();

        // An interval has two ends, so at most two batches can straddle it — and a batch
        // wholly inside is forwarded untouched rather than sliced.
        assert!(
            calls(CallKind::UnloadRange) <= 2,
            "nested-limits at {name}: {} batches sliced at the boundary",
            calls(CallKind::UnloadRange)
        );
        // Two sources, and each is satisfied by its first batch: the mid-plan limit wants
        // 28 rows and a row group holds far more, so the scan under it never reads a
        // second one whatever the batching.
        let pulled = calls(CallKind::NextBatch);
        assert_eq!(
            pulled, 2,
            "nested-limits at {name}: the sources were pulled {pulled} times, and one \
             batch each is what their limits need"
        );
        assert!(
            !report.satisfied.is_empty(),
            "nested-limits at {name}: the run drained rather than stopping at its limit"
        );
        // The mid-plan limit holds nothing whatever its offset, which is what the slice
        // symbol buys: its queue never carries more than the one batch it was handed. The
        // executor's own residency is a unit case; this is the claim the driver makes.
        let limit = limit_node(tree.as_ref());
        assert!(
            report.peak_queued[limit] <= 1,
            "nested-limits at {name}: the limit queued {} batches",
            report.peak_queued[limit]
        );
        most_offered = most_offered.max(offered);
    }
    // Without this the count above is a claim about a plan that had nothing to stop: at
    // the single-batch modes a mapping offers one batch per lane and stopping is free.
    assert!(
        most_offered > 2,
        "no mode offered more batches than were pulled, so nothing here was stopped"
    );

    /// The mid-plan limit's index in the driver's pre-order numbering, which is the tree
    /// walked children-after-self.
    fn limit_node(root: &dyn peacockdb_core::batch_partitioned::GpuNode) -> usize {
        fn walk(
            node: &dyn peacockdb_core::batch_partitioned::GpuNode,
            next: &mut usize,
        ) -> Option<usize> {
            let here = *next;
            *next += 1;
            if matches!(as_node_ref(node), NodeRef::Limit(_)) {
                return Some(here);
            }
            node.children()
                .into_iter()
                .find_map(|child| walk(child, next))
        }
        walk(root, &mut 0).expect("nested-limits carries a mid-plan limit")
    }

    /// How many batches every source in the plan could produce — the mapping's own count,
    /// which is what a scan that ran to the end would have read.
    fn batches_offered(node: &dyn peacockdb_core::batch_partitioned::GpuNode) -> usize {
        let here = match as_node_ref(node) {
            NodeRef::LoadParquet(load) => load.partition_groups.iter().map(Vec::len).sum(),
            _ => 0,
        };
        here + node
            .children()
            .into_iter()
            .map(batches_offered)
            .sum::<usize>()
    }
}

// ── the accounting, on a real plan ──────────────────────────────────────────

/// The budget path, on a query rather than on a mock.
///
/// Eight `Executor` impls report a residency and a transient, and until this ran on a real
/// plan the only things reading them were two unit cases: every end-to-end run above
/// passes `None`, which is the accountant watching rather than enforcing.
///
/// The budget is searched for rather than taken from the watching run's peak, and the
/// reason is the finding: a pre-call check tests the MODELLED transient, which can exceed
/// what the run was ever observed holding — this query peaks at 8,222 bytes and the
/// smallest budget it fits in is 9,556, which is what its nested-loop join asks for before
/// a call. A budget equal to the observed peak therefore trips, which is the accounting
/// working rather than failing.
///
/// So the claim is that a boundary exists and is one byte wide: the smallest budget that
/// completes, and the byte below it that does not.
#[tokio::test]
async fn a_query_has_a_smallest_budget_that_fits_and_trips_a_byte_below_it() {
    let data_dir = data_dir_for("tpch", "1");
    let sql = std::fs::read_to_string(queries_dir_for("tpch").join("nested-loop-join.sql"))
        .expect("the query text");
    let mode = &BP_MODES[3];
    let name = mode.name;
    let ctx = peacockdb_core::register_tables_for(
        peacockdb_core::build_session_state(mode.target_partitions),
        &data_dir,
    )
    .await
    .expect("register the tables");
    let plan = ctx
        .sql(&sql)
        .await
        .expect("the query plans")
        .create_physical_plan()
        .await
        .expect("the query has a physical plan");
    let (tree, _memory) = plan_batch_partitioned(&plan, mode.knobs())
        .unwrap_or_else(|error| panic!("nested-loop-join at {name}: {error}"));

    let watching = batch_partitioned_driver::<CpuBackend>(tree.as_ref(), &ctx.task_ctx(), None)
        .expect("the unbudgeted run finishes");
    assert!(
        watching.peak_bytes > 0,
        "an unbudgeted run observed nothing"
    );

    // Why the two figures differ, since a reader who tries to remove the double count
    // finds this test red: a probing join's `resident_bytes` is its build side plus its
    // accumulated keys, and its `scratch_bytes` adds the build side AGAIN, because the
    // call is about to read it. The spec has scratch consult `self`, so it is deliberate.
    // Measured here: peak 8,222, smallest fitting budget 9,556, and region — this join's
    // build side — is 920 bytes on its own. So the gap is that side plus whatever else was
    // resident at the instant the check ran, not that side alone.
    let fits = |budget: usize| {
        batch_partitioned_driver::<CpuBackend>(tree.as_ref(), &ctx.task_ctx(), Some(budget)).is_ok()
    };
    // The observed peak is a floor for the search and not the answer, per the block above.
    let (mut low, mut high) = (watching.peak_bytes, watching.peak_bytes * 8);
    assert!(fits(high), "the query does not run at eight times its peak");
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if fits(middle) {
            high = middle
        } else {
            low = middle
        }
    }
    assert!(fits(high), "the smallest fitting budget does not fit");
    // Asserted as a shape rather than as a number, so the case survives a fixture change:
    // the excess is a real quantity of this plan's — one side of one join and the rest of
    // what was live — rather than slack.
    let excess = high - watching.peak_bytes;
    assert!(
        excess > 0 && excess < watching.peak_bytes,
        "the smallest fitting budget is {high} against a peak of {}, and the gap should be \
         one side of one join rather than a multiple of the run",
        watching.peak_bytes
    );

    match batch_partitioned_driver::<CpuBackend>(tree.as_ref(), &ctx.task_ctx(), Some(high - 1)) {
        Err(RunError::BudgetExceeded { when, message }) => {
            assert_eq!(when, When::PreCall, "{message}");
            assert!(
                message.contains("GpuNestedLoopJoin"),
                "the failure names the node it happened at: {message}"
            );
        }
        other => panic!("a byte below {high} should not fit, got {other:?}"),
    }
}

/// What a call measured, against what its executor said it would take.
///
/// `CallStats::scratch_bytes` is the measured half of the accounting, and every CPU
/// executor returned `None` for it until the exec nodes began reporting — so
/// `underestimates` was empty on every query for want of an input rather than for want of
/// an underestimate.
///
/// So `measured_calls` is asserted first and the empty list second, in that order: an
/// empty list means the model held only where something measured. Red with
/// `CallStats::default()` back in `CpuExec::exec` — which is the regression this exists to
/// catch, and which `calls > 0` could not see, since a call counts whether it measured or
/// not.
#[tokio::test]
async fn the_model_is_compared_against_what_the_calls_measured() {
    let data_dir = data_dir_for("tpch", "1");
    let sql = std::fs::read_to_string(queries_dir_for("tpch").join("filter-project.sql"))
        .expect("the query text");
    let mode = &BP_MODES[3];
    let name = mode.name;
    let ctx = peacockdb_core::register_tables_for(
        peacockdb_core::build_session_state(mode.target_partitions),
        &data_dir,
    )
    .await
    .expect("register the tables");
    let plan = ctx
        .sql(&sql)
        .await
        .expect("the query plans")
        .create_physical_plan()
        .await
        .expect("the query has a physical plan");
    let (tree, _memory) = plan_batch_partitioned(&plan, mode.knobs())
        .unwrap_or_else(|error| panic!("filter-project at {name}: {error}"));
    let report = batch_partitioned_driver::<CpuBackend>(tree.as_ref(), &ctx.task_ctx(), None)
        .expect("the run finishes");
    assert!(
        report.measured_calls > 0,
        "no call reported a measured transient, so there was nothing to compare the model \
         against"
    );
    assert!(
        report.underestimates.is_empty(),
        "an exec node needed more than its model allowed: {:?}",
        report.underestimates
    );
}

// ── that each dimension does something ──────────────────────────────────────

/// Three of the four dimensions, made visible in the calls rather than in the rows.
///
/// Every injected run asserts the same answer, which is exactly what a dimension that did
/// nothing would also produce: a setting that never fired would pass every case above and
/// prove nothing. So each is read off the trace against the same plan uninjected.
#[tokio::test]
async fn an_injected_run_makes_different_calls_from_the_plan_it_came_from() {
    use common::injection::{Drain, Empties, Rebatch};
    use peacockdb_core::batch_partitioned::driver::CallKind;
    use peacockdb_core::batch_partitioned::nodes::{NodeRef, as_node_ref};

    let data_dir = data_dir_for("tpch", "1");
    let sql = std::fs::read_to_string(queries_dir_for("tpch").join("nested-loop-join.sql"))
        .expect("the query text");
    // The one mode with batching off, so the small-table rule leaves every source at four
    // lanes and there is a lane to drain.
    let mode = &BP_MODES[2];
    let name = mode.name;
    let ctx = peacockdb_core::register_tables_for(
        peacockdb_core::build_session_state(mode.target_partitions),
        &data_dir,
    )
    .await
    .expect("register the tables");
    let plan = ctx
        .sql(&sql)
        .await
        .expect("the query plans")
        .create_physical_plan()
        .await
        .expect("the query has a physical plan");
    let (tree, _memory) = plan_batch_partitioned(&plan, mode.knobs())
        .unwrap_or_else(|error| panic!("nested-loop-join at {name}: {error}"));

    let run = |injection: Injection| {
        let injected = apply(tree.as_ref(), injection, SEED);
        let context = InjectedContext::new(ctx.task_ctx(), injection, SEED);
        let report = batch_partitioned_driver::<Injected>(injected.as_ref(), &context, None)
            .unwrap_or_else(|error| panic!("nested-loop-join at {}: {error}", injection.label()));
        let pulls = report
            .trace
            .iter()
            .filter(|event| event.call == CallKind::NextBatch)
            .count();
        let lane_pulls = |lane: u32| {
            report
                .trace
                .iter()
                .filter(|event| event.call == CallKind::NextBatch && event.lane == lane)
                .count()
        };
        (report.lanes_of.len(), pulls, lane_pulls(0))
    };

    let (nodes, pulls, first_lane) = run(Injection::NONE);
    assert!(
        first_lane > 0,
        "lane 0 read nothing before anything was injected, so draining it proves nothing"
    );

    // A rebatcher is a node the plan did not ask for, so the tree it runs is a longer one.
    let (rebatched, _, _) = run(Injection {
        rebatch: Rebatch::AboveSources,
        ..Injection::NONE
    });
    let sources = sources_in(tree.as_ref());
    assert_eq!(
        rebatched,
        nodes + sources,
        "a rebatcher above each of the {sources} sources added {} nodes",
        rebatched - nodes
    );

    // A drained lane is live and reads nothing; its row groups went to its neighbour, so
    // the pulls are the same count from one lane fewer.
    let (_, drained_pulls, drained_first) = run(Injection {
        drain: Drain::FirstLane,
        ..Injection::NONE
    });
    assert_eq!(drained_first, 0, "the drained lane read {drained_first} times");
    assert_eq!(
        drained_pulls, pulls,
        "draining a lane changed how many batches were read, so rows moved rather than \
         being re-lanes"
    );

    // An empty batch is a call that produced no rows, so the sources are pulled more often
    // for the same rows.
    let (_, empty_pulls, _) = run(Injection {
        empties: Empties::Sometimes(50),
        ..Injection::NONE
    });
    assert!(
        empty_pulls > pulls,
        "the empty-batch setting fired on none of the {pulls} pulls"
    );

    fn sources_in(node: &dyn peacockdb_core::batch_partitioned::GpuNode) -> usize {
        usize::from(matches!(as_node_ref(node), NodeRef::LoadParquet(_)))
            + node
                .children()
                .into_iter()
                .map(sources_in)
                .sum::<usize>()
    }
}

/// The one plan shape the degenerate hash is not run against, and why it is a refusal
/// rather than a defect.
///
/// A hash that puts every key in lane 0 leaves every other lane's build side empty, and
/// Right, Full and RightAnti answer an empty build side with their probe side — a call
/// over a build table that does not exist ([#175](../../llm-wiki/tickets.md#t175)). The
/// candidate set drops the dimension for those plans, so this is what says the drop is a
/// refusal the engine makes rather than a shape nobody tried.
#[tokio::test]
async fn a_degenerate_hash_under_a_right_outer_is_refused_by_name() {
    use common::injection::{Hash, planned_mode};

    let data_dir = data_dir_for("tpcds", "1");
    let sql =
        std::fs::read_to_string(queries_dir_for("tpcds").join("q93.sql")).expect("the query text");
    let mode = &BP_MODES[2];
    let name = mode.name;
    let ctx = peacockdb_core::register_tables_for(
        peacockdb_core::build_session_state(mode.target_partitions),
        &data_dir,
    )
    .await
    .expect("register the tables");
    let plan = ctx
        .sql(&sql)
        .await
        .expect("the query plans")
        .create_physical_plan()
        .await
        .expect("the query has a physical plan");
    let (tree, _memory) = plan_batch_partitioned(&plan, mode.knobs())
        .unwrap_or_else(|error| panic!("q93 at {name}: {error}"));
    assert!(
        planned_mode(name, tree.as_ref()).owes_probe_when_empty,
        "q93 is here because its Right outer owes its probe side, and this plan has none"
    );

    let injected = apply(
        tree.as_ref(),
        Injection {
            hash: Hash::Degenerate,
            ..Injection::NONE
        },
        SEED,
    );
    let context = InjectedContext::new(
        ctx.task_ctx(),
        Injection {
            hash: Hash::Degenerate,
            ..Injection::NONE
        },
        SEED,
    );
    match batch_partitioned_driver::<Injected>(injected.as_ref(), &context, None) {
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("#175") && message.contains("build side is empty"),
                "the refusal names neither the ticket nor the cause: {message}"
            );
        }
        Ok(_) => panic!("a lane with no build side should have been refused"),
    }
}

/// The hole the row encoding leaves, and the assert that covers it.
///
/// Two batches of the same values under different column names encode identically — names
/// are not an input to the encoding — so the rendered fallback, which does carry them in
/// its header, is never reached on a names-only divergence. Constructed rather than read
/// off a query: no query answers with the wrong names today, which is exactly why the
/// substitution could weaken this without anything going red.
#[test]
fn an_answer_under_the_wrong_column_names_is_not_the_same_answer() {
    use datafusion::arrow::array::{ArrayRef, Int64Array};

    let values: ArrayRef = std::sync::Arc::new(Int64Array::from(vec![1i64, 2, 3]));
    let expected = RecordBatch::try_from_iter_with_nullable([("sum(v)", values.clone(), true)])
        .expect("the oracle's answer");
    let renamed = RecordBatch::try_from_iter_with_nullable([("v", values, true)])
        .expect("the same values, misnamed");

    assert_eq!(
        sorted_rows(std::slice::from_ref(&expected)),
        sorted_rows(std::slice::from_ref(&renamed)),
        "the encoding is supposed to be blind to names, and this case exists because it is"
    );
    assert_ne!(
        columns_of(std::slice::from_ref(&expected)),
        columns_of(std::slice::from_ref(&renamed)),
        "the column check does not see a rename, so nothing does"
    );
}

/// The injected set is a list, and four of its entries are the only carriers of a shape.
///
/// `end_to_end!` and the injected form differ by one word, so a query leaving the set
/// leaves it silently: `bp_tpcds_q33` would still pass, the tier's test count would not
/// move, and the four-lane interleave — the one operator whose correctness IS a lane
/// correspondence — would stop being injected at all. The list generates the fixtures, so
/// leaving the set means leaving the list, and this reads the list.
#[test]
fn the_injected_set_keeps_the_shapes_only_one_query_has() {
    for carrier in [
        "tpcds/q33",
        "tpch/nested_loop_join",
        "tpch/nested_loop_left_join",
        "tpch/nested_limits",
    ] {
        assert!(
            INJECTED.contains(&carrier),
            "{carrier} carries a shape no other injected query has: {INJECTED:?}"
        );
    }
    assert_eq!(
        INJECTED.len(),
        11,
        "the injected set is eleven queries: {INJECTED:?}"
    );
}

/// A query planned at one mode, so the runs a boundary search takes share one plan.
struct PlannedQuery {
    ctx: SessionContext,
    tree: Box<dyn GpuNode>,
    /// Read off the tree, since the small-table rule can leave a query at one lane whatever
    /// `target_partitions` asked for — and a one-lane query is one a drain cannot reach.
    lanes: usize,
}

impl PlannedQuery {
    async fn plan(dataset: &str, query: &str, mode: &BpMode) -> Self {
        let name = mode.name;
        let sql = std::fs::read_to_string(queries_dir_for(dataset).join(format!("{query}.sql")))
            .expect("the query text");
        let ctx = peacockdb_core::register_tables_for(
            peacockdb_core::build_session_state(mode.target_partitions),
            &data_dir_for(dataset, "1"),
        )
        .await
        .expect("register the tables");
        let plan = ctx
            .sql(&sql)
            .await
            .expect("the query plans")
            .create_physical_plan()
            .await
            .expect("the query has a physical plan");
        let (tree, _memory) = plan_batch_partitioned(&plan, mode.knobs())
            .unwrap_or_else(|error| panic!("{query} at {name}: {error}"));
        let lanes = planned_mode(name, tree.as_ref()).lanes;
        Self { ctx, tree, lanes }
    }

    fn run(&self, injection: Injection, budget: Option<usize>) -> Result<RunReport, RunError> {
        let injected = apply(self.tree.as_ref(), injection, SEED);
        let context = InjectedContext::new(self.ctx.task_ctx(), injection, SEED);
        batch_partitioned_driver::<Injected>(injected.as_ref(), &context, budget)
    }

    /// The smallest budget a shape completes at, and the peak it was seen holding. Searched
    /// from that peak rather than taken from it: a pre-call check tests the MODELLED
    /// transient, which can exceed anything the run was seen holding, so the peak is a
    /// floor rather than the answer.
    fn boundary(&self, injection: Injection) -> (usize, usize) {
        let peak = self
            .run(injection, None)
            .expect("the unbudgeted run finishes")
            .peak_bytes;
        let (mut low, mut high) = (peak, peak * 8);
        assert!(
            self.run(injection, Some(high)).is_ok(),
            "{} does not run at eight times its peak",
            injection.label()
        );
        while low + 1 < high {
            let middle = low + (high - low) / 2;
            if self.run(injection, Some(middle)).is_ok() {
                high = middle
            } else {
                low = middle
            }
        }
        (peak, high)
    }
}

/// The `(peak, smallest fitting budget)` pair of one query as planned and under one
/// injected shape. Asserts what both callers claim: the shape moved what the query holds,
/// and the injected plan's budget is a boundary — it completes at that budget and trips a
/// byte below, at the phase and the node named here. Whether the budget moved with the
/// peak is the caller's claim rather than this one's, so the pairs are returned.
fn boundaries_under(
    planned: &PlannedQuery,
    injection: Injection,
    trip: (When, &str),
) -> ((usize, usize), (usize, usize)) {
    let (as_planned, injected) = (
        planned.boundary(Injection::NONE),
        planned.boundary(injection),
    );
    assert_ne!(
        as_planned.0,
        injected.0,
        "{} did not move what the query holds, so this proves nothing about the accounting \
         under injection",
        injection.label()
    );
    match planned.run(injection, Some(injected.1 - 1)) {
        Err(RunError::BudgetExceeded { when, message }) => {
            assert_eq!(when, trip.0, "{message}");
            assert!(
                message.contains(trip.1),
                "the failure names the node it happened at: {message}"
            );
        }
        other => panic!("a byte below {} should not fit, got {other:?}", injected.1),
    }
    (as_planned, injected)
}

/// The accounting under an injected shape, which is the half no other case reaches.
///
/// Every other injected run passes no budget, so the accountant watches rather than
/// enforces, and the two rewrites reach it by different halves: a drained lane moves row
/// groups between lanes, a rebatcher moves the batch sizes the accountant prices.
/// `q16` at four lanes peaks at 104.7 MB as planned and 77.9 MB with lane 0 drained, and
/// its budget follows, 131.5 MB against 104.7 MB. `nested-loop-join` at bp-tp4-rowgroup
/// peaks at 8,222 bytes and 8,540 under a rebatcher, and its budget does not move: the
/// pre-call check tests the join's own transient, which merging a lane's batches leaves
/// alone.
#[tokio::test]
async fn an_injected_shape_moves_what_the_query_holds() {
    // Batching off, so the small-table rule leaves the sources at four lanes and there is a
    // lane whose row groups can move.
    let q16 = PlannedQuery::plan("tpcds", "q16", &BP_MODES[2]).await;
    let (as_planned, drained) = boundaries_under(
        &q16,
        Injection {
            drain: Drain::FirstLane,
            ..Injection::NONE
        },
        (When::PreCall, "GpuEmitPartitions"),
    );
    assert_ne!(
        as_planned.1, drained.1,
        "the peak moved and the budget it needs did not"
    );

    // The drain's complement: a query whose residency only a rebatcher reaches, because a
    // lane it could move row groups out of is what this one does not have.
    let nested_loop = PlannedQuery::plan("tpch", "nested-loop-join", &BP_MODES[3]).await;
    assert_eq!(
        nested_loop.lanes, 1,
        "nested-loop-join at {} planned more than one lane, so a drain reaches it too",
        BP_MODES[3].name
    );
    let (as_planned, rebatched) = boundaries_under(
        &nested_loop,
        Injection {
            rebatch: Rebatch::AboveSources,
            ..Injection::NONE
        },
        (When::PreCall, "GpuNestedLoopJoin"),
    );
    assert_eq!(
        as_planned.1, rebatched.1,
        "the budget this query needs is its join's transient, which a rebatcher above the \
         sources does not reach"
    );
}
