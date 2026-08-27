//! SQL in, rows out: the batch-partitioned mode end to end on the CPU backend.
//!
//! Every other test of this mode proves one layer against a fixture of the last one's
//! shape — a recipe against a plan, an executor against a recipe, a driver against a mock.
//! This one starts at a query's text and ends at its rows, so what it tests is the join
//! between the pieces.
//!
//! The oracle is DataFusion on the same SQL, never the legacy CPU executor: a second
//! engine of ours agrees with us wherever we are consistently wrong, and the finalize
//! expression this mode evaluates is the one it also sends to a device. DataFusion is the
//! implementation in reach that decomposed the aggregate differently.
//!
//! Each query runs at every mode below rather than one, because a lane count and a batch
//! count are what this mode can get wrong: a plan re-planned at another shape must answer
//! the same rows or the shape is load-bearing, which it must not be.
//!
//! # What is varied here, and what is not
//!
//! Re-planned rather than edited, which is the prototype's own rule: a node's partitioning
//! is not a field, so a shape is reached by planning at it. What varies is the pair the
//! five modes carry — lanes at 1 and 4, and batching at one per lane, one per row group,
//! and the estimator's size. Every query answers the same rows at all five.
//!
//! Three of the prototype's modes are not here, and the omissions are not equal:
//!
//! - **`small_table_bytes` is one constant at every mode**, and it is the sharpest of the
//!   three: it decides whether a join co-partitions or collapses both sides onto one lane,
//!   which is a different PLAN, not a different batch shape. Varying it would re-plan every
//!   join in the list into its other form. That is the injection this path most wants and
//!   least has.
//! - **No rebatcher above the sources.** The batching modes move where batch boundaries
//!   fall, but every boundary here is one the planner chose; a node that re-cuts them
//!   underneath an operator is a shape no mode produces.
//! - **No zero-row batches.** An empty batch reaches the executors' own tests and the
//!   mock's, and no plan here emits one, so the operators are proved on it and this path
//!   is not.
//!
//! The entry calls the injector a model rather than a specification, so a shorter list is
//! allowed — but silence about which parts were dropped would leave a reader to assume the
//! coverage is the injector's. It is not: it is the mode axis, whole, and one knob of the
//! three.
mod common;

use datafusion::arrow::array::RecordBatch;

use peacockdb_core::batch_partitioned::cpu_backend::backend::CpuBackend;
use peacockdb_core::batch_partitioned::driver::batch_partitioned_driver;
use peacockdb_core::batch_partitioned::plan::{BatchSizing, PlanKnobs, plan_batch_partitioned};
use peacockdb_core::config::MemoryLimit;

use common::{assert_results_match, data_dir_for, queries_dir_for};

/// Where a Welford merge is the only divergence: this mode decomposes the aggregate into
/// an init, two merges and a finalize, and DataFusion computes it in one pass, so the last
/// digits differ by reassociation. The legacy GPU tier uses the same figure for the same
/// reason (`golden_approx_std`).
const WELFORD_TOLERANCE: f64 = 1e-11;

/// The tier the mode goldens are written at, so a failure here reads against a committed
/// plan rather than against a shape nothing else records.
const BUDGET: u64 = MemoryLimit::Mini.bytes() as u64;
const SMALL_TABLE_BYTES: u64 = 5 * 1024 * 1024;

/// The five shapes the plan goldens hold, which is the injection set: one lane and one
/// batch is the degenerate end, row-group granularity is the finest the mapping expresses,
/// and the sized mode is the only one a budget moves. A query answering differently at two
/// of these has a bug that no single shape would have shown.
const MODES: [(&str, usize, BatchSizing); 5] = [
    ("bp-tp1-single", 1, BatchSizing::OneBatchPerLane),
    ("bp-tp1-rowgroup", 1, BatchSizing::OneBatchPerRowGroup),
    ("bp-tp4-single", 4, BatchSizing::OneBatchPerLane),
    ("bp-tp4-rowgroup", 4, BatchSizing::OneBatchPerRowGroup),
    ("bp-tp4-sized", 4, BatchSizing::Budgeted),
];

fn knobs(target_partitions: usize, sizing: BatchSizing) -> PlanKnobs {
    PlanKnobs {
        target_partitions,
        sizing,
        budget: BUDGET,
        small_table_bytes: SMALL_TABLE_BYTES,
    }
}

/// One query at every mode, against DataFusion on the same text.
async fn answers_match_datafusion(dataset: &str, query: &str, tolerance: Option<f64>) {
    let sql_path = queries_dir_for(dataset).join(format!("{query}.sql"));
    let sql = std::fs::read_to_string(&sql_path)
        .unwrap_or_else(|_| panic!("query file not found: {}", sql_path.display()));
    sql_answers_match_datafusion(dataset, query, &sql, tolerance).await;
}

async fn sql_answers_match_datafusion(
    dataset: &str,
    query: &str,
    sql: &str,
    tolerance: Option<f64>,
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

    for (name, target_partitions, sizing) in MODES {
        let ctx = peacockdb_core::register_tables_for(
            peacockdb_core::build_session_state(target_partitions),
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
        let (tree, _memory) = plan_batch_partitioned(&plan, knobs(target_partitions, sizing))
            .unwrap_or_else(|error| panic!("{dataset}/{query} at {name}: {error}"));
        let report = batch_partitioned_driver::<CpuBackend>(tree.as_ref(), &ctx.task_ctx(), None)
            .unwrap_or_else(|error| panic!("{dataset}/{query} at {name}: {error}"));
        let actual: Vec<RecordBatch> = report
            .batches
            .into_iter()
            .map(|batch| batch.into_record_batch())
            .collect();
        assert_results_match(&expected, &actual, tolerance, &format!("{query} at {name}"));
        // Zero at the end of any correct run: a batch held and never released is a leak on
        // the CPU and a resident table on a device, and neither shows in the rows.
        assert_eq!(
            report.in_flight_bytes, 0,
            "{dataset}/{query} at {name} ended holding batches"
        );
        assert_eq!(
            report.holds, report.releases,
            "{dataset}/{query} at {name} held {} batches and released {}",
            report.holds, report.releases
        );
    }
}

/// `end_to_end!(dataset, query)` — one test per query, so a failure names it.
macro_rules! end_to_end {
    ($dataset:ident, $query:ident) => {
        end_to_end!($dataset, $query, None);
    };
    ($dataset:ident, $query:ident, $tolerance:expr) => {
        paste::paste! {
            #[tokio::test]
            async fn [<bp_ $dataset _ $query>]() {
                answers_match_datafusion(
                    stringify!($dataset),
                    &stringify!($query).replace('_', "-"),
                    $tolerance,
                )
                .await;
            }
        }
    };
}

// ── the join capability matrix ──────────────────────────────────────────────
// Chosen by cover over the matrix rather than by taste, off the mode goldens: every join
// type this mode claims, crossed with a residual filter, null_equals_null and multi-key.
// Eleven of the seventeen carry a cell no other query here does.
//
// nested-loop Inner, and the smallest plan in the corpus that carries a join at all.
end_to_end!(tpch, nested_loop_join);
// nested-loop Left: the one mode whose probe side is a single batch, so its call takes the
// build side rather than a copy of it.
end_to_end!(tpch, nested_loop_left_join);
// RightAnti — the probe-side semi family, answered per batch with no finish pass.
end_to_end!(tpch, anti_join);
// Left outer: the finish pass, and the only cell with no device path at all (#152).
end_to_end!(tpch, left_join);
// Inner, multi-key Inner, Inner with a residual filter, LeftSemi and RightSemi in one plan.
end_to_end!(tpch, q20);
// LeftAnti with a residual filter, which the matrix answers in one legacy call over a
// probe side the planner made single-batch.
end_to_end!(tpch, q21);
// Full outer: Right's per-batch call and Left's finish, in one node.
end_to_end!(tpcds, q97);
// LeftAnti, and a LeftSemi carrying a residual filter.
end_to_end!(tpcds, q16);
// LeftMark, which scatters a boolean into an all-false column.
end_to_end!(tpcds, q45);
// LeftSemi with null_equals_null — a set operation lowered to a join, where NULL = NULL.
end_to_end!(tpcds, q8);
// RightSemi with null_equals_null.
end_to_end!(tpcds, q38);
// RightAnti with null_equals_null.
end_to_end!(tpcds, q87);
// Right outer, and its keys are composite — the only multi-key outer in either corpus.
end_to_end!(tpcds, q93);

// ── the shapes a join cover does not reach ──────────────────────────────────
// A union executed as an INTERLEAVE at four lanes: output lane p is built from lane p of
// each branch, which is the whole claim and is invisible at one lane.
end_to_end!(tpcds, q33);
// A union that cannot interleave: its two unions carry branches that disagree on lane
// count, which is what makes them unions rather than interleaves, and the lanes above are
// their sum. q77 is the shape this claim was written for — 4+1+4 — and it is not here
// because it is refused rather than untested: one of its Right outers gets a lane whose
// build side is empty, and what that owes is its probe side padded, which takes a call the
// recipe does not publish ([#175](../../llm-wiki/tickets.md#t175)).
end_to_end!(tpcds, q2);
// Both row-interval lowerings on one root-to-leaf path — the root-adjacent one becoming
// the unload's skip/fetch and the mid-plan one a limit over the scan — and the only
// OFFSETs in either corpus. It is also where the matrix's cross join is covered: what
// connects the two intervals has to be one, so the cell has no query of its own here.
end_to_end!(tpch, nested_limits);
// A merge over state worth merging: the Welford init, both merges and the finalize
// project. Every other aggregate here merges a sum.
end_to_end!(tpch, shuffle_stddev, Some(WELFORD_TOLERANCE));

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
    for (name, target_partitions, sizing) in MODES {
        let ctx = peacockdb_core::register_tables_for(
            peacockdb_core::build_session_state(target_partitions),
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
        let (tree, _memory) = plan_batch_partitioned(&plan, knobs(target_partitions, sizing))
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
            report.early_exit,
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
    let (name, target_partitions, sizing) = MODES[3];
    let ctx = peacockdb_core::register_tables_for(
        peacockdb_core::build_session_state(target_partitions),
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
    let (tree, _memory) = plan_batch_partitioned(&plan, knobs(target_partitions, sizing))
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
        Err(peacockdb_core::batch_partitioned::RunError::BudgetExceeded { message, .. }) => {
            assert!(
                message.contains("Gpu") && message.contains("budget"),
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
    let (name, target_partitions, sizing) = MODES[3];
    let ctx = peacockdb_core::register_tables_for(
        peacockdb_core::build_session_state(target_partitions),
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
    let (tree, _memory) = plan_batch_partitioned(&plan, knobs(target_partitions, sizing))
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
