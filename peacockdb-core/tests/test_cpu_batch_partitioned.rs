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
// OFFSETs in either corpus.
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
