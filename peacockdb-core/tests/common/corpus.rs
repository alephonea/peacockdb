//! One corpus query at one mode: plan it, run it, and hold the answer to the oracle.
//!
//! What a `corpus_query!` case does, minus the golden — the two corpus binaries define the
//! macro and this is the body both of their cases reach. The declaration list is
//! [`corpus_cases.inc`](corpus_cases.inc), included by each, so the two engines' coverage
//! is read off one line per query rather than two lists that can disagree.

use datafusion::arrow::array::RecordBatch;
use datafusion::execution::context::SessionContext;
use peacockdb_core::batch_partitioned::cpu_backend::backend::CpuBackend;
use peacockdb_core::batch_partitioned::driver::{RunReport, batch_partitioned_driver};
use peacockdb_core::batch_partitioned::plan::plan_batch_partitioned;
use peacockdb_core::batch_partitioned::{GpuNode, validate};

use super::bp_mode::{BpMode, mode_named};
use super::exec_mode::{CpuOracle, cpu_oracle_mode};
use super::{
    assert_results_match, batches_to_sorted_str, data_dir_for, queries_dir_for, total_rows,
};

/// A query planned and run at one mode, with everything a caller needs to check it.
pub struct CpuRun {
    pub tree: Box<dyn GpuNode>,
    pub report: RunReport,
    pub batches: Vec<RecordBatch>,
}

/// Plan `query` at `mode` and run it on the CPU backend. Panics naming the query and the
/// mode: a corpus case's whole context is those two, and a bare planner error names
/// neither.
pub async fn run_cpu(dataset: &str, sf: &str, query: &str, mode: &BpMode) -> CpuRun {
    let what = format!("{dataset}/{query} at {}", mode.name);
    let ctx = session_for(dataset, sf, mode.target_partitions).await;
    let plan = ctx
        .sql(&query_text(dataset, query))
        .await
        .unwrap_or_else(|e| panic!("{what}: the query does not plan: {e}"))
        .create_physical_plan()
        .await
        .unwrap_or_else(|e| panic!("{what}: no physical plan: {e}"));
    let (tree, _memory) = plan_batch_partitioned(&plan, mode.knobs())
        .unwrap_or_else(|e| panic!("{what}: this mode refuses it: {e}"));
    // The planner's own check, made again: the driver asks only for canonical form, so a
    // tree that met neither would run and answer.
    validate::validate(tree.as_ref()).unwrap_or_else(|e| panic!("{what} is not a plan: {e}"));
    let report = batch_partitioned_driver::<CpuBackend>(tree.as_ref(), &ctx.task_ctx(), None)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let batches = report
        .batches
        .iter()
        .map(|batch| batch.record_batch().clone())
        .collect();
    assert_eq!(report.in_flight_bytes, 0, "{what} ended holding batches");
    assert_eq!(
        report.holds, report.releases,
        "{what} held {} batches and released {}",
        report.holds, report.releases
    );
    CpuRun {
        tree,
        report,
        batches,
    }
}

/// The answer against plain DataFusion at `target_partitions = 1`, asked whichever of the
/// three ways this query's declaration names. Runs on every case, regenerating or not: a
/// wrong answer must not reach a golden, and this is the check that stops it.
pub async fn assert_answer(
    dataset: &str,
    sf: &str,
    query: &str,
    mode: &BpMode,
    oracle: &str,
    batches: &[RecordBatch],
) {
    let what = format!("{dataset}/{query} at {}", mode.name);
    let oracle = cpu_oracle_mode(oracle);
    let sql = query_text(dataset, query);
    let ctx = session_for(dataset, sf, 1).await;
    match oracle {
        CpuOracle::DataFusionSubset => {
            let (unlimited, skip, fetch) = without_its_limit(&sql, &what);
            let whole = collect(&ctx, &unlimited, &what).await;
            assert_subset_of(batches, &whole, skip, fetch, &what);
        }
        _ => {
            let expected = collect(&ctx, &sql, &what).await;
            assert_results_match(&expected, batches, oracle.rel_tol(), &what);
        }
    }
}

/// What an unordered `LIMIT` does determine: how many rows come back, and that each was in
/// the unlimited answer. Compared as a multiset — set membership passes a run that returned
/// one row twice where the oracle has it once, which a limit over a join can produce.
fn assert_subset_of(
    batches: &[RecordBatch],
    whole: &[RecordBatch],
    skip: u64,
    fetch: Option<u64>,
    what: &str,
) {
    let available = total_rows(whole) as u64;
    let wanted = match fetch {
        Some(n) => n.min(available.saturating_sub(skip)),
        None => available.saturating_sub(skip),
    };
    assert_eq!(
        total_rows(batches) as u64,
        wanted,
        "{what}: an unordered limit does not fix which rows, but it fixes how many"
    );
    let mut held = counted(whole);
    for row in rows_of(batches) {
        let count = held.get_mut(&row).unwrap_or_else(|| {
            panic!("{what}: a row came back that the unlimited answer does not hold:\n{row}")
        });
        assert!(
            *count > 0,
            "{what}: a row came back more often than the unlimited answer holds it:\n{row}"
        );
        *count -= 1;
    }
}

fn counted(batches: &[RecordBatch]) -> std::collections::HashMap<String, usize> {
    let mut counts = std::collections::HashMap::new();
    for row in rows_of(batches) {
        *counts.entry(row).or_insert(0) += 1;
    }
    counts
}

/// One rendered line per row, which is what makes a multiset comparison cheap and its
/// failure legible. The same rendering the result goldens use, minus its header.
fn rows_of(batches: &[RecordBatch]) -> Vec<String> {
    batches_to_sorted_str(batches)
        .lines()
        .map(str::to_string)
        .collect()
}

/// The query without its trailing `LIMIT n [OFFSET m]`, and the interval it carried. The
/// last `limit` in the text, since a query can hold an inner one; anything else after it
/// panics rather than being trimmed, so a query declaring this oracle without an unordered
/// limit fails loudly instead of being compared against itself.
fn without_its_limit(sql: &str, what: &str) -> (String, u64, Option<u64>) {
    let body = sql.trim().trim_end_matches(';');
    let at = body
        .to_ascii_lowercase()
        .rfind("limit")
        .unwrap_or_else(|| panic!("{what}: declared data_fusion_subset and has no limit"));
    let mut words = body[at..].split_whitespace();
    let bad = || -> ! {
        panic!(
            "{what}: declared data_fusion_subset and its tail is not `LIMIT n [OFFSET m]`: {}",
            &body[at..]
        )
    };
    if !words.next().is_some_and(|word| word.eq_ignore_ascii_case("limit")) {
        bad();
    }
    let fetch: u64 = words.next().and_then(|n| n.parse().ok()).unwrap_or_else(|| bad());
    let skip = match words.next() {
        None => 0,
        Some(word) if word.eq_ignore_ascii_case("offset") => {
            words.next().and_then(|n| n.parse().ok()).unwrap_or_else(|| bad())
        }
        Some(_) => bad(),
    };
    if words.next().is_some() {
        bad();
    }
    (body[..at].to_string(), skip, Some(fetch))
}

async fn collect(ctx: &SessionContext, sql: &str, what: &str) -> Vec<RecordBatch> {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|e| panic!("{what}: the oracle does not plan it: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("{what}: the oracle does not run it: {e}"))
}

async fn session_for(dataset: &str, sf: &str, target_partitions: usize) -> SessionContext {
    peacockdb_core::register_tables_for(
        peacockdb_core::build_session_state(target_partitions),
        &data_dir_for(dataset, sf),
    )
    .await
    .expect("register the tables")
}

fn query_text(dataset: &str, query: &str) -> String {
    let path = queries_dir_for(dataset).join(format!("{query}.sql"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("query file not found: {}", path.display()))
}

/// The whole of a cpu corpus case as it stands before the goldens arrive: plan, run,
/// answer. `mode` is the macro's ident spelling, decoded here rather than at the call site.
pub async fn cpu_case(dataset: &str, sf: &str, query: &str, mode: &str, cpu_oracle: &str) {
    let mode = mode_named(mode);
    let run = run_cpu(dataset, sf, query, mode).await;
    assert_answer(dataset, sf, query, mode, cpu_oracle, &run.batches).await;
}
