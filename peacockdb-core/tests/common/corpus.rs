//! One corpus query at one mode: plan it, run it, and hold the answer to the oracle.
//!
//! What a `corpus_query!` case does, minus the golden — the two corpus binaries define the
//! macro and this is the body both of their cases reach. The declaration list is
//! [`corpus_cases.inc`](corpus_cases.inc), included by each, so the two engines' coverage
//! is read off one line per query rather than two lists that can disagree.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use datafusion::arrow::array::RecordBatch;
use datafusion::execution::context::SessionContext;
use peacockdb_core::batch_partitioned::cpu_backend::backend::CpuBackend;
use peacockdb_core::batch_partitioned::driver::{RunReport, batch_partitioned_driver};
use peacockdb_core::batch_partitioned::plan::plan_batch_partitioned;
use peacockdb_core::batch_partitioned::plan_text::render_run;
use peacockdb_core::batch_partitioned::{GpuNode, validate};

use super::result_text::ResultDigest;
use super::bp_mode::{BP_MODES, BpMode, mode_named};
use super::cost_model::CostModel;
use super::exec_mode::{CpuOracle, cpu_oracle_mode};
use super::{
    RESULT_GOLDEN_MAX_BYTES, assert_results_match, batches_to_sorted_str, corpus_golden,
    data_dir_for, queries_dir_for, registry, result_text, total_rows,
};

/// A query planned and run at one mode, with everything a caller needs to check it.
pub struct CpuRun {
    pub tree: Box<dyn GpuNode>,
    pub report: RunReport,
    pub batches: Vec<RecordBatch>,
}

/// Plan `query` at `mode`. Panics naming the query and the mode: a corpus case's whole
/// context is those two, and a bare planner error names neither. The session comes back
/// with the tree because both engines need it — the cpu backend runs against its task
/// context and the device's oracle runs against the same session.
pub async fn plan_at(
    dataset: &str,
    sf: &str,
    query: &str,
    mode: &BpMode,
) -> (SessionContext, Box<dyn GpuNode>) {
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
    (ctx, tree)
}

/// Plan and run on the CPU backend, with the two accounting assertions every run makes.
pub async fn run_cpu(dataset: &str, sf: &str, query: &str, mode: &BpMode) -> CpuRun {
    let what = format!("{dataset}/{query} at {}", mode.name);
    let (ctx, tree) = plan_at(dataset, sf, query, mode).await;
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
    match cpu_oracle_mode(oracle) {
        CpuOracle::DataFusionSubset => {
            assert_subset_of_unlimited(dataset, sf, query, batches, &what).await
        }
        CpuOracle::DataFusionExact => {
            let expected = oracle_digest(dataset, sf, query, &what).await;
            let actual = result_text::digest_of(batches);
            if expected != actual {
                // The rows are fetched only to say HOW they differ, which is the one path
                // with any use for them.
                let rows = oracle_rows(dataset, sf, query, &what).await;
                panic!(
                    "result for {what} differs from oracle ({} rows against {})\n{}",
                    actual.rows(),
                    expected.rows(),
                    result_text::first_difference(&rows, batches)
                );
            }
        }
        // The tolerance arm indexes the rows and cannot work from a digest, so its oracle is
        // run per mode. `shuffle-stddev` is what declares it: the Welford merge reassociates
        // its sums and drifts a few ULP from a single-partition pass.
        oracle => {
            let expected = oracle_rows(dataset, sf, query, &what).await;
            assert_results_match(&expected, batches, oracle.rel_tol(), &what);
        }
    }
}

/// The oracle's answer as its digest, once per query rather than once per (query, mode).
/// Five modes are held to one answer, so running it five times is four runs of the most
/// expensive thing this tier does.
///
/// The DIGEST is what is held, not the answer: eight bytes a row, so a hundred queries cost
/// what one large one would have. Holding the rows would put `anti-join`'s 240 MB back in
/// memory, which is what this comparator was rewritten to stop doing.
async fn oracle_digest(dataset: &str, sf: &str, query: &str, what: &str) -> ResultDigest {
    static DIGESTS: OnceLock<Mutex<HashMap<String, ResultDigest>>> = OnceLock::new();
    let digests = DIGESTS.get_or_init(|| Mutex::new(HashMap::new()));
    let key = format!("{dataset}/{sf}/{query}");
    if let Some(held) = digests.lock().expect("the oracle cache").get(&key) {
        return held.clone();
    }
    let digest = result_text::digest_of(&oracle_rows(dataset, sf, query, what).await);
    digests
        .lock()
        .expect("the oracle cache")
        .insert(key, digest.clone());
    digest
}

async fn oracle_rows(dataset: &str, sf: &str, query: &str, what: &str) -> Vec<RecordBatch> {
    let ctx = session_for(dataset, sf, 1).await;
    collect(&ctx, &query_text(dataset, query), what).await
}

/// What an unordered `LIMIT` does determine, asked of the same session: how many rows come
/// back, and that each was in the unlimited answer.
///
/// Neither question collects that answer. The count is a `count(*)` over the same body, and
/// the containment streams the body past a map of the rows that came back — six million
/// rows for `scan-limit`, which collected would be an OOM on a 15 GiB host rather than a
/// slow test.
async fn assert_subset_of_unlimited(
    dataset: &str,
    sf: &str,
    query: &str,
    batches: &[RecordBatch],
    what: &str,
) {
    let (body, skip, fetch) = without_its_limit(&query_text(dataset, query), what);
    let ctx = session_for(dataset, sf, 1).await;
    let available = count_of(&ctx, &body, what).await;
    assert_eq!(
        total_rows(batches) as u64,
        wanted_rows(available, skip, fetch),
        "{what}: an unordered limit does not fix which rows, but it fixes how many"
    );
    assert_contained_in(&ctx, &body, batches, what).await;
}

/// `max(0, min(n, |unlimited| - m))` for `LIMIT n OFFSET m`: an offset past the end returns
/// nothing, and a limit past what is left returns what is left.
pub fn wanted_rows(available: u64, skip: u64, fetch: Option<u64>) -> u64 {
    let after_skip = available.saturating_sub(skip);
    match fetch {
        Some(n) => n.min(after_skip),
        None => after_skip,
    }
}

/// Every returned row was in the unlimited answer, counted as a MULTISET: set membership
/// passes a run that returned one row twice where the oracle holds it once, which a limit
/// over a join can produce. The unlimited side is streamed and never held.
async fn assert_contained_in(
    ctx: &SessionContext,
    body: &str,
    batches: &[RecordBatch],
    what: &str,
) {
    use futures::StreamExt;

    let mut owed = owed_rows(batches);
    let mut held_at_all: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut stream = ctx
        .sql(body)
        .await
        .unwrap_or_else(|e| panic!("{what}: the oracle does not plan the unlimited body: {e}"))
        .execute_stream()
        .await
        .unwrap_or_else(|e| panic!("{what}: the oracle does not run the unlimited body: {e}"));
    while !owed.is_empty() {
        let Some(batch) = stream.next().await else {
            break;
        };
        let batch = batch.unwrap_or_else(|e| panic!("{what}: the oracle stream failed: {e}"));
        take_rows(&mut owed, std::slice::from_ref(&batch), &mut held_at_all);
    }
    // The two failures are different bugs and the second is the likelier: a row the
    // unlimited answer does not hold at all is a wrong row, where one held too few times is
    // a duplicated one.
    let absent: Vec<&String> = owed
        .keys()
        .filter(|row| !held_at_all.contains(*row))
        .collect();
    assert!(
        owed.is_empty(),
        "{what}: {} row(s) the unlimited answer does not account for. {} of them are not in \
         it at all, the first being:\n{}",
        owed.values().sum::<usize>(),
        absent.len(),
        absent
            .first()
            .copied()
            .unwrap_or_else(|| owed.keys().next().expect("a row"))
    );
}

/// The rows a run returned, counted — what the containment check owes the oracle.
pub fn owed_rows(batches: &[RecordBatch]) -> HashMap<String, usize> {
    let mut owed: HashMap<String, usize> = HashMap::new();
    for row in rows_of(batches) {
        *owed.entry(row).or_insert(0) += 1;
    }
    owed
}

/// Strike off what this slice of the unlimited answer accounts for, and record which owed
/// rows it holds at all — the two failures a leftover can mean are told apart by that.
pub fn take_rows(
    owed: &mut HashMap<String, usize>,
    batches: &[RecordBatch],
    held_at_all: &mut std::collections::HashSet<String>,
) {
    for row in rows_of(batches) {
        if let Some(count) = owed.get_mut(&row) {
            held_at_all.insert(row.clone());
            *count -= 1;
            if *count == 0 {
                owed.remove(&row);
            }
        }
    }
}

async fn count_of(ctx: &SessionContext, body: &str, what: &str) -> u64 {
    let counted = collect(ctx, &format!("SELECT count(*) FROM ({body})"), what).await;
    let column = counted
        .first()
        .expect("a count returns a row")
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("count(*) is Int64");
    column.value(0) as u64
}

/// The rows a comparison sees: unpadded, one string per row. NOT the golden's rendering —
/// padding is a function of the whole answer, so the ten rows a limit returned and the
/// millions the oracle streams past them would render the same row differently and nothing
/// would ever be struck off. Its header and borders are not rows either.
fn rows_of(batches: &[RecordBatch]) -> Vec<String> {
    result_text::rendered_rows(batches)
}

/// The query without its trailing `LIMIT n [OFFSET m]`, and the interval it carried. The
/// last `limit` in the text, since a query can hold an inner one; anything else after it
/// panics rather than being trimmed, so a query declaring this oracle without an unordered
/// limit fails loudly instead of being compared against itself.
pub fn without_its_limit(sql: &str, what: &str) -> (String, u64, Option<u64>) {
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
    if !words
        .next()
        .is_some_and(|word| word.eq_ignore_ascii_case("limit"))
    {
        bad();
    }
    let fetch: u64 = words
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| bad());
    let skip = match words.next() {
        None => 0,
        Some(word) if word.eq_ignore_ascii_case("offset") => words
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| bad()),
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

/// The whole of a cpu corpus case: plan, run, answer, and the three goldens. `mode` is the
/// macro's ident spelling, decoded here rather than at the call site.
///
/// The oracle comparison comes first and runs whether or not this is a regenerating run —
/// a wrong answer must never reach a golden, and freezing one is the only way this tier
/// could record something no later run would question.
pub async fn cpu_case(dataset: &str, sf: &str, query: &str, mode: &str, cpu_oracle: &str) {
    let mode = mode_named(mode);
    let what = format!("{dataset}/{query} at {}", mode.name);
    let run = run_cpu(dataset, sf, query, mode).await;
    assert_answer(dataset, sf, query, mode, cpu_oracle, &run.batches).await;

    let column = cpu_column(mode);
    let cpu_text = render_run(run.tree.as_ref(), &run.report);
    corpus_golden::assert_or_merge(
        &corpus_golden::cpu_golden(dataset, sf, mode.name),
        dataset,
        sf,
        &[&column],
        query,
        &cpu_text,
    );
    // Derived from the text just written rather than from the report it came from, so the
    // cost is a function of the golden exactly as `test_cost_model` re-derives it.
    let cost = CostModel::load().cost_text_from_cpu(&cpu_text, &what);
    corpus_golden::assert_or_merge(
        &corpus_golden::cost_golden(dataset, sf, mode.name),
        dataset,
        sf,
        &[&column],
        query,
        &cost,
    );
    if authoritative_mode(dataset, sf, query).is_some_and(|author| author.name == mode.name) {
        assert_result_section(dataset, sf, query, mode, &run.batches);
    }
}

/// The `bp_cpu_` column this mode's cells live in.
fn cpu_column(mode: &BpMode) -> String {
    format!("bp_cpu_{}", mode.ident().trim_start_matches("bp_"))
}

/// Which mode authors `.result.txt`: the last mode the query DECLARES, in the fixed
/// sequence of five. Its authority is a property of the declaration and not of what
/// happened to run, which is what keeps the one golden with no mode in its key well defined
/// under a filtered regeneration — a run without the authority leaves the section alone.
pub fn authoritative_mode(dataset: &str, sf: &str, query: &str) -> Option<&'static BpMode> {
    let rows = registry::load_csv();
    let row = rows
        .iter()
        .find(|row| row.dataset == dataset && row.sf == sf && registry::stem(&row.query) == query)?;
    BP_MODES.iter().rev().find(|mode| {
        row.states
            .get(&cpu_column(mode))
            .is_some_and(|state| state == "enabled" || state == "skip")
    })
}

/// Why a result has no section, and which mode decided it. The size is named where it is
/// known — where the lower bound tripped, the run stopped counting on purpose and says so
/// rather than finishing the sum to put a number in a marker.
///
/// The marker keeps first position and `mode=` follows it: `corpus_gpu` reads a leading
/// SKIPPED as "this section holds no rows", so a mode line ahead of it would let a
/// `golden_exact` declaration pass against a section with nothing to compare.
pub fn over_cap(bytes: Option<usize>, mode: &BpMode) -> String {
    let size = match bytes {
        Some(bytes) => format!("is {bytes} bytes, at or above"),
        None => "is at or above".to_string(),
    };
    format!(
        "{}the result {size} the {RESULT_GOLDEN_MAX_BYTES}-byte cap\nmode={}\n",
        corpus_golden::SKIPPED,
        mode.name
    )
}

/// The one entry this query has, and the mode that wrote it. A result at or above the cap
/// keeps its section and says why rather than being deleted: absent and not-applicable read
/// alike, and only one of them is a regression.
fn assert_result_section(
    dataset: &str,
    sf: &str,
    query: &str,
    mode: &BpMode,
    batches: &[RecordBatch],
) {
    // Sized before it is built, and the cells alone are a lower bound on the table: an
    // answer far above the cap costs one row of memory rather than being rendered whole,
    // sorted, and then discarded for being too large. Under the bound it is rendered once
    // and measured exactly, so the cap still means the same bytes it always did.
    let body = match result_text::exceeds_rendered_size(batches, RESULT_GOLDEN_MAX_BYTES) {
        true => over_cap(None, mode),
        false => {
            let rendered = batches_to_sorted_str(batches);
            match rendered.len() >= RESULT_GOLDEN_MAX_BYTES {
                true => over_cap(Some(rendered.len()), mode),
                false => format!("mode={}\n{rendered}\n", mode.name),
            }
        }
    };
    let columns: Vec<String> = BP_MODES.iter().map(cpu_column).collect();
    let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
    corpus_golden::assert_or_merge(
        &corpus_golden::result_golden(dataset, sf),
        dataset,
        sf,
        &columns,
        query,
        &body,
    );
}
