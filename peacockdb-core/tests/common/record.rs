//! The cost-model calibration record: one line per cuDF call.
//!
//! Built from the two halves of a measurement: the driver's call log says where a call was
//! and what went into it, the device's regions say what it cost and what came back. Neither
//! half is a row on its own.
//!
//! TSV, and the conditions of the run — timing mode, build profile, allocator — are in
//! the file's `#` heading rather than in every row: they are constant across a run, and
//! the heading is free text where the allocator's description keeps its commas.
//!
//! `hbm_bytes` is deliberately absent: it comes from Nsight and is joined in later on the
//! same tuple.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use peacockdb_core::batch_partitioned::driver::{Measured, Measurements, RunReport};
use peacockdb_core::batch_partitioned::executor::AbiCall;
use peacockdb_core::batch_partitioned::recipe::{RecipePlan, Seq};

/// Env var naming the file rows are APPENDED to. Unset ⇒ no record is written, which
/// is why every caller can emit unconditionally.
pub const RECORD_PATH_ENV: &str = "PEACOCK_RECORD_PATH";

/// One row per cuDF call, keyed by `(dataset, sf, query, plan node, recipe step, call)`.
///
/// Six coordinates because that is what identifies a call: one plan node publishes several
/// recipe steps, and a batched run drives each of them once per batch per lane. Fewer, and
/// a row cannot say what it measured.
///
/// `node_seq` is the POST-order position, which is what recipes are addressed by and what
/// `recipe_seq` lives in the same space as. The driver numbers nodes pre-order — its
/// `emitted[0]` is the root — so a writer walking the report has to translate, and a writer
/// that forgets produces two plausible numbers from two different orders. The only symptom
/// is a join against `--- recipes ---` that quietly matches the wrong node.
///
/// `lane` is the DRIVING lane: the one the node was called on, not the one it emits into.
/// The two differ at a scatter and at a cross-lane accumulator, and it is the driving one
/// that a call belongs to.
///
/// `run_index` is which of the case's executions the row came from — the seventh
/// coordinate, and the one that makes a row unique. It is derivable, and written anyway:
/// a repeat of `call_index` 0 marks where an execution ended, so every consumer could
/// count. `hbm.tsv` is such a consumer, and two implementations of one counting rule are
/// two chances to disagree about which execution a row belongs to.
///
/// The cost CATEGORY is deliberately not among them: it is a lookup in `cost_model.conf`,
/// which changes on its own, and a column would freeze the taxonomy as it stood when the
/// rows were written.
pub const COLUMNS: &[&str] = &[
    "dataset",
    "sf",
    "query",
    "mode",
    "node_seq",
    "node_type",
    "lane",
    "recipe_seq",
    "recipe_kind",
    "call_index",
    "run_index",
    "in_rows",
    "in_bytes",
    "out_rows",
    "out_bytes",
    "peacock_host_us",
    "cudf_host_us",
    "device_us",
];

/// What a row cannot be recovered from: which engine produced it, over what data, and
/// under what conditions.
///
/// The last three are constant across a run and go into the file's `#` heading rather
/// than into every row — see [`record_header`]. They are still part of this struct
/// because the heading is written from it.
pub struct RunMeta<'a> {
    pub dataset: &'a str,
    pub sf: &'a str,
    pub query: &'a str,
    /// The batch-partitioned planning mode, `bp-tp4-sized` and the like. The same query
    /// at two modes is a different plan and a different set of calls.
    pub mode: &'a str,
    pub timing_mode: &'a str,
    pub build_profile: &'a str,
    pub allocator: &'a str,
}
/// One row per call the run made, in the order the driver made them.
///
/// `nodes` is [`nodes_as_recorded`](peacockdb_core::batch_partitioned::driver::nodes_as_recorded):
/// each node's type and POST-order position, in the driver's pre-order — the order the
/// report is indexed by. The translation is the whole reason it is taken rather than
/// derived here; see the note on [`COLUMNS`].
///
/// A call the device did not measure still gets a row, with its measured fields empty. The
/// alternative is dropping it, and a record silently missing the calls nobody measured is a
/// record whose totals cannot be checked against the plan.
pub fn record_rows(
    nodes: &[(&str, usize)],
    report: &RunReport,
    measured: &Measurements,
    meta: &RunMeta<'_>,
    run_index: usize,
) -> Vec<String> {
    let mut rows = Vec::new();
    for (node, (node_type, post_order)) in nodes.iter().enumerate() {
        for (lane, calls) in report.abi_calls[node].iter().enumerate() {
            for made in calls.iter().filter_map(|made| made.recorded()) {
                for call in made {
                    let cost = measured.call(call.seq, call.call_index);
                    rows.push(row(meta, node_type, *post_order, lane, run_index, call, cost));
                }
            }
        }
    }
    rows
}

/// Each row carries ITS call's measurement, looked up by `(seq, call_index)`.
///
/// Not the driver call's total. A driver call can address several seqs — an aggregate
/// concatenates and then merges — and the device measured each of them separately. Handing
/// every row the total would report a merge that produced one row as having produced the
/// six its whole entry did.
fn row(
    meta: &RunMeta<'_>,
    node_type: &str,
    post_order: usize,
    lane: usize,
    run_index: usize,
    call: &AbiCall,
    cost: Option<Measured>,
) -> String {
    let empty = String::new();
    let or_empty = |value: Option<u64>| value.map_or(empty.clone(), |v| v.to_string());
    let measured = cost.filter(|cost| cost.regions > 0);
    [
        meta.dataset.to_string(),
        meta.sf.to_string(),
        meta.query.to_string(),
        meta.mode.to_string(),
        post_order.to_string(),
        node_type.to_string(),
        lane.to_string(),
        call.seq.to_string(),
        call.kind.to_string(),
        call.call_index.to_string(),
        run_index.to_string(),
        call.in_rows.to_string(),
        or_empty(call.in_bytes),
        or_empty(measured.map(|m| m.out_rows)),
        or_empty(measured.map(|m| m.out_bytes)),
        or_empty(measured.map(|m| m.host_setup_us)),
        or_empty(measured.map(|m| m.host_submit_us)),
        or_empty(measured.map(|m| m.device_us)),
    ]
    .join("\t")
}

/// What the plan declares, in the record's own coordinates: the seqs each node's recipe
/// publishes, indexed by the POST-order position rows carry in `node_seq`.
///
/// Read through the same two steps `--- recipes ---` renders from — `RecipePlan::get` at a
/// post-order position, then each call's target — so rows checked against this are rows
/// checked against that section. That the two readings really do line up is asserted on a
/// planned query in `test_batch_partitioned_plans`, where a plan can be built without a
/// device.
pub fn declared_steps(recipes: &RecipePlan) -> BTreeMap<usize, BTreeSet<Seq>> {
    (0..recipes.nodes())
        .map(|node| {
            let seqs = recipes.get(node).map(|recipe| recipe.seqs()).unwrap_or_default();
            (node, seqs.into_iter().collect())
        })
        .collect()
}

/// One execution's rows against what its plan declares.
///
/// Two statements, and the row count follows from them rather than being counted: every
/// row names a step its own node publishes, and the calls to a step are numbered `0..n`
/// with no gap. A run's rows are then exactly Σ over the declared steps of the calls made
/// to each — a total the plan predicts, not one the producer reports about itself.
///
/// It exists for the two ways this record can be wrong while looking right. A `node_seq`
/// taken from the driver's pre-order walk still names a node that exists and still pairs
/// with a seq that exists; only the PAIR is wrong, and a join against `--- recipes ---`
/// then matches the wrong line and says nothing. A dropped call leaves every remaining row
/// well formed and the totals merely smaller.
///
/// Rows of ONE execution, which it also checks: a case appends ten, and their
/// `call_index` sequences restart, so two executions handed here at once would read as a
/// step called twice. `run_index` says which execution a row is, and a mixed batch is
/// caught by name rather than surfacing as that.
pub fn rows_match_the_recipes(
    rows: &[String],
    declared: &BTreeMap<usize, BTreeSet<Seq>>,
) -> Result<(), String> {
    // Keyed by seq alone, as the driver's counter is: `call_index` counts a seq's calls
    // across the whole run, so a per-node tally would read a lane's share as a gap.
    let mut calls: BTreeMap<Seq, BTreeSet<u64>> = BTreeMap::new();
    let mut run: Option<u64> = None;
    for row in rows {
        let node = field(row, "node_seq")? as usize;
        let seq = field(row, "recipe_seq")? as Seq;
        let call = field(row, "call_index")?;
        let at = field(row, "run_index")?;
        match run {
            Some(first) if first != at => {
                return Err(format!("rows of executions {first} and {at} were checked together"));
            }
            _ => run = Some(at),
        }
        match declared.get(&node) {
            None => {
                return Err(format!(
                    "a row is at node {node}, and the plan has {} — {row:?}",
                    declared.len()
                ));
            }
            Some(seqs) if !seqs.contains(&seq) => {
                return Err(format!(
                    "a row pairs node {node} with step #{seq}, whose recipe publishes {seqs:?} \
                     — the pair is what a join against `--- recipes ---` reads, and both \
                     halves of a wrong one exist"
                ));
            }
            Some(_) => {}
        }
        if !calls.entry(seq).or_default().insert(call) {
            return Err(format!("step #{seq} has two rows for call {call} — {row:?}"));
        }
    }
    for (seq, made) in &calls {
        let expected: BTreeSet<u64> = (0..made.len() as u64).collect();
        if *made != expected {
            return Err(format!(
                "step #{seq} has {} rows numbered {made:?} — a gap is a call that was made \
                 and never written down",
                made.len()
            ));
        }
    }
    Ok(())
}

/// One numeric column of a row, by the name [`COLUMNS`] gives it — so the checker reads
/// the record the way a consumer does, by column name, rather than by a position it would
/// have to be kept in step with.
fn field(row: &str, column: &str) -> Result<u64, String> {
    let at = COLUMNS
        .iter()
        .position(|name| *name == column)
        .expect("a column this module names");
    let text = row
        .split('\t')
        .nth(at)
        .ok_or_else(|| format!("a row has no {column} column — {row:?}"))?;
    text.parse()
        .map_err(|_| format!("{column} is {text:?}, which is not a number — {row:?}"))
}

const RUN_PREFIX: &str = "# run: ";

/// The conditions this run measured under, as heading lines. Constant across a run —
/// which is why they are here and not columns — but each one changes what the
/// microseconds MEAN, so a file mixing two of them is a file whose rows cannot be
/// compared. [`append_records`] refuses to write one.
fn run_conditions(meta: &RunMeta<'_>) -> Vec<String> {
    vec![
        format!("{RUN_PREFIX}timing_mode={}", meta.timing_mode),
        format!("{RUN_PREFIX}build_profile={}", meta.build_profile),
        format!("{RUN_PREFIX}allocator={}", meta.allocator),
    ]
}

/// The `#` preamble, written once per file. A record has to be readable without this
/// source, and every line below is one a reader would otherwise guess wrong.
pub fn record_header(meta: &RunMeta<'_>) -> String {
    format!("{HEADER_NOTES}\n{}\n{}", run_conditions(meta).join("\n"), COLUMNS.join("\t"))
}

const HEADER_NOTES: &str = "\
# peacockdb cost-model calibration record. One row per CALL — per
# (plan node, recipe step, call index), not per node and not per output partition: one
#   plan node publishes several recipe steps, and a batched run drives each of them once
#   per batch per lane. A call answering with several output partitions is still ONE row;
#   its cost is the sum over its regions, because the shared prologue is charged to p0.
# A benchmark executes its plan several times and writes EVERY measured execution, in
#   the order they ran; the .benchmark.txt beside it reports one chosen run instead. So
#   the same (query, mode, node_seq, recipe_seq, call_index) recurs once per execution,
#   told apart by run_index. Spread across executions is data.
# ONE FILE IS ONE RUN. The `# run:` lines below hold what is constant across it, and
#   each of them changes what the microseconds MEAN — so appending a run that disagrees
#   with them is refused rather than merged.
# mode = the batch-partitioned planning mode, `bp-tp4-sized` and the like. The same query
#   at two modes is a different plan and a different set of calls.
# node_seq = the node's position in the TREE, post-order — the same space recipe_seq is
#   in. The driver numbers nodes pre-order, so this is a translation and not the index a
#   report is walked by.
# lane = the lane the node was DRIVEN on. A scatter is driven on one and emits into four,
#   a cross-lane accumulator the other way round; a call belongs to the driving one.
# recipe_seq = the seq the call addressed in the FlatBuffers plan. Deliberately a different
#   index from node_seq: one plan node with two calls is one tree position and two fb ones.
# recipe_kind = the fb node kind the call addressed, `CudfAggregate{Partial}` and the
#   like. Redundant with recipe_seq given the plan, and written anyway so a row is
#   readable without one.
# call_index = which call of this recipe_seq the session had reached, 0 for the first.
#   The number C++ counts to independently, and the key the two halves of a measurement
#   meet on.
# run_index = which execution of the case, 0 for the first MEASURED one — the warm-up is
#   not written. Derivable from a repeat of call_index 0 and written anyway, so that a
#   file joined against this one (hbm.tsv) names the same execution by the same number
#   rather than by its own count of the same boundary.
# out_rows/out_bytes = what the call answered with, summed over its output partitions.
#   Both come from the device's region: a call in the middle of a node's chain hands the
#   raw handle on, so this side never built a batch from it and never priced it.
# in_rows/in_bytes = what the CALLER handed over, summed over the call's input slots.
#   Empty where the input was the previous call's output: nothing on this side priced it,
#   and that call's own out_bytes is the figure.
# peacock_host_us = host time before the region's first device touch — the prologue only
#   peacockdb pays.
# cudf_host_us = host time from that touch to the end of the region. NOT launch cost:
#   cuDF and rmm synchronize internally, so it follows device_us closely.
# device_us = between the region's CUDA events, on the one stream everything is issued
#   to. An interval of that stream, not a figure for the device as a whole.
# hbm_bytes is NOT here: it comes from Nsight and joins on the same tuple.";

/// Append this run's rows to `$PEACOCK_RECORD_PATH`, or do nothing if it is unset.
///
/// Appends rather than writes: a collection run is many queries in one process, and
/// one file per run is what the fit reads. The header goes in only when the file is
/// created, so concatenating runs stays a valid file.
pub fn append_records(rows: &[String], meta: &RunMeta<'_>) {
    let Ok(path) = std::env::var(RECORD_PATH_ENV) else { return };
    let path = PathBuf::from(path);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let fresh = std::fs::metadata(&path).map(|m| m.len() == 0).unwrap_or(true);
    if !fresh {
        assert_run_conditions_match(&path, meta);
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("cannot open {} for records: {e}", path.display()));
    let mut text = String::new();
    if fresh {
        text.push_str(&record_header(meta));
        text.push('\n');
    }
    for row in rows {
        text.push_str(row);
        text.push('\n');
    }
    f.write_all(text.as_bytes())
        .unwrap_or_else(|e| panic!("cannot append records to {}: {e}", path.display()));
}

/// Refuse to append to a file whose heading describes a different run.
///
/// This is the check the three dropped columns used to make possible. As columns, a file
/// mixing two runs was legal and only detectable afterwards by whoever thought to look;
/// as a heading written once, the mixing is what has to be caught, and here is the only
/// place it can be. Reads the heading, not the file: it stops at the first row.
fn assert_run_conditions_match(path: &Path, meta: &RunMeta<'_>) {
    let f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("cannot read the heading of {}: {e}", path.display()));
    let found: Vec<String> = std::io::BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .take_while(|l| l.starts_with('#'))
        .filter(|l| l.starts_with(RUN_PREFIX))
        .collect();
    let want = run_conditions(meta);
    assert!(
        found == want,
        "{} was written by a different run and this one would not be comparable with \
         it.\n  file: {}\n  this run: {}\nA record file is one run: point \
         {RECORD_PATH_ENV} at a new path, or remove that one.",
        path.display(),
        found.join(" | "),
        want.join(" | "),
    );
}
