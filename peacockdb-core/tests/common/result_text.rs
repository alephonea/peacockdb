//! Comparing and sizing a result without holding it whole.
//!
//! Decide, then materialize. The comparator renders a row, hashes it and drops the string,
//! so a green run holds eight bytes a row rather than the answer — `anti-join` is 1.2
//! million rows and 240 MB of `String` under the old exact arm, which `assert_eq!`
//! materialized on both sides before comparing a byte. The cap check is the same shape from
//! the other end: an answer that will never be written is never rendered whole.
//!
//! On a mismatch the rows are re-streamed and a bounded excerpt printed. That is not
//! politeness — a CI log drops a line past a thousand characters, so a half-gigabyte diff
//! spends the memory and produces something nobody can read.

use std::hash::{Hash, Hasher};

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};

/// How many rows either side of the first difference a failure prints.
const EXCERPT: usize = 3;

/// One formatter per column, built once for the batch. Arrow's is cheap but not free, and
/// the alternative is building one per cell — 1.2 million rows times sixteen columns.
fn formatters(batch: &RecordBatch) -> Vec<ArrayFormatter<'_>> {
    let options = FormatOptions::default();
    batch
        .columns()
        .iter()
        .map(|column| ArrayFormatter::try_new(column, &options).expect("a formattable column"))
        .collect()
}

/// One row rendered as its cells, tab-separated. Not the golden's padded form — padding is
/// a function of the whole answer, so it cannot be computed a row at a time — and the
/// verdict is the same either way: two answers agree cell for cell exactly when they agree
/// padded.
fn render_row(columns: &[ArrayFormatter<'_>], row: usize, out: &mut String) {
    out.clear();
    for formatter in columns {
        out.push_str(&formatter.value(row).to_string());
        out.push('\t');
    }
}

/// Every row's digest, paired with its position, sorted by digest — the same
/// order-independence the golden's sorted rendering has, at eight bytes a row.
fn row_digests(batches: &[RecordBatch]) -> Vec<(u64, usize)> {
    let mut digests = Vec::new();
    let mut rendered = String::new();
    let mut at = 0usize;
    for batch in batches {
        let columns = formatters(batch);
        for row in 0..batch.num_rows() {
            render_row(&columns, row, &mut rendered);
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            rendered.hash(&mut hasher);
            digests.push((hasher.finish(), at));
            at += 1;
        }
    }
    digests.sort_unstable();
    digests
}

/// The schema as one digest, so a right answer under the wrong column names is still wrong.
/// The old exact arm compared the golden's header line and would have caught it; nothing
/// else here reads a name.
fn schema_digest(batches: &[RecordBatch]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    if let Some(batch) = batches.first() {
        for field in batch.schema().fields() {
            field.name().hash(&mut hasher);
            format!("{:?}", field.data_type()).hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Whether the two answers are the same multiset of rows under the same schema.
pub fn results_agree(expected: &[RecordBatch], actual: &[RecordBatch]) -> bool {
    schema_digest(expected) == schema_digest(actual)
        && digests_only(expected) == digests_only(actual)
}

fn digests_only(batches: &[RecordBatch]) -> Vec<u64> {
    row_digests(batches)
        .into_iter()
        .map(|(digest, _)| digest)
        .collect()
}

/// What a failure prints: the first row the two disagree on, with a few either side, from a
/// second pass over the rows the digests named. Bounded on purpose.
pub fn first_difference(expected: &[RecordBatch], actual: &[RecordBatch]) -> String {
    if schema_digest(expected) != schema_digest(actual) {
        return format!(
            "the columns differ — expected {:?}, actual {:?}",
            columns_of(expected),
            columns_of(actual)
        );
    }
    let (want, got) = (row_digests(expected), row_digests(actual));
    let at = want
        .iter()
        .zip(got.iter())
        .position(|((want, _), (got, _))| want != got)
        .unwrap_or_else(|| want.len().min(got.len()));
    let mut said = format!(
        "{} rows expected, {} actual; first difference at sorted row {at}\n",
        want.len(),
        got.len()
    );
    let from = at.saturating_sub(EXCERPT);
    for (label, side) in [("expected", &want), ("actual", &got)] {
        let batches = match label {
            "expected" => expected,
            _ => actual,
        };
        for (offset, (_, row)) in side.iter().enumerate().skip(from).take(EXCERPT * 2 + 1) {
            said.push_str(&format!("  {label} [{offset}] {}\n", row_at(batches, *row)));
        }
    }
    said
}

fn columns_of(batches: &[RecordBatch]) -> Vec<String> {
    batches
        .first()
        .map(|batch| {
            batch
                .schema()
                .fields()
                .iter()
                .map(|field| format!("{}:{:?}", field.name(), field.data_type()))
                .collect()
        })
        .unwrap_or_default()
}

/// One row by its position across the batches, rendered for a message.
fn row_at(batches: &[RecordBatch], mut at: usize) -> String {
    for batch in batches {
        if at < batch.num_rows() {
            let mut rendered = String::new();
            render_row(&formatters(batch), at, &mut rendered);
            return rendered.trim_end().to_string();
        }
        at -= batch.num_rows();
    }
    "(no such row)".to_string()
}

/// A lower bound on the rendered size: the cells alone, without the padding and borders the
/// table adds. Stops the moment it passes `cap`, so an answer far above it costs one row of
/// memory and no full rendering.
pub fn exceeds_rendered_size(batches: &[RecordBatch], cap: usize) -> bool {
    let mut total = 0usize;
    let mut rendered = String::new();
    for batch in batches {
        let columns = formatters(batch);
        for row in 0..batch.num_rows() {
            render_row(&columns, row, &mut rendered);
            total += rendered.len() + 1;
            if total >= cap {
                return true;
            }
        }
    }
    false
}
