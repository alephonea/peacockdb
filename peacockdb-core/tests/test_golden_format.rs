//! The golden text format, read back: `== <query>` sections and the node lines in one.
//!
//! Strings in, strings out — no dataset, no executor — because what is under test is the
//! reader every tier shares. The comparator's cases came from
//! `test_batch_partitioned_plans.rs` with the code they cover; the node-line cases are new
//! with the fields the corpus tiers put on that line.
#[macro_use]
mod common;

use common::golden_text::{ordered_sections, parse_node_line, section_differences};

// --- the section comparator --------------------------------------------------

#[test]
fn a_section_that_moved_is_named_with_the_column_that_moved() {
    let golden = "== q1\nGpuUnload\n  GpuSort: lanes=1\n== q2\nGpuUnload\n";
    let run = "== q1\nGpuUnload\n  GpuSort: lanes=4\n== q2\nGpuUnload\n";
    let said = section_differences(golden, run);
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(said[0].starts_with("q1: line 2, column "), "{said:?}");
    assert!(
        said[0].contains("lanes=1") && said[0].contains("lanes=4"),
        "{said:?}"
    );
}

#[test]
fn a_section_missing_from_one_side_says_which_side() {
    let both = "== q1\nGpuUnload\n== q2\nGpuUnload\n";
    let short = "== q1\nGpuUnload\n";
    assert!(
        section_differences(both, short)[0].starts_with("q2: in the golden"),
        "{:?}",
        section_differences(both, short)
    );
    assert!(
        section_differences(short, both)[0].starts_with("q2: produced by the run"),
        "{:?}",
        section_differences(short, both)
    );
}

#[test]
fn sections_out_of_order_are_named_by_position() {
    let golden = "== q1\nGpuUnload\n== q2\nGpuUnload\n";
    let run = "== q2\nGpuUnload\n== q1\nGpuUnload\n";
    let said = section_differences(golden, run);
    assert!(
        said[0].starts_with("section 0: `q1` in the golden"),
        "{said:?}"
    );
}

#[test]
fn a_section_body_is_every_line_under_its_header() {
    let sections =
        ordered_sections("== q1\nGpuUnload\n\n--- memory ---\nbudget=1\n== q2\nrefused: #180\n");
    assert_eq!(sections.len(), 2);
    assert_eq!(sections[0].0, "q1");
    assert_eq!(sections[0].1, "GpuUnload\n\n--- memory ---\nbudget=1\n");
    assert_eq!(sections[1].1, "refused: #180\n");
}

// --- the node line -----------------------------------------------------------

#[test]
fn a_node_line_carries_its_name_its_depth_and_every_field() {
    let line = "    GpuFilter: predicate=l_shipdate@4 = 5, lanes=4, batches=multiple, output_rows=7, output_bytes=91";
    let node = parse_node_line(line).expect("a node line");
    assert_eq!(node.name, "GpuFilter");
    assert_eq!(node.depth, 2);
    assert_eq!(node.field("predicate"), Some("l_shipdate@4 = 5"));
    assert_eq!(node.field("batches"), Some("multiple"));
    assert_eq!(node.count("output_rows"), Some(7));
    assert_eq!(node.count("output_bytes"), Some(91));
    assert_eq!(node.field("lanes"), Some("4"));
    assert_eq!(node.field("fetch"), None);
}

#[test]
fn a_comma_inside_a_value_does_not_split_a_field() {
    let line = "GpuHashJoin: on=[(c_custkey@0, o_custkey@1)], filter=CAST(x@0 AS Decimal128(38, 15)) > y@1, lanes=1";
    let node = parse_node_line(line).expect("a node line");
    assert_eq!(node.field("on"), Some("[(c_custkey@0, o_custkey@1)]"));
    assert_eq!(
        node.field("filter"),
        Some("CAST(x@0 AS Decimal128(38, 15)) > y@1")
    );
    assert_eq!(node.field("lanes"), Some("1"));
    assert_eq!(node.fields.len(), 3, "{:?}", node.fields);
}

#[test]
fn a_quoted_comma_does_not_split_a_field_either() {
    let node = parse_node_line(r#"GpuFilter: predicate=c_name@1 = Utf8("a,b"), lanes=1"#)
        .expect("a node line");
    assert_eq!(node.field("predicate"), Some(r#"c_name@1 = Utf8("a,b")"#));
    assert_eq!(node.fields.len(), 2, "{:?}", node.fields);
}

#[test]
fn a_nested_mapping_survives_as_one_field() {
    let line = "  GpuLoadParquet: table=lineitem, partition_groups=[[[0, 1], [2]], [[3]]], lanes=2";
    let node = parse_node_line(line).expect("a node line");
    assert_eq!(node.depth, 1);
    assert_eq!(
        node.field("partition_groups"),
        Some("[[[0, 1], [2]], [[3]]]")
    );
}

/// The legacy tier renders a node with no fields of its own as a bare name and appends the
/// cost fields after a comma, so the separator is not always a colon.
#[test]
fn a_node_with_no_fields_of_its_own_still_carries_the_cost_fields() {
    let node = parse_node_line(
        "  GpuCoalescePartitionsExec, partitions=1, output_rows=14, output_bytes=560",
    )
    .expect("a node line");
    assert_eq!(node.name, "GpuCoalescePartitionsExec");
    assert_eq!(node.count("output_bytes"), Some(560));
}

#[test]
fn a_bare_node_name_is_a_node_line() {
    let node = parse_node_line("GpuUnload").expect("a node line");
    assert_eq!(node.name, "GpuUnload");
    assert!(node.fields.is_empty(), "{:?}", node.fields);
}

/// Every other line the corpus goldens hold. A reader that took one of these for a node
/// would bin a continuation line's bytes into the cost, or diff a header as a tree.
#[test]
fn nothing_else_in_a_golden_reads_as_a_node_line() {
    for line in [
        "== q1",
        "--- memory ---",
        "  in_rows=[[860160]] batch_rows=[[14]] batch_bytes=[[560]]",
        "    p0: in_rows=4 out_rows=4 out_bytes=160",
        "storage_read=560 # GpuLoadParquet",
        "peacockdb_cost=1120",
        "",
        "  budget=2147483648, accumulators=1, certain=true",
    ] {
        assert!(
            parse_node_line(line).is_none(),
            "read as a node line: {line}"
        );
    }
}

// --- the subset oracle -------------------------------------------------------
//
// `data_fusion_subset` has no user among the twenty — `scan-limit` is its only one and
// arrives with T19 — so its parts are exercised here rather than first running during a
// rollout, where a bad tail parse fails on somebody else's change.

use common::corpus::{owed_rows, take_rows, wanted_rows, without_its_limit};

fn tail_of(sql: &str) -> (String, u64, Option<u64>) {
    without_its_limit(sql, "a test")
}

#[test]
fn a_trailing_limit_is_taken_off_the_body() {
    assert_eq!(
        tail_of("SELECT * FROM lineitem LIMIT 10;"),
        ("SELECT * FROM lineitem ".to_string(), 0, Some(10))
    );
    assert_eq!(
        tail_of("SELECT * FROM lineitem LIMIT 10 OFFSET 5"),
        ("SELECT * FROM lineitem ".to_string(), 5, Some(10))
    );
    assert_eq!(
        tail_of("select * from t limit 3 offset 1 ;").1,
        1,
        "the keywords are matched without regard to case"
    );
}

/// The inner limit stays: a query can carry one, and what this strips is the outer one.
#[test]
fn only_the_last_limit_comes_off() {
    let (body, skip, fetch) =
        tail_of("SELECT k FROM (SELECT k FROM t LIMIT 40 OFFSET 5) x LIMIT 20 OFFSET 3");
    assert!(body.contains("LIMIT 40 OFFSET 5"), "{body}");
    assert_eq!((skip, fetch), (3, Some(20)));
}

#[test]
#[should_panic(expected = "has no limit")]
fn a_query_with_no_limit_declaring_this_oracle_fails() {
    tail_of("SELECT * FROM lineitem");
}

#[test]
#[should_panic(expected = "is not `LIMIT n [OFFSET m]`")]
fn junk_after_the_interval_fails_rather_than_being_trimmed() {
    tail_of("SELECT * FROM t LIMIT 10 ORDER BY k");
}

#[test]
#[should_panic(expected = "is not `LIMIT n [OFFSET m]`")]
fn an_offset_with_no_number_fails() {
    tail_of("SELECT * FROM t LIMIT 10 OFFSET");
}

/// A tail that merely contains the word is not an interval. `rfind` reaches inside an
/// identifier, so this is the case that says the parse checks what it found.
#[test]
#[should_panic(expected = "is not `LIMIT n [OFFSET m]`")]
fn a_column_named_for_the_keyword_is_not_an_interval() {
    tail_of("SELECT credit_limit FROM t");
}

#[test]
fn the_count_is_what_the_interval_leaves() {
    assert_eq!(wanted_rows(100, 0, Some(10)), 10, "the plain case");
    assert_eq!(
        wanted_rows(100, 95, Some(10)),
        5,
        "a limit past what is left"
    );
    assert_eq!(wanted_rows(4, 10, Some(10)), 0, "an offset past the end");
    assert_eq!(wanted_rows(0, 0, Some(10)), 0, "nothing to limit");
    assert_eq!(wanted_rows(100, 3, None), 97, "an offset with no limit");
}

/// The case set membership passes and this must not: a run returning one row twice where
/// the unlimited answer holds it once.
#[test]
fn a_row_returned_twice_is_not_contained_in_an_answer_holding_it_once() {
    let twice = rows(&["a", "a"]);
    let once = rows(&["a", "b"]);
    let mut owed = owed_rows(&twice);
    take_rows(&mut owed, &once, &mut Default::default());
    assert_eq!(
        owed.values().sum::<usize>(),
        1,
        "the second copy was accounted for by a row the oracle holds once"
    );

    let mut owed = owed_rows(&rows(&["a"]));
    take_rows(&mut owed, &once, &mut Default::default());
    assert!(
        owed.is_empty(),
        "one copy is contained in an answer holding one"
    );
}

/// Two answers of different widths, which is what the case above cannot show: the rows are
/// rendered per side, so a padded rendering makes the same logical row two different
/// strings and nothing is ever struck off. Both fixtures being one character wide made the
/// padding cancel.
#[test]
fn containment_holds_when_the_two_sides_render_at_different_widths() {
    let returned = rows(&["a"]);
    let unlimited = rows(&["a", "bbbbbb"]);
    let mut owed = owed_rows(&returned);
    take_rows(&mut owed, &unlimited, &mut Default::default());
    assert!(
        owed.is_empty(),
        "`a` was not struck off an answer that holds it — left owing {owed:?}"
    );
}

/// The separator has to be a character the data cannot hold, which a tab is not: with one,
/// a row of `("a\tb", "c")` and a row of `("a", "b\tc")` render to one string and are one
/// digest — two different answers agreeing, on the comparison nothing sits behind.
#[test]
fn a_separator_in_the_data_does_not_make_two_rows_one() {
    let left = two_columns(&[("a\tb", "c")]);
    let right = two_columns(&[("a", "b\tc")]);
    assert!(
        !common::result_text::results_agree(&left, &right),
        "two different answers came out equal"
    );
}

fn two_columns(rows: &[(&str, &str)]) -> Vec<datafusion::arrow::array::RecordBatch> {
    use std::sync::Arc;

    use datafusion::arrow::array::{RecordBatch, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let first: Vec<&str> = rows.iter().map(|(a, _)| *a).collect();
    let second: Vec<&str> = rows.iter().map(|(_, b)| *b).collect();
    vec![
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(first)),
                Arc::new(StringArray::from(second)),
            ],
        )
        .expect("a batch"),
    ]
}

fn rows(values: &[&str]) -> Vec<datafusion::arrow::array::RecordBatch> {
    use std::sync::Arc;

    use datafusion::arrow::array::{RecordBatch, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Utf8, true)]));
    let column = Arc::new(StringArray::from(values.to_vec()));
    vec![RecordBatch::try_new(schema, vec![column]).expect("a batch")]
}

// --- a doc block a split left behind -----------------------------------------

/// Two `///` groups separated by a blank line above one item: the lower group is that
/// item's, and the upper one belonged to a declaration that is no longer there. Splitting a
/// function in two and moving only the code leaves this, and nothing reads as missing —
/// every cap still holds and the build is green.
///
/// It covers the SPLIT shape only. The other half of the antipattern is a declaration
/// INSERTED above an existing one, which takes its whole block and leaves no blank line
/// anywhere; that is a property of a diff rather than of a tree, and no check here sees it.
/// The reading habit named in `coding-style.md` is what covers it.
#[test]
fn no_declaration_carries_a_block_left_behind_by_a_split() {
    let mut stranded: Vec<String> = Vec::new();
    for path in rust_sources() {
        let text = std::fs::read_to_string(&path).expect("a source file");
        for (line, at) in stranded_blocks(&text) {
            stranded.push(format!("{}:{}: {line}", path.display(), at + 1));
        }
    }
    assert!(
        stranded.is_empty(),
        "a doc block sits above a blank line and another block — the upper one belonged to \
         a declaration that is gone:\n{}",
        stranded.join("\n")
    );
}

/// The first line of each stranded group, with its line number. A group qualifies when a
/// `///` run is followed by a blank line, then another `///` run, then an item.
fn stranded_blocks(text: &str) -> Vec<(String, usize)> {
    let lines: Vec<&str> = text.lines().collect();
    let doc = |at: usize| {
        lines
            .get(at)
            .is_some_and(|l| l.trim_start().starts_with("///"))
    };
    let blank = |at: usize| lines.get(at).is_some_and(|l| l.trim().is_empty());
    let mut found = Vec::new();
    let mut at = 0;
    while at < lines.len() {
        if !doc(at) || (at > 0 && (doc(at - 1) || blank_run_above(&lines, at))) {
            at += 1;
            continue;
        }
        let upper = at;
        let mut cursor = at;
        while doc(cursor) {
            cursor += 1;
        }
        if blank(cursor) && doc(cursor + 1) {
            let mut lower = cursor + 1;
            while doc(lower) {
                lower += 1;
            }
            if lines.get(lower).is_some_and(|l| !l.trim().is_empty()) {
                found.push((lines[upper].trim().to_string(), upper));
            }
        }
        at = cursor;
    }
    found
}

/// Whether the line above is blank AND the one above that is a doc line — which means this
/// run is the LOWER half of a pair already reported, not the start of a new one.
fn blank_run_above(lines: &[&str], at: usize) -> bool {
    at >= 2 && lines[at - 1].trim().is_empty() && lines[at - 2].trim_start().starts_with("///")
}

fn rust_sources() -> Vec<std::path::PathBuf> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut found = Vec::new();
    for dir in ["src", "tests"] {
        walk(&root.join(dir), &mut found);
    }
    assert!(found.len() > 50, "only {} sources were read", found.len());
    found
}

fn walk(dir: &std::path::Path, into: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        match path.is_dir() {
            true => walk(&path, into),
            false if path.extension().is_some_and(|e| e == "rs") => into.push(path),
            false => {}
        }
    }
}

/// The guard's own cases, since the tree is green and a check with nothing to find is one
/// nobody can tell from a check that does not work.
#[test]
fn the_split_guard_finds_the_shape_and_nothing_else() {
    let split = "/// Left behind by a split.\n\n/// The item's own block.\nfn item() {}\n";
    assert_eq!(stranded_blocks(split).len(), 1, "the shape it exists for");

    let contiguous = "/// One block.\n/// Second line.\nfn item() {}\n";
    assert!(
        stranded_blocks(contiguous).is_empty(),
        "an insertion takes the whole block and leaves no blank line — out of scope"
    );

    let two_items = "/// One item.\nfn a() {}\n\n/// Another item.\nfn b() {}\n";
    assert!(
        stranded_blocks(two_items).is_empty(),
        "two documented items in a row are not a stranded block"
    );

    let module_doc = "//! A module.\n\n/// An item.\nfn a() {}\n";
    assert!(
        stranded_blocks(module_doc).is_empty(),
        "a module doc is not one"
    );
}

// --- the convention two crates read ------------------------------------------

/// The same committed sample `cost-report`'s tests read, asserted to the same values from
/// the other side of the crate boundary. Nothing is shared but the file: `cost-report`'s
/// `[dependencies]` is empty on purpose, so it builds in seconds in the cheap tier, and
/// depending on this crate to share fifteen lines of `key=` reading would trade a stated
/// property for nothing. What is wanted is that the two readers diverge RED, and this is
/// what makes them.
#[test]
fn the_sectioned_cost_fixture_reads_the_same_from_this_side() {
    let text = std::fs::read_to_string(common::testdata_root().join("fixtures/sectioned-cost.txt"))
        .expect("the committed fixture");
    let sections = ordered_sections(&text);
    assert_eq!(
        sections.iter().map(|(q, _)| q.as_str()).collect::<Vec<_>>(),
        ["q6", "q14"]
    );
    let total = |query: &str| -> u64 {
        sections
            .iter()
            .find(|(name, _)| name == query)
            .and_then(|(_, body)| {
                body.lines()
                    .find_map(|line| line.strip_prefix("peacockdb_cost="))
            })
            .expect("a section with a total")
            .parse()
            .expect("a number")
    };
    assert_eq!(total("q6"), 54_772_928);
    assert_eq!(total("q14"), 28_000_000);
}
