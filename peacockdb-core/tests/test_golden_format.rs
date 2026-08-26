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
