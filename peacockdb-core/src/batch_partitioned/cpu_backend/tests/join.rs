//! The join capability matrix, run: one row per mode, each against rows written down here.
//!
//! `dim(k, label)` is the build side and `fact(fk, v)` the probe, the pair the matrix works
//! its examples on. The probe arrives in two batches on purpose — a build row unmatched in
//! the first and matched in the second is the whole reason the finishing types keep the
//! probe keys and answer at done rather than per call.

use super::*;
use crate::batch_partitioned::cpu_backend::join::CpuJoin;
use crate::batch_partitioned::layout::PartitionLayout;
use crate::batch_partitioned::nodes::join::{JoinFilterColumn, JoinSide, NestedLoopJoinType};
use crate::batch_partitioned::nodes::{GpuCrossJoin, GpuJoin, GpuNestedLoopJoin};
use datafusion::common::JoinType;

const DIM: [(i64, &str); 3] = [(1, "a"), (2, "b"), (3, "c")];
/// Split as `[(2,20), (2,21)]` then `[(4,40)]`: `k=2` matches only the first batch and
/// `k=1` and `k=3` match neither, so a per-batch answer and a finished one differ.
const FACT: [&[(i64, i64)]; 2] = [&[(2, 20), (2, 21)], &[(4, 40)]];

fn dim_columns() -> Vec<(&'static str, DataType)> {
    vec![("k", DataType::Int64), ("label", DataType::Utf8)]
}

fn fact_columns() -> Vec<(&'static str, DataType)> {
    vec![("fk", DataType::Int64), ("v", DataType::Int64)]
}

fn dim() -> CpuBatch {
    let keys: ArrayRef = Arc::new(Int64Array::from(
        DIM.iter().map(|(k, _)| Some(*k)).collect::<Vec<_>>(),
    ));
    let labels: ArrayRef = Arc::new(StringArray::from(
        DIM.iter().map(|(_, l)| Some(*l)).collect::<Vec<_>>(),
    ));
    CpuBatch::new(
        RecordBatch::try_new(
            Arc::new(schema_of(&dim_columns()).fields.as_ref().clone()),
            vec![keys, labels],
        )
        .expect("the build side fits its schema"),
    )
}

fn fact(rows: &[(i64, i64)]) -> CpuBatch {
    let fks: ArrayRef = Arc::new(Int64Array::from(
        rows.iter().map(|(fk, _)| Some(*fk)).collect::<Vec<_>>(),
    ));
    let vs: ArrayRef = Arc::new(Int64Array::from(
        rows.iter().map(|(_, v)| Some(*v)).collect::<Vec<_>>(),
    ));
    CpuBatch::new(
        RecordBatch::try_new(
            Arc::new(schema_of(&fact_columns()).fields.as_ref().clone()),
            vec![fks, vs],
        )
        .expect("a probe batch fits its schema"),
    )
}

/// A stub input with the layout a join requires of that side: the build is one batch per
/// lane, the probe streams.
fn side(columns: &[(&str, DataType)], batches: BatchLayout) -> Box<dyn GpuNode> {
    Box::new(Given {
        kind: NodeKind::Intermediate {
            layout: PartitionLayout {
                batch_layout: batches,
                ..PartitionLayout::new(1)
            },
            schema: schema_of(columns),
        },
    })
}

fn hash_join(join_type: JoinType, output: &[(&str, DataType)]) -> GpuJoin {
    GpuJoin::new(
        side(&dim_columns(), BatchLayout::SingleBatch),
        side(&fact_columns(), BatchLayout::MultipleBatches),
        join_type,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        None,
        schema_of(output),
    )
}

/// Every row the join emitted, rendered and sorted: a join's output order is its probe
/// batches' order and its finish pass comes last, so what a mode owes is a multiset.
fn rows_of(batches: Vec<CpuBatch>) -> Vec<String> {
    let mut rows: Vec<String> = Vec::new();
    for batch in batches {
        let batch = batch.record_batch();
        for row in 0..batch.num_rows() {
            let cells: Vec<String> = (0..batch.num_columns())
                .map(|column| {
                    match ScalarValue::try_from_array(batch.column(column), row)
                        .expect("a value at every position")
                    {
                        value if value.is_null() => "-".to_string(),
                        ScalarValue::Utf8(Some(text)) => text,
                        other => other.to_string(),
                    }
                })
                .collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    rows
}

/// The build side set, both probe batches through, then the finish — the call sequence a
/// driver makes, in the order the typestate allows.
fn drive(join: CpuJoin) -> Vec<String> {
    let (mut probing, _) = join.set_build(dim()).expect("the build side is set");
    let mut out = Vec::new();
    for batch in FACT {
        let (produced, _) = probing
            .probe_and_fetch(fact(batch))
            .expect("the probe runs");
        out.extend(produced);
    }
    let (finished, _) = probing.finish_and_fetch().expect("the finish runs");
    out.extend(finished);
    rows_of(out)
}

fn both_sides() -> Vec<(&'static str, DataType)> {
    [dim_columns(), fact_columns()].concat()
}

fn rows(rows: &[&str]) -> Vec<String> {
    let mut rows: Vec<String> = rows.iter().map(|row| row.to_string()).collect();
    rows.sort();
    rows
}

/// The four types that emit both sides. Each owes the same two matched rows plus whatever
/// its preserved side adds — and for Left and Full those extra rows can only be known once
/// every probe batch has gone past, which is what the finish pass is.
#[test]
fn the_types_that_emit_both_sides_answer_what_the_matrix_says() {
    let matched = ["2|b|2|20", "2|b|2|21"];
    let build_unmatched = ["1|a|-|-", "3|c|-|-"];
    let probe_unmatched = ["-|-|4|40"];
    let cases: [(JoinType, Vec<&str>); 4] = [
        (JoinType::Inner, matched.to_vec()),
        (
            JoinType::Left,
            [matched.as_slice(), build_unmatched.as_slice()].concat(),
        ),
        (
            JoinType::Right,
            [matched.as_slice(), probe_unmatched.as_slice()].concat(),
        ),
        (
            JoinType::Full,
            [
                matched.as_slice(),
                build_unmatched.as_slice(),
                probe_unmatched.as_slice(),
            ]
            .concat(),
        ),
    ];
    for (join_type, expected) in cases {
        let node = hash_join(join_type, &both_sides());
        let join = CpuJoin::hash(
            &node,
            &schema_of(&dim_columns()).fields,
            &schema_of(&fact_columns()).fields,
            ctx(),
        )
        .expect("the join builds");
        assert_eq!(drive(join), rows(&expected), "{join_type:?}");
    }
}

/// The build-side semi family: no per-call join at all, so the whole answer is the finish
/// pass over the keys the probe batches contributed.
#[test]
fn the_build_side_semi_family_answers_out_of_its_finish_pass_alone() {
    let mark = [dim_columns(), vec![("mark", DataType::Boolean)]].concat();
    let cases: [(JoinType, Vec<(&str, DataType)>, Vec<&str>); 3] = [
        (JoinType::LeftSemi, dim_columns(), vec!["2|b"]),
        (JoinType::LeftAnti, dim_columns(), vec!["1|a", "3|c"]),
        (
            JoinType::LeftMark,
            mark,
            vec!["1|a|false", "2|b|true", "3|c|false"],
        ),
    ];
    for (join_type, output, expected) in cases {
        let node = hash_join(join_type, &output);
        let join = CpuJoin::hash(
            &node,
            &schema_of(&dim_columns()).fields,
            &schema_of(&fact_columns()).fields,
            ctx(),
        )
        .expect("the join builds");
        assert_eq!(drive(join), rows(&expected), "{join_type:?}");
    }
}

/// The probe-side semi family: membership in a complete build side is a per-row question,
/// so every row leaves with its own batch and there is no finish at all.
#[test]
fn the_probe_side_semi_family_answers_per_batch_and_finishes_with_nothing() {
    let cases: [(JoinType, Vec<&str>); 2] = [
        (JoinType::RightSemi, vec!["2|20", "2|21"]),
        (JoinType::RightAnti, vec!["4|40"]),
    ];
    for (join_type, expected) in cases {
        let node = hash_join(join_type, &fact_columns());
        let join = CpuJoin::hash(
            &node,
            &schema_of(&dim_columns()).fields,
            &schema_of(&fact_columns()).fields,
            ctx(),
        )
        .expect("the join builds");
        let (mut probing, _) = join.set_build(dim()).expect("the build side is set");
        let mut out = Vec::new();
        for batch in FACT {
            let (produced, _) = probing
                .probe_and_fetch(fact(batch))
                .expect("the probe runs");
            out.extend(produced);
        }
        let (finished, _) = probing.finish_and_fetch().expect("the finish runs");
        assert!(finished.is_empty(), "{join_type:?} owes nothing at done");
        assert_eq!(rows_of(out), rows(&expected), "{join_type:?}");
    }
}

/// The one thing a per-batch answer cannot get right on its own: `k=2` matches the first
/// probe batch and nothing in the second. A Left join that answered per batch would pad it
/// as unmatched against the second and emit the row twice.
#[test]
fn a_build_row_matched_in_one_batch_is_not_padded_against_another() {
    let node = hash_join(JoinType::Left, &both_sides());
    let join = CpuJoin::hash(
        &node,
        &schema_of(&dim_columns()).fields,
        &schema_of(&fact_columns()).fields,
        ctx(),
    )
    .expect("the join builds");
    let (mut probing, _) = join.set_build(dim()).expect("the build side is set");
    let mut per_batch = Vec::new();
    for batch in FACT {
        let (produced, _) = probing
            .probe_and_fetch(fact(batch))
            .expect("the probe runs");
        per_batch.extend(produced);
    }
    assert_eq!(
        rows_of(per_batch),
        rows(&["2|b|2|20", "2|b|2|21"]),
        "the probe calls emit matches only — a padded row before done would be a guess"
    );
    let (finished, _) = probing.finish_and_fetch().expect("the finish runs");
    assert_eq!(
        rows_of(finished),
        rows(&["1|a|-|-", "3|c|-|-"]),
        "and the finish emits exactly the build rows nothing ever matched"
    );
}

/// No key to co-locate on, so both sides are one lane and every build row meets every row
/// of the batch. Nothing at done: a cross join's answer is decided by (build, this batch).
#[test]
fn a_cross_join_pairs_every_build_row_with_every_probe_row() {
    let node = GpuCrossJoin::new(
        side(&dim_columns(), BatchLayout::SingleBatch),
        side(&fact_columns(), BatchLayout::MultipleBatches),
        None,
        schema_of(&both_sides()),
    );
    let join = CpuJoin::cross(
        &node,
        &schema_of(&dim_columns()).fields,
        &schema_of(&fact_columns()).fields,
        ctx(),
    )
    .expect("the cross join builds");
    let answered = drive(join);
    assert_eq!(answered.len(), 9, "three build rows by three probe rows");
    assert!(answered.contains(&"1|a|4|40".to_string()));
    assert!(answered.contains(&"3|c|2|20".to_string()));
}

fn greater(build: u32, probe: u32) -> (Expr, Vec<JoinFilterColumn>) {
    (
        Expr::binary(
            Expr::column(0, "k"),
            BinaryOp::Gt,
            Expr::column(1, "fk"),
            DataType::Boolean,
        ),
        vec![
            JoinFilterColumn {
                side: JoinSide::Build,
                index: build,
            },
            JoinFilterColumn {
                side: JoinSide::Probe,
                index: probe,
            },
        ],
    )
}

/// A predicate rather than a key: `k > fk` pairs only `k=3` with the two `fk=2` rows.
#[test]
fn a_nested_loop_inner_join_emits_the_pairs_its_predicate_keeps() {
    let (filter, columns) = greater(0, 0);
    let node = GpuNestedLoopJoin::new(
        side(&dim_columns(), BatchLayout::SingleBatch),
        side(&fact_columns(), BatchLayout::MultipleBatches),
        NestedLoopJoinType::Inner,
        filter,
        columns,
        None,
        schema_of(&both_sides()),
    );
    let join = CpuJoin::nested_loop(
        &node,
        &schema_of(&dim_columns()).fields,
        &schema_of(&fact_columns()).fields,
        ctx(),
    )
    .expect("the nested loop join builds");
    assert_eq!(drive(join), rows(&["3|c|2|20", "3|c|2|21"]));
}

/// The left form pads the build rows no pair kept — and it can, because the matrix gives
/// it a single-batch probe: with one call there is no later batch for a padded row to be
/// wrong about, which is why it needs no finish pass of its own.
#[test]
fn a_nested_loop_left_join_pads_the_build_rows_no_pair_kept() {
    let (filter, columns) = greater(0, 0);
    let node = GpuNestedLoopJoin::new(
        side(&dim_columns(), BatchLayout::SingleBatch),
        side(&fact_columns(), BatchLayout::SingleBatch),
        NestedLoopJoinType::Left,
        filter,
        columns,
        None,
        schema_of(&both_sides()),
    );
    let join = CpuJoin::nested_loop(
        &node,
        &schema_of(&dim_columns()).fields,
        &schema_of(&fact_columns()).fields,
        ctx(),
    )
    .expect("the nested loop join builds");
    let (mut probing, _) = join.set_build(dim()).expect("the build side is set");
    let whole_probe: Vec<(i64, i64)> = FACT.concat();
    let (produced, _) = probing
        .probe_and_fetch(fact(&whole_probe))
        .expect("the probe runs");
    assert_eq!(
        rows_of(produced),
        rows(&["1|a|-|-", "2|b|-|-", "3|c|2|20", "3|c|2|21"]),
    );
}

/// `null_equals_null` reaches the finish join too, and it has to: the pass substitutes for
/// a single legacy call, so a NULL key matching a NULL key there and nowhere else would
/// make the answer depend on how many batches the probe arrived in.
#[test]
fn a_null_key_matches_a_null_key_in_the_finish_pass_when_the_node_says_so() {
    let with_nulls = |null_equals_null: bool| {
        GpuJoin::new(
            side(&dim_columns(), BatchLayout::SingleBatch),
            side(&fact_columns(), BatchLayout::MultipleBatches),
            JoinType::LeftAnti,
            vec![(0, 0)],
            None,
            Vec::new(),
            null_equals_null,
            None,
            schema_of(&dim_columns()),
        )
    };
    let keyed_null = |node: &GpuJoin| {
        let join = CpuJoin::hash(
            node,
            &schema_of(&dim_columns()).fields,
            &schema_of(&fact_columns()).fields,
            ctx(),
        )
        .expect("the join builds");
        let build: ArrayRef = Arc::new(Int64Array::from(vec![None, Some(1i64)]));
        let labels: ArrayRef = Arc::new(StringArray::from(vec![Some("null key"), Some("one")]));
        let build = CpuBatch::new(
            RecordBatch::try_new(
                Arc::new(schema_of(&dim_columns()).fields.as_ref().clone()),
                vec![build, labels],
            )
            .expect("the build side fits"),
        );
        let (mut probing, _) = join.set_build(build).expect("the build side is set");
        let probe_keys: ArrayRef = Arc::new(Int64Array::from(vec![None, Some(9i64)]));
        let probe_values: ArrayRef = Arc::new(Int64Array::from(vec![Some(0i64), Some(9)]));
        let probe = CpuBatch::new(
            RecordBatch::try_new(
                Arc::new(schema_of(&fact_columns()).fields.as_ref().clone()),
                vec![probe_keys, probe_values],
            )
            .expect("the probe fits"),
        );
        probing.probe_and_fetch(probe).expect("the probe runs");
        let (finished, _) = probing.finish_and_fetch().expect("the finish runs");
        rows_of(finished)
    };
    assert_eq!(
        keyed_null(&with_nulls(true)),
        rows(&["1|one"]),
        "the NULL build key found the NULL probe key, so only `1` is unmatched"
    );
    assert_eq!(
        keyed_null(&with_nulls(false)),
        rows(&["-|null key", "1|one"]),
        "and with SQL's own semantics a NULL key matches nothing, so both are unmatched"
    );
}

/// The matrix's second dimension. A residual filter is not decoration: it decides whether
/// a shape streams its probe, drops to one legacy call over a probe the planner made
/// single-batch, or is refused outright — so every cell above has a twin here.
///
/// `k * 10 = v` reads both sides, which is what makes it a residual rather than something
/// DataFusion would have pushed below the join. It keeps the `(2, 20)` pair and drops
/// `(2, 21)`.
fn residual() -> (Expr, Vec<JoinFilterColumn>) {
    (
        Expr::binary(
            Expr::binary(
                Expr::column(0, "k"),
                BinaryOp::Multiply,
                Expr::Literal(ScalarValue::Int64(Some(10))),
                DataType::Int64,
            ),
            BinaryOp::Eq,
            Expr::column(1, "v"),
            DataType::Boolean,
        ),
        vec![
            JoinFilterColumn {
                side: JoinSide::Build,
                index: 0,
            },
            JoinFilterColumn {
                side: JoinSide::Probe,
                index: 1,
            },
        ],
    )
}

fn filtered(join_type: JoinType, output: &[(&str, DataType)]) -> GpuJoin {
    let (filter, columns) = residual();
    GpuJoin::new(
        side(&dim_columns(), BatchLayout::SingleBatch),
        // The planner makes a refusing shape's probe a single batch, and this is that
        // batch: what the matrix calls the legacy call is one call over the whole of it.
        side(&fact_columns(), BatchLayout::SingleBatch),
        join_type,
        vec![(0, 0)],
        Some(filter),
        columns,
        false,
        None,
        schema_of(output),
    )
}

/// Where a filtered cell's rows come out, which is the whole difference the residual makes:
/// a shape that still streams answers per call, one that dropped to a single call answers
/// there too but from one batch, and the finish pass is gone in both.
#[test]
fn the_filtered_column_answers_in_one_call_and_never_at_done() {
    let cases: [(JoinType, Vec<(&str, DataType)>, Vec<&str>); 4] = [
        (JoinType::Inner, both_sides(), vec!["2|b|2|20"]),
        (JoinType::LeftSemi, dim_columns(), vec!["2|b"]),
        (JoinType::LeftAnti, dim_columns(), vec!["1|a", "3|c"]),
        (
            JoinType::LeftMark,
            [dim_columns(), vec![("mark", DataType::Boolean)]].concat(),
            vec!["1|a|false", "2|b|true", "3|c|false"],
        ),
    ];
    for (join_type, output, expected) in cases {
        let node = filtered(join_type, &output);
        let join = CpuJoin::hash(
            &node,
            &schema_of(&dim_columns()).fields,
            &schema_of(&fact_columns()).fields,
            ctx(),
        )
        .expect("the join builds");
        let (mut probing, _) = join.set_build(dim()).expect("the build side is set");
        let whole_probe: Vec<(i64, i64)> = FACT.concat();
        let (per_call, _) = probing
            .probe_and_fetch(fact(&whole_probe))
            .expect("the probe runs");
        let answered = rows_of(per_call);
        let (finished, _) = probing.finish_and_fetch().expect("the finish runs");
        assert!(
            finished.is_empty(),
            "{join_type:?} with a residual is one call and owes nothing at done"
        );
        assert_eq!(answered, rows(&expected), "{join_type:?} with a residual");
    }
}

/// The four cells the matrix refuses, each for a defect or a missing cuDF variant rather
/// than for anything this mode decided — so each names the ticket a reader would follow.
#[test]
fn the_filtered_column_refuses_four_cells_by_ticket() {
    let cases: [(JoinType, &str); 5] = [
        (JoinType::Left, "#153"),
        (JoinType::Right, "#153"),
        (JoinType::Full, "#153"),
        (JoinType::RightSemi, "#159"),
        (JoinType::RightAnti, "#159"),
    ];
    for (join_type, ticket) in cases {
        let node = filtered(join_type, &both_sides());
        let refused = match CpuJoin::hash(
            &node,
            &schema_of(&dim_columns()).fields,
            &schema_of(&fact_columns()).fields,
            ctx(),
        ) {
            Err(refused) => refused,
            Ok(_) => panic!("{join_type:?} with a residual is refused by the matrix"),
        };
        let message = format!("{refused}");
        assert!(
            message.contains(ticket),
            "{join_type:?} has to name {ticket}: {message}"
        );
    }
}
