//! The joins and the scatter on a device, and the one call the frozen surface cannot make.
//!
//! The finish pass is the reason this file exists: probe keys per batch, the concat at
//! done, the anti join and the pad project. T21 left it as the one shape this mode invented
//! with no device behind it, and two of its defects were found by reading rather than by
//! running — so a run is what settles it.

use super::*;

use datafusion::common::JoinType;
use peacockdb_core::batch_partitioned::gpu_backend::emit::GpuEmitter;
use peacockdb_core::batch_partitioned::gpu_backend::join::GpuJoin as GpuJoinExec;
use peacockdb_core::batch_partitioned::nodes::{GpuEmitPartitions, GpuFilter, GpuJoin};

/// The joined row: both sides' columns, which for this fixture is `k, v` twice.
fn joined_columns() -> ArrowSchema {
    schema_of(&[
        ("k", DataType::Utf8),
        ("v", DataType::Int64),
        ("k2", DataType::Utf8),
        ("v2", DataType::Int64),
    ])
}

/// `build` is row group 2 — `(a, 6)` and `(b, 5)` — and the probe is row group 0 filtered
/// to `v > 1`, which is `(a, 2)` alone. So `a` matches and `b` is the row only the finish
/// pass can find.
fn left_join_tree(join_type: JoinType, output: &ArrowSchema) -> Box<dyn GpuNode> {
    let probe = GpuFilter::new(
        mapped(vec![vec![vec![0]]]),
        Expr::binary(
            Expr::column(1, "v"),
            BinaryOp::Gt,
            Expr::Literal(ScalarValue::Int64(Some(1))),
            DataType::Boolean,
        ),
        None,
        Schema::new(Arc::new(columns())),
    );
    Box::new(GpuJoin::new(
        mapped(vec![vec![vec![2]]]),
        Box::new(probe),
        join_type,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        None,
        Schema::new(Arc::new(output.clone())),
    ))
}

/// A Left join cannot run on a device at all, and not only past its first probe batch: its
/// key project and its per-call join both read the same batch, and the ABI has no copy of
/// one any more than of a build side. So the pad project — the shape whose columns were
/// wrong until this morning — still has no device behind it, and says so here rather than
/// in a comment nothing checks.
#[test]
fn a_left_joins_probe_batch_cannot_be_read_twice_and_says_which_ticket() {
    let out = joined_columns();
    let tree = left_join_tree(JoinType::Left, &out);
    let session = Session::open(tree.as_ref());
    let keys = schema_of(&[("k", DataType::Utf8)]);
    let join = GpuJoinExec::new(session.executor, session.recipe(3), Some(&keys), &out)
        .expect("the join builds");
    let (mut probing, _) = join
        .set_build(session.scan(&[2]))
        .expect("the build side is set");
    let probe = session
        .exec(2, &columns())
        .exec(session.scan(&[0]))
        .expect("the filter runs")
        .0;
    let refused = probing
        .probe_and_fetch(probe)
        .expect_err("the batch cannot be read by both calls");
    let message = format!("{refused}");
    assert!(
        message.contains("#152") && message.contains("probe batch"),
        "the refusal has to name the ticket and which handle: {message}"
    );
}

/// The build-side semi family is the one shape that streams a multi-batch probe today: its
/// probe call is the key project alone, so the build side is untouched until the finish
/// consumes it and no copy is ever needed.
#[test]
fn a_semi_join_streams_every_probe_batch_and_answers_at_done() {
    let out = columns();
    let tree: Box<dyn GpuNode> = Box::new(GpuJoin::new(
        mapped(vec![vec![vec![2]]]),
        mapped(vec![vec![vec![0], vec![1]]]),
        JoinType::LeftAnti,
        vec![(1, 1)],
        None,
        Vec::new(),
        false,
        None,
        Schema::new(Arc::new(out.clone())),
    ));
    let session = Session::open(tree.as_ref());
    let keys = schema_of(&[("v", DataType::Int64)]);
    let join = GpuJoinExec::new(session.executor, session.recipe(2), Some(&keys), &out)
        .expect("the join builds");
    let (mut probing, _) = join
        .set_build(session.scan(&[2]))
        .expect("the build side is set");
    for group in [0u32, 1] {
        let (produced, _) = probing
            .probe_and_fetch(session.scan(&[group]))
            .expect("the probe runs");
        assert!(
            produced.is_empty(),
            "a semi join's probe call keeps keys and emits nothing"
        );
    }
    let (finished, _) = probing.finish_and_fetch().expect("the finish runs");
    let [answer] = <[GpuBatch; 1]>::try_from(finished).expect("one batch at done");
    let (back, _) = session
        .export(&out)
        .unload(answer, RowRange::WHOLE)
        .expect("the rows cross the boundary");
    assert_eq!(
        by_key(&back),
        vec![
            vec![string("a"), ScalarValue::Int64(Some(6))],
            vec![string("b"), ScalarValue::Int64(Some(5))],
        ],
        "no v of 6 or 5 appears in the probe, so the anti join keeps both build rows"
    );
}

/// The call the frozen surface cannot make. `execute_node` erases what it reads, so the
/// join call for the first probe batch takes the build side with it; a second batch has
/// nothing to be given, and saying so is better than a handle that is not there.
#[test]
fn a_second_probe_batch_of_a_copying_join_is_refused_by_name() {
    let out = joined_columns();
    let tree: Box<dyn GpuNode> = Box::new(GpuJoin::new(
        mapped(vec![vec![vec![2]]]),
        mapped(vec![vec![vec![0], vec![1]]]),
        JoinType::Inner,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        None,
        Schema::new(Arc::new(out.clone())),
    ));
    let session = Session::open(tree.as_ref());
    let join =
        GpuJoinExec::new(session.executor, session.recipe(2), None, &out).expect("the join builds");
    let (mut probing, _) = join
        .set_build(session.scan(&[2]))
        .expect("the build side is set");
    let (matched, _) = probing
        .probe_and_fetch(session.scan(&[0]))
        .expect("the first probe batch has the build side");
    let [joined] = <[GpuBatch; 1]>::try_from(matched).expect("one batch out per probe batch");
    let (back, _) = session
        .export(&out)
        .unload(joined, RowRange::WHOLE)
        .expect("the rows cross the boundary");
    assert_eq!(
        by_key(&back),
        vec![
            vec![
                string("a"),
                ScalarValue::Int64(Some(6)),
                string("a"),
                ScalarValue::Int64(Some(2))
            ],
            vec![
                string("b"),
                ScalarValue::Int64(Some(5)),
                string("b"),
                ScalarValue::Int64(Some(1))
            ],
        ],
        "the first batch's matches, which is what makes this cell proved and not merely reached"
    );
    let refused = probing
        .probe_and_fetch(session.scan(&[1]))
        .expect_err("the second has none");
    let message = format!("{refused}");
    assert!(
        message.contains("#152") && message.contains("batch 2"),
        "the refusal has to name the ticket and which batch: {message}"
    );
}

/// The scatter: one call per batch, N handles back, and the count is the contract.
#[test]
fn a_scatter_answers_with_one_handle_per_lane() {
    const LANES: usize = 3;
    let tree: Box<dyn GpuNode> = Box::new(GpuEmitPartitions::new(source(), vec![0], LANES));
    let session = Session::open(tree.as_ref());
    let out = columns();
    let mut emitter =
        GpuEmitter::new(session.executor, session.recipe(1), &out).expect("the scatter builds");
    let (lanes, _) = emitter
        .emit(session.scan(&ROW_GROUPS))
        .expect("the scatter runs");
    assert_eq!(lanes.len(), LANES);
    let scattered: usize = lanes.iter().map(Batch::num_rows).sum();
    assert_eq!(
        scattered,
        VALUES.len(),
        "every row landed in exactly one lane"
    );
}
