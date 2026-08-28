//! The joins and the scatter on a device, and the one call the frozen surface cannot make.
//!
//! The finish pass is the reason this file exists: probe keys per batch, the concat at
//! done, the anti join and the pad project. T21 left it as the one shape this mode invented
//! with no device behind it, and two of its defects were found by reading rather than by
//! running — so a run is what settles it.

use peacockdb_core::batch_partitioned::nodes::join::joined_schema;
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
        joined_schema(&columns(), &columns(), join_type),
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
    let join = GpuJoinExec::new(
        session.executor,
        session.recipe(3),
        Some(JoinType::Left),
        Some(&keys),
        &out,
    )
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
/// consumes it and no copy is ever needed. All three members, because they answer three
/// different questions out of the same pass — anti gathers what did not match, semi what
/// did, and mark scatters a boolean into a column of its own.
#[test]
fn every_member_of_the_semi_family_streams_and_answers_at_done() {
    let build_rows = || {
        vec![
            vec![string("a"), ScalarValue::Int64(Some(6))],
            vec![string("b"), ScalarValue::Int64(Some(5))],
        ]
    };
    let marked = |mark: bool| {
        vec![
            vec![
                string("a"),
                ScalarValue::Int64(Some(6)),
                ScalarValue::Boolean(Some(mark)),
            ],
            vec![
                string("b"),
                ScalarValue::Int64(Some(5)),
                ScalarValue::Boolean(Some(mark)),
            ],
        ]
    };
    // On `v` nothing matches — the probe holds 1, 2, 3 and 4 and the build holds 5 and 6 —
    // and on `k` everything does, which is what tells the three members apart.
    let cases: [(
        JoinType,
        (u32, u32),
        Vec<(&str, DataType)>,
        Vec<Vec<ScalarValue>>,
    ); 3] = [
        (JoinType::LeftAnti, (1, 1), vec![], build_rows()),
        (JoinType::LeftSemi, (0, 0), vec![], build_rows()),
        (
            JoinType::LeftMark,
            (0, 0),
            vec![("mark", DataType::Boolean)],
            marked(true),
        ),
    ];
    for (join_type, keys, extra, expected) in cases {
        let out = schema_of(
            &[
                &[("k", DataType::Utf8), ("v", DataType::Int64)][..],
                &extra[..],
            ]
            .concat(),
        );
        let tree: Box<dyn GpuNode> = Box::new(GpuJoin::new(
            mapped(vec![vec![vec![2]]]),
            mapped(vec![vec![vec![0], vec![1]]]),
            join_type,
            vec![keys],
            None,
            Vec::new(),
            false,
            None,
            Schema::new(Arc::new(out.clone())),
            joined_schema(&columns(), &columns(), join_type),
        ));
        let session = Session::open(tree.as_ref());
        let key_column = match keys.1 {
            0 => ("k", DataType::Utf8),
            _ => ("v", DataType::Int64),
        };
        let key_schema = schema_of(&[key_column]);
        let join = GpuJoinExec::new(
            session.executor,
            session.recipe(2),
            Some(join_type),
            Some(&key_schema),
            &out,
        )
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
                "{join_type:?}: a probe call keeps keys and emits nothing"
            );
        }
        let (finished, _) = probing.finish_and_fetch().expect("the finish runs");
        let [answer] = <[GpuBatch; 1]>::try_from(finished).expect("one batch at done");
        let (back, _) = session
            .export(&out)
            .unload(answer, RowRange::WHOLE)
            .expect("the rows cross the boundary");
        assert_eq!(by_key(&back), expected, "{join_type:?}");
    }
}

/// The narrowing project, on a device. A semi join's finish emits the build side, so a
/// node keeping one of its two columns owes a call that cuts it down — and until the
/// recipe published one, the device emitted both columns while the CPU emitted one.
///
/// `v` alone, so the column that survives is not the key: keeping the key would pass with
/// a project that emitted the wrong column as readily as the right one.
#[test]
fn a_projecting_semi_joins_finish_emits_the_column_the_node_declares() {
    let out = schema_of(&[("v", DataType::Int64)]);
    let tree: Box<dyn GpuNode> = Box::new(GpuJoin::new(
        mapped(vec![vec![vec![2]]]),
        mapped(vec![vec![vec![0], vec![1]]]),
        JoinType::LeftSemi,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        Some(vec![1]),
        Schema::new(Arc::new(out.clone())),
        joined_schema(&columns(), &columns(), JoinType::LeftSemi),
    ));
    let session = Session::open(tree.as_ref());
    let keys = schema_of(&[("k", DataType::Utf8)]);
    let join = GpuJoinExec::new(
        session.executor,
        session.recipe(2),
        Some(JoinType::LeftSemi),
        Some(&keys),
        &out,
    )
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
            "a probe call keeps keys and emits nothing"
        );
    }
    let (finished, _) = probing.finish_and_fetch().expect("the finish runs");
    let [answer] = <[GpuBatch; 1]>::try_from(finished).expect("one batch at done");
    let (back, _) = session
        .export(&out)
        .unload(answer, RowRange::WHOLE)
        .expect("the rows cross the boundary");
    let mut rows = rows(&back);
    rows.sort_by_key(|row| format!("{:?}", row));
    assert_eq!(
        rows,
        vec![
            vec![ScalarValue::Int64(Some(5))],
            vec![ScalarValue::Int64(Some(6))],
        ],
        "one column out, and it is v"
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
        joined_schema(&columns(), &columns(), JoinType::Inner),
    ));
    let session = Session::open(tree.as_ref());
    let join = GpuJoinExec::new(
        session.executor,
        session.recipe(2),
        Some(JoinType::Inner),
        None,
        &out,
    )
    .expect("the join builds");
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
        values().len(),
        "every row landed in exactly one lane"
    );
}

/// The device's half of the same claim: a lane whose probe was empty owes every build row
/// from an anti join, and this backend answers it without a call — the concat of no keys
/// is a refusal, and what the finish would have computed is the build side itself.
#[test]
fn a_finish_over_no_probe_keys_hands_the_build_side_up() {
    let out = columns();
    let tree: Box<dyn GpuNode> = Box::new(GpuJoin::new(
        mapped(vec![vec![vec![2]]]),
        mapped(vec![vec![vec![0]]]),
        JoinType::LeftAnti,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        None,
        Schema::new(Arc::new(out.clone())),
        joined_schema(&columns(), &columns(), JoinType::LeftAnti),
    ));
    let session = Session::open(tree.as_ref());
    let keys = schema_of(&[("k", DataType::Utf8)]);
    let join = GpuJoinExec::new(
        session.executor,
        session.recipe(2),
        Some(JoinType::LeftAnti),
        Some(&keys),
        &out,
    )
    .expect("the join builds");
    let (probing, _) = join
        .set_build(session.scan(&[2]))
        .expect("the build side is set");
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
        "no probe batch arrived, so both build rows are unmatched"
    );
}
