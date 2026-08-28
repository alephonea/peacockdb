//! The joins: what each of the three requires of its inputs, and the distribution a join
//! declares about its own output — the claim nothing downstream re-checks.

use super::*;

#[test]
fn a_join_whose_build_side_is_many_batches_names_the_node_that_fixes_it() {
    let build = Given::input(one_lane(BatchLayout::MultipleBatches), &["k"]);
    let probe = Given::input(one_lane(BatchLayout::MultipleBatches), &["fk"]);
    let schema = Schema::new(Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("fk", DataType::Int64, true),
    ])));
    let join = GpuCrossJoin::new(build, probe, None, schema);
    invalid(
        join.validate_schemas_and_partitions(),
        "GpuCoalesceAllBatches",
    );
}

#[test]
fn a_join_filter_column_mapped_to_the_wrong_side_is_caught_at_plan_time() {
    use super::join::{JoinFilterColumn, JoinSide};
    let build = Given::input(one_lane(BatchLayout::SingleBatch), &["k"]);
    let probe = Given::input(one_lane(BatchLayout::MultipleBatches), &["fk"]);
    let schema = Schema::new(Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("fk", DataType::Int64, true),
    ])));
    let filter = Expr::binary(
        Expr::column(0, "k"),
        BinaryOp::Lt,
        Expr::column(1, "fk"),
        DataType::Boolean,
    );
    // Both filter columns pointed at the probe: @0 then reads a valid column of the
    // wrong table, which nothing downstream could detect.
    let mapping = vec![
        JoinFilterColumn {
            side: JoinSide::Probe,
            index: 0,
        },
        JoinFilterColumn {
            side: JoinSide::Probe,
            index: 0,
        },
    ];
    let join = GpuNestedLoopJoin::new(
        build,
        probe,
        NestedLoopJoinType::Inner,
        filter,
        mapping,
        None,
        schema,
    );
    invalid(
        join.validate_schemas_and_partitions(),
        "maps to fk on the Probe side",
    );
}

#[test]
fn a_finalize_expression_is_checked_against_the_intermediate_and_not_the_input() {
    // @2 is a column of the input and not of the intermediate table the finalize reads,
    // which is the whole reason the two schemas are separate arguments.
    let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["k", "other", "n"]);
    let intermediate = columns(&["k", "n"]);
    let aggregate = GpuAggregate::new(
        input,
        summing(
            vec![Expr::column(0, "k")],
            vec![Expr::column(2, "n")],
            Some(vec![NamedExpr::new(Expr::column(2, "n"), "n")]),
        ),
        intermediate.clone(),
        columns(&["k", "n"]),
    );
    invalid(
        aggregate.validate_schemas_and_partitions(),
        "GpuAggregate: column n@2 is past the 2 columns",
    );
}

fn hashed(keys: Vec<u32>, n: usize) -> PartitionLayout {
    PartitionLayout {
        key_distribution: KeyDistribution::ByHash { hash_keys: keys },
        batch_layout: BatchLayout::SingleBatch,
        ..PartitionLayout::new(n)
    }
}

fn joining(
    build: Box<dyn GpuNode>,
    probe: Box<dyn GpuNode>,
    join_type: datafusion::common::JoinType,
    filter: Option<Expr>,
    filter_columns: Vec<join::JoinFilterColumn>,
) -> GpuJoin {
    GpuJoin::new(
        build,
        probe,
        join_type,
        vec![(0, 0)],
        filter,
        filter_columns,
        false,
        None,
        columns(&["k", "fk"]),
        join::joined_schema(&columns(&["k"]).fields, &columns(&["fk"]).fields, join_type),
    )
}

#[test]
fn a_join_whose_sides_carry_different_lane_counts_is_refused() {
    let join = joining(
        Given::input(hashed(vec![0], 4), &["k"]),
        Given::input(hashed(vec![0], 8), &["fk"]),
        datafusion::common::JoinType::Inner,
        None,
        Vec::new(),
    );
    invalid(
        join.validate_schemas_and_partitions(),
        "the sides carry 4 and 8 lanes",
    );
}

#[test]
fn a_join_over_lanes_hashed_on_other_columns_names_the_node_that_fixes_it() {
    // Hashed on a column that is not the join key: lane p of one side then holds rows
    // whose matches are in some other lane of the other, and each lane answers for part
    // of a key it cannot see.
    let join = joining(
        Given::input(hashed(vec![1], 4), &["k", "other"]),
        Given::input(hashed(vec![0], 4), &["fk"]),
        datafusion::common::JoinType::Inner,
        None,
        Vec::new(),
    );
    invalid(join.validate_schemas_and_partitions(), "GpuEmitPartitions");
}

#[test]
fn an_equi_join_whose_build_side_is_many_batches_names_the_node_that_fixes_it() {
    let join = joining(
        Given::input(one_lane(BatchLayout::MultipleBatches), &["k"]),
        Given::input(one_lane(BatchLayout::SingleBatch), &["fk"]),
        datafusion::common::JoinType::Inner,
        None,
        Vec::new(),
    );
    invalid(
        join.validate_schemas_and_partitions(),
        "its build side must be one batch per lane",
    );
}

#[test]
fn a_join_that_cannot_stream_its_probe_over_many_batches_names_the_node_that_fixes_it() {
    // A filtered semi join is one legacy call over the whole probe side, so the planner
    // owes it a single batch there — the capability matrix is what says so.
    let join = joining(
        Given::input(one_lane(BatchLayout::SingleBatch), &["k"]),
        Given::input(one_lane(BatchLayout::MultipleBatches), &["fk"]),
        datafusion::common::JoinType::LeftSemi,
        Some(Expr::binary(
            Expr::column(0, "k"),
            BinaryOp::Lt,
            Expr::column(1, "fk"),
            DataType::Boolean,
        )),
        vec![
            join::JoinFilterColumn {
                side: join::JoinSide::Build,
                index: 0,
            },
            join::JoinFilterColumn {
                side: join::JoinSide::Probe,
                index: 0,
            },
        ],
    );
    invalid(
        join.validate_schemas_and_partitions(),
        "cannot stream its probe",
    );
}

#[test]
fn a_join_shape_the_capability_matrix_refuses_is_refused_at_validation() {
    let join = joining(
        Given::input(one_lane(BatchLayout::SingleBatch), &["k"]),
        Given::input(one_lane(BatchLayout::SingleBatch), &["fk"]),
        datafusion::common::JoinType::Left,
        Some(Expr::binary(
            Expr::column(0, "k"),
            BinaryOp::Lt,
            Expr::column(1, "fk"),
            DataType::Boolean,
        )),
        vec![
            join::JoinFilterColumn {
                side: join::JoinSide::Build,
                index: 0,
            },
            join::JoinFilterColumn {
                side: join::JoinSide::Probe,
                index: 0,
            },
        ],
    );
    unsupported(join.validate_schemas_and_partitions(), "#153");
}

#[test]
fn a_join_filter_reference_past_its_map_is_caught_at_plan_time() {
    let join = joining(
        Given::input(one_lane(BatchLayout::SingleBatch), &["k"]),
        Given::input(one_lane(BatchLayout::SingleBatch), &["fk"]),
        datafusion::common::JoinType::Inner,
        Some(Expr::binary(
            Expr::column(0, "k"),
            BinaryOp::Lt,
            Expr::column(1, "fk"),
            DataType::Boolean,
        )),
        vec![join::JoinFilterColumn {
            side: join::JoinSide::Build,
            index: 0,
        }],
    );
    invalid(
        join.validate_schemas_and_partitions(),
        "filter column fk@1 is past the 1 its map has",
    );
}

#[test]
fn a_cross_join_over_several_lanes_names_the_node_that_fixes_it() {
    let join = GpuCrossJoin::new(
        Given::input(hashed(vec![0], 4), &["k"]),
        Given::input(hashed(vec![0], 4), &["fk"]),
        None,
        columns(&["k", "fk"]),
    );
    invalid(
        join.validate_schemas_and_partitions(),
        "both inputs must be one lane, not 4 and 4",
    );
}

#[test]
fn a_left_nested_loop_join_streaming_its_probe_names_the_node_that_fixes_it() {
    // Its finish pass accumulates keys and a predicate join has none, so the probe side
    // has to be whole before the join runs.
    let join = GpuNestedLoopJoin::new(
        Given::input(one_lane(BatchLayout::SingleBatch), &["k"]),
        Given::input(one_lane(BatchLayout::MultipleBatches), &["fk"]),
        NestedLoopJoinType::Left,
        Expr::binary(
            Expr::column(0, "k"),
            BinaryOp::Lt,
            Expr::column(1, "fk"),
            DataType::Boolean,
        ),
        vec![
            join::JoinFilterColumn {
                side: join::JoinSide::Build,
                index: 0,
            },
            join::JoinFilterColumn {
                side: join::JoinSide::Probe,
                index: 0,
            },
        ],
        None,
        columns(&["k", "fk"]),
    );
    invalid(
        join.validate_schemas_and_partitions(),
        "GpuCoalesceAllBatches",
    );
}

#[test]
fn an_interleave_of_branches_with_different_lane_counts_is_refused() {
    let interleave = GpuInterleave::new(
        vec![
            Given::input(hashed(vec![0], 4), &["k"]),
            Given::input(hashed(vec![0], 2), &["k"]),
        ],
        columns(&["k"]),
    );
    invalid(
        interleave.validate_schemas_and_partitions(),
        "same hash distribution",
    );
}

/// The distribution a join declares about its own output, which is the claim nothing
/// downstream re-checks: a co-partitioned join above it is refused or allowed on this.
fn joined_distribution(
    build: (PartitionLayout, &[&str]),
    probe: (PartitionLayout, &[&str]),
    join_type: datafusion::common::JoinType,
    keys: Vec<(u32, u32)>,
    output: &[&str],
) -> KeyDistribution {
    let join = GpuJoin::new(
        Given::input(build.0, build.1),
        Given::input(probe.0, probe.1),
        join_type,
        keys,
        None,
        Vec::new(),
        false,
        None,
        columns(output),
        join::joined_schema(&columns(build.1).fields, &columns(probe.1).fields, join_type),
    );
    input_layout(&join).key_distribution
}

#[test]
fn a_join_over_lanes_nothing_hashed_declares_no_hash() {
    // One lane, no shuffle anywhere below either side: there is no hash, so a claim that
    // rows are where a hash put them is false however the keys are named.
    let unhashed = || (one_lane(BatchLayout::SingleBatch), &["k"][..]);
    assert_eq!(
        joined_distribution(
            unhashed(),
            (one_lane(BatchLayout::MultipleBatches), &["fk"]),
            datafusion::common::JoinType::Inner,
            vec![(0, 0)],
            &["k", "fk"],
        ),
        KeyDistribution::NotSpecified
    );
}

#[test]
fn a_full_join_declares_no_hash_however_its_sides_are_partitioned() {
    // Full pads both ways: an unmatched build row reads NULL where the probe key was and
    // an unmatched probe row reads NULL where the build key was, so neither column holds
    // the value that placed every row.
    assert_eq!(
        joined_distribution(
            (hashed(vec![0], 4), &["k"]),
            (hashed(vec![0], 4), &["fk"]),
            datafusion::common::JoinType::Full,
            vec![(0, 0)],
            &["k", "fk"],
        ),
        KeyDistribution::NotSpecified
    );
}

#[test]
fn a_left_join_names_the_side_whose_rows_are_never_padded() {
    // Its unmatched build rows carry NULL in the probe key, so the build column is the
    // one that holds the placement value for every output row.
    assert_eq!(
        joined_distribution(
            (hashed(vec![0], 4), &["k"]),
            (hashed(vec![0], 4), &["fk"]),
            datafusion::common::JoinType::Left,
            vec![(0, 0)],
            &["k", "fk"],
        ),
        KeyDistribution::ByHash { hash_keys: vec![0] }
    );
}

#[test]
fn a_right_join_names_the_other_one() {
    assert_eq!(
        joined_distribution(
            (hashed(vec![0], 4), &["k"]),
            (hashed(vec![0], 4), &["fk"]),
            datafusion::common::JoinType::Right,
            vec![(0, 0)],
            &["k", "fk"],
        ),
        KeyDistribution::ByHash { hash_keys: vec![1] }
    );
}

#[test]
fn an_inner_join_carries_the_claim_both_its_sides_earned() {
    assert_eq!(
        joined_distribution(
            (hashed(vec![0], 4), &["k"]),
            (hashed(vec![0], 4), &["fk"]),
            datafusion::common::JoinType::Inner,
            vec![(0, 0)],
            &["k", "fk"],
        ),
        KeyDistribution::ByHash { hash_keys: vec![0] }
    );
}

#[test]
fn a_semi_join_carries_the_claim_of_the_side_it_emits() {
    // The output holds the build side alone, at its own ordinals.
    assert_eq!(
        joined_distribution(
            (hashed(vec![1], 4), &["other", "k"]),
            (hashed(vec![0], 4), &["fk"]),
            datafusion::common::JoinType::LeftSemi,
            vec![(1, 0)],
            &["other", "k"],
        ),
        KeyDistribution::ByHash { hash_keys: vec![1] }
    );
}

#[test]
fn sides_hashed_on_different_things_leave_the_output_with_no_claim() {
    // Only one side was scattered on its join key, so the other's rows are not where the
    // first one's are — and an aggregate above would answer per lane for a group that is
    // spread across them.
    assert_eq!(
        joined_distribution(
            (hashed(vec![0], 4), &["k"]),
            (
                PartitionLayout {
                    key_distribution: KeyDistribution::ByHash { hash_keys: vec![1] },
                    batch_layout: BatchLayout::SingleBatch,
                    ..PartitionLayout::new(4)
                },
                &["fk", "other"]
            ),
            datafusion::common::JoinType::Inner,
            vec![(0, 0)],
            &["k", "fk", "other"],
        ),
        KeyDistribution::NotSpecified
    );
}

#[test]
fn a_projection_moves_the_claim_to_where_the_key_actually_lands() {
    // The output ordinal is neither side's: the projection drops a build column, so the
    // build key lands at 0 having been 1 in the crossed table.
    let join = GpuJoin::new(
        Given::input(hashed(vec![1], 4), &["other", "k"]),
        Given::input(hashed(vec![0], 4), &["fk"]),
        datafusion::common::JoinType::Inner,
        vec![(1, 0)],
        None,
        Vec::new(),
        false,
        Some(vec![1, 2]),
        columns(&["k", "fk"]),
        join::joined_schema(
            &columns(&["other", "k"]).fields,
            &columns(&["fk"]).fields,
            datafusion::common::JoinType::Inner,
        ),
    );
    assert_eq!(
        input_layout(&join).key_distribution,
        KeyDistribution::ByHash { hash_keys: vec![0] }
    );
}

#[test]
fn a_key_name_the_output_holds_twice_still_resolves() {
    // Both sides call the key `k`, so a lookup by name has two candidates and can only
    // give up. The ordinal is structural: build width 1, so the probe's key is at 1.
    assert_eq!(
        joined_distribution(
            (hashed(vec![0], 4), &["k"]),
            (hashed(vec![0], 4), &["k"]),
            datafusion::common::JoinType::Right,
            vec![(0, 0)],
            &["k", "k"],
        ),
        KeyDistribution::ByHash { hash_keys: vec![1] }
    );
}

#[test]
fn a_mark_join_carries_the_claim_of_the_side_it_marks() {
    // Its output is the build side plus the boolean, so the build key is still at its own
    // ordinal — a mark join treated as emitting neither side loses a claim it earned, and
    // every co-partitioned node above it then collapses to one lane.
    assert_eq!(
        joined_distribution(
            (hashed(vec![0], 4), &["k"]),
            (hashed(vec![0], 4), &["fk"]),
            datafusion::common::JoinType::LeftMark,
            vec![(0, 0)],
            &["k", "mark"],
        ),
        KeyDistribution::ByHash { hash_keys: vec![0] }
    );
}

#[test]
fn a_semi_joins_projection_is_bounded_by_the_side_it_emits() {
    // A semi join emits the build side alone, so @2 is past its table even though the two
    // sides hold four columns between them — the bound that reads both sides would let it
    // through.
    let join = GpuJoin::new(
        Given::input(one_lane(BatchLayout::SingleBatch), &["k", "other"]),
        Given::input(one_lane(BatchLayout::SingleBatch), &["fk", "spare"]),
        datafusion::common::JoinType::LeftSemi,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        Some(vec![2]),
        columns(&["other"]),
        join::joined_schema(
            &columns(&["k", "other"]).fields,
            &columns(&["fk", "spare"]).fields,
            datafusion::common::JoinType::LeftSemi,
        ),
    );
    invalid(
        join.validate_schemas_and_partitions(),
        "projected column @2 is past the 2 a LeftSemi join emits",
    );
}
