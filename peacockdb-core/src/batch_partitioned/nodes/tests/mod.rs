//! One input per rule, each built to break the rule it is aimed at. A guard that cannot
//! go red is not a guard, and a plan that violates one of these is unreachable from sql
//! precisely because the translation layer is what inserts the fix — so a hand-built
//! input is the only thing that can show the guard working.

use super::*;
use crate::batch_partitioned::aggregates::{AggCall, AggFunc, PlanAgg};
use crate::batch_partitioned::expr::{BinaryOp, NamedExpr};
use crate::batch_partitioned::layout::{BatchLayout, ColumnOrder, NodeKind};
use crate::batch_partitioned::node::RowInterval;
use crate::batch_partitioned::nodes::join::NestedLoopJoinType;
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use std::any::Any;
use std::sync::Arc;

/// An input with a layout and schema chosen by the test: the guards below are about
/// what a node requires of its input, and a plan that violates one is unreachable
/// from sql precisely because the translation layer is what inserts the fix.
#[derive(Debug)]
struct Given {
    kind: NodeKind,
}

impl Given {
    fn input(layout: PartitionLayout, columns: &[&str]) -> Box<dyn GpuNode> {
        let fields: Vec<Field> = columns
            .iter()
            .map(|name| Field::new(*name, DataType::Int64, true))
            .collect();
        let schema = Schema::new(Arc::new(ArrowSchema::new(fields)));
        Box::new(Given {
            kind: NodeKind::Intermediate { layout, schema },
        })
    }
}

impl GpuNode for Given {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        Vec::new()
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn one_lane(batches: BatchLayout) -> PartitionLayout {
    PartitionLayout {
        batch_layout: batches,
        ..PartitionLayout::new(1)
    }
}

fn invalid(result: Result<(), PlanError>, mentions: &str) {
    match result {
        Err(PlanError::Invalid(what)) => assert!(
            what.contains(mentions),
            "the error names the wrong fix: {what}"
        ),
        other => panic!("expected an invalid plan naming {mentions}, got {other:?}"),
    }
}

mod joins;

#[test]
fn a_reference_past_its_inputs_columns_is_caught_at_plan_time() {
    let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["a", "b"]);
    let schema = Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
        "a",
        DataType::Int64,
        true,
    )])));
    let filter = GpuFilter::new(input, Expr::column(5, "a"), None, schema);
    invalid(
        filter.validate_schemas_and_partitions(),
        "past the 2 columns",
    );
}

#[test]
fn a_reference_whose_name_does_not_match_its_position_is_caught_at_plan_time() {
    // The rebasing the layer does at every inserted node is what makes this the
    // likely slip: the ordinal stays valid and starts reading a different column.
    let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["a", "b"]);
    let project = GpuProject::new(
        input,
        vec![NamedExpr::new(Expr::column(1, "a"), "a")],
        Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int64,
            true,
        )]))),
    );
    invalid(
        project.validate_schemas_and_partitions(),
        "reads b at that position",
    );
}

#[test]
fn a_limit_over_several_lanes_names_the_node_that_fixes_it() {
    let input = Given::input(PartitionLayout::new(4), &["a"]);
    let limit = GpuLimit::new(
        input,
        RowInterval {
            skip: 0,
            fetch: Some(10),
        },
    );
    invalid(
        limit.validate_schemas_and_partitions(),
        "GpuMergePartitions",
    );
}

#[test]
fn an_accumulating_sort_over_unsorted_batches_names_the_node_that_fixes_it() {
    let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["a"]);
    let keys = vec![ColumnOrder {
        column: 0,
        ascending: true,
        nulls_first: false,
    }];
    let accumulator = GpuAccumulateBatchesAndSort::new(input, keys, None);
    invalid(accumulator.validate_schemas_and_partitions(), "GpuSort");
}

#[test]
fn a_scatter_over_several_lanes_names_the_node_that_fixes_it() {
    let input = Given::input(PartitionLayout::new(4), &["k"]);
    let emit = GpuEmitPartitions::new(input, vec![0], 4);
    invalid(emit.validate_schemas_and_partitions(), "GpuMergePartitions");
}

#[test]
fn a_union_branch_of_another_type_names_the_cast_that_fixes_it() {
    // Routing cannot retype anything, so a branch that does not already match the
    // declared output is a missing project rather than work for the executor (#41).
    let wide = Given::input(one_lane(BatchLayout::MultipleBatches), &["n"]);
    let narrow: Box<dyn GpuNode> = Box::new(Given {
        kind: NodeKind::Intermediate {
            layout: one_lane(BatchLayout::MultipleBatches),
            schema: Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
                "n",
                DataType::Int32,
                true,
            )]))),
        },
    });
    let declared = Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
        "n",
        DataType::Int64,
        true,
    )])));
    let union = GpuUnion::new(vec![wide, narrow], declared);
    invalid(
        union.validate_schemas_and_partitions(),
        "casting GpuProject",
    );
}

#[test]
fn an_interleave_of_differently_hashed_branches_is_refused() {
    let hashed = |keys: Vec<u32>| PartitionLayout {
        key_distribution: KeyDistribution::ByHash { hash_keys: keys },
        ..PartitionLayout::new(4)
    };
    let schema = Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
        "k",
        DataType::Int64,
        true,
    )])));
    let interleave = GpuInterleave::new(
        vec![
            Given::input(hashed(vec![0]), &["k"]),
            Given::input(hashed(vec![1]), &["k"]),
        ],
        schema,
    );
    // Lane p is lane p of every branch, so a branch hashed on another key would put
    // rows that cannot meet into the same lane.
    invalid(
        interleave.validate_schemas_and_partitions(),
        "same hash distribution",
    );
}

#[test]
fn a_finalizing_merge_over_lanes_hashed_on_other_columns_is_refused() {
    let hashed = PartitionLayout {
        key_distribution: KeyDistribution::ByHash { hash_keys: vec![1] },
        ..PartitionLayout::new(4)
    };
    let input = Given::input(hashed, &["k", "other", "n"]);
    let schema = Schema::new(Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("n", DataType::Int64, true),
    ])));
    let body = AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![AggCall {
            func: PlanAgg::Sum,
            args: vec![Expr::column(2, "n")],
            outputs: vec![Field::new("n", DataType::Int64, true)],
        }],
        finalize: Some(vec![NamedExpr::new(Expr::column(1, "n"), "n")]),
    };
    let intermediate = Schema::new(Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("n", DataType::Int64, true),
    ])));
    let merge = GpuAggregateBatches::new(input, body, intermediate, schema);
    // Hashed on a column it does not group by, so a group's rows are spread across
    // lanes and each lane would answer for part of it.
    invalid(
        merge.validate_schemas_and_partitions(),
        "subset of its group columns",
    );
}

fn unsupported(result: Result<(), PlanError>, mentions: &str) {
    match result {
        Err(PlanError::Unsupported(what)) => assert!(
            what.contains(mentions),
            "the refusal names the wrong shape: {what}"
        ),
        other => panic!("expected an unsupported shape naming {mentions}, got {other:?}"),
    }
}

fn columns(names: &[&str]) -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(
        names
            .iter()
            .map(|name| Field::new(*name, DataType::Int64, true))
            .collect::<Vec<Field>>(),
    )))
}

fn ascending(column: u32) -> ColumnOrder {
    ColumnOrder {
        column,
        ascending: true,
        nulls_first: false,
    }
}

/// Sorted within each batch and not across them — what a `GpuSort` alone leaves, and the
/// shape a prefix of the stream is not the top of.
fn batch_sorted_lane() -> PartitionLayout {
    PartitionLayout {
        sort_order: SortOrder::batch_sorted(vec![ascending(0)]),
        ..one_lane(BatchLayout::MultipleBatches)
    }
}

#[test]
fn a_scan_the_partitioner_gave_no_lanes_is_caught_at_plan_time() {
    let scan = crate::batch_partitioned::parquet_meta::ScanMetadata {
        file: "/nation.parquet".to_string(),
        groups: Vec::new(),
        can_be_null: vec![false],
    };
    let load = GpuLoadParquet::new(
        "nation".to_string(),
        vec![0],
        Vec::new(),
        &scan,
        None,
        columns(&["n_nationkey"]),
    );
    invalid(
        load.validate_schemas_and_partitions(),
        "the partitioner returned no lanes",
    );
}

#[test]
fn a_sort_on_no_keys_is_caught_at_plan_time() {
    let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["a"]);
    let sort = GpuSort::new(input, Vec::new(), None);
    invalid(sort.validate_schemas_and_partitions(), "no sort keys");
}

#[test]
fn a_sort_key_past_its_inputs_columns_is_caught_at_plan_time() {
    let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["a"]);
    let sort = GpuSort::new(input, vec![ascending(3)], None);
    invalid(
        sort.validate_schemas_and_partitions(),
        "sort key @3 is past the 1 columns",
    );
}

#[test]
fn a_limit_over_batches_sorted_only_within_themselves_names_the_node_that_fixes_it() {
    // The rows a limit names depend on the order they arrive in, so this shape reads as
    // ordered and is not: the prefix is whatever the batch boundaries made it.
    let limit = GpuLimit::new(
        Given::input(batch_sorted_lane(), &["a"]),
        RowInterval {
            skip: 0,
            fetch: Some(10),
        },
    );
    invalid(
        limit.validate_schemas_and_partitions(),
        "GpuAccumulateBatchesAndSort",
    );
}

#[test]
fn an_unload_carrying_an_interval_owes_the_same_ordered_prefix() {
    let carried = GpuUnload::new(
        Given::input(batch_sorted_lane(), &["a"]),
        Some(RowInterval {
            skip: 0,
            fetch: Some(10),
        }),
    );
    invalid(
        carried.validate_schemas_and_partitions(),
        "GpuAccumulateBatchesAndSort",
    );
    // The rule is the interval's, not the unload's: without one there is no prefix to be
    // wrong about, and an unordered stream is unaffected either way.
    let none = GpuUnload::new(Given::input(batch_sorted_lane(), &["a"]), None);
    assert_eq!(none.validate_schemas_and_partitions(), Ok(()));
}

fn summing(
    group_by: Vec<Expr>,
    args: Vec<Expr>,
    finalize: Option<Vec<NamedExpr>>,
) -> AggregateBody {
    AggregateBody {
        group_by,
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![AggCall {
            func: PlanAgg::Sum,
            args,
            outputs: vec![Field::new("n", DataType::Int64, true)],
        }],
        finalize,
    }
}

#[test]
fn a_group_key_past_the_aggregates_input_is_caught_at_plan_time() {
    let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["k", "n"]);
    let intermediate = columns(&["k", "n"]);
    let aggregate = GpuAggregate::new(
        input,
        summing(vec![Expr::column(4, "k")], vec![Expr::column(1, "n")], None),
        intermediate.clone(),
        intermediate,
    );
    invalid(
        aggregate.validate_schemas_and_partitions(),
        "GpuAggregate: column k@4 is past the 2 columns",
    );
}

#[test]
fn an_aggregators_argument_reading_another_column_is_caught_at_plan_time() {
    let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["k", "n"]);
    let intermediate = columns(&["k", "n"]);
    let aggregate = GpuAggregate::new(
        input,
        summing(vec![Expr::column(0, "k")], vec![Expr::column(0, "n")], None),
        intermediate.clone(),
        intermediate,
    );
    invalid(
        aggregate.validate_schemas_and_partitions(),
        "GpuAggregate: column n@0 reads k at that position",
    );
}

#[test]
fn a_hash_key_past_the_scattered_columns_is_caught_at_plan_time() {
    let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["k"]);
    let emit = GpuEmitPartitions::new(input, vec![2], 4);
    invalid(
        emit.validate_schemas_and_partitions(),
        "hash key @2 is past the 1 columns",
    );
}

#[test]
fn a_sorted_merge_over_unsorted_batches_names_the_node_that_fixes_it() {
    let input = Given::input(PartitionLayout::new(4), &["a"]);
    let merge = GpuMergeSortedPartitions::new(input, vec![ascending(0)], None);
    invalid(merge.validate_schemas_and_partitions(), "GpuSort");
}

#[test]
fn a_union_branch_emitting_fewer_columns_than_the_output_is_refused() {
    let union = GpuUnion::new(
        vec![
            Given::input(one_lane(BatchLayout::MultipleBatches), &["a", "b"]),
            Given::input(one_lane(BatchLayout::MultipleBatches), &["a"]),
        ],
        columns(&["a", "b"]),
    );
    invalid(
        union.validate_schemas_and_partitions(),
        "branch 1 has 1 columns and the output declares 2",
    );
}

#[test]
fn a_merge_on_keys_its_input_is_not_sorted_by_is_refused() {
    // Batch-sorted is not the rule — sorted on THESE keys is. A merge takes the smallest
    // head row of each input, so on the wrong key it emits rows in no order and then
    // declares them sorted, which is what a top-N above it acts on.
    let sorted_on_a = PartitionLayout {
        sort_order: SortOrder::batch_sorted(vec![ascending(0)]),
        ..PartitionLayout::new(4)
    };
    let merge = GpuMergeSortedPartitions::new(
        Given::input(sorted_on_a, &["a", "b"]),
        vec![ascending(1)],
        None,
    );
    invalid(
        merge.validate_schemas_and_partitions(),
        "it merges on @1 asc nulls last at position 0 where its input's batches are sorted on @0",
    );
}

#[test]
fn a_merge_that_disagrees_about_direction_is_refused() {
    let descending = ColumnOrder {
        column: 0,
        ascending: false,
        nulls_first: false,
    };
    let sorted_ascending = PartitionLayout {
        sort_order: SortOrder::batch_sorted(vec![ascending(0)]),
        ..one_lane(BatchLayout::MultipleBatches)
    };
    let accumulator = GpuAccumulateBatchesAndSort::new(
        Given::input(sorted_ascending, &["a"]),
        vec![descending],
        None,
    );
    invalid(
        accumulator.validate_schemas_and_partitions(),
        "it merges on @0 desc nulls last at position 0",
    );
}

#[test]
fn a_merge_on_more_keys_than_its_input_carries_is_refused() {
    let sorted_on_one = PartitionLayout {
        sort_order: SortOrder::batch_sorted(vec![ascending(0)]),
        ..PartitionLayout::new(4)
    };
    let merge = GpuMergeSortedPartitions::new(
        Given::input(sorted_on_one, &["a", "b"]),
        vec![ascending(0), ascending(1)],
        None,
    );
    invalid(
        merge.validate_schemas_and_partitions(),
        "it merges on 2 keys and its input's batches are sorted on 1",
    );
}

#[test]
fn a_merge_on_a_prefix_of_its_inputs_order_is_allowed() {
    // Batches sorted on [a, b] merged on [a] come out ordered by a, which is what the
    // node then declares.
    let sorted_on_two = PartitionLayout {
        sort_order: SortOrder::batch_sorted(vec![ascending(0), ascending(1)]),
        ..PartitionLayout::new(4)
    };
    let merge = GpuMergeSortedPartitions::new(
        Given::input(sorted_on_two, &["a", "b"]),
        vec![ascending(0)],
        None,
    );
    assert_eq!(merge.validate_schemas_and_partitions(), Ok(()));
}

/// An input declaring one aggregate's state at the positions given, as a partial's output
/// is `[group key, state columns…]`.
fn declaring_state(func: AggFunc, columns: &[&str], positions: Vec<u32>) -> Box<dyn GpuNode> {
    let mut schema = self::columns(columns);
    schema.group_keys = vec![0];
    schema.agg_state = vec![crate::batch_partitioned::schema::AggStateColumns {
        output: "n".to_string(),
        func,
        ddof: 0,
        positions,
    }];
    Box::new(Given {
        kind: NodeKind::Intermediate {
            layout: one_lane(BatchLayout::MultipleBatches),
            schema,
        },
    })
}

fn merging(input: Box<dyn GpuNode>, aggs: Vec<AggCall>) -> GpuAggregateBatches {
    let intermediate = columns(&["k", "n"]);
    GpuAggregateBatches::new(
        input,
        AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs,
            finalize: None,
        },
        intermediate.clone(),
        intermediate,
    )
}

fn call(func: PlanAgg, reads: u32, name: &str) -> AggCall {
    AggCall {
        func,
        args: vec![Expr::column(reads, name)],
        outputs: vec![Field::new("n", DataType::Int64, true)],
    }
}

#[test]
fn a_count_merged_by_counting_it_again_is_refused() {
    // The one exception in the decomposition table: a count merges by SUM, because what
    // arrives is already counts. Counting them counts the groups instead of the rows.
    let merge = merging(
        declaring_state(AggFunc::Count, &["k", "n"], vec![1]),
        vec![call(PlanAgg::Count, 1, "n")],
    );
    invalid(
        merge.validate_schemas_and_partitions(),
        "@1 is n state and is merged by count rather than sum",
    );
}

#[test]
fn state_columns_the_decomposition_does_not_merge_are_refused() {
    // avg decomposes into two columns, so an input declaring one of them has either lost
    // a column or annotated a different aggregate's state.
    let merge = merging(
        declaring_state(AggFunc::Avg, &["k", "n"], vec![1]),
        vec![call(PlanAgg::Sum, 1, "n")],
    );
    invalid(
        merge.validate_schemas_and_partitions(),
        "n declares 1 state columns and its decomposition merges 2",
    );
}

#[test]
fn a_state_column_nothing_merges_is_refused() {
    // avg decomposes into two columns and this merge reads only one of them, so the
    // count would arrive at the finalize unmerged.
    let merge = merging(
        declaring_state(AggFunc::Avg, &["k", "n", "c"], vec![1, 2]),
        vec![call(PlanAgg::Sum, 1, "n")],
    );
    invalid(
        merge.validate_schemas_and_partitions(),
        "nothing merges @2, which its input declares as n state",
    );
}

#[test]
fn a_welford_triple_merged_column_by_column_is_refused() {
    // Its three columns merge in one call: the merged mean needs the counts and the
    // cross term, so summing each column on its own is a different answer.
    let merge = merging(
        declaring_state(AggFunc::Stddev, &["k", "c", "m", "s"], vec![1, 2, 3]),
        vec![
            call(PlanAgg::Sum, 1, "c"),
            call(PlanAgg::Sum, 2, "m"),
            call(PlanAgg::Sum, 3, "s"),
        ],
    );
    invalid(
        merge.validate_schemas_and_partitions(),
        "merges its 3 state columns in one call",
    );
}

#[test]
fn a_merge_that_reads_the_declared_state_with_the_declared_aggregators_passes() {
    let merge = merging(
        declaring_state(AggFunc::Avg, &["k", "n", "c"], vec![1, 2]),
        vec![call(PlanAgg::Sum, 1, "n"), call(PlanAgg::Sum, 2, "c")],
    );
    assert_eq!(merge.validate_schemas_and_partitions(), Ok(()));
}

#[test]
fn a_union_branch_naming_its_columns_differently_is_refused() {
    // The declared names are what every name@ordinal check above the union resolves
    // against, so a branch that names the same column something else makes one of the two
    // readings wrong wherever they meet.
    let union = GpuUnion::new(
        vec![
            Given::input(one_lane(BatchLayout::MultipleBatches), &["a"]),
            Given::input(one_lane(BatchLayout::MultipleBatches), &["b"]),
        ],
        columns(&["a"]),
    );
    invalid(
        union.validate_schemas_and_partitions(),
        "branch 1 emits b as Int64 where the output declares a",
    );
}

fn loading(partition_groups: Vec<Vec<Vec<u32>>>, survivors: Vec<u32>) -> GpuLoadParquet {
    let scan = crate::batch_partitioned::parquet_meta::ScanMetadata {
        file: "/nation.parquet".to_string(),
        groups: survivors
            .into_iter()
            .map(
                |index| crate::batch_partitioned::partitioner::RowGroupMeta {
                    index,
                    rows: 100,
                    bytes: 1000,
                },
            )
            .collect(),
        can_be_null: vec![false],
    };
    GpuLoadParquet::new(
        "nation".to_string(),
        vec![0],
        partition_groups,
        &scan,
        None,
        columns(&["n_nationkey"]),
    )
}

#[test]
fn a_batch_reading_no_row_group_is_refused() {
    invalid(
        loading(vec![vec![vec![0]], vec![Vec::new()]], vec![0, 1])
            .validate_schemas_and_partitions(),
        "lane 1 batch 0 reads no row group",
    );
}

#[test]
fn a_lane_reading_a_row_group_pruning_left_out_is_refused() {
    invalid(
        loading(vec![vec![vec![0, 3]]], vec![0, 1]).validate_schemas_and_partitions(),
        "lane 0 reads row group 3, which pruning left out",
    );
}

#[test]
fn a_lane_with_no_batches_is_what_more_lanes_than_row_groups_looks_like() {
    // Four lanes over two row groups: the mapping says two of them read nothing rather
    // than inventing work, so this is the shape the goldens carry and not a defect.
    assert_eq!(
        loading(
            vec![vec![vec![0]], vec![vec![1]], Vec::new(), Vec::new()],
            vec![0, 1]
        )
        .validate_schemas_and_partitions(),
        Ok(())
    );
}
