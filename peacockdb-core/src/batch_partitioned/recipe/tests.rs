//! The kinds whose recipe is more than one call, `GpuJoin` first: per join type, the seq
//! set it emits and when each call is made, against the capability matrix. The trivial
//! kinds are not tested here — the plan goldens run them over every corpus query, which is
//! more coverage than a hand-built node would be.

use super::*;
use crate::batch_partitioned::aggregates::{AggCall, PlanAgg};
use crate::batch_partitioned::expr::{BinaryOp, Expr, NamedExpr};
use crate::batch_partitioned::layout::{BatchLayout, ColumnOrder, NodeKind, PartitionLayout};
use crate::batch_partitioned::nodes::aggregate::AggregateBody;
use crate::batch_partitioned::nodes::join::{JoinFilterColumn, JoinSide, NestedLoopJoinType};
use crate::batch_partitioned::nodes::{GpuJoin, GpuNestedLoopJoin};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::JoinType;
use std::any::Any;
use std::sync::Arc;

/// An input whose layout and schema the test writes. A recipe function is handed its
/// node and the schemas that node declares it consumes, so that is all a case needs to
/// build — and a stub is what keeps a case about the join rather than about a subtree.
#[derive(Debug)]
struct Given {
    kind: NodeKind,
}

impl Given {
    fn input(batches: BatchLayout, columns: &[&str]) -> Box<dyn GpuNode> {
        Box::new(Given {
            kind: NodeKind::Intermediate {
                layout: PartitionLayout {
                    batch_layout: batches,
                    ..PartitionLayout::new(1)
                },
                schema: columns_of(columns),
            },
        })
    }
}

impl GpuNode for Given {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn name(&self) -> &'static str {
        "GpuGiven"
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        Vec::new()
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), crate::batch_partitioned::PlanError> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn columns_of(columns: &[&str]) -> Schema {
    Schema::new(Arc::new(ArrowSchema::new(
        columns
            .iter()
            .map(|name| Field::new(*name, DataType::Int64, true))
            .collect::<Vec<Field>>(),
    )))
}

/// `dim(k, label)` joined to `fact(fk, v)`, the pair the capability matrix works its
/// examples on. `filter` is a residual over both sides, which is what moves a mode off
/// the streaming path.
fn join(join_type: JoinType, filter: bool, projection: Option<Vec<u32>>) -> GpuJoin {
    let build = Given::input(BatchLayout::SingleBatch, &["k", "label"]);
    let probe = Given::input(BatchLayout::MultipleBatches, &["fk", "v"]);
    let (residual, columns) = if filter {
        (
            Some(Expr::binary(
                Expr::column(0, "label"),
                BinaryOp::Lt,
                Expr::column(1, "v"),
                DataType::Boolean,
            )),
            vec![
                JoinFilterColumn {
                    side: JoinSide::Build,
                    index: 1,
                },
                JoinFilterColumn {
                    side: JoinSide::Probe,
                    index: 1,
                },
            ],
        )
    } else {
        (None, Vec::new())
    };
    GpuJoin::new(
        build,
        probe,
        join_type,
        vec![(0, 0)],
        residual,
        columns,
        false,
        projection,
        columns_of(&["k", "label", "fk", "v"]),
    )
}

fn recipe_for(node: &GpuJoin) -> Recipe {
    let build = columns_of(&["k", "label"]);
    let probe = columns_of(&["fk", "v"]);
    join::hash_join(node, &[&build, &probe], &mut Seqs::default()).expect("a join drives the ABI")
}

/// `(kind, when)` per call, which is the pair the mapping table states.
fn shape(recipe: &Recipe) -> Vec<(FbKind, CallPattern)> {
    recipe
        .calls
        .iter()
        .map(|call| {
            (
                call.target.expect("a join call addresses a seq").1,
                call.when,
            )
        })
        .collect()
}

fn hash_join(join_type: JoinType) -> FbKind {
    FbKind::HashJoin { join_type }
}

#[test]
fn a_probe_local_join_is_one_call_per_batch_against_a_copy_of_the_build_side() {
    // Inner, Right and the probe-side semi family: every emitted row is decided by
    // (build, this batch), so nothing accumulates and there is no finish.
    for join_type in [
        JoinType::Inner,
        JoinType::Right,
        JoinType::RightSemi,
        JoinType::RightAnti,
    ] {
        let recipe = recipe_for(&join(join_type, false, None));
        assert_eq!(
            shape(&recipe),
            vec![(hash_join(join_type), CallPattern::PerProbeBatch)],
            "{join_type:?}"
        );
        assert_eq!(
            recipe.calls[0].inputs,
            vec![Input::BuildSideCopy, Input::Batch],
            "{join_type:?}: a streamed probe needs the build side again next batch (#152)"
        );
    }
}

/// Full outer reaches the goldens nowhere — no corpus query plans one — so this is the
/// only place its five calls are checked at all.
#[test]
fn an_outer_join_that_preserves_its_build_side_keeps_the_keys_and_finishes_with_an_anti_join() {
    // Left emits this batch's matches as an Inner; Full also emits the probe rows this
    // batch had no match for, which is batch-local because the build side is complete.
    for (join_type, per_call) in [
        (JoinType::Left, JoinType::Inner),
        (JoinType::Full, JoinType::Right),
    ] {
        let recipe = recipe_for(&join(join_type, false, None));
        assert_eq!(
            shape(&recipe),
            vec![
                (
                    FbKind::Project(ProjectRole::ProbeKeys),
                    CallPattern::PerProbeBatch
                ),
                (hash_join(per_call), CallPattern::PerProbeBatch),
                (FbKind::CoalescePartitions, CallPattern::AtDone),
                (hash_join(JoinType::LeftAnti), CallPattern::AtDone),
                (
                    FbKind::Project(ProjectRole::NullPad { nulls: 2 }),
                    CallPattern::AtDone
                ),
            ],
            "{join_type:?}"
        );
        assert_eq!(
            recipe.calls[0].inputs,
            vec![Input::BatchCopy],
            "{join_type:?}: the join below consumes the batch, so the keys come off a copy"
        );
        assert_eq!(
            recipe.calls[3].inputs,
            vec![Input::BuildSide, Input::PriorOutput],
            "{join_type:?}: the finish is the last use of the build side"
        );
    }
}

#[test]
fn the_build_side_semi_family_makes_no_join_call_until_its_finish() {
    for join_type in [JoinType::LeftSemi, JoinType::LeftAnti, JoinType::LeftMark] {
        let recipe = recipe_for(&join(join_type, false, None));
        assert_eq!(
            shape(&recipe),
            vec![
                (
                    FbKind::Project(ProjectRole::ProbeKeys),
                    CallPattern::PerProbeBatch
                ),
                (FbKind::CoalescePartitions, CallPattern::AtDone),
                (hash_join(join_type), CallPattern::AtDone),
            ],
            "{join_type:?}"
        );
        // The claim the matrix makes about this family, as a check rather than a
        // sentence: its per-batch call touches neither the build side nor a copy of it,
        // which is why it streams at no copy cost at all.
        assert_eq!(recipe.calls[0].inputs, vec![Input::Batch], "{join_type:?}");
        assert!(
            !recipe.calls[0]
                .inputs
                .iter()
                .any(|input| matches!(input, Input::BuildSide | Input::BuildSideCopy)),
            "{join_type:?}: the build side is not touched until the finish"
        );
    }
}

#[test]
fn a_single_batch_probe_is_the_legacy_call_and_hands_the_build_side_over() {
    // A residual filter takes the build-side semi family off the streaming path, and the
    // planner puts a GpuCoalesceAllBatches under the probe: one node, one call, no finish.
    let recipe = recipe_for(&join(JoinType::LeftSemi, true, None));
    assert_eq!(
        shape(&recipe),
        vec![(hash_join(JoinType::LeftSemi), CallPattern::PerProbeBatch)]
    );
    assert_eq!(
        recipe.calls[0].inputs,
        vec![Input::BuildSide, Input::Batch],
        "one call means the build handle is handed over rather than copied"
    );
}

#[test]
fn the_pad_project_appends_one_null_per_probe_column_the_projection_keeps() {
    // Without a projection every probe column is kept, so both are padded. With one, the
    // ordinals at or above the build width are the probe's — here `v` alone.
    let kinds = |projection: Option<Vec<u32>>| {
        shape(&recipe_for(&join(JoinType::Left, false, projection)))
            .into_iter()
            .filter_map(|(kind, _)| match kind {
                FbKind::Project(ProjectRole::NullPad { nulls }) => Some(nulls),
                _ => None,
            })
            .collect::<Vec<usize>>()
    };
    assert_eq!(kinds(None), vec![2]);
    assert_eq!(kinds(Some(vec![0, 3])), vec![1]);
    assert_eq!(
        kinds(Some(vec![0, 1])),
        vec![0],
        "a projection that keeps no probe column pads nothing"
    );
}

#[test]
fn seqs_ascend_with_call_order_and_no_two_calls_share_one() {
    let mut seqs = Seqs::default();
    let build = columns_of(&["k", "label"]);
    let probe = columns_of(&["fk", "v"]);
    let first = join::hash_join(
        &join(JoinType::Left, false, None),
        &[&build, &probe],
        &mut seqs,
    )
    .expect("a join drives the ABI");
    let second = join::hash_join(
        &join(JoinType::Inner, false, None),
        &[&build, &probe],
        &mut seqs,
    )
    .expect("a join drives the ABI");
    assert_eq!(first.seqs(), vec![0, 1, 2, 3, 4]);
    assert_eq!(
        second.seqs(),
        vec![5],
        "seqs are handed out across the plan, not per node"
    );
}

#[test]
fn a_nested_loop_join_copies_its_build_side_only_where_the_probe_streams() {
    let nested_loop = |join_type| {
        let build = Given::input(BatchLayout::SingleBatch, &["k"]);
        let batches = match join_type {
            NestedLoopJoinType::Left => BatchLayout::SingleBatch,
            NestedLoopJoinType::Inner => BatchLayout::MultipleBatches,
        };
        let probe = Given::input(batches, &["fk"]);
        GpuNestedLoopJoin::new(
            build,
            probe,
            join_type,
            Expr::binary(
                Expr::column(0, "k"),
                BinaryOp::Lt,
                Expr::column(1, "fk"),
                DataType::Boolean,
            ),
            vec![
                JoinFilterColumn {
                    side: JoinSide::Build,
                    index: 0,
                },
                JoinFilterColumn {
                    side: JoinSide::Probe,
                    index: 0,
                },
            ],
            None,
            columns_of(&["k", "fk"]),
        )
    };
    let inputs_of = |join_type| {
        let node = nested_loop(join_type);
        let schema = columns_of(&["k"]);
        join::nested_loop_join(&node, &[&schema, &schema], &mut Seqs::default())
            .expect("a nested-loop join drives the ABI")
            .calls[0]
            .inputs
            .clone()
    };
    assert_eq!(
        inputs_of(NestedLoopJoinType::Inner),
        vec![Input::BuildSideCopy, Input::Batch]
    );
    assert_eq!(
        inputs_of(NestedLoopJoinType::Left),
        vec![Input::BuildSide, Input::Batch],
        "a single-batch probe is one call, so the build side is handed over"
    );
}

#[test]
fn an_accumulating_sort_sorts_every_batch_and_merges_what_the_lane_kept() {
    let node = GpuAccumulateBatchesAndSort::new(
        Given::input(BatchLayout::MultipleBatches, &["a"]),
        vec![ColumnOrder {
            column: 0,
            ascending: true,
            nulls_first: false,
        }],
        Some(10),
    );
    let recipe = accumulate_and_sort(&node, &[&columns_of(&["a"])], &mut Seqs::default())
        .expect("an accumulating sort drives the ABI");
    assert_eq!(
        recipe
            .calls
            .iter()
            .map(|call| (call.target.unwrap().1, call.when, call.inputs.clone()))
            .collect::<Vec<_>>(),
        vec![
            (FbKind::Sort, CallPattern::PerBatch, vec![Input::Batch]),
            (
                FbKind::SortPreservingMerge,
                CallPattern::AtDone,
                vec![Input::LaneBatches]
            ),
        ]
    );
}

#[test]
fn a_batch_aggregate_runs_the_same_pair_at_a_compaction_and_at_done() {
    let body = AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![AggCall {
            func: PlanAgg::Sum,
            args: vec![Expr::column(1, "n")],
            outputs: vec![Field::new("sum(n)", DataType::Int64, true)],
        }],
        finalize: None,
    };
    let node = GpuAggregateBatches::new(
        Given::input(BatchLayout::MultipleBatches, &["k", "n"]),
        body,
        columns_of(&["k", "sum(n)"]),
        columns_of(&["k", "sum(n)"]),
    );
    let recipe = aggregate_batches(&node, &[&columns_of(&["k", "n"])], &mut Seqs::default())
        .expect("a batch aggregate drives the ABI");
    assert_eq!(
        shape(&recipe),
        vec![
            (FbKind::CoalescePartitions, CallPattern::PerCompaction),
            (FbKind::Aggregate, CallPattern::PerCompaction),
        ],
        "the compaction is the done pass run early, which is what makes the threshold a \
         scheduling decision"
    );
    assert_eq!(recipe.calls[1].inputs, vec![Input::PriorOutput]);
}

/// A finalizing merge is the same pair — what a node emits is not what it calls.
#[test]
fn the_finalizing_form_of_a_batch_aggregate_emits_the_same_seq_set() {
    let with_finalize = AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![AggCall {
            func: PlanAgg::Sum,
            args: vec![Expr::column(1, "n")],
            outputs: vec![Field::new("sum(n)", DataType::Int64, true)],
        }],
        finalize: Some(vec![NamedExpr::new(Expr::column(1, "sum(n)"), "sum(n)")]),
    };
    let node = GpuAggregateBatches::new(
        Given::input(BatchLayout::MultipleBatches, &["k", "n"]),
        with_finalize,
        columns_of(&["k", "sum(n)"]),
        columns_of(&["k", "sum(n)"]),
    );
    let recipe = aggregate_batches(&node, &[&columns_of(&["k", "n"])], &mut Seqs::default())
        .expect("a batch aggregate drives the ABI");
    assert_eq!(
        shape(&recipe),
        vec![
            (FbKind::CoalescePartitions, CallPattern::PerCompaction),
            (FbKind::Aggregate, CallPattern::PerCompaction),
        ]
    );
}
