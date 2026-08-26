//! The kinds whose recipe is more than one call, `GpuJoin` first: per join type, the seq
//! set it emits and when each call is made, against the capability matrix. The trivial
//! kinds are not tested here — the plan goldens run them over every corpus query, which is
//! more coverage than a hand-built node would be.

use super::writer::Writer;
use super::*;
use crate::batch_partitioned::aggregates::{AggCall, PlanAgg};
use crate::batch_partitioned::expr::{BinaryOp, Expr, NamedExpr};
use crate::batch_partitioned::layout::{BatchLayout, ColumnOrder, NodeKind, PartitionLayout};
use crate::batch_partitioned::nodes::GpuFilter;
use crate::batch_partitioned::nodes::aggregate::AggregateBody;
use crate::batch_partitioned::nodes::join::{JoinFilterColumn, JoinSide, NestedLoopJoinType};
use crate::batch_partitioned::nodes::{GpuAggregate, GpuJoin, GpuNestedLoopJoin};
use crate::generated::gpu_plan_generated::peacock::plan as fb;
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::JoinType;
use datafusion::common::ScalarValue;
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
    join::hash_join(node, &[&build, &probe], &mut Writer::new())
        .expect("the join's payloads are writable")
        .expect("a join drives the ABI")
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

/// Seqs are the post-order positions of the tree that was built, stubs included — which
/// is why they are not dense, and why they cannot be counted from the recipes alone.
#[test]
fn seqs_are_the_post_order_positions_of_what_was_built() {
    let mut writer = Writer::new();
    let build = columns_of(&["k", "label"]);
    let probe = columns_of(&["fk", "v"]);
    let first = join::hash_join(
        &join(JoinType::Left, false, None),
        &[&build, &probe],
        &mut writer,
    )
    .expect("writable")
    .expect("a join drives the ABI");
    let second = join::hash_join(
        &join(JoinType::Inner, false, None),
        &[&build, &probe],
        &mut writer,
    )
    .expect("writable")
    .expect("a join drives the ABI");
    // #0 and #2 are the stubs the key project and the per-batch join took for slots
    // nothing else filled; #5 is the finish join's.
    assert_eq!(first.seqs(), vec![1, 3, 4, 6, 7]);
    assert_eq!(
        second.seqs(),
        vec![9],
        "seqs run on across the plan rather than restarting per node"
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
        join::nested_loop_join(&node, &[&schema, &schema], &mut Writer::new())
            .expect("the join's payload is writable")
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
    let recipe = accumulate_and_sort(&node, &[&columns_of(&["a"])], &mut Writer::new())
        .expect("the sort's payloads are writable")
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
    let recipe = aggregate_batches(&node, &[&columns_of(&["k", "n"])], &mut Writer::new())
        .expect("the aggregate's payloads are writable")
        .expect("a batch aggregate drives the ABI");
    assert_eq!(
        shape(&recipe),
        vec![
            (FbKind::CoalescePartitions, CallPattern::PerCompaction),
            (
                FbKind::Aggregate { merge: true },
                CallPattern::PerCompaction
            ),
        ],
        "the compaction is the done pass run early, which is what makes the threshold a \
         scheduling decision"
    );
    assert_eq!(recipe.calls[1].inputs, vec![Input::PriorOutput]);
}

/// A finalizing merge adds the project that carries the finalize — ours, so the two
/// engines evaluate the same expression rather than agreeing because two implementations
/// happen to match.
/// One key and one summed column, finalized: the shape the three tests below read, so the
/// list a finalize carries and the row a project has to emit differ by exactly the key.
fn finalizing_merge(finalize: Vec<NamedExpr>, state: &[&str]) -> GpuAggregateBatches {
    GpuAggregateBatches::new(
        Given::input(BatchLayout::MultipleBatches, &["k", "n"]),
        AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![AggCall {
                func: PlanAgg::Sum,
                args: vec![Expr::column(1, "n")],
                outputs: vec![Field::new("sum(n)", DataType::Int64, true)],
            }],
            finalize: Some(finalize),
        },
        columns_of(state),
        columns_of(&["k", "sum(n)"]),
    )
}

/// The other node that finalizes: the translation hands its finalize to the init itself
/// where one batch in one lane is already the whole of every group, so no merge exists to
/// carry it.
fn finalizing_init(finalize: Vec<NamedExpr>, state: &[&str]) -> GpuAggregate {
    GpuAggregate::new(
        Given::input(BatchLayout::SingleBatch, &["k", "n"]),
        AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![AggCall {
                func: PlanAgg::Sum,
                args: vec![Expr::column(1, "n")],
                outputs: vec![Field::new("sum(n)", DataType::Int64, true)],
            }],
            finalize: Some(finalize),
        },
        columns_of(state),
        columns_of(&["k", "sum(n)"]),
    )
}

fn summed() -> Vec<NamedExpr> {
    vec![NamedExpr::new(Expr::column(1, "sum(n)"), "sum(n)")]
}

#[test]
fn a_finalizing_merge_carries_its_finalize_in_a_project_of_its_own() {
    let node = finalizing_merge(summed(), &["k", "sum(n)"]);
    let recipe = aggregate_batches(&node, &[&columns_of(&["k", "n"])], &mut Writer::new())
        .expect("the aggregate's payloads are writable")
        .expect("a batch aggregate drives the ABI");
    assert_eq!(
        shape(&recipe),
        vec![
            (FbKind::CoalescePartitions, CallPattern::PerCompaction),
            (
                FbKind::Aggregate { merge: true },
                CallPattern::PerCompaction
            ),
            (FbKind::Project(ProjectRole::Finalize), CallPattern::AtDone),
        ],
        "the merge runs per compaction and the finalize once, at done"
    );
}

/// The init's mirror of the merge below. Both branches of `recipe::aggregate` reach the
/// same `finalize_project`, so what this pins is the call list: a second call, per batch
/// rather than at done, since an init that finalizes itself has no compaction to wait for.
#[test]
fn an_init_that_finalizes_itself_carries_the_finalize_in_a_project_of_its_own() {
    let node = finalizing_init(summed(), &["k", "sum(n)"]);
    let recipe = aggregate(&node, &[&columns_of(&["k", "n"])], &mut Writer::new())
        .expect("the aggregate's payloads are writable")
        .expect("an aggregate drives the ABI");
    assert_eq!(
        shape(&recipe),
        vec![
            (FbKind::Aggregate { merge: false }, CallPattern::PerBatch),
            (
                FbKind::Project(ProjectRole::Finalize),
                CallPattern::PerBatch
            ),
        ],
        "the init builds state and finalizes it, both per batch"
    );
}

/// The keys are the half the finalize list does not carry, and a project replaces the row:
/// finalized columns alone answer with values and nothing to read them by. Asserted on the
/// buffer rather than on the recipe, since the recipe names the call and not its payload.
#[test]
fn a_finalize_project_emits_the_group_keys_the_finalize_list_leaves_out() {
    let node = finalizing_merge(summed(), &["k", "sum(n)"]);
    let mut writer = Writer::new();
    let recipe = aggregate_batches(&node, &[&columns_of(&["k", "n"])], &mut writer)
        .expect("the aggregate's payloads are writable")
        .expect("a batch aggregate drives the ABI");
    let (bytes, _) = writer.finish().expect("one root");
    let plan = flatbuffers::root::<fb::GpuPlan>(&bytes).expect("the buffer verifies");
    let seq = *recipe.seqs().last().expect("the finalize is the last call");
    let project = node_at(&plan, seq)
        .and_then(|node| node.node_as_cudf_project())
        .expect("the last seq is the finalize project");
    let aliases: Vec<&str> = project.aliases().expect("named columns").iter().collect();
    assert_eq!(
        aliases,
        vec!["k", "sum(n)"],
        "the project has to emit the node's whole declared output"
    );
}

/// The keys are taken from the state by position, so a state whose first columns are not
/// the keys fills the row to the right width out of the wrong columns. Held by
/// `GpuAggregate::intermediate`'s documented order today, and by this if that ever moves.
#[test]
fn a_state_whose_first_columns_are_not_the_keys_is_refused() {
    let node = finalizing_merge(summed(), &["j", "sum(n)"]);
    let refused = aggregate_batches(&node, &[&columns_of(&["k", "n"])], &mut Writer::new())
        .expect_err("a state that does not lead with the keys is not projectable");
    let message = format!("{refused}");
    assert!(
        message.contains("column 0 `k`") && message.contains("names it `j`"),
        "the refusal has to name the two spellings: {message}"
    );
}

/// The width check, shown going red: a finalize list that does not account for every
/// non-key output column is a plan that answers with a column missing, which is a wrong
/// answer rather than an error unless this refuses.
#[test]
fn a_finalize_that_does_not_fill_the_declared_output_is_refused() {
    let node = finalizing_merge(Vec::new(), &["k", "sum(n)"]);
    let refused = aggregate_batches(&node, &[&columns_of(&["k", "n"])], &mut Writer::new())
        .expect_err("a finalize short of the declared output is not writable");
    let message = format!("{refused}");
    assert!(
        message.contains("1 keys and 0 columns") && message.contains("declares 2 output"),
        "the refusal has to name both counts: {message}"
    );
}

/// The one literal the wire cannot carry, in a node with an input — so the walk has
/// something to lose. #168 is the corpus case (`mixed-join`'s residual adds an interval to
/// a column), and the failure mode it guards is structural: a placeholder that dropped the
/// inputs its node had taken would leave them unreachable, and the tree would then be
/// shorter than the numbering.
///
/// A leaf would prove nothing here — the counts agree either way when nothing was taken.
fn filter_over_scan_with_an_interval() -> GpuFilter {
    let scan = crate::batch_partitioned::parquet_meta::ScanMetadata {
        file: "/orders.parquet".to_string(),
        groups: vec![crate::batch_partitioned::partitioner::RowGroupMeta {
            index: 0,
            rows: 10,
            bytes: 100,
        }],
        can_be_null: vec![false],
    };
    let load = crate::batch_partitioned::nodes::GpuLoadParquet::new(
        "orders".to_string(),
        vec![0],
        vec![vec![vec![0]]],
        &scan,
        None,
        columns_of(&["o_orderdate"]),
    );
    let ninety_days = ScalarValue::IntervalMonthDayNano(Some(
        datafusion::arrow::datatypes::IntervalMonthDayNano::new(0, 90, 0),
    ));
    GpuFilter::new(
        Box::new(load),
        Expr::binary(
            Expr::column(0, "o_orderdate"),
            BinaryOp::Lt,
            Expr::Literal(ninety_days),
            DataType::Boolean,
        ),
        None,
        columns_of(&["o_orderdate"]),
    )
}

#[test]
fn a_payload_the_wire_cannot_carry_fails_the_plan_and_names_where() {
    let refused = attach_recipes(&filter_over_scan_with_an_interval())
        .err()
        .expect("the wire has no interval, so this plan cannot be built");
    let said = refused.to_string();
    // The reason, its ticket, and the seq the node would have taken — the three things a
    // reader of the golden line needs, and the assertion that keeps the Err arm real: it
    // would go quietly green the day the writer learns intervals, having stopped testing
    // anything, so the message is checked rather than merely the failure.
    assert!(said.contains("scalar value"), "{said}");
    assert!(said.contains("(#168)"), "{said}");
    assert!(said.contains(" at #"), "{said}");
}

/// `dim(k, label)` joined to `fact(fk, v)`, keeping one column of each side. The finish
/// pass is the half that has to answer in the node's declared shape: the anti join emits
/// every build column whatever the projection says, so what the pad project keeps is the
/// whole question.
fn projecting_left_join() -> GpuJoin {
    GpuJoin::new(
        Given::input(BatchLayout::SingleBatch, &["k", "label"]),
        Given::input(BatchLayout::MultipleBatches, &["fk", "v"]),
        JoinType::Left,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        Some(vec![1, 3]),
        columns_of(&["label", "v"]),
    )
}

/// The buffer a join's recipe addresses, so a case can read the payload rather than the
/// call list.
fn written(node: &GpuJoin) -> (Recipe, Vec<u8>) {
    let mut writer = Writer::new();
    let recipe = join::hash_join(
        node,
        &[&columns_of(&["k", "label"]), &columns_of(&["fk", "v"])],
        &mut writer,
    )
    .expect("the join's payloads are writable")
    .expect("a hash join drives the ABI");
    let (bytes, _) = writer.finish().expect("one root");
    (recipe, bytes)
}

#[test]
fn a_finishing_joins_pad_project_emits_the_columns_the_node_declares() {
    let node = projecting_left_join();
    let (recipe, bytes) = written(&node);
    let plan = flatbuffers::root::<fb::GpuPlan>(&bytes).expect("the buffer verifies");
    let seq = *recipe
        .seqs()
        .last()
        .expect("the pad project is the last call");
    let project = node_at(&plan, seq)
        .and_then(|node| node.node_as_cudf_project())
        .expect("the last seq is the pad project");
    let aliases: Vec<&str> = project.aliases().expect("named columns").iter().collect();
    assert_eq!(
        aliases,
        vec!["label", "v"],
        "the unmatched build rows leave in the node's own shape, not the build side's"
    );
}

/// `dim(k, label)` semi-joined to `fact(fk, v)`, keeping one build column. A semi join's
/// output is the build side, so its projection only narrows — and publishing no call for
/// that narrowing is what left the device emitting every build column while the CPU
/// emitted what the node declared.
fn projecting_semi_join(join_type: JoinType, kept: Vec<u32>, output: &[&str]) -> GpuJoin {
    GpuJoin::new(
        Given::input(BatchLayout::SingleBatch, &["k", "label"]),
        Given::input(BatchLayout::MultipleBatches, &["fk", "v"]),
        join_type,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        Some(kept),
        columns_of(output),
    )
}

#[test]
fn a_projecting_semi_joins_recipe_narrows_what_its_finish_emitted() {
    let node = projecting_semi_join(JoinType::LeftSemi, vec![1], &["label"]);
    let (recipe, bytes) = written(&node);
    let plan = flatbuffers::root::<fb::GpuPlan>(&bytes).expect("the buffer verifies");
    assert_eq!(
        recipe
            .calls
            .last()
            .map(|call| call.target.map(|(_, kind)| kind)),
        Some(Some(FbKind::Project(ProjectRole::Narrow))),
        "the finish is followed by the project that cuts it down: {recipe}"
    );
    let seq = *recipe.seqs().last().expect("the narrow project is last");
    let project = node_at(&plan, seq)
        .and_then(|node| node.node_as_cudf_project())
        .expect("the last seq is the narrow project");
    let aliases: Vec<&str> = project.aliases().expect("named columns").iter().collect();
    assert_eq!(
        aliases,
        vec!["label"],
        "one column named, one column emitted"
    );
}

/// A mark join's finish emits the build side AND the boolean it appends, so its projection
/// indexes one column past the build side. The pad project would read that ordinal as a
/// probe column and write a typed NULL where the mark belongs, which is why the narrowing
/// project walks what the finish emitted rather than build-plus-probe.
#[test]
fn a_projecting_mark_joins_recipe_keeps_the_mark_as_a_column_and_not_as_a_null() {
    let node = projecting_semi_join(JoinType::LeftMark, vec![1, 2], &["label", "mark"]);
    let (recipe, bytes) = written(&node);
    let plan = flatbuffers::root::<fb::GpuPlan>(&bytes).expect("the buffer verifies");
    let seq = *recipe.seqs().last().expect("the narrow project is last");
    let project = node_at(&plan, seq)
        .and_then(|node| node.node_as_cudf_project())
        .expect("the last seq is the narrow project");
    let aliases: Vec<&str> = project.aliases().expect("named columns").iter().collect();
    assert_eq!(aliases, vec!["label", "mark"]);
    let exprs = project.exprs().expect("the project has expressions");
    for position in 0..exprs.len() {
        assert!(
            exprs.get(position).node_as_column_ref().is_some(),
            "column {position} of a narrowing project is a column, never a literal"
        );
    }
}

/// A semi join with no projection publishes no narrowing call: the finish already emits
/// the row, and a project that keeps every column is a call for nothing.
#[test]
fn a_semi_join_that_narrows_nothing_publishes_no_project_after_its_finish() {
    let node = GpuJoin::new(
        Given::input(BatchLayout::SingleBatch, &["k", "label"]),
        Given::input(BatchLayout::MultipleBatches, &["fk", "v"]),
        JoinType::LeftSemi,
        vec![(0, 0)],
        None,
        Vec::new(),
        false,
        None,
        columns_of(&["k", "label"]),
    );
    let (recipe, _) = written(&node);
    assert_eq!(
        recipe
            .calls
            .last()
            .map(|call| call.target.map(|(_, kind)| kind)),
        Some(Some(FbKind::HashJoin {
            join_type: JoinType::LeftSemi
        })),
        "the finish join is the last call: {recipe}"
    );
}

/// One name for one column. The key project builds the accumulated keys table and names
/// its columns; the finish join is its only reader, so the name it uses to read them has
/// to be the name they were given.
#[test]
fn the_finish_join_reads_the_accumulated_keys_under_the_names_they_carry() {
    let node = projecting_left_join();
    let (recipe, bytes) = written(&node);
    let plan = flatbuffers::root::<fb::GpuPlan>(&bytes).expect("the buffer verifies");
    let seqs = recipe.seqs();
    let keys = node_at(&plan, seqs[0])
        .and_then(|node| node.node_as_cudf_project())
        .expect("the first call is the key project");
    let named: Vec<&str> = keys.aliases().expect("named columns").iter().collect();

    let finish = node_at(&plan, seqs[3])
        .and_then(|node| node.node_as_cudf_hash_join())
        .expect("the fourth call is the finish join");
    let read: Vec<String> = finish
        .keys()
        .expect("the finish join has keys")
        .iter()
        .map(|key| {
            key.right()
                .and_then(|expr| expr.node_as_column_ref())
                .expect("a key read by ordinal is a column")
                .name()
                .expect("a column has a name")
                .to_string()
        })
        .collect();
    assert_eq!(
        read, named,
        "the finish join reads the keys under other names than the project gave them"
    );
}

/// Every cell of both dimensions — nine join types crossed with a residual filter — and
/// one claim about each: the path this writer publishes and the path the CPU executor
/// builds are the same path.
///
/// That is the assertion the defect was. A cell where one keeps probe keys and the other
/// makes a single call is a residual dropped or a finish pass that never runs, and neither
/// shows up as a failed call — so this goes red for whichever cell a later reader gets
/// wrong rather than for the ones we thought to write down.
#[test]
fn the_recipe_and_the_executor_take_the_same_path_through_every_cell() {
    use crate::batch_partitioned::cpu_backend::join::CpuJoin;
    use datafusion::execution::context::SessionContext;

    let types = [
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
        JoinType::LeftMark,
        JoinType::RightSemi,
        JoinType::RightAnti,
    ];
    let build = columns_of(&["k", "label"]);
    let probe = columns_of(&["fk", "v"]);
    let mut cells = 0;
    for join_type in types {
        for residual in [false, true] {
            let node = join(join_type, residual, None);
            let executor = CpuJoin::hash(
                &node,
                &build.fields,
                &probe.fields,
                SessionContext::new().task_ctx(),
            );
            if node.capability().is_err() {
                assert!(
                    executor.is_err(),
                    "{join_type:?} with residual={residual} is refused by the matrix and \
                     the executor built it anyway"
                );
                cells += 1;
                continue;
            }
            let recipe = recipe_for(&node);
            let recipe_finishes = recipe
                .calls
                .iter()
                .any(|call| call.when == CallPattern::AtDone);
            let executor = executor.expect("a cell the matrix allows is one the executor builds");
            assert_eq!(
                recipe_finishes,
                executor.makes_a_finish_pass(),
                "{join_type:?} with residual={residual}: the recipe {} and the executor {}",
                if recipe_finishes {
                    "finishes"
                } else {
                    "does not"
                },
                if executor.makes_a_finish_pass() {
                    "does"
                } else {
                    "does not"
                }
            );
            cells += 1;
        }
    }
    assert_eq!(
        cells, 18,
        "nine types by two residuals, and every cell answered"
    );
}
