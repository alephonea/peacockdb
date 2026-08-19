//! One test per node kind, per expression kind and per planner rule, each from the
//! smallest plan that shows it. The corpus goldens are a regression net over whole plans
//! and a different question from whether a rule is right.

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

use super::Translator;
use crate::batch_partitioned::aggregates::PlanAgg;
use crate::batch_partitioned::error::PlanError;
use crate::batch_partitioned::expr::{BinaryOp, Expr};
use crate::batch_partitioned::layout::KeyDistribution;
use crate::batch_partitioned::node::{GpuNode, RowInterval};
use crate::batch_partitioned::nodes::join::{NestedLoopJoinType, capability};
use crate::batch_partitioned::nodes::{NodeRef, as_node_ref};
use crate::batch_partitioned::partitioner::Batching;

/// Plain DataFusion planning at tp1 over the committed minimal dataset — this mode
/// translates the physical plan rather than annotating it, so no GPU rule runs.
async fn plan_at(sql: &str, target_partitions: usize) -> Arc<dyn ExecutionPlan> {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal");
    let ctx = crate::register_tables_for(crate::build_session_state(target_partitions), &data)
        .await
        .expect("register the minimal tables");
    ctx.sql(sql)
        .await
        .expect("plan the query")
        .create_physical_plan()
        .await
        .expect("physical plan")
}

async fn plan(sql: &str) -> Arc<dyn ExecutionPlan> {
    plan_at(sql, 1).await
}

async fn translated(sql: &str) -> Box<dyn GpuNode> {
    Translator::new(1, Batching::Off)
        .translate(&plan(sql).await)
        .expect("translate the plan")
}

/// tp4 with batching on, which is what makes the small-table threshold bite.
async fn translated_at_tp4(sql: &str, small_table_rows: u64) -> Box<dyn GpuNode> {
    Translator::new(
        4,
        Batching::On {
            target_batch_bytes: 1 << 20,
        },
    )
    .with_small_table_rows(small_table_rows)
    .translate(&plan_at(sql, 4).await)
    .expect("translate the plan")
}

async fn refused(sql: &str) -> PlanError {
    Translator::new(1, Batching::Off)
        .translate(&plan(sql).await)
        .expect_err("this shape should not plan")
}

fn name_of(node: &dyn GpuNode) -> &'static str {
    match as_node_ref(node) {
        NodeRef::LoadParquet(_) => "LoadParquet",
        NodeRef::Filter(_) => "Filter",
        NodeRef::Project(_) => "Project",
        NodeRef::Sort(_) => "Sort",
        NodeRef::CoalesceAllBatches(_) => "CoalesceAllBatches",
        NodeRef::AccumulateBatchesAndSort(_) => "AccumulateBatchesAndSort",
        NodeRef::Limit(_) => "Limit",
        NodeRef::Aggregate(_) => "Aggregate",
        NodeRef::AggregateBatches(_) => "AggregateBatches",
        NodeRef::Join(_) => "Join",
        NodeRef::CrossJoin(_) => "CrossJoin",
        NodeRef::NestedLoopJoin(_) => "NestedLoopJoin",
        NodeRef::MergePartitions(_) => "MergePartitions",
        NodeRef::EmitPartitions(_) => "EmitPartitions",
        NodeRef::MergeSortedPartitions(_) => "MergeSortedPartitions",
        NodeRef::Union(_) => "Union",
        NodeRef::Interleave(_) => "Interleave",
        NodeRef::Unload(_) => "Unload",
    }
}

/// The tree as `Parent(Child, Child)` — enough to assert which nodes were emitted and
/// where, without standing in for the plan renderer.
fn shape(node: &dyn GpuNode) -> String {
    let children = node.children();
    if children.is_empty() {
        return name_of(node).to_string();
    }
    let inner: Vec<String> = children.iter().map(|c| shape(*c)).collect();
    format!("{}({})", name_of(node), inner.join(", "))
}

fn descend(root: &dyn GpuNode, depth: usize) -> &dyn GpuNode {
    let mut node = root;
    for _ in 0..depth {
        node = node.children()[0];
    }
    node
}

fn validate_all(node: &dyn GpuNode) {
    node.validate_schemas_and_partitions()
        .unwrap_or_else(|e| panic!("{} failed its own validation: {e}", name_of(node)));
    for child in node.children() {
        validate_all(child);
    }
}

#[tokio::test]
async fn a_scan_becomes_a_loader_carrying_the_partitioners_mapping() {
    let tree = translated("SELECT * FROM nation").await;
    assert_eq!(shape(tree.as_ref()), "Unload(LoadParquet)");

    let NodeRef::LoadParquet(load) = as_node_ref(descend(tree.as_ref(), 1)) else {
        panic!("expected a loader");
    };
    assert_eq!(load.table, "nation");
    // One lane at tp1, one batch per lane with batching off, whole row groups inside.
    assert_eq!(load.partition_groups, vec![vec![vec![0]]]);
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn a_filter_becomes_one_node_and_keeps_its_predicate() {
    let tree = translated("SELECT * FROM nation WHERE n_regionkey > 1").await;
    // The CoalesceBatchesExec DataFusion puts above the filter leaves no node: its
    // target says nothing about the batches a lane will hold in this mode.
    assert_eq!(shape(tree.as_ref()), "Unload(Filter(LoadParquet))");

    let NodeRef::Filter(filter) = as_node_ref(descend(tree.as_ref(), 1)) else {
        panic!("expected a filter");
    };
    assert!(matches!(filter.predicate, Expr::Binary { .. }));
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn a_projection_becomes_one_node_per_output_column() {
    let tree = translated("SELECT n_regionkey + 1 AS shifted FROM nation").await;
    assert_eq!(shape(tree.as_ref()), "Unload(Project(LoadParquet))");

    let NodeRef::Project(project) = as_node_ref(descend(tree.as_ref(), 1)) else {
        panic!("expected a project");
    };
    assert_eq!(project.exprs.len(), 1);
    assert_eq!(project.exprs[0].name, "shifted");
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn a_sort_becomes_a_per_batch_sort_under_an_accumulator() {
    let tree = translated("SELECT * FROM nation ORDER BY n_name").await;
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(AccumulateBatchesAndSort(Sort(LoadParquet)))"
    );
    // Sorting each batch leaves them individually ordered and collectively not; the
    // accumulator is what makes the stream sorted.
    let accumulated = descend(tree.as_ref(), 1);
    assert!(accumulated.kind().layout().unwrap().is_stream_sorted());
    assert!(
        !descend(tree.as_ref(), 2)
            .kind()
            .layout()
            .unwrap()
            .is_stream_sorted()
    );
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn a_top_n_replicates_its_fetch_onto_every_stage() {
    let tree = translated("SELECT * FROM nation ORDER BY n_name LIMIT 3").await;
    let NodeRef::AccumulateBatchesAndSort(accumulator) = as_node_ref(descend(tree.as_ref(), 1))
    else {
        panic!("expected the accumulator");
    };
    let NodeRef::Sort(sort) = as_node_ref(descend(tree.as_ref(), 2)) else {
        panic!("expected the per-batch sort");
    };
    // Each stage holds at most n rows per live batch instead of its whole input.
    assert_eq!((sort.fetch, accumulator.fetch), (Some(3), Some(3)));
    assert_eq!(sort.keys.len(), 1);
}

#[tokio::test]
async fn a_root_adjacent_limit_is_not_a_node_at_all() {
    let tree = translated("SELECT * FROM nation LIMIT 3 OFFSET 2").await;
    assert_eq!(shape(tree.as_ref()), "Unload(LoadParquet)");
    // The interval rides the boundary crossing, so the skip prefix never moves.
    assert_eq!(
        tree.row_interval(),
        Some(RowInterval {
            skip: 2,
            fetch: Some(3)
        })
    );
}

#[tokio::test]
async fn a_mid_plan_limit_is_a_node_over_a_one_lane_stream() {
    let tree =
        translated("SELECT count(*) FROM (SELECT * FROM nation WHERE n_regionkey > 1 LIMIT 3) t")
            .await;
    // DataFusion's limit pushdown parks this one on a CoalesceBatchesExec as a
    // fetch, so a translation that merely drops that node counts every row.
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(AggregateBatches(Aggregate(Project(Limit(Filter(LoadParquet))))))"
    );
    assert_eq!(
        descend(tree.as_ref(), 4).row_interval(),
        Some(RowInterval {
            skip: 0,
            fetch: Some(3)
        })
    );
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn an_aggregate_over_many_batches_splits_into_init_and_merge() {
    // A count DataFusion can answer from parquet statistics never reaches an
    // aggregate node at all, so the predicate is what makes this a real aggregate.
    let tree = translated("SELECT count(*) FROM nation WHERE n_regionkey > 1").await;
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(AggregateBatches(Aggregate(Project(Filter(LoadParquet)))))"
    );

    let NodeRef::AggregateBatches(merge) = as_node_ref(descend(tree.as_ref(), 1)) else {
        panic!("expected the merging aggregate");
    };
    let NodeRef::Aggregate(init) = as_node_ref(descend(tree.as_ref(), 2)) else {
        panic!("expected the init aggregate");
    };
    // A count merges by SUM — the one place naming the merge separately from the init
    // is the difference between a right and a wrong answer.
    assert_eq!(init.body.aggs[0].func, PlanAgg::Count);
    assert_eq!(merge.body.aggs[0].func, PlanAgg::Sum);
    assert!(init.body.finalize.is_none() && merge.body.finalize.is_some());
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn an_aggregate_over_one_batch_needs_no_merge() {
    // The accumulator below it emits a single batch in a single lane, so the init
    // node is already looking at whole groups and finishes them itself.
    let tree = translated(
        "SELECT count(*) FROM (SELECT n_regionkey, count(*) c FROM nation GROUP BY n_regionkey) t",
    )
    .await;
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(Aggregate(Project(AggregateBatches(Aggregate(LoadParquet)))))"
    );
    let NodeRef::Aggregate(single) = as_node_ref(descend(tree.as_ref(), 1)) else {
        panic!("expected one aggregate");
    };
    assert!(
        single.body.finalize.is_some(),
        "it finishes the aggregate itself"
    );
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn avg_becomes_two_state_columns_and_a_divide() {
    let tree = translated("SELECT avg(n_regionkey) FROM nation").await;
    let NodeRef::Aggregate(init) = as_node_ref(descend(tree.as_ref(), 2)) else {
        panic!("expected the init aggregate");
    };
    let NodeRef::AggregateBatches(merge) = as_node_ref(descend(tree.as_ref(), 1)) else {
        panic!("expected the merging aggregate");
    };
    // Never a mean of means: a sum and a count, each merged by sum, divided at the end.
    assert_eq!(
        init.body.aggs.iter().map(|a| a.func).collect::<Vec<_>>(),
        vec![PlanAgg::Sum, PlanAgg::Count]
    );
    assert_eq!(
        merge.body.aggs.iter().map(|a| a.func).collect::<Vec<_>>(),
        vec![PlanAgg::Sum, PlanAgg::Sum]
    );
    let finalize = &merge.body.finalize.as_ref().unwrap()[0];
    assert!(matches!(
        finalize.expr,
        Expr::Binary {
            op: BinaryOp::Divide,
            ..
        }
    ));
}

#[tokio::test]
async fn stddev_becomes_welford_state_merged_by_merge_m2() {
    let tree = translated("SELECT stddev(n_regionkey) FROM nation").await;
    let NodeRef::Aggregate(init) = as_node_ref(descend(tree.as_ref(), 2)) else {
        panic!("expected the init aggregate");
    };
    let NodeRef::AggregateBatches(merge) = as_node_ref(descend(tree.as_ref(), 1)) else {
        panic!("expected the merging aggregate");
    };
    assert_eq!(
        init.body.aggs.iter().map(|a| a.func).collect::<Vec<_>>(),
        vec![PlanAgg::Count, PlanAgg::Mean, PlanAgg::M2]
    );
    // The combine is not a per-column reduction, so it is one call over all three.
    assert_eq!(merge.body.aggs.len(), 1);
    assert_eq!(merge.body.aggs[0].func, PlanAgg::MergeM2);
    assert_eq!(merge.body.aggs[0].outputs.len(), 3);
    let finalize = &merge.body.finalize.as_ref().unwrap()[0];
    assert!(
        matches!(finalize.expr, Expr::Case { .. }),
        "count <= ddof yields NULL"
    );
}

#[tokio::test]
async fn a_cross_join_takes_its_build_side_as_one_batch() {
    let tree = translated("SELECT * FROM nation, region").await;
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(Project(CrossJoin(CoalesceAllBatches(LoadParquet), LoadParquet)))"
    );
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn a_non_equi_predicate_becomes_a_nested_loop_join() {
    let tree =
        translated("SELECT * FROM nation n, region r WHERE n.n_regionkey < r.r_regionkey").await;
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(Project(NestedLoopJoin(CoalesceAllBatches(LoadParquet), LoadParquet)))"
    );
    let NodeRef::NestedLoopJoin(join) = as_node_ref(descend(tree.as_ref(), 2)) else {
        panic!("expected a nested-loop join");
    };
    assert_eq!(join.join_type, NestedLoopJoinType::Inner);
    assert!(matches!(join.filter, Expr::Binary { .. }));
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn an_unrecognized_plan_node_is_refused_and_named() {
    // A count DataFusion answers from parquet statistics is a PlaceholderRowExec and a
    // projection, with no aggregate to translate — refused by name until #158 settles
    // which way out to take.
    let err = refused("SELECT count(*) FROM nation").await;
    assert!(
        matches!(&err, PlanError::Unsupported(what) if what.contains("PlaceholderRowExec")),
        "{err}"
    );
}

#[tokio::test]
async fn a_hash_repartition_becomes_a_merge_and_a_scatter() {
    let tree = translated_at_tp4(
        "SELECT c_nationkey, count(*) FROM customer GROUP BY c_nationkey",
        0,
    )
    .await;
    // The per-lane merge shrinks what crosses the shuffle; the coalesce below the
    // scatter is what makes it one call rather than one per merged batch.
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(AggregateBatches(EmitPartitions(CoalesceAllBatches(MergePartitions(\
         AggregateBatches(Aggregate(LoadParquet)))))))"
    );
    let NodeRef::EmitPartitions(emit) = as_node_ref(descend(tree.as_ref(), 2)) else {
        panic!("expected the scatter");
    };
    assert_eq!(emit.hash_keys, vec![0]);
    assert_eq!(emit.kind().layout().unwrap().n, 4);
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn a_round_robin_repartition_leaves_no_node() {
    // It carries no key, so it says nothing this mode acts on: lanes come from the
    // partitioner's mapping instead.
    let tree = translated_at_tp4("SELECT * FROM customer WHERE c_nationkey > 1", 0).await;
    assert_eq!(shape(tree.as_ref()), "Unload(Filter(LoadParquet))");
    assert_eq!(descend(tree.as_ref(), 1).kind().layout().unwrap().n, 4);
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn a_keyless_aggregate_collapses_its_lanes_instead_of_shuffling() {
    // There is no key to re-land rows by, so DataFusion coalesces and the sequence
    // merges into one lane rather than scattering into four.
    let tree = translated_at_tp4("SELECT sum(c_acctbal) FROM customer", 0).await;
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(AggregateBatches(MergePartitions(AggregateBatches(Aggregate(LoadParquet)))))"
    );
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn a_sort_preserving_merge_replaces_the_per_lane_accumulator() {
    let tree = translated_at_tp4("SELECT * FROM customer ORDER BY c_name", 0).await;
    // One accumulator, not two: the N-into-1 merge is what the parent needs, so the
    // per-lane accumulate-and-sort is not emitted at all.
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(MergeSortedPartitions(Sort(LoadParquet)))"
    );
    assert!(
        descend(tree.as_ref(), 1)
            .kind()
            .layout()
            .unwrap()
            .is_stream_sorted()
    );
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn an_equi_join_is_co_partitioned_by_a_scatter_on_each_side() {
    let tree = translated_at_tp4(
        "SELECT c.c_name, s.s_name FROM customer c JOIN supplier s ON c.c_nationkey = s.s_nationkey",
        0,
    )
    .await;
    // Both sides scatter on their join key, so lane p of one holds exactly what can
    // match lane p of the other; the round-robin repartition below the small side
    // leaves no node at all.
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(Project(Join(CoalesceAllBatches(EmitPartitions(MergePartitions(LoadParquet))), \
         EmitPartitions(MergePartitions(LoadParquet)))))"
    );
    let NodeRef::Join(join) = as_node_ref(descend(tree.as_ref(), 2)) else {
        panic!("expected an equi-join");
    };
    // Lane p of one side holds exactly the rows that can match lane p of the other.
    assert_eq!(join.keys.len(), 1);
    assert!(!join.null_equals_null);
    assert!(join.capability().unwrap().probe_streams);
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn a_broadcast_shaped_join_runs_in_one_lane_until_140() {
    // Both tables are tiny, so DataFusion collects the left rather than hashing both
    // sides. Nothing co-locates them, and this mode has no broadcast to do it with.
    let tree = translated_at_tp4(
        "SELECT n.n_name, r.r_name FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey",
        0,
    )
    .await;
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(Project(Join(CoalesceAllBatches(MergePartitions(LoadParquet)), \
         MergePartitions(LoadParquet))))"
    );
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn an_outer_join_with_a_residual_filter_is_refused() {
    // Not a limitation of this mode: the shipping executor applies the filter after
    // the outer gather, so a padded row's NULLs drop it (#153).
    let err = capability(datafusion::common::JoinType::Left, true).unwrap_err();
    assert!(
        matches!(&err, PlanError::Unsupported(what) if what.contains("#153")),
        "{err}"
    );
    // The build-side semi family keeps its finish pass and gives up streaming instead.
    let filtered_semi = capability(datafusion::common::JoinType::LeftSemi, true).unwrap();
    assert!(!filtered_semi.probe_streams && filtered_semi.needs_finish);
}

#[tokio::test]
async fn a_union_forwards_its_branches_into_one_lane_numbering() {
    let tree = translated_at_tp4(
        "SELECT c_nationkey FROM customer UNION ALL SELECT s_nationkey FROM supplier",
        0,
    )
    .await;
    assert_eq!(
        shape(tree.as_ref()),
        "Unload(Union(LoadParquet, Project(LoadParquet)))"
    );
    // Lane counts sum, and no branch's hash survives the renumbering.
    let union = descend(tree.as_ref(), 1);
    assert_eq!(union.kind().layout().unwrap().n, 8);
    assert_eq!(
        union.kind().layout().unwrap().key_distribution,
        KeyDistribution::NotSpecified
    );
    validate_all(tree.as_ref());
}

#[tokio::test]
async fn a_small_source_plans_one_lane_and_the_shuffle_around_it_disappears() {
    let query =
        |table: &str, key: &str| format!("SELECT {key}, count(*) FROM {table} GROUP BY {key}");
    // nation is 25 rows, under the threshold: one lane, so nothing to merge back and
    // no shuffle to re-land rows the lane never split.
    let small = translated_at_tp4(&query("nation", "n_regionkey"), 1_000).await;
    assert_eq!(
        shape(small.as_ref()),
        "Unload(AggregateBatches(Aggregate(LoadParquet)))"
    );
    assert_eq!(descend(small.as_ref(), 2).kind().layout().unwrap().n, 1);

    // customer is 150k rows at the same threshold, and gets the whole sequence.
    let large = translated_at_tp4(&query("customer", "c_nationkey"), 1_000).await;
    assert_eq!(
        shape(large.as_ref()),
        "Unload(AggregateBatches(EmitPartitions(CoalesceAllBatches(MergePartitions(\
         AggregateBatches(Aggregate(LoadParquet)))))))"
    );
    assert_eq!(descend(large.as_ref(), 5).kind().layout().unwrap().n, 4);
    validate_all(small.as_ref());
    validate_all(large.as_ref());
}

#[tokio::test]
async fn the_threshold_is_inert_while_batching_is_off() {
    // With one batch per lane there is nothing for a batch-size threshold to size, so
    // the rule that drops a source to one lane has nothing to act on either.
    let tree = Translator::new(4, Batching::Off)
        .with_small_table_rows(1_000)
        .translate(
            &plan_at(
                "SELECT n_regionkey, count(*) FROM nation GROUP BY n_regionkey",
                4,
            )
            .await,
        )
        .expect("translate the plan");
    assert_eq!(descend(tree.as_ref(), 5).kind().layout().unwrap().n, 4);
}

#[tokio::test]
async fn a_window_function_is_refused_at_plan_time() {
    let err = refused("SELECT sum(n_regionkey) OVER () FROM nation").await;
    assert!(
        matches!(&err, PlanError::Unsupported(what) if what.contains("#143")),
        "{err}"
    );
}
