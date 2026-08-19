//! What every node declares about its columns: the arrow types, and the annotations a
//! merging or finalizing node reads.
//!
//! Types are asserted on the tree rather than on the rendered plan. Both engines derive
//! their per-node byte counts from the same declared schema, so a column typed wrongly
//! costs the same on either and no golden byte moves — `avg`'s two state columns were
//! once typed backwards, and only a real divide would have diverged.

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;

use super::Translator;
use crate::batch_partitioned::aggregates::AggFunc;
use crate::batch_partitioned::expr::{BinaryOp, Expr};
use crate::batch_partitioned::node::GpuNode;
use crate::batch_partitioned::nodes::{NodeRef, as_node_ref};
use crate::batch_partitioned::partitioner::Batching;
use crate::batch_partitioned::plan::{BatchSizing, PlanKnobs, plan_batch_partitioned};
use crate::batch_partitioned::schema::{AggStateColumns, Schema};

/// The committed minimal dataset, whose `p_retailprice` and `c_acctbal` are
/// `Decimal128(15,2)` — the two columns every decimal assertion below starts from.
async fn translated(sql: &str, target_partitions: usize) -> Box<dyn GpuNode> {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal");
    let ctx = crate::register_tables_for(crate::build_session_state(target_partitions), &data)
        .await
        .expect("register the minimal tables");
    let plan = ctx
        .sql(sql)
        .await
        .expect("plan the query")
        .create_physical_plan()
        .await
        .expect("physical plan");
    Translator::new(
        target_partitions,
        Batching::Sized {
            target_batch_bytes: 4096,
        },
    )
    // Every table here is small, and what these tests need is the four-lane sequence.
    .with_small_table_bytes(0)
    .translate(&plan)
    .expect("translate the plan")
}

/// Every node the predicate accepts, deepest first — the order the aggregate sequence
/// runs in, so `[init, per-lane merge, finalizing merge]` reads as written.
fn deepest_first<'a>(
    root: &'a dyn GpuNode,
    accept: &dyn Fn(&dyn GpuNode) -> bool,
    into: &mut Vec<&'a dyn GpuNode>,
) {
    for child in root.children() {
        deepest_first(child, accept, into);
    }
    if accept(root) {
        into.push(root);
    }
}

fn aggregates(root: &dyn GpuNode) -> Vec<&dyn GpuNode> {
    let mut found = Vec::new();
    deepest_first(
        root,
        &|node| {
            matches!(
                as_node_ref(node),
                NodeRef::Aggregate(_) | NodeRef::AggregateBatches(_)
            )
        },
        &mut found,
    );
    found
}

fn projects(root: &dyn GpuNode) -> Vec<&dyn GpuNode> {
    let mut found = Vec::new();
    deepest_first(
        root,
        &|node| matches!(as_node_ref(node), NodeRef::Project(_)),
        &mut found,
    );
    found
}

fn schema_of(node: &dyn GpuNode) -> &Schema {
    node.kind().schema().expect("an aggregate is not a sink")
}

fn types_of(node: &dyn GpuNode) -> Vec<(String, DataType)> {
    schema_of(node)
        .fields
        .fields()
        .iter()
        .map(|field| (field.name().clone(), field.data_type().clone()))
        .collect()
}

const AVG_AT_TP4: &str = "SELECT p_brand, avg(p_retailprice) FROM part GROUP BY p_brand";

fn avg_state(positions: Vec<u32>) -> AggStateColumns {
    AggStateColumns {
        output: "avg(part.p_retailprice)".to_string(),
        func: AggFunc::Avg,
        ddof: 0,
        positions,
    }
}

#[tokio::test]
async fn one_aggregate_is_three_schemas_and_each_declares_what_it_holds() {
    let tree = translated(AVG_AT_TP4, 4).await;
    let sequence = aggregates(tree.as_ref());
    assert_eq!(sequence.len(), 3, "init, per-lane merge, finalizing merge");

    // The init and the per-lane merge both emit state: one group key, then the two
    // columns `avg` decomposed into.
    for holding_state in &sequence[..2] {
        let schema = schema_of(*holding_state);
        assert_eq!(schema.group_keys, vec![0]);
        assert_eq!(schema.agg_state, vec![avg_state(vec![1, 2])]);
        assert_eq!(
            schema.state_for("avg(part.p_retailprice)"),
            Some(&avg_state(vec![1, 2])),
            "a merge finds the state by the output it belongs to"
        );
    }

    // The finalizing merge emits the aggregate, so the keys are still keys and there is
    // no state left to declare.
    let finalized = schema_of(sequence[2]);
    assert_eq!(finalized.group_keys, vec![0]);
    assert_eq!(finalized.agg_state, Vec::new());
    assert_eq!(finalized.state_for("avg(part.p_retailprice)"), None);
}

#[tokio::test]
async fn avgs_state_columns_are_typed_by_what_they_hold_and_not_by_position() {
    // DataFusion declares avg's state as [count, sum] and this mode's decomposition reads
    // [sum, count]. Pairing them by position types both backwards — a sum in a UInt64 and
    // a count in a decimal — which no per-node byte count can show, because both engines
    // read the same declared schema.
    let tree = translated(AVG_AT_TP4, 4).await;
    let init = aggregates(tree.as_ref())[0];
    assert_eq!(
        types_of(init),
        vec![
            ("p_brand".to_string(), DataType::Utf8View),
            (
                "avg(part.p_retailprice)$sum".to_string(),
                DataType::Decimal128(15, 2)
            ),
            (
                "avg(part.p_retailprice)$count".to_string(),
                DataType::UInt64
            ),
        ]
    );

    let state = &schema_of(init).agg_state[0];
    let fields = schema_of(init).fields.fields().clone();
    assert_eq!(
        fields[state.positions[0] as usize].data_type(),
        &DataType::Decimal128(15, 2),
        "the sum keeps the scale of the column it sums"
    );
    assert_eq!(
        fields[state.positions[1] as usize].data_type(),
        &DataType::UInt64,
        "the count is a count"
    );
}

#[tokio::test]
async fn the_divide_that_finishes_an_avg_hits_the_scale_datafusion_declared() {
    let tree = translated(AVG_AT_TP4, 4).await;
    let sequence = aggregates(tree.as_ref());
    let NodeRef::AggregateBatches(finalizing) = as_node_ref(sequence[2]) else {
        panic!("the sequence finishes in a merge");
    };
    let out_type = schema_of(sequence[2]).fields.field(1).data_type().clone();
    assert_eq!(out_type, DataType::Decimal128(19, 6));

    let finalize = finalizing
        .body
        .finalize
        .as_ref()
        .expect("the last node finalizes");
    // One entry per aggregate: the group keys pass through unnamed by this list.
    assert_eq!(finalize.len(), 1);
    let Expr::Binary {
        left,
        op: BinaryOp::Divide,
        right,
        out_type: declared,
    } = &finalize[0].expr
    else {
        panic!("avg finishes as a divide, got {:?}", finalize[0].expr);
    };
    assert_eq!(declared, &out_type);
    // The denominator is an exact integer-valued decimal of the same precision, so cuDF's
    // own divide scale (s_left - s_right) lands on the scale declared above rather than on
    // one it derived.
    let target_of = |expr: &Expr| match expr {
        Expr::Cast { target, .. } => target.clone(),
        other => panic!("both sides of the divide are casts, got {other:?}"),
    };
    assert_eq!(target_of(left), DataType::Decimal128(19, 6));
    assert_eq!(target_of(right), DataType::Decimal128(19, 0));
}

#[tokio::test]
async fn a_union_branch_is_cast_to_the_declared_output_by_a_project() {
    // Two decimals of different scale meeting at a union: routing cannot retype, so the
    // planner owes each branch a project that lands on the declared type (#41), and the
    // declared type is DataFusion's coercion rather than one this mode derived.
    let tree = translated(
        "SELECT p_retailprice AS v FROM part \
         UNION ALL SELECT c_acctbal * 1.5 FROM customer",
        1,
    )
    .await;
    let union = tree.children()[0];
    let declared = DataType::Decimal128(30, 15);
    assert_eq!(types_of(union), vec![("v".to_string(), declared.clone())]);

    let branches = projects(tree.as_ref());
    assert_eq!(branches.len(), 2, "one casting project per branch");
    for branch in &branches {
        assert_eq!(types_of(*branch), vec![("v".to_string(), declared.clone())]);
        let NodeRef::Project(project) = as_node_ref(*branch) else {
            panic!("a branch is a project");
        };
        let Expr::Cast { target, .. } = &project.exprs[0].expr else {
            panic!("the branch expression ends in a cast");
        };
        assert_eq!(target, &declared);
        // The scan below still reads what parquet holds: the widening is the project's.
        assert_eq!(
            schema_of(branch.children()[0]).fields.field(0).data_type(),
            &DataType::Decimal128(15, 2)
        );
    }
}

#[tokio::test]
async fn a_project_over_a_finished_aggregate_carries_its_types_and_renames_them() {
    // A project, an aggregate and a union in one plan: the sum's declared type is
    // DataFusion's widening of the column it sums, the project renames without retyping,
    // and the branches meet a union that declares the same type both carry.
    let tree = translated(
        "SELECT p_brand AS k, sum(p_retailprice) AS total FROM part GROUP BY p_brand \
         UNION ALL SELECT c_mktsegment, sum(c_acctbal) FROM customer GROUP BY c_mktsegment",
        4,
    )
    .await;
    let summed = DataType::Decimal128(25, 2);
    let sequence = aggregates(tree.as_ref());
    assert_eq!(sequence.len(), 6, "the sequence runs once per branch");

    let init = schema_of(sequence[0]);
    assert_eq!(
        init.agg_state,
        vec![AggStateColumns {
            output: "sum(part.p_retailprice)".to_string(),
            func: AggFunc::Sum,
            ddof: 0,
            positions: vec![1],
        }]
    );
    assert_eq!(init.fields.field(1).data_type(), &summed);

    for branch in projects(tree.as_ref()) {
        assert_eq!(
            types_of(branch),
            vec![
                ("k".to_string(), DataType::Utf8View),
                ("total".to_string(), summed.clone()),
            ]
        );
        // The names are the query's and the types are the aggregate's: renaming a column
        // is not retyping it.
        assert_eq!(
            schema_of(branch.children()[0]).fields.field(1).data_type(),
            &summed
        );
        // Nothing reads a group-key annotation through a projection, and nothing claims
        // one either: only the aggregate nodes declare them.
        assert_eq!(schema_of(branch).group_keys, Vec::<u32>::new());
        assert_eq!(schema_of(branch).agg_state, Vec::new());
    }

    let routed = tree.children()[0];
    assert_eq!(
        types_of(routed),
        vec![
            ("k".to_string(), DataType::Utf8View),
            ("total".to_string(), summed),
        ]
    );
}

/// The arrow schema is shared rather than rebuilt where a node passes its input's columns
/// through, which is what makes "the types are DataFusion's own" true by construction
/// rather than by a second derivation agreeing.
#[tokio::test]
async fn a_pass_through_node_carries_its_inputs_own_schema() {
    let tree = translated("SELECT * FROM nation WHERE n_nationkey > 3", 1).await;
    let filter = tree.children()[0];
    assert!(Arc::ptr_eq(
        &schema_of(filter).fields,
        &schema_of(filter.children()[0]).fields
    ));
}

/// The planner end to end, through `plan_batch_partitioned` rather than through a rule
/// called directly: the two tests below are what pins that validation is wired into it at
/// all — every other test here calls the translator or a node's own check.
async fn planned(
    sql: &str,
    target_partitions: usize,
) -> Result<(), crate::batch_partitioned::PlanError> {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal");
    let ctx = crate::register_tables_for(crate::build_session_state(target_partitions), &data)
        .await
        .expect("register the minimal tables");
    let plan = ctx
        .sql(sql)
        .await
        .expect("plan the query")
        .create_physical_plan()
        .await
        .expect("physical plan");
    plan_batch_partitioned(&plan, knobs(target_partitions)).map(|_| ())
}

fn knobs(target_partitions: usize) -> PlanKnobs {
    PlanKnobs {
        target_partitions,
        sizing: BatchSizing::OneBatchPerLane,
        budget: 2 * 1024 * 1024 * 1024,
        small_table_bytes: 5 * 1024 * 1024,
    }
}

#[tokio::test]
async fn a_limit_inside_a_limit_plans() {
    // DataFusion merges adjacent limits, so the nested form only survives with something
    // between them: a limited aggregate subquery under a cross join, with a limit at the
    // root, arrives as an interval on the GpuUnload and another on a GpuLimit below it.
    // Each counts the stream its own node is handed, which is what nesting means.
    for tp in [1, 4] {
        assert_eq!(
            planned(
                "SELECT * FROM (SELECT p_brand FROM part GROUP BY p_brand LIMIT 7) x, \
                 nation n LIMIT 2",
                tp
            )
            .await,
            Ok(())
        );
    }
}

#[tokio::test]
async fn the_planner_refuses_a_tree_its_validation_rejects() {
    // A merge over batches nobody sorted. DataFusion never emits this shape, so it is
    // built by hand — and the point is where the refusal comes from: `plan_batch_partitioned`
    // itself, not `validate` called directly. Deleting the call from the planner turns this
    // red and nothing else.
    use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion::physical_plan::expressions::Column;
    use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal");
    let ctx = crate::register_tables_for(crate::build_session_state(1), &data)
        .await
        .expect("register the minimal tables");
    let scan = ctx
        .sql("SELECT p_partkey FROM part")
        .await
        .expect("plan the query")
        .create_physical_plan()
        .await
        .expect("physical plan");
    let unsorted_merge: Arc<dyn datafusion::physical_plan::ExecutionPlan> =
        Arc::new(SortPreservingMergeExec::new(
            LexOrdering::new(vec![PhysicalSortExpr::new_default(Arc::new(Column::new(
                "p_partkey",
                0,
            )))]),
            scan,
        ));

    match plan_batch_partitioned(&unsorted_merge, knobs(1)) {
        Err(crate::batch_partitioned::PlanError::Invalid(said)) => assert!(
            said.contains("GpuMergeSortedPartitions") && said.contains("GpuSort"),
            "the planner's refusal names the wrong fix: {said}"
        ),
        other => panic!("the planner planned a tree its own validation rejects: {other:?}"),
    }
}

#[tokio::test]
async fn the_planner_refuses_a_root_that_does_not_emit_what_the_query_asked_for() {
    // A bare Partial aggregate as the root. DataFusion declares `avg`'s state as
    // [count, sum] and this mode's decomposition reads [sum, count] — a deliberate
    // reordering (see `decompose`), so a root that stops at the partial emits its state
    // columns the other way round from the schema its own plan node declares. That is the
    // one shape where the two schemas differ without translation being wrong, which is
    // what makes it the staging for the root check. If the order ever stops being ours,
    // this test moves with it.
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};

    fn partial_of(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        let is_partial = plan
            .as_any()
            .downcast_ref::<AggregateExec>()
            .is_some_and(|aggregate| *aggregate.mode() == AggregateMode::Partial);
        if is_partial {
            return Some(plan.clone());
        }
        plan.children()
            .into_iter()
            .find_map(|child| partial_of(&child.clone()))
    }

    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal");
    let ctx = crate::register_tables_for(crate::build_session_state(4), &data)
        .await
        .expect("register the minimal tables");
    let plan = ctx
        .sql("SELECT p_brand, avg(p_retailprice) FROM part GROUP BY p_brand")
        .await
        .expect("plan the query")
        .create_physical_plan()
        .await
        .expect("physical plan");
    let partial = partial_of(&plan).expect("a partial aggregate to root at");

    match plan_batch_partitioned(&partial, knobs(4)) {
        Err(crate::batch_partitioned::PlanError::Invalid(said)) => assert!(
            said.contains("the plan emits") && said.contains("$sum"),
            "the planner's refusal names the wrong columns: {said}"
        ),
        other => panic!("the planner emitted columns the query did not ask for: {other:?}"),
    }
}
