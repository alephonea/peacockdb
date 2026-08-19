//! The join capability matrix, at plan time, over parquet this test writes itself.
//!
//! No goldens: a golden says a file moved, and what is under test here is which rule
//! decided what. Two lists, both positive: the hand-built matrix of every join type
//! crossed with a residual filter, and the sql that reaches each type. The shapes that are
//! REFUSED live in `test_planner_join_refusals.rs`, since what plans and what does not
//! read differently — except the three refusals no query reaches, which are at the end of
//! this file because a constructor is the only way to build them.
//!
//! The fixture is `common::join_fixture`, shared with that file.

mod common;

use common::join_fixture::{Fixture, LANES, join_types_in, planned};

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::JoinType;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, Column};
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::physical_plan::joins::{
    CrossJoinExec, HashJoinExec, NestedLoopJoinExec, PartitionMode,
};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning, PhysicalExpr};

use peacockdb_core::batch_partitioned::error::PlanError;
use peacockdb_core::batch_partitioned::layout::KeyDistribution;
use peacockdb_core::batch_partitioned::node::GpuNode;
use peacockdb_core::batch_partitioned::nodes::{NodeRef, as_node_ref};

fn equi(left: usize, right: usize) -> (Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>) {
    (
        Arc::new(Column::new("k", left)),
        Arc::new(Column::new("k", right)),
    )
}

/// A residual filter over one column of each side — the shape that decides three of the
/// refusals.
fn residual() -> JoinFilter {
    // The filter's own table names its columns as the sides do, which is what validation
    // checks a reference against.
    let schema = ArrowSchema::new(vec![
        Field::new("v", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
    ]);
    JoinFilter::new(
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new("v", 0)),
            Operator::Lt,
            Arc::new(Column::new("v", 1)),
        )),
        vec![
            ColumnIndex {
                index: 1,
                side: datafusion::common::JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: datafusion::common::JoinSide::Right,
            },
        ],
        Arc::new(schema),
    )
}

fn hash_join(
    build: Arc<dyn ExecutionPlan>,
    probe: Arc<dyn ExecutionPlan>,
    join_type: JoinType,
    filter: Option<JoinFilter>,
    null_equals_null: bool,
) -> Arc<dyn ExecutionPlan> {
    Arc::new(
        HashJoinExec::try_new(
            build,
            probe,
            vec![equi(0, 0)],
            filter,
            &join_type,
            None,
            PartitionMode::Partitioned,
            null_equals_null,
        )
        .expect("a hash join"),
    )
}

fn find<'a>(
    root: &'a dyn GpuNode,
    accept: &dyn Fn(&dyn GpuNode) -> bool,
) -> Option<&'a dyn GpuNode> {
    if accept(root) {
        return Some(root);
    }
    root.children()
        .into_iter()
        .find_map(|child| find(child, accept))
}

fn the_join(tree: &dyn GpuNode) -> &dyn GpuNode {
    find(tree, &|node| {
        matches!(
            as_node_ref(node),
            NodeRef::Join(_) | NodeRef::CrossJoin(_) | NodeRef::NestedLoopJoin(_)
        )
    })
    .expect("a join in the tree")
}

/// What the matrix promises for one mode, written from the spec rather than read back from
/// the planner's own `capability`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The probe streams; `finish` is whether the lane owes a pass at done.
    Streams {
        finish: bool,
    },
    /// The probe is made one batch first, and the join is then one legacy call.
    SingleBatchProbe {
        finish: bool,
    },
    Refused(&'static str),
}

/// Exhaustive by construction: a tenth join type stops this compiling rather than going
/// silently uncovered.
fn expected(join_type: JoinType, has_filter: bool) -> Outcome {
    match join_type {
        JoinType::Inner => Outcome::Streams { finish: false },
        // The build side is complete before the first probe call, so an unmatched probe row
        // is unmatched everywhere and nothing is owed at done.
        JoinType::Right => {
            if has_filter {
                Outcome::Refused("#153")
            } else {
                Outcome::Streams { finish: false }
            }
        }
        // A preserved build side is what the finish pass answers for.
        JoinType::Left | JoinType::Full => {
            if has_filter {
                Outcome::Refused("#153")
            } else {
                Outcome::Streams { finish: true }
            }
        }
        // The per-call join disappears — a probe call is only the key project — so the
        // filtered form gives up streaming instead of the finish.
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
            if has_filter {
                Outcome::SingleBatchProbe { finish: true }
            } else {
                Outcome::Streams { finish: true }
            }
        }
        // Membership in a complete build side is a per-row question, and no swapped mixed_*
        // variant exists to ask it with a residual.
        JoinType::RightSemi | JoinType::RightAnti => {
            if has_filter {
                Outcome::Refused("mixed_*")
            } else {
                Outcome::Streams { finish: false }
            }
        }
    }
}

const HASH_JOIN_TYPES: [JoinType; 9] = [
    JoinType::Inner,
    JoinType::Left,
    JoinType::Right,
    JoinType::Full,
    JoinType::LeftSemi,
    JoinType::RightSemi,
    JoinType::LeftAnti,
    JoinType::RightAnti,
    JoinType::LeftMark,
];

#[tokio::test]
async fn every_hash_join_type_plans_or_refuses_as_the_matrix_says() {
    let fixture = Fixture::new("matrix").await;
    for join_type in HASH_JOIN_TYPES {
        for has_filter in [false, true] {
            let filter = has_filter.then(residual);
            // Both keys are NULL-free here, so only the filter and the type decide.
            let plan = hash_join(
                fixture.scattered("tiny").await,
                fixture.scattered("big").await,
                join_type,
                filter,
                false,
            );
            let outcome = expected(join_type, has_filter);
            match (planned(&plan), outcome) {
                (Err(PlanError::Unsupported(what)), Outcome::Refused(names)) => {
                    assert!(
                        what.contains(names),
                        "{join_type:?} filter={has_filter}: refused without naming {names}: {what}"
                    );
                }
                (Ok(tree), Outcome::Streams { finish } | Outcome::SingleBatchProbe { finish }) => {
                    let join = the_join(tree.as_ref());
                    let NodeRef::Join(node) = as_node_ref(join) else {
                        panic!("{join_type:?}: not an equi-join");
                    };
                    let capability = node.capability().expect("a capability");
                    assert_eq!(
                        capability.needs_finish, finish,
                        "{join_type:?} filter={has_filter}: finish"
                    );
                    let streams = matches!(outcome, Outcome::Streams { .. });
                    assert_eq!(
                        capability.probe_streams, streams,
                        "{join_type:?} filter={has_filter}: streaming"
                    );
                    // A probe that cannot stream is made one batch by the planner, which is
                    // the tree consequence rather than the node's own answer.
                    let probe_coalesced = matches!(
                        as_node_ref(join.children()[1]),
                        NodeRef::CoalesceAllBatches(_)
                    );
                    assert_eq!(
                        probe_coalesced, !streams,
                        "{join_type:?} filter={has_filter}: probe coalesce"
                    );
                }
                (planned, expected) => {
                    panic!(
                        "{join_type:?} filter={has_filter}: got {planned:?}, expected {expected:?}"
                    )
                }
            }
        }
    }
}

#[tokio::test]
async fn a_co_partitioned_join_keeps_its_lanes_and_both_sides_carry_the_same_hash() {
    let fixture = Fixture::new("copartitioned").await;
    let plan = hash_join(
        fixture.scattered("tiny").await,
        fixture.scattered("big").await,
        JoinType::Inner,
        None,
        false,
    );
    let tree = planned(&plan).expect("a co-partitioned join plans");
    let join = the_join(tree.as_ref());
    assert_eq!(join.kind().layout().unwrap().n, LANES);
    let hash_of = |node: &dyn GpuNode| node.kind().layout().unwrap().key_distribution.clone();
    // Equal lane counts are not co-location: both sides must be scattered on their key.
    assert_eq!(
        hash_of(join.children()[0]),
        KeyDistribution::ByHash { hash_keys: vec![0] }
    );
    assert_eq!(
        hash_of(join.children()[1]),
        KeyDistribution::ByHash { hash_keys: vec![0] }
    );
}

#[tokio::test]
async fn a_join_whose_sides_are_not_co_located_runs_in_one_lane() {
    let fixture = Fixture::new("broadcast").await;
    // Neither side scattered: four lanes each, agreeing on nothing.
    let plan = hash_join(
        fixture.scan("tiny").await,
        fixture.scan("big").await,
        JoinType::Inner,
        None,
        false,
    );
    let tree = planned(&plan).expect("it plans");
    let join = the_join(tree.as_ref());
    assert_eq!(
        join.kind().layout().unwrap().n,
        1,
        "#140 is what would keep the lanes"
    );
}

#[tokio::test]
async fn a_small_build_side_plans_one_lane_and_a_large_one_does_not() {
    let fixture = Fixture::new("threshold").await;
    for (table, lanes) in [("tiny", 1), ("big", LANES)] {
        let plan = hash_join(
            fixture.scan(table).await,
            fixture.scan(table).await,
            JoinType::Inner,
            None,
            false,
        );
        let tree = planned(&plan).expect("it plans");
        let loader = find(tree.as_ref(), &|node| {
            matches!(as_node_ref(node), NodeRef::LoadParquet(_))
        })
        .expect("a loader");
        assert_eq!(
            loader.kind().layout().unwrap().n,
            lanes,
            "{table} is on the wrong side of the threshold"
        );
    }
}

#[tokio::test]
async fn an_anti_join_refuses_only_when_nulls_can_meet_on_both_sides() {
    let fixture = Fixture::new("nulls").await;
    for join_type in [JoinType::LeftAnti, JoinType::RightAnti, JoinType::LeftMark] {
        // Only the probe side carries NULLs: a NULL there matches nothing on a NULL-free
        // build, so the hardcoded EQUAL cannot invent a match. This must plan — it is what
        // tpcds q10, q35 and q69 rest on.
        let one_side = hash_join(
            fixture.scattered("tiny").await,
            fixture.scattered("nulls").await,
            join_type,
            None,
            false,
        );
        assert!(
            planned(&one_side).is_ok(),
            "{join_type:?}: a NULL-free build side has no NULL to match"
        );

        // Both sides carry them, and the executor matches NULL to NULL whatever the flag
        // says — which is set semantics, not SQL's.
        let both_sides = hash_join(
            fixture.scattered("nulls").await,
            fixture.scattered("nulls").await,
            join_type,
            None,
            false,
        );
        let err = planned(&both_sides).expect_err("both sides nullable must refuse");
        assert!(
            matches!(&err, PlanError::Unsupported(what) if what.contains("#59") && what.contains("#80")),
            "{join_type:?}: {err}"
        );

        // The same plan asking for set semantics is asking for what the executor does.
        let set_semantics = hash_join(
            fixture.scattered("nulls").await,
            fixture.scattered("nulls").await,
            join_type,
            None,
            true,
        );
        assert!(
            planned(&set_semantics).is_ok(),
            "{join_type:?}: null_equals_null=true is the equality the executor hardcodes"
        );
    }
}

#[tokio::test]
async fn a_semi_join_plans_with_nulls_on_both_sides_because_it_honours_the_flag() {
    let fixture = Fixture::new("semi").await;
    for join_type in [JoinType::LeftSemi, JoinType::RightSemi] {
        let plan = hash_join(
            fixture.scattered("nulls").await,
            fixture.scattered("nulls").await,
            join_type,
            None,
            false,
        );
        assert!(
            planned(&plan).is_ok(),
            "{join_type:?}: semi takes null_equality from the plan, so SQL semantics are available"
        );
    }
}

#[tokio::test]
async fn an_outer_join_makes_its_padded_side_nullable_for_the_join_above_it() {
    let fixture = Fixture::new("padding").await;
    // The Left join pads its probe side, so the anti join above it meets NULLs on both
    // sides even though no column in either file holds one.
    let padded = hash_join(
        fixture.scattered("tiny").await,
        fixture.scattered("tiny").await,
        JoinType::Left,
        None,
        false,
    );
    let scattered = Arc::new(
        RepartitionExec::try_new(
            padded,
            Partitioning::Hash(vec![Arc::new(Column::new("k", 2))], LANES),
        )
        .expect("a hash repartition"),
    );
    let anti = Arc::new(
        HashJoinExec::try_new(
            fixture.scattered("nulls").await,
            scattered,
            // Column 2 of the Left join's output is its padded side's key.
            vec![(Arc::new(Column::new("k", 0)), Arc::new(Column::new("k", 2)))],
            None,
            &JoinType::LeftAnti,
            None,
            PartitionMode::Partitioned,
            false,
        )
        .expect("a hash join"),
    ) as Arc<dyn ExecutionPlan>;
    let err = planned(&anti).expect_err("the padding is a NULL the analysis must see");
    assert!(
        matches!(&err, PlanError::Unsupported(what) if what.contains("#59")),
        "{err}"
    );
}

#[tokio::test]
async fn cross_and_nested_loop_joins_take_both_sides_to_one_lane() {
    let fixture = Fixture::new("keyless").await;
    let cross: Arc<dyn ExecutionPlan> = Arc::new(CrossJoinExec::new(
        fixture.scan("tiny").await,
        fixture.scan("big").await,
    ));
    let tree = planned(&cross).expect("a cross join plans");
    let join = the_join(tree.as_ref());
    assert!(matches!(as_node_ref(join), NodeRef::CrossJoin(_)));
    for side in join.children() {
        assert_eq!(side.kind().layout().unwrap().n, 1, "no key to co-locate on");
    }

    let nested: Arc<dyn ExecutionPlan> = Arc::new(
        NestedLoopJoinExec::try_new(
            fixture.scan("tiny").await,
            fixture.scan("big").await,
            Some(residual()),
            &JoinType::Inner,
            None,
        )
        .expect("a nested-loop join"),
    );
    let tree = planned(&nested).expect("a nested-loop join plans");
    let join = the_join(tree.as_ref());
    assert!(matches!(as_node_ref(join), NodeRef::NestedLoopJoin(_)));
    for side in join.children() {
        assert_eq!(side.kind().layout().unwrap().n, 1);
    }
}

#[tokio::test]
async fn a_nested_loop_join_beyond_inner_and_left_is_refused() {
    let fixture = Fixture::new("nlj-types").await;
    for join_type in [JoinType::Right, JoinType::Full, JoinType::LeftSemi] {
        let plan: Arc<dyn ExecutionPlan> = Arc::new(
            NestedLoopJoinExec::try_new(
                fixture.scan("tiny").await,
                fixture.scan("big").await,
                Some(residual()),
                &join_type,
                None,
            )
            .expect("a nested-loop join"),
        );
        let err = planned(&plan).expect_err("the executor rejects these outright");
        assert!(
            matches!(&err, PlanError::Unsupported(what) if what.contains("Inner and Left")),
            "{join_type:?}: {err}"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// What a QUERY reaches
// ════════════════════════════════════════════════════════════════════════════
// The matrix above proves the planner handles every join type. It says nothing about
// which of them a query produces, and a refusal guarding a shape nothing reaches is a
// cost with no return. These are hand-written queries, one per type, asserting what each
// plans as — so the file records the mapping from sql to join type rather than assuming
// it, and a type with no query is visible as a gap rather than absent.

/// The query that produces each join type against this fixture. Exhaustive by
/// construction, like the matrix: no wildcard arm, so a tenth variant stops the build.
///
/// Four of them depend on the fixture's row counts rather than on their own text —
/// DataFusion puts the smaller side on the build, which is what turns the Left forms into
/// the Right ones. Those are marked.
fn sql_for(join_type: JoinType) -> &'static str {
    match join_type {
        JoinType::Inner => "SELECT t.v FROM tiny t JOIN big b ON t.k = b.k",
        JoinType::Left => "SELECT t.v FROM tiny t LEFT JOIN big b ON t.k = b.k",
        // SWAP: the same LEFT JOIN with the big table on the outside. DataFusion swaps
        // the sides so the small one builds, and remaps the type to Right.
        JoinType::Right => "SELECT b.v FROM big b LEFT JOIN tiny t ON b.k = t.k",
        JoinType::Full => "SELECT t.v FROM tiny t FULL OUTER JOIN big b ON t.k = b.k",
        JoinType::LeftSemi => {
            "SELECT t.v FROM tiny t WHERE EXISTS (SELECT 1 FROM big b WHERE b.k = t.k)"
        }
        // SWAP: big on the outside, tiny inside the EXISTS.
        JoinType::RightSemi => {
            "SELECT b.v FROM big b WHERE EXISTS (SELECT 1 FROM tiny t WHERE t.k = b.k)"
        }
        JoinType::LeftAnti => {
            "SELECT t.v FROM tiny t WHERE NOT EXISTS (SELECT 1 FROM big b WHERE b.k = t.k)"
        }
        // SWAP: as RightSemi, negated.
        JoinType::RightAnti => {
            "SELECT b.v FROM big b WHERE NOT EXISTS (SELECT 1 FROM tiny t WHERE t.k = b.k)"
        }
        // A bare IN is a semi join; the mark form wants the membership to be one arm of a
        // disjunction, where a boolean per left row is the only thing that answers it.
        JoinType::LeftMark => "SELECT t.v FROM tiny t WHERE t.k IN (SELECT k FROM big) OR t.v > 5",
    }
}

#[tokio::test]
async fn every_join_type_is_reached_by_a_query() {
    let fixture = Fixture::new("sql-matrix").await;
    for join_type in HASH_JOIN_TYPES {
        let plan = fixture.plan(sql_for(join_type)).await;
        assert_eq!(
            join_types_in(&plan),
            vec![join_type],
            "{}\nwas expected to plan as {join_type:?}",
            sql_for(join_type)
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Refusals no query reaches
// ════════════════════════════════════════════════════════════════════════════
// Three shapes the planner refuses that nothing in sql produces. Each is built here
// rather than left uncovered, and each says what was tried — an unreachable refusal is a
// cost, and the next reader should be able to weigh it without redoing the search.

#[tokio::test]
async fn a_join_key_that_is_not_a_bare_column_is_refused_and_no_query_makes_one() {
    let fixture = Fixture::new("refuse-expr-key").await;
    // `... ON t.k + 1 = b.k` plans as an ordinary hash join on columns: DataFusion
    // projects the expression below the join, so the key it hashes is always a column.
    let projected = fixture
        .plan("SELECT t.v FROM tiny t JOIN big b ON t.k + 1 = b.k")
        .await;
    assert_eq!(join_types_in(&projected), vec![JoinType::Inner]);

    // So the shape only exists by construction.
    let build = fixture.scattered("tiny").await;
    let probe = fixture.scattered("big").await;
    let computed: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("k", 0)),
        Operator::Plus,
        Arc::new(datafusion::physical_expr::expressions::Literal::new(
            datafusion::common::ScalarValue::Int64(Some(1)),
        )),
    ));
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            build,
            probe,
            vec![(computed, Arc::new(Column::new("k", 0)))],
            None,
            &JoinType::Inner,
            None,
            PartitionMode::Partitioned,
            false,
        )
        .expect("a hash join on an expression"),
    );
    let err = planned(&plan).expect_err("an expression key has no ordinal to hash");
    assert!(
        matches!(&err, PlanError::Unsupported(what) if what.contains("join key")),
        "{err}"
    );
}

#[tokio::test]
async fn a_final_aggregate_over_something_other_than_a_partial_is_refused_and_no_query_makes_one() {
    let fixture = Fixture::new("refuse-final-alone").await;
    // DataFusion emits Final only as the upper half of a pair it built itself, so this is
    // an internal-consistency refusal: it catches this mode assembling a sequence wrongly,
    // never a query.
    use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
    let scan = fixture.scan("tiny").await;
    let group = PhysicalGroupBy::new_single(vec![(
        Arc::new(Column::new("k", 0)) as Arc<dyn PhysicalExpr>,
        "k".to_string(),
    )]);
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Final,
            group,
            vec![],
            vec![],
            scan.clone(),
            scan.schema(),
        )
        .expect("a final aggregate"),
    );
    let err = planned(&plan).expect_err("a final over a scan is not a sequence");
    assert!(
        matches!(&err, PlanError::Unsupported(what) if what.contains("rather than a partial")),
        "{err}"
    );
}

#[tokio::test]
async fn a_repartition_by_an_unstated_rule_is_refused_and_no_query_makes_one() {
    let fixture = Fixture::new("refuse-unknown-partitioning").await;
    // DataFusion's planner emits RoundRobinBatch and Hash; UnknownPartitioning is a
    // property a node DECLARES about itself, not a repartition target, so no query builds
    // one. Refused rather than guessed at: a lane assignment nobody chose is worse than a
    // plan that does not run.
    let scan = fixture.scan("tiny").await;
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        RepartitionExec::try_new(scan, Partitioning::UnknownPartitioning(LANES))
            .expect("an unknown repartition"),
    );
    let err = planned(&plan).expect_err("an unstated rule is not a rule");
    assert!(
        matches!(&err, PlanError::Unsupported(what) if what.contains("unstated rule")),
        "{err}"
    );
}
