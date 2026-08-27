//! A recipe plan on a live GPU, driven by hand: the first time anything this mode plans
//! meets a device.
//!
//! One helper walks the tree making exactly the calls each recipe names, threading every
//! output handle into the next call's input and exporting at the root; one test per query,
//! so a failure names the query rather than a stage. No driver and no scheduling — every
//! shape here plans one batch per lane, so a recipe's own call order is the schedule.
//!
//! The oracle is DataFusion on the same SQL: a golden would pin a wrong finalize on its
//! first run, and our CPU executor evaluates the very finalize the device is sent.
#![cfg(not(feature = "rust-only"))]
#[macro_use]
mod common;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::ipc::reader::StreamReader;
use datafusion::common::JoinType;

use peacockdb_core::batch_partitioned::forwarder::{BatchForwarder, forwarder_for};
use peacockdb_core::batch_partitioned::node::GpuNode;
use peacockdb_core::batch_partitioned::nodes::{NodeRef, as_node_ref};
use peacockdb_core::batch_partitioned::plan::{BatchSizing, PlanKnobs, plan_batch_partitioned};
use peacockdb_core::batch_partitioned::recipe::{
    AbiSymbol, Call, CallPattern, FbKind, Input, ProjectRole, Recipe, RecipePlan, Seq,
    attach_recipes,
};
use peacockdb_core::batch_partitioned::{ExecutorCategory, category_of};
use peacockdb_ffi::raw::{
    PeacockExecutor, PeacockNodeStats, peacock_executor_begin_plan, peacock_executor_create,
    peacock_executor_destroy, peacock_executor_end_plan, peacock_executor_execute_node,
    peacock_executor_execute_scan_rowgroups, peacock_last_error, peacock_result_free,
    peacock_result_from_handle,
};

use common::exec_mode::CpuOracle;
use common::{GPU_BUDGET, assert_results_match, data_dir_for, total_rows};

/// A scan reading less than this stops being worth splitting. The value the plan goldens
/// are canonized at, so every shape below is one that tier already renders.
const SMALL_TABLE_BYTES: u64 = 5 * 1024 * 1024;

/// Everything but the aggregates: one lane and one batch, which makes every recipe a
/// single call per node and the walk a straight line.
const ONE_LANE: PlanKnobs = PlanKnobs {
    target_partitions: 1,
    sizing: BatchSizing::OneBatchPerLane,
    budget: GPU_BUDGET as u64,
    small_table_bytes: SMALL_TABLE_BYTES,
};

/// The aggregates, at one batch and two lanes: a merge is the operator this mode adds and
/// one lane never performs one.
const TWO_LANES: PlanKnobs = PlanKnobs {
    target_partitions: 2,
    sizing: BatchSizing::OneBatchPerLane,
    budget: GPU_BUDGET as u64,
    small_table_bytes: SMALL_TABLE_BYTES,
};

/// The handles a node produced: one entry per lane, holding that lane's batches in
/// arrival order.
type Lanes = Vec<Vec<u64>>;

/// An executor with a recipe plan loaded, torn down in the order the header requires.
struct Session {
    executor: *mut PeacockExecutor,
}

impl Session {
    /// `begin_plan`'s `out_node_count` is asserted against the fb nodes the writer created
    /// — never against the plan tree's own count, which differs in most plans. Until a
    /// device has parsed a buffer we wrote, our agreement with the C++ post-order rested on
    /// two child-order functions having been read side by side; this is the first place
    /// both numbers exist at once.
    fn open(recipes: &RecipePlan) -> Self {
        let mut executor: *mut PeacockExecutor = std::ptr::null_mut();
        assert_eq!(
            unsafe { peacock_executor_create(GPU_BUDGET as u64, &mut executor) },
            0,
            "peacock_executor_create failed"
        );
        let session = Self { executor };
        let bytes = recipes.bytes();
        let mut nodes = 0u64;
        let rc = unsafe {
            peacock_executor_begin_plan(executor, bytes.as_ptr(), bytes.len() as u64, &mut nodes)
        };
        assert_eq!(rc, 0, "begin_plan failed: {}", session.last_error());
        assert_eq!(
            nodes as usize,
            recipes.wire_nodes(),
            "the C++ post-order holds {nodes} nodes and the writer created {} — every seq \
             a recipe publishes is an index into that walk, so the two numbering the same \
             tree is what makes a call address the node it names",
            recipes.wire_nodes()
        );
        session
    }

    fn last_error(&self) -> String {
        let message = unsafe { peacock_last_error(self.executor) };
        if message.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    }

    /// One batch's worth of a scan: the row groups the mapping named for it, overriding
    /// the list the node carries.
    fn scan(&self, seq: Seq, row_groups: &[u32]) -> u64 {
        let mut handle = 0u64;
        let mut stats = PeacockNodeStats::default();
        let rc = unsafe {
            peacock_executor_execute_scan_rowgroups(
                self.executor,
                seq as u64,
                row_groups.as_ptr(),
                row_groups.len() as u64,
                &mut handle,
                &mut stats,
            )
        };
        assert_eq!(
            rc,
            0,
            "execute_scan_rowgroups(#{seq}, {row_groups:?}) failed: {}",
            self.last_error()
        );
        handle
    }

    /// One `execute_node`, its input handles grouped by the child slot each fills.
    fn execute(&self, seq: Seq, inputs: &[Vec<u64>], out_cap: usize) -> Vec<u64> {
        let counts: Vec<u64> = inputs.iter().map(|group| group.len() as u64).collect();
        let flat: Vec<u64> = inputs.concat();
        let mut handles = vec![0u64; out_cap];
        let mut stats = vec![PeacockNodeStats::default(); out_cap];
        let mut produced = 0u64;
        let rc = unsafe {
            peacock_executor_execute_node(
                self.executor,
                seq as u64,
                flat.as_ptr(),
                counts.as_ptr(),
                counts.len() as u64,
                handles.as_mut_ptr(),
                out_cap as u64,
                &mut produced,
                stats.as_mut_ptr(),
            )
        };
        assert_eq!(
            rc,
            0,
            "execute_node(#{seq}, {counts:?} handles) failed: {}",
            self.last_error()
        );
        handles.truncate(produced as usize);
        handles
    }

    /// The whole handle across the boundary. No shape here plans a limit, so the sink's
    /// range is always the batch it is handed.
    fn export(&self, handle: u64) -> Vec<RecordBatch> {
        let mut ipc: *mut u8 = std::ptr::null_mut();
        let mut len = 0u64;
        let rc = unsafe {
            peacock_result_from_handle(self.executor, handle, 0, u64::MAX, &mut ipc, &mut len)
        };
        assert_eq!(rc, 0, "result_from_handle failed: {}", self.last_error());
        if len == 0 {
            return Vec::new();
        }
        let bytes = unsafe { std::slice::from_raw_parts(ipc, len as usize) };
        let batches = StreamReader::try_new(std::io::Cursor::new(bytes), None)
            .and_then(|stream| stream.collect::<Result<Vec<_>, _>>())
            .expect("the exported IPC stream decodes");
        unsafe { peacock_result_free(ipc) };
        batches
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            peacock_executor_end_plan(self.executor);
            peacock_executor_destroy(self.executor);
        }
    }
}

/// Where a call's named inputs are, at the moment the walk makes it. Every one is a handle
/// the walk already holds, which is what lets a recipe be driven without re-reading the
/// node it came from.
#[derive(Default)]
struct At {
    batch: Option<u64>,
    build: Option<u64>,
    lane: Vec<u64>,
    all_lanes: Vec<u64>,
    prior: Option<u64>,
}

/// Where a call's named input comes from, in the handles the walk is already holding.
///
/// The two copies resolve to the handle itself. They name a device copy of something a
/// later call still needs and the ABI has no symbol for one (#152) — but every shape here
/// plans a single probe batch, so the handle is used once and handed over, which is what
/// the join arm asserts before anything reaches this.
fn resolve(input: Input, at: &At) -> Vec<u64> {
    let held = |handle: Option<u64>| vec![handle.unwrap_or_else(|| panic!("no {input:?} here"))];
    match input {
        Input::Batch | Input::BatchCopy => held(at.batch),
        Input::BuildSide | Input::BuildSideCopy => held(at.build),
        Input::PriorOutput => held(at.prior),
        Input::LaneBatches => at.lane.clone(),
        Input::AllLanes => at.all_lanes.clone(),
        Input::AccumulatedKeys | Input::RowGroups | Input::RowRange => {
            panic!("{input:?} is not a handle the walk holds")
        }
    }
}

/// A node's result where exactly one handle was expected.
fn only(handles: Vec<u64>, what: &str) -> u64 {
    match handles.as_slice() {
        [handle] => *handle,
        other => panic!("{what}: {} handles rather than one", other.len()),
    }
}

/// The walk: where the recipes are, which node is next in the post-order they are indexed
/// by, and what reached the sink.
struct Walk<'a> {
    session: &'a Session,
    recipes: &'a RecipePlan,
    next_node: usize,
    exported: Vec<RecordBatch>,
    /// Every call made, in the order it was made. A test reads it when its claim is about
    /// the calls rather than the answer, and a failure prints it: a wrong table is the
    /// symptom of one call, and the seq is what a reader looks up in the payload golden.
    made: Vec<(Seq, FbKind)>,
}

impl Walk<'_> {
    /// Children first, then this node: the same post-order `attach_recipes` indexed its
    /// recipes by, so the position it takes here is the position they are stored at.
    fn node(&mut self, node: &dyn GpuNode) -> Lanes {
        let kids: Vec<Lanes> = node
            .children()
            .into_iter()
            .map(|child| self.node(child))
            .collect();
        let index = self.next_node;
        self.next_node += 1;
        let category = category_of(node);
        if category == ExecutorCategory::BatchForwarder {
            return route(node, &kids);
        }
        let recipe = self
            .recipes
            .get(index)
            .unwrap_or_else(|| panic!("{} makes ABI calls and carries no recipe", node.name()));
        match category {
            ExecutorCategory::Source => self.source(node, recipe),
            ExecutorCategory::Exec => self.per_batch(node, recipe, &kids[0]),
            ExecutorCategory::PartitionEmitter => self.emit_partitions(node, recipe, &kids[0]),
            ExecutorCategory::BatchAccumulator => self.per_lane(node, recipe, &kids[0]),
            ExecutorCategory::PartitionAccumulator => self.over_all_lanes(recipe, &kids[0]),
            ExecutorCategory::Join => self.join(node, recipe, &kids[0], &kids[1]),
            ExecutorCategory::Unload => self.unload(node, recipe, &kids[0]),
            ExecutorCategory::BatchForwarder => unreachable!("returned above"),
        }
    }

    /// The recipe's calls in order, each one's prior output being the last one's.
    fn chain(&mut self, calls: &[&Call], at: &mut At) -> Vec<u64> {
        let mut produced = Vec::new();
        for call in calls {
            produced = self.make(call, at);
            at.prior = produced.first().copied();
        }
        produced
    }

    fn make(&mut self, call: &Call, at: &At) -> Vec<u64> {
        let (seq, kind) = call.target.unwrap_or_else(|| {
            panic!(
                "{} takes runtime bounds rather than a seq, and no shape here plans one",
                call.symbol.name()
            )
        });
        let inputs: Vec<Vec<u64>> = call.inputs.iter().map(|from| resolve(*from, at)).collect();
        let out_cap = match kind {
            FbKind::Repartition { lanes } => lanes as usize,
            _ => 1,
        };
        self.made.push((seq, kind));
        self.session.execute(seq, &inputs, out_cap)
    }

    fn source(&mut self, node: &dyn GpuNode, recipe: &Recipe) -> Lanes {
        let NodeRef::LoadParquet(load) = as_node_ref(node) else {
            unreachable!("the source category holds one node kind")
        };
        let [call] = recipe.calls.as_slice() else {
            panic!("a scan's recipe is one call per batch")
        };
        assert_eq!(call.symbol, AbiSymbol::ExecuteScanRowGroups);
        let (seq, kind) = call.target.expect("a scan addresses its own node");
        let mut lanes = Vec::with_capacity(load.partition_groups.len());
        for lane in &load.partition_groups {
            let mut batches = Vec::with_capacity(lane.len());
            for row_groups in lane {
                self.made.push((seq, kind));
                batches.push(self.session.scan(seq, row_groups));
            }
            lanes.push(batches);
        }
        lanes
    }

    /// The map arms: one call chain per batch, output keeping its input's lane and batch
    /// structure.
    fn per_batch(&mut self, node: &dyn GpuNode, recipe: &Recipe, input: &Lanes) -> Lanes {
        let calls: Vec<&Call> = recipe.calls.iter().collect();
        let mut lanes = Vec::with_capacity(input.len());
        for lane in input {
            let mut batches = Vec::with_capacity(lane.len());
            for handle in lane {
                let mut at = At {
                    batch: Some(*handle),
                    ..At::default()
                };
                batches.push(only(self.chain(&calls, &mut at), node.name()));
            }
            lanes.push(batches);
        }
        lanes
    }

    /// The emitter's one call answers with a handle per output lane, so a batch of lane p
    /// is the p-th handle of every call its input made.
    fn emit_partitions(&mut self, node: &dyn GpuNode, recipe: &Recipe, input: &Lanes) -> Lanes {
        let calls: Vec<&Call> = recipe.calls.iter().collect();
        let out_lanes = match recipe.calls.first().and_then(|call| call.target) {
            Some((_, FbKind::Repartition { lanes })) => lanes as usize,
            other => panic!("{}: expected a repartition, got {other:?}", node.name()),
        };
        let mut lanes = vec![Vec::new(); out_lanes];
        for lane in input {
            for handle in lane {
                let mut at = At {
                    batch: Some(*handle),
                    ..At::default()
                };
                let scattered = self.chain(&calls, &mut at);
                assert_eq!(
                    scattered.len(),
                    out_lanes,
                    "{}: the scatter answered with {} handles",
                    node.name(),
                    scattered.len()
                );
                for (out, handle) in lanes.iter_mut().zip(scattered) {
                    out.push(handle);
                }
            }
        }
        lanes
    }

    /// An accumulator: whatever it does per batch, then its at-done calls once over the
    /// lane it accumulated. The two phases are the recipe's own grouping.
    fn per_lane(&mut self, node: &dyn GpuNode, recipe: &Recipe, input: &Lanes) -> Lanes {
        let (streamed, at_done) = phases(recipe);
        assert!(
            !at_done.is_empty(),
            "{}: nothing runs at done, so the lane it accumulated has no output — a \
             streaming limit is the shape that reaches here, and none is planned",
            node.name()
        );
        let mut lanes = Vec::with_capacity(input.len());
        for lane in input {
            let mut held = Vec::with_capacity(lane.len());
            for handle in lane {
                if streamed.is_empty() {
                    held.push(*handle);
                    continue;
                }
                let mut at = At {
                    batch: Some(*handle),
                    ..At::default()
                };
                held.push(only(self.chain(&streamed, &mut at), node.name()));
            }
            let mut at = At {
                lane: held,
                ..At::default()
            };
            lanes.push(vec![only(self.chain(&at_done, &mut at), node.name())]);
        }
        lanes
    }

    /// One call over every lane's handle, partition-major, answering with one lane.
    fn over_all_lanes(&mut self, recipe: &Recipe, input: &Lanes) -> Lanes {
        let calls: Vec<&Call> = recipe.calls.iter().collect();
        let mut at = At {
            all_lanes: input.concat(),
            ..At::default()
        };
        vec![vec![only(
            self.chain(&calls, &mut at),
            "a partition accumulator",
        )]]
    }

    fn join(&mut self, node: &dyn GpuNode, recipe: &Recipe, build: &Lanes, probe: &Lanes) -> Lanes {
        assert!(
            recipe
                .calls
                .iter()
                .all(|call| call.when == CallPattern::PerProbeBatch),
            "{}: a finish pass accumulates probe keys across batches (#136), which no shape \
             here plans",
            node.name()
        );
        assert_eq!(
            build.len(),
            probe.len(),
            "{}: lane p of one side must hold what can match lane p of the other",
            node.name()
        );
        let calls: Vec<&Call> = recipe.calls.iter().collect();
        let mut lanes = Vec::with_capacity(build.len());
        for (build_lane, probe_lane) in build.iter().zip(probe) {
            assert_eq!(
                probe_lane.len(),
                1,
                "{}: {} probe batches, and the call consumes the build handle with no ABI \
                 symbol to copy it (#152) — every shape here plans one probe batch",
                node.name(),
                probe_lane.len()
            );
            let mut at = At {
                batch: Some(probe_lane[0]),
                build: Some(only(build_lane.clone(), "a join's build side")),
                ..At::default()
            };
            lanes.push(vec![only(self.chain(&calls, &mut at), node.name())]);
        }
        lanes
    }

    /// The sink produces results rather than handles, so it answers with no lanes.
    fn unload(&mut self, node: &dyn GpuNode, recipe: &Recipe, input: &Lanes) -> Lanes {
        assert!(
            node.row_interval().is_none(),
            "a root-adjacent limit gives the sink a row range per handle, and no shape here \
             plans one"
        );
        let [call] = recipe.calls.as_slice() else {
            panic!("a sink's recipe is one call per handle")
        };
        assert_eq!(call.symbol, AbiSymbol::ResultFromHandle);
        for lane in input {
            for handle in lane {
                let batches = self.session.export(*handle);
                self.exported.extend(batches);
            }
        }
        Vec::new()
    }
}

/// A recipe's calls split into the two phases a walk drives: what runs as batches arrive,
/// and what runs once the lane is complete. A compaction runs exactly what done runs, so
/// at one batch per lane the two are the same call.
fn phases(recipe: &Recipe) -> (Vec<&Call>, Vec<&Call>) {
    recipe.calls.iter().partition(|call| {
        matches!(
            call.when,
            CallPattern::PerBatch | CallPattern::PerProbeBatch
        )
    })
}

/// No calls at all: a forwarder renumbers lanes, and `forwarder_for` is the routing the
/// drivers read off the same node. One batch per visit, cycling the sources in the order
/// the forwarder lists them.
fn route(node: &dyn GpuNode, kids: &[Lanes]) -> Lanes {
    let forwarder = forwarder_for(node);
    let out_lanes = node.kind().layout().map_or(1, |layout| layout.n);
    (0..out_lanes)
        .map(|out_lane| {
            let mut queues: Vec<&[u64]> = forwarder
                .sources_of(out_lane)
                .iter()
                .map(|(child, lane)| kids[*child][*lane].as_slice())
                .collect();
            let mut forwarded = Vec::new();
            while queues.iter().any(|queue| !queue.is_empty()) {
                for queue in queues.iter_mut() {
                    if let Some((first, rest)) = queue.split_first() {
                        forwarded.push(*first);
                        *queue = rest;
                    }
                }
            }
            forwarded
        })
        .collect()
}

/// What one walk of one query produced.
struct Walked {
    batches: Vec<RecordBatch>,
    calls: Vec<(Seq, FbKind)>,
}

/// The calls in the order they were made, `#14 CudfAggregate{Merge}` each. Three of the
/// four defects this file found were found by whoever had just written the code and the
/// fourth by a grep; neither is available to whoever runs it next, and a wrong table names
/// no call on its own.
fn trail(calls: &[(Seq, FbKind)]) -> String {
    calls
        .iter()
        .map(|(seq, kind)| format!("#{seq} {kind}"))
        .collect::<Vec<_>>()
        .join(", ")
}

async fn context(target_partitions: usize) -> datafusion::execution::context::SessionContext {
    peacockdb_core::register_tables_for(
        peacockdb_core::build_session_state(target_partitions),
        &data_dir_for("tpch", "1"),
    )
    .await
    .expect("register the tpch sf1 tables")
}

/// Plan the query in this mode, hand the recipe plan to a device, and make the calls.
async fn walk(sql: &str, knobs: PlanKnobs) -> Walked {
    let ctx = context(knobs.target_partitions).await;
    let plan = ctx
        .sql(sql)
        .await
        .expect("datafusion plans it")
        .create_physical_plan()
        .await
        .expect("datafusion lowers it");
    let (tree, _) = plan_batch_partitioned(&plan, knobs).expect("this mode plans it");
    let recipes = attach_recipes(tree.as_ref()).expect("a planned tree has recipes");
    let session = Session::open(&recipes);
    let mut walk = Walk {
        session: &session,
        recipes: &recipes,
        next_node: 0,
        exported: Vec::new(),
        made: Vec::new(),
    };
    let left = walk.node(tree.as_ref());
    assert!(left.is_empty(), "the sink answered with resident handles");
    assert_eq!(
        walk.next_node,
        recipes.nodes(),
        "the walk visited {} of the plan's {} nodes",
        walk.next_node,
        recipes.nodes()
    );
    Walked {
        batches: walk.exported,
        calls: walk.made,
    }
}

/// The walk's answer against DataFusion's on the same SQL, compared as sorted multisets
/// since a GPU join's output order is not deterministic. Returns the calls it made.
async fn assert_walk_matches_datafusion(sql: &str, knobs: PlanKnobs) -> Vec<(Seq, FbKind)> {
    let walked = walk(sql, knobs).await;
    // An exact compare of two empty results holds having compared nothing, so a query whose
    // predicate selected none would prove only that the walk did not crash.
    assert!(
        total_rows(&walked.batches) > 0,
        "the walk exported no rows for {sql}"
    );
    let expected = context(1)
        .await
        .sql(sql)
        .await
        .expect("the oracle plans it")
        .collect()
        .await
        .expect("the oracle runs it");
    assert_results_match(
        &expected,
        &walked.batches,
        CpuOracle::DataFusionExact.rel_tol(),
        &format!(
            "{sql}\n  the walk made: {}\n  and its",
            trail(&walked.calls)
        ),
    );
    walked.calls
}

fn times(calls: &[(Seq, FbKind)], kind: FbKind) -> usize {
    calls.iter().filter(|(_, made)| *made == kind).count()
}

/// Finalize projects whose call directly before them is `kind`. Which aggregate a finalize
/// belongs to is not in the counts — both branches emit one — so the adjacency is what
/// separates an init that finalized itself from a merge that did.
fn finalizing(calls: &[(Seq, FbKind)], kind: FbKind) -> usize {
    calls
        .windows(2)
        .filter(|pair| pair[0].1 == kind && pair[1].1 == FbKind::Project(ProjectRole::Finalize))
        .count()
}

/// The queries, named so the coverage read at the end of the file is over the very set the
/// tests above run rather than a second list that could drift from it.
const BARE_SCAN: &str = "SELECT * FROM nation";
const FILTER: &str = "SELECT o_orderkey, o_totalprice FROM orders WHERE o_totalprice > 500000";
const PROJECT_OVER_FILTER: &str =
    "SELECT c_custkey * 2 AS doubled FROM customer WHERE c_nationkey = 3";
const INNER_JOIN: &str =
    "SELECT n_name, r_name FROM nation JOIN region ON n_regionkey = r_regionkey";
const SEMI_JOIN: &str = "SELECT c_custkey FROM customer c WHERE c_nationkey = 3 AND EXISTS \
     (SELECT 1 FROM orders o WHERE o.o_custkey = c.c_custkey AND o.o_totalprice > c.c_acctbal)";
const SUM_BY_FLAG: &str =
    "SELECT l_returnflag, sum(l_quantity) FROM lineitem GROUP BY l_returnflag";
const AVG_BY_FLAG: &str =
    "SELECT l_returnflag, avg(l_quantity) FROM lineitem GROUP BY l_returnflag";
const MAX_OF_SUMS: &str = "SELECT max(per_flag) FROM \
     (SELECT l_returnflag, sum(l_quantity) AS per_flag FROM lineitem GROUP BY l_returnflag)";
const ROLLUP: &str = "SELECT l_returnflag, l_linestatus, sum(l_quantity) FROM lineitem \
     GROUP BY ROLLUP(l_returnflag, l_linestatus)";

#[tokio::test]
async fn a_bare_scan_comes_back_as_the_table() {
    assert_walk_matches_datafusion(BARE_SCAN, ONE_LANE).await;
}

#[tokio::test]
async fn a_filter_keeps_the_rows_its_predicate_names() {
    assert_walk_matches_datafusion(FILTER, ONE_LANE).await;
}

#[tokio::test]
async fn a_project_over_a_filter_evaluates_on_what_the_filter_left() {
    assert_walk_matches_datafusion(PROJECT_OVER_FILTER, ONE_LANE).await;
}

#[tokio::test]
async fn an_inner_join_matches_the_oracle_as_a_multiset() {
    assert_walk_matches_datafusion(INNER_JOIN, ONE_LANE).await;
}

/// A filtered semi join is the build-preserving type whose single probe batch takes the
/// legacy one-call form: the capability matrix makes the probe single-batch, so the one
/// call hands the build side over rather than needing a copy of it.
#[tokio::test]
async fn a_semi_join_that_keeps_its_build_side_runs_as_one_call() {
    let calls = assert_walk_matches_datafusion(SEMI_JOIN, ONE_LANE).await;
    assert_eq!(
        times(
            &calls,
            FbKind::HashJoin {
                join_type: JoinType::LeftSemi
            }
        ),
        1,
        "the join ran as something other than one LeftSemi call: {}",
        trail(&calls)
    );
}

/// Two lanes each merge their own state, the cross-lane merge folds them, and the finalize
/// runs once per lane at done — the first time `AggregateMode::Merge` meets a device.
#[tokio::test]
async fn each_lane_merges_its_own_state_before_the_cross_lane_merge_folds_them() {
    let calls = assert_walk_matches_datafusion(SUM_BY_FLAG, TWO_LANES).await;
    assert_eq!(
        times(&calls, FbKind::Aggregate { merge: true }),
        4,
        "two lanes below the shuffle and two above it merge state: {}",
        trail(&calls)
    );
    assert_eq!(
        times(&calls, FbKind::Project(ProjectRole::Finalize)),
        2,
        "the finalize runs once per lane, at done: {}",
        trail(&calls)
    );
}

/// `avg` finalizes by dividing a decimal by a count, and cuDF takes a divide's result
/// scale from its operands where arrow takes it from the declared output type — so a wrong
/// cast is invisible on a CPU host and wrong here, in a column whose type reads correctly
/// either way. The compare is on the rendered digits.
#[tokio::test]
async fn an_average_finalizes_to_the_digits_the_oracle_computes() {
    let calls = assert_walk_matches_datafusion(AVG_BY_FLAG, TWO_LANES).await;
    assert_eq!(
        times(&calls, FbKind::Project(ProjectRole::Finalize)),
        2,
        "the divide never reached the device: {}",
        trail(&calls)
    );
}

/// Two aggregates, one per branch of the rule. The inner one groups the whole table and
/// its lane holds many batches, so it takes the merge route and the merge carries the
/// finalize. The outer `max` reads that merge's one output batch, which is already the
/// whole of its single group, so the translation hands the finalize to the init itself and
/// no merge is built for it — the shape the goldens hold 39 of and no device had run.
///
/// The adjacency is the claim, not the counts: both branches emit one init and one
/// finalize, so a plan that lost the self-finalizing arm would still show two of each.
#[tokio::test]
async fn an_aggregate_whose_input_is_one_batch_finalizes_without_a_merge() {
    let calls = assert_walk_matches_datafusion(MAX_OF_SUMS, ONE_LANE).await;
    assert_eq!(
        finalizing(&calls, FbKind::Aggregate { merge: false }),
        1,
        "no init finalized itself, so the single-batch arm never ran: {}",
        trail(&calls)
    );
    assert_eq!(
        finalizing(&calls, FbKind::Aggregate { merge: true }),
        1,
        "the inner aggregate's merge stopped carrying its finalize: {}",
        trail(&calls)
    );
}

/// A ROLLUP is the one shape whose payload is not derivable from the plan line: the masks
/// and the per-position NULL placeholders decide how many groups come back. Dropping them
/// made this query refuse on the device rather than answer wrongly, because the state then
/// carried no `__grouping_id` and the merge read a state column as a key — a width
/// coincidence of this shape, not a property of the class. Where the widths line up, the
/// same omission answers with one grouping set out of three and looks right.
#[tokio::test]
async fn a_rollup_answers_with_every_grouping_set() {
    let calls = assert_walk_matches_datafusion(ROLLUP, ONE_LANE).await;
    assert_eq!(
        times(&calls, FbKind::Aggregate { merge: false }),
        1,
        "the sets are expanded once, by the init: {}",
        trail(&calls)
    );
}

/// Which fb kinds a device has now run, and which this walk refuses — the set T15 and T16
/// inherit as already proven, and the one they inherit as still open.
#[derive(Debug, PartialEq, Eq)]
enum Driven {
    Handled,
    /// Why the walk cannot make this call, in the words its panic uses.
    Refused(&'static str),
}

/// A match rather than a list: a kind added to the mapping stops this compiling, where a
/// list would go quietly short. The two facts a reader wants are here and nowhere else —
/// the spec's prose says the same thing and cannot go red.
fn driven(kind: FbKind) -> Driven {
    match kind {
        FbKind::Scan
        | FbKind::Filter
        | FbKind::PlainProject
        | FbKind::CoalescePartitions
        | FbKind::Repartition { .. }
        | FbKind::Aggregate { .. }
        | FbKind::Project(ProjectRole::Finalize) => Driven::Handled,
        FbKind::HashJoin { join_type } => match join_type {
            JoinType::Inner | JoinType::LeftSemi => Driven::Handled,
            _ => Driven::Refused("a join type no shape here plans"),
        },
        FbKind::Project(ProjectRole::ProbeKeys)
        | FbKind::Project(ProjectRole::NullPad { .. })
        | FbKind::Project(ProjectRole::Narrow) => {
            Driven::Refused("the finish pass accumulates probe keys across batches (#136)")
        }
        FbKind::Sort | FbKind::SortPreservingMerge => Driven::Refused("no shape here plans a sort"),
        FbKind::CrossJoin | FbKind::NestedLoopJoin => {
            Driven::Refused("both copy their build side, and the ABI has no copy (#152)")
        }
    }
}

/// The kinds the queries above put on a device. Checked against the walk both ways: a kind
/// here that no query produces is a claim on paper, and a kind produced that is not here is
/// an arm that quietly gained a shape.
const PROVEN: [FbKind; 10] = [
    FbKind::Scan,
    FbKind::Filter,
    FbKind::PlainProject,
    FbKind::CoalescePartitions,
    FbKind::Repartition { lanes: 2 },
    FbKind::Aggregate { merge: false },
    FbKind::Aggregate { merge: true },
    FbKind::Project(ProjectRole::Finalize),
    FbKind::HashJoin {
        join_type: JoinType::Inner,
    },
    FbKind::HashJoin {
        join_type: JoinType::LeftSemi,
    },
];

#[tokio::test]
async fn the_kinds_a_device_has_run_are_the_kinds_this_file_claims() {
    let mut made: Vec<FbKind> = Vec::new();
    for (sql, knobs) in [
        (BARE_SCAN, ONE_LANE),
        (FILTER, ONE_LANE),
        (PROJECT_OVER_FILTER, ONE_LANE),
        (INNER_JOIN, ONE_LANE),
        (SEMI_JOIN, ONE_LANE),
        (MAX_OF_SUMS, ONE_LANE),
        (ROLLUP, ONE_LANE),
        (SUM_BY_FLAG, TWO_LANES),
        (AVG_BY_FLAG, TWO_LANES),
    ] {
        for (_, kind) in walk(sql, knobs).await.calls {
            if !made.contains(&kind) {
                made.push(kind);
            }
        }
    }
    for kind in &made {
        assert_eq!(
            driven(*kind),
            Driven::Handled,
            "{kind} reached a device and is classified as refused"
        );
        assert!(
            PROVEN.contains(kind),
            "{kind} reached a device and is not in PROVEN"
        );
    }
    for kind in PROVEN {
        assert!(
            made.contains(&kind),
            "PROVEN claims {kind} and no query here produces it"
        );
    }
}
