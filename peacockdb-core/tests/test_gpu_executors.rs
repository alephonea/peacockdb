//! The exec executors on a live GPU: each one handed its node's recipe and one batch.
//!
//! Every plan here is hand-built over six synthetic rows the test writes itself, so a case
//! names the shape it means and its expected answer is written down rather than computed.
//! Parquet is the transport and not the subject: the ABI has no way to put a table on a
//! device except to read one, so the source is a scan the test drives by hand — as T21's
//! walk does, and for the same reason: a source executor is nobody's task yet.
#![cfg(not(feature = "rust-only"))]
#[macro_use]
mod common;

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::reader::{FileReader, SerializedFileReader};

use peacockdb_core::batch_partitioned::aggregates::{AggCall, PlanAgg};
use peacockdb_core::batch_partitioned::executor::RowRange;
use peacockdb_core::batch_partitioned::expr::{BinaryOp, Expr, NamedExpr};
use peacockdb_core::batch_partitioned::gpu_backend::{GpuExec, GpuExport};
use peacockdb_core::batch_partitioned::layout::ColumnOrder;
use peacockdb_core::batch_partitioned::node::GpuNode;
use peacockdb_core::batch_partitioned::nodes::aggregate::AggregateBody;
use peacockdb_core::batch_partitioned::nodes::{
    GpuAggregate, GpuFilter, GpuLoadParquet, GpuProject, GpuSort,
};
use peacockdb_core::batch_partitioned::parquet_meta::ScanMetadata;
use peacockdb_core::batch_partitioned::partitioner::RowGroupMeta;
use peacockdb_core::batch_partitioned::recipe::{AbiSymbol, RecipePlan, attach_recipes};
use peacockdb_core::batch_partitioned::schema::Schema;
use peacockdb_core::batch_partitioned::{CpuBatch, GpuBatch};
use peacockdb_ffi::raw::{
    PeacockExecutor, PeacockNodeStats, peacock_executor_begin_plan, peacock_executor_create,
    peacock_executor_destroy, peacock_executor_end_plan, peacock_executor_execute_scan_rowgroups,
    peacock_last_error,
};

use common::GPU_BUDGET;

/// Six rows in one row group, two groups of three under `k`. The values are chosen so that
/// every expected answer below is exact in a float: an average of 4 and an average of 3.
const KEYS: [&str; 6] = ["a", "b", "a", "b", "a", "b"];
const VALUES: [i64; 6] = [2, 1, 4, 3, 6, 5];

fn columns() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("k", DataType::Utf8, true),
        Field::new("v", DataType::Int64, true),
    ])
}

/// The synthetic table, written once per process into a path the C++ scan will open.
fn table() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "peacock-gpu-executors-{}.parquet",
        std::process::id()
    ));
    if path.exists() {
        return path;
    }
    let k: ArrayRef = Arc::new(StringArray::from(KEYS.to_vec()));
    let v: ArrayRef = Arc::new(Int64Array::from(VALUES.to_vec()));
    let batch = RecordBatch::try_new(Arc::new(columns()), vec![k, v]).expect("six rows");
    let file = std::fs::File::create(&path).expect("a writable temp dir");
    let mut writer =
        ArrowWriter::try_new(file, Arc::new(columns()), None).expect("the writer opens");
    writer.write(&batch).expect("the rows are written");
    writer.close().expect("the footer is written");
    path
}

/// A one-lane, one-batch scan of that table — the shape every case below starts from.
fn source() -> Box<dyn GpuNode> {
    let path = table();
    let file = std::fs::File::open(&path).expect("the file just written");
    let reader = SerializedFileReader::new(file).expect("a parquet file");
    let group = reader.metadata().row_group(0);
    let scan = ScanMetadata {
        file: path.to_string_lossy().into_owned(),
        groups: vec![RowGroupMeta {
            index: 0,
            rows: group.num_rows() as u64,
            bytes: group.total_byte_size() as u64,
        }],
        can_be_null: vec![false, false],
    };
    Box::new(GpuLoadParquet::new(
        "t".to_string(),
        vec![0, 1],
        vec![vec![vec![0]]],
        &scan,
        None,
        Schema::new(Arc::new(columns())),
    ))
}

/// An executor with the recipe plan loaded, torn down in the order the header requires.
struct Session {
    executor: *mut PeacockExecutor,
    recipes: RecipePlan,
}

impl Session {
    /// The tree's recipes attached and its buffer handed to the device, which is the whole
    /// of what an executor needs before it can address a seq.
    fn open(tree: &dyn GpuNode) -> Self {
        let recipes = attach_recipes(tree).expect("every node's payload is writable");
        let mut executor: *mut PeacockExecutor = std::ptr::null_mut();
        assert_eq!(
            unsafe { peacock_executor_create(GPU_BUDGET as u64, &mut executor) },
            0,
            "peacock_executor_create failed"
        );
        let bytes = recipes.bytes();
        let mut nodes = 0u64;
        let rc = unsafe {
            peacock_executor_begin_plan(executor, bytes.as_ptr(), bytes.len() as u64, &mut nodes)
        };
        assert_eq!(rc, 0, "begin_plan failed: {}", error_of(executor));
        assert_eq!(nodes as usize, recipes.wire_nodes());
        Self { executor, recipes }
    }

    /// The scan's one batch, by the call its own recipe names. The source is driven here
    /// rather than by an executor because nothing in this task owns one.
    fn scan(&self) -> GpuBatch {
        let recipe = self.recipes.get(0).expect("the scan is the first node");
        let [call] = recipe.calls.as_slice() else {
            panic!("a scan's recipe is one call per batch")
        };
        assert_eq!(call.symbol, AbiSymbol::ExecuteScanRowGroups);
        let (seq, _) = call.target.expect("a scan addresses its own node");
        let groups = [0u32];
        let mut handle = 0u64;
        let mut stats = PeacockNodeStats::default();
        let rc = unsafe {
            peacock_executor_execute_scan_rowgroups(
                self.executor,
                seq as u64,
                groups.as_ptr(),
                groups.len() as u64,
                &mut handle,
                &mut stats,
            )
        };
        assert_eq!(rc, 0, "the scan failed: {}", error_of(self.executor));
        GpuBatch::new(self.executor, handle, stats.rows as usize, 0)
    }

    /// The executor for the node at `index` in the tree's post-order, which is what the
    /// recipes are indexed by.
    fn exec(&self, index: usize, schema: &ArrowSchema) -> GpuExec {
        let recipe = self.recipes.get(index).expect("the node makes ABI calls");
        GpuExec::new(self.executor, recipe, schema).expect("the recipe is an exec node's")
    }

    fn export(&self, schema: &ArrowSchema) -> GpuExport {
        GpuExport::new(self.executor, schema)
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

fn error_of(executor: *mut PeacockExecutor) -> String {
    let message = unsafe { peacock_last_error(executor) };
    if message.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

/// What came back, one row at a time, so an expected answer reads as rows written down.
fn rows(batch: &CpuBatch) -> Vec<Vec<ScalarValue>> {
    let batch = batch.record_batch();
    (0..batch.num_rows())
        .map(|row| {
            (0..batch.num_columns())
                .map(|column| {
                    ScalarValue::try_from_array(batch.column(column), row)
                        .expect("a value at every position")
                })
                .collect()
        })
        .collect()
}

/// A grouped answer is in whatever order the hash table held it, so every claim about one
/// is a claim about which group got which value.
fn by_key(batch: &CpuBatch) -> Vec<Vec<ScalarValue>> {
    let mut rows = rows(batch);
    rows.sort_by_key(|row| format!("{:?}", row[0]));
    rows
}

fn string(value: &str) -> ScalarValue {
    ScalarValue::Utf8(Some(value.to_string()))
}

fn schema_of(columns: &[(&str, DataType)]) -> ArrowSchema {
    ArrowSchema::new(
        columns
            .iter()
            .map(|(name, kind)| Field::new(*name, kind.clone(), true))
            .collect::<Vec<Field>>(),
    )
}

/// Run one exec node over the scan's batch and bring the answer home.
fn one_node(tree: Box<dyn GpuNode>, out: &ArrowSchema) -> CpuBatch {
    let session = Session::open(tree.as_ref());
    let batch = session.scan();
    let (produced, _) = session.exec(1, out).exec(batch).expect("the node runs");
    let (result, _) = session
        .export(out)
        .unload(produced, RowRange::WHOLE)
        .expect("the rows cross the boundary");
    result
}

fn greater_than(bound: i64) -> Expr {
    Expr::binary(
        Expr::column(1, "v"),
        BinaryOp::Gt,
        Expr::Literal(ScalarValue::Int64(Some(bound))),
        DataType::Boolean,
    )
}

#[test]
fn a_filter_answers_with_the_rows_its_predicate_keeps() {
    let out = columns();
    let node = GpuFilter::new(
        source(),
        greater_than(3),
        None,
        Schema::new(Arc::new(out.clone())),
    );
    assert_eq!(
        rows(&one_node(Box::new(node), &out)),
        vec![
            vec![string("a"), ScalarValue::Int64(Some(4))],
            vec![string("a"), ScalarValue::Int64(Some(6))],
            vec![string("b"), ScalarValue::Int64(Some(5))],
        ],
        "the three rows above 3, in the order the scan read them"
    );
}

/// A filter that keeps nothing still answers with a batch, because the node above counts
/// calls and not rows — and on this side that batch is a resident table of zero rows.
#[test]
fn a_filter_that_keeps_nothing_still_answers_with_a_batch() {
    let out = columns();
    let node = GpuFilter::new(
        source(),
        greater_than(100),
        None,
        Schema::new(Arc::new(out.clone())),
    );
    let answer = one_node(Box::new(node), &out);
    assert_eq!(answer.record_batch().num_rows(), 0);
    assert_eq!(
        answer.record_batch().schema().fields().len(),
        2,
        "an empty answer still has the node's columns"
    );
}

#[test]
fn a_project_evaluates_its_expressions_under_the_names_it_declares() {
    let out = schema_of(&[("twice", DataType::Int64)]);
    let node = GpuProject::new(
        source(),
        vec![NamedExpr::new(
            Expr::binary(
                Expr::column(1, "v"),
                BinaryOp::Multiply,
                Expr::Literal(ScalarValue::Int64(Some(2))),
                DataType::Int64,
            ),
            "twice",
        )],
        Schema::new(Arc::new(out.clone())),
    );
    let answer = one_node(Box::new(node), &out);
    assert_eq!(
        answer.record_batch().schema().field(0).name(),
        "twice",
        "the output name is the project's, not the input's"
    );
    assert_eq!(
        rows(&answer)
            .into_iter()
            .map(|row| row[0].clone())
            .collect::<Vec<ScalarValue>>(),
        VALUES
            .iter()
            .map(|v| ScalarValue::Int64(Some(v * 2)))
            .collect::<Vec<ScalarValue>>()
    );
}

#[test]
fn a_sort_orders_the_batch_it_was_given() {
    let out = columns();
    let node = GpuSort::new(
        source(),
        vec![ColumnOrder {
            column: 1,
            ascending: true,
            nulls_first: false,
        }],
        None,
    );
    assert_eq!(
        rows(&one_node(Box::new(node), &out))
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        (1..=6)
            .map(|v| ScalarValue::Int64(Some(v)))
            .collect::<Vec<ScalarValue>>()
    );
}

/// The per-batch sort's `fetch` is a top-N inside the batch it was handed. Ordering a whole
/// stream is `GpuAccumulateBatchesAndSort`, a different node.
#[test]
fn a_sort_with_a_fetch_keeps_the_top_of_its_own_batch() {
    let out = columns();
    let node = GpuSort::new(
        source(),
        vec![ColumnOrder {
            column: 1,
            ascending: false,
            nulls_first: false,
        }],
        Some(2),
    );
    assert_eq!(
        rows(&one_node(Box::new(node), &out))
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        vec![ScalarValue::Int64(Some(6)), ScalarValue::Int64(Some(5))],
        "the two largest, in order"
    );
}

fn summing(output: &str) -> AggregateBody {
    AggregateBody {
        group_by: vec![Expr::column(0, "k")],
        grouping_sets: Vec::new(),
        null_exprs: Vec::new(),
        aggs: vec![AggCall {
            func: PlanAgg::Sum,
            args: vec![Expr::column(1, "v")],
            outputs: vec![Field::new(output, DataType::Int64, true)],
        }],
        finalize: None,
    }
}

#[test]
fn a_partial_aggregate_emits_the_state_its_node_declared() {
    let out = schema_of(&[("k", DataType::Utf8), ("sum(v)", DataType::Int64)]);
    let state = Schema::new(Arc::new(out.clone()));
    let node = GpuAggregate::new(source(), summing("sum(v)"), state.clone(), state);
    assert_eq!(
        by_key(&one_node(Box::new(node), &out)),
        vec![
            vec![string("a"), ScalarValue::Int64(Some(12))],
            vec![string("b"), ScalarValue::Int64(Some(9))],
        ],
        "2 + 4 + 6 under a, 1 + 3 + 5 under b"
    );
}

/// The single-node shortcut: state and finalize on one node, which is two calls on one
/// executor. The finalize is this mode's own expression — the same one the CPU backend
/// hands DataFusion — rather than an aggregate mode that also finalizes.
#[test]
fn an_aggregate_that_finalizes_runs_both_of_its_calls() {
    let state_columns = schema_of(&[
        ("k", DataType::Utf8),
        ("avg(v)$sum", DataType::Int64),
        ("avg(v)$count", DataType::Int64),
    ]);
    let out = schema_of(&[("k", DataType::Utf8), ("avg(v)", DataType::Float64)]);
    let average = Expr::binary(
        Expr::Cast {
            expr: Box::new(Expr::column(1, "avg(v)$sum")),
            target: DataType::Float64,
        },
        BinaryOp::Divide,
        Expr::Cast {
            expr: Box::new(Expr::column(2, "avg(v)$count")),
            target: DataType::Float64,
        },
        DataType::Float64,
    );
    let value = |output: &str, func| AggCall {
        func,
        args: vec![Expr::column(1, "v")],
        outputs: vec![Field::new(output, DataType::Int64, true)],
    };
    let node = GpuAggregate::new(
        source(),
        AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            grouping_sets: Vec::new(),
            null_exprs: Vec::new(),
            aggs: vec![
                value("avg(v)$sum", PlanAgg::Sum),
                value("avg(v)$count", PlanAgg::Count),
            ],
            finalize: Some(vec![NamedExpr::new(average, "avg(v)")]),
        },
        Schema::new(Arc::new(state_columns)),
        Schema::new(Arc::new(out.clone())),
    );
    assert_eq!(
        by_key(&one_node(Box::new(node), &out)),
        vec![
            vec![string("a"), ScalarValue::Float64(Some(4.0))],
            vec![string("b"), ScalarValue::Float64(Some(3.0))],
        ],
        "12/3 and 9/3, and the state columns are gone from the row"
    );
}

/// The whole handle, which is what every case above asks for after its node has run. Named
/// on its own so that an export which trimmed says so, rather than showing up as whichever
/// operator the failing test was about.
#[test]
fn an_export_of_the_whole_handle_brings_back_every_row() {
    let session = Session::open(source().as_ref());
    let batch = session.scan();
    let (answer, _) = session
        .export(&columns())
        .unload(batch, RowRange::WHOLE)
        .expect("the rows cross the boundary");
    assert_eq!(
        rows(&answer)
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        VALUES
            .iter()
            .map(|v| ScalarValue::Int64(Some(*v)))
            .collect::<Vec<ScalarValue>>(),
        "six rows, in the order the scan read them"
    );
}

/// The range is what a straddling batch costs: the rows wanted cross the boundary rather
/// than the batch they sit in.
#[test]
fn an_export_with_a_range_answers_with_the_rows_it_names() {
    let session = Session::open(source().as_ref());
    let batch = session.scan();
    let (answer, _) = session
        .export(&columns())
        .unload(
            batch,
            RowRange {
                offset: 1,
                length: 2,
            },
        )
        .expect("the rows cross the boundary");
    assert_eq!(
        rows(&answer)
            .into_iter()
            .map(|row| row[1].clone())
            .collect::<Vec<ScalarValue>>(),
        vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(4))]
    );
}

/// A limit's fetch legitimately overruns the batch it straddles, so a range past the end
/// clamps rather than failing.
#[test]
fn an_export_whose_range_runs_past_the_end_stops_at_the_end() {
    let session = Session::open(source().as_ref());
    let batch = session.scan();
    let (answer, _) = session
        .export(&columns())
        .unload(
            batch,
            RowRange {
                offset: 4,
                length: 100,
            },
        )
        .expect("the rows cross the boundary");
    assert_eq!(answer.record_batch().num_rows(), 2);
}

/// A range naming no rows of a table that has them exports nothing at all, and the answer
/// is still a batch of the sink's columns rather than a missing one.
#[test]
fn an_export_whose_offset_is_past_the_end_answers_empty() {
    let session = Session::open(source().as_ref());
    let batch = session.scan();
    let (answer, _) = session
        .export(&columns())
        .unload(
            batch,
            RowRange {
                offset: 99,
                length: 1,
            },
        )
        .expect("the export succeeds");
    assert_eq!(answer.record_batch().num_rows(), 0);
    assert_eq!(answer.record_batch().schema().fields().len(), 2);
}

/// An exec executor drives a straight line of per-batch calls. A recipe that waits for
/// done belongs to an accumulator, and building one here would call it once per batch —
/// a wrong answer rather than an error, since every call succeeds.
#[test]
fn a_recipe_whose_calls_wait_for_done_is_refused_by_an_exec_executor() {
    let keys = vec![ColumnOrder {
        column: 1,
        ascending: true,
        nulls_first: false,
    }];
    let tree: Box<dyn GpuNode> = Box::new(
        peacockdb_core::batch_partitioned::nodes::GpuAccumulateBatchesAndSort::new(
            source(),
            keys,
            None,
        ),
    );
    let recipes = attach_recipes(tree.as_ref()).expect("the payloads are writable");
    let refused = match GpuExec::new(
        std::ptr::null_mut(),
        recipes.get(1).expect("the accumulator makes ABI calls"),
        &columns(),
    ) {
        Err(refused) => refused,
        Ok(_) => panic!("an accumulator's recipe is not an exec node's"),
    };
    let message = format!("{refused}");
    assert!(
        message.contains("AtDone"),
        "the refusal has to name the pattern it found: {message}"
    );
}
