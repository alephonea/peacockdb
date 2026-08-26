//! The batch-partitioned executors on a live GPU: each one handed its node's recipe and
//! the batches a lane would give it.
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

use datafusion::parquet::file::properties::WriterProperties;
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

use peacockdb_core::batch_partitioned::recipe::{AbiSymbol, Recipe, RecipePlan, attach_recipes};

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
    // Two rows per row group, so a lane can be read as one batch or as three.
    let properties = WriterProperties::builder().set_max_row_group_size(2).build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(columns()), Some(properties))
        .expect("the writer opens");
    writer.write(&batch).expect("the rows are written");
    writer.close().expect("the footer is written");
    path
}

/// A one-lane scan of that table, reading every row group in one batch.
fn source() -> Box<dyn GpuNode> {
    mapped(vec![vec![ROW_GROUPS.to_vec()]])
}

/// The same table as one lane of three batches, which is what an accumulator needs to
/// have something to accumulate.
fn source_per_row_group() -> Box<dyn GpuNode> {
    mapped(vec![ROW_GROUPS.iter().map(|group| vec![*group]).collect()])
}

/// The table as two lanes: one row group in the first, two in the second. A partition
/// accumulator needs several lanes to have anything to merge.
fn source_two_lanes() -> Box<dyn GpuNode> {
    mapped(vec![vec![vec![0]], vec![vec![1], vec![2]]])
}

const ROW_GROUPS: [u32; 3] = [0, 1, 2];

fn mapped(partition_groups: Vec<Vec<Vec<u32>>>) -> Box<dyn GpuNode> {
    let path = table();
    let file = std::fs::File::open(&path).expect("the file just written");
    let reader = SerializedFileReader::new(file).expect("a parquet file");
    let groups = (0..reader.metadata().num_row_groups())
        .map(|index| {
            let group = reader.metadata().row_group(index);
            RowGroupMeta {
                index: index as u32,
                rows: group.num_rows() as u64,
                bytes: group.total_byte_size() as u64,
            }
        })
        .collect();
    let scan = ScanMetadata {
        file: path.to_string_lossy().into_owned(),
        groups,
        can_be_null: vec![false, false],
    };
    Box::new(GpuLoadParquet::new(
        "t".to_string(),
        vec![0, 1],
        partition_groups,
        &scan,
        None,
        Schema::new(Arc::new(columns())),
    ))
}

/// An executor with the recipe plan loaded, torn down in the order the header requires.
struct Session {
    /// Borrowed by every executor drawn from it, and outliving all of them.
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
    fn scan(&self, groups: &[u32]) -> GpuBatch {
        let recipe = self.recipes.get(0).expect("the scan is the first node");
        let [call] = recipe.calls.as_slice() else {
            panic!("a scan's recipe is one call per batch")
        };
        assert_eq!(call.symbol, AbiSymbol::ExecuteScanRowGroups);
        let (seq, _) = call.target.expect("a scan addresses its own node");
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

    /// The accumulator for the node at `index`, built from the recipe that node published.
    fn recipe(&self, index: usize) -> &Recipe {
        self.recipes.get(index).expect("the node makes ABI calls")
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
    let batch = session.scan(&ROW_GROUPS);
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

// A test target's child modules resolve against tests/ itself, and a file there would be
// another target. The path keeps them under a directory named for this one.
#[path = "test_gpu_executors/accumulate.rs"]
mod accumulate;
#[path = "test_gpu_executors/exec.rs"]
mod exec;
