//! The three-table parquet fixture the join planner tests share, and the knobs they plan
//! with.
//!
//! Written rather than checked in: the null analysis reads parquet statistics, so a column
//! is nullable here because a row group holds a NULL and never because it was declared so,
//! and `tpch.minimal` holds none. Two test targets use it — what plans, and what is
//! refused — and the row counts below decide both.

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::JoinType;
use datafusion::execution::context::SessionContext;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
use datafusion::prelude::ParquetReadOptions;

use peacockdb_core::batch_partitioned::error::PlanError;
use peacockdb_core::batch_partitioned::node::GpuNode;
use peacockdb_core::batch_partitioned::plan::{BatchSizing, PlanKnobs, plan_batch_partitioned};

/// Lanes for the co-partitioned cases.
pub const LANES: usize = 4;

/// Between `tiny` and `big` below, so the small-source rule is reachable without writing a
/// five-megabyte fixture.
pub const SMALL_SOURCE_BYTES: u64 = 4 * 1024;

pub fn knobs(sizing: BatchSizing) -> PlanKnobs {
    PlanKnobs {
        target_partitions: LANES,
        sizing,
        budget: 2 * 1024 * 1024 * 1024,
        small_table_bytes: SMALL_SOURCE_BYTES,
    }
}

/// Three tables. `tiny` and `nulls` differ only in what the null analysis reads off them —
/// ten rows each, one with NULL keys — and `big` is a thousand rows.
///
/// The ROW COUNTS are load-bearing twice over. DataFusion's join-order swap reads the
/// parquet footer's own statistics and puts the smaller side on the build, which is what
/// turns a LeftSemi into a RightSemi and a LeftAnti into a RightAnti; it is a comparison
/// and not a threshold, so a thousand against ten is as decisive as a million against a
/// hundred and both files stay tiny. And `big` sits above the small-source byte threshold
/// while the other two sit below it. Changing either count silently changes which join
/// types the SQL below plans as.
pub struct Fixture {
    dir: PathBuf,
    ctx: SessionContext,
}

impl Fixture {
    pub async fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "peacockdb-join-fixture-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("a fixture directory");

        write(&dir, "tiny", &(1..=10).map(Some).collect::<Vec<_>>(), 0);
        write(
            &dir,
            "nulls",
            &[
                Some(1),
                None,
                Some(3),
                None,
                Some(5),
                Some(6),
                Some(7),
                Some(8),
                None,
                Some(10),
            ],
            0,
        );
        // Padded so it also sits above the byte threshold; the rows are never read.
        write(&dir, "big", &(1..=1000).map(Some).collect::<Vec<_>>(), 64);

        let ctx =
            SessionContext::new_with_state(peacockdb_core::build_session_state(LANES).state());
        for table in ["tiny", "nulls", "big"] {
            ctx.register_parquet(
                table,
                dir.join(format!("{table}.parquet")).to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await
            .expect("register the fixture");
        }
        Self { dir, ctx }
    }

    /// A whole query, planned by DataFusion — which is what decides the join type.
    pub async fn plan(&self, sql: &str) -> Arc<dyn ExecutionPlan> {
        self.ctx
            .sql(sql)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .create_physical_plan()
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    /// The planner's refusal for a query, which must be a PlanError rather than a panic.
    pub async fn refused(&self, sql: &str) -> PlanError {
        let plan = self.plan(sql).await;
        plan_batch_partitioned(&plan, knobs(BatchSizing::OneBatchPerRowGroup))
            .map(|_| ())
            .expect_err(sql)
    }

    pub async fn scan(&self, table: &str) -> Arc<dyn ExecutionPlan> {
        self.ctx
            .sql(&format!("SELECT k, v FROM {table}"))
            .await
            .expect("plan the scan")
            .create_physical_plan()
            .await
            .expect("physical plan")
    }

    /// The same scan, hash-partitioned on its key — what a shuffle would have left, and
    /// what a co-partitioned join needs on both sides.
    pub async fn scattered(&self, table: &str) -> Arc<dyn ExecutionPlan> {
        let scan = self.scan(table).await;
        Arc::new(
            RepartitionExec::try_new(
                scan,
                Partitioning::Hash(vec![Arc::new(Column::new("k", 0))], LANES),
            )
            .expect("a hash repartition"),
        )
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// One parquet file: a key column carrying exactly the NULLs asked for, a value, and
/// `padding` bytes per row to move the file across the size threshold.
fn write(dir: &std::path::Path, name: &str, keys: &[Option<i64>], padding: usize) {
    let mut fields = vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
    ];
    if padding > 0 {
        fields.push(Field::new("pad", DataType::Utf8, true));
    }
    let schema = Arc::new(ArrowSchema::new(fields));
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(keys.iter().copied().collect::<Int64Array>()),
        Arc::new((0..keys.len() as i64).map(Some).collect::<Int64Array>()),
    ];
    if padding > 0 {
        let filler = "x".repeat(padding);
        columns.push(Arc::new(
            keys.iter()
                .map(|_| Some(filler.as_str()))
                .collect::<StringArray>(),
        ));
    }
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("a batch");
    let file = std::fs::File::create(dir.join(format!("{name}.parquet"))).expect("create");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("a writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
}

pub fn planned(plan: &Arc<dyn ExecutionPlan>) -> Result<Box<dyn GpuNode>, PlanError> {
    plan_batch_partitioned(plan, knobs(BatchSizing::OneBatchPerRowGroup)).map(|(tree, _)| tree)
}

/// Every hash join in a plan, in tree order.
pub fn join_types_in(plan: &Arc<dyn ExecutionPlan>) -> Vec<JoinType> {
    let mut found = Vec::new();
    fn walk(plan: &Arc<dyn ExecutionPlan>, found: &mut Vec<JoinType>) {
        if let Some(join) = plan.as_any().downcast_ref::<HashJoinExec>() {
            found.push(*join.join_type());
        }
        for child in plan.children() {
            walk(&child.clone(), found);
        }
    }
    walk(plan, &mut found);
    found
}
