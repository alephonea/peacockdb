pub mod gpu_rule;
pub mod cpu_executor;
pub mod gpu_rowgroup_prune;
#[cfg(not(feature = "rust-only"))]
pub mod gpu_executor;
#[allow(unused_imports, dead_code, clippy::all)]
pub mod generated {
    pub mod gpu_plan_generated {
        include!(concat!(env!("OUT_DIR"), "/gpu_plan_generated.rs"));
    }
}
pub mod plan_serializer;
pub mod resident;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl};
use datafusion::execution::context::SessionContext;
use datafusion::execution::SessionStateBuilder;
use datafusion::error::Result;

use cpu_executor::{execute_node_by_node, NodeMemoryStats};
use gpu_rule::{GpuExecutionRule, GpuMemoryBudgetRule};

pub fn build_session_state_with_gpu_rules(
    target_partitions: usize,
    gpu_memory_budget: usize
) -> SessionContext {
    let base = SessionContext::new();
    let mut config = base.state().config().clone();
    config.options_mut().execution.target_partitions = target_partitions;
    let state = SessionStateBuilder::new_from_existing(base.state())
        .with_config(config)
        .with_physical_optimizer_rule(Arc::new(GpuExecutionRule))
        .with_physical_optimizer_rule(Arc::new(GpuMemoryBudgetRule::new(gpu_memory_budget)))
        .build();
    
    SessionContext::new_with_state(state)
}

pub fn build_session_state(
    target_partitions: usize
) -> SessionContext {
    let base = SessionContext::new();
    let mut config = base.state().config().clone();
    config.options_mut().execution.target_partitions = target_partitions;
    let state = SessionStateBuilder::new_from_existing(base.state())
        .with_config(config)
        .build();
    
    SessionContext::new_with_state(state)
}

async fn read_table(path: PathBuf, ctx: &SessionContext) -> Result<(String, Arc<ListingTable>), ()> {
    if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
        ()
    }

    let table_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| datafusion::error::DataFusionError::Plan(
            format!("could not derive table name from path: {}", path.display()),
        )).unwrap()
        .to_string();

    let table_url = ListingTableUrl::parse(path.to_str().unwrap()).unwrap();
    let format = Arc::new(ParquetFormat::default().with_enable_pruning(true));
    let listing_options = ListingOptions::new(format).with_file_extension(".parquet");

    let resolved_schema = listing_options.infer_schema(&ctx.state(), &table_url).await.unwrap();

    let config = ListingTableConfig::new(table_url)
        .with_listing_options(listing_options)
        .with_schema(resolved_schema);

    let table = Arc::new(ListingTable::try_new(config).unwrap());

    Ok((table_name, table))
}

pub async fn register_tables_for(
    ctx: SessionContext,
    data_dir: &Path
) -> Result<SessionContext> {
    for entry in std::fs::read_dir(data_dir)? {
        let path = entry?.path();
        let Ok((table_name, table)) = read_table(path, &ctx).await else { continue; }; 
        ctx.register_table(&table_name, table)?;
    }

    Ok(ctx)
}

pub async fn create_context_with_tables(
    data_dir: &Path,
    target_partitions: usize,
    gpu_memory_budget: usize,
) -> Result<SessionContext> {
    let ctx = build_session_state_with_gpu_rules(target_partitions, gpu_memory_budget);
    register_tables_for(ctx, data_dir).await
}

// ---------------------------------------------------------------------------
// CpuExecutor
// ---------------------------------------------------------------------------

/// Executes SQL queries on CPU by building a GPU-annotated physical plan
/// and running it through [`execute_node_by_node`].
///
/// This is the idiomatic entry point: callers only see SQL in and
/// `Vec<RecordBatch>` out — the GPU plan construction, node stripping, and
/// `TaskContext` wiring are all hidden inside.
///
/// ```
/// # use std::path::Path;
/// # use peacockdb_core::CpuExecutor;
/// # async fn example() -> datafusion::error::Result<()> {
/// let exec = CpuExecutor::new(Path::new("./data"), 8, 2 * 1024 * 1024 * 1024).await?;
/// let batches = exec.execute("SELECT count(*) FROM orders WHERE o_totalprice > 100").await?;
/// # Ok(())
/// # }
/// ```
pub struct CpuExecutor {
    ctx: SessionContext,
}

impl CpuExecutor {
    /// Build a `CpuExecutor` from a directory of `.parquet` files.
    ///
    /// Internally calls [`create_context_with_tables`] so the `SessionContext`
    /// already has `GpuExecutionRule` and `GpuMemoryBudgetRule` registered.
    /// The resulting physical plans are GPU-annotated but executed on CPU.
    pub async fn new(
        data_dir: &Path,
        target_partitions: usize,
        gpu_memory_budget: usize,
    ) -> Result<Self> {
        let ctx = create_context_with_tables(data_dir, target_partitions, gpu_memory_budget).await?;
        Ok(Self { ctx })
    }

    /// Execute a SQL query and return all result batches.
    ///
    /// Steps (all hidden from the caller):
    /// 1. `ctx.sql(sql)` → DataFusion `DataFrame` (SQL parse + logical plan)
    /// 2. `.create_physical_plan()` → GPU-annotated `ExecutionPlan` tree
    /// 3. `execute_node_by_node` → strip GPU wrappers, run each CPU node bottom-up
    pub async fn execute(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let plan = self.ctx.sql(sql).await?.create_physical_plan().await?;
        execute_node_by_node(plan, self.ctx.task_ctx(), &mut |_, _| {}).await
    }

    /// Like [`execute`] but also returns the physical plan and per-node memory stats
    /// in post-order. The plan is the GPU-annotated plan (before CPU stripping) and
    /// its tree structure matches the stat ordering, enabling tree-shaped formatting.
    pub async fn execute_instrumented(
        &self,
        sql: &str,
    ) -> Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>, Vec<NodeMemoryStats>)> {
        let plan = self.ctx.sql(sql).await?.create_physical_plan().await?;
        let mut stats = Vec::new();
        let batches = execute_node_by_node(plan.clone(), self.ctx.task_ctx(), &mut |_, s| {
            stats.push(s.clone());
        })
        .await?;
        Ok((batches, plan, stats))
    }
}