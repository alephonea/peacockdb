mod gpu_rule;

use std::path::Path;
use std::sync::Arc;

use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl};
use datafusion::execution::context::SessionContext;
use datafusion::execution::SessionStateBuilder;
use datafusion::error::Result;

use gpu_rule::{GpuExecutionRule, GpuMemoryBudgetRule};

const DEFAULT_GPU_MEMORY_BUDGET: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

/// Scans `data_dir` for `.parquet` files and registers each as a table in a new
/// `SessionContext`. The table name is the file stem (e.g. `orders.parquet` → `orders`).
pub async fn create_context_with_tables(data_dir: &Path) -> Result<SessionContext> {
    create_context_with_tables_and_budget(data_dir, DEFAULT_GPU_MEMORY_BUDGET).await
}

pub async fn create_context_with_tables_and_budget(
    data_dir: &Path,
    gpu_memory_budget: usize,
) -> Result<SessionContext> {
    let state = SessionStateBuilder::new_from_existing(SessionContext::new().state())
        .with_physical_optimizer_rule(Arc::new(GpuExecutionRule))
        .with_physical_optimizer_rule(Arc::new(GpuMemoryBudgetRule::new(gpu_memory_budget)))
        .build();
    let ctx = SessionContext::new_with_state(state);

    let entries = std::fs::read_dir(data_dir).map_err(|e| {
        datafusion::error::DataFusionError::IoError(e)
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| datafusion::error::DataFusionError::IoError(e))?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
            continue;
        }

        let table_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| datafusion::error::DataFusionError::Plan(
                format!("could not derive table name from path: {}", path.display()),
            ))?
            .to_string();

        let table_url = ListingTableUrl::parse(path.to_str().unwrap())?;
        let format = Arc::new(ParquetFormat::default().with_enable_pruning(true));
        let listing_options = ListingOptions::new(format).with_file_extension(".parquet");

        let resolved_schema = listing_options.infer_schema(&ctx.state(), &table_url).await?;

        let config = ListingTableConfig::new(table_url)
            .with_listing_options(listing_options)
            .with_schema(resolved_schema);

        let table = Arc::new(ListingTable::try_new(config)?);
        ctx.register_table(&table_name, table)?;
    }

    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Int64Array;
    use datafusion::physical_plan::ExecutionPlan;
    use gpu_rule::{analyze_memory, row_width, GpuScanExec};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn testdata_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../testdata/tpchsf1")
    }

    async fn count(ctx: &SessionContext, query: &str) -> i64 {
        let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }

    use datafusion::physical_plan::display::DisplayableExecutionPlan;

    fn plans_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/plans")
    }

    /// Render the plan to a string, normalizing `ParquetExec:` lines to
    /// `ParquetExec: table=<stem>` so canonical files are path-independent.
    fn plan_str(plan: &Arc<dyn ExecutionPlan>) -> String {
        let raw = DisplayableExecutionPlan::new(plan.as_ref())
            .indent(false)
            .to_string();
        raw.lines()
            .filter(|l| !l.is_empty())
            .map(|line| {
                if line.trim_start().starts_with("ParquetExec:") {
                    let indent = line.len() - line.trim_start().len();
                    let table = line.find(".parquet")
                        .and_then(|end| line[..end].rfind('/').map(|sep| &line[sep + 1..end]))
                        .unwrap_or("unknown");
                    format!("{}ParquetExec: table={table}", &line[..indent])
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Render per-node memory analysis: row_width and subtree_max_row_bytes.
    fn memory_str(plan: &Arc<dyn ExecutionPlan>) -> String {
        fn walk(plan: &Arc<dyn ExecutionPlan>, indent: usize, lines: &mut Vec<String>) {
            let mem = analyze_memory(plan);
            let rw = row_width(&plan.schema());
            let prefix = " ".repeat(indent);
            lines.push(format!(
                "{}{}: row_width={}, subtree_max_row_bytes={}",
                prefix,
                plan.name(),
                rw,
                mem.subtree_max_row_bytes
            ));
            for child in plan.children() {
                walk(child, indent + 2, lines);
            }
        }
        let mut lines = Vec::new();
        walk(plan, 0, &mut lines);
        lines.join("\n")
    }

    fn assert_plan_matches_canonical(plan: &Arc<dyn ExecutionPlan>, name: &str) {
        let canonical_path = plans_dir().join(format!("{name}.txt"));
        let canonical = std::fs::read_to_string(&canonical_path)
            .unwrap_or_else(|_| panic!("canonical file not found: {}", canonical_path.display()));
        let actual = format!("{}\n--- memory ---\n{}", plan_str(plan), memory_str(plan));
        assert_eq!(
            actual,
            canonical.trim_end(),
            "plan for '{name}' does not match {}",
            canonical_path.display()
        );
    }

    // ── Basic correctness ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_nation_row_count() {
        let ctx = create_context_with_tables(&testdata_dir()).await.unwrap();
        assert_eq!(count(&ctx, "SELECT count(*) FROM nation").await, 25);
    }

    #[tokio::test]
    async fn test_region_nation_join() {
        let ctx = create_context_with_tables(&testdata_dir()).await.unwrap();
        let n = count(
            &ctx,
            "SELECT count(*) FROM nation JOIN region ON nation.n_regionkey = region.r_regionkey",
        )
        .await;
        assert_eq!(n, 25);
    }

    // ── GPU plan node tests ──────────────────────────────────────────────────

    /// Filter + aggregate: SELECT count(*) FROM customer WHERE c_acctbal > 0
    /// Expected GPU nodes: GpuAggregateExec (partial + final), GpuFilterExec
    #[tokio::test]
    async fn test_gpu_nodes_filter_agg() {
        let ctx = create_context_with_tables(&testdata_dir()).await.unwrap();
        let query = "SELECT count(*) FROM customer WHERE c_acctbal > 0";

        let plan = ctx.sql(query).await.unwrap().create_physical_plan().await.unwrap();
        assert_plan_matches_canonical(&plan, "filter_agg");

        let n = count(&ctx, query).await;
        assert!(n > 0 && n <= 150_000, "unexpected count {n}");
    }

    /// Hash join + sort: nations joined with their region, sorted by name.
    /// Expected GPU nodes: GpuSortExec, GpuHashJoinExec
    #[tokio::test]
    async fn test_gpu_nodes_join_sort() {
        let ctx = create_context_with_tables(&testdata_dir()).await.unwrap();
        let query = "
            SELECT n.n_name, r.r_name
            FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey
            ORDER BY n.n_name";

        let plan = ctx.sql(query).await.unwrap().create_physical_plan().await.unwrap();
        assert_plan_matches_canonical(&plan, "join_sort");

        // Result: 25 rows (every nation has exactly one region)
        let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 25);
    }

    /// Group by + join + sort: nations per region, sorted descending by count.
    /// Expected GPU nodes: GpuSortExec, GpuAggregateExec, GpuHashJoinExec
    #[tokio::test]
    async fn test_gpu_nodes_group_join_sort() {
        let ctx = create_context_with_tables(&testdata_dir()).await.unwrap();
        let query = "
            SELECT r.r_name, count(*) AS nation_count
            FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey
            GROUP BY r.r_name
            ORDER BY nation_count DESC, r.r_name";

        let plan = ctx.sql(query).await.unwrap().create_physical_plan().await.unwrap();
        assert_plan_matches_canonical(&plan, "group_join_sort");

        // Result: 5 regions, each with exactly 5 nations.
        let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 5);
        let counts = batches[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..counts.len() {
            assert_eq!(counts.value(i), 5, "region {} has {} nations, expected 5", i, counts.value(i));
        }
    }

    // ── Memory budget tests ──────────────────────────────────────────────────

    /// Find all GpuScanExec nodes and return their batch sizes.
    fn scan_batch_sizes(plan: &Arc<dyn ExecutionPlan>) -> Vec<usize> {
        let mut sizes = Vec::new();
        if let Some(scan) = plan.as_any().downcast_ref::<GpuScanExec>() {
            sizes.push(scan.gpu_batch_size);
        }
        for child in plan.children() {
            sizes.extend(scan_batch_sizes(child));
        }
        sizes
    }

    /// With a tight GPU memory budget, the batch size should be reduced below
    /// the default 8192. Results must still be correct.
    #[tokio::test]
    async fn test_memory_budget_reduces_batch_size() {
        // 10 KiB budget → should force a very small batch size.
        let ctx =
            create_context_with_tables_and_budget(&testdata_dir(), 10 * 1024).await.unwrap();
        let query = "
            SELECT n.n_name, r.r_name
            FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey
            ORDER BY n.n_name";

        let plan = ctx.sql(query).await.unwrap().create_physical_plan().await.unwrap();
        let sizes = scan_batch_sizes(&plan);
        assert!(!sizes.is_empty(), "expected GpuScanExec nodes in plan");
        for &bs in &sizes {
            assert!(bs < 8192, "expected batch_size < 8192 with 10KiB budget, got {bs}");
            assert!(bs >= 1, "batch_size must be at least 1");
        }

        // Results must still be correct despite smaller batches.
        let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 25);
    }
}
