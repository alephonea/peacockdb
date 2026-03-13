use std::path::Path;
use std::sync::Arc;

use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl};
use datafusion::execution::context::SessionContext;
use datafusion::error::Result;

/// Scans `data_dir` for `.parquet` files and registers each as a table in a new
/// `SessionContext`. The table name is the file stem (e.g. `orders.parquet` → `orders`).
pub async fn create_context_with_tables(data_dir: &Path) -> Result<SessionContext> {
    let ctx = SessionContext::new();

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
    use std::path::PathBuf;

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

    #[tokio::test]
    async fn test_nation_row_count() {
        let ctx = create_context_with_tables(&testdata_dir()).await.unwrap();
        assert_eq!(count(&ctx, "SELECT count(*) FROM nation").await, 25);
    }

    #[tokio::test]
    async fn test_region_nation_join() {
        let ctx = create_context_with_tables(&testdata_dir()).await.unwrap();
        // Every nation belongs to exactly one region; joined count equals nation count.
        let n = count(
            &ctx,
            "SELECT count(*) FROM nation JOIN region ON nation.n_regionkey = region.r_regionkey",
        )
        .await;
        assert_eq!(n, 25);
    }
}
