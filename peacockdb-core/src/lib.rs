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
