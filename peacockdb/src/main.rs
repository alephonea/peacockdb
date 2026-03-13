use std::path::PathBuf;

use clap::Parser;
use datafusion::arrow::util::pretty::print_batches;
use peacockdb_core::create_context_with_tables;

#[derive(Parser)]
#[command(name = "peacockdb", about = "GPU-accelerated analytical database")]
struct Cli {
    /// Directory of Parquet files; each file becomes a table named after its stem.
    #[arg(long)]
    data_dir: PathBuf,

    /// SQL query to execute.
    #[arg(long)]
    query: String,

    /// GPU memory limit in bytes (reserved for future use).
    #[arg(long)]
    gpu_memory_limit: Option<u64>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let ctx = create_context_with_tables(&cli.data_dir).await?;
    let df = ctx.sql(&cli.query).await?;
    let batches = df.collect().await?;
    print_batches(&batches)?;

    Ok(())
}
