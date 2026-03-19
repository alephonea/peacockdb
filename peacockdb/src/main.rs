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

    /// Number of CPU partitions for parallel execution (defaults to number of CPUs).
    #[arg(long)]
    target_partitions: Option<usize>,

    /// GPU memory budget in bytes (defaults to 2 GiB).
    #[arg(long, default_value_t = 2 * 1024 * 1024 * 1024)]
    gpu_memory_budget: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let target_partitions = cli.target_partitions.unwrap_or_else(num_cpus::get);
    let ctx = create_context_with_tables(&cli.data_dir, target_partitions, cli.gpu_memory_budget).await?;
    let df = ctx.sql(&cli.query).await?;
    let batches = df.collect().await?;
    print_batches(&batches)?;

    Ok(())
}
