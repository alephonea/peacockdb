//! Row-group metadata for the partitioner, read at plan time.
//!
//! Pruning is legacy's — the same survivors the existing scan reads — and what is added
//! here is the per-group rows and bytes the mapping needs. Bytes are the parquet
//! column-chunk totals over the projected columns: a varchar's width is a property of
//! the data, and the file already knows it.

use std::collections::BTreeSet;

use datafusion::datasource::physical_plan::ParquetExec;
use datafusion::parquet::file::reader::{FileReader, SerializedFileReader};

use super::error::PlanError;
use super::partitioner::RowGroupMeta;
use crate::gpu_rowgroup_prune::surviving_row_groups;

pub fn survivor_metadata(parquet: &ParquetExec) -> Result<Vec<RowGroupMeta>, PlanError> {
    let config = parquet.base_config();
    // At tp>1 DataFusion splits ONE file into several byte-range entries, all with the same
    // path, so entries are not files. Genuinely several files would each have their own row
    // groups, and this mapping addresses one file's — measuring the first and sizing the
    // whole plan from it would be a wrong answer rather than an error.
    let paths: BTreeSet<String> = config
        .file_groups
        .iter()
        .flatten()
        .map(|file| format!("/{}", file.object_meta.location))
        .collect();
    if paths.len() > 1 {
        return Err(PlanError::Unsupported(format!(
            "a scan over {} files: the row-group mapping addresses one file",
            paths.len()
        )));
    }
    // A local path, since the mapping is read off the file here at plan time. An object
    // store's location would simply not open, which is an error rather than a wrong answer.
    let path = paths
        .into_iter()
        .next()
        .ok_or_else(|| PlanError::Invalid("a scan with no files".to_string()))?;

    let file =
        std::fs::File::open(&path).map_err(|e| PlanError::Invalid(format!("{path}: {e}")))?;
    let reader =
        SerializedFileReader::new(file).map_err(|e| PlanError::Invalid(format!("{path}: {e}")))?;

    // The projection indexes the file schema, and a column chunk sits at the same
    // position: every table this engine reads is flat.
    let projected: Vec<usize> = match &config.projection {
        Some(columns) => columns.clone(),
        None => (0..config.file_schema.fields().len()).collect(),
    };
    let survivors = surviving_row_groups(parquet);

    let mut metadata = Vec::new();
    for (index, group) in reader.metadata().row_groups().iter().enumerate() {
        let index = index as u32;
        if survivors
            .as_ref()
            .is_some_and(|kept| !kept.contains(&index))
        {
            continue;
        }
        // A projection index with no column chunk would silently make the source look
        // smaller, and bytes decide both the batch size and the lane count.
        let mut bytes: i64 = 0;
        for column in &projected {
            let chunk = group.columns().get(*column).ok_or_else(|| {
                PlanError::Invalid(format!(
                    "{path}: projected column {column} has no chunk in row group {index}, so \
                     the schema is not flat and a projection index is not a chunk position"
                ))
            })?;
            bytes += chunk.uncompressed_size();
        }
        metadata.push(RowGroupMeta {
            index,
            rows: group.num_rows() as u64,
            bytes: bytes as u64,
        });
    }
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::execution::context::SessionContext;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::prelude::ParquetReadOptions;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn minimal() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal")
    }

    /// The scan under a query, which is what carries the projection and the pruning
    /// predicate this file reads.
    async fn scan_of(sql: &str) -> Arc<dyn ExecutionPlan> {
        let ctx = crate::register_tables_for(crate::build_session_state(1), &minimal())
            .await
            .expect("register the minimal tables");
        let plan = ctx
            .sql(sql)
            .await
            .expect("plan the query")
            .create_physical_plan()
            .await
            .expect("physical plan");
        find_scan(&plan).expect("a scan")
    }

    fn find_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        if plan.as_any().is::<ParquetExec>() {
            return Some(plan.clone());
        }
        plan.children()
            .into_iter()
            .find_map(|child| find_scan(&child.clone()))
    }

    async fn survivors(sql: &str) -> Vec<RowGroupMeta> {
        let scan = scan_of(sql).await;
        survivor_metadata(scan.as_any().downcast_ref::<ParquetExec>().unwrap())
            .expect("read the metadata")
    }

    /// The file's own numbers, read independently of the code under test.
    fn chunk_bytes(table: &str, columns: &[usize]) -> u64 {
        let file = std::fs::File::open(minimal().join(format!("{table}.parquet"))).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        reader
            .metadata()
            .row_groups()
            .iter()
            .flat_map(|group| columns.iter().map(|c| group.column(*c).uncompressed_size()))
            .sum::<i64>() as u64
    }

    #[tokio::test]
    async fn bytes_are_the_column_chunk_totals_of_the_projected_columns() {
        // n_name is field 1 of the file schema, and its chunks are what a scan projecting
        // it reads — not a width derived from the type, which is the whole point of
        // measuring the file.
        let measured: u64 = survivors("SELECT n_name FROM nation")
            .await
            .iter()
            .map(|g| g.bytes)
            .sum();
        assert_eq!(measured, chunk_bytes("nation", &[1]));
    }

    #[tokio::test]
    async fn a_narrower_projection_reads_fewer_bytes() {
        let one: u64 = survivors("SELECT n_name FROM nation")
            .await
            .iter()
            .map(|g| g.bytes)
            .sum();
        let all: u64 = survivors("SELECT * FROM nation")
            .await
            .iter()
            .map(|g| g.bytes)
            .sum();
        assert_eq!(all, chunk_bytes("nation", &[0, 1, 2, 3]));
        assert!(one < all, "{one} is not less than {all}");
    }

    #[tokio::test]
    async fn rows_are_the_row_groups_own_counts() {
        let measured = survivors("SELECT * FROM customer").await;
        // customer is two row groups at this scale, the second one short.
        assert_eq!(measured.len(), 2);
        assert_eq!(measured.iter().map(|g| g.rows).sum::<u64>(), 150_000);
        assert_eq!(measured[0].index, 0);
        assert_eq!(measured[1].index, 1);
    }

    #[tokio::test]
    async fn pruning_leaves_only_the_row_groups_that_survived_it() {
        // c_custkey rises with the row groups, so a predicate past the first group's
        // maximum leaves the second — and the index it keeps is the file's, not a position
        // in the survivor list.
        let kept = survivors("SELECT c_custkey FROM customer WHERE c_custkey > 140000").await;
        assert_eq!(kept.len(), 1, "{kept:?}");
        assert_eq!(kept[0].index, 1);
        assert_eq!(
            kept[0].bytes,
            chunk_bytes("customer", &[0]) - customer_group_zero_bytes()
        );
    }

    fn customer_group_zero_bytes() -> u64 {
        let file = std::fs::File::open(minimal().join("customer.parquet")).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        reader.metadata().row_group(0).column(0).uncompressed_size() as u64
    }

    #[tokio::test]
    async fn a_scan_over_several_files_is_refused_rather_than_measured_from_one() {
        // The mapping addresses one file's row groups, so a second file would be sized
        // from the first's — a wrong answer where an error belongs.
        let dir = std::env::temp_dir().join(format!(
            "peacockdb-batch-partitioned-multifile-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["nation-a.parquet", "nation-b.parquet"] {
            std::fs::copy(minimal().join("nation.parquet"), dir.join(name)).unwrap();
        }
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "nations",
            dir.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
        .expect("register the directory");
        let plan = ctx
            .sql("SELECT n_name FROM nations")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let scan = find_scan(&plan).expect("a scan");
        let err = survivor_metadata(scan.as_any().downcast_ref::<ParquetExec>().unwrap())
            .expect_err("a multi-file scan has no single mapping");
        assert!(
            matches!(&err, PlanError::Unsupported(what) if what.contains("2 files")),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
