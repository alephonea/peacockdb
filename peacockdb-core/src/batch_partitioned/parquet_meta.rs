//! Row-group metadata for the partitioner, read at plan time.
//!
//! Pruning is legacy's — the same survivors the existing scan reads — and what is added
//! here is the per-group rows and bytes the mapping needs. Bytes are the parquet
//! column-chunk totals over the projected columns: a varchar's width is a property of
//! the data, and the file already knows it.

use datafusion::datasource::physical_plan::ParquetExec;
use datafusion::parquet::file::reader::{FileReader, SerializedFileReader};

use super::error::PlanError;
use super::partitioner::RowGroupMeta;
use crate::gpu_rowgroup_prune::surviving_row_groups;

pub fn survivor_metadata(parquet: &ParquetExec) -> Result<Vec<RowGroupMeta>, PlanError> {
    let config = parquet.base_config();
    let path = config
        .file_groups
        .first()
        .and_then(|group| group.first())
        .map(|file| format!("/{}", file.object_meta.location))
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
        let bytes: i64 = projected
            .iter()
            .filter_map(|column| group.columns().get(*column))
            .map(|chunk| chunk.uncompressed_size())
            .sum();
        metadata.push(RowGroupMeta {
            index,
            rows: group.num_rows() as u64,
            bytes: bytes as u64,
        });
    }
    Ok(metadata)
}
