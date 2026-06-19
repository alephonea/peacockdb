//! Compute surviving parquet row-group indices for a `GpuScanExec`, so the GPU
//! scan (cuDF `read_parquet` `set_row_groups`) decodes ONLY the groups that can
//! match the scan's static predicate — matching the CPU oracle, which DataFusion's
//! `ParquetExec` already prunes the same way.
//!
//! PARITY BY CONSTRUCTION: we reuse the SAME machinery DataFusion's `ParquetExec`
//! uses for CPU-side row-group pruning — its pushed-down `predicate()` run through a
//! `PruningPredicate` over `RowGroupPruningStatistics` (a faithful port of
//! datafusion-44's internal `row_group_filter.rs`). Same predicate + same row-group
//! stats => identical surviving set. SCOPE: single-source scans with a static
//! pushdown predicate (TPC-H q6/q1/q14/q15-style). Multi-file, no predicate, or
//! join/dynamic (date_dim) ranges => `None` (read all groups, as today; #16).

use std::sync::Arc;

use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::datatypes::Schema;
use datafusion::common::Column;
use datafusion::datasource::physical_plan::ParquetExec;
use datafusion::parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use datafusion::parquet::file::metadata::RowGroupMetaData;
use datafusion::parquet::file::reader::{FileReader, SerializedFileReader};
use datafusion::parquet::schema::types::SchemaDescriptor;
use datafusion::physical_optimizer::pruning::{PruningPredicate, PruningStatistics};

/// Faithful port of datafusion-44 `row_group_filter::RowGroupPruningStatistics`:
/// adapts parquet row-group column-chunk statistics to the `PruningStatistics`
/// the `PruningPredicate` consumes. Kept identical so GPU pruning == CPU pruning.
struct RowGroupPruningStatistics<'a> {
    parquet_schema: &'a SchemaDescriptor,
    row_group_metadatas: Vec<&'a RowGroupMetaData>,
    arrow_schema: &'a Schema,
}

impl<'a> RowGroupPruningStatistics<'a> {
    fn metadata_iter(&'a self) -> impl Iterator<Item = &'a RowGroupMetaData> + 'a {
        self.row_group_metadatas.iter().copied()
    }

    fn converter<'b>(&'a self, column: &'b Column) -> Option<StatisticsConverter<'a>> {
        StatisticsConverter::try_new(&column.name, self.arrow_schema, self.parquet_schema).ok()
    }
}

impl PruningStatistics for RowGroupPruningStatistics<'_> {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.converter(column)?.row_group_mins(self.metadata_iter()).ok()
    }
    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.converter(column)?.row_group_maxes(self.metadata_iter()).ok()
    }
    fn num_containers(&self) -> usize {
        self.row_group_metadatas.len()
    }
    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        self.converter(column)?
            .row_group_null_counts(self.metadata_iter())
            .ok()
            .map(|c| Arc::new(c) as ArrayRef)
    }
    fn row_counts(&self, column: &Column) -> Option<ArrayRef> {
        self.converter(column)?
            .row_group_row_counts(self.metadata_iter())
            .ok()
            .flatten()
            .map(|c| Arc::new(c) as ArrayRef)
    }
    fn contained(
        &self,
        _column: &Column,
        _values: &std::collections::HashSet<datafusion::scalar::ScalarValue>,
    ) -> Option<datafusion::arrow::array::BooleanArray> {
        None
    }
}

/// Surviving row-group indices for `parquet`'s single source under its pushdown
/// predicate. `None` => no pruning applicable (read all groups): no predicate,
/// not exactly one file, unreadable metadata, or the predicate prunes nothing.
pub fn surviving_row_groups(parquet: &ParquetExec) -> Option<Vec<u32>> {
    let predicate = parquet.predicate()?; // None when nothing was pushed down (e.g. #16 dynamic)
    let config = parquet.base_config();

    // Single-source only — cuDF set_row_groups is per-source; keep scope tight.
    let files: Vec<_> = config.file_groups.iter().flatten().collect();
    if files.len() != 1 {
        return None;
    }
    let path = format!("/{}", files[0].object_meta.location);

    // Read row-group metadata (sync; the parquet is local at serialize time).
    let file = std::fs::File::open(&path).ok()?;
    let reader = SerializedFileReader::new(file).ok()?;
    let meta = reader.metadata();
    let groups: Vec<&RowGroupMetaData> = meta.row_groups().iter().collect();
    if groups.is_empty() {
        return None;
    }
    let parquet_schema = meta.file_metadata().schema_descr();

    let pruning = PruningPredicate::try_new(predicate.clone(), config.file_schema.clone()).ok()?;
    let stats = RowGroupPruningStatistics {
        parquet_schema,
        row_group_metadatas: groups,
        arrow_schema: config.file_schema.as_ref(),
    };
    // `keep[i] == false` => row group i cannot match the predicate (prune it).
    let keep = pruning.prune(&stats).ok()?;
    let survivors: Vec<u32> = keep
        .iter()
        .enumerate()
        .filter_map(|(i, &k)| if k { Some(i as u32) } else { None })
        .collect();

    // Nothing pruned -> behave exactly as today (empty list = cuDF reads all groups).
    if survivors.len() == keep.len() {
        None
    } else {
        Some(survivors)
    }
}
