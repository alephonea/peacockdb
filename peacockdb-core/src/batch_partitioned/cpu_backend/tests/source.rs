//! The loader: a lane reads the row groups the mapping gave it, one batch per call.
//!
//! Parquet is written here rather than taken from the corpus, because what is under test
//! is the mapping — which row groups this lane's third call reads — and a fixture whose
//! row groups nobody chose cannot state that.

use super::*;
use crate::batch_partitioned::backend::{Backend, NodeExecutors};
use crate::batch_partitioned::cpu_backend::backend::CpuBackend;
use crate::batch_partitioned::executor::{SourceExecutor, SourceStep};
use crate::batch_partitioned::nodes::GpuLoadParquet;
use crate::batch_partitioned::parquet_meta::ScanMetadata;
use crate::batch_partitioned::partitioner::RowGroupMeta;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::WriterProperties;
use std::path::PathBuf;

/// Six rows in three row groups of two, written once per process.
///
/// Once per PROCESS and not per call: libtest runs these cases in parallel, and a
/// check-then-create is true the moment the file is created — before its footer is
/// written — so the second thread would read a parquet with no footer.
fn table() -> PathBuf {
    static PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    PATH.get_or_init(write_table).clone()
}

fn write_table() -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("peacock-cpu-source-{}.parquet", std::process::id()));
    let batch = grouped(
        vec![
            Some("a"),
            Some("b"),
            Some("c"),
            Some("d"),
            Some("e"),
            Some("f"),
        ],
        (1..=6).map(Some).collect(),
    );
    let file = std::fs::File::create(&path).expect("a writable temp dir");
    let properties = WriterProperties::builder()
        .set_max_row_group_size(2)
        .build();
    let mut writer = ArrowWriter::try_new(file, batch.record_batch().schema(), Some(properties))
        .expect("the writer opens");
    writer.write(batch.record_batch()).expect("six rows");
    writer.close().expect("the footer is written");
    path
}

fn loader(projection: Vec<u32>, partition_groups: Vec<Vec<Vec<u32>>>) -> GpuLoadParquet {
    let path = table();
    let scan = ScanMetadata {
        file: path.to_string_lossy().into_owned(),
        groups: (0..3)
            .map(|index| RowGroupMeta {
                index,
                rows: 2,
                bytes: 16,
            })
            .collect(),
        can_be_null: vec![false, false],
    };
    let columns: Vec<(&str, DataType)> = projection
        .iter()
        .map(|column| GROUPED[*column as usize].clone())
        .collect();
    GpuLoadParquet::new(
        "t".to_string(),
        projection,
        partition_groups,
        &scan,
        None,
        schema_of(&columns),
    )
}

/// Every batch the lane produced, as the values in it, so a case reads as the rows the
/// mapping named.
fn lane(node: &GpuLoadParquet, lane: usize) -> Vec<Vec<i64>> {
    let NodeExecutors::Source(mut source) =
        CpuBackend::executors_for(&ctx(), node, 0, lane).expect("a loader builds a source")
    else {
        panic!("a loader is a source");
    };
    let mut batches = Vec::new();
    loop {
        match source.next_batch().expect("the read succeeds") {
            SourceStep::Batch {
                batch,
                source: next,
                ..
            } => {
                batches.push(values_in(&batch));
                source = next;
            }
            SourceStep::Exhausted => return batches,
        }
    }
}

fn values_in(batch: &CpuBatch) -> Vec<i64> {
    let record = batch.record_batch();
    let column = record
        .column(record.num_columns() - 1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("the value column");
    (0..record.num_rows())
        .map(|row| column.value(row))
        .collect()
}

/// One call per mapping entry, and its rows are that entry's row groups — not the
/// reader's own chunking, which would make a lane's batch count the decoder's business.
#[test]
fn a_lane_emits_one_batch_per_entry_of_its_mapping() {
    let node = loader(vec![0, 1], vec![vec![vec![0], vec![1, 2]], vec![vec![2]]]);
    assert_eq!(
        lane(&node, 0),
        vec![vec![1, 2], vec![3, 4, 5, 6]],
        "two entries, and the second reads both of its row groups as one batch"
    );
    assert_eq!(
        lane(&node, 1),
        vec![vec![5, 6]],
        "the other lane's own entry"
    );
}

/// The mask maps what the projection names: column 1 alone comes back as one column, and
/// it is the value column rather than the key the reader would hand over first.
#[test]
fn a_lane_reads_the_columns_its_projection_names_and_no_others() {
    let node = loader(vec![1], vec![vec![vec![0], vec![1]]]);
    let NodeExecutors::Source(source) =
        CpuBackend::executors_for(&ctx(), &node, 0, 0).expect("a loader builds a source")
    else {
        panic!("a loader is a source");
    };
    let SourceStep::Batch { batch, .. } = source.next_batch().expect("the read succeeds") else {
        panic!("the lane has two entries");
    };
    assert_eq!(
        batch.record_batch().num_columns(),
        1,
        "one column named, one column read"
    );
    assert_eq!(values_in(&batch), vec![1, 2], "and it is v, not k");
}

/// A lane the mapping gave nothing is exhausted on its first step rather than an error:
/// the partitioner is free to produce one, and the driver's answer is an empty lane.
#[test]
fn a_lane_with_no_entries_is_exhausted_at_once() {
    let node = loader(vec![0, 1], vec![vec![vec![0]], Vec::new()]);
    assert!(lane(&node, 1).is_empty());
}

/// `ProjectionMask::roots` is a set, so a descending projection would come back permuted
/// against the declared schema — on this backend and, identically, on the device, which
/// sends the same list to cuDF. Two engines wrong the same way is what the goldens cannot
/// catch, so the rule is stated where it can break.
#[test]
fn a_projection_that_does_not_ascend_is_refused() {
    let node = loader(vec![1, 0], vec![vec![vec![0]]]);
    let Err(refused) = CpuBackend::executors_for(&ctx(), &node, 0, 0) else {
        panic!("a projection out of order built a source");
    };
    assert!(
        format!("{refused:?}").contains("must ascend"),
        "{refused:?}"
    );
}

/// A lane the mapping does not have is a plan the partitioner did not produce, and the
/// message says which lane against how many.
#[test]
fn a_lane_past_the_mapping_is_refused_by_number() {
    let node = loader(vec![0, 1], vec![vec![vec![0]]]);
    let Err(refused) = CpuBackend::executors_for(&ctx(), &node, 0, 3) else {
        panic!("a lane past the mapping built a source");
    };
    assert!(format!("{refused:?}").contains("lane 3"), "{refused:?}");
}
