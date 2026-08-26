//! `GpuLoadParquet` on the CPU: one call per batch, reading the row groups the mapping
//! gave this lane.
//!
//! The partitioner decided at plan time which row groups each of a lane's batches reads,
//! and this reads exactly those — so a lane's batch count and its rows are the plan's
//! rather than a reader's chunking. That is what lets the two backends and the goldens
//! agree on what a batch is.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{Schema as ArrowSchema, SchemaRef};
use datafusion::parquet::arrow::ProjectionMask;
use datafusion::parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};

use super::super::cpu_batch::CpuBatch;
use super::super::error::PlanError;
use super::super::executor::{BackendError, CallStats};
use super::super::nodes::GpuLoadParquet;
use super::declared_as;

/// A lane's reads, in the order the mapping named them.
pub struct CpuSource {
    file: String,
    /// The footer, parsed once: a lane reads the same file once per batch, and re-parsing
    /// it per call is the whole of what a scan does besides decoding.
    metadata: ArrowReaderMetadata,
    projection: ProjectionMask,
    /// The row groups per batch this lane still owes, front first.
    batches: std::collections::VecDeque<Vec<usize>>,
    schema: SchemaRef,
}

impl CpuSource {
    pub fn new(
        node: &GpuLoadParquet,
        lane: usize,
        schema: &ArrowSchema,
    ) -> Result<Self, PlanError> {
        let batches = node.partition_groups.get(lane).ok_or_else(|| {
            PlanError::Invalid(format!(
                "lane {lane} of a scan the partitioner mapped into {} lanes",
                node.partition_groups.len()
            ))
        })?;
        // `ProjectionMask::roots` is a SET: the reader emits the columns in file order
        // whatever order they are named in, so a projection that is not ascending would
        // hand this lane its columns permuted against the schema the node declares — and
        // the device, which sends the same list to cuDF, would permute them identically,
        // so the goldens would agree and be wrong together.
        if node.projection.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PlanError::Invalid(format!(
                "a scan's projection is read as a set, so it must ascend: {:?}",
                node.projection
            )));
        }
        let mut file = std::fs::File::open(&node.file)
            .map_err(|error| PlanError::Invalid(format!("opening {}: {error}", node.file)))?;
        let metadata = ArrowReaderMetadata::load(&mut file, ArrowReaderOptions::new())
            .map_err(|error| PlanError::Invalid(format!("reading {}: {error}", node.file)))?;
        let projection = ProjectionMask::roots(
            metadata.parquet_schema(),
            node.projection.iter().map(|column| *column as usize),
        );
        Ok(Self {
            file: node.file.clone(),
            metadata,
            projection,
            batches: batches
                .iter()
                .map(|groups| groups.iter().map(|group| *group as usize).collect())
                .collect(),
            schema: Arc::new(schema.clone()),
        })
    }

    /// The next batch, or `None` where the mapping gave this lane nothing more.
    pub fn read_next(&mut self) -> Result<Option<(CpuBatch, CallStats)>, BackendError> {
        let Some(groups) = self.batches.pop_front() else {
            return Ok(None);
        };
        let file = std::fs::File::open(&self.file)
            .map_err(|error| BackendError::new(format!("{}: {error}", self.file)))?;
        let reader =
            ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.metadata.clone())
                .with_row_groups(groups)
                .with_projection(self.projection.clone())
                .build()
                .map_err(|error| BackendError::new(format!("reading {}: {error}", self.file)))?;
        let read: Vec<RecordBatch> = reader
            .collect::<Result<Vec<RecordBatch>, _>>()
            .map_err(|error| BackendError::new(format!("reading {}: {error}", self.file)))?;
        // One batch per call whatever the reader's own chunking: the plan decided how many
        // batches this lane has, and a node above counts calls.
        let batch = concat_batches(
            &read
                .first()
                .map(|first| first.schema())
                .unwrap_or_else(|| self.schema.clone()),
            read.iter(),
        )
        .map_err(|error| BackendError::new(format!("joining the row groups read: {error}")))?;
        let batch = declared_as(batch, &self.schema)?;
        Ok(Some((CpuBatch::new(batch), CallStats::default())))
    }
}
