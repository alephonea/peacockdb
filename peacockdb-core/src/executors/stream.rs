//! Streaming plumbing shared by the recursive full-table CPU driver and the
//! per-node CPU primitive: the resident-memory enforcer, the adapter that feeds
//! precomputed child batches back into a plan node, and the instrumented stream
//! that measures each node as it drains.
//!
//! This file exists because [`super::full_table_cpu_executor`] and
//! [`super::single_node`] BOTH need this machinery — neither can own it.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use futures::Stream;

use super::executor::NodeMemoryStats;
use crate::cpu_executor::{assert_type_accountable, batch_allocated_size, ColAccum};

// ---------------------------------------------------------------------------
// Resident-memory enforcement (Part 2): strict "GPU" memory budget, mid-run.
// ---------------------------------------------------------------------------

/// Tracks the modeled concurrently-resident data set during node-by-node
/// execution and trips a budget the MOMENT it is crossed (before the query
/// completes). The accounting is delegated to `resident::peak_from_skeleton`, the
/// SAME path-sum logic as the offline `resident::check_resident_budget`, so the
/// mid-run verdict can't drift from the reference. As each node's output stream
/// completes its `output_bytes` is recorded; the peak is recomputed and grows
/// monotonically toward the true peak, so the crossing is detected exactly when
/// it happens.
pub(crate) struct ResidentEnforcer {
    budget: usize,
    state: Mutex<EnforcerState>,
}

struct EnforcerState {
    /// seq -> (stripped node name, child seqs). Built during `build_stream`.
    skeleton: HashMap<usize, (String, Vec<usize>)>,
    /// Post-order means the LAST node registered is the root.
    root: usize,
    /// Completed nodes' output_bytes (missing = not yet materialized = 0).
    output_bytes: HashMap<usize, usize>,
    tripped: Option<String>,
}

impl ResidentEnforcer {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            budget,
            state: Mutex::new(EnforcerState {
                skeleton: HashMap::new(),
                root: 0,
                output_bytes: HashMap::new(),
                tripped: None,
            }),
        }
    }

    pub(crate) fn register(&self, seq: usize, name: String, children: Vec<usize>) {
        let mut s = self.state.lock().unwrap();
        s.skeleton.insert(seq, (name, children));
        s.root = seq;
    }

    pub(crate) fn on_complete(&self, seq: usize, output_bytes: usize) {
        let mut s = self.state.lock().unwrap();
        s.output_bytes.insert(seq, output_bytes);
        if s.tripped.is_none() {
            let peak = crate::resident::peak_from_skeleton(s.root, &s.skeleton, &s.output_bytes);
            if peak > self.budget {
                s.tripped = Some(format!(
                    "resident GPU memory budget exceeded: peak {peak} bytes > budget {} bytes",
                    self.budget
                ));
            }
        }
    }

    pub(crate) fn tripped_error(&self) -> Option<DataFusionError> {
        self.state
            .lock()
            .unwrap()
            .tripped
            .clone()
            .map(DataFusionError::ResourcesExhausted)
    }
}

pub(crate) async fn drain_stream(mut stream: SendableRecordBatchStream) -> Result<Vec<RecordBatch>> {
    use futures::StreamExt;
    let mut out = Vec::new();
    while let Some(batch) = stream.next().await {
        out.push(batch?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// StreamSourceExec — adapt a live SendableRecordBatchStream as a child node
// ---------------------------------------------------------------------------

/// `ExecutionPlan` that returns a pre-built `SendableRecordBatchStream` from
/// `execute(0, _)`. Single-partition, single-use: the stream is taken on first
/// `execute()` call; subsequent calls error.
pub(crate) struct StreamSourceExec {
    schema: SchemaRef,
    stream: Mutex<Option<SendableRecordBatchStream>>,
    cache: PlanProperties,
}

impl fmt::Debug for StreamSourceExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamSourceExec")
            .field("schema", &self.schema)
            .finish()
    }
}

impl StreamSourceExec {
    pub(crate) fn new(
        schema: SchemaRef,
        eq_properties: EquivalenceProperties,
        stream: SendableRecordBatchStream,
    ) -> Self {
        let cache = PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            schema,
            stream: Mutex::new(Some(stream)),
            cache,
        }
    }
}

impl DisplayAs for StreamSourceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StreamSourceExec")
    }
}

impl ExecutionPlan for StreamSourceExec {
    fn name(&self) -> &str {
        "StreamSourceExec"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
    fn properties(&self) -> &PlanProperties {
        &self.cache
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }
    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }
    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.stream
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| DataFusionError::Internal("StreamSourceExec executed twice".into()))
    }
}

// ---------------------------------------------------------------------------
// InstrumentedStream — accumulate NodeMemoryStats as batches flow through
// ---------------------------------------------------------------------------

pub(crate) struct InstrumentedStream {
    seq: usize,
    node_name: String,
    schema: SchemaRef,
    inner: SendableRecordBatchStream,
    allocated_bytes: usize,
    // Per-column accumulators (one per output field). Each level's bitmap/offset
    // overhead is charged ONCE at finalize from the accumulated totals, so
    // `output_bytes` is deterministic regardless of how rows are chunked into
    // batches (i.e. regardless of target_partitions). See `ColAccum`.
    cols: Vec<ColAccum>,
    row_count: usize,
    max_batch_rows: usize,
    collector: Arc<Mutex<Vec<(usize, NodeMemoryStats)>>>,
    /// Resident-budget enforcer (Part 2). `None` = no enforcement.
    enforcer: Option<Arc<ResidentEnforcer>>,
    done: bool,
}

impl InstrumentedStream {
    pub(crate) fn new(
        seq: usize,
        node_name: String,
        schema: SchemaRef,
        inner: SendableRecordBatchStream,
        collector: Arc<Mutex<Vec<(usize, NodeMemoryStats)>>>,
        enforcer: Option<Arc<ResidentEnforcer>>,
    ) -> Self {
        // Fail fast (here, not in Drop) if any output column type isn't accountable.
        for f in schema.fields() {
            assert_type_accountable(f.data_type());
        }
        let cols = vec![ColAccum::default(); schema.fields().len()];
        Self {
            seq,
            node_name,
            schema,
            inner,
            allocated_bytes: 0,
            cols,
            row_count: 0,
            max_batch_rows: 0,
            collector,
            enforcer,
            done: false,
        }
    }
}

impl InstrumentedStream {
    fn push_stat(&mut self) {
        if !self.done {
            self.done = true;
            // Logical size of the whole node output: each column's (and nested
            // level's) overhead charged ONCE from the accumulated totals —
            // independent of batch boundaries.
            let output_bytes: usize = self
                .cols
                .iter()
                .zip(self.schema.fields())
                .map(|(c, f)| c.size(f.data_type()))
                .sum();
            // Feed the resident enforcer this node's materialized size so it can
            // recompute the concurrent peak and trip the budget mid-run.
            if let Some(enf) = &self.enforcer {
                enf.on_complete(self.seq, output_bytes);
            }
            self.collector.lock().unwrap().push((
                self.seq,
                NodeMemoryStats {
                    node_name: self.node_name.clone(),
                    allocated_bytes: self.allocated_bytes,
                    output_bytes,
                    row_count: self.row_count,
                    max_batch_rows: self.max_batch_rows,
                    // #11 single-node execution = one output partition (coalesced):
                    // empty ⇒ the golden renders partitions=1, no sub-lines.
                    part_stats: Vec::new(),
                },
            ));
        }
    }
}

// Fires even when the stream is dropped before being fully exhausted
// (e.g. the child of a GlobalLimitExec is abandoned after LIMIT rows).
impl Drop for InstrumentedStream {
    fn drop(&mut self) {
        self.push_stat();
    }
}

impl Stream for InstrumentedStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let me = &mut *self;
        // Strict resident control: if a node's completion (possibly elsewhere in the
        // tree) has already pushed the modeled concurrent-resident over budget, fail
        // here rather than finish the query.
        if let Some(enf) = &me.enforcer {
            if let Some(e) = enf.tripped_error() {
                me.done = true;
                return Poll::Ready(Some(Err(e)));
            }
        }
        let poll = Pin::new(&mut me.inner).poll_next(cx);
        if let Poll::Ready(item) = &poll {
            match item {
                Some(Ok(batch)) => {
                    me.allocated_bytes += batch_allocated_size(batch);
                    for (acc, col) in me.cols.iter_mut().zip(batch.columns()) {
                        acc.add(col.as_ref());
                    }
                    me.row_count += batch.num_rows();
                    if batch.num_rows() > me.max_batch_rows {
                        me.max_batch_rows = batch.num_rows();
                    }
                }
                None => me.push_stat(),
                _ => {}
            }
        }
        poll
    }
}

impl RecordBatchStream for InstrumentedStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
