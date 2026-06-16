use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use datafusion::arrow::array::{
    Array, BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray, StringArray,
    StringViewArray,
};
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    execute_stream, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::physical_plan::union::{InterleaveExec, UnionExec};
use futures::Stream;

use crate::gpu_rule::{
    GpuAggregateExec, GpuCoalesceBatchesExec, GpuCoalescePartitionsExec, GpuFilterExec,
    GpuHashJoinExec, GpuProjectExec, GpuRepartitionExec, GpuScanExec, GpuSortExec,
    GpuSortPreservingMergeExec,
};

// 
fn strip_gpu(node: Arc<dyn ExecutionPlan>) -> (Arc<dyn ExecutionPlan>, Option<usize>) {
    macro_rules! try_strip {
        ($ty:ty) => {
            if let Some(n) = node.as_any().downcast_ref::<$ty>() {
                return (n.inner().clone(), None);
            }
        };
    }

    // GpuScanExec is special: it carries the memory-budget batch size.
    if let Some(scan) = node.as_any().downcast_ref::<GpuScanExec>() {
        return (scan.inner().clone(), Some(scan.gpu_batch_size));
    }

    try_strip!(GpuFilterExec);
    try_strip!(GpuProjectExec);
    try_strip!(GpuAggregateExec);
    try_strip!(GpuHashJoinExec);
    try_strip!(GpuSortExec);
    try_strip!(GpuCoalesceBatchesExec);
    try_strip!(GpuCoalescePartitionsExec);
    try_strip!(GpuRepartitionExec);
    try_strip!(GpuSortPreservingMergeExec);

    // Plain CPU node — pass through unchanged.
    (node, None)
}

/// Apply a batch-size override to a `TaskContext`, returning the updated context.
/// The override comes from `GpuScanExec.gpu_batch_size`, which was computed by
/// `GpuMemoryBudgetRule` to keep GPU memory within budget.  We honour the same
/// limit on CPU so that peak working-set size stays within the same bound.
fn with_batch_size(ctx: Arc<TaskContext>, batch_size: usize) -> Arc<TaskContext> {
    let new_config = ctx.session_config().clone().with_batch_size(batch_size);
    Arc::new(TaskContext::new(
        ctx.task_id(),
        ctx.session_id(),
        new_config,
        ctx.scalar_functions().clone(),
        ctx.aggregate_functions().clone(),
        ctx.window_functions().clone(),
        ctx.runtime_env(),
    ))
}

// ---------------------------------------------------------------------------
// Node-by-node CPU execution
// ---------------------------------------------------------------------------

/// Per-node memory stats collected via the `on_node` callback.
#[derive(Clone)]
pub struct NodeMemoryStats {
    /// Name of the CPU node that was executed (GPU wrapper already stripped).
    pub node_name: String,
    /// Sum of `get_array_memory_size()` across all output batches (allocated upper bound).
    pub allocated_bytes: usize,
    /// Logical byte size of all batches produced by this node.
    pub output_bytes: usize,
    /// Total number of output rows across all batches.
    pub row_count: usize,
    /// Largest single batch (in rows) produced by this node.
    /// Compare against `GpuScanExec.gpu_batch_size` to verify the memory contract.
    pub max_batch_rows: usize,
}

/// Recursively strip all GPU wrapper nodes from a plan tree, returning a
/// structurally identical tree composed of plain DataFusion CPU nodes.
pub fn strip_gpu_tree(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    let (cpu_node, _) = strip_gpu(plan);
    let stripped_children = cpu_node
        .children()
        .into_iter()
        .map(|c| strip_gpu_tree(c.clone()))
        .collect::<Result<Vec<_>>>()?;
    cpu_node.with_new_children(stripped_children)
}

/// Execute a physical plan one node at a time, bottom-up, on CPU.
///
/// GPU wrapper nodes (`GpuFilterExec`, `GpuScanExec`, …) are stripped to their
/// inner DataFusion CPU nodes before execution.  The memory boundary encoded in
/// `GpuScanExec.gpu_batch_size` is preserved: the `TaskContext` batch size is
/// overridden to that value so the Parquet reader produces the same batch sizes
/// the GPU planner computed.
///
/// Execution is streaming: each node's input is a live `SendableRecordBatchStream`
/// wrapped in a `StreamSourceExec` rather than a fully-materialized `MemoryExec`.
/// This keeps peak memory bounded by what the underlying operators themselves
/// hold (e.g. a hash join still buffers its build side internally) plus a few
/// in-flight batches, not the full output of every intermediate node.
///
/// For each node the function:
/// 1. Strips the GPU wrapper (if any) → CPU node + optional batch_size.
/// 2. Applies the batch_size override to `TaskContext` if present.
/// 3. Recurses into the CPU node's children to obtain child streams.
/// 4. Wraps each child stream in `StreamSourceExec`.
/// 5. Calls `execute(0, ctx)` on the isolated CPU node with its stream stubs.
/// 6. Wraps the resulting stream in `InstrumentedStream`, which fires `on_node`
///    with the accumulated `NodeMemoryStats` once the stream is fully drained.
pub async fn execute_node_by_node(
    root: Arc<dyn ExecutionPlan>,
    task_ctx: Arc<TaskContext>,
    on_node: &mut dyn FnMut(&str, &NodeMemoryStats),
) -> Result<Vec<RecordBatch>> {
    let collector: Arc<Mutex<Vec<(usize, NodeMemoryStats)>>> = Arc::new(Mutex::new(Vec::new()));
    let seq_counter = Arc::new(AtomicUsize::new(0));
    let stream = build_stream(root.clone(), task_ctx, collector.clone(), seq_counter)?;
    let batches = drain_stream(stream).await?;
    let mut stats = std::mem::take(&mut *collector.lock().unwrap());
    // Stats are pushed in stream-completion/Drop order. That is normally
    // post-order, but a LIMIT parent finalizes (returns None) before its
    // abandoned child stream is dropped, which inverts the two. Sort by the
    // post-order `seq` assigned at build time so consumers (e.g. cpu_stats_str)
    // can rely on strict children-before-parent ordering.
    stats.sort_by_key(|(seq, _)| *seq);
    for (_, s) in &stats {
        on_node(&s.node_name, s);
    }
    Ok(batches)
}

/// Convenience wrapper: runs [`execute_node_by_node`] and collects
/// [`NodeMemoryStats`] per node in post-order (stream-completion order).
pub async fn execute_node_by_node_instrumented(
    root: Arc<dyn ExecutionPlan>,
    task_ctx: Arc<TaskContext>,
    stats: &mut Vec<NodeMemoryStats>,
) -> Result<Vec<RecordBatch>> {
    execute_node_by_node(root, task_ctx, &mut |_, s| stats.push(s.clone())).await
}

fn build_stream(
    root: Arc<dyn ExecutionPlan>,
    task_ctx: Arc<TaskContext>,
    collector: Arc<Mutex<Vec<(usize, NodeMemoryStats)>>>,
    seq_counter: Arc<AtomicUsize>,
) -> Result<SendableRecordBatchStream> {
    let (cpu_node, batch_size_override) = strip_gpu(root);

    let task_ctx = match batch_size_override {
        Some(n) => with_batch_size(task_ctx, n),
        None => task_ctx,
    };

    let mut stream_children: Vec<Arc<dyn ExecutionPlan>> = Vec::new();
    for child in cpu_node.children() {
        let child_schema = child.schema();
        // Carry the child's equivalence properties (notably its output ordering)
        // into the stub. The data really is ordered — a SortExec produced it —
        // but a stub that reports no ordering makes order-sensitive parents like
        // BoundedWindowAggExec (mode=Sorted) reject their input
        // ("PARTITION BY expression to be ordered").
        let child_eq = child.properties().equivalence_properties().clone();
        let child_stream =
            build_stream(child.clone(), task_ctx.clone(), collector.clone(), seq_counter.clone())?;
        stream_children.push(Arc::new(StreamSourceExec::new(
            child_schema,
            child_eq,
            child_stream,
        )));
    }

    let node_name = cpu_node.name().to_string();
    let node_schema = cpu_node.schema();
    // InterleaveExec requires Hash-partitioned children and rejects the
    // StreamSourceExec stubs (UnknownPartitioning(1)); for that one node UnionExec
    // is a semantically-equivalent single-stream substitute. Any *other*
    // with_new_children failure is a real error and must propagate (master used
    // `?` here — don't blanket-swallow it).
    let node = match cpu_node.clone().with_new_children(stream_children.clone()) {
        Ok(n) => n,
        Err(_) if cpu_node.as_any().is::<InterleaveExec>() => {
            Arc::new(UnionExec::new(stream_children))
        }
        Err(e) => return Err(e),
    };
    // Assign this node's post-order sequence *after* its children were built
    // (they incremented the counter first), so children always sort before their
    // parent regardless of stream-completion/Drop timing (see I2 / LIMIT case).
    let seq = seq_counter.fetch_add(1, Ordering::Relaxed);
    // Use execute_stream (not execute(0)) so multi-partition nodes (UnionExec,
    // RepartitionExec, …) are coalesced into a single stream instead of
    // silently dropping all partitions but one.
    let inner = execute_stream(node, task_ctx)?;
    Ok(Box::pin(InstrumentedStream::new(
        seq,
        node_name,
        node_schema,
        inner,
        collector,
    )))
}

async fn drain_stream(mut stream: SendableRecordBatchStream) -> Result<Vec<RecordBatch>> {
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
struct StreamSourceExec {
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
    fn new(
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

struct InstrumentedStream {
    seq: usize,
    node_name: String,
    schema: SchemaRef,
    inner: SendableRecordBatchStream,
    allocated_bytes: usize,
    output_bytes: usize,
    row_count: usize,
    max_batch_rows: usize,
    collector: Arc<Mutex<Vec<(usize, NodeMemoryStats)>>>,
    done: bool,
}

impl InstrumentedStream {
    fn new(
        seq: usize,
        node_name: String,
        schema: SchemaRef,
        inner: SendableRecordBatchStream,
        collector: Arc<Mutex<Vec<(usize, NodeMemoryStats)>>>,
    ) -> Self {
        Self {
            seq,
            node_name,
            schema,
            inner,
            allocated_bytes: 0,
            output_bytes: 0,
            row_count: 0,
            max_batch_rows: 0,
            collector,
            done: false,
        }
    }
}

impl InstrumentedStream {
    fn push_stat(&mut self) {
        if !self.done {
            self.done = true;
            self.collector.lock().unwrap().push((
                self.seq,
                NodeMemoryStats {
                    node_name: self.node_name.clone(),
                    allocated_bytes: self.allocated_bytes,
                    output_bytes: self.output_bytes,
                    row_count: self.row_count,
                    max_batch_rows: self.max_batch_rows,
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
        let poll = Pin::new(&mut me.inner).poll_next(cx);
        if let Poll::Ready(item) = &poll {
            match item {
                Some(Ok(batch)) => {
                    me.allocated_bytes += batch_allocated_size(batch);
                    me.output_bytes += batch_logical_size(batch);
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

// ---------------------------------------------------------------------------
// Memory size helpers
// ---------------------------------------------------------------------------

/// Sum of allocated buffer capacities across all columns.
///
/// Uses `get_array_memory_size()` which walks all buffers recursively
/// (validity bitmap + values + offsets + children). Safe upper bound —
/// may over-report for sliced batches or over-allocated builders.
pub fn batch_allocated_size(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|col| col.get_array_memory_size())
        .sum()
}

/// Exact logical byte size of a `RecordBatch`.
///
/// For fixed-width types this is derived from the schema and row count.
/// For variable-width types (`Utf8`, `Binary`, etc.) the offsets buffer is
/// read to get the exact data byte count.  Unknown / nested types contribute 0.
pub fn batch_logical_size(batch: &RecordBatch) -> usize {
    let rows = batch.num_rows();
    batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns().iter())
        .map(|(field, col)| {
            let bitmap_bytes = (rows + 7) / 8;
            let data_bytes = match field.data_type() {
                DataType::Boolean => (rows + 7) / 8,
                DataType::Int8 | DataType::UInt8 => rows,
                DataType::Int16 | DataType::UInt16 => rows * 2,
                DataType::Int32
                | DataType::UInt32
                | DataType::Float32
                | DataType::Date32 => rows * 4,
                DataType::Int64
                | DataType::UInt64
                | DataType::Float64
                | DataType::Date64 => rows * 8,
                DataType::Timestamp(_, _) => rows * 8,
                DataType::Utf8 => {
                    let offset_bytes = (rows + 1) * 4; // i32 offsets
                    let data = col
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .map(|arr| {
                            let offsets = arr.value_offsets();
                            if offsets.is_empty() {
                                0usize
                            } else {
                                (offsets[rows] - offsets[0]) as usize
                            }
                        })
                        .unwrap_or(0);
                    offset_bytes + data
                }
                DataType::LargeUtf8 => {
                    let offset_bytes = (rows + 1) * 8; // i64 offsets
                    let data = col
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .map(|arr| {
                            let offsets = arr.value_offsets();
                            if offsets.is_empty() {
                                0usize
                            } else {
                                (offsets[rows] - offsets[0]) as usize
                            }
                        })
                        .unwrap_or(0);
                    offset_bytes + data
                }
                DataType::Binary => {
                    let offset_bytes = (rows + 1) * 4;
                    let data = col
                        .as_any()
                        .downcast_ref::<BinaryArray>()
                        .map(|arr| {
                            let offsets = arr.value_offsets();
                            if offsets.is_empty() {
                                0usize
                            } else {
                                (offsets[rows] - offsets[0]) as usize
                            }
                        })
                        .unwrap_or(0);
                    offset_bytes + data
                }
                DataType::LargeBinary => {
                    let offset_bytes = (rows + 1) * 8;
                    let data = col
                        .as_any()
                        .downcast_ref::<LargeBinaryArray>()
                        .map(|arr| {
                            let offsets = arr.value_offsets();
                            if offsets.is_empty() {
                                0usize
                            } else {
                                (offsets[rows] - offsets[0]) as usize
                            }
                        })
                        .unwrap_or(0);
                    offset_bytes + data
                }
                DataType::Decimal128(_, _) => rows * 16,
                DataType::Decimal256(_, _) => rows * 32,
                DataType::FixedSizeBinary(n) => rows * (*n as usize),
                // View layouts: count CONTENT (Σ value byte lengths + offsets),
                // mirroring the Utf8/Binary arms. Must NOT use get_array_memory_size
                // here — that's allocation-dependent (buffer capacity) and varies
                // run-to-run, making the goldens non-deterministic.
                DataType::Utf8View => {
                    let data = col
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .map(|a| (0..a.len()).filter(|&i| a.is_valid(i)).map(|i| a.value(i).len()).sum::<usize>())
                        .unwrap_or(0);
                    (rows + 1) * 4 + data
                }
                DataType::BinaryView => {
                    let data = col
                        .as_any()
                        .downcast_ref::<BinaryViewArray>()
                        .map(|a| (0..a.len()).filter(|&i| a.is_valid(i)).map(|i| a.value(i).len()).sum::<usize>())
                        .unwrap_or(0);
                    (rows + 1) * 4 + data
                }
                // Dictionary: count the keys deterministically (rows × key width).
                // Values are deduped/small; omitting them slightly undercounts but
                // keeps the golden deterministic (no allocation-size dependency).
                DataType::Dictionary(key_type, _) => rows * key_type.primitive_width().unwrap_or(4),
                // HARD fail on an unhandled type (in debug AND release): the old
                // silent 0 undercounted decimals/Utf8View, and an allocation-based
                // fallback (get_array_memory_size) would make goldens
                // non-deterministic. Panicking forces a deterministic per-type arm to
                // be added rather than silently producing a wrong/unstable size.
                other => panic!("batch_logical_size: unhandled DataType {other:?} — add a deterministic arm"),
            };
            bitmap_bytes + data_bytes
        })
        .sum()
}

// Tests live in tests/test_cpu_executor.rs