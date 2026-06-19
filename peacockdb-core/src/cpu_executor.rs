use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use datafusion::arrow::array::{
    Array, BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray, ListArray, StringArray,
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
    GpuHashJoinExec, GpuInterleaveExec, GpuProjectExec, GpuRepartitionExec, GpuScanExec,
    GpuSortExec, GpuSortPreservingMergeExec,
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
    // Strip to the bare InterleaveExec so build_stream's substitution sees it:
    // with single-partition stream stubs InterleaveExec::try_new can't interleave,
    // so it's rebuilt as a (semantically-equivalent) UnionExec. Without stripping,
    // the wrapper's with_new_children surfaces that error and it isn't recognised.
    try_strip!(GpuInterleaveExec);

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
    // Per-column accumulators (one per output field). Each level's bitmap/offset
    // overhead is charged ONCE at finalize from the accumulated totals, so
    // `output_bytes` is deterministic regardless of how rows are chunked into
    // batches (i.e. regardless of target_partitions). See `ColAccum`.
    cols: Vec<ColAccum>,
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
                    // Logical size of the whole node output: each column's (and
                    // nested level's) overhead charged ONCE from the accumulated
                    // totals — independent of batch boundaries.
                    output_bytes: self
                        .cols
                        .iter()
                        .zip(self.schema.fields())
                        .map(|(c, f)| c.size(f.data_type()))
                        .sum(),
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

/// Per-column STRUCTURAL byte size: the part that depends only on the column
/// type and the row count, NOT on how rows are split into batches — the
/// validity bitmap plus either the fixed-width data buffer or the var-length
/// OFFSET buffer. This is the single source of truth for per-type widths.
///
/// Because it is batch-independent it can be evaluated once per node from the
/// total row count, which is what makes `output_bytes` deterministic at
/// `target_partitions > 1` (the per-batch overhead — bitmap rounding + the
/// offset buffer's `+1` — was the only thing that wobbled with batch boundaries).
fn type_structural_size(dt: &DataType, rows: usize) -> usize {
    let bitmap_bytes = (rows + 7) / 8;
    let data_bytes = match dt {
        DataType::Boolean => (rows + 7) / 8,
        DataType::Int8 | DataType::UInt8 => rows,
        DataType::Int16 | DataType::UInt16 => rows * 2,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => rows * 4,
        DataType::Int64 | DataType::UInt64 | DataType::Float64 | DataType::Date64 => rows * 8,
        DataType::Timestamp(_, _) => rows * 8,
        // Var-length: only the offset buffer is structural; the content is
        // accumulated separately (see `array_content_size`). View layouts also
        // carry an (rows+1)*4 offset-equivalent, mirroring the old formula.
        DataType::Utf8 | DataType::Binary | DataType::Utf8View | DataType::BinaryView => {
            (rows + 1) * 4 // i32 offsets
        }
        DataType::LargeUtf8 | DataType::LargeBinary => (rows + 1) * 8, // i64 offsets
        DataType::Decimal128(_, _) => rows * 16,
        DataType::Decimal256(_, _) => rows * 32,
        DataType::FixedSizeBinary(n) => rows * (*n as usize),
        // Dictionary: count the keys deterministically (rows × key width).
        // Values are deduped/small; omitting them slightly undercounts but
        // keeps the golden deterministic (no allocation-size dependency).
        DataType::Dictionary(key_type, _) => rows * key_type.primitive_width().unwrap_or(4),
        // Nested types are NOT handled here — `ColAccum` computes them from
        // per-level totals (List child overhead can't be derived from the parent
        // row count alone). `assert_type_accountable` recurses into them.
        //
        // HARD fail on any other unhandled type: the old silent 0 undercounted
        // decimals/Utf8View, and an allocation-based fallback
        // (get_array_memory_size) would make goldens non-deterministic. Panicking
        // forces a deterministic per-type arm to be added rather than silently
        // producing a wrong/unstable size. The guard is reached at stream
        // construction (see `assert_type_accountable`), NOT in a destructor, so it
        // unwinds as a normal test failure instead of aborting the process.
        other => panic!("type_structural_size: unhandled DataType {other:?} — add a deterministic arm"),
    };
    bitmap_bytes + data_bytes
}

/// Per-column var-length CONTENT bytes for one batch: `offsets[rows]-offsets[0]`
/// for offset layouts, or Σ value byte lengths for View layouts. Fixed-width
/// types contribute 0. This term telescopes across batches (the sum over batches
/// equals the value for the whole node), so it carries NO per-batch overhead and
/// is safe to accumulate as batches arrive.
fn array_content_size(dt: &DataType, col: &dyn Array, rows: usize) -> usize {
    // offsets[rows]-offsets[0]; offsets are i32 (Utf8/Binary) or i64 (Large*).
    macro_rules! offset_content {
        ($arr:ty) => {
            col.as_any()
                .downcast_ref::<$arr>()
                .map(|a| {
                    let o = a.value_offsets();
                    if o.is_empty() {
                        0usize
                    } else {
                        (o[rows] - o[0]) as usize
                    }
                })
                .unwrap_or(0)
        };
    }
    match dt {
        DataType::Utf8 => offset_content!(StringArray),
        DataType::LargeUtf8 => offset_content!(LargeStringArray),
        DataType::Binary => offset_content!(BinaryArray),
        DataType::LargeBinary => offset_content!(LargeBinaryArray),
        // View layouts: Σ value byte lengths. Must NOT use get_array_memory_size
        // here — that's allocation-dependent (buffer capacity) and varies
        // run-to-run, making the goldens non-deterministic.
        DataType::Utf8View => col
            .as_any()
            .downcast_ref::<StringViewArray>()
            .map(|a| (0..a.len()).filter(|&i| a.is_valid(i)).map(|i| a.value(i).len()).sum())
            .unwrap_or(0),
        DataType::BinaryView => col
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .map(|a| (0..a.len()).filter(|&i| a.is_valid(i)).map(|i| a.value(i).len()).sum())
            .unwrap_or(0),
        _ => 0,
    }
}

/// Per-column accumulator that makes `output_bytes` deterministic for nested
/// (`List`) columns too, not just flat ones.
///
/// The wobble being removed is per-batch OVERHEAD (validity bitmap rounding +
/// the offset buffer's `+1`) double-counted across batch boundaries. For flat
/// columns the level total is just the row count, so the overhead can be
/// computed once from the schema. For a `List`, the CHILD level's element count
/// is data-dependent and is NOT a function of the parent row count — so we must
/// accumulate it. `ColAccum` mirrors the array's nesting, summing each level's
/// element `count` and the leaf var-length `content` bytes across all batches;
/// `size` then charges every level's bitmap/offset overhead ONCE from its total.
/// Counts and content are order-independent sums, so the result is identical
/// regardless of how the coalesced stream chunks rows into batches.
#[derive(Clone, Default)]
struct ColAccum {
    count: usize,            // total elements at this level across all batches
    content: usize,          // leaf var-length content bytes (telescopes)
    children: Vec<ColAccum>, // sub-array accumulators (List child)
}

impl ColAccum {
    fn child(&mut self) -> &mut ColAccum {
        if self.children.is_empty() {
            self.children.push(ColAccum::default());
        }
        &mut self.children[0]
    }

    /// Fold one batch's array (for this column) into the running totals.
    fn add(&mut self, array: &dyn Array) {
        let len = array.len();
        self.count += len;
        match array.data_type() {
            DataType::List(_) => {
                let la = array.as_any().downcast_ref::<ListArray>().unwrap();
                let o = la.value_offsets();
                // Slice the child to just the range these rows reference (the
                // array may itself be a slice, so start at o[0] not 0).
                let (start, end) = (o[0] as usize, o[len] as usize);
                let child = la.values().slice(start, end - start);
                self.child().add(child.as_ref());
            }
            dt => self.content += array_content_size(dt, array, len),
        }
    }

    /// Logical size of this whole accumulated level: bitmap + (offset|fixed) +
    /// content / child, all charged ONCE from the accumulated totals.
    fn size(&self, dt: &DataType) -> usize {
        match dt {
            DataType::List(field) => {
                let bitmap = (self.count + 7) / 8;
                let offsets = (self.count + 1) * 4; // i32 offsets
                bitmap + offsets + self.children.first().map_or(0, |c| c.size(field.data_type()))
            }
            // Flat: type_structural_size already counts bitmap + fixed/offset; add
            // the accumulated var-length content (0 for fixed-width types).
            flat => type_structural_size(flat, self.count) + self.content,
        }
    }
}

/// Fail at stream CONSTRUCTION (not in `Drop`) if a column type has no
/// deterministic accounting, recursing into `List` children. Calling the guard
/// here means an unhandled type unwinds as a normal test failure rather than
/// aborting the process from inside `InstrumentedStream`'s destructor.
fn assert_type_accountable(dt: &DataType) {
    match dt {
        DataType::List(field) => assert_type_accountable(field.data_type()),
        other => {
            let _ = type_structural_size(other, 0); // panics here if unhandled
        }
    }
}

/// Exact logical byte size of a single `RecordBatch` (structural + content).
///
/// Note: the per-node `output_bytes` metric is NOT this summed per batch — it is
/// a [`ColAccum`] over the whole node output, so each level's overhead is charged
/// once and the value does not depend on batch boundaries. This helper is
/// retained for callers that genuinely want a single batch's size.
pub fn batch_logical_size(batch: &RecordBatch) -> usize {
    batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns().iter())
        .map(|(field, col)| {
            let mut acc = ColAccum::default();
            acc.add(col.as_ref());
            acc.size(field.data_type())
        })
        .sum()
}

// Tests live in tests/test_cpu_executor.rs