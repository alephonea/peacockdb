//! CPU-side memory accounting: how many bytes a batch/schema actually costs.
//!
//! Single source of truth for node `output_bytes` on BOTH backends — the GPU
//! backend reconstructs its stats through `logical_size_from_schema` — so CPU and
//! GPU costs are identical by construction whenever per-node row counts match.

use datafusion::arrow::array::{
    Array, BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray, ListArray, StringArray,
    StringViewArray,
};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::arrow::record_batch::RecordBatch;

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
pub(crate) fn type_structural_size(dt: &DataType, rows: usize) -> usize {
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

/// Logical `output_bytes` for a node from its output schema, total row count, and
/// the Σ var-length CONTENT bytes (the data-dependent term). The ColAccum
/// metric reconstructed from rows+schema+content — the SINGLE source of the
/// byte-accounting overhead (validity bitmap + fixed-width + var-length offset
/// buffers). The GPU node-executor calls this with the content bytes measured by
/// C++, so CPU-emulated and GPU costs are identical whenever rows (and content)
/// match. (Flat columns only — nested `List` appears at tp>1 two-phase aggregation
/// and is handled by `ColAccum`.)
pub fn logical_size_from_schema(schema: &Schema, rows: usize, varlen_content_bytes: usize) -> usize {
    schema
        .fields()
        .iter()
        .map(|f| type_structural_size(f.data_type(), rows))
        .sum::<usize>()
        + varlen_content_bytes
}

/// Σ var-length CONTENT bytes across all columns of one batch — the data-dependent
/// term of [`logical_size_from_schema`]. Used by the CPU hash-repartition to
/// compute each output partition's `output_bytes` identically to the map-op
/// `ColAccum` path (and to the GPU's per-partition `varlen_content_bytes`). Flat
/// columns only (the repartition input is post-partial-agg group keys + additive
/// scalar state — no nested `List`).
pub fn batch_varlen_content_bytes(batch: &RecordBatch) -> usize {
    let schema = batch.schema();
    let rows = batch.num_rows();
    (0..schema.fields().len())
        .map(|i| array_content_size(schema.field(i).data_type(), batch.column(i).as_ref(), rows))
        .sum()
}

/// Per-column var-length CONTENT bytes for one batch: `offsets[rows]-offsets[0]`
/// for offset layouts, or Σ value byte lengths for View layouts. Fixed-width
/// types contribute 0. This term telescopes across batches (the sum over batches
/// equals the value for the whole node), so it carries NO per-batch overhead and
/// is safe to accumulate as batches arrive.
pub(crate) fn array_content_size(dt: &DataType, col: &dyn Array, rows: usize) -> usize {
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
pub(crate) struct ColAccum {
    count: usize,            // total elements at this level across all batches
    content: usize,          // leaf var-length content bytes (telescopes)
    children: Vec<ColAccum>, // sub-array accumulators (List child)
}

impl ColAccum {
    pub(crate) fn child(&mut self) -> &mut ColAccum {
        if self.children.is_empty() {
            self.children.push(ColAccum::default());
        }
        &mut self.children[0]
    }

    /// Fold one batch's array (for this column) into the running totals.
    pub(crate) fn add(&mut self, array: &dyn Array) {
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
    pub(crate) fn size(&self, dt: &DataType) -> usize {
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
pub(crate) fn assert_type_accountable(dt: &DataType) {
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
