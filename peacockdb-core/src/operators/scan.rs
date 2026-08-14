//! Scan family: the ParquetExec wrapper carrying the memory-budget batch size
//! and the row-group -> partition map.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};

#[allow(unused_imports)]
use super::{gpu_exec_node, GpuExtraDisplay};
#[allow(unused_imports)]
use super::operator::{Operator, PartitionTopology};
use datafusion::datasource::physical_plan::ParquetExec;


// ---------------------------------------------------------------------------
// GpuScanExec — wraps ParquetExec to override batch_size at execution time
// ---------------------------------------------------------------------------

/// One entry of the scan's explicit row-group→batch→partition MAP:
/// a group of WHOLE row groups read as one batch, landing in output `partition`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanBatchMap {
    pub row_groups: Vec<u32>,
    pub partition: u32,
}

/// Build the RG→batch→partition map: split the row groups to read
/// (`rgs`) into `n_parts` CONTIGUOUS chunks, one batch each, chunk p → partition p.
/// Deterministic + reader-independent (explicit RG indices). At a large budget each
/// partition is one batch (>per-partition-budget multi-batch splitting is deferred).
/// Returns EMPTY (= legacy single-partition) when there's nothing to split N-way
/// (n_parts<=1, no RGs, or fewer RGs than would yield >1 partition) — so tp1 and
/// tiny single-RG scans stay byte-identical.
pub fn build_scan_map(rgs: &[u32], n_parts: usize) -> Vec<ScanBatchMap> {
    if rgs.is_empty() || n_parts <= 1 {
        return Vec::new();
    }
    let n = n_parts.min(rgs.len());
    if n <= 1 {
        return Vec::new();
    }
    let per = rgs.len() / n;
    let rem = rgs.len() % n;
    let mut map = Vec::with_capacity(n);
    let mut idx = 0usize;
    for p in 0..n {
        let cnt = per + if p < rem { 1 } else { 0 };
        map.push(ScanBatchMap { row_groups: rgs[idx..idx + cnt].to_vec(), partition: p as u32 });
        idx += cnt;
    }
    map
}

#[derive(Debug)]
pub struct GpuScanExec {
    inner: Arc<dyn ExecutionPlan>,
    pub gpu_batch_size: usize,
    /// Explicit surviving row-group override. Set ONLY when reconstructed from a
    /// flatbuffer (deserialize), so re-serialization emits the SAME indices and the
    /// serialize -> deserialize -> serialize bytes stay equal (the reconstructed
    /// ParquetExec has no predicate to recompute from). None for a fresh plan, where
    /// the serializer computes survivors from the ParquetExec pushdown predicate.
    row_groups: Option<Vec<u32>>,
    /// Explicit RG→batch→partition MAP. EMPTY = legacy
    /// single-partition read (tp1, byte-unchanged). When set, the scan emits one
    /// batch per entry into its `partition` — both the CPU golden generator and
    /// the GPU replay this identical map. Set on a deserialized plan (carried
    /// verbatim) or by the tp8 partitioning optimizer step.
    batches: Vec<ScanBatchMap>,
}

impl GpuScanExec {
    pub fn new(inner: Arc<dyn ExecutionPlan>, gpu_batch_size: usize) -> Self {
        Self {
            inner,
            gpu_batch_size,
            row_groups: None,
            batches: Vec::new(),
        }
    }
    /// Reconstruction constructor (deserialize): carries the stored row-group override.
    pub fn with_row_groups(
        inner: Arc<dyn ExecutionPlan>,
        gpu_batch_size: usize,
        row_groups: Option<Vec<u32>>,
    ) -> Self {
        Self {
            inner,
            gpu_batch_size,
            row_groups,
            batches: Vec::new(),
        }
    }
    /// Builder: attach the explicit RG→batch→partition map (deserialize / optimizer).
    pub fn with_batches(mut self, batches: Vec<ScanBatchMap>) -> Self {
        self.batches = batches;
        self
    }
    pub fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    /// Explicit survivor override (Some only on a deserialized plan); None => the
    /// serializer computes survivors from the predicate.
    pub fn row_groups_override(&self) -> Option<&Vec<u32>> {
        self.row_groups.as_ref()
    }
    /// The explicit RG→batch→partition map (empty = legacy single-partition).
    pub fn batches_map(&self) -> &[ScanBatchMap] {
        &self.batches
    }
}

/// Scanned table name for a parquet scan = its file stem
/// (`.../lineitem.parquet` -> `lineitem`).
pub fn parquet_table_name(parquet: &ParquetExec) -> Option<String> {
    let file = parquet.base_config().file_groups.first()?.first()?;
    file.object_meta
        .location
        .to_string()
        .rsplit('/')
        .next()?
        .strip_suffix(".parquet")
        .map(String::from)
}

// Scan annotation flows through GpuExtraDisplay so the `.txt` plan goldens and the
// `.cpu.txt` cost tree label the scan identically (one shared source). We surface
// table + projections (both round-trip through serialization); the pushed-down
// parquet predicate is deliberately NOT shown here because CudfScan serialization
// doesn't carry it, so it would break the flatbuffer roundtrip's plan_str equality
// after deserialize.
//
// `batch_size` is deliberately NOT displayed either. It is budget-derived
// (`gpu_memory_budget / subtree_max_row_bytes`), so rendering it pinned every golden
// to a `MemoryLimit` tier value — yet it is vestigial for everything a golden
// records: it only sets the parquet reader's chunk size, never WHICH rows are read
// (the row-group→partition map comes from `build_scan_map(rgs, n_parts)`, which the
// budget is not an input to), and each golden quantity is a per-node aggregate over
// the full stream, so chunking cancels. It is still SERIALIZED — this is display-only.
impl GpuExtraDisplay for GpuScanExec {
    fn extra_display_info(&self) -> String {
        let Some(parquet) = self.inner.as_any().downcast_ref::<ParquetExec>() else {
            return String::new();
        };
        let config = parquet.base_config();
        let mut parts = Vec::new();
        if let Some(t) = parquet_table_name(parquet) {
            parts.push(format!("table={t}"));
        }
        let names: Vec<String> = match &config.projection {
            Some(p) => p.iter().map(|&i| config.file_schema.field(i).name().clone()).collect(),
            None => config.file_schema.fields().iter().map(|f| f.name().clone()).collect(),
        };
        parts.push(format!("projections=[{}]", names.join(", ")));
        parts.join(", ")
    }
}

impl DisplayAs for GpuScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        // Same empty-extra handling as `gpu_exec_node!`, so a non-ParquetExec inner
        // (now the only way extra can be empty) renders without a dangling colon.
        let extra = self.extra_display_info();
        if extra.is_empty() {
            write!(f, "GpuScanExec")
        } else {
            write!(f, "GpuScanExec: {extra}")
        }
    }
}

impl ExecutionPlan for GpuScanExec {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
    fn properties(&self) -> &PlanProperties {
        self.inner.properties()
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inner.children()
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let new_inner = self.inner.clone().with_new_children(children)?;
        Ok(Arc::new(Self::new(new_inner, self.gpu_batch_size)))
    }
    fn name(&self) -> &str {
        "GpuScanExec"
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let new_config = context
            .session_config()
            .clone()
            .with_batch_size(self.gpu_batch_size);
        let new_ctx = Arc::new(TaskContext::new(
            context.task_id(),
            context.session_id(),
            new_config,
            context.scalar_functions().clone(),
            context.aggregate_functions().clone(),
            context.window_functions().clone(),
            context.runtime_env(),
        ));
        self.inner.execute(partition, new_ctx)
    }
}


// ---------------------------------------------------------------------------
// FlatBuffer wire format
//
// STATEMENT ORDER IS THE WIRE FORMAT. FlatBufferBuilder is a no-interning bump
// arena, so every builder call appends and returns an offset — reordering the
// statements below, or hoisting a create_string, changes the bytes even though the
// values are identical. Do not "tidy" these bodies. testdata/goldens/plan_bytes.sha256
// pins them; the C++ side reads what they emit.
// ---------------------------------------------------------------------------

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::generated::gpu_plan_generated::peacock::plan as fb;
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::listing::PartitionedFile;
use crate::plan_serializer::deserialize_schema;
use crate::plan_serializer::serialize_schema;

pub(crate) fn serialize_cudf_scan<'a>(
    b: &mut FlatBufferBuilder<'a>,
    scan: &GpuScanExec,
) -> Result<(fb::PlanNodeKind, WIPOffset<flatbuffers::UnionWIPOffset>), String> {
    // The inner plan is a ParquetExec.
    let inner = scan.inner();
    let parquet = inner
        .as_any()
        .downcast_ref::<ParquetExec>()
        .ok_or_else(|| "GpuScanExec inner is not ParquetExec".to_string())?;

    let config = parquet.base_config();

    // The wire format requires absolute filesystem paths. `object_meta.location`
    // is an `object_store::path::Path`, which strips the leading `/` during
    // normalization, so we re-add it here. ListingTableUrl canonicalizes to
    // absolute at registration time, so the original input was always absolute
    // by the time we get here.
    // File-path emission is gated on whether this scan carries the explicit
    // RG→batch→partition map — `batches_map()` is the single source of truth (the
    // real-partitioning gate in GpuMemoryBudgetRule decides map presence; map presence alone
    // decides dedup here — no second, divergent notion of "partitioned").
    //
    // MAP PRESENT (real-partitioning device, e.g. tp8-standard): the peacock model
    // reads WHOLE row groups from each DISTINCT physical file per the map, not byte
    // ranges. DataFusion splits ONE file into several byte-RANGE PartitionedFile
    // entries (all the same path) at target_partitions>1; collapse them to distinct
    // paths, else cuDF's source_info sees N identical sources but receives a single
    // row-group vector ("Must specify row groups for each source"). Dedup preserves
    // first-seen order; genuine multi-file scans keep all distinct paths.
    //
    // NO MAP (tp1, or the tight tp8-mini determinism device): emit EVERY
    // PartitionedFile verbatim — byte-identical to the legacy serialization — so the deserialized scan reconstructs the SAME file_group
    // count and the flatbuffer roundtrip's GpuRepartitionExec.input_partitions
    // stays stable (no 8↔1 flip).
    let dedup = !scan.batches_map().is_empty();
    let mut path_strings: Vec<String> = Vec::new();
    for group in &config.file_groups {
        for pf in group {
            let p = format!("/{}", pf.object_meta.location);
            if !dedup || !path_strings.contains(&p) {
                path_strings.push(p);
            }
        }
    }
    let paths: Vec<_> = path_strings.iter().map(|s| b.create_string(s)).collect();
    let file_paths = b.create_vector(&paths);

    let file_schema = serialize_schema(b, &config.file_schema);

    let projection = config.projection.as_ref().map(|proj| {
        let indices: Vec<u32> = proj.iter().map(|&i| i as u32).collect();
        b.create_vector(&indices)
    });

    let limit = config.limit.unwrap_or(0) as u64;

    // Surviving row groups under the static pushdown predicate (same PruningPredicate
    // DataFusion uses for the CPU path -> identical set -> parity with the .cpu.txt
    // oracle). A deserialized scan carries the override verbatim (its reconstructed
    // ParquetExec has no predicate to recompute from) so the roundtrip stays
    // byte-stable; a fresh plan computes from the predicate. None/empty for the
    // no-predicate / multi-file / #16-dynamic cases -> read all.
    let survivors = scan
        .row_groups_override()
        .cloned()
        .or_else(|| crate::gpu_rowgroup_prune::surviving_row_groups(parquet));
    let row_groups = survivors
        .filter(|v| !v.is_empty())
        .map(|v| b.create_vector(&v));

    // Explicit RG→batch→partition map. Empty = legacy
    // single-partition read (tp1, byte-unchanged: None adds no bytes).
    let batches = if scan.batches_map().is_empty() {
        None
    } else {
        let offsets: Vec<_> = scan
            .batches_map()
            .iter()
            .map(|sb| {
                let rgs = b.create_vector(&sb.row_groups);
                fb::ScanBatch::create(
                    b,
                    &fb::ScanBatchArgs { row_groups: Some(rgs), partition: sb.partition },
                )
            })
            .collect();
        Some(b.create_vector(&offsets))
    };

    let cudf_scan = fb::CudfScan::create(
        b,
        &fb::CudfScanArgs {
            file_paths: Some(file_paths),
            file_schema: Some(file_schema),
            projection,
            batch_size: scan.gpu_batch_size as u32,
            limit,
            row_groups,
            batches,
        },
    );

    Ok((fb::PlanNodeKind::CudfScan, cudf_scan.as_union_value()))
}

// ---------------------------------------------------------------------------
// Wire format, READ side. Co-located with the writer above ON PURPOSE: the two are
// one contract, and the round-trip identity this file's twin halves guarantee is
// easy to break by editing one side alone. Keep them together.
// ---------------------------------------------------------------------------

pub(crate) fn deserialize_cudf_scan(
    scan: &fb::CudfScan,
    node: &fb::PlanNode,
) -> Result<Arc<dyn ExecutionPlan>, String> {
    let file_schema = node
        .output_schema()
        .map(|s| deserialize_schema(&s))
        .unwrap_or_else(|| {
            scan.file_schema()
                .map(|s| deserialize_schema(&s))
                .unwrap_or_else(|| Arc::new(datafusion::arrow::datatypes::Schema::empty()))
        });

    let full_schema = scan
        .file_schema()
        .map(|s| deserialize_schema(&s))
        .unwrap_or_else(|| file_schema.clone());

    let file_groups: Vec<Vec<PartitionedFile>> = scan
        .file_paths()
        .map(|v| {
            (0..v.len())
                .map(|i| {
                    let path = v.get(i);
                    vec![PartitionedFile::new(path.to_string(), 0)]
                })
                .collect()
        })
        .unwrap_or_default();

    let projection = scan.projection().map(|v| {
        (0..v.len()).map(|i| v.get(i) as usize).collect::<Vec<_>>()
    });

    let limit = if scan.limit() > 0 {
        Some(scan.limit() as usize)
    } else {
        None
    };

    let config = FileScanConfig::new(
        datafusion::execution::object_store::ObjectStoreUrl::local_filesystem(),
        full_schema,
    )
    .with_file_groups(file_groups)
    .with_projection(projection)
    .with_limit(limit);

    let parquet = ParquetExec::builder(config).build_arc();

    // Carry the surviving row groups verbatim so re-serialization reproduces them
    // (the reconstructed ParquetExec has no predicate to recompute from).
    let row_groups: Option<Vec<u32>> = scan
        .row_groups()
        .map(|v| (0..v.len()).map(|i| v.get(i)).collect::<Vec<u32>>())
        .filter(|v| !v.is_empty());

    // Carry the explicit RG→batch→partition map verbatim.
    let batches: Vec<crate::gpu_rule::ScanBatchMap> = scan
        .batches()
        .map(|v| {
            (0..v.len())
                .map(|i| {
                    let sb = v.get(i);
                    let rgs = sb
                        .row_groups()
                        .map(|r| (0..r.len()).map(|j| r.get(j)).collect())
                        .unwrap_or_default();
                    crate::gpu_rule::ScanBatchMap { row_groups: rgs, partition: sb.partition() }
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(Arc::new(
        GpuScanExec::with_row_groups(parquet, scan.batch_size() as usize, row_groups)
            .with_batches(batches),
    ))
}


// --- Operator: partition topology + strip behavior ------------------------

impl Operator for GpuScanExec {
    fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
    fn partition_topology(&self) -> PartitionTopology {
        PartitionTopology::ScanEmit
    }
}
