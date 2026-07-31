//! CPU backend for the node-by-node driver: the DataFusion oracle.
//!
//! Handles are `Vec<RecordBatch>` in a local registry; stats come from the Part-1
//! `ColAccum` over the actual batches. Also owns the N-partition machinery (scan
//! partitioning, Spark-murmur3 hash repartition, partitioned join arity) — those
//! are backend internals, not mode configuration, so they live here rather than in
//! [`super::super::partitioned_cpu_executor`], which only picks the config.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, UInt32Array};
use datafusion::arrow::compute::{cast, concat_batches, take};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::physical_plan::parquet::{ParquetAccessPlan, RowGroupAccess};
use datafusion::datasource::physical_plan::ParquetExec;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::memory::MemoryExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{execute_stream, ExecutionPlan, Partitioning, PhysicalExpr};
use datafusion_comet_spark_expr::hash_funcs::murmur3::create_murmur3_hashes;

use crate::cpu_executor::{batch_varlen_content_bytes, logical_size_from_schema};
use crate::gpu_rowgroup_prune::all_row_groups;
use crate::gpu_rule::{
    GpuCoalescePartitionsExec, GpuHashJoinExec, GpuRepartitionExec, GpuScanExec,
    GpuSortPreservingMergeExec,
};

use crate::executors::executor::{NodeMemoryStats, PartitionStat};
use crate::executors::node_by_node::NodeExecutor;
use crate::executors::single_node::{execute_single_node, merge_stats};

/// Spark HashPartitioning seed (comet + the GPU kernel both init the hash to 42).
const SPARK_HASH_SEED: u32 = 42;

/// Spark `pmod` (positive modulo): map a signed murmur3 hash into `[0, n)`. MUST
/// match the GPU kernel's `pmod` exactly (negative hashes wrap identically), else
/// per-partition row counts diverge from the golden.
fn pmod(h: i32, n: i32) -> i32 {
    ((h % n) + n) % n
}

/// If `node` is a *lowered* Hash `GpuRepartitionExec` (the `1→N` form produced by the
/// GpuMemoryBudgetRule under RealMultiPartition — its input is a single partition, a
/// GpuCoalescePartitions), return its (hash key exprs, N). The CPU executor then
/// hash-partitions via Spark-murmur3 (comet) to match the GPU kernel — NOT
/// DataFusion's ahash-based `RepartitionExec` (whose partition NUMBERS differ).
///
/// Requiring a SINGLE-partition input is what keeps this real-N-way split confined to
/// the lowered shape: an UN-lowered `M→N` Hash repartition (SinglePartition mode, or
/// any plan where the map/lowering didn't fire) must fall through to the generic path
/// (execute_single_node → coalesced 1 output), matching the recursive baseline — else
/// #13's Σ-over-partitions stats diverge from it. RoundRobin is never intercepted.
fn hash_repartition_of(node: &Arc<dyn ExecutionPlan>) -> Option<(Vec<Arc<dyn PhysicalExpr>>, usize)> {
    let gpu_rp = node.as_any().downcast_ref::<GpuRepartitionExec>()?;
    let rp = gpu_rp.inner().as_any().downcast_ref::<RepartitionExec>()?;
    if rp.input().properties().output_partitioning().partition_count() != 1 {
        return None; // un-lowered M→N shuffle → generic (coalesced) path
    }
    match rp.partitioning() {
        Partitioning::Hash(exprs, n) => Some((exprs.clone(), *n)),
        _ => None,
    }
}

/// Partition-collapsing nodes: N input partitions → 1 output (the CPU oracle
/// realizes this by concatenating all child partitions into a single input). Every
/// other single-child node maps over its input partitions (count preserved).
fn collapses_partitions(node: &Arc<dyn ExecutionPlan>) -> bool {
    let any = node.as_any();
    any.is::<GpuCoalescePartitionsExec>()
        || any.is::<GpuSortPreservingMergeExec>()
        || any.is::<CoalescePartitionsExec>()
        || any.is::<SortPreservingMergeExec>()
}

/// (#96) If `node` is a `PartitionMode::Partitioned` `HashJoinExec` (possibly
/// GPU-wrapped) whose two children carry the SAME partition count N>1, return N so
/// the join runs PER-PARTITION (`child0[p] ⋈ child1[p]` for p in 0..N) — matching
/// DataFusion's Partitioned `output_partitioning` (verified: all q17 joins at tp8 are
/// Partitioned 8→8) and the GPU C++ MAP arm. DataFusion's Partitioned join REQUIRES
/// both inputs Hash-partitioned on the join keys, realized here by the lowered
/// `GpuRepartition` feeding each side with the SAME comet-murmur3 hash (the Inc2
/// kernel) — so matching keys (incl. nulls, per the join's own `null_equals_null`)
/// co-locate in bucket p ⇒ the per-partition inner join is complete and ∪ₚ = the full
/// join. Returns None for `CollectLeft` (a tracked latent gap — none in the current
/// flip set), non-joins, unequal-N, or N≤1: all fall through to the collapsed
/// single-partition path (which keeps tp1 byte-identical).
fn partitioned_join_arity(
    node: &Arc<dyn ExecutionPlan>,
    child_parts: &[Vec<Vec<RecordBatch>>],
) -> Option<usize> {
    let inner =
        node.as_any().downcast_ref::<GpuHashJoinExec>().map(|g| g.inner()).unwrap_or(node);
    let join = inner.as_any().downcast_ref::<HashJoinExec>()?;
    if *join.partition_mode() != datafusion::physical_plan::joins::PartitionMode::Partitioned {
        return None;
    }
    if child_parts.len() != 2 {
        return None;
    }
    let n = child_parts[0].len();
    if n <= 1 || child_parts[1].len() != n {
        return None;
    }
    Some(n)
}

/// Execute a `GpuScanExec` carrying a non-empty RG→batch→partition map, returning
/// one materialized partition per map entry plus the Σ-over-partitions stats.
///
/// Each partition restricts the underlying `ParquetExec` to exactly that entry's
/// row groups via a [`ParquetAccessPlan`] on the file's `extensions` — the SAME
/// row-group→partition assignment the GPU replays (cuDF `set_row_groups`), so
/// per-partition row counts match by construction. The scan's predicate + parquet
/// options are preserved, so each partition's rows are identical to what the full
/// DataFusion scan would yield for those groups (peacock RG-prunes but does NOT
/// push row filters into the scan, so all rows of the selected groups are read).
async fn cpu_scan_partitions(
    scan: &GpuScanExec,
    task_ctx: Arc<TaskContext>,
) -> DfResult<(Vec<Vec<RecordBatch>>, NodeMemoryStats)> {
    let parquet = scan.inner().as_any().downcast_ref::<ParquetExec>().ok_or_else(|| {
        DataFusionError::Internal("cpu_scan_partitions: GpuScanExec inner is not a ParquetExec".into())
    })?;
    // Access-plan length MUST equal the file's true row-group count or DataFusion
    // rejects the plan.
    let total_rgs = all_row_groups(parquet).map(|v| v.len()).ok_or_else(|| {
        DataFusionError::Internal(
            "cpu_scan_partitions: cannot determine row-group count for partitioned scan".into(),
        )
    })?;
    let base_file = parquet
        .base_config()
        .file_groups
        .iter()
        .flatten()
        .next()
        .cloned()
        .ok_or_else(|| DataFusionError::Internal("cpu_scan_partitions: scan has no source file".into()))?;

    let mut out_parts: Vec<Vec<RecordBatch>> = Vec::with_capacity(scan.batches_map().len());
    let mut acc: Option<NodeMemoryStats> = None;
    let mut part_stats: Vec<PartitionStat> = Vec::with_capacity(scan.batches_map().len());
    for entry in scan.batches_map() {
        // Scan ONLY this partition's row groups; skip every other group.
        let mut access = ParquetAccessPlan::new(vec![RowGroupAccess::Skip; total_rgs]);
        for &rg in &entry.row_groups {
            access.scan(rg as usize);
        }
        let mut file = base_file.clone();
        // The base file may carry a byte RANGE (DataFusion's tp8 split); clear it so
        // the access plan ALONE decides which row groups this partition reads.
        file.range = None;
        file.extensions = Some(Arc::new(access) as Arc<dyn std::any::Any + Send + Sync>);

        let mut config = parquet.base_config().clone();
        config.file_groups = vec![vec![file]];
        // Preserve the scan predicate per-partition. It only ROW-GROUP-prunes here
        // (pushdown_filters is off → no per-row filtering at the scan), and the map's
        // RGs are already the predicate survivors, so it removes nothing — the
        // partition reads exactly its mapped RGs, matching the GPU's set_row_groups.
        // CPU and GPU agree as long as no SUB-row-group (page) pruning diverges; today
        // it can't (our DuckDB parquet carries no page index / bloom filters).
        let mut builder = ParquetExec::builder(config)
            .with_table_parquet_options(parquet.table_parquet_options().clone());
        if let Some(pred) = parquet.predicate() {
            builder = builder.with_predicate(pred.clone());
        }
        let part_parquet: Arc<dyn ExecutionPlan> = Arc::new(builder.build());
        // Re-wrap so execute_single_node applies the same gpu_batch_size override
        // and records the node as "ParquetExec" (matching the recursive oracle).
        let part_scan: Arc<dyn ExecutionPlan> =
            Arc::new(GpuScanExec::new(part_parquet, scan.gpu_batch_size));
        let (batches, stat) = execute_single_node(&part_scan, vec![], task_ctx.clone()).await?;
        // Per-partition sub-line: this partition's row groups + its own out rows/bytes
        // (the SAME map the GPU replays via set_row_groups → identical by construction).
        part_stats.push(PartitionStat {
            out_rows: stat.row_count,
            out_bytes: stat.output_bytes,
            row_groups: entry.row_groups.clone(),
        });
        out_parts.push(batches);
        merge_stats(&mut acc, stat);
    }
    let mut stat = acc.unwrap_or_else(|| NodeMemoryStats {
        node_name: "ParquetExec".to_string(),
        ..Default::default()
    });
    // Only N>1 carries sub-lines; a 1-entry map renders as partitions=1.
    if part_stats.len() > 1 {
        stat.part_stats = part_stats;
    }
    Ok((out_parts, stat))
}

/// CPU Spark-murmur3 hash-repartition (Inc2): concat the (already coalesced) input
/// into one table, assign each row to `pmod(spark_murmur3(keys, seed=42), n)` via the
/// comet helper — EXACTLY the GPU `peacock::partitioning::spark_hash_partition`
/// kernel (the live conformance gate proves bit-equality) — and scatter rows into
/// `n` output partitions in row order. Count-preserving (Σ out_rows == input rows);
/// the per-partition `out_rows`/`out_bytes` are the load-bearing murmur3-fidelity
/// numbers the golden records and the GPU must reproduce. Uses NOT DataFusion's
/// `RepartitionExec` (ahash → different partition NUMBERS).
fn cpu_hash_repartition(
    node: &Arc<dyn ExecutionPlan>,
    hash_exprs: &[Arc<dyn PhysicalExpr>],
    n_parts: usize,
    input: Vec<RecordBatch>,
) -> DfResult<(Vec<Vec<RecordBatch>>, NodeMemoryStats)> {
    let schema = node.schema();
    // One table — matches the GPU's single cudf::table hash_partition input.
    let batch = concat_batches(&schema, input.iter()).map_err(DataFusionError::from)?;
    let rows = batch.num_rows();

    // Evaluate the hash key columns, then fold them left-to-right with comet's
    // Spark-murmur3 (buffer pre-seeded to 42 — Spark's HashPartitioning seed).
    // comet's hasher rejects the Arrow "view" string/binary layouts (Utf8View/
    // BinaryView) that DataFusion 45's Parquet reader emits; cast those to the
    // canonical offset layout — same bytes ⇒ same Spark hash as the GPU (which
    // hashes the cudf STRING offset layout), so partition assignment still matches.
    let keys: Vec<ArrayRef> = hash_exprs
        .iter()
        .map(|e| {
            let arr = e.evaluate(&batch).and_then(|v| v.into_array(rows))?;
            match arr.data_type() {
                DataType::Utf8View => cast(&arr, &DataType::Utf8).map_err(DataFusionError::from),
                DataType::BinaryView => cast(&arr, &DataType::Binary).map_err(DataFusionError::from),
                _ => Ok(arr),
            }
        })
        .collect::<DfResult<Vec<_>>>()?;
    let mut hashes = vec![SPARK_HASH_SEED; rows];
    if rows > 0 {
        create_murmur3_hashes(&keys, &mut hashes)
            .map_err(|e| DataFusionError::External(format!("comet murmur3: {e}").into()))?;
    }
    let n = n_parts as i32;
    let mut idx: Vec<Vec<u32>> = vec![Vec::new(); n_parts];
    for (r, &h) in hashes.iter().enumerate() {
        idx[pmod(h as i32, n) as usize].push(r as u32);
    }

    let mut out_parts: Vec<Vec<RecordBatch>> = Vec::with_capacity(n_parts);
    let mut part_stats: Vec<PartitionStat> = Vec::with_capacity(n_parts);
    let mut acc = NodeMemoryStats { node_name: node.name().to_string(), ..Default::default() };
    for indices in &idx {
        let take_idx = UInt32Array::from(indices.clone());
        let cols: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|c| take(c.as_ref(), &take_idx, None).map_err(DataFusionError::from))
            .collect::<DfResult<Vec<_>>>()?;
        let part = RecordBatch::try_new(schema.clone(), cols)?;
        let out_rows = part.num_rows();
        // output_bytes via the single-source ColAccum overhead formula = the GPU's
        // per-partition accounting (identical rows ⇒ identical strings ⇒ bytes).
        let out_bytes =
            logical_size_from_schema(&schema, out_rows, batch_varlen_content_bytes(&part));
        acc.row_count += out_rows;
        acc.output_bytes += out_bytes;
        acc.max_batch_rows = acc.max_batch_rows.max(out_rows);
        part_stats.push(PartitionStat { out_rows, out_bytes, row_groups: Vec::new() });
        // Empty partition ⇒ no batch (the mapped-over child stream is just empty).
        out_parts.push(if out_rows == 0 { Vec::new() } else { vec![part] });
    }
    // A Hash repartition always targets N>1 partitions ⇒ always carries sub-lines.
    acc.part_stats = part_stats;
    Ok((out_parts, acc))
}

/// CPU backend: handles are `Vec<RecordBatch>` held in a local registry; each
/// node runs through the same DataFusion machinery as the recursive executor, so
/// its stats are byte-identical to `execute_node_by_node`.
pub struct CpuNodeExecutor {
    task_ctx: Arc<TaskContext>,
    registry: HashMap<u64, Vec<RecordBatch>>,
    next_handle: u64,
}

impl CpuNodeExecutor {
    pub fn new(task_ctx: Arc<TaskContext>) -> Self {
        Self { task_ctx, registry: HashMap::new(), next_handle: 1 }
    }

    fn store(&mut self, batches: Vec<RecordBatch>) -> u64 {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.registry.insert(handle, batches);
        handle
    }
}

impl NodeExecutor for CpuNodeExecutor {
    async fn execute_node(
        &mut self,
        _seq: usize,
        node: &Arc<dyn ExecutionPlan>,
        input_handles: &[Vec<u64>],
    ) -> DfResult<(Vec<u64>, NodeMemoryStats)> {
        // (iii) SCAN with an explicit RG→batch→partition map → N partition handles.
        if let Some(scan) = node.as_any().downcast_ref::<GpuScanExec>() {
            if !scan.batches_map().is_empty() {
                let (parts, stat) = cpu_scan_partitions(scan, self.task_ctx.clone()).await?;
                let handles: Vec<u64> = parts.into_iter().map(|b| self.store(b)).collect();
                return Ok((handles, stat));
            }
        }

        // Materialize each child's partition batches (consuming registry handles).
        let child_parts: Vec<Vec<Vec<RecordBatch>>> = input_handles
            .iter()
            .map(|child| child.iter().map(|h| self.registry.remove(h).unwrap_or_default()).collect())
            .collect();

        // (Inc2) HASH REPARTITION → N partitions via Spark-murmur3 (comet), matching
        // the GPU kernel. The lowering feeds it ONE input partition (a preceding
        // GpuCoalescePartitions concats M→1); we concat whatever we get and scatter.
        // Intercepted BEFORE the generic single-child map, which would otherwise run
        // DataFusion's ahash `RepartitionExec` and coalesce it back to one stream.
        if let Some((hash_exprs, n_parts)) = hash_repartition_of(node) {
            let input: Vec<RecordBatch> =
                child_parts.into_iter().next().unwrap_or_default().into_iter().flatten().collect();
            let (parts, stat) = cpu_hash_repartition(node, &hash_exprs, n_parts, input)?;
            let handles: Vec<u64> = parts.into_iter().map(|b| self.store(b)).collect();
            return Ok((handles, stat));
        }

        // Ordinary single-child op with the multi-partition map active → MAP the
        // node over each input partition (count preserved); the Σ-over-partitions
        // stat falls out of summing per-partition runs. Partition-collapsing ops
        // (CoalescePartitions / SortPreservingMerge) and any multi-/zero-child node
        // fall through to the concat-into-one path — which also covers tp1 (single
        // partition) byte-identically (one partition in → one run → one out).
        if !collapses_partitions(node) && child_parts.len() == 1 && !child_parts[0].is_empty() {
            let mut handles = Vec::with_capacity(child_parts[0].len());
            let mut acc: Option<NodeMemoryStats> = None;
            let mut part_stats: Vec<PartitionStat> = Vec::with_capacity(child_parts[0].len());
            for part in &child_parts[0] {
                let (batches, stat) =
                    execute_single_node(node, vec![part.clone()], self.task_ctx.clone()).await?;
                // Per-partition sub-line: this node's own out rows/bytes for output
                // partition k (no row_groups — non-scan node; in_rows is derived from
                // the child's out_rows by the golden formatter).
                part_stats.push(PartitionStat {
                    out_rows: stat.row_count,
                    out_bytes: stat.output_bytes,
                    row_groups: Vec::new(),
                });
                handles.push(self.store(batches));
                merge_stats(&mut acc, stat);
            }
            let mut acc = acc.expect("non-empty child has at least one partition");
            // Only N>1 carries sub-lines (count-preserving map: N == child N).
            if part_stats.len() > 1 {
                acc.part_stats = part_stats;
            }
            return Ok((handles, acc));
        }

        // (#96) PARTITIONED multi-child JOIN with the multi-partition map active → run
        // the join PER-PARTITION, producing N output partitions (one per co-partitioned
        // bucket), matching DataFusion's Partitioned join + the GPU. Both inputs are
        // Hash-repartitioned on the join key with the same comet-murmur3, so bucket p of
        // each side holds all rows whose join keys hash to p ⇒ child0[p] ⋈ child1[p] is
        // complete. Gated (partitioned_join_arity) on Partitioned mode + equal-N>1;
        // CollectLeft / unequal-N fall through to the collapsed path below.
        if let Some(n) = partitioned_join_arity(node, &child_parts) {
            let mut handles = Vec::with_capacity(n);
            let mut acc: Option<NodeMemoryStats> = None;
            let mut part_stats: Vec<PartitionStat> = Vec::with_capacity(n);
            for p in 0..n {
                let (batches, stat) = execute_single_node(
                    node,
                    vec![child_parts[0][p].clone(), child_parts[1][p].clone()],
                    self.task_ctx.clone(),
                )
                .await?;
                part_stats.push(PartitionStat {
                    out_rows: stat.row_count,
                    out_bytes: stat.output_bytes,
                    row_groups: Vec::new(),
                });
                handles.push(self.store(batches));
                merge_stats(&mut acc, stat);
            }
            let mut acc = acc.expect("partitioned join has N>1 partitions");
            acc.part_stats = part_stats; // N>1 by the gate → always carries sub-lines
            return Ok((handles, acc));
        }

        // (#99) SortPreservingMerge = K-WAY MERGE the child's N SORTED partitions
        // (respecting the SPM's sort keys + fetch), NOT concat. The generic collapse
        // below concats the N partitions into ONE and runs SPM on it — but SPM merges
        // N input partitions, so on 1 it's a NO-OP: the output is only per-partition
        // sorted, and a downstream LIMIT then picks the wrong global top-N. Feed a
        // MemoryExec of the N partitions to the inner SortPreservingMergeExec so its
        // real merge + fetch run. (N==1 falls through to concat = byte-identical.)
        if let Some(spm) = node.as_any().downcast_ref::<GpuSortPreservingMergeExec>() {
            if child_parts.len() == 1 && child_parts[0].len() > 1 {
                let parts = child_parts.into_iter().next().unwrap();
                let inner = spm.inner().clone();
                let child_schema = inner.children()[0].schema();
                let mem = Arc::new(MemoryExec::try_new(&parts, child_schema, None)?);
                let merged = inner.with_new_children(vec![mem])?;
                let stream = execute_stream(merged, self.task_ctx.clone())?;
                let batches: Vec<RecordBatch> = {
                    use futures::TryStreamExt;
                    stream.try_collect().await?
                };
                let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                let varlen: usize = batches.iter().map(batch_varlen_content_bytes).sum();
                let stat = NodeMemoryStats {
                    node_name: node.name().to_string(),
                    output_bytes: logical_size_from_schema(node.schema().as_ref(), rows, varlen),
                    row_count: rows,
                    max_batch_rows: batches.iter().map(|b| b.num_rows()).max().unwrap_or(0),
                    ..Default::default()
                };
                let handle = self.store(batches);
                return Ok((vec![handle], stat));
            }
        }

        // Concat-into-one: each child's partitions concatenated into a single input,
        // the node runs once → one output partition.
        let inputs: Vec<Vec<RecordBatch>> =
            child_parts.into_iter().map(|child| child.into_iter().flatten().collect()).collect();
        let (batches, stat) = execute_single_node(node, inputs, self.task_ctx.clone()).await?;
        let handle = self.store(batches);
        Ok((vec![handle], stat))
    }

    async fn materialize(&mut self, handles: &[u64]) -> DfResult<Vec<RecordBatch>> {
        let mut out = Vec::new();
        for h in handles {
            out.extend(self.registry.remove(h).unwrap_or_default());
        }
        Ok(out)
    }

    fn release(&mut self, handles: &[u64]) {
        for h in handles {
            self.registry.remove(h);
        }
    }
}
