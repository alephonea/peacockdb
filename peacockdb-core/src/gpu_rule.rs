use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::Result;
use datafusion::datasource::physical_plan::ParquetExec;
use datafusion::execution::TaskContext;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::utils::JoinFilter;
use datafusion::physical_plan::joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::union::{InterleaveExec, UnionExec};
use datafusion::physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, InListExpr, NotExpr};
use datafusion::logical_expr::Operator;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};

// ---------------------------------------------------------------------------
// GPU exec node stubs (delegate to inner CPU node)
// ---------------------------------------------------------------------------

/// Optional extra display info appended after the node name in plan output.
/// Implement with a non-empty string to annotate a specific GPU node type.
trait GpuExtraDisplay {
    fn extra_display_info(&self) -> String {
        String::new()
    }
}

macro_rules! gpu_exec_node {
    ($name:ident) => {
        #[derive(Debug)]
        pub struct $name {
            inner: Arc<dyn ExecutionPlan>,
        }

        impl $name {
            pub fn new(inner: Arc<dyn ExecutionPlan>) -> Self {
                Self { inner }
            }
            pub fn inner(&self) -> &Arc<dyn ExecutionPlan> {
                &self.inner
            }
        }

        impl DisplayAs for $name {
            fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
                let extra = self.extra_display_info();
                if extra.is_empty() {
                    write!(f, "{}", stringify!($name))
                } else {
                    write!(f, "{}: {}", stringify!($name), extra)
                }
            }
        }

        impl ExecutionPlan for $name {
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
                Ok(Arc::new(Self::new(new_inner)))
            }
            fn name(&self) -> &str {
                stringify!($name)
            }
            fn execute(
                &self,
                partition: usize,
                context: Arc<TaskContext>,
            ) -> Result<SendableRecordBatchStream> {
                self.inner.execute(partition, context)
            }
        }
    };
}

gpu_exec_node!(GpuFilterExec);
impl GpuExtraDisplay for GpuFilterExec {
    fn extra_display_info(&self) -> String {
        let fe = self.inner.as_any().downcast_ref::<FilterExec>().unwrap();
        let mut s = format!("predicate={}", fe.predicate());
        if let Some(proj) = fe.projection() {
            let cols: Vec<String> = proj.iter().map(|i| i.to_string()).collect();
            s.push_str(&format!(", projection=[{}]", cols.join(", ")));
        }
        s
    }
}

gpu_exec_node!(GpuProjectExec);
impl GpuExtraDisplay for GpuProjectExec {
    fn extra_display_info(&self) -> String {
        let pe = self.inner.as_any().downcast_ref::<ProjectionExec>().unwrap();
        let exprs: Vec<String> = pe
            .expr()
            .iter()
            .map(|(e, alias)| format!("{e} as {alias}"))
            .collect();
        format!("expr=[{}]", exprs.join(", "))
    }
}

gpu_exec_node!(GpuAggregateExec);
impl GpuExtraDisplay for GpuAggregateExec {
    fn extra_display_info(&self) -> String {
        let agg = self.inner.as_any().downcast_ref::<AggregateExec>().unwrap();
        let groups: Vec<&str> = agg.group_expr().expr().iter()
            .map(|(_, name): &(_, String)| name.as_str())
            .collect();
        let aggrs: Vec<&str> = agg.aggr_expr().iter()
            .map(|e| e.name())
            .collect();
        format!("group_by=[{}], aggr=[{}]", groups.join(", "), aggrs.join(", "))
    }
}

gpu_exec_node!(GpuHashJoinExec);
impl GpuExtraDisplay for GpuHashJoinExec {
    fn extra_display_info(&self) -> String {
        let hj = self.inner.as_any().downcast_ref::<HashJoinExec>().unwrap();
        let on: Vec<String> = hj
            .on()
            .iter()
            .map(|(l, r)| format!("({l}, {r})"))
            .collect();
        let mut s = format!("join_type={:?}, on=[{}]", hj.join_type(), on.join(", "));
        if let Some(jf) = hj.filter() {
            s.push_str(&format!(", filter={}", jf.expression()));
        }
        if let Some(proj) = hj.projection.as_ref() {
            let cols: Vec<String> = proj.iter().map(|i| i.to_string()).collect();
            s.push_str(&format!(", projection=[{}]", cols.join(", ")));
        }
        s
    }
}

gpu_exec_node!(GpuCrossJoinExec);
impl GpuExtraDisplay for GpuCrossJoinExec {}

gpu_exec_node!(GpuNestedLoopJoinExec);
impl GpuExtraDisplay for GpuNestedLoopJoinExec {
    fn extra_display_info(&self) -> String {
        let nlj = self
            .inner
            .as_any()
            .downcast_ref::<NestedLoopJoinExec>()
            .unwrap();
        let mut s = format!("join_type={:?}", nlj.join_type());
        if let Some(jf) = nlj.filter() {
            s.push_str(&format!(", filter={}", jf.expression()));
        }
        if let Some(proj) = nlj.projection() {
            let cols: Vec<String> = proj.iter().map(|i| i.to_string()).collect();
            s.push_str(&format!(", projection=[{}]", cols.join(", ")));
        }
        s
    }
}

gpu_exec_node!(GpuSortExec);
impl GpuExtraDisplay for GpuSortExec {
    fn extra_display_info(&self) -> String {
        let se = self.inner.as_any().downcast_ref::<SortExec>().unwrap();
        let mut s = format!("expr=[{}]", se.expr());
        if let Some(f) = se.fetch() {
            s.push_str(&format!(", fetch={f}"));
        }
        s
    }
}

gpu_exec_node!(GpuCoalesceBatchesExec);
impl GpuExtraDisplay for GpuCoalesceBatchesExec {
    fn extra_display_info(&self) -> String {
        let cb = self.inner.as_any().downcast_ref::<CoalesceBatchesExec>().unwrap();
        format!("target_batch_size={}", cb.target_batch_size())
    }
}

gpu_exec_node!(GpuCoalescePartitionsExec);
impl GpuExtraDisplay for GpuCoalescePartitionsExec {}

gpu_exec_node!(GpuRepartitionExec);
impl GpuExtraDisplay for GpuRepartitionExec {
    fn extra_display_info(&self) -> String {
        let rp = self.inner.as_any().downcast_ref::<RepartitionExec>().unwrap();
        let partitioning = rp.partitioning();
        let input_partitions = rp.input().properties().output_partitioning().partition_count();
        format!("partitioning={partitioning}, input_partitions={input_partitions}")
    }
}

gpu_exec_node!(GpuSortPreservingMergeExec);
impl GpuExtraDisplay for GpuSortPreservingMergeExec {
    fn extra_display_info(&self) -> String {
        let spm = self.inner.as_any().downcast_ref::<SortPreservingMergeExec>().unwrap();
        format!("[{}]", spm.expr())
    }
}

gpu_exec_node!(GpuUnionExec);
impl GpuExtraDisplay for GpuUnionExec {}

gpu_exec_node!(GpuInterleaveExec);
impl GpuExtraDisplay for GpuInterleaveExec {}

gpu_exec_node!(GpuGlobalLimitExec);
impl GpuExtraDisplay for GpuGlobalLimitExec {
    fn extra_display_info(&self) -> String {
        let gl = self.inner.as_any().downcast_ref::<GlobalLimitExec>().unwrap();
        match gl.fetch() {
            Some(f) => format!("skip={}, fetch={}", gl.skip(), f),
            None => format!("skip={}, fetch=None", gl.skip()),
        }
    }
}

gpu_exec_node!(GpuWindowExec);
impl GpuExtraDisplay for GpuWindowExec {
    fn extra_display_info(&self) -> String {
        // Window exprs live on either WindowAggExec or BoundedWindowAggExec.
        let names: Vec<String> = if let Some(w) =
            self.inner.as_any().downcast_ref::<WindowAggExec>()
        {
            w.window_expr().iter().map(|e| e.name().to_string()).collect()
        } else if let Some(w) = self.inner.as_any().downcast_ref::<BoundedWindowAggExec>() {
            w.window_expr().iter().map(|e| e.name().to_string()).collect()
        } else {
            vec![]
        };
        format!("wdw=[{}]", names.join(", "))
    }
}

// ---------------------------------------------------------------------------
// GpuScanExec — wraps ParquetExec to override batch_size at execution time
// ---------------------------------------------------------------------------

/// One entry of the scan's explicit row-group→batch→partition MAP (Phase 2 Inc1):
/// a group of WHOLE row groups read as one batch, landing in output `partition`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanBatchMap {
    pub row_groups: Vec<u32>,
    pub partition: u32,
}

/// Build the RG→batch→partition map (Phase 2 Inc1): split the row groups to read
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
    /// Explicit RG→batch→partition MAP (Phase 2 Inc1). EMPTY = legacy
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
// table + projections (both round-trip through serialization) and batch_size; the
// pushed-down parquet predicate is deliberately NOT shown here because GpuScan
// serialization doesn't carry it, so it would break the flatbuffer roundtrip's
// plan_str equality after deserialize.
impl GpuExtraDisplay for GpuScanExec {
    fn extra_display_info(&self) -> String {
        let Some(parquet) = self.inner.as_any().downcast_ref::<ParquetExec>() else {
            return format!("batch_size={}", self.gpu_batch_size);
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
        parts.push(format!("batch_size={}", self.gpu_batch_size));
        parts.join(", ")
    }
}

impl DisplayAs for GpuScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GpuScanExec: {}", self.extra_display_info())
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
// GpuExecutionRule — replace CPU nodes with GPU wrappers
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GpuExecutionRule;

/// Expand `x IN (a, b, c)` into `((x = a) OR (x = b)) OR (x = c)` — or its
/// `NOT(...)` form for `NOT IN`. cuDF's AST has no IN opcode, so this lowering
/// must happen before execution; doing it here (in the plan) rather than inside
/// the serializer keeps serialization a verbatim encoding of the plan.
fn expand_in_list(in_list: &InListExpr) -> Result<Arc<dyn PhysicalExpr>> {
    let list = in_list.list();
    if list.is_empty() {
        return Err(datafusion::error::DataFusionError::NotImplemented(
            "IN with empty list".into(),
        ));
    }
    let target = in_list.expr();
    let eq = |item: &Arc<dyn PhysicalExpr>| -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(target.clone(), Operator::Eq, item.clone()))
    };
    let mut acc = eq(&list[0]);
    for item in &list[1..] {
        acc = Arc::new(BinaryExpr::new(acc, Operator::Or, eq(item)));
    }
    if in_list.negated() {
        acc = Arc::new(NotExpr::new(acc));
    }
    Ok(acc)
}

/// Recursively replace every `InListExpr` in `expr` with its OR-chain form.
fn lower_in_lists(expr: Arc<dyn PhysicalExpr>) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
    expr.transform_up(|e| {
        if let Some(in_list) = e.as_any().downcast_ref::<InListExpr>() {
            Ok(Transformed::yes(expand_in_list(in_list)?))
        } else {
            Ok(Transformed::no(e))
        }
    })
}

impl PhysicalOptimizerRule for GpuExecutionRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let result = plan.transform_up(|node: Arc<dyn ExecutionPlan>| {
            let new_node: Arc<dyn ExecutionPlan> = if node.as_any().is::<FilterExec>() {
                // Lower any IN-lists in the predicate before wrapping.
                let rebuilt: Option<Arc<dyn ExecutionPlan>> = {
                    let fe = node.as_any().downcast_ref::<FilterExec>().unwrap();
                    let lowered = lower_in_lists(fe.predicate().clone())?;
                    if lowered.transformed {
                        let mut f = FilterExec::try_new(lowered.data, fe.input().clone())?;
                        if let Some(proj) = fe.projection() {
                            f = f.with_projection(Some(proj.clone()))?;
                        }
                        Some(Arc::new(f) as Arc<dyn ExecutionPlan>)
                    } else {
                        None
                    }
                };
                Arc::new(GpuFilterExec::new(rebuilt.unwrap_or(node)))
            } else if node.as_any().is::<ProjectionExec>() {
                let rebuilt: Option<Arc<dyn ExecutionPlan>> = {
                    let pe = node.as_any().downcast_ref::<ProjectionExec>().unwrap();
                    let mut changed = false;
                    let mut new_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
                        Vec::with_capacity(pe.expr().len());
                    for (e, alias) in pe.expr() {
                        let lowered = lower_in_lists(e.clone())?;
                        changed |= lowered.transformed;
                        new_exprs.push((lowered.data, alias.clone()));
                    }
                    if changed {
                        Some(Arc::new(ProjectionExec::try_new(new_exprs, pe.input().clone())?)
                            as Arc<dyn ExecutionPlan>)
                    } else {
                        None
                    }
                };
                Arc::new(GpuProjectExec::new(rebuilt.unwrap_or(node)))
            } else if node.as_any().is::<AggregateExec>() {
                Arc::new(GpuAggregateExec::new(node))
            } else if node.as_any().is::<HashJoinExec>() {
                // Lower any IN-lists in the residual join filter before wrapping.
                let rebuilt: Option<Arc<dyn ExecutionPlan>> = {
                    let hj = node.as_any().downcast_ref::<HashJoinExec>().unwrap();
                    match hj.filter() {
                        Some(jf) => {
                            let lowered = lower_in_lists(jf.expression().clone())?;
                            if lowered.transformed {
                                let new_filter = JoinFilter::new(
                                    lowered.data,
                                    jf.column_indices().to_vec(),
                                    jf.schema().clone(),
                                );
                                let h = HashJoinExec::try_new(
                                    hj.left().clone(),
                                    hj.right().clone(),
                                    hj.on().to_vec(),
                                    Some(new_filter),
                                    hj.join_type(),
                                    hj.projection.clone(),
                                    *hj.partition_mode(),
                                    hj.null_equals_null(),
                                )?;
                                Some(Arc::new(h) as Arc<dyn ExecutionPlan>)
                            } else {
                                None
                            }
                        }
                        None => None,
                    }
                };
                Arc::new(GpuHashJoinExec::new(rebuilt.unwrap_or(node)))
            } else if node.as_any().is::<CrossJoinExec>() {
                Arc::new(GpuCrossJoinExec::new(node))
            } else if node.as_any().is::<NestedLoopJoinExec>() {
                Arc::new(GpuNestedLoopJoinExec::new(node))
            } else if node.as_any().is::<SortExec>() {
                Arc::new(GpuSortExec::new(node))
            } else if node.as_any().is::<CoalesceBatchesExec>() {
                Arc::new(GpuCoalesceBatchesExec::new(node))
            } else if node.as_any().is::<CoalescePartitionsExec>() {
                Arc::new(GpuCoalescePartitionsExec::new(node))
            } else if node.as_any().is::<RepartitionExec>() {
                Arc::new(GpuRepartitionExec::new(node))
            } else if node.as_any().is::<SortPreservingMergeExec>() {
                Arc::new(GpuSortPreservingMergeExec::new(node))
            } else if node.as_any().is::<UnionExec>() {
                Arc::new(GpuUnionExec::new(node))
            } else if node.as_any().is::<InterleaveExec>() {
                Arc::new(GpuInterleaveExec::new(node))
            } else if node.as_any().is::<GlobalLimitExec>() {
                Arc::new(GpuGlobalLimitExec::new(node))
            } else if node.as_any().is::<WindowAggExec>()
                || node.as_any().is::<BoundedWindowAggExec>()
            {
                Arc::new(GpuWindowExec::new(node))
            } else {
                return Ok(Transformed::no(node));
            };
            Ok(Transformed::yes(new_node))
        })?;
        Ok(result.data)
    }

    fn name(&self) -> &str {
        "gpu_execution"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Memory analysis
// ---------------------------------------------------------------------------

/// Estimated byte width of a single row for the given schema.
/// Uses `DataType::primitive_width()` for fixed-size types,
/// falls back to 32 bytes for variable-length types (Utf8, Binary, etc.).
pub fn row_width(schema: &SchemaRef) -> usize {
    schema
        .fields()
        .iter()
        .map(|f| f.data_type().primitive_width().unwrap_or(32))
        .sum::<usize>()
        .max(1) // at least 1 byte per row
}

// ---------------------------------------------------------------------------
// Estimator traits
// ---------------------------------------------------------------------------

/// Estimates the selectivity of a filter operator: the fraction of input rows
/// that pass the predicate (0.0 = nothing passes, 1.0 = everything passes).
///
/// TODO: Implement a statistics-based estimator that uses DataFusion's
/// `ExecutionPlan::statistics()` — e.g., histogram-based estimation for range
/// filters, NDV-based estimation for equality filters.
pub(crate) trait SelectivityEstimator {
    fn estimate(&self, plan: &Arc<dyn ExecutionPlan>) -> f64;
}

/// Estimates the cardinality ratio of a join: output_rows / max(left_rows, right_rows).
/// A ratio of 1.0 means 1:1, >1.0 means fan-out, <1.0 means a semi-join or filtering join.
///
/// TODO: Implement a statistics-based estimator that uses DataFusion's
/// `ExecutionPlan::statistics()` — e.g., foreign-key detection for 1:1 joins,
/// NDV-based join selectivity for many-to-many.
pub(crate) trait CardinalityEstimator {
    fn estimate(&self, plan: &Arc<dyn ExecutionPlan>) -> f64;
}

/// Assumes all filters pass 100% of rows.
pub(crate) struct TrivialSelectivityEstimator;

impl SelectivityEstimator for TrivialSelectivityEstimator {
    fn estimate(&self, _plan: &Arc<dyn ExecutionPlan>) -> f64 {
        1.0
    }
}

/// Assumes all joins are 1:1 (output rows = input rows).
pub(crate) struct TrivialCardinalityEstimator;

impl CardinalityEstimator for TrivialCardinalityEstimator {
    fn estimate(&self, _plan: &Arc<dyn ExecutionPlan>) -> f64 {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Subtree memory model
// ---------------------------------------------------------------------------

/// Result of analyzing a plan subtree's memory usage.
///
/// Memory is modeled as a linear function of the scan batch size N.
/// `output_row_ratio` tracks the cumulative row multiplier: if a filter has
/// 50% selectivity, downstream operators see 0.5 × N rows instead of N.
#[derive(Clone, Copy)]
pub struct SubtreeMemory {
    /// Peak GPU memory as bytes per scan-batch-row N:
    /// `peak_bytes = subtree_max_row_bytes * N`.
    pub subtree_max_row_bytes: usize,
    /// Output row width in bytes (per output row).
    pub output_width: usize,
    /// Ratio of output rows to original batch size N.
    /// 1.0 means row count is preserved; <1.0 after filters; >1.0 after fan-out joins.
    pub output_row_ratio: f64,
    /// Estimated input bytes flowing into this node per scan-batch-row N.
    pub input_row_bytes: usize,
    /// Estimated output bytes produced by this node per scan-batch-row N.
    pub output_row_bytes: usize,
}

/// Walk the plan tree and compute peak memory as a linear function of batch size N.
///
/// Per-operator memory = input batch + output batch, where the row counts are
/// adjusted by selectivity (filters) and cardinality (joins) estimators.
pub fn analyze_memory(plan: &Arc<dyn ExecutionPlan>) -> SubtreeMemory {
    analyze_memory_with(
        plan,
        &TrivialSelectivityEstimator,
        &TrivialCardinalityEstimator,
    )
}

/// Compute a node's `SubtreeMemory` given already-computed child results.
/// Does not recurse — callers are responsible for walking children first.
pub(crate) fn node_memory_with(
    plan: &Arc<dyn ExecutionPlan>,
    child_mems: &[SubtreeMemory],
    selectivity: &dyn SelectivityEstimator,
    cardinality: &dyn CardinalityEstimator,
) -> SubtreeMemory {
    let output_width = row_width(&plan.schema());

    if child_mems.is_empty() {
        return SubtreeMemory {
            subtree_max_row_bytes: output_width,
            output_width,
            output_row_ratio: 1.0,
            input_row_bytes: 0,
            output_row_bytes: output_width,
        };
    }

    match plan.name() {
        "GpuFilterExec" => {
            let child = child_mems[0];
            let sel = selectivity.estimate(plan);
            let input_rows_bytes = (child.output_row_ratio * child.output_width as f64) as usize;
            let output_rows_bytes = (child.output_row_ratio * sel * output_width as f64) as usize;
            SubtreeMemory {
                subtree_max_row_bytes: child
                    .subtree_max_row_bytes
                    .max(input_rows_bytes + output_rows_bytes),
                output_width,
                output_row_ratio: child.output_row_ratio * sel,
                input_row_bytes: input_rows_bytes,
                output_row_bytes: output_rows_bytes,
            }
        }
        "GpuProjectExec" | "GpuAggregateExec" => {
            let child = child_mems[0];
            let input_rows_bytes = (child.output_row_ratio * child.output_width as f64) as usize;
            let output_rows_bytes = (child.output_row_ratio * output_width as f64) as usize;
            SubtreeMemory {
                subtree_max_row_bytes: child
                    .subtree_max_row_bytes
                    .max(input_rows_bytes + output_rows_bytes),
                output_width,
                output_row_ratio: child.output_row_ratio,
                input_row_bytes: input_rows_bytes,
                output_row_bytes: output_rows_bytes,
            }
        }
        "GpuHashJoinExec" => {
            let (build, probe) = (child_mems[0], child_mems[1]);
            let card = cardinality.estimate(plan);
            let build_bytes = (build.output_row_ratio * build.output_width as f64) as usize;
            let probe_bytes = (probe.output_row_ratio * probe.output_width as f64) as usize;
            let output_ratio = build.output_row_ratio.max(probe.output_row_ratio) * card;
            let output_bytes = (output_ratio * output_width as f64) as usize;
            let own = build_bytes + probe_bytes + output_bytes;
            SubtreeMemory {
                subtree_max_row_bytes: build
                    .subtree_max_row_bytes
                    .max(probe.subtree_max_row_bytes)
                    .max(own),
                output_width,
                output_row_ratio: output_ratio,
                input_row_bytes: build_bytes + probe_bytes,
                output_row_bytes: output_bytes,
            }
        }
        "CrossJoinExec" | "NestedLoopJoinExec" => {
            let (left, right) = (child_mems[0], child_mems[1]);
            let card = cardinality.estimate(plan);
            let left_bytes = (left.output_row_ratio * left.output_width as f64) as usize;
            let right_bytes = (right.output_row_ratio * right.output_width as f64) as usize;
            let output_ratio = left.output_row_ratio * right.output_row_ratio * card;
            let output_bytes = (output_ratio * output_width as f64) as usize;
            let own = left_bytes + right_bytes + output_bytes;
            SubtreeMemory {
                subtree_max_row_bytes: left
                    .subtree_max_row_bytes
                    .max(right.subtree_max_row_bytes)
                    .max(own),
                output_width,
                output_row_ratio: output_ratio,
                input_row_bytes: left_bytes + right_bytes,
                output_row_bytes: output_bytes,
            }
        }
        "GpuSortExec" => {
            let child = child_mems[0];
            let input_bytes = (child.output_row_ratio * child.output_width as f64) as usize;
            SubtreeMemory {
                subtree_max_row_bytes: child.subtree_max_row_bytes.max(2 * input_bytes),
                output_width,
                output_row_ratio: child.output_row_ratio,
                input_row_bytes: input_bytes,
                output_row_bytes: input_bytes,
            }
        }
        // UNION ALL (concatenate the rows of all inputs). Must mirror what
        // execute_union actually does today, branching on input count:
        //   - single input  → std::move (true pass-through): peak = child peak.
        //   - multiple inputs → all input tables are held live, then the
        //     cudf::concatenate output is allocated → peak ≈ Σ(inputs) + output,
        //     and the output row count is the *sum* of child cardinalities.
        // Undercounting here would feed GpuMemoryBudgetRule too small a
        // subtree_max_row_bytes → too large a batch size → OOM.
        // (Once the multi-partition / true-pass-through model lands in #34, the
        // concat moves up into GpuCoalescePartitionsExec and this can revert to a
        // plain pass-through.)
        "GpuUnionExec" | "GpuInterleaveExec" => {
            let max_child_peak = child_mems
                .iter()
                .map(|c| c.subtree_max_row_bytes)
                .max()
                .unwrap_or(output_width);
            if child_mems.len() <= 1 {
                let child = child_mems.first().copied().unwrap_or(SubtreeMemory {
                    subtree_max_row_bytes: output_width,
                    output_width,
                    output_row_ratio: 1.0,
                    input_row_bytes: 0,
                    output_row_bytes: output_width,
                });
                let input_bytes = (child.output_row_ratio * child.output_width as f64) as usize;
                let output_bytes = (child.output_row_ratio * output_width as f64) as usize;
                SubtreeMemory {
                    subtree_max_row_bytes: max_child_peak,
                    output_width,
                    output_row_ratio: child.output_row_ratio,
                    input_row_bytes: input_bytes,
                    output_row_bytes: output_bytes,
                }
            } else {
                let inputs_bytes: usize = child_mems
                    .iter()
                    .map(|c| (c.output_row_ratio * c.output_width as f64) as usize)
                    .sum();
                let output_row_ratio: f64 = child_mems.iter().map(|c| c.output_row_ratio).sum();
                let output_bytes = (output_row_ratio * output_width as f64) as usize;
                let own = inputs_bytes + output_bytes;
                SubtreeMemory {
                    subtree_max_row_bytes: max_child_peak.max(own),
                    output_width,
                    output_row_ratio,
                    input_row_bytes: inputs_bytes,
                    output_row_bytes: output_bytes,
                }
            }
        }
        // Everything else (CoalescePartitions, Repartition, CoalesceBatches, etc.):
        // pass-through — peak is the max of children, ratio is max of children.
        _ => {
            let max_peak = child_mems
                .iter()
                .map(|c| c.subtree_max_row_bytes)
                .max()
                .unwrap_or(output_width);
            let max_ratio = child_mems
                .iter()
                .map(|c| c.output_row_ratio)
                .fold(1.0_f64, f64::max);
            let input_bytes: usize = child_mems
                .iter()
                .map(|c| (c.output_row_ratio * c.output_width as f64) as usize)
                .sum();
            let output_bytes = (max_ratio * output_width as f64) as usize;
            SubtreeMemory {
                subtree_max_row_bytes: max_peak,
                output_width,
                output_row_ratio: max_ratio,
                input_row_bytes: input_bytes,
                output_row_bytes: output_bytes,
            }
        }
    }
}

pub(crate) fn analyze_memory_with(
    plan: &Arc<dyn ExecutionPlan>,
    selectivity: &dyn SelectivityEstimator,
    cardinality: &dyn CardinalityEstimator,
) -> SubtreeMemory {
    let child_mems: Vec<SubtreeMemory> = plan
        .children()
        .iter()
        .map(|c| analyze_memory_with(c, selectivity, cardinality))
        .collect();
    node_memory_with(plan, &child_mems, selectivity, cardinality)
}

/// Walk the plan tree once and return per-node memory info in pre-order.
/// Each entry is `(name, depth, SubtreeMemory)`. O(n) — each node is visited once.
pub fn analyze_memory_nodes(plan: &Arc<dyn ExecutionPlan>) -> Vec<(String, usize, SubtreeMemory)> {
    fn walk(
        plan: &Arc<dyn ExecutionPlan>,
        depth: usize,
        result: &mut Vec<(String, usize, SubtreeMemory)>,
    ) -> SubtreeMemory {
        let my_idx = result.len();
        result.push((plan.name().to_string(), depth, SubtreeMemory {
            subtree_max_row_bytes: 0,
            output_width: 0,
            output_row_ratio: 0.0,
            input_row_bytes: 0,
            output_row_bytes: 0,
        }));
        let child_mems: Vec<SubtreeMemory> = plan
            .children()
            .iter()
            .map(|c| walk(c, depth + 1, result))
            .collect();
        let mem = node_memory_with(
            plan,
            &child_mems,
            &TrivialSelectivityEstimator,
            &TrivialCardinalityEstimator,
        );
        result[my_idx].2 = mem;
        mem
    }
    let mut result = Vec::new();
    walk(plan, 0, &mut result);
    result
}

// ---------------------------------------------------------------------------
// GpuMemoryBudgetRule — compute batch size from memory budget, wrap scans
// ---------------------------------------------------------------------------

/// Minimum GPU memory budget at which a multi-partition scan emits a real
/// RG→batch→partition map (Phase 2 Inc1). BELOW this, the device is treated as
/// memory-constrained: the scan stays single-partition (the tp8-mem2gib #11
/// determinism path), so the plan serializes byte-identically to the legacy
/// scan and the flatbuffer roundtrip is stable (deserialize does not reconstruct
/// N partitions, so GpuRepartitionExec.input_partitions does not flip 1→8).
/// The real-partitioning device (H200, tp8-mem120gib) sits well above this and
/// gets the map; tp1 never attaches one regardless (target_partitions == 1).
/// How a target-partitioned plan is realized on the multi-handle node-executor
/// path. This — NOT the memory budget — is the sole discriminator for whether the
/// scan gets an RG→batch→partition map and the Hash repartition is lowered into an
/// explicit GpuCoalescePartitions(M→1) + GpuRepartition(1→N).
///
/// Decoupling real-partitioning from budget (dmitry, replacing the old 16 GiB
/// `REAL_PARTITION_MIN_BUDGET` threshold) keeps a memory-constrained real-8-way
/// device (e.g. a genuine 8-way at 2 GiB) EXPRESSIBLE — that execution/enforcer
/// work is GitHub #91 (post-Phase-2; do not conflate with this policy flag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionMode {
    /// Legacy single-partition-coalesced path — tp1 and the tp8-mem2gib #11
    /// determinism device. No scan map, no repartition lowering (byte-identical
    /// serialized plan, stable flatbuffer roundtrip).
    SinglePartition,
    /// Real N-way partitioning (H200 / tp8-mem120gib): the scan emits N partitions
    /// per its map and a Hash repartition is lowered + Spark-murmur3 partitioned.
    RealMultiPartition,
}

/// True when a plan should be realized with real N-way partitioning: multi-partition
/// (`n_parts > 1`) AND the explicit `RealMultiPartition` policy. Gates BOTH the scan
/// map AND the Hash-repartition lowering, so a device either gets the whole real-8-way
/// treatment or none of it.
fn real_partitioning(mode: PartitionMode, n_parts: usize) -> bool {
    n_parts > 1 && mode == PartitionMode::RealMultiPartition
}

#[derive(Debug)]
pub struct GpuMemoryBudgetRule {
    gpu_memory_budget: usize,
    partition_mode: PartitionMode,
}

impl GpuMemoryBudgetRule {
    pub fn new(gpu_memory_budget: usize, partition_mode: PartitionMode) -> Self {
        Self { gpu_memory_budget, partition_mode }
    }
}

impl PhysicalOptimizerRule for GpuMemoryBudgetRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mem = analyze_memory(&plan);
        let max_n = if mem.subtree_max_row_bytes > 0 {
            self.gpu_memory_budget / mem.subtree_max_row_bytes
        } else {
            config.execution.batch_size
        };
        let batch_size = max_n.max(1);

        let result = plan.transform_up(|node: Arc<dyn ExecutionPlan>| {
            if node.as_any().is::<ParquetExec>() {
                // Real-partitioning device only (target_partitions>1 AND
                // PartitionMode::RealMultiPartition — see `real_partitioning`): attach
                // the explicit RG→batch→partition map so the scan emits N partitions
                // (dissolving RoundRobin into the scan). The RGs to read = #12 survivors
                // if a static predicate prunes, else all groups. tp1 OR the single-
                // partition mode (tp8-mem2gib) gets an empty map = legacy single-
                // partition scan, keeping the serialized plan / roundtrip byte-stable.
                let n_parts = config.execution.target_partitions;
                let batches = if real_partitioning(self.partition_mode, n_parts) {
                    let parquet = node.as_any().downcast_ref::<ParquetExec>().unwrap();
                    crate::gpu_rowgroup_prune::surviving_row_groups(parquet)
                        .or_else(|| crate::gpu_rowgroup_prune::all_row_groups(parquet))
                        .map(|rgs| build_scan_map(&rgs, n_parts))
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                Ok(Transformed::yes(
                    Arc::new(GpuScanExec::new(node, batch_size).with_batches(batches))
                        as Arc<dyn ExecutionPlan>,
                ))
            } else if node.as_any().is::<GpuCoalesceBatchesExec>() {
                let gpu_cb = node.as_any().downcast_ref::<GpuCoalesceBatchesExec>().unwrap();
                let coalesce = gpu_cb.inner().as_any().downcast_ref::<CoalesceBatchesExec>().unwrap();
                let input = coalesce.input().clone();
                let new_inner: Arc<dyn ExecutionPlan> = Arc::new(CoalesceBatchesExec::new(input, batch_size));
                Ok(Transformed::yes(
                    Arc::new(GpuCoalesceBatchesExec::new(new_inner)) as Arc<dyn ExecutionPlan>,
                ))
            } else if node.as_any().is::<GpuRepartitionExec>()
                && real_partitioning(self.partition_mode, config.execution.target_partitions)
            {
                // Real-partitioning device only: lower a multi-input Hash repartition
                // GpuRepartition(Hash, M→N) into the EXPLICIT 2-node form (dmitry's
                // amendment) — GpuCoalescePartitions(M→1, BUFFERING concat) feeding
                // GpuRepartition(Hash, 1→N). This makes the shuffle's concat a visible,
                // cost-/memory-accounted plan node (Task-A renders it below the Hash),
                // and gives the CPU/GPU node executors a single 1-partition input to
                // hash-partition into N via Spark-murmur3 (Inc2). RoundRobin and
                // already-1→N repartitions are left untouched. Off the real device
                // (tp8-mem2gib/tp1) the plan is unchanged → roundtrip byte-stable.
                let gpu_rp = node.as_any().downcast_ref::<GpuRepartitionExec>().unwrap();
                let rp = gpu_rp.inner().as_any().downcast_ref::<RepartitionExec>().unwrap();
                let input_parts = rp.input().properties().output_partitioning().partition_count();
                if matches!(rp.partitioning(), Partitioning::Hash(_, _)) && input_parts > 1 {
                    let coalesced: Arc<dyn ExecutionPlan> = Arc::new(GpuCoalescePartitionsExec::new(
                        Arc::new(CoalescePartitionsExec::new(rp.input().clone())),
                    ));
                    let new_rp = RepartitionExec::try_new(coalesced, rp.partitioning().clone())?;
                    Ok(Transformed::yes(
                        Arc::new(GpuRepartitionExec::new(Arc::new(new_rp) as Arc<dyn ExecutionPlan>))
                            as Arc<dyn ExecutionPlan>,
                    ))
                } else {
                    Ok(Transformed::no(node))
                }
            } else {
                Ok(Transformed::no(node))
            }
        })?;
        Ok(result.data)
    }

    fn name(&self) -> &str {
        "gpu_memory_budget"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::empty::EmptyExec;

    fn empty(fields: Vec<Field>) -> Arc<dyn ExecutionPlan> {
        Arc::new(EmptyExec::new(Arc::new(Schema::new(fields))))
    }

    fn child(subtree_max_row_bytes: usize, output_width: usize, output_row_ratio: f64) -> SubtreeMemory {
        SubtreeMemory {
            subtree_max_row_bytes,
            output_width,
            output_row_ratio,
            input_row_bytes: 0,
            output_row_bytes: 0,
        }
    }

    // Focused coverage of the CrossJoin/NestedLoopJoin cost arm in
    // node_memory_with (CrossJoinExec and NestedLoopJoinExec share the arm).
    // Both estimators are trivial (selectivity = cardinality = 1.0).
    #[test]
    fn cross_nlj_cost_arm() {
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CrossJoinExec::new(
            empty(vec![Field::new("a", DataType::Int64, false)]),
            empty(vec![Field::new("b", DataType::Int64, false)]),
        ));
        assert_eq!(plan.name(), "CrossJoinExec");

        // Distinct ratios/widths so each term is individually checkable; small
        // child peaks so the join's own footprint dominates subtree_max.
        let left = child(10, 8, 2.0);
        let right = child(20, 16, 3.0);

        let mem = node_memory_with(
            &plan,
            &[left, right],
            &TrivialSelectivityEstimator,
            &TrivialCardinalityEstimator,
        );

        let card = 1.0; // TrivialCardinalityEstimator
        let output_width = row_width(&plan.schema());
        let left_bytes = (left.output_row_ratio * left.output_width as f64) as usize; // 16
        let right_bytes = (right.output_row_ratio * right.output_width as f64) as usize; // 48
        let output_ratio = left.output_row_ratio * right.output_row_ratio * card; // 6.0
        let output_bytes = (output_ratio * output_width as f64) as usize;
        let own = left_bytes + right_bytes + output_bytes;

        assert_eq!(mem.output_width, output_width);
        assert_eq!(mem.output_row_ratio, output_ratio);
        assert_eq!(mem.input_row_bytes, left_bytes + right_bytes);
        assert_eq!(mem.output_row_bytes, output_bytes);
        assert_eq!(
            mem.subtree_max_row_bytes,
            left.subtree_max_row_bytes
                .max(right.subtree_max_row_bytes)
                .max(own)
        );
        // own (>= 64) dominates the deliberately tiny child peaks (10, 20).
        assert_eq!(mem.subtree_max_row_bytes, own);
    }
}