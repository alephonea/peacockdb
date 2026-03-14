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
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};

// ---------------------------------------------------------------------------
// GPU exec node stubs (delegate to inner CPU node)
// ---------------------------------------------------------------------------

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
        }

        impl DisplayAs for $name {
            fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}", stringify!($name))
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
gpu_exec_node!(GpuProjectExec);
gpu_exec_node!(GpuAggregateExec);
gpu_exec_node!(GpuHashJoinExec);
gpu_exec_node!(GpuSortExec);

// ---------------------------------------------------------------------------
// GpuScanExec — wraps ParquetExec to override batch_size at execution time
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GpuScanExec {
    inner: Arc<dyn ExecutionPlan>,
    pub gpu_batch_size: usize,
}

impl GpuScanExec {
    pub fn new(inner: Arc<dyn ExecutionPlan>, gpu_batch_size: usize) -> Self {
        Self {
            inner,
            gpu_batch_size,
        }
    }
}

impl DisplayAs for GpuScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GpuScanExec: batch_size={}", self.gpu_batch_size)
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

impl PhysicalOptimizerRule for GpuExecutionRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let result = plan.transform_up(|node: Arc<dyn ExecutionPlan>| {
            let new_node: Arc<dyn ExecutionPlan> = if node.as_any().is::<FilterExec>() {
                Arc::new(GpuFilterExec::new(node))
            } else if node.as_any().is::<ProjectionExec>() {
                Arc::new(GpuProjectExec::new(node))
            } else if node.as_any().is::<AggregateExec>() {
                Arc::new(GpuAggregateExec::new(node))
            } else if node.as_any().is::<HashJoinExec>() {
                Arc::new(GpuHashJoinExec::new(node))
            } else if node.as_any().is::<SortExec>() {
                Arc::new(GpuSortExec::new(node))
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
pub(crate) fn row_width(schema: &SchemaRef) -> usize {
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
pub(crate) struct SubtreeMemory {
    /// Peak GPU memory as bytes per scan-batch-row N:
    /// `peak_bytes = subtree_max_row_bytes * N`.
    pub(crate) subtree_max_row_bytes: usize,
    /// Output row width in bytes (per output row).
    pub(crate) output_width: usize,
    /// Ratio of output rows to original batch size N.
    /// 1.0 means row count is preserved; <1.0 after filters; >1.0 after fan-out joins.
    pub(crate) output_row_ratio: f64,
}

/// Walk the plan tree and compute peak memory as a linear function of batch size N.
///
/// Per-operator memory = input batch + output batch, where the row counts are
/// adjusted by selectivity (filters) and cardinality (joins) estimators.
pub(crate) fn analyze_memory(plan: &Arc<dyn ExecutionPlan>) -> SubtreeMemory {
    analyze_memory_with(
        plan,
        &TrivialSelectivityEstimator,
        &TrivialCardinalityEstimator,
    )
}

pub(crate) fn analyze_memory_with(
    plan: &Arc<dyn ExecutionPlan>,
    selectivity: &dyn SelectivityEstimator,
    cardinality: &dyn CardinalityEstimator,
) -> SubtreeMemory {
    let output_width = row_width(&plan.schema());
    let children = plan.children();

    // Leaf node (ParquetExec, etc.): just the output batch.
    if children.is_empty() {
        return SubtreeMemory {
            subtree_max_row_bytes: output_width,
            output_width,
            output_row_ratio: 1.0,
        };
    }

    match plan.name() {
        // Filter: input batch (child rows) + output batch (filtered rows).
        "GpuFilterExec" => {
            let child = analyze_memory_with(children[0], selectivity, cardinality);
            let sel = selectivity.estimate(plan);
            let output_rows_bytes = (sel * output_width as f64) as usize;
            let input_rows_bytes = (child.output_row_ratio * child.output_width as f64) as usize;
            SubtreeMemory {
                subtree_max_row_bytes: child
                    .subtree_max_row_bytes
                    .max(input_rows_bytes + output_rows_bytes),
                output_width,
                output_row_ratio: child.output_row_ratio * sel,
            }
        }
        // Projection / aggregation: input batch + output batch, row count preserved.
        "GpuProjectExec" | "GpuAggregateExec" => {
            let child = analyze_memory_with(children[0], selectivity, cardinality);
            let input_rows_bytes = (child.output_row_ratio * child.output_width as f64) as usize;
            let output_rows_bytes = (child.output_row_ratio * output_width as f64) as usize;
            SubtreeMemory {
                subtree_max_row_bytes: child
                    .subtree_max_row_bytes
                    .max(input_rows_bytes + output_rows_bytes),
                output_width,
                output_row_ratio: child.output_row_ratio,
            }
        }
        // Hash join: build side + probe batch + output batch.
        "GpuHashJoinExec" => {
            let build = analyze_memory_with(children[0], selectivity, cardinality);
            let probe = analyze_memory_with(children[1], selectivity, cardinality);
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
            }
        }
        // Sort: input + workspace (index array, ~2× input).
        "GpuSortExec" => {
            let child = analyze_memory_with(children[0], selectivity, cardinality);
            let input_bytes = (child.output_row_ratio * child.output_width as f64) as usize;
            SubtreeMemory {
                subtree_max_row_bytes: child.subtree_max_row_bytes.max(2 * input_bytes),
                output_width,
                output_row_ratio: child.output_row_ratio,
            }
        }
        // Everything else (CoalescePartitions, Repartition, CoalesceBatches, etc.):
        // pass-through — peak is the max of children, ratio is max of children.
        _ => {
            let child_results: Vec<_> = children
                .iter()
                .map(|c| analyze_memory_with(c, selectivity, cardinality))
                .collect();
            let max_peak = child_results
                .iter()
                .map(|c| c.subtree_max_row_bytes)
                .max()
                .unwrap_or(output_width);
            let max_ratio = child_results
                .iter()
                .map(|c| c.output_row_ratio)
                .fold(1.0_f64, f64::max);
            SubtreeMemory {
                subtree_max_row_bytes: max_peak,
                output_width,
                output_row_ratio: max_ratio,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GpuMemoryBudgetRule — compute batch size from memory budget, wrap scans
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GpuMemoryBudgetRule {
    gpu_memory_budget: usize,
}

impl GpuMemoryBudgetRule {
    pub fn new(gpu_memory_budget: usize) -> Self {
        Self { gpu_memory_budget }
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
                Ok(Transformed::yes(
                    Arc::new(GpuScanExec::new(node, batch_size)) as Arc<dyn ExecutionPlan>,
                ))
            } else if let Some(coalesce) = node.as_any().downcast_ref::<CoalesceBatchesExec>() {
                let input = coalesce.input().clone();
                Ok(Transformed::yes(
                    Arc::new(CoalesceBatchesExec::new(input, batch_size))
                        as Arc<dyn ExecutionPlan>,
                ))
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
