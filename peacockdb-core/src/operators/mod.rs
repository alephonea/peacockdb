//! GPU operator wrappers, grouped by family, plus the single dispatch point.
//!
//! Every `Gpu*Exec` is a passthrough wrapper around a DataFusion node (see
//! `gpu_exec_node!`). They exist so `gpu_executor`/the serializer can pattern-match
//! on intent, and `cpu_executor::strip_gpu` undoes them at execution time.
//!
//! ONE PLACE FOR DISPATCH: `as_operator()` below is the only downcast ladder over
//! these types. It is still a match over concrete types — Rust gives us no other way
//! to go from `&dyn ExecutionPlan` to a trait we defined — but there is exactly one
//! of it, instead of one per call site.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

pub mod operator;

/// Optional extra display info appended after the node name in plan output.
/// Implement with a non-empty string to annotate a specific GPU node type.
pub(crate) trait GpuExtraDisplay {
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

// Make the macro and the display trait reachable from the family modules without
// `#[macro_use]` ordering games.
pub(crate) use gpu_exec_node;

pub mod aggregate;
pub mod coalesce;
pub mod filter;
pub mod join;
pub mod limit;
pub mod project;
pub mod repartition;
pub mod scan;
pub mod sort;
pub mod union;
pub mod window;


// ---------------------------------------------------------------------------
// THE dispatch point. One downcast ladder for the whole crate.
//
// It is still a match over concrete types — Rust offers no way to go from
// `&dyn ExecutionPlan` to a trait we defined without one — but there is exactly ONE
// of them now, instead of a copy at every call site that needed to know what a node
// was. Adding an operator means adding a line here and nowhere else.
// ---------------------------------------------------------------------------

/// Borrow a plan node as an [`Operator`], or `None` for a plain DataFusion node.
pub fn as_operator(plan: &dyn ExecutionPlan) -> Option<&dyn Operator> {
    macro_rules! try_as {
        ($($ty:ty),+ $(,)?) => {
            $(
                if let Some(op) = plan.as_any().downcast_ref::<$ty>() {
                    return Some(op as &dyn Operator);
                }
            )+
        };
    }
    try_as!(
        scan::GpuScanExec,
        filter::GpuFilterExec,
        project::GpuProjectExec,
        aggregate::GpuAggregateExec,
        join::GpuHashJoinExec,
        join::GpuCrossJoinExec,
        join::GpuNestedLoopJoinExec,
        sort::GpuSortExec,
        sort::GpuSortPreservingMergeExec,
        coalesce::GpuCoalesceBatchesExec,
        coalesce::GpuCoalescePartitionsExec,
        repartition::GpuRepartitionExec,
        union::GpuUnionExec,
        union::GpuInterleaveExec,
        limit::GpuGlobalLimitExec,
        window::GpuWindowExec,
    );
    None
}

/// The node the recursive CPU driver should actually execute, plus any batch-size
/// override the wrapper carried.
///
/// The asymmetry is LOAD-BEARING: five operators return `strips_to_inner() == false`
/// and pass through wrapped — each per-operator impl documents why.
pub fn strip_target(plan: &Arc<dyn ExecutionPlan>) -> (Arc<dyn ExecutionPlan>, Option<usize>) {
    match as_operator(plan.as_ref()) {
        // The scan is the only wrapper carrying the memory-budget batch size.
        Some(_) if plan.as_any().is::<scan::GpuScanExec>() => {
            let scan = plan.as_any().downcast_ref::<scan::GpuScanExec>().unwrap();
            (scan.inner().clone(), Some(scan.gpu_batch_size))
        }
        Some(op) if op.strips_to_inner() => (op.inner().clone(), None),
        // Either a non-stripping operator or a plain CPU node: pass through.
        _ => (plan.clone(), None),
    }
}

pub use operator::{Operator, PartitionTopology};
