//! The per-node CPU primitive and the helpers shared across executor modules.
//!
//! `execute_single_node` runs exactly ONE plan node against its children's
//! already-computed batches — it is the primitive [`super::backend::cpu_node_executor`]
//! is built on, while [`super::full_table_cpu_executor`] drives whole subtrees. Both
//! reach the helpers here, which is why they live in their own file.

use std::sync::{Arc, Mutex};

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{execute_stream, ExecutionPlan};
use datafusion::physical_plan::union::{InterleaveExec, UnionExec};


use super::executor::NodeMemoryStats;
use super::stream::{drain_stream, InstrumentedStream, StreamSourceExec};

// 
/// Delegates to the ONE dispatch point, `operators::strip_target` (Inc3). The old
/// `try_strip!` chain lived here; it is gone, not duplicated. Behavior is unchanged,
/// asymmetry included — see the per-operator `strips_to_inner` impls for which
/// wrappers deliberately pass through unstripped, and why.
pub(crate) fn strip_gpu(node: Arc<dyn ExecutionPlan>) -> (Arc<dyn ExecutionPlan>, Option<usize>) {
    crate::operators::strip_target(&node)
}

/// Apply a batch-size override to a `TaskContext`, returning the updated context.
/// The override comes from `GpuScanExec.gpu_batch_size`, which was computed by
/// `GpuMemoryBudgetRule` to keep GPU memory within budget.  We honour the same
/// limit on CPU so that peak working-set size stays within the same bound.
pub(crate) fn with_batch_size(ctx: Arc<TaskContext>, batch_size: usize) -> Arc<TaskContext> {
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

/// Execute exactly ONE plan node, fed by its children's already-computed output
/// batches (in child order), and return this node's output batches + stats.
///
/// This is the CPU backend of the unified node-executor interface
/// (`node_executor::CpuNodeExecutor`): the orchestrator drives the tree post-order
/// and hands each node its child outputs. Reuses the same machinery as the
/// recursive executor — `strip_gpu`, the `StreamSourceExec` child stubs, the
/// `InterleaveExec`→`UnionExec` substitution, and `InstrumentedStream` — so the
/// per-node `NodeMemoryStats` are byte-identical to `execute_node_by_node`.
pub(crate) async fn execute_single_node(
    node: &Arc<dyn ExecutionPlan>,
    inputs: Vec<Vec<RecordBatch>>,
    task_ctx: Arc<TaskContext>,
) -> Result<(Vec<RecordBatch>, NodeMemoryStats)> {
    let (cpu_node, batch_size_override) = strip_gpu(node.clone());
    let task_ctx = match batch_size_override {
        Some(n) => with_batch_size(task_ctx, n),
        None => task_ctx,
    };

    let children = cpu_node.children();
    if children.len() != inputs.len() {
        return Err(DataFusionError::Internal(format!(
            "execute_single_node: {} expects {} inputs, got {}",
            cpu_node.name(),
            children.len(),
            inputs.len()
        )));
    }
    let mut stubs: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(children.len());
    for (child, batches) in children.iter().zip(inputs.into_iter()) {
        let child_schema = child.schema();
        let child_eq = child.properties().equivalence_properties().clone();
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            child_schema.clone(),
            futures::stream::iter(batches.into_iter().map(Ok)),
        )) as SendableRecordBatchStream;
        stubs.push(Arc::new(StreamSourceExec::new(child_schema, child_eq, stream)));
    }

    let node_schema = cpu_node.schema();
    let node_name = cpu_node.name().to_string();
    let executable = match cpu_node.clone().with_new_children(stubs.clone()) {
        Ok(n) => n,
        Err(_) if cpu_node.as_any().is::<InterleaveExec>() => Arc::new(UnionExec::new(stubs)),
        Err(e) => return Err(e),
    };

    let inner = execute_stream(executable, task_ctx)?;
    let collector: Arc<Mutex<Vec<(usize, NodeMemoryStats)>>> = Arc::new(Mutex::new(Vec::new()));
    let instrumented = Box::pin(InstrumentedStream::new(
        0,
        node_name,
        node_schema,
        inner,
        collector.clone(),
        None,
    ));
    let batches = drain_stream(instrumented).await?;
    let stat = collector
        .lock()
        .unwrap()
        .pop()
        .map(|(_, s)| s)
        .expect("InstrumentedStream must record one stat on completion");
    Ok((batches, stat))
}

/// Σ-over-partitions accumulation of per-partition [`NodeMemoryStats`]: row counts
/// and (per-partition `ColAccum`) byte sizes ADD; `max_batch_rows` takes the max.
/// This is exactly the GPU's per-node accounting (each partition charged its own
/// overhead), so the tp8 cost golden generated here matches the real 8-way GPU.
pub(crate) fn merge_stats(acc: &mut Option<NodeMemoryStats>, s: NodeMemoryStats) {
    match acc {
        None => *acc = Some(s),
        Some(a) => {
            a.allocated_bytes += s.allocated_bytes;
            a.output_bytes += s.output_bytes;
            a.row_count += s.row_count;
            a.max_batch_rows = a.max_batch_rows.max(s.max_batch_rows);
        }
    }
}
