//! `GpuVectorSearchExec` — the physical node `VectorTopK` lowers to. The MVP
//! executes on CPU (strategy pinned to ExactBrute): score every input row's
//! embedding against the query via the `distance` physical expr (which routes
//! through `vector::cpu`), keep the `k` nearest with a `BinaryHeap`, emit them
//! nearest-first. The real GPU/cuVS path replaces `execute()` in a later ticket;
//! the serialized `GpuVectorSearch` IR is what crosses to C++ then.

use std::any::Any;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{Array, Float32Array, RecordBatch, UInt32Array};
use datafusion::arrow::compute::{concat_batches, take};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{internal_err, DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};

/// A (distance, row) pair ordered as a max-heap by distance (ties broken by row
/// for determinism), so a size-k heap that pops its max keeps the k smallest.
/// `f32::total_cmp` gives a total order without pulling in an ordered-float crate.
struct Scored {
    dist: f32,
    row: u32,
}
impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist).then(self.row.cmp(&other.row))
    }
}

#[derive(Debug)]
pub struct GpuVectorSearchExec {
    input: Arc<dyn ExecutionPlan>,
    /// The scoring expression, `l2_distance(embedding, query)`. `None` on a node
    /// reconstructed from IR bytes (deserialize) — that carries the immediate
    /// query/dim for re-serialization but is not runnable (see module docs).
    distance: Option<Arc<dyn PhysicalExpr>>,
    k: usize,
    /// Query vector, little-endian element bytes (dim * sizeof(F16)) — for the
    /// serialized IR. Empty when the query wasn't a literal.
    query: Vec<u8>,
    dim: u32,
    cache: PlanProperties,
}

impl GpuVectorSearchExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        distance: Option<Arc<dyn PhysicalExpr>>,
        k: usize,
        query: Vec<u8>,
        dim: u32,
    ) -> Self {
        let cache = PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self { input, distance, k, query, dim, cache }
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn query(&self) -> &[u8] {
        &self.query
    }
    pub fn dim(&self) -> u32 {
        self.dim
    }
}

impl DisplayAs for GpuVectorSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GpuVectorSearchExec: k={}, metric=L2, strategy=ExactBrute", self.k)
    }
}

impl ExecutionPlan for GpuVectorSearchExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.distance.clone(),
            self.k,
            self.query.clone(),
            self.dim,
        )))
    }

    fn name(&self) -> &str {
        "GpuVectorSearchExec"
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("GpuVectorSearchExec has a single output partition");
        }
        let Some(distance) = self.distance.clone() else {
            return internal_err!(
                "GpuVectorSearchExec reconstructed from IR is not executable (no distance expr)"
            );
        };
        let input = self.input.clone();
        let schema = self.schema();
        let k = self.k;

        let fut = async move {
            // Brute force: pull the whole (single-partition) input, score every row.
            let batches =
                datafusion::physical_plan::common::collect(input.execute(0, context)?).await?;
            let combined = concat_batches(&input.schema(), &batches)?;
            let n = combined.num_rows();

            let scores = distance
                .evaluate(&combined)?
                .into_array(n)?;
            let scores = scores
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    DataFusionError::Execution(
                        "GpuVectorSearchExec: distance did not evaluate to Float32".to_string(),
                    )
                })?;

            // Max-heap of size k -> the k smallest distances survive. A null score
            // (shouldn't occur for valid vectors) sorts last.
            let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(k + 1);
            for row in 0..n {
                let dist = if scores.is_null(row) {
                    f32::INFINITY
                } else {
                    scores.value(row)
                };
                heap.push(Scored { dist, row: row as u32 });
                if heap.len() > k {
                    heap.pop();
                }
            }

            // Emit nearest-first (ascending distance), matching ORDER BY ... ASC LIMIT k.
            let mut top = heap.into_vec();
            top.sort();
            let indices = UInt32Array::from(top.iter().map(|s| s.row).collect::<Vec<_>>());
            let cols = combined
                .columns()
                .iter()
                .map(|c| take(c, &indices, None))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let out = RecordBatch::try_new(combined.schema(), cols)?;
            Ok(out)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}
