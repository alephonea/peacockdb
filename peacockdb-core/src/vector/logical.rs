//! `VectorTopK` — the logical node a `Sort(l2_distance)+Limit` collapses into
//! (see [`super::analyzer`]). It scores its input by `distance` and keeps the `k`
//! nearest rows; output schema == input schema (a row selection, same columns).

use std::fmt;

use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};

/// Top-k nearest rows of `input` under vector distance `distance` (an
/// `l2_distance(embedding, query)` call). `query`/`dim` are the query vector's
/// little-endian element bytes + dimensionality, carried for lowering into the
/// serialized `GpuVectorSearch` IR (the CPU exec scores via `distance` directly).
#[derive(PartialEq, Eq, PartialOrd, Hash, Debug)]
pub struct VectorTopK {
    input: LogicalPlan,
    distance: Expr,
    k: usize,
    query: Vec<u8>,
    dim: u32,
}

impl VectorTopK {
    pub fn new(input: LogicalPlan, distance: Expr, k: usize, query: Vec<u8>, dim: u32) -> Self {
        Self { input, distance, k, query, dim }
    }

    pub fn input(&self) -> &LogicalPlan {
        &self.input
    }
    pub fn distance(&self) -> &Expr {
        &self.distance
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

impl UserDefinedLogicalNodeCore for VectorTopK {
    fn name(&self) -> &str {
        "VectorTopK"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        // A row selection: same columns as the input.
        self.input.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.distance.clone()]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VectorTopK: k={}, metric=L2, distance={}", self.k, self.distance)
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            input: inputs.swap_remove(0),
            distance: exprs.into_iter().next().unwrap_or_else(|| self.distance.clone()),
            k: self.k,
            query: self.query.clone(),
            dim: self.dim,
        })
    }
}
