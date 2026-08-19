//! The aggregate vocabulary the schema annotations refer to. `AggFunc` is what sql
//! asked for; the decomposition into what a node runs (`PlanAgg`) lands with the
//! planner that needs it — see the spec's Aggregators section.

/// What sql asked for, which is what a merge must agree with: a `sum` merged as if it
/// were a `mean`'s sum-half reads the right column and computes nonsense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Sum,
    Min,
    Max,
    Count,
    Avg,
    Stddev,
    Var,
}
