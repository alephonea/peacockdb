//! Strict resident "GPU"-memory control.
//!
//! Simulates a fixed GPU memory budget. Streaming operators (filter, project,
//! coalesce, repartition, sort-preserving merge, bounded window) process
//! batch-by-batch and release their input as they go, so their resident
//! footprint is bounded by a batch. PIPELINE BREAKERS must hold a full
//! materialized data set resident at once:
//!   - `SortExec` — the whole input is buffered to sort it,
//!   - `AggregateExec` — the group hash table holds every output group,
//!   - `HashJoinExec` — the entire BUILD side is hashed and held while the probe
//!     side streams,
//!   - `CrossJoinExec` / `NestedLoopJoinExec` — the left side is held resident.
//!
//! If a breaker's resident requirement exceeds the budget the query would OOM on
//! the GPU regardless of batch size (a 10 GiB sort can't fit a 2 GiB budget). The
//! resident size uses the SAME per-node logical basis as the `output_bytes`
//! metric (deterministic, schema+rows derived — see `memory::ColAccum`),
//! NOT `allocated_bytes`/`get_array_memory_size`, which is allocation-padded and
//! non-deterministic.
//!
//! Peak resident is the PATH-SUM of concurrently-live breakers: a join's build
//! side stays charged while its probe subtree (possibly nested joins) runs, so
//! their build sides stack; sort/aggregate are sequential with their descendants.
//! See [`ResidentNode::peak`].
//!
//! CAVEAT — window functions: `WindowAggExec` is classified as streaming (charge
//! 0), which holds for the BOUNDED window variant only. An UNBOUNDED
//! `WindowAggExec` buffers its whole partition resident; if a window query ever
//! enters the OOM/tight-budget set, this would UNDER-count the peak and the
//! classification must be revisited.

use std::sync::Arc;

use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::ExecutionPlan;

use crate::cpu_executor::NodeMemoryStats;

/// A plan node paired with its post-order execution stats and children, mirroring
/// the (plan, stats) lockstep used by the cpu cost-tree formatter.
///
/// `pub(crate)` so the in-engine enforcer (`executors::stream`) builds the SAME node
/// tree and calls the SAME [`ResidentNode::peak`] — the offline reference path
/// ([`peak_resident`]) and the mid-run enforcement must never drift.
pub(crate) struct ResidentNode {
    /// Stripped CPU node name (e.g. `SortExec`, `AggregateExec`, `HashJoinExec`).
    pub(crate) name: String,
    /// Logical bytes this node materialized as output (telescoped `ColAccum`).
    pub(crate) output_bytes: usize,
    pub(crate) children: Vec<ResidentNode>,
}

impl ResidentNode {
    /// Peak concurrently-resident bytes over this subtree (PATH-SUM model):
    /// resident data sets that are alive AT THE SAME TIME add up.
    ///
    /// - Hash/cross/nested-loop join: the BUILD side (first child) is fully
    ///   materialized and held resident WHILE the probe side (second child)
    ///   streams — and the probe subtree may itself contain nested joins whose
    ///   build sides are live concurrently. So the probe phase stacks:
    ///   `build_bytes + probe.peak()`. The build phase (before the table is
    ///   complete) costs only `build.peak()`. Peak = max of the two phases.
    /// - Sort / aggregate: buffers its whole input (or group table); the input
    ///   subtree has finished and released by the time the buffer is full, so
    ///   these are SEQUENTIAL with their descendants: `max(child.peak(),
    ///   output_bytes)` — not stacked.
    /// - Everything else streams: `max` over children, contributing nothing
    ///   itself.
    pub(crate) fn peak(&self) -> usize {
        let child_peak = |i: usize| self.children.get(i).map_or(0, |c| c.peak());
        match self.name.as_str() {
            "HashJoinExec" | "CrossJoinExec" | "NestedLoopJoinExec" => {
                let build_bytes = self.children.first().map_or(0, |c| c.output_bytes);
                child_peak(0).max(build_bytes + child_peak(1))
            }
            // CoalescePartitionsExec at tp>1 CONCATENATES all input partitions into
            // one resident table — a buffering breaker, not streaming. Like
            // Sort/Aggregate it's sequential with its descendants (the inputs are done
            // and released as the concat completes), so `max(child.peak(),
            // output_bytes)`. At single-partition (tp1) its output flows through ≤ the
            // child peak, so this is a no-op there.
            "SortExec" | "AggregateExec" | "CoalescePartitionsExec" => {
                child_peak(0).max(self.output_bytes)
            }
            _ => self.children.iter().map(|c| c.peak()).max().unwrap_or(0),
        }
    }
}

/// Build the (plan, post-order stats) lockstep tree. `idx` walks `stats` in the
/// same order the executor pushed them: all children before their parent.
fn build(plan: &Arc<dyn ExecutionPlan>, stats: &[NodeMemoryStats], idx: &mut usize) -> ResidentNode {
    let children: Vec<ResidentNode> = plan.children().iter().map(|c| build(c, stats, idx)).collect();
    let stat = &stats[*idx];
    *idx += 1;
    ResidentNode {
        name: stat.node_name.clone(),
        output_bytes: stat.output_bytes,
        children,
    }
}

/// Peak concurrently-resident bytes for the plan (path-sum model; see
/// [`ResidentNode::peak`]).
pub fn peak_resident(plan: &Arc<dyn ExecutionPlan>, stats: &[NodeMemoryStats]) -> usize {
    let mut idx = 0;
    build(plan, stats, &mut idx).peak()
}

/// Peak resident over a skeleton (seq -> (node_name, child seqs)) and a partial
/// map of completed nodes' output_bytes (missing = not yet materialized = 0).
/// Used by the in-engine enforcer: it shares THIS path-sum logic so the mid-run
/// verdict matches [`peak_resident`] exactly. Because output_bytes is only ever
/// added (never removed) as nodes complete, the value grows monotonically toward
/// the true peak, so a budget crossing is detected the moment it actually occurs.
pub(crate) fn peak_from_skeleton(
    root: usize,
    skeleton: &std::collections::HashMap<usize, (String, Vec<usize>)>,
    output_bytes: &std::collections::HashMap<usize, usize>,
) -> usize {
    fn node(
        seq: usize,
        skel: &std::collections::HashMap<usize, (String, Vec<usize>)>,
        ob: &std::collections::HashMap<usize, usize>,
    ) -> ResidentNode {
        let (name, kids) = &skel[&seq];
        ResidentNode {
            name: name.clone(),
            output_bytes: ob.get(&seq).copied().unwrap_or(0),
            children: kids.iter().map(|k| node(*k, skel, ob)).collect(),
        }
    }
    node(root, skeleton, output_bytes).peak()
}

/// `Ok(())` if the plan's peak resident fits the budget, else `ResourcesExhausted`
/// with the bytes-vs-budget in the message so the `_error` tests assert a
/// meaningful OOM rather than a bare failure.
pub fn check_resident_budget(
    plan: &Arc<dyn ExecutionPlan>,
    stats: &[NodeMemoryStats],
    budget: usize,
) -> Result<()> {
    let peak = peak_resident(plan, stats);
    if peak > budget {
        return Err(DataFusionError::ResourcesExhausted(format!(
            "resident GPU memory budget exceeded: peak {peak} bytes > budget {budget} bytes"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ResidentNode;

    fn node(name: &str, output_bytes: usize, children: Vec<ResidentNode>) -> ResidentNode {
        ResidentNode { name: name.into(), output_bytes, children }
    }
    fn scan(ob: usize) -> ResidentNode {
        node("ParquetExec", ob, vec![])
    }

    #[test]
    fn nested_join_build_sides_stack() {
        // outer build (100) held while the probe subtree — itself a join with
        // build (50) — runs: 100 + 50 = 150.
        let inner = node("HashJoinExec", 5, vec![scan(50), scan(3)]);
        let outer = node("HashJoinExec", 10, vec![scan(100), inner]);
        assert_eq!(outer.peak(), 150);
    }

    #[test]
    fn join_build_on_build_side_does_not_stack() {
        // A nested join on the BUILD side runs to completion (and releases) before
        // the outer build table is held → max(build phase, probe phase), not sum.
        let inner = node("HashJoinExec", 90, vec![scan(120), scan(4)]); // build phase peak 120
        let outer = node("HashJoinExec", 10, vec![inner, scan(7)]); // probe phase = 90 + 0
        assert_eq!(outer.peak(), 120);
    }

    #[test]
    fn sort_is_sequential_with_descendants() {
        // Sort buffers its input; the descendant join has finished & released by the
        // time the buffer is full → max(child.peak, output), NOT summed.
        let j = node("HashJoinExec", 10, vec![scan(100), scan(3)]); // peak 100
        let sort = node("SortExec", 20, vec![j]);
        assert_eq!(sort.peak(), 100);
    }

    #[test]
    fn aggregate_buffer_stacks_under_an_ancestor_join() {
        // Agg group table (40) is live on the probe path while the outer join holds
        // its build (80): 80 + 40 = 120.
        let agg = node("AggregateExec", 40, vec![scan(200)]);
        let join = node("HashJoinExec", 5, vec![scan(80), agg]);
        assert_eq!(join.peak(), 120);
    }

    #[test]
    fn streaming_ops_contribute_zero() {
        let plan = node(
            "GpuCoalesceBatchesExec",
            999,
            vec![node("FilterExec", 999, vec![scan(123)])],
        );
        assert_eq!(plan.peak(), 0);
    }
}
