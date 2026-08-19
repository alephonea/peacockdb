//! The plan-time memory model: what each node holds, and how large a source's batches may
//! be for the whole plan to fit a budget.
//!
//! Two passes. The first walks from every source up to the nearest accumulator, because
//! that is exactly where resident stops scaling with batch size — a join's build side holds
//! a whole relation and an aggregate's state one row per group, so those come off the
//! budget as constants before anything is divided. The second spends what is left.
//!
//! Every cardinality here is the trivial estimate the planner has today (#19, #73): a
//! filter passes everything and a join is 1:1. Widths are real, from the declared schemas.

use std::collections::HashMap;

use datafusion::arrow::datatypes::{Fields, Schema as ArrowSchema};

use super::error::PlanError;
use super::layout::NodeKind;
use super::node::GpuNode;
use super::nodes::{NodeRef, as_node_ref};
use crate::memory::logical_size_from_schema;

/// What the `.plan.mem.txt` golden renders: a figure per node in canonical post-order, and
/// the batch size each source was given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryModel {
    pub budget: u64,
    /// Σ over the accumulators — held whatever the batch size, so it is spent first.
    pub accumulator_bytes: u64,
    /// `estimated_max_resident_size` per node, indexed by post-order sequence.
    pub resident: Vec<u64>,
    /// Post-order sequence of each source, and the batch size derived for it.
    pub target_batch_bytes: HashMap<usize, u64>,
}

/// A batch size below this is not worth deriving: the mapping is quantized to whole row
/// groups anyway, so the model would be pretending to a precision the plan cannot use.
const MIN_TARGET_BATCH_BYTES: u64 = 1 << 20;

pub fn estimate(root: &dyn GpuNode, budget: u64) -> Result<MemoryModel, PlanError> {
    let tree = Tree::of(root);
    let accumulator_bytes: u64 = (0..tree.nodes.len())
        .filter_map(|seq| tree.held_by_accumulator(seq))
        .sum();

    // If the constants alone exceed the budget no batch size helps, and saying so is the
    // point: this is the common shape of a build side larger than vram, not an edge case.
    let remainder = budget
        .checked_sub(accumulator_bytes)
        .filter(|left| *left > 0);
    let Some(remainder) = remainder else {
        return Err(PlanError::Invalid(format!(
            "what the accumulators hold ({accumulator_bytes} bytes) is already the whole \
             budget ({budget}), so no batch size makes this plan run"
        )));
    };

    let sources = tree.sources();
    // Equal shares rather than proportional ones: a proportional split hands the most
    // budget to the source already producing the most bytes.
    let share = remainder / sources.len().max(1) as u64;
    let mut target_batch_bytes = HashMap::new();
    for source in &sources {
        let amplification = tree.amplification(*source);
        let target = (share as f64 / amplification).floor() as u64;
        target_batch_bytes.insert(*source, coarse(target));
    }

    let batch_bytes = tree.batch_bytes(&target_batch_bytes);
    let resident = (0..tree.nodes.len())
        .map(|seq| match tree.held_by_accumulator(seq) {
            // A join holds its build side and is handed a probe batch per lane on top of
            // it; every other accumulator's held already is what it received.
            Some(held) => held + tree.streamed_into(seq, &batch_bytes),
            None => batch_bytes[seq] * tree.lanes(seq),
        })
        .collect();

    Ok(MemoryModel {
        budget,
        accumulator_bytes,
        resident,
        target_batch_bytes,
    })
}

/// Rounds down onto a coarse grid. The inputs are the optimizer's estimates, and an
/// estimate that drifts slightly should not regenerate every golden.
fn coarse(bytes: u64) -> u64 {
    if bytes < MIN_TARGET_BATCH_BYTES {
        return MIN_TARGET_BATCH_BYTES;
    }
    1 << (63 - bytes.leading_zeros() as u64)
}

/// The tree in canonical post-order — children left to right, then the node — which is the
/// order the goldens render and the order every per-node figure is indexed by.
struct Tree<'a> {
    nodes: Vec<&'a dyn GpuNode>,
    children: Vec<Vec<usize>>,
    parent: Vec<Option<usize>>,
}

impl<'a> Tree<'a> {
    fn of(root: &'a dyn GpuNode) -> Self {
        let mut tree = Tree {
            nodes: Vec::new(),
            children: Vec::new(),
            parent: Vec::new(),
        };
        tree.visit(root);
        for seq in 0..tree.nodes.len() {
            for child in tree.children[seq].clone() {
                tree.parent[child] = Some(seq);
            }
        }
        tree
    }

    fn visit(&mut self, node: &'a dyn GpuNode) -> usize {
        let children: Vec<usize> = node.children().into_iter().map(|c| self.visit(c)).collect();
        self.nodes.push(node);
        self.children.push(children);
        self.parent.push(None);
        self.nodes.len() - 1
    }

    fn sources(&self) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|seq| matches!(self.nodes[*seq].kind(), NodeKind::Source { .. }))
            .collect()
    }

    fn lanes(&self, seq: usize) -> u64 {
        match self.nodes[seq].kind().layout() {
            Some(layout) => layout.n as u64,
            // A sink holds nothing of its own; its input's lane count is that node's.
            None => 1,
        }
    }

    fn width(&self, seq: usize) -> u64 {
        match self.nodes[seq].kind().schema() {
            Some(schema) => logical_size_from_schema(&schema.fields, 1, 0) as u64,
            None => self.children[seq]
                .first()
                .map(|child| self.width(*child))
                .unwrap_or(1),
        }
        .max(1)
    }

    /// Rows a node emits. Selectivity and join cardinality are the trivial estimates the
    /// planner has today, so a filter passes everything and a join is 1:1 against its
    /// larger side; a limit is the one node that knows better.
    fn rows(&self, seq: usize) -> u64 {
        match as_node_ref(self.nodes[seq]) {
            NodeRef::LoadParquet(load) => load.rows,
            NodeRef::Limit(limit) => {
                let input = self.rows(self.children[seq][0]);
                match limit.interval.fetch {
                    Some(fetch) => input.min(limit.interval.skip + fetch),
                    None => input.saturating_sub(limit.interval.skip),
                }
            }
            NodeRef::Union(_) | NodeRef::Interleave(_) => {
                self.children[seq].iter().map(|c| self.rows(*c)).sum()
            }
            NodeRef::Join(_) | NodeRef::CrossJoin(_) | NodeRef::NestedLoopJoin(_) => self.children
                [seq]
                .iter()
                .map(|c| self.rows(*c))
                .max()
                .unwrap_or(0),
            _ => self.children[seq]
                .first()
                .map(|child| self.rows(*child))
                .unwrap_or(0),
        }
    }

    fn bytes(&self, seq: usize) -> u64 {
        self.rows(seq) * self.width(seq)
    }

    /// What a node holds whatever the batch size — `None` for a node whose residency
    /// scales with its input batch. A mid-plan limit is a `BatchAccumulator` by category
    /// and holds nothing at all, which is why it is not one of these.
    fn held_by_accumulator(&self, seq: usize) -> Option<u64> {
        match as_node_ref(self.nodes[seq]) {
            NodeRef::CoalesceAllBatches(_) | NodeRef::MergeSortedPartitions(_) => {
                Some(self.bytes(self.children[seq][0]))
            }
            NodeRef::AccumulateBatchesAndSort(accumulator) => {
                let input = self.children[seq][0];
                let rows = match accumulator.fetch {
                    // A top-N holds n rows per lane, which is what makes it bounded.
                    Some(fetch) => self.rows(input).min(fetch as u64 * self.lanes(input)),
                    None => self.rows(input),
                };
                Some(rows * self.width(seq))
            }
            // One row per group, and with no cardinality estimate the worst case is one
            // group per input row (#19 is what would sharpen it).
            NodeRef::AggregateBatches(_) => {
                Some(self.rows(self.children[seq][0]) * self.width(seq))
            }
            NodeRef::Join(join) => {
                let build = self.bytes(self.children[seq][0]);
                // A build-preserving join on the frozen surface also holds the key columns
                // of every probe row it has seen, per lane, until the finish pass — the
                // term that decides whether such a plan fits (#136).
                let keys = match join.capability() {
                    Ok(capability) if capability.needs_finish => {
                        let probe = self.children[seq][1];
                        self.rows(probe) * self.key_width(join, probe)
                    }
                    _ => 0,
                };
                Some(build + keys)
            }
            NodeRef::CrossJoin(_) | NodeRef::NestedLoopJoin(_) => {
                Some(self.bytes(self.children[seq][0]))
            }
            _ => None,
        }
    }

    /// What arrives at a node while it holds its state: a join's probe batch, per lane.
    fn streamed_into(&self, seq: usize, batch_bytes: &[u64]) -> u64 {
        match as_node_ref(self.nodes[seq]) {
            NodeRef::Join(_) | NodeRef::CrossJoin(_) | NodeRef::NestedLoopJoin(_) => {
                batch_bytes[self.children[seq][1]] * self.lanes(seq)
            }
            _ => 0,
        }
    }

    /// The key columns a finish pass accumulates, per probe row.
    fn key_width(&self, join: &super::nodes::GpuJoin, probe: usize) -> u64 {
        let Some(schema) = self.nodes[probe].kind().schema() else {
            return 0;
        };
        join.keys
            .iter()
            .filter_map(|(_, ordinal)| schema.fields.fields().get(*ordinal as usize))
            .map(|field| {
                let one = ArrowSchema::new(Fields::from(vec![field.as_ref().clone()]));
                logical_size_from_schema(&one, 1, 0) as u64
            })
            .sum()
    }

    /// A source's amplification: the largest its batch gets anywhere on the way to the
    /// accumulator that ends the walk, counting the lanes live at that point. The maximum
    /// rather than either end — a batch is rarely widest where it starts, and above a
    /// merge the same batch costs one lane's worth rather than N.
    fn amplification(&self, source: usize) -> f64 {
        let (rows, width) = (self.rows(source).max(1), self.width(source));
        let mut node = source;
        let mut amplification = self.lanes(source) as f64;
        while let Some(parent) = self.parent[node] {
            if self.held_by_accumulator(parent).is_some() {
                break;
            }
            let factor = (self.rows(parent) as f64 / rows as f64)
                * (self.width(parent) as f64 / width as f64)
                * self.lanes(parent) as f64;
            amplification = amplification.max(factor);
            node = parent;
        }
        amplification.max(1.0)
    }

    /// One in-flight batch at each node, bottom up: a source emits its target, an
    /// accumulator emits what it held as one batch per lane, and everything between scales
    /// its input by the rows and width it changes.
    fn batch_bytes(&self, targets: &HashMap<usize, u64>) -> Vec<u64> {
        let mut batch = vec![0u64; self.nodes.len()];
        for seq in 0..self.nodes.len() {
            batch[seq] = if let Some(target) = targets.get(&seq) {
                *target
            } else if let Some(held) = self.held_by_accumulator(seq) {
                held / self.lanes(seq).max(1)
            } else {
                self.children[seq]
                    .iter()
                    .map(|child| {
                        let child_bytes = self.bytes(*child).max(1);
                        (batch[*child] as f64 * self.bytes(seq) as f64 / child_bytes as f64) as u64
                    })
                    .max()
                    .unwrap_or(0)
            };
        }
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_partitioned::partitioner::Batching;
    use crate::batch_partitioned::translate::Translator;
    use std::path::PathBuf;

    const BUDGET: u64 = 2 * 1024 * 1024 * 1024;

    async fn modelled(sql: &str, target_partitions: usize, budget: u64) -> (Vec<u64>, MemoryModel) {
        let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal");
        let ctx = crate::register_tables_for(crate::build_session_state(target_partitions), &data)
            .await
            .expect("register the minimal tables");
        let plan = ctx
            .sql(sql)
            .await
            .expect("plan the query")
            .create_physical_plan()
            .await
            .expect("physical plan");
        let tree = Translator::new(
            target_partitions,
            Batching::On {
                target_batch_bytes: 1 << 20,
            },
        )
        .translate(&plan)
        .expect("translate the plan");
        let model = estimate(tree.as_ref(), budget).expect("estimate the plan");
        let mut targets: Vec<u64> = model.target_batch_bytes.values().copied().collect();
        targets.sort_unstable();
        (targets, model)
    }

    async fn refused(sql: &str, target_partitions: usize, budget: u64) -> PlanError {
        let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/tpch.minimal");
        let ctx = crate::register_tables_for(crate::build_session_state(target_partitions), &data)
            .await
            .expect("register the minimal tables");
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let tree = Translator::new(
            target_partitions,
            Batching::On {
                target_batch_bytes: 1 << 20,
            },
        )
        .translate(&plan)
        .expect("translate the plan");
        estimate(tree.as_ref(), budget).expect_err("this plan should not fit")
    }

    #[tokio::test]
    async fn what_the_accumulators_hold_comes_off_the_budget_first() {
        let (targets, model) = modelled(
            "SELECT c_nationkey, count(*) FROM customer GROUP BY c_nationkey",
            4,
            BUDGET,
        )
        .await;
        // The aggregate's state is held whatever the batch size, so it is spent before
        // anything is divided, and what is left is what the source may spend.
        assert!(model.accumulator_bytes > 0);
        assert_eq!(targets.len(), 1);
        assert!(targets[0] <= BUDGET - model.accumulator_bytes);
    }

    #[tokio::test]
    async fn constants_over_the_budget_are_a_plan_time_error_not_a_plan_that_cannot_run() {
        // A build side larger than vram is the common shape of this, not an edge case.
        let err = refused(
            "SELECT c_nationkey, count(*) FROM customer GROUP BY c_nationkey",
            4,
            1 << 16,
        )
        .await;
        assert!(
            matches!(&err, PlanError::Invalid(what) if what.contains("no batch size")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn a_batch_is_charged_once_per_lane_in_force() {
        let (one_lane, _) = modelled("SELECT c_nationkey FROM customer", 1, BUDGET).await;
        let (four_lanes, _) = modelled("SELECT c_nationkey FROM customer", 4, BUDGET).await;
        // Four lanes hold four batches at once, so each may be a quarter the size.
        assert_eq!(one_lane[0], four_lanes[0] * 4);
    }

    #[tokio::test]
    async fn amplification_is_the_widest_point_on_the_path_not_either_end() {
        // The aggregate above the scan is where a batch is widest — its output row carries
        // a key and a count where the input row carried a key — and that is what sizes the
        // source, though it is neither end of the walk.
        let (plain, _) = modelled("SELECT c_nationkey FROM customer", 4, BUDGET).await;
        let (widened, _) = modelled(
            "SELECT c_nationkey, count(*) FROM customer GROUP BY c_nationkey",
            4,
            BUDGET,
        )
        .await;
        assert!(
            widened[0] < plain[0],
            "the widening above the source did not size it: {widened:?} vs {plain:?}"
        );
    }

    #[tokio::test]
    async fn an_accumulator_ends_the_walk() {
        // Everything above the aggregate's merge is served by what the merge emits, so a
        // project up there cannot change what the source may spend.
        let (bare, _) = modelled(
            "SELECT c_nationkey, count(*) AS n FROM customer GROUP BY c_nationkey",
            4,
            BUDGET,
        )
        .await;
        let (with_project, _) = modelled(
            "SELECT c_nationkey, count(*) * 1000 AS n FROM customer GROUP BY c_nationkey",
            4,
            BUDGET,
        )
        .await;
        assert_eq!(bare, with_project);
    }

    #[tokio::test]
    async fn two_sources_get_equal_shares_rather_than_proportional_ones() {
        let (targets, _) = modelled(
            "SELECT c.c_name, s.s_name FROM customer c JOIN supplier s ON c.c_nationkey = s.s_nationkey",
            4,
            BUDGET,
        )
        .await;
        // customer is 150k rows and supplier 10k; a proportional split would hand the most
        // budget to the source already producing the most bytes.
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0], targets[1]);
    }

    #[tokio::test]
    async fn a_target_lands_on_the_coarse_grid() {
        let (targets, _) = modelled(
            "SELECT c.c_name, s.s_name FROM customer c JOIN supplier s ON c.c_nationkey = s.s_nationkey",
            4,
            BUDGET,
        )
        .await;
        // Powers of two, and never below the granularity the mapping can express — an
        // estimate that drifts slightly must not regenerate every golden.
        for target in targets {
            assert!(target.is_power_of_two(), "{target} is not on the grid");
            assert!(target >= MIN_TARGET_BATCH_BYTES);
        }
    }

    #[tokio::test]
    async fn a_build_preserving_join_is_charged_for_the_keys_it_accumulates() {
        // The finish pass holds the key columns of every probe row it has seen (#136),
        // and the small table on the left is what keeps this a Left join — DataFusion
        // swaps the sides, and remaps the type, when the right one is smaller.
        let (_, left) = modelled(
            "SELECT s.s_name, c.c_name FROM supplier s LEFT JOIN customer c ON s.s_nationkey = c.c_nationkey",
            4,
            BUDGET,
        )
        .await;
        let (_, inner) = modelled(
            "SELECT s.s_name, c.c_name FROM supplier s JOIN customer c ON s.s_nationkey = c.c_nationkey",
            4,
            BUDGET,
        )
        .await;
        assert!(
            left.accumulator_bytes > inner.accumulator_bytes,
            "a streamed probe under a build-preserving join costs nothing: {} vs {}",
            left.accumulator_bytes,
            inner.accumulator_bytes
        );
    }

    #[tokio::test]
    async fn a_limit_bounds_what_the_nodes_above_it_hold() {
        let (_, limited) = modelled(
            "SELECT c_name FROM (SELECT * FROM customer WHERE c_nationkey > 1 LIMIT 10) t \
             ORDER BY c_name",
            1,
            BUDGET,
        )
        .await;
        let (_, whole) = modelled(
            "SELECT c_name FROM customer WHERE c_nationkey > 1 ORDER BY c_name",
            1,
            BUDGET,
        )
        .await;
        assert!(
            limited.accumulator_bytes < whole.accumulator_bytes,
            "the sort above a limit accumulates the whole table: {} vs {}",
            limited.accumulator_bytes,
            whole.accumulator_bytes
        );
    }
}
