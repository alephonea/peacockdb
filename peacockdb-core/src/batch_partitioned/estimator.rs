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
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryModel {
    pub budget: u64,
    /// Σ over the accumulators — held whatever the batch size, so it is spent first.
    pub accumulator_bytes: u64,
    /// The part of it that cannot be an overestimate — a build side is its input's rows,
    /// where an aggregate's state rests on a cardinality estimate. Only this part can
    /// refuse a plan.
    pub certain_accumulator_bytes: u64,
    /// What each source may spend, before its own amplification and size narrow it.
    pub share_per_source: u64,
    /// `estimated_max_resident_size` per node, indexed by post-order sequence.
    pub resident: Vec<u64>,
    /// One per source, in post-order sequence — which is the order translation reaches
    /// them, so the second pass consumes them in this order.
    pub sources: Vec<SourceEstimate>,
}

/// What the walk from one source found, and what it was given for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceEstimate {
    pub seq: usize,
    /// The largest a batch from this source gets on its way to the accumulator that ends
    /// the walk, counting the lanes live at that point.
    pub amplification: f64,
    pub target_batch_bytes: u64,
}

/// A batch size below this is not worth deriving: the mapping is quantized to whole row
/// groups anyway, so the model would be pretending to a precision the plan cannot use.
pub(crate) const MIN_TARGET_BATCH_BYTES: u64 = 1 << 20;

pub fn estimate(root: &dyn GpuNode, budget: u64) -> Result<MemoryModel, PlanError> {
    let tree = Tree::of(root);
    let accumulator_bytes: u64 = (0..tree.nodes.len())
        .filter_map(|seq| tree.held_by_accumulator(seq))
        .sum();

    let certain_accumulator_bytes: u64 = (0..tree.nodes.len())
        .filter(|seq| tree.rows_are_certain(*seq))
        .filter_map(|seq| tree.held_by_accumulator(seq))
        .sum();

    // The planner refuses on a constant only where the constant cannot be an overestimate.
    // A build side over a scan is its input's rows; an aggregate's state rests on the
    // cardinality estimate, which today says one row per input row, so refusing on it
    // would turn "we do not know" into "you cannot run this".
    if certain_accumulator_bytes >= budget {
        return Err(PlanError::Invalid(format!(
            "what the accumulators hold and cannot be holding less of \
             ({certain_accumulator_bytes} bytes) is already the whole budget ({budget}), so \
             no batch size makes this plan run"
        )));
    }
    // Where the estimated constants alone exhaust the budget, batches are sized against the
    // ones we know and the enforcer owns the rest.
    let remainder = budget
        .checked_sub(accumulator_bytes)
        .filter(|left| *left > 0)
        .unwrap_or(budget - certain_accumulator_bytes);

    // Equal shares rather than proportional ones: a proportional split hands the most
    // budget to the source already producing the most bytes.
    let seqs = tree.sources();
    let share_per_source = remainder / seqs.len().max(1) as u64;
    let sources: Vec<SourceEstimate> = seqs
        .iter()
        .map(|seq| {
            let amplification = tree.amplification(*seq);
            let afforded = coarse((share_per_source as f64 / amplification).floor() as u64);
            SourceEstimate {
                seq: *seq,
                amplification,
                // A batch is never larger than what the source holds — a scan pruned to a
                // few row groups is where that is orders of magnitude.
                target_batch_bytes: afforded.min(tree.source_bytes(*seq)),
            }
        })
        .collect();

    let targets: HashMap<usize, u64> = sources
        .iter()
        .map(|source| (source.seq, source.target_batch_bytes))
        .collect();
    let batch_bytes = tree.batch_bytes(&targets);
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
        certain_accumulator_bytes,
        share_per_source,
        resident,
        sources,
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

    /// Whether a node's row count is a fact rather than an estimate. A filter's
    /// selectivity, an aggregate's group count and a join's cardinality are all the
    /// trivial estimates the planner has today (#19, #73), and each of them can only
    /// overstate what its subtree really produces.
    fn rows_are_certain(&self, seq: usize) -> bool {
        let guessed = matches!(
            as_node_ref(self.nodes[seq]),
            NodeRef::Filter(_)
                | NodeRef::Aggregate(_)
                | NodeRef::AggregateBatches(_)
                | NodeRef::Join(_)
                | NodeRef::CrossJoin(_)
                | NodeRef::NestedLoopJoin(_)
        );
        !guessed && self.children[seq].iter().all(|c| self.rows_are_certain(*c))
    }

    /// What a source holds in total, over the columns it projects.
    fn source_bytes(&self, seq: usize) -> u64 {
        match as_node_ref(self.nodes[seq]) {
            NodeRef::LoadParquet(load) => load.bytes().max(1),
            _ => u64::MAX,
        }
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
            NodeRef::LoadParquet(load) => load.rows(),
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
        let _ = targets;
        let mut batch = vec![0u64; self.nodes.len()];
        for seq in 0..self.nodes.len() {
            batch[seq] = if let NodeRef::LoadParquet(load) = as_node_ref(self.nodes[seq]) {
                // What this mapping emits, which is the budget's target only where the
                // budget is what cut the batches.
                load.largest_batch_bytes()
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
    use crate::batch_partitioned::plan::BatchSizing;
    use crate::batch_partitioned::translate::Translator;
    use std::path::PathBuf;

    const BUDGET: u64 = 2 * 1024 * 1024 * 1024;

    async fn modelled(sql: &str, target_partitions: usize, budget: u64) -> MemoryModel {
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
            Batching::Sized {
                target_batch_bytes: 1 << 20,
            },
        )
        .translate(&plan)
        .expect("translate the plan");
        estimate(tree.as_ref(), budget).expect("estimate the plan")
    }

    /// Planned end to end at one of the three batching forms, which is what decides the
    /// mapping the model then prices.
    async fn modelled_as(
        sql: &str,
        target_partitions: usize,
        sizing: crate::batch_partitioned::plan::BatchSizing,
    ) -> MemoryModel {
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
        crate::batch_partitioned::plan::plan_batch_partitioned(
            &plan,
            crate::batch_partitioned::plan::PlanKnobs {
                target_partitions,
                sizing,
                budget: BUDGET,
                small_table_bytes: 0,
            },
        )
        .expect("plan and estimate")
        .1
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
            Batching::Sized {
                target_batch_bytes: 1 << 20,
            },
        )
        .translate(&plan)
        .expect("translate the plan");
        estimate(tree.as_ref(), budget).expect_err("this plan should not fit")
    }

    #[tokio::test]
    async fn what_the_accumulators_hold_comes_off_the_budget_first() {
        let model = modelled(
            "SELECT c_nationkey, count(*) FROM customer GROUP BY c_nationkey",
            4,
            BUDGET,
        )
        .await;
        // The aggregate's state is held whatever the batch size, so it is spent before
        // anything is divided, and what is left is what the source may spend.
        assert!(model.accumulator_bytes > 0);
        assert_eq!(model.share_per_source, BUDGET - model.accumulator_bytes);
    }

    #[tokio::test]
    async fn constants_that_cannot_be_less_and_exceed_the_budget_are_a_plan_time_error() {
        // A sort over a plain scan holds the whole table, and no estimate stands between
        // that number and the truth — which is the shape of a build side larger than vram.
        let err = refused("SELECT * FROM customer ORDER BY c_name", 1, 1 << 16).await;
        assert!(
            matches!(&err, PlanError::Invalid(what) if what.contains("no batch size")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn a_constant_that_rests_on_an_estimate_does_not_refuse_a_plan() {
        // An aggregate's state is one row per input row only because there is no
        // cardinality estimate (#19); tpch q1 groups six million rows into four. Refusing
        // on that would turn "we do not know" into "you cannot run this".
        let model = modelled(
            "SELECT c_nationkey, count(*) FROM customer GROUP BY c_nationkey",
            4,
            1 << 16,
        )
        .await;
        assert!(model.accumulator_bytes > model.budget);
        assert_eq!(model.certain_accumulator_bytes, 0);
        assert!(model.sources[0].target_batch_bytes > 0);
    }

    #[tokio::test]
    async fn a_batch_is_charged_once_per_lane_in_force() {
        let one_lane = modelled("SELECT c_nationkey FROM customer", 1, BUDGET).await;
        let four_lanes = modelled("SELECT c_nationkey FROM customer", 4, BUDGET).await;
        // Four lanes hold four batches at once, so each may be a quarter the size.
        assert_eq!(
            four_lanes.sources[0].amplification,
            one_lane.sources[0].amplification * 4.0
        );
    }

    #[tokio::test]
    async fn amplification_is_the_widest_point_on_the_path_not_either_end() {
        // The aggregate above the scan is where a batch is widest — its output row carries
        // a key and a count where the input row carried a key — and that is what sizes the
        // source, though it is neither end of the walk.
        let plain = modelled("SELECT c_nationkey FROM customer", 4, BUDGET).await;
        let widened = modelled(
            "SELECT c_nationkey, count(*) FROM customer GROUP BY c_nationkey",
            4,
            BUDGET,
        )
        .await;
        assert!(
            widened.sources[0].amplification > plain.sources[0].amplification,
            "the widening above the source did not size it: {} vs {}",
            widened.sources[0].amplification,
            plain.sources[0].amplification
        );
    }

    #[tokio::test]
    async fn an_accumulator_ends_the_walk() {
        // Everything above the aggregate's merge is served by what the merge emits, so a
        // project up there cannot change what the source may spend.
        let bare = modelled(
            "SELECT c_nationkey, count(*) AS n FROM customer GROUP BY c_nationkey",
            4,
            BUDGET,
        )
        .await;
        let with_project = modelled(
            "SELECT c_nationkey, count(*) * 1000 AS n FROM customer GROUP BY c_nationkey",
            4,
            BUDGET,
        )
        .await;
        assert_eq!(bare.sources, with_project.sources);
    }

    #[tokio::test]
    async fn two_sources_get_equal_shares_rather_than_proportional_ones() {
        let model = modelled(
            "SELECT c.c_name, s.s_name FROM customer c JOIN supplier s ON c.c_nationkey = s.s_nationkey",
            4,
            BUDGET,
        )
        .await;
        // customer is 150k rows and supplier 10k; a proportional split would hand the most
        // budget to the source already producing the most bytes. Both get the same share,
        // and what separates their targets afterwards is their own size.
        assert_eq!(model.sources.len(), 2);
        assert_eq!(
            model.share_per_source,
            (BUDGET - model.accumulator_bytes) / 2
        );
        for source in &model.sources {
            assert!(
                source.target_batch_bytes as f64 <= model.share_per_source as f64,
                "a source was given more than its share"
            );
        }
    }

    #[tokio::test]
    async fn a_target_lands_on_the_coarse_grid() {
        // The share these sources can afford is far above what they hold, so each target
        // is the source's own size: a batch is never larger than what there is to read.
        let capped = modelled(
            "SELECT c.c_name, s.s_name FROM customer c JOIN supplier s ON c.c_nationkey = s.s_nationkey",
            4,
            BUDGET,
        )
        .await;
        for source in &capped.sources {
            assert!(source.target_batch_bytes < capped.share_per_source);
        }
        // Where the budget is what binds, the target lands on the grid — powers of two,
        // so an estimate that drifts slightly does not regenerate every golden.
        let bound = modelled("SELECT * FROM customer", 4, 64 * 1024 * 1024).await;
        let target = bound.sources[0].target_batch_bytes;
        assert!(target.is_power_of_two(), "{target} is not on the grid");
        assert!(target >= MIN_TARGET_BATCH_BYTES);
    }

    #[tokio::test]
    async fn a_build_preserving_join_is_charged_for_the_keys_it_accumulates() {
        // The finish pass holds the key columns of every probe row it has seen (#136),
        // and the small table on the left is what keeps this a Left join — DataFusion
        // swaps the sides, and remaps the type, when the right one is smaller.
        let left = modelled(
            "SELECT s.s_name, c.c_name FROM supplier s LEFT JOIN customer c ON s.s_nationkey = c.c_nationkey",
            4,
            BUDGET,
        )
        .await;
        let inner = modelled(
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
    async fn a_loader_is_priced_by_the_batches_its_mapping_makes() {
        // Three batching forms, one budget: what a loader holds is the largest batch the
        // mapping produces, so per-row-group holds a row group where one-batch-per-lane
        // holds the lane. A model reading the budget's target instead would say the same
        // number for all three.
        let per_lane = modelled_as("SELECT * FROM customer", 1, BatchSizing::OneBatchPerLane).await;
        let per_group = modelled_as(
            "SELECT * FROM customer",
            1,
            BatchSizing::OneBatchPerRowGroup,
        )
        .await;
        assert!(
            per_group.resident[0] < per_lane.resident[0],
            "per-row-group holds {} where per-lane holds {}",
            per_group.resident[0],
            per_lane.resident[0]
        );
    }

    #[tokio::test]
    async fn a_limit_bounds_what_the_nodes_above_it_hold() {
        let limited = modelled(
            "SELECT c_name FROM (SELECT * FROM customer WHERE c_nationkey > 1 LIMIT 10) t \
             ORDER BY c_name",
            1,
            BUDGET,
        )
        .await;
        let whole = modelled(
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
