//! DataFusion physical plan → the mode's node tree.
//!
//! A conscious decision per DataFusion node kind: nothing is carried over implicitly and
//! an unrecognized node is a plan-time error naming it. What is reused is DataFusion's
//! planning — the coercions, the decimal scales, the per-aggregate state schemas — and
//! not its execution semantics, which is what annotating a tree with wrappers carries
//! along by accident.

use std::sync::Arc;

use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::datasource::physical_plan::ParquetExec;
use datafusion::physical_expr::{LexOrdering, PhysicalExpr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::Partitioning;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::utils::JoinFilter;
use datafusion::physical_plan::joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec};
use datafusion::physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion::physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::union::{InterleaveExec, UnionExec};
use datafusion::physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};

use super::error::PlanError;
use super::expr::{Expr, NamedExpr};
use super::expr_translate::translate_expr;
use super::layout::{BatchLayout, ColumnOrder, KeyDistribution};
use super::node::{GpuNode, RowInterval};
use super::nodes::join::{JoinFilterColumn, JoinSide, NestedLoopJoinType, capability};
use super::nodes::{
    GpuAccumulateBatchesAndSort, GpuCoalesceAllBatches, GpuCrossJoin, GpuEmitPartitions, GpuFilter,
    GpuInterleave, GpuJoin, GpuLimit, GpuLoadParquet, GpuMergePartitions, GpuMergeSortedPartitions,
    GpuNestedLoopJoin, GpuProject, GpuSort, GpuUnion, GpuUnload,
};
use super::parquet_meta::survivor_metadata;
use super::partitioner::{Batching, RowGroupMeta, partition};
use super::schema::Schema;
use crate::operators::scan::parquet_table_name;

mod aggregate;
#[cfg(test)]
mod tests;

pub struct Translator {
    /// Lanes a source is partitioned into, before the small-table rule.
    pub target_partitions: usize,
    pub batching: Batching,
    /// Batch sizes the estimator derived, one per source in the order translation reaches
    /// them. Empty on the first pass, when there is nothing derived yet.
    source_targets: Vec<u64>,
    next_source: std::cell::Cell<usize>,
    /// A source reading fewer bytes than this plans one lane whatever the target, and the
    /// nodes a one-lane region does not need are then not emitted at all. Bytes rather
    /// than rows because a narrow table of many rows reads less than a wide table of few;
    /// measured on the columns this scan projects, of the row groups pruning left it, so
    /// it is a property of the scan and not of the table.
    pub small_table_bytes: u64,
}

impl Translator {
    pub fn new(target_partitions: usize, batching: Batching) -> Self {
        Self {
            target_partitions,
            batching,
            source_targets: Vec::new(),
            next_source: std::cell::Cell::new(0),
            small_table_bytes: 0,
        }
    }

    /// The second pass: each source gets the batch size the estimator solved for it,
    /// rather than the one number the first pass had to assume.
    pub fn with_source_targets(mut self, targets: Vec<u64>) -> Self {
        self.source_targets = targets;
        self
    }

    pub fn with_small_table_bytes(mut self, bytes: u64) -> Self {
        self.small_table_bytes = bytes;
        self
    }

    /// The root, with the limit lowering rule applied: a root-adjacent limit is not a
    /// node at all — its interval becomes the unload's, because a limit over a stream
    /// about to leave the device is a statement about which rows are worth moving.
    pub fn translate(&self, root: &Arc<dyn ExecutionPlan>) -> Result<Box<dyn GpuNode>, PlanError> {
        match limit_interval(root) {
            Some((input, interval)) => {
                let input = self.node(&input)?;
                Ok(Box::new(GpuUnload::new(input, Some(interval))))
            }
            None => Ok(Box::new(GpuUnload::new(self.node(root)?, None))),
        }
    }

    fn node(&self, plan: &Arc<dyn ExecutionPlan>) -> Result<Box<dyn GpuNode>, PlanError> {
        let any = plan.as_any();

        if let Some(parquet) = any.downcast_ref::<ParquetExec>() {
            return self.source(parquet);
        }
        if let Some(filter) = any.downcast_ref::<FilterExec>() {
            let input = self.node(filter.input())?;
            let predicate = translate_expr(filter.predicate(), &filter.input().schema())?;
            let projection = filter
                .projection()
                .map(|columns| columns.iter().map(|c| *c as u32).collect());
            return Ok(Box::new(GpuFilter::new(
                input,
                predicate,
                projection,
                Schema::new(filter.schema()),
            )));
        }
        if let Some(project) = any.downcast_ref::<ProjectionExec>() {
            let input = self.node(project.input())?;
            let input_schema = project.input().schema();
            let mut exprs = Vec::with_capacity(project.expr().len());
            for (expr, name) in project.expr() {
                exprs.push(NamedExpr::new(translate_expr(expr, &input_schema)?, name));
            }
            return Ok(Box::new(GpuProject::new(
                input,
                exprs,
                Schema::new(project.schema()),
            )));
        }
        if let Some(sort) = any.downcast_ref::<SortExec>() {
            return self.sort(sort);
        }
        if let Some(coalesce) = any.downcast_ref::<CoalesceBatchesExec>() {
            // The target batch size goes; batching is this mode's own concern (#139).
            // A fetch does not: DataFusion's limit pushdown parks a limit here, and
            // dropping the node with it would answer a count over three rows with the
            // count of all of them.
            if let Some(fetch) = coalesce.fetch() {
                let interval = RowInterval {
                    skip: 0,
                    fetch: Some(fetch as u64),
                };
                return self.mid_plan_limit(coalesce.input(), interval);
            }
            return self.node(coalesce.input());
        }
        if any.is::<GlobalLimitExec>() || any.is::<LocalLimitExec>() {
            let (input, interval) = limit_interval(plan).expect("a limit carries an interval");
            return self.mid_plan_limit(&input, interval);
        }
        if let Some(aggregate) = any.downcast_ref::<AggregateExec>() {
            return self.aggregate(aggregate);
        }
        if let Some(cross) = any.downcast_ref::<CrossJoinExec>() {
            let build = self.build_side(cross.left())?;
            // With no key to co-locate on, both sides are one lane — the probe as much as
            // the build, though only the build is also one batch.
            let probe = self.merged(self.node(cross.right())?);
            return Ok(Box::new(GpuCrossJoin::new(
                build,
                probe,
                Schema::new(cross.schema()),
            )));
        }
        if let Some(nested) = any.downcast_ref::<NestedLoopJoinExec>() {
            return self.nested_loop_join(nested);
        }
        if let Some(repartition) = any.downcast_ref::<RepartitionExec>() {
            return self.repartition(repartition);
        }
        if let Some(coalesce) = any.downcast_ref::<CoalescePartitionsExec>() {
            let input = self.node(coalesce.input())?;
            return Ok(self.merged(input));
        }
        if let Some(merge) = any.downcast_ref::<SortPreservingMergeExec>() {
            return self.sort_preserving_merge(merge);
        }
        if let Some(join) = any.downcast_ref::<HashJoinExec>() {
            return self.hash_join(join);
        }
        if let Some(union) = any.downcast_ref::<UnionExec>() {
            let branches = self.branches(union.inputs().iter(), &union.schema())?;
            return Ok(Box::new(GpuUnion::new(
                branches,
                Schema::new(union.schema()),
            )));
        }
        if let Some(interleave) = any.downcast_ref::<InterleaveExec>() {
            let branches = self.branches(interleave.inputs().iter(), &interleave.schema())?;
            // DataFusion interleaves where its branches share a hash, so lane p of one
            // belongs beside lane p of the next. This mode decides its own lane counts —
            // the small-source rule can put one branch on a single lane — and branches
            // that no longer agree have no shared distribution left to interleave on.
            let first = branches[0].kind().layout().expect("a branch is not a sink");
            let agreed = branches.iter().all(|branch| {
                let layout = branch.kind().layout().expect("a branch is not a sink");
                layout.n == first.n && layout.key_distribution == first.key_distribution
            });
            let schema = Schema::new(interleave.schema());
            return Ok(if agreed {
                Box::new(GpuInterleave::new(branches, schema))
            } else {
                Box::new(GpuUnion::new(branches, schema))
            });
        }
        if any.is::<WindowAggExec>() || any.is::<BoundedWindowAggExec>() {
            return Err(PlanError::Unsupported(format!(
                "{}: window functions do not plan in batch-partitioned mode (#143)",
                plan.name()
            )));
        }

        if any.is::<PlaceholderRowExec>() {
            return Err(PlanError::Unsupported(format!(
                "{}: an aggregate DataFusion answered from statistics, so there is no \
                 aggregate left to translate (#158)",
                plan.name()
            )));
        }

        Err(PlanError::Unsupported(format!("plan node {}", plan.name())))
    }

    /// A hash repartition is the shuffle: merge the lanes into one, then scatter that one
    /// into N by the same murmur3 both engines use. Round-robin carries no key, so it says
    /// nothing this mode acts on and leaves no node.
    fn repartition(&self, repartition: &RepartitionExec) -> Result<Box<dyn GpuNode>, PlanError> {
        let input = self.node(repartition.input())?;
        match repartition.partitioning() {
            Partitioning::Hash(exprs, n) => {
                let keys = hash_key_ordinals(exprs, &repartition.input().schema())?;
                Ok(self.shuffled(input, keys, *n))
            }
            // Round-robin carries no key, so it says nothing this mode acts on; an unknown
            // partitioning is a claim we cannot read, and guessing at it would be a lane
            // assignment nobody chose.
            Partitioning::RoundRobinBatch(_) => Ok(input),
            Partitioning::UnknownPartitioning(n) => Err(PlanError::Unsupported(format!(
                "a repartition into {n} lanes by an unstated rule"
            ))),
        }
    }

    /// The shuffle, as both halves: nothing to merge below one lane, and a lane that
    /// arrived as one batch per lane is worth re-coalescing so the scatter is one call
    /// rather than one per lane's batch.
    fn shuffled(&self, input: Box<dyn GpuNode>, keys: Vec<u32>, n: usize) -> Box<dyn GpuNode> {
        let was_single_batch = batches(input.as_ref()) == BatchLayout::SingleBatch;
        let mut input = self.merged(input);
        if was_single_batch && batches(input.as_ref()) != BatchLayout::SingleBatch {
            input = Box::new(GpuCoalesceAllBatches::new(input));
        }
        Box::new(GpuEmitPartitions::new(input, keys, n))
    }

    /// A merge over one lane is not a no-op to be optimized away later — it is a node
    /// this plan does not have, and the golden shows it as one.
    fn merged(&self, input: Box<dyn GpuNode>) -> Box<dyn GpuNode> {
        if lanes(input.as_ref()) > 1 {
            Box::new(GpuMergePartitions::new(input))
        } else {
            input
        }
    }

    /// An N-into-1 order-preserving merge: the per-lane sort below it stays a per-batch
    /// sort, and this node is the accumulator, so no per-lane accumulator is emitted.
    fn sort_preserving_merge(
        &self,
        merge: &SortPreservingMergeExec,
    ) -> Result<Box<dyn GpuNode>, PlanError> {
        let (input, keys) = match merge.input().as_any().downcast_ref::<SortExec>() {
            Some(sort) => {
                let sorted = self.per_batch_sort(sort)?;
                (sorted.node, sorted.keys)
            }
            None => {
                let input = self.node(merge.input())?;
                let keys = sort_key_ordinals(merge.expr(), &merge.input().schema())?;
                (input, keys)
            }
        };
        Ok(Box::new(GpuMergeSortedPartitions::new(
            input,
            keys,
            merge.fetch(),
        )))
    }

    fn hash_join(&self, join: &HashJoinExec) -> Result<Box<dyn GpuNode>, PlanError> {
        let has_filter = join.filter().is_some();
        let capability = capability(*join.join_type(), has_filter)?;

        let mut keys = Vec::with_capacity(join.on().len());
        for (left, right) in join.on() {
            // Each side's key is an ordinal into that side, so each is read against that
            // side's schema.
            keys.push((
                column_ordinal_of(left, &join.left().schema(), "join key")?,
                column_ordinal_of(right, &join.right().schema(), "join key")?,
            ));
        }

        // DataFusion collects its left input, so the build side is already left; what the
        // planner adds is the one batch per lane the join reads it as.
        let mut build = self.node(join.left())?;
        let mut probe = self.node(join.right())?;
        if !co_partitioned(build.as_ref(), probe.as_ref(), &keys) {
            // DataFusion broadcasts a small build side rather than hashing both, so the
            // sides are not co-located and this mode has no broadcast to do it with
            // (#140). One lane is what is left.
            build = self.merged(build);
            probe = self.merged(probe);
        }
        if batches(build.as_ref()) != BatchLayout::SingleBatch {
            build = Box::new(GpuCoalesceAllBatches::new(build));
        }
        if !capability.probe_streams && batches(probe.as_ref()) != BatchLayout::SingleBatch {
            probe = Box::new(GpuCoalesceAllBatches::new(probe));
        }

        let (filter, filter_columns) = match join.filter() {
            Some(filter) => (
                Some(translate_expr(filter.expression(), filter.schema())?),
                filter_column_map(filter)?,
            ),
            None => (None, Vec::new()),
        };
        Ok(Box::new(GpuJoin::new(
            build,
            probe,
            *join.join_type(),
            keys,
            filter,
            filter_columns,
            join.null_equals_null(),
            join.projection
                .as_ref()
                .map(|columns| columns.iter().map(|c| *c as u32).collect()),
            Schema::new(join.schema()),
        )))
    }

    /// Union branches are planned independently, so one column can arrive as a different
    /// type per branch. The cast is a node rather than something the executor does behind
    /// the plan's back (#41), and it is the same rule as everywhere else: every cast is
    /// explicit.
    fn branches<'a>(
        &self,
        inputs: impl Iterator<Item = &'a Arc<dyn ExecutionPlan>>,
        declared: &ArrowSchema,
    ) -> Result<Vec<Box<dyn GpuNode>>, PlanError> {
        let mut branches = Vec::new();
        for input in inputs {
            let translated = self.node(input)?;
            branches.push(self.cast_branch(translated, &input.schema(), declared));
        }
        Ok(branches)
    }

    fn cast_branch(
        &self,
        branch: Box<dyn GpuNode>,
        schema: &ArrowSchema,
        declared: &ArrowSchema,
    ) -> Box<dyn GpuNode> {
        let differs = schema
            .fields()
            .iter()
            .zip(declared.fields().iter())
            .any(|(field, out)| field.data_type() != out.data_type());
        if !differs {
            return branch;
        }
        let exprs: Vec<NamedExpr> = schema
            .fields()
            .iter()
            .zip(declared.fields().iter())
            .enumerate()
            .map(|(index, (field, out))| {
                let column = Expr::column(index as u32, field.name());
                let expr = if field.data_type() == out.data_type() {
                    column
                } else {
                    Expr::Cast {
                        expr: Box::new(column),
                        target: out.data_type().clone(),
                    }
                };
                NamedExpr::new(expr, out.name())
            })
            .collect();
        Box::new(GpuProject::new(
            branch,
            exprs,
            Schema::new(Arc::new(declared.clone())),
        ))
    }

    /// Sources are reached in one order, left to right, so the estimator's figures are
    /// consumed in that order — the same order it indexed them in.
    fn batching_for_source(&self) -> Batching {
        let seq = self.next_source.get();
        self.next_source.set(seq + 1);
        match (self.batching, self.source_targets.get(seq)) {
            (Batching::Sized { .. }, Some(target)) => Batching::Sized {
                target_batch_bytes: *target as usize,
            },
            (batching, _) => batching,
        }
    }

    /// The small-table rule: a source under the threshold plans one lane whatever the
    /// target, and the nodes a one-lane region does not need are then never emitted. It
    /// bites only while batching is on, which is what makes the threshold mean anything.
    fn lanes_for(&self, survivors: &[RowGroupMeta], limit: Option<usize>) -> usize {
        // A limit DataFusion pushed into the scan is the whole answer wherever it erased
        // the limit node above it — `SELECT * FROM nation LIMIT 3` at tp4 plans as a bare
        // scan carrying limit=3. Every lane would honour it, so N lanes would return N
        // times the rows; one lane is what makes the loader's own limit the answer.
        if limit.is_some() {
            return 1;
        }
        let bytes: u64 = survivors.iter().map(|group| group.bytes).sum();
        match self.batching {
            // With one batch per lane there is nothing for a size threshold to size, so the
            // rule that drops a source to one lane has nothing to act on either.
            Batching::Off => self.target_partitions,
            _ if bytes < self.small_table_bytes => 1,
            _ => self.target_partitions,
        }
    }

    fn source(&self, parquet: &ParquetExec) -> Result<Box<dyn GpuNode>, PlanError> {
        let config = parquet.base_config();
        let scan = survivor_metadata(parquet)?;
        let lanes = self.lanes_for(&scan.groups, config.limit);
        let partition_groups = partition(&scan.groups, lanes, self.batching_for_source())?;

        let files = config
            .file_groups
            .iter()
            .flatten()
            .map(|file| format!("/{}", file.object_meta.location))
            .collect();
        let projection = match &config.projection {
            Some(columns) => columns.iter().map(|c| *c as u32).collect(),
            None => (0..config.file_schema.fields().len() as u32).collect(),
        };
        Ok(Box::new(GpuLoadParquet::new(
            parquet_table_name(parquet).unwrap_or_default(),
            files,
            projection,
            partition_groups,
            &scan,
            config.limit,
            Schema::new(parquet.schema()),
        )))
    }

    /// A sort becomes a per-batch sort plus the accumulator that makes the whole stream
    /// ordered, and the `fetch` is replicated onto both: the top n of a union is the top
    /// n of each part's top n, which is what keeps a top-N memory-bounded.
    fn sort(&self, sort: &SortExec) -> Result<Box<dyn GpuNode>, PlanError> {
        let sorted = self.per_batch_sort(sort)?;
        Ok(Box::new(GpuAccumulateBatchesAndSort::new(
            sorted.node,
            sorted.keys,
            sorted.fetch,
        )))
    }

    /// The per-batch half of the decomposition, without deciding which accumulator goes
    /// above it: one lane's stream sorted is an accumulate-and-sort, an N-into-1 is a
    /// merge, and only the parent knows which it needs.
    fn per_batch_sort(&self, sort: &SortExec) -> Result<PerBatchSort, PlanError> {
        let input = self.node(sort.input())?;
        let keys = sort_key_ordinals(sort.expr(), &sort.input().schema())?;
        let fetch = sort.fetch();
        Ok(PerBatchSort {
            node: Box::new(GpuSort::new(input, keys.clone(), fetch)),
            keys,
            fetch,
        })
    }

    /// Mid-plan, the interval is a real node over a one-lane stream. Its input is NOT
    /// required to be one batch: requiring that would read the whole of a subquery's
    /// table to answer for a hundred rows.
    fn mid_plan_limit(
        &self,
        input: &Arc<dyn ExecutionPlan>,
        interval: RowInterval,
    ) -> Result<Box<dyn GpuNode>, PlanError> {
        let mut input = self.node(input)?;
        if lanes(input.as_ref()) > 1 {
            input = Box::new(GpuMergePartitions::new(input));
        }
        Ok(Box::new(GpuLimit::new(input, interval)))
    }

    /// A join's build side is always one batch, and the planner is what makes it one.
    fn build_side(&self, plan: &Arc<dyn ExecutionPlan>) -> Result<Box<dyn GpuNode>, PlanError> {
        let mut build = self.merged(self.node(plan)?);
        if batches(build.as_ref()) != BatchLayout::SingleBatch {
            build = Box::new(GpuCoalesceAllBatches::new(build));
        }
        Ok(build)
    }

    fn nested_loop_join(&self, join: &NestedLoopJoinExec) -> Result<Box<dyn GpuNode>, PlanError> {
        use datafusion::common::JoinType;
        let join_type = match join.join_type() {
            JoinType::Inner => NestedLoopJoinType::Inner,
            JoinType::Left => NestedLoopJoinType::Left,
            other => {
                return Err(PlanError::Unsupported(format!(
                    "nested-loop join type {other:?} — the executor rejects anything but \
                     Inner and Left (#160)"
                )));
            }
        };
        // DataFusion uses a nested-loop join with no predicate where the product itself is
        // what the query asked for — tpcds q9 pairs one-row aggregates that way — and that
        // is a cross join by any other name.
        let Some(filter) = join.filter() else {
            let build = self.build_side(join.left())?;
            let probe = self.node(join.right())?;
            return Ok(Box::new(GpuCrossJoin::new(
                build,
                probe,
                Schema::new(join.schema()),
            )));
        };
        let build = self.build_side(join.left())?;
        let mut probe = self.merged(self.node(join.right())?);
        // A Left form emits its unmatched build rows in the same pass, so it cannot
        // stream: #136's finish trick accumulates keys and a predicate join has none.
        if join_type == NestedLoopJoinType::Left
            && batches(probe.as_ref()) != BatchLayout::SingleBatch
        {
            probe = Box::new(GpuCoalesceAllBatches::new(probe));
        }
        // The filter is written against a table of its own, so its ordinals travel with
        // the map that says which side each of them came from.
        let predicate = translate_expr(filter.expression(), filter.schema())?;
        let filter_columns = filter_column_map(filter)?;
        Ok(Box::new(GpuNestedLoopJoin::new(
            build,
            probe,
            join_type,
            predicate,
            filter_columns,
            Schema::new(join.schema()),
        )))
    }
}

/// A sort's per-batch half, before the parent decides which accumulator goes above it.
struct PerBatchSort {
    node: Box<dyn GpuNode>,
    keys: Vec<ColumnOrder>,
    fetch: Option<usize>,
}

/// Lane p of one side holds what can match lane p of the other only if both were scattered
/// on their own join keys, in key order — equal lane counts are not the same fact, and two
/// four-lane scans agree on nothing.
pub(crate) fn co_partitioned(
    build: &dyn GpuNode,
    probe: &dyn GpuNode,
    keys: &[(u32, u32)],
) -> bool {
    let scattered_on = |node: &dyn GpuNode, on: Vec<u32>| {
        let layout = node.kind().layout().expect("a sink cannot be an input");
        layout.n == 1 || layout.key_distribution == KeyDistribution::ByHash { hash_keys: on }
    };
    lanes(build) == lanes(probe)
        && scattered_on(build, keys.iter().map(|(b, _)| *b).collect())
        && scattered_on(probe, keys.iter().map(|(_, p)| *p).collect())
}

/// Hash keys are ordinals into the node's input, so anything else is a shape this mode
/// does not plan rather than something to evaluate on the way to the hash.
pub(crate) fn hash_key_ordinals(
    exprs: &[Arc<dyn PhysicalExpr>],
    input_schema: &ArrowSchema,
) -> Result<Vec<u32>, PlanError> {
    let mut keys = Vec::with_capacity(exprs.len());
    for expr in exprs {
        keys.push(column_ordinal_of(expr, input_schema, "hash key")?);
    }
    Ok(keys)
}

pub(crate) fn sort_key_ordinals(
    exprs: &LexOrdering,
    input_schema: &ArrowSchema,
) -> Result<Vec<ColumnOrder>, PlanError> {
    let mut keys = Vec::with_capacity(exprs.len());
    for key in exprs.iter() {
        keys.push(ColumnOrder {
            column: column_ordinal_of(&key.expr, input_schema, "sort key")?,
            ascending: !key.options.descending,
            nulls_first: key.options.nulls_first,
        });
    }
    Ok(keys)
}

pub(crate) fn column_ordinal_of(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &ArrowSchema,
    site: &str,
) -> Result<u32, PlanError> {
    match translate_expr(expr, input_schema)? {
        Expr::Column(reference) => Ok(reference.index),
        _ => Err(PlanError::Unsupported(format!(
            "{site} {expr} is an expression rather than a column"
        ))),
    }
}

/// Which side each column of a join filter's own table came from. Exhaustive: DataFusion
/// has a third side for a mark join's synthesized column, and a filter reading it would
/// otherwise be silently rebased onto the probe.
pub(crate) fn filter_column_map(filter: &JoinFilter) -> Result<Vec<JoinFilterColumn>, PlanError> {
    let mut columns = Vec::with_capacity(filter.column_indices().len());
    for column in filter.column_indices() {
        let side = match column.side {
            datafusion::common::JoinSide::Left => JoinSide::Build,
            datafusion::common::JoinSide::Right => JoinSide::Probe,
            datafusion::common::JoinSide::None => {
                return Err(PlanError::Unsupported(
                    "a join filter reading the mark column: it belongs to neither side, so \
                     this mode has no ordinal to rebase it onto"
                        .to_string(),
                ));
            }
        };
        columns.push(JoinFilterColumn {
            side,
            index: column.index as u32,
        });
    }
    Ok(columns)
}

/// A limit's interval, whichever node carries it. `LocalLimitExec` has no skip.
pub(crate) fn limit_interval(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<(Arc<dyn ExecutionPlan>, RowInterval)> {
    let any = plan.as_any();
    if let Some(global) = any.downcast_ref::<GlobalLimitExec>() {
        return Some((
            global.input().clone(),
            RowInterval {
                skip: global.skip() as u64,
                fetch: global.fetch().map(|n| n as u64),
            },
        ));
    }
    if let Some(local) = any.downcast_ref::<LocalLimitExec>() {
        return Some((
            local.input().clone(),
            RowInterval {
                skip: 0,
                fetch: Some(local.fetch() as u64),
            },
        ));
    }
    // Not reachable from today's planner: where a limit is root-adjacent DataFusion leaves
    // a GlobalLimitExec there and uses a coalesce's fetch only as the bound pushed below
    // it. The arm is here so both paths agree if that ever changes — one query must not
    // have two plan shapes depending on where the fetch was parked.
    if let Some(coalesce) = any.downcast_ref::<CoalesceBatchesExec>() {
        return coalesce.fetch().map(|fetch| {
            (
                coalesce.input().clone(),
                RowInterval {
                    skip: 0,
                    fetch: Some(fetch as u64),
                },
            )
        });
    }
    None
}

pub(crate) fn lanes(node: &dyn GpuNode) -> usize {
    node.kind().layout().expect("a sink cannot be an input").n
}

pub(crate) fn batches(node: &dyn GpuNode) -> BatchLayout {
    node.kind()
        .layout()
        .expect("a sink cannot be an input")
        .batch_layout
}
