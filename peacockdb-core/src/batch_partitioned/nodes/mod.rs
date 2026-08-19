//! The plan nodes, grouped by family, and the one downcast registry over them.

pub mod accumulators;
pub mod aggregate;
pub mod exec_ops;
pub mod join;
pub mod partition_ops;
pub mod source;
pub mod union;
pub mod unload;

use super::error::PlanError;
use super::expr::Expr;
use super::layout::{KeyDistribution, PartitionLayout, SortOrder};
use super::node::GpuNode;
use super::schema::Schema;

pub use accumulators::{GpuAccumulateBatchesAndSort, GpuCoalesceAllBatches, GpuLimit};
pub use aggregate::{AggregateBody, GpuAggregate, GpuAggregateBatches};
pub use exec_ops::{GpuFilter, GpuProject, GpuSort};
pub use join::{GpuCrossJoin, GpuJoin, GpuNestedLoopJoin};
pub use partition_ops::{GpuEmitPartitions, GpuMergePartitions, GpuMergeSortedPartitions};
pub use source::GpuLoadParquet;
pub use union::{GpuInterleave, GpuUnion};
pub use unload::GpuUnload;

/// Every node kind, as a borrow of the concrete node. Adding a node is one line here and
/// an exhaustive match everywhere it is consumed — the renderer, a backend's executor
/// match, the serializer — rather than a downcast chain per consumer.
pub enum NodeRef<'a> {
    LoadParquet(&'a GpuLoadParquet),
    Filter(&'a GpuFilter),
    Project(&'a GpuProject),
    Sort(&'a GpuSort),
    CoalesceAllBatches(&'a GpuCoalesceAllBatches),
    AccumulateBatchesAndSort(&'a GpuAccumulateBatchesAndSort),
    Limit(&'a GpuLimit),
    Aggregate(&'a GpuAggregate),
    AggregateBatches(&'a GpuAggregateBatches),
    Join(&'a GpuJoin),
    CrossJoin(&'a GpuCrossJoin),
    NestedLoopJoin(&'a GpuNestedLoopJoin),
    MergePartitions(&'a GpuMergePartitions),
    EmitPartitions(&'a GpuEmitPartitions),
    MergeSortedPartitions(&'a GpuMergeSortedPartitions),
    Union(&'a GpuUnion),
    Interleave(&'a GpuInterleave),
    Unload(&'a GpuUnload),
}

pub fn as_node_ref(node: &dyn GpuNode) -> NodeRef<'_> {
    let any = node.as_any();
    if let Some(n) = any.downcast_ref::<GpuLoadParquet>() {
        NodeRef::LoadParquet(n)
    } else if let Some(n) = any.downcast_ref::<GpuFilter>() {
        NodeRef::Filter(n)
    } else if let Some(n) = any.downcast_ref::<GpuProject>() {
        NodeRef::Project(n)
    } else if let Some(n) = any.downcast_ref::<GpuSort>() {
        NodeRef::Sort(n)
    } else if let Some(n) = any.downcast_ref::<GpuCoalesceAllBatches>() {
        NodeRef::CoalesceAllBatches(n)
    } else if let Some(n) = any.downcast_ref::<GpuAccumulateBatchesAndSort>() {
        NodeRef::AccumulateBatchesAndSort(n)
    } else if let Some(n) = any.downcast_ref::<GpuLimit>() {
        NodeRef::Limit(n)
    } else if let Some(n) = any.downcast_ref::<GpuAggregate>() {
        NodeRef::Aggregate(n)
    } else if let Some(n) = any.downcast_ref::<GpuAggregateBatches>() {
        NodeRef::AggregateBatches(n)
    } else if let Some(n) = any.downcast_ref::<GpuJoin>() {
        NodeRef::Join(n)
    } else if let Some(n) = any.downcast_ref::<GpuCrossJoin>() {
        NodeRef::CrossJoin(n)
    } else if let Some(n) = any.downcast_ref::<GpuNestedLoopJoin>() {
        NodeRef::NestedLoopJoin(n)
    } else if let Some(n) = any.downcast_ref::<GpuMergePartitions>() {
        NodeRef::MergePartitions(n)
    } else if let Some(n) = any.downcast_ref::<GpuEmitPartitions>() {
        NodeRef::EmitPartitions(n)
    } else if let Some(n) = any.downcast_ref::<GpuMergeSortedPartitions>() {
        NodeRef::MergeSortedPartitions(n)
    } else if let Some(n) = any.downcast_ref::<GpuUnion>() {
        NodeRef::Union(n)
    } else if let Some(n) = any.downcast_ref::<GpuInterleave>() {
        NodeRef::Interleave(n)
    } else if let Some(n) = any.downcast_ref::<GpuUnload>() {
        NodeRef::Unload(n)
    } else {
        panic!("a plan node outside the registry reached a consumer of it")
    }
}

/// The layout a node inherits from its input. A sink is the root, so it is never one.
pub(crate) fn input_layout(input: &dyn GpuNode) -> PartitionLayout {
    input
        .kind()
        .layout()
        .expect("a sink cannot be an input")
        .clone()
}

pub(crate) fn input_schema(input: &dyn GpuNode) -> Schema {
    input
        .kind()
        .schema()
        .expect("a sink cannot be an input")
        .clone()
}

/// Every column reference must be in range of the schema it reads AND carry the name of
/// the field at that position. The name is redundant on purpose: an ordinal read in the
/// wrong order is otherwise invisible until the final result (#135), and the layer
/// rebases ordinals at every node it inserts, so a stale reference is the likely slip.
pub(crate) fn check_column_refs(
    expr: &Expr,
    against: &Schema,
    site: &str,
) -> Result<(), PlanError> {
    match expr {
        Expr::Column(reference) => {
            let field = against
                .fields
                .fields()
                .get(reference.index as usize)
                .ok_or_else(|| {
                    PlanError::Invalid(format!(
                        "{site}: column {}@{} is past the {} columns its input has",
                        reference.name,
                        reference.index,
                        against.fields.fields().len()
                    ))
                })?;
            if field.name() != &reference.name {
                return Err(PlanError::Invalid(format!(
                    "{site}: column {}@{} reads {} at that position",
                    reference.name,
                    reference.index,
                    field.name()
                )));
            }
            Ok(())
        }
        Expr::Literal(_) => Ok(()),
        Expr::Binary { left, right, .. } => {
            check_column_refs(left, against, site)?;
            check_column_refs(right, against, site)
        }
        Expr::Unary { arg, .. } => check_column_refs(arg, against, site),
        Expr::Cast { expr, .. } => check_column_refs(expr, against, site),
        Expr::Like { expr, pattern, .. } => {
            check_column_refs(expr, against, site)?;
            check_column_refs(pattern, against, site)
        }
        Expr::Case {
            comparand,
            when_then,
            else_expr,
        } => {
            for part in comparand.iter().chain(else_expr.iter()) {
                check_column_refs(part, against, site)?;
            }
            for (when, then) in when_then {
                check_column_refs(when, against, site)?;
                check_column_refs(then, against, site)?;
            }
            Ok(())
        }
        Expr::ScalarFunction { args, .. } => {
            for arg in args {
                check_column_refs(arg, against, site)?;
            }
            Ok(())
        }
    }
}

/// Carry a layout's key distribution and sort order through a projection, keeping only
/// what a bare column reference re-exposes: a projected-away or computed column takes
/// its property with it, and a declaration that outlived its column would be a lie the
/// nodes above it act on.
pub(crate) fn rebase_through_projection(
    layout: &PartitionLayout,
    projected: &[Expr],
) -> PartitionLayout {
    let new_index = |old: u32| -> Option<u32> {
        projected
            .iter()
            .position(|expr| match expr {
                Expr::Column(reference) => reference.index == old,
                _ => false,
            })
            .map(|position| position as u32)
    };

    let key_distribution = match &layout.key_distribution {
        KeyDistribution::NotSpecified => KeyDistribution::NotSpecified,
        KeyDistribution::ByHash { hash_keys } => {
            match hash_keys
                .iter()
                .map(|k| new_index(*k))
                .collect::<Option<Vec<_>>>()
            {
                Some(hash_keys) => KeyDistribution::ByHash { hash_keys },
                None => KeyDistribution::NotSpecified,
            }
        }
    };

    let sort_order = match &layout.sort_order {
        SortOrder::NotSpecified => SortOrder::NotSpecified,
        SortOrder::BatchSorted { columns } => {
            let mapped: Option<Vec<_>> = columns
                .iter()
                .map(|order| {
                    new_index(order.column)
                        .map(|column| super::layout::ColumnOrder { column, ..*order })
                })
                .collect();
            // A prefix of the keys would still hold, but a sort key that vanished mid-list
            // leaves an order nothing downstream can name.
            mapped
                .map(SortOrder::batch_sorted)
                .unwrap_or(SortOrder::NotSpecified)
        }
    };

    PartitionLayout {
        key_distribution,
        sort_order,
        ..layout.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_partitioned::aggregates::{AggCall, PlanAgg};
    use crate::batch_partitioned::expr::{BinaryOp, NamedExpr};
    use crate::batch_partitioned::layout::{BatchLayout, ColumnOrder, NodeKind};
    use crate::batch_partitioned::node::RowInterval;
    use crate::batch_partitioned::nodes::join::NestedLoopJoinType;
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use std::any::Any;
    use std::sync::Arc;

    /// An input with a layout and schema chosen by the test: the guards below are about
    /// what a node requires of its input, and a plan that violates one is unreachable
    /// from sql precisely because the translation layer is what inserts the fix.
    #[derive(Debug)]
    struct Given {
        kind: NodeKind,
    }

    impl Given {
        fn input(layout: PartitionLayout, columns: &[&str]) -> Box<dyn GpuNode> {
            let fields: Vec<Field> = columns
                .iter()
                .map(|name| Field::new(*name, DataType::Int64, true))
                .collect();
            let schema = Schema::new(Arc::new(ArrowSchema::new(fields)));
            Box::new(Given {
                kind: NodeKind::Intermediate { layout, schema },
            })
        }
    }

    impl GpuNode for Given {
        fn kind(&self) -> &NodeKind {
            &self.kind
        }

        fn children(&self) -> Vec<&dyn GpuNode> {
            Vec::new()
        }

        fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
            Ok(())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn one_lane(batches: BatchLayout) -> PartitionLayout {
        PartitionLayout {
            batch_layout: batches,
            ..PartitionLayout::new(1)
        }
    }

    fn invalid(result: Result<(), PlanError>, mentions: &str) {
        match result {
            Err(PlanError::Invalid(what)) => assert!(
                what.contains(mentions),
                "the error names the wrong fix: {what}"
            ),
            other => panic!("expected an invalid plan naming {mentions}, got {other:?}"),
        }
    }

    #[test]
    fn a_reference_past_its_inputs_columns_is_caught_at_plan_time() {
        let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["a", "b"]);
        let schema = Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int64,
            true,
        )])));
        let filter = GpuFilter::new(input, Expr::column(5, "a"), None, schema);
        invalid(
            filter.validate_schemas_and_partitions(),
            "past the 2 columns",
        );
    }

    #[test]
    fn a_reference_whose_name_does_not_match_its_position_is_caught_at_plan_time() {
        // The rebasing the layer does at every inserted node is what makes this the
        // likely slip: the ordinal stays valid and starts reading a different column.
        let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["a", "b"]);
        let project = GpuProject::new(
            input,
            vec![NamedExpr::new(Expr::column(1, "a"), "a")],
            Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
                "a",
                DataType::Int64,
                true,
            )]))),
        );
        invalid(
            project.validate_schemas_and_partitions(),
            "reads b at that position",
        );
    }

    #[test]
    fn a_limit_over_several_lanes_names_the_node_that_fixes_it() {
        let input = Given::input(PartitionLayout::new(4), &["a"]);
        let limit = GpuLimit::new(
            input,
            RowInterval {
                skip: 0,
                fetch: Some(10),
            },
        );
        invalid(
            limit.validate_schemas_and_partitions(),
            "GpuMergePartitions",
        );
    }

    #[test]
    fn a_join_whose_build_side_is_many_batches_names_the_node_that_fixes_it() {
        let build = Given::input(one_lane(BatchLayout::MultipleBatches), &["k"]);
        let probe = Given::input(one_lane(BatchLayout::MultipleBatches), &["fk"]);
        let schema = Schema::new(Arc::new(ArrowSchema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("fk", DataType::Int64, true),
        ])));
        let join = GpuCrossJoin::new(build, probe, schema);
        invalid(
            join.validate_schemas_and_partitions(),
            "GpuCoalesceAllBatches",
        );
    }

    #[test]
    fn a_join_filter_column_mapped_to_the_wrong_side_is_caught_at_plan_time() {
        use super::join::{JoinFilterColumn, JoinSide};
        let build = Given::input(one_lane(BatchLayout::SingleBatch), &["k"]);
        let probe = Given::input(one_lane(BatchLayout::MultipleBatches), &["fk"]);
        let schema = Schema::new(Arc::new(ArrowSchema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("fk", DataType::Int64, true),
        ])));
        let filter = Expr::binary(
            Expr::column(0, "k"),
            BinaryOp::Lt,
            Expr::column(1, "fk"),
            DataType::Boolean,
        );
        // Both filter columns pointed at the probe: @0 then reads a valid column of the
        // wrong table, which nothing downstream could detect.
        let mapping = vec![
            JoinFilterColumn {
                side: JoinSide::Probe,
                index: 0,
            },
            JoinFilterColumn {
                side: JoinSide::Probe,
                index: 0,
            },
        ];
        let join = GpuNestedLoopJoin::new(
            build,
            probe,
            NestedLoopJoinType::Inner,
            filter,
            mapping,
            schema,
        );
        invalid(
            join.validate_schemas_and_partitions(),
            "maps to fk on the Probe side",
        );
    }

    #[test]
    fn an_accumulating_sort_over_unsorted_batches_names_the_node_that_fixes_it() {
        let input = Given::input(one_lane(BatchLayout::MultipleBatches), &["a"]);
        let keys = vec![ColumnOrder {
            column: 0,
            ascending: true,
            nulls_first: false,
        }];
        let accumulator = GpuAccumulateBatchesAndSort::new(input, keys, None);
        invalid(accumulator.validate_schemas_and_partitions(), "GpuSort");
    }

    #[test]
    fn a_scatter_over_several_lanes_names_the_node_that_fixes_it() {
        let input = Given::input(PartitionLayout::new(4), &["k"]);
        let emit = GpuEmitPartitions::new(input, vec![0], 4);
        invalid(emit.validate_schemas_and_partitions(), "GpuMergePartitions");
    }

    #[test]
    fn a_union_branch_of_another_type_names_the_cast_that_fixes_it() {
        // Routing cannot retype anything, so a branch that does not already match the
        // declared output is a missing project rather than work for the executor (#41).
        let wide = Given::input(one_lane(BatchLayout::MultipleBatches), &["n"]);
        let narrow: Box<dyn GpuNode> = Box::new(Given {
            kind: NodeKind::Intermediate {
                layout: one_lane(BatchLayout::MultipleBatches),
                schema: Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
                    "n",
                    DataType::Int32,
                    true,
                )]))),
            },
        });
        let declared = Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
            "n",
            DataType::Int64,
            true,
        )])));
        let union = GpuUnion::new(vec![wide, narrow], declared);
        invalid(
            union.validate_schemas_and_partitions(),
            "casting GpuProject",
        );
    }

    #[test]
    fn an_interleave_of_differently_hashed_branches_is_refused() {
        let hashed = |keys: Vec<u32>| PartitionLayout {
            key_distribution: KeyDistribution::ByHash { hash_keys: keys },
            ..PartitionLayout::new(4)
        };
        let schema = Schema::new(Arc::new(ArrowSchema::new(vec![Field::new(
            "k",
            DataType::Int64,
            true,
        )])));
        let interleave = GpuInterleave::new(
            vec![
                Given::input(hashed(vec![0]), &["k"]),
                Given::input(hashed(vec![1]), &["k"]),
            ],
            schema,
        );
        // Lane p is lane p of every branch, so a branch hashed on another key would put
        // rows that cannot meet into the same lane.
        invalid(
            interleave.validate_schemas_and_partitions(),
            "same hash distribution",
        );
    }

    #[test]
    fn a_finalizing_merge_over_lanes_hashed_on_other_columns_is_refused() {
        let hashed = PartitionLayout {
            key_distribution: KeyDistribution::ByHash { hash_keys: vec![1] },
            ..PartitionLayout::new(4)
        };
        let input = Given::input(hashed, &["k", "other", "n"]);
        let schema = Schema::new(Arc::new(ArrowSchema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("n", DataType::Int64, true),
        ])));
        let body = AggregateBody {
            group_by: vec![Expr::column(0, "k")],
            aggs: vec![AggCall {
                func: PlanAgg::Sum,
                args: vec![Expr::column(2, "n")],
                outputs: vec![Field::new("n", DataType::Int64, true)],
            }],
            finalize: Some(vec![NamedExpr::new(Expr::column(1, "n"), "n")]),
        };
        let intermediate = Schema::new(Arc::new(ArrowSchema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("n", DataType::Int64, true),
        ])));
        let merge = GpuAggregateBatches::new(input, body, intermediate, schema);
        // Hashed on a column it does not group by, so a group's rows are spread across
        // lanes and each lane would answer for part of it.
        invalid(
            merge.validate_schemas_and_partitions(),
            "subset of its group columns",
        );
    }
}
