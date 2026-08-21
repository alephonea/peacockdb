//! The fb payload each addressed node carries, one function per kind.
//!
//! Each is handed its plan node, the schemas that node declares it consumes, and the
//! child offsets the walk chose for its structural slots — so a payload is written from
//! the node alone, as the mapping table is a per-node claim.

use datafusion::arrow::datatypes::Field;
use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::generated::gpu_plan_generated::peacock::plan as fb;
use crate::plan_serializer::serialize_schema;

use super::super::error::PlanError;
use super::super::expr::Expr;
use super::super::layout::ColumnOrder;
use super::super::nodes::{
    GpuAccumulateBatchesAndSort, GpuEmitPartitions, GpuFilter, GpuLoadParquet,
    GpuMergeSortedPartitions, GpuProject, GpuSort,
};
use super::super::schema::Schema;
use super::expr_writer::write_expr;
use super::writer::Payload;

type Kids<'a, 'b> = &'b [WIPOffset<fb::PlanNode<'a>>];

/// `fetch` on the wire: `-1` is unlimited, which is how the frozen nodes spell `None`.
fn fetch_of(fetch: Option<usize>) -> i64 {
    fetch.map(|rows| rows as i64).unwrap_or(-1)
}

/// A bare column reference, named from the schema at that position — the same pairing of
/// ordinal and name the plan text carries, and for the same reason (#135).
fn column_ref<'a>(
    b: &mut FlatBufferBuilder<'a>,
    ordinal: u32,
    schema: &Schema,
) -> WIPOffset<fb::Expr<'a>> {
    let field = schema
        .fields
        .fields()
        .get(ordinal as usize)
        .expect("a plan that validated references a column it has");
    write_expr(b, &Expr::column(ordinal, field.name()))
        .expect("a column reference is always writable")
}

fn sort_keys<'a>(
    b: &mut FlatBufferBuilder<'a>,
    keys: &[ColumnOrder],
    input: &Schema,
) -> WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::SortExprNode<'a>>>> {
    let written: Vec<WIPOffset<fb::SortExprNode>> = keys
        .iter()
        .map(|key| {
            let expr = column_ref(b, key.column, input);
            fb::SortExprNode::create(
                b,
                &fb::SortExprNodeArgs {
                    expr: Some(expr),
                    asc: key.ascending,
                    nulls_first: key.nulls_first,
                },
            )
        })
        .collect();
    b.create_vector(&written)
}

/// The scan the driver overrides per batch.
///
/// `file_schema` is the projected fields and `projection` is left empty, which the C++
/// reads as "every column of the schema I was given". It selects columns by NAME
/// (`scan.cpp` builds `projected_names` and hands them to cuDF), so this reads exactly the
/// columns the node declares — where naming the file's own ordinals would need the file's
/// full column list, which a plan node does not carry.
pub(super) fn scan<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuLoadParquet,
    output: &Schema,
) -> Result<Payload, PlanError> {
    let paths: Vec<WIPOffset<&str>> = node.files.iter().map(|f| b.create_string(f)).collect();
    let paths = b.create_vector(&paths);
    let schema = serialize_schema(b, &output.fields);
    let scan = fb::CudfScan::create(
        b,
        &fb::CudfScanArgs {
            file_paths: Some(paths),
            file_schema: Some(schema),
            limit: node.limit.unwrap_or(0) as u64,
            ..Default::default()
        },
    );
    Ok(Payload {
        kind: fb::PlanNodeKind::CudfScan,
        value: scan.as_union_value(),
    })
}

pub(super) fn filter<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuFilter,
    kids: Kids<'a, '_>,
) -> Result<Payload, PlanError> {
    let predicate = write_expr(b, &node.predicate)?;
    let projection = node
        .projection
        .as_ref()
        .map(|columns| b.create_vector(columns));
    let filter = fb::CudfFilter::create(
        b,
        &fb::CudfFilterArgs {
            predicate: Some(predicate),
            input: Some(kids[0]),
            projection,
        },
    );
    Ok(Payload {
        kind: fb::PlanNodeKind::CudfFilter,
        value: filter.as_union_value(),
    })
}

pub(super) fn project<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuProject,
    kids: Kids<'a, '_>,
) -> Result<Payload, PlanError> {
    let mut exprs = Vec::with_capacity(node.exprs.len());
    for named in &node.exprs {
        exprs.push(write_expr(b, &named.expr)?);
    }
    let names: Vec<WIPOffset<&str>> = node
        .exprs
        .iter()
        .map(|named| b.create_string(&named.name))
        .collect();
    Ok(project_payload(b, exprs, names, kids[0]))
}

/// A project built from expressions the caller already wrote — the join's key and pad
/// projects come this way, since neither is a plan node of its own.
pub(super) fn project_payload<'a>(
    b: &mut FlatBufferBuilder<'a>,
    exprs: Vec<WIPOffset<fb::Expr<'a>>>,
    names: Vec<WIPOffset<&'a str>>,
    input: WIPOffset<fb::PlanNode<'a>>,
) -> Payload {
    let exprs = b.create_vector(&exprs);
    let aliases = b.create_vector(&names);
    let project = fb::CudfProject::create(
        b,
        &fb::CudfProjectArgs {
            exprs: Some(exprs),
            aliases: Some(aliases),
            input: Some(input),
        },
    );
    Payload {
        kind: fb::PlanNodeKind::CudfProject,
        value: project.as_union_value(),
    }
}

/// Per batch, so partitions are preserved: the collapse to one stream is a node of its
/// own in this mode, never a side effect of a sort.
pub(super) fn sort<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuSort,
    input: &Schema,
    kids: Kids<'a, '_>,
) -> Result<Payload, PlanError> {
    let exprs = sort_keys(b, &node.keys, input);
    let sort = fb::CudfSort::create(
        b,
        &fb::CudfSortArgs {
            exprs: Some(exprs),
            fetch: fetch_of(node.fetch),
            input: Some(kids[0]),
            preserve_partitioning: true,
        },
    );
    Ok(Payload {
        kind: fb::PlanNodeKind::CudfSort,
        value: sort.as_union_value(),
    })
}

/// The sort an accumulating node runs per batch before its merge: same node, and the
/// `fetch` rides the merge rather than the per-batch sorts, which must keep every row a
/// later batch could outrank.
pub(super) fn accumulating_sort<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuAccumulateBatchesAndSort,
    input: &Schema,
    kids: Kids<'a, '_>,
) -> Result<Payload, PlanError> {
    let exprs = sort_keys(b, &node.keys, input);
    let sort = fb::CudfSort::create(
        b,
        &fb::CudfSortArgs {
            exprs: Some(exprs),
            fetch: -1,
            input: Some(kids[0]),
            preserve_partitioning: true,
        },
    );
    Ok(Payload {
        kind: fb::PlanNodeKind::CudfSort,
        value: sort.as_union_value(),
    })
}

pub(super) fn merge_sorted<'a>(
    b: &mut FlatBufferBuilder<'a>,
    keys: &[ColumnOrder],
    fetch: Option<usize>,
    input: &Schema,
    kids: Kids<'a, '_>,
) -> Result<Payload, PlanError> {
    let exprs = sort_keys(b, keys, input);
    let merge = fb::CudfSortPreservingMerge::create(
        b,
        &fb::CudfSortPreservingMergeArgs {
            exprs: Some(exprs),
            fetch: fetch_of(fetch),
            input: Some(kids[0]),
        },
    );
    Ok(Payload {
        kind: fb::PlanNodeKind::CudfSortPreservingMerge,
        value: merge.as_union_value(),
    })
}

/// The merge a `GpuMergeSortedPartitions` runs, which is the same node one level up: k
/// sorted lanes into one stream, the `fetch` applied to the result.
pub(super) fn merge_partitions<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuMergeSortedPartitions,
    input: &Schema,
    kids: Kids<'a, '_>,
) -> Result<Payload, PlanError> {
    merge_sorted(b, &node.keys, node.fetch, input, kids)
}

/// The collapse arm: it concatenates whatever k handles the call passes, which is why the
/// node carries nothing but its input.
pub(super) fn coalesce_partitions<'a>(
    b: &mut FlatBufferBuilder<'a>,
    kids: Kids<'a, '_>,
) -> Payload {
    let coalesce = fb::CudfCoalescePartitions::create(
        b,
        &fb::CudfCoalescePartitionsArgs {
            input: Some(kids[0]),
        },
    );
    Payload {
        kind: fb::PlanNodeKind::CudfCoalescePartitions,
        value: coalesce.as_union_value(),
    }
}

/// One lane into N by Spark murmur3 on the hash keys — the routing both engines share,
/// so a row lands in the same lane on either.
pub(super) fn repartition<'a>(
    b: &mut FlatBufferBuilder<'a>,
    node: &GpuEmitPartitions,
    input: &Schema,
    lanes: u32,
    kids: Kids<'a, '_>,
) -> Result<Payload, PlanError> {
    let keys: Vec<WIPOffset<fb::Expr>> = node
        .hash_keys
        .iter()
        .map(|ordinal| column_ref(b, *ordinal, input))
        .collect();
    let keys = b.create_vector(&keys);
    let repartition = fb::CudfRepartition::create(
        b,
        &fb::CudfRepartitionArgs {
            kind: fb::PartitioningKind::Hash,
            num_partitions: lanes,
            hash_exprs: Some(keys),
            input: Some(kids[0]),
        },
    );
    Ok(Payload {
        kind: fb::PlanNodeKind::CudfRepartition,
        value: repartition.as_union_value(),
    })
}

/// A typed NULL literal, which is what pads a probe column an anti join never emitted.
pub(super) fn null_literal<'a>(
    b: &mut FlatBufferBuilder<'a>,
    field: &Field,
) -> Result<WIPOffset<fb::Expr<'a>>, PlanError> {
    let null = datafusion::common::ScalarValue::try_from(field.data_type())
        .map_err(|e| PlanError::Unsupported(format!("no null scalar for {}: {e}", field.name())))?;
    write_expr(b, &Expr::Literal(null))
}

/// One column's declared field, for a caller that needs its name or its type — a key
/// project naming what it kept, a pad literal typed by what it stands in for.
pub(super) fn field_at(schema: &Schema, ordinal: u32) -> &Field {
    schema
        .fields
        .fields()
        .get(ordinal as usize)
        .expect("a plan that validated references a column it has")
}
