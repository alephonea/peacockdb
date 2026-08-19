//! The 1:1-per-batch nodes: filter, project, and the per-batch sort.

use std::any::Any;

use super::super::error::PlanError;
use super::super::expr::{Expr, NamedExpr};
use super::super::layout::{ColumnOrder, NodeKind, SortOrder};
use super::super::node::GpuNode;
use super::super::schema::Schema;
use super::{check_column_refs, input_layout, input_schema, rebase_through_projection};

#[derive(Debug)]
pub struct GpuFilter {
    kind: NodeKind,
    pub predicate: Expr,
    input: Box<dyn GpuNode>,
}

impl GpuFilter {
    pub fn new(input: Box<dyn GpuNode>, predicate: Expr) -> Self {
        // Dropping rows leaves every declared property standing: the lanes, the order
        // within a batch, and the key each lane holds.
        let kind = NodeKind::Intermediate {
            layout: input_layout(input.as_ref()),
            schema: input_schema(input.as_ref()),
        };
        Self {
            kind,
            predicate,
            input,
        }
    }
}

impl GpuNode for GpuFilter {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        vec![self.input.as_ref()]
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        check_column_refs(
            &self.predicate,
            &input_schema(self.input.as_ref()),
            "GpuFilter",
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
pub struct GpuProject {
    kind: NodeKind,
    pub exprs: Vec<NamedExpr>,
    input: Box<dyn GpuNode>,
}

impl GpuProject {
    /// `schema` is DataFusion's own for this projection — the types it coerced to, not a
    /// second derivation of them.
    pub fn new(input: Box<dyn GpuNode>, exprs: Vec<NamedExpr>, schema: Schema) -> Self {
        let projected: Vec<Expr> = exprs.iter().map(|e| e.expr.clone()).collect();
        let layout = rebase_through_projection(&input_layout(input.as_ref()), &projected);
        Self {
            kind: NodeKind::Intermediate { layout, schema },
            exprs,
            input,
        }
    }
}

impl GpuNode for GpuProject {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        vec![self.input.as_ref()]
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        let against = input_schema(self.input.as_ref());
        for expr in &self.exprs {
            check_column_refs(&expr.expr, &against, "GpuProject")?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Sorts each input batch independently, so the batches are individually ordered and
/// collectively not — an accumulator above it is what makes a stream sorted. `fetch` is
/// DataFusion's, replicated onto every stage of the decomposition, which is sound
/// because the top n of a union is the top n of each part's top n.
#[derive(Debug)]
pub struct GpuSort {
    kind: NodeKind,
    pub keys: Vec<ColumnOrder>,
    pub fetch: Option<usize>,
    input: Box<dyn GpuNode>,
}

impl GpuSort {
    pub fn new(input: Box<dyn GpuNode>, keys: Vec<ColumnOrder>, fetch: Option<usize>) -> Self {
        let mut layout = input_layout(input.as_ref());
        layout.sort_order = SortOrder::batch_sorted(keys.clone());
        let kind = NodeKind::Intermediate {
            layout,
            schema: input_schema(input.as_ref()),
        };
        Self {
            kind,
            keys,
            fetch,
            input,
        }
    }
}

impl GpuNode for GpuSort {
    fn kind(&self) -> &NodeKind {
        &self.kind
    }

    fn children(&self) -> Vec<&dyn GpuNode> {
        vec![self.input.as_ref()]
    }

    fn validate_schemas_and_partitions(&self) -> Result<(), PlanError> {
        if self.keys.is_empty() {
            return Err(PlanError::Invalid("GpuSort: no sort keys".to_string()));
        }
        let columns = input_schema(self.input.as_ref()).fields.fields().len();
        for key in &self.keys {
            if key.column as usize >= columns {
                return Err(PlanError::Invalid(format!(
                    "GpuSort: sort key @{} is past the {columns} columns its input has",
                    key.column
                )));
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
