//! What a node declares about its output *columns*: arrow types, plus the semantics a
//! consumer can check.
//!
//! Types come from the planner's own arrow schema, so decimal precision and scale are
//! not a second copy that can drift. The annotations are what a merging or finalizing
//! node checks before trusting a position — the class #135 describes, where an ordinal
//! read in the wrong order produces identical per-node numbers everywhere and surfaces
//! only in the final result.

use std::sync::Arc;

use datafusion::arrow::datatypes::Schema as ArrowSchema;

use super::aggregates::AggFunc;

/// One aggregate's state columns in a partial's output. `positions` index the declaring
/// node's fields; `func` and `ddof` are carried so a merge can confirm it is merging the
/// aggregate it thinks it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggStateColumns {
    pub output: String,
    pub func: AggFunc,
    pub ddof: u32,
    pub positions: Vec<u32>,
}

/// Column types in order — the index *is* the ordinal every plan reference uses — plus
/// what those columns mean.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub fields: Arc<ArrowSchema>,
    /// ordinals of the group-by keys, including a synthesized `__grouping_id`
    pub group_keys: Vec<u32>,
    /// one entry per aggregate whose state this output carries
    pub agg_state: Vec<AggStateColumns>,
}

impl Schema {
    /// Columns with no group keys and no aggregate state — a scan, a filter, a project.
    pub fn new(fields: Arc<ArrowSchema>) -> Self {
        Self {
            fields,
            group_keys: Vec::new(),
            agg_state: Vec::new(),
        }
    }

    pub fn position_of(&self, column: &str) -> Option<u32> {
        self.fields.index_of(column).ok().map(|i| i as u32)
    }

    pub fn state_for(&self, output: &str) -> Option<&AggStateColumns> {
        self.agg_state.iter().find(|s| s.output == output)
    }
}
