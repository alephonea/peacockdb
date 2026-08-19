//! What a node declares about its output: how many lanes, how rows were routed into
//! them, what order they carry and whether a lane is one batch. Declarations only —
//! nothing here executes, and the vocabulary is fixed by
//! `llm-wiki/tasks/batch_partitioned_executor.md`.

use super::schema::Schema;

/// A sink structurally has no layout and no schema; everything else always has both,
/// which is why they live inside the kind rather than beside it as two `Option`s that
/// have to be `None` together.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Source {
        layout: PartitionLayout,
        schema: Schema,
    },
    Intermediate {
        layout: PartitionLayout,
        schema: Schema,
    },
    Sink,
}

/// One sort key: a column ordinal into the declaring node's schema, and its direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnOrder {
    pub column: u32,
    pub ascending: bool,
    pub nulls_first: bool,
}

/// How rows were routed into lanes. `ByHash` is Spark murmur3 seed 42 — the only
/// routing `GpuEmitPartitions` has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyDistribution {
    NotSpecified,
    ByHash { hash_keys: Vec<u32> },
}

impl KeyDistribution {
    /// The `hash_keys ⊆ group columns` rule a final aggregate's input must satisfy:
    /// rows of one group are co-located only when the shuffle keyed on a subset of
    /// the columns being grouped.
    pub fn is_subset_of(&self, group_columns: &[u32]) -> bool {
        match self {
            Self::NotSpecified => false,
            Self::ByHash { hash_keys } => hash_keys.iter().all(|k| group_columns.contains(k)),
        }
    }
}

/// Two-valued on purpose. A whole-stream order is `BatchSorted` meeting `SingleBatch`,
/// derived by [`PartitionLayout::is_stream_sorted`] rather than declared, so there is no
/// second way to say it. It becomes a real third state only under #138's ranged merge
/// emission, which orders a stream across several batches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SortOrder {
    NotSpecified,
    BatchSorted { columns: Vec<ColumnOrder> },
}

impl SortOrder {
    /// An order on no columns is no order, so it canonicalizes — otherwise two layouts
    /// that mean the same thing compare unequal.
    pub fn batch_sorted(columns: Vec<ColumnOrder>) -> Self {
        if columns.is_empty() {
            Self::NotSpecified
        } else {
            Self::BatchSorted { columns }
        }
    }

    pub fn is_batch_sorted(&self) -> bool {
        matches!(self, Self::BatchSorted { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchLayout {
    SingleBatch,
    MultipleBatches,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionLayout {
    pub n: usize,
    pub key_distribution: KeyDistribution,
    pub sort_order: SortOrder,
    pub batch_layout: BatchLayout,
}

impl PartitionLayout {
    /// N lanes, nothing else declared — what a scan or a shuffle-free chain emits.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            key_distribution: KeyDistribution::NotSpecified,
            sort_order: SortOrder::NotSpecified,
            batch_layout: BatchLayout::MultipleBatches,
        }
    }

    /// Whole stream ordered, not merely each batch — what a top-N after a sort needs.
    pub fn is_stream_sorted(&self) -> bool {
        self.sort_order.is_batch_sorted() && self.batch_layout == BatchLayout::SingleBatch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(column: u32) -> ColumnOrder {
        ColumnOrder {
            column,
            ascending: true,
            nulls_first: false,
        }
    }

    #[test]
    fn empty_sort_columns_canonicalize_to_not_specified() {
        assert_eq!(SortOrder::batch_sorted(vec![]), SortOrder::NotSpecified);
        assert!(!SortOrder::batch_sorted(vec![]).is_batch_sorted());
        assert!(SortOrder::batch_sorted(vec![key(0)]).is_batch_sorted());
    }

    #[test]
    fn stream_sorted_is_batch_sorted_meeting_single_batch() {
        let mut layout = PartitionLayout::new(1);
        assert!(!layout.is_stream_sorted());

        layout.sort_order = SortOrder::batch_sorted(vec![key(0)]);
        assert!(
            !layout.is_stream_sorted(),
            "many batches, each sorted, is not a sorted stream"
        );

        layout.batch_layout = BatchLayout::SingleBatch;
        assert!(layout.is_stream_sorted());

        layout.sort_order = SortOrder::NotSpecified;
        assert!(
            !layout.is_stream_sorted(),
            "one batch is not an ordered one"
        );
    }

    #[test]
    fn layouts_are_equal_exactly_when_every_field_agrees() {
        let base = PartitionLayout {
            n: 4,
            key_distribution: KeyDistribution::ByHash {
                hash_keys: vec![0, 2],
            },
            sort_order: SortOrder::batch_sorted(vec![key(1)]),
            batch_layout: BatchLayout::SingleBatch,
        };
        assert_eq!(base, base.clone());

        let mut lanes = base.clone();
        lanes.n = 8;
        assert_ne!(base, lanes);

        let mut keys = base.clone();
        keys.key_distribution = KeyDistribution::ByHash {
            hash_keys: vec![2, 0],
        };
        assert_ne!(base, keys, "hash key order is the routing, not a set");

        let mut sorted = base.clone();
        sorted.sort_order = SortOrder::batch_sorted(vec![ColumnOrder {
            nulls_first: true,
            ..key(1)
        }]);
        assert_ne!(base, sorted);
    }

    #[test]
    fn hash_keys_must_be_a_subset_of_the_group_columns() {
        let hashed = KeyDistribution::ByHash {
            hash_keys: vec![1, 3],
        };
        assert!(hashed.is_subset_of(&[0, 1, 3]));
        assert!(!hashed.is_subset_of(&[1, 2]));
        assert!(!KeyDistribution::NotSpecified.is_subset_of(&[0, 1]));
    }
}
