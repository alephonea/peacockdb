//! The row-group → (partition, batch) mapping: one pure policy, computed once at plan
//! time, whose output everything else consumes verbatim — `GpuLoadParquet` stores it,
//! the plan golden renders it as `partition_groups=[...]`, the loader executes it,
//! validation checks the declared lane count against it.
//!
//! Balancing is by row count and batching is by bytes, and the bytes are the parquet
//! column-chunk totals over the projected columns: a varchar's width is a property of
//! the data, and the file metadata already holds the answer.

use super::error::PlanError;

/// One surviving row group, in file order. `index` is its index in the file, which is
/// what the scan passes to `set_row_groups` — survivors are post-pruning, so it is not
/// the position in this slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowGroupMeta {
    pub index: u32,
    pub rows: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Batching {
    Off,
    On { target_batch_bytes: usize },
}

/// Survivors → partitions → batches → row-group indices.
///
/// Contiguous chunks balanced by row count; within a chunk, consecutive row groups pack
/// greedily while bytes stay under target. A row group over target is still its own
/// batch — one row group is the minimum granularity, and the planner always emits a plan
/// (the enforcer owns the runtime consequence, #142). Contiguity is policy, not a cuDF
/// requirement; changing it regenerates goldens and is treated as that.
pub fn partition(
    survivors: &[RowGroupMeta],
    n_partitions: usize,
    batching: Batching,
) -> Result<Vec<Vec<Vec<u32>>>, PlanError> {
    if survivors.is_empty() {
        return Err(PlanError::Invalid(
            "no surviving row groups: what an empty scan means is the caller's decision, \
             not an empty map — the wire format reads that as legacy single-partition"
                .to_string(),
        ));
    }
    if n_partitions == 0 {
        return Err(PlanError::Invalid(
            "a source needs at least one lane to emit into".to_string(),
        ));
    }

    Ok(balanced_chunks(survivors, n_partitions)
        .into_iter()
        .map(|chunk| batches_of(&survivors[chunk], batching))
        .collect())
}

/// Contiguous ranges over the survivors, each holding about its share of the rows.
/// Chunks are empty only where there were fewer survivors than lanes: an empty lane is
/// an ordinary shape here, as it is for a hash that lands no key.
fn balanced_chunks(survivors: &[RowGroupMeta], n_partitions: usize) -> Vec<std::ops::Range<usize>> {
    let total: u64 = survivors.iter().map(|g| g.rows).sum();
    let mut chunks = Vec::with_capacity(n_partitions);
    let mut index = 0;
    let mut taken = 0;

    for part in 0..n_partitions {
        let want = (total - taken).div_ceil((n_partitions - part) as u64);
        let start = index;
        let mut got = 0;
        while index < survivors.len() {
            // Stop where taking the next group would land further from this lane's share
            // than stopping does. Overshooting instead costs the balance bound: two full
            // row groups against a want of one and a half puts the whole tail in one lane.
            let next = got + survivors[index].rows;
            if index > start && next.abs_diff(want) >= got.abs_diff(want) {
                break;
            }
            got = next;
            index += 1;
        }
        chunks.push(start..index);
        taken += got;
    }

    // The last lane's share is everything left, and every group takes it closer to that.
    debug_assert_eq!(
        index,
        survivors.len(),
        "a lane's worth of row groups went nowhere"
    );
    chunks
}

fn batches_of(chunk: &[RowGroupMeta], batching: Batching) -> Vec<Vec<u32>> {
    if chunk.is_empty() {
        return Vec::new();
    }
    let target = match batching {
        Batching::Off => return vec![chunk.iter().map(|g| g.index).collect()],
        Batching::On { target_batch_bytes } => target_batch_bytes as u64,
    };

    let mut batches: Vec<Vec<u32>> = Vec::new();
    let mut current: Vec<u32> = Vec::new();
    let mut bytes = 0;
    for group in chunk {
        if !current.is_empty() && bytes + group.bytes > target {
            batches.push(std::mem::take(&mut current));
            bytes = 0;
        }
        current.push(group.index);
        bytes += group.bytes;
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row groups with the file indices they would have after pruning dropped groups 0
    /// and 3 — the mapping carries file indices, never positions in the survivor list.
    fn pruned(sizes: &[(u64, u64)]) -> Vec<RowGroupMeta> {
        let skipped = [0, 3];
        (0..)
            .filter(|i| !skipped.contains(i))
            .zip(sizes)
            .map(|(index, &(rows, bytes))| RowGroupMeta { index, rows, bytes })
            .collect()
    }

    fn uniform(count: usize, rows: u64) -> Vec<RowGroupMeta> {
        (0..count as u32)
            .map(|index| RowGroupMeta {
                index,
                rows,
                bytes: rows * 4,
            })
            .collect()
    }

    fn lane_rows(survivors: &[RowGroupMeta], mapping: &[Vec<Vec<u32>>]) -> Vec<u64> {
        mapping
            .iter()
            .map(|batches| {
                batches
                    .iter()
                    .flatten()
                    .map(|g| survivors.iter().find(|m| m.index == *g).unwrap().rows)
                    .sum()
            })
            .collect()
    }

    #[test]
    fn fewer_survivors_than_lanes_leaves_lanes_empty() {
        let survivors = uniform(2, 100);
        let mapping = partition(&survivors, 4, Batching::Off).unwrap();
        assert_eq!(mapping, vec![vec![vec![0]], vec![vec![1]], vec![], vec![]]);
    }

    #[test]
    fn three_lanes_take_contiguous_chunks_of_the_survivors() {
        let survivors = uniform(6, 100);
        let mapping = partition(&survivors, 3, Batching::Off).unwrap();
        assert_eq!(
            mapping,
            vec![vec![vec![0, 1]], vec![vec![2, 3]], vec![vec![4, 5]]]
        );
    }

    #[test]
    fn a_row_group_over_target_is_its_own_batch() {
        let survivors = pruned(&[(10, 40), (10, 500), (10, 30), (10, 30)]);
        let mapping = partition(
            &survivors,
            1,
            Batching::On {
                target_batch_bytes: 100,
            },
        )
        .unwrap();
        assert_eq!(mapping, vec![vec![vec![1], vec![2], vec![4, 5]]]);
    }

    #[test]
    fn batching_off_is_one_batch_per_chunk() {
        let survivors = pruned(&[(10, 4_000), (10, 4_000), (10, 4_000), (10, 4_000)]);
        let mapping = partition(&survivors, 2, Batching::Off).unwrap();
        assert_eq!(mapping, vec![vec![vec![1, 2]], vec![vec![4, 5]]]);
        assert!(mapping.iter().all(|batches| batches.len() == 1));
    }

    #[test]
    fn an_empty_survivor_set_is_an_error_rather_than_an_empty_map() {
        let err = partition(&[], 4, Batching::Off).unwrap_err();
        assert!(matches!(err, PlanError::Invalid(_)), "{err}");
        assert!(partition(&uniform(2, 10), 0, Batching::Off).is_err());
    }

    /// The bound holds for what a parquet writer emits: one row-group size per file and a
    /// short last group. It is not universal — contiguity is the stronger rule, so row
    /// groups differing by orders of magnitude inside one file can beat it.
    #[test]
    fn lane_rows_differ_by_at_most_one_row_group() {
        for groups in 1..=12usize {
            for tail in [1, 7_431, 122_880] {
                for lanes in 1..=groups {
                    let mut survivors = uniform(groups, 122_880);
                    survivors.last_mut().unwrap().rows = tail;
                    let mapping = partition(&survivors, lanes, Batching::Off).unwrap();

                    let rows = lane_rows(&survivors, &mapping);
                    let spread = rows.iter().max().unwrap() - rows.iter().min().unwrap();
                    let largest = survivors.iter().map(|g| g.rows).max().unwrap();
                    assert!(
                        spread <= largest,
                        "{groups} groups (tail {tail}) over {lanes} lanes: \
                         spread {spread} exceeds one row group"
                    );
                }
            }
        }
    }

    #[test]
    fn the_mapping_is_a_fixed_function_of_its_inputs() {
        let survivors = pruned(&[(90, 900), (10, 100), (50, 500), (50, 500), (30, 300)]);
        let batching = Batching::On {
            target_batch_bytes: 600,
        };
        let expected = vec![vec![vec![1], vec![2]], vec![vec![4], vec![5], vec![6]]];
        assert_eq!(partition(&survivors, 2, batching).unwrap(), expected);
        assert_eq!(partition(&survivors, 2, batching).unwrap(), expected);
    }
}
