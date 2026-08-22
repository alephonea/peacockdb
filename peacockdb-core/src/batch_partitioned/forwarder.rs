//! Batch forwarding: the nodes that renumber lanes without touching rows.
//!
//! `GpuMergePartitions`, `GpuUnion` and `GpuInterleave` are one driver arm over three
//! mappings. A visit to output lane p cycles `sources_of(p)` in listed order, forwarding
//! one batch per visit, skipping sources with nothing queued and retiring those whose
//! producer has finished — the merge's round-robin and the interleave's per-lane child
//! rotation are that same rule.

/// Routes whole batches into a new lane numbering; never touches rows, never buffers.
/// No backends and no `CallStats` — routing is driver work, and a batch's bytes are
/// already accounted as driver-held in flight.
pub trait BatchForwarder {
    /// The (child index, child lane) pairs feeding output lane p, in service order.
    fn sources_of(&self, out_lane: usize) -> Vec<(usize, usize)>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Forwarder {
    /// N -> 1: lane 0 takes every lane of the one child, round-robin.
    MergePartitions { n: usize },
    /// Lane counts sum: output lane k is served by exactly one (child, lane).
    Union { lanes: Vec<(usize, usize)> },
    /// Child-major: output lane p is lane p of each child, which is why the inputs must
    /// share a hash distribution.
    Interleave { children: usize, n: usize },
}

impl BatchForwarder for Forwarder {
    fn sources_of(&self, out_lane: usize) -> Vec<(usize, usize)> {
        match self {
            Self::MergePartitions { n } => {
                assert_eq!(out_lane, 0, "GpuMergePartitions has one output lane");
                (0..*n).map(|lane| (0, lane)).collect()
            }
            Self::Union { lanes } => vec![lanes[out_lane]],
            Self::Interleave { children, n } => {
                assert!(
                    out_lane < *n,
                    "output lane {out_lane} is outside the interleave"
                );
                (0..*children).map(|child| (child, out_lane)).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_takes_every_lane_of_its_one_child() {
        assert_eq!(
            Forwarder::MergePartitions { n: 3 }.sources_of(0),
            vec![(0, 0), (0, 1), (0, 2)]
        );
    }

    #[test]
    fn union_lanes_have_exactly_one_source_each() {
        let union = Forwarder::Union {
            lanes: vec![(0, 0), (0, 1), (1, 0)],
        };
        assert_eq!(union.sources_of(0), vec![(0, 0)]);
        assert_eq!(union.sources_of(2), vec![(1, 0)]);
    }

    #[test]
    fn interleave_serves_lane_p_from_lane_p_of_every_child() {
        let interleave = Forwarder::Interleave { children: 3, n: 2 };
        assert_eq!(interleave.sources_of(1), vec![(0, 1), (1, 1), (2, 1)]);
    }
}
