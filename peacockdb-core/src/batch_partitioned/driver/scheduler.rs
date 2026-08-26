//! Which node runs next: smallest height among the runnable, ties leftmost.
//!
//! Maintained incrementally rather than rescanned. The predicate the prototype evaluates
//! per step — every node, every lane, and both hold chains to the root — is
//! O(nodes × lanes × depth) in front of work that is O(lanes) executor calls. Here each
//! event that can change an answer updates the one node it touches, and a pick is a scan
//! for the lowest set bit of a bitset ordered by (height, order).
//!
//! Nothing here knows about backends, batches or executors: it takes events and returns a
//! node. That is what makes a scheduling corner reachable from a test that builds no plan.

/// A join, as the schedule sees one: the range of its probe subtree, and how many lanes
/// have yet to leave their build phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JoinShape {
    pub node: usize,
    /// `[start, end)` of the probe child's subtree in pre-order.
    pub probe: (usize, usize),
    pub lanes: usize,
}

/// The tree as plain numbers. Node indices are pre-order, so a subtree is a contiguous
/// range and "order" — the leftmost tie-break — is the index itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlanShape {
    pub heights: Vec<u32>,
    pub lanes: Vec<usize>,
    pub subtree: Vec<(usize, usize)>,
    pub joins: Vec<JoinShape>,
}

impl PlanShape {
    pub(crate) fn node_count(&self) -> usize {
        self.heights.len()
    }
}

pub(crate) struct Scheduler {
    /// rank -> node, sorted by (height, index): the first set bit is the pick.
    node_of_rank: Vec<usize>,
    rank_of_node: Vec<usize>,
    words: Vec<u64>,
    /// per node, how many of its lanes can step. A recheck is this against the holds,
    /// never a sweep over lanes.
    ready_lanes: Vec<u32>,
    lane_ready: Vec<bool>,
    lane_base: Vec<usize>,
    /// Per node, how many readiness indices it has. Kept so a caller passing an output-lane
    /// index where a readiness index belongs is caught here rather than silently writing
    /// into the next node's flags.
    lane_count: Vec<usize>,
    /// Counters, not flags: one node can sit in two joins' probe subtrees, and a lift of
    /// one of them must not free it from the other.
    build_holds: Vec<u32>,
    limit_holds: Vec<u32>,
    /// per join, lanes that have yet to leave their build phase
    building: Vec<usize>,
    joins: Vec<JoinShape>,
    satisfied: Vec<bool>,
    subtree: Vec<(usize, usize)>,
}

impl Scheduler {
    /// A join holds its probe subtree from time zero: no lane has left build yet, and a
    /// probe batch produced before `set_build` has nowhere to go.
    pub(crate) fn new(shape: &PlanShape) -> Self {
        let n = shape.node_count();
        let mut node_of_rank: Vec<usize> = (0..n).collect();
        node_of_rank.sort_by_key(|node| (shape.heights[*node], *node));
        let mut rank_of_node = vec![0; n];
        for (rank, node) in node_of_rank.iter().enumerate() {
            rank_of_node[*node] = rank;
        }
        let mut lane_base = Vec::with_capacity(n);
        let mut lanes_total = 0;
        for lanes in &shape.lanes {
            lane_base.push(lanes_total);
            lanes_total += lanes;
        }
        let mut scheduler = Self {
            node_of_rank,
            rank_of_node,
            words: vec![0; n.div_ceil(64)],
            ready_lanes: vec![0; n],
            lane_ready: vec![false; lanes_total],
            lane_base,
            lane_count: shape.lanes.clone(),
            build_holds: vec![0; n],
            limit_holds: vec![0; n],
            building: shape.joins.iter().map(|join| join.lanes).collect(),
            joins: shape.joins.clone(),
            satisfied: vec![false; n],
            subtree: shape.subtree.clone(),
        };
        for join in &shape.joins {
            for node in join.probe.0..join.probe.1 {
                scheduler.build_holds[node] += 1;
            }
        }
        scheduler
    }

    /// The pick: the runnable node of smallest height, ties leftmost. `None` ends the run.
    pub(crate) fn next(&self) -> Option<usize> {
        self.words
            .iter()
            .enumerate()
            .find(|(_, word)| **word != 0)
            .map(|(index, word)| self.node_of_rank[index * 64 + word.trailing_zeros() as usize])
    }

    /// Whether this lane of this node can make progress. Idempotent, so the driver may
    /// recompute a node's lanes after any step without tracking what changed.
    pub(crate) fn set_lane_ready(&mut self, node: usize, lane: usize, ready: bool) {
        debug_assert!(
            lane < self.lane_count[node],
            "readiness index {lane} is outside the {} this node has",
            self.lane_count[node]
        );
        let slot = self.lane_base[node] + lane;
        if self.lane_ready[slot] == ready {
            return;
        }
        self.lane_ready[slot] = ready;
        if ready {
            self.ready_lanes[node] += 1;
        } else {
            self.ready_lanes[node] -= 1;
        }
        self.refresh(node);
    }

    /// One lane of a join has run `set_build`. The hold lifts when every lane has, since
    /// a lane still building still cannot take a probe batch.
    pub(crate) fn lane_left_build(&mut self, join_node: usize) {
        let index = self
            .joins
            .iter()
            .position(|join| join.node == join_node)
            .expect("a join reports its own node");
        assert!(
            self.building[index] > 0,
            "a join lane left its build phase twice"
        );
        self.building[index] -= 1;
        if self.building[index] > 0 {
            return;
        }
        let probe = self.joins[index].probe;
        for node in probe.0..probe.1 {
            self.build_holds[node] -= 1;
            self.refresh(node);
        }
    }

    /// Enough rows have passed this node that no later one can change its answer. It
    /// holds its whole subtree, itself included, and the hold never lifts — so this is
    /// where a run ends with lanes not done and queues non-empty.
    pub(crate) fn satisfy(&mut self, node: usize) {
        if self.satisfied[node] {
            return;
        }
        self.satisfied[node] = true;
        let (start, end) = self.subtree[node];
        for held in start..end {
            self.limit_holds[held] += 1;
            self.refresh(held);
        }
    }

    /// Any satisfied node at all — the run ended early, so in-flight batches are dropped
    /// rather than reported as stranded.
    pub(crate) fn is_satisfied(&self, node: usize) -> bool {
        self.satisfied[node]
    }

    pub(crate) fn any_satisfied(&self) -> bool {
        self.satisfied.iter().any(|satisfied| *satisfied)
    }

    pub(crate) fn is_held(&self, node: usize) -> bool {
        self.build_holds[node] > 0 || self.limit_holds[node] > 0
    }

    fn refresh(&mut self, node: usize) {
        let runnable = self.ready_lanes[node] > 0 && !self.is_held(node);
        let rank = self.rank_of_node[node];
        let (word, bit) = (rank / 64, 1u64 << (rank % 64));
        if runnable {
            self.words[word] |= bit;
        } else {
            self.words[word] &= !bit;
        }
    }
}

#[cfg(test)]
mod tests;
