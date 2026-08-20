//! The corners are enumerated rather than sampled, and then a differential test asserts
//! the incremental schedule agrees with a naive rescan on randomized shapes: if the two
//! disagree the incremental one is wrong by definition, which is the only cheap way to
//! keep it honest as the holds grow.

use super::*;

/// A shape from parents in pre-order. Subtrees are contiguous because the numbering is,
/// which is the property every hold range rests on.
fn shape(parents: &[Option<usize>], lanes: &[usize], joins: &[(usize, usize)]) -> PlanShape {
    let n = parents.len();
    let mut heights = vec![0u32; n];
    for node in 1..n {
        heights[node] = heights[parents[node].expect("only the root has no parent")] + 1;
    }
    let mut end = vec![0usize; n];
    for node in (0..n).rev() {
        end[node] = node + 1;
        for child in node + 1..n {
            if parents[child] == Some(node) {
                end[node] = end[node].max(end[child]);
            }
        }
    }
    let subtree: Vec<(usize, usize)> = (0..n).map(|node| (node, end[node])).collect();
    PlanShape {
        heights,
        lanes: lanes.to_vec(),
        subtree: subtree.clone(),
        joins: joins
            .iter()
            .map(|(node, probe)| JoinShape {
                node: *node,
                probe: subtree[*probe],
                lanes: lanes[*node],
            })
            .collect(),
    }
}

/// A chain root -> ... -> leaf, one lane each.
fn chain(len: usize) -> PlanShape {
    let parents: Vec<Option<usize>> = (0..len)
        .map(|node| if node == 0 { None } else { Some(node - 1) })
        .collect();
    shape(&parents, &vec![1; len], &[])
}

#[test]
fn nothing_ready_is_how_a_run_ends() {
    let mut scheduler = Scheduler::new(&chain(3));
    assert_eq!(scheduler.next(), None);
    scheduler.set_lane_ready(2, 0, true);
    assert_eq!(scheduler.next(), Some(2));
    scheduler.set_lane_ready(2, 0, false);
    assert_eq!(scheduler.next(), None);
}

#[test]
fn one_node_is_its_own_schedule() {
    let mut scheduler = Scheduler::new(&shape(&[None], &[1], &[]));
    assert_eq!(scheduler.next(), None);
    scheduler.set_lane_ready(0, 0, true);
    assert_eq!(scheduler.next(), Some(0));
}

#[test]
fn the_pick_is_the_smallest_height_among_the_ready() {
    let mut scheduler = Scheduler::new(&chain(4));
    for node in [3, 1, 2] {
        scheduler.set_lane_ready(node, 0, true);
    }
    assert_eq!(scheduler.next(), Some(1), "height 1 beats heights 2 and 3");
}

#[test]
fn ties_at_one_height_break_leftmost() {
    // root(0) over two subtrees: 1 -> 2 and 3 -> 4. Nodes 1 and 3 share a height, and
    // the build-side-drains-first argument is exactly this tie going to 1.
    let mut scheduler = Scheduler::new(&shape(
        &[None, Some(0), Some(1), Some(0), Some(3)],
        &[1; 5],
        &[],
    ));
    scheduler.set_lane_ready(3, 0, true);
    assert_eq!(scheduler.next(), Some(3));
    scheduler.set_lane_ready(1, 0, true);
    assert_eq!(
        scheduler.next(),
        Some(1),
        "equal heights go to the leftmost"
    );
}

#[test]
fn a_node_is_ready_while_any_of_its_lanes_is() {
    let mut scheduler = Scheduler::new(&shape(&[None, Some(0)], &[1, 3], &[]));
    for lane in 0..3 {
        scheduler.set_lane_ready(1, lane, true);
    }
    scheduler.set_lane_ready(1, 0, false);
    scheduler.set_lane_ready(1, 2, false);
    assert_eq!(scheduler.next(), Some(1), "lane 1 can still step");
    scheduler.set_lane_ready(1, 1, false);
    assert_eq!(scheduler.next(), None);
}

#[test]
fn a_lane_reported_ready_twice_is_counted_once() {
    let mut scheduler = Scheduler::new(&shape(&[None, Some(0)], &[1, 2], &[]));
    scheduler.set_lane_ready(1, 0, true);
    scheduler.set_lane_ready(1, 0, true);
    scheduler.set_lane_ready(1, 0, false);
    assert_eq!(
        scheduler.next(),
        None,
        "a repeated report must not leave a phantom ready lane behind"
    );
}

#[test]
fn readiness_comes_and_goes_and_comes_back() {
    let mut scheduler = Scheduler::new(&chain(2));
    scheduler.set_lane_ready(1, 0, true);
    assert_eq!(scheduler.next(), Some(1));
    scheduler.set_lane_ready(1, 0, false);
    assert_eq!(scheduler.next(), None);
    scheduler.set_lane_ready(1, 0, true);
    assert_eq!(scheduler.next(), Some(1));
}

/// unload(0) <- join(1) <- [build 2 <- 3, probe 4 <- 5]
fn one_join(lanes: usize) -> PlanShape {
    shape(
        &[None, Some(0), Some(1), Some(2), Some(1), Some(4)],
        &[lanes; 6],
        &[(1, 4)],
    )
}

#[test]
fn a_join_holds_its_probe_subtree_from_time_zero() {
    let mut scheduler = Scheduler::new(&one_join(1));
    for node in [3, 5] {
        scheduler.set_lane_ready(node, 0, true);
    }
    assert!(scheduler.is_held(4) && scheduler.is_held(5));
    assert_eq!(
        scheduler.next(),
        Some(3),
        "only the build subtree is schedulable before set_build"
    );
    scheduler.set_lane_ready(3, 0, false);
    assert_eq!(
        scheduler.next(),
        None,
        "the probe side has a batch to make and is still held"
    );
    scheduler.lane_left_build(1);
    assert_eq!(scheduler.next(), Some(5), "the probe subtree is free now");
}

#[test]
fn the_hold_lifts_only_when_every_lane_has_left_build() {
    let mut scheduler = Scheduler::new(&one_join(3));
    scheduler.set_lane_ready(5, 0, true);
    scheduler.lane_left_build(1);
    scheduler.lane_left_build(1);
    assert_eq!(scheduler.next(), None, "one lane is still building");
    scheduler.lane_left_build(1);
    assert_eq!(scheduler.next(), Some(5));
}

#[test]
fn a_node_under_two_builds_stays_held_when_the_inner_one_lifts() {
    // unload(0) <- outer join(1) <- [build 2, probe 3 = inner join <- [build 4, probe 5]]
    let inner_probe_holder = 5;
    let mut scheduler = Scheduler::new(&shape(
        &[None, Some(0), Some(1), Some(1), Some(3), Some(3)],
        &[1; 6],
        &[(1, 3), (3, 5)],
    ));
    scheduler.set_lane_ready(inner_probe_holder, 0, true);
    assert!(scheduler.is_held(inner_probe_holder));
    scheduler.lane_left_build(3);
    assert!(
        scheduler.is_held(inner_probe_holder),
        "the outer join's hold is a second count, not the same one"
    );
    assert_eq!(scheduler.next(), None);
    scheduler.lane_left_build(1);
    assert_eq!(scheduler.next(), Some(inner_probe_holder));
}

#[test]
fn a_join_inside_another_joins_build_subtree_is_not_held() {
    // unload(0) <- outer(1) <- [build 2 = inner join <- [3, 4], probe 5]
    let mut scheduler = Scheduler::new(&shape(
        &[None, Some(0), Some(1), Some(2), Some(2), Some(1)],
        &[1; 6],
        &[(1, 5), (2, 4)],
    ));
    scheduler.set_lane_ready(3, 0, true);
    assert!(!scheduler.is_held(3), "a build subtree is never held");
    assert_eq!(scheduler.next(), Some(3));
}

#[test]
fn a_satisfied_limit_and_a_join_hold_the_same_node_independently() {
    let mut scheduler = Scheduler::new(&one_join(1));
    scheduler.set_lane_ready(5, 0, true);
    scheduler.satisfy(4);
    scheduler.lane_left_build(1);
    assert!(
        scheduler.is_held(5),
        "the join's hold lifted; the limit's did not"
    );
    assert_eq!(scheduler.next(), None);
}

#[test]
fn a_limit_of_zero_is_satisfied_before_a_single_step() {
    let mut scheduler = Scheduler::new(&chain(3));
    for node in 0..3 {
        scheduler.set_lane_ready(node, 0, true);
    }
    scheduler.satisfy(0);
    assert!(scheduler.any_satisfied());
    assert_eq!(
        scheduler.next(),
        None,
        "a satisfied root holds the whole plan, itself included"
    );
}

#[test]
fn satisfying_twice_holds_once() {
    let mut scheduler = Scheduler::new(&chain(3));
    scheduler.set_lane_ready(2, 0, true);
    scheduler.satisfy(1);
    scheduler.satisfy(1);
    assert!(scheduler.is_held(2));
    // Nothing lifts a limit's hold, so the check that a second satisfy did not double the
    // counter is that the node is held by exactly one: drop it once and it is free.
    scheduler.limit_holds[2] -= 1;
    scheduler.refresh(2);
    assert_eq!(scheduler.next(), Some(2));
}

// -- the differential test ---------------------------------------------------------

/// The predicate rewritten as the prototype states it: rescan every node, walk both hold
/// relations from scratch. Slow and obviously correct, which is the whole point.
struct Naive {
    shape: PlanShape,
    lane_ready: Vec<Vec<bool>>,
    building: Vec<usize>,
    satisfied: Vec<bool>,
}

impl Naive {
    fn new(shape: &PlanShape) -> Self {
        Self {
            lane_ready: shape.lanes.iter().map(|n| vec![false; *n]).collect(),
            building: shape.joins.iter().map(|join| join.lanes).collect(),
            satisfied: vec![false; shape.node_count()],
            shape: shape.clone(),
        }
    }

    fn held(&self, node: usize) -> bool {
        let building = self
            .shape
            .joins
            .iter()
            .zip(&self.building)
            .any(|(join, left)| *left > 0 && (join.probe.0..join.probe.1).contains(&node));
        let satisfied = (0..self.shape.node_count()).any(|other| {
            let (start, end) = self.shape.subtree[other];
            self.satisfied[other] && (start..end).contains(&node)
        });
        building || satisfied
    }

    fn pick(&self) -> Option<usize> {
        (0..self.shape.node_count())
            .filter(|node| self.lane_ready[*node].iter().any(|ready| *ready))
            .filter(|node| !self.held(*node))
            .min_by_key(|node| (self.shape.heights[*node], *node))
    }
}

/// xorshift64*, seeded and fixed: the determinism rules apply to tests too.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// Grown by expansion rather than by attaching to an arbitrary earlier node: only that
/// gives the pre-order numbering `PlanIndex` produces, and the contiguous subtrees every
/// hold range rests on. At most two children, so joins are the shape rather than a fan.
fn random_shape(rng: &mut Rng, size: usize) -> PlanShape {
    fn grow(
        rng: &mut Rng,
        parents: &mut Vec<Option<usize>>,
        parent: Option<usize>,
        left: &mut usize,
    ) {
        let me = parents.len();
        parents.push(parent);
        *left -= 1;
        let children = if *left == 0 { 0 } else { rng.below(3) };
        for _ in 0..children {
            if *left == 0 {
                break;
            }
            grow(rng, parents, Some(me), left);
        }
    }

    let mut parents = Vec::new();
    let mut left = size;
    grow(rng, &mut parents, None, &mut left);

    let children_of = |node: usize| -> Vec<usize> {
        (0..parents.len())
            .filter(|child| parents[*child] == Some(node))
            .collect()
    };
    let lanes: Vec<usize> = (0..parents.len()).map(|_| 1 + rng.below(3)).collect();
    let joins: Vec<(usize, usize)> = (0..parents.len())
        .filter_map(|node| {
            let children = children_of(node);
            (children.len() == 2 && rng.below(2) == 0).then(|| (node, children[1]))
        })
        .collect();
    shape(&parents, &lanes, &joins)
}

/// Every subtree a shape declares is contiguous, which is what the hold ranges are.
fn assert_contiguous(shape: &PlanShape) {
    for node in 0..shape.node_count() {
        let (start, end) = shape.subtree[node];
        assert_eq!(start, node, "a subtree starts at its own node");
        for inside in start..end {
            assert!(
                inside == node || shape.heights[inside] > shape.heights[node],
                "node {inside} is inside node {node}'s range and is not below it"
            );
        }
    }
}

#[test]
fn the_incremental_schedule_picks_what_a_full_rescan_would() {
    let mut rng = Rng(0x5eed_1234_9876_4321);
    for _ in 0..200 {
        // Growth stops when a node draws no children, so the tree is at most this big and
        // every index below comes off what it actually produced.
        let budget = 1 + rng.below(12);
        let shape = random_shape(&mut rng, budget);
        assert_contiguous(&shape);
        let size = shape.node_count();
        let mut scheduler = Scheduler::new(&shape);
        let mut naive = Naive::new(&shape);
        assert_eq!(scheduler.next(), naive.pick());
        for _ in 0..80 {
            let node = rng.below(size);
            match rng.below(6) {
                0..=3 => {
                    let lane = rng.below(shape.lanes[node]);
                    let ready = rng.below(2) == 0;
                    scheduler.set_lane_ready(node, lane, ready);
                    naive.lane_ready[node][lane] = ready;
                }
                4 => {
                    let join = shape.joins.iter().position(|join| join.node == node);
                    if let Some(index) = join.filter(|index| naive.building[*index] > 0) {
                        scheduler.lane_left_build(node);
                        naive.building[index] -= 1;
                    }
                }
                _ => {
                    scheduler.satisfy(node);
                    naive.satisfied[node] = true;
                }
            }
            assert_eq!(
                scheduler.next(),
                naive.pick(),
                "the two schedules disagree on shape {shape:?}"
            );
        }
    }
}
