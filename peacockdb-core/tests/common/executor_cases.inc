// The contract the two backends answer to, as data.
//
// Each backend is proved against its own oracle over its own fixture, which says nothing
// about the two agreeing — and a mode whose whole claim is that one plan runs on either
// engine needs that said somewhere. The instrument is `gpu_cases.inc`'s: a case list
// included by several targets, so a case added here reaches every engine claiming the
// shape. `INPUT` is the fixture for both, and the device writes its parquet from it — a
// table one side does not read is a table that proves the CPU twice.

/// The rows every case starts from, as `(k, v)`. Small enough to write an answer down, and
/// split into three batches by the device fixture's row groups.
pub const INPUT: [(&str, i64); 6] = [
    ("a", 2),
    ("b", 1),
    ("a", 4),
    ("b", 3),
    ("a", 6),
    ("b", 5),
];

/// What a case asks of a backend. One node each, since what is under test is the answer
/// rather than a plan: a shape both engines run, driven the way that engine drives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// `v > n`, per batch.
    Filter { above: i64 },
    /// `v * 2`, per batch.
    Double,
    /// The whole lane ordered by `v`, with an optional top-N.
    SortLane { fetch: Option<usize> },
    /// `sum(v) GROUP BY k`, built per batch and merged at done.
    SumByKey { finalize: bool },
    /// `sum(v) GROUP BY k, __grouping_id` — the merge side of a grouping-set plan, where
    /// the state's key width is one wider than the group list and this branch's own
    /// `key_width` change is what decides which column is whose.
    SumByKeyAndGroupingId,
    /// Every batch of the lane concatenated at done.
    CoalesceLane,
    /// The hash scatter: the answer is which lane each row landed in.
    ScatterLanes { lanes: usize },
}

impl Shape {
    /// Whether the order of the answer is part of it. A sort's is — comparing its rows as
    /// a set is an assertion no unsorted sort could fail — while a grouped aggregate's is
    /// its hash table's and a scatter's is the lane walk's.
    pub fn order_is_the_answer(&self) -> bool {
        matches!(self, Shape::SortLane { .. })
    }
}

/// One case: what it is called, what it does, and the answer both engines owe.
pub struct Case {
    pub name: &'static str,
    pub shape: Shape,
    /// The answer as `k|v` rows, sorted — or `lane|k|v` where the shape is a scatter.
    pub expect: &'static [&'static str],
}

/// The lane numbers below are a golden: they were taken from a run, and they are the
/// contract rather than an observation, because co-partitioning is what every partitioned
/// join rests on. The two engines hash with one rule — comet's Spark-murmur3 on this side,
/// `spark_hash_partition` on the device, held bit-equal by `test_inc2_conformance` — so a
/// row landing in a different lane on either engine means a join would silently drop
/// matches, and this is where that goes red.
pub const CASES: &[Case] = &[
    Case {
        name: "a filter keeps the rows above its bound",
        shape: Shape::Filter { above: 3 },
        expect: &["a|4", "a|6", "b|5"],
    },
    Case {
        name: "a filter that keeps nothing still answers",
        shape: Shape::Filter { above: 100 },
        expect: &[],
    },
    Case {
        name: "a project evaluates its expression",
        shape: Shape::Double,
        expect: &["a|12", "a|4", "a|8", "b|10", "b|2", "b|6"],
    },
    Case {
        name: "a sort orders the whole lane",
        shape: Shape::SortLane { fetch: None },
        expect: &["b|1", "a|2", "b|3", "a|4", "b|5", "a|6"],
    },
    Case {
        name: "a sort with a fetch keeps the smallest of the lane",
        shape: Shape::SortLane { fetch: Some(2) },
        expect: &["b|1", "a|2"],
    },
    Case {
        name: "a coalesce holds every row of the lane",
        shape: Shape::CoalesceLane,
        expect: &["a|2", "a|4", "a|6", "b|1", "b|3", "b|5"],
    },
    Case {
        name: "a merge folds the partials its init produced",
        shape: Shape::SumByKey { finalize: false },
        expect: &["a|12", "b|9"],
    },
    Case {
        name: "a merge that finalizes answers in its declared columns",
        shape: Shape::SumByKey { finalize: true },
        expect: &["a|12", "b|9"],
    },
    Case {
        name: "a merge over a state whose keys include a grouping id",
        shape: Shape::SumByKeyAndGroupingId,
        expect: &["a|12", "b|9"],
    },
    Case {
        name: "the scatter puts each row in the lane its key hashes to",
        shape: Shape::ScatterLanes { lanes: 4 },
        expect: &["1|b|1", "1|b|3", "1|b|5", "2|a|2", "2|a|4", "2|a|6"],
    },
    Case {
        name: "the scatter at a lane count past any one batch's rows",
        shape: Shape::ScatterLanes { lanes: 64 },
        expect: &["18|a|2", "18|a|4", "18|a|6", "1|b|1", "1|b|3", "1|b|5"],
    },
];
