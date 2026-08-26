//! What a recipe is made of: the four ABI symbols, the legacy node kinds they address,
//! the handles a call is passed, and when the driver makes it.

use std::fmt;

use datafusion::common::JoinType;

use crate::generated::gpu_plan_generated::peacock::plan as fb;

/// A node of the recipe plan, addressed by its position in it. The number is the whole
/// content of an address, which is why a call carries nothing else about the node it runs.
pub type Seq = u32;

/// The frozen entry points. Two of them take runtime bounds instead of a seq, which is
/// the reason they exist: a frozen node cannot carry a number that is known only once the
/// rows have been counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiSymbol {
    ExecuteNode,
    ExecuteScanRowGroups,
    SliceHandle,
    ResultFromHandle,
}

impl AbiSymbol {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ExecuteNode => "execute_node",
            Self::ExecuteScanRowGroups => "execute_scan_rowgroups",
            Self::SliceHandle => "slice_handle",
            Self::ResultFromHandle => "result_from_handle",
        }
    }
}

/// Which of a join's two projects a seq is. Both are `CudfProject`, and telling them
/// apart in a recipe of five calls is the difference between reading it and decoding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectRole {
    /// The probe keys this batch contributes to the accumulation (#136).
    ProbeKeys,
    /// An aggregate's finalize, which is ours rather than the executor's: both engines
    /// evaluate this expression, so they agree by construction.
    Finalize,
    /// Build columns straight through, plus one typed NULL per probe column the join's
    /// projection keeps — what makes the anti join's output the joined schema.
    NullPad { nulls: usize },
}

/// The legacy node kinds this mode addresses, carrying the fields a reader cannot get
/// from the plan line above: a per-call join type is not the node's own, and a
/// repartition's lane count is the one number a recipe repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FbKind {
    Scan,
    Filter,
    Project(ProjectRole),
    /// The one a map arm runs; a `CudfProject` with no role of its own.
    PlainProject,
    /// `Partial` builds state from raw values and `Merge` merges state into state.
    /// Never `Final`, which would also finalize, and a finalize here is a project.
    Aggregate {
        merge: bool,
    },
    Sort,
    SortPreservingMerge,
    CoalescePartitions,
    Repartition {
        lanes: u32,
    },
    HashJoin {
        join_type: JoinType,
    },
    CrossJoin,
    NestedLoopJoin,
}

impl FbKind {
    /// The node kind on the wire. What a recipe claims and what the buffer holds are
    /// checked against each other through this — see `read::check_seq_kinds`.
    pub fn wire_kind(&self) -> fb::PlanNodeKind {
        match self {
            Self::Scan => fb::PlanNodeKind::CudfScan,
            Self::Filter => fb::PlanNodeKind::CudfFilter,
            Self::Project(_) | Self::PlainProject => fb::PlanNodeKind::CudfProject,
            Self::Aggregate { .. } => fb::PlanNodeKind::CudfAggregate,
            Self::Sort => fb::PlanNodeKind::CudfSort,
            Self::SortPreservingMerge => fb::PlanNodeKind::CudfSortPreservingMerge,
            Self::CoalescePartitions => fb::PlanNodeKind::CudfCoalescePartitions,
            Self::Repartition { .. } => fb::PlanNodeKind::CudfRepartition,
            Self::HashJoin { .. } => fb::PlanNodeKind::CudfHashJoin,
            Self::CrossJoin => fb::PlanNodeKind::CudfCrossJoin,
            Self::NestedLoopJoin => fb::PlanNodeKind::CudfNestedLoopJoin,
        }
    }
}

/// Where a call's input comes from. The driver owns every one; naming them is what makes
/// a recipe checkable against consume-on-use (#152), since a copy appears here as a copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    /// The batch the call was scheduled for.
    Batch,
    /// A copy of it, because the call below consumes it and something else needs it too.
    /// The surface has no copy symbol, so a recipe naming this is one an executor refuses
    /// until [#145](../../../../llm-wiki/tickets.md#t145) — see #152.
    BatchCopy,
    /// The build side, handed over: the call consumes it and nothing needs it again.
    BuildSide,
    /// A copy of the build side, because the next probe batch needs it as well. Same
    /// absence as [`Input::BatchCopy`]: the first call gets the handle and a second is
    /// refused naming #152.
    BuildSideCopy,
    /// Every batch this lane accumulated.
    LaneBatches,
    /// Every lane's handle, partition-major.
    AllLanes,
    /// The probe keys kept per batch, which the finish pass joins against (#136).
    AccumulatedKeys,
    /// What the previous call in this recipe returned.
    PriorOutput,
    /// Not a handle: the row groups this batch reads, overriding the node's own list.
    RowGroups,
    /// Not a handle either: an offset and a length counted at run time.
    RowRange,
}

impl Input {
    /// Whether this input is the build side, handed over or copied. Asked by an executor
    /// pricing a call: a probe call that names neither is a call the build side does not
    /// have to be there for.
    pub fn is_build_side(&self) -> bool {
        matches!(self, Self::BuildSide | Self::BuildSideCopy)
    }

    fn text(&self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::BatchCopy => "batch copy",
            Self::BuildSide => "build",
            Self::BuildSideCopy => "build copy",
            Self::LaneBatches => "lane batches",
            Self::AllLanes => "all lanes",
            Self::AccumulatedKeys => "accumulated keys",
            Self::PriorOutput => "prior output",
            Self::RowGroups => "row groups",
            Self::RowRange => "row range",
        }
    }
}

/// When the driver makes a call. Two nodes emitting the same seq set differ by this and
/// nothing else — a sort per batch and a sort at done are different plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallPattern {
    PerBatch,
    PerProbeBatch,
    /// Once, when the node's input is complete.
    AtDone,
    /// Every time the accumulation compacts, and once more at done.
    PerCompaction,
    /// The two batches that straddle an interval's ends; a batch wholly inside is
    /// forwarded untouched and one wholly outside is released uncalled.
    PerStraddlingBatch,
    /// Once per handle that reaches the sink.
    PerHandle,
}

impl CallPattern {
    fn text(&self) -> &'static str {
        match self {
            Self::PerBatch => "per batch",
            Self::PerProbeBatch => "per probe batch",
            Self::AtDone => "at done",
            Self::PerCompaction => "per compaction and at done",
            Self::PerStraddlingBatch => "per straddling batch",
            Self::PerHandle => "per handle",
        }
    }
}

/// One ABI call: the symbol, the seq it addresses where it takes one, and what it passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub symbol: AbiSymbol,
    /// `None` for the two symbols whose arguments are runtime row counts.
    pub target: Option<(Seq, FbKind)>,
    pub inputs: Vec<Input>,
    pub when: CallPattern,
}

impl Call {
    /// A call against a recipe-plan node. The symbol follows from the kind: only a scan
    /// takes the row-group override, and everything else is the generic entry point.
    pub(super) fn seq(seq: Seq, kind: FbKind, inputs: Vec<Input>, when: CallPattern) -> Self {
        Self {
            symbol: match kind {
                FbKind::Scan => AbiSymbol::ExecuteScanRowGroups,
                _ => AbiSymbol::ExecuteNode,
            },
            target: Some((seq, kind)),
            inputs,
            when,
        }
    }

    pub(super) fn bare(symbol: AbiSymbol, inputs: Vec<Input>, when: CallPattern) -> Self {
        Self {
            symbol,
            target: None,
            inputs,
            when,
        }
    }
}

/// What one node does to the device, in call order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    pub calls: Vec<Call>,
}

impl Recipe {
    pub(super) fn of(calls: Vec<Call>) -> Self {
        Self { calls }
    }

    /// The seqs this node addresses, in the order it emits them.
    pub fn seqs(&self) -> Vec<Seq> {
        self.calls
            .iter()
            .filter_map(|call| call.target.map(|(seq, _)| seq))
            .collect()
    }
}

/// `per batch: execute_node(#3 CudfFilter, batch)`, the calls grouped under the pattern
/// that drives them — which is how the mapping table reads them too, per probe call and
/// at finish. What the plan line already carries is not repeated: a golden that states a
/// thing twice is one a reader stops checking.
impl fmt::Display for Recipe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut groups: Vec<(CallPattern, Vec<String>)> = Vec::new();
        for call in &self.calls {
            let target = match call.target {
                Some((seq, kind)) => format!("#{seq} {kind}, "),
                None => String::new(),
            };
            let inputs: Vec<&str> = call.inputs.iter().map(Input::text).collect();
            let text = format!("{}({target}{})", call.symbol.name(), inputs.join(", "));
            match groups.last_mut() {
                Some((when, calls)) if *when == call.when => calls.push(text),
                _ => groups.push((call.when, vec![text])),
            }
        }
        let phases: Vec<String> = groups
            .iter()
            .map(|(when, calls)| format!("{}: {}", when.text(), calls.join(", ")))
            .collect();
        write!(f, "{}", phases.join("; "))
    }
}

impl fmt::Display for FbKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scan => write!(f, "CudfScan"),
            Self::Filter => write!(f, "CudfFilter"),
            Self::Project(ProjectRole::ProbeKeys) => write!(f, "CudfProject{{probe keys}}"),
            Self::Project(ProjectRole::Finalize) => write!(f, "CudfProject{{finalize}}"),
            Self::Project(ProjectRole::NullPad { nulls }) => {
                write!(f, "CudfProject{{build columns + {nulls} null}}")
            }
            Self::PlainProject => write!(f, "CudfProject"),
            Self::Aggregate { merge } => {
                write!(
                    f,
                    "CudfAggregate{{{}}}",
                    if *merge { "Merge" } else { "Partial" }
                )
            }
            Self::Sort => write!(f, "CudfSort"),
            Self::SortPreservingMerge => write!(f, "CudfSortPreservingMerge"),
            Self::CoalescePartitions => write!(f, "CudfCoalescePartitions"),
            Self::Repartition { lanes } => write!(f, "CudfRepartition{{Hash, 1→{lanes}}}"),
            Self::HashJoin { join_type } => write!(f, "CudfHashJoin{{{join_type:?}}}"),
            Self::CrossJoin => write!(f, "CudfCrossJoin"),
            Self::NestedLoopJoin => write!(f, "CudfNestedLoopJoin"),
        }
    }
}
