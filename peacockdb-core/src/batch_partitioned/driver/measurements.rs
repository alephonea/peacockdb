//! Joining the two halves of a measurement: what this side recorded per call, and what the
//! device reported per region.
//!
//! Neither half is a measurement on its own. The driver knows WHERE a call was — which
//! node, which lane, which batch — and the device knows WHAT it cost and what it produced.
//! They meet on `(seq, call_index)`, a key both sides count to independently and in the
//! same order.
//!
//! What comes out is a [`Measured`] per call, and the module is named for that rather than
//! for time: the output of a call in the middle of a node's chain is reported here and
//! nowhere else, because nothing on this side ever built a batch from it.

use std::collections::HashMap;

use super::RunReport;
use crate::batch_partitioned::recipe::Seq;

/// One measured region, as C++ reports it: which call it belonged to and what that call
/// cost. Its key is `(seq, call_index)`, which is what an `AbiCall` carries — the two
/// records are halves of one row and meet there.
///
/// Here rather than in `gpu_backend`, for the reason `AbiCalls` lives in `executor`: this
/// is what a measurement IS, and the backend fills one in. It is not the ABI struct — that
/// is `PeacockNodeRegion`, and `collect_regions` copies out of it field by field — so
/// nothing about it needs the FFI, and a plain `rust-only` build that has no backend at
/// all still has a driver that compiles.
///
/// One call can answer with several of these, one per output partition: a scatter's lanes,
/// a scan driven off a batch map. The shared prologue is charged to partition 0, so a
/// call's cost is the SUM over its partitions and never one of them.
#[derive(Debug, Clone, Copy)]
pub struct Region {
    pub seq: Seq,
    pub partition: usize,
    pub call_index: u64,
    pub host_setup_us: u64,
    pub host_submit_us: u64,
    /// Zero where the region recorded no complete event pair — it touched no device, or
    /// the events could not be created. Absent regions are dropped by C++, not zeroed;
    /// this zero is the other case.
    pub device_us: u64,
    /// Rows and bytes this call answered with, for this output partition. The only place a
    /// middle call's output exists: a node driving several hands its caller the last one's
    /// and drops the rest.
    pub out_rows: u64,
    /// C++'s own reconstruction of the byte total. Compared against this side's wherever
    /// this side has one, and consumed where it does not: a call in the middle of a node's
    /// chain hands the raw handle on, so nothing here ever priced it.
    pub logical_bytes: u64,
    /// Whether the produced types were one for one the declared ones, which is what scopes
    /// the comparison above.
    pub schema_faithful: bool,
}

/// What the device reported about one call, summed over the regions it produced.
///
/// Summed rather than picked: a call that answers with several output partitions charges
/// its shared prologue to partition 0, so any single region is a fraction of the call and
/// only the total is the call. The same holds for what it produced — the partitions of one
/// call are that call's output.
///
/// Named for the measurement rather than for the time because it carries both: the output
/// of a call in the middle of a node's chain exists on this side nowhere else.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Measured {
    pub host_setup_us: u64,
    pub host_submit_us: u64,
    pub device_us: u64,
    pub out_rows: u64,
    /// Priced by C++'s own reconstruction of the byte rule — the only figure that exists
    /// for a chained call, which this side never built a batch from.
    pub out_bytes: u64,
    /// Regions that answered for this call. Zero means the device recorded none — a call
    /// the measurement did not see, which is not the same as a call that cost nothing.
    pub regions: usize,
}

impl Measured {
    /// The host side of the region: the prologue plus the submission.
    ///
    /// There is deliberately no term that adds `device_us` to this. Under events the two
    /// OVERLAP — the host submits while the device runs, and `cudf_host_us` is submission
    /// and not execution — so their sum is the duration of nothing. Which of the three a
    /// record reports is the record format's decision, not this module's.
    pub fn host_us(&self) -> u64 {
        self.host_setup_us + self.host_submit_us
    }
}

/// Every call of a run, measured — at both granularities a reader needs.
///
/// The device answers per `(seq, call_index)`, so **the split between the seqs of one
/// driver call is measured**, not guessed. Keeping only a per-entry total would throw that
/// away and then attribute the total to each seq, which reports a merge that produced one
/// row as having produced six.
///
/// The two consumers want different units and both are honest:
///
/// | | unit | why |
/// |---|---|---|
/// | `.benchmark.txt` | a driver call | its axis is lanes × batches, and a batch is a driver call |
/// | `records.tsv` | one cuDF call | its row is one call, and the device measured each |
pub struct Measurements {
    /// Node → driving lane → the calls that lane made, each the sum over the seqs it
    /// addressed. `None` for a call the run did not measure, so an unmeasured backend
    /// reads as absent rather than as free.
    per_entry: Vec<Vec<Vec<Option<Measured>>>>,
    per_call: HashMap<(Seq, u64), Measured>,
}

impl Measurements {
    /// What one driver call cost, summed over the seqs it addressed — the unit a batch is
    /// rendered in.
    pub fn entry(&self, node: usize, lane: usize, position: usize) -> Option<Measured> {
        self.per_entry[node][lane][position]
    }

    /// Lanes of a node, each with one entry per call it made. For walking the shape without
    /// knowing its lengths.
    pub fn lanes(&self, node: usize) -> &[Vec<Option<Measured>>] {
        &self.per_entry[node]
    }

    pub fn nodes(&self) -> usize {
        self.per_entry.len()
    }

    /// What one cuDF call cost, as the device reported it — the unit a record row is in.
    pub fn call(&self, seq: Seq, call_index: u64) -> Option<Measured> {
        self.per_call.get(&(seq, call_index)).copied()
    }
}

/// Cost every recorded call from the regions the device answered with.
///
/// Regions the join did not claim are returned beside the measurements rather than
/// dropped: a region nobody asked for means the two sides disagree about what ran, and
/// silently discarding it would hide exactly that.
pub fn join_regions(report: &RunReport, regions: &[Region]) -> (Measurements, Vec<Region>) {
    // One `Measured` per cuDF call, summed over that call's output partitions. Summed
    // because a call answering with several partitions charges its shared prologue to
    // partition 0, so any single region is a fraction of the call.
    let mut per_call: HashMap<(Seq, u64), Measured> = HashMap::new();
    for region in regions {
        let call = per_call.entry((region.seq, region.call_index)).or_default();
        call.host_setup_us += region.host_setup_us;
        call.host_submit_us += region.host_submit_us;
        call.device_us += region.device_us;
        call.out_rows += region.out_rows;
        call.out_bytes += region.logical_bytes;
        call.regions += 1;
    }
    let mut claimed: HashMap<(Seq, u64), ()> = HashMap::new();
    let per_entry = report
        .abi_calls
        .iter()
        .map(|lanes| {
            lanes
                .iter()
                .map(|calls| {
                    calls
                        .iter()
                        .map(|made| {
                            let made = made.recorded()?;
                            let mut total = Measured::default();
                            for call in made {
                                let Some(found) = per_call.get(&(call.seq, call.call_index))
                                else {
                                    continue;
                                };
                                claimed.insert((call.seq, call.call_index), ());
                                total.host_setup_us += found.host_setup_us;
                                total.host_submit_us += found.host_submit_us;
                                total.device_us += found.device_us;
                                total.out_rows += found.out_rows;
                                total.out_bytes += found.out_bytes;
                                total.regions += found.regions;
                            }
                            Some(total)
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let unclaimed = match claimed.len() == per_call.len() {
        true => Vec::new(),
        false => regions
            .iter()
            .filter(|region| !claimed.contains_key(&(region.seq, region.call_index)))
            .copied()
            .collect(),
    };
    (Measurements { per_entry, per_call }, unclaimed)
}

/// One node's whole cost, summed over every lane and every call it made. `None` where the
/// node was not measured at all, which is not the same as a node that cost nothing.
///
/// Three terms and no total, for the reason argued on [`Measured::host_us`].
pub fn node_measured(times: &Measurements, node: usize) -> Option<Measured> {
    let mut total: Option<Measured> = None;
    for lane in times.lanes(node) {
        for call in lane.iter().flatten() {
            let sum = total.get_or_insert(Measured::default());
            sum.host_setup_us += call.host_setup_us;
            sum.host_submit_us += call.host_submit_us;
            sum.device_us += call.device_us;
            sum.out_rows += call.out_rows;
            sum.out_bytes += call.out_bytes;
            sum.regions += call.regions;
        }
    }
    total
}
