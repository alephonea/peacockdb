//! The resident total the enforcer checks, kept incrementally:
//!
//! ```text
//! resident = Σ byte_size of driver-held in-flight batches
//!          + Σ cached resident_bytes() over live executors
//! ```
//!
//! The executor half is cached per slot and refreshed one instance at a time. That is not
//! a saving so much as what lets this type exist at all: summing over live executors means
//! holding a reference to each, and the driver owns them mutably. A refresh takes the
//! figure after the call, never a handle to what produced it.

use crate::batch_partitioned::error::{RunError, When};
use crate::batch_partitioned::executor::{CallStats, Executor};

/// Which executor instance a figure belongs to. Dense `index` for the cache; `node` and
/// `lane` ride along so a diagnostic can name the site with no formatting per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Slot {
    pub index: usize,
    pub node: u32,
    pub lane: u32,
}

/// A crossing before it is a message: the accountant holds no names, and the driver
/// formats one only on the path that ends the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Trip {
    pub slot: Slot,
    pub when: When,
    pub bytes: usize,
    pub budget: usize,
}

/// A call whose modelled scratch came in under what it measured. Expected — a join's model
/// rests on a cardinality estimate — so it is recorded with its magnitude rather than
/// asserted away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Underestimate {
    pub node: u32,
    pub lane: u32,
    pub modelled: usize,
    pub measured: usize,
}

impl Underestimate {
    /// How far under: 2.0 means the call used twice what was modelled.
    pub fn ratio(&self) -> f64 {
        if self.modelled == 0 {
            f64::INFINITY
        } else {
            self.measured as f64 / self.modelled as f64
        }
    }
}

/// A batch and the figure it was accounted at. Reading the size once is what keeps the
/// released figure equal to the held one, and what keeps an arrow batch from walking every
/// array twice per hop.
#[derive(Debug)]
pub(crate) struct Held<T> {
    pub batch: T,
    pub bytes: usize,
}

impl<T: crate::batch_partitioned::batch::Batch> Held<T> {
    pub(crate) fn of(batch: T) -> Self {
        let bytes = batch.byte_size();
        Self { batch, bytes }
    }

    pub(crate) fn rows(&self) -> u64 {
        self.batch.num_rows() as u64
    }
}

pub(crate) struct ResidentAccountant {
    /// `None` accounts and reports without ever tripping.
    budget: Option<usize>,
    in_flight_bytes: usize,
    executor_bytes: usize,
    cached: Vec<usize>,
    peak: usize,
    calls: usize,
    holds: usize,
    releases: usize,
    underestimates: Vec<Underestimate>,
}

impl ResidentAccountant {
    pub(crate) fn new(slots: usize, budget: Option<usize>) -> Self {
        Self {
            budget,
            in_flight_bytes: 0,
            executor_bytes: 0,
            cached: vec![0; slots],
            peak: 0,
            calls: 0,
            holds: 0,
            releases: 0,
            underestimates: Vec::new(),
        }
    }

    pub(crate) fn resident(&self) -> usize {
        self.in_flight_bytes + self.executor_bytes
    }

    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight_bytes
    }

    pub(crate) fn peak(&self) -> usize {
        self.peak
    }

    pub(crate) fn calls(&self) -> usize {
        self.calls
    }

    /// Held and released, counted rather than netted: a total back at zero is also what a
    /// release of something never held would leave behind.
    pub(crate) fn hops(&self) -> (usize, usize) {
        (self.holds, self.releases)
    }

    pub(crate) fn underestimates(&self) -> &[Underestimate] {
        &self.underestimates
    }

    /// A batch enters a driver-held queue. Bytes rather than the batch, so the figure is
    /// read once and the same one is released: an arrow batch recomputes its size by
    /// walking every array, and two walks are two chances to disagree.
    pub(crate) fn hold(&mut self, bytes: usize) {
        self.holds += 1;
        self.in_flight_bytes += bytes;
        self.observe();
    }

    /// A batch leaves a driver-held queue: consumed, handed to the caller, or dropped.
    pub(crate) fn release(&mut self, bytes: usize) -> Result<(), RunError> {
        self.releases += 1;
        self.in_flight_bytes = self.in_flight_bytes.checked_sub(bytes).ok_or_else(|| {
            RunError::Protocol(format!(
                "released {bytes} bytes against {} in flight: a batch was released without \
                 having been held",
                self.in_flight_bytes
            ))
        })?;
        Ok(())
    }

    /// Pre-check. Returns the modelled scratch so the matching `end_*` can compare it.
    pub(crate) fn begin_call<E: Executor + ?Sized>(
        &mut self,
        slot: Slot,
        executor: &E,
        n_rows: u64,
        n_bytes: usize,
    ) -> Result<usize, Trip> {
        let modelled = executor.scratch_bytes(n_rows, n_bytes);
        self.check(self.resident() + modelled, slot, When::PreCall)?;
        self.calls += 1;
        Ok(modelled)
    }

    /// Post-check for a call whose executor survived it — including one that changed type,
    /// as a join does at `set_build`: the successor reports for the same slot.
    pub(crate) fn end_call<E: Executor + ?Sized>(
        &mut self,
        slot: Slot,
        executor: &E,
        stats: CallStats,
        modelled: usize,
    ) -> Result<(), Trip> {
        self.end(slot, Some(executor.resident_bytes()), stats, modelled)
    }

    /// Post-check for a call that consumed its executor — `mark_done_and_fetch`,
    /// `finish_and_fetch`, an exhausted source. Its state is gone, so it stops
    /// contributing and there is nothing left to ask for a figure.
    pub(crate) fn end_consuming_call(
        &mut self,
        slot: Slot,
        stats: CallStats,
        modelled: usize,
    ) -> Result<(), Trip> {
        self.end(slot, None, stats, modelled)
    }

    /// A finished executor stops contributing. Idempotent, so a step that ends a lane
    /// without calling anything can say so too.
    pub(crate) fn forget(&mut self, slot: Slot) {
        self.executor_bytes -= self.cached[slot.index];
        self.cached[slot.index] = 0;
    }

    fn end(
        &mut self,
        slot: Slot,
        residency: Option<usize>,
        stats: CallStats,
        modelled: usize,
    ) -> Result<(), Trip> {
        match residency {
            Some(current) => self.refresh(slot, current),
            None => self.forget(slot),
        }
        if let Some(measured) = stats.scratch_bytes.filter(|measured| modelled < *measured) {
            self.underestimates.push(Underestimate {
                node: slot.node,
                lane: slot.lane,
                modelled,
                measured,
            });
        }
        self.observe();
        self.check(self.resident(), slot, When::PostCall)
    }

    fn refresh(&mut self, slot: Slot, current: usize) {
        self.executor_bytes = self.executor_bytes + current - self.cached[slot.index];
        self.cached[slot.index] = current;
    }

    fn observe(&mut self) {
        self.peak = self.peak.max(self.resident());
    }

    fn check(&self, value: usize, slot: Slot, when: When) -> Result<(), Trip> {
        match self.budget {
            Some(budget) if value > budget => Err(Trip {
                slot,
                when,
                bytes: value,
                budget,
            }),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests;
