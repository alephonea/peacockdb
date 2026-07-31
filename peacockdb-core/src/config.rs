//! Run-configuration tiers shared by the executors, the test harness, and the
//! golden-file device labels.
//!
//! A device label is `tp<N>-<memtier>` (e.g. `tp8-standard`), and both halves are
//! authoritative: the golden path AND the session config derive from the same
//! label, so a mislabeled test runs the config it claims instead of silently
//! diverging from its name.

/// Realized partition count for [`TargetPartitions::Multi`].
pub const TARGET_PARTITIONS: usize = 8;

/// Deliberately-tiny budget used to force many small batches in the batch-sizing
/// stress tests. NOT a [`MemoryLimit`] tier — no device is ever configured this
/// low; it only exists to push the batch-size math to its floor.
pub const BATCH_STRESS_BUDGET: usize = 10 * 1024;

/// `target_partitions` requested from DataFusion. The *realized* partition count
/// is plan-derived and only ever ≤ this hint (a scan over fewer files, or a
/// coalescing parent, lands below it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetPartitions {
    Single,
    Multi,
}

impl TargetPartitions {
    pub const fn hint(self) -> usize {
        match self {
            TargetPartitions::Single => 1,
            TargetPartitions::Multi => TARGET_PARTITIONS,
        }
    }

    /// Label component of a device string (`tp1` / `tp8`).
    pub const fn label(self) -> &'static str {
        match self {
            TargetPartitions::Single => "tp1",
            TargetPartitions::Multi => "tp8",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "tp1" => Some(TargetPartitions::Single),
            "tp8" => Some(TargetPartitions::Multi),
            _ => None,
        }
    }
}

/// Resident-memory budget tier handed to `GpuMemoryBudgetRule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLimit {
    /// 100 MiB — sits in the gap between the corpus' top query (tpcds q78 ≈ 135.5
    /// MB) and the next one down, so the OOM tests get a real boundary to cross.
    Micro,
    /// 2 GiB.
    Mini,
    /// 12 GiB.
    Standard,
    /// 70 GiB.
    Full,
}

impl MemoryLimit {
    pub const fn bytes(self) -> usize {
        match self {
            MemoryLimit::Micro => 100 * 1024 * 1024,
            MemoryLimit::Mini => 2 * 1024 * 1024 * 1024,
            MemoryLimit::Standard => 12 * 1024 * 1024 * 1024,
            MemoryLimit::Full => 70 * 1024 * 1024 * 1024,
        }
    }

    /// Label component of a device string.
    pub const fn label(self) -> &'static str {
        match self {
            MemoryLimit::Micro => "micro",
            MemoryLimit::Mini => "mini",
            MemoryLimit::Standard => "standard",
            MemoryLimit::Full => "full",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "micro" => Some(MemoryLimit::Micro),
            "mini" => Some(MemoryLimit::Mini),
            "standard" => Some(MemoryLimit::Standard),
            "full" => Some(MemoryLimit::Full),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_round_trip() {
        for t in [TargetPartitions::Single, TargetPartitions::Multi] {
            assert_eq!(TargetPartitions::from_label(t.label()), Some(t));
        }
        for m in [
            MemoryLimit::Micro,
            MemoryLimit::Mini,
            MemoryLimit::Standard,
            MemoryLimit::Full,
        ] {
            assert_eq!(MemoryLimit::from_label(m.label()), Some(m));
        }
        assert_eq!(TargetPartitions::from_label("tp4"), None);
        // The retired pre-tier label spelling must NOT resolve.
        assert_eq!(MemoryLimit::from_label("mem2gib"), None);
    }

    #[test]
    fn tiers_are_strictly_increasing() {
        assert!(MemoryLimit::Micro.bytes() < MemoryLimit::Mini.bytes());
        assert!(MemoryLimit::Mini.bytes() < MemoryLimit::Standard.bytes());
        assert!(MemoryLimit::Standard.bytes() < MemoryLimit::Full.bytes());
        assert!(BATCH_STRESS_BUDGET < MemoryLimit::Micro.bytes());
    }
}
