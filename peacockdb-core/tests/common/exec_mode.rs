//! Execution mode as an explicit test parameter, and the golden-label spelling
//! derived from it.
//!
//! Mode is stated by the MACRO NAME at every call site (`cpu_full_table_result_test!`,
//! `gpu_partitioned_test!`) and passed down as a parameter. It is never recovered from
//! a device label: a label picking an executor made the executor a side effect of how
//! a golden file was named — see the "routing on a label" antipattern in
//! `llm-wiki/coding-style.md`. The one surviving label→mode decode is
//! `common::plan_partition_mode`, and it is the plan tier only.

use peacockdb_core::PartitionMode;

/// Which executor drives a CPU/GPU run, and the [`PartitionMode`] its context is
/// built with. The enum — NOT the memory budget — is the sole discriminator, so a
/// memory-constrained genuine-8-way device (#91) is a call-site change, not a
/// budget change.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExecMode {
    /// The #11 instrumented-enforced executor: streams each node
    /// single-partition-coalesced regardless of `target_partitions`.
    FullTable,
    /// The `CpuNodeExecutor` / GPU node-by-node path: maintains N partitions across
    /// nodes (partial-agg = Σ-over-partitions, CoalescePartitions concat N→1),
    /// matching the real 8-way GPU.
    Partitioned,
}

impl ExecMode {
    pub fn partition_mode(self) -> PartitionMode {
        match self {
            ExecMode::FullTable => PartitionMode::SinglePartition,
            ExecMode::Partitioned => PartitionMode::RealMultiPartition,
        }
    }

    /// The mode's golden-filename component.
    pub fn label(self) -> &'static str {
        match self {
            ExecMode::FullTable => "full_table",
            ExecMode::Partitioned => "partitioned",
        }
    }
}

/// The `<mode>-<tp>-<tier>` component of a `.cpu.txt` / `.cost.txt` / `.result.txt`
/// golden filename (`full_table-tp8-mini`).
///
/// The ONE place the two halves are joined. The CPU macros pass their device
/// straight in; the GPU macros carry the joined form at the call site and split it
/// with [`gpu_label_device`] before rejoining here — so a GPU call site and its CPU
/// counterpart cannot disagree about the filename they read.
pub fn golden_label(mode: ExecMode, device: &str) -> String {
    format!("{}-{}", mode.label(), device)
}

/// Split a GPU call site's combined golden label (`full_table_tp1_standard`, the
/// ident as written in `gpu_*_test!`) into its `tp<N>-<tier>` device, which
/// `device_config` decodes into the run config.
///
/// The label's mode prefix MUST equal the macro's own mode. A crossed pair —
/// `gpu_full_table_test!` with a `partitioned_…` label — would run one mode against
/// the other mode's goldens, so it panics here rather than failing later as an
/// inscrutable cost-tree mismatch.
pub fn gpu_label_device(mode: ExecMode, label: &str) -> String {
    let prefix = mode.label();
    let device = label
        .strip_prefix(prefix)
        .and_then(|rest| rest.strip_prefix('_'))
        .unwrap_or_else(|| {
            panic!(
                "gpu label '{label}' does not start with '{prefix}_' — a \
                 gpu_{prefix}_test! must carry a {prefix}_… label, or the run would \
                 assert against the other mode's goldens"
            )
        });
    device.replace('_', "-")
}

/// How a CPU run's result is compared against the DataFusion oracle.
///
/// BOTH variants run the SAME oracle: plain DataFusion at `target_partitions = 1`
/// (`build_session_state(1)`). Only the float tolerance differs — which is why this is
/// an oracle-comparison mode and not a different kind of test, and why it is an
/// argument rather than a second macro name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuOracle {
    /// Exact sorted-string equality. The default.
    DataFusionExact,
    /// 1e-12 relative tolerance on Float64 columns. ONLY for queries whose sole
    /// divergence from the oracle is float summation reassociation: at tp>1 the
    /// executor sums in a different association order than DataFusion's
    /// single-partition pass, drifting ~1 ULP. The `output_bytes` cost golden stays
    /// EXACT either way — a ULP does not change a float's byte width — so this
    /// loosens the result compare only, never `assert_cpu_cost_canonical`.
    DataFusionApproximate,
}

impl CpuOracle {
    /// The `rel_tol` handed to the result compare. `None` = exact.
    pub fn rel_tol(self) -> Option<f64> {
        match self {
            CpuOracle::DataFusionExact => None,
            CpuOracle::DataFusionApproximate => Some(1e-12),
        }
    }
}

/// Map a CPU macro's oracle keyword to its [`CpuOracle`]. Mirrors
/// [`result_golden_mode`]: unknown keyword panics naming the accepted set.
pub fn cpu_oracle_mode(s: &str) -> CpuOracle {
    match s {
        "data_fusion_exact" => CpuOracle::DataFusionExact,
        "data_fusion_approximate" => CpuOracle::DataFusionApproximate,
        other => panic!(
            "cpu result test: unknown oracle keyword '{other}' \
             (expected data_fusion_exact|data_fusion_approximate)"
        ),
    }
}

/// Whether a CPU run writes the frozen `.result.txt` golden under UPDATE_CANONICAL.
///
/// Legacy-only, and retires with the legacy modes: the batch-partitioned mode declares a
/// query's CPU and GPU coverage in one macro, so whether a golden is consumed is derived
/// there rather than declared (T18 in the task spec).
///
/// INVARIANT: [`Write`] exactly for the (query, golden-label) pairs a golden-asserting GPU
/// test consumes (`golden_exact`/`golden_approx`/`golden_approx_std`) — not
/// `full_table-tp8-mini`, which has no consumer, nor `oracle` queries. Skip-when-write
/// fails loud; write-when-skip is an orphan golden, the silent case this gate prevents.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResultGolden {
    Write,
    Skip,
}

/// Map a CPU macro's trailing keyword to its [`ResultGolden`]. Mirrors
/// `gpu_result_mode`: unknown keyword panics naming the accepted set.
pub fn result_golden_mode(s: &str) -> ResultGolden {
    match s {
        "result_golden" => ResultGolden::Write,
        "no_result_golden" => ResultGolden::Skip,
        other => panic!(
            "cpu result test: unknown result-golden keyword '{other}' \
             (expected result_golden|no_result_golden)"
        ),
    }
}
