//! GPU backend for the node-by-node driver: the C++/cuDF FFI.
//!
//! Handles are GPU-resident tables; per-node stats are reconstructed in Rust from
//! the FFI's `{rows, var-len content}` via the single-source
//! [`crate::cpu_executor::logical_size_from_schema`], so CPU and GPU costs are
//! identical by construction whenever per-node row counts match.
//!
//! SAFETY: the raw `*mut PeacockExecutor` here is BORROWED — it is owned by the
//! [`crate::executors::all_at_once_gpu_executor::GpuExecutor`] that handed it over,
//! which must outlive this backend. That contract is documented, not enforced by
//! lifetimes; keep the two in view of each other when changing either.


use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result as DfResult;
use datafusion::physical_plan::ExecutionPlan;

use crate::executors::executor::{NodeMemoryStats, PartitionStat};
use crate::executors::node_by_node::NodeExecutor;

use arrow::ipc::reader::StreamReader;
use datafusion::error::DataFusionError;

use peacockdb_ffi::raw::{
    peacock_executor_begin_plan, peacock_executor_collect_node_times, peacock_executor_end_plan,
    peacock_executor_execute_node, peacock_handle_release, peacock_install_rmm_pool,
    peacock_last_error, peacock_result_free, peacock_result_from_handle,
    peacock_measure_timing_floor_us, peacock_set_node_timing, peacock_set_nvtx_ranges,
    PeacockExecutor,
    PeacockNodeDeviceTime, PeacockNodeStats, PeacockRmmPoolInfo, PEACOCK_NODE_TIMING_EVENTS,
    PEACOCK_NODE_TIMING_OFF, PEACOCK_NODE_TIMING_SYNC, PEACOCK_RMM_POOL_INSTALLED,
};

use crate::cpu_executor::logical_size_from_schema;

/// Which device allocator a measurement was taken under — the outcome of
/// [`install_rmm_pool`], not the request.
///
/// cuDF routes every intermediate through rmm's current device resource, and the
/// difference between a pool and rmm's default (a `cudaMalloc`/`cudaFree` per
/// allocation) is far larger than run-to-run noise — worst on exactly the nodes with
/// the largest outputs. So `Unavailable` does not describe a slower run to be recorded
/// and compared; it describes a run whose times mean nothing, and the benchmark harness
/// refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmmPool {
    /// Sizes are what the pool was actually built with, not what was requested.
    Pool { integrated: bool, free_bytes: u64, initial_bytes: u64, maximum_bytes: u64 },
    /// The pool could not be built — typically a neighbour holding the device when the
    /// reservation was computed — so rmm's default resource is in place and nobody
    /// chose that.
    Unavailable,
}

impl std::fmt::Display for RmmPool {
    /// The `allocator=` line of a benchmark record. One line, no spaces around `=`,
    /// sizes in GiB because that is the unit the sizing rule is written in.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const GIB: f64 = 1073741824.0;
        match *self {
            RmmPool::Pool { integrated, free_bytes, initial_bytes, maximum_bytes } => write!(
                f,
                "rmm-pool initial={:.1}GiB max={:.1}GiB of {:.1}GiB free on {} device",
                initial_bytes as f64 / GIB,
                maximum_bytes as f64 / GIB,
                free_bytes as f64 / GIB,
                if integrated { "an integrated" } else { "a discrete" },
            ),
            RmmPool::Unavailable => {
                write!(f, "rmm-default (pool unavailable), cudaMalloc per allocation")
            }
        }
    }
}

/// Install the pooled device allocator and report what happened.
///
/// Idempotent, in C++ (`peacock::install_rmm_pool`) rather than behind a `OnceLock` here,
/// so the process has one guard whichever side calls first — the gtest binaries from
/// `main()`, this path per case. A second call rebuilding the pool would drop a resource
/// live allocations still point into.
///
/// Must run before any GPU work; cheap afterwards, since it returns the first call's
/// outcome, so the caller can ask for the label at write time.
///
/// The engine does not install this for itself — a shipping query still allocates the
/// expensive way and `gpu_memory_limit` is accepted and ignored. That is #148, a product
/// decision rather than a measurement one, which is why this entry point exists.
pub fn install_rmm_pool() -> RmmPool {
    let mut info = PeacockRmmPoolInfo::default();
    // Non-zero only for a null pointer, which cannot happen here.
    let _ = unsafe { peacock_install_rmm_pool(&mut info) };
    match info.state {
        PEACOCK_RMM_POOL_INSTALLED => RmmPool::Pool {
            integrated: info.integrated != 0,
            free_bytes: info.free_bytes,
            initial_bytes: info.initial_bytes,
            maximum_bytes: info.maximum_bytes,
        },
        // Includes any state a newer C++ side might add: an unrecognised outcome is not
        // an installed pool, and treating it as one is the mistake that matters.
        _ => RmmPool::Unavailable,
    }
}

/// How per-node GPU regions are measured. `Off` by default, because neither mode
/// is free and one of them changes how the engine SCHEDULES.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum NodeTiming {
    /// No measurement. Every timing field stays 0.
    #[default]
    Off,
    /// Host clock per region, closed by a `cudaStreamSynchronize` into
    /// [`PartitionStat::host_submit_us`]. The sync serializes what cuDF would pipeline,
    /// so measuring changes what is measured. Kept as the baseline for `Events`.
    Sync,
    /// CUDA events around the device work, host clock around the host work, no sync
    /// inside the region. Device numbers arrive via `collect_device_times` after the
    /// root materialize, into [`PartitionStat::device_us`].
    Events,
}

/// Select the per-node GPU timing mode (process-global, [`NodeTiming::Off`] by default).
///
/// Why it is opt-in in either mode, and why the split into host setup / host submit /
/// device exists, is argued once on `set_node_timing` and `mark_device_start` in
/// `cpp/src/plan_executor.h`. The GPU suite runs `--test-threads=1` (cuDF/RMM share one
/// process-wide pool), so the global needs no cross-test guard.
pub fn set_node_timing(mode: NodeTiming) {
    let raw = match mode {
        NodeTiming::Off => PEACOCK_NODE_TIMING_OFF,
        NodeTiming::Sync => PEACOCK_NODE_TIMING_SYNC,
        NodeTiming::Events => PEACOCK_NODE_TIMING_EVENTS,
    };
    unsafe { peacock_set_node_timing(raw) };
}

/// Emit NVTX ranges around plan nodes and their output partitions (process-global, off
/// by default).
///
/// Why this is not folded into [`set_node_timing`] is argued on `set_nvtx_ranges` in
/// `cpp/src/plan_executor.h`. Nothing reads the ranges unless a profiler is attached,
/// so this is for capture runs, not for the benchmark tree.
pub fn set_nvtx_ranges(on: bool) {
    unsafe { peacock_set_nvtx_ranges(i32::from(on)) };
}

/// Microseconds the measurement costs under [`NodeTiming::Sync`]: that mode's timed
/// region around no work at all.
///
/// The resolution floor of every sync-mode time here. A node's number is its real work
/// plus one of these, so a node at or below the floor is not cheap but unresolvable, and
/// the two are indistinguishable unless the floor is printed beside them —
/// `bench_stats_str` writes it into each record as `sync_floor_us`.
///
/// Do not subtract it: node measurements vary by more than the floor itself, so
/// subtracting manufactures zeros and clamped noise, hiding what reporting it exposes.
///
/// Requires a live CUDA context (construct a `GpuExecutor` first) and an idle default
/// stream. Returns 0 on CUDA error — self-announcing, since it is never actually free.
pub fn measure_timing_floor_us(samples: u32) -> u64 {
    unsafe { peacock_measure_timing_floor_us(samples) }
}

/// GPU backend: intermediates stay GPU-resident behind handles in the C++
/// `NodeSession`; the executor pointer is BORROWED (owned by `GpuExecutor`).
/// On drop, `peacock_executor_end_plan` frees the session + all remaining
/// resident handles — the VRAM-safety net for mid-walk errors.
pub struct GpuNodeExecutor {
    executor: *mut PeacockExecutor,
    /// Post-order node count from `begin_plan`, kept only to size the device-time
    /// collection buffer.
    node_count: usize,
}

/// Output partition count is bounded by `target_partitions`; a fixed caller buffer
/// avoids an FFI allocation/free per node for the handle array.
const OUT_CAP: usize = 64;

impl GpuNodeExecutor {
    /// Load the serialized plan into the C++ session (indexes post-order).
    pub fn new(executor: *mut PeacockExecutor, plan_bytes: &[u8]) -> DfResult<Self> {
        let mut node_count: u64 = 0;
        let rc = unsafe {
            peacock_executor_begin_plan(
                executor,
                plan_bytes.as_ptr(),
                plan_bytes.len() as u64,
                &mut node_count,
            )
        };
        if rc != 0 {
            return Err(last_error(executor, "peacock_executor_begin_plan"));
        }
        Ok(Self { executor, node_count: node_count as usize })
    }
}

fn last_error(executor: *mut PeacockExecutor, ctx: &str) -> DataFusionError {
    let msg = unsafe {
        std::ffi::CStr::from_ptr(peacock_last_error(executor))
            .to_string_lossy()
            .into_owned()
    };
    DataFusionError::External(format!("{ctx} failed: {msg}").into())
}

impl NodeExecutor for GpuNodeExecutor {
    async fn execute_node(
        &mut self,
        seq: usize,
        node: &Arc<dyn ExecutionPlan>,
        input_handles: &[Vec<u64>],
    ) -> DfResult<(Vec<u64>, NodeMemoryStats)> {
        // Flatten the per-child partition handles + per-child counts.
        let counts: Vec<u64> = input_handles.iter().map(|c| c.len() as u64).collect();
        let flat: Vec<u64> = input_handles.iter().flatten().copied().collect();
        let mut out_buf = [0u64; OUT_CAP];
        let mut out_count: u64 = 0;
        // Per-partition stats (parallel to out_handles); see the FFI contract.
        let mut out_stats = [PeacockNodeStats::default(); OUT_CAP];
        let rc = unsafe {
            peacock_executor_execute_node(
                self.executor,
                seq as u64,
                flat.as_ptr(),
                counts.as_ptr(),
                counts.len() as u64,
                out_buf.as_mut_ptr(),
                OUT_CAP as u64,
                &mut out_count,
                out_stats.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(last_error(self.executor, "peacock_executor_execute_node"));
        }
        let n = out_count as usize;
        let out_handles: Vec<u64> = out_buf[..n].to_vec();
        // Cost = Σ-over-partitions of the PER-PARTITION ColAccum overhead (each
        // partition charged its own bitmap/offset +1 fixed terms) + the var-len
        // content C++ measured — matching the #13 CpuNodeExecutor's Σ-over-
        // partition golden, NOT ColAccum(Σ rows). Rust owns the byte formula
        // (logical_size_from_schema), single-sourced → no CPU/GPU drift.
        let schema = node.schema();
        // Scan's per-partition row groups come from the SAME RG→batch→partition
        // map the C++ side replays via set_row_groups — so the GPU's per-partition
        // sub-lines match the #13 CPU golden by construction (the golden is
        // GPU-VERIFIED, not just CPU-printed).
        let scan_map = node
            .as_any()
            .downcast_ref::<crate::gpu_rule::GpuScanExec>()
            .map(|s| s.batches_map())
            .unwrap_or(&[]);
        let mut rows = 0usize;
        let mut output_bytes = 0usize;
        let mut max_batch_rows = 0usize;
        // Σ over partitions, matching how C++ charges shared work (the hash-scatter
        // prologue lands on partition 0). Zero unless node timing is on. No `device_us`
        // here: it is not known yet, and the driver merges it in after materialize.
        let mut host_setup_us = 0u64;
        let mut host_submit_us = 0u64;
        let mut part_stats: Vec<PartitionStat> = Vec::with_capacity(n);
        for (k, st) in out_stats[..n].iter().enumerate() {
            let rp = st.rows as usize;
            let bp = logical_size_from_schema(&schema, rp, st.varlen_content_bytes as usize);
            // C++ reconstructs the same total from cuDF types (#153). Unused — `bp`
            // is — but it must agree: the bare-cuDF sf40 tests never enter Rust and
            // report THEIR number, so if the two ends count bytes differently, every
            // coefficient fitted across both is wrong by an undetectable factor.
            //
            // Debug-only: `[profile.benchmarks]` inherits release, so the measured path
            // pays nothing while every `cargo test` GPU run checks the whole corpus.
            //
            // Gated on the device having materialized the types the node DECLARES —
            // two implementations of one byte rule can only be compared when both cost
            // the same columns. Legitimate shape divergences: a grouping-set/ROLLUP
            // Partial AVG emits one MEAN where DataFusion declares `[count]`+`[sum]`,
            // `__grouping_id` is built INT32 against a declared UInt8 (#196), and a union
            // branch holds a decimal literal as FLOAT64 until `execute_union` retypes it
            // (#41). None can arise on the bare-cuDF sf40 path this protects, so skipping
            // them costs the calibration nothing. Flag set by `types_match_declared`
            // (execute_plan.cpp); the one such divergence that was a real bug is #195.
            if st.schema_faithful != 0 {
                debug_assert_eq!(
                    st.logical_bytes as usize,
                    bp,
                    "{} partition {k}: C++ logical_bytes={} != Rust logical_size_from_schema={bp} \
                     (rows={rp}, varlen={}); schema={:?}",
                    node.name(),
                    st.logical_bytes,
                    st.varlen_content_bytes,
                    schema,
                );
            }
            rows += rp;
            output_bytes += bp;
            max_batch_rows = max_batch_rows.max(rp);
            host_setup_us += st.host_setup_us;
            host_submit_us += st.host_submit_us;
            part_stats.push(PartitionStat {
                out_rows: rp,
                out_bytes: bp,
                row_groups: scan_map.get(k).map(|e| e.row_groups.clone()).unwrap_or_default(),
                host_setup_us: st.host_setup_us,
                host_submit_us: st.host_submit_us,
                device_us: 0,
            });
        }
        let stat = NodeMemoryStats {
            node_name: node.name().to_string(),
            allocated_bytes: 0, // not modeled on GPU (VRAM layout not compared)
            output_bytes,
            row_count: rows,
            max_batch_rows,
            // Only N>1 carries sub-lines (matches the CPU golden's N==1 ⇒ none).
            part_stats: if n > 1 { part_stats } else { Vec::new() },
            host_setup_us,
            host_submit_us,
            device_us: 0,
        };
        Ok((out_handles, stat))
    }

    async fn materialize(&mut self, handles: &[u64]) -> DfResult<Vec<RecordBatch>> {
        let mut out = Vec::new();
        for &handle in handles {
            let mut out_ptr: *mut u8 = std::ptr::null_mut();
            let mut out_len: u64 = 0;
            // The whole handle: a root materialization has no row interval to apply.
            let rc = unsafe {
                peacock_result_from_handle(
                    self.executor,
                    handle,
                    0,
                    u64::MAX,
                    &mut out_ptr,
                    &mut out_len,
                )
            };
            if rc != 0 {
                return Err(last_error(self.executor, "peacock_result_from_handle"));
            }
            if out_len == 0 || out_ptr.is_null() {
                continue;
            }
            let ipc = unsafe { std::slice::from_raw_parts(out_ptr, out_len as usize) };
            let batches = StreamReader::try_new(std::io::Cursor::new(ipc), None)
                .and_then(|r| r.collect::<Result<Vec<_>, _>>())
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            unsafe { peacock_result_free(out_ptr) };
            out.extend(batches);
        }
        Ok(out)
    }

    fn release(&mut self, handles: &[u64]) {
        for &handle in handles {
            unsafe { peacock_handle_release(self.executor, handle) };
        }
    }

    async fn collect_device_times(&mut self) -> DfResult<Vec<(usize, usize, u64)>> {
        // One entry per (node, output partition) that recorded both events, so
        // nodes × OUT_CAP is the exact upper bound. C++ reports how many it HAD, not how
        // many fit, and fails if that exceeds `cap` — an under-sized buffer surfaces as
        // an error rather than as a device that quietly did less work.
        let cap = self.node_count * OUT_CAP;
        let mut buf = vec![PeacockNodeDeviceTime::default(); cap];
        let mut count: u64 = 0;
        let rc = unsafe {
            peacock_executor_collect_node_times(
                self.executor,
                buf.as_mut_ptr(),
                cap as u64,
                &mut count,
            )
        };
        if rc != 0 {
            return Err(last_error(self.executor, "peacock_executor_collect_node_times"));
        }
        Ok(buf[..count as usize]
            .iter()
            .map(|t| (t.seq as usize, t.partition as usize, t.device_us))
            .collect())
    }
}
impl Drop for GpuNodeExecutor {
    fn drop(&mut self) {
        unsafe { peacock_executor_end_plan(self.executor) };
    }
}
