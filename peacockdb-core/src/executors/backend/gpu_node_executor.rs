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
    peacock_executor_begin_plan, peacock_executor_end_plan, peacock_executor_execute_node,
    peacock_handle_release, peacock_install_rmm_pool, peacock_last_error, peacock_result_free,
    peacock_result_from_handle, peacock_measure_timing_floor_us, peacock_set_node_timing,
    PeacockExecutor, PeacockNodeStats, PeacockRmmPoolInfo, PEACOCK_RMM_POOL_INSTALLED,
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
    /// A pooled resource is installed. Sizes are what it was actually built with.
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
/// Idempotent, and the idempotency lives in C++ (`peacock::install_rmm_pool`) rather
/// than behind a `OnceLock` here, so the process has exactly one guard no matter which
/// side calls first — the gtest binaries call it from `main()`, this path calls it per
/// case. A second call rebuilding the pool would drop a resource that live allocations
/// still point into; the benchmark target, 127 `#[test]` functions sharing one process,
/// is precisely the shape that would find that.
///
/// Must run before any GPU work. Cheap to call again afterwards — it returns the first
/// call's outcome — which is why the caller can just ask for the label at write time.
///
/// The engine does not install this for itself: a shipping query still allocates the
/// expensive way, and `gpu_memory_limit` is still accepted and ignored. Changing that is
/// `llm-wiki/tickets.md` #148, and it is a decision about the product rather than about
/// measurement, which is why this entry point exists in the meantime.
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

/// Turn per-node GPU timing on or off (process-global, OFF by default).
///
/// With it on, every unit of work inside the C++ `NodeSession` is bracketed by a
/// `cudaStreamSynchronize`, and [`NodeMemoryStats::time_us`] /
/// [`PartitionStat::time_us`] carry real microseconds instead of zeros. Why the sync
/// is both what makes the number real and what makes it costly — hence opt-in — is
/// argued once, on `set_node_timing` in `cpp/src/plan_executor.h`.
///
/// Process-global, and the GPU suite already runs `--test-threads=1` (cuDF/RMM share
/// one process-wide pool), so there is no cross-test interleaving to guard against.
pub fn set_node_timing(enabled: bool) {
    unsafe { peacock_set_node_timing(if enabled { 1 } else { 0 }) };
}

/// Microseconds the MEASUREMENT costs: [`set_node_timing`]'s timed region wrapped
/// around no work at all.
///
/// This is the resolution floor of every `time_us` in this module. A node's number
/// is its real work PLUS one of these, so a node reporting at or below the floor is
/// not cheap — it is unresolvable, and the two look identical unless the floor is
/// printed next to them. That is the whole reason this exists; `bench_stats_str`
/// writes it into each record as `sync_floor_us`.
///
/// Do NOT subtract it from node times. Individual node measurements vary by more
/// than the floor itself, so subtracting manufactures zeros and negative-clamped
/// noise — it would hide precisely what reporting the floor is meant to expose.
///
/// Requires a live CUDA context (construct a `GpuExecutor` first) and an idle
/// default stream; returns 0 if CUDA errored, which is a self-announcing value
/// since the instrumentation is never actually free.
pub fn measure_timing_floor_us(samples: u32) -> u64 {
    unsafe { peacock_measure_timing_floor_us(samples) }
}

/// GPU backend: intermediates stay GPU-resident behind handles in the C++
/// `NodeSession`; the executor pointer is BORROWED (owned by `GpuExecutor`).
/// On drop, `peacock_executor_end_plan` frees the session + all remaining
/// resident handles — the VRAM-safety net for mid-walk errors.
pub struct GpuNodeExecutor {
    executor: *mut PeacockExecutor,
}

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
        Ok(Self { executor })
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
        // Output partition count is bounded by target_partitions; a fixed
        // caller buffer avoids an FFI allocation/free for the handle array.
        const OUT_CAP: usize = 64;
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
        // prologue lands on partition 0) — so this is the node's total either way.
        // Zero unless node timing is on; see `peacock_set_node_timing`.
        let mut time_us = 0u64;
        let mut part_stats: Vec<PartitionStat> = Vec::with_capacity(n);
        for (k, st) in out_stats[..n].iter().enumerate() {
            let rp = st.rows as usize;
            let bp = logical_size_from_schema(&schema, rp, st.varlen_content_bytes as usize);
            rows += rp;
            output_bytes += bp;
            max_batch_rows = max_batch_rows.max(rp);
            time_us += st.time_us;
            part_stats.push(PartitionStat {
                out_rows: rp,
                out_bytes: bp,
                row_groups: scan_map.get(k).map(|e| e.row_groups.clone()).unwrap_or_default(),
                time_us: st.time_us,
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
            time_us,
        };
        Ok((out_handles, stat))
    }

    async fn materialize(&mut self, handles: &[u64]) -> DfResult<Vec<RecordBatch>> {
        let mut out = Vec::new();
        for &handle in handles {
            let mut out_ptr: *mut u8 = std::ptr::null_mut();
            let mut out_len: u64 = 0;
            let rc = unsafe {
                peacock_result_from_handle(self.executor, handle, &mut out_ptr, &mut out_len)
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
}
impl Drop for GpuNodeExecutor {
    fn drop(&mut self) {
        unsafe { peacock_executor_end_plan(self.executor) };
    }
}
