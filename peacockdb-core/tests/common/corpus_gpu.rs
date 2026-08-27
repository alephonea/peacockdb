//! One corpus query at one mode, on the device.
//!
//! The device side reads what the CPU side wrote and never writes: per-node shape and
//! statistics against that mode's `.cpu.txt` always, and the result where the declaration
//! names a golden. It ignores the regeneration variables rather than honouring them — a
//! device that can author its own golden proves nothing against it.
//!
//! What the tree assertion is worth is worth stating: `batch_partitioned_driver` is generic
//! over the backend, so both engines walk one driver over one plan and produce the same
//! shape by construction. The evidence is the rows and the bytes under it. The shape check
//! stays because it costs nothing and goes red on the day that construction stops holding.

use datafusion::arrow::array::RecordBatch;
use peacockdb_core::batch_partitioned::driver::batch_partitioned_driver;
use peacockdb_core::batch_partitioned::gpu_backend::backend::{GpuBackend, GpuContext};
use peacockdb_core::batch_partitioned::plan_text::render_run;
use peacockdb_core::batch_partitioned::recipe::{RecipePlan, attach_recipes};
use peacockdb_ffi::raw::{
    PeacockExecutor, peacock_executor_begin_plan, peacock_executor_create,
    peacock_executor_destroy, peacock_executor_end_plan, peacock_last_error,
};

use super::bp_mode::{BUDGET, BpMode, mode_named};
use super::corpus::{plan_at, run_cpu};
use super::corpus_golden;
use super::{GpuResultMode, assert_results_match, batches_to_sorted_str, gpu_result_mode};

/// A device session over one plan: the recipes attached and the buffer handed across, which
/// is the whole of what a `GpuContext` needs. Owns the executor and ends the plan on drop,
/// so a failing case releases the device rather than leaving it holding a plan.
struct Session {
    executor: *mut PeacockExecutor,
    recipes: Option<RecipePlan>,
}

impl Session {
    fn open(tree: &dyn peacockdb_core::batch_partitioned::GpuNode, what: &str) -> Self {
        let recipes = attach_recipes(tree).unwrap_or_else(|e| panic!("{what}: no recipes: {e}"));
        let mut executor: *mut PeacockExecutor = std::ptr::null_mut();
        assert_eq!(
            unsafe { peacock_executor_create(BUDGET, &mut executor) },
            0,
            "{what}: peacock_executor_create failed"
        );
        let bytes = recipes.bytes();
        let mut nodes = 0u64;
        let rc = unsafe {
            peacock_executor_begin_plan(executor, bytes.as_ptr(), bytes.len() as u64, &mut nodes)
        };
        assert_eq!(rc, 0, "{what}: begin_plan failed: {}", error_of(executor));
        assert_eq!(nodes as usize, recipes.wire_nodes());
        Self {
            executor,
            recipes: Some(recipes),
        }
    }

    fn context(&mut self) -> GpuContext {
        GpuContext {
            executor: self.executor,
            recipes: self.recipes.take().expect("the recipes are taken once"),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            peacock_executor_end_plan(self.executor);
            peacock_executor_destroy(self.executor);
        }
    }
}

fn error_of(executor: *mut PeacockExecutor) -> String {
    let message = unsafe { peacock_last_error(executor) };
    match message.is_null() {
        true => "(no message)".to_string(),
        false => unsafe { std::ffi::CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned(),
    }
}

/// The whole of a device corpus case: plan, run on the device, then the two read-only
/// assertions — the mode's `.cpu.txt` section, and the result the declaration names.
pub async fn gpu_case(dataset: &str, sf: &str, query: &str, mode: &str, gpu_oracle: &str) {
    let mode = mode_named(mode);
    let what = format!("{dataset}/{query} at {} on a device", mode.name);
    let (_ctx, tree) = plan_at(dataset, sf, query, mode).await;
    let mut session = Session::open(tree.as_ref(), &what);
    let ctx = session.context();
    let report = batch_partitioned_driver::<GpuBackend>(tree.as_ref(), &ctx, None)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(report.in_flight_bytes, 0, "{what} ended holding batches");
    assert_eq!(
        report.holds, report.releases,
        "{what} held {} batches and released {}",
        report.holds, report.releases
    );
    corpus_golden::assert_section(
        &corpus_golden::cpu_golden(dataset, sf, mode.name),
        query,
        &render_run(tree.as_ref(), &report),
    );
    let batches: Vec<RecordBatch> = report
        .batches
        .iter()
        .map(|batch| batch.record_batch().clone())
        .collect();
    assert_result(dataset, sf, query, mode, gpu_oracle, &batches).await;
}

/// The two conditions that decide which authority is available, and the check that the
/// declaration named the right one. Derivable is why a CHECK can exist, never why the value
/// would be absent: a `golden_exact` where the section is a marker is a test that fails on
/// correct behaviour, and a `live_cpu` where a committed section serves spends a device-side
/// cpu run on a comparison a file makes faster and harder.
fn assert_oracle_suits_the_golden(dataset: &str, sf: &str, query: &str, gpu_oracle: &str, what: &str) {
    let section = corpus_golden::section_of(&corpus_golden::result_golden(dataset, sf), query);
    let frozen = !section.starts_with(corpus_golden::SKIPPED);
    match gpu_result_mode(gpu_oracle) {
        GpuResultMode::LiveCpu => assert!(
            !frozen,
            "{what}: gpu_oracle is live_cpu and `.result.txt` holds this query's rows — a \
             device-side cpu run for a comparison the committed section already makes"
        ),
        GpuResultMode::Skip => {}
        _ => assert!(
            frozen,
            "{what}: gpu_oracle names a golden and `.result.txt` has none for this query — \
             the section says `{}`, so this compare fails on correct behaviour",
            section.trim_end()
        ),
    }
}

/// The device's answer against whichever authority the declaration names.
async fn assert_result(
    dataset: &str,
    sf: &str,
    query: &str,
    mode: &BpMode,
    gpu_oracle: &str,
    batches: &[RecordBatch],
) {
    let what = format!("{dataset}/{query} at {} on a device", mode.name);
    assert_oracle_suits_the_golden(dataset, sf, query, gpu_oracle, &what);
    let tolerance = match gpu_result_mode(gpu_oracle) {
        GpuResultMode::Skip => return,
        // A live cpu run at the SAME mode, because where the SQL does not fix the row set,
        // another mode's answer is not an authority on this one's — which is the reason
        // this value exists rather than a frozen section serving all five.
        GpuResultMode::LiveCpu => {
            let run = run_cpu(dataset, sf, query, mode).await;
            assert_results_match(&run.batches, batches, None, &what);
            return;
        }
        GpuResultMode::GoldenExact => None,
        GpuResultMode::GoldenApprox => Some(1e-12),
        GpuResultMode::GoldenApproxStddev => Some(1e-11),
    };
    let golden = corpus_golden::section_of(&corpus_golden::result_golden(dataset, sf), query);
    let (author, rows) = golden
        .split_once('\n')
        .expect("a result section names the mode that wrote it, then its rows");
    let author = author
        .strip_prefix("mode=")
        .expect("a result section opens with `mode=`");
    // The frozen section is one mode's answer serving every mode's run. Where that cannot
    // hold, the declaration says `live_cpu` — so a golden compare here is also the claim
    // that the modes agree, and it is worth naming the author when they do not.
    let actual = batches_to_sorted_str(batches);
    match tolerance {
        None => assert_eq!(
            actual.trim_end(),
            rows.trim_end(),
            "{what}: the device's answer differs from the result golden, written at {author}"
        ),
        Some(tolerance) => {
            super::assert_sorted_str_approx(rows.trim_end(), actual.trim_end(), tolerance, &what)
        }
    }
}
