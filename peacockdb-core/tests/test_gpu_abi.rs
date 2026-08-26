//! Driving the C ABI from Rust on a live GPU: a scan read one row-group subset at a
//! time, a row range on the export, and slicing a resident handle.
//!
//! The C++ suite proves the same kernels against cuDF tables; what only this side can
//! prove is the boundary the driver will actually cross — the exported IPC decoded back
//! into rows, and `GpuBatch`'s release skipped exactly when an FFI call consumed the
//! handle.
#![cfg(not(feature = "rust-only"))]
#[macro_use]
mod common;
use datafusion::arrow::array::{Array, Int64Array, RecordBatch};
use datafusion::arrow::ipc::reader::StreamReader;

use peacockdb_core::batch_partitioned::GpuBatch;
use peacockdb_core::create_context_with_tables;
use peacockdb_core::plan_serializer::serialize_plan;
use peacockdb_ffi::raw::{
    PeacockExecutor, PeacockNodeStats, peacock_executor_begin_plan, peacock_executor_create,
    peacock_executor_destroy, peacock_executor_end_plan, peacock_executor_execute_scan_rowgroups,
    peacock_executor_slice_handle, peacock_result_free, peacock_result_from_handle,
};

use common::{GPU_BUDGET, testdata_minimal_dir};

/// customer projected to c_custkey: the one committed fixture with two row groups
/// (122880 + 27120), and a narrow column to read back.
const SQL: &str = "SELECT c_custkey FROM customer";
const TABLE_ROWS: usize = 150_000;
/// Post-order seq 0 is the deepest-leftmost leaf, so it is the scan in any plan.
const SCAN_SEQ: u64 = 0;

/// An executor with the plan loaded, torn down in the order the header requires.
struct LoadedPlan {
    executor: *mut PeacockExecutor,
}

impl LoadedPlan {
    async fn new() -> Self {
        let ctx = create_context_with_tables(&testdata_minimal_dir(), 1, GPU_BUDGET)
            .await
            .unwrap();
        let plan = ctx
            .sql(SQL)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let bytes = serialize_plan(&plan).unwrap();

        let mut executor: *mut PeacockExecutor = std::ptr::null_mut();
        assert_eq!(
            unsafe { peacock_executor_create(GPU_BUDGET as u64, &mut executor) },
            0
        );
        let mut nodes = 0u64;
        let rc = unsafe {
            peacock_executor_begin_plan(executor, bytes.as_ptr(), bytes.len() as u64, &mut nodes)
        };
        assert_eq!(rc, 0, "begin_plan failed");
        Self { executor }
    }

    /// One batch's worth of the scan: the row groups named, and nothing else.
    fn scan(&self, row_groups: &[u32]) -> (u64, PeacockNodeStats) {
        let mut handle = 0u64;
        let mut stats = PeacockNodeStats::default();
        let rc = unsafe {
            peacock_executor_execute_scan_rowgroups(
                self.executor,
                SCAN_SEQ,
                row_groups.as_ptr(),
                row_groups.len() as u64,
                &mut handle,
                &mut stats,
            )
        };
        assert_eq!(rc, 0, "execute_scan_rowgroups({row_groups:?}) failed");
        (handle, stats)
    }

    fn slice(&self, handle: u64, offset: u64, length: u64) -> u64 {
        let mut out = 0u64;
        let rc = unsafe {
            peacock_executor_slice_handle(self.executor, handle, offset, length, &mut out)
        };
        assert_eq!(rc, 0, "slice_handle failed");
        out
    }

    fn export(&self, handle: u64, offset: u64, length: u64) -> Result<Vec<RecordBatch>, i32> {
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let mut len = 0u64;
        let rc = unsafe {
            peacock_result_from_handle(self.executor, handle, offset, length, &mut ptr, &mut len)
        };
        if rc != 0 {
            return Err(rc);
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        let ipc = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
        let batches = StreamReader::try_new(std::io::Cursor::new(ipc), None)
            .and_then(|r| r.collect::<Result<Vec<_>, _>>())
            .unwrap();
        unsafe { peacock_result_free(ptr) };
        Ok(batches)
    }

    fn keys(&self, handle: u64, offset: u64, length: u64) -> Vec<i64> {
        self.export(handle, offset, length)
            .expect("export failed")
            .iter()
            .flat_map(|b| {
                let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                (0..col.len()).map(|i| col.value(i)).collect::<Vec<_>>()
            })
            .collect()
    }
}

impl Drop for LoadedPlan {
    fn drop(&mut self) {
        unsafe {
            peacock_executor_end_plan(self.executor);
            peacock_executor_destroy(self.executor);
        }
    }
}

#[tokio::test]
async fn scanning_one_row_group_at_a_time_reads_the_whole_table_between_them() {
    let plan = LoadedPlan::new().await;
    let (first, first_stats) = plan.scan(&[0]);
    let (second, second_stats) = plan.scan(&[1]);

    // The same seq twice: the scan arm holds nothing between calls, which is what lets
    // one plan node be a batch loader.
    assert_eq!(
        first_stats.rows + second_stats.rows,
        TABLE_ROWS as u64,
        "the two row groups did not add up to the table"
    );
    let mut keys = plan.keys(first, 0, u64::MAX);
    keys.extend(plan.keys(second, 0, u64::MAX));
    keys.sort_unstable();
    assert_eq!(keys.len(), TABLE_ROWS);
    // Strictly increasing over a unique key: the two reads overlap nowhere, so the
    // count above is coverage rather than one group read twice.
    assert!(
        keys.windows(2).all(|w| w[0] < w[1]),
        "a key came back from both reads"
    );
}

#[tokio::test]
async fn exported_ranges_are_the_rows_they_name() {
    let plan = LoadedPlan::new().await;
    let (handle, stats) = plan.scan(&[1]);
    let rows = stats.rows;

    let whole = plan.keys(handle, 0, u64::MAX);
    assert_eq!(whole.len() as u64, rows);

    let mut in_pieces = plan.keys(handle, 0, 100);
    assert_eq!(in_pieces.len(), 100);
    in_pieces.extend(plan.keys(handle, 100, u64::MAX));
    assert_eq!(in_pieces, whole, "the pieces are not the whole");

    // A fetch overrunning the batch it straddles clamps; past the end there is nothing
    // to ship, which is the case the driver hits on a satisfied limit.
    assert_eq!(
        plan.keys(handle, rows - 10, 1000),
        whole[whole.len() - 10..].to_vec()
    );
    assert!(plan.keys(handle, rows, 10).is_empty());
}

#[tokio::test]
async fn slicing_keeps_the_rows_named_and_consumes_the_handle() {
    let plan = LoadedPlan::new().await;
    let (handle, _) = plan.scan(&[1]);
    let whole = plan.keys(handle, 0, u64::MAX);

    let head = plan.slice(handle, 0, 100);
    assert_eq!(plan.keys(head, 0, u64::MAX), whole[..100].to_vec());
    assert!(
        plan.export(handle, 0, u64::MAX).is_err(),
        "the sliced handle survived the call that consumed it"
    );
}

#[tokio::test]
async fn a_dropped_batch_releases_its_handle_and_a_consumed_one_does_not() {
    let plan = LoadedPlan::new().await;

    let (dropped, stats) = plan.scan(&[1]);
    drop(GpuBatch::new(
        plan.executor,
        dropped,
        stats.rows as usize,
        0,
    ));
    assert!(
        plan.export(dropped, 0, u64::MAX).is_err(),
        "the handle outlived the batch that owned it"
    );

    let (kept, stats) = plan.scan(&[1]);
    let (executor, handle) = GpuBatch::new(plan.executor, kept, stats.rows as usize, 0).consume();
    assert_eq!(handle, kept);
    assert_eq!(executor, plan.executor);
    assert!(
        plan.export(kept, 0, u64::MAX).is_ok(),
        "consume released a handle it was handing to an FFI call"
    );
}
