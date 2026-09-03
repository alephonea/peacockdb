//! A device session over one plan, shared by the tiers that open one.

use peacockdb_core::batch_partitioned::GpuNode;
use peacockdb_core::batch_partitioned::gpu_backend::backend::GpuContext;
use peacockdb_core::batch_partitioned::driver::Region;
use peacockdb_core::batch_partitioned::gpu_backend::collect_regions;
use peacockdb_core::batch_partitioned::recipe::{RecipePlan, attach_recipes};
use peacockdb_ffi::raw::{
    PeacockExecutor, peacock_executor_begin_plan, peacock_executor_create,
    peacock_executor_destroy, peacock_executor_end_plan, peacock_last_error,
};

use super::bp_mode::BUDGET;

/// The recipes attached and the buffer handed across, which is the whole of what a
/// `GpuContext` needs. Owns the executor and ends the plan on drop, so a failing case
/// releases the device rather than leaving it holding a plan.
pub struct Session {
    executor: *mut PeacockExecutor,
    recipes: Option<RecipePlan>,
}

impl Session {
    pub fn open(tree: &dyn GpuNode, what: &str) -> Self {
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

    /// What the device recorded, drained while the session is still open: `end_plan`
    /// destroys the events, so a caller reading after the drop reads nothing.
    ///
    /// `cap` bounds what the run can have produced — C++ refuses rather than truncating,
    /// and by then the drain has happened.
    pub fn regions(&self, cap: usize, what: &str) -> Vec<Region> {
        collect_regions(self.executor, cap).unwrap_or_else(|e| panic!("{what}: {e}"))
    }

    pub fn context(&mut self) -> GpuContext {
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

pub fn error_of(executor: *mut PeacockExecutor) -> String {
    let message = unsafe { peacock_last_error(executor) };
    match message.is_null() {
        true => "(no message)".to_string(),
        false => unsafe { std::ffi::CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned(),
    }
}
