//! `GpuBatch`'s own surface: what it reports, and which way out of it releases the
//! handle. The release semantics against a live registry are in test_gpu_abi.rs, which
//! needs a GPU; these need none, because `peacock_handle_release` is null-guarded on the
//! executor pointer, so a batch built on a null one drops as a no-op. Not rust-only
//! either way: `GpuBatch` exists only where the FFI is linked.
#![cfg(not(feature = "rust-only"))]

use std::ptr;

use peacockdb_core::batch_partitioned::{Batch, GpuBatch};

fn detached(handle: u64, num_rows: usize, byte_size: usize) -> GpuBatch {
    GpuBatch::new(ptr::null_mut(), handle, num_rows, byte_size)
}

#[test]
fn a_batch_reports_what_it_was_built_with() {
    let batch = detached(7, 100, 4096);
    assert_eq!(batch.handle(), 7);
    assert!(batch.executor().is_null());
    assert_eq!(batch.num_rows(), 100);
    assert_eq!(batch.byte_size(), 4096);
    assert!(format!("{batch:?}").contains("handle: 7"));
}

#[test]
fn dropping_a_detached_batch_releases_against_nothing() {
    // The null guard this whole target rests on: a batch whose executor is null drops
    // without reaching a registry, so the surface can be tested off the GPU. A guard
    // removed on the C++ side crashes here rather than in a GPU tier.
    drop(detached(1, 10, 32));
    drop(detached(1, 10, 32));
}

#[test]
fn consume_yields_the_pair_the_ffi_call_needs() {
    let batch = detached(42, 10, 32);
    let (executor, handle) = batch.consume();
    assert!(executor.is_null());
    assert_eq!(handle, 42);
}
