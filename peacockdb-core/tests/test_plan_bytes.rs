//! Wire-format guard: the EXACT bytes `serialize_plan_mode` emits, per query.
//!
//! Why this exists. Nothing else in the suite pins the serialized layout:
//!   - `.plan.txt` goldens are rendered TEXT. FlatBuffer field write-order changes
//!     none of it, so a completely different binary layout leaves them identical.
//!   - The round-trip oracle asserts `reserialize(deserialize(bytes)) == bytes`
//!     WITHIN ONE BUILD. That proves idempotency, not stability across versions: a
//!     different-but-internally-consistent layout passes it 100%.
//! So a refactor could silently change every byte on the wire and the whole suite
//! would stay green — while the C++ side, which READS these bytes, is a live
//! cross-language consumer. This file closes that gap: the digests are committed, so
//! any layout shift turns the suite red and has to be justified rather than noticed.
//!
//! FlatBufferBuilder is a no-interning bump arena, so byte identity is sensitive to
//! statement ORDER inside each serializer arm, not just to field values. That is
//! precisely the property this guard protects.
//!
//! Regenerate deliberately (a diff here means the wire format moved):
//!   UPDATE_CANONICAL=1 cargo test --features rust-only -p peacockdb-core --test test_plan_bytes

#[macro_use]
mod common;

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use peacockdb_core::plan_serializer::serialize_plan_mode;

/// A FIXED-LENGTH path that stands in for the testdata root when building plans.
///
/// The serialized plan legitimately EMBEDS absolute parquet paths — the C++ side has
/// to open those files, so they cannot be serialized away (operators/scan.rs). That
/// makes the raw bytes depend on where the repo is checked out: this guard would
/// false-red in CI (/home/runner/work/...) and on any dev box off /media/data, and a
/// guard that cries wolf every run is a guard people learn to ignore.
///
/// Byte-substituting the path afterwards does NOT fix it: a FlatBuffer string is
/// [uint32 len][bytes][pad], so a different root also changes the length prefix,
/// every subsequent offset, and (when the delta is not a multiple of 4) the padding.
/// Measured: an 8-path plan moved 192 bytes for a +24-char root. Normalizing that
/// away means rewriting offsets, i.e. parsing the buffer — and the result would be a
/// digest of something that is not a real buffer.
///
/// So instead we hold the path CONSTANT: point the plan build at a symlink whose path
/// is the same on every machine. The bytes are then identical by construction, and the
/// digest still describes a buffer that genuinely exists.
fn canonical_root() -> std::path::PathBuf {
    let link = std::path::PathBuf::from("/tmp/peacock-plan-bytes-root");
    let real = common::testdata_root();
    // Re-point every run: a stale link from another checkout would silently digest
    // the wrong tree.
    let _ = std::fs::remove_file(&link);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&real, &link)
        .unwrap_or_else(|e| panic!("cannot create {} -> {}: {e}", link.display(), real.display()));
    link
}

fn canonical_data_dir(dataset: &str, sf: &str) -> std::path::PathBuf {
    canonical_root().join(format!("{dataset}.sf{sf}"))
}

/// Digest golden: `<dataset>.sf<sf>/<query>.<device>` -> sha256 + byte length.
fn digest_path() -> std::path::PathBuf {
    common::testdata_root().join("goldens/plan_bytes.sha256")
}

/// The corpus is DERIVED from the committed `.plan.txt` goldens rather than listed
/// here, so it cannot drift out of sync with the plan suite: add a plan golden and
/// this guard covers it automatically.
fn corpus() -> Vec<(String, String, String, String)> {
    let mut out = Vec::new();
    let root = common::testdata_root().join("goldens");
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", root.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for dir in dirs {
        // `<dataset>.sf<sf>`
        let dname = dir.file_name().unwrap().to_string_lossy().to_string();
        let Some((dataset, sf)) = dname.split_once(".sf") else { continue };
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|f| f.ends_with(".plan.txt"))
            .collect();
        files.sort();
        for f in files {
            let stem = f.trim_end_matches(".plan.txt");
            // `<query>.<device>` — device is the LAST dot-separated component.
            let Some((query, device)) = stem.rsplit_once('.') else { continue };
            out.push((dataset.to_string(), sf.to_string(), query.to_string(), device.to_string()));
        }
    }
    out
}

/// Fail LOUD and EARLY when the sf1 parquet is absent.
///
/// This test builds real physical plans before serializing them, so it needs the
/// dataset — it is NOT a pure-text check over committed artifacts. Without this
/// guard a parquet-less tier surfaces the problem as a deep `NotFound` inside
/// `create_context_with_tables_mode`, which reads as "the wire-format guard is
/// broken" rather than "this tier has no parquet".
///
/// Deliberately FAIL, never SKIP. A skip exits 0 — green, having verified nothing —
/// which is exactly how a wire-format guard quietly stops guarding (the same lesson
/// as the `PASSED 0 tests` / `ran_any` guards in the test runners).
fn assert_dataset_present() {
    let root = common::testdata_root();
    let probe = root.join("tpch.sf1/lineitem.parquet");
    assert!(
        probe.exists(),
        "test_plan_bytes requires the sf1 parquet (it builds physical plans before \
         serializing them, so it is NOT a committed-artifacts-only check).\n\
         Missing: {}\n\
         Provision the dataset (testdata/generate_testdata.sh --bench tpch) or run this \
         target in the CPU tier that already provisions parquet — NOT the golden-only tier.",
        probe.display()
    );
}

#[tokio::test]
async fn serialized_plan_bytes_are_stable() {
    assert_dataset_present();

    let mut actual: BTreeMap<String, String> = BTreeMap::new();

    for (dataset, sf, query, device) in corpus() {
        let key = format!("{dataset}.sf{sf}/{query}.{device}");
        let (partitions, budget) = common::device_config(&device);
        let data_dir = canonical_data_dir(&dataset, &sf);
        let sql_path = common::queries_dir_for(&dataset).join(format!("{query}.sql"));
        let Ok(sql) = std::fs::read_to_string(&sql_path) else { continue };

        let ctx = peacockdb_core::create_context_with_tables_mode(
            &data_dir,
            partitions,
            budget,
            common::partition_mode(&device),
        )
        .await
        .unwrap();
        let plan = ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();

        // Record UNSUPPORTED rather than skipping: a node becoming (un)serializable is
        // itself a wire-format change, and silently dropping it would hide that.
        let value = match serialize_plan_mode(&plan, common::partition_mode(&device)) {
            Ok(bytes) => {
                let mut h = Sha256::new();
                h.update(&bytes);
                format!("{:x} {}", h.finalize(), bytes.len())
            }
            Err(e) => format!("UNSUPPORTED {e}"),
        };
        actual.insert(key, value);
    }

    // The three INLINE-SQL plans. They have no testdata/*-queries/*.sql file, so the
    // corpus walk above cannot see them — yet they are exactly the plans
    // test_plan_serialiser.rs round-trips, i.e. the serializer's own test plans were
    // the unguarded ones. SQL kept verbatim in sync with test_plan_serialiser.rs.
    for (name, sql) in [
        ("filter_agg", "SELECT count(*) FROM customer WHERE c_acctbal > 0"),
        (
            "join_sort",
            "SELECT n.n_name, r.r_name \
             FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey \
             ORDER BY n.n_name",
        ),
        (
            "group_join_sort",
            "SELECT r.r_name, count(*) AS nation_count \
             FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey \
             GROUP BY r.r_name \
             ORDER BY nation_count DESC, r.r_name",
        ),
    ] {
        let device = "tp8-mini";
        let (partitions, budget) = common::device_config(device);
        let ctx = peacockdb_core::create_context_with_tables_mode(
            &canonical_data_dir("tpch", "1"),
            partitions,
            budget,
            common::partition_mode(device),
        )
        .await
        .unwrap();
        let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
        let value = match serialize_plan_mode(&plan, common::partition_mode(device)) {
            Ok(bytes) => {
                let mut h = Sha256::new();
                h.update(&bytes);
                format!("{:x} {}", h.finalize(), bytes.len())
            }
            Err(e) => format!("UNSUPPORTED {e}"),
        };
        actual.insert(format!("tpch.sf1/{name}.{device} [inline]"), value);
    }

    assert!(!actual.is_empty(), "corpus was empty — no .plan.txt goldens found");

    let rendered: String =
        actual.iter().map(|(k, v)| format!("{k}  {v}\n")).collect::<Vec<_>>().concat();

    let path = digest_path();
    if std::env::var("UPDATE_CANONICAL").is_ok() {
        // Deliberately obstructive. These digests exist to be a FIXED expectation from
        // before a refactor; regenerating them on a red test photographs the new
        // behavior and asserts it against itself, which proves nothing. Requiring a
        // second, explicit variable makes that a decision rather than a reflex.
        if std::env::var("PEACOCK_REWRITE_PLAN_BYTES").is_err() {
            panic!(
                "REFUSING to regenerate {}.\n\
                 A red digest test means the serialized bytes MOVED. The C++ side READS \
                 these bytes, so this is a cross-language wire-format change — find out WHY \
                 before touching this file. Regenerating to make the test pass destroys the \
                 only guard on the layout.\n\
                 If the change really is intended and reviewed, set \
                 PEACOCK_REWRITE_PLAN_BYTES=1 as well.",
                path.display()
            );
        }
        let header = concat!(
            "# serialize_plan_mode() wire-format digests: sha256 + byte length per query.\n",
            "# REGENERATING THIS DEFEATS ITS PURPOSE. A red digest test means the serialized\n",
            "# bytes MOVED — find out WHY (the C++ side reads these bytes); do NOT regenerate\n",
            "# to make it pass. See tests/test_plan_bytes.rs for the full rationale.\n",
            "# Plans are built via the FIXED symlink /tmp/peacock-plan-bytes-root, so the\n",
            "# absolute parquet paths the wire format embeds are identical on every machine\n",
            "# and these digests are portable. Path VALUES are therefore held constant, not\n",
            "# normalized out; the round-trip oracle covers them. Regenerated at c4ddca8 —\n",
            "# equal to the 9cc44c9 capture by the 131/131 0-drifted proof across Inc3.\n",
        );
        std::fs::write(&path, format!("{header}{rendered}")).unwrap();
        eprintln!("Updated plan-bytes digests: {} ({} entries)", path.display(), actual.len());
        return;
    }

    let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing {}\nRun with UPDATE_CANONICAL=1 to generate it.",
            path.display()
        )
    });

    // Report every drifted query, not just the first — a layout change moves all of
    // them at once, and seeing one line is misleading about the blast radius.
    let expected_map: BTreeMap<&str, &str> = expected
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once("  "))
        .collect();
    let mut drifted = Vec::new();
    for (k, v) in &actual {
        match expected_map.get(k.as_str()) {
            Some(e) if *e == v.as_str() => {}
            Some(e) => drifted.push(format!("  {k}\n    expected {e}\n    actual   {v}")),
            None => drifted.push(format!("  {k}\n    (absent from the digest golden)")),
        }
    }
    let missing: Vec<&&str> =
        expected_map.keys().filter(|k| !actual.contains_key(**k)).collect();

    assert!(
        drifted.is_empty() && missing.is_empty(),
        "serialized plan bytes moved for {} of {} queries (and {} golden entries vanished).\n\
         The FlatBuffer wire format changed — the C++ side reads these bytes, so this is a \
         cross-language change, not a test nit. Justify it, then regenerate with \
         UPDATE_CANONICAL=1.\n{}\n{}",
        drifted.len(),
        actual.len(),
        missing.len(),
        drifted.join("\n"),
        if missing.is_empty() { String::new() } else { format!("vanished: {missing:?}") }
    );
}
