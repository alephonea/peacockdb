//! Bespoke (non-macro) CPU-executor tests. The parameterized result/cost suite
//! lives in test_cpu_full_table.rs / test_cpu_partitioned.rs; shared helpers live
//! in common/mod.rs.
#[macro_use]
mod common;

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::physical_plan::ExecutionPlan;

use peacockdb_core::create_context_with_tables_mode;
use peacockdb_core::cpu_executor::NodeMemoryStats;
use peacockdb_core::executors::full_table_cpu_executor::execute_full_table_instrumented;

use common::exec_mode::{CpuOracle, ExecMode, ResultGolden};
use common::{
    all_node_names, data_dir_for, device_config, fmt_plan, has_gpu_node, make_ctx,
    plan_is_partitioned_executable, queries_dir_for, scan_batch_sizes, FULL_BUDGET,
    BATCH_STRESS_BUDGET,
};

/// Lock the routing predicate — the gate is AGG-KIND (state-mergeability), not the
/// mere presence of a repartition:
///   - q6 (global additive agg, no shuffle)                    → partitioned-executable
///   - shuffle_additive (GROUP BY + Hash shuffle, SUM/COUNT)   → partitioned-executable
///   - q1 (GROUP BY + Hash shuffle, AVG state = sum/count)     → partitioned-executable
///   - shuffle_stddev (GROUP BY + Hash shuffle, STDDEV M2)     → partitioned-executable
/// All are evaluated at tp8-standard so they carry a scan map; the ONLY discriminator
/// is whether every Final-aggregate is state-mergeable. Fails LOUDLY if the predicate
/// is narrowed to reject a legitimate mergeable shuffle (sum/count/min/max/avg/stddev/var)
/// — an UNKNOWN aggregate still defaults to NON-mergeable (whitelist fail-safe).
#[tokio::test]
async fn routing_predicate_gates_on_agg_kind() {
    async fn plan_for(query: &str, device: &str) -> Arc<dyn ExecutionPlan> {
        let (parts, budget) = device_config(device);
        // Real-partitioning mode so the scan carries its map (the predicate keys on
        // has_scan_map); the enum — not the budget — is the gate. Stated here rather
        // than derived from `device`: this builds an EXECUTION context.
        let ctx = create_context_with_tables_mode(
            &data_dir_for("tpch", "1"),
            parts,
            budget,
            ExecMode::Partitioned.partition_mode(),
        )
        .await
        .unwrap();
        let sql =
            std::fs::read_to_string(queries_dir_for("tpch").join(format!("{query}.sql"))).unwrap();
        ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap()
    }
    let q6 = plan_for("q6", "tp8-standard").await;
    assert!(plan_is_partitioned_executable(&q6), "q6 (additive global agg) must route to the partitioned executor");
    let shuffle = plan_for("shuffle-additive", "tp8-standard").await;
    assert!(
        plan_is_partitioned_executable(&shuffle),
        "shuffle_additive (SUM/COUNT over a Hash shuffle) must route to the partitioned executor"
    );
    let q1 = plan_for("q1", "tp8-standard").await;
    assert!(
        plan_is_partitioned_executable(&q1),
        "q1 (AVG = additive sum/count state) must route to the partitioned executor (Inc4)"
    );
    let stddev = plan_for("shuffle-stddev", "tp8-standard").await;
    assert!(
        plan_is_partitioned_executable(&stddev),
        "shuffle_stddev (STDDEV = Welford count/mean/m2 state) must route to the partitioned executor (Inc5)"
    );
}

/// A STDDEV Final-agg query driven through the partitioned CpuNodeExecutor at a
/// real-partitioning device (tp8-standard) MERGES CORRECTLY — its Welford
/// [count, mean, m2] state combines across the 8 hash partitions (each group lands
/// wholly in one bucket, so the per-bucket Final is exact). Asserts the #13 result
/// matches DataFusion (the oracle). An UNKNOWN aggregate still stays full-table
/// (whitelist fail-safe).
#[tokio::test]
async fn inc5_stddev_final_agg_at_tp8_partitioned_matches_datafusion() {
    // DataFusionApproximate (see CpuOracle for the general rationale): here the
    // reassociated sum is the Welford M2 across the 8 partitions — observed rel diff
    // ~3e-14, so exact-string compare cannot be used.
    // Write, not Skip: the golden_approx_std consumer at
    // shuffle-stddev/partitioned-tp8-standard is enabled again (#103 was an
    // uninitialized OutBuild field, not the merge), so this golden is read rather than
    // orphaned — which is the thing the `ResultGolden` gate is there to prevent.
    common::assert_cpu_results_match_datafusion(
        "tpch",
        "1",
        "shuffle-stddev",
        "tp8-standard",
        CpuOracle::DataFusionApproximate,
        ExecMode::Partitioned,
        ResultGolden::Write,
    )
    .await;
}

/// Every Gpu* node must be stripped before CPU execution; no stat may name one.
#[tokio::test]
async fn test_execution_strips_gpu_nodes() {
    let ctx = make_ctx(FULL_BUDGET).await;
    let plan = ctx
        .sql("SELECT count(*) FROM nation WHERE n_regionkey >= 0")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    assert!(has_gpu_node(&plan), "expected GPU nodes in plan, got: {:?}", all_node_names(&plan));

    let mut stats: Vec<NodeMemoryStats> = vec![];
    execute_full_table_instrumented(plan, ctx.task_ctx(), &mut stats).await.unwrap();

    assert!(!stats.is_empty(), "no nodes were executed");
    let gpu_names: Vec<&str> =
        stats.iter().filter(|s| s.node_name.starts_with("Gpu")).map(|s| s.node_name.as_str()).collect();
    assert!(gpu_names.is_empty(), "GPU nodes not stripped: {gpu_names:?}");
}

#[tokio::test]
async fn test_memory_boundary_preserved_tight_budget() {
    let query = "SELECT count(*) FROM customer WHERE c_custkey > 0";

    let ctx_full = make_ctx(FULL_BUDGET).await;
    let plan_full = ctx_full.sql(query).await.unwrap().create_physical_plan().await.unwrap();

    let ctx_tight = make_ctx(BATCH_STRESS_BUDGET).await;
    let plan_tight = ctx_tight.sql(query).await.unwrap().create_physical_plan().await.unwrap();

    eprintln!("\n=== FULL BUDGET ({} GiB) plan ===\n{}", FULL_BUDGET / (1024 * 1024 * 1024), fmt_plan(&plan_full));
    eprintln!("=== TIGHT BUDGET ({} KiB) plan ===\n{}", BATCH_STRESS_BUDGET / 1024, fmt_plan(&plan_tight));

    let tight_scan_sizes = scan_batch_sizes(&plan_tight);
    assert!(
        !tight_scan_sizes.is_empty(),
        "expected GpuScanExec in tight plan; node names: {:?}",
        all_node_names(&plan_tight)
    );
    let gpu_batch_size = *tight_scan_sizes.iter().max().unwrap();

    let full_scan_sizes = scan_batch_sizes(&plan_full);
    let full_batch_size = *full_scan_sizes.iter().max().unwrap();

    eprintln!("GpuScanExec batch_size — full budget: {full_batch_size}, tight budget: {gpu_batch_size}");
    assert!(
        gpu_batch_size < full_batch_size,
        "tight budget batch_size ({gpu_batch_size}) should be smaller than full budget ({full_batch_size})"
    );

    let mut stats: Vec<NodeMemoryStats> = vec![];
    let batches =
        execute_full_table_instrumented(plan_tight, ctx_tight.task_ctx(), &mut stats).await.unwrap();

    let count = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
    assert_eq!(count, 150_000, "customer table must have 150 000 rows");

    let scan_stats: Vec<&NodeMemoryStats> =
        stats.iter().filter(|s| s.node_name == "ParquetExec").collect();
    assert!(!scan_stats.is_empty(), "expected ParquetExec in stats");

    eprintln!("Per-node stats (post-order):");
    for s in &stats {
        eprintln!(
            "  {}: rows={}, max_batch={}, alloc={}B, out={}B",
            s.node_name, s.row_count, s.max_batch_rows, s.allocated_bytes, s.output_bytes
        );
    }

    for s in &scan_stats {
        assert!(
            s.max_batch_rows <= gpu_batch_size,
            "ParquetExec batch {} rows exceeds gpu_batch_size={}",
            s.max_batch_rows,
            gpu_batch_size
        );
    }

    let gpu_names: Vec<&str> =
        stats.iter().filter(|s| s.node_name.starts_with("Gpu")).map(|s| s.node_name.as_str()).collect();
    assert!(gpu_names.is_empty(), "GPU nodes in stats: {gpu_names:?}");
}

#[tokio::test]
async fn test_instrumented_stats_are_populated() {
    let ctx = make_ctx(FULL_BUDGET).await;
    let plan = ctx
        .sql("SELECT n_name, n_regionkey FROM nation WHERE n_regionkey = 1")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    let mut stats: Vec<NodeMemoryStats> = vec![];
    let batches =
        execute_full_table_instrumented(plan, ctx.task_ctx(), &mut stats).await.unwrap();

    let final_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let root_stat = stats.last().unwrap();
    assert_eq!(root_stat.row_count, final_rows, "root node row_count in stats does not match actual output");
    assert!(root_stat.allocated_bytes > 0, "root node allocated_bytes should be > 0");
    assert!(root_stat.output_bytes > 0, "root node output_bytes should be > 0");
    assert!(
        root_stat.allocated_bytes >= root_stat.output_bytes,
        "allocated_bytes ({}) must be >= output_bytes ({})",
        root_stat.allocated_bytes,
        root_stat.output_bytes
    );
}

// --- the calibration record (#153 C5) ---------------------------------------

/// Every region the record emits, in emission order, as `(out_rows, out_bytes)` read
/// off a `.cpu.txt` golden: a node's `p<k>:` sub-lines when it has them, its own
/// `output_rows`/`output_bytes` when it does not (the golden's N=1 convention).
fn golden_regions(text: &str) -> Vec<(usize, usize)> {
    fn field(line: &str, key: &str) -> Option<usize> {
        let rest = &line[line.find(key)? + key.len()..];
        rest.chars().take_while(char::is_ascii_digit).collect::<String>().parse().ok()
    }
    let mut regions: Vec<(usize, usize)> = Vec::new();
    // Index in `regions` of the node line still eligible to be replaced by its own
    // sub-lines. Reset at each node line; a node with sub-lines contributes them
    // INSTEAD of itself, which is exactly the record's rule.
    let mut pending: Option<usize> = None;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with('p') && t.split_once(':').is_some_and(|(k, _)| k[1..].chars().all(|c| c.is_ascii_digit())) {
            if let Some(i) = pending.take() {
                regions.truncate(i);
            }
            regions.push((field(t, "out_rows=").unwrap(), field(t, "out_bytes=").unwrap()));
        } else if let (Some(r), Some(b)) = (field(t, "output_rows="), field(t, "output_bytes=")) {
            pending = Some(regions.len());
            regions.push((r, b));
        }
    }
    regions
}

/// C5's proving test, and the half of the calibration record that can be checked
/// without a GPU: which regions exist, in what order, and with what rows and bytes.
/// The timing columns are zero here — those are gated by `test_node_timing` on the
/// device. Split that way on purpose: the structural half is what silently rots when
/// the plan shape or the partition accounting changes, and it costs nothing to run in
/// every CPU tier.
///
/// Both modes, because they are the two the record's partition column has to survive:
/// full-table is one region per node, partitioned is eight for every node below the
/// coalesce, and only the golden knows which.
#[tokio::test]
async fn calibration_record_regions_match_the_cpu_golden() {
    use common::cost_model::CostModel;
    use common::record::{record_rows, RunMeta, COLUMNS};
    use common::{cpu_golden, data_dir_for, exec_mode::golden_label, queries_dir_for};
    use peacockdb_core::executors::full_table_cpu_executor::execute_full_table_instrumented_enforced;
    use peacockdb_core::node_executor::{execute_node_by_node, CpuNodeExecutor};

    let model = CostModel::load();
    for (query, mode, device) in
        [("q6", ExecMode::FullTable, "tp8-mini"), ("q6", ExecMode::Partitioned, "tp8-standard")]
    {
        let (partitions, budget) = device_config(device);
        let ctx = create_context_with_tables_mode(
            &data_dir_for("tpch", "1"),
            partitions,
            budget,
            mode.partition_mode(),
        )
        .await
        .unwrap();
        let sql =
            std::fs::read_to_string(queries_dir_for("tpch").join(format!("{query}.sql"))).unwrap();
        let plan = ctx.sql(&sql).await.unwrap().create_physical_plan().await.unwrap();

        let stats: Vec<NodeMemoryStats> = match mode {
            ExecMode::Partitioned => {
                let mut backend = CpuNodeExecutor::new(ctx.task_ctx());
                execute_node_by_node(&plan, &mut backend).await.unwrap().1
            }
            ExecMode::FullTable => {
                let mut stats = vec![];
                execute_full_table_instrumented_enforced(
                    plan.clone(),
                    ctx.task_ctx(),
                    budget,
                    &mut stats,
                )
                .await
                .unwrap();
                stats
            }
        };

        let label = golden_label(mode, device);
        let meta = RunMeta {
            source: "peacockdb",
            dataset: "tpch",
            sf: "1",
            query,
            label: &label,
            timing_mode: "off",
            build_profile: "test",
            allocator: "none",
        };
        let rows = record_rows(&plan, &stats, &meta, &model);
        let golden =
            std::fs::read_to_string(cpu_golden("tpch", "1", query, &label)).unwrap();
        let expect = golden_regions(&golden);

        assert_eq!(
            rows.len(),
            expect.len(),
            "{query} @ {label}: record has {} regions, the golden {}",
            rows.len(),
            expect.len()
        );
        for (i, (row, (want_rows, want_bytes))) in rows.iter().zip(&expect).enumerate() {
            let f: Vec<&str> = row.split('\t').collect();
            assert_eq!(
                f.len(),
                COLUMNS.len(),
                "{query} @ {label} region {i}: {} fields, {} columns",
                f.len(),
                COLUMNS.len()
            );
            let at = |name: &str| f[COLUMNS.iter().position(|c| *c == name).unwrap()];
            assert_eq!(
                (at("out_rows"), at("out_bytes")),
                (want_rows.to_string().as_str(), want_bytes.to_string().as_str()),
                "{query} @ {label} region {i} ({} p{}): record disagrees with the golden",
                at("node_type"),
                at("partition"),
            );
            // The regressor is the category's bytes, and on this source that is
            // out_bytes — the property the sf40 side cannot check for us.
            assert_eq!(at("cuda_bytes"), at("out_bytes"), "{query} @ {label} region {i}");
            assert!(
                !at("category").is_empty() && at("category") != "-",
                "{query} @ {label} region {i}: node type {} is untagged",
                at("node_type")
            );
        }

        // Σ over a node's regions must be its children's totals. The bytes columns are
        // per-partition but the fit reads them as a decomposition of the node's input,
        // and a partition mapping that drops part of a child is invisible row by row —
        // a coalesce reporting one eighth of what it concatenates still looks plausible.
        let mut sums: HashMap<usize, (usize, usize, usize, usize)> = HashMap::new();
        for row in &rows {
            let f: Vec<&str> = row.split('\t').collect();
            let at = |name: &str| {
                f[COLUMNS.iter().position(|c| *c == name).unwrap()].parse::<usize>().unwrap()
            };
            let e = sums.entry(at("node_seq")).or_default();
            e.0 += at("in_rows");
            e.1 += at("in_bytes");
            e.2 += at("out_rows");
            e.3 += at("out_bytes");
        }
        let mut edges: Vec<(usize, Vec<usize>)> = Vec::new();
        fn walk_edges(
            plan: &Arc<dyn ExecutionPlan>,
            idx: &mut usize,
            out: &mut Vec<(usize, Vec<usize>)>,
        ) -> usize {
            let kids: Vec<usize> =
                plan.children().iter().map(|c| walk_edges(c, idx, out)).collect();
            let seq = *idx;
            *idx += 1;
            out.push((seq, kids));
            seq
        }
        walk_edges(&plan, &mut 0, &mut edges);
        for (seq, kids) in &edges {
            let want = kids.iter().fold((0, 0), |a, k| (a.0 + sums[k].2, a.1 + sums[k].3));
            let got = (sums[seq].0, sums[seq].1);
            assert_eq!(
                got, want,
                "{query} @ {label} node {seq}: Σ input over regions is {got:?}, children \
                 produced {want:?}"
            );
        }
    }
}
