mod common;
use common::data_dir_for;
use peacockdb_core::batch_partitioned::plan::{BatchSizing, PlanKnobs, plan_batch_partitioned};
use peacockdb_core::batch_partitioned::plan_text::render_plan;

#[tokio::test]
async fn probe() {
    let knobs = PlanKnobs {
        target_partitions: 4,
        sizing: BatchSizing::OneBatchPerRowGroup,
        budget: 512 * 1024 * 1024,
        small_table_bytes: 5 * 1024 * 1024,
    };
    let ctx = peacockdb_core::register_tables_for(
        peacockdb_core::build_session_state(4),
        &data_dir_for("tpch", "1"),
    )
    .await
    .expect("tables");
    for (label, sql) in [
        ("join", "SELECT c_nationkey, c_mktsegment, count(*) FROM customer JOIN nation ON c_nationkey = n_nationkey GROUP BY ROLLUP(c_nationkey, c_mktsegment)"),
        ("inner-group", "SELECT c_nationkey, c_mktsegment, count(*) FROM (SELECT c_nationkey, c_mktsegment FROM customer GROUP BY c_nationkey, c_mktsegment) s GROUP BY ROLLUP(c_nationkey, c_mktsegment)"),
    ] {
        let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap();
        match plan_batch_partitioned(&plan, knobs) {
            Ok((tree, _)) => {
                let text = render_plan(tree.as_ref());
                println!("=== {label}");
                for line in text.lines() {
                    let head: String = line.chars().take(150).collect();
                    println!("{head}{}", if line.contains("hashed_on=") { "   <<< HASHED" } else { "" });
                }
            }
            Err(e) => println!("=== {label}\nrefused: {e}"),
        }
    }
}
