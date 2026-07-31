//! GPU-coverage & output-size report generator (Task 6).
//!
//! Reads only committed goldens, so it runs in the CI CPU tier with no GPU and
//! no executor build:
//!   - PeacockDB Σout = Σ `output_bytes` over a query's `<q>.cpu.txt` cost tree
//!                      (every CPU operator's output size).
//!   - DuckDB Σout    = pipeline-breaker materialized bytes computed from the
//!                      `<q>.duckdb_cost.txt` profiling tree (see [`duckdb_cost`]).
//!   - GPU coverage   = whether the query's GPU result test is enabled in
//!                      `test_gpu.rs` (uncommented macro invocation).
//!
//! Both sides are deterministic, measured byte sums — NOT wall-clock cost, and
//! the two engines emit different plan trees, so the ratio is a provisional,
//! directional-only proxy (the report displays it, asserts nothing).
//!
//! Emits a self-contained HTML page (inline CSS, for GitHub Pages) and a compact
//! Markdown blob (for the upserted PR comment, keyed on [`SENTINEL`]).
//!
//! Usage:
//!   cost-report [--testdata DIR] [--tests FILE] [--html FILE] [--md FILE]
//!               [--pages-url URL] [--sha SHA] [--repo OWNER/REPO] [--published]

use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// PeacockDB/DuckDB Σout ratio at or below this renders green; above gets the
/// light-red row highlight (🔴 in the markdown comment). Single configurable
/// threshold for both renderers — directional only; revisit as the models converge.
const RATIO_GREEN_MAX: f64 = 1.4;

const PAGES_URL_DEFAULT: &str = "https://asymptote-tech.github.io/peacockdb/";
const DEFAULT_REPO: &str = "asymptote-tech/peacockdb";
/// Device label of the CPU-cost goldens (8 partitions / 2 GiB), part of the
/// `.cpu.txt` filename under the unified golden layout. MUST track the device the
/// `cpu_result_test!` goldens are canonized at — #11 renamed these tp1 → tp8, so a
/// stale label here makes every PeacockDB cell render "—" (now guarded in `main`).
const CPU_DEVICE: &str = "tp8-mini";
/// Hidden marker so CI can find-and-update its single PR comment in place.
const SENTINEL: &str = "<!-- peacockdb-cost-report -->";
/// Separate marker for the cost-regression gate widget, so it upserts as its own
/// PR comment independently of the coverage/ratio report above.
const DIFF_SENTINEL: &str = "<!-- peacockdb-cost-regression -->";

/// The execution-mode columns, in display order. `all_at_once` deliberately has no
/// column: it is a whole-plan-at-once GPU path with no per-node breakdown, so a
/// per-query enabled/disabled cell would be meaningless.
const MODE_COLUMNS: [&str; 5] = [
    "ftc_tp1",
    "ftc_tp8",
    "partitioned_cpu",
    "full_table_gpu",
    "partitioned_gpu",
];

/// One row of `testdata/cost-registry.csv` — the widget's source of truth.
///
/// This REPLACES the old textual scrape of `test_gpu.rs`, which could only see one
/// file and inferred "enabled" from whether a macro line was commented out. The CSV
/// is verified against the test suite's own link-time inventory by the registry
/// tests in peacockdb-core (both directions), so a stale cell fails the build rather
/// than silently mis-rendering a tick here.
struct Registry {
    rows: Vec<RegistryRow>,
}

struct RegistryRow {
    dataset: String,
    query: String,
    states: BTreeMap<String, String>,
    features: Vec<String>,
    tickets: Vec<String>,
}

impl Registry {
    fn load(path: &Path) -> Registry {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read registry CSV {}: {e}", path.display()));
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().expect("registry CSV is empty").split(',').collect();
        let cols: Vec<String> = header.iter().map(|s| s.to_string()).collect();
        let idx = |name: &str| -> usize {
            cols.iter()
                .position(|c| c == name)
                .unwrap_or_else(|| panic!("registry CSV has no {name:?} column"))
        };
        let (i_ds, i_q, i_feat, i_tick) =
            (idx("dataset"), idx("query"), idx("features"), idx("tickets"));
        let mode_idx: Vec<(String, usize)> =
            MODE_COLUMNS.iter().map(|m| (m.to_string(), idx(m))).collect();

        let mut rows = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f.len(), cols.len(), "registry CSV: ragged row: {line}");
            let states = mode_idx
                .iter()
                .map(|(m, i)| (m.clone(), f[*i].to_string()))
                .collect();
            rows.push(RegistryRow {
                dataset: f[i_ds].to_string(),
                query: f[i_q].to_string(),
                states,
                features: f[i_feat].split_whitespace().map(str::to_string).collect(),
                tickets: f[i_tick].split_whitespace().map(str::to_string).collect(),
            });
        }
        assert!(!rows.is_empty(), "registry CSV has no rows");
        Registry { rows }
    }

    fn for_dataset(&self, dataset: &str) -> impl Iterator<Item = &RegistryRow> {
        self.rows.iter().filter(move |r| r.dataset == dataset)
    }
}

/// Render a cell state as its glyph. `enabled` means the mode runs AND its result is
/// validated (golden or live oracle — both count); `skip` means it runs but nothing
/// checks the result, which is why it gets its own glyph rather than being folded
/// into either ✓ or ✗.
fn state_glyph(state: &str) -> &'static str {
    match state {
        "enabled" => "✓",
        "skip" => "~",
        "disabled" => "✗",
        _ => "—",
    }
}

struct Row {
    /// Underscore form as written in the tests/CSV (`q1`, `shuffle_stddev`).
    query: String,
    /// Query number when the name is `q<N>` — used for the numeric sort and the
    /// `q<n>.sql` link. `None` for the synthetic micro-queries (aggregate_groupby,
    /// mixed_join, …), which the spec requires the widget to include.
    n: Option<u32>,
    states: BTreeMap<String, String>,
    features: Vec<String>,
    tickets: Vec<String>,
    peacockdb: Option<u64>,
    duckdb: Option<u64>,
}

impl Row {
    /// Golden/SQL file stem: the CSV's underscore form maps to hyphenated paths.
    fn stem(&self) -> String {
        self.query.replace('_', "-")
    }

    fn state(&self, col: &str) -> &str {
        self.states.get(col).map(String::as_str).unwrap_or("na")
    }

    /// "Operational" for the summary line = the single-partition GPU mode is
    /// enabled. That is the mode the old test_gpu.rs scrape counted, so the
    /// headline number stays comparable across this change.
    fn operational(&self) -> bool {
        self.state("full_table_gpu") == "enabled"
    }
}

impl Row {
    /// PeacockDB Σout / DuckDB Σout (the spec's ratio direction).
    fn ratio(&self) -> Option<f64> {
        match (self.peacockdb, self.duckdb) {
            (Some(p), Some(d)) if d > 0 => Some(p as f64 / d as f64),
            _ => None,
        }
    }

    /// Color bucket: green (within budget), red (over), grey (skip / no cost).
    fn bucket(&self) -> &'static str {
        match self.ratio() {
            _ if !self.operational() => "grey",
            Some(r) if r <= RATIO_GREEN_MAX => "green",
            Some(_) => "red",
            None => "grey",
        }
    }

    /// full_table_cpu is one logical mode run at two target-partition counts, so it
    /// renders as a single cell showing the split rather than two columns.
    fn ftc_cell(&self) -> String {
        format!(
            "tp1{} tp8{}",
            state_glyph(self.state("ftc_tp1")),
            state_glyph(self.state("ftc_tp8"))
        )
    }
}

struct Dataset {
    label: &'static str,
    total: usize,
    /// Repo-relative golden dir, e.g. "testdata/goldens/tpch.sf1" — used for cell links.
    canon_rel: &'static str,
    /// Repo-relative query-SQL dir, e.g. "testdata/tpch-queries" — used to link the
    /// Query column to each query's `q<n>.sql` source.
    query_rel: &'static str,
    rows: Vec<Row>,
}

impl Dataset {
    fn operational(&self) -> usize {
        self.rows.iter().filter(|r| r.operational()).count()
    }
}

/// Where golden files live, and how to link to them at a given commit.
struct Links {
    repo: String,
    sha: Option<String>,
}

impl Links {
    /// `stem` is the hyphenated file stem (`q6`, `scan-limit`), so numbered and
    /// synthetic queries share one path builder.
    fn golden_url(&self, canon_rel: &str, stem: &str, ext: &str) -> Option<String> {
        let sha = self.sha.as_ref()?;
        Some(format!("https://github.com/{}/blob/{sha}/{canon_rel}/{stem}.{ext}", self.repo))
    }

    /// Link to a query's SQL source (`<query_rel>/<stem>.sql`) at the report's
    /// commit; `None` on dry runs (no sha), mirroring [`golden_url`]. The synthetic
    /// micro-queries have real .sql files too (aggregate-groupby.sql, …), so they
    /// link exactly like the numbered ones.
    fn query_url(&self, query_rel: &str, stem: &str) -> Option<String> {
        let sha = self.sha.as_ref()?;
        Some(format!("https://github.com/{}/blob/{sha}/{query_rel}/{stem}.sql", self.repo))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |name: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    let env = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());

    let testdata = PathBuf::from(opt("--testdata", "testdata"));
    // The registry CSV lives beside the goldens it describes; --registry overrides
    // for out-of-tree runs. (`--tests`, which pointed at test_gpu.rs for the old
    // textual scrape, is gone — coverage is read from this CSV now.)
    let registry_csv = PathBuf::from(opt(
        "--registry",
        testdata.join("cost-registry.csv").to_str().unwrap_or("testdata/cost-registry.csv"),
    ));
    let html_out = opt("--html", "cost_report.html");
    // When set, assemble the page-per-sha Pages site here instead of writing a
    // single --html file (master deploy); see `assemble_site`.
    let site = opt("--site", "");
    let md_out = opt("--md", "");
    let pages_url = opt("--pages-url", PAGES_URL_DEFAULT);
    let published = args.iter().any(|a| a == "--published");

    // Code version the report was generated from, for golden + query cell links.
    // Degrades to plain (unlinked) cells when unavailable (e.g. a local dry run).
    // Resolved before the cost-diff branch so its Query column can link too.
    let sha = if let Some(s) = args.iter().position(|a| a == "--sha").and_then(|i| args.get(i + 1)) {
        Some(s.clone())
    } else {
        env("GITHUB_SHA")
    };
    let repo = opt("--repo", &env("GITHUB_REPOSITORY").unwrap_or_else(|| DEFAULT_REPO.to_string()));
    let links = Links { repo, sha };

    // Cost-regression gate (separate mode): diff this tree's .cost.txt totals
    // against a base, render a per-query change widget, and exit non-zero on any
    // regression. Self-contained — does not build the coverage report below.
    if args.iter().any(|a| a == "--cost-diff") {
        run_cost_diff(
            &testdata,
            &opt("--base", ""),
            &opt("--html", "cost_diff.html"),
            &md_out,
            &links,
        );
        return;
    }

    // Render-time UTC, supplied by CI (`date -u '+%Y-%m-%d %H:%M UTC'`) so the bin
    // stays std-only (no date crate). Omitted on local dry runs → no freshness line.
    let generated_at = args
        .iter()
        .position(|a| a == "--generated-at")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .or_else(|| env("COST_REPORT_GENERATED_AT"));

    // Coverage now comes from the committed registry CSV (verified against the test
    // suite's link-time inventory by peacockdb-core's registry tests), NOT from
    // scraping test_gpu.rs for uncommented macro lines.
    let registry = Registry::load(&registry_csv);

    let tpch = build_dataset("TPC-H", 22, "testdata/goldens/tpch.sf1", "testdata/tpch-queries", &testdata.join("goldens/tpch.sf1"), &registry, "tpch");
    let tpcds = build_dataset("TPC-DS", 99, "testdata/goldens/tpcds.sf1", "testdata/tpcds-queries", &testdata.join("goldens/tpcds.sf1"), &registry, "tpcds");
    let datasets = [tpch, tpcds];

    // CI gate: a query whose Σout we EXPECT must actually have one. A missing value
    // silently renders "—" (e.g. a stale CPU_DEVICE or an absent golden) and CI
    // would stay green — so fail loudly instead.
    //
    // "Expect" = the query is GPU-operational AND its full_table_cpu run at the
    // CPU_DEVICE tier is enabled, because that is the exact golden the Σout is read
    // from (`<query>.{CPU_DEVICE}.cost.txt`). Requiring it of every operational
    // query would be wrong now that the registry includes the synthetic
    // micro-queries: tpch/scan_limit is GPU-operational but runs full_table_cpu at
    // tp1-mini only, so no tp8-mini cost golden exists or should. Its Σout cell is a
    // dash, and the ftc column shows why.
    let mut missing: Vec<String> = Vec::new();
    for d in &datasets {
        for r in &d.rows {
            if r.operational() && r.state("ftc_tp8") == "enabled" && r.peacockdb.is_none() {
                missing.push(format!("{} {}", d.label, r.query));
            }
        }
    }
    if !missing.is_empty() {
        eprintln!(
            "cost-report: missing PeacockDB cost for {} operational queries (stale CPU_DEVICE='{CPU_DEVICE}' or absent .cost.txt goldens): {}",
            missing.len(),
            missing.join(", ")
        );
        std::process::exit(1);
    }

    let freshness = freshness_line(links.sha.as_deref(), generated_at.as_deref());

    if site.is_empty() {
        let html = render_html(&datasets, &pages_url, &links, generated_at.as_deref(), None);
        std::fs::write(&html_out, &html).unwrap_or_else(|e| panic!("write {html_out}: {e}"));
        eprintln!("wrote {html_out}");
    } else {
        assemble_site(Path::new(&site), &datasets, &pages_url, &links, generated_at.as_deref(), links.sha.as_deref());
        eprintln!("assembled page-per-sha site at {site}/");
    }

    let md = render_markdown(&datasets, &pages_url, published, &links, freshness.as_deref());
    if md_out.is_empty() {
        print!("{md}");
    } else {
        std::fs::write(&md_out, &md).unwrap_or_else(|e| panic!("write {md_out}: {e}"));
        eprintln!("wrote {md_out}");
    }
}

fn build_dataset(
    label: &'static str,
    total: usize,
    canon_rel: &'static str,
    query_rel: &'static str,
    canon: &Path,
    registry: &Registry,
    dataset_key: &str,
) -> Dataset {
    // Each golden carries its own explicit total footer (peacockdb_cost= /
    // duckdb_cost=), the single source of truth for that side's number; the
    // per-node output_bytes/materialized values above it are the contribution
    // breakdown that sums to it. We read the footer (sum_field over a key that
    // appears once == that value).
    // Rows come from the registry CSV, not a 1..=N range, so the synthetic
    // micro-queries (aggregate_groupby, mixed_join, …) appear alongside the numbered
    // ones as the spec requires. Numbered queries sort first, by number.
    let mut rows: Vec<Row> = registry
        .for_dataset(dataset_key)
        .map(|r| {
            let n = r
                .query
                .strip_prefix('q')
                .and_then(|d| d.parse::<u32>().ok());
            let stem = r.query.replace('_', "-");
            Row {
                query: r.query.clone(),
                n,
                states: r.states.clone(),
                features: r.features.clone(),
                tickets: r.tickets.clone(),
                // PeacockDB total now lives in the cheap-to-regenerate .cost.txt (the
                // .cpu.txt no longer carries a footer); same `peacockdb_cost=` key.
                peacockdb: read_total(
                    &canon.join(format!("{stem}.{CPU_DEVICE}.cost.txt")),
                    "peacockdb_cost=",
                ),
                duckdb: read_total(&canon.join(format!("{stem}.duckdb_cost.txt")), "duckdb_cost="),
            }
        })
        .collect();
    rows.sort_by(|a, b| match (a.n, b.n) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.query.cmp(&b.query),
    });
    Dataset { label, total, canon_rel, query_rel, rows }
}

/// The explicit total carried by a golden's footer line (`<key><n>`), e.g.
/// `peacockdb_cost=`/`duckdb_cost=`. `None` if the file is absent OR the footer
/// is missing — so a footerless/malformed golden renders grey (—), never a false
/// green 0.
fn read_total(path: &Path, key: &str) -> Option<u64> {
    read_total_str(&std::fs::read_to_string(path).ok()?, key)
}

/// `Some(value)` iff `key` appears on some line; `None` if it's absent.
fn read_total_str(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|l| l.find(key).map(|_| field(l, key)))
}

/// The digit run immediately after the first `key` on `line` (0 if absent).
fn field(line: &str, key: &str) -> u64 {
    match line.find(key) {
        Some(pos) => line[pos + key.len()..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .unwrap_or(0),
        None => 0,
    }
}

fn fmt_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[i])
    }
}

fn ratio_or_dash(r: Option<f64>) -> String {
    r.map(|r| format!("{r:.2}")).unwrap_or_else(|| "—".to_string())
}

/// A byte-cost table cell: linked to its golden when the value and a link both
/// exist, plain text otherwise (no broken links on dry runs / missing goldens).
fn cost_cell_html(value: Option<u64>, url: Option<String>) -> String {
    match (value, url) {
        (Some(v), Some(u)) => format!("<a href=\"{u}\">{}</a>", fmt_bytes(v)),
        (Some(v), None) => fmt_bytes(v),
        (None, _) => "—".to_string(),
    }
}

fn cost_cell_md(value: Option<u64>, url: Option<String>) -> String {
    match (value, url) {
        (Some(v), Some(u)) => format!("[{}]({u})", fmt_bytes(v)),
        (Some(v), None) => fmt_bytes(v),
        (None, _) => "—".to_string(),
    }
}

/// PeacockDB Σout cell: the cost total, then "plan"/"cost" links to the `.cpu.txt`
/// (per-node tree) and `.cost.txt` (cost components) goldens. Links render only
/// when the value AND a sha-based URL exist (plain bytes on dry runs / no sha);
/// a missing value is an em-dash, never a link.
fn peacock_cell_html(value: Option<u64>, plan_url: Option<String>, cost_url: Option<String>) -> String {
    let Some(v) = value else { return "—".to_string() };
    let mut links = Vec::new();
    if let Some(u) = plan_url {
        links.push(format!("<a href=\"{u}\">plan</a>"));
    }
    if let Some(u) = cost_url {
        links.push(format!("<a href=\"{u}\">cost</a>"));
    }
    if links.is_empty() {
        fmt_bytes(v)
    } else {
        format!("{} ({})", fmt_bytes(v), links.join(", "))
    }
}

fn peacock_cell_md(value: Option<u64>, plan_url: Option<String>, cost_url: Option<String>) -> String {
    let Some(v) = value else { return "—".to_string() };
    let mut links = Vec::new();
    if let Some(u) = plan_url {
        links.push(format!("[plan]({u})"));
    }
    if let Some(u) = cost_url {
        links.push(format!("[cost]({u})"));
    }
    if links.is_empty() {
        fmt_bytes(v)
    } else {
        format!("{} ({})", fmt_bytes(v), links.join(", "))
    }
}

/// Query-column cell: the `q<n>` label linked to its SQL source when a URL exists
/// (sha present), plain `q<n>` otherwise (dry run) — mirrors the golden-link cells.
fn query_cell_html(name: &str, url: Option<String>) -> String {
    match url {
        Some(u) => format!("<a href=\"{u}\">{name}</a>"),
        None => name.to_string(),
    }
}

/// Feature codes as small chips. Empty renders as an em-dash rather than a blank
/// cell, so "no features" is visibly deliberate.
fn features_html(features: &[String]) -> String {
    if features.is_empty() {
        return "—".to_string();
    }
    features
        .iter()
        .map(|f| format!("<span class=\"chip\">{f}</span>"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Ticket numbers as GitHub issue links. Bare numbers in the CSV; the repo comes
/// from the report's `--repo`, so forks link to their own issues.
fn tickets_html(tickets: &[String], repo: &str) -> String {
    if tickets.is_empty() {
        return "—".to_string();
    }
    tickets
        .iter()
        .map(|t| format!("<a href=\"https://github.com/{repo}/issues/{t}\">#{t}</a>"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn tickets_md(tickets: &[String], repo: &str) -> String {
    if tickets.is_empty() {
        return "—".to_string();
    }
    tickets
        .iter()
        .map(|t| format!("[#{t}](https://github.com/{repo}/issues/{t})"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn query_cell_md(name: &str, url: Option<String>) -> String {
    match url {
        Some(u) => format!("[{name}]({u})"),
        None => name.to_string(),
    }
}

/// Visible "regenerated each run" marker for the upserted PR comment (and HTML
/// header). Needs both the commit and a render timestamp; omitted otherwise.
fn freshness_line(sha: Option<&str>, generated_at: Option<&str>) -> Option<String> {
    let sha = sha?;
    let at = generated_at?;
    let short = &sha[..sha.len().min(7)];
    Some(format!("♻️ _Cost report regenerated for `{short}` at {at}_"))
}

/// `nav_prefix`: `Some(p)` adds a "latest · all reports" nav whose links are
/// relative to `p` (`""` on the site root `index.html`, `"../"` on a per-sha page);
/// `None` (standalone / PR / dry run) omits the nav, since those pages aren't part
/// of the published page-per-sha site.
fn render_html(
    datasets: &[Dataset],
    pages_url: &str,
    links: &Links,
    generated_at: Option<&str>,
    nav_prefix: Option<&str>,
) -> String {
    let mut s = String::new();
    s.push_str("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">");
    s.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
    s.push_str("<title>PeacockDB GPU coverage report</title><style>");
    s.push_str(
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;color:#1b1f23;}\
         h1{font-size:1.5rem;}h2{margin-top:2rem;font-size:1.2rem;}\
         .summary{font-size:1.05rem;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:.6rem .9rem;}\
         table{border-collapse:collapse;width:100%;margin-top:.5rem;}\
         th,td{border:1px solid #d0d7de;padding:.35rem .6rem;text-align:left;font-variant-numeric:tabular-nums;}\
         th{background:#f6f8fa;}td.num{text-align:right;}\
         tr.green td:first-child{border-left:4px solid #1a7f37;}\
         tr.red td:first-child{border-left:4px solid #cf222e;}\
         tr.grey td:first-child{border-left:4px solid #8c959f;}\
         tr.green{background:#e9f7ee;}tr.red{background:#ffe0e0;}tr.grey{background:#f3f4f6;color:#57606a;}\
         .foot{margin-top:1.5rem;color:#57606a;font-size:.85rem;}\
         td.mode{text-align:center;white-space:nowrap;font-variant-numeric:normal;}\
         /* The 4 mode headers are the widest thing in their columns by far — the \
            data cells below them are a single glyph — so they, not the data, set \
            those columns' min-content width and pushed the table into horizontal \
            scrolling. Shrinking just these headers narrows all four columns. */\
         th.modeh{font-size:.72rem;letter-spacing:-.01em;}\
         /* Same reasoning for the features column: its width is set by the longest \
            single chip (an unbreakable token), so the cell font is what controls it. */\
         td.feat{font-size:.68rem;}\
         .chip{display:inline-block;background:#eef2f6;border:1px solid #d0d7de;border-radius:10px;\
               padding:0 .4rem;font-size:.95em;color:#38434f;margin:.05rem 0;}\
         .legend{margin-top:.6rem;color:#57606a;font-size:.85rem;}\
         .caveat{margin-top:.8rem;background:#fff8c5;border:1px solid #d4a72c;border-radius:6px;padding:.6rem .9rem;font-size:.9rem;}",
    );
    s.push_str("</style></head><body>");
    s.push_str("<h1>PeacockDB GPU coverage &amp; output-size report</h1>");

    if let Some(p) = nav_prefix {
        let _ = write!(
            s,
            "<p class=\"foot\"><a href=\"{p}index.html\">latest</a> · <a href=\"{p}history.html\">all reports</a></p>"
        );
    }

    let mut summary = String::new();
    for d in datasets {
        let _ = write!(summary, "{}: {}/{} GPU-operational. ", d.label, d.operational(), d.total);
    }
    let _ = write!(s, "<p class=\"summary\">{}</p>", summary.trim_end());

    if let (Some(sha), Some(at)) = (links.sha.as_deref(), generated_at) {
        let _ = write!(s, "<p class=\"foot\">Generated from {} at {at}</p>", &sha[..sha.len().min(7)]);
    }

    s.push_str(
        "<p class=\"caveat\"><strong>Provisional proxy, not execution cost.</strong> Both columns are \
         deterministic, measured byte sums — not wall-clock time. <em>PeacockDB Σout</em> = Σ per-operator \
         output bytes (Arrow logical size). <em>DuckDB Σout</em> = bytes materialized at pipeline breakers; \
         joins now count both inputs + their own output (aligned with PeacockDB), and a TABLE_SCAN counts \
         bytes_read from storage <em>plus</em> its post-filter output (mirroring PeacockDB's split scan + filter). \
         <strong>Remaining structural skew → red rows on selective queries:</strong> PeacockDB does NOT push \
         predicates into the scan — its GpuScanExec reads ALL projected rows and a separate GpuFilterExec applies \
         the predicate (the predicate is dropped at GpuScan serialization, so the GPU path is zero row-group prune), \
         whereas DuckDB prunes + filters inline in TABLE_SCAN. So peacockdb scan output stays full-size while \
         DuckDB's is post-filter — a real, explainable efficiency gap (no scan-level pushdown), not noise. \
         (Also: group-by still counts buffered input on DuckDB vs output on PeacockDB.) The ratio is \
         <strong>directional only</strong>, to be replaced by a proper cost model; it asserts nothing and gates nothing.</p>",
    );

    s.push_str(
        "<p class=\"legend\"><strong>Mode columns</strong> come from \
         <code>testdata/cost-registry.csv</code>, which is verified against the test suite's own \
         link-time inventory (both directions) — a tick here means a test really exists. \
         <strong>✓</strong> enabled (result validated, by golden or live oracle) · \
         <strong>~</strong> skip (runs, result NOT validated) · \
         <strong>✗</strong> disabled (deliberately off — see Tickets) · \
         <strong>—</strong> n/a (mode does not apply to this query). \
         <em>full_table_cpu</em> shows both target-partition counts in one cell. The \
         <em>all_at_once</em> GPU path has no column: it executes a whole plan in one shot with no \
         per-node breakdown, so a per-query cell would be meaningless.</p>",
    );

    for d in datasets {
        let _ = write!(
            s,
            "<h2>{}</h2><table><tr><th>Query</th><th class=\"modeh\">full_table_cpu</th>\
             <th class=\"modeh\">partitioned_cpu</th><th class=\"modeh\">full_table_gpu</th>\
             <th class=\"modeh\">partitioned_gpu</th>\
             <th>PeacockDB Σout</th><th>DuckDB Σout</th><th>Ratio</th>\
             <th>Features</th><th>Tickets</th></tr>",
            d.label
        );
        for r in &d.rows {
            let stem = r.stem();
            let plan_url = r.peacockdb.and_then(|_| links.golden_url(d.canon_rel, &stem, &format!("{CPU_DEVICE}.cpu.txt")));
            let cost_url = r.peacockdb.and_then(|_| links.golden_url(d.canon_rel, &stem, &format!("{CPU_DEVICE}.cost.txt")));
            let dk_url = r.duckdb.and_then(|_| links.golden_url(d.canon_rel, &stem, "duckdb_cost.txt"));
            let _ = write!(
                s,
                "<tr class=\"{}\"><td>{}</td><td class=\"mode\">{}</td><td class=\"mode\">{}</td>\
                 <td class=\"mode\">{}</td><td class=\"mode\">{}</td><td class=\"num\">{}</td>\
                 <td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"feat\">{}</td><td>{}</td></tr>",
                r.bucket(),
                query_cell_html(&r.query, links.query_url(d.query_rel, &stem)),
                r.ftc_cell(),
                state_glyph(r.state("partitioned_cpu")),
                state_glyph(r.state("full_table_gpu")),
                state_glyph(r.state("partitioned_gpu")),
                peacock_cell_html(r.peacockdb, plan_url, cost_url),
                cost_cell_html(r.duckdb, dk_url),
                ratio_or_dash(r.ratio()),
                features_html(&r.features),
                tickets_html(&r.tickets, &links.repo),
            );
        }
        s.push_str("</table>");
    }

    let _ = write!(
        s,
        "<p class=\"foot\">Ratio = PeacockDB Σout / DuckDB Σout (directional only, see above); \
         green &le; {RATIO_GREEN_MAX}, red &gt; {RATIO_GREEN_MAX}, grey = skipped or no PeacockDB \
         size. <a href=\"{pages_url}\">{pages_url}</a></p>",
    );
    s.push_str("</body></html>");
    s
}

/// The "Full report" reference for the PR comment. Pre-deploy (PR runs) the
/// Pages site has nothing for this change, so render plain pending text instead
/// of a live link that would 404; on the deploying master run, a live link.
fn full_report_ref(published: bool, pages_url: &str) -> String {
    if published {
        format!("[Full report]({pages_url})")
    } else {
        "Full report _(published on merge to master)_".to_string()
    }
}

fn render_markdown(datasets: &[Dataset], pages_url: &str, published: bool, links: &Links, freshness: Option<&str>) -> String {
    let mut s = String::new();
    s.push_str(SENTINEL);
    s.push('\n');

    let mut summary = String::new();
    for d in datasets {
        let _ = write!(summary, "{} {}/{}", d.label, d.operational(), d.total);
        if !std::ptr::eq(d, datasets.last().unwrap()) {
            summary.push_str(", ");
        }
    }
    let _ = write!(
        s,
        "**PeacockDB GPU coverage & output-size report** — {summary} GPU-operational. {}\n",
        full_report_ref(published, pages_url)
    );
    if let Some(f) = freshness {
        let _ = write!(s, "{f}\n");
    }
    s.push('\n');

    for d in datasets {
        let _ = write!(
            s,
            "<details><summary>{} — {}/{} operational</summary>\n\n\
             | Query | ft_cpu | p_cpu | ft_gpu | p_gpu | PeacockDB Σout | DuckDB Σout | Ratio | Features | Tickets |\n\
             |---|---|---|---|---|---:|---:|---:|---|---|\n",
            d.label,
            d.operational(),
            d.total
        );
        for r in &d.rows {
            let stem = r.stem();
            let plan_url = r.peacockdb.and_then(|_| links.golden_url(d.canon_rel, &stem, &format!("{CPU_DEVICE}.cpu.txt")));
            let cost_url = r.peacockdb.and_then(|_| links.golden_url(d.canon_rel, &stem, &format!("{CPU_DEVICE}.cost.txt")));
            let dk_url = r.duckdb.and_then(|_| links.golden_url(d.canon_rel, &stem, "duckdb_cost.txt"));
            // Markdown can't set a row background, so flag the >threshold rows
            // with 🔴 — the comment-side equivalent of the HTML light-red row.
            let ratio_cell = match r.bucket() {
                "red" => format!("{} 🔴", ratio_or_dash(r.ratio())),
                _ => ratio_or_dash(r.ratio()),
            };
            let _ = write!(
                s,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                query_cell_md(&r.query, links.query_url(d.query_rel, &stem)),
                r.ftc_cell(),
                state_glyph(r.state("partitioned_cpu")),
                state_glyph(r.state("full_table_gpu")),
                state_glyph(r.state("partitioned_gpu")),
                peacock_cell_md(r.peacockdb, plan_url, cost_url),
                cost_cell_md(r.duckdb, dk_url),
                ratio_cell,
                if r.features.is_empty() { "—".to_string() } else { r.features.join(" ") },
                tickets_md(&r.tickets, &links.repo),
            );
        }
        s.push_str("\n</details>\n\n");
    }
    let _ = write!(
        s,
        "_🔴 = ratio > {RATIO_GREEN_MAX} (PeacockDB Σout over {RATIO_GREEN_MAX}× DuckDB). Σout = measured \
         output bytes; DuckDB side counts bytes materialized at pipeline breakers. The two engines emit \
         different plan trees, so the ratio is a **provisional, directional-only** proxy — not a comparable \
         execution cost — and asserts nothing._\n"
    );
    s
}

// --- page-per-sha Pages site (ticket #77) -----------------------------------
/// Assemble the page-per-sha site under `dir` (master deploy):
///   `<dir>/index.html`        latest report (this run)
///   `<dir>/<sha>/index.html`  this run, addressable by commit
///   `<dir>/history.tsv`       manifest, newest-first: `<sha>\t<generated_at>`
///   `<dir>/history.html`      rendered history index, newest-first
///
/// Prior `<sha>/` pages and `history.tsv` are pre-seeded into `dir` from the live
/// site by the CI step (the Pages deploy replaces the whole site, so carrying them
/// forward is what keeps old reports reachable). With no `sha` (local dry run) only
/// `index.html` is written.
fn assemble_site(
    dir: &Path,
    datasets: &[Dataset],
    pages_url: &str,
    links: &Links,
    generated_at: Option<&str>,
    sha: Option<&str>,
) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let root = render_html(datasets, pages_url, links, generated_at, Some(""));
    std::fs::write(dir.join("index.html"), &root).unwrap_or_else(|e| panic!("write index.html: {e}"));

    let Some(sha) = sha else { return };
    let per_sha = render_html(datasets, pages_url, links, generated_at, Some("../"));
    let sha_dir = dir.join(sha);
    std::fs::create_dir_all(&sha_dir).unwrap_or_else(|e| panic!("create {}: {e}", sha_dir.display()));
    std::fs::write(sha_dir.join("index.html"), &per_sha).unwrap_or_else(|e| panic!("write {sha}/index.html: {e}"));

    let prior = std::fs::read_to_string(dir.join("history.tsv")).unwrap_or_default();
    let manifest = update_history(&prior, sha, generated_at.unwrap_or(""));
    std::fs::write(dir.join("history.tsv"), serialize_history(&manifest)).unwrap_or_else(|e| panic!("write history.tsv: {e}"));
    std::fs::write(dir.join("history.html"), render_history(&manifest)).unwrap_or_else(|e| panic!("write history.html: {e}"));
}

/// Parse the `<sha>\t<generated_at>` manifest (newest-first), skipping blank lines.
fn parse_history(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (sha, at) = l.split_once('\t').unwrap_or((l, ""));
            (sha.trim().to_string(), at.trim().to_string())
        })
        .collect()
}

/// Prepend `(sha, at)` newest-first, dropping any prior entry for the same sha (a
/// re-run of a commit moves to the front carrying its new timestamp).
fn update_history(prior: &str, sha: &str, at: &str) -> Vec<(String, String)> {
    let mut out = vec![(sha.to_string(), at.to_string())];
    out.extend(parse_history(prior).into_iter().filter(|(s, _)| s != sha));
    out
}

fn serialize_history(manifest: &[(String, String)]) -> String {
    let mut s = manifest.iter().map(|(sha, at)| format!("{sha}\t{at}")).collect::<Vec<_>>().join("\n");
    s.push('\n');
    s
}

/// Self-contained history index: every report newest-first, linking to `<sha>/`.
fn render_history(manifest: &[(String, String)]) -> String {
    let mut s = String::new();
    s.push_str("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">");
    s.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
    s.push_str("<title>PeacockDB cost report history</title><style>");
    s.push_str(
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;color:#1b1f23;}\
         h1{font-size:1.4rem;}ul{line-height:1.7;}code{background:#f6f8fa;padding:.1rem .3rem;border-radius:4px;}\
         .foot{margin-top:1.5rem;color:#57606a;font-size:.85rem;}",
    );
    s.push_str("</style></head><body>");
    s.push_str("<h1>PeacockDB cost report — history</h1>");
    s.push_str("<p class=\"foot\"><a href=\"index.html\">latest</a></p><ul>");
    for (i, (sha, at)) in manifest.iter().enumerate() {
        let short = &sha[..sha.len().min(7)];
        let when = if at.is_empty() { String::new() } else { format!(" — {at}") };
        let latest = if i == 0 { " <em>(latest)</em>" } else { "" };
        let _ = write!(s, "<li><a href=\"{sha}/index.html\"><code>{short}</code></a>{when}{latest}</li>");
    }
    s.push_str("</ul></body></html>");
    s
}

// --- cost-regression gate (cost-diff mode) ----------------------------------
/// One query's PeacockDB CPU cost (`peacockdb_cost=` total) between base and PR.
struct DiffRow {
    label: String,
    old: u64,
    new: u64,
}

impl DiffRow {
    fn is_regression(&self) -> bool {
        self.new > self.old
    }
    fn is_improvement(&self) -> bool {
        self.new < self.old
    }
    fn changed(&self) -> bool {
        self.new != self.old
    }
    /// `(new - old) / old * 100`. `None` when `old == 0` (delta undefined — shown
    /// as "—"); classification still uses the exact integer compare above.
    fn delta_pct(&self) -> Option<f64> {
        (self.old != 0).then(|| (self.new as f64 - self.old as f64) / self.old as f64 * 100.0)
    }
}

/// Pure comparison seam (unit-testable with no git/fs): one row per query present
/// in BOTH maps, sorted by label. A query missing from `old` (no baseline — a new
/// query, or a base that predates the .cost.txt goldens) is omitted, never counted
/// as a regression — this is what stops the introducing PR from self-failing.
fn cost_diff(old: &BTreeMap<String, u64>, new: &BTreeMap<String, u64>) -> Vec<DiffRow> {
    let mut rows: Vec<DiffRow> = new
        .iter()
        .filter_map(|(label, &n)| old.get(label).map(|&o| DiffRow { label: label.clone(), old: o, new: n }))
        .collect();
    rows.sort_by(|a, b| a.label.cmp(&b.label));
    rows
}

fn fmt_delta(pct: Option<f64>) -> String {
    pct.map(|p| format!("{p:+.2}%")).unwrap_or_else(|| "—".to_string())
}

/// Display label for a `.cost.txt` golden: `<dataset>/<query>` (device segment and
/// extension dropped — each query has exactly one golden, so this stays unique).
fn diff_label(rel: &Path) -> String {
    let dataset = rel.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("?");
    let file = rel.file_name().and_then(|s| s.to_str()).unwrap_or("?");
    let query = file.split('.').next().unwrap_or(file);
    format!("{dataset}/{query}")
}

/// Link a diff-widget label (`<dataset>.sfN/q<n>`, e.g. `tpch.sf1/q1`) to its query
/// SQL at the report's commit. `None` when there's no sha, the label doesn't parse,
/// or the query isn't a numbered `q<n>` (synthetic goldens degrade to plain text).
fn diff_query_url(links: &Links, label: &str) -> Option<String> {
    let (dataset, query) = label.split_once('/')?;
    let bench = dataset.split('.').next()?; // "tpch.sf1" -> "tpch"
    let n: u32 = query.strip_prefix('q')?.parse().ok()?;
    links.query_url(&format!("testdata/{bench}-queries"), &format!("q{n}"))
}

fn diff_query_cell_html(links: &Links, label: &str) -> String {
    match diff_query_url(links, label) {
        Some(u) => format!("<a href=\"{u}\">{label}</a>"),
        None => label.to_string(),
    }
}

fn diff_query_cell_md(links: &Links, label: &str) -> String {
    match diff_query_url(links, label) {
        Some(u) => format!("[{label}]({u})"),
        None => label.to_string(),
    }
}

/// Walk the working tree's `.cost.txt` goldens. Returns `(label, working path, repo
/// path)` where the repo path (`<testdata_arg>/goldens/…`) is what `git show
/// <ref>:<repo path>` reads for the base side.
fn collect_cost_goldens(testdata: &Path) -> Vec<(String, PathBuf, String)> {
    let mut out = Vec::new();
    for sub in ["goldens/tpch.sf1", "goldens/tpcds.sf1"] {
        let dir = testdata.join(sub);
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            if !path.to_str().map(|s| s.ends_with(".cost.txt")).unwrap_or(false) {
                continue;
            }
            let rel = Path::new(sub).join(path.file_name().unwrap());
            let label = diff_label(&rel);
            let repo_path = format!("{}/{}", testdata.display(), rel.display());
            out.push((label, path, repo_path));
        }
    }
    out.sort();
    out
}

/// Base-side `.cost.txt` total. `base` is a directory (read `<base>/goldens/…`) or
/// else a git ref (`git show <base>:<repo path>`). `None` when the file is absent
/// in the base → that query has no baseline and is omitted by [`cost_diff`].
fn base_total(base: &str, repo_path: &str, testdata: &Path) -> Option<u64> {
    let base_dir = Path::new(base);
    if base_dir.is_dir() {
        // repo_path is "<testdata>/goldens/…"; strip the testdata prefix to re-root
        // it under the base dir.
        let rel = Path::new(repo_path).strip_prefix(testdata).unwrap_or(Path::new(repo_path));
        return read_total(&base_dir.join(rel), "peacockdb_cost=");
    }
    let out = std::process::Command::new("git").args(["show", &format!("{base}:{repo_path}")]).output().ok()?;
    if !out.status.success() {
        return None; // not in base → no baseline
    }
    read_total_str(&String::from_utf8_lossy(&out.stdout), "peacockdb_cost=")
}

/// Self-contained HTML artifact: only CHANGED queries, green row = improvement,
/// red row = regression (same row classes as the coverage report).
fn render_diff_html(rows: &[DiffRow], links: &Links) -> String {
    let changed: Vec<&DiffRow> = rows.iter().filter(|r| r.changed()).collect();
    let regr = changed.iter().filter(|r| r.is_regression()).count();
    let impr = changed.iter().filter(|r| r.is_improvement()).count();
    let mut s = String::new();
    s.push_str("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">");
    s.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
    s.push_str("<title>PeacockDB CPU cost change vs base</title><style>");
    s.push_str(
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;color:#1b1f23;}\
         h1{font-size:1.4rem;}table{border-collapse:collapse;max-width:680px;margin-top:.5rem;}\
         th,td{border:1px solid #d0d7de;padding:.35rem .6rem;text-align:left;font-variant-numeric:tabular-nums;}\
         th{background:#f6f8fa;}td.num{text-align:right;}\
         tr.green{background:#e9f7ee;}tr.red{background:#ffe0e0;}\
         tr.green td:first-child{border-left:4px solid #1a7f37;}tr.red td:first-child{border-left:4px solid #cf222e;}\
         .foot{margin-top:1.2rem;color:#57606a;font-size:.85rem;}",
    );
    s.push_str("</style></head><body>");
    s.push_str("<h1>PeacockDB CPU cost change vs base</h1>");
    let _ = write!(s, "<p>{impr} improvement(s), {regr} regression(s).</p>");
    if changed.is_empty() {
        s.push_str("<p>No PeacockDB CPU cost change across compared queries.</p></body></html>");
        return s;
    }
    s.push_str("<table><tr><th>Query</th><th>Base Σout</th><th>PR Σout</th><th>Δ%</th></tr>");
    for r in &changed {
        let cls = if r.is_regression() { "red" } else { "green" };
        let _ = write!(
            s,
            "<tr class=\"{cls}\"><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
            diff_query_cell_html(links, &r.label),
            fmt_bytes(r.old),
            fmt_bytes(r.new),
            fmt_delta(r.delta_pct()),
        );
    }
    s.push_str("</table><p class=\"foot\">Σout = Σ per-operator output bytes (PeacockDB CPU cost). \
        A regression (PeacockDB CPU Σout increased vs base) fails CI; exact integer compare, no tolerance. \
        New queries with no base .cost.txt are omitted.</p></body></html>");
    s
}

/// PR-comment markdown: only CHANGED queries, 🔴 regression / 🟢 improvement (GitHub
/// comments can't set a row background — same marker convention as the ratio report).
fn render_diff_markdown(rows: &[DiffRow], links: &Links) -> String {
    let changed: Vec<&DiffRow> = rows.iter().filter(|r| r.changed()).collect();
    let regr = changed.iter().filter(|r| r.is_regression()).count();
    let impr = changed.iter().filter(|r| r.is_improvement()).count();
    let mut s = String::new();
    s.push_str(DIFF_SENTINEL);
    s.push('\n');
    if changed.is_empty() {
        s.push_str("**PeacockDB CPU cost gate** — ✅ no cost change vs base across compared queries.\n");
        return s;
    }
    let verdict = if regr > 0 { " — 🔴 **build failing**" } else { " — ✅" };
    let _ = write!(s, "**PeacockDB CPU cost gate** — {impr} improvement(s), {regr} regression(s) vs base{verdict}\n\n");
    s.push_str("| Query | Base Σout | PR Σout | Δ% |\n|---|---:|---:|---:|\n");
    for r in &changed {
        let mark = if r.is_regression() { "🔴" } else { "🟢" };
        let _ = write!(
            s,
            "| {} | {} | {} | {mark} {} |\n",
            diff_query_cell_md(links, &r.label),
            fmt_bytes(r.old),
            fmt_bytes(r.new),
            fmt_delta(r.delta_pct()),
        );
    }
    let _ = write!(
        s,
        "\n_🔴 regression (PeacockDB CPU Σout increased) fails CI; 🟢 improvement. Exact integer byte-sum \
         compare, no tolerance. New queries with no base .cost.txt are omitted (never a regression)._\n"
    );
    s
}

/// Drive the gate: build the PR (working-tree) and base cost maps, render the
/// widget (always), write the artifacts, and exit non-zero iff ≥1 regression.
/// With no resolvable base (e.g. a master run), every query is omitted → 0
/// regressions → clean exit 0.
fn run_cost_diff(testdata: &Path, base: &str, html_out: &str, md_out: &str, links: &Links) {
    let goldens = collect_cost_goldens(testdata);
    let mut new_map = BTreeMap::new();
    let mut old_map = BTreeMap::new();
    for (label, path, repo_path) in &goldens {
        if let Some(n) = read_total(path, "peacockdb_cost=") {
            new_map.insert(label.clone(), n);
        }
        if !base.is_empty() {
            if let Some(o) = base_total(base, repo_path, testdata) {
                old_map.insert(label.clone(), o);
            }
        }
    }

    let rows = cost_diff(&old_map, &new_map);
    let regressions = rows.iter().filter(|r| r.is_regression()).count();

    // Render ALWAYS; the exit-code decision is separate from rendering.
    std::fs::write(html_out, render_diff_html(&rows, links)).unwrap_or_else(|e| panic!("write {html_out}: {e}"));
    eprintln!("wrote {html_out}");
    let md = render_diff_markdown(&rows, links);
    if md_out.is_empty() {
        print!("{md}");
    } else {
        std::fs::write(md_out, &md).unwrap_or_else(|e| panic!("write {md_out}: {e}"));
        eprintln!("wrote {md_out}");
    }

    let changed = rows.iter().filter(|r| r.changed()).count();
    eprintln!("cost-diff: {} compared, {changed} changed, {regressions} regression(s)", rows.len());
    if regressions > 0 {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Row for tests. `modes` maps each mode column to its state; anything
    /// unlisted defaults to "na". Keeps the tests readable now that a Row carries
    /// five mode cells plus features and tickets.
    fn test_row(query: &str, modes: &[(&str, &str)], peacockdb: Option<u64>, duckdb: Option<u64>) -> Row {
        let mut states = BTreeMap::new();
        for m in MODE_COLUMNS {
            states.insert(m.to_string(), "na".to_string());
        }
        for (k, v) in modes {
            states.insert(k.to_string(), v.to_string());
        }
        Row {
            query: query.to_string(),
            n: query.strip_prefix('q').and_then(|d| d.parse::<u32>().ok()),
            states,
            features: vec![],
            tickets: vec![],
            peacockdb,
            duckdb,
        }
    }

    // `operational_set_honors_comment_convention` is GONE with the function it
    // tested. Coverage no longer comes from parsing test_gpu.rs for uncommented
    // macro lines — it comes from testdata/cost-registry.csv, which peacockdb-core's
    // registry tests check against the suite's link-time inventory in both
    // directions. That is a strictly stronger guarantee than the comment convention:
    // the scrape could only ever see one file and could not notice a query whose
    // test had been deleted outright.

    /// The glyph mapping is the whole visual contract of the widget, so pin it.
    /// `skip` must NOT collapse into ✓ or ✗ — it means "ran, result unvalidated",
    /// which is exactly the state a reader must be able to distinguish from a
    /// verified pass.
    #[test]
    fn state_glyphs_are_distinct_and_total() {
        assert_eq!(state_glyph("enabled"), "✓");
        assert_eq!(state_glyph("skip"), "~");
        assert_eq!(state_glyph("disabled"), "✗");
        assert_eq!(state_glyph("na"), "—");
        // Anything unrecognized renders as na rather than panicking mid-report...
        assert_eq!(state_glyph("bogus"), "—");
        // ...but the four real states must all differ.
        let g: Vec<&str> = ["enabled", "skip", "disabled", "na"].iter().map(|s| state_glyph(s)).collect();
        let uniq: BTreeSet<&&str> = g.iter().collect();
        assert_eq!(uniq.len(), 4, "glyphs collide: {g:?}");
    }

    #[test]
    fn ticket_and_feature_cells_render_links_and_dashes() {
        assert_eq!(tickets_html(&[], "o/r"), "—");
        assert_eq!(features_html(&[]), "—");
        let t = tickets_html(&["103".to_string()], "asymptote-tech/peacockdb");
        assert!(t.contains("https://github.com/asymptote-tech/peacockdb/issues/103"), "{t}");
        assert!(t.contains("#103"), "{t}");
        assert!(features_html(&["stddev_var".to_string()]).contains("stddev_var"));
    }

    /// full_table_cpu is ONE column showing both target-partition counts.
    #[test]
    fn ftc_cell_shows_the_tp_split() {
        let mut states = BTreeMap::new();
        states.insert("ftc_tp1".to_string(), "enabled".to_string());
        states.insert("ftc_tp8".to_string(), "disabled".to_string());
        let r = Row {
            query: "q1".into(),
            n: Some(1),
            states,
            features: vec![],
            tickets: vec![],
            peacockdb: None,
            duckdb: None,
        };
        assert_eq!(r.ftc_cell(), "tp1✓ tp8✗");
    }

    #[test]
    fn read_total_reads_footer_and_none_when_absent() {
        // cost-report reads the explicit footer total, not the per-node breakdown.
        let cpu = "GpuScanExec: ..., output_bytes=58, output_rows=6\npeacockdb_cost=12345\n";
        assert_eq!(read_total_str(cpu, "peacockdb_cost="), Some(12345));
        let duck = "TABLE_SCAN: output_bytes=240, materialized=2640, bytes_read_est=2400\nduckdb_cost=98765\n";
        assert_eq!(read_total_str(duck, "duckdb_cost="), Some(98765));
        // Missing footer → None (grey/—), NOT Some(0) (which would be a false green).
        assert_eq!(read_total_str(cpu, "duckdb_cost="), None);
        assert_eq!(read_total_str("", "peacockdb_cost="), None);
    }

    #[test]
    fn cost_cells_link_only_when_value_and_url_present() {
        let v = Some(43_308_088u64);
        let url = Some("https://x/blob/abc/testdata/goldens/tpch.sf1/q1.tp1-mini.cpu.txt".to_string());
        assert!(cost_cell_html(v, url.clone()).starts_with("<a href="));
        assert!(cost_cell_md(v, url).starts_with("[41.30 MB]("));
        // value but no sha/url → plain text, no link.
        assert_eq!(cost_cell_html(v, None), "41.30 MB");
        assert_eq!(cost_cell_md(v, None), "41.30 MB");
        // missing value → em-dash, never a link.
        assert_eq!(cost_cell_html(None, Some("u".into())), "—");
        assert_eq!(cost_cell_md(None, Some("u".into())), "—");
    }

    #[test]
    fn full_report_ref_pending_on_pr_live_on_master() {
        let url = "https://asymptote-tech.github.io/peacockdb/";
        let pr = full_report_ref(false, url);
        assert!(!pr.contains(url));
        assert!(pr.to_lowercase().contains("master"));
        assert_eq!(full_report_ref(true, url), format!("[Full report]({url})"));
    }

    #[test]
    fn freshness_line_needs_both_sha_and_time() {
        assert_eq!(
            freshness_line(Some("d9bc04f1abc"), Some("2026-06-16 18:38 UTC")),
            Some("♻️ _Cost report regenerated for `d9bc04f` at 2026-06-16 18:38 UTC_".to_string())
        );
        assert_eq!(freshness_line(None, Some("t")), None);
        assert_eq!(freshness_line(Some("abc"), None), None);
        // sha shorter than 7 chars must not panic.
        assert_eq!(freshness_line(Some("abc"), Some("t")), Some("♻️ _Cost report regenerated for `abc` at t_".to_string()));
    }

    #[test]
    fn bucket_threshold_is_1_4() {
        let row = |p: u64, d: u64, op: bool| test_row(
            "q1",
            &[("full_table_gpu", if op { "enabled" } else { "na" })],
            Some(p),
            Some(d),
        );
        assert_eq!(row(14, 10, true).bucket(), "green"); // ratio 1.4 → green (≤)
        assert_eq!(row(141, 100, true).bucket(), "red"); // ratio 1.41 → red
        assert_eq!(row(14, 10, false).bucket(), "grey"); // not operational → grey
    }

    #[test]
    fn peacock_cell_renders_plan_and_cost_links() {
        let plan = Some("https://x/q1.tp8-mini.cpu.txt".to_string());
        let cost = Some("https://x/q1.tp8-mini.cost.txt".to_string());
        let html = peacock_cell_html(Some(43_308_088), plan.clone(), cost.clone());
        assert!(html.contains(">plan</a>") && html.contains(">cost</a>") && html.starts_with("41.30 MB ("));
        let md = peacock_cell_md(Some(43_308_088), plan, cost);
        assert_eq!(md, "41.30 MB ([plan](https://x/q1.tp8-mini.cpu.txt), [cost](https://x/q1.tp8-mini.cost.txt))");
        // value but no urls (dry run) → plain bytes, no links.
        assert_eq!(peacock_cell_html(Some(43_308_088), None, None), "41.30 MB");
        assert_eq!(peacock_cell_md(Some(43_308_088), None, None), "41.30 MB");
        // missing value → em-dash regardless of urls.
        assert_eq!(peacock_cell_html(None, Some("u".into()), Some("v".into())), "—");
        assert_eq!(peacock_cell_md(None, Some("u".into()), Some("v".into())), "—");
    }

    #[test]
    fn history_prepends_newest_first_and_dedups_sha() {
        let prior = "old1\t2026-06-20 10:00\nold2\t2026-06-19 09:00\n";
        let m = update_history(prior, "new", "2026-06-23 12:00");
        assert_eq!(m[0], ("new".to_string(), "2026-06-23 12:00".to_string()));
        assert_eq!(m.len(), 3);
        // re-running an existing sha moves it to the front with the new timestamp,
        // never duplicating it.
        let m2 = update_history(&serialize_history(&m), "old1", "2026-06-24 08:00");
        assert_eq!(m2[0], ("old1".to_string(), "2026-06-24 08:00".to_string()));
        assert_eq!(m2.iter().filter(|(s, _)| s == "old1").count(), 1);
        assert_eq!(m2.len(), 3);
        // parse/serialize roundtrip is stable.
        assert_eq!(parse_history(&serialize_history(&m2)), m2);
    }

    fn map(pairs: &[(&str, u64)]) -> BTreeMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn cost_diff_classifies_and_omits_no_baseline() {
        let old = map(&[("a", 100), ("b", 100), ("c", 100)]);
        // a improves, b regresses, c unchanged, d is brand-new (no baseline).
        let new = map(&[("a", 80), ("b", 120), ("c", 100), ("d", 50)]);
        let rows = cost_diff(&old, &new);
        assert_eq!(rows.len(), 3); // d omitted — no base entry
        assert!(rows.iter().all(|r| r.label != "d"));
        let a = rows.iter().find(|r| r.label == "a").unwrap();
        assert!(a.is_improvement() && a.changed() && !a.is_regression());
        assert_eq!(a.delta_pct(), Some(-20.0));
        let b = rows.iter().find(|r| r.label == "b").unwrap();
        assert!(b.is_regression() && b.changed());
        assert_eq!(b.delta_pct(), Some(20.0));
        let c = rows.iter().find(|r| r.label == "c").unwrap();
        assert!(!c.changed() && !c.is_regression() && !c.is_improvement());
    }

    #[test]
    fn regression_count_drives_exit_decision() {
        // Mixed (1 improvement, 1 regression, 1 unchanged) → exactly 1 regression.
        let rows = cost_diff(&map(&[("a", 100), ("b", 100), ("c", 100)]), &map(&[("a", 90), ("b", 110), ("c", 100)]));
        assert_eq!(rows.iter().filter(|r| r.is_regression()).count(), 1); // → exit 1
        // Pure improvement → 0 regressions (→ exit 0).
        let rows = cost_diff(&map(&[("a", 100)]), &map(&[("a", 50)]));
        assert_eq!(rows.iter().filter(|r| r.is_regression()).count(), 0);
        // No baseline at all (master / pre-task-1 base) → 0 comparable, 0 regressions.
        let rows = cost_diff(&map(&[]), &map(&[("a", 50), ("b", 99)]));
        assert!(rows.is_empty());
    }

    /// A sha-less `Links` (dry run): every URL helper returns `None`, so labels /
    /// cells render plain — keeps the label-substring assertions below unambiguous.
    fn no_links() -> Links {
        Links { repo: "o/r".into(), sha: None }
    }

    #[test]
    fn diff_markdown_marks_regressions_and_omits_unchanged() {
        let rows = cost_diff(&map(&[("a", 100), ("b", 100), ("c", 100)]), &map(&[("a", 80), ("b", 120), ("c", 100)]));
        let md = render_diff_markdown(&rows, &no_links());
        assert!(md.starts_with(DIFF_SENTINEL));
        assert!(md.contains("1 improvement(s), 1 regression(s)") && md.contains("build failing"));
        assert!(md.contains("| a |") && md.contains("🟢") && md.contains("-20.00%"));
        assert!(md.contains("| b |") && md.contains("🔴") && md.contains("+20.00%"));
        assert!(!md.contains("| c |")); // unchanged omitted from the widget
        // No-change case still upserts a benign comment (clears a prior regression).
        let clean = render_diff_markdown(&cost_diff(&map(&[("a", 100)]), &map(&[("a", 100)])), &no_links());
        assert!(clean.starts_with(DIFF_SENTINEL) && clean.contains("no cost change"));
    }

    #[test]
    fn delta_pct_guards_zero_base() {
        let rows = cost_diff(&map(&[("a", 0)]), &map(&[("a", 5)]));
        assert_eq!(rows[0].delta_pct(), None); // undefined, shown as "—"
        assert!(rows[0].is_regression()); // classification still works (0 → 5)
        assert_eq!(fmt_delta(None), "—");
    }

    #[test]
    fn query_url_links_only_with_sha() {
        let links = Links { repo: "o/r".into(), sha: Some("abc123".into()) };
        assert_eq!(
            links.query_url("testdata/tpch-queries", "q6"),
            Some("https://github.com/o/r/blob/abc123/testdata/tpch-queries/q6.sql".to_string())
        );
        assert_eq!(
            links.query_url("testdata/tpcds-queries", "q14"),
            Some("https://github.com/o/r/blob/abc123/testdata/tpcds-queries/q14.sql".to_string())
        );
        // Synthetic micro-queries link the same way, via their hyphenated stem.
        assert_eq!(
            links.query_url("testdata/tpch-queries", "scan-limit"),
            Some("https://github.com/o/r/blob/abc123/testdata/tpch-queries/scan-limit.sql".to_string())
        );
        // no sha (dry run) → no link.
        assert_eq!(no_links().query_url("testdata/tpch-queries", "q1"), None);
    }

    fn one_row_dataset() -> Dataset {
        Dataset {
            label: "TPC-H",
            total: 1,
            canon_rel: "testdata/goldens/tpch.sf1",
            query_rel: "testdata/tpch-queries",
            rows: vec![test_row("q1", &[("full_table_gpu", "enabled")], Some(100), Some(100))],
        }
    }

    #[test]
    fn query_cell_linked_with_sha_plain_without() {
        let linked = Links { repo: "o/r".into(), sha: Some("deadbeef".into()) };
        let url = "https://github.com/o/r/blob/deadbeef/testdata/tpch-queries/q1.sql";

        let html = render_html(&[one_row_dataset()], "https://p/", &linked, None, None);
        assert!(html.contains(&format!("<a href=\"{url}\">q1</a>")));
        let md = render_markdown(&[one_row_dataset()], "https://p/", false, &linked, None);
        assert!(md.contains(&format!("[q1]({url})")));

        // No sha → plain q1 cell, no query link.
        let html_plain = render_html(&[one_row_dataset()], "https://p/", &no_links(), None, None);
        assert!(html_plain.contains("<td>q1</td>") && !html_plain.contains("q1.sql"));
        let md_plain = render_markdown(&[one_row_dataset()], "https://p/", false, &no_links(), None);
        assert!(md_plain.contains("| q1 |") && !md_plain.contains("q1.sql"));
    }

    #[test]
    fn diff_query_url_parses_dataset_and_qn() {
        let links = Links { repo: "o/r".into(), sha: Some("cafe".into()) };
        assert_eq!(
            diff_query_url(&links, "tpch.sf1/q1"),
            Some("https://github.com/o/r/blob/cafe/testdata/tpch-queries/q1.sql".to_string())
        );
        assert_eq!(
            diff_query_url(&links, "tpcds.sf1/q14"),
            Some("https://github.com/o/r/blob/cafe/testdata/tpcds-queries/q14.sql".to_string())
        );
        // Synthetic (non-qN) golden → no link.
        assert_eq!(diff_query_url(&links, "tpch.sf1/scan_limit"), None);
        // No sha → no link even for a well-formed label.
        assert_eq!(diff_query_url(&no_links(), "tpch.sf1/q1"), None);
    }

    #[test]
    fn diff_widget_links_labels_when_sha_present() {
        let links = Links { repo: "o/r".into(), sha: Some("cafe".into()) };
        let rows = cost_diff(&map(&[("tpch.sf1/q1", 100)]), &map(&[("tpch.sf1/q1", 120)]));
        let url = "https://github.com/o/r/blob/cafe/testdata/tpch-queries/q1.sql";

        let md = render_diff_markdown(&rows, &links);
        assert!(md.contains(&format!("[tpch.sf1/q1]({url})")));
        let html = render_diff_html(&rows, &links);
        assert!(html.contains(&format!("<a href=\"{url}\">tpch.sf1/q1</a>")));

        // No sha → plain label, no link.
        let md_plain = render_diff_markdown(&rows, &no_links());
        assert!(md_plain.contains("| tpch.sf1/q1 |") && !md_plain.contains("q1.sql"));
    }

    #[test]
    fn render_history_lists_newest_first_with_links() {
        let m = vec![
            ("abc1234def".to_string(), "2026-06-23 12:00".to_string()),
            ("0009999aaa".to_string(), "2026-06-20 10:00".to_string()),
        ];
        let html = render_history(&m);
        assert!(html.contains("href=\"abc1234def/index.html\""));
        assert!(html.contains("<code>abc1234</code>")); // shortened
        assert!(html.contains("(latest)")); // first entry flagged
        // newest entry appears before the older one.
        assert!(html.find("abc1234def").unwrap() < html.find("0009999aaa").unwrap());
    }
}
