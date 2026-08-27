//! GPU-coverage & output-size report generator.
//!
//! Reads only committed goldens, so it runs in the CI CPU tier with no GPU and
//! no executor build:
//!   - PeacockDB Σout = Σ `output_bytes` over a query's `<q>.cpu.txt` cost tree
//!                      (every CPU operator's output size).
//!   - DuckDB Σout    = pipeline-breaker materialized bytes computed from the
//!                      `<q>.duckdb_cost.txt` profiling tree (see [`duckdb_cost`]).
//!   - GPU coverage   = whether the query's GPU result test is enabled in
//!                      `test_gpu_full_table.rs` / `test_gpu_partitioned.rs`
//!                      (uncommented macro invocation).
//!
//! Both sides are deterministic, measured byte sums — NOT wall-clock cost, and
//! the two engines emit different plan trees, so the ratio is a provisional,
//! directional-only proxy (the report displays it, asserts nothing).
//!
//! Emits a self-contained HTML page (inline CSS, for GitHub Pages) and a compact
//! Markdown blob (for the upserted PR comment, keyed on [`SENTINEL`]).
//!
//! Usage:
//!   cost-report [--testdata DIR] [--registry FILE] [--html FILE] [--md FILE]
//!               [--site DIR] [--pages-url URL] [--sha SHA] [--repo OWNER/REPO]
//!               [--generated-at TS] [--published]
//!               [--cost-diff --base REF|DIR]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// PeacockDB/DuckDB Σout ratio at or below this renders green; above gets the
/// light-red row highlight (🔴 in the markdown comment). Single configurable
/// threshold for both renderers — directional only; revisit as the models converge.
const RATIO_GREEN_MAX: f64 = 1.4;

const PAGES_URL_DEFAULT: &str = "https://asymptote-tech.github.io/peacockdb/";
const DEFAULT_REPO: &str = "asymptote-tech/peacockdb";
/// Golden label of the CPU-cost goldens (full-table execution, 8 partitions /
/// 2 GiB), the `<mode>-<tp>-<tier>` component of the `.cpu.txt` filename. MUST
/// track the mode + device the `cpu_full_table_result_test!` goldens are canonized
/// at — a stale label here makes every PeacockDB cell render "—" (guarded in `main`).
const CPU_DEVICE: &str = "full_table-tp8-mini";
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
/// The CSV is verified against the test suite's own link-time inventory by the
/// registry tests in peacockdb-core (both directions), so a stale cell fails the
/// build rather than silently mis-rendering a tick here.
struct Registry {
    rows: Vec<RegistryRow>,
}

struct RegistryRow {
    dataset: String,
    query: String,
    states: BTreeMap<String, String>,
    /// "ok" | "fail" — whether `create_physical_plan` succeeds for this query.
    plan_status: String,
    features: Vec<String>,
    tickets: Vec<String>,
}

/// How a row renders its four execution-mode cells.
///
/// The two non-executable categories collapse those four columns into ONE spanning
/// cell: a query that cannot execute has nothing to say per-mode, and four identical
/// em-dashes invite the reader to hunt for a distinction that isn't there.
#[derive(PartialEq, Debug, Clone, Copy)]
enum RowKind {
    /// Plans AND has at least one execution mode enabled — four normal mode cells.
    Executable,
    /// Plans, but no execution mode is enabled yet.
    PlanOnly,
    /// `create_physical_plan` itself fails (see the plan-attempt probe).
    PlanFailed,
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
        let (i_ds, i_q, i_feat, i_tick, i_ps) =
            (idx("dataset"), idx("query"), idx("features"), idx("tickets"), idx("plan_status"));
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
                plan_status: f[i_ps].to_string(),
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

/// The CPU columns whose `✓` links to a golden, and the single `<mode>-<tp>-<tier>`
/// label each links at.
///
/// ONE label per column, so the link target is predictable from the column alone.
/// `scan_limit` is the query that makes this worth stating: it is registered at BOTH
/// tp1_mini and tp1_standard and `column_for` keys on the tp count rather than the
/// memory tier, so its single `ftc_tp1` cell aggregates two runs and it owns two
/// goldens. It links at tp1-standard like every other tp1 row; a hyperlink cannot
/// express "two runs" and inventing a second candidate for it would add config that
/// nothing else can reach.
///
/// The existence check in `build_dataset` plus the gate in `main` is what makes one
/// label safe: a query registered ONLY at some other tier has no golden here and
/// fails by name, rather than rendering a dead link.
///
/// Deliberately NOT derived from `CPU_DEVICE`: that const is the Σout cells' golden,
/// and folding the two together would let a Σout retarget silently move these links.
const CPU_GOLDEN_LABEL: [(&str, &str); 3] = [
    ("ftc_tp1", "full_table-tp1-standard"),
    ("ftc_tp8", "full_table-tp8-mini"),
    ("partitioned_cpu", "partitioned-tp8-standard"),
];

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
    plan_status: String,
    features: Vec<String>,
    tickets: Vec<String>,
    peacockdb: Option<u64>,
    duckdb: Option<u64>,
    /// CPU column -> the `<mode>-<tp>-<tier>` golden label ([`CPU_GOLDEN_LABEL`])
    /// whose `.cpu.txt` exists on disk for this query, for the ✓ links. Absent for
    /// non-`enabled` cells, which never link, and for an `enabled` cell whose golden
    /// is missing — which `main` then reports rather than rendering unlinked.
    cpu_golden: BTreeMap<String, String>,
}

impl Row {
    /// Golden/SQL file stem: the CSV's underscore form maps to hyphenated paths.
    fn stem(&self) -> String {
        self.query.replace('_', "-")
    }

    fn state(&self, col: &str) -> &str {
        self.states.get(col).map(String::as_str).unwrap_or("na")
    }

    /// Which of the three row shapes this is. Plan failure dominates: a query that
    /// cannot be planned cannot have a meaningful execution mode, so it is reported
    /// as PlanFailed even if the registry somehow also marked a mode enabled (that
    /// combination would itself be a bug worth seeing as "plan ✗").
    fn kind(&self) -> RowKind {
        if self.plan_status == "fail" {
            return RowKind::PlanFailed;
        }
        let any_enabled = MODE_COLUMNS.iter().any(|c| self.state(c) == "enabled");
        if any_enabled { RowKind::Executable } else { RowKind::PlanOnly }
    }

    /// "Operational" for the summary line = the single-partition GPU mode is
    /// enabled.
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

    /// One CPU mode's glyph, hyperlinked to the `.cpu.txt` it was verified against
    /// when the cell is `enabled` and a URL exists.
    ///
    /// ONLY `✓` links: `~`, `✗` and `—` have no golden behind them, so a link would
    /// promise a file that is not there. Falls back to the bare glyph on a dry run
    /// (no sha ⇒ `golden_url` is `None`), exactly like the Σout cells.
    fn cpu_glyph(&self, col: &str, links: &Links, canon_rel: &str) -> String {
        let glyph = state_glyph(self.state(col));
        let url = self
            .cpu_golden
            .get(col)
            .and_then(|label| links.golden_url(canon_rel, &self.stem(), &format!("{label}.cpu.txt")));
        match url {
            Some(u) => format!("<a href=\"{u}\">{glyph}</a>"),
            None => glyph.to_string(),
        }
    }

    /// full_table_cpu is one logical mode run at two target-partition counts, so it
    /// renders as a single cell showing the split rather than two columns.
    ///
    /// Returns MARKUP, not a label: the two glyphs link to different goldens
    /// (tp1-standard/tp1-mini vs tp8-mini), so they cannot be one escaped string.
    /// Both renders emit HTML here — the PR comment's table is raw HTML too.
    fn ftc_cell(&self, links: &Links, canon_rel: &str) -> String {
        format!(
            "tp1{} tp8{}",
            self.cpu_glyph("ftc_tp1", links, canon_rel),
            self.cpu_glyph("ftc_tp8", links, canon_rel)
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
    tickets: TicketIndex,
}

/// Which wiki file each ticket number's anchor lives in. `tickets.md` holds open work,
/// `tasks/bp-tickets.md` the rollout's, and `archive/archived-tickets.md` the rest — and a
/// closed ticket keeps its registry cell, since a `na` should say which decision it rests
/// on, so the link has to follow the number rather than assume the file. The id space is
/// shared across all three, so a number is never two things.
#[derive(Debug, Default, Clone)]
struct TicketIndex {
    open: BTreeSet<String>,
    rollout: BTreeSet<String>,
    archived: BTreeSet<String>,
}

/// The number an anchor line declares, or `None` for a line that has none — including the
/// two pages' own prose about the convention, which spells the anchor with `tNN`.
fn anchor_number(line: &str) -> Option<String> {
    let (_, rest) = line.split_once("<a id=\"t")?;
    let number: String = rest.chars().take_while(char::is_ascii_digit).collect();
    (!number.is_empty()).then_some(number)
}

/// The ticket numbers a wiki file anchors, each checked against the header it sits above.
/// Placement is the half existence cannot see: a header inserted above an existing anchor
/// takes that anchor's link, and both directions of an existence check still pass, since
/// every header has an anchor and every anchor a header. Stacked anchors are skipped past
/// rather than refused outright, so the one they resolve to is judged on the header it
/// actually reaches.
fn anchored_numbers(text: &str) -> Result<BTreeSet<String>, String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut found = BTreeSet::new();
    for (at, line) in lines.iter().enumerate() {
        let Some(number) = anchor_number(line) else {
            continue;
        };
        let header = lines[at + 1..]
            .iter()
            .find(|next| !next.trim().is_empty() && anchor_number(next).is_none());
        match header {
            Some(header) if names_ticket(header, &number) => {
                found.insert(number);
            }
            Some(header) => {
                return Err(format!(
                    "<a id=\"t{number}\"> resolves to {}, which is not #{number}",
                    header.trim()
                ));
            }
            None => return Err(format!("<a id=\"t{number}\"> has no header below it")),
        }
    }
    Ok(found)
}

/// Whether a header line is this ticket's own. The trailing digit check is what keeps
/// `### #17` from answering for `t170`.
fn names_ticket(header: &str, number: &str) -> bool {
    header
        .trim_start()
        .strip_prefix("### #")
        .and_then(|rest| rest.strip_prefix(number))
        .is_some_and(|tail| !tail.starts_with(|c: char| c.is_ascii_digit()))
}

impl TicketIndex {
    /// All three files, by their anchors: every ticket carries `<a id="tNN">` wherever it
    /// lives, which is the same thing the links point at, and each above its own header.
    fn load(wiki: &Path) -> Self {
        let numbers = |path: PathBuf| -> BTreeSet<String> {
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                eprintln!("cost-report: cannot read {}: {e}", path.display());
                std::process::exit(1);
            });
            anchored_numbers(&text).unwrap_or_else(|misplaced| {
                eprintln!("cost-report: {}: {misplaced}", path.display());
                std::process::exit(1);
            })
        };
        Self {
            open: numbers(wiki.join("tickets.md")),
            rollout: numbers(wiki.join("tasks/bp-tickets.md")),
            archived: numbers(wiki.join("archive/archived-tickets.md")),
        }
    }

    /// The repo-relative file a number resolves to, or `None` where it is in neither —
    /// which is a link to nowhere and is refused rather than rendered.
    fn path_for(&self, ticket: &str) -> Option<&'static str> {
        if self.open.contains(ticket) {
            Some("llm-wiki/tickets.md")
        } else if self.rollout.contains(ticket) {
            Some("llm-wiki/tasks/bp-tickets.md")
        } else if self.archived.contains(ticket) {
            Some("llm-wiki/archive/archived-tickets.md")
        } else {
            None
        }
    }
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
    // for out-of-tree runs.
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
    // The wiki sits beside the testdata tree, so an out-of-tree run finds both or neither.
    let wiki = testdata
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("llm-wiki");
    let links = Links {
        repo,
        sha,
        tickets: TicketIndex::load(&wiki),
    };

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

    // Coverage comes from the committed registry CSV, verified against the test
    // suite's link-time inventory by peacockdb-core's registry tests.
    let registry = Registry::load(&registry_csv);
    // A ticket a row names must resolve to a file that has its anchor. Rendering a link
    // to nowhere is the silent half of the failure the archive's own header warns about,
    // so it is checked once here rather than discovered by a reader clicking it.
    let unresolved: BTreeSet<&String> = registry
        .rows
        .iter()
        .flat_map(|row| row.tickets.iter())
        .filter(|ticket| links.tickets.path_for(ticket).is_none())
        .collect();
    if !unresolved.is_empty() {
        eprintln!(
            "cost-report: {} ticket(s) named in the registry are in neither \
             llm-wiki/tickets.md, llm-wiki/tasks/bp-tickets.md nor \
             llm-wiki/archive/archived-tickets.md: {}",
            unresolved.len(),
            unresolved.into_iter().cloned().collect::<Vec<_>>().join(", ")
        );
        std::process::exit(1);
    }

    let tpch = build_dataset("TPC-H", "testdata/goldens/tpch.sf1", "testdata/tpch-queries", &testdata.join("goldens/tpch.sf1"), &registry, "tpch");
    let tpcds = build_dataset("TPC-DS", "testdata/goldens/tpcds.sf1", "testdata/tpcds-queries", &testdata.join("goldens/tpcds.sf1"), &registry, "tpcds");
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
    // tp1-mini only, so no full_table-tp8-mini cost golden exists or should. Its Σout
    // cell is a dash, and the ftc column shows why.
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

    // Every `enabled` CPU cell must have a .cpu.txt: assert_cpu_cost_canonical runs on
    // EVERY CPU macro invocation, independent of the ResultGolden keyword (which gates
    // only .result.txt). That invariant is the whole premise of these links, so a cell
    // that resolves to nothing is a real breakage — a new device label, or a golden
    // renamed without updating CPU_GOLDEN_LABEL — and must be loud rather than a
    // silently unlinked ✓.
    let mut unlinked: Vec<String> = Vec::new();
    for d in &datasets {
        for r in &d.rows {
            for (col, label) in CPU_GOLDEN_LABEL {
                if r.state(col) == "enabled" && !r.cpu_golden.contains_key(col) {
                    unlinked.push(format!("{} {} [{col}] (expected {label})", d.label, r.query));
                }
            }
        }
    }
    if !unlinked.is_empty() {
        eprintln!(
            "cost-report: {} enabled CPU cell(s) have no .cpu.txt golden under any known \
             label. Every enabled CPU cell owns one (assert_cpu_cost_canonical is \
             unconditional), so this means a query registered at a tier CPU_GOLDEN_LABEL \
             does not name, or a golden rename it has not caught up with:\n  {}",
            unlinked.len(),
            unlinked.join("\n  ")
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
    canon_rel: &'static str,
    query_rel: &'static str,
    canon: &Path,
    registry: &Registry,
    dataset_key: &str,
) -> Dataset {
    // Each golden carries its own explicit total footer (peacockdb_cost= /
    // duckdb_cost=), the single source of truth for that side's number; the
    // per-node output_bytes/materialized values above it are the contribution
    // breakdown that sums to it. We read only the footer (`read_total`).
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
                plan_status: r.plan_status.clone(),
                features: r.features.clone(),
                tickets: r.tickets.clone(),
                // PeacockDB total lives in the cheap-to-regenerate .cost.txt (the
                // .cpu.txt carries no footer); `peacockdb_cost=` key.
                peacockdb: read_total(
                    &canon.join(format!("{stem}.{CPU_DEVICE}.cost.txt")),
                    "peacockdb_cost=",
                ),
                duckdb: read_total(&canon.join(format!("{stem}.duckdb_cost.txt")), "duckdb_cost="),
                // Check the golden exists rather than assuming it: see
                // CPU_GOLDEN_LABEL. Only `enabled` cells are resolved — the others
                // never render a link, and checking them would make the main() gate
                // below fire on cells that legitimately have no golden.
                cpu_golden: CPU_GOLDEN_LABEL
                    .iter()
                    .filter(|(col, _)| r.states.get(*col).map(String::as_str) == Some("enabled"))
                    .filter(|(_, label)| canon.join(format!("{stem}.{label}.cpu.txt")).exists())
                    .map(|(col, label)| (col.to_string(), label.to_string()))
                    .collect(),
            }
        })
        .collect();
    rows.sort_by(|a, b| match (a.n, b.n) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.query.cmp(&b.query),
    });
    // `total` is the number of registry rows for this dataset, NOT a hardcoded
    // 22/99: the registry includes the synthetic micro-queries, so the denominator
    // must describe the table, not the numbered-query count.
    let total = rows.len();
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
        (Some(v), Some(u)) => format!("<a href=\"{u}\">{}</a>", fmt_bytes(v)),
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
    // HTML anchors, not markdown links — this cell lands inside a raw <td>.
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

/// The four execution-mode `<td>`s for a row — or ONE `colspan=4` cell when the
/// query has no per-mode story to tell.
///
/// Executable rows get the full cells (full_table_cpu split + 3 glyphs). The other
/// two kinds merge, because four repeated em-dashes read as "look for the
/// difference" when the real statement is a single fact about the whole row: it
/// plans but nothing runs it yet, or it does not plan at all.
fn mode_cells_html(r: &Row, links: &Links, canon_rel: &str) -> String {
    match r.kind() {
        RowKind::Executable => format!(
            "<td class=\"mode\">{}</td><td class=\"mode\">{}</td>\
             <td class=\"mode\">{}</td><td class=\"mode\">{}</td>",
            r.ftc_cell(links, canon_rel),
            r.cpu_glyph("partitioned_cpu", links, canon_rel),
            state_glyph(r.state("full_table_gpu")),
            state_glyph(r.state("partitioned_gpu")),
        ),
        RowKind::PlanOnly => {
            "<td class=\"mode span\" colspan=\"4\">plan ✓</td>".to_string()
        }
        RowKind::PlanFailed => {
            "<td class=\"mode span\" colspan=\"4\">plan ✗</td>".to_string()
        }
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

/// Ticket links into the wiki (GitHub issues are retired; the wiki is the registry). Each
/// ticket carries an `<a id="tNN">` anchor wherever it lives, so `#tNN` is stable even when
/// a title is reworded — and which file it lives in comes from [`TicketIndex`], since a
/// closed ticket keeps its registry cell and moves to the archive. Bare numbers in the CSV;
/// `--repo` keeps forks linking to their own copy.
fn ticket_link(t: &str, links: &Links) -> String {
    let path = links.tickets.path_for(t).unwrap_or_else(|| {
        // Unreachable after the startup gate; if it ever is reached, a dead link is the
        // one outcome worse than no report.
        eprintln!("cost-report: ticket #{t} is in none of the three ticket files");
        std::process::exit(1);
    });
    format!(
        "<a href=\"https://github.com/{}/blob/master/{path}#t{t}\">#{t}</a>",
        links.repo
    )
}

fn tickets_html(tickets: &[String], links: &Links) -> String {
    if tickets.is_empty() {
        return "—".to_string();
    }
    tickets.iter().map(|t| ticket_link(t, links)).collect::<Vec<_>>().join(" ")
}

fn tickets_md(tickets: &[String], links: &Links) -> String {
    if tickets.is_empty() {
        return "—".to_string();
    }
    tickets.iter().map(|t| ticket_link(t, links)).collect::<Vec<_>>().join(" ")
}

/// Markdown counterpart of [`mode_cells_html`]. The PR comment's table is raw HTML
/// (GFM pipe tables support neither `colspan` nor font control), so the same shapes
/// apply: four `<td>` mode cells for an Executable row, one `colspan=4`
/// plan-status cell otherwise. `<sub>` shrinks text — GitHub strips class/style.
fn mode_cells_md(r: &Row, links: &Links, canon_rel: &str) -> String {
    match r.kind() {
        RowKind::Executable => format!(
            "<td><sub>{}</sub></td><td>{}</td><td>{}</td><td>{}</td>",
            r.ftc_cell(links, canon_rel),
            r.cpu_glyph("partitioned_cpu", links, canon_rel),
            state_glyph(r.state("full_table_gpu")),
            state_glyph(r.state("partitioned_gpu")),
        ),
        RowKind::PlanOnly => {
            "<td colspan=\"4\"><sub>plan ✓</sub></td>".to_string()
        }
        RowKind::PlanFailed => {
            "<td colspan=\"4\"><sub>plan ✗</sub></td>".to_string()
        }
    }
}

/// NOTE: emits an HTML anchor, not markdown `[text](url)`. The comment's table is
/// raw HTML (needed for colspan + <sub>), and GitHub does NOT process markdown link
/// syntax inside raw HTML block elements — it would render the brackets literally.
fn query_cell_md(name: &str, url: Option<String>) -> String {
    match url {
        Some(u) => format!("<a href=\"{u}\">{name}</a>"),
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
         /* The synthetic micro-queries (aggregate_groupby, shuffle_stddev, …) are \
            long unbreakable tokens, unlike the q<N> names, so they alone set the \
            Query column's min-content width. Shrinking just those keeps the numbered \
            rows at full size. Row::number is None for exactly this set. */\
         td.micro{font-size:.72rem;}\
         /* Both Sigma-out columns, header and values: byte counts are the widest \
            numeric cells and are reference data, not the headline. */\
         th.sigma,td.sigma{font-size:.72rem;}\
         /* merged plan-status cell spanning the 4 mode columns */\
         td.span{font-size:.75rem;color:#57606a;font-style:italic;}\
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
         the predicate (the predicate is dropped at CudfScan serialization, so the GPU path is zero row-group prune), \
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
         Mode columns are abbreviated to keep the table within one screen: \
         <em>ft_cpu</em> = full_table_cpu, <em>p_cpu</em> = partitioned_cpu, \
         <em>ft_gpu</em> = full_table_gpu, <em>p_gpu</em> = partitioned_gpu. \
         <em>ft_cpu</em> shows both target-partition counts in one cell. The \
         <em>all_at_once</em> GPU path has no column: it executes a whole plan in one shot with no \
         per-node breakdown, so a per-query cell would be meaningless.</p>",
    );

    for d in datasets {
        let _ = write!(
            s,
            // Short header labels (ft_cpu/p_cpu/ft_gpu/p_gpu) are DISPLAY TEXT ONLY —
            // CSV column names and all code paths keep the long forms. These four
            // headers set their columns' min-content width (the cells below are one
            // glyph), so the long names put the table back into horizontal scroll.
            "<h2>{}</h2><table><tr><th>Query</th><th class=\"modeh\">ft_cpu</th>\
             <th class=\"modeh\">p_cpu</th><th class=\"modeh\">ft_gpu</th>\
             <th class=\"modeh\">p_gpu</th>\
             <th class=\"sigma\">PeacockDB Σout</th><th class=\"sigma\">DuckDB Σout</th><th>Ratio</th>\
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
                "<tr class=\"{}\"><td{}>{}</td>{}<td class=\"num sigma\">{}</td>\
                 <td class=\"num sigma\">{}</td><td class=\"num\">{}</td><td class=\"feat\">{}</td><td>{}</td></tr>",
                r.bucket(),
                // Attribute omitted entirely for q<N> rows — an empty class="" is
                // noise in the output and is what the plain-cell test asserts against.
                if r.n.is_none() { " class=\"micro\"" } else { "" },
                query_cell_html(&r.query, links.query_url(d.query_rel, &stem)),
                mode_cells_html(r, links, d.canon_rel),
                peacock_cell_html(r.peacockdb, plan_url, cost_url),
                cost_cell_html(r.duckdb, dk_url),
                ratio_or_dash(r.ratio()),
                features_html(&r.features),
                tickets_html(&r.tickets, links),
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
            // A raw HTML table, NOT a GFM pipe table. Two things the widget's
            // structure needs are impossible in pipe tables: colspan (for the merged
            // plan-status cell) and any font control. GitHub renders inline HTML in
            // comments, and <sub> shrinks text — `class`/`style` are stripped by its
            // sanitizer, so <sub> is the mechanism, not CSS.
            "<details><summary>{} — {}/{} operational</summary>\n\n\
             <table>\n<tr><th>Query</th><th><sub>ft_cpu</sub></th>\
             <th><sub>p_cpu</sub></th><th><sub>ft_gpu</sub></th>\
             <th><sub>p_gpu</sub></th><th><sub>PeacockDB Σout</sub></th>\
             <th><sub>DuckDB Σout</sub></th><th>Ratio</th><th><sub>Features</sub></th>\
             <th>Tickets</th></tr>\n",
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
                "<tr><td>{}</td>{}<td><sub>{}</sub></td>\
                 <td><sub>{}</sub></td><td>{}</td>\
                 <td><sub>{}</sub></td><td>{}</td></tr>\n",
                // <sub> only for the non-numeric names; q<N> keeps full size. GitHub
                // strips class/style, so <sub> is the only lever here.
                match r.n {
                    None => format!("<sub>{}</sub>", query_cell_md(&r.query, links.query_url(d.query_rel, &stem))),
                    Some(_) => query_cell_md(&r.query, links.query_url(d.query_rel, &stem)),
                },
                mode_cells_md(r, links, d.canon_rel),
                peacock_cell_md(r.peacockdb, plan_url, cost_url),
                cost_cell_md(r.duckdb, dk_url),
                ratio_cell,
                if r.features.is_empty() { "—".to_string() } else { r.features.join(" ") },
                tickets_md(&r.tickets, links),
            );
        }
        s.push_str("</table>\n</details>\n\n");
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

    /// A dry-run `Links` (no sha), the shape every cell falls back to when the
    /// report is generated without a commit: `golden_url` yields None, so glyphs and
    /// costs render as plain text. Tests that assert bare glyph output use this.
    fn dry_links() -> Links {
        Links { repo: "o/r".to_string(), sha: None, tickets: TicketIndex::default() }
    }

    /// A `Links` WITH a sha, so link-emitting paths are exercised.
    fn sha_links() -> Links {
        Links { repo: "o/r".to_string(), sha: Some("abc123".to_string()) , tickets: TicketIndex::default() }
    }

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
            plan_status: "ok".to_string(),
            features: vec![],
            tickets: vec![],
            peacockdb,
            duckdb,
            cpu_golden: BTreeMap::new(),
        }
    }

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
        let links = links_with_tickets(&["103"], &[]);
        assert_eq!(tickets_html(&[], &links), "—");
        assert_eq!(features_html(&[]), "—");
        let t = tickets_html(&["103".to_string()], &links);
        assert!(t.contains("llm-wiki/tickets.md#t103"), "{t}");
        assert!(t.contains(">#103<"), "{t}");
        assert!(features_html(&["stddev_var".to_string()]).contains("stddev_var"));
    }

    fn links_with_tickets(open: &[&str], archived: &[&str]) -> Links {
        links_with_every_file(open, &[], archived)
    }

    fn links_with_every_file(open: &[&str], rollout: &[&str], archived: &[&str]) -> Links {
        let numbers = |list: &[&str]| list.iter().map(|t| t.to_string()).collect();
        Links {
            repo: "asymptote-tech/peacockdb".into(),
            sha: None,
            tickets: TicketIndex {
                open: numbers(open),
                rollout: numbers(rollout),
                archived: numbers(archived),
            },
        }
    }

    /// The rollout file is the third the index reads, and a number in it links there. It
    /// exists because a sweep files and closes tickets in bulk, which is not what a triage
    /// pass reads for — and cost-report exits 1 on a ticket in no file, so the first one
    /// filed would fail this job until the index read it.
    #[test]
    fn a_rollout_ticket_links_into_the_rollout_file() {
        let links = links_with_every_file(&["170"], &["180"], &["103"]);
        let rendered = tickets_html(&["180".to_string()], &links);
        assert!(
            rendered.contains("llm-wiki/tasks/bp-tickets.md#t180"),
            "{rendered}"
        );
        assert!(rendered.contains(">#180<"), "{rendered}");
    }

    /// A closed ticket keeps its registry cell and moves file, so the link has to follow
    /// it. The two files are told apart by which one holds the anchor, never by the
    /// number — an archived number links into the archive and an open one does not.
    ///
    /// The numbers are examples and have to stay one open and one archived; this one builds
    /// its own index, so it goes red only if the LINKING breaks, while its sibling below
    /// reads the wiki and goes red when a ticket is archived.
    #[test]
    fn an_archived_ticket_links_into_the_archive() {
        let links = links_with_tickets(&["170"], &["103"]);
        let archived = tickets_html(&["103".to_string()], &links);
        assert!(
            archived.contains("llm-wiki/archive/archived-tickets.md#t103"),
            "{archived}"
        );
        assert!(archived.contains(">#103<"), "{archived}");
        // The open one is unmoved by the archive existing.
        let open = tickets_html(&["170".to_string()], &links);
        assert!(open.contains("llm-wiki/tickets.md#t170"), "{open}");
    }

    /// The index is read off the anchors, which is what the links point at — so a ticket
    /// that has one resolves and a number that has none anywhere does not.
    ///
    /// It reads the real wiki, so the numbers below are examples that must stay one per
    /// file: archiving the ticket named here turns this red, and the fix is to swap in a
    /// currently-open number rather than to look for a bug in `path_for`. That is the price
    /// of reading the files rather than a fixture, and it is the reason this caught #103
    /// moving.
    #[test]
    fn the_index_reads_the_anchors_of_every_wiki_file() {
        let wiki = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../llm-wiki");
        let index = TicketIndex::load(&wiki);
        assert_eq!(index.path_for("170"), Some("llm-wiki/tickets.md"));
        assert_eq!(index.path_for("180"), Some("llm-wiki/tasks/bp-tickets.md"));
        assert_eq!(
            index.path_for("103"),
            Some("llm-wiki/archive/archived-tickets.md")
        );
        // A number nothing has ever used is a link to nowhere, and says so.
        assert_eq!(index.path_for("99999"), None);
    }

    /// Placement, which the case above cannot see: it asserts `170` resolves, and it passed
    /// while #170's anchor sat above #177's header. Built from strings rather than from the
    /// wiki, so the red case is reachable without an edit to `tickets.md` that would then
    /// have to be undone.
    #[test]
    fn an_anchor_must_sit_above_its_own_header() {
        let numbers = |text: &str| {
            anchored_numbers(text).map(|found| found.into_iter().collect::<Vec<_>>())
        };
        let placed = "<a id=\"t170\"></a>\n### #170 — a ticket\n\nbody\n\n\
                      <a id=\"t177\"></a>\n\n### #177 — another\n";
        assert_eq!(
            numbers(placed),
            Ok(vec!["170".to_string(), "177".to_string()]),
            "an anchor above its own header loads, blank lines between them or not"
        );
        // The shape that shipped: #177 inserted above #170's anchor, so every link to
        // #t170 landed on #177. The inner anchor is skipped past rather than refused, so
        // the error names the header #170 actually reached.
        let stacked = numbers("<a id=\"t170\"></a>\n<a id=\"t177\"></a>\n### #177 — another\n")
            .expect_err("a header that is not the anchor's own is refused");
        assert!(stacked.contains("t170") && stacked.contains("#177"), "{stacked}");
        // A prefix is not a match, or #17 would answer for #170.
        assert!(numbers("<a id=\"t170\"></a>\n### #17 — a ticket\n").is_err());
        assert!(numbers("<a id=\"t17\"></a>\n### #170 — a ticket\n").is_err());
        // An anchor with nothing under it resolves to nowhere, which is the same defect.
        assert!(numbers("<a id=\"t170\"></a>\n\n").is_err());
        // The convention as the two pages describe it in prose is not an anchor.
        assert_eq!(numbers("each ticket carries an `<a id=\"tNN\">` anchor\n"), Ok(vec![]));
    }

    /// The three row shapes, and the merge. PlanOnly is covered here deliberately:
    /// no query in the current registry is plan-only (the 4 non-executable TPC-DS
    /// queries all FAIL to plan), so without this test that branch would ship
    /// unexercised and could rot silently until the first query lands in it.
    #[test]
    fn row_kind_and_merged_mode_cells() {
        let exec = test_row("q1", &[("full_table_gpu", "enabled")], None, None);
        assert_eq!(exec.kind(), RowKind::Executable);
        let h = mode_cells_html(&exec, &dry_links(), "testdata/goldens/tpch.sf1");
        assert_eq!(h.matches("<td").count(), 4, "executable rows keep 4 cells: {h}");
        assert!(!h.contains("colspan"), "executable rows must not merge: {h}");

        // plans, but nothing enabled -> one spanning cell
        let plan_only = test_row("q42", &[], None, None);
        assert_eq!(plan_only.kind(), RowKind::PlanOnly);
        let h = mode_cells_html(&plan_only, &dry_links(), "testdata/goldens/tpch.sf1");
        assert_eq!(h.matches("<td").count(), 1);
        assert!(h.contains("colspan=\"4\"") && h.contains("plan ✓"), "{h}");

        // does not plan -> one spanning cell, and plan failure DOMINATES an
        // enabled mode (that combination is itself a bug and must read as ✗).
        let mut failed = test_row("q27", &[("full_table_gpu", "enabled")], None, None);
        failed.plan_status = "fail".to_string();
        assert_eq!(failed.kind(), RowKind::PlanFailed);
        let h = mode_cells_html(&failed, &dry_links(), "testdata/goldens/tpch.sf1");
        assert!(h.contains("colspan=\"4\"") && h.contains("plan ✗"), "{h}");

        // The markdown comment now mirrors the HTML structure (raw <table>), so the
        // same invariants hold there: 4 cells when executable, 1 spanning cell
        // otherwise. <sub> is what shrinks text — GitHub strips class/style.
        assert_eq!(mode_cells_md(&exec, &dry_links(), "testdata/goldens/tpch.sf1").matches("<td").count(), 4);
        assert!(mode_cells_md(&exec, &dry_links(), "testdata/goldens/tpch.sf1").contains("<sub>"));
        assert_eq!(mode_cells_md(&plan_only, &dry_links(), "testdata/goldens/tpch.sf1").matches("<td").count(), 1);
        assert!(mode_cells_md(&plan_only, &dry_links(), "testdata/goldens/tpch.sf1").contains("colspan=\"4\""));
        assert!(mode_cells_md(&failed, &dry_links(), "testdata/goldens/tpch.sf1").contains("plan ✗"));
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
            plan_status: "ok".to_string(),
            features: vec![],
            tickets: vec![],
            peacockdb: None,
            duckdb: None,
            cpu_golden: BTreeMap::new(),
        };
        // Dry run (no sha): no URLs, so both glyphs are bare — the same degradation
        // the Sigma-out cells have always had.
        assert_eq!(r.ftc_cell(&dry_links(), "testdata/goldens/tpch.sf1"), "tp1✓ tp8✗");
    }

    /// The ✓ links, resolved PER COLUMN from [`CPU_GOLDEN_LABEL`].
    ///
    /// scan_limit earns a case here because it is the one query where the answer is
    /// not obvious: it owns TWO tp1 goldens (registered at tp1_mini and tp1_standard,
    /// both mapping to the single ftc_tp1 cell), yet that cell shows one link, at
    /// tp1-standard like every other tp1 row. Pinning it stops a future reader from
    /// "fixing" the link to tp1-mini on the theory that the mini golden is unreachable
    /// by mistake — it is unreachable by decision (see the task spec).
    #[test]
    fn enabled_cpu_glyphs_link_to_their_own_golden() {
        let canon = "testdata/goldens/tpch.sf1";

        let mut r = test_row("q1", &[("ftc_tp1", "enabled"), ("ftc_tp8", "enabled")], None, None);
        r.cpu_golden.insert("ftc_tp1".into(), "full_table-tp1-standard".into());
        r.cpu_golden.insert("ftc_tp8".into(), "full_table-tp8-mini".into());
        let cell = r.ftc_cell(&sha_links(), canon);
        assert!(cell.contains(&format!("{canon}/q1.full_table-tp1-standard.cpu.txt")), "{cell}");
        assert!(cell.contains(&format!("{canon}/q1.full_table-tp8-mini.cpu.txt")), "{cell}");
        assert_eq!(cell.matches("<a href=").count(), 2, "both glyphs link: {cell}");

        // scan_limit owns BOTH tp1 goldens (registered at tp1_mini and tp1_standard,
        // one ftc_tp1 cell). It links at tp1-standard like every other tp1 row — the
        // decision recorded in the task spec — and its stem is hyphenated.
        let mut sl = test_row("scan_limit", &[("ftc_tp1", "enabled"), ("ftc_tp8", "disabled")], None, None);
        sl.cpu_golden.insert("ftc_tp1".into(), "full_table-tp1-standard".into());
        let cell = sl.ftc_cell(&sha_links(), canon);
        assert!(cell.contains(&format!("{canon}/scan-limit.full_table-tp1-standard.cpu.txt")), "{cell}");

        // Only ✓ links: a disabled cell has no golden behind it, so it stays plain
        // even when a sha is present.
        assert_eq!(sl.cpu_glyph("ftc_tp8", &sha_links(), canon), "✗");
        // ...and a `skip` cell likewise (runs, but nothing verified it).
        let sk = test_row("q2", &[("partitioned_cpu", "skip")], None, None);
        assert_eq!(sk.cpu_glyph("partitioned_cpu", &sha_links(), canon), "~");
    }

    /// Every mode column is accounted for: either CPU_GOLDEN_LABEL probes it, or it is
    /// a GPU column that owns no golden.
    ///
    /// The list that DEFINES the golden check was itself unchecked. Add a seventh mode
    /// column, or rename one, and CPU_GOLDEN_LABEL simply never probes it: its ✓ renders
    /// unlinked and the missing-golden gate stays silent, because that gate only
    /// iterates the labels it already knows. Same shape as an exemption list nothing
    /// validates, one level up. This forces the new column to be classified —
    /// linkable-with-a-golden, or GPU/unlinkable — rather than degrading quietly.
    #[test]
    fn every_mode_column_is_either_probed_or_declared_goldenless() {
        use std::collections::BTreeSet;
        // GPU columns read the CPU golden rather than owning one, so they are
        // deliberately not linked — see the CPU_GOLDEN_LABEL doc.
        const GOLDENLESS: [&str; 2] = ["full_table_gpu", "partitioned_gpu"];

        let probed: BTreeSet<&str> = CPU_GOLDEN_LABEL.iter().map(|(c, _)| *c).collect();
        let declared: BTreeSet<&str> = GOLDENLESS.into_iter().collect();
        let covered: BTreeSet<&str> = probed.union(&declared).copied().collect();
        let all: BTreeSet<&str> = MODE_COLUMNS.into_iter().collect();

        assert!(
            probed.is_disjoint(&declared),
            "a column cannot be both probed for a golden and declared goldenless: {:?}",
            probed.intersection(&declared).collect::<Vec<_>>()
        );
        assert_eq!(
            covered, all,
            "unclassified mode column(s): {:?}. Add it to CPU_GOLDEN_LABEL with the \
             golden label its ✓ should link to, or to GOLDENLESS if it owns no golden.",
            all.difference(&covered).collect::<Vec<_>>()
        );
    }

    /// The Query column shrinks ONLY the non-numeric names, in both renders.
    #[test]
    fn micro_query_names_render_small() {
        assert!(test_row("shuffle_stddev", &[], None, None).n.is_none());
        assert!(test_row("q1", &[], None, None).n.is_some());
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
        let url = Some("https://x/blob/abc/testdata/goldens/tpch.sf1/q1.full_table-tp1-mini.cpu.txt".to_string());
        assert!(cost_cell_html(v, url.clone()).starts_with("<a href="));
        assert!(cost_cell_md(v, url).starts_with("<a href="));
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
        let plan = Some("https://x/q1.full_table-tp8-mini.cpu.txt".to_string());
        let cost = Some("https://x/q1.full_table-tp8-mini.cost.txt".to_string());
        let html = peacock_cell_html(Some(43_308_088), plan.clone(), cost.clone());
        assert!(html.contains(">plan</a>") && html.contains(">cost</a>") && html.starts_with("41.30 MB ("));
        let md = peacock_cell_md(Some(43_308_088), plan, cost);
        // HTML anchors: the comment's table is raw HTML, where markdown link
        // syntax would render literally as brackets.
        assert!(md.contains("<a href=\"https://x/q1.full_table-tp8-mini.cpu.txt\">plan</a>"), "{md}");
        assert!(md.contains("<a href=\"https://x/q1.full_table-tp8-mini.cost.txt\">cost</a>"), "{md}");
        assert!(md.starts_with("41.30 MB ("));
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
        Links { repo: "o/r".into(), sha: None, tickets: TicketIndex::default() }
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
        let links = Links {
            repo: "o/r".into(),
            sha: Some("abc123".into()),
            tickets: TicketIndex::default(),
        };
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
        let linked = Links { repo: "o/r".into(), sha: Some("deadbeef".into()) , tickets: TicketIndex::default() };
        let url = "https://github.com/o/r/blob/deadbeef/testdata/tpch-queries/q1.sql";

        let html = render_html(&[one_row_dataset()], "https://p/", &linked, None, None);
        assert!(html.contains(&format!("<a href=\"{url}\">q1</a>")));
        let md = render_markdown(&[one_row_dataset()], "https://p/", false, &linked, None);
        assert!(md.contains(&format!("<a href=\"{url}\">q1</a>")));

        // No sha → plain q1 cell, no query link.
        let html_plain = render_html(&[one_row_dataset()], "https://p/", &no_links(), None, None);
        assert!(html_plain.contains("<td>q1</td>") && !html_plain.contains("q1.sql"));
        let md_plain = render_markdown(&[one_row_dataset()], "https://p/", false, &no_links(), None);
        assert!(md_plain.contains("<td>q1</td>") && !md_plain.contains("q1.sql"));
    }

    #[test]
    fn diff_query_url_parses_dataset_and_qn() {
        let links = Links { repo: "o/r".into(), sha: Some("cafe".into()) , tickets: TicketIndex::default() };
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
        let links = Links { repo: "o/r".into(), sha: Some("cafe".into()) , tickets: TicketIndex::default() };
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
