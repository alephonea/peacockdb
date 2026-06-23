//! Cost taxonomy + multiplier config (loaded from `cost_model.conf`) and the
//! `.cost.txt` generator.
//!
//! The `.cost.txt` golden is derived PURELY from the sibling `.cpu.txt` text:
//! parse each node line's type + `output_bytes`, bin it into a category, then sum
//! `multiplier * bytes` over the categories. No executor run — recomputing cost is
//! a cheap text parse, so cost goldens regenerate without the expensive plan run.
//!
//! The taxonomy + multipliers live in the text file `testdata/cost_model.conf`
//! (read at runtime, not compiled in): a multiplier can be retuned and the cost
//! goldens regenerated without recompiling, and the config stays a plain editable
//! file with no parser-crate dependency (the format is trivial whitespace columns).

const OUTPUT_BYTES_KEY: &str = "output_bytes=";

/// One cost category: where its bytes come from and how they are weighted.
pub struct Category {
    pub name: String,
    pub multiplier: f64,
    /// Gpu node types binned into this category (empty = placeholder phase).
    pub nodes: Vec<String>,
}

/// The parsed `cost_model.conf`, in file (= `.cost.txt` line) order.
pub struct CostModel {
    pub categories: Vec<Category>,
}

impl CostModel {
    /// Load + parse `cost_model.conf`. It lives under the testdata root (and so is
    /// relocated by `PEACOCK_TESTDATA_DIR` together with the goldens it drives, e.g.
    /// when the suite runs against shipped artifacts on a remote host).
    pub fn load() -> CostModel {
        let path = super::testdata_root().join("cost_model.conf");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read cost config {}: {e}", path.display()));
        CostModel::parse(&text)
    }

    /// Parse the config text. Each non-comment, non-blank line is
    /// `<category> <multiplier> [comma,separated,nodes]`.
    pub fn parse(text: &str) -> CostModel {
        let mut categories = Vec::new();
        for line in text.lines() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let mut cols = line.split_whitespace();
            let name = cols.next().expect("cost config: missing category name").to_string();
            let multiplier = cols
                .next()
                .unwrap_or_else(|| panic!("cost config: '{name}' missing multiplier"))
                .parse()
                .unwrap_or_else(|e| panic!("cost config: '{name}' bad multiplier: {e}"));
            let nodes = match cols.next() {
                Some(list) => list.split(',').map(str::to_string).collect(),
                None => Vec::new(),
            };
            categories.push(Category { name, multiplier, nodes });
        }
        CostModel { categories }
    }

    /// Category index for a node type, or `None` if it is not in the taxonomy.
    fn category_of(&self, node_type: &str) -> Option<usize> {
        self.categories.iter().position(|c| c.nodes.iter().any(|n| n == node_type))
    }

    /// Derive the `.cost.txt` body from a `.cpu.txt` body. One line per category
    /// `<category>=<raw bytes> # <node types>`, then a `peacockdb_cost=<total>`
    /// footer where `total = Σ(multiplier * bytes)`. Panics (via `ctx` for the
    /// message) on a node type absent from the taxonomy — the taxonomy must be total.
    pub fn cost_text_from_cpu(&self, cpu_text: &str, ctx: &str) -> String {
        let mut bytes = vec![0u64; self.categories.len()];
        for line in cpu_text.lines() {
            let Some((node_type, ob)) = parse_node_line(line) else { continue };
            let cat = self
                .category_of(node_type)
                .unwrap_or_else(|| panic!("{ctx}: node type '{node_type}' is not in the cost taxonomy"));
            bytes[cat] += ob;
        }
        let mut total = 0.0f64;
        let mut out = String::new();
        for (i, c) in self.categories.iter().enumerate() {
            total += c.multiplier * bytes[i] as f64;
            let comment = if c.nodes.is_empty() {
                "(placeholder, no node mapping)".to_string()
            } else {
                c.nodes.join(", ")
            };
            out.push_str(&format!("{}={} # {comment}\n", c.name, bytes[i]));
        }
        out.push_str(&format!("peacockdb_cost={}", total.round() as u64));
        out
    }
}

/// Parse one `.cpu.txt` node line into `(node_type, output_bytes)`. `None` for any
/// line without `output_bytes=` (blank lines, the trailing total footer if present).
fn parse_node_line(line: &str) -> Option<(&str, u64)> {
    let pos = line.find(OUTPUT_BYTES_KEY)?;
    // Node name = the leading identifier. A node with no args renders bare (e.g.
    // `GpuCoalescePartitionsExec, output_bytes=…`), so stop at the first non-ident
    // char (`:`, `,` or space), not at `:` alone.
    let trimmed = line.trim_start();
    let end = trimmed.find(|c: char| !c.is_alphanumeric()).unwrap_or(trimmed.len());
    let node_type = &trimmed[..end];
    let bytes: u64 = line[pos + OUTPUT_BYTES_KEY.len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    Some((node_type, bytes))
}
