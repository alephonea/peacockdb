//! Reading the golden text format: `== <query>` sections, and the node lines inside one.
//!
//! One reader, because the format is one format. The plan goldens, the cost derivation and
//! the corpus tiers all split on the same header and all read the same node line, and three
//! readers that agree by luck is what this replaces — the divergence is the cost, never the
//! duplication.

/// A node line: its name, its depth in the tree, and every field it carries.
///
/// Indentation is the tree, two spaces per level, so `depth` is what a caller pairs a node
/// with its parent by. Fields keep file order and are borrowed from the line.
pub struct NodeLine<'a> {
    pub name: &'a str,
    pub depth: usize,
    pub fields: Vec<(&'a str, &'a str)>,
}

impl<'a> NodeLine<'a> {
    pub fn field(&self, key: &str) -> Option<&'a str> {
        self.fields
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| *value)
    }

    /// The field as a count. `None` when the field is absent; panics when it is present and
    /// not a number, since that is a renderer defect rather than a line of another kind.
    pub fn count(&self, key: &str) -> Option<u64> {
        self.field(key).map(|value| {
            value.parse().unwrap_or_else(|e| {
                panic!("{}: field `{key}={value}` is not a count: {e}", self.name)
            })
        })
    }
}

/// Parse one line of a golden as a node line, or `None` for a line of any other kind.
///
/// A node line is an indented name — the leading identifier, capitalized as every node kind
/// in both families is — followed by the end of the line, its fields after `: `, or more
/// fields after `, `. That rule is what separates it from the per-batch continuation line,
/// a legacy `pK:` sub-line, a `== ` header, a `--- memory ---` marker and a cost category,
/// none of which start with a capital.
pub fn parse_node_line(line: &str) -> Option<NodeLine<'_>> {
    let indent = line.len() - line.trim_start().len();
    let trimmed = &line[indent..];
    if !trimmed.starts_with(|c: char| c.is_ascii_uppercase()) {
        return None;
    }
    let end = trimmed
        .find(|c: char| !c.is_alphanumeric())
        .unwrap_or(trimmed.len());
    let (name, rest) = trimmed.split_at(end);
    let rest = match rest.chars().next() {
        None => "",
        Some(':') | Some(',') => &rest[1..],
        Some(_) => return None,
    };
    Some(NodeLine {
        name,
        depth: indent / 2,
        fields: split_fields(rest),
    })
}

/// The comma-separated fields of a node line, split where a comma is not inside brackets or
/// a quoted string — an expression carries both (`on=[(c_custkey@0, o_custkey@1)]`,
/// `Decimal128(38, 15)`), and splitting inside one would cut a value in half.
fn split_fields(rest: &str) -> Vec<(&str, &str)> {
    let mut fields = Vec::new();
    let mut nesting = 0usize;
    let mut quoted = false;
    let mut start = 0usize;
    for (at, c) in rest.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '(' | '[' | '{' if !quoted => nesting += 1,
            ')' | ']' | '}' if !quoted => nesting = nesting.saturating_sub(1),
            ',' if !quoted && nesting == 0 => {
                fields.push(split_key(&rest[start..at]));
                start = at + 1;
            }
            _ => {}
        }
    }
    fields.push(split_key(&rest[start..]));
    fields.retain(|(key, value)| !key.is_empty() || !value.is_empty());
    fields
}

/// At the first `=`, since a value carries its own (`predicate=l_shipdate@4 = 5`). A field
/// with none is its own key and an empty value.
fn split_key(field: &str) -> (&str, &str) {
    let field = field.trim();
    match field.find('=') {
        Some(at) => (field[..at].trim(), field[at + 1..].trim()),
        None => (field, ""),
    }
}

/// Sections in file order: `(query, body)` at each `== ` header, names as the file writes
/// them.
pub fn ordered_sections(text: &str) -> Vec<(String, String)> {
    let mut sections: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        match line.strip_prefix("== ") {
            Some(header) => sections.push((header.to_string(), String::new())),
            None => {
                if let Some((_, body)) = sections.last_mut() {
                    body.push_str(line);
                    body.push('\n');
                }
            }
        }
    }
    sections
}

/// Every query whose section moved, one short line each. One file holds every query, so a
/// whole-file `assert_eq!` says only that a two-megabyte golden differs — and a plan line
/// runs past a thousand characters, which the CI log drops, so even the dump it prints
/// arrives unreadable. One line per query, each naming the column that moved, is what
/// survives the log and what a person can scan.
pub fn section_differences(canonical: &str, actual: &str) -> Vec<String> {
    let expected = ordered_sections(canonical);
    let produced = ordered_sections(actual);
    let mut said = Vec::new();
    for (position, (name, body)) in expected.iter().enumerate() {
        let Some((produced_name, produced_body)) = produced.get(position) else {
            said.push(format!(
                "{name}: in the golden, and the run produced no section there"
            ));
            continue;
        };
        if produced_name != name {
            said.push(format!(
                "section {position}: `{name}` in the golden and `{produced_name}` in the run"
            ));
            continue;
        }
        if produced_body != body {
            said.push(format!("{name}: {}", line_difference(body, produced_body)));
        }
    }
    for (name, _) in produced.iter().skip(expected.len()) {
        said.push(format!(
            "{name}: produced by the run, and the golden has no such section"
        ));
    }
    if said.is_empty() && canonical != actual {
        said.push("every section matches — a header or a trailing byte moved".to_string());
    }
    said
}

/// The first line of a section that differs, as the column it differs at and a window
/// either side. A plan line is long enough that printing two of them whole says less than
/// pointing at the character, and a long line is the one the log drops.
pub fn line_difference(expected: &str, actual: &str) -> String {
    for (number, (want, got)) in expected.lines().zip(actual.lines()).enumerate() {
        if want == got {
            continue;
        }
        let at = want
            .char_indices()
            .zip(got.char_indices())
            .find(|((_, a), (_, b))| a != b)
            .map(|((index, _), _)| index)
            .unwrap_or_else(|| want.len().min(got.len()));
        let rest = expected
            .lines()
            .zip(actual.lines())
            .skip(number + 1)
            .filter(|(want, got)| want != got)
            .count();
        return format!(
            "line {}, column {at} — expected `{}` — actual `{}`{}",
            number + 1,
            window(want, at),
            window(got, at),
            if rest > 0 {
                format!(" (+{rest} more lines)")
            } else {
                String::new()
            }
        );
    }
    format!(
        "{} lines in the golden and {} in the run",
        expected.lines().count(),
        actual.lines().count()
    )
}

/// 40 characters either side of `at`, elided at both ends — narrow enough that a run of
/// them stays inside what a CI log carries per line.
fn window(line: &str, at: usize) -> String {
    let start = at.saturating_sub(40);
    let end = (at + 40).min(line.len());
    let start = (start..=at)
        .find(|i| line.is_char_boundary(*i))
        .unwrap_or(at);
    let end = (at..=end)
        .rev()
        .find(|i| line.is_char_boundary(*i))
        .unwrap_or(at);
    format!(
        "{}{}{}",
        if start > 0 { "…" } else { "" },
        &line[start..end],
        if end < line.len() { "…" } else { "" }
    )
}

/// A node as an execution golden records it: its line, and the per-batch record beneath.
///
/// `in_rows` is nested by child and then by that child's lane; `batch_rows` and
/// `batch_bytes` by this node's lane and then by batch. `abandoned` is per lane and
/// present only where a run left something behind, and `rows_skipped` is a total.
pub struct RunNode<'a> {
    pub line: NodeLine<'a>,
    pub in_rows: Vec<Vec<u64>>,
    pub batch_rows: Vec<Vec<u64>>,
    pub batch_bytes: Vec<Vec<u64>>,
    pub abandoned: Vec<u64>,
    pub rows_skipped: u64,
    /// Indices into the same vector. Depth is the tree, so the parent of a node is the
    /// nearest one above it at one less.
    pub children: Vec<usize>,
}

/// One `.cpu.txt` section: the early-exit marker and every node under it, in file order.
///
/// Panics on a line the format does not allow, since a section is written by our own
/// renderer — a shape nobody produces is a defect to name, not a case to tolerate.
pub fn parse_run_section(body: &str) -> (String, Vec<RunNode<'_>>) {
    let mut lines = body.lines();
    let marker = lines
        .next()
        .and_then(|line| line.strip_prefix("early_exit="))
        .expect("a section opens with its early_exit marker")
        .to_string();
    let mut nodes: Vec<RunNode<'_>> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(node) = parse_node_line(line) {
            let depth = node.depth;
            open.truncate(depth);
            if let Some(parent) = open.last() {
                let at = nodes.len();
                nodes[*parent].children.push(at);
            }
            open.push(nodes.len());
            nodes.push(RunNode {
                line: node,
                in_rows: Vec::new(),
                batch_rows: Vec::new(),
                batch_bytes: Vec::new(),
                abandoned: Vec::new(),
                rows_skipped: 0,
                children: Vec::new(),
            });
            continue;
        }
        let node = nodes.last_mut().expect("a per-batch line under no node");
        for field in line.split_whitespace() {
            let (key, value) = field
                .split_once('=')
                .expect("a `key=value` per-batch field");
            match key {
                "in_rows" => node.in_rows = nested_numbers(value),
                "batch_rows" => node.batch_rows = nested_numbers(value),
                "batch_bytes" => node.batch_bytes = nested_numbers(value),
                "abandoned" => node.abandoned = flat_numbers(value),
                "rows_skipped" => node.rows_skipped = value.parse().expect("a row count"),
                other => panic!("unknown per-batch field `{other}` in: {line}"),
            }
        }
    }
    (marker, nodes)
}

/// `[[1,2],[3]]` — the bracket style every nested list in the format renders in.
fn nested_numbers(text: &str) -> Vec<Vec<u64>> {
    let inner = text
        .strip_prefix('[')
        .and_then(|text| text.strip_suffix(']'))
        .unwrap_or_else(|| panic!("not a nested list: {text}"));
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split_inclusive(']')
        .map(|lane| flat_numbers(lane.trim_start_matches(',')))
        .collect()
}

fn flat_numbers(text: &str) -> Vec<u64> {
    let inner = text
        .strip_prefix('[')
        .and_then(|text| text.strip_suffix(']'))
        .unwrap_or_else(|| panic!("not a list: {text}"));
    match inner.is_empty() {
        true => Vec::new(),
        false => inner
            .split(',')
            .map(|number| number.parse().expect("a number in a list"))
            .collect(),
    }
}
