# Task: link CPU ✓ cells to their goldens; small-font Query and Σout columns

Branch `ENS-widget-golden-links` (off `ENS-test-exec-mode`). Cost-report widget only —
`cost-report/src/main.rs`. No changes to the registry CSV, the test suite, or any golden.

## 1. The premise, verified

Every `enabled` cell in `ftc_tp1`, `ftc_tp8` and `partitioned_cpu` **has a committed
`.cpu.txt`**. `assert_cpu_cost_canonical` runs unconditionally on every CPU macro
invocation (`common/mod.rs:847`), before and independent of the `ResultGolden` keyword —
that keyword gates only `.result.txt`. So a `✓` in those three columns always has a
golden to point at.

This does *not* extend to the GPU columns: those read the CPU golden rather than owning
one, so leave `full_table_gpu` / `partitioned_gpu` cells alone.

## 2. Which golden each ✓ points at

The device label is **not** in the CSV, and it is not one-per-column:

| Column | Golden label |
|---|---|
| `ftc_tp8` | `full_table-tp8-mini` |
| `partitioned_cpu` | `partitioned-tp8-standard` |
| `ftc_tp1` | `full_table-tp1-standard` |

**Corrected 2026-08-04.** This section originally said `ftc_tp1` is tp1-standard "except
`scan_limit`, which is `full_table-tp1-mini`". That is wrong, and the developer caught it:
`scan_limit` has **both** goldens. It is registered twice — `tp1_mini` at
`test_cpu_full_table.rs:24` and `tp1_standard` at `:188` — and `column_for` keys on the tp
count, not the memory tier, so both land in the single `ftc_tp1` cell. It is not the
exception to the column's label; it is the one query whose cell aggregates two runs.

**Decision: one label per column, `full_table-tp1-standard`.** It is the label every tp1
row uses including `scan_limit`, so the link target is predictable from the column alone.
Drop the tp1-mini candidate rather than carrying config that nothing can reach — this repo
already treats unreachable config as a hazard in its own right (`GOLDEN_INVARIANT_EXEMPT`
and `INTENTIONALLY_NOT_IN_CI` both carry staleness assertions for exactly that).

What a cell aggregating two runs should render is a real question, but a hyperlink cannot
express it and the answer would change the cell shape. Out of scope here; raise it as its
own task if the tp1-mini run ever needs to be reachable from the widget.

The fail-loud check below is what makes dropping the candidate safe: a future query
registered ONLY at tp1-mini has no golden under the single label, so the widget fails
naming it instead of rendering a dead link. That is strictly better than a silent second
candidate, because it forces the decision rather than guessing.

Link target: the same `links.golden_url(canon_rel, stem, "<label>.cpu.txt")` helper the
Σout cell uses, so dry runs with no sha degrade to plain text exactly as they do today.

Only `✓` becomes a link. `~`, `✗` and `—` stay plain — there is no golden behind them.

## 3. Small font

- **Query column:** non-numeric queries only — `aggregate_groupby`, `scan_limit`,
  `shuffle_stddev`, `hash_join`, … The discriminator already exists: `Row::number` is
  `None` for exactly these (it is `Some(n)` for `q<N>`). Numbered queries keep today's
  size.
- **PeacockDB Σout and DuckDB Σout:** small font for **both the header and the values**.
  The Ratio column is not in scope.

Both renders. They use different mechanisms and the difference is load-bearing: the HTML
report can use a CSS class (`th.modeh` is the precedent), while the PR-comment table must
use `<sub>` because GitHub strips `class`/`style` — see the `mode_cells_md` doc comment.

## 4. Watch for

- `MODE_COLUMNS` and `registry.rs::COLUMNS` are the CSV header contract. This task is
  display-only: do not touch either, and do not change any cell value.
- `ftc_cell()` currently renders `tp1✓ tp8✓` as one string inside a single `<td>`. Both
  glyphs need to become independently linkable, so that function has to return markup
  rather than a plain label — check its callers in both renders before changing its shape.
- `CPU_DEVICE` is the const the Σout cells resolve through. The new per-column labels are
  related but not the same thing; do not fold them together in a way that makes a Σout
  change silently move a mode link.

## 5. Verification

- `cargo test -p cost-report` — the widget's own unit tests, including the
  `mode_cells_md` shape asserts.
- `scripts/cost-report-preview.sh` and eyeball the generated HTML: a linked `✓` per
  enabled CPU cell, `scan_limit`'s tp1 link resolving to `full_table-tp1-mini`, small-font
  micro-query names, small-font Σout headers and values.
- Confirm the PR-comment render still parses as HTML on GitHub (`<sub>`, no class/style).
- No golden, CSV or test-suite file appears in `git status`.
