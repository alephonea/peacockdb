#!/usr/bin/env python3
"""Draw the calibration record, before anything is fitted to it.

    /usr/bin/python3 scripts/calibration/plot.py \
        --record testdata/calibration/records.tsv \
        --hbm testdata/calibration/hbm.tsv \
        --out-dir testdata/calibration/plots

THE one script. Every picture under `plots/`, and the `index.html` that shows them on one
page, comes from this call and from nothing else. That is the point rather than a tidiness
preference: a second generator is a second reading of the format, and two readings of one
file disagree the first time a column moves.

A fit answers "what are the coefficients"; it cannot answer "is a line the right shape",
because it returns coefficients either way. These are the pictures that decide whether
fitting is meaningful at all, so nothing here reduces a call to one number before it is
drawn -- every measured execution is a point.

Not matplotlib's default python: this repo's `python3` is a linuxbrew build with neither
numpy nor matplotlib. `/usr/bin/python3` has both.

WHAT IS DRAWN

  load/     the SCAN, alone. `CudfScan` is storage and PCIe; everything else is compute.
            See "load is not processing" below -- this separation is the reason the
            directory exists rather than a panel inside `compute/`.
  compute/  per cuDF step kind: device_us against out_bytes and in_bytes, log-log, one
            point per execution. The shape the cost model assumes is a straight line
            through the origin; a per-kind panel is where it either looks like one or
            does not.
  spread/   per step kind: every execution's device_us as a ratio to that call's median
            across executions. A fit collapses those executions to one number, and this
            is the plot that says what that discards.
  query/    per (query, mode): where the time went, by node and by term.
  hbm/      hbm_bytes against out_bytes, where an HBM capture was joined in.
  icicle/   per (query, mode): plan node -> recipe step -> cuDF call, width in device
            microseconds. The record's three-level tuple, drawn as the three levels it is.

LOAD IS NOT PROCESSING

A scan at sf40 is most of the query and is entirely `read_parquet`: decompression, page
decode, PCIe. A filter is arithmetic over resident columns. Putting both on one axis
draws one picture about two different machines -- one bounded by storage, the other by
compute -- and a line through the pair describes neither. So no panel here mixes them,
and every panel says which of the two it is.

READING THE RECORD

The parser is this file's own, deliberately NOT imported from `fit.py`. That module reads
the pre-batch-partitioned format -- `wall_us`, `label`, `partition` -- and is the fit's,
which is a separate piece of work; importing it would tie this script to the columns it
was written to leave behind.

Everything is read BY COLUMN NAME. The record's columns have already changed twice, and a
reader that counts fields survives such a change quietly, drawing whatever slid into the
slot it wanted.
"""

import argparse
import collections
import html
import pathlib
import statistics
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# The cuDF step kinds that are STORAGE rather than compute. One name today, and a set
# rather than a comparison because the next loader (a cached scan, a row-group reader)
# belongs on this side of the line the moment it exists, and a reader that tests
# `== "CudfScan"` would put it on the wrong one silently.
LOAD_KINDS = {"CudfScan"}

# The tuple a row is addressed by, and the prefix of it that names one CALL within one
# execution of one case. `run_index` last because dropping it is exactly how you go from
# "this call" to "this call across the ten executions".
CASE = ("dataset", "sf", "query", "mode")
CALL = CASE + ("node_seq", "recipe_seq", "call_index")


def read_tsv(path):
    """Rows as dicts, keyed by the file's own column line.

    By name and not by position, for the reason the module docstring gives. A row whose
    field count disagrees with the header is an error rather than a shorter dict: the two
    ways it happens -- a tab inside a value, a writer half-updated -- both produce rows
    that would plot as real points.
    """
    names, rows = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if names is None:
                names = f
                continue
            if len(f) != len(names):
                sys.exit(f"{path}: row of {len(f)} fields against {len(names)} columns")
            rows.append(dict(zip(names, f)))
    if names is None:
        sys.exit(f"{path} has no column line")
    return rows


def require(rows, path, *columns):
    missing = [c for c in columns if c not in rows[0]]
    if missing:
        sys.exit(f"{path} has no {missing}; it has {sorted(rows[0])}")


def num(row, column):
    """A measured field, or None where the run did not measure it.

    Empty is not zero and the difference is the whole point: a call the device recorded
    no region for took time, and a zero would say it did not. Such a call is dropped from
    a plot rather than drawn at the origin, where it would pull every fit through it.
    """
    text = row.get(column, "")
    return int(text) if text != "" else None


def is_load(row):
    return row["recipe_kind"] in LOAD_KINDS


def case_label(row):
    return f"{row['dataset']}.sf{row['sf']} {row['query']} {row['mode']}"


def executions(rows):
    """Rows grouped by call, each group being that call's executions in run order.

    `run_index` is a column now, so this is a group-by rather than the recovery from row
    ORDER the format used to promise. That change is why this file no longer imports
    `fit.py`'s `region_groups`: what it recovered is now written down.
    """
    by_call = collections.defaultdict(list)
    for row in rows:
        by_call[tuple(row[c] for c in CALL)].append(row)
    for key, group in by_call.items():
        group.sort(key=lambda r: int(r["run_index"]))
        seen = [int(r["run_index"]) for r in group]
        if seen != list(range(len(seen))):
            sys.exit(f"call {key} has executions {seen}, which is not 0..n")
    return by_call


# ---------------------------------------------------------------------------
# scatter panels: compute/ and load/
# ---------------------------------------------------------------------------

def scatter(ax, points, title, xlabel, ylabel):
    """One log-log panel, one point per EXECUTION.

    Log-log because the model's claim is `t = a * bytes`, which is a straight line of
    slope 1 here whatever `a` is -- so the eye checks the SHAPE without knowing the
    coefficient. A panel whose points bend, or sit at two heights for one x, is a panel
    the model cannot describe, and that is visible before any fitting.
    """
    if not points:
        ax.set_axis_off()
        return False
    by_sf = collections.defaultdict(list)
    for sf, x, y in points:
        if x > 0 and y > 0:
            by_sf[sf].append((x, y))
    if not by_sf:
        ax.set_axis_off()
        return False
    for sf in sorted(by_sf, key=lambda s: int(s)):
        xs = [p[0] for p in by_sf[sf]]
        ys = [p[1] for p in by_sf[sf]]
        ax.scatter(xs, ys, s=14, alpha=0.55, label=f"sf{sf}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, which="both")
    if len(by_sf) > 1:
        ax.legend(fontsize=7)
    return True


def plot_kinds(rows, out_dir, which):
    """One figure per cuDF step kind, for one side of the load/compute line.

    Per KIND rather than per plan node: the record's `recipe_kind` is what the device was
    actually asked to do, and one plan node publishes several of them -- an aggregate
    concatenates, merges and finalizes. A per-node panel would average three different
    kernels into one cloud and call the result a node's cost.
    """
    want = LOAD_KINDS if which == "load" else None
    kinds = sorted({
        r["recipe_kind"] for r in rows
        if (r["recipe_kind"] in want if want else not is_load(r))
    })
    made = []
    for kind in kinds:
        of_kind = [r for r in rows if r["recipe_kind"] == kind]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        drawn = False
        for ax, xcol in zip(axes, ("out_bytes", "in_bytes")):
            points = [
                (r["sf"], num(r, xcol), num(r, "device_us"))
                for r in of_kind
                if num(r, xcol) is not None and num(r, "device_us") is not None
            ]
            drawn |= scatter(ax, points, f"{kind}: device_us vs {xcol}",
                             xcol, "device_us")
        note = ("STORAGE — this is read_parquet: decompression, page decode, PCIe. "
                "Never drawn beside compute; see the module docstring."
                if which == "load" else
                "COMPUTE — arithmetic over resident columns. The scan is in load/.")
        fig.suptitle(f"{kind}   ({len(of_kind)} executions)   {note}", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        if drawn:
            path = out_dir / which / f"{safe(kind)}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=110)
            made.append(path)
        plt.close(fig)
    return made


def safe(name):
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


# ---------------------------------------------------------------------------
# spread/ — what collapsing executions to one number throws away
# ---------------------------------------------------------------------------

def plot_spread(by_call, out_dir):
    """Per step kind: each execution's device_us as a ratio to its own call's median.

    The question is not "how fast is this call" but "how repeatable is it". A fit takes
    one number per call; this says how much of a number there was to take. A kind whose
    ratios sit inside a few percent is one where a median means something, and a kind
    with a long tail is one where the coefficient is being fitted to scheduling noise.

    Ratio to the CALL's own median, not to the kind's: calls of one kind differ by orders
    of magnitude in size, and a ratio across them would measure the query, not the noise.
    """
    by_kind = collections.defaultdict(list)
    for group in by_call.values():
        times = [num(r, "device_us") for r in group]
        times = [t for t in times if t is not None and t > 0]
        if len(times) < 2:
            continue
        mid = statistics.median(times)
        if mid <= 0:
            continue
        by_kind[group[0]["recipe_kind"]].extend(t / mid for t in times)
    if not by_kind:
        return []
    kinds = sorted(by_kind)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(kinds)), 4))
    # Ticks set after rather than through boxplot's own parameter: it was renamed
    # `labels` -> `tick_labels` in matplotlib 3.9, and this way the call works on both.
    ax.boxplot([by_kind[k] for k in kinds], showfliers=True,
               flierprops={"markersize": 2, "alpha": 0.4})
    ax.set_xticks(range(1, len(kinds) + 1))
    ax.set_xticklabels(kinds)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("device_us / this call's median", fontsize=8)
    ax.set_title("Spread across executions, per cuDF step kind — what a median discards",
                 fontsize=9)
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "spread" / "by-kind.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# query/ — where one case's time went
# ---------------------------------------------------------------------------

def median_run(rows):
    """One real execution: the one whose total device_us is the median.

    A real execution rather than a per-call median, for the reason the benchmark tree
    reports a whole run: per-call medians give a picture belonging to no execution, whose
    parts can sum to less than any that actually happened.
    """
    by_run = collections.defaultdict(int)
    for r in rows:
        t = num(r, "device_us")
        if t is not None:
            by_run[r["run_index"]] += t
    if not by_run:
        return None
    ordered = sorted(by_run, key=lambda k: by_run[k])
    return ordered[len(ordered) // 2]


def plot_queries(rows, out_dir):
    """Per case: device time by node, and the three time terms, for one execution.

    Two panels because they answer different questions. The left says which node to look
    at. The right says whether the answer is the device at all -- at sf1 the host
    prologue is most of a small node and none of a big one, and that difference is why
    the record carries three terms rather than a total.

    Never their sum: under CUDA events the host submission CONTAINS the device execution,
    so `peacock_host_us + cudf_host_us + device_us` describes no interval. Drawn side by
    side, never stacked on one bar.
    """
    made = []
    by_case = collections.defaultdict(list)
    for r in rows:
        by_case[tuple(r[c] for c in CASE)].append(r)
    for key, case_rows in sorted(by_case.items()):
        run = median_run(case_rows)
        if run is None:
            continue
        one = [r for r in case_rows if r["run_index"] == run]
        nodes = collections.defaultdict(lambda: [0, 0, 0])
        for r in one:
            label = f"{r['node_seq']} {r['node_type']}"
            for i, col in enumerate(("peacock_host_us", "cudf_host_us", "device_us")):
                v = num(r, col)
                if v is not None:
                    nodes[label][i] += v
        order = sorted(nodes, key=lambda n: -nodes[n][2])
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        axes[0].barh(range(len(order)), [nodes[n][2] for n in order], color="tab:blue")
        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels(order, fontsize=7)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("device_us", fontsize=8)
        axes[0].set_title("Device time by plan node", fontsize=9)
        axes[0].grid(axis="x", alpha=0.25)

        width = 0.27
        for i, (col, colour) in enumerate((
                ("peacock_host_us", "tab:green"),
                ("cudf_host_us", "tab:orange"),
                ("device_us", "tab:blue"))):
            axes[1].barh([y + (i - 1) * width for y in range(len(order))],
                         [nodes[n][i] for n in order], height=width, color=colour,
                         label=col)
        axes[1].set_yticks(range(len(order)))
        axes[1].set_yticklabels([], fontsize=7)
        axes[1].invert_yaxis()
        axes[1].set_xscale("symlog")
        axes[1].set_xlabel("microseconds (log)", fontsize=8)
        axes[1].set_title("Three terms, side by side — NOT summed:\n"
                          "under events the host interval contains the device one",
                          fontsize=8)
        axes[1].legend(fontsize=7)
        axes[1].grid(axis="x", alpha=0.25)
        fig.suptitle(f"{case_label(one[0])} — execution {run} of "
                     f"{len({r['run_index'] for r in case_rows})}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        path = out_dir / "query" / f"{safe(case_label(one[0]))}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        made.append(path)
    return made


# ---------------------------------------------------------------------------
# icicle/ — the record's three levels, drawn as three levels
# ---------------------------------------------------------------------------

def plot_icicle(rows, out_dir):
    """Per case: plan node -> recipe step -> cuDF call, width in device microseconds.

    The record's tuple has exactly these three levels and they NEST: a plan node
    publishes several recipe steps, and a batched run drives each step once per batch per
    lane. So the picture is not a choice of layout -- it is what the tuple already says,
    laid out so a level's children sit under it and end where it ends.

    Nodes run left to right in POST-order, which is both the order the record numbers
    them in and the order the device executes them. The plan tree renders the other way
    round, root first; the two are one tree read from opposite ends.

    Sums are the structure, not decoration: level 1 is grouped from level 2 and level 0
    from level 1, so a child overflowing its parent would mean the grouping is wrong.
    What can be checked against ANOTHER source is the set of steps under a node, and that
    is `--- recipes ---` (asserted in test_batch_partitioned_plans).

    TWO PANELS, and the second is the same rule the scatter panels follow. A scan is 95%
    of q6 at sf1, so on a truthful axis every compute frame is a sliver with no room for
    its own name -- the picture says "the scan dominates" and then nothing else. The
    lower panel drops the load frames and rescales, which is not a second truth but the
    same one at the magnification the compute side needs. Its title says so, because an
    icicle whose widths do not mean what the axis says is worse than no icicle.

    One execution, chosen the way `query/` chooses it and for the same reason.
    """
    made = []
    by_case = collections.defaultdict(list)
    for r in rows:
        by_case[tuple(r[c] for c in CASE)].append(r)

    for key, case_rows in sorted(by_case.items()):
        run = median_run(case_rows)
        if run is None:
            continue
        one = [r for r in case_rows if r["run_index"] == run]

        tree = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in one:
            t = num(r, "device_us")
            if t is None:
                continue
            tree[(int(r["node_seq"]), r["node_type"], is_load(r))][
                (int(r["recipe_seq"]), r["recipe_kind"])].append((int(r["call_index"]), t))
        if not tree:
            continue

        compute = {k: v for k, v in tree.items() if not k[2]}
        panels = [(tree, "every node, on a truthful axis")]
        if compute and len(compute) < len(tree):
            dropped = sorted(k[1] for k in tree if k[2])
            panels.append((compute, f"load frames dropped ({', '.join(dropped)}) and "
                                    "rescaled — the same numbers, magnified"))

        fig, axes = plt.subplots(len(panels), 1,
                                 figsize=(15, 1.0 + 2.3 * len(panels)), squeeze=False)
        for ax, (nodes, note) in zip(axes[:, 0], panels):
            total, legend = draw_icicle(ax, nodes)
            ax.set_title(f"{note} — {total}us", fontsize=8)
            # Every frame named, including the ones too narrow to hold their own label.
            # An icicle whose small frames are anonymous coloured slivers tells the reader
            # that something is there and refuses to say what, which is worse than a plain
            # table: on q19 only three of nine nodes clear the width a label needs, and
            # the six that do not are exactly the ones a reader cannot guess.
            ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1.005, 0.5),
                      fontsize=6.5, frameon=False, handlelength=1.1, borderaxespad=0)
        axes[-1, 0].set_xlabel(
            "device_us — width is time; a level's children fill it exactly", fontsize=8)
        fig.suptitle(f"{case_label(one[0])} — plan node / recipe step / cuDF call, "
                     f"execution {run}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = out_dir / "icicle" / f"{safe(case_label(one[0]))}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        made.append(path)
    return made


def draw_icicle(ax, nodes):
    """Three rows, each child starting where the previous sibling ended.

    Returns the total and the legend handles — one per plan node, in plan order, carrying
    the microseconds and the share. The legend is not decoration: widths here span four
    orders of magnitude, so most frames are too narrow for their own text, and a colour
    with no name is a frame the reader can see and cannot identify.
    """
    import matplotlib.patches as mpatches

    cmap = plt.get_cmap("tab20")
    total = sum(t for steps in nodes.values()
                for calls in steps.values() for _, t in calls)
    ax.set_xlim(0, max(1, total))
    legend = []
    x = 0
    for ci, node in enumerate(sorted(nodes)):
        steps = nodes[node]
        node_us = sum(t for calls in steps.values() for _, t in calls)
        colour = cmap(ci % 20)
        legend.append(mpatches.Patch(
            color=colour,
            label=f"{node[0]} {node[1]}  {node_us}us  {100 * node_us / max(1, total):.2f}%"))
        bar(ax, x, 0, node_us, f"{node[0]} {node[1]}", colour, 0.9, total)
        sx = x
        for step in sorted(steps):
            calls = sorted(steps[step])
            step_us = sum(t for _, t in calls)
            bar(ax, sx, 1, step_us, f"#{step[0]} {step[1]}", lighten(colour, 0.35),
                0.9, total)
            cx = sx
            for call_index, t in calls:
                bar(ax, cx, 2, t, f"c{call_index}", lighten(colour, 0.62), 0.9, total)
                cx += t
            sx += step_us
        x += node_us
    ax.set_ylim(2.6, -0.6)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["plan node", "recipe step", "cuDF call"], fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    return total, legend


def bar(ax, x, y, width, label, colour, height, total):
    ax.barh([y], [width], left=[x], height=height, color=colour,
            edgecolor="white", linewidth=0.6)
    # Only where the text fits. A label wider than its bar overflows into its neighbour
    # and reads as that neighbour's -- the one failure mode of an icicle that makes it
    # actively misleading rather than merely dense.
    if total and width > 0.055 * total:
        ax.text(x + width / 2, y, label, ha="center", va="center", fontsize=6.2,
                clip_on=True)


def lighten(rgba, amount):
    r, g, b = rgba[:3]
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount, 1.0)


# ---------------------------------------------------------------------------
# hbm/ — traffic against the bytes this side counted
# ---------------------------------------------------------------------------

def plot_hbm(hbm_rows, rows, out_dir):
    """hbm_bytes against out_bytes, and the bandwidth that falls out of the pair.

    Two different runs meet here, and this function is where they meet: the traffic is a
    capture's, `out_bytes` is a clean run's, and neither file holds both. That join is
    the reason the record carries a tuple at all, so a panel drawing both is the one
    picture that proves it works — see the key below for what the join may and may not
    carry across the two runs.

    An hbm row with no matching record row is not dropped quietly. The two files come
    from different runs of the same cases, so a miss means they are not the same cases —
    a filter that differed, a capture from before a rename — and a plot silently drawn
    from the half that matched would look exactly like a complete one.

    Calls under ten metric samples are drawn hollow. The integral over a call shorter
    than the sampling period is an estimate from too few points, and a reader who cannot
    see which points those are will fit a line through them.
    """
    if not hbm_rows:
        return []

    # Keyed by the CALL, deliberately WITHOUT `run_index`.
    #
    # The two files are different runs, so execution 3 of one and execution 3 of the
    # other are not the same event and pairing them would be arbitrary. What is joined
    # here does not depend on which execution it came from: `out_bytes` is a property of
    # the call -- same plan, same data, same output -- and the check below is what makes
    # that a fact rather than an assumption. Measured on this record: constant across all
    # ten executions of all 42 calls, while `device_us` moves 7% at the median and 36% at
    # the worst.
    #
    # So NOTHING here crosses the two runs on a per-execution basis. The bandwidth panel
    # divides an hbm row's bytes by that same row's `device_busy_us`, both from the
    # capture; the scatter puts the capture's bytes against a constant of the call. A
    # microsecond from the clean run never meets a byte from the captured one.
    key = lambda r: tuple(r[c] for c in CALL)
    logical = {}
    for r in rows:
        seen = logical.setdefault(key(r), r["out_bytes"])
        if seen != r["out_bytes"]:
            sys.exit(
                f"call {key(r)} reports out_bytes {seen} and {r['out_bytes']} in "
                "different executions. The join below takes it as a property of the "
                "call; it is not, and every hbm point would be against whichever "
                "execution happened to be read first."
            )
    missed = [r for r in hbm_rows if key(r) not in logical]
    if missed:
        sys.exit(
            f"{len(missed)} of {len(hbm_rows)} hbm rows join to no record row, the first "
            f"being {key(missed[0])}. The two files are not from the same cases; a plot "
            "drawn from the half that matched would look complete."
        )
    for r in hbm_rows:
        r["out_bytes"] = logical[key(r)]
    thick = [r for r in hbm_rows if int(r["samples"]) >= 10]
    thin = [r for r in hbm_rows if int(r["samples"]) < 10]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for group, style in ((thick, {"s": 18, "alpha": 0.7}),
                         (thin, {"s": 18, "facecolors": "none", "edgecolors": "grey"})):
        pts = [(num(r, "out_bytes"), int(r["hbm_bytes"])) for r in group]
        pts = [(x, y) for x, y in pts if x and y]
        if pts:
            axes[0].scatter([p[0] for p in pts], [p[1] for p in pts], **style)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("out_bytes (what this side counted)", fontsize=8)
    axes[0].set_ylabel("hbm_bytes (what the device moved)", fontsize=8)
    axes[0].set_title("Traffic vs logical output — hollow: under 10 samples", fontsize=9)
    axes[0].grid(alpha=0.25, which="both")

    by_kind = collections.defaultdict(list)
    for r in thick:
        us = num(r, "device_busy_us")
        if us:
            by_kind[r["recipe_kind"]].append(int(r["hbm_bytes"]) / 1e9 / (us / 1e6))
    if by_kind:
        kinds = sorted(by_kind)
        axes[1].boxplot([by_kind[k] for k in kinds], showfliers=False)
        axes[1].set_xticks(range(1, len(kinds) + 1))
        axes[1].set_xticklabels(kinds)
        axes[1].set_ylabel("GB/s while the device was busy", fontsize=8)
        axes[1].set_title("Achieved bandwidth per step kind", fontsize=9)
        axes[1].tick_params(axis="x", labelrotation=30, labelsize=7)
        for label in axes[1].get_xticklabels():
            label.set_horizontalalignment("right")
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].set_axis_off()
    fig.tight_layout()
    path = out_dir / "hbm" / "traffic.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# index.html
# ---------------------------------------------------------------------------

SECTIONS = [
    ("icicle", "Icicle: plan node → recipe step → cuDF call",
     "The record's three-level tuple, drawn as three levels. Width is device time; a "
     "level's children sum to it. One real execution — the one whose total is the "
     "median — because per-call medians give a picture belonging to no execution."),
    ("query", "Where a case's time went",
     "Left: device time per plan node. Right: the three time terms side by side, NEVER "
     "summed — under CUDA events the host submission interval contains the device one, "
     "so adding them describes no interval."),
    ("load", "Load — the scan, alone",
     "read_parquet: decompression, page decode, PCIe. Kept apart from compute on "
     "purpose: one axis holding both draws a picture about two different machines."),
    ("compute", "Compute — everything else",
     "Arithmetic over resident columns. Log-log, because the model's claim is "
     "t = a·bytes, a straight line of slope 1 whatever a is — so the eye checks the "
     "shape without knowing the coefficient."),
    ("spread", "What a median discards",
     "Each execution's device_us as a ratio to its own call's median. A kind with a long "
     "tail is one whose coefficient would be fitted to scheduling noise."),
    ("hbm", "HBM traffic",
     "Times from a clean run, traffic from a capture under GPU memory counters, joined "
     "on the tuple. The counters cost the query ~7%, which is why they are two runs."),
]

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>peacockdb calibration</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1200px;
        color: #222; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
 p.note {{ color: #555; max-width: 70ch; }}
 figure {{ margin: 1rem 0; }} img {{ max-width: 100%; border: 1px solid #ddd; }}
 figcaption {{ font: 12px monospace; color: #666; }}
 .empty {{ color: #999; font-style: italic; }}
</style>
<h1>peacockdb cost-model calibration</h1>
<p class="note">Every picture here comes from one call to
<code>scripts/calibration/plot.py</code>. Sources: {sources}</p>
{body}
"""


def write_index(out_dir, made, sources):
    """One page, images as files beside it rather than data-URIs.

    Thirty-odd panels inline as base64 would be a page of tens of megabytes that no
    browser opens comfortably and no diff can read. Relative paths keep it openable
    straight off the filesystem, which is the whole requirement.
    """
    by_dir = collections.defaultdict(list)
    for path in made:
        by_dir[path.parent.name].append(path)
    body = []
    for name, title, note in SECTIONS:
        body.append(f"<h2>{html.escape(title)}</h2>")
        body.append(f'<p class="note">{html.escape(note)}</p>')
        paths = sorted(by_dir.get(name, []))
        if not paths:
            body.append('<p class="empty">nothing drawn — no rows of this kind in the '
                        'record given</p>')
            continue
        for path in paths:
            rel = f"{name}/{path.name}"
            body.append(f'<figure><img src="{rel}" alt="{html.escape(rel)}">'
                        f'<figcaption>{html.escape(rel)}</figcaption></figure>')
    page = PAGE.format(sources=html.escape(", ".join(str(s) for s in sources)),
                       body="\n".join(body))
    path = out_dir / "index.html"
    path.write_text(page)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="append", required=True,
                    help="a records.tsv; repeat for several (sf1 and sf40)")
    ap.add_argument("--hbm", action="append", default=[],
                    help="an hbm.tsv from nsys_hbm.py; optional")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rows = []
    for path in args.record:
        got = read_tsv(path)
        require(got, path, *CALL, "run_index", "recipe_kind", "device_us", "out_bytes")
        rows += got
    hbm = []
    for path in args.hbm:
        got = read_tsv(path)
        require(got, path, *CALL, "run_index", "hbm_bytes", "samples")
        hbm += got

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_call = executions(rows)

    made = []
    made += plot_icicle(rows, out_dir)
    made += plot_queries(rows, out_dir)
    made += plot_kinds(rows, out_dir, "load")
    made += plot_kinds(rows, out_dir, "compute")
    made += plot_spread(by_call, out_dir)
    made += plot_hbm(hbm, rows, out_dir)
    index = write_index(out_dir, made, args.record + args.hbm)

    print(f"{len(rows)} rows, {len(by_call)} calls, {len(hbm)} hbm rows")
    print(f"{len(made)} panels under {out_dir}")
    print(f"wrote {index}")


if __name__ == "__main__":
    main()
