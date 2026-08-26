#!/usr/bin/env python3
"""Draw the calibration record, before anything is fitted to it.

    /usr/bin/python3 scripts/calibration/plot.py \
        --record testdata/calibration/records-sf40.tsv \
        --record testdata/calibration/records-sf1.tsv \
        --out-dir testdata/calibration/plots

A fit answers "what are the coefficients"; it cannot answer "is a line the right shape",
because it returns coefficients either way. These are the pictures that decide whether
fitting is meaningful at all, so nothing here reduces a region to one number before it is
drawn -- every measured execution is a point.

Not matplotlib's default python: this repo's `python3` is a linuxbrew build with neither
numpy nor matplotlib. `/usr/bin/python3` has both.

WHAT IS DRAWN

  nodes/    per node type: wall_us and device_us against out_bytes and in_bytes, log-log,
            one point per execution, coloured by sf. The shape the cost model assumes is
            a straight line through the origin; a per-node-type panel is where it either
            looks like one or does not.
  spread/   per node type: every execution's wall_us as a ratio to its region's median.
            The fit collapses a region's executions to their median, and this is the plot
            that says what that discards.
  query/    per query: the three time terms stacked, and the same total stacked by node
            type. Answers whether the host prologue dominates at sf1 and stops at sf40 --
            the reason the model fits `const_peacock` separately at all.
  hbm/      hbm_bytes against out_bytes, where a capture exists. The fit works on
            hbm_bytes; this is what the difference between the two looks like.

REGIONS, RUNS, AND THE COLUMNS THAT ARE NOT THERE

The record has no partition column and no run index -- both are recoverable from row
ORDER, which is what the format promises instead, and fit.py's `region_groups` is where
that recovery lives. It is imported rather than repeated here. A file whose queries
disagree about how many executions they hold is a file whose order assumption is wrong,
and this script says so rather than plotting.

The cost category is likewise not a column: fit.py reads it out of cost_model.conf, which
is the point of having dropped it -- the taxonomy moves, and old rows should follow it.
"""

import argparse
import collections
import pathlib
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The record's order-to-regions recovery and the category lookup live in fit.py, which is
# the tool that has to keep working. A second copy here would be a second thing to keep
# in step with the format, and the first one to be forgotten is always the one only the
# pictures depend on.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fit import read_categories, region_groups, split_runs  # noqa: E402,F401

# Points at or below this are the measurement's own resolution, not the node's cost. The
# runs report it as sync_floor_us in every .benchmark.txt; it is an argument here because
# the record does not carry it and a plot must not invent one.
DEFAULT_FLOOR_US = 1

HOST_TERMS = ("peacock_host_us", "cudf_host_us")
TIME_TERMS = HOST_TERMS + ("device_us",)


def read_record(path):
    """Record rows as dicts, in file order. Order is data here -- see the module doc."""
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


def regions(rows):
    """Record rows -> one dict per region, each carrying every execution of itself.

    `region_groups` decides what a region is; this only reshapes its output into what the
    panels below want: the identity taken from the first execution, since a plan's node
    types and counts do not move between executions, and every execution's times kept.
    """
    groups, runs_seen = region_groups(rows)
    out = []
    for execs in groups:
        head = execs[0]
        out.append(dict(
            source=head["source"], sf=head["sf"], query=head["query"],
            label=head["label"], node_seq=int(head["node_seq"]),
            node_type=head["node_type"],
            dataset=head["dataset"],
            in_bytes=int(head["in_bytes"]), out_bytes=int(head["out_bytes"]),
            cuda_bytes=int(head["cuda_bytes"]),
            execs=[{t: int(e[t]) for t in TIME_TERMS + ("wall_us",)} for e in execs],
        ))
    return out, runs_seen


def scatter(ax, points, floor_us, title, xlabel, ylabel):
    """One log-log panel, coloured by sf, with the measurement floor drawn across it.

    A log axis cannot show a zero, and zeros are common here -- a passthrough node does
    no device work at all. They are counted in the title rather than dropped quietly,
    because a panel that plots 3 of 1110 points and says nothing looks like a node type
    with three measurements.
    """
    by_sf = collections.defaultdict(list)
    for sf, x, y in points:
        by_sf[sf].append((x, y))
    below = sum(1 for _, _, y in points if y <= floor_us)
    plotted = 0
    for sf in sorted(by_sf, key=float):
        xy = [(x, y) for x, y in by_sf[sf] if x > 0 and y > 0]
        plotted += len(xy)
        ax.scatter([x for x, _ in xy], [y for _, y in xy], s=6, alpha=0.5,
                   label=f"sf{sf} ({len(xy)}/{len(by_sf[sf])})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — {plotted}/{len(points)} plotted, "
                 f"{below} at or below {floor_us}us", fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    if not plotted:
        # Ticks off before returning. The early return skips the log scales below, and a
        # linear empty axis is drawn by matplotlib over its own default -0.05..0.05 --
        # numbers that are not data, on a panel whose whole point is that there is none.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "nothing positive on both axes", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="crimson")
        return
    ax.axhline(floor_us, color="crimson", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=6)


def plot_nodes(regs, categories, floor_us, out_dir):
    by_type = collections.defaultdict(list)
    for r in regs:
        by_type[r["node_type"]].append(r)

    written = []
    for node_type, rs in sorted(by_type.items()):
        cat = categories.get(node_type, "UNBINNED")
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (yterm, xterm) in zip(axes.flat, [
            ("wall_us", "out_bytes"), ("wall_us", "in_bytes"),
            ("device_us", "out_bytes"), ("device_us", "in_bytes"),
        ]):
            pts = [(r["sf"], r[xterm], e[yterm]) for r in rs for e in r["execs"]]
            scatter(ax, pts, floor_us, f"{yterm} vs {xterm}", xterm, yterm)
        fig.suptitle(f"{node_type}  [{cat}]  {len(rs)} regions, "
                     f"{sum(len(r['execs']) for r in rs)} executions")
        fig.tight_layout()
        path = out_dir / "nodes" / f"{cat}.{node_type}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def plot_spread(regs, floor_us, out_dir):
    """Every execution as a ratio to its region's median -- what the median discards.

    Regions whose median is at the floor are dropped, not plotted: a ratio between two
    numbers that are both the clock's resolution says nothing about the node, and on a
    log axis it is the loudest thing in the picture.
    """
    by_type = collections.defaultdict(list)
    for r in regs:
        med = statistics.median(e["wall_us"] for e in r["execs"])
        if med <= floor_us or len(r["execs"]) < 2:
            continue
        by_type[r["node_type"]].append((med, [e["wall_us"] / med for e in r["execs"]]))

    written = []
    for node_type, rs in sorted(by_type.items()):
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = [med for med, ratios in rs for _ in ratios]
        ys = [ratio for _, ratios in rs for ratio in ratios]
        ax.scatter(xs, ys, s=6, alpha=0.4)
        ax.axhline(1.0, color="crimson", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("region median wall_us")
        ax.set_ylabel("execution / median")
        worst = max((max(ratios) for _, ratios in rs), default=float("nan"))
        ax.set_title(f"{node_type}: {len(rs)} regions above the floor, "
                     f"worst execution {worst:.2f}x its median", fontsize=10)
        ax.grid(True, lw=0.3, alpha=0.4)
        fig.tight_layout()
        path = out_dir / "spread" / f"{node_type}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def query_order(query):
    """q2 before q10. Alphabetical order puts them the other way round."""
    head = query.rstrip("0123456789")
    tail = query[len(head):]
    return (head, int(tail) if tail else -1)


def plot_queries(regs, out_dir):
    """Per query: how long it took, and what the time was made of.

    Three panels rather than one, because the totals span three decades and composition
    does not survive that. Absolute time gets a log axis of its own; the two composition
    panels are shares of it. A stack drawn on a log axis is not a stack -- a segment's
    height there is not its contribution -- which is the mistake this shape avoids.

    Median over executions first: this plot is about composition, and ten overlapping
    stacks answer no question `spread/` does not answer better.
    """
    # Split by dataset as well as by run: tpch and tpcds are different corpora, and one
    # figure holding both is a hundred bars wide and legible at no size.
    by_run = collections.defaultdict(list)
    for r in regs:
        by_run[(r["source"], r["dataset"], r["sf"], r["label"])].append(r)

    written = []
    for (source, dataset, sf, label), rs in sorted(by_run.items()):
        queries = sorted({r["query"] for r in rs}, key=query_order)
        per_query = {q: [r for r in rs if r["query"] == q] for q in queries}
        med = {id(r): {t: statistics.median(e[t] for e in r["execs"])
                       for t in TIME_TERMS + ("wall_us",)} for r in rs}

        # Only the two host terms are stacked: they sum to wall_us by definition, while
        # device_us is a concurrent span of the same clock — the host waits through most
        # of it. Stacking all three makes every query look half device, which is an
        # artefact of adding a number to the thing that contains it.
        terms = {t: [sum(med[id(r)][t] for r in per_query[q]) for q in queries]
                 for t in HOST_TERMS}
        node_types = sorted({r["node_type"] for r in rs})
        by_node = {nt: [sum(med[id(r)]["wall_us"] for r in per_query[q]
                            if r["node_type"] == nt) for q in queries]
                   for nt in node_types}
        wall = [sum(med[id(r)]["wall_us"] for r in per_query[q]) for q in queries]

        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1, figsize=(max(9, len(queries) * 0.42), 11))
        ax0.bar(queries, wall, color="0.4")
        ax0.set_yscale("log")
        ax0.set_ylabel("wall_us")
        ax0.set_title("host time over all regions", fontsize=10)

        for ax, series, title in (
            (ax1, terms, "share of host time (wall_us) by term"),
            (ax2, by_node, "share of wall_us by node type"),
        ):
            totals = [sum(series[name][i] for name in series) or 1.0
                      for i in range(len(queries))]
            bottom = [0.0] * len(queries)
            for name, values in series.items():
                share = [100 * v / t for v, t in zip(values, totals)]
                ax.bar(queries, share, bottom=bottom, label=name)
                bottom = [b + v for b, v in zip(bottom, share)]
            ax.set_ylabel("% of the query")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=6, ncol=3)

        device = [sum(med[id(r)]["device_us"] for r in per_query[q]) for q in queries]
        ax1.plot(queries, [100 * d / w if w else 0 for d, w in zip(device, wall)],
                 "k.", ms=4, label="device_us / wall_us (concurrent, not a share)")
        ax1.legend(fontsize=6, ncol=3)

        for ax in (ax0, ax1, ax2):
            ax.tick_params(axis="x", rotation=90, labelsize=7)
            ax.grid(True, axis="y", lw=0.3, alpha=0.4)
        fig.suptitle(f"{source} {dataset}.sf{sf} [{label}] — median over executions, "
                     "summed over regions")
        fig.tight_layout()
        path = out_dir / "query" / f"{source}.{dataset}.sf{sf}.{label}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def plot_hbm(regs, hbm, out_dir):
    pts = []
    for r in regs:
        h = hbm.get((r["query"], r["label"], r["node_seq"]))
        if h and r["out_bytes"] > 0 and int(h["hbm_bytes"]) > 0:
            pts.append((r["node_type"], r["out_bytes"], int(h["hbm_bytes"])))
    if not pts:
        return []

    fig, ax = plt.subplots(figsize=(8, 7))
    by_type = collections.defaultdict(list)
    for nt, x, y in pts:
        by_type[nt].append((x, y))
    for nt, xy in sorted(by_type.items()):
        ax.scatter([x for x, _ in xy], [y for _, y in xy], s=8, alpha=0.6, label=nt)
    lo = min(min(x, y) for _, x, y in pts)
    hi = max(max(x, y) for _, x, y in pts)
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--", label="hbm = out_bytes")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("out_bytes")
    ax.set_ylabel("hbm_bytes")
    ax.set_title(f"what the region is charged vs what the device moved (n={len(pts)})",
                 fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(fontsize=6)
    fig.tight_layout()
    path = out_dir / "hbm" / "hbm-vs-out-bytes.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="append", required=True,
                    help="calibration record TSV; repeat for several runs")
    ap.add_argument("--hbm", action="append", default=[],
                    help="hbm TSV from nsys_hbm.py; joined on (query, label, node_seq)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conf", default="testdata/cost_model.conf")
    ap.add_argument("--floor-us", type=int, default=DEFAULT_FLOOR_US,
                    help="sync_floor_us of the run, as its .benchmark.txt reports it")
    args = ap.parse_args()

    categories = read_categories(args.conf)
    hbm = {}
    for path in args.hbm:
        for r in read_record(path):
            hbm[(r["query"], r["label"], int(r["node_seq"]))] = r

    regs, runs_seen = [], collections.Counter()
    for path in args.record:
        rows = read_record(path)
        rs, seen = regions(rows)
        # One file is one run, so every (query, label) in it was executed the same number
        # of times. Disagreement means the period search found something that is not the
        # execution count, and every plot below would be drawing a fiction.
        if len(seen) != 1:
            sys.exit(f"{path}: derived execution counts disagree: {dict(seen)}. "
                     "Row order does not carry runs the way the format says it does.")
        regs += rs
        runs_seen += seen
        print(f"{path}: {len(rows)} rows, {len(rs)} regions, "
              f"{next(iter(seen))} executions each")

    unbinned = sorted({r["node_type"] for r in regs} - set(categories))
    if unbinned:
        print(f"!! not in {args.conf}: {unbinned} — plotted as UNBINNED")

    out_dir = pathlib.Path(args.out_dir)
    written = (plot_nodes(regs, categories, args.floor_us, out_dir)
               + plot_spread(regs, args.floor_us, out_dir)
               + plot_queries(regs, out_dir)
               + plot_hbm(regs, hbm, out_dir))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
