#!/usr/bin/env python3
"""Fit the cost model's per-category coefficients from calibration records.

    walltime = const + A * cuda_bytes + B * hbm_bytes

fitted per cost category, where cuda_bytes is what cost_model.conf charges the category
for and hbm_bytes is what the device actually moved.

    scripts/calibration/fit.py --record records.tsv --record cudf-sf40.tsv \
                              --hbm cudf-sf40-hbm.tsv --out fit.tsv

Emits a data file rather than a report: a report that refits every time it renders cannot
be re-rendered without rerunning the fit, and then a wording change and a coefficient
change are the same edit.

The intercept is not a free parameter, it is the check. Both constants in the model are
per-region intercepts and therefore perfectly collinear with each other, so the plan was
to measure them and fit only the slopes on the residual. An intercept surviving that is
either a constant the measurement missed or a slope the model does not have.

Each category is fitted on both regressors and on each one alone, and all three go in
the file under a `model` column. The two are not independent -- what a region is charged
and what it moved rise together -- so on a collinear category the joint fit's split
between A and B is arbitrary along the collinear direction while its R2 stays high. One
fit cannot show that; a coefficient that changes sign or magnitude across the variants
can, and the reader needs to see it before quoting either number.

Regressors are scaled to megabytes before fitting. In bytes the design matrix pairs a
column of ones with a column near 1e9, and the resulting condition number reports the
choice of unit rather than anything about the data.

Regions are the unit, not record rows: a benchmark repeats each region and those rows
are one measurement observed several times, so they collapse to their median first.
Which rows are repeats of which is not a column -- the record carries neither a partition
nor a run index -- it is recoverable from row order, and `region_groups` recovers it.
The cost category is likewise no longer a column and is read out of cost_model.conf.
Rows are dropped when their hbm_bytes came from too few samples to integrate -- see
nsys_hbm.py, which reports how much traffic that omits.
"""

import argparse
import collections
import statistics
import sys

import numpy as np

MB = 1e6

# What a node type whose name is in no cost_model.conf line is fitted under. Kept separate
# rather than skipped: a type missing from the taxonomy is a thing to notice, and dropping
# its regions silently would make the fit look complete while it is not.
UNBINNED = "UNBINNED"

# Fitting a k-parameter model needs more than k points to say anything about the fit's
# uncertainty; at exactly k it interpolates and reports R2 = 1 having learnt nothing.
MIN_DOF = 3


def read_tsv(path):
    rows, header = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if header is None:
                header = f
                continue
            rows.append(dict(zip(header, f)))
    return rows


def read_categories(conf):
    """node_type -> category, from cost_model.conf.

    The record used to carry the category as a column and deliberately no longer does: it
    is a lookup, and a copy of a lookup is a copy of a taxonomy as it stood on the day of
    the run. Reading it here is what lets an old record be refitted under the current one.
    """
    out = {}
    for line in open(conf):
        f = line.split("#", 1)[0].split()
        if len(f) < 3:
            continue
        for node in f[2].split(","):
            out[node] = f[0]
    return out


def split_runs(rows):
    """Rows of one (query, label) -> (regions per execution, executions).

    A benchmark repeats an identical region sequence once per measured execution, so the
    sequence has a period and the period is the number of regions in one execution. No
    period divides the length only when there is one execution, which is what a
    correctness run writes.

    The period is matched on the node identity AND on its row and byte counts, which are
    deterministic across executions of one plan. Matching node_seq alone would find a
    false period inside a partitioned plan, where one node contributes several
    consecutive regions.
    """
    def key(r):
        return (r["node_seq"], r["node_type"], r["in_rows"], r["in_bytes"],
                r["out_rows"], r["out_bytes"], r["cuda_bytes"])

    keys = [key(r) for r in rows]
    n = len(keys)
    for period in range(1, n + 1):
        if n % period:
            continue
        if all(keys[i] == keys[i % period] for i in range(n)):
            return period, n // period
    return n, 1


def region_groups(rows):
    """Rows of ONE record file -> ([executions of one region], executions-per-key tally).

    A region is (source, dataset, sf, query, label, ordinal within the execution). Every
    component earns its place. Source and dataset because query names are reused: the two
    sources number nodes independently, and tpch q10 and tpcds q10 are different plans.
    The ordinal rather than node_seq because the record's row is a region, not a node --
    three of the four execute branches time each output partition separately, and keying
    on the node would average a repartition's p0 prologue into the partitions that never
    pay it. A key short of this silently medians unrelated regions together, and the
    result looks like ordinary data.

    One file at a time, never several concatenated: the recovery reads row order, and two
    runs appended end to end are not one period.
    """
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["source"], r["dataset"], r["sf"], r["query"], r["label"])].append(r)

    out, seen = [], collections.Counter()
    for rs in groups.values():
        period, runs = split_runs(rs)
        seen[runs] += 1
        out += [rs[i::period] for i in range(period)]
    return out, seen


def fit_ols(X, y):
    """OLS with an intercept. X columns are already in the units the coefficients report."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    if dof > 0:
        cov = (rss / dof) * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.abs(np.diag(cov)))
    else:
        se = np.full(A.shape[1], float("nan"))
    return beta, se, r2, float(np.linalg.cond(A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="append", required=True,
                    help="calibration record TSV; repeat for several sources")
    ap.add_argument("--hbm", action="append", default=[],
                    help="hbm TSV from nsys_hbm.py; joined on (query, label, node_seq)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", default="testdata/cost_model.conf",
                    help="cost_model.conf the categories are read from")
    ap.add_argument("--min-samples", type=int, default=10,
                    help="drop regions whose hbm came from fewer GPU metric samples")
    args = ap.parse_args()

    hbm = {}
    for path in args.hbm:
        for r in read_tsv(path):
            hbm[(r["query"], r["label"], int(r["node_seq"]))] = r

    categories = read_categories(args.conf)

    regions = []
    for path in args.record:
        rows = read_tsv(path)
        if not rows:
            sys.exit(f"{path}: no record rows")
        groups, seen = region_groups(rows)
        # One file is one run, so its (query, label) groups must agree on how many
        # executions they hold. Disagreement means the period found is not the execution
        # count, and every median below would be taken over rows that are not repeats of
        # each other -- which no later number would reveal.
        if len(seen) != 1:
            sys.exit(f"{path}: derived execution counts disagree: {dict(seen)}. "
                     "Row order does not carry runs the way the format says it does.")
        for rs in groups:
            head = rs[0]
            # hbm comes from the bare-cuDF source only, where every region is one
            # partition.
            h = (hbm.get((head["query"], head["label"], int(head["node_seq"])))
                 if head["source"] != "peacockdb" else None)
            regions.append(dict(
                source=head["source"],
                category=categories.get(head["node_type"], UNBINNED),
                node_type=head["node_type"],
                execs=len(rs),
                wall_us=statistics.median(int(x["wall_us"]) for x in rs),
                cuda_mb=int(head["cuda_bytes"]) / MB,
                hbm_mb=int(h["hbm_bytes"]) / MB if h else None,
                samples=int(h["samples"]) if h else None,
            ))
    if not regions:
        sys.exit("no record rows")

    unbinned = sorted({r["node_type"] for r in regions if r["category"] == UNBINNED})
    if unbinned:
        print(f"!! not in {args.conf}: {unbinned} — fitted as {UNBINNED}", file=sys.stderr)

    out = []
    by = collections.defaultdict(list)
    for r in regions:
        by[(r["source"], r["category"])].append(r)

    for (source, category), rs in sorted(by.items()):
        thin = [r for r in rs if r["samples"] is not None and r["samples"] < args.min_samples]
        usable = [r for r in rs if r not in thin]
        has_hbm = bool(usable) and all(r["hbm_mb"] is not None for r in usable)

        y = np.array([r["wall_us"] for r in usable], float)
        cuda = np.array([r["cuda_mb"] for r in usable], float)
        hbm_mb = np.array([r["hbm_mb"] for r in usable], float) if has_hbm else None

        base = dict(source=source, category=category, regions=len(rs), dropped=len(thin),
                    n=len(usable), median_cuda_mb=0.0, median_hbm_mb="",
                    median_wall_us=0.0, corr_regressors="")
        if usable:
            base["median_cuda_mb"] = statistics.median(r["cuda_mb"] for r in usable)
            base["median_wall_us"] = statistics.median(r["wall_us"] for r in usable)
            if has_hbm:
                base["median_hbm_mb"] = statistics.median(r["hbm_mb"] for r in usable)
        if has_hbm and cuda.std() > 0 and hbm_mb.std() > 0:
            base["corr_regressors"] = round(float(np.corrcoef(cuda, hbm_mb)[0, 1]), 4)

        # Every applicable variant, not just the full model. Where the regressors are
        # collinear the full model's coefficients are arbitrary along the collinear
        # direction, and no single fit says so out loud -- a coefficient that changes
        # sign or magnitude between variants does. Reporting only the model as specified
        # would present one arbitrary point on that ridge as a measurement.
        variants = []
        if cuda.std() > 0 and has_hbm and hbm_mb.std() > 0:
            variants.append(("cuda+hbm", ["A", "B"], [cuda, hbm_mb]))
        if cuda.std() > 0:
            variants.append(("cuda", ["A"], [cuda]))
        if has_hbm and hbm_mb.std() > 0:
            variants.append(("hbm", ["B"], [hbm_mb]))

        if not variants:
            # Not a failure to report as one: a category whose charged bytes never vary
            # and whose regions take no measurable time is telling us its coefficient is
            # arbitrary, which is a result about the taxonomy.
            out.append(dict(base, model="", A_us_per_mb="", se_A="", B_us_per_mb="",
                            se_B="", intercept_us="", se_intercept="", r2="", cond="",
                            note="no regressor varies"))
            continue

        for model, names, arrays in variants:
            rec = dict(base, model=model, A_us_per_mb="", se_A="", B_us_per_mb="",
                       se_B="", intercept_us="", se_intercept="", r2="", cond="", note="")
            if len(usable) - (len(arrays) + 1) < MIN_DOF:
                rec["note"] = f"only {len(usable)} regions for {len(arrays) + 1} parameters"
            elif y.std() == 0:
                # R^2 would be 0/0. The fit is exact and the coefficients are all zero,
                # but writing that as a fit invites reading a measured slope of zero into
                # what is really an absence of anything to measure.
                rec["note"] = f"wall_us is identically {y[0]:g} over all {len(usable)} regions"
            else:
                beta, se, r2, cond = fit_ols(np.column_stack(arrays), y)
                rec["intercept_us"] = round(float(beta[0]), 3)
                rec["se_intercept"] = round(float(se[0]), 3)
                for i, nm in enumerate(names, start=1):
                    rec[f"{nm}_us_per_mb"] = round(float(beta[i]), 6)
                    rec[f"se_{nm}"] = round(float(se[i]), 6)
                rec["r2"] = round(r2, 4)
                rec["cond"] = round(cond, 1)
            out.append(rec)

    cols = ["source", "category", "model", "regions", "dropped", "n", "median_cuda_mb",
            "median_hbm_mb", "median_wall_us", "A_us_per_mb", "se_A", "B_us_per_mb",
            "se_B", "intercept_us", "se_intercept", "r2", "cond", "corr_regressors", "note"]
    with open(args.out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for rec in out:
            fh.write("\t".join(str(rec[c]) for c in cols) + "\n")

    # How many rows became how many regions, per source. Printed because the collapse is
    # where an incomplete region key hides: it does not fail, it medians unrelated rows
    # together, and the only visible trace is a region count quietly below the row count.
    per_source = collections.defaultdict(lambda: [0, 0, set()])
    for r in regions:
        st = per_source[r["source"]]
        st[0] += r["execs"]
        st[1] += 1
        st[2].add(r["execs"])
    for source, (nrows, nregions, mult) in sorted(per_source.items()):
        print(f"{source}: {nrows} rows -> {nregions} regions, "
              f"rows per region {sorted(mult)}")

    cells = {(r["source"], r["category"]) for r in out}
    fitted = {(r["source"], r["category"]) for r in out if r["r2"] != ""}
    print(f"{len(cells)} (source, category) cells, {len(fitted)} fitted, {len(out)} fits")
    # Per cell, not per row: a cell contributes several rows, one per model variant.
    dropped = sum({(r["source"], r["category"]): r["dropped"] for r in out}.values())
    if dropped:
        print(f"{dropped} regions dropped for fewer than {args.min_samples} hbm samples")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
