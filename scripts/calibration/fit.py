#!/usr/bin/env python3
"""Fit the cost model's per-category coefficients from calibration records (#153).

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
Rows are dropped when their hbm_bytes came from too few samples to integrate -- see
nsys_hbm.py, which reports how much traffic that omits.
"""

import argparse
import collections
import statistics
import sys

import numpy as np

MB = 1e6

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
    ap.add_argument("--min-samples", type=int, default=10,
                    help="drop regions whose hbm came from fewer GPU metric samples")
    args = ap.parse_args()

    hbm = {}
    for path in args.hbm:
        for r in read_tsv(path):
            hbm[(r["query"], r["label"], int(r["node_seq"]))] = r

    rows = []
    for path in args.record:
        rows += read_tsv(path)
    if not rows:
        sys.exit("no record rows")

    # A region is (source, dataset, sf, query, label, node_seq, partition), and every
    # component earns its place. Source and dataset because query names are reused: the
    # two sources number nodes independently, and tpch q10 and tpcds q10 are different
    # plans. Partition because the record's row is a region, not a node -- three of the
    # four execute branches time each output partition separately, and collapsing them
    # averages a repartition's p0 prologue into the partitions that never pay it.
    # A key short of this silently medians unrelated regions together, and the result
    # looks like ordinary data.
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["source"], r["dataset"], r["sf"], r["query"], r["label"],
                int(r["node_seq"]), int(r["partition"]))].append(r)

    regions = []
    for (source, _dataset, _sf, query, label, seq, _partition), rs in groups.items():
        # hbm comes from the bare-cuDF source only, where every region is one partition.
        h = hbm.get((query, label, seq)) if source != "peacockdb" else None
        regions.append(dict(
            source=source, category=rs[0]["category"], node_type=rs[0]["node_type"],
            wall_us=statistics.median(int(x["wall_us"]) for x in rs),
            cuda_mb=int(rs[0]["cuda_bytes"]) / MB,
            hbm_mb=int(h["hbm_bytes"]) / MB if h else None,
            samples=int(h["samples"]) if h else None,
        ))

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
    for key, rs in groups.items():
        st = per_source[key[0]]
        st[0] += len(rs)
        st[1] += 1
        st[2].add(len(rs))
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
