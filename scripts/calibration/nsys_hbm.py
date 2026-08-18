#!/usr/bin/env python3
"""hbm_bytes per timed region, from an Nsight Systems capture (#153).

The calibration record deliberately has no hbm_bytes column: nothing inside the process
can count HBM traffic, so it comes from a profiled run and joins back on
(query, label, node_seq). This is that join's producer.

    nsys profile --trace=nvtx,cuda --sample=none --cpuctxsw=none \
                 --gpu-metrics-device=0 --gpu-metrics-set=<arch> \
                 --gpu-metrics-frequency=20000 -o cap -- <binary>
    nsys export --type=sqlite --force-overwrite=true -o cap.sqlite cap.nsys-rep
    scripts/calibration/nsys_hbm.py --capture cap.sqlite --record run.tsv --out hbm.tsv

Nsight's general metric set reports DRAM traffic as a percentage of the device's peak
bandwidth, not as bytes, so bytes are the integral of that percentage over the sampling
period times a peak the caller supplies. That makes --peak-bw part of the measurement
rather than a display detail, and a wrong one is a silent scale error on every row.
Checked once against a region whose traffic is known exactly -- a cast of N rows from an
8-byte to a 16-byte type has to read 8N and write 16N -- which came out inside 2.4% on
both directions at the default peak.

Metric ids are looked up by name because they are a property of the capture, not of the
tool: a run whose ids were assumed rather than read produced a read column holding what
was really the write, and the error was only visible because a cast's traffic is known
in advance. Nothing downstream of here could have caught it.

The remaining checks are invariants a healthy capture cannot violate -- no sample above
100% of peak, no gap in the sampling cadence -- and they are asserted rather than trusted
because both failures degrade quietly: an interrupted collection leaves later regions
with an hbm_bytes of zero, which is what a region that moved no data also looks like.
"""

import argparse
import bisect
import sqlite3
import statistics
import sys

# nsys metric ids within the general set. Names are checked against the capture, since a
# different --gpu-metrics-set numbers them differently.
DRAM_READ = "DRAM Read Bandwidth"
DRAM_WRITE = "DRAM Write Bandwidth"

# NVTX_EVENTS.eventType. 59 is a push/pop range, 75 the domain's own name record.
NVTX_PUSHPOP_RANGE = 59
NVTX_DOMAIN_CREATE = 75


def metric_id(conn, name):
    row = conn.execute(
        "select metricId from TARGET_INFO_GPU_METRICS where metricName = ?", (name,)
    ).fetchone()
    if row is None:
        have = [r[0] for r in conn.execute("select metricName from TARGET_INFO_GPU_METRICS")]
        sys.exit(f"capture has no metric {name!r}; it has {have}")
    return row[0]


def domain_id(conn, name):
    row = conn.execute(
        f"""select e.domainId from NVTX_EVENTS e left join StringIds s on s.id = e.textId
            where e.eventType = {NVTX_DOMAIN_CREATE} and coalesce(e.text, s.value) = ?""",
        (name,),
    ).fetchone()
    if row is None:
        sys.exit(f"capture has no NVTX domain {name!r} -- was the binary run with ranges on?")
    return row[0]


def samples(conn, mid):
    return list(
        conn.execute(
            "select timestamp, value from GPU_METRICS where metricId = ? order by timestamp",
            (mid,),
        )
    )


def busy_union(conn):
    """Disjoint device-busy intervals, merged across all streams.

    Merged rather than summed: the capture has 32 streams and the parquet reader keeps
    several of them busy at once, so summing durations would count concurrent work twice
    and report a region as busier than it was long.
    """
    spans = []
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ):
        spans += list(conn.execute(f"select start, end from {table}"))
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="sqlite export of the .nsys-rep")
    ap.add_argument("--record", required=True, help="record TSV written by the same run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="peacockdb", help="NVTX domain holding the regions")
    ap.add_argument(
        "--peak-bw",
        type=float,
        default=4.8e12,
        help="device peak HBM bandwidth in bytes/s; default is H200 SXM",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(args.capture)
    dom = domain_id(conn, args.domain)
    ranges = list(
        conn.execute(
            f"""select e.start, e.end, coalesce(e.text, s.value) from NVTX_EVENTS e
                left join StringIds s on s.id = e.textId
                where e.eventType = {NVTX_PUSHPOP_RANGE} and e.domainId = ?
                order by e.start""",
            (dom,),
        )
    )

    read = samples(conn, metric_id(conn, DRAM_READ))
    write = samples(conn, metric_id(conn, DRAM_WRITE))
    if not read:
        sys.exit("capture has no GPU metric samples -- was --gpu-metrics-device given?")
    stamps = [t for t, _ in read]
    deltas = [b - a for a, b in zip(stamps, stamps[1:])]
    period_ns = statistics.median(deltas)
    gaps = sum(1 for d in deltas if d > 2 * period_ns)
    if gaps:
        sys.exit(
            f"{gaps} gaps in GPU metric sampling at a median period of {period_ns / 1e3:.1f}us: "
            "collection stopped or stalled, so later regions would read as zero-traffic. "
            "Lower --gpu-metrics-frequency and recapture."
        )
    over = max(r[1] + w[1] for r, w in zip(read, write))
    if over > 100:
        sys.exit(
            f"a sample reports {over}% of peak DRAM bandwidth, which the device cannot do. "
            "The sampling frequency is above what it sustains; lower it and recapture."
        )

    rows = []
    with open(args.record) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[0] == "source":
                continue
            rows.append(f)
    if len(rows) != len(ranges):
        sys.exit(
            f"{len(ranges)} NVTX ranges against {len(rows)} record rows. The two come from "
            "one run and pair up in execution order; a mismatch means they do not."
        )

    merged = busy_union(conn)
    starts = [m[0] for m in merged]

    def integral(series, a, b):
        lo, hi = bisect.bisect_left(stamps, a), bisect.bisect_right(stamps, b)
        pct = sum(v for _, v in series[lo:hi])
        return pct / 100.0 * args.peak_bw * period_ns / 1e9, hi - lo

    def busy_ns(a, b):
        total = 0
        i = max(0, bisect.bisect_right(starts, a) - 1)
        while i < len(merged) and merged[i][0] < b:
            total += max(0, min(merged[i][1], b) - max(merged[i][0], a))
            i += 1
        return total

    out = []
    for (a, b, text), f in zip(ranges, rows):
        # The range is named "<seq> <op>" by the recorder; seq is the join key and this
        # asserts the positional pairing above rather than assuming it.
        seq = int(text.split()[0])
        if seq != int(f[5]):
            sys.exit(f"range {text!r} pairs with record row for node_seq {f[5]}")
        r, n = integral(read, a, b)
        w, _ = integral(write, a, b)
        busy = busy_ns(a, b)
        out.append(
            (f[3], f[4], seq, round(r), round(w), round(r + w), n,
             busy // 1000, (b - a - busy) // 1000)
        )

    with open(args.out, "w") as fh:
        fh.write(
            "query\tlabel\tnode_seq\thbm_read_bytes\thbm_write_bytes\thbm_bytes\t"
            "samples\tdevice_busy_us\tdevice_idle_us\n"
        )
        for row in out:
            fh.write("\t".join(str(x) for x in row) + "\n")

    thin = [r for r in out if r[6] < 10]
    total = sum(r[5] for r in out)
    print(f"{len(out)} regions, {total / 1e9:.2f} GB HBM, sampling period {period_ns / 1e3:.1f}us")
    print(
        f"{len(thin)} regions under 10 samples, carrying "
        f"{sum(r[5] for r in thin) / 1e9:.3f} GB ({100 * sum(r[5] for r in thin) / total:.2f}%) "
        "-- too few to integrate, and the fit should weight or drop them"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
