#!/usr/bin/env python3
"""What a timed region spends its time on, one level down, from an Nsight capture.

    PCK_BENCH_NSYS=1 PCK_TEST_FILTER=bench_tpch_sf40_q9_ \
        ./scripts/build-test-shadgpu.sh --run-benchmarks --pull-benchmarks
    scripts/calibration/nsys_calls.py --capture testdata/calibration/capture.sqlite \
        --out testdata/calibration/calls.tsv

The calibration record has one number per region and one input size for it. For a hash
join that input size is the SUM over both children -- `input_for` adds every child's
bytes together -- and a join's cost does not depend on its two sides the same way, so no
coefficient fitted against that sum can be right. This reads the split back out of a
capture instead of adding columns to the record: libcudf pushes an NVTX range around
every public call it makes, so the build side (`hash_join`), the probe (`inner_join`) and
the materialisation (`gather`) are already three separate spans inside our region. The
engine is not touched, and neither is the format.

WHAT A ROW IS

One row per (node_seq, call, depth), aggregated over the executions in the capture. The
region is the `p<k>` range our own domain pushes per output partition; `depth` is how
deep the call sits inside it among calls of the same domain, so **depth 0 rows partition
the region and deeper rows break those down**. Summing across depths double-counts.

Every region also gets a `(unattributed)` row at depth 0: the part of it that is inside
no call of the traced domain at all. Without it a reader has no way to tell a region
explained by its calls from one where they cover a third of the span, and the second is
the interesting case -- it means the cost is in our code, not in cuDF's.

HOST AND DEVICE ARE DIFFERENT COLUMNS

`host_us` is the NVTX range itself: the wall time the calling thread spent inside that
cuDF call. `device_us` is the sum of kernel, memcpy and memset durations whose launching
runtime call falls inside the range, joined through CUPTI's correlationId.

The two are not the same measurement and neither replaces the other. A cuDF call that
submits asynchronously and returns has a small host span and a large device one; a call
that synchronizes has host >= device. And `device_us` is a SUM over device operations,
not a union of their spans -- concurrent work on several streams is counted once per
operation. It answers "how much device work did this call cause", not "how long was the
device busy".

WHAT THIS CANNOT SEE

A capture is not a measurement of the unprofiled run. nsys serializes some of what it
traces, and the .benchmark.txt written by a captured run is not comparable with any
other -- which is why the capture is a knob and not part of the ordinary run.

Threads other than the one that pushed the region are not attributed to it: a device
operation launched from a pool thread lands in the `(off-thread)` tally printed at the
end rather than in a region. For the single-threaded execute path that tally should be
the parquet reader's and nothing else.
"""

import argparse
import bisect
import collections
import sqlite3
import statistics
import sys

# NVTX_EVENTS.eventType. 59 is a push/pop range, 75 the domain's own name record.
NVTX_PUSHPOP_RANGE = 59
NVTX_DOMAIN_CREATE = 75

# Our own domain, and the name the region range carries. See node_session.cpp: a node
# pushes "<seq> <PlanNodeKind>" and the per-partition timer pushes "p<k>" inside it, so
# the pair is what a record row calls a region.
OWN_DOMAIN = "peacockdb"

DEVICE_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL",
                 "CUPTI_ACTIVITY_KIND_MEMCPY",
                 "CUPTI_ACTIVITY_KIND_MEMSET")

UNATTRIBUTED = "(unattributed)"


def domain_ids(conn):
    """name -> domainId for every NVTX domain the capture knows."""
    return {
        name: did
        for did, name in conn.execute(
            f"""select e.domainId, coalesce(e.text, s.value) from NVTX_EVENTS e
                left join StringIds s on s.id = e.textId
                where e.eventType = {NVTX_DOMAIN_CREATE}"""
        )
    }


def ranges(conn, domain_id):
    """(start, end, text, tid) for one domain, in start order."""
    return list(conn.execute(
        f"""select e.start, e.end, coalesce(e.text, s.value), e.globalTid
            from NVTX_EVENTS e left join StringIds s on s.id = e.textId
            where e.eventType = {NVTX_PUSHPOP_RANGE} and e.domainId = ?
              and e.end is not null
            order by e.start""",
        (domain_id,),
    ))


def regions(own):
    """Our domain's ranges -> [(node_seq, node_type, partition, start, end, tid)].

    The two levels are told apart by their names rather than by nesting depth: a node
    range is "<seq> <Kind>" and a partition range is "p<k>". Reading the level off the
    name and then checking containment is what catches a capture whose two levels do not
    line up -- a partition range outside every node range means the ranges did not come
    from the code this script thinks they did.
    """
    nodes = [r for r in own if not r[2].startswith("p") or " " in r[2]]
    parts = [r for r in own if r not in nodes]
    starts = [n[0] for n in nodes]

    out = []
    for start, end, text, tid in parts:
        i = bisect.bisect_right(starts, start) - 1
        if i < 0 or nodes[i][1] < end or nodes[i][3] != tid:
            sys.exit(f"partition range at {start} is inside no node range of its thread")
        seq, kind = nodes[i][2].split(" ", 1)
        out.append((int(seq), kind, int(text[1:]), start, end, tid))
    return out


class DeviceWork:
    """Device nanoseconds, addressable by the host interval that launched them.

    A kernel does not carry the NVTX range it belongs to; it carries a correlationId
    pointing back at the CUDA runtime call that launched it. So the join is: sum every
    device operation per correlationId, look up where on the host that launch happened,
    and index those launch points by thread. Asking "how much device work did this
    range cause" is then a range sum over the launch points inside it.

    Attributing at the launch site and not at the kernel's own timestamps is the only
    choice that stays true for asynchronous work: the kernel a call submits may still be
    running after the call returns, and placing it by its own start would credit it to
    whatever range happened to be open on the host at that moment -- a different node.

    A nested call's launches lie inside the outer call's interval too, so an outer range
    counts its children's device work. That is deliberate: it makes `device_us` mean the
    same thing as `host_us`, which is a span and likewise contains its children.
    """

    def __init__(self, conn):
        per_corr = collections.Counter()
        for table in DEVICE_TABLES:
            for corr, ns in conn.execute(
                    f"select correlationId, sum(end - start) from {table} group by 1"):
                per_corr[corr] += ns

        launches = collections.defaultdict(list)
        for start, tid, corr in conn.execute(
                "select start, globalTid, correlationId "
                "from CUPTI_ACTIVITY_KIND_RUNTIME"):
            ns = per_corr.get(corr)
            if ns:
                launches[tid].append((start, ns))

        self.total = sum(per_corr.values())
        self.by_tid = {}
        for tid, rows in launches.items():
            rows.sort()
            starts = [r[0] for r in rows]
            # Prefix sums, so a range sum is two bisects and a subtraction. There are
            # ~200k launches and a region asks about every call it contains; a linear
            # scan per question would be quadratic on the big scans.
            running, total = [0], 0
            for _, ns in rows:
                total += ns
                running.append(total)
            self.by_tid[tid] = (starts, running)

    def span(self, tid, start, end):
        """Device ns launched from `tid` in [start, end)."""
        starts, running = self.by_tid.get(tid, ((), (0,)))
        return (running[bisect.bisect_left(starts, end)]
                - running[bisect.bisect_left(starts, start)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="sqlite export of the .nsys-rep")
    ap.add_argument("--domain", action="append", default=[],
                    help="NVTX domain to break regions down by; repeatable, "
                         "default libcudf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=12,
                    help="how many calls per node to print in the summary")
    args = ap.parse_args()
    call_domains = args.domain or ["libcudf"]

    conn = sqlite3.connect(args.capture)
    doms = domain_ids(conn)
    if OWN_DOMAIN not in doms:
        sys.exit(f"capture has no NVTX domain {OWN_DOMAIN!r}: "
                 "the run needs PEACOCK_NVTX=1, which PCK_BENCH_NSYS sets")
    missing = [d for d in call_domains if d not in doms]
    if missing:
        sys.exit(f"capture has no NVTX domain(s) {missing}; it has {sorted(doms)}")

    regs = regions(ranges(conn, doms[OWN_DOMAIN]))
    if not regs:
        sys.exit("capture has no partition ranges -- nothing was executed under them")

    calls = []
    for name in call_domains:
        calls += [(s, e, f"{name}:{t}" if len(call_domains) > 1 else t, tid)
                  for s, e, t, tid in ranges(conn, doms[name])]
    calls.sort()

    # Per thread, sorted by start: a region asks only about the calls on its own thread,
    # and inside a thread the ranges are properly nested, which is what lets the walk
    # below track depth with a stack instead of testing containment pairwise.
    by_tid = collections.defaultdict(list)
    for c in calls:
        by_tid[c[3]].append(c)
    call_index = {tid: ([c[0] for c in cs], cs) for tid, cs in by_tid.items()}

    device = DeviceWork(conn)

    # One bucket per (region identity, call name, depth), holding a list over the
    # executions in the capture. A list and not a running total: the capture holds the
    # warm-up execution too, and a mean over it and the measured ones is neither.
    per_exec = collections.defaultdict(lambda: collections.defaultdict(
        lambda: [0, 0, 0]))  # exec key -> (call, depth) -> [count, host_ns, device_ns]
    region_span = collections.defaultdict(list)
    seen = collections.Counter()
    in_regions_ns = 0

    for seq, kind, part, r_start, r_end, tid in regs:
        ident = (seq, kind, part)
        run = seen[ident]
        seen[ident] += 1
        region_span[ident].append(r_end - r_start)
        bucket = per_exec[(ident, run)]

        region_dev = device.span(tid, r_start, r_end)
        in_regions_ns += region_dev

        starts, rows = call_index.get(tid, ((), ()))
        lo = bisect.bisect_left(starts, r_start)
        stack = []
        host_covered = dev_covered = 0
        for start, end, text, _ in rows[lo:]:
            if start >= r_end:
                break
            if end > r_end:
                sys.exit(f"call {text!r} straddles the end of region {ident}")
            # The stack holds the end timestamps of the calls still open at `start`;
            # everything that ended earlier is popped, so its height is the depth.
            while stack and stack[-1] <= start:
                stack.pop()
            depth = len(stack)
            dev = device.span(tid, start, end)
            if depth == 0:
                # Only the top level counts towards coverage. A nested call's time is
                # already inside its parent's span, and adding it would make the
                # residual below negative.
                host_covered += end - start
                dev_covered += dev
            slot = bucket[(text, depth)]
            slot[0] += 1
            slot[1] += end - start
            slot[2] += dev
            stack.append(end)

        # The residual: the part of the region that is in no traced call. It exists so
        # that the depth-0 rows add up to the region and a reader can see at a glance
        # how much of a node its cuDF calls actually explain.
        rest = bucket[(UNATTRIBUTED, 0)]
        rest[0] += 1
        rest[1] += (r_end - r_start) - host_covered
        rest[2] += region_dev - dev_covered

    # Median over executions, per (region, call, depth). Median rather than the mean for
    # the same reason the fit takes one: the warm-up execution is in here, and on a first
    # touch of a column the parquet reader does work no later execution repeats.
    rows = []
    keys = {(ident, cd) for (ident, _), b in per_exec.items() for cd in b}
    for ident, (call, depth) in sorted(keys, key=lambda k: (k[0], k[1][1], k[1][0])):
        runs = [per_exec[(ident, r)].get((call, depth), [0, 0, 0])
                for r in range(seen[ident])]
        seq, kind, part = ident
        rows.append(dict(
            node_seq=seq, node_type=kind, partition=part, call=call, depth=depth,
            executions=seen[ident],
            calls_per_exec=statistics.median(r[0] for r in runs),
            host_us=round(statistics.median(r[1] for r in runs) / 1000),
            device_us=round(statistics.median(r[2] for r in runs) / 1000),
            region_us=round(statistics.median(region_span[ident]) / 1000),
        ))

    cols = ["node_seq", "node_type", "partition", "call", "depth", "executions",
            "calls_per_exec", "host_us", "device_us", "region_us"]
    with open(args.out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    runs_seen = set(seen.values())
    print(f"{len(regs)} region ranges, {len(seen)} distinct regions, "
          f"{sorted(runs_seen)} executions each")
    if len(runs_seen) != 1:
        print("!! regions disagree about how many times they ran; the capture holds a "
              "filter that matched more than one case, or an execution died partway")
    print(f"{(device.total - in_regions_ns) / 1e6:.1f} ms of {device.total / 1e6:.1f} ms "
          "of device work was launched outside every region "
          "(reader threads, allocator warm-up, teardown)")

    for ident in sorted(seen):
        top = [r for r in rows if (r["node_seq"], r["node_type"], r["partition"]) == ident
               and r["depth"] == 0]
        top.sort(key=lambda r: -r["host_us"])
        span = top[0]["region_us"] if top else 0
        print(f"\n#{ident[0]} {ident[1]} p{ident[2]}  region {span} us")
        for r in top[:args.top]:
            share = 100 * r["host_us"] / span if span else 0
            print(f"    {r['host_us']:>9} us  {share:5.1f}%  "
                  f"dev {r['device_us']:>8} us  x{r['calls_per_exec']:<5g} {r['call']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
