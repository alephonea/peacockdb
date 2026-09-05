#!/usr/bin/env python3
"""What a timed region spends its time on, one level down, from an Nsight capture.

    PCK_BENCH_NSYS=1 PCK_TEST_FILTER=bench_tpch_sf40_q9_ \
        ./scripts/build-test-shadgpu.sh --run-benchmarks --pull-benchmarks
    scripts/calibration/nsys_calls.py --capture testdata/calibration/capture.sqlite \
        --out testdata/calibration/calls.tsv \
        --plans testdata/goldens/tpch.sf40/bp-tp1-single.plans.txt

Which query a region belongs to is READ OFF THE CAPTURE: the harness wraps each case in an
NVTX range naming it, so nothing has to be told on the command line. `--plans` is still a
path because the goldens live where they live, and its mode selects which of the capture's
cases it can check.

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
import pathlib
import re
import sys

import nvtx_names

# Our own domain. The two levels it carries, and how their names are read, are
# `nvtx_names` — shared with `nsys_hbm.py`, which reads the same capture for a different
# number.
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
                where e.eventType = {nvtx_names.DOMAIN_CREATE}"""
        )
    }


def ranges(conn, domain_id):
    """(start, end, text, tid) for one domain, in start order."""
    return list(conn.execute(
        f"""select e.start, e.end, coalesce(e.text, s.value), e.globalTid
            from NVTX_EVENTS e left join StringIds s on s.id = e.textId
            where e.eventType = {nvtx_names.PUSHPOP_RANGE} and e.domainId = ?
              and e.end is not null
            order by e.start""",
        (domain_id,),
    ))


def recipe_seqs(plans_path, query):
    """The seqs a query's `--- recipes ---` section names, as {seq: kind}.

    The golden is the planner's own statement of what the C++ will be asked to run, so
    checking a capture against it answers a question the capture alone cannot: whether the
    regions in it are the ones this plan was supposed to produce. A capture of a different
    query, or of a plan that has since moved, looks perfectly self-consistent.
    """
    text = pathlib.Path(plans_path).read_text()
    marker = f"\n== {query}\n"
    if marker not in text:
        sys.exit(f"{plans_path} has no `== {query}` section")
    section = text.split(marker, 1)[1].split("\n== ", 1)[0]
    if "--- recipes ---" not in section:
        sys.exit(f"{plans_path}: `{query}` has no recipes section")
    recipes = section.split("--- recipes ---", 1)[1].split("--- memory ---", 1)[0]
    # `execute_node(#4 CudfAggregate{Merge}, prior output)` -> 4, CudfAggregate. The brace
    # payload is the recipe's own annotation and is not what a capture's range carries.
    found = {}
    for seq, kind in re.findall(r"execute_\w+\(#(\d+) (\w+)", recipes):
        found[int(seq)] = kind
    return found


def check_against_recipes(regs, plans_path):
    """Every seq the recipes name was driven, with the kind they name, and no others.

    Which query is read off the CAPTURE, not off the command line. A plans golden holds
    every query of one mode, so the capture's own cases select their sections — and a case
    from another mode is skipped rather than checked against a plan it never ran from.
    """
    # `bp-tp1-single.plans.txt` -> `bp-tp1-single`. The mode is in the filename because
    # that is how the goldens are laid out.
    mode = pathlib.Path(plans_path).name.split(".plans.txt")[0]
    by_case = collections.defaultdict(dict)
    for case, seq, _, kind, _, _, _, _ in regs:
        by_case[case][seq] = kind
    checked = sorted(c for c in by_case if c[3] == mode)
    if not checked:
        sys.exit(
            f"{plans_path} is mode {mode!r} and the capture holds "
            f"{sorted({c[3] for c in by_case})} -- nothing to check it against."
        )
    for case in checked:
        _check_one(by_case[case], plans_path, case[2])


def _check_one(seen, plans_path, query):
    declared = recipe_seqs(plans_path, query)

    missing = sorted(set(declared) - set(seen))
    extra = sorted(set(seen) - set(declared))
    wrong = sorted((seq, declared[seq], seen[seq]) for seq in set(declared) & set(seen)
                   if declared[seq] != seen[seq])
    if missing or extra or wrong:
        parts = []
        if missing:
            parts.append(f"declared but never driven: {missing}")
        if extra:
            parts.append(f"driven but not declared: {extra}")
        if wrong:
            parts.append("kind differs: " + ", ".join(
                f"#{s} is {d} in the plan and {c} in the capture" for s, d, c in wrong))
        sys.exit(f"capture does not match {query}'s recipes -- " + "; ".join(parts))
    print(f"recipes: {len(declared)} seqs declared for {query}, all driven with the "
          f"kinds the plan names")


def regions(own):
    """Our domain's ranges -> [(case, seq, call_index, kind, partition, start, end, tid)].

    The three levels are told apart by their names rather than by nesting depth -- that
    rule is `nvtx_names`. What is added here is the CONTAINMENT check, twice: a partition
    range outside every call range of its thread, or a call outside every case range,
    means the ranges did not come from the code this script thinks they did, and a level
    rule alone cannot see that.

    `case` is `(dataset, sf, query, mode)`, read off the harness's own range. It used to
    be a `--query` the CALLER typed, which is the shape of every quiet mistake: a capture
    of q19 analysed under the name q6 is self-consistent all the way down, and even its
    seq numbers line up, because seq numbering restarts with every plan.
    """
    cases = [(a, b, nvtx_names.case_of(t)) for a, b, t, _ in own if nvtx_names.is_case(t)]
    if not cases:
        sys.exit(
            "the capture has no case range. It predates the harness pushing one, so the "
            "query a region belonged to cannot be recovered from it -- retake it with a "
            "build that does."
        )
    calls = [r for r in own
             if not nvtx_names.is_partition(r[2]) and not nvtx_names.is_case(r[2])]
    parts = [r for r in own if nvtx_names.is_partition(r[2])]
    starts = [c[0] for c in calls]

    out = []
    for start, end, text, tid in parts:
        i = bisect.bisect_right(starts, start) - 1
        if i < 0 or calls[i][1] < end or calls[i][3] != tid:
            sys.exit(f"partition range at {start} is inside no call range of its thread")
        case = next((c for a, b, c in cases if a <= calls[i][0] < b), None)
        if case is None:
            sys.exit(
                f"call range {calls[i][2]!r} at {calls[i][0]} is inside no case range. "
                "Every call the harness makes is inside the case it belongs to."
            )
        seq, call, kind = nvtx_names.call_of(calls[i][2])
        out.append((case, seq, call, kind, nvtx_names.partition_of(text), start, end, tid))
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
    ap.add_argument("--plans", help="one <mode>.plans.txt to check the capture against")
    ap.add_argument("--plans-dir",
                    help="a goldens directory; each case's <mode>.plans.txt is found in "
                         "it, so a capture spanning several modes checks against all of "
                         "them without the caller listing which")
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
    # executions in the capture. A list and not a running total: executions differ by
    # more than noise — 7% at the median and 36% at the worst, measured — and a mean
    # hides which of them the number came from. (It used to say the warm-up was in here
    # too; it is not, since the harness turns ranges on after it.)
    per_exec = collections.defaultdict(lambda: collections.defaultdict(
        lambda: [0, 0, 0]))  # exec key -> (call, depth) -> [count, host_ns, device_ns]
    region_span = collections.defaultdict(list)
    seen = collections.Counter()
    by_call = collections.defaultdict(set)
    in_regions_ns = 0

    for case, seq, call, kind, part, r_start, r_end, tid in regs:
        ident = (seq, kind, part)
        # `run` counts occurrences across the capture; `call` is the index within one
        # execution, which is what a record row carries. They are not the same number and
        # must not be conflated: a benchmark opens a session per run, so the C++ counter
        # restarts at every one of them and a seq driven once per run is call 0 ten times
        # over. What the two together say is how many executions the capture holds.
        run = seen[ident]
        seen[ident] += 1
        by_call[ident].add(call)
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
    if args.plans:
        check_against_recipes(regs, args.plans)
    if args.plans_dir:
        # One goldens file per mode, found rather than listed: the capture already says
        # which modes are in it, and a caller retyping that list is the same class of
        # mistake `--query` was. A mode with no golden is reported, not skipped — the
        # check silently covering less than the capture is how it stops meaning anything.
        modes = sorted({case[3] for case, *_ in regs})
        for mode in modes:
            path = pathlib.Path(args.plans_dir) / f"{mode}.plans.txt"
            if not path.exists():
                print(f"recipes: no {path} — {mode} unchecked")
                continue
            check_against_recipes(regs, str(path))

    calls_per_exec = {len(v) for v in by_call.values()}
    print(f"call indices per region: {sorted(calls_per_exec)} "
          f"(1 means every execution drove each seq once)")
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
