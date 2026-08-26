# Layout injection over the corpus — what a run costs

Every batch-partitioned end-to-end query runs at five modes and, for eleven of them, at
injected shapes as well: a rebatcher, a drained lane, zero-row batches and a degenerate
hash, crossed with the modes and cut to a cover. This is what those runs cost, so the cap
is chosen against a table rather than guessed.

Host: `dmitry-socrates`, Intel i5-10210U (4 cores, 8 threads), 15 GiB, tpch/tpcds sf1 on
local disk. **One thread** (`--test-threads=1`), because a number taken under four-way
contention cannot be compared with one taken under different contention. A CI runner is a
different machine and these are not its numbers; what carries across is the ratios.

Taken from `cargo test --features rust-only -p peacockdb-core --test
test_cpu_batch_partitioned -- --test-threads=1 --nocapture`, which prints one
`[injection]` line per run. Each figure is one run, not a minimum of several: what is being
decided is which rows to cut, and a repeat would not change that ordering.

Measured at `8052258`, and not re-taken since. One rule has moved: the interior rebatcher cut
the plan's top eligible edge when these were taken and takes a seeded one per candidate as of
`1ce0a37`, so every `rebatch=interior` cell, and the 1.04x row that averages them, describes
the earlier build.

## What a run costs, and where it goes

| Query | result rows | runs | total | mean | slowest |
|---|--:|--:|--:|--:|--:|
| `tpch/anti-join` | 1,196,041 | 30 | 61.8 s | 2.06 s | 2.43 s |
| `tpcds/q8` | 5 | 30 | 58.3 s | 1.94 s | 2.55 s |
| `tpcds/q2` | 2,513 | 30 | 20.6 s | 0.69 s | 0.81 s |
| `tpcds/q97` | 1 | 30 | 19.1 s | 0.64 s | 0.80 s |
| `tpcds/q93` | 100 | 30 | 18.2 s | 0.61 s | 0.70 s |
| `tpcds/q33` | 100 | 30 | 15.8 s | 0.53 s | 1.25 s |
| `tpcds/q45` | 19 | 30 | 10.7 s | 0.36 s | 0.57 s |
| `tpcds/q16` | 1 | 30 | 7.5 s | 0.25 s | 0.39 s |
| `tpch/nested-limits` | 20 | 30 | 0.1 s | 0.00 s | 0.00 s |
| `tpch/nested-loop-left-join` | 51 | 30 | 0.0 s | 0.00 s | 0.00 s |
| `tpch/nested-loop-join` | 50 | 30 | 0.0 s | 0.00 s | 0.00 s |
| **every query** | | 330 | **212 s** | | |

Two things the table settles.

**The comparison was the tier, and is not any more.** Every run compares its answer against
the oracle, and `assert_results_match` renders both sides per call — right for five runs
and wrong for thirty-five. Before it was changed, `tpch/anti-join` — `SELECT *` over 1.2M
rows — spent more than eight minutes on its five uninjected runs without reaching an
injected one. The oracle is now encoded once per query, the comparison is over arrow's row
encoding rather than rendered text, and the same query runs in 2.06 s.

**After that, cost is the query rather than the answer.** `tpcds/q8` returns five rows and
costs what `anti-join` costs returning 1.2M, because both scan around 3M and join them.
That restores the axis the eleven were chosen on — cheapest by rows scanned — which the
rendering comparison had quietly replaced with result size.

## The cap

**30 per query**, and 212 s for the eleven serially. The cover is at most one candidate per
requirement — five modes and nine dimension values — so a cap of 14 carries everything the
selection is asserted to carry, and `the_selection_covers_every_mode_and_every_boundary`
asserts that too. Nothing here asks for it: the three `tpch` nested-loop queries cost 0.1 s
for all 90 of their runs, and `q8` and `anti-join` are half the total between them.

The cover is not the argument for keeping it at 30, though: a cover guarantees each
dimension value once and says nothing about pairs, and the sixteen runs above the cover are
where an interaction between two settings would sit. `q33` at `bp-tp4-rowgroup` is 499 ms
with a lane drained and 281 ms drained with a degenerate hash on top; neither cell is in a
cover, and a pair is what makes the second number worth having.

At default parallelism the same file has measured 160 s, 213 s and 401 s on this host, the
same binary and the same 330 runs each time — a spread wider than any effect worth claiming
from it, and the reason this page measures serially. It was 279 s before this task with no
injected runs, which is the comparison that tempts; it is not one this page can make, because
the before figure is a single unrepeated run at a parallelism it does not control. What the
serial table supports is narrower and enough: the comparison stopped being the tier's cost.

If a runner turns out to need less, cut runs and not cover: the selection rule guarantees
a cover at any cap it accepts, and what must survive any trim is one carrier per dimension
and the shapes only one query has — `q33`'s four-lane interleave, the two nested-loop
forms, and `nested-limits`' two row-interval lowerings.

## What each setting costs

| Setting | runs carrying it | against the same plan as-planned |
|---|--:|--:|
| `rebatch=sources` | 47 | 1.07x |
| `empties=50%` | 71 | 1.06x |
| `rebatch=interior` | 42 | 1.04x |
| `drain=lane0` | 23 | 1.02x |
| `as-planned` | 32 | 1.00x |

## Every run

One cell per (query, mode, setting), in milliseconds. A `·` is a candidate the mode
cannot carry — one lane has no lane to drain, a plan with no scatter has no hash — or one
the cover did not need.

### `tpch/anti-join` — 1,196,041 result rows, 61.8 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 1722 | 1733 | · | 2157 | · |
| `drain=lane0` | · | · | · | 2139 | · |
| `drain=lane0/empties=50%` | · | · | 2239 | 2156 | 2024 |
| `empties=50%` | 1674 | 1813 | · | 2114 | · |
| `rebatch=interior` | · | 1813 | · | 2245 | 2140 |
| `rebatch=interior/drain=lane0` | · | · | · | 2289 | · |
| `rebatch=interior/drain=lane0/empties=50%` | · | · | 2428 | 2157 | · |
| `rebatch=interior/empties=50%` | · | 1931 | · | 2259 | · |
| `rebatch=sources` | 1742 | 1755 | 2186 | · | 2017 |
| `rebatch=sources/drain=lane0` | · | · | 2244 | · | 2139 |
| `rebatch=sources/drain=lane0/empties=50%` | · | · | 2303 | 2228 | 2108 |
| `rebatch=sources/empties=50%` | · | 1712 | 2272 | · | 2095 |

### `tpcds/q8` — 5 result rows, 58.3 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 2118 | 1860 | · | · | · |
| `drain=lane0` | · | · | 2036 | 1906 | 1891 |
| `drain=lane0/empties=50%` | · | · | · | · | 1901 |
| `drain=lane0/hash=one-lane` | · | · | 2069 | 1867 | 1764 |
| `empties=50%` | · | 1871 | · | · | · |
| `empties=50%/hash=one-lane` | · | · | 1958 | 1909 | · |
| `rebatch=interior/drain=lane0` | · | · | 1977 | 1885 | · |
| `rebatch=interior/drain=lane0/empties=50%` | · | · | · | 1875 | 1890 |
| `rebatch=interior/drain=lane0/empties=50%/hash=one-lane` | · | · | · | · | 1796 |
| `rebatch=interior/drain=lane0/hash=one-lane` | · | · | 1842 | 1765 | 1822 |
| `rebatch=interior/empties=50%/hash=one-lane` | · | · | 1895 | · | · |
| `rebatch=sources` | · | 1961 | · | · | · |
| `rebatch=sources/drain=lane0` | · | · | 2519 | 1916 | · |
| `rebatch=sources/drain=lane0/empties=50%` | · | · | · | 1888 | 1904 |
| `rebatch=sources/drain=lane0/hash=one-lane` | · | · | 2033 | 1837 | 1842 |
| `rebatch=sources/empties=50%/hash=one-lane` | · | · | 2546 | · | · |

### `tpcds/q2` — 2,513 result rows, 20.6 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 589 | 635 | · | · | · |
| `drain=lane0` | · | · | 714 | 751 | 713 |
| `drain=lane0/empties=50%` | · | · | · | · | 805 |
| `drain=lane0/hash=one-lane` | · | · | 629 | 647 | 617 |
| `empties=50%` | · | 716 | · | · | · |
| `empties=50%/hash=one-lane` | · | · | 622 | 640 | · |
| `rebatch=interior/drain=lane0` | · | · | 724 | 762 | · |
| `rebatch=interior/drain=lane0/empties=50%` | · | · | · | 741 | 717 |
| `rebatch=interior/drain=lane0/empties=50%/hash=one-lane` | · | · | · | · | 622 |
| `rebatch=interior/drain=lane0/hash=one-lane` | · | · | 622 | 638 | 656 |
| `rebatch=interior/empties=50%/hash=one-lane` | · | · | 626 | · | · |
| `rebatch=sources` | · | 640 | · | · | · |
| `rebatch=sources/drain=lane0` | · | · | 729 | 733 | · |
| `rebatch=sources/drain=lane0/empties=50%` | · | · | · | 754 | 799 |
| `rebatch=sources/drain=lane0/hash=one-lane` | · | · | 630 | 673 | 789 |
| `rebatch=sources/empties=50%/hash=one-lane` | · | · | 643 | · | · |

### `tpcds/q97` — 1 result rows, 19.1 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 577 | 488 | · | 669 | · |
| `drain=lane0` | · | · | · | 693 | · |
| `drain=lane0/empties=50%` | · | · | 691 | 656 | 669 |
| `empties=50%` | 577 | 480 | · | 654 | · |
| `rebatch=interior` | · | 474 | · | 660 | 801 |
| `rebatch=interior/drain=lane0` | · | · | · | 667 | · |
| `rebatch=interior/drain=lane0/empties=50%` | · | · | 671 | 664 | · |
| `rebatch=interior/empties=50%` | · | 483 | · | 668 | · |
| `rebatch=sources` | 564 | 572 | 686 | · | 667 |
| `rebatch=sources/drain=lane0` | · | · | 661 | · | 653 |
| `rebatch=sources/drain=lane0/empties=50%` | · | · | 668 | 684 | 794 |
| `rebatch=sources/empties=50%` | · | 584 | 681 | · | 677 |

### `tpcds/q93` — 100 result rows, 18.2 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 448 | 550 | · | 678 | · |
| `drain=lane0` | · | · | · | 701 | · |
| `drain=lane0/empties=50%` | · | · | 605 | 687 | 595 |
| `empties=50%` | 442 | 657 | · | 687 | · |
| `rebatch=interior` | · | 531 | · | 695 | 610 |
| `rebatch=interior/drain=lane0` | · | · | · | 673 | · |
| `rebatch=interior/drain=lane0/empties=50%` | · | · | 611 | 691 | · |
| `rebatch=interior/empties=50%` | · | 612 | · | 692 | · |
| `rebatch=sources` | 445 | 447 | 592 | · | 599 |
| `rebatch=sources/drain=lane0` | · | · | 641 | · | 618 |
| `rebatch=sources/drain=lane0/empties=50%` | · | · | 634 | 616 | 626 |
| `rebatch=sources/empties=50%` | · | 483 | 660 | · | 637 |

### `tpcds/q33` — 100 result rows, 15.8 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 805 | 804 | · | · | · |
| `drain=lane0` | · | · | 1252 | 499 | 465 |
| `drain=lane0/empties=50%` | · | · | · | · | 461 |
| `drain=lane0/hash=one-lane` | · | · | 728 | 281 | 294 |
| `empties=50%` | · | 776 | · | · | · |
| `empties=50%/hash=one-lane` | · | · | 701 | 279 | · |
| `rebatch=interior/drain=lane0` | · | · | 481 | 468 | · |
| `rebatch=interior/drain=lane0/empties=50%` | · | · | · | 476 | 458 |
| `rebatch=interior/drain=lane0/empties=50%/hash=one-lane` | · | · | · | · | 288 |
| `rebatch=interior/drain=lane0/hash=one-lane` | · | · | 321 | 294 | 293 |
| `rebatch=interior/empties=50%/hash=one-lane` | · | · | 308 | · | · |
| `rebatch=sources` | · | 912 | · | · | · |
| `rebatch=sources/drain=lane0` | · | · | 831 | 484 | · |
| `rebatch=sources/drain=lane0/empties=50%` | · | · | · | 486 | 513 |
| `rebatch=sources/drain=lane0/hash=one-lane` | · | · | 456 | 319 | 305 |
| `rebatch=sources/empties=50%/hash=one-lane` | · | · | 780 | · | · |

### `tpcds/q45` — 19 result rows, 10.7 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 257 | 268 | · | · | · |
| `drain=lane0` | · | · | 508 | 562 | 567 |
| `drain=lane0/empties=50%` | · | · | · | · | 497 |
| `drain=lane0/hash=one-lane` | · | · | 237 | 274 | 243 |
| `empties=50%` | · | 267 | · | · | · |
| `empties=50%/hash=one-lane` | · | · | 235 | 242 | · |
| `rebatch=interior/drain=lane0` | · | · | 510 | 503 | · |
| `rebatch=interior/drain=lane0/empties=50%` | · | · | · | 535 | 492 |
| `rebatch=interior/drain=lane0/empties=50%/hash=one-lane` | · | · | · | · | 236 |
| `rebatch=interior/drain=lane0/hash=one-lane` | · | · | 242 | 254 | 240 |
| `rebatch=interior/empties=50%/hash=one-lane` | · | · | 241 | · | · |
| `rebatch=sources` | · | 267 | · | · | · |
| `rebatch=sources/drain=lane0` | · | · | 509 | 566 | · |
| `rebatch=sources/drain=lane0/empties=50%` | · | · | · | 503 | 496 |
| `rebatch=sources/drain=lane0/hash=one-lane` | · | · | 240 | 236 | 234 |
| `rebatch=sources/empties=50%/hash=one-lane` | · | · | 244 | · | · |

### `tpcds/q16` — 1 result rows, 7.5 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 217 | 176 | · | · | · |
| `drain=lane0` | · | · | 297 | 308 | 305 |
| `drain=lane0/empties=50%` | · | · | · | · | 309 |
| `drain=lane0/hash=one-lane` | · | · | 186 | 191 | 182 |
| `empties=50%` | · | 178 | · | · | · |
| `empties=50%/hash=one-lane` | · | · | 188 | 268 | · |
| `rebatch=interior/drain=lane0` | · | · | 302 | 310 | · |
| `rebatch=interior/drain=lane0/empties=50%` | · | · | · | 298 | 300 |
| `rebatch=interior/drain=lane0/empties=50%/hash=one-lane` | · | · | · | · | 185 |
| `rebatch=interior/drain=lane0/hash=one-lane` | · | · | 362 | 190 | 189 |
| `rebatch=interior/empties=50%/hash=one-lane` | · | · | 186 | · | · |
| `rebatch=sources` | · | 218 | · | · | · |
| `rebatch=sources/drain=lane0` | · | · | 391 | 330 | · |
| `rebatch=sources/drain=lane0/empties=50%` | · | · | · | 324 | 321 |
| `rebatch=sources/drain=lane0/hash=one-lane` | · | · | 213 | 205 | 202 |
| `rebatch=sources/empties=50%/hash=one-lane` | · | · | 201 | · | · |

### `tpch/nested-limits` — 20 result rows, 0.1 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 3 | 2 | 3 | 2 | 3 |
| `empties=50%` | 4 | 2 | 3 | 2 | 4 |
| `rebatch=interior` | 3 | 2 | 3 | 2 | 3 |
| `rebatch=interior/empties=50%` | 3 | 2 | 3 | 2 | 3 |
| `rebatch=sources` | 3 | 3 | 3 | 3 | 3 |
| `rebatch=sources/empties=50%` | 3 | 3 | 3 | 3 | 3 |

### `tpch/nested-loop-left-join` — 51 result rows, 0.0 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 0 | 0 | 0 | 0 | · |
| `drain=lane0/empties=50%` | · | · | 0 | · | · |
| `empties=50%` | 0 | 0 | · | 0 | 0 |
| `rebatch=interior` | 0 | 0 | 0 | 0 | 0 |
| `rebatch=interior/drain=lane0/empties=50%` | · | · | 1 | · | · |
| `rebatch=interior/empties=50%` | · | 0 | · | 0 | 0 |
| `rebatch=sources` | 0 | 0 | 0 | 0 | 1 |
| `rebatch=sources/drain=lane0` | · | · | 0 | · | · |
| `rebatch=sources/drain=lane0/empties=50%` | · | · | 1 | · | · |
| `rebatch=sources/empties=50%` | 0 | 0 | 1 | 0 | 1 |

### `tpch/nested-loop-join` — 50 result rows, 0.0 s over 30 runs, ms per run

| Setting | tp1-single | tp1-rowgroup | tp4-single | tp4-rowgroup | tp4-sized |
|---|--:|--:|--:|--:|--:|
| `as-planned` | 0 | 0 | 0 | 0 | · |
| `drain=lane0/empties=50%` | · | · | 0 | · | · |
| `empties=50%` | 0 | 0 | · | 0 | 1 |
| `rebatch=interior` | 0 | 0 | 0 | 1 | 0 |
| `rebatch=interior/drain=lane0/empties=50%` | · | · | 1 | · | · |
| `rebatch=interior/empties=50%` | · | 0 | · | 1 | 0 |
| `rebatch=sources` | 0 | 0 | 0 | 0 | 0 |
| `rebatch=sources/drain=lane0` | · | · | 0 | · | · |
| `rebatch=sources/drain=lane0/empties=50%` | · | · | 1 | · | · |
| `rebatch=sources/empties=50%` | 0 | 0 | 1 | 0 | 1 |
