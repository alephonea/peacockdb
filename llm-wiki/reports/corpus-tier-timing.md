# The batch-partitioned corpus tier — what a run costs

T18's cpu tier at sf1: 18 queries, 87 (query, mode) runs, one process per query. Taken serially
(`--test-threads=1`) on `dmitry-socrates`, one row per query, the way
[injection-timing.md](injection-timing.md) was — a number taken under different contention cannot
be compared with one taken under this.

| query | cases | wall s | peak RSS MB |
|---|--:|--:|--:|
| `tpch/q19` | 5 | 9.04 | 1283 |
| `tpcds/q15` | 5 | 8.54 | 162 |
| `tpch/q10` | 5 | 4.44 | 465 |
| `tpch/q13` | 5 | 4.14 | 295 |
| `tpcds/q82` | 5 | 3.42 | 561 |
| `tpcds/q37` | 5 | 3.24 | 561 |
| `tpch/q3` | 5 | 2.51 | 442 |
| `tpcds/q43` | 5 | 2.24 | 251 |
| `tpch/q12` | 5 | 1.82 | 169 |
| `tpcds/q3` | 5 | 1.28 | 237 |
| `tpcds/q42` | 5 | 1.19 | 258 |
| `tpcds/q52` | 5 | 1.12 | 235 |
| `tpcds/q55` | 5 | 1.12 | 238 |
| `tpcds/q96` | 2 | 0.61 | 229 |
| `tpch/q15` | 5 | 0.61 | 94 |
| `tpch/q6` | 5 | 0.56 | 150 |
| `tpch/q14` | 5 | 0.45 | 91 |
| `tpcds/q41` | 5 | 0.33 | 57 |
| **whole tier, one thread** | 87 | **45.93** | **1341** |
| whole tier, two threads | 87 | 26.15 | 1502 |

Each row is its own process and pays one parquet registration, so the rows do not sum to the
tier — the tier line is the one to compare against another tier.

**Two queries are most of it.** `tpch/q19` and `tpcds/q15` are 17.58 s between them — 38% of the
tier's 45.93 s, or 38% of the rows' own 46.66 s, which differ by one registration and so agree
here. q19 alone is 1.28 GB of the 1.34 GB peak: the tier's memory is one query's. The other
sixteen average 1.8 s.

**What this does not support.** Scaling it to T19's 119 queries gives 5.1 minutes serial by
query, or 5.2 by case, if the distribution holds — well under the fifty-eight minutes that has
lost the cpp-cpu leg before. Distrust that projection rather than plan against it: these twenty were chosen for having
the smallest plans in the corpus, so the distribution is the one thing about them that is not
representative. The number worth carrying forward is the shape — that one or two queries dominate
both axes — because that is what decides whether a tier that grows fivefold grows evenly.
