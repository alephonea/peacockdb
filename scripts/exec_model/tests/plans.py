"""Render the corpus plan goldens, without executing anything.

    python3 scripts/exec_model/tests/plans.py                # print every plan
    python3 scripts/exec_model/tests/plans.py q1 q3          # print some
    python3 scripts/exec_model/tests/plans.py --write        # rewrite the golden files

A plan is a function of the schemas: every builder in `plans_tpch.py` / `plans_tpcds.py`
takes a table provider and hands the frames to scan nodes that do not read them until a
driver runs. So the goldens can be produced from parquet footers in a tenth of a second,
where running the queries that produce the same trees takes minutes. That difference is the
reason this file exists — a plan golden should be cheap to look at and cheap to regenerate,
and only its *verification* should cost a corpus run.

The golden itself is written by `corpus.check_plan`, the same function the tests compare
against, so there is one definition of what a section looks like.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/plans.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

import os
import sys

from . import corpus
from .. import plan_text
from . import plans_tpch, plans_tpcds

#: bench → its query registry. A benchmark with no queries yet simply has none.
BENCHES = {"tpch": plans_tpch.QUERIES, "tpcds": plans_tpcds.QUERIES}


def selected(names):
    """Every (bench, query, builder) the arguments ask for."""
    chosen = [
        (bench, query, build)
        for bench, queries in BENCHES.items()
        for query, build in queries.items()
        if not names or query in names
    ]
    if not chosen:
        raise SystemExit(f"no corpus query matches {names}; have {_known()}")
    return chosen


def _known():
    return ", ".join(f"{bench}:{query}" for bench, queries in BENCHES.items() for query in queries)


def main(argv) -> int:
    write = "--write" in argv
    names = [arg for arg in argv if not arg.startswith("-")]
    if write:
        # check_plan writes when this is set, and compares when it is not — one code path
        # for both, so the file this produces is by construction the file tests read.
        os.environ["UPDATE_CANONICAL"] = "1"
    for bench, query, build in selected(names):
        root = build(corpus.schema_reader(bench))
        if write:
            corpus.check_plan(bench, query, root)
            print(f"wrote {bench}.plans.txt :: {query}")
        else:
            print(plan_text.render(root, f"{bench} {query}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
