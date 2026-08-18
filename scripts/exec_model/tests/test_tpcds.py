"""TPC-DS corpus queries: run every lowering at every layout, against DuckDB.

The lowerings live in `plans_tpcds*.py` — a plan is a function of the schemas, so they are
built from parquet footers alone when `plans.py` renders the goldens. What lives here is the
driver and the oracle.

**The oracle is the query's own text.** DuckDB runs `testdata/tpcds-queries/qN.sql` over the
same parquet files, and the result is compared against the prototype's. A hand-written
pandas equivalent would catch only what the *mode* gets wrong — batching, lanes, a finish
pass — because both sides would share one reading of the SQL; this catches the reading too,
which is the circularity #80 complains of in the legacy tiers. It also pins the output
column names, since those come from the SQL's own aliases and nothing else in the prototype
knows them. Seventy-one hand-written oracles would also have been seventy-one new places to
be wrong.

Whole tables, the spec's own parameters: sampling was tried and abandoned (see `corpus.py`
for what it does to date-clustered tables). Manual dispatch only — `exec-model-corpus.yml`,
sharded by `PCK_SHARD` and `PCK_LAYOUT`.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

from . import corpus
from .harness import main
from .plans_tpcds import ORDER_BY, QUERIES


def _corpus_test(query):
    """One query's test: DuckDB once, then the lowering at each layout against it."""
    def test():
        want = corpus.duckdb_answer("tpcds", query)
        assert len(want) or query in EMPTY_BY_DESIGN, (
            f"{query}: DuckDB returns no rows, so the run would assert nothing"
        )
        for label, got in corpus.run_layouts("tpcds", query,
                                             QUERIES[query](corpus.reader("tpcds"))):
            corpus.matches_oracle(got, want, label, order_by=ORDER_BY[query])

    test.__name__ = f"test_corpus_{query}"
    test.__doc__ = f"TPC-DS {query}, at every layout, against DuckDB's answer."
    return test


#: Queries whose correct answer at sf1 is the empty set. An empty result is normally the
#: shape of a test that verifies nothing, so it has to be declared rather than tolerated.
#: q17's three-channel join over one quarter genuinely matches no row at this scale.
EMPTY_BY_DESIGN = frozenset({"q17"})

# Generated rather than written out: the body is identical for all of them, and seventy-one
# copies of it would be seventy-one chances for one to drift. The harness collects by name
# from the namespace, so these are ordinary tests to it — `test_tpcds.py q42` selects one,
# `PCK_SHARD=k/n` takes a share.
for _query in QUERIES:
    globals()[f"test_corpus_{_query}"] = _corpus_test(_query)

if __name__ == "__main__":
    raise SystemExit(main(globals()))
