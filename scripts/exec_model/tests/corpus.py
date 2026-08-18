"""Reading the generated benchmark datasets, for the two corpus test files.

`test_tpch.py` and `test_tpcds.py` both run real query shapes over real tables, so they
share a reader, a budget and an oracle comparison. Nothing here is a test.

**Which dataset.** `testdata/tpch.sf1` and `testdata/tpcds.sf1`, which
`generate_testdata.sh` produces and CI's cpp-cpu job regenerates every run — which is why
both corpus files run there rather than in cost-report, where there is no dataset. TPC-H
falls back to the committed `testdata/tpch.minimal` for the five tables it holds; a query
naming a table neither dataset has **fails**, loudly, naming the generator. It does not
skip: a skipped corpus suite reads exactly like a passing one.

**Two casts on read**, both standing in for typing the prototype does not have:

- **decimals to float64.** TPC-H money is `decimal128(15, 2)` and pandas has no decimal
  dtype, so pyarrow hands back an object column of `Decimal`s that will not multiply by a
  float literal. `frame.py` names this divergence already; it is also why the real engine
  carries precision and scale in the flat buffers rather than letting cuDF re-derive them.
- **dates to `datetime64[ns]`.** A parquet `date32` arrives as an object column of
  `datetime.date`, which compares to nothing the expression IR can hold. cuDF's own type is
  `TIMESTAMP_DAYS`, a typed scalar, so datetime64 is the closer model.

**The corpus queries read whole tables.** They are the real queries with the spec's own
parameters, so they get the real data: sampling was tried and it is a trap. Both benchmarks
are written clustered by date (`generate_testdata.sh` ~L273, ~L322: lineitem `ORDER BY
l_shipdate, l_orderkey, l_linenumber`, every TPC-DS fact by its date_sk), which row-group
pruning depends on — so a row prefix is one quarter of 1992, a row-group sample is a set of
date windows, and two tables sampled independently join to nothing. Every one of those bit.
Whole tables have none of the problems and answer the question the corpus is asked to
answer.

`limit` therefore serves the short plan-shape tests only, where a few thousand rows of a
dimension table is the point.

The read cache is deliberately small: a full lineitem is a gigabyte of python objects, and
holding one per column-set across a file's worth of queries is how a runner runs out of
memory. Two reads of the same columns in one query share; a later query re-reads.
"""

from __future__ import annotations

import datetime
import decimal
import functools
import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..batch_partitioned_driver import batch_partitioned_driver
from ..node import CpuBackendSelector, RecipeJoinBackendSelector
from ..operators import aggregates as A
from ..operators.frame import concatenate
from ..plan import Plan

#: Generous against the plan-shape tests' real peaks (~tens of MB) and far from unbounded,
#: so the enforcer is exercised on every call and a blown resident set fails the test.
BUDGET = 256 * 1024 * 1024

#: What a corpus query runs under. The legacy tiers' "mini" device is 2 GiB and this is the
#: same number, which is the point: a row costs more in pandas than in cuDF, so a budget
#: that binds here binds harder than the GPU's would — a plan that fits is not flattered.
#: A build side that does not fit is a plan to fix (project it down, put the small side on
#: the build) rather than a budget to raise; q18 found one.
CORPUS_BUDGET = 2 * 1024 * 1024 * 1024

_ROOT = pathlib.Path(__file__).resolve().parents[3] / "testdata"

#: bench → the directories to try, in order. Only TPC-H has a committed fallback.
_DATASETS = {
    "tpch": ("tpch.sf1", "tpch.minimal"),
    "tpcds": ("tpcds.sf1",),
}


def dataset_dir(bench: str, name: str) -> pathlib.Path:
    """The directory holding `name`, or a failure that says how to make one."""
    for candidate in _DATASETS[bench]:
        if (_ROOT / candidate / f"{name}.parquet").exists():
            return _ROOT / candidate
    raise FileNotFoundError(
        f"no {name}.parquet under {' or '.join(_DATASETS[bench])} in {_ROOT}. "
        "The corpus tests need the generated sf1 tables: run testdata/generate_testdata.sh "
        "(CI's cpp-cpu job does this before running these files)."
    )


@functools.lru_cache(maxsize=4)
def _read(bench: str, name: str, columns: tuple[str, ...], limit: int | None) -> pd.DataFrame:
    path = dataset_dir(bench, name) / f"{name}.parquet"
    if limit is None:
        arrow = pq.read_table(path, columns=list(columns))
    else:
        reader = pq.ParquetFile(path)
        batches, seen = [], 0
        for batch in reader.iter_batches(batch_size=65_536, columns=list(columns)):
            batches.append(batch)
            seen += batch.num_rows
            if seen >= limit:
                break
        arrow = pa.Table.from_batches(batches, schema=batches[0].schema).slice(0, limit)
    return _typed(arrow)


def _typed(arrow: pa.Table) -> pd.DataFrame:
    """An arrow table as a frame, with the two casts this file's header explains.

    Applied to the DuckDB oracle's answer as well as to the tables, so a comparison is
    never about how the two sides were transported.
    """
    frame = arrow.to_pandas()
    for field in arrow.schema:
        if pa.types.is_decimal(field.type):
            frame[field.name] = frame[field.name].astype("float64")
        elif pa.types.is_date(field.type):
            frame[field.name] = pd.to_datetime(frame[field.name])
    return frame


def table(bench: str, name: str, columns, limit: int | None = None) -> pd.DataFrame:
    """`name`'s columns, whole, or the first `limit` rows.

    A copy per call: the frames go into plans that mutate nothing, but the cache holds one
    frame per read and a caller that edited it would poison every later reader.
    """
    return _read(bench, name, tuple(columns), limit).copy()


def build_join(name, build_frame, build_node, probe, build_key, probe_key, join_type=None):
    """One join in a star: the small side is the build, the fact stream is the probe.

    The shape every corpus query here has, and the one v1 leans on — a build side
    collected into a single batch, a probe side that streams past it. `build_frame` is the
    frame the build node reads, needed only for the schema an empty lane would emit.
    """
    from ..operators import nodes as N
    from ..operators.join_types import JoinType

    return N.hash_join(
        name,
        N.coalesce_all(f"{name}_build", build_node, schema=dict(build_frame.dtypes)),
        probe,
        join_type or JoinType.INNER,
        [build_key],
        [probe_key],
    )


def reader(bench: str):
    """`(name, columns) -> frame` over the real tables, for running a query."""
    return lambda name, columns: table(bench, name, columns)


def _sample_value(arrow_type):
    """A valid value of `arrow_type`, so a schema frame types like the real one.

    One row rather than none: a zero-row frame has no row groups, and the scan builder
    refuses an empty survivor set — correctly, since an empty work set is an error and not
    a plan. The value is never read; only its type reaches the plan.
    """
    if pa.types.is_boolean(arrow_type):
        return False
    if pa.types.is_integer(arrow_type):
        return 0
    if pa.types.is_floating(arrow_type):
        return 0.0
    if pa.types.is_decimal(arrow_type):
        return decimal.Decimal(0)
    if pa.types.is_date(arrow_type) or pa.types.is_timestamp(arrow_type):
        return datetime.date(1970, 1, 1)
    return ""


@functools.lru_cache(maxsize=64)
def _schema_frame(bench: str, name: str, columns: tuple[str, ...]) -> pd.DataFrame:
    fields = {
        field.name: field.type
        for field in pq.read_schema(dataset_dir(bench, name) / f"{name}.parquet")
    }
    frame = pa.table(
        {column: pa.array([_sample_value(fields[column])], type=fields[column])
         for column in columns}
    ).to_pandas()
    for column in columns:
        if pa.types.is_decimal(fields[column]):
            frame[column] = frame[column].astype("float64")
        elif pa.types.is_date(fields[column]):
            frame[column] = pd.to_datetime(frame[column])
    return frame


def schema_reader(bench: str):
    """`(name, columns) -> a one-row frame of the right types`, from the parquet footer.

    A plan is a function of the schemas, not of the rows: the builders use a frame for its
    dtypes and hand it to a scan node that never reads it until the driver runs. So the
    same builder produces the same plan text over empty frames, in milliseconds instead of
    the seconds a six-million-row read costs — which is what lets `plans.py` emit the plan
    goldens without executing anything.
    """
    return lambda name, columns: _schema_frame(bench, name, tuple(columns)).copy()


# -- the oracle -------------------------------------------------------------------


def duckdb_binary() -> str:
    """A DuckDB CLI, or a failure saying where one comes from."""
    for candidate in (os.environ.get("DUCKDB"), "duckdb",
                      str(pathlib.Path.home() / ".duckdb/cli/latest/duckdb")):
        found = shutil.which(candidate) if candidate else None
        if found:
            return found
    raise FileNotFoundError(
        "no duckdb binary for the corpus oracle: set $DUCKDB, or install the CLI. CI's "
        "corpus workflow downloads the pinned v1.5.4 release the goldens are generated with."
    )


@functools.lru_cache(maxsize=32)
def duckdb_answer(bench: str, query: str) -> pd.DataFrame:
    """The query's **own text**, run by DuckDB over the same parquet files.

    An oracle that is neither this prototype nor pandas. A hand-written pandas equivalent
    only catches what the *mode* gets wrong — batching, lanes, a finish pass — because both
    sides would share one reading of the SQL; this catches the reading too, which is the
    circularity #80 complains of in the legacy tiers. It also checks the lowering produces
    the query's declared output columns, since the names come from the SQL's own aliases.

    The answer comes back as **parquet**, not CSV: CSV carries no types, so a column of
    surrogate keys came back as float, a zip code came back as an integer, and the compare
    was then arguing about the transport rather than the answer. Read with the same
    conversions `_read` applies to the tables, so both sides of the compare are typed alike.
    """
    sql = (_ROOT / f"{bench}-queries" / f"{query}.sql").read_text().strip()
    directory = dataset_dir(bench, "lineitem" if bench == "tpch" else "store_sales")
    views = "".join(
        f"CREATE VIEW {path.stem} AS SELECT * FROM read_parquet('{path}');\n"
        for path in sorted(directory.glob("*.parquet"))
    )
    with tempfile.TemporaryDirectory() as scratch:
        answer = pathlib.Path(scratch) / "answer.parquet"
        subprocess.run(
            [duckdb_binary(), "-c",
             f"{views}COPY ({sql.rstrip().rstrip(';')}) TO '{answer}' (FORMAT PARQUET);"],
            capture_output=True, text=True, check=True,
        )
        return _typed(pq.read_table(answer))


def _compare_columns(got: pd.DataFrame, want: pd.DataFrame, columns, label: str, what: str,
                     oracle: str = "DuckDB"):
    """Column by column, aligning only what the transport could not carry.

    A date that reached the oracle as text is parsed against the engine's own dtype — the
    DuckDB side comes back typed through parquet, but a hand-written pandas oracle need not
    be. Money is compared with a tolerance, since summing six million floats in a different
    order moves the last bits.
    """
    for column in columns:
        left, right = got[column], want[column]
        if pd.api.types.is_datetime64_any_dtype(left):
            right = pd.to_datetime(right)
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            assert np.allclose(
                left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                rtol=1e-9, atol=1e-2, equal_nan=True,
            ), f"{label}: {what} {column} differs from {oracle}"
        else:
            def nulls_alike(values):
                return [None if pd.isna(v) else v for v in values]

            assert nulls_alike(left) == nulls_alike(right), (
                f"{label}: {what} {column} differs from {oracle}"
            )


def _canonical(frame: pd.DataFrame, inexact) -> pd.DataFrame:
    """One frame's rows in an order that depends on nothing but their values.

    Exact columns lead and the inexact ones follow, rounded before they are sorted on: two
    engines that summed the same money in a different order disagree in the last bits, and
    an unrounded sort key would let that noise reorder the rows it is meant to align.
    Rounding is only for the ordering — the values themselves are still compared at full
    precision, with a tolerance.

    `inexact` comes from both frames at once and not from this one. A column of surrogate
    keys arrives from parquet as float where some are null and from DuckDB's CSV as int, and
    a per-frame split would then sort the two sides by different column orders — which
    looked exactly like a wrong answer the first time it happened.
    """
    keys = frame.copy()
    for column in inexact:
        keys[column] = pd.to_numeric(keys[column], errors="coerce").round(4)
    order = [c for c in keys.columns if c not in inexact] + list(inexact)
    return frame.loc[
        keys.sort_values(order, kind="stable", na_position="first").index
    ].reset_index(drop=True)


def matches_oracle(got: pd.DataFrame, want: pd.DataFrame, label: str, order_by=None,
                   oracle: str = "DuckDB"):
    """Compare an engine result against an oracle's, exactly as strictly as SQL allows.

    Column names and their order are asserted rather than aligned: they are the query's
    declared output, and a lowering that renamed or reordered them answered a different
    query.

    Two checks, because `ORDER BY` promises less than a positional compare asserts. The
    rows are compared as a **multiset** — both sides canonically sorted — which is the whole
    of what the query determines. Then the `order_by` columns are compared **positionally**,
    which is the whole of what the sort determines: where the sort keys tie, SQL leaves the
    order of those rows open and two correct engines may disagree. Comparing full rows
    positionally would make a tie into a failure, which is how tpch q11 first failed.

    `order_by` is the query's ORDER BY as output column names; a query with no ORDER BY
    passes None and gets the multiset check alone.
    """
    assert list(got.columns) == list(want.columns), (
        f"{label}: {list(got.columns)} vs the query's {list(want.columns)}"
    )
    assert len(got) == len(want), f"{label}: {len(got)} rows vs {oracle}'s {len(want)}"
    if order_by:
        missing = [column for column in order_by if column not in want.columns]
        assert not missing, f"{label}: ORDER BY names {missing}, which the output has not"
        _compare_columns(got, want, order_by, label, "sort key", oracle)
    inexact = [
        column for column in want.columns
        if pd.api.types.is_float_dtype(got[column]) or pd.api.types.is_float_dtype(want[column])
    ]
    _compare_columns(_canonical(got, inexact), _canonical(want, inexact),
                     list(want.columns), label, "column", oracle)


# -- running a corpus query ---------------------------------------------------------

#: The layouts every corpus query is run at. `default` is the plan as the test wrote it;
#: the other two are `LayoutInjector` rewrites of it. Three rather than the injector's
#: five: on an unshuffled join the injector varies batching and not lane count (a join's
#: lanes are load-bearing unless both sides are hash-partitioned), so the extra presets
#: mostly re-prove what the synthetic sweeps already prove per operator — and each one
#: multiplies a query that reads six million rows.
LAYOUTS = ("default", "one_partition_one_batch", "many_small_partitions")


def selected_layouts() -> tuple:
    """The layouts this process runs. `PCK_LAYOUT` names one, so three processes cover the
    three; unset runs all of them, so a single-process run is still complete."""
    chosen = os.environ.get("PCK_LAYOUT")
    if not chosen:
        return LAYOUTS
    if chosen not in LAYOUTS:
        raise SystemExit(f"PCK_LAYOUT={chosen} is not one of {LAYOUTS}")
    return (chosen,)


def run_layouts(bench: str, query: str, root, budget: int | None = None):
    """Yield `(label, result)` for this query at each layout, and pin its plan.

    The plan golden is written from the **default** layout only: the injected ones are
    rewrites of it, so pinning them would pin the injector rather than the lowering.
    """
    from ..operators.injection import HashMode, LayoutInjector, LayoutPreset

    for label in selected_layouts():
        if label == "default":
            plan, checked = root, check_plan(bench, query, root)
            del checked
        else:
            injector = LayoutInjector(LayoutPreset(label), HashMode.SPREAD, 0.2, 17)
            plan = injector.apply(root)
        got, driver = execute(plan, budget=budget or CORPUS_BUDGET)
        # What must hold whatever the layout was: nothing stranded, nothing still held,
        # and a peak that the enforcer was actually watching.
        assert driver.accountant.in_flight_bytes == 0, f"{query}/{label}: batches still held"
        assert 0 < driver.accountant.peak <= (budget or CORPUS_BUDGET), (
            f"{query}/{label}: peak {driver.accountant.peak}"
        )
        yield f"{query}/{label}", got


# -- the plan golden ----------------------------------------------------------------

#: One file per benchmark, holding one section per query, accumulated as queries land.
_PLANS = pathlib.Path(__file__).resolve().parents[1]


def check_plan(bench: str, query: str, root) -> str:
    """Compare this query's plan against its golden section, or write it.

    `UPDATE_CANONICAL=1` inserts or replaces the section, as everywhere else in this repo;
    without it a changed lowering is a red test and the diff says which node moved.
    """
    from .. import plan_text

    rendered = plan_text.render(root, query)
    path = _PLANS / f"{bench}.plans.txt"
    sections = _sections(path)
    if os.environ.get("UPDATE_CANONICAL"):
        sections[query] = rendered
        path.write_text("".join(sections[name] for name in sorted(sections)))
        return rendered
    if query not in sections:
        raise AssertionError(
            f"{path.name} has no plan for {query}; regenerate with UPDATE_CANONICAL=1"
        )
    if sections[query] != rendered:
        raise AssertionError(
            f"{path.name}: {query}'s plan changed\n--- golden\n{sections[query]}"
            f"--- now\n{rendered}"
        )
    return rendered


def _sections(path: pathlib.Path) -> dict:
    """The file split by its `== <query>` headers."""
    if not path.exists():
        return {}
    sections, current = {}, None
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("== "):
            current = line[3:].strip()
            sections[current] = line
        elif current is not None:
            sections[current] += line
    return sections


def schema_of(*frames, **computed) -> pd.DataFrame:
    """A zero-row frame with every column these frames carry, plus the computed ones.

    What an aggregate needs in order to emit an empty batch: a lane that filtered
    everything away still owes its parent one batch, and a batch has typed columns. The
    real engine takes that from the node's declared output schema (T7); the prototype takes
    it from the frames the plan was built over, which is the same information a schema is.
    """
    merged = {}
    for frame in frames:
        merged.update(dict(frame.dtypes))
    merged.update({column: dtype for column, dtype in computed.items()})
    return pd.DataFrame({column: pd.Series([], dtype=dtype) for column, dtype in merged.items()})


def agg_schemas(df, keys, aggs):
    """Typed `{column: dtype}` per aggregate phase, derived over a zero-row slice — what an
    empty lane's `aggregate_batches` has to emit."""
    state = A.partial(df.iloc[0:0], list(keys), aggs)
    return dict(state.dtypes), dict(A.final(state, list(keys), aggs).dtypes)


def backend_selector():
    """Which backend the corpus runs its joins on.

    `PCK_BACKEND=recipe` routes every join through the FlatBuffers emulation
    (`operators/recipe_join.py`): the builders emit the same `Cudf*` node sequence the C++
    reads off the wire, and a python model of the cuDF calls interprets it. Running the whole
    corpus that way is the claim under test — that the frozen fbs and C++ surface executes
    these queries' joins without bending over backwards — checked against the same DuckDB
    answers as the pandas backend, so the two backends cannot agree on a wrong answer by
    sharing an implementation.

    Unset it and joins run on plain pandas, which is faster and is what the default sweep
    uses.
    """
    chosen = os.environ.get("PCK_BACKEND", "pandas")
    if chosen == "pandas":
        return CpuBackendSelector()
    if chosen == "recipe":
        return RecipeJoinBackendSelector()
    raise SystemExit(f"PCK_BACKEND={chosen} is not one of pandas, recipe")


def execute(root, budget: int | None = BUDGET):
    driver = batch_partitioned_driver(Plan.build(root), backend_selector(), budget)
    driver.run()
    # The plan's own concatenate, not pandas': it keeps the first batch's schema when every
    # batch is empty, and a query whose answer is the empty set still has to report its
    # columns. TPC-DS q17 is that query — dropping the empty batches here left a frame with
    # no columns at all, which reads as a lowering that produced the wrong ones.
    frames = [batch.frame for batch in driver.results]
    got = concatenate(frames) if frames else pd.DataFrame()
    return got, driver


def same(got: pd.DataFrame, want: pd.DataFrame, label: str) -> None:
    """Row order is not part of the contract unless a sort says so."""
    assert list(got.columns) == list(want.columns), f"{label}: {list(got.columns)}"
    got = got.sort_values(list(got.columns)).reset_index(drop=True)
    want = want.sort_values(list(want.columns)).reset_index(drop=True)
    assert len(got) == len(want), f"{label}: {len(got)} rows vs {len(want)}"
    for column in want.columns:
        left, right = got[column].to_numpy(), want[column].to_numpy()
        # pandas' predicate rather than np.issubdtype: a pandas 3 string column is an
        # extension dtype numpy refuses to classify.
        if pd.api.types.is_numeric_dtype(want[column]):
            assert np.allclose(left.astype(float), right.astype(float), equal_nan=True), (
                f"{label}: column {column}"
            )
        else:
            # NaN != NaN: normalize nulls before comparing a non-numeric column.
            def nulls_alike(values):
                return [None if pd.isna(v) else v for v in values]

            assert nulls_alike(left) == nulls_alike(right), f"{label}: column {column}"


def in_order(got: pd.DataFrame, want: pd.DataFrame, label: str, order_by=None) -> None:
    """For a query whose ORDER BY is part of the answer — compare positionally.

    `order_by` names the columns the query actually sorts on. Given them, only those are
    compared positionally and the rows themselves as a multiset — `matches_oracle`, the same
    split the TPC-DS side uses, and the whole of what SQL determines. q11 is why: it sorts
    on `value` alone, two of its parts hold the same value, and which of them comes first is
    open. Comparing every column positionally called that a wrong answer.

    Without `order_by` every column is compared positionally, which asserts more than SQL
    promises. The rest of the TPC-H corpus satisfies it because their sort keys are unique
    in this data. A query that starts failing there has found a tie, and the fix is to name
    its `order_by` here — not to re-sort the oracle until the two agree.
    """
    if order_by is not None:
        matches_oracle(got, want, label, order_by, oracle="the oracle")
        return
    assert list(got.columns) == list(want.columns), f"{label}: {list(got.columns)}"
    assert len(got) == len(want), f"{label}: {len(got)} rows vs {len(want)}"
    for column in want.columns:
        left, right = got[column].to_numpy(), want[column].to_numpy()
        if pd.api.types.is_numeric_dtype(want[column]):
            assert np.allclose(left.astype(float), right.astype(float), equal_nan=True), (
                f"{label}: column {column} out of order or wrong"
            )
        else:
            assert list(left) == list(right), f"{label}: column {column}"
