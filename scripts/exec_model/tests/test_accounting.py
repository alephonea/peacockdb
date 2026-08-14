"""Resident accounting, against the formula the spec states.

    resident = Σ byte_size of driver-held in-flight batches
             + Σ cached resident_bytes() over live executors

Two properties get most of the attention. The cached executor total must equal what a live
sum would give — caching is the whole point (it is what lets the Rust port drop the
accountant's references to every executor) and a cache that drifts is worse than no cache.
Model-versus-measured is deliberately NOT among them. `scratch_bytes` is an estimate — a
join's rests on the optimizer's cardinality figure, a filter's on assumed selectivity — so
it will sometimes come in under, and the enforcer is built for that ("fail cleanly when the
accounted peak exceeds the budget"). What is tested is that under-estimates are *recorded*
with their magnitude, not that they never happen.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

import pandas as pd

from .harness import main, raises
from ..accounting import ResidentAccountant
from ..batch import CallStats
from ..batch_partitioned_driver import batch_partitioned_driver
from ..errors import ResidentBudgetExceeded
from ..node import CpuBackendSelector
from ..operators import aggregates as A
from ..operators import nodes as N
from ..operators.expressions import Binary, Col, Lit
from ..operators.frame import PandasBatch
from ..operators.joins import HashJoin, JoinType
from ..plan import Plan
from .mocks import MockBatch, MockSelector, coalesce_all, sink, source


class StatefulExecutor:
    """Minimal Executor: residency the test sets, model the test states."""

    def __init__(self, resident=0, model=0):
        self._resident = resident
        self._model = model

    def resident_bytes(self) -> int:
        return self._resident

    def scratch_bytes(self, n_rows: int, n_bytes: int) -> int:
        return self._model

    def set_resident(self, value):
        self._resident = value


# -- the formula ------------------------------------------------------------------


def test_resident_is_in_flight_plus_executor_state():
    acct = ResidentAccountant()
    a, b = MockBatch("a", 3), MockBatch("b", 5)
    acct.hold(a)
    acct.hold(b)
    assert acct.in_flight_bytes == a.byte_size() + b.byte_size()

    executor = StatefulExecutor(resident=100)
    acct.refresh("e0", executor)
    assert acct.resident() == a.byte_size() + b.byte_size() + 100

    acct.release(a)
    assert acct.resident() == b.byte_size() + 100


def test_the_cached_executor_total_tracks_a_live_sum():
    # The cache is refreshed one instance at a time; it must still equal the sum over all
    # of them, or the enforcer is deciding on a stale number.
    acct = ResidentAccountant()
    executors = {f"e{i}": StatefulExecutor(resident=i * 10) for i in range(5)}
    for label, executor in executors.items():
        acct.refresh(label, executor)
    assert acct.executor_bytes == sum(e.resident_bytes() for e in executors.values())

    executors["e2"].set_resident(999)
    acct.refresh("e2", executors["e2"])  # only the one that changed
    assert acct.executor_bytes == sum(e.resident_bytes() for e in executors.values())

    executors["e4"].set_resident(0)
    acct.refresh("e4", executors["e4"])
    assert acct.executor_bytes == sum(e.resident_bytes() for e in executors.values())


def test_forgetting_an_executor_removes_its_contribution():
    acct = ResidentAccountant()
    acct.refresh("e0", StatefulExecutor(resident=70))
    acct.refresh("e1", StatefulExecutor(resident=30))
    assert acct.executor_bytes == 100
    acct.forget("e0")
    assert acct.executor_bytes == 30


def test_releasing_a_batch_that_was_never_held_is_an_error():
    # usize underflow in Rust. Caught here rather than silently going negative — this
    # fired for real when the unit-test harness seeded queues without holding.
    acct = ResidentAccountant()
    with raises(AssertionError, match="in-flight went negative"):
        acct.release(MockBatch("ghost", 4))


# -- the enforcer -----------------------------------------------------------------


def test_the_pre_check_trips_before_the_call_runs():
    # scratch is modelled BEFORE the call, so a call that would exceed the budget never
    # happens — which is the difference between failing cleanly and OOMing the device.
    acct = ResidentAccountant(budget=100)
    with raises(ResidentBudgetExceeded, match="pre-call"):
        acct.begin_call("node", StatefulExecutor(model=500), 10, 10)


def test_the_post_check_trips_on_residency_the_model_did_not_predict():
    acct = ResidentAccountant(budget=100)
    executor = StatefulExecutor(resident=0, model=0)
    acct.begin_call("node", executor, 0, 0)
    executor.set_resident(500)  # the call kept far more than it modelled
    with raises(ResidentBudgetExceeded, match="post-call"):
        acct.end_call("node", executor)


def test_no_budget_means_it_accounts_without_tripping():
    acct = ResidentAccountant()
    acct.begin_call("node", StatefulExecutor(model=10**9), 0, 0)
    assert acct.peak >= 0


def test_the_peak_is_the_high_water_mark_not_the_final_value():
    acct = ResidentAccountant()
    big = MockBatch("big", 1000)
    acct.hold(big)
    peak = acct.peak
    acct.release(big)
    assert acct.peak == peak > 0
    assert acct.resident() == 0


# -- through the drivers ----------------------------------------------------------


def shuffle_plan(rows=40, lanes=4):
    df = pd.DataFrame({"g": ["x", "y"] * (rows // 2), "v": list(range(rows))})
    aggs = [A.Agg(A.SUM, "v", "s"), A.Agg(A.MEAN, "v", "m")]
    scan = N.scan("scan", df, lanes, 5, 10)
    filtered = N.filter_("filter", scan, Binary(">", Col("v"), Lit(3)))
    partial = N.partial_aggregate("agg_partial", filtered, ["g"], aggs)
    compacted = N.aggregate_batches("agg_batches", partial, ["g"], aggs, final=False)
    emitted = N.emit_partitions("emit", N.merge_partitions("merge", compacted), ["g"], lanes)
    return N.unload("unload", N.aggregate_batches("agg_final", emitted, ["g"], aggs, True))


def join_plan():
    dim = pd.DataFrame({"k": [0, 1, 2], "label": list("ABC")})
    fact = pd.DataFrame({"k": [0, 1, 1, 2, 2, 2], "v": list(range(6))})
    build = N.coalesce_all("build", N.scan("dim", dim, 1, 2), schema=list(dim.columns))
    probe = N.scan("fact", fact, 1, 2)
    return N.unload("unload", N.hash_join("join", build, probe, JoinType.INNER, ["k"], ["k"]))


def run(root, budget=None):
    driver = batch_partitioned_driver(Plan.build(root), CpuBackendSelector(), budget)
    driver.run()
    return driver


def test_in_flight_returns_to_zero_when_a_query_completes():
    for plan in (shuffle_plan(), join_plan()):
        driver = run(plan)
        assert driver.accountant.in_flight_bytes == 0


def test_model_accuracy_is_recorded_rather_than_enforced():
    # Both plans run to completion whatever the model does. The figures are reported so a
    # developer can see model quality; nothing here asserts the model was right.
    for label, plan in (("shuffle", shuffle_plan()), ("join", join_plan())):
        driver = run(plan)
        assert driver.accountant.calls > 0, label
        assert driver.accountant.worst_underestimate() >= 1.0, label


def test_an_under_predicting_model_is_recorded_with_its_magnitude():
    # The recording mechanism itself, driven with a model that is deliberately too small.
    acct = ResidentAccountant()
    executor = StatefulExecutor(model=10)
    modelled = acct.begin_call("stingy", executor, 1, 10)
    acct.end_call("stingy", executor, CallStats(scratch_bytes=40), modelled)

    assert [u.label for u in acct.underestimates] == ["stingy"]
    assert acct.underestimates[0].ratio == 4.0
    assert acct.worst_underestimate() == 4.0


def test_a_cardinality_estimate_lets_the_join_model_its_own_scratch():
    # The estimate is a NODE property the optimizer supplies, carried into the executor at
    # construction — so it reaches scratch_bytes through `&self` without the signature
    # changing. With a fan-out figure the join models the merged frame it is about to
    # build; with none it falls back to the constant estimator's 1.0 (#19).
    dim = pd.DataFrame({"k": [0, 1, 2], "label": list("ABC")})
    fact = pd.DataFrame({"k": [0, 1, 1, 2, 2, 2], "v": list(range(6))})

    modelled = []
    for fanout in (1.0, 3.0):
        join = HashJoin(JoinType.INNER, ["k"], ["k"], fanout=fanout)
        join.set_build(PandasBatch(dim, "B"))
        probe = PandasBatch(fact, "P")
        modelled.append(join.scratch_bytes(probe.num_rows(), probe.byte_size()))
        join.probe_and_fetch(probe)

    # A larger estimate means a larger model — the estimate is actually consulted.
    assert modelled[1] > modelled[0]


def test_accumulator_residency_is_visible_while_it_holds_rows():
    # An accumulator is where mandatory residency lives, so the accountant must see it
    # rise before the flush rather than only after.
    plan = Plan.build(sink("u", coalesce_all("collect", source("load", [[10, 10, 10]]))))
    driver = batch_partitioned_driver(plan, MockSelector())
    seen = []
    while driver.step():
        seen.append(driver.accountant.executor_bytes)
    assert max(seen) > 0, "the accumulator's held bytes never appeared in the total"


def test_a_tight_budget_fails_the_query_cleanly():
    with raises(ResidentBudgetExceeded):
        run(shuffle_plan(rows=400), budget=500)


def test_a_generous_budget_completes_and_records_a_peak():
    driver = run(shuffle_plan(), budget=50_000_000)
    assert 0 < driver.accountant.peak <= 50_000_000


def test_an_absent_measurement_is_not_recorded_as_an_underestimate():
    # None means an un-instrumented run; it must not be read as zero and flagged.
    acct = ResidentAccountant()
    executor = StatefulExecutor(model=0)
    modelled = acct.begin_call("gpu", executor, 0, 0)
    acct.end_call("gpu", executor, CallStats(scratch_bytes=None), modelled)
    assert not acct.underestimates


if __name__ == "__main__":
    raise SystemExit(main(globals()))
