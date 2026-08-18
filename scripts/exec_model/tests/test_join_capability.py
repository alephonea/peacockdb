"""Every join mode the engine can be asked for, on two backends that share no join code.

The batch-partitioned mode keeps the FlatBuffers schema and the C++ operators frozen, and
that claim is worth exactly what the join can do — the join being the one operator whose
state has to survive a call. So each mode here runs twice: once on the pandas backend
(`operators/joins.py`), once on the backend that answers every call by emitting fb nodes
and making `execute_node` calls against the cuDF model (`operators/recipe_join.py`). A
disagreement between them is a real one, and the spec's join capability matrix is what they
are both asserting.

Split out of `test_end_to_end.py`, which it shares helpers with: same drivers, same
oracle-comparison rules, same budget. The join block simply outgrew that file.
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python scripts/exec_model/tests/<file>.py`
    import pathlib as _pathlib, sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "scripts.exec_model.tests"

import inspect
import pathlib
import re

import numpy as np
import pandas as pd

from .harness import main, raises
from .test_end_to_end import BUDGET, CONFIGS, canonical, execute, fixture, same
from ..node import CpuBackendSelector, RecipeJoinBackendSelector
from ..operators import nodes as N
from ..operators import recipe as R
from ..operators import recipe_join
from ..operators.cudf_calls import Table
from ..operators.expressions import Binary, Col, Lit
from ..operators.injection import HashMode, LayoutInjector, LayoutPreset
from ..operators.join_types import JoinType, capability, joined_projection
from ..operators.recipe_join import RecipeJoin

# -- the shape with same-named keys ------------------------------------------------
#
# These three join on `k` against `k`, so the join's `projection` has a duplicate to
# drop; the matrix below joins `k` against `fk`, where both key columns survive. Two
# different paths through `joined_projection`, and both are the general case for some
# query.


def dim_fixture():
    return pd.DataFrame({"k": [0, 1, 2, 3, 9], "label": list("ABCDE")})


def test_inner_join_with_a_shuffle_on_both_sides():
    fact, dim = fixture(), dim_fixture()
    want = dim.merge(fact, how="inner", on="k")
    for parts, group, target in CONFIGS:
        build = N.coalesce_all(
            "build_collect",
            N.emit_partitions(
                "build_emit",
                N.merge_partitions("build_merge", N.scan("dim", dim, parts, group, target)),
                ["k"],
                parts,
            ),
            schema=dict(dim.dtypes),
        )
        probe = N.emit_partitions(
            "probe_emit",
            N.merge_partitions("probe_merge", N.scan("fact", fact, parts, group, target)),
            ["k"],
            parts,
        )
        root = N.unload(
            "unload", N.hash_join("join", build, probe, JoinType.INNER, ["k"], ["k"])
        )
        got, _ = execute(root)
        same(got, want, f"inner join {(parts, group, target)}")


def test_left_outer_join_finish_pass_matches_the_oracle():
    fact, dim = fixture(), dim_fixture()
    want = dim.merge(fact, how="left", on="k")
    for parts, group, target in CONFIGS:
        build = N.coalesce_all(
            "build_collect",
            N.emit_partitions(
                "build_emit",
                N.merge_partitions("build_merge", N.scan("dim", dim, parts, group, target)),
                ["k"],
                parts,
            ),
            schema=dict(dim.dtypes),
        )
        probe = N.emit_partitions(
            "probe_emit",
            N.merge_partitions("probe_merge", N.scan("fact", fact, parts, group, target)),
            ["k"],
            parts,
        )
        root = N.unload(
            "unload",
            # probe_schema: a post-shuffle probe lane can be empty, and the finish must
            # still null-pad the probe columns it never saw.
            N.hash_join(
                "join", build, probe, JoinType.LEFT, ["k"], ["k"],
                probe_schema=list(fact.columns),
            ),
        )
        got, _ = execute(root)
        same(got, want, f"left outer join {(parts, group, target)}")


def test_semi_and_anti_joins_match_the_oracle():
    fact, dim = fixture(), dim_fixture()
    present = set(fact.k.unique())
    for join_type, keep in ((JoinType.LEFT_SEMI, True), (JoinType.LEFT_ANTI, False)):
        want = dim[dim.k.isin(present) == keep].reset_index(drop=True)
        for parts, group, target in CONFIGS:
            build = N.coalesce_all(
                "build_collect",
                N.emit_partitions(
                    "build_emit",
                    N.merge_partitions("build_merge", N.scan("dim", dim, parts, group, target)),
                    ["k"],
                    parts,
                ),
                schema=dict(dim.dtypes),
            )
            probe = N.emit_partitions(
                "probe_emit",
                N.merge_partitions("probe_merge", N.scan("fact", fact, parts, group, target)),
                ["k"],
                parts,
            )
            root = N.unload("unload", N.hash_join("join", build, probe, join_type, ["k"], ["k"]))
            got, _ = execute(root)
            same(got, want, f"{join_type.value} {(parts, group, target)}")


# -- the join capability matrix ----------------------------------------------------
#
# Every join shape the engine can be asked for, run on both backends at every batching and
# partitioning config. The two backends share no join code: `joins.py` joins with pandas,
# `recipe_join.py` answers each call by emitting FlatBuffers nodes and making `execute_node`
# calls against the cuDF model in `cudf_calls.py`. Agreement is therefore evidence, and
# what it is evidence *for* is the mode's load-bearing claim: that every join type is
# expressible on the frozen C++ surface, with a probe side arriving in batches.
#
# The build side is always left and always one batch; the probe side streams unless the
# matrix says it cannot. Key columns are named differently on the two sides (`k` / `fk`) so
# both survive the join, which is the general case — the same-name case, where the join's
# `projection` drops the duplicate, is what the three tests above it exercise.


#: (label, selector) — the constructor, so each run gets its own.
BACKENDS = (("pandas", CpuBackendSelector), ("recipe", RecipeJoinBackendSelector))


def build_side():
    """The small side. `k=9` matches no probe row, so every build-preserving type has an
    unmatched row to prove its finish pass on."""
    return pd.DataFrame({"k": [0, 1, 2, 3, 9], "label": list("ABCDE")})


def probe_side(n=48, seed=11):
    """The streamed side. `fk` reaches 5, which no build row has, so the probe-preserving
    types have unmatched rows of their own."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "fk": rng.integers(0, 6, n),
            "g": rng.choice(list("xyz"), n),
            "v": rng.integers(1, 100, n),
        }
    )


#: One oracle per join type, in pandas, written from the SQL meaning rather than from the
#: engine's mechanics. `label` going null in an outer result is the pad; `mark` is the
#: extra column DataFusion's mark join appends.
ORACLES = {
    JoinType.INNER: lambda b, p: b.merge(p, how="inner", left_on="k", right_on="fk"),
    JoinType.LEFT: lambda b, p: b.merge(p, how="left", left_on="k", right_on="fk"),
    JoinType.RIGHT: lambda b, p: b.merge(p, how="right", left_on="k", right_on="fk"),
    JoinType.FULL: lambda b, p: b.merge(p, how="outer", left_on="k", right_on="fk"),
    JoinType.LEFT_SEMI: lambda b, p: b[b.k.isin(p.fk)],
    JoinType.LEFT_ANTI: lambda b, p: b[~b.k.isin(p.fk)],
    JoinType.LEFT_MARK: lambda b, p: b.assign(mark=b.k.isin(p.fk)),
    JoinType.RIGHT_SEMI: lambda b, p: p[p.fk.isin(b.k)],
    JoinType.RIGHT_ANTI: lambda b, p: p[~p.fk.isin(b.k)],
}


def shuffled_sides(build_frame, probe_frame, config, single_batch_probe=False):
    """Both sides hash-partitioned on their join key — the shape a shuffle join has.

    `single_batch_probe` is the planner's other lowering: a `GpuCoalesceAllBatches` under
    the probe side, for the shapes the matrix refuses to stream.
    """
    parts, group, target = config
    build = N.coalesce_all(
        "build_collect",
        N.emit_partitions(
            "build_emit",
            N.merge_partitions("build_merge", N.scan("build_scan", build_frame, parts, group, target)),
            ["k"],
            parts,
        ),
        schema=dict(build_frame.dtypes),
    )
    probe = N.emit_partitions(
        "probe_emit",
        N.merge_partitions("probe_merge", N.scan("probe_scan", probe_frame, parts, group, target)),
        ["fk"],
        parts,
    )
    if single_batch_probe:
        probe = N.coalesce_all("probe_collect", probe, schema=dict(probe_frame.dtypes))
    return build, probe


def join_plan(build_frame, probe_frame, config, join_type, **kwargs):
    single_batch = not capability(join_type, has_filter=kwargs.get("residual") is not None).streams
    build, probe = shuffled_sides(build_frame, probe_frame, config, single_batch)
    return N.unload(
        "unload",
        N.hash_join(
            "join", build, probe, join_type, ["k"], ["fk"],
            # A post-shuffle probe lane can be empty, and a build-preserving finish must
            # still null-pad the probe columns it never saw.
            probe_schema=list(probe_frame.columns),
            **kwargs,
        ),
    )


def collapsed_sides(build_frame, probe_frame, config, single_batch_probe=False):
    """One lane each — what a join with no keys to shuffle on gets."""
    parts, group, target = config
    build = N.coalesce_all(
        "build_collect",
        N.merge_partitions("build_merge", N.scan("build_scan", build_frame, parts, group, target)),
        schema=dict(build_frame.dtypes),
    )
    probe = N.merge_partitions(
        "probe_merge", N.scan("probe_scan", probe_frame, parts, group, target)
    )
    if single_batch_probe:
        probe = N.coalesce_all("probe_collect", probe, schema=dict(probe_frame.dtypes))
    return build, probe


def test_every_join_type_matches_the_oracle_on_both_backends():
    build, probe = build_side(), probe_side()
    for join_type, oracle in ORACLES.items():
        want = oracle(build, probe)
        for label, selector in BACKENDS:
            for config in CONFIGS:
                got, _ = execute(
                    join_plan(build, probe, config, join_type), selector=selector()
                )
                same(got, want, f"{join_type.name} on {label} at {config}")


def test_a_cross_join_streams_its_probe_side_on_both_backends():
    build, probe = build_side(), probe_side(n=12)
    want = build.merge(probe, how="cross")
    for label, selector in BACKENDS:
        for config in CONFIGS:
            b, p = collapsed_sides(build, probe, config)
            got, _ = execute(N.unload("unload", N.cross_join("cross", b, p)), selector=selector())
            same(got, want, f"cross join on {label} at {config}")


def test_a_nested_loop_join_matches_the_oracle_on_both_backends():
    # An inequality, which is what makes DataFusion choose a nested-loop join over a hash
    # join: there is no equality to hash on.
    build, probe = build_side(), probe_side(n=12)
    predicate = Binary("<", Col("k"), Col("fk"))
    crossed = build.merge(probe, how="cross")
    inner_want = crossed[crossed.k < crossed.fk].reset_index(drop=True)
    unmatched = build[~build.k.isin(set(inner_want.k))]
    left_want = pd.concat(
        [inner_want, unmatched.assign(**{c: np.nan for c in probe.columns})],
        ignore_index=True,
    )
    for label, selector in BACKENDS:
        for config in CONFIGS:
            for join_type, want in ((JoinType.INNER, inner_want), (JoinType.LEFT, left_want)):
                single_batch = not capability(join_type, has_filter=True, nested_loop=True).streams
                b, p = collapsed_sides(build, probe, config, single_batch)
                got, _ = execute(
                    N.unload("unload", N.nested_loop_join("nlj", b, p, join_type, predicate)),
                    selector=selector(),
                )
                same(got, want, f"nested-loop {join_type.name} on {label} at {config}")


def test_a_residual_filter_rides_the_join_on_both_backends():
    # `CudfHashJoin.filter`: an equi-join plus a condition over both sides' columns — TPC-H
    # q17's and q21's shape, and the only shapes the corpus has (residual filters appear on
    # Inner, LeftSemi and LeftAnti and nowhere else). Inner keeps a streamed probe, because
    # the filter is evaluated per (build, probe batch) pair and every emitted row is decided
    # locally; the semi family falls back to a single-batch probe, because #136's finish
    # pass sees accumulated keys and a keys-only table cannot evaluate a predicate over both
    # sides.
    build, probe = build_side(), probe_side()
    residual = Binary(">", Col("v"), Lit(50))
    joined = build.merge(probe, how="inner", left_on="k", right_on="fk")
    passing = joined[joined.v > 50]
    wanted = {
        JoinType.INNER: passing.reset_index(drop=True),
        JoinType.LEFT_SEMI: build[build.k.isin(set(passing.k))],
        JoinType.LEFT_ANTI: build[~build.k.isin(set(passing.k))],
    }
    assert capability(JoinType.INNER, has_filter=True).streams
    assert not capability(JoinType.LEFT_SEMI, has_filter=True).streams
    for label, selector in BACKENDS:
        for config in CONFIGS:
            for join_type, want in wanted.items():
                got, _ = execute(
                    join_plan(build, probe, config, join_type, residual=residual),
                    selector=selector(),
                )
                same(got, want, f"residual {join_type.name} on {label} at {config}")


def test_an_outer_join_with_a_residual_filter_is_refused_not_answered_wrongly():
    # Found by writing this suite: `execute_hash_join` applies a residual filter with
    # apply_boolean_mask AFTER the outer gather (join.cpp ~L353), so a padded row's NULL
    # columns make the predicate NULL and the row is dropped — an ON-condition demoted to a
    # WHERE. Latent in the legacy engine, which no corpus query reaches, and #153. Until it
    # is fixed the shape is refused at plan time, which is the spec's rule for an
    # unsupported shape inside a supported feature.
    for join_type in (JoinType.LEFT, JoinType.RIGHT, JoinType.FULL):
        assert capability(join_type, has_filter=True).refusal is not None
        assert capability(join_type, has_filter=False).streams
    build, probe = build_side(), probe_side()
    for _label, selector in BACKENDS:
        with raises(NotImplementedError):
            execute(
                join_plan(build, probe, (1, 5, 10), JoinType.LEFT,
                          residual=Binary(">", Col("v"), Lit(50))),
                selector=selector(),
            )


def test_a_single_batch_probe_join_refuses_a_second_batch():
    # The mutation check for the test above it: without the planner's GpuCoalesceAllBatches
    # on the probe side, the join is handed batch after batch and has to say so. Both
    # backends refuse, because both read the same capability function.
    build, probe = build_side(), probe_side()
    for _label, selector in BACKENDS:
        b, p = shuffled_sides(build, probe, (1, 5, 10), single_batch_probe=False)
        root = N.unload(
            "unload",
            N.hash_join("join", b, p, JoinType.LEFT_SEMI, ["k"], ["fk"],
                        residual=Binary(">", Col("v"), Lit(50))),
        )
        with raises(AssertionError):
            execute(root, selector=selector())


def test_a_right_semi_join_with_a_residual_filter_is_refused_not_approximated():
    # The one shape with no path on the frozen surface: the swapped mixed_* variant does
    # not exist, so the C++ throws (join.cpp ~L155). The translation layer's answer is to
    # keep the emitted side as the build, which keeps the join a Left form — and until it
    # does, the refusal is loud rather than a wrong answer.
    assert capability(JoinType.RIGHT_SEMI, has_filter=True).refusal is not None
    assert capability(JoinType.RIGHT_ANTI, has_filter=True).refusal is not None
    assert capability(JoinType.RIGHT_SEMI, has_filter=False).streams
    build, probe = build_side(), probe_side()
    for _label, selector in BACKENDS:
        with raises(NotImplementedError):
            execute(
                join_plan(build, probe, (1, 5, 10), JoinType.RIGHT_SEMI,
                          residual=Binary(">", Col("v"), Lit(50))),
                selector=selector(),
            )


# -- what the recipe backend actually issues ---------------------------------------


def join_executors(driver):
    """The recipe join executors a run created, one per join lane.

    Reaching for the lane driver's executor rather than constructing one: the claim under
    test is about the calls a *run* made.
    """
    found = []
    for state in driver.states:
        for lane_driver in state.lane_drivers.values():
            executor = lane_driver._executor
            if isinstance(executor, RecipeJoin):
                found.append(executor)
    return found


#: The recipe plan each join type emits — the fourth column of the spec's join table, as
#: an assertion. A type whose probe calls emit nothing (semi / anti / mark) has no
#: per-batch join node at all: its probe calls are the key project.
SEQUENCES = {
    JoinType.INNER: ["CudfHashJoin"],
    JoinType.RIGHT: ["CudfHashJoin"],
    JoinType.RIGHT_SEMI: ["CudfHashJoin"],
    JoinType.RIGHT_ANTI: ["CudfHashJoin"],
    JoinType.LEFT: [
        "CudfHashJoin", "CudfProject", "CudfCoalescePartitions", "CudfHashJoin", "CudfProject",
    ],
    JoinType.FULL: [
        "CudfHashJoin", "CudfProject", "CudfCoalescePartitions", "CudfHashJoin", "CudfProject",
    ],
    JoinType.LEFT_SEMI: ["CudfProject", "CudfCoalescePartitions", "CudfHashJoin"],
    JoinType.LEFT_ANTI: ["CudfProject", "CudfCoalescePartitions", "CudfHashJoin"],
    JoinType.LEFT_MARK: ["CudfProject", "CudfCoalescePartitions", "CudfHashJoin"],
}


def test_each_join_type_emits_the_documented_node_sequence():
    build, probe = build_side(), probe_side()
    for join_type, expected in SEQUENCES.items():
        _got, driver = execute(
            join_plan(build, probe, (4, 5, 10), join_type), selector=RecipeJoinBackendSelector()
        )
        emitted = [
            [type(node).__name__ for node in executor.session.plan]
            for executor in join_executors(driver)
        ]
        assert emitted, f"{join_type.name}: no join executor was built"
        for plan in emitted:
            # A lane that saw no probe batch and needs no finish emits nothing at all.
            assert plan in (expected, []), f"{join_type.name}: {plan} vs {expected}"
        assert any(plan for plan in emitted), f"{join_type.name}: no lane emitted a plan"


def test_a_streamed_probe_needs_the_build_side_once_per_batch():
    # #152, quantified. Every execute_node call erases the handles it reads, so a join
    # whose probe streams needs its build side again for the next batch and there is no
    # node that duplicates a handle. What that costs, per type:
    #
    #   INNER (and the other probe-local types): one build copy per probe batch.
    #   LEFT / FULL: the same, plus one copy of each probe batch, since the join consumes
    #                the batch and #136's key accumulation needs it too.
    #   LEFT_SEMI / LEFT_ANTI / LEFT_MARK: none — the probe calls are the key project, so
    #                the build side is untouched until the finish consumes it.
    build, probe = build_side(), probe_side()
    expected = {
        JoinType.INNER: ("build",),
        JoinType.LEFT: ("build", "probe"),
        JoinType.FULL: ("build", "probe"),
        JoinType.LEFT_SEMI: (),
        JoinType.LEFT_ANTI: (),
        JoinType.LEFT_MARK: (),
    }
    for join_type, reasons in expected.items():
        _got, driver = execute(
            join_plan(build, probe, (4, 5, 10), join_type), selector=RecipeJoinBackendSelector()
        )
        probed, copied = 0, 0
        for executor in join_executors(driver):
            session = executor.session
            probed += executor.probe_calls
            copied += sum(session.copied_bytes.values())
            assert set(session.copies) == set(reasons), (
                f"{join_type.name}: copied {sorted(session.copies)}, expected {sorted(reasons)}"
            )
            for reason in reasons:
                assert session.copies[reason] == executor.probe_calls, (
                    f"{join_type.name}: {session.copies[reason]} {reason} copies against "
                    f"{executor.probe_calls} probe calls"
                )
        assert (copied > 0) == bool(reasons), f"{join_type.name}: {copied} bytes copied"
        assert probed > len(join_executors(driver)), (
            f"{join_type.name}: no lane streamed more than one probe batch, so this "
            "asserts nothing"
        )


def test_a_single_batch_probe_costs_no_copy_at_all():
    # The other side of the same coin, and the reason #152 offers it as the alternative:
    # with one probe batch the build side is used once and can be handed over outright.
    build, probe = build_side(), probe_side()
    _got, driver = execute(
        join_plan(build, probe, (4, 5, 10), JoinType.LEFT_ANTI, residual=Binary(">", Col("v"), Lit(50))),
        selector=RecipeJoinBackendSelector(),
    )
    for executor in join_executors(driver):
        assert executor.session.copies == {}, executor.session.copies


def legacy_call(join_type, build, probe, residual=None):
    """One `CudfHashJoin` over whole tables — the plan the legacy modes emit.

    The same layer the batch-partitioned lowering uses, driven the way a single-batch probe
    drives it, which is what makes the fallback in the capability matrix a known quantity
    rather than a hope.
    """
    session = R.NodeSession()
    keys = ((list(build.columns).index("k"), list(probe.columns).index("fk")),)
    seq = session.add(
        R.CudfHashJoin(
            join_type,
            keys,
            recipe_join.join_filter_of(residual, list(build.columns), list(probe.columns)),
            joined_projection(list(build.columns), list(probe.columns), keys, join_type),
        )
    )
    handle = session.execute_node(
        seq, [session.register(Table.from_frame(build)), session.register(Table.from_frame(probe))]
    )
    return session.table_for(handle).to_frame()


def test_the_recipe_layer_reproduces_the_legacy_single_call_join():
    # The outer types as one call over whole tables: `cudf::left_join` / `full_join` and the
    # NULLIFY gather on the side that can be unmatched — the arms the streamed lowering does
    # not take, since it emits Inner or Right per batch and finishes with an anti join.
    build, probe = build_side(), probe_side()
    for join_type, how in ((JoinType.LEFT, "left"), (JoinType.FULL, "outer"),
                           (JoinType.RIGHT, "right"), (JoinType.INNER, "inner")):
        want = build.merge(probe, how=how, left_on="k", right_on="fk")
        same(legacy_call(join_type, build, probe), want, f"legacy {join_type.name}")


def test_an_outer_join_with_a_residual_filter_is_wrong_today_which_is_153():
    # The defect, pinned. `execute_hash_join` applies the residual after the outer gather,
    # so the LEFT join below answers as an inner one: the unmatched build row (k=9) has NULL
    # probe columns, `v > 50` over NULL is NULL, and apply_boolean_mask drops it. Asserted
    # as it behaves rather than as it should.
    #
    # What this pin can and cannot do. It asserts `recipe.py`, and `recipe.py` mirrors
    # join.cpp by hand — no import links them — so fixing the C++ does not turn this red by
    # itself. #153 names the two files that have to move with it, and this comment is the
    # other half of that pairing: change the model, and this test is what tells you the
    # refusal in the capability matrix can come out.
    build, probe = build_side(), probe_side()
    got = legacy_call(JoinType.LEFT, build, probe, residual=Binary(">", Col("v"), Lit(50)))
    inner = build.merge(probe, how="inner", left_on="k", right_on="fk")
    same(got, inner[inner.v > 50].reset_index(drop=True), "#153: LEFT answers as INNER")
    assert 9 not in set(got.k), "the unmatched build row survived — #153 may be fixed"


# -- null keys, where the two backends could disagree for a reason -----------------


def null_key_sides():
    """Both sides carry a null key, and the shuffle co-locates them.

    The hash skips null columns (comet-mandated), so an all-null key lands in
    `pmod(seed, N)` on both sides — which is what lets a null build key and a null probe
    key meet at all. `k=2` matches nothing, so the anti result is never empty.
    """
    build = pd.DataFrame({"k": [0.0, 1.0, 2.0, None], "label": list("ABCD")})
    probe = pd.DataFrame({"fk": [0.0, 0.0, 1.0, None, None], "v": [1, 2, 3, 4, 5]})
    return build, probe


def labels(frame):
    return sorted(frame.label) if len(frame) and "label" in frame.columns else []


def test_null_keys_follow_the_c_side_asymmetry_on_both_backends():
    # join.cpp's most deliberate choice, and the one with two tickets behind it: semi takes
    # the plan's null_equals_null, while anti and mark are hardcoded EQUAL, because
    # `x NOT IN (…, NULL)` is neither EQUAL nor UNEQUAL (#80, #59). Every other fixture in
    # this file has non-null keys, so this is the only case where that line is alive —
    # invert it and nothing else here notices.
    build, probe = null_key_sides()
    got = {}
    for label, selector in BACKENDS:
        for join_type in (JoinType.LEFT_SEMI, JoinType.LEFT_ANTI, JoinType.LEFT_MARK):
            for flag in (False, True):
                frame, _ = execute(
                    join_plan(build, probe, (4, 5, 10), join_type, null_equals_null=flag),
                    selector=selector(),
                )
                got[(label, join_type, flag)] = frame

    for label, _ in BACKENDS:
        # Semi honours the flag: NULL=NULL makes the null-keyed build row a match.
        assert labels(got[(label, JoinType.LEFT_SEMI, False)]) == ["A", "B"], label
        assert labels(got[(label, JoinType.LEFT_SEMI, True)]) == ["A", "B", "D"], label
        # Anti ignores it. Under EQUAL the null build key matches the null probe key, so
        # row D counts as matched and never appears — where SQL's NOT EXISTS would emit it,
        # since a null key matches nothing. That is #80's divergence, reproduced rather
        # than corrected, and identical under both flags.
        for flag in (False, True):
            assert labels(got[(label, JoinType.LEFT_ANTI, flag)]) == ["C"], (label, flag)
        # Mark is EQUAL for the same reason: D is marked true whatever the flag says.
        for flag in (False, True):
            marks = got[(label, JoinType.LEFT_MARK, flag)].set_index("label").mark
            assert list(marks[["A", "B", "C", "D"]]) == [True, True, False, True], (label, flag)

    for key, frame in got.items():
        if key[0] == "pandas":
            same(frame, got[("recipe",) + key[1:]], f"backends disagree on {key[1:]}")


def test_an_inner_join_honours_null_equals_null_on_both_backends():
    # The control for the test above: on the equi path the flag is not ignored, so a
    # backend that dropped it would fail here rather than only in the semi case.
    build, probe = null_key_sides()
    for label, selector in BACKENDS:
        rows = {}
        for flag in (False, True):
            frame, _ = execute(
                join_plan(build, probe, (4, 5, 10), JoinType.INNER, null_equals_null=flag),
                selector=selector(),
            )
            rows[flag] = len(frame)
        # Three matching pairs on the non-null keys; NULL=NULL adds the two null probe rows.
        assert rows[False] == 3, label
        assert rows[True] == 5, label


# -- the refusals, tied to the thing that justifies them ---------------------------


def test_a_filtered_right_semi_throws_in_the_recipe_layer_as_the_c_side_does():
    # The matrix refuses RightSemi/RightAnti with a residual filter. Asserting
    # `capability(...).refusal is not None` would only restate that decision, so this drives
    # the layer the decision is *about*: the C++ throws (join.cpp ~L155, no swapped mixed_*
    # variant), and the recipe throws with it. Remove the refusal and this is what the
    # planner would have run into.
    build, probe = build_side(), probe_side()
    for join_type in (JoinType.RIGHT_SEMI, JoinType.RIGHT_ANTI):
        with raises(NotImplementedError):
            legacy_call(join_type, build, probe, residual=Binary(">", Col("v"), Lit(50)))


# -- the model against the code it models ------------------------------------------

#: The C++ this whole file is an argument about. Read, never built.
JOIN_CPP = pathlib.Path(__file__).resolve().parents[3] / "cpp/src/operators/join.cpp"

#: cuDF calls join.cpp makes that `cudf_calls.py` deliberately does not model, and why.
_NOT_MODELLED = {
    # A zero-row slice used only to build a type-only table_view for the is_ast_able
    # check. The prototype's expressions carry their types in pandas, so there is no
    # type-only view to construct.
    "slice",
}


def cpp_source() -> str:
    assert JOIN_CPP.exists(), f"{JOIN_CPP} is missing — the model has nothing to check against"
    return JOIN_CPP.read_text()


def test_the_recipe_covers_every_join_type_join_cpp_dispatches_on():
    # The recipe mirrors join.cpp by hand and no import links them, so a branch added there
    # would be invisible here. This is the cheap half of that guard: the set of join types
    # the C++ names must be the set this prototype has an enum member for. Text-level, so
    # it cannot see a semantic drift — but "a new join type appeared" stops being silent.
    named = {re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()
             for name in re.findall(r"fb::JoinType_(\w+)", cpp_source())}
    modelled = {member.name for member in JoinType}
    assert named <= modelled, f"join.cpp dispatches on types the prototype has no member for: {named - modelled}"
    assert modelled <= named, f"the prototype has members join.cpp never names: {modelled - named}"


def test_the_recipe_covers_every_cudf_call_join_cpp_makes():
    # The other half: every `cudf::x(...)` join.cpp calls is either modelled in
    # cudf_calls.py or listed above with a reason. A cuDF call appearing in the C++ that
    # the model has never heard of is exactly the drift this file cannot otherwise see.
    called = set(re.findall(r"cudf::([a-z_]+)\s*\(", cpp_source()))
    # Functions *defined* here, not every attribute: `dir()` would count an import that
    # happened to share a cuDF name as coverage, which would make the guard's reach an
    # accident of the import list rather than something it enforces.
    module = cudf_calls_module()
    modelled = {
        name
        for name, value in vars(module).items()
        if inspect.isfunction(value) and value.__module__ == module.__name__
    }
    missing = {name for name in called if name not in modelled} - _NOT_MODELLED
    assert not missing, f"join.cpp calls cuDF functions the model does not have: {sorted(missing)}"
    stale = _NOT_MODELLED - called
    assert not stale, f"the not-modelled list names calls join.cpp no longer makes: {sorted(stale)}"


def cudf_calls_module():
    from ..operators import cudf_calls

    return cudf_calls


# -- every layout, every type ------------------------------------------------------


def test_every_join_type_survives_every_layout_on_both_backends():
    # `test_tpch.py` does this to whole queries over real tables; this does it to the join
    # matrix. A join is the operator most exposed to layout — its build side must arrive as
    # exactly one batch, its lanes must be co-located, and its finish pass runs per lane —
    # so an empty lane or a hash that puts every row in one place is where it would break.
    # The hash mode rotates with the pair rather than nesting inside it: every preset and
    # every placement still appears against every type across the sweep, at a fifth of the
    # runs a full cross product would cost. Derived from the indices, so it reproduces.
    build, probe = build_side(), probe_side()
    modes = list(HashMode)
    for preset_index, preset in enumerate(LayoutPreset):
        for type_index, (join_type, oracle) in enumerate(ORACLES.items()):
            want = oracle(build, probe)
            plan = join_plan(build, probe, (2, 5, 10), join_type)
            hash_mode = modes[(preset_index + type_index) % len(modes)]
            injector = LayoutInjector(preset, hash_mode, empty_batch_probability=0.2)
            for label, selector in BACKENDS:
                got, _ = execute(injector.apply(plan), selector=selector())
                same(got, want, f"{join_type.name} on {label} at {injector.label}")


if __name__ == "__main__":
    raise SystemExit(main(globals()))
