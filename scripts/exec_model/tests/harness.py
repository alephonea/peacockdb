"""Minimal stdlib test harness, so a test file runs with plain `python <file>`.

The prototype has no project dependencies and the system python here has no pytest, so
the two things this suite actually used from pytest — a `raises` context manager and a
runner — live here instead. Under pytest only `raises` is used, and it behaves the same,
so both entry points run the identical tests.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import re
import sys
import traceback
from contextlib import contextmanager


@contextmanager
def raises(exception, match=None):
    """Assert the block raises `exception`, optionally with a message matching `match`."""
    try:
        yield
    except exception as caught:
        if match is not None and not re.search(match, str(caught)):
            raise AssertionError(
                f"{exception.__name__} raised, but {str(caught)!r} does not match {match!r}"
            ) from None
        return
    raise AssertionError(f"{exception.__name__} was not raised")


#: `async` matters: an async test lands in the namespace and is callable, so it would be
#: called, return a coroutine, and be reported ok having asserted nothing.
_DEF = re.compile(r"^(?:async )?def (test_\w+)", re.M)


def _source_problems(namespace, collected) -> list[str]:
    """Ways the source file's tests can fail to reach the runner, in the file's own text.

    `main` runs from the module's `__main__` footer, so anything defined *below* that
    footer does not exist yet — it would be skipped here while pytest still collected it.
    Duplicate names are the other direction: both are in the source, only the second is in
    the namespace, and pytest is blind to it too.
    """
    defined = _DEF.findall(pathlib.Path(namespace["__file__"]).read_text())
    problems = []
    missed = [name for name in defined if name not in collected]
    if missed:
        problems.append(f"defined below the __main__ footer, so never run: {', '.join(missed)}")
    duplicated = sorted({name for name in defined if defined.count(name) > 1})
    if duplicated:
        problems.append(f"defined more than once, so only the last runs: {', '.join(duplicated)}")
    return problems


def _selected(tests, path):
    """The subset this process runs: every test, a named few, or one shard of them.

    Two selectors, because two callers want different things. A person debugging wants
    names — `python3 tests/test_tpch.py q3 q12`. A CI matrix wants a share of the file
    without naming anything, so `PCK_SHARD=k/n` takes every n-th test starting at k; the
    corpus files are minutes long and their queries are independent, so n processes finish
    in a fraction of the time one does.

    Both refuse to select nothing. A shard or a name that matches no test would otherwise
    print `0 passed` and exit 0, which is the "green having verified nothing" failure the
    empty-glob and PASSED-0 guards elsewhere exist to prevent.
    """
    names = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if names:
        # An argument that *is* a test's name selects that test and nothing else. Without
        # this, `test_tpcds.py test_corpus_q1` also runs q10, q11, q13, q15, q16 and q19 —
        # substring matching is what a person wants when they type `q3`, and exactly what
        # they do not want when they have named a test in full.
        available = {name for name, _ in tests}
        selected = [
            (name, obj) for name, obj in tests
            if any(part == name if part in available else part in name for part in names)
        ]
        if not selected:
            raise SystemExit(f"  FAIL {path}: no test matches {names}")
        tests = selected
    # `-substring` drops matching tests. One file can then serve two CI steps: the corpus
    # queries in `test_tpch.py` are minutes long and run on manual dispatch, while the same
    # file's short plan tests keep running on every push as `test_tpch.py -corpus`.
    excluded = [arg[1:] for arg in sys.argv[1:] if arg.startswith("-") and len(arg) > 1]
    if excluded:
        tests = [(n, o) for n, o in tests if not any(part in n for part in excluded)]
        if not tests:
            raise SystemExit(f"  FAIL {path}: excluding {excluded} leaves no test")
    shard = os.environ.get("PCK_SHARD")
    if shard:
        index, total = (int(part) for part in shard.split("/"))
        if not 0 <= index < total:
            raise SystemExit(f"  FAIL {path}: PCK_SHARD={shard} is not k/n with 0 <= k < n")
        tests = tests[index::total]
        if not tests:
            raise SystemExit(
                f"  FAIL {path}: shard {shard} selects no test — fewer tests than shards"
            )
    return tests


def main(namespace) -> int:
    """Run every `test_*` callable in `namespace`, in definition order.

    Pass `globals()`. Returns a process exit code. See `_selected` for running a subset.
    """
    tests = [
        (name, obj)
        for name, obj in namespace.items()
        if name.startswith("test_") and callable(obj)
    ]
    path = namespace["__file__"]
    for problem in _source_problems(namespace, {name for name, _ in tests}):
        print(f"  FAIL {path}: {problem}")
        return 1
    # After the source checks, which must see the whole file however little this process
    # is about to run.
    tests = _selected(tests, path)
    # A file that runs nothing must not report success: the CI step prints one line per
    # file, so zero tests reads exactly like a healthy run. Same guard as the per-binary
    # "PASSED 0 tests" check the GPU job makes.
    if not tests:
        print(f"  FAIL {path}: no test_* functions to run")
        return 1
    failed = []
    for name, test in tests:
        try:
            result = test()
            # An async test would return an un-awaited coroutine and report ok having
            # asserted nothing. The regex above already refuses to let one through; this
            # is the same refusal at the point of call, where a decorator could hide it.
            if inspect.iscoroutine(result):
                result.close()
                raise AssertionError("returned a coroutine — this runner does not await")
            print(f"  ok   {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL {name}\n{traceback.format_exc()}")
    print(f"{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0
