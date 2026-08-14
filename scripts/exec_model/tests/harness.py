"""Minimal stdlib test harness, so a test file runs with plain `python <file>`.

The prototype has no project dependencies and the system python here has no pytest, so
the two things this suite actually used from pytest — a `raises` context manager and a
runner — live here instead. Under pytest only `raises` is used, and it behaves the same,
so both entry points run the identical tests.
"""

from __future__ import annotations

import inspect
import pathlib
import re
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


def main(namespace) -> int:
    """Run every `test_*` callable in `namespace`, in definition order.

    Pass `globals()`. Returns a process exit code.
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
