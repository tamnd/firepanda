"""`firepanda.test`, which runs the test suite the way `pandas.test` runs pandas'.

pandas ships its tests inside the package, so it points pytest at its own
directory. firepanda's tests sit next to the package in a source checkout and
are not in the wheel, so the suite is found there when it exists and the
package directory is used otherwise, which still collects the doctests.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

PKG = os.path.dirname(__file__)
_MISSING = "Missing optional dependency '{}'.  Use pip or conda to install {}."


def _suite() -> str:
    tests = os.path.join(os.path.dirname(PKG), "tests")
    return tests if os.path.isdir(tests) else PKG


def test(extra_args: list[str] | None = None, run_doctests: bool = False) -> None:
    """Run the test suite with pytest and exit with its status.

    By default the marks are `-m "not slow and not network and not db"`, as in
    pandas. `extra_args` replaces them, and `run_doctests` runs only the
    doctests in the package.

    Raises:
        ImportError: If pytest is not installed, with pandas' sentence.
        SystemExit: Always, with pytest's exit status, as pandas does.
    """
    try:
        pytest: Any = importlib.import_module("pytest")
    except ImportError:
        raise ImportError(_MISSING.format("pytest", "pytest")) from None
    cmd = ["-m not slow and not network and not db"]
    if extra_args:
        if not isinstance(extra_args, list):
            extra_args = [extra_args]
        cmd = extra_args
    if run_doctests:
        cmd = ["--doctest-modules", PKG]
    else:
        cmd += [_suite()]
    print(f"running: pytest {' '.join(cmd)}")
    sys.exit(pytest.main(cmd))
