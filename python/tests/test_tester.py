"""`firepanda.test`, which runs the suite the way `pandas.test` runs pandas'.

The runner is swapped for one that records what it was asked to run, so these
tests never start a second pytest inside the first.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def run(firepanda: ModuleType, monkeypatch: Any, capsys: Any, **kwargs: Any) -> tuple[Any, ...]:
    seen: list[list[str]] = []

    def main(cmd: list[str]) -> int:
        seen.append(cmd)
        return 3

    monkeypatch.setattr(pytest, "main", main)
    with pytest.raises(SystemExit) as stop:
        firepanda.test(**kwargs)
    return stop.value.code, seen[0], capsys.readouterr().out


def test_the_default_marks_are_pandas_marks(
    firepanda: ModuleType, monkeypatch: Any, capsys: Any
) -> None:
    code, cmd, out = run(firepanda, monkeypatch, capsys)
    assert code == 3
    assert cmd[0] == "-m not slow and not network and not db"
    assert os.path.isdir(cmd[1])
    assert out == f"running: pytest {' '.join(cmd)}\n"


def test_extra_args_replace_the_marks(firepanda: ModuleType, monkeypatch: Any, capsys: Any) -> None:
    _, cmd, _ = run(firepanda, monkeypatch, capsys, extra_args=["-x", "-q"])
    assert cmd[:2] == ["-x", "-q"] and len(cmd) == 3
    _, cmd, _ = run(firepanda, monkeypatch, capsys, extra_args="-x")
    assert cmd[0] == "-x" and len(cmd) == 2


def test_doctests_run_over_the_package(
    firepanda: ModuleType, monkeypatch: Any, capsys: Any
) -> None:
    _, cmd, _ = run(firepanda, monkeypatch, capsys, run_doctests=True)
    assert cmd == ["--doctest-modules", os.path.dirname(firepanda.__file__)]


@needs_pandas
def test_the_signature_is_pandas(firepanda: ModuleType) -> None:
    import pandas

    assert str(inspect.signature(firepanda.test)) == str(inspect.signature(pandas.test))
