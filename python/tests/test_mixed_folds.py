"""Reductions down the columns of a frame that mixes flags with numbers or text.

pandas adds up flags as whole numbers, so a sum or product over flags beside
integers stays int64. The largest or smallest value of columns whose answers
share no type is kept as it is in an object answer, and a row of flags beside
numbers is reduced in Python's own arithmetic into an object answer. Each test
here runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def mixed(lib: Any) -> Any:
    return lib.DataFrame({"a": [1, 2], "b": [True, False]})


def floats(lib: Any) -> Any:
    return lib.DataFrame({"a": [1, 2], "b": [True, False], "c": [1.5, 2.5]})


def gapped(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.0, None, 3.0], "b": [True, False, True]})


BUILDS = {
    "sum": lambda lib: mixed(lib).sum(),
    "prod": lambda lib: mixed(lib).prod(),
    "max": lambda lib: mixed(lib).max(),
    "min": lambda lib: mixed(lib).min(),
    "flags only": lambda lib: lib.DataFrame({"a": [True, True], "b": [True, False]}).sum(),
    "flags and floats": lambda lib: lib.DataFrame({"a": [True, False], "b": [1.5, 2.0]}).max(),
    "a gap": lambda lib: lib.DataFrame({"a": [1, None], "b": [True, False]}).max(),
    "text": lambda lib: lib.DataFrame({"a": [1, 2], "b": ["x", "y"]}).max(),
    "min_count": lambda lib: mixed(lib).sum(min_count=3),
    "row sum": lambda lib: mixed(lib).sum(axis=1),
    "row sum floats": lambda lib: floats(lib).sum(axis=1),
    "row prod": lambda lib: mixed(lib).prod(axis=1),
    "row max": lambda lib: mixed(lib).max(axis=1),
    "row min": lambda lib: floats(lib).min(axis=1),
    "row mean": lambda lib: mixed(lib).mean(axis=1),
    "row gap sum": lambda lib: gapped(lib).sum(axis=1),
    "row gap kept": lambda lib: gapped(lib).sum(axis=1, skipna=False),
    "row gap max": lambda lib: gapped(lib).max(axis=1),
    "row gap mean": lambda lib: gapped(lib).mean(axis=1),
    "row min_count": lambda lib: gapped(lib).sum(axis=1, min_count=2),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_mixed_column_folds_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
