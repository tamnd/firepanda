"""Reductions down the columns of a frame that mixes flags with numbers or text.

pandas adds up flags as whole numbers, so a sum or product over flags beside
integers stays int64. The largest or smallest value of columns whose answers
share no type is kept as it is in an object answer. Each test here runs the
same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def mixed(lib: Any) -> Any:
    return lib.DataFrame({"a": [1, 2], "b": [True, False]})


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
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_mixed_column_folds_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
