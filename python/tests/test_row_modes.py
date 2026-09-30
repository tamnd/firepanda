"""`DataFrame.mode` along the rows, and the pieces it leans on.

pandas answers `mode(axis=1)` with the most common values of each row, a
column per place, NaN where a row has fewer. Text stays text through the turn
on its side. A column of flags that can be missing picks rows by its true
flags, a missing one counting as false. Each test here runs the same code on
both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def numbers(lib: Any) -> Any:
    return lib.DataFrame({"a": [1, 2, 3], "b": [1, 3, 4], "c": [2, 3, 5]}, index=["x", "y", "z"])


def gaps(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.5, None, 2.0], "b": [1.5, None, 3.0], "c": [None, None, 3.0]})


def mask(lib: Any) -> Any:
    return lib.Series([True, None, False], index=["a", "b", "c"], dtype="boolean")


BUILDS = {
    "numbers": lambda lib: numbers(lib).mode(axis=1),
    "columns word": lambda lib: numbers(lib).mode(axis="columns"),
    "gaps": lambda lib: gaps(lib).mode(axis=1),
    "gaps kept": lambda lib: gaps(lib).mode(axis=1, dropna=False),
    "numeric only": lambda lib: lib.DataFrame({"a": [1, 2], "b": ["p", "q"], "c": [1.0, 2.0]}).mode(
        axis=1, numeric_only=True
    ),
    "text": lambda lib: lib.DataFrame({"a": ["p", "q"], "b": ["p", "r"]}).mode(axis=1),
    "text turned": lambda lib: [
        str(kind) for kind in lib.DataFrame({"a": ["p", "q"], "b": ["p", "r"]}).T.dtypes
    ],
    "string mode": lambda lib: lib.Series(["p", "q", "q"], dtype="string").mode(),
    "mask": lambda lib: lib.Series([1, 2, 3], index=["a", "b", "c"])[mask(lib)],
    "mask loc": lambda lib: lib.Series([1, 2, 3], index=["a", "b", "c"]).loc[mask(lib)],
    "frame mask": lambda lib: lib.DataFrame({"v": [1, 2, 3]}, index=["a", "b", "c"])[mask(lib)],
    "filled mask": lambda lib: mask(lib).fillna(False),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_row_modes_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
