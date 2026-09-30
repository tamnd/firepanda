"""Running totals, products, least and greatest values across the rows of a frame.

pandas runs a scan along `axis=1` as the same scan down the columns of the
frame turned on its side, and turns the answer back. Columns of one kind keep
it; a mix of flags and numbers turns into objects scanned in Python's own
arithmetic, where a gap is passed over as the value that leaves the scan
where it was. Each test here runs the same code on both libraries and
compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def whole(lib: Any) -> Any:
    return lib.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}, index=["x", "y"])


def holed(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.0, None], "b": [3.0, 4.0], "c": [None, 6.0]})


def flagged(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.0, None], "b": [True, False], "c": [2, 3]})


BUILDS = {
    "cumsum": lambda lib: whole(lib).cumsum(axis=1),
    "types kept": lambda lib: whole(lib).cumsum(axis=1).dtypes,
    "cumprod": lambda lib: whole(lib).cumprod(axis=1),
    "cummax by name": lambda lib: whole(lib).cummax(axis="columns"),
    "cummin": lambda lib: whole(lib).cummin(axis=1),
    "gaps": lambda lib: holed(lib).cumsum(axis=1),
    "gaps kept": lambda lib: holed(lib).cumsum(axis=1, skipna=False),
    "gaps cummax": lambda lib: holed(lib).cummax(axis=1),
    "whole and floats": lambda lib: lib.DataFrame({"a": [1, 2], "b": [1.5, 2.5]}).cumsum(axis=1),
    "flags": lambda lib: lib.DataFrame({"a": [1, 2], "b": [True, False]}).cumsum(axis=1),
    "flags types": lambda lib: (
        lib.DataFrame({"a": [1, 2], "b": [True, False]}).cumsum(axis=1).dtypes
    ),
    "flags and gaps": lambda lib: flagged(lib).cumsum(axis=1),
    "flags gaps kept": lambda lib: flagged(lib).cumsum(axis=1, skipna=False),
    "flags cumprod": lambda lib: flagged(lib).cumprod(axis=1),
    "flags cummax": lambda lib: flagged(lib).cummax(axis=1),
    "objects": lambda lib: lib.Series([1, None, True, 5], dtype=object).cumsum(),
    "objects cummin": lambda lib: lib.Series([1, None, True, 5], dtype=object).cummin(),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_scans_across_rows_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
