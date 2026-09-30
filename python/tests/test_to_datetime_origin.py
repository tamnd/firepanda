"""`to_datetime(origin=...)`, which counts numbers from somewhere other than 1970.

pandas takes `julian` for Julian days, or any instant with no zone, and moves
the numbers to counts from 1970 before reading them, so a list still answers
an index, a column a column and a number an instant. Each test runs the same
code on both libraries and compares the answer or the error.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

CALLS = {
    "instant": lambda lib: lib.to_datetime([1, 2, 3], unit="D", origin=lib.Timestamp("1960-01-01")),
    "text": lambda lib: lib.to_datetime([1, 2, 3], unit="D", origin="1960-01-01"),
    "column": lambda lib: lib.to_datetime(lib.Series([1, 2]), unit="s", origin="2000-01-01"),
    "fractions": lambda lib: lib.to_datetime(
        lib.Series([1.5, None]), unit="h", origin="2000-01-01"
    ),
    "number": lambda lib: lib.to_datetime(5, unit="D", origin="2000-01-01"),
    "nanoseconds": lambda lib: lib.to_datetime([10], origin="2000-01-01"),
    "index": lambda lib: lib.to_datetime(
        lib.Index([1, 2]), unit="ms", origin="2000-01-01 00:00:00.5"
    ),
    "julian column": lambda lib: lib.to_datetime(
        lib.Series([2451544.5, 2451545.0]), unit="D", origin="julian"
    ),
    "julian list": lambda lib: lib.to_datetime([2451544.5], unit="D", origin="julian"),
    "julian number": lambda lib: lib.to_datetime(2451544.5, unit="D", origin="julian"),
    "unix": lambda lib: lib.to_datetime([1], unit="D", origin="unix"),
}

MISTAKES = {
    "julian seconds": lambda lib: lib.to_datetime(
        lib.Series([2451544.5]), unit="s", origin="julian"
    ),
    "julian too far": lambda lib: lib.to_datetime([1e12], unit="D", origin="julian"),
    "text values": lambda lib: lib.to_datetime(["2020-01-01"], origin="2000-01-01"),
    "not an instant": lambda lib: lib.to_datetime([1], unit="D", origin="nonsense"),
    "zoned": lambda lib: lib.to_datetime(
        [1], unit="D", origin=lib.Timestamp("2000-01-01", tz="UTC")
    ),
}


@pytest.mark.parametrize("make", CALLS.values(), ids=CALLS.keys())
def test_origin_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


@pytest.mark.parametrize("make", MISTAKES.values(), ids=MISTAKES.keys())
def test_what_pandas_refuses_is_refused(firepanda: Any, make: Any) -> None:
    with pytest.raises(ValueError) as theirs:
        make(pd)
    with pytest.raises(ValueError) as mine:
        make(firepanda)
    assert str(mine.value) == str(theirs.value)
    assert type(theirs.value).__name__ in {c.__name__ for c in type(mine.value).__mro__}
