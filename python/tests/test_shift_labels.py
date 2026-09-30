"""Shifting by a frequency on the column axis and on an index of periods.

pandas moves the labels rather than the values when `shift` is given a
frequency. It does so across the columns under `axis=1`, it accepts an offset
that names a period index's own frequency, and it refuses any other frequency
on periods and any index that is not one of instants, spans or periods.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def months(lib: Any) -> Any:
    return lib.Series([1, 2, 3], index=lib.period_range("2024-01", periods=3, freq="M"))


def across(lib: Any) -> Any:
    return lib.DataFrame([[1, 2]], columns=lib.date_range("2024-01-01", periods=2))


BUILDS = {
    "months by count": lambda lib: months(lib).shift(-1, freq="M"),
    "months by offset": lambda lib: months(lib).shift(2, freq=lib.offsets.MonthEnd(1)),
    "months frame": lambda lib: months(lib).to_frame("v").shift(1, freq="M"),
    "columns": lambda lib: across(lib).shift(1, freq="D", axis=1),
    "columns named": lambda lib: across(lib).shift(-2, freq="h", axis="columns"),
    "hours offset": lambda lib: lib.Series(
        [1, 2], index=lib.date_range("2024-01-01", periods=2)
    ).shift(2, freq=lib.offsets.Hour(3)),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_shift_labels_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


REFUSED = {
    "months two": (lambda lib: months(lib).shift(1, freq="2M"), ValueError, "does not match"),
    "months days": (lambda lib: months(lib).shift(1, freq="D"), ValueError, "freq D does not"),
    "months offset": (
        lambda lib: months(lib).shift(1, freq=lib.offsets.MonthEnd(2)),
        ValueError,
        "freq 2M does not",
    ),
    "numbered rows": (
        lambda lib: lib.Series([1, 2]).shift(1, freq="D"),
        NotImplementedError,
        "Got type RangeIndex",
    ),
    "named columns": (
        lambda lib: lib.DataFrame({"a": [1]}).shift(1, freq="D", axis=1),
        NotImplementedError,
        "Got type Index",
    ),
    "series columns": (
        lambda lib: months(lib).shift(1, freq="M", axis=1),
        ValueError,
        "No axis named 1",
    ),
}


@pytest.mark.parametrize(("make", "error", "words"), REFUSED.values(), ids=REFUSED.keys())
def test_shift_labels_refuses_as_pandas(firepanda: Any, make: Any, error: Any, words: str) -> None:
    with pytest.raises(error, match=words):
        make(pd)
    with pytest.raises(error, match=words):
        make(firepanda)
