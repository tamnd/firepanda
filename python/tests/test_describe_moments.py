"""`describe` of moments and spans, and of the columns `include` and `exclude` pick.

pandas describes a column of moments by its count, mean, extremes and
percentiles in an object column, NaT where there are no values, and a column of
spans with its spread as well. A frame describes its numbers, spans and moments
without a zone by default, and picks columns as `select_dtypes` does when handed
types. Each test here runs the same code on both libraries and compares.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def moments(lib: Any) -> Any:
    days = ["2024-01-01", "2024-01-03", None, "2024-01-02"]
    return lib.Series(lib.to_datetime(days), name="d")


def spans(lib: Any) -> Any:
    return lib.Series(lib.to_timedelta(["1D", "2D", "4D"]), name="t")


def frame(lib: Any) -> Any:
    days = lib.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"])
    return lib.DataFrame({"n": [1, 2, 3], "s": ["a", "b", "a"], "d": days})


BUILDS = {
    "moments": lambda lib: moments(lib).describe(),
    "percentiles": lambda lib: moments(lib).describe(percentiles=[0.1]),
    "spans": lambda lib: spans(lib).describe(),
    "zoned": lambda lib: moments(lib).dt.tz_localize("UTC").describe(),
    "empty": lambda lib: moments(lib).iloc[:0].describe(),
    "all gaps": lambda lib: moments(lib).iloc[2:3].describe(),
    "frame": lambda lib: frame(lib).describe(),
    "frame all": lambda lib: frame(lib).describe(include="all"),
    "frame moments": lambda lib: frame(lib)[["d"]].describe(),
    "frame spans": lambda lib: lib.DataFrame({"t": spans(lib), "n": [1, 2, 3]}).describe(),
    "frame zoned": lambda lib: lib.DataFrame(
        {"z": moments(lib).dt.tz_localize("UTC"), "n": [1, 2, 3, 4]}
    ).describe(),
    "include": lambda lib: frame(lib).describe(include="object"),
    "include list": lambda lib: frame(lib).describe(include=["number", "object"]),
    "exclude": lambda lib: frame(lib).describe(exclude="number"),
    "exclude list": lambda lib: frame(lib).describe(exclude=["object"]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_describe_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_types_that_pick_nothing_are_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="No objects to concatenate"):
        frame(firepanda).describe(include="bool")
