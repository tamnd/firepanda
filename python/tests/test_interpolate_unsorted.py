"""`interpolate` by the labels when the labels do not rise.

pandas sorts the points of the values it has and fills each gap by numpy's
`interp` along them, and it still counts `limit`, `limit_direction` and
`limit_area` by position. Each test runs the same code on both libraries and
compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

NAN = float("nan")


def shuffled(lib: Any) -> Any:
    return lib.Series([1.0, NAN, 3.0, NAN, 10.0], index=[0, 4, 1, 3, 2])


BUILDS = {
    "index": lambda lib: shuffled(lib).interpolate(method="index"),
    "values": lambda lib: shuffled(lib).interpolate(method="values"),
    "limit": lambda lib: shuffled(lib).interpolate(method="index", limit=1),
    "backward": lambda lib: shuffled(lib).interpolate(method="index", limit_direction="backward"),
    "both": lambda lib: shuffled(lib).interpolate(method="index", limit_direction="both"),
    "inside": lambda lib: shuffled(lib).interpolate(method="index", limit_area="inside"),
    "outside": lambda lib: shuffled(lib).interpolate(method="index", limit_area="outside"),
    "repeated": lambda lib: lib.Series([1.0, NAN, 5.0, NAN], index=[3, 1, 1, 2]).interpolate(
        method="index"
    ),
    "falling": lambda lib: lib.Series(
        [NAN, 1.0, NAN, 4.0, NAN], index=[5.0, 4.0, 3.0, 1.0, 0.5]
    ).interpolate(method="index", limit_direction="both"),
    "time": lambda lib: lib.Series(
        [1.0, NAN, 3.0], index=lib.to_datetime(["2024-01-03", "2024-01-02", "2024-01-01"])
    ).interpolate(method="time"),
    "frame": lambda lib: lib.DataFrame(
        {"a": [1.0, NAN, 3.0], "b": [NAN, 2.0, 6.0]}, index=[2, 0, 1]
    ).interpolate(method="index", limit_direction="both"),
    "infinity": lambda lib: lib.Series([1.0, NAN, float("inf")], index=[2, 1, 0]).interpolate(
        method="index"
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_interpolate_unsorted_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
