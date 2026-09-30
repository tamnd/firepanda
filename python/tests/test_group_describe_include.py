"""A group by's `describe` with `include` and `exclude`, as pandas picks and lays out the columns.

The columns are picked the way `DataFrame.describe` picks them, and when the
picked columns answer different statistics each takes them all, the ones it has
no answer for missing. Each test runs the same code on both libraries and
compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "k": ["a", "b", "a", "b"],
            "v": [1.0, 2.0, 3.0, 4.0],
            "s": ["x", "y", "x", "z"],
            "i": [1, 2, 3, 4],
        }
    )


BUILDS = {
    "all": lambda lib: frame(lib).groupby("k").describe(include="all"),
    "object": lambda lib: frame(lib).groupby("k").describe(include="object"),
    "str": lambda lib: frame(lib).groupby("k").describe(include=["str"]),
    "exclude number": lambda lib: frame(lib).groupby("k").describe(exclude="number"),
    "include number": lambda lib: frame(lib).groupby("k").describe(include="number"),
    "exclude float": lambda lib: frame(lib).groupby("k").describe(exclude=["float"]),
    "percentiles": lambda lib: frame(lib).groupby("k").describe([0.5], include="number"),
    "text column": lambda lib: frame(lib).groupby("k")["s"].describe(),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_group_describe_include_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


@pytest.mark.parametrize(
    "build",
    [
        lambda lib: frame(lib).groupby("k").describe(include="all", exclude="number"),
        lambda lib: frame(lib).groupby("k")[["v"]].describe(include="object"),
    ],
)
def test_a_pick_pandas_refuses_is_refused(firepanda: Any, build: Any) -> None:
    with pytest.raises(ValueError) as theirs:
        build(pd)
    with pytest.raises(ValueError) as mine:
        build(firepanda)
    assert str(mine.value) == str(theirs.value)
