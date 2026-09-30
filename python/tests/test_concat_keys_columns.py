"""`concat(keys=, axis=1)`, which labels the columns with each part's key.

Series take their keys as their column names. Frames put each key in front
of every one of their columns' labels, as a MultiIndex over the columns, and a
series beside frames is a frame of one column. `names` names the levels from
the front, and a key that is a tuple is one level a value. `transform` with a
list of functions answers through the same door. Each test here runs the same
code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def first(lib: Any) -> Any:
    return lib.DataFrame({"x": [1, 2], "y": [3.5, 4.0]})


def second(lib: Any) -> Any:
    return lib.DataFrame({"x": [5, 6], "z": ["p", "q"]}, index=[1, 2])


def column(lib: Any, index: Any = None) -> Any:
    return lib.Series([1, 2], name="s", index=index)


def both(lib: Any, **options: Any) -> Any:
    return lib.concat([first(lib), second(lib)], axis=1, keys=["A", "B"], **options)


BUILDS = {
    "frames": lambda lib: both(lib),
    "inner": lambda lib: both(lib, join="inner"),
    "named": lambda lib: both(lib, names=["part"]),
    "named twice": lambda lib: both(lib, names=["part", "col"]),
    "columns": lambda lib: both(lib).columns,
    "picked": lambda lib: both(lib)["A"],
    "sorted": lambda lib: lib.concat([second(lib), first(lib)], axis=1, keys=["B", "A"], sort=True),
    "series": lambda lib: lib.concat(
        [column(lib), column(lib, [1, 2])], axis=1, keys=["one", "two"]
    ),
    "series named": lambda lib: lib.concat(
        [column(lib), column(lib)], axis=1, keys=["one", "two"], names=["k"]
    ),
    "series numbers": lambda lib: lib.concat(
        [column(lib), column(lib)], axis="columns", keys=[1, 2]
    ),
    "tuples": lambda lib: lib.concat([first(lib), first(lib)], axis=1, keys=[("p", 1), ("q", 2)]),
    "mapping": lambda lib: lib.concat({"A": first(lib), "B": first(lib)}, axis=1),
    "series mapping": lambda lib: lib.concat({"u": column(lib), "v": column(lib)}, axis=1),
    "beside": lambda lib: lib.concat([first(lib), column(lib)], axis=1, keys=["A", "B"]),
    "unnamed beside": lambda lib: lib.concat(
        [first(lib), lib.Series([7, 8]), lib.Series([9, 9])], axis=1, keys=["A", "B", "C"]
    ),
    "series first": lambda lib: lib.concat([column(lib), first(lib)], axis=1, keys=["S", "A"]),
    "levels again": lambda lib: lib.concat(
        [
            lib.concat([first(lib)], axis=1, keys=["L"]),
            lib.concat([first(lib)], axis=1, keys=["M"]),
        ],
        axis=1,
        keys=["top", "bottom"],
    ),
    "transform list": lambda lib: first(lib).transform(["sqrt", "exp"]).round(2),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_keys_across_the_columns_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
