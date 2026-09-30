"""A key column selected and reduced with `as_index=False`.

pandas puts each key in a column of the answer only when no column of the
answer already has its name, so a key that is also reduced as a value shows
the reduced value alone, and a key from outside the frame named after a
column is left out. Each test here runs the same code on both libraries and
compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame({"k": ["a", "b", "a"], "j": [1, 1, 2], "v": [1.0, 2.0, 3.0]})


def flat(lib: Any, by: Any = "k") -> Any:
    return frame(lib).groupby(by, as_index=False)


BUILDS = {
    "count": lambda lib: flat(lib)["k"].count(),
    "list count": lambda lib: flat(lib)[["k"]].count(),
    "list beside a value": lambda lib: flat(lib)[["k", "v"]].sum(),
    "two keys": lambda lib: flat(lib, ["k", "j"])["j"].sum(),
    "two keys list": lambda lib: flat(lib, ["k", "j"])[["j", "v"]].max(),
    "nunique": lambda lib: flat(lib)["k"].nunique(),
    "size": lambda lib: flat(lib)["k"].size(),
    "first": lambda lib: flat(lib)["k"].first(),
    "transform": lambda lib: flat(lib)["k"].transform("count"),
    "agg list": lambda lib: flat(lib)["k"].agg(["count", "max"]),
    "outside key named after a column": lambda lib: (
        frame(lib)[["k", "v"]].groupby(lib.Series(["x", "y", "x"], name="v"), as_index=False).sum()
    ),
    "outside key and a selection": lambda lib: (
        frame(lib).groupby(lib.Series(["x", "y", "x"], name="v"), as_index=False)["k"].count()
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_selected_keys_without_labels_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
