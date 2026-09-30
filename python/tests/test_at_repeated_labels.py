"""`at` on a label that is there more than once.

pandas answers such a label as `loc` does, with every value under it as a
series named after the column, and a label there once as the value itself.
Each test here runs the same code on both libraries and compares what they
print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame({"v": [1, 2, 3], "w": ["a", "b", "c"]}, index=["x", "y", "x"])


def series(lib: Any) -> Any:
    return lib.Series([1.5, 2.5, 3.5], index=["x", "y", "x"], name="s")


BUILDS = {
    "frame": lambda lib: frame(lib).at["x", "v"],
    "frame text": lambda lib: frame(lib).at["x", "w"],
    "frame once": lambda lib: int(frame(lib).at["y", "v"]),
    "series": lambda lib: series(lib).at["x"],
    "series once": lambda lib: float(series(lib).at["y"]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_a_repeated_label_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_missing_label_is_refused(firepanda: Any) -> None:
    with pytest.raises(KeyError):
        series(firepanda).at["z"]
