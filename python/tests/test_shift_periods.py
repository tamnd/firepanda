"""`shift` by a list of periods, which pandas answers with a shift by each side by side.

Each shift names its columns after the column and the period, with `suffix`
between them when one is given, and a series answers a frame of its one column
shifted by each. A group by shifts within each group the same way. Each test
here runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame({"k": ["a", "b", "a", "b"], "v": [1, 2, 3, 4], "w": [1.5, 2.5, 3.5, 4.5]})


BUILDS = {
    "frame": lambda lib: frame(lib)[["v", "w"]].shift([0, 1, -1]),
    "suffix": lambda lib: frame(lib)[["v"]].shift([1, 2], suffix="_lag"),
    "fill": lambda lib: frame(lib)[["v"]].shift([1, 2], fill_value=0),
    "range": lambda lib: frame(lib)[["v"]].shift(range(1, 3)),
    "series": lambda lib: lib.Series([1, 2, 3], name="s").shift([1, 2]),
    "unnamed": lambda lib: lib.Series([1, 2, 3]).shift([1]),
    "series suffix left out": lambda lib: lib.Series([1, 2, 3], name="s").shift([1], suffix="x"),
    "series int suffix": lambda lib: lib.Series([1, 2, 3], name="s").shift(1, suffix="x"),
    "groups": lambda lib: frame(lib).groupby("k").shift([1, 2]),
    "group column": lambda lib: frame(lib).groupby("k")["v"].shift([0, 1]),
    "group suffix": lambda lib: frame(lib).groupby("k")["v"].shift([1], suffix="_p"),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_shifts_by_each_period_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


MISTAKES = {
    "no periods": lambda lib: frame(lib).shift([]),
    "across rows": lambda lib: frame(lib).shift([1], axis=1),
    "a fraction": lambda lib: frame(lib).shift([1.5]),
    "suffix on one": lambda lib: frame(lib).shift(1, suffix="x"),
    "no group periods": lambda lib: frame(lib).groupby("k").shift([]),
    "group suffix on one": lambda lib: frame(lib).groupby("k").shift(1, suffix="x"),
}


@pytest.mark.parametrize("make", MISTAKES.values(), ids=MISTAKES.keys())
def test_mistakes_are_refused_in_pandas_words(firepanda: Any, make: Any) -> None:
    with pytest.raises(Exception) as theirs:
        make(pd)
    with pytest.raises(type(theirs.value), match=str(theirs.value).replace("`", ".")[:30]):
        make(firepanda)
