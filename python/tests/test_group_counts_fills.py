"""Counting from the end, counting gaps and filling a shift within groups.

pandas numbers each group's rows or the groups themselves from the end under
`ascending=False`, counts a gap as one more distinct value under
`nunique(dropna=False)`, and fills the rows a group shift opens with
`fill_value`, rows with a missing key included, while `freq` shifts each
group's labels, keys and all, one group after another. Each case runs the same
code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def holed(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "k": ["a", "b", "a", None, "a", "b"],
            "v": [1.0, None, 3.0, 4.0, None, None],
            "s": ["x", None, "x", "y", "z", None],
            "i": [1, 2, 3, 4, 5, 6],
        }
    )


def dated(lib: Any) -> Any:
    return holed(lib).set_index(lib.date_range("2024-01-01", periods=6))


BUILDS = {
    "cumcount from the end": lambda lib: holed(lib).groupby("k").cumcount(ascending=False),
    "ngroup from the end": lambda lib: holed(lib).groupby("k").ngroup(ascending=False),
    "ngroup missing keys kept": lambda lib: (
        holed(lib).groupby("k", dropna=False).ngroup(ascending=False)
    ),
    "cumcount missing keys kept": lambda lib: (
        holed(lib).groupby("k", dropna=False).cumcount(ascending=False)
    ),
    "one column cumcount": lambda lib: holed(lib).groupby("k")["v"].cumcount(ascending=False),
    "nunique with gaps": lambda lib: holed(lib).groupby("k").nunique(dropna=False),
    "one column nunique": lambda lib: holed(lib).groupby("k")["s"].nunique(dropna=False),
    "nunique keys as columns": lambda lib: (
        holed(lib).groupby("k", as_index=False).nunique(dropna=False)
    ),
    "nunique selected": lambda lib: holed(lib).groupby("k")[["v"]].nunique(dropna=False),
    "nunique missing keys kept": lambda lib: (
        holed(lib).groupby("k", dropna=False).nunique(dropna=False)
    ),
    "shift fill text": lambda lib: holed(lib).groupby("k")["s"].shift(-1, fill_value="q"),
    "shift fill whole": lambda lib: holed(lib).groupby("k")["i"].shift(1, fill_value=0),
    "shift fill frame": lambda lib: holed(lib)[["k", "v", "i"]].groupby("k").shift(2, fill_value=0),
    "shift fill list": lambda lib: holed(lib).groupby("k")[["i"]].shift([1, 2], fill_value=-1),
    "shift freq": lambda lib: dated(lib).groupby("k").shift(1, freq="D"),
    "shift freq one column": lambda lib: dated(lib).groupby("k")["v"].shift(2, freq="h"),
    "series fill all rows": lambda lib: lib.Series([1.0]).shift(1, fill_value=0),
    "series fill narrow floats": lambda lib: lib.Series([1.5, 2.5], dtype="float32").shift(
        1, fill_value=0
    ),
    "series fill narrow whole": lambda lib: lib.Series([1, 2], dtype="int32").shift(
        5, fill_value=3
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_group_counts_and_fills_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_number_cannot_fill_text(firepanda: Any) -> None:
    for lib in (pd, firepanda):
        with pytest.raises(TypeError, match="Invalid value '0' for dtype 'str'"):
            holed(lib).groupby("k").shift(1, fill_value=0)
