"""`align`, `combine_first` and `combine` where one side repeats a row label.

pandas lines up two sets of row labels that differ, where one side repeats a
label, by joining them the way `Index.join` does, so every row of a label on
one side meets every row of it on the other. A frame's `combine_first` and
`combine` align that way first. A column's `combine_first` reindexes each side
instead, which pandas refuses when a label repeats, and so does a frame whose
rows move to meet a column. Each test runs the same
code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def twice(lib: Any) -> Any:
    return lib.Series([1.0, float("nan")], index=["a", "a"])


def once(lib: Any) -> Any:
    return lib.Series([5.0, 6.0], index=["a", "b"])


def doubled(lib: Any) -> Any:
    return lib.DataFrame({"v": [1.0, float("nan")]}, index=["a", "a"])


def single(lib: Any) -> Any:
    return lib.DataFrame({"v": [5.0, 6.0], "w": [1, 2]}, index=["b", "a"])


BUILDS = {
    "align outer": lambda lib: once(lib).align(twice(lib)),
    "align inner": lambda lib: twice(lib).align(once(lib), join="inner"),
    "align left": lambda lib: twice(lib).align(once(lib), join="left"),
    "align right": lambda lib: twice(lib).align(once(lib), join="right"),
    "align both repeat": lambda lib: twice(lib).align(lib.Series([7, 8, 9], index=["b", "a", "a"])),
    "align filled": lambda lib: twice(lib).align(once(lib), fill_value=0.0),
    "align frames": lambda lib: doubled(lib).align(single(lib)),
    "align frame rows": lambda lib: doubled(lib).align(single(lib), axis=0, join="inner"),
    "align frame and repeated column": lambda lib: single(lib).align(twice(lib), axis=0),
    "align column and repeated frame": lambda lib: once(lib).align(doubled(lib)),
    "align repeated frame inner": lambda lib: doubled(lib).align(once(lib), axis=0, join="inner"),
    "combine_first frames": lambda lib: doubled(lib).combine_first(single(lib)),
    "combine_first flipped": lambda lib: single(lib).combine_first(doubled(lib)),
    "combine_first whole numbers": lambda lib: lib.DataFrame(
        {"v": [1, 2]}, index=["a", "a"]
    ).combine_first(single(lib)),
    "combine_first both repeat": lambda lib: doubled(lib).combine_first(
        lib.DataFrame({"v": [7.0, 8.0, 9.0]}, index=["a", "a", "c"])
    ),
    "combine": lambda lib: doubled(lib).combine(
        single(lib), lambda x, y: x.fillna(0) + y.fillna(0)
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_repeated_labels_are_joined_as_pandas_joins_them(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


REFUSED = {
    "frame moved to a column": lambda lib: doubled(lib).align(once(lib), axis=0),
    "this repeats": lambda lib: twice(lib).combine_first(once(lib)),
    "other repeats": lambda lib: once(lib).combine_first(twice(lib)),
    "same labels other type": lambda lib: lib.Series([1, 2], index=["a", "a"]).combine_first(
        twice(lib)
    ),
}


@pytest.mark.parametrize("make", REFUSED.values(), ids=REFUSED.keys())
def test_a_repeated_label_reindexed_is_refused(firepanda: Any, make: Any) -> None:
    with pytest.raises(ValueError, match="cannot reindex on an axis with duplicate labels"):
        make(pd)
    with pytest.raises(ValueError, match="cannot reindex on an axis with duplicate labels"):
        make(firepanda)


def test_a_column_on_the_same_labels_fills_in_place(firepanda: Any) -> None:
    other = [3.0, 4.0]
    got = twice(firepanda).combine_first(firepanda.Series(other, index=["a", "a"]))
    assert repr(got) == repr(twice(pd).combine_first(pd.Series(other, index=["a", "a"])))
