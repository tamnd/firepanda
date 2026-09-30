"""`set_index` handed labels rather than the name of a column.

pandas takes an index, a column, an array or a list as the new row labels,
read by position rather than aligned, and names the level after what was
handed in: an index or a column keeps its name, and a list has none. A lone
index becomes the row labels whole, so a date range keeps its step. Beside the
names of columns each becomes a level of a MultiIndex. Each test here runs the
same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


BUILDS = {
    "date range": lambda lib: frame(lib).set_index(lib.date_range("2024-01-01", periods=3)),
    "step kept": lambda lib: frame(lib).set_index(lib.date_range("2024-01-01", periods=3)).index,
    "named index": lambda lib: frame(lib).set_index(lib.Index(["p", "q", "r"], name="k")),
    "numbers": lambda lib: frame(lib).set_index(lib.Index([10, 20, 30])),
    "column": lambda lib: frame(lib).set_index(lib.Series([7, 8, 9], name="s")),
    "by position": lambda lib: frame(lib).set_index(
        lib.Series([7, 8, 9], index=[2, 0, 1], name="s")
    ),
    "own column": lambda lib: frame(lib).set_index(frame(lib)["b"]),
    "list": lambda lib: frame(lib).set_index([[4, 5, 6]]),
    "iterator": lambda lib: frame(lib).set_index(iter([4, 5, 6])),
    "index and column": lambda lib: frame(lib).set_index([lib.Index([1, 2, 3]), "b"]),
    "column and series": lambda lib: frame(lib).set_index(["b", lib.Series([4, 5, 6])]),
    "kept": lambda lib: frame(lib).set_index(["b", lib.Index([4, 5, 6])], drop=False),
    "append": lambda lib: frame(lib).set_index("b").set_index(lib.Index([1, 2, 3]), append=True),
    "periods": lambda lib: frame(lib).set_index(lib.period_range("2024-01", periods=3, freq="M")),
    "spans": lambda lib: frame(lib).set_index(lib.timedelta_range("1D", periods=3)),
    "resampled": lambda lib: (
        frame(lib)[["a"]].set_index(lib.date_range("2024", periods=3)).resample("2D").sum()
    ),
    "rolled": lambda lib: (
        frame(lib)[["a"]].set_index(lib.date_range("2024", periods=3)).rolling("2D").sum()
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_labels_handed_in_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_labels_of_another_length_are_refused_in_pandas_words(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="Length mismatch: Expected 3 rows, received array of"):
        frame(firepanda).set_index(firepanda.Index([1, 2]))


def test_a_range_is_looked_up_among_the_columns(firepanda: Any) -> None:
    with pytest.raises(KeyError, match="None of"):
        frame(firepanda).set_index(range(3))


def test_labels_are_set_in_place(firepanda: Any) -> None:
    mine, theirs = frame(firepanda), frame(pd)
    assert mine.set_index(firepanda.Index([4, 5, 6], name="q"), inplace=True) is None
    theirs.set_index(pd.Index([4, 5, 6], name="q"), inplace=True)
    assert repr(mine) == repr(theirs)
