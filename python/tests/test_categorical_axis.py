"""A `CategoricalIndex` kept as the labels wherever it is handed over as an index.

pandas keeps the categories, their order and the name when a categorical index
is given to the constructors as `index=`, set with `set_axis` or the `index`
setter, or reindexed onto. A reindex of numbers by text finds nothing and
leaves every row missing rather than refusing. Each test runs the same code on
both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def cats(lib: Any) -> Any:
    return lib.CategoricalIndex(["a", "b"], categories=["b", "a", "c"], ordered=True, name="n")


def assigned(lib: Any) -> Any:
    made = lib.Series([1, 2])
    made.index = cats(lib)
    return made


def framed(lib: Any) -> Any:
    made = lib.DataFrame({"v": [1, 2]})
    made.index = cats(lib)
    return made


BUILDS = {
    "series index": lambda lib: lib.Series([1, 2], index=cats(lib)),
    "frame index": lambda lib: lib.DataFrame({"v": [1, 2]}, index=cats(lib)),
    "frame rows": lambda lib: lib.DataFrame([[1, 2], [3, 4]], index=cats(lib)),
    "setter": assigned,
    "frame setter": framed,
    "set_axis": lambda lib: lib.Series([1, 2]).set_axis(cats(lib)),
    "frame set_axis": lambda lib: lib.DataFrame({"v": [1, 2]}).set_axis(cats(lib)),
    "reindex": lambda lib: lib.Series([1, 2], index=["a", "q"]).reindex(cats(lib)),
    "frame reindex": lambda lib: lib.DataFrame({"v": [1, 2]}, index=["a", "q"]).reindex(cats(lib)),
    "numbers by categories": lambda lib: lib.Series([1, 2]).reindex(cats(lib)),
    "numbers by text": lambda lib: lib.Series([1, 2]).reindex(["a", 1]),
    "frame numbers by text": lambda lib: lib.DataFrame({"v": [1, 2]}).reindex(["a", "b"]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_categorical_axis_is_pandas(firepanda: Any, make: Any) -> None:
    mine, theirs = make(firepanda), make(pd)
    assert repr(mine) == repr(theirs)
    assert type(mine.index).__name__ == type(theirs.index).__name__
    assert repr(mine.index) == repr(theirs.index)
