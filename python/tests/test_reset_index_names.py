"""`DataFrame.reset_index(names=...)` on row labels of one level.

pandas names the column the old labels land in from `names`, a name or the
first name of a list, and an empty list keeps the labels' own name. Each test
runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def keyed(lib: Any) -> Any:
    return lib.DataFrame({"v": [1, 2]}, index=lib.Index([10, 20], name="k"))


BUILDS = {
    "name": lambda lib: keyed(lib).reset_index(names="x"),
    "list": lambda lib: keyed(lib).reset_index(names=["x"]),
    "longer list": lambda lib: keyed(lib).reset_index(names=["x", "y"]),
    "empty list": lambda lib: keyed(lib).reset_index(names=[]),
    "dropped": lambda lib: keyed(lib).reset_index(names="x", drop=True),
    "number": lambda lib: keyed(lib).reset_index(names=5),
    "unnamed": lambda lib: lib.DataFrame({"v": [1]}).reset_index(names="i"),
    "col_level": lambda lib: keyed(lib).reset_index(names="x", col_level=0),
    "categories": lambda lib: lib.DataFrame(
        {"v": [1, 2]}, index=lib.CategoricalIndex(["a", "b"])
    ).reset_index(names="c"),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_named_reset_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_in_place_names_the_column(firepanda: Any) -> None:
    mine, theirs = keyed(firepanda), keyed(pd)
    assert mine.reset_index(names="x", inplace=True) is None
    theirs.reset_index(names="x", inplace=True)
    assert repr(mine) == repr(theirs)


@pytest.mark.parametrize(
    "build",
    [
        lambda lib: keyed(lib).reset_index(names=("x",)),
        lambda lib: keyed(lib).reset_index(names="v"),
    ],
)
def test_what_pandas_refuses_is_refused(firepanda: Any, build: Any) -> None:
    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(type(theirs.value)) as mine:
        build(firepanda)
    assert str(mine.value) == str(theirs.value)
