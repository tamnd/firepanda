"""`melt` on columns of several levels, and `col_level` to melt just one of them.

pandas gives each level of columns a column of its own in the long frame,
named for the level or numbered when the names do not tell them apart, and
`col_level` melts one level as if it were the only one. Each test runs the
same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def named(lib: Any) -> Any:
    pairs = [("A", "a"), ("B", "b"), ("C", "c")]
    columns = lib.MultiIndex.from_tuples(pairs, names=["up", "low"])
    return lib.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns)


def unnamed(lib: Any) -> Any:
    columns = lib.MultiIndex.from_tuples([("A", "a"), ("B", "b")])
    return lib.DataFrame([[1, 2], [4, 5]], columns=columns)


BUILDS = {
    "col_level 0": lambda lib: named(lib).melt(col_level=0, id_vars=["A"], value_vars=["B"]),
    "col_level 1": lambda lib: named(lib).melt(col_level=1, id_vars=["a"]),
    "col_level name": lambda lib: named(lib).melt(col_level="low", value_vars=["b", "c"]),
    "col_level labels": lambda lib: lib.melt(
        named(lib), col_level=0, id_vars="A", ignore_index=False
    ),
    "col_level flat": lambda lib: lib.DataFrame({"a": [1], "b": [2]}).melt(col_level=0),
    "col_level flat name": lambda lib: (
        lib.DataFrame({"a": [1], "b": [2]}).rename_axis(columns="n").melt(col_level="n")
    ),
    "levels": lambda lib: named(lib).melt(),
    "levels id": lambda lib: named(lib).melt(id_vars=[("A", "a")]),
    "levels numbered": lambda lib: unnamed(lib).melt(),
    "levels var_name": lambda lib: unnamed(lib).melt(var_name=["p", "q"]),
    "levels var_name short": lambda lib: unnamed(lib).melt(var_name=["p"]),
    "levels labels": lambda lib: unnamed(lib).melt(ignore_index=False, value_name="v"),
    "levels none": lambda lib: unnamed(lib).melt(id_vars=[("A", "a")], value_vars=[]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_melt_by_level_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


@pytest.mark.parametrize(
    "build",
    [
        lambda lib: lib.DataFrame({"a": [1]}).melt(col_level="zz"),
        lambda lib: named(lib).melt(col_level=2),
    ],
)
def test_a_level_that_is_not_there_is_pandas_mistake(firepanda: Any, build: Any) -> None:
    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(type(theirs.value)) as mine:
        build(firepanda)
    assert str(mine.value) == str(theirs.value)
