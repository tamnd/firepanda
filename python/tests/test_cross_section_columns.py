"""`xs(level=)` across columns of several levels, and `df[key]` or `loc` by a first label.

pandas crosses the column labels the way it crosses row labels: the key names
a value on each given level, the columns holding it are kept, and the named
levels leave the labels unless `drop_level` is False. A label of the first
level in `df[key]` or `df.loc[rows, key]` names the columns under it, and the
names of the levels that stay are kept. Each test runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def layered(lib: Any) -> Any:
    columns = lib.MultiIndex.from_tuples(
        [("a", "x"), ("a", "y"), ("b", "x"), ("b", "z")], names=["one", "two"]
    )
    return lib.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=columns)


def deep(lib: Any) -> Any:
    columns = lib.MultiIndex.from_tuples(
        [("a", "x", 1), ("a", "y", 2), ("b", "x", 1)], names=["p", "q", "r"]
    )
    return lib.DataFrame([[1, 2, 3]], columns=columns)


BUILDS = {
    "first level": lambda lib: layered(lib).xs("a", axis=1, level=0),
    "second level": lambda lib: layered(lib).xs("x", axis=1, level=1),
    "level by name": lambda lib: layered(lib).xs("x", axis=1, level="two"),
    "drop_level": lambda lib: layered(lib).xs("x", axis=1, level="two", drop_level=False),
    "every level": lambda lib: layered(lib).xs(("b", "z"), axis=1, level=[0, 1]),
    "no level": lambda lib: layered(lib).xs("a", axis=1),
    "three levels": lambda lib: deep(lib).xs("x", axis=1, level=1),
    "getitem": lambda lib: layered(lib)["a"],
    "getitem deep": lambda lib: deep(lib)["a"].columns.names,
    "loc every row": lambda lib: layered(lib).loc[:, "b"],
    "loc rows": lambda lib: layered(lib).loc[[0], "a"],
    "loc one row": lambda lib: layered(lib).loc[0, "a"],
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_cross_section_columns_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_missing_key_and_flat_columns_are_refused(firepanda: Any) -> None:
    with pytest.raises(KeyError):
        layered(firepanda).xs("q", axis=1, level=1)
    with pytest.raises(TypeError, match="Index must be a MultiIndex"):
        firepanda.DataFrame({"a": [1]}).xs("a", axis=1, level=0)
