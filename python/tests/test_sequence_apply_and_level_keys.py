"""Three things a styled table leans on, checked against pandas.

A function given to `apply` that answers a list, a tuple or an array for each
column makes a frame of them, as pandas makes one. A `loc` key with a slice or
a list for some level of the rows picks rows level by level, in the order the
lists give. A column of numbers or flags compared with a piece of text is False
for `==` and True for `!=` on every row, and an order between them is refused
in pandas' words.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word."""
    return ["missing" if value is None or value != value else value for value in values]


def agrees(got: Any, want: Any) -> None:
    """The same labels, values, types and names."""
    assert list(got.index) == list(want.index)
    assert list(got.index.names) == list(want.index.names)
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            agrees(got[name], want[name])
        return
    assert plain(got.tolist()) == plain(want.tolist())
    assert str(got.dtype).replace("string", "str") == str(want.dtype)
    assert got.name == want.name


def mixed(m: ModuleType) -> Any:
    """Floats with a gap, whole numbers and text, with text row labels."""
    return m.DataFrame(
        {"a": [1.5, -2.0, None, 4.25], "b": [1, 2, 3, 4], "s": ["x", "y", None, "z"]},
        index=["r0", "r1", "r2", "r3"],
    )


def levels(m: ModuleType) -> Any:
    """Rows labelled by two levels."""
    index = m.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1), ("B", 2)], names=["k", "n"])
    return m.DataFrame({"a": [1, 2, 3, 4], "b": [5.0, 6.0, 7.0, 8.0]}, index=index)


def numpy_answer(column: Any) -> Any:
    """An array of text as long as the column."""
    import numpy as np

    return np.array(["u"] * len(column))


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: mixed(m)[["a", "b"]].apply(lambda c: ["q"] * len(c)),
    lambda m: mixed(m)[["a", "b"]].apply(lambda c: ("q",) * len(c)),
    lambda m: mixed(m)[["a", "b"]].apply(lambda c: [1, 2]),
    lambda m: mixed(m)[["a", "b"]].apply(lambda c: [v * 2 for v in c]),
    lambda m: mixed(m)[["a", "b"]].apply(lambda c: ["" if v != v else "on" for v in c]),
    lambda m: mixed(m)[["a", "b"]].apply(lambda c: [1] if c.name == "a" else [1, 2]),
    lambda m: mixed(m)[["a", "b"]].apply(lambda c: ["q"] * len(c), axis=1),
    lambda m: levels(m).loc[("A", slice(None)), :],
    lambda m: levels(m).loc[m.IndexSlice["A", :], :],
    lambda m: levels(m).loc[(slice(None), 1), :],
    lambda m: levels(m).loc[(["B", "A"], slice(None)), :],
    lambda m: levels(m).loc[(slice(None), [2, 1]), :],
    lambda m: levels(m).loc[(slice("A", "B"), [2, 1]), :],
    lambda m: levels(m).loc[(slice(None, None, -1), 1), :],
    lambda m: levels(m).loc[("B", [2, 1]), :],
    lambda m: levels(m).loc[[("A", 1), ("B", 2)], :],
    lambda m: levels(m).loc[("A", slice(None))],
    lambda m: levels(m).loc[(["B"], "a")],
    lambda m: levels(m).loc[(slice(None), "a")],
    lambda m: levels(m).loc[:, "a"],
    lambda m: levels(m).loc[(("A", slice(None)), "a")],
    lambda m: levels(m).loc[(slice(None),)],
    lambda m: levels(m).loc[(["B"],)],
    lambda m: levels(m)["a"].loc[(slice("A", "A"),)],
    lambda m: mixed(m)["a"] == "",
    lambda m: mixed(m)["b"] != "x",
    lambda m: mixed(m) == "x",
    lambda m: mixed(m).ne("x"),
    lambda m: mixed(m)[["a", "b"]].eq(""),
    lambda m: m.Series([True, False]) == "a",
    lambda m: m.Series([1, 2], dtype="uint8").eq("a"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Sequences framed, rows picked by level, and numbers compared with text."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_an_array_for_each_column_is_a_frame(firepanda: ModuleType) -> None:
    """A numpy array of text for each column makes a frame of text."""
    pytest.importorskip("numpy")
    import pandas as pd

    agrees(
        mixed(firepanda)[["a", "b"]].apply(numpy_answer),
        mixed(pd)[["a", "b"]].apply(numpy_answer),
    )


def test_get_locs_follows_the_lists(firepanda: ModuleType) -> None:
    """The positions `get_locs` answers are in the order the lists name their values."""
    import pandas as pd

    for key in ([["B", "A"], 1], [slice(None), [2, 1]], ["A", slice(None)]):
        mine = [int(at) for at in levels(firepanda).index.get_locs(key)]
        assert mine == [int(at) for at in levels(pd).index.get_locs(key)]


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: mixed(m)["a"] < "x",
    lambda m: mixed(m)["b"] >= "x",
    lambda m: mixed(m)["a"].lt("x"),
    lambda m: m.Series([True, False]) <= "a",
    lambda m: mixed(m)[["a", "b"]] > "x",
    lambda m: levels(m).loc[(["A", "B"], 1)],
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class, and the same message for an order between numbers and text."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    if isinstance(theirs.value, TypeError):
        assert str(mine.value) == str(theirs.value)
