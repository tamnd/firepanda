"""`combine_first` on a frame and a column, checked against pandas.

pandas fills every missing value from the same row and column of the other
object. The rows are both sides' rows, sorted when the two differ, and a frame
has its own columns first and then the ones only the other frame has. A column
both sides hold takes the type the two types have in common.
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
    """The same labels, index name and values, and types column by column."""
    assert list(got.index) == list(want.index)
    assert got.index.name == want.index.name
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            agrees(got[name], want[name])
        return
    assert plain(got.tolist()) == plain(want.tolist())
    assert str(got.dtype).replace("string", "str") == str(want.dtype)
    assert got.name == want.name


def gaps(m: ModuleType) -> Any:
    """A frame with a gap in each column and labels out of order."""
    return m.DataFrame({"b": [1.0, None, 3], "a": [None, 2.0, 3.0]}, index=[2, 0, 1])


def other(m: ModuleType) -> Any:
    """A frame sharing some rows and columns with `gaps`, and adding some."""
    return m.DataFrame({"c": [9, 9], "a": [7.0, 8.0], "b": [5, 6]}, index=[0, 5])


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: gaps(m).combine_first(other(m)),
    lambda m: other(m).combine_first(gaps(m)),
    lambda m: gaps(m).combine_first(gaps(m).fillna(0.0)),
    lambda m: m.DataFrame({"x": [1, 2]}).combine_first(m.DataFrame({"x": [5, 6, 7]})),
    lambda m: m.DataFrame({"x": [1, 2]}).combine_first(m.DataFrame({"x": [1.5, None]})),
    lambda m: m.DataFrame({"s": ["a", None]}).combine_first(
        m.DataFrame({"s": ["z", "y"], "t": [1, 2]})
    ),
    lambda m: m.DataFrame({"x": [1, 2]}, index=["b", "a"]).combine_first(
        m.DataFrame({"x": [5]}, index=["c"])
    ),
    lambda m: m.DataFrame({"x": [1, 2]}, index=m.Index([3, 1], name="k")).combine_first(
        m.DataFrame({"y": [5]}, index=m.Index([1], name="k"))
    ),
    lambda m: m.DataFrame({"x": [1.0, None]}, index=[4, 4]).combine_first(
        m.DataFrame({"x": [5.0, 6.0]}, index=[4, 4])
    ),
    lambda m: m.DataFrame({"x": [1.0, None]}).combine_first(m.DataFrame(index=[0, 1])),
    lambda m: m.Series([1.0, None, 3], index=[2, 0, 1]).combine_first(
        m.Series([7, 8], index=[0, 5])
    ),
    lambda m: m.Series([1, 2], name="x").combine_first(m.Series([7, 8, 9], name="y")),
    lambda m: m.Series([None, "b"], name="s").combine_first(m.Series(["a", "z"])),
    lambda m: m.Series([1.0, None], index=m.Index([0, 1], name="k")).combine_first(
        m.Series([5.0, 6.0])
    ),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Shared and new rows and columns, types in common, text, names and repeats."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_a_column_is_not_a_frame(firepanda: ModuleType) -> None:
    """pandas reads the other object's columns, so a column fails where it does."""
    import pandas as pd

    with pytest.raises(AttributeError) as theirs:
        pd.DataFrame({"x": [1]}).combine_first(pd.Series([1]))
    with pytest.raises(AttributeError) as mine:
        firepanda.DataFrame({"x": [1]}).combine_first(firepanda.Series([1]))
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.DataFrame({"x": [1.0, None]}).combine_first(m.DataFrame({"x": ["a", "b"]})),
        lambda m: m.Series([True, False]).combine_first(m.Series([1, 2])),
        lambda m: m.Series([1.0, None], index=[1, 1]).combine_first(m.Series([5.0], index=[2])),
    ],
)
def test_what_pandas_answers_as_objects_is_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Two types with nothing in common, and repeated labels that pandas joins."""
    with pytest.raises(NotImplementedError, match="combine_first"):
        build(firepanda)
