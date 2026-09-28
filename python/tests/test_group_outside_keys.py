"""Grouping by keys that are not one of the frame's columns, and grouping a column.

pandas takes a column of values lined up on the row labels, an array or a list
as long as the frame, a function or a dictionary of the row labels, and the row
labels themselves by `level` or by their name. Each is named the way pandas
names it, and a key from outside the frame is never one of the columns reduced
or handed back. `Series.groupby` takes the same keys.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def frame(m: ModuleType) -> Any:
    return m.DataFrame(
        {"a": [1, 2, 1, 3], "b": [1.5, 2.5, 3.5, 4.5]}, index=m.Index([10, 11, 12, 13], name="ix")
    )


def unnamed(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1, 2, 1, 3], "b": [1.5, 2.5, 3.5, 4.5]}, index=[10, 11, 12, 13])


def repeated(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1, 2, 3]}, index=[1, 1, 2])


def outside(m: ModuleType) -> Any:
    return m.Series([1, 1, 2, 2], index=[13, 12, 11, 10], name="z")


def by_position(m: ModuleType) -> Any:
    return m.Series([1, 1, 2, 2], index=[10, 11, 12, 13], name="z")


def own(table: Any, key: Callable[[Any], Any], **kwargs: Any) -> Any:
    """`table` grouped by a column made from itself, as `df.groupby(df["a"])` is written."""
    return table.groupby(key(table), **kwargs)


def evens(label: int) -> int:
    return label % 2


LETTERS = {10: "x", 11: "y", 12: "x", 13: "y"}

ANSWERS: dict[str, Callable[[ModuleType], Any]] = {
    "own column": lambda m: own(frame(m), lambda df: df["a"]).sum(),
    "own column as columns": lambda m: own(frame(m), lambda df: df["a"], as_index=False).sum(),
    "a column worked on": lambda m: own(frame(m), lambda df: df["a"] // 2).sum(),
    "another frame's column": lambda m: (
        frame(m).groupby(unnamed(m)["a"].set_axis([10, 11, 12, 13]) + 1).sum()
    ),
    "a gap in a key of floats": lambda m: (
        m.DataFrame({"k": [2.0, 1.0, float("nan")], "v": [1, 2, 3]}).groupby("k").sum()
    ),
    "a gap in a key kept": lambda m: (
        m.DataFrame({"k": [2.0, 1.0, float("nan")], "v": [1, 2, 3]})
        .groupby("k", dropna=False)
        .sum()
    ),
    "a gap in a key transformed": lambda m: (
        m.DataFrame({"k": [2.0, 1.0, float("nan")], "v": [1, 2, 3]})
        .groupby("k")["v"]
        .transform("sum")
    ),
    "size of a column by name": lambda m: frame(m).groupby("a")["b"].size(),
    "series lined up": lambda m: frame(m).groupby(outside(m)).sum(),
    "series as columns": lambda m: frame(m).groupby(outside(m), as_index=False).sum(),
    "series with gaps": lambda m: frame(m).groupby(m.Series([1, 2], index=[11, 10])).sum(),
    "series on repeated labels": lambda m: (
        repeated(m).groupby(m.Series([1, 2], index=[2, 1])).sum()
    ),
    "list": lambda m: frame(m).groupby([1, 1, 2, 2]).sum(),
    "list as columns": lambda m: frame(m).groupby([1, 1, 2, 2], as_index=False).mean(),
    "list of text": lambda m: frame(m).groupby(["x", "y", "x", "y"]).sum(),
    "array": lambda m: frame(m).groupby(__import__("numpy").array([2, 2, 1, 1])).max(),
    "index": lambda m: frame(m).groupby(m.Index([1, 1, 2, 2], name="w")).sum(),
    "function": lambda m: frame(m).groupby(evens).sum(),
    "function unnamed": lambda m: unnamed(m).groupby(evens).sum(),
    "dictionary": lambda m: frame(m).groupby(LETTERS).sum(),
    "dictionary with a gap": lambda m: frame(m).groupby({10: "x", 11: "y"}).sum(),
    "level": lambda m: frame(m).groupby(level=0).sum(),
    "level from the back": lambda m: frame(m).groupby(level=-1).count(),
    "level by name": lambda m: frame(m).groupby(level="ix").sum(),
    "level in a list": lambda m: frame(m).groupby(level=[0]).sum(),
    "labels by name": lambda m: frame(m).groupby("ix").sum(),
    "level as columns": lambda m: frame(m).groupby(level=0, as_index=False).sum(),
    "unnamed level as columns": lambda m: unnamed(m).groupby(level=0, as_index=False).sum(),
    "two unnamed as columns": lambda m: (
        frame(m).groupby([[1, 1, 2, 2], [0, 1, 0, 1]], as_index=False).sum()
    ),
    "name and values as columns": lambda m: (
        frame(m).groupby(["a", [0, 1, 0, 1]], as_index=False).sum()
    ),
    "unsorted": lambda m: frame(m).groupby([2, 1, 2, 1], sort=False).sum(),
    "size": lambda m: frame(m).groupby(outside(m)).size(),
    "size as columns": lambda m: frame(m).groupby([1, 1, 2, 2], as_index=False).size(),
    "one column": lambda m: frame(m).groupby(outside(m))["b"].sum(),
    "some columns": lambda m: frame(m).groupby(outside(m))[["b"]].mean(),
    "agg": lambda m: frame(m).groupby(outside(m)).agg({"a": "sum", "b": "max"}),
    "agg a function": lambda m: frame(m).groupby([1, 1, 2, 2])["b"].agg(lambda g: g.max() - 1),
    "transform": lambda m: frame(m).groupby(outside(m)).transform("sum"),
    "cumsum": lambda m: frame(m).groupby([1, 1, 2, 2]).cumsum(),
    "head": lambda m: frame(m).groupby(outside(m)).head(1),
    "tail": lambda m: frame(m).groupby([1, 1, 2, 2]).tail(1),
    "nth": lambda m: frame(m).groupby(outside(m)).nth(0),
    "filter": lambda m: frame(m).groupby(outside(m)).filter(lambda g: g["a"].sum() > 3),
    "get_group": lambda m: frame(m).groupby(outside(m)).get_group(1),
    "apply": lambda m: frame(m).groupby(outside(m)).apply(lambda g: g["b"].sum()),
    "describe": lambda m: frame(m).groupby(outside(m))["b"].describe(),
    "series by a series": lambda m: frame(m)["b"].groupby(frame(m)["a"]).sum(),
    "series by a list": lambda m: frame(m)["b"].groupby([1, 1, 2, 2]).mean(),
    "series by level": lambda m: frame(m)["b"].groupby(level=0).sum(),
    "series by the labels' name": lambda m: frame(m)["b"].groupby("ix").sum(),
    "series by a function": lambda m: frame(m)["b"].groupby(evens).sum(),
    "series by a dictionary": lambda m: frame(m)["b"].groupby(LETTERS).max(),
    "unnamed series": lambda m: frame(m)["b"].rename(None).groupby([1, 1, 2, 2]).sum(),
    "a column by a series lined up": lambda m: frame(m)["b"].groupby(outside(m)).sum(),
    "series agg list": lambda m: frame(m)["b"].groupby(by_position(m)).agg(["sum", "min"]),
    "series transform": lambda m: frame(m)["b"].groupby(by_position(m)).transform("mean"),
    "series head": lambda m: frame(m)["b"].groupby(by_position(m)).head(1),
    "series filter": lambda m: frame(m)["b"].groupby(by_position(m)).filter(lambda g: len(g) > 1),
    "series apply": lambda m: frame(m)["b"].groupby(by_position(m)).apply(lambda g: g.max()),
    "series describe": lambda m: frame(m)["b"].groupby(by_position(m)).describe(),
    "series size": lambda m: frame(m)["b"].groupby(by_position(m)).size(),
    "series nunique": lambda m: frame(m)["a"].groupby(by_position(m)).nunique(),
}

MISTAKES: dict[str, Callable[[ModuleType], Any]] = {
    "a short list names columns": lambda m: frame(m).groupby([1, 2, 3]),
    "a short array": lambda m: frame(m).groupby(__import__("numpy").array([1, 2, 3])),
    "a level past the first": lambda m: frame(m).groupby(level=1),
    "a level by a name it does not have": lambda m: frame(m).groupby(level="zz"),
    "a name that is neither": lambda m: frame(m).groupby("zz"),
    "a column grouped as columns": lambda m: frame(m)["b"].groupby([1, 1, 2, 2], as_index=False),
    "a column by a name it does not have": lambda m: frame(m)["b"].groupby("zz"),
    "a column with no key": lambda m: frame(m)["b"].groupby(),
}


def shown(answer: Any) -> Any:
    """An answer's shape, names, labels and values, spelled the same in both."""
    if hasattr(answer, "columns"):
        values = {name: repr(answer[name].tolist()) for name in answer.columns}
        return list(answer.columns), answer.index.name, repr(answer.index.tolist()), values
    return answer.name, answer.index.name, repr(answer.index.tolist()), repr(answer.tolist())


@needs_pandas
@pytest.mark.parametrize("name", list(ANSWERS))
def test_an_answer_matches_pandas(firepanda: ModuleType, name: str) -> None:
    """The columns, the names, the labels and every value."""
    import pandas as pd

    assert shown(ANSWERS[name](firepanda)) == shown(ANSWERS[name](pd))


@needs_pandas
@pytest.mark.parametrize("name", list(MISTAKES))
def test_a_mistake_raises_what_pandas_raises(firepanda: ModuleType, name: str) -> None:
    """The same kind of error and the same words."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        MISTAKES[name](pd)
    with pytest.raises(theirs.type) as mine:
        MISTAKES[name](firepanda)
    assert str(mine.value) == str(theirs.value)


@needs_pandas
def test_iterating_hands_back_scalar_keys_and_no_hidden_column(firepanda: ModuleType) -> None:
    """Each group is the frame's own columns, and one key is not a tuple."""
    import pandas as pd

    def groups(m: ModuleType) -> Any:
        return [(k, list(g.columns)) for k, g in frame(m).groupby([1, 1, 2, 2])]

    def pieces(m: ModuleType) -> Any:
        return [(k, g.name, g.tolist()) for k, g in frame(m)["b"].groupby(outside(m))]

    assert groups(firepanda) == groups(pd)
    assert pieces(firepanda) == pieces(pd)


def test_the_group_by_is_still_its_class(firepanda: ModuleType) -> None:
    """The grouping that shows outside keys is a `DataFrameGroupBy` or a `SeriesGroupBy`."""
    table = frame(firepanda)
    from firepanda._frame import DataFrameGroupBy, SeriesGroupBy

    assert isinstance(table.groupby([1, 1, 2, 2]), DataFrameGroupBy)
    assert isinstance(table["b"].groupby([1, 1, 2, 2]), SeriesGroupBy)
    assert type(table.groupby(level=0)).__name__ == "DataFrameGroupBy"


def test_by_and_level_together_are_refused(firepanda: ModuleType) -> None:
    """Both at once pick the levels of a MultiIndex."""
    with pytest.raises(NotImplementedError):
        frame(firepanda).groupby("a", level=0)
