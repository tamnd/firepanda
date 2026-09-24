"""`align` on a frame and a column, checked against pandas.

pandas gives both objects the same labels and answers the pair. An outer join
takes every label, sorted when the two sides differ, an inner join the labels
both sides hold, and left and right one side's labels. Two frames line up on
both axes unless `axis` names one, and a frame and a column need an axis.
"""

from __future__ import annotations

import importlib.util
import inspect
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


def left(m: ModuleType) -> Any:
    """A frame with labels out of order and a named index."""
    return m.DataFrame(
        {"b": [1.0, 2.0, 3.0], "a": [4.0, 5.0, 6.0]}, index=m.Index([2, 0, 1], name="k")
    )


def right(m: ModuleType) -> Any:
    """A frame sharing some rows and columns with `left`, and adding some."""
    return m.DataFrame({"c": [7.0, 8.0], "a": [9.0, 10.0]}, index=m.Index([0, 5], name="j"))


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: left(m).align(right(m)),
    lambda m: left(m).align(right(m), join="inner"),
    lambda m: left(m).align(right(m), join="left"),
    lambda m: left(m).align(right(m), join="right"),
    lambda m: left(m).align(right(m), axis=0),
    lambda m: left(m).align(right(m), axis="columns"),
    lambda m: left(m).align(right(m), fill_value=0.0),
    lambda m: left(m).align(left(m), join="bogus"),
    lambda m: left(m).align(m.Series([1.0, 2.0], index=[0, 9]), axis=0),
    lambda m: left(m).align(m.Series([1.0, 2.0], index=["a", "z"]), axis=1),
    lambda m: m.Series([1.0, 2.0], index=["b", "a"]).align(m.Series([3.0], index=["c"])),
    lambda m: m.Series([1.0, 2.0], name="x").align(m.Series([3.0, 4.0, 5.0]), join="inner"),
    lambda m: m.Series(["p", "q"]).align(m.Series([3.0], index=[7]), join="left"),
    lambda m: m.Series([1.0, 2.0], index=[0, 9]).align(left(m)),
    lambda m: m.Series([1.0, 2.0], index=[4, 4]).align(m.Series([5.0, 6.0], index=[4, 4])),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Each join, each axis, a fill, a frame with a column, and labels already equal."""
    import pandas as pd

    mine, yours = build(firepanda), build(pd)
    assert isinstance(mine, tuple)
    assert len(mine) == 2
    agrees(mine[0], yours[0])
    agrees(mine[1], yours[1])


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: left(m).align(right(m), join="bogus"),
    lambda m: left(m).align(right(m), axis=2),
    lambda m: left(m).align(m.Series([1.0])),
    lambda m: left(m).align([1, 2]),
    lambda m: m.Series([1.0]).align(m.Series([2.0], index=[3]), axis=1),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class or a subclass of it, and the same message."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.Series([1.0, 2.0], index=[4, 4]).align(m.Series([5.0], index=[3])),
        lambda m: m.Series([1.0]).align(m.Series([2.0], index=[3]), level=1),
    ],
)
def test_what_is_not_written_is_refused(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Repeated labels that pandas joins, and a level of a MultiIndex."""
    with pytest.raises(NotImplementedError, match="align"):
        build(firepanda)


@pytest.mark.parametrize("owner", ["Series", "DataFrame"])
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str) -> None:
    """Parameter for parameter, with the same defaults but for `copy`, which is unused."""
    import pandas as pd

    ours = inspect.signature(getattr(firepanda, owner).align).parameters
    yours = inspect.signature(getattr(pd, owner).align).parameters
    assert [(p.name, p.kind) for p in ours.values()] == [(p.name, p.kind) for p in yours.values()]
    assert [p.default for p in ours.values() if p.name != "copy"] == [
        p.default for p in yours.values() if p.name != "copy"
    ]
