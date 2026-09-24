"""`repeat`, `set_axis`, a frame's `idxmax` and `idxmin`, and `iterrows`, checked against pandas.

`repeat` takes each value and its label as many times as it is told, one count
for all or one count each. `set_axis` puts new labels on the rows or the
columns. A frame's `idxmax` is each column's own, or across a row the column
holding the extreme, and `iterrows` walks the rows as columns named by label.
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
    if hasattr(want, "dtype") and not hasattr(want, "index"):
        return
    assert got.name == want.name


def column(m: ModuleType) -> Any:
    """A named column with text labels."""
    return m.Series([1, 2, 3], index=["a", "b", "c"], name="x")


def frame(m: ModuleType) -> Any:
    """Two number columns, one with a gap, and a named index."""
    return m.DataFrame(
        {"a": [1, 5, 3], "b": [4.0, None, 6.0]}, index=m.Index(["x", "y", "z"], name="k")
    )


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: column(m).repeat(2),
    lambda m: column(m).repeat([1, 0, 2]),
    lambda m: column(m).repeat([2]),
    lambda m: column(m).repeat(2.0),
    lambda m: column(m).repeat([1.5, 2, 0.2]),
    lambda m: column(m).repeat(0),
    lambda m: column(m).set_axis([7, 8, 9]),
    lambda m: column(m).set_axis(m.Index([7, 8, 9], name="q")),
    lambda m: column(m).set_axis(["p", "q", "r"], axis="index"),
    lambda m: frame(m).set_axis(["p", "q", "r"]),
    lambda m: frame(m).set_axis(["b", "a"], axis=1),
    lambda m: frame(m).set_axis(["c", "d"], axis="columns"),
    lambda m: frame(m).idxmax(),
    lambda m: frame(m).idxmin(),
    lambda m: frame(m).idxmax(axis=1),
    lambda m: frame(m).idxmin(axis="columns"),
    lambda m: m.DataFrame({"a": [1, 2], "s": ["x", "y"]}).idxmax(),
    lambda m: m.DataFrame({"a": [1, 2], "s": ["x", "y"]}).idxmin(numeric_only=True),
    lambda m: m.DataFrame({"a": [2.0, 1.0], "b": [2.0, 3.0]}).idxmax(axis=1),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Counts, labels on each axis, extremes both ways, text and ties."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_an_index_repeats_its_labels(firepanda: ModuleType) -> None:
    """An index keeps its name, and one count each works as on a column."""
    import pandas as pd

    for build in (
        lambda m: m.Index([1, 2], name="k").repeat(2),
        lambda m: m.Index([1, 2]).repeat([0, 3]),
    ):
        got, want = build(firepanda), build(pd)
        assert got.tolist() == want.tolist()
        assert got.name == want.name


def test_iterrows_walks_the_rows(firepanda: ModuleType) -> None:
    """Each label, and its row as a column named by the label."""
    import pandas as pd

    mine, yours = list(frame(firepanda).iterrows()), list(frame(pd).iterrows())
    assert [label for label, _ in mine] == [label for label, _ in yours]
    for (_, got), (_, want) in zip(mine, yours, strict=True):
        agrees(got, want)


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: column(m).repeat([1, 2]),
    lambda m: column(m).repeat(-1),
    lambda m: column(m).repeat([1, -1, 2]),
    lambda m: column(m).repeat(2, axis=0),
    lambda m: m.Index([1, 2]).repeat(2, axis=0),
    lambda m: column(m).set_axis([1, 2]),
    lambda m: column(m).set_axis([1, 2, 3], axis=1),
    lambda m: frame(m).set_axis(["p"], axis=1),
    lambda m: frame(m).set_axis([1, 2], axis=0),
    lambda m: frame(m).set_axis(["p", "q"], axis=2),
    lambda m: frame(m).idxmax(skipna=False),
    lambda m: frame(m).idxmax(axis=1, skipna=False),
    lambda m: frame(m).idxmax(axis=2),
    lambda m: m.DataFrame({"a": [None, None]}, dtype="float64").idxmax(),
    lambda m: m.DataFrame({"a": [None, 1.0], "b": [None, 2.0]}).idxmin(axis=1),
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
    ("owner", "method"),
    [
        ("Series", "repeat"),
        ("Index", "repeat"),
        ("Series", "set_axis"),
        ("DataFrame", "set_axis"),
        ("DataFrame", "idxmax"),
        ("DataFrame", "idxmin"),
        ("DataFrame", "iterrows"),
    ],
)
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str, method: str) -> None:
    """Parameter for parameter, with the same defaults but for `copy`, which is unused."""
    import pandas as pd

    ours = inspect.signature(getattr(getattr(firepanda, owner), method)).parameters
    yours = inspect.signature(getattr(getattr(pd, owner), method)).parameters
    assert [(p.name, p.kind) for p in ours.values()] == [(p.name, p.kind) for p in yours.values()]
    assert [p.default for p in ours.values() if p.name != "copy"] == [
        p.default for p in yours.values() if p.name != "copy"
    ]
