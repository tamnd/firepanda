"""`map`, `apply`, `combine`, `update` and `transpose` on a column, and `map` on a
frame, checked against pandas.

pandas calls the function on each value in Python, hands a missing value over
as NaN unless asked to leave it alone, and reads the answer's type from what
came back. A mapping answers a key it lacks as missing, and a column is read as
a mapping from its labels.
"""

from __future__ import annotations

import collections
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
    """The same labels, values, types and names, or the same single value."""
    if not hasattr(want, "index"):
        assert got == want
        return
    assert list(got.index) == list(want.index)
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            agrees(got[name], want[name])
        return
    assert plain(got.tolist()) == plain(want.tolist())
    assert str(got.dtype).replace("string", "str") == str(want.dtype)
    assert got.name == want.name


def column(m: ModuleType) -> Any:
    """Whole numbers with labels and a name."""
    return m.Series([1, 2, 3], index=["x", "y", "z"], name="a")


def gaps(m: ModuleType) -> Any:
    """Floats with a gap."""
    return m.Series([1.0, None, 3.0], name="g")


def text(m: ModuleType) -> Any:
    """Text with a gap."""
    return m.Series(["a", None, "c"])


def updated(m: ModuleType, other: Callable[[ModuleType], Any]) -> Any:
    """A column after `update`, which answers nothing and changes the column."""
    target = m.Series([1, 2, 3], name="u")
    assert target.update(other(m)) is None
    return target


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: column(m).map(lambda v: v * 2),
    lambda m: column(m).map(lambda v: v > 1),
    lambda m: column(m).map(lambda v: v / 2),
    lambda m: column(m).map(str),
    lambda m: column(m).map({1: "one", 3: "three"}),
    lambda m: column(m).map(collections.defaultdict(lambda: "other", {1: "one"})),
    lambda m: column(m).map(m.Series(["p", "q"], index=[2, 1])),
    lambda m: column(m).map(lambda v, k: v + k, k=10),
    lambda m: gaps(m).map(lambda v: v * 2),
    lambda m: gaps(m).map(lambda v: v * 2, na_action="ignore"),
    lambda m: text(m).map(str.upper, na_action="ignore"),
    lambda m: text(m).map(lambda v: type(v).__name__),
    lambda m: text(m).map({"a": 1}),
    lambda m: m.Series([], dtype="int64").map(str),
    lambda m: column(m).apply(lambda v: v + 1),
    lambda m: column(m).apply(lambda v, k: v * k, args=(3,)),
    lambda m: column(m).apply("sum"),
    lambda m: column(m).apply(["sum", "max"]),
    lambda m: column(m).combine(m.Series([5, 1], index=["y", "w"], name="b"), max),
    lambda m: column(m).combine(m.Series([5, 1], index=["y", "w"]), max, fill_value=0),
    lambda m: column(m).combine(m.Series([5, 0, 1], index=["x", "y", "z"], name="a"), min),
    lambda m: column(m).combine(2, lambda p, q: p * q),
    lambda m: updated(m, lambda n: n.Series([9, None, 7], index=[0, 1, 5])),
    lambda m: updated(m, lambda n: [9, 8, 7]),
    lambda m: updated(m, lambda n: {1: 5}),
    lambda m: column(m).T,
    lambda m: column(m).transpose(),
    lambda m: m.DataFrame({"a": [1, 2], "b": [3.5, None]}).map(lambda v: v * 2),
    lambda m: m.DataFrame({"a": [1, 2], "b": [3.5, None]}).map(lambda v, k: v + k, k=1),
    lambda m: m.DataFrame({"a": ["x", None]}, index=[5, 6]).map(str.upper, na_action="ignore"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Functions, mappings, columns as mappings, gaps, names and each method."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_text_updated_stays_text(firepanda: ModuleType) -> None:
    """A text column takes text from the other column."""
    target = firepanda.Series(["a", "b"])
    target.update(firepanda.Series(["z"], index=[1]))
    assert target.tolist() == ["a", "z"]


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.Series([1]).map(1),
    lambda m: m.Series([1]).map(str, na_action="x"),
    lambda m: m.Series([1, 2]).map(m.Series(["a", "b"], index=[1, 1])),
    lambda m: updated(m, lambda n: n.Series([1.5], index=[0])),
    lambda m: m.DataFrame({"a": [1]}).map({1: 2}),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class or a subclass of it, and the same message."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert type(mine.value).__name__ == type(theirs.value).__name__ or isinstance(
        mine.value, type(theirs.value)
    )
    assert str(mine.value) == str(theirs.value)


def test_a_category_column_is_refused(firepanda: ModuleType) -> None:
    """pandas maps the categories, which is not done here yet."""
    with pytest.raises(NotImplementedError, match="category"):
        firepanda.Series(["a"], dtype="category").map(str.upper)


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("Series", "map"),
        ("Series", "apply"),
        ("Series", "combine"),
        ("Series", "update"),
        ("Series", "transpose"),
        ("DataFrame", "map"),
    ],
)
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(getattr(firepanda, owner), name)).parameters
    yours = inspect.signature(getattr(getattr(pd, owner), name)).parameters
    assert [(p.name, p.kind, repr(p.default)) for p in ours.values()] == [
        (p.name, p.kind, repr(p.default)) for p in yours.values()
    ]
