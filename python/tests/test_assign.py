"""`DataFrame.assign` and a column read as an attribute, checked against pandas.

`assign` takes its keywords in order, each one seeing the frame the ones before
it made, and reads a value the way pandas reads the right side of `df[name] =`:
a series is lined up on the row labels, a list has to be as long as the frame
and anything else is one value for every row. Every test builds the same frame
in both libraries and compares the columns, the types, the labels and the rows.
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


def frame(m: Any) -> Any:
    """Three rows on labels that are not their positions."""
    return m.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "i": [10, 20, 30]}).set_index("i")


def keyed(m: Any, labels: list[int], values: list[Any]) -> Any:
    """A series on chosen labels, built the one way both libraries share."""
    return m.DataFrame({"i": labels, "v": values}).set_index("i")["v"]


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row, with NaN equal to NaN."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or a == b for a, b in zip(got, want, strict=True)
    )


def agrees(got: Any, want: Any) -> None:
    """The same columns, types, rows and row labels as pandas."""
    assert list(got.columns) == list(want.columns)
    assert list(got.index) == list(want.index)
    for name in want.columns:
        printed = str(want[name].dtype)
        assert got[name].dtype == ("string" if printed == "str" else printed), name
        assert same(got[name].tolist(), want[name].tolist()), name


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: frame(m).assign(c=1, d=1.5, e="s", f=True),
    lambda m: frame(m).assign(c=[4, 5, 6]),
    lambda m: frame(m).assign(c=(4.5, 5, 6)),
    lambda m: frame(m).assign(c=range(3)),
    lambda m: frame(m).assign(c=keyed(m, [30, 10, 20], [7, 8, 9])),
    lambda m: frame(m).assign(c=keyed(m, [30, 99], [7, 8])),
    lambda m: frame(m).assign(c=lambda f: f.a * 2, d=lambda f: f.c + 1),
    lambda m: frame(m).assign(a=lambda f: f["a"] * 10),
    lambda m: frame(m).assign(b=0, z="last"),
    lambda m: frame(m).assign(c=frame(m)[["a"]]),
    lambda m: frame(m).assign(c={30: 1.5, 10: 2.5}),
    lambda m: frame(m).assign(),
    lambda m: frame(m).assign(**{"two words": 1}),
    lambda m: m.DataFrame().assign(a=[1, 2]),
    lambda m: m.DataFrame().assign(a=1),
    lambda m: m.DataFrame({"a": [1, 2, 3]}).assign(c=m.Series([5, 6, 7])),
    lambda m: m.DataFrame({"a": [1, 2, 3]}).head(0).assign(c=1),
    lambda m: m.DataFrame({"a": [1]}).assign(c="only"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_an_assign_is_pandas_assign(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Scalars, lists, aligned series, functions, replacements and empty frames."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_numpy_values_are_read_as_a_list(firepanda: ModuleType) -> None:
    """An array is as long as the frame, like a list."""
    np = pytest.importorskip("numpy")
    import pandas as pd

    agrees(
        frame(firepanda).assign(c=np.array([1.5, 2.0, 3.0])),
        frame(pd).assign(c=np.array([1.5, 2.0, 3.0])),
    )


def test_the_frame_assigned_to_is_unchanged(firepanda: ModuleType) -> None:
    """A new frame comes back and the old one keeps its columns."""
    before = frame(firepanda)
    after = before.assign(c=1, a=0)
    assert list(before.columns) == ["a", "b"]
    assert before["a"].tolist() == [1, 2, 3]
    assert list(after.columns) == ["a", "b", "c"]


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: frame(m).assign(c=[4, 5]),
    lambda m: frame(m).assign(c=frame(m)),
    lambda m: frame(m).assign(c={1, 2, 3}),
    lambda m: frame(m).assign(c=lambda f: f.nope),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class, or a subclass of it, and the same first line."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value).split("\n")[0] == str(theirs.value).split("\n")[0]


def test_none_is_refused_rather_than_guessed(firepanda: ModuleType) -> None:
    """pandas makes an object column, which firepanda does not have."""
    with pytest.raises(NotImplementedError):
        frame(firepanda).assign(c=None)


def test_a_column_reads_as_an_attribute(firepanda: ModuleType) -> None:
    """The column by name, a method before a column, and the columns in `dir`."""
    got = firepanda.DataFrame({"a": [1, 2], "sum": [3, 4], "two words": [5, 6]})
    assert got.a.tolist() == [1, 2]
    assert callable(got.sum)
    assert "a" in dir(got)
    assert "two words" not in dir(got)


def test_the_signature_is_pandas_signature(firepanda: ModuleType) -> None:
    """Keywords only, as pandas takes them."""
    import pandas as pd

    mine = inspect.signature(firepanda.DataFrame.assign).parameters
    yours = inspect.signature(pd.DataFrame.assign).parameters
    assert [(p.name, p.kind) for p in mine.values()] == [(p.name, p.kind) for p in yours.values()]
