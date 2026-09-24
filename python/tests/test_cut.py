"""`cut` and `qcut`, checked against pandas.

pandas puts each value in a bin and answers the bin's position or its label.
A count of bins spreads them evenly over the values, and `qcut` puts the edges
at quantiles. pandas labels bins with intervals by default, which firepanda
does not have, so it answers positions with `labels=False` and ordered
categories of text with a list of labels.
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
    """The same labels, name, values and type, and the same categories in order."""
    import numpy as np

    if isinstance(want, tuple):
        agrees(got[0], want[0])
        assert isinstance(got[1], np.ndarray)
        assert got[1].dtype == want[1].dtype
        assert np.allclose(got[1], want[1])
        return
    if isinstance(want, np.ndarray):
        assert isinstance(got, np.ndarray)
        assert got.dtype == want.dtype
        assert plain(got.tolist()) == plain(want.tolist())
        return
    assert list(got.index) == list(want.index)
    assert got.name == want.name
    assert plain(got.tolist()) == plain(want.tolist())
    assert str(got.dtype) == str(want.dtype)
    if str(want.dtype) == "category":
        assert list(got.cat.categories) == list(want.cat.categories)
        assert got.cat.ordered == want.cat.ordered


def numbers(m: ModuleType) -> Any:
    """A named column with labels out of order, a gap and a repeat."""
    return m.Series([4.0, 1.0, None, 7.5, 2.0, 7.5], index=[9, 3, 5, 1, 0, 2], name="v")


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.cut(numbers(m), 3, labels=False),
    lambda m: m.cut(numbers(m), 3, labels=False, right=False),
    lambda m: m.cut(numbers(m), [1, 4, 8], labels=False),
    lambda m: m.cut(numbers(m), [1, 4, 8], labels=False, include_lowest=True),
    lambda m: m.cut(numbers(m), [1, 4, 8], labels=False, right=False),
    lambda m: m.cut(m.Series([1, 2, 3, 10]), 2, labels=False),
    lambda m: m.cut(m.Series([5, 5, 5]), 2, labels=False),
    lambda m: m.cut(m.Series([0, 0]), 1, labels=False),
    lambda m: m.cut(m.Series([1, 2, 3]), [0, 2, 3], labels=["low", "high"]),
    lambda m: m.cut(numbers(m), [0, 3, 5, 8], labels=["a", "b", "c"]),
    lambda m: m.cut(numbers(m), [0, 3, 5, 8], labels=["x", "y", "x"], ordered=False),
    lambda m: m.cut(numbers(m), [0, 3, 5, 8], labels=("c", "b", "a")),
    lambda m: m.cut(numbers(m), [1, 1, 4, 8], labels=False, duplicates="drop"),
    lambda m: m.cut(numbers(m), [1, 4, 8], labels=False, retbins=True),
    lambda m: m.cut(numbers(m), 2, labels=["s", "t"], retbins=True),
    lambda m: m.cut([1, 5, 9], 2, labels=False),
    lambda m: m.cut([1.0, float("nan"), 9.0], [0, 5, 10], labels=False),
    lambda m: m.cut(m.Series([True, False, True]), 2, labels=False),
    lambda m: m.qcut(m.Series(range(10)), 4, labels=False),
    lambda m: m.qcut(numbers(m), 2, labels=False),
    lambda m: m.qcut(numbers(m), [0, 0.3, 1], labels=["low", "high"]),
    lambda m: m.qcut(m.Series(range(7), name="n"), 3, labels=["a", "b", "c"], retbins=True),
    lambda m: m.qcut(m.Series([1, 1, 1, 2]), 4, labels=False, duplicates="drop"),
    lambda m: m.qcut([3, 1, 2], 3, labels=False, retbins=True),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Counts and edges, both sides, gaps, labels, repeats and the edges back."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.cut(m.Series([1, 2]), 0),
        lambda m: m.cut(m.Series([], dtype="float64"), 2),
        lambda m: m.cut(m.Series([1.0, float("inf")]), 2),
        lambda m: m.cut(m.Series([1, 2]), [3, 1, 2]),
        lambda m: m.cut(m.Series([1, 2]), [1, 1, 2]),
        lambda m: m.cut(m.Series([1, 2]), [1, 2, 3], duplicates="keep"),
        lambda m: m.cut(m.Series([1, 2]), [1, 2, 3], labels=["a"]),
        lambda m: m.cut(m.Series([1, 2]), [1, 2, 3], labels=["a", "a"]),
        lambda m: m.cut(m.Series([1, 2]), [1, 2, 3], labels="ab"),
        lambda m: m.cut(m.Series([1, 2]), [1, 2, 3], ordered=False),
        lambda m: m.qcut(m.Series([1, 1, 1, 2]), 4, labels=False),
        lambda m: m.cut([[1, 2]], 2, labels=False),
    ],
)
def test_mistakes_fail_as_pandas_fails(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """No bins, no values, infinity, edges out of order or repeated, and bad labels."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert type(mine.value) is type(theirs.value)
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.cut(m.Series([1, 2, 3]), 2),
        lambda m: m.qcut(m.Series([1, 2, 3]), 2),
        lambda m: m.cut(m.Series([1, 2, 3]), 2, labels=[1, 2]),
        lambda m: m.cut([1, 2, 3], 2, labels=["a", "b"]),
    ],
)
def test_what_firepanda_cannot_hold_is_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Interval labels, labels that are not text, and a Categorical of a list."""
    with pytest.raises(NotImplementedError, match="cut"):
        build(firepanda)
